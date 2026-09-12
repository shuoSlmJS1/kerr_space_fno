"""Focused CPU contracts for the unified benchmark, without formal experiments."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.models.benchmark_track_a import (
    MODEL_CONFIGS, DeepONet, build_model, model_config, parameter_counts,
)
from src.models.timesnet1d import select_dominant_frequencies
from src.training.benchmark_data import (
    Trajectories, fit_normalization, restore_normalization, recover_xyz,
    validate_pair, TRAIN_TASKS, EVAL_TASKS,
    sha256_file,
)
from src.training.benchmark_track_a import (
    TRAINING, SCHEMA, checkpoint_contract, fit_epochs, seed_training,
    predict_frozen, resolution_routing, robustness, planned_matrix,
    save_checkpoint, train_run, evaluate_run, output_path, inspect_device,
)


COUNTS = {"bilstm": 1093331, "resnet": 1096119, "timesnet": 1077059,
          "transformer": 1088003, "fno1d": 1069763, "deeponet": 1085699}


def synthetic(n=3, t=9, offset=0):
    q = np.array([2.7, 1.8, 2.3, 2.0][:n], dtype=np.float64)[:, None] + offset
    grid = np.linspace(0, 5.995, t)
    y = np.stack([q + grid, q - grid, np.broadcast_to(q, (n, t))], axis=-1)
    return Trajectories(q, grid, y)


def metadata(name, native=1200):
    return {"schema": SCHEMA, "training": dict(TRAINING), "model_name": name,
            "model_config": model_config(name), "train_resolution": native,
            "capacity": parameter_counts(build_model(name)), "dataset_provenance": {"synthetic": True}}


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_builders_dynamic_shapes_capacity_and_backward(self):
        for name in MODEL_CONFIGS:
            with self.subTest(name=name):
                model = build_model(name)
                counts = parameter_counts(model)
                self.assertEqual(counts["tensor_numel"], COUNTS[name])
                self.assertEqual(counts["real_scalar_parameter_count"], COUNTS[name])
                self.assertTrue(900000 <= counts["real_scalar_parameter_count"] <= 1300000)
                for t in (8, 13):
                    x = torch.from_numpy(synthetic(n=2, t=t).raw_input())
                    model.zero_grad(set_to_none=True)
                    y = model(x)
                    self.assertEqual(y.shape, (2, t, 3))
                    y.square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_counts_complex_frozen_buffers_shared(self):
        model = nn.Module()
        model.real = nn.Parameter(torch.zeros(3))
        model.shared = model.real
        model.complex = nn.Parameter(torch.zeros(5, dtype=torch.complex64))
        model.frozen = nn.Parameter(torch.zeros(7), requires_grad=False)
        model.register_buffer("buffer", torch.zeros(11))
        self.assertEqual(parameter_counts(model), {
            "tensor_numel": 8, "real_scalar_parameter_count": 13, "trainable_parameter_bytes": 52})

    def test_resnet_rf(self):
        model = build_model("resnet")
        self.assertEqual(model.dilations, tuple(2**i for i in range(11)))
        self.assertEqual(len(model.blocks), 11)
        self.assertEqual(model.theoretical_receptive_field, 12349)
        self.assertTrue(all(b.local_conv.dilation == (1,) and b.local_conv.kernel_size == (7,) for b in model.blocks))

    def test_transformer_standard_attention_runtime_coordinates(self):
        model = build_model("transformer").eval()
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(all(isinstance(layer, nn.TransformerEncoderLayer) for layer in model.layers))
        self.assertFalse(any(isinstance(layer, nn.Embedding) for layer in model.modules()))
        self.assertFalse(any("pos" in name for name, _ in model.named_parameters()))
        x = torch.from_numpy(synthetic(n=2).raw_input())
        order = torch.randperm(x.shape[1])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x[:, order]), model(x)[:, order], atol=2e-6))
            changed = x.clone()
            changed[:, :, 1] += 0.5
            self.assertFalse(torch.allclose(model(changed), model(x)))

    def test_fno_native_resolutions(self):
        model = build_model("fno1d").eval()
        with torch.no_grad():
            for t in EVAL_TASKS:
                self.assertEqual(model(torch.from_numpy(synthetic(n=1, t=t).raw_input())).shape, (1, t, 3))
        self.assertFalse(any(p.is_complex() for p in model.parameters()))

    def test_deeponet_information_and_queries(self):
        model = build_model("deeponet").eval()
        self.assertIsInstance(model, DeepONet)
        self.assertEqual(model.branch[0].in_features, 1)
        self.assertEqual(model.trunk[0].in_features, 1)
        x = torch.from_numpy(synthetic(n=2).raw_input())
        with torch.no_grad():
            full = model(x)
            self.assertTrue(torch.allclose(full[:, ::2], model(x[:, ::2]), atol=1e-6))
            changed = x.clone()
            changed[1] = x[0]
            self.assertTrue(torch.equal(model(changed)[0], model(changed)[1]))
            x[0, 1, 0] += 1
            with self.assertRaises(ValueError):
                model(x)

    def test_timesnet_canonical_fft_and_batch_shared_topk(self):
        t = torch.arange(32, dtype=torch.float32)
        x = torch.stack([torch.sin(2 * torch.pi * 3 * t / 32),
                         10 * torch.sin(2 * torch.pi * 5 * t / 32)])[:, :, None]
        frequencies, periods, weights = select_dominant_frequencies(x, 2)
        self.assertEqual(set(frequencies.tolist()), {3, 5})
        self.assertTrue(torch.equal(periods, 32 // frequencies))
        self.assertEqual(weights.shape, (2, 2))
        model = build_model("timesnet")
        self.assertEqual(model.in_dim, 2)
        self.assertEqual(model.top_k, 2)
        self.assertTrue(any(isinstance(layer, nn.Conv2d) for layer in model.modules()))


class WorkflowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loader_physical_lambda_raw_truth_and_order(self):
        data = synthetic()
        x, y = data[0]
        self.assertTrue(np.array_equal(data.q[:, 0], [2.7, 1.8, 2.3]))
        self.assertTrue(np.array_equal(x[:, 1].numpy(), data.lambda_grid.astype(np.float32)))
        self.assertEqual(float(data.lambda_grid[-1]), 5.995)
        self.assertTrue(torch.equal(x[:, 0], torch.full((9,), np.float32(2.7))))
        self.assertEqual(data.xyz.dtype, np.float64)
        self.assertTrue(np.array_equal(y.numpy(), data.xyz[0].astype(np.float32)))

    def test_train_only_stats_and_restoration(self):
        train, val = synthetic(), synthetic(offset=100)
        stats = fit_normalization(train)
        before = stats.to_dict()
        val.stats = stats
        x, y = val[0]
        self.assertEqual(stats.to_dict(), before)
        self.assertGreater(float(x[:, 0].mean()), 100)
        self.assertTrue(np.allclose(stats.x_mean, train.raw_input().mean(axis=(0, 1))))
        restored = restore_normalization(json.loads(json.dumps(before)))
        self.assertEqual(restored.to_dict(), before)
        self.assertTrue(np.allclose(recover_xyz(y.numpy()[None], restored)[0], val.xyz[0], atol=1e-5))
        self.assertNotEqual(fit_normalization(synthetic(t=13)).x_std[1], stats.x_std[1])

    def test_checkpoint_restore_prediction_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="track_a_test_") as tmp:
            root = Path(tmp)
            full, resumed = root / "full", root / "resumed"
            full.mkdir()
            resumed.mkdir()
            data = {"train": synthetic(), "val": synthetic(n=2, offset=0.1)}
            stats = fit_normalization(data["train"])
            meta = metadata("transformer")
            seed_training()
            model = build_model("transformer")
            complete = fit_epochs(model, data, stats, "cpu", full, meta, epochs=2)
            seed_training()
            model2 = build_model("transformer")
            fit_epochs(model2, data, stats, "cpu", resumed, meta, epochs=1)
            state = torch.load(resumed / "epoch_0001.pt", weights_only=True)
            loaded, restored = checkpoint_contract(state)
            self.assertEqual(restored.to_dict(), stats.to_dict())
            continuing = fit_epochs(loaded, data, restored, "cpu", resumed, meta, epochs=2, resume=state)
            self.assertEqual(complete["history"], continuing["history"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
            self.assertAlmostEqual(complete["history"][1]["learning_rate"], 1e-3 * 0.995)
            self.assertEqual(complete["best_epoch"], min(complete["history"], key=lambda x: x["validation_normalized_mse"])["epoch"])
            prediction, timing = predict_frozen(loaded, synthetic(), restored, "cpu")
            self.assertEqual(prediction.shape, (3, 9, 3))
            self.assertGreater(timing["inference_wall_seconds"], 0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(resumed / "epoch_0001.pt", state)
        self.assertFalse(root.exists())

    def test_all_models_one_synthetic_epoch_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="track_a_training_test_") as tmp:
            root = Path(tmp)
            for name in MODEL_CONFIGS:
                with self.subTest(model=name):
                    directory = root / name
                    directory.mkdir()
                    data = {"train": synthetic(), "val": synthetic(n=2, offset=0.1)}
                    stats = fit_normalization(data["train"])
                    model = build_model(name)
                    result = fit_epochs(model, data, stats, "cpu", directory, metadata(name), epochs=1)
                    self.assertEqual(result["completed_epochs"], 1)
                    state = torch.load(directory / "epoch_0001.pt", weights_only=True)
                    restored_model, restored_stats = checkpoint_contract(state)
                    predicted, _ = predict_frozen(model, synthetic(), stats, "cpu")
                    restored_prediction, _ = predict_frozen(restored_model, synthetic(), restored_stats, "cpu")
                    self.assertTrue(np.array_equal(predicted, restored_prediction))
        self.assertFalse(root.exists())

    def test_best_weights_survive_a_worse_last_epoch(self):
        with tempfile.TemporaryDirectory(prefix="track_a_selection_test_") as tmp:
            data = {"train": synthetic(), "val": synthetic(n=2)}
            model = build_model("bilstm")
            calls = iter((1.0, 0.2, 0.8, 0.4))
            def epoch_stub(model, loader, device, optimizer=None):
                if optimizer is not None:
                    with torch.no_grad():
                        next(model.parameters()).add_(0.01)
                    optimizer.zero_grad()
                    optimizer.step()
                return next(calls)
            with patch("src.training.benchmark_track_a.epoch_mse", side_effect=epoch_stub):
                result = fit_epochs(model, data, fit_normalization(data["train"]), "cpu",
                                    Path(tmp), metadata("bilstm"), epochs=2)
            self.assertEqual(result["best_epoch"], 1)
            state = torch.load(Path(tmp) / "epoch_0002.pt", weights_only=True)
            key = next(iter(state["model_state_dict"]))
            self.assertFalse(torch.equal(state["best_model_state_dict"][key], state["model_state_dict"][key]))

    def test_resolution_matrix_and_native_robustness(self):
        matrix = planned_matrix()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(sum(len(row["evaluation_resolutions"]) for row in matrix), 48)
        self.assertEqual(resolution_routing(2399), {1200: "reverse", 2399: "native", 3598: "cross", 4797: "cross"})
        metrics = {t: {"global_relative_l2": float(i + 1), "mean_per_q_relative_l2": float(i + 2)}
                   for i, t in enumerate(EVAL_TASKS)}
        result = robustness(metrics, 2399)
        self.assertEqual(result[2399]["global_relative_l2"]["ratio_to_native"], 1)
        self.assertEqual(result[4797]["global_relative_l2"]["ratio_to_native"], 2)
        self.assertEqual(result[1200]["global_relative_l2"]["delta_from_native"], -1)
        with self.assertRaises(ValueError):
            resolution_routing(3598)

    def test_pair_rejects_split_row_permutation(self):
        arrays = {f"x_{split}": synthetic().q for split in ("train", "val", "test")}
        original = SimpleNamespace(arrays=arrays)
        altered = dict(arrays)
        altered["x_val"] = arrays["x_val"][[1, 0, 2]]
        with self.assertRaisesRegex(ValueError, "Q split identity"):
            validate_pair(original, SimpleNamespace(arrays=altered))

    def test_frozen_evaluator_routes_all_resolutions_without_refit(self):
        # 合成 checkpoint 的完成标记仅用于隔离测试，绝不登记为正式实验。
        with tempfile.TemporaryDirectory(prefix="track_a_evaluator_test_") as tmp:
            root = Path(tmp)
            checkpoint = root / "best_model.pt"
            stats = fit_normalization(synthetic())
            payload = {**metadata("fno1d", 2399), "normalization": stats.to_dict(),
                       "model_state_dict": build_model("fno1d").state_dict(), "epoch": 1,
                       "completed_epochs": 500, "source_provenance": {"source_sha256": {"synthetic": "test"}}}
            save_checkpoint(checkpoint, payload)
            (root / "summary.json").write_text(json.dumps({"completed_epochs": 500,
                "best_checkpoint": checkpoint.name, "best_epoch": 1,
                "best_checkpoint_sha256": sha256_file(checkpoint)}), encoding="utf-8")
            datasets = {t: synthetic(t=9 + i * 2) for i, t in enumerate(EVAL_TASKS)}
            for data in datasets.values():
                order = np.argsort(data.q[:, 0])
                data.q, data.xyz = data.q[order], data.xyz[order]
            with patch("src.training.benchmark_track_a.output_path", side_effect=Path), \
                 patch("src.training.benchmark_track_a.source_provenance", return_value={"source_sha256": {"synthetic": "test"}}), \
                 patch("src.training.benchmark_track_a.load_training", return_value=({}, {"synthetic": True})), \
                 patch("src.training.benchmark_track_a.load_evaluation", side_effect=lambda t: (datasets[t], {"synthetic": True})) as routed, \
                 patch("src.training.benchmark_track_a.fit_normalization", side_effect=AssertionError("No refit allowed")):
                result = evaluate_run(checkpoint, root / "evaluation", approved=True)
            self.assertEqual([call.args[0] for call in routed.call_args_list], list(EVAL_TASKS))
            self.assertEqual(result["metrics"][2399]["route"], "native")
            self.assertEqual(result["resolution_robustness"][2399]["global_relative_l2"]["ratio_to_native"], 1)
            self.assertEqual(result["normalization"], stats.to_dict())
            for t in EVAL_TASKS:
                with np.load(root / "evaluation" / f"prediction_t{t}.npz") as saved:
                    self.assertTrue(np.array_equal(saved["Q"], datasets[t].q))
                    self.assertTrue(np.array_equal(saved["lambda_grid"], datasets[t].lambda_grid))
                self.assertIn("p99", result["metrics"][t]["per_q_relative_l2_summary"])
        self.assertFalse(root.exists())

    def test_no_implicit_formal_authorization_or_path_escape(self):
        with self.assertRaises(PermissionError):
            train_run("fno1d", 1200, "unused")
        with self.assertRaises(PermissionError):
            evaluate_run("unused", "unused")
        with self.assertRaises(ValueError):
            output_path("data/test")
        with self.assertRaises(ValueError):
            checkpoint_contract({"model_state_dict": {}})
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaises(ValueError):
                inspect_device("cuda:0", 0)


if __name__ == "__main__":
    unittest.main()
