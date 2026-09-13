"""DeepONet preparation checks using CPU synthetic data, never formal runs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.models.benchmark_track_a import build_model, parameter_counts
from src.training.benchmark_data import EVAL_TASKS, fit_normalization, sha256_file
from src.training.benchmark_track_a import (
    checkpoint_contract, evaluate_run, fit_epochs, predict_frozen,
    save_checkpoint, seed_training,
)
from tests.test_benchmark_track_a import metadata, synthetic


class DeepONetPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_capacity_and_grouped_operator_construction(self):
        model = build_model("deeponet").eval()
        self.assertEqual(parameter_counts(model.branch)["real_scalar_parameter_count"], 592128)
        self.assertEqual(parameter_counts(model.trunk)["real_scalar_parameter_count"], 493568)
        self.assertEqual(model.bias.numel(), 3)
        self.assertEqual(parameter_counts(model), {
            "real_scalar_parameter_count": 1085699, "tensor_numel": 1085699,
            "trainable_parameter_bytes": 4342796})
        for net, output in ((model.branch, 384), (model.trunk, 128)):
            linear = [layer for layer in net if isinstance(layer, nn.Linear)]
            self.assertEqual([(layer.in_features, layer.out_features) for layer in linear],
                             [(1, 384)] + [(384, 384)] * 3 + [(384, output)])
            self.assertEqual(sum(isinstance(layer, nn.Tanh) for layer in net), 4)
            self.assertIsInstance(net[-1], nn.Linear)
        x = torch.from_numpy(synthetic(n=2).raw_input())
        with torch.no_grad():
            branch = model.branch(x[:, 0, :1]).reshape(2, 3, 128)
            trunk = model.trunk(x[:, :, 1:2])
            expected = (branch[:, None] * trunk[:, :, None]).sum(-1) + model.bias
            torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-7)

    def test_full_resolutions_and_independent_nonuniform_queries(self):
        seed_training()
        model = build_model("deeponet").eval()
        stats = fit_normalization(synthetic())
        with torch.no_grad():
            for t in (1, 17, *EVAL_TASKS):
                with self.subTest(T=t):
                    data = synthetic(n=1, t=t)
                    data.stats = stats
                    output = model(data[0][0][None])
                    self.assertEqual(output.shape, (1, t, 3))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(output).all())
            data = synthetic(n=2, t=13)
            data.lambda_grid = np.linspace(0, 1, 13) ** 2 * 5.995
            data.stats = stats
            x = torch.stack([data[i][0] for i in range(2)])
            full = model(x)
            order = torch.tensor([12, 0, 4, 4, 7])
            torch.testing.assert_close(model(x[:, order]), full[:, order], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(model(x[:1]), full[:1], rtol=1e-5, atol=1e-7)
            changed = x.clone()
            changed[:, 5, 1] += 0.2
            unchanged = torch.arange(13) != 5
            torch.testing.assert_close(model(changed)[:, unchanged], full[:, unchanged])
            self.assertFalse(torch.allclose(model(changed)[:, 5], full[:, 5]))
            changed = x.clone()
            changed[:, :, 0] += 0.3
            self.assertFalse(torch.allclose(model(changed), full))

    def test_synthetic_resume_and_frozen_evaluator_for_both_native_routes(self):
        with tempfile.TemporaryDirectory(prefix="deeponet_track_a_test_") as tmp:
            root = Path(tmp)
            for native in (1200, 2399):
                with self.subTest(native=native):
                    full, resumed = root / f"full_{native}", root / f"resume_{native}"
                    full.mkdir()
                    resumed.mkdir()
                    data = {"train": synthetic(), "val": synthetic(n=2, offset=0.1)}
                    stats = fit_normalization(data["train"])
                    stored = stats.to_dict()
                    meta = metadata("deeponet", native)
                    seed_training()
                    model = build_model("deeponet")
                    complete = fit_epochs(model, data, stats, "cpu", full, meta, epochs=2)
                    seed_training()
                    interrupted = build_model("deeponet")
                    fit_epochs(interrupted, data, stats, "cpu", resumed, meta, epochs=1)
                    state = torch.load(resumed / "epoch_0001.pt", weights_only=True)
                    loaded, restored = checkpoint_contract(state)
                    self.assertEqual(restored.to_dict(), stored)
                    before, _ = predict_frozen(interrupted, synthetic(), stats, "cpu")
                    after, _ = predict_frozen(loaded, synthetic(), restored, "cpu")
                    np.testing.assert_array_equal(before, after)
                    continued = fit_epochs(loaded, data, restored, "cpu", resumed, meta,
                                           epochs=2, resume=state)
                    self.assertEqual(complete["history"], continued["history"])
                    self.assertEqual([row["epoch"] for row in continued["history"]], [1, 2])
                    self.assertEqual(continued["history"][1]["learning_rate"], 1e-3 * 0.995)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
                    final = torch.load(resumed / "epoch_0002.pt", weights_only=True)
                    reference = torch.load(full / "epoch_0002.pt", weights_only=True)
                    self.assertEqual(final["scheduler_state_dict"], reference["scheduler_state_dict"])
                    self.assertEqual(final["optimizer_state_dict"]["param_groups"],
                                     reference["optimizer_state_dict"]["param_groups"])
                    for key, values in final["optimizer_state_dict"]["state"].items():
                        for field, value in values.items():
                            self.assertTrue(torch.equal(value, reference["optimizer_state_dict"]["state"][key][field]))
                    self.assertTrue(torch.equal(final["rng"]["torch"], reference["rng"]["torch"]))
                    self.assertEqual(final["normalization"], stored)
                    self.assertEqual(final["best_epoch"], complete["best_epoch"])
                    for key, value in final["best_model_state_dict"].items():
                        self.assertTrue(torch.equal(value, reference["best_model_state_dict"][key]))

                    # 完成标记仅为隔离 evaluator 门禁的合成夹具，不代表跑过正式 500 轮。
                    source = {"source_sha256": {"synthetic": "test"}}
                    checkpoint = resumed / "best_model.pt"
                    payload = {**meta, "normalization": stored, "source_provenance": source,
                               "model_state_dict": final["best_model_state_dict"],
                               "epoch": complete["best_epoch"], "completed_epochs": 500}
                    save_checkpoint(checkpoint, payload)
                    (resumed / "summary.json").write_text(json.dumps({"completed_epochs": 500,
                        "best_checkpoint": checkpoint.name, "best_epoch": complete["best_epoch"],
                        "best_checkpoint_sha256": sha256_file(checkpoint)}), encoding="utf-8")
                    datasets = {t: synthetic(t=9 + i * 2) for i, t in enumerate(EVAL_TASKS)}
                    for dataset in datasets.values():
                        order = np.argsort(dataset.q[:, 0])
                        dataset.q, dataset.xyz = dataset.q[order], dataset.xyz[order]
                    with patch("src.training.benchmark_track_a.output_path", side_effect=Path), \
                         patch("src.training.benchmark_track_a.source_provenance", return_value=source), \
                         patch("src.training.benchmark_track_a.load_training", return_value=({}, {"synthetic": True})), \
                         patch("src.training.benchmark_track_a.load_evaluation", side_effect=lambda t: (datasets[t], {"synthetic": True})) as routed, \
                         patch("src.training.benchmark_track_a.fit_normalization", side_effect=AssertionError("No evaluation refit allowed")):
                        result = evaluate_run(checkpoint, resumed / "evaluation", approved=True)
                    self.assertEqual([call.args[0] for call in routed.call_args_list], list(EVAL_TASKS))
                    self.assertEqual(result["normalization"], stored)
                    self.assertEqual(result["resolution_robustness"][native]["global_relative_l2"]["ratio_to_native"], 1)
                    for t, metric in result["metrics"].items():
                        self.assertEqual(metric["route"], "native" if t == native else "reverse" if t < native else "cross")
                        self.assertTrue(np.isfinite(metric["global_relative_l2"]))
                        with np.load(resumed / "evaluation" / f"prediction_t{t}.npz") as saved:
                            self.assertEqual(saved["prediction_xyz"].shape, datasets[t].xyz.shape)
                            np.testing.assert_array_equal(saved["Q"], datasets[t].q)
                            np.testing.assert_array_equal(saved["lambda_grid"], datasets[t].lambda_grid)
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
