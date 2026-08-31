"""Plan B matched T2399 reverse-workflow synthetic tests."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.data_generation.plan_b_matched_training import (
    HISTORICAL_T1200_TRAINING_CONTRACT,
    raw_float64_q_sha256,
    replay_matched_t2399_training_dataset,
    source_split_q_hashes,
    validate_historical_t1200_source,
)
from src.data_generation.plan_b_paired import COARSE_N_STEPS, COARSE_STEP_SIZE, FINE_N_STEPS, FINE_STEP_SIZE, build_lambda_grid
from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_plan_b_bidirectional_reverse_2d.py"
SPEC = importlib.util.spec_from_file_location("plan_b_bidirectional", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


FIXED_PARAMS = {
    "M": 1.0,
    "a": 0.5,
    "E": 0.95,
    "Lz": 3.0,
    "r0": 10.0,
    "theta0": 1.2,
    "phi0": 0.0,
    "sign_r": -1,
    "sign_th": 1,
}


def _split_q() -> dict[str, np.ndarray]:
    return {
        "train": np.array([[1.7], [1.9]], dtype=np.float64),
        "val": np.array([[2.1]], dtype=np.float64),
        "test": np.array([[2.3]], dtype=np.float64),
    }


def _trajectory(q: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = q[:, 0, None]
    lam = grid[None, :]
    return np.stack((values + lam, values * 2.0 - lam, 1.0 + values * lam), axis=2).astype(np.float64)


def _contract(q: dict[str, np.ndarray]):
    return replace(
        HISTORICAL_T1200_TRAINING_CONTRACT,
        source_task_name="synthetic_q_1p6-3_n2000_t1200",
        split_counts={name: int(values.shape[0]) for name, values in q.items()},
        split_q_sha256={name: raw_float64_q_sha256(values) for name, values in q.items()},
        original_candidate_count=sum(values.shape[0] for values in q.values()),
    )


def _metadata(contract, *, n_steps: int, step_size: float) -> dict[str, object]:
    return {
        "task_name": contract.source_task_name,
        "task_spec": {
            "vary_params": ["Q"],
            "vary_ranges": {"Q": [1.6, 3.0]},
            "sample_shape": [contract.original_candidate_count],
            "fixed_params": FIXED_PARAMS,
            "n_steps": n_steps,
            "step_size": step_size,
            "seed": 10,
            "sampling_mode": "grid",
            "metadata": {"orbit_solver": "second_order_rk4", "orbit_solver_version": "v1"},
        },
        "generation_status": {
            "success_count": contract.original_candidate_count,
            "initial_candidate_count": contract.original_candidate_count,
            "failure_count": 0,
            "used_completion_sampling": False,
            "successful_points_strictly_uniform": True,
        },
    }


def _write_source(directory: Path, contract, q: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True)
    grid = build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
    arrays: dict[str, np.ndarray] = {"vary_params_order": np.array(["Q"]), "lambda_grid": grid}
    for split, values in q.items():
        arrays[f"x_{split}"] = values
        arrays[f"y_{split}"] = _trajectory(values, grid)
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(json.dumps(_metadata(contract, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)), encoding="utf-8", newline="\n")
    (directory / "failed_samples.json").write_text("[]\n", encoding="utf-8", newline="\n")


def _simulator(*, Q: float, n_steps: int, step_size: float, **_: object) -> dict[str, object]:
    grid = build_lambda_grid(n_steps, step_size)
    values = np.asarray([[Q]], dtype=np.float64)
    return {"lambda_grid": grid, "xyz": _trajectory(values, grid)[0]}


class PlanBMatchedReplayTests(unittest.TestCase):
    def test_source_hashes_and_split_order_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = _split_q()
            contract = _contract(q)
            source = root / "source"
            _write_source(source, contract, q)
            artifact, evidence = validate_historical_t1200_source(source, contract=contract)
            self.assertTrue(evidence["all_reference_hashes_match"])
            self.assertEqual(source_split_q_hashes(artifact), contract.split_q_sha256)
            with np.load(source / "dataset.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["x_train"] = arrays["x_train"][::-1].copy()
            np.savez_compressed(source / "dataset.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "Q SHA256"):
                validate_historical_t1200_source(source, contract=contract)

    def test_replay_preserves_identity_and_never_replaces_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = _split_q()
            contract = _contract(q)
            source = root / "source"
            fine = root / "fine"
            _write_source(source, contract, q)
            result = replay_matched_t2399_training_dataset(source, fine, simulator=_simulator, progress_every=100, contract=contract)
            self.assertTrue(result["matched_dataset_complete"])
            self.assertTrue(result["training_permitted"])
            self.assertTrue(result["fine_common_nodes_equal_source_grid"])
            self.assertEqual(result["generated_split_q_sha256"], contract.split_q_sha256)
            with np.load(source / "dataset.npz", allow_pickle=False) as coarse, np.load(fine / "dataset.npz", allow_pickle=False) as generated:
                self.assertTrue(np.array_equal(generated["lambda_grid"][::2], coarse["lambda_grid"]))
                for split in ("train", "val", "test"):
                    self.assertTrue(np.array_equal(generated[f"x_{split}"], coarse[f"x_{split}"]))


class PlanBTrainingContractTests(unittest.TestCase):
    def test_historical_training_values_are_explicit(self) -> None:
        contract = HISTORICAL_T1200_TRAINING_CONTRACT
        self.assertEqual(contract.model_config["modes1"], 16)
        self.assertEqual(contract.model_config["modes2"], 32)
        self.assertEqual(contract.model_config["width"], 64)
        self.assertEqual(contract.model_config["depth"], 4)
        self.assertEqual(contract.training_config, {
            "epochs": 500, "batch_size": 1, "lr": 0.001, "weight_decay": 0.0001,
            "scheduler_gamma": 0.995, "training_seed": 27, "normalization": "standard",
            "target_transform": "raw", "lambda_reference_index": 0,
        })
        self.assertIn("best_model.pt", contract.historical_checkpoint_path)

    def test_t2399_stats_are_newly_fitted_from_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = _split_q()
            contract = _contract(q)
            source = root / "source"; fine = root / "fine"
            _write_source(source, contract, q)
            replay_matched_t2399_training_dataset(source, fine, simulator=_simulator, progress_every=100, contract=contract)
            fields, stats = workflow.build_matched_training_fields(fine)
            expected = np.mean(fields["train"].x_2d, axis=(0, 1, 2))
            np.testing.assert_allclose(np.asarray(stats.x_mean), expected.astype(np.float32))
            self.assertEqual(stats.method, "standard")
            self.assertNotEqual(stats.to_dict()["x_mean"], [2.2, 3.0])

    def test_tiny_training_smoke_writes_best_last_history_and_cleanup_is_external(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = _split_q()
            contract = _contract(q)
            source = root / "source"; fine = root / "fine"; output = root / "run"
            _write_source(source, contract, q)
            replay = replay_matched_t2399_training_dataset(source, fine, simulator=_simulator, progress_every=100, contract=contract)
            tiny = workflow.TrainingRunSpec(
                model_config={"model_type": "fno2d", "in_dim": 2, "out_dim": 3, "modes1": 1, "modes2": 2, "width": 4, "depth": 1, "hidden_dim": 8, "activation": "gelu"},
                training_config={"epochs": 1, "batch_size": 1, "lr": 0.001, "weight_decay": 0.0001, "scheduler_gamma": 0.995, "training_seed": 27, "normalization": "standard", "target_transform": "raw", "lambda_reference_index": 0},
            )
            summary = workflow.train_matched_t2399_model(dataset_dir=fine, output_dir=output, device="cpu", source_validation=replay["source_validation"], spec=tiny)
            self.assertEqual(summary["normalization_fit_split"], "train")
            self.assertFalse(summary["normalization_reused_from_t1200"])
            self.assertTrue(Path(summary["best_checkpoint"]).is_file())
            self.assertTrue(Path(summary["last_checkpoint"]).is_file())
            self.assertTrue(Path(summary["history_path"]).is_file())


class PlanBEvaluationAndMatrixTests(unittest.TestCase):
    def _checkpoint(self, path: Path) -> None:
        stats = FieldNormalizationStats(method="standard", x_mean=[2.0, 3.0], x_std=[1.0, 2.0], y_mean=[1.0, 2.0, 3.0], y_std=[2.0, 3.0, 4.0])
        config = {
            "experiment_type": workflow.EXPERIMENT_TYPE,
            "model_config": workflow._training_spec_from_contract().model_config,
            "dataset_summary": {"train": {"num_lambda": FINE_N_STEPS}, "normalization_stats": stats.to_dict(), "target_transform_config": TargetTransformConfig(mode="raw").to_dict()},
            "plan_b_reverse_contract": {"normalization_reused_from_t1200": False},
        }
        torch.save({"epoch": 1, "config": config, "model_state_dict": {}}, path)

    def test_checkpoint_contract_uses_frozen_t2399_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            self._checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            frozen = workflow.validate_t2399_checkpoint_contract(checkpoint)
            self.assertEqual(frozen.provenance["normalization_fit_split"], "T2399_train")
            self.assertFalse(frozen.provenance["normalization_refit_during_evaluation"])

    def test_matrix_reuses_completed_theta1200_metrics_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_path = root / "theta1200.json"
            metrics_path.write_text(json.dumps({"coarse_t1200": {"global_relative_l2": 0.1}, "fine_t2399_full_grid_primary": {"global_relative_l2": 0.2}}), encoding="utf-8", newline="\n")
            theta1200 = workflow.load_completed_theta1200_metrics(metrics_path)
            theta2399 = {"metrics": {"theta2399_reverse_t1200": {"global_relative_l2": 0.3}, "theta2399_native_t2399": {"global_relative_l2": 0.25}}}
            matrix = workflow.assemble_bidirectional_matrix(theta1200, theta2399)
            self.assertTrue(theta1200["reuse_completed_theta1200_inference"])
            self.assertFalse(matrix["theta1200_inference_rerun"])
            self.assertEqual(matrix["training_rows"]["theta1200"]["test_t2399"]["global_relative_l2"], 0.2)

    def test_native_and_reverse_forward_share_one_frozen_stats_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "checkpoint.pt"
            self._checkpoint(checkpoint_path)
            coarse = SimpleNamespace(
                canonical_q=np.array([1.7], dtype=np.float64),
                canonical_truth=np.ones((1, COARSE_N_STEPS, 3), dtype=np.float32),
            )
            fine = SimpleNamespace(
                canonical_q=np.array([1.7], dtype=np.float64),
                canonical_truth=np.ones((1, FINE_N_STEPS, 3), dtype=np.float32),
             )
            qualification = {"structural_valid": True, "numerical_metrics": {"available": True}}
            received = []

            def fake_forward(*, contract, field, **_):
                received.append(contract.normalization_stats)
                return np.zeros_like(field.canonical_truth)

            with patch.object(workflow, "validate_plan_b_inputs", return_value=(qualification, coarse, fine)), patch.object(workflow, "load_fno2d_checkpoint_model", return_value=object()), patch.object(workflow, "run_frozen_forward", side_effect=fake_forward):
                result, _, _ = workflow.evaluate_theta2399_on_q400(
                    checkpoint_path=checkpoint_path,
                    q400_t1200_dataset_dir="unused_coarse",
                    q400_t2399_dataset_dir="unused_fine",
                    device="cpu",
                )
            self.assertEqual(len(received), 2)
            self.assertIs(received[0], received[1])
            self.assertFalse(result["frozen_inference"]["normalization_refit"])


class PlanBInterfaceTests(unittest.TestCase):
    def test_cli_and_scope(self) -> None:
        args = workflow.parse_args([
            "--source-t1200-dataset-dir", "source", "--matched-t2399-dataset-dir", "fine",
            "--q400-t1200-dataset-dir", "q400c", "--q400-t2399-dataset-dir", "q400f",
            "--theta1200-metrics-json", "metrics.json", "--output-dir", "output",
        ])
        self.assertEqual(args.device, "cuda")
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertIn("load_completed_theta1200_metrics", SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("evaluate_plan_b(", SCRIPT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
