"""Plan B frozen FNO2D resolution evaluator 的合成测试。"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    build_lambda_grid,
)
from src.models.registry_2d import build_model_2d
from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_plan_b_resolution_generalization_2d.py"
SPEC = importlib.util.spec_from_file_location("plan_b_resolution_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


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


def _q_splits() -> dict[str, np.ndarray]:
    return {
        "train": np.array([[2.1], [1.7]], dtype=np.float64),
        "val": np.array([[2.4]], dtype=np.float64),
        "test": np.array([[2.7]], dtype=np.float64),
    }


def _trajectory(q_values: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    q = q_values[:, 0, None]
    lam = lambda_grid[None, :]
    return np.stack((q + lam, 2.0 * q - lam, 1.0 + q * lam), axis=2).astype(np.float64)


def _metadata(n_steps: int, step_size: float, *, paired: bool) -> dict[str, object]:
    metadata: dict[str, object] = {
        "task_name": "synthetic_plan_b_q400_field",
        "task_spec": {
            "vary_params": ["Q"],
            "fixed_params": FIXED_PARAMS,
            "n_steps": n_steps,
            "step_size": step_size,
            "metadata": {
                "orbit_solver": "second_order_rk4",
                "orbit_solver_version": "v1",
            },
        },
        "integration": {
            "n_steps": n_steps,
            "step_size": step_size,
            "lambda_min": 0.0,
            "lambda_max": (n_steps - 1) * step_size,
        },
    }
    if paired:
        metadata["plan_b_protocol_version"] = "v1"
        metadata["paired_replay"] = {
            "paired_completeness": True,
            "replacement_policy": "forbidden",
        }
    return metadata


def _write_dataset(
    directory: Path,
    *,
    n_steps: int,
    step_size: float,
    q_splits: dict[str, np.ndarray] | None = None,
    paired: bool,
) -> None:
    directory.mkdir(parents=True)
    q_splits = q_splits or _q_splits()
    lambda_grid = build_lambda_grid(n_steps, step_size)
    arrays: dict[str, np.ndarray] = {
        "vary_params_order": np.array(["Q"]),
        "lambda_grid": lambda_grid,
    }
    for split, q_values in q_splits.items():
        arrays[f"x_{split}"] = q_values.copy()
        arrays[f"y_{split}"] = _trajectory(q_values, lambda_grid)
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps(_metadata(n_steps, step_size, paired=paired), indent=2),
        encoding="utf-8",
        newline="\n",
    )
    (directory / "failed_samples.json").write_text("[]\n", encoding="utf-8", newline="\n")


def _model_config() -> dict[str, object]:
    return {
        "model_type": "fno2d",
        "in_dim": 2,
        "out_dim": 3,
        "modes1": 1,
        "modes2": 2,
        "width": 4,
        "depth": 1,
        "hidden_dim": 8,
        "activation": "gelu",
    }


def _stats() -> FieldNormalizationStats:
    return FieldNormalizationStats(
        method="standard",
        x_mean=[2.2, 3.0],
        x_std=[0.5, 1.5],
        y_mean=[1.0, 2.0, 3.0],
        y_std=[2.0, 2.5, 3.0],
    )


def _checkpoint(path: Path) -> dict[str, object]:
    torch.manual_seed(7)
    model = build_model_2d(**_model_config())
    checkpoint: dict[str, object] = {
        "epoch": 4,
        "model_state_dict": model.state_dict(),
        "config": {
            "model_type": "fno2d",
            "normalization": "standard",
            "target_transform": "raw",
            "lambda_reference_index": 0,
            "model_config": _model_config(),
            "dataset_summary": {
                "param_name": "Q",
                "vary_params_order": ["Q"],
                "normalization_stats": _stats().to_dict(),
                "target_transform_config": TargetTransformConfig(mode="raw").to_dict(),
                "train": {"num_lambda": COARSE_N_STEPS},
            },
        },
    }
    torch.save(checkpoint, path)
    return checkpoint


def _paired_directories(root: Path) -> tuple[Path, Path]:
    coarse = root / "coarse"
    fine = root / "fine"
    _write_dataset(coarse, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE, paired=False)
    _write_dataset(fine, n_steps=FINE_N_STEPS, step_size=FINE_STEP_SIZE, paired=True)
    return coarse, fine


class PlanBMetricTests(unittest.TestCase):
    def test_full_fine_metric_correctness(self) -> None:
        truth = np.array([[[1.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])
        prediction = np.array([[[2.0, 0.0, 0.0]], [[12.0, 0.0, 0.0]]])
        metrics = evaluator.compute_raw_metrics(prediction, truth, np.array([1.7, 2.1]))
        self.assertAlmostEqual(metrics["global_mse"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["global_relative_l2"], np.sqrt(5.0 / 101.0))
        self.assertAlmostEqual(metrics["mean_per_q_relative_l2"], 0.6)
        self.assertAlmostEqual(metrics["per_q_relative_l2_summary"]["p99"], 0.992)

    def test_common_node_metric_correctness(self) -> None:
        truth_fine = np.ones((1, 5, 3), dtype=np.float64)
        prediction_fine = truth_fine.copy()
        prediction_fine[0, 2, 0] += 3.0
        metrics = evaluator.compute_raw_metrics(
            prediction_fine[:, ::2, :],
            truth_fine[:, ::2, :],
            np.array([1.7]),
        )
        self.assertEqual(metrics["point_count_per_q"], 3)
        self.assertAlmostEqual(metrics["global_mse"], 1.0)
        self.assertAlmostEqual(metrics["global_relative_l2"], 1.0)

    def test_generalization_ratio_and_gap(self) -> None:
        coarse = {"global_relative_l2": 0.2, "mean_per_q_relative_l2": 0.4}
        fine = {"global_relative_l2": 0.5, "mean_per_q_relative_l2": 0.6}
        gap = evaluator.compute_generalization_gap(fine, coarse)
        self.assertAlmostEqual(gap["global_relative_l2"]["fine_minus_coarse"], 0.3)
        self.assertAlmostEqual(gap["global_relative_l2"]["fine_divided_by_coarse"], 2.5)
        self.assertAlmostEqual(gap["mean_per_q_relative_l2"]["fine_divided_by_coarse"], 1.5)


class PlanBProtocolTests(unittest.TestCase):
    def test_q_mismatch_is_rejected_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse, fine = _paired_directories(root)
            with np.load(fine / "dataset.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["x_train"] = arrays["x_train"][::-1].copy()
            np.savez_compressed(fine / "dataset.npz", **arrays)
            with patch.object(evaluator, "load_checkpoint_2d", side_effect=AssertionError("checkpoint load forbidden")):
                with self.assertRaisesRegex(ValueError, "structural validation failed"):
                    evaluator.evaluate_plan_b(
                        checkpoint_path=root / "checkpoint.pt",
                        coarse_dataset_dir=coarse,
                        fine_dataset_dir=fine,
                        device="cpu",
                    )

    def test_t2400_is_rejected_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse = root / "coarse"
            invalid_fine = root / "invalid_fine"
            _write_dataset(coarse, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE, paired=False)
            _write_dataset(invalid_fine, n_steps=2400, step_size=FINE_STEP_SIZE, paired=True)
            with patch.object(evaluator, "load_checkpoint_2d", side_effect=AssertionError("checkpoint load forbidden")):
                with self.assertRaisesRegex(ValueError, "structural validation failed"):
                    evaluator.evaluate_plan_b(
                        checkpoint_path=root / "checkpoint.pt",
                        coarse_dataset_dir=coarse,
                        fine_dataset_dir=invalid_fine,
                        device="cpu",
                    )

    def test_frozen_normalization_comes_only_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            checkpoint = _checkpoint(path)
            contract = evaluator.validate_checkpoint_contract(checkpoint)
            self.assertEqual(contract.normalization_stats.to_dict(), _stats().to_dict())
            self.assertFalse(contract.provenance["normalization_refit_during_evaluation"])
            self.assertNotIn("compute_field_normalization_stats", SCRIPT_PATH.read_text(encoding="utf-8"))

    def test_variable_t_fno_forward_accepts_t1200_and_t2399(self) -> None:
        model = build_model_2d(**_model_config()).eval()
        with torch.no_grad():
            coarse_output = model(torch.zeros((1, 2, COARSE_N_STEPS, 2), dtype=torch.float32))
            fine_output = model(torch.zeros((1, 2, FINE_N_STEPS, 2), dtype=torch.float32))
        self.assertEqual(tuple(coarse_output.shape), (1, 2, COARSE_N_STEPS, 3))
        self.assertEqual(tuple(fine_output.shape), (1, 2, FINE_N_STEPS, 3))


class PlanBSmokeTests(unittest.TestCase):
    def test_tiny_frozen_evaluation_smoke_and_output_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse, fine = _paired_directories(root)
            checkpoint_path = root / "checkpoint.pt"
            _checkpoint(checkpoint_path)
            result, coarse_prediction, fine_prediction = evaluator.evaluate_plan_b(
                checkpoint_path=checkpoint_path,
                coarse_dataset_dir=coarse,
                fine_dataset_dir=fine,
                device="cpu",
            )
            self.assertTrue(result["ground_truth_qualification"]["structural_valid"])
            self.assertEqual(coarse_prediction.shape, (4, COARSE_N_STEPS, 3))
            self.assertEqual(fine_prediction.shape, (4, FINE_N_STEPS, 3))
            output_dir = root / "output"
            evaluator.write_evaluation_outputs(
                output_dir=output_dir,
                result=result,
                coarse_prediction=coarse_prediction,
                fine_prediction=fine_prediction,
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "metrics.json",
                    "experiment_summary.json",
                    "coarse_prediction_raw_xyz.npy",
                    "fine_prediction_raw_xyz.npy",
                },
            )


class PlanBInterfaceTests(unittest.TestCase):
    def test_cli_and_source_scope(self) -> None:
        args = evaluator.parse_args(
            [
                "--checkpoint", "checkpoint.pt",
                "--coarse-dataset-dir", "coarse",
                "--fine-dataset-dir", "fine",
                "--output-dir", "output",
            ]
        )
        self.assertEqual(args.device, "cuda")
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for prohibited in ("optimizer", ".backward(", "scheduler", "torch.save(", "compute_field_normalization_stats"):
            self.assertNotIn(prohibited, source)
        self.assertIn("validate_plan_b_ground_truth", source)
        self.assertIn("predict_2d_loader", source)
        self.assertIn("truth_used_as_model_input", source)


if __name__ == "__main__":
    unittest.main()
