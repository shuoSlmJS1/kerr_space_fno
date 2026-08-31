"""Plan B resolution-range sweep synthetic orchestration tests."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    build_lambda_grid,
    generate_paired_fine_dataset,
)
from src.data_generation.plan_b_resolution_range import (
    RESOLUTION_RANGE_SPECS,
    endpoint_refinement_spec,
    generate_paired_refinement_dataset,
    validate_refinement_ground_truth,
)
from src.models.registry_2d import build_model_2d
from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_plan_b_resolution_range_sweep_2d.py"
SPEC = importlib.util.spec_from_file_location("plan_b_resolution_range_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


FIXED_PARAMS = {
    "M": 1.0, "a": 0.5, "E": 0.95, "Lz": 3.0, "r0": 10.0,
    "theta0": 1.2, "phi0": 0.0, "sign_r": -1, "sign_th": 1,
}


def _q_splits() -> dict[str, np.ndarray]:
    return {
        "train": np.array([[1.7], [2.1]], dtype=np.float64),
        "val": np.array([[2.4]], dtype=np.float64),
        "test": np.array([[2.7]], dtype=np.float64),
    }


def _trajectory(q: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values, lam = q[:, 0, None], grid[None, :]
    return np.stack((values + lam, 2.0 * values - lam, 1.0 + values * lam), axis=2).astype(np.float64)


def _metadata(n_steps: int, step_size: float) -> dict[str, object]:
    return {
        "task_name": "synthetic_q400_t1200",
        "task_spec": {
            "vary_params": ["Q"], "vary_ranges": {"Q": [1.6, 3.0]},
            "fixed_params": FIXED_PARAMS, "n_steps": n_steps, "step_size": step_size,
            "metadata": {"orbit_solver": "second_order_rk4", "orbit_solver_version": "v1"},
        },
        "integration": {"n_steps": n_steps, "step_size": step_size, "lambda_min": 0.0, "lambda_max": (n_steps - 1) * step_size},
    }


def _write_anchor(directory: Path) -> None:
    directory.mkdir(parents=True)
    grid = build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
    arrays: dict[str, np.ndarray] = {"vary_params_order": np.array(["Q"]), "lambda_grid": grid}
    for split, values in _q_splits().items():
        arrays[f"x_{split}"] = values
        arrays[f"y_{split}"] = _trajectory(values, grid)
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(json.dumps(_metadata(COARSE_N_STEPS, COARSE_STEP_SIZE)), encoding="utf-8", newline="\n")
    (directory / "failed_samples.json").write_text("[]\n", encoding="utf-8", newline="\n")


def _simulator(*, Q: float, n_steps: int, step_size: float, **_: object) -> dict[str, object]:
    grid = build_lambda_grid(n_steps, step_size)
    return {"lambda_grid": grid, "xyz": _trajectory(np.array([[Q]], dtype=np.float64), grid)[0]}


def _failing_simulator(*, Q: float, **kwargs: object) -> dict[str, object]:
    if Q == 2.1:
        raise RuntimeError("synthetic source-Q failure")
    return _simulator(Q=Q, **kwargs)


def _metrics(scale: float) -> dict[str, object]:
    per_q = {"mean": scale, "median": scale, "p95": scale, "p99": scale, "max": scale, "worst_q_value": 1.7}
    return {"global_mse": scale * scale, "global_relative_l2": scale, "mean_per_q_relative_l2": scale, "per_q_relative_l2_summary": per_q}


def _completed_metrics(root: Path) -> tuple[Path, Path, Path]:
    theta1200 = root / "theta1200_metrics.json"
    theta2399 = root / "theta2399_metrics.json"
    matrix = root / "bidirectional_matrix.json"
    a, b, c, d = _metrics(0.10), _metrics(0.12), _metrics(0.13), _metrics(0.11)
    theta1200.write_text(json.dumps({"coarse_t1200": a, "fine_t2399_full_grid_primary": b}), encoding="utf-8", newline="\n")
    theta2399.write_text(json.dumps({"theta2399_reverse_t1200": c, "theta2399_native_t2399": d}), encoding="utf-8", newline="\n")
    matrix.write_text(json.dumps({"training_rows": {"theta1200": {"test_t1200": a, "test_t2399": b}, "theta2399": {"test_t1200": c, "test_t2399": d}}}), encoding="utf-8", newline="\n")
    return theta1200, matrix, theta2399


def _contract(label: str) -> SimpleNamespace:
    stats = FieldNormalizationStats(
        method="standard", x_mean=[1.0 if label == "theta1200" else 2.0, 3.0], x_std=[1.0, 1.0],
        y_mean=[0.0, 0.0, 0.0], y_std=[1.0, 1.0, 1.0],
    )
    return SimpleNamespace(
        normalization_stats=stats, target_transform_config=TargetTransformConfig(mode="raw"),
        model_config={"model_type": "fno2d", "in_dim": 2, "out_dim": 3},
    )


class ResolutionSpecTests(unittest.TestCase):
    def test_t3598_and_t4797_endpoint_and_common_nodes(self) -> None:
        coarse = build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
        for factor, expected_t, expected_h in ((3, 3598, COARSE_STEP_SIZE / 3), (4, 4797, 0.00125)):
            spec = endpoint_refinement_spec(factor)
            self.assertEqual(spec.n_steps, expected_t)
            self.assertAlmostEqual(spec.step_size, expected_h)
            fine = build_lambda_grid(spec.n_steps, spec.step_size)
            self.assertAlmostEqual(float(fine[-1]), 5.995)
            np.testing.assert_allclose(fine[::factor], coarse, rtol=0.0, atol=1e-12)

    def test_actual_tiny_fno_accepts_new_grid_lengths(self) -> None:
        model = build_model_2d(model_type="fno2d", in_dim=2, out_dim=3, modes1=1, modes2=2, width=4, depth=1, hidden_dim=8, activation="gelu").eval()
        with torch.no_grad():
            for t_value in (3598, 4797):
                result = model(torch.zeros((1, 1, t_value, 2), dtype=torch.float32))
                self.assertEqual(tuple(result.shape), (1, 1, t_value, 3))


class PairedTruthTests(unittest.TestCase):
    def test_exact_replay_and_qualification_for_both_new_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); anchor = root / "anchor"; _write_anchor(anchor)
            for spec in RESOLUTION_RANGE_SPECS.values():
                output = root / f"t{spec.n_steps}"
                metadata = generate_paired_refinement_dataset(anchor, output, spec=spec, simulator=_simulator, progress_every=10)
                self.assertTrue(metadata["paired_replay"]["paired_completeness"])
                result = validate_refinement_ground_truth(anchor, output, spec=spec)
                self.assertTrue(result["structural_valid"])
                self.assertIsNotNone(result["numerical_metrics"])
                self.assertEqual(result["numerical_metrics"]["comparison"], f"fine_xyz[:, ::{spec.refinement_factor}, :] versus coarse_xyz")

    def test_failure_and_nonfinite_truth_block_numerical_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); anchor = root / "anchor"; _write_anchor(anchor)
            failed = root / "failed"
            generate_paired_refinement_dataset(anchor, failed, spec=RESOLUTION_RANGE_SPECS[3], simulator=_failing_simulator, progress_every=10)
            failure_result = validate_refinement_ground_truth(anchor, failed, spec=RESOLUTION_RANGE_SPECS[3])
            self.assertFalse(failure_result["structural_valid"])
            self.assertIsNone(failure_result["numerical_metrics"])
            valid = root / "valid"
            generate_paired_refinement_dataset(anchor, valid, spec=RESOLUTION_RANGE_SPECS[4], simulator=_simulator, progress_every=10)
            with np.load(valid / "dataset.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["y_train"] = arrays["y_train"].copy(); arrays["y_train"][0, 0, 0] = np.nan
            np.savez_compressed(valid / "dataset.npz", **arrays)
            nonfinite_result = validate_refinement_ground_truth(anchor, valid, spec=RESOLUTION_RANGE_SPECS[4])
            self.assertFalse(nonfinite_result["structural_valid"])
            self.assertIsNone(nonfinite_result["numerical_metrics"])


class WorkflowTests(unittest.TestCase):
    def _run(self, root: Path, *, resume: bool = False, forward_calls: list[tuple[str, int, object]] | None = None) -> dict[str, object]:
        anchor, existing = root / "anchor", root / "existing_t2399"
        if not anchor.exists():
            _write_anchor(anchor)
            generate_paired_fine_dataset(anchor, existing, simulator=_simulator, progress_every=10)
        theta1200_metrics, matrix, theta2399_metrics = _completed_metrics(root)
        contracts = {"theta1200": _contract("theta1200"), "theta2399": _contract("theta2399")}
        def fake_loader(path: Path, _device: str) -> dict[str, object]:
            return {"label": "theta1200" if "1200" in path.name else "theta2399", "epoch": 1}
        def fake_model_loader(checkpoint: dict[str, object], _device: str) -> object:
            return checkpoint["label"]
        def fake_forward(*, model: object, contract: object, field: object, **_: object) -> np.ndarray:
            if forward_calls is not None:
                forward_calls.append((str(model), int(field.lambda_grid.size), contract.normalization_stats))
            return np.asarray(field.canonical_truth, dtype=np.float32) + (0.01 if model == "theta1200" else 0.02)
        with patch.object(workflow, "validate_checkpoint_contract", return_value=contracts["theta1200"]), patch.object(workflow, "validate_t2399_checkpoint_contract", return_value=contracts["theta2399"]):
            return workflow.run_resolution_range_workflow(
                q400_t1200_dataset_dir=anchor, q400_t2399_dataset_dir=existing,
                q400_t3598_dataset_dir=root / "t3598", q400_t4797_dataset_dir=root / "t4797",
                theta1200_checkpoint=root / "theta1200.pt", theta2399_checkpoint=root / "theta2399.pt",
                theta1200_metrics_json=theta1200_metrics, bidirectional_matrix_json=matrix, theta2399_metrics_json=theta2399_metrics,
                output_dir=root / "sweep", device="cpu", resume=resume, progress_every=10, simulator=_simulator,
                checkpoint_loader=fake_loader, model_loader=fake_model_loader, forward_runner=fake_forward,
            )

    def test_tiny_full_workflow_reuses_existing_cells_and_builds_matrix_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); calls: list[tuple[str, int, object]] = []
            result = self._run(root, forward_calls=calls)
            self.assertEqual(len(calls), 4)
            self.assertEqual({item[1] for item in calls}, {3598, 4797})
            self.assertIs(calls[0][2], calls[2][2])
            self.assertIs(calls[1][2], calls[3][2])
            self.assertIsNot(calls[0][2], calls[1][2])
            matrix = result["matrix"]
            self.assertTrue(matrix["existing_t1200_t2399_metrics_reused"])
            self.assertEqual(matrix["existing_t1200_t2399_forward_passes_rerun"], 0)
            self.assertEqual(set(matrix["training_rows"]["theta1200"]), {"test_t1200", "test_t2399", "test_t3598", "test_t4797"})
            curve = json.loads((root / "sweep" / "resolution_curve.json").read_text(encoding="utf-8"))["curve"]
            reverse = next(item for item in curve if item["training_model"] == "theta2399" and item["evaluation_T"] == 1200)
            self.assertAlmostEqual(reverse["h_train_divided_by_h_test"], 0.5)
            self.assertEqual(len(curve), 8)
            resumed_calls: list[tuple[str, int, object]] = []
            self._run(root, resume=True, forward_calls=resumed_calls)
            self.assertEqual(resumed_calls, [])

    def test_cli_resume_and_no_training_scope(self) -> None:
        args = workflow.parse_args([
            "--q400-t1200-dataset-dir", "q1200", "--q400-t2399-dataset-dir", "q2399",
            "--q400-t3598-dataset-dir", "q3598", "--q400-t4797-dataset-dir", "q4797",
            "--theta1200-checkpoint", "theta1200.pt", "--theta2399-checkpoint", "theta2399.pt",
            "--theta1200-metrics-json", "old1200.json", "--bidirectional-matrix-json", "matrix.json",
            "--theta2399-metrics-json", "old2399.json", "--output-dir", "output", "--resume",
        ])
        self.assertTrue(args.resume)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for forbidden in ("train_matched_t2399_model", "optimizer", ".backward(", "scheduler", "compute_field_normalization_stats"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
