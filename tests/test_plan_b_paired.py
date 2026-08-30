"""Plan B Protocol v1 paired replay and qualification tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data_generation.orbit_solver_second_order import SecondOrderDiagnostics
from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    build_lambda_grid,
    generate_paired_fine_dataset,
    load_dataset_artifact,
    protocol_grid,
    q_identity_metadata,
    validate_plan_b_ground_truth,
)


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
        "train": np.array([[1.7], [2.1]], dtype=np.float64),
        "val": np.array([[2.4]], dtype=np.float64),
        "test": np.array([[2.7]], dtype=np.float64),
    }


def _meta(n_steps: int, step_size: float, *, paired_completeness: bool | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "task_name": "synthetic_plan_b_source",
        "task_spec": {
            "vary_params": ["Q"],
            "vary_ranges": {"Q": [1.6, 3.0]},
            "fixed_params": FIXED_PARAMS,
            "sample_shape": [4],
            "n_steps": n_steps,
            "step_size": step_size,
            "split_ratios": [0.5, 0.25, 0.25],
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
    if paired_completeness is not None:
        metadata["plan_b_protocol_version"] = "v1"
        metadata["paired_replay"] = {
            "paired_completeness": paired_completeness,
            "replacement_policy": "forbidden",
        }
    return metadata


def _trajectory(q_values: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    q = q_values[:, 0, None]
    steps = lambda_grid[None, :]
    return np.stack((q + steps, 2.0 * q - steps, q * steps + 1.0), axis=2)


def _write_dataset(
    directory: Path,
    *,
    n_steps: int,
    step_size: float,
    q_splits: dict[str, np.ndarray] | None = None,
    trajectories: dict[str, np.ndarray] | None = None,
    paired_completeness: bool | None = None,
    failed_samples: list[dict[str, object]] | None = None,
) -> None:
    directory.mkdir(parents=True)
    q_splits = q_splits or _q_splits()
    lambda_grid = build_lambda_grid(n_steps, step_size)
    arrays: dict[str, np.ndarray] = {
        "vary_params_order": np.array(["Q"]),
        "lambda_grid": lambda_grid,
    }
    for split in ("train", "val", "test"):
        q_values = q_splits[split]
        arrays[f"x_{split}"] = q_values.copy()
        arrays[f"y_{split}"] = (
            trajectories[split].copy()
            if trajectories is not None
            else _trajectory(q_values, lambda_grid)
        )
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps(_meta(n_steps, step_size, paired_completeness=paired_completeness), indent=2),
        encoding="utf-8",
    )
    (directory / "failed_samples.json").write_text(
        json.dumps(failed_samples or [], indent=2), encoding="utf-8"
    )


def _fake_simulator(**kwargs: object) -> dict[str, object]:
    q_value = float(kwargs["Q"])
    n_steps = int(kwargs["n_steps"])
    step_size = float(kwargs["step_size"])
    lambda_grid = build_lambda_grid(n_steps, step_size)
    xyz = _trajectory(np.array([[q_value]], dtype=np.float64), lambda_grid)[0]
    return {
        "lambda_grid": lambda_grid,
        "xyz": xyz,
        "diagnostics": SecondOrderDiagnostics(),
    }


def _failing_simulator(**kwargs: object) -> dict[str, object]:
    if float(kwargs["Q"]) == 2.1:
        raise RuntimeError("synthetic solver failure")
    return _fake_simulator(**kwargs)


def _rewrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def test_grid_refinement_is_endpoint_fixed() -> None:
    coarse = protocol_grid("coarse")
    fine = protocol_grid("fine")
    assert coarse.size == COARSE_N_STEPS
    assert fine.size == FINE_N_STEPS
    assert coarse[-1] == 5.995
    assert fine[-1] == 5.995
    assert np.array_equal(fine[::2], coarse)
    assert FINE_N_STEPS - 1 == 2 * (COARSE_N_STEPS - 1)


def test_q_identity_replay_preserves_values_order_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fine = tmp_path / "fine"
    _write_dataset(source, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)
    metadata = generate_paired_fine_dataset(source, fine, simulator=_fake_simulator, progress_every=10)
    source_artifact = load_dataset_artifact(source)
    fine_artifact = load_dataset_artifact(fine)
    for split in ("train", "val", "test"):
        assert np.array_equal(source_artifact.arrays[f"x_{split}"], fine_artifact.arrays[f"x_{split}"])
    source_identity = q_identity_metadata(source_artifact.arrays)
    generated_identity = metadata["paired_replay"]["generated_q_identity"]
    assert generated_identity["source_q_identity_sha256"] == source_identity["source_q_identity_sha256"]
    assert generated_identity["canonical_q_identity_sha256"] == source_identity["canonical_q_identity_sha256"]


def test_qualification_rejects_q_mismatch_before_metrics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fine = tmp_path / "fine"
    _write_dataset(source, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)
    generate_paired_fine_dataset(source, fine, simulator=_fake_simulator, progress_every=10)
    with np.load(fine / "dataset.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["x_train"] = arrays["x_train"][::-1].copy()
    _rewrite_npz(fine / "dataset.npz", arrays)
    result = validate_plan_b_ground_truth(source, fine)
    assert result["structural_valid"] is False
    assert result["structural_checks"]["q_values_and_order"]["passed"] is False
    assert result["numerical_metrics"] is None


def test_qualification_rejects_nonfinite_trajectory_before_metrics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fine = tmp_path / "fine"
    _write_dataset(source, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)
    generate_paired_fine_dataset(source, fine, simulator=_fake_simulator, progress_every=10)
    with np.load(fine / "dataset.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["y_val"] = arrays["y_val"].copy()
    arrays["y_val"][0, 0, 0] = np.nan
    _rewrite_npz(fine / "dataset.npz", arrays)
    result = validate_plan_b_ground_truth(source, fine)
    assert result["structural_valid"] is False
    assert result["structural_checks"]["trajectory_finiteness"]["passed"] is False
    assert result["numerical_metrics"] is None

def test_qualification_rejects_t2400_under_endpoint_fixed_protocol(tmp_path: Path) -> None:
    source = tmp_path / "source"
    invalid_fine = tmp_path / "invalid_fine"
    _write_dataset(source, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)
    _write_dataset(
        invalid_fine,
        n_steps=2400,
        step_size=FINE_STEP_SIZE,
        paired_completeness=True,
    )
    result = validate_plan_b_ground_truth(source, invalid_fine)
    assert result["structural_valid"] is False
    assert result["structural_checks"]["fine_protocol_grid"]["passed"] is False
    assert result["numerical_metrics"] is None


def test_common_node_extraction_and_metrics(tmp_path: Path) -> None:
    q_splits = {
        "train": np.array([[1.8]], dtype=np.float64),
        "val": np.empty((0, 1), dtype=np.float64),
        "test": np.empty((0, 1), dtype=np.float64),
    }
    coarse = tmp_path / "coarse"
    fine = tmp_path / "fine"
    coarse_grid = protocol_grid("coarse")
    fine_grid = protocol_grid("fine")
    coarse_trajectory = np.ones((1, COARSE_N_STEPS, 3), dtype=np.float64)
    fine_trajectory = np.ones((1, FINE_N_STEPS, 3), dtype=np.float64)
    fine_trajectory[:, ::2, :] = coarse_trajectory
    fine_trajectory[0, 10, 0] += 2.0
    _write_dataset(
        coarse,
        n_steps=COARSE_N_STEPS,
        step_size=COARSE_STEP_SIZE,
        q_splits=q_splits,
        trajectories={"train": coarse_trajectory, "val": np.empty((0, COARSE_N_STEPS, 3)), "test": np.empty((0, COARSE_N_STEPS, 3))},
    )
    _write_dataset(
        fine,
        n_steps=FINE_N_STEPS,
        step_size=FINE_STEP_SIZE,
        q_splits=q_splits,
        trajectories={"train": fine_trajectory, "val": np.empty((0, FINE_N_STEPS, 3)), "test": np.empty((0, FINE_N_STEPS, 3))},
        paired_completeness=True,
    )
    assert np.array_equal(fine_grid[::2], coarse_grid)
    result = validate_plan_b_ground_truth(coarse, fine)
    assert result["structural_valid"] is True
    metrics = result["numerical_metrics"]
    assert metrics is not None
    record = metrics["per_q"][0]
    assert np.isclose(record["mse"], 4.0 / (COARSE_N_STEPS * 3))
    assert np.isclose(record["relative_l2"], 2.0 / np.sqrt(COARSE_N_STEPS * 3))


def test_failed_q_is_recorded_without_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fine = tmp_path / "fine"
    _write_dataset(source, n_steps=COARSE_N_STEPS, step_size=COARSE_STEP_SIZE)
    metadata = generate_paired_fine_dataset(source, fine, simulator=_failing_simulator, progress_every=10)
    replay = metadata["paired_replay"]
    assert replay["paired_completeness"] is False
    assert replay["failure_count"] == 1
    assert replay["failed_q_values"] == [2.1]
    artifact = load_dataset_artifact(fine)
    assert np.array_equal(artifact.arrays["x_train"], _q_splits()["train"])
    assert artifact.arrays["success_mask_train"].tolist() == [True, False]
    assert len(artifact.failed_samples) == 1


class TestPlanBPaired(unittest.TestCase):
    """用标准库临时目录覆盖 Plan B 的合成协议测试。"""

    def _run(self, function) -> None:
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_grid_refinement_is_endpoint_fixed(self) -> None:
        test_grid_refinement_is_endpoint_fixed()

    def test_q_identity_replay_preserves_values_order_and_hashes(self) -> None:
        self._run(test_q_identity_replay_preserves_values_order_and_hashes)

    def test_qualification_rejects_q_mismatch_before_metrics(self) -> None:
        self._run(test_qualification_rejects_q_mismatch_before_metrics)

    def test_qualification_rejects_nonfinite_trajectory_before_metrics(self) -> None:
        self._run(test_qualification_rejects_nonfinite_trajectory_before_metrics)

    def test_qualification_rejects_t2400_under_endpoint_fixed_protocol(self) -> None:
        self._run(test_qualification_rejects_t2400_under_endpoint_fixed_protocol)

    def test_common_node_extraction_and_metrics(self) -> None:
        self._run(test_common_node_extraction_and_metrics)

    def test_failed_q_is_recorded_without_replacement(self) -> None:
        self._run(test_failed_q_is_recorded_without_replacement)


if __name__ == "__main__":
    unittest.main()
