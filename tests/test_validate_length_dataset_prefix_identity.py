"""严格长度数据集身份验证器的合成数据测试。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_length_dataset_prefix_identity.py"
SPEC = importlib.util.spec_from_file_location("prefix_identity_validator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _meta(n_steps: int, lambda_max: float, solver: str = "second_order_rk4") -> dict[str, object]:
    return {
        "task_spec": {
            "vary_params": ["Q"], "vary_ranges": {"Q": [1.6, 3.0]},
            "fixed_params": {"M": 1.0, "a": 0.5, "E": 0.95, "Lz": 3.0, "r0": 10.0,
                             "theta0": 1.2, "phi0": 0.0, "sign_r": -1, "sign_th": 1},
            "split_ratios": [0.5, 0.25, 0.25], "seed": 10, "sampling_mode": "grid",
            "completion_policy": "target_success", "n_steps": n_steps,
            "step_size": 0.005, "lambda_max": lambda_max,
            "metadata": {"orbit_solver": solver, "orbit_solver_version": "v1"},
        },
        "generation_status": {"completion_policy": "target_success", "successful_points_strictly_uniform": True},
    }


def _write_dataset(directory: Path, length: int, *, q_splits: dict[str, np.ndarray] | None = None,
                   source_length: int | None = None, perturb: tuple[str, int, int, int, float] | None = None,
                   solver: str = "second_order_rk4", nonfinite: tuple[str, str, str] | None = None) -> None:
    directory.mkdir(parents=True)
    q_splits = q_splits or {
        "train": np.array([[1.6], [1.8]], dtype=np.float64),
        "val": np.array([[2.0]], dtype=np.float64), "test": np.array([[2.2]], dtype=np.float64),
    }
    source_length = max(length, source_length or length)
    arrays: dict[str, np.ndarray] = {"vary_params_order": np.array(["Q"]), "lambda_grid": np.arange(length, dtype=np.float64) * 0.005}
    for split, q in q_splits.items():
        arrays[f"x_{split}"] = q
        base = np.stack([q[:, 0, None] + np.arange(source_length) * 0.01 + channel for channel in range(3)], axis=2)
        arrays[f"y_{split}"] = base[:, :length, :].astype(np.float64)
    if perturb is not None:
        split, index, step, channel, value = perturb
        arrays[f"y_{split}"][index, step, channel] += value
    if nonfinite is not None:
        key, kind, split = nonfinite
        target = arrays[key] if split == "" else arrays[f"{key}_{split}"]
        target.reshape(-1)[0] = np.nan if kind == "nan" else np.inf
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(json.dumps(_meta(length, (length - 1) * 0.005, solver)), encoding="utf-8")


def _triplet(tmp_path: Path, **long_kwargs: object) -> tuple[Path, Path, Path]:
    short, medium, long = tmp_path / "short", tmp_path / "medium", tmp_path / "long"
    _write_dataset(short, 4)
    _write_dataset(medium, 6, source_length=4)
    _write_dataset(long, 8, source_length=4, **long_kwargs)
    return short, medium, long


def _result(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    short, medium, long = _triplet(tmp_path, **kwargs)
    return validator.validate_datasets(short, medium, long)


def test_exact_prefix_all_pairs_and_json_serializable(tmp_path: Path) -> None:
    result = _result(tmp_path)
    assert result["classification"] == "EXACT_PREFIX"
    assert set(result["pair_classification"].values()) == {"EXACT_PREFIX"}
    assert result["trajectory_prefix_checks"]["short_to_medium"]["train"]["rmse"] == 0.0
    json.dumps(result, allow_nan=False)


def test_q_order_and_q_value_mismatch_are_not_paired(tmp_path: Path) -> None:
    reordered = {"train": np.array([[1.8], [1.6]]), "val": np.array([[2.0]]), "test": np.array([[2.2]])}
    result = _result(tmp_path, q_splits=reordered)
    split = result["split_identity"]["short_to_long"]["train"]
    assert split["same_values_different_order"] is True
    assert split["secondary_deterministic_q_match"]["eligible"] is True
    assert result["pair_classification"]["short_to_long"] == "NOT_PAIRED"
    other = tmp_path / "other"
    short, medium, long = _triplet(other)
    bad = np.array([[2.3]])
    with np.load(long / "dataset.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["x_test"] = bad
    np.savez_compressed(long / "replacement.npz", **arrays)
    (long / "dataset.npz").unlink()
    (long / "replacement.npz").rename(long / "dataset.npz")
    assert validator.validate_datasets(short, medium, long)["classification"] == "NOT_PAIRED"


def test_lambda_and_trajectory_tolerance_classifications(tmp_path: Path) -> None:
    short, medium, long = _triplet(tmp_path)
    with np.load(long / "dataset.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["lambda_grid"][1] += 5e-13
    arrays["y_test"][0, 1, 0] += 5e-13
    np.savez_compressed(long / "changed.npz", **arrays)
    (long / "dataset.npz").unlink(); (long / "changed.npz").rename(long / "dataset.npz")
    result = validator.validate_datasets(short, medium, long)
    assert result["pair_classification"]["short_to_long"] == "NUMERICALLY_EQUIVALENT_PREFIX"
    outside = _result(tmp_path / "outside", perturb=("test", 0, 1, 0, 1e-6))
    assert outside["classification"] == "NOT_PAIRED"
    short_two, medium_two, long_two = _triplet(tmp_path / "lambda_outside")
    with np.load(long_two / "dataset.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["lambda_grid"][1] += 1e-6
    np.savez_compressed(long_two / "changed.npz", **arrays)
    (long_two / "dataset.npz").unlink()
    (long_two / "changed.npz").rename(long_two / "dataset.npz")
    lambda_outside = validator.validate_datasets(short_two, medium_two, long_two)
    assert lambda_outside["pair_classification"]["short_to_long"] == "NOT_PAIRED"


def test_nonfinite_and_metadata_mismatch_are_not_paired(tmp_path: Path, kind: str) -> None:
    result = _result(tmp_path, nonfinite=("y", kind, "test"))
    assert result["classification"] == "NOT_PAIRED"
    other = tmp_path / "metadata"
    short, medium, long = _triplet(other, solver="different_solver")
    assert validator.validate_datasets(short, medium, long)["classification"] == "NOT_PAIRED"


def test_medium_can_pass_while_long_fails_and_reuse_rules_follow_pairs(tmp_path: Path) -> None:
    result = _result(tmp_path, perturb=("test", 0, 1, 0, 1e-6))
    assert result["pair_classification"]["short_to_medium"] == "EXACT_PREFIX"
    assert result["scientific_reuse"]["historical_t1800_reusable"] is True
    assert result["scientific_reuse"]["t2400_ready_for_future_a1"] is False


def test_exclusive_output_cli_help_and_no_torch_or_checkpoint_dependency(tmp_path: Path) -> None:
    short, medium, long = _triplet(tmp_path)
    output = tmp_path / "result.json"
    command = [sys.executable, str(SCRIPT_PATH), "--short-dataset-dir", str(short), "--medium-dataset-dir", str(medium), "--long-dataset-dir", str(long), "--output-json", str(output)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Classification: EXACT_PREFIX" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == "EXACT_PREFIX"
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    help_result = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], check=True, capture_output=True, text=True)
    assert "--trajectory-batch-size" in help_result.stdout
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "checkpoint" not in source.lower()


class TestLengthDatasetPrefixIdentity(unittest.TestCase):
    """使用标准库临时目录运行全部合成验证场景。"""

    def _run(self, function) -> None:
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_exact_prefix_all_pairs_and_json_serializable(self) -> None:
        self._run(test_exact_prefix_all_pairs_and_json_serializable)

    def test_q_order_and_q_value_mismatch_are_not_paired(self) -> None:
        self._run(test_q_order_and_q_value_mismatch_are_not_paired)

    def test_lambda_and_trajectory_tolerance_classifications(self) -> None:
        self._run(test_lambda_and_trajectory_tolerance_classifications)

    def test_nan_is_not_paired(self) -> None:
        self._run(lambda path: test_nonfinite_and_metadata_mismatch_are_not_paired(path, "nan"))

    def test_inf_is_not_paired(self) -> None:
        self._run(lambda path: test_nonfinite_and_metadata_mismatch_are_not_paired(path, "inf"))

    def test_medium_can_pass_while_long_fails_and_reuse_rules_follow_pairs(self) -> None:
        self._run(test_medium_can_pass_while_long_fails_and_reuse_rules_follow_pairs)

    def test_exclusive_output_cli_help_and_no_torch_or_checkpoint_dependency(self) -> None:
        self._run(test_exclusive_output_cli_help_and_no_torch_or_checkpoint_dependency)


if __name__ == "__main__":
    unittest.main()
