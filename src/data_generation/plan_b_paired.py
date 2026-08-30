"""Plan B paired-Q replay generation and ground-truth qualification.

该模块为 Protocol v1 提供窄接口：从既有 coarse 数据集逐字节复用 Q 身份，
只改变 lambda 轴离散，并在不满足结构条件时拒绝计算数值一致性指标。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.data_generation.dataset_builder import (
    build_initial_state,
    build_kerr_params,
)
from src.data_generation.orbit_solver_second_order import (
    simulate_one_orbit_second_order,
)
from src.data_generation.validity import validate_single_sample_hard_constraints


PLAN_B_PROTOCOL_VERSION = "v1"
COARSE_N_STEPS = 1200
COARSE_STEP_SIZE = 0.005
FINE_N_STEPS = 2399
FINE_STEP_SIZE = 0.0025
SPLITS = ("train", "val", "test")
FIXED_PARAMETER_KEYS = (
    "M",
    "a",
    "E",
    "Lz",
    "r0",
    "theta0",
    "phi0",
    "sign_r",
    "sign_th",
)

OrbitSimulator = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class DatasetArtifact:
    """已加载的 Plan B 输入数据集及其最小必需信息。"""

    directory: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    failed_samples: list[dict[str, Any]]


def build_lambda_grid(n_steps: int, step_size: float) -> np.ndarray:
    """按 solver 的既有定义构造 lambda grid。"""
    if not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps must be an integer greater than or equal to 2.")
    if not isinstance(step_size, (int, float)) or float(step_size) <= 0.0:
        raise ValueError("step_size must be positive.")
    return np.arange(n_steps, dtype=np.float64) * float(step_size)


def protocol_grid(role: str) -> np.ndarray:
    """返回 Protocol v1 唯一允许的 coarse 或 fine grid。"""
    if role == "coarse":
        return build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
    if role == "fine":
        return build_lambda_grid(FINE_N_STEPS, FINE_STEP_SIZE)
    raise ValueError("role must be 'coarse' or 'fine'.")


def _json_value(value: Any) -> Any:
    """将 numpy 标量和数组递归转换为 JSON 可序列化值。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return value


def _load_failed_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"failed_samples.json must contain a JSON list: {path}")
    return [dict(item) for item in value]


def load_dataset_artifact(dataset_dir: str | Path) -> DatasetArtifact:
    """读取标准 dataset.npz、meta.json 和可选失败记录，不改变源文件。"""
    directory = Path(dataset_dir).resolve()
    dataset_path = directory / "dataset.npz"
    meta_path = directory / "meta.json"
    if not dataset_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "Dataset directory must contain dataset.npz and meta.json: "
            f"{directory}"
        )
    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    _require_dataset_arrays(arrays, directory)
    return DatasetArtifact(
        directory=directory,
        arrays=arrays,
        metadata=_load_json(meta_path),
        failed_samples=_load_failed_samples(directory / "failed_samples.json"),
    )


def _require_dataset_arrays(arrays: dict[str, np.ndarray], directory: Path) -> None:
    required = {"vary_params_order", "lambda_grid"}
    required.update({f"x_{split}" for split in SPLITS})
    required.update({f"y_{split}" for split in SPLITS})
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Dataset is missing required arrays in {directory}: {missing}")
    order = [str(item) for item in arrays["vary_params_order"].tolist()]
    if order != ["Q"]:
        raise ValueError(
            "Plan B paired replay requires exactly one varying parameter ordered as ['Q']."
        )
    for split in SPLITS:
        q_values = arrays[f"x_{split}"]
        trajectories = arrays[f"y_{split}"]
        if q_values.ndim != 2 or q_values.shape[1] != 1:
            raise ValueError(f"x_{split} must have shape [N, 1].")
        if trajectories.ndim != 3 or trajectories.shape[0] != q_values.shape[0]:
            raise ValueError(f"y_{split} must have shape [N, T, 3] aligned with x_{split}.")
        if trajectories.shape[2] != 3:
            raise ValueError(f"y_{split} must have xyz channel count 3.")


def _task_spec(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("task_spec")
    if not isinstance(value, dict):
        raise ValueError("Dataset metadata is missing task_spec.")
    return value


def _fixed_params(metadata: dict[str, Any]) -> dict[str, Any]:
    fixed = _task_spec(metadata).get("fixed_params")
    if not isinstance(fixed, dict):
        raise ValueError("Dataset task_spec is missing fixed_params.")
    missing = [key for key in FIXED_PARAMETER_KEYS if key not in fixed]
    if missing:
        raise ValueError(f"Dataset fixed_params is missing Protocol v1 keys: {missing}")
    return {key: fixed[key] for key in FIXED_PARAMETER_KEYS}


def _integration_metadata(metadata: dict[str, Any]) -> tuple[int, float, float]:
    task_spec = _task_spec(metadata)
    try:
        n_steps = int(task_spec["n_steps"])
        step_size = float(task_spec["step_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Dataset task_spec must record n_steps and step_size.") from error
    return n_steps, step_size, float((n_steps - 1) * step_size)


def _solver_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    task_metadata = _task_spec(metadata).get("metadata", {})
    if not isinstance(task_metadata, dict):
        raise ValueError("Dataset task_spec.metadata must be a JSON object.")
    solver = task_metadata.get("orbit_solver")
    version = task_metadata.get("orbit_solver_version")
    if not isinstance(solver, str) or not isinstance(version, str):
        raise ValueError("Dataset metadata must record orbit_solver and orbit_solver_version.")
    return {"orbit_solver": solver, "orbit_solver_version": version}


def _hash_arrays(named_arrays: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in named_arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def q_identity_metadata(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """返回同时覆盖原始 split 顺序和 canonical sorted-Q 的机器身份信息。"""
    split_arrays = [(split, arrays[f"x_{split}"]) for split in SPLITS]
    combined = np.concatenate([array[:, 0] for _, array in split_arrays])
    canonical = np.sort(combined, kind="stable")
    return {
        "source_split_order": list(SPLITS),
        "source_row_order_preserved": True,
        "global_q_strictly_ascending_in_source_order": bool(
            np.all(np.diff(combined) > 0.0)
        ) if combined.size > 1 else True,
        "source_q_count": int(combined.size),
        "source_q_identity_sha256": _hash_arrays(split_arrays),
        "canonical_ordering": "stable_ascending_Q",
        "canonical_q_identity_sha256": _hash_arrays([("canonical_Q", canonical)]),
    }


def _require_protocol_grid(
    lambda_grid: np.ndarray,
    n_steps: int,
    step_size: float,
    role: str,
) -> None:
    expected = protocol_grid(role)
    if n_steps != expected.size or not math.isclose(step_size, expected[1], rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"Plan B Protocol v1 {role} grid requires T={expected.size} and "
            f"step_size={expected[1]!r}."
        )
    if not np.array_equal(lambda_grid, expected):
        raise ValueError(f"Plan B Protocol v1 {role} lambda_grid does not match arange(T) * step_size.")


def _diagnostics_record(orbit: dict[str, object]) -> dict[str, Any]:
    diagnostics = orbit.get("diagnostics")
    if diagnostics is None:
        return {"available": False}
    record = asdict(diagnostics)
    return {"available": True, **_json_value(record)}


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def generate_paired_fine_dataset(
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    n_steps: int = FINE_N_STEPS,
    step_size: float = FINE_STEP_SIZE,
    simulator: OrbitSimulator = simulate_one_orbit_second_order,
    progress_every: int = 50,
) -> dict[str, Any]:
    """从 coarse Q 身份逐项 replay 生成 Plan B fine truth，绝不补样或替换失败 Q。"""
    if n_steps != FINE_N_STEPS or not math.isclose(step_size, FINE_STEP_SIZE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Plan B Protocol v1 only permits T=2399 and step_size=0.0025 for fine replay.")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")

    source = load_dataset_artifact(source_dataset_dir)
    source_n_steps, source_step_size, _ = _integration_metadata(source.metadata)
    _require_protocol_grid(source.arrays["lambda_grid"], source_n_steps, source_step_size, "coarse")
    source_identity = q_identity_metadata(source.arrays)
    fixed_params = _fixed_params(source.metadata)
    solver_provenance = _solver_provenance(source.metadata)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    kerr_params = build_kerr_params(fixed_params)
    initial_state = build_initial_state(fixed_params)
    lambda_grid = protocol_grid("fine")
    payload: dict[str, Any] = {
        "vary_params_order": source.arrays["vary_params_order"].copy(),
        "lambda_grid": lambda_grid,
    }
    diagnostics_by_q: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    success_count = 0
    source_q_count = int(source_identity["source_q_count"])
    processed = 0

    for split in SPLITS:
        source_q = source.arrays[f"x_{split}"]
        xyz = np.full((source_q.shape[0], n_steps, 3), np.nan, dtype=np.float64)
        success_mask = np.zeros(source_q.shape[0], dtype=bool)
        payload[f"x_{split}"] = source_q.copy()
        payload[f"y_{split}"] = xyz
        payload[f"success_mask_{split}"] = success_mask
        for row_index, q_row in enumerate(source_q):
            q_value = float(q_row[0])
            processed += 1
            try:
                validate_single_sample_hard_constraints({"Q": q_value}, fixed_params)
                orbit = simulator(
                    p=kerr_params,
                    init=initial_state,
                    Q=q_value,
                    n_steps=n_steps,
                    step_size=step_size,
                )
                orbit_grid = np.asarray(orbit["lambda_grid"], dtype=np.float64)
                orbit_xyz = np.asarray(orbit["xyz"], dtype=np.float64)
                if not np.array_equal(orbit_grid, lambda_grid):
                    raise RuntimeError("Solver returned a lambda grid inconsistent with Plan B Protocol v1.")
                if orbit_xyz.shape != (n_steps, 3) or not np.all(np.isfinite(orbit_xyz)):
                    raise RuntimeError("Solver returned invalid xyz trajectory data.")
                payload[f"y_{split}"][row_index] = orbit_xyz
                payload[f"success_mask_{split}"][row_index] = True
                success_count += 1
                diagnostics_by_q.append({
                    "split": split,
                    "source_row_index": int(row_index),
                    "Q": q_value,
                    "success": True,
                    "diagnostics": _diagnostics_record(orbit),
                })
            except Exception as error:  # noqa: BLE001 - generation must preserve every source-Q failure.
                failure = {
                    "split": split,
                    "source_row_index": int(row_index),
                    "Q": q_value,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "replacement_attempted": False,
                }
                failed_samples.append(failure)
                diagnostics_by_q.append({**failure, "success": False, "diagnostics": {"available": False}})
            if processed % progress_every == 0 or processed == source_q_count:
                print(f"Paired replay progress: {processed}/{source_q_count}")

    generated_identity = q_identity_metadata(payload)
    paired_completeness = bool(
        success_count == source_q_count
        and source_identity["source_q_identity_sha256"] == generated_identity["source_q_identity_sha256"]
        and not failed_samples
    )
    source_task_spec = _task_spec(source.metadata)
    fine_task_spec = json.loads(json.dumps(_json_value(source_task_spec)))
    fine_task_spec["n_steps"] = int(n_steps)
    fine_task_spec["step_size"] = float(step_size)
    fine_task_spec.setdefault("metadata", {})["orbit_solver"] = solver_provenance["orbit_solver"]
    fine_task_spec["metadata"]["orbit_solver_version"] = solver_provenance["orbit_solver_version"]
    metadata = {
        "schema_version": "1.0",
        "experiment_type": "plan_b_paired_fine_replay_generation",
        "plan_b_protocol_version": PLAN_B_PROTOCOL_VERSION,
        "task_spec": fine_task_spec,
        "integration": {
            "n_steps": int(n_steps),
            "step_size": float(step_size),
            "lambda_min": float(lambda_grid[0]),
            "lambda_max": float(lambda_grid[-1]),
        },
        "paired_replay": {
            "source_dataset_identifier": source.metadata.get("task_name", str(source.directory)),
            "source_dataset_path": str(source.directory),
            "source_q_count": source_q_count,
            "source_q_identity": source_identity,
            "generated_q_identity": generated_identity,
            "q_ordering": {
                "generation_order": "source_split_row_order",
                "canonical_ordering": "stable_ascending_Q",
                "source_and_generated_raw_order_match": bool(
                    source_identity["source_q_identity_sha256"] == generated_identity["source_q_identity_sha256"]
                ),
            },
            "fixed_kerr_parameters": {key: fixed_params[key] for key in ("M", "a", "E", "Lz")},
            "initial_conditions": {key: fixed_params[key] for key in ("r0", "theta0", "phi0", "sign_r", "sign_th")},
            "solver_provenance": solver_provenance,
            "success_count": int(success_count),
            "failure_count": int(len(failed_samples)),
            "failed_q_values": [item["Q"] for item in failed_samples],
            "paired_completeness": paired_completeness,
            "replacement_policy": "forbidden",
            "target_success_replacement_used": False,
        },
        "generation_status": {
            "completed": paired_completeness,
            "success_count": int(success_count),
            "failure_count": int(len(failed_samples)),
            "paired_completeness": paired_completeness,
        },
    }
    np.savez_compressed(output / "dataset.npz", **payload)
    _write_json_exclusive(output / "meta.json", metadata)
    _write_json_exclusive(output / "failed_samples.json", failed_samples)
    _write_json_exclusive(output / "solver_diagnostics.json", diagnostics_by_q)
    return _json_value(metadata)


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **_json_value(details)}


def _artifact_q_arrays(artifact: DatasetArtifact) -> list[tuple[str, np.ndarray]]:
    return [(split, artifact.arrays[f"x_{split}"]) for split in SPLITS]


def _load_solver_diagnostics(directory: Path) -> list[dict[str, Any]] | None:
    path = directory / "solver_diagnostics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else None


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def validate_plan_b_ground_truth(
    coarse_dataset_dir: str | Path,
    fine_dataset_dir: str | Path,
) -> dict[str, Any]:
    """验证 Plan B 配对结构；结构不合法时绝不计算数值一致性指标。"""
    coarse = load_dataset_artifact(coarse_dataset_dir)
    fine = load_dataset_artifact(fine_dataset_dir)
    coarse_n_steps, coarse_step_size, coarse_lambda_max = _integration_metadata(coarse.metadata)
    fine_n_steps, fine_step_size, fine_lambda_max = _integration_metadata(fine.metadata)
    coarse_q_identity = q_identity_metadata(coarse.arrays)
    fine_q_identity = q_identity_metadata(fine.arrays)
    checks: dict[str, dict[str, Any]] = {}
    checks["q_count"] = _check(
        "q_count",
        coarse_q_identity["source_q_count"] == fine_q_identity["source_q_count"],
        coarse_count=coarse_q_identity["source_q_count"],
        fine_count=fine_q_identity["source_q_count"],
    )
    checks["q_values_and_order"] = _check(
        "q_values_and_order",
        all(np.array_equal(coarse.arrays[f"x_{split}"], fine.arrays[f"x_{split}"]) for split in SPLITS),
        coarse_q_identity_sha256=coarse_q_identity["source_q_identity_sha256"],
        fine_q_identity_sha256=fine_q_identity["source_q_identity_sha256"],
    )
    checks["canonical_q_identity"] = _check(
        "canonical_q_identity",
        coarse_q_identity["canonical_q_identity_sha256"] == fine_q_identity["canonical_q_identity_sha256"],
        coarse_canonical_q_identity_sha256=coarse_q_identity["canonical_q_identity_sha256"],
        fine_canonical_q_identity_sha256=fine_q_identity["canonical_q_identity_sha256"],
    )
    checks["fixed_physics"] = _check(
        "fixed_physics", _fixed_params(coarse.metadata) == _fixed_params(fine.metadata)
    )
    checks["solver_provenance"] = _check(
        "solver_provenance", _solver_provenance(coarse.metadata) == _solver_provenance(fine.metadata)
    )
    checks["coarse_protocol_grid"] = _check(
        "coarse_protocol_grid",
        coarse_n_steps == COARSE_N_STEPS
        and coarse_step_size == COARSE_STEP_SIZE
        and np.array_equal(coarse.arrays["lambda_grid"], protocol_grid("coarse")),
        n_steps=coarse_n_steps,
        step_size=coarse_step_size,
    )
    checks["fine_protocol_grid"] = _check(
        "fine_protocol_grid",
        fine_n_steps == FINE_N_STEPS
        and fine_step_size == FINE_STEP_SIZE
        and np.array_equal(fine.arrays["lambda_grid"], protocol_grid("fine")),
        n_steps=fine_n_steps,
        step_size=fine_step_size,
    )
    checks["lambda_endpoints"] = _check(
        "lambda_endpoints",
        coarse_lambda_max == fine_lambda_max == protocol_grid("coarse")[-1]
        and float(coarse.arrays["lambda_grid"][0]) == float(fine.arrays["lambda_grid"][0]) == 0.0,
        coarse_lambda_max=coarse_lambda_max,
        fine_lambda_max=fine_lambda_max,
    )
    checks["common_node_grid"] = _check(
        "common_node_grid",
        np.array_equal(fine.arrays["lambda_grid"][::2], coarse.arrays["lambda_grid"]),
    )
    paired_replay = fine.metadata.get("paired_replay", {})
    checks["paired_completeness"] = _check(
        "paired_completeness",
        isinstance(paired_replay, dict) and paired_replay.get("paired_completeness") is True,
        reported_value=paired_replay.get("paired_completeness") if isinstance(paired_replay, dict) else None,
    )
    structural_valid = all(check["passed"] for check in checks.values())
    base_result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_type": "plan_b_ground_truth_consistency_qualification",
        "plan_b_protocol_version": PLAN_B_PROTOCOL_VERSION,
        "coarse_dataset_path": str(coarse.directory),
        "fine_dataset_path": str(fine.directory),
        "structural_valid": structural_valid,
        "structural_checks": checks,
        "anomalies": {
            "coarse_failed_samples": coarse.failed_samples,
            "fine_failed_samples": fine.failed_samples,
        },
    }
    if not structural_valid:
        base_result["numerical_metrics"] = None
        base_result["turning_point_diagnostics"] = {"available": False, "reason": "structural_validation_failed"}
        return _json_value(base_result)

    trajectory_shape_failures: list[dict[str, Any]] = []
    nonfinite_records: list[dict[str, Any]] = []
    for split in SPLITS:
        coarse_y = coarse.arrays[f"y_{split}"].astype(np.float64, copy=False)
        fine_y = fine.arrays[f"y_{split}"][:, ::2, :].astype(np.float64, copy=False)
        if coarse_y.shape != fine_y.shape:
            trajectory_shape_failures.append({
                "split": split,
                "coarse_shape": list(coarse_y.shape),
                "fine_common_shape": list(fine_y.shape),
            })
            continue
        for row_index, q_row in enumerate(coarse.arrays[f"x_{split}"]):
            coarse_row = coarse_y[row_index]
            fine_row = fine_y[row_index]
            if not np.all(np.isfinite(coarse_row)) or not np.all(np.isfinite(fine_row)):
                nonfinite_records.append({"split": split, "source_row_index": int(row_index), "Q": float(q_row[0])})
    if trajectory_shape_failures:
        base_result["structural_valid"] = False
        base_result["structural_checks"]["trajectory_shapes"] = _check(
            "trajectory_shapes", False, failures=trajectory_shape_failures
        )
        base_result["numerical_metrics"] = None
        base_result["turning_point_diagnostics"] = {"available": False, "reason": "trajectory_shape_mismatch"}
        return _json_value(base_result)
    if nonfinite_records:
        base_result["structural_valid"] = False
        base_result["structural_checks"]["trajectory_finiteness"] = _check(
            "trajectory_finiteness", False, nonfinite_records=nonfinite_records
        )
        base_result["numerical_metrics"] = None
        base_result["turning_point_diagnostics"] = {"available": False, "reason": "nonfinite_trajectory"}
        base_result["anomalies"]["nonfinite_trajectory_records"] = nonfinite_records
        return _json_value(base_result)

    per_q_relative_l2: list[float] = []
    per_q_mse: list[float] = []
    per_q_records: list[dict[str, Any]] = []
    for split in SPLITS:
        coarse_y = coarse.arrays[f"y_{split}"].astype(np.float64, copy=False)
        fine_y = fine.arrays[f"y_{split}"][:, ::2, :].astype(np.float64, copy=False)
        for row_index, q_row in enumerate(coarse.arrays[f"x_{split}"]):
            coarse_row = coarse_y[row_index]
            fine_row = fine_y[row_index]
            difference = fine_row - coarse_row
            relative_l2 = float(np.linalg.norm(difference.ravel()) / (np.linalg.norm(coarse_row.ravel()) + 1e-12))
            mse = float(np.mean(difference * difference))
            per_q_relative_l2.append(relative_l2)
            per_q_mse.append(mse)
            per_q_records.append({
                "split": split,
                "source_row_index": int(row_index),
                "Q": float(q_row[0]),
                "relative_l2": relative_l2,
                "mse": mse,
            })

    fine_diagnostics = _load_solver_diagnostics(fine.directory)
    base_result["numerical_metrics"] = {
        "comparison": "fine_xyz[:, ::2, :] versus coarse_xyz",
        "per_q": per_q_records,
        "relative_l2_summary": _metric_summary(np.asarray(per_q_relative_l2, dtype=np.float64)),
        "mse_summary": _metric_summary(np.asarray(per_q_mse, dtype=np.float64)),
    }
    base_result["turning_point_diagnostics"] = {
        "available": fine_diagnostics is not None,
        "fine_solver_records": fine_diagnostics,
    } if fine_diagnostics is not None else {"available": False, "reason": "fine_solver_diagnostics_not_recorded"}
    return _json_value(base_result)


def write_consistency_result_exclusively(result: dict[str, Any], output_path: str | Path) -> None:
    """以排他方式写入 machine-readable qualification JSON。"""
    path = Path(output_path).resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    _write_json_exclusive(path, result)
