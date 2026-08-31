"""Plan B endpoint-fixed resolution-range paired truth utilities.

该模块服务于 T1200 anchor 到整数 interval-refinement grids 的独立 sweep；
不改变 Protocol v1 的 T2399 专用实现，也不允许 Q 重采样或失败样本替换。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data_generation.dataset_builder import build_initial_state, build_kerr_params
from src.data_generation.orbit_solver_second_order import simulate_one_orbit_second_order
from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FIXED_PARAMETER_KEYS,
    SPLITS,
    DatasetArtifact,
    _fixed_params,
    _integration_metadata,
    _json_value,
    _load_solver_diagnostics,
    _metric_summary,
    _solver_provenance,
    build_lambda_grid,
    load_dataset_artifact,
    q_identity_metadata,
)
from src.data_generation.validity import validate_single_sample_hard_constraints


GRID_ATOL = 1e-12


def _grid_equal(left: np.ndarray, right: np.ndarray, *, rtol: float = 0.0, atol: float = GRID_ATOL) -> bool:
    """先检查 shape，再以明确容差比较浮点 lambda grids。"""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    return left_values.shape == right_values.shape and bool(np.allclose(left_values, right_values, rtol=rtol, atol=atol))

@dataclass(frozen=True)
class ResolutionSpec:
    """一个相对于 T1200 anchor 的 endpoint-preserving 整数 refinement grid。"""

    refinement_factor: int
    n_steps: int
    step_size: float

    @property
    def interval_count(self) -> int:
        return self.n_steps - 1


def endpoint_refinement_spec(refinement_factor: int) -> ResolutionSpec:
    """构造保持 sampled endpoint 的整数区间细化规格。"""

    if not isinstance(refinement_factor, int) or refinement_factor < 1:
        raise ValueError("refinement_factor must be an integer greater than or equal to 1.")
    return ResolutionSpec(
        refinement_factor=refinement_factor,
        n_steps=(COARSE_N_STEPS - 1) * refinement_factor + 1,
        step_size=COARSE_STEP_SIZE / refinement_factor,
    )


RESOLUTION_RANGE_SPECS = {
    3: endpoint_refinement_spec(3),
    4: endpoint_refinement_spec(4),
}


def validate_resolution_spec(spec: ResolutionSpec) -> np.ndarray:
    """验证 endpoint 与 common-node nesting；浮点 grid 比较使用严格容差。"""

    expected = endpoint_refinement_spec(spec.refinement_factor)
    if spec.n_steps != expected.n_steps or not math.isclose(
        spec.step_size, expected.step_size, rel_tol=0.0, abs_tol=GRID_ATOL
    ):
        raise ValueError("Resolution spec is not an endpoint-preserving T1200 integer refinement.")
    coarse = build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
    fine = build_lambda_grid(spec.n_steps, spec.step_size)
    if not math.isclose(float(fine[-1]), float(coarse[-1]), rel_tol=0.0, abs_tol=GRID_ATOL):
        raise ValueError("Resolution spec changes the sampled physical endpoint.")
    if not _grid_equal(fine[::spec.refinement_factor], coarse, rtol=0.0, atol=GRID_ATOL):
        raise ValueError("Resolution spec does not preserve coarse nodes as fine-grid nodes.")
    return fine


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _diagnostics_record(orbit: dict[str, object]) -> dict[str, Any]:
    diagnostics = orbit.get("diagnostics")
    return {"available": False} if diagnostics is None else {"available": True, **_json_value(asdict(diagnostics))}


def generate_paired_refinement_dataset(
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    spec: ResolutionSpec,
    simulator=simulate_one_orbit_second_order,
    progress_every: int = 50,
) -> dict[str, Any]:
    """逐 row replay anchor Q，生成一个新 refinement truth；从不补样。"""

    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")
    fine_grid = validate_resolution_spec(spec)
    source = load_dataset_artifact(source_dataset_dir)
    source_steps, source_h, source_endpoint = _integration_metadata(source.metadata)
    expected_coarse = build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE)
    if source_steps != COARSE_N_STEPS or not math.isclose(source_h, COARSE_STEP_SIZE, rel_tol=0.0, abs_tol=GRID_ATOL):
        raise ValueError("Resolution-range replay requires the T1200/h=0.005 anchor dataset.")
    if not _grid_equal(source.arrays["lambda_grid"], expected_coarse, rtol=0.0, atol=GRID_ATOL):
        raise ValueError("Anchor lambda grid is not the required T1200 grid.")
    if not math.isclose(source_endpoint, float(fine_grid[-1]), rel_tol=0.0, abs_tol=GRID_ATOL):
        raise ValueError("Source and refinement sampled endpoints differ.")

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing paired refinement dataset: {output}")
    output.mkdir(parents=True, exist_ok=False)
    fixed_params = _fixed_params(source.metadata)
    if tuple(fixed_params) != FIXED_PARAMETER_KEYS:
        raise ValueError("Source fixed-parameter contract is incomplete.")
    source_identity = q_identity_metadata(source.arrays)
    payload: dict[str, Any] = {"vary_params_order": source.arrays["vary_params_order"].copy(), "lambda_grid": fine_grid}
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    success_count = 0
    processed = 0
    kerr_params = build_kerr_params(fixed_params)
    initial_state = build_initial_state(fixed_params)

    for split in SPLITS:
        q_values = source.arrays[f"x_{split}"].copy()
        trajectories = np.full((q_values.shape[0], spec.n_steps, 3), np.nan, dtype=np.float64)
        success_mask = np.zeros(q_values.shape[0], dtype=bool)
        payload[f"x_{split}"] = q_values
        payload[f"y_{split}"] = trajectories
        payload[f"success_mask_{split}"] = success_mask
        for row_index, q_row in enumerate(q_values):
            q_value = float(q_row[0])
            processed += 1
            try:
                validate_single_sample_hard_constraints({"Q": q_value}, fixed_params)
                orbit = simulator(p=kerr_params, init=initial_state, Q=q_value, n_steps=spec.n_steps, step_size=spec.step_size)
                orbit_grid = np.asarray(orbit["lambda_grid"], dtype=np.float64)
                orbit_xyz = np.asarray(orbit["xyz"], dtype=np.float64)
                if not _grid_equal(orbit_grid, fine_grid, rtol=0.0, atol=GRID_ATOL):
                    raise RuntimeError("Solver returned a lambda grid inconsistent with the requested refinement.")
                if orbit_xyz.shape != (spec.n_steps, 3) or not np.all(np.isfinite(orbit_xyz)):
                    raise RuntimeError("Solver returned invalid xyz trajectory data.")
                trajectories[row_index] = orbit_xyz
                success_mask[row_index] = True
                success_count += 1
                diagnostics.append({"split": split, "source_row_index": row_index, "Q": q_value, "success": True, "diagnostics": _diagnostics_record(orbit)})
            except Exception as error:  # noqa: BLE001 - source identities must remain intact after any failure.
                failure = {"split": split, "source_row_index": row_index, "Q": q_value, "error_type": type(error).__name__, "error": str(error), "replacement_attempted": False}
                failures.append(failure)
                diagnostics.append({**failure, "success": False, "diagnostics": {"available": False}})
            if processed % progress_every == 0 or processed == source_identity["source_q_count"]:
                print(f"Paired refinement replay progress: {processed}/{source_identity['source_q_count']}")

    generated_identity = q_identity_metadata(payload)
    complete = bool(success_count == source_identity["source_q_count"] and not failures and source_identity["source_q_identity_sha256"] == generated_identity["source_q_identity_sha256"])
    task_spec = json.loads(json.dumps(_json_value(source.metadata["task_spec"])))
    task_spec["n_steps"] = spec.n_steps
    task_spec["step_size"] = spec.step_size
    provenance = _solver_provenance(source.metadata)
    task_spec.setdefault("metadata", {})["orbit_solver"] = provenance["orbit_solver"]
    task_spec["metadata"]["orbit_solver_version"] = provenance["orbit_solver_version"]
    metadata = {
        "schema_version": "1.0",
        "experiment_type": "plan_b_resolution_range_paired_replay_generation",
        "task_spec": task_spec,
        "integration": {"n_steps": spec.n_steps, "step_size": spec.step_size, "lambda_min": float(fine_grid[0]), "lambda_max": float(fine_grid[-1])},
        "paired_replay": {
            "source_dataset_identifier": source.metadata.get("task_name", str(source.directory)),
            "source_dataset_path": str(source.directory),
            "source_q_count": source_identity["source_q_count"],
            "source_q_identity": source_identity,
            "generated_q_identity": generated_identity,
            "source_and_generated_raw_order_match": source_identity["source_q_identity_sha256"] == generated_identity["source_q_identity_sha256"],
            "canonical_ordering": "stable_ascending_Q",
            "refinement_factor_relative_to_t1200": spec.refinement_factor,
            "fixed_kerr_parameters": {key: fixed_params[key] for key in ("M", "a", "E", "Lz")},
            "initial_conditions": {key: fixed_params[key] for key in ("r0", "theta0", "phi0", "sign_r", "sign_th")},
            "solver_provenance": provenance,
            "success_count": success_count,
            "failure_count": len(failures),
            "failed_q_values": [item["Q"] for item in failures],
            "paired_completeness": complete,
            "replacement_policy": "forbidden",
            "target_success_replacement_used": False,
        },
        "generation_status": {"completed": complete, "success_count": success_count, "failure_count": len(failures), "paired_completeness": complete},
    }
    np.savez_compressed(output / "dataset.npz", **payload)
    _write_json_exclusive(output / "meta.json", metadata)
    _write_json_exclusive(output / "failed_samples.json", failures)
    _write_json_exclusive(output / "solver_diagnostics.json", diagnostics)
    return _json_value(metadata)


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **_json_value(details)}


def validate_refinement_ground_truth(
    coarse_dataset_dir: str | Path,
    fine_dataset_dir: str | Path,
    *,
    spec: ResolutionSpec,
) -> dict[str, Any]:
    """在任何 evaluation 前验证 paired structure；失败时不计算数值指标。"""

    expected_grid = validate_resolution_spec(spec)
    coarse = load_dataset_artifact(coarse_dataset_dir)
    fine = load_dataset_artifact(fine_dataset_dir)
    coarse_steps, coarse_h, coarse_endpoint = _integration_metadata(coarse.metadata)
    fine_steps, fine_h, fine_endpoint = _integration_metadata(fine.metadata)
    coarse_identity, fine_identity = q_identity_metadata(coarse.arrays), q_identity_metadata(fine.arrays)
    replay = fine.metadata.get("paired_replay", {})
    expected_source_path = str(coarse.directory)
    checks: dict[str, dict[str, Any]] = {
        "q_count": _check(coarse_identity["source_q_count"] == fine_identity["source_q_count"], coarse_count=coarse_identity["source_q_count"], fine_count=fine_identity["source_q_count"]),
        "q_values_and_order": _check(all(np.array_equal(coarse.arrays[f"x_{split}"], fine.arrays[f"x_{split}"]) for split in SPLITS), coarse_q_identity_sha256=coarse_identity["source_q_identity_sha256"], fine_q_identity_sha256=fine_identity["source_q_identity_sha256"]),
        "canonical_q_identity": _check(coarse_identity["canonical_q_identity_sha256"] == fine_identity["canonical_q_identity_sha256"]),
        "fixed_physics": _check(_fixed_params(coarse.metadata) == _fixed_params(fine.metadata)),
        "solver_provenance": _check(_solver_provenance(coarse.metadata) == _solver_provenance(fine.metadata)),
        "coarse_anchor_grid": _check(coarse_steps == COARSE_N_STEPS and math.isclose(coarse_h, COARSE_STEP_SIZE, rel_tol=0.0, abs_tol=GRID_ATOL) and _grid_equal(coarse.arrays["lambda_grid"], build_lambda_grid(COARSE_N_STEPS, COARSE_STEP_SIZE), rtol=0.0, atol=GRID_ATOL)),
        "fine_refinement_grid": _check(fine_steps == spec.n_steps and math.isclose(fine_h, spec.step_size, rel_tol=0.0, abs_tol=GRID_ATOL) and _grid_equal(fine.arrays["lambda_grid"], expected_grid, rtol=0.0, atol=GRID_ATOL)),
        "sampled_endpoints": _check(math.isclose(coarse_endpoint, fine_endpoint, rel_tol=0.0, abs_tol=GRID_ATOL) and math.isclose(fine_endpoint, float(expected_grid[-1]), rel_tol=0.0, abs_tol=GRID_ATOL)),
        "common_node_grid": _check(_grid_equal(fine.arrays["lambda_grid"][::spec.refinement_factor], coarse.arrays["lambda_grid"], rtol=0.0, atol=GRID_ATOL)),
        "paired_completeness": _check(isinstance(replay, dict) and replay.get("paired_completeness") is True and replay.get("target_success_replacement_used") is False and not fine.failed_samples, reported_value=replay.get("paired_completeness") if isinstance(replay, dict) else None),
        "source_dataset_provenance": _check(isinstance(replay, dict) and replay.get("source_dataset_path") == expected_source_path, expected_source_dataset_path=expected_source_path, reported_source_dataset_path=replay.get("source_dataset_path") if isinstance(replay, dict) else None),
    }
    result: dict[str, Any] = {
        "schema_version": "1.0", "experiment_type": "plan_b_resolution_range_ground_truth_qualification",
        "coarse_dataset_path": str(coarse.directory), "fine_dataset_path": str(fine.directory),
        "refinement_factor_relative_to_t1200": spec.refinement_factor,
        "structural_valid": all(item["passed"] for item in checks.values()), "structural_checks": checks,
        "anomalies": {"coarse_failed_samples": coarse.failed_samples, "fine_failed_samples": fine.failed_samples},
    }
    if not result["structural_valid"]:
        result["numerical_metrics"] = None
        result["turning_point_diagnostics"] = {"available": False, "reason": "structural_validation_failed"}
        return _json_value(result)
    per_q_rel: list[float] = []
    per_q_mse: list[float] = []
    per_q: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    for split in SPLITS:
        coarse_y = np.asarray(coarse.arrays[f"y_{split}"], dtype=np.float64)
        fine_y = np.asarray(fine.arrays[f"y_{split}"][:, ::spec.refinement_factor, :], dtype=np.float64)
        if coarse_y.shape != fine_y.shape:
            result["structural_valid"] = False
            result["structural_checks"]["trajectory_shapes"] = _check(False, split=split, coarse_shape=list(coarse_y.shape), fine_common_shape=list(fine_y.shape))
            result["numerical_metrics"] = None
            result["turning_point_diagnostics"] = {"available": False, "reason": "trajectory_shape_mismatch"}
            return _json_value(result)
        for index, q_row in enumerate(coarse.arrays[f"x_{split}"]):
            coarse_row, fine_row = coarse_y[index], fine_y[index]
            if not np.all(np.isfinite(coarse_row)) or not np.all(np.isfinite(fine_row)):
                nonfinite.append({"split": split, "source_row_index": index, "Q": float(q_row[0])})
                continue
            delta = fine_row - coarse_row
            rel = float(np.linalg.norm(delta.ravel()) / (np.linalg.norm(coarse_row.ravel()) + 1e-12))
            mse = float(np.mean(delta * delta))
            per_q_rel.append(rel); per_q_mse.append(mse)
            per_q.append({"split": split, "source_row_index": index, "Q": float(q_row[0]), "relative_l2": rel, "mse": mse})
    if nonfinite:
        result["structural_valid"] = False
        result["structural_checks"]["trajectory_finiteness"] = _check(False, nonfinite_records=nonfinite)
        result["anomalies"]["nonfinite_trajectory_records"] = nonfinite
        result["numerical_metrics"] = None
        result["turning_point_diagnostics"] = {"available": False, "reason": "nonfinite_trajectory"}
        return _json_value(result)
    result["numerical_metrics"] = {"comparison": f"fine_xyz[:, ::{spec.refinement_factor}, :] versus coarse_xyz", "per_q": per_q, "relative_l2_summary": _metric_summary(np.asarray(per_q_rel)), "mse_summary": _metric_summary(np.asarray(per_q_mse))}
    diagnostics = _load_solver_diagnostics(fine.directory)
    result["turning_point_diagnostics"] = {"available": True, "fine_solver_records": diagnostics} if diagnostics is not None else {"available": False, "reason": "fine_solver_diagnostics_not_recorded"}
    return _json_value(result)
