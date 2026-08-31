"""Unified server-facing Plan B resolution-range paired-truth and frozen-FNO sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_plan_b_resolution_generalization_2d import (  # noqa: E402
    build_canonical_field_from_artifact,
    compute_raw_metrics,
    run_frozen_forward,
    validate_checkpoint_contract,
)
from scripts.run_analysis_2d import load_checkpoint_2d, load_fno2d_checkpoint_model  # noqa: E402
from scripts.run_plan_b_bidirectional_reverse_2d import validate_t2399_checkpoint_contract  # noqa: E402
from src.data_generation.plan_b_paired import (  # noqa: E402
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    _json_value,
    load_dataset_artifact,
    q_identity_metadata,
)
from src.data_generation.plan_b_resolution_range import (  # noqa: E402
    RESOLUTION_RANGE_SPECS,
    ResolutionSpec,
    generate_paired_refinement_dataset,
    validate_refinement_ground_truth,
)


EXPERIMENT_TYPE = "plan_b_fixed_domain_resolution_range_sweep"
EVALUATION_T = (COARSE_N_STEPS, FINE_N_STEPS, 3598, 4797)
EVALUATION_H = {
    COARSE_N_STEPS: COARSE_STEP_SIZE,
    FINE_N_STEPS: FINE_STEP_SIZE,
    3598: RESOLUTION_RANGE_SPECS[3].step_size,
    4797: RESOLUTION_RANGE_SPECS[4].step_size,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _path_text(path: str | Path) -> str:
    return str(Path(path).resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析统一 server workflow 的显式输入与可恢复输出目录。"""

    parser = argparse.ArgumentParser(
        description=(
            "Generate and qualify endpoint-fixed Q400/T3598 and Q400/T4797 truth, "
            "then run only the new frozen theta1200/theta2399 cells and assemble a 2x4 matrix."
        )
    )
    parser.add_argument("--q400-t1200-dataset-dir", required=True, type=Path)
    parser.add_argument("--q400-t2399-dataset-dir", required=True, type=Path)
    parser.add_argument("--q400-t3598-dataset-dir", required=True, type=Path)
    parser.add_argument("--q400-t4797-dataset-dir", required=True, type=Path)
    parser.add_argument("--theta1200-checkpoint", required=True, type=Path)
    parser.add_argument("--theta2399-checkpoint", required=True, type=Path)
    parser.add_argument("--theta1200-metrics-json", required=True, type=Path)
    parser.add_argument("--bidirectional-matrix-json", required=True, type=Path)
    parser.add_argument("--theta2399-metrics-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def _manifest(
    *,
    q400_t1200_dataset_dir: str | Path,
    q400_t2399_dataset_dir: str | Path,
    generated_dirs: dict[int, str | Path],
    theta1200_checkpoint: str | Path,
    theta2399_checkpoint: str | Path,
    theta1200_metrics_json: str | Path,
    bidirectional_matrix_json: str | Path,
    theta2399_metrics_json: str | Path,
) -> dict[str, Any]:
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "anchor_dataset": _path_text(q400_t1200_dataset_dir),
        "existing_t2399_dataset": _path_text(q400_t2399_dataset_dir),
        "new_dataset_dirs": {str(key): _path_text(value) for key, value in generated_dirs.items()},
        "checkpoints": {"theta1200": _path_text(theta1200_checkpoint), "theta2399": _path_text(theta2399_checkpoint)},
        "completed_metrics": {
            "theta1200": _path_text(theta1200_metrics_json),
            "bidirectional_matrix": _path_text(bidirectional_matrix_json),
            "theta2399": _path_text(theta2399_metrics_json),
        },
        "new_refinement_specs": {str(key): asdict(value) for key, value in RESOLUTION_RANGE_SPECS.items()},
    }


def _initialize_state(output_dir: Path, manifest: dict[str, Any], resume: bool) -> tuple[Path, dict[str, Any]]:
    state_path = output_dir / "workflow_state.json"
    if output_dir.exists() and not resume:
        raise FileExistsError("Sweep output exists; use --resume only with identical provenance.")
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("--resume requires an existing workflow_state.json.")
        state = _read_json(state_path)
        if state.get("manifest") != manifest:
            raise ValueError("Resume provenance does not match the existing sweep state.")
        return state_path, state
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE, "manifest": manifest, "stages": {}}
    _write_json(state_path, state, exclusive=True)
    return state_path, state


def _update_state(path: Path, state: dict[str, Any], stage: str, value: dict[str, Any]) -> None:
    state.setdefault("stages", {})[stage] = value
    _write_json(path, state, exclusive=False)


def _same_q_field(anchor_dir: str | Path, candidate_dir: str | Path, *, candidate_name: str) -> None:
    anchor, candidate = load_dataset_artifact(anchor_dir), load_dataset_artifact(candidate_dir)
    anchor_identity, candidate_identity = q_identity_metadata(anchor.arrays), q_identity_metadata(candidate.arrays)
    if anchor_identity["source_q_identity_sha256"] != candidate_identity["source_q_identity_sha256"]:
        raise ValueError(f"{candidate_name} does not preserve the anchor Q identities and ordering.")
    if anchor_identity["canonical_q_identity_sha256"] != candidate_identity["canonical_q_identity_sha256"]:
        raise ValueError(f"{candidate_name} does not preserve the anchor canonical Q ordering.")


def _load_completed_metrics(
    theta1200_metrics_json: str | Path,
    bidirectional_matrix_json: str | Path,
    theta2399_metrics_json: str | Path,
) -> dict[str, dict[int, dict[str, Any]]]:
    """只读取既有 T1200/T2399 结果；绝不调用已有 cell 的 forward。"""

    theta1200_raw = _read_json(Path(theta1200_metrics_json))
    theta2399_raw = _read_json(Path(theta2399_metrics_json))
    matrix = _read_json(Path(bidirectional_matrix_json))
    required_1200 = {"coarse_t1200", "fine_t2399_full_grid_primary"}
    if not required_1200.issubset(theta1200_raw):
        raise ValueError("Completed theta1200 metrics lack the required T1200/T2399 cells.")
    required_2399 = {"theta2399_native_t2399", "theta2399_reverse_t1200"}
    if not required_2399.issubset(theta2399_raw):
        raise ValueError("Completed theta2399 metrics lack the required native/reverse cells.")
    rows = matrix.get("training_rows")
    if not isinstance(rows, dict) or "theta1200" not in rows or "theta2399" not in rows:
        raise ValueError("Bidirectional matrix lacks both frozen-model rows.")
    completed = {
        "theta1200": {1200: theta1200_raw["coarse_t1200"], 2399: theta1200_raw["fine_t2399_full_grid_primary"]},
        "theta2399": {1200: theta2399_raw["theta2399_reverse_t1200"], 2399: theta2399_raw["theta2399_native_t2399"]},
    }
    for model_name, values in completed.items():
        for t_value, metrics in values.items():
            matrix_metrics = rows[model_name][f"test_t{t_value}"]
            for metric_name in ("global_mse", "global_relative_l2", "mean_per_q_relative_l2"):
                if metric_name not in metrics or metric_name not in matrix_metrics:
                    raise ValueError(f"Completed {model_name}/T{t_value} metrics are incomplete.")
                if not math.isclose(float(metrics[metric_name]), float(matrix_metrics[metric_name]), rel_tol=0.0, abs_tol=1e-14):
                    raise ValueError(f"Completed metrics and bidirectional matrix disagree for {model_name}/T{t_value}.")
    return completed


def _checkpoint_payload(
    *,
    name: str,
    path: str | Path,
    device: str,
    checkpoint_loader: Callable[..., dict[str, Any]],
    model_loader: Callable[..., torch.nn.Module],
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = checkpoint_loader(Path(path), device)
    contract = validate_checkpoint_contract(checkpoint) if name == "theta1200" else validate_t2399_checkpoint_contract(checkpoint)
    model = model_loader(checkpoint, device)
    return model, contract, checkpoint


def _grid_record(spec: ResolutionSpec) -> dict[str, Any]:
    return {
        "T": spec.n_steps,
        "step_size": spec.step_size,
        "sampled_lambda_interval": [0.0, (COARSE_N_STEPS - 1) * COARSE_STEP_SIZE],
        "refinement_factor_relative_to_t1200": spec.refinement_factor,
    }


def _evaluate_new_cell(
    *,
    model_name: str,
    model: torch.nn.Module,
    contract: Any,
    checkpoint: dict[str, Any],
    field: Any,
    qualification: dict[str, Any],
    spec: ResolutionSpec,
    device: str,
    forward_runner: Callable[..., np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    started = perf_counter()
    prediction = forward_runner(model=model, contract=contract, field=field, device=device)
    metrics = compute_raw_metrics(prediction, field.canonical_truth, field.canonical_q)
    result = {
        "schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE, "status": "completed",
        "model_name": model_name,
        "checkpoint": {
            "epoch": checkpoint.get("epoch"), "model_config": contract.model_config,
            "normalization": contract.normalization_stats.to_dict(),
            "target_transform": contract.target_transform_config.to_dict(),
            "normalization_refit_during_evaluation": False,
        },
        "test_grid": _grid_record(spec),
        "ground_truth_qualification": {"structural_valid": qualification["structural_valid"], "numerical_metrics": qualification["numerical_metrics"]},
        "frozen_inference": {"forward_passes": 1, "one_shot": True, "retraining": False, "fine_tuning": False, "adaptation": "none", "normalization_refit": False, "seconds": perf_counter() - started},
        "metrics": metrics,
    }
    return _json_value(result), np.asarray(prediction, dtype=np.float32)


def _write_new_cell(output_dir: Path, result: dict[str, Any], prediction: np.ndarray, *, resume: bool, manifest: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_dir / "experiment_summary.json"
    if output_dir.exists():
        if not resume or not summary_path.is_file():
            raise FileExistsError(f"Evaluation output already exists: {output_dir}")
        existing = _read_json(summary_path)
        if existing.get("workflow_manifest") != manifest:
            raise ValueError(f"Evaluation output provenance mismatch: {output_dir}")
        return existing
    output_dir.mkdir(parents=True, exist_ok=False)
    materialized = {**result, "workflow_manifest": manifest}
    _write_json(output_dir / "metrics.json", result["metrics"], exclusive=True)
    _write_json(summary_path, materialized, exclusive=True)
    np.save(output_dir / f"{result['model_name']}_t{result['test_grid']['T']}_prediction_raw_xyz.npy", prediction)
    return materialized


def _native_analysis(rows: dict[str, dict[int, dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training_t = {"theta1200": 1200, "theta2399": 2399}
    analysis: dict[str, Any] = {}
    curve: list[dict[str, Any]] = []
    for model_name, native_t in training_t.items():
        native = rows[model_name][native_t]
        h_train = EVALUATION_H[native_t]
        per_model: dict[str, Any] = {}
        for evaluation_t in EVALUATION_T:
            metrics = rows[model_name][evaluation_t]
            item: dict[str, Any] = {
                "training_model": model_name, "training_T": native_t, "evaluation_T": evaluation_t,
                "evaluation_step_size": EVALUATION_H[evaluation_t],
                "h_train_divided_by_h_test": h_train / EVALUATION_H[evaluation_t],
                "refinement_factor_relative_to_t1200": (evaluation_t - 1) / (COARSE_N_STEPS - 1),
                "global_relative_l2": metrics["global_relative_l2"],
                "mean_per_q_relative_l2": metrics["mean_per_q_relative_l2"],
            }
            gap: dict[str, Any] = {}
            for metric_name in ("global_relative_l2", "mean_per_q_relative_l2"):
                current, baseline = float(metrics[metric_name]), float(native[metric_name])
                gap[metric_name] = {"ratio_to_native": current / baseline if baseline != 0.0 else None, "absolute_gap_to_native": current - baseline}
            item["native_normalized"] = gap
            per_model[f"T{evaluation_t}"] = item
            curve.append(item)
        analysis[model_name] = per_model
    return analysis, curve


def _matrix(rows: dict[str, dict[int, dict[str, Any]]], existing_sources: dict[str, str]) -> dict[str, Any]:
    fields = ("global_mse", "global_relative_l2", "mean_per_q_relative_l2")
    return {
        "schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE,
        "evaluation_space": "raw_physical_xyz_float64", "evaluation_T": list(EVALUATION_T),
        "training_rows": {name: {f"test_t{t_value}": rows[name][t_value] for t_value in EVALUATION_T} for name in ("theta1200", "theta2399")},
        "metric_matrices": {field: {name: {f"T{t_value}": float(rows[name][t_value][field]) for t_value in EVALUATION_T} for name in ("theta1200", "theta2399")} for field in fields},
        "existing_t1200_t2399_metrics_reused": True, "existing_metric_sources": existing_sources,
        "existing_t1200_t2399_forward_passes_rerun": 0,
    }


def run_resolution_range_workflow(
    *,
    q400_t1200_dataset_dir: str | Path,
    q400_t2399_dataset_dir: str | Path,
    q400_t3598_dataset_dir: str | Path,
    q400_t4797_dataset_dir: str | Path,
    theta1200_checkpoint: str | Path,
    theta2399_checkpoint: str | Path,
    theta1200_metrics_json: str | Path,
    bidirectional_matrix_json: str | Path,
    theta2399_metrics_json: str | Path,
    output_dir: str | Path,
    device: str,
    resume: bool = False,
    progress_every: int = 50,
    simulator: Callable[..., dict[str, object]] | None = None,
    checkpoint_loader: Callable[..., dict[str, Any]] = load_checkpoint_2d,
    model_loader: Callable[..., torch.nn.Module] = load_fno2d_checkpoint_model,
    forward_runner: Callable[..., np.ndarray] = run_frozen_forward,
) -> dict[str, Any]:
    """生成、qualification、四个新 frozen cells 与 2x4 matrix 的唯一 server workflow。"""

    generated_dirs = {3598: q400_t3598_dataset_dir, 4797: q400_t4797_dataset_dir}
    manifest = _manifest(q400_t1200_dataset_dir=q400_t1200_dataset_dir, q400_t2399_dataset_dir=q400_t2399_dataset_dir, generated_dirs=generated_dirs, theta1200_checkpoint=theta1200_checkpoint, theta2399_checkpoint=theta2399_checkpoint, theta1200_metrics_json=theta1200_metrics_json, bidirectional_matrix_json=bidirectional_matrix_json, theta2399_metrics_json=theta2399_metrics_json)
    output = Path(output_dir).resolve()
    state_path, state = _initialize_state(output, manifest, resume)
    _same_q_field(q400_t1200_dataset_dir, q400_t2399_dataset_dir, candidate_name="Existing Q400/T2399 dataset")
    completed = _load_completed_metrics(theta1200_metrics_json, bidirectional_matrix_json, theta2399_metrics_json)
    rows: dict[str, dict[int, dict[str, Any]]] = {name: dict(values) for name, values in completed.items()}
    qualifications: dict[int, dict[str, Any]] = {}
    fields: dict[int, Any] = {}
    for t_value, spec in ((3598, RESOLUTION_RANGE_SPECS[3]), (4797, RESOLUTION_RANGE_SPECS[4])):
        target = Path(generated_dirs[t_value]).resolve()
        if not target.exists():
            generator = simulator if simulator is not None else None
            kwargs: dict[str, Any] = {"source_dataset_dir": q400_t1200_dataset_dir, "output_dir": target, "spec": spec, "progress_every": progress_every}
            if generator is not None:
                kwargs["simulator"] = generator
            generate_paired_refinement_dataset(**kwargs)
        elif not resume:
            raise FileExistsError(f"Generated truth already exists outside a resume run: {target}")
        qualification = validate_refinement_ground_truth(q400_t1200_dataset_dir, target, spec=spec)
        qualification_path = output / "ground_truth_qualification" / f"t{t_value}.json"
        if qualification_path.exists():
            if not resume:
                raise FileExistsError(f"Qualification output already exists: {qualification_path}")
            recorded = _read_json(qualification_path)
            if recorded.get("fine_dataset_path") != qualification["fine_dataset_path"]:
                raise ValueError(f"Qualification provenance mismatch for T{t_value}.")
        else:
            _write_json(qualification_path, qualification, exclusive=True)
        _update_state(state_path, state, f"t{t_value}_truth_qualified", {"dataset": str(target), "structural_valid": qualification["structural_valid"]})
        if qualification.get("structural_valid") is not True:
            _update_state(state_path, state, "stopped", {"reason": f"T{t_value} structural qualification failed; frozen evaluation forbidden."})
            raise RuntimeError(f"T{t_value} structural qualification failed; frozen evaluation is forbidden.")
        qualifications[t_value] = qualification
        fields[t_value] = build_canonical_field_from_artifact(target)

    models: dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]] = {}
    for model_name, checkpoint_path in (("theta1200", theta1200_checkpoint), ("theta2399", theta2399_checkpoint)):
        models[model_name] = _checkpoint_payload(name=model_name, path=checkpoint_path, device=device, checkpoint_loader=checkpoint_loader, model_loader=model_loader)
    for t_value, spec in ((3598, RESOLUTION_RANGE_SPECS[3]), (4797, RESOLUTION_RANGE_SPECS[4])):
        for model_name, (model, contract, checkpoint) in models.items():
            evaluation_dir = output / "new_evaluations" / f"{model_name}_t{t_value}"
            reused_evaluation = evaluation_dir.exists()
            if reused_evaluation and resume:
                existing = _write_new_cell(evaluation_dir, {}, np.empty((0,)), resume=True, manifest=manifest)
                rows[model_name][t_value] = existing["metrics"]
            else:
                result, prediction = _evaluate_new_cell(model_name=model_name, model=model, contract=contract, checkpoint=checkpoint, field=fields[t_value], qualification=qualifications[t_value], spec=spec, device=device, forward_runner=forward_runner)
                materialized = _write_new_cell(evaluation_dir, result, prediction, resume=False, manifest=manifest)
                rows[model_name][t_value] = materialized["metrics"]
            _update_state(state_path, state, f"{model_name}_t{t_value}_evaluated", {"output_dir": str(evaluation_dir), "reused_on_resume": reused_evaluation})

    source_paths = {"theta1200_metrics": _path_text(theta1200_metrics_json), "bidirectional_matrix": _path_text(bidirectional_matrix_json), "theta2399_metrics": _path_text(theta2399_metrics_json)}
    matrix = {**_matrix(rows, source_paths), "workflow_manifest": manifest}
    analysis, curve = _native_analysis(rows)
    matrix_path, curve_path = output / "resolution_range_matrix.json", output / "resolution_curve.json"
    curve_document = {"schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE, "workflow_manifest": manifest, "curve": curve}
    if matrix_path.exists() or curve_path.exists():
        if not resume:
            raise FileExistsError("Sweep matrix outputs already exist; use a provenance-matched --resume run.")
        if not matrix_path.is_file() or not curve_path.is_file():
            raise FileNotFoundError("Resume requires both existing resolution-range matrix outputs.")
        if _read_json(matrix_path).get("workflow_manifest") != manifest or _read_json(curve_path).get("workflow_manifest") != manifest:
            raise ValueError("Resolution-range matrix output provenance mismatch.")
    else:
        _write_json(matrix_path, matrix, exclusive=True); _write_json(curve_path, curve_document, exclusive=True)
    final = {"schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE, "status": "completed", "manifest": manifest, "truth_qualifications": qualifications, "matrix": matrix, "native_normalized_analysis": analysis, "resolution_curve_path": str(curve_path)}
    summary_path = output / "experiment_summary.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError("Experiment summary already exists.")
        if _read_json(summary_path).get("manifest") != manifest:
            raise ValueError("Experiment summary provenance mismatch.")
    else:
        _write_json(summary_path, final, exclusive=True)
    _update_state(state_path, state, "completed", {"matrix": str(matrix_path), "summary": str(summary_path)})
    return _json_value(final)


def main() -> None:
    args = parse_args()
    result = run_resolution_range_workflow(
        q400_t1200_dataset_dir=args.q400_t1200_dataset_dir, q400_t2399_dataset_dir=args.q400_t2399_dataset_dir,
        q400_t3598_dataset_dir=args.q400_t3598_dataset_dir, q400_t4797_dataset_dir=args.q400_t4797_dataset_dir,
        theta1200_checkpoint=args.theta1200_checkpoint, theta2399_checkpoint=args.theta2399_checkpoint,
        theta1200_metrics_json=args.theta1200_metrics_json, bidirectional_matrix_json=args.bidirectional_matrix_json,
        theta2399_metrics_json=args.theta2399_metrics_json, output_dir=args.output_dir, device=str(args.device),
        resume=bool(args.resume), progress_every=int(args.progress_every),
    )
    print("Plan B resolution-range sweep workflow completed.")
    print(f"Output directory: {args.output_dir}")
    print(f"Matrix path: {args.output_dir / 'resolution_range_matrix.json'}")
    print(f"Resolution cells: {', '.join(str(item) for item in result['matrix']['evaluation_T'])}")


if __name__ == "__main__":
    main()
