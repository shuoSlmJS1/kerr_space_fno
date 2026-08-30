"""R3-B1：同一 validation-Q 的七长度冻结响应开发诊断。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_formal_length_extrapolation_2d as formal  # noqa: E402
from scripts import evaluate_formal_length_extrapolation_r3_2d as r3formal  # noqa: E402
from scripts import evaluate_r1_validation_length_response_2d as r1response  # noqa: E402
from scripts.run_analysis_2d import load_checkpoint_2d  # noqa: E402

DEFAULT_LENGTHS = (600, 700, 800, 900, 1000, 1100, 1200)
OUTPUT_FILENAMES = (
    "r3_validation_length_response_summary.json",
    "r3_validation_length_response_by_length.csv",
    "r3_validation_length_response_per_q.csv",
)
BY_LENGTH_FIELDS = (
    "total_length",
    "gradient_seen",
    "checkpoint_selection_seen",
    "length_status",
    "mse",
    "global_relative_l2",
    "mean_per_q_relative_l2",
    "median_per_q_relative_l2",
    "p95_per_q_relative_l2",
    "max_per_q_relative_l2",
)
PER_Q_FIELDS = (
    "total_length",
    "Q",
    "gradient_seen",
    "checkpoint_selection_seen",
    "mse",
    "relative_l2",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 R3 validation-Q 七长度开发诊断参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen R3 checkpoint on validation-Q strict prefixes. "
            "This is a development diagnostic, not formal long-domain test evidence."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def validate_requested_lengths(
    lengths: Sequence[int],
    *,
    source_length: int,
) -> tuple[int, ...]:
    """验证唯一 T1200 source 内的严格 prefix 长度。"""
    requested = tuple(int(value) for value in lengths)
    if not requested:
        raise ValueError("lengths must not be empty.")
    if any(value <= 0 for value in requested):
        raise ValueError("lengths must contain strictly positive values.")
    if len(set(requested)) != len(requested):
        raise ValueError("lengths must be unique.")
    if any(value > int(source_length) for value in requested):
        raise ValueError(f"lengths cannot exceed source length {source_length}.")
    return tuple(sorted(requested))


def verify_r3_validation_provenance(
    *,
    checkpoint: Mapping[str, Any],
    task_name: str,
) -> r3formal.R3CheckpointProvenance:
    """验证 R3 provenance 与唯一 validation source 一致，禁止臆造 exposure 标签。"""
    provenance = r3formal.validate_r3_checkpoint_provenance(checkpoint)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("R3 checkpoint has no config mapping.")
    if str(config.get("source_task")) != str(task_name):
        raise ValueError("R3 checkpoint source_task does not match the requested validation source task.")
    selection = config.get("checkpoint_selection")
    if not isinstance(selection, Mapping) or selection.get("split") != "validation_Q_only":
        raise ValueError("R3 checkpoint lacks validation-Q-only checkpoint-selection provenance.")
    if tuple(int(value) for value in selection.get("lengths", ())) != provenance.validation_lengths:
        raise ValueError("R3 checkpoint selection lengths are inconsistent with validation_lengths provenance.")
    return provenance


def strict_prefix_field(
    field: r1response.ValidationQField,
    total_length: int,
) -> r1response.ValidationQField:
    """构造同一 validation-Q identity 的严格 raw float64 prefix field。"""
    lambda_prefix, truth_prefix = r1response.strict_prefix(field, total_length)
    return r1response.ValidationQField(
        task_name=field.task_name,
        canonical_q=np.asarray(field.canonical_q, dtype=np.float64),
        canonical_truth=np.asarray(truth_prefix, dtype=np.float64),
        lambda_grid=np.asarray(lambda_prefix, dtype=np.float64),
        canonical_to_source_index=np.asarray(field.canonical_to_source_index, dtype=np.int64),
    )


def _length_status(*, total_length: int, training_lengths: set[int], validation_lengths: set[int]) -> str:
    """为每个长度给出不误称为 untouched test 的 provenance 标签。"""
    gradient_seen = total_length in training_lengths
    selection_seen = total_length in validation_lengths
    if gradient_seen and selection_seen:
        return "gradient_seen_and_checkpoint_selection_seen"
    if gradient_seen:
        return "gradient_seen"
    if selection_seen:
        return "non_gradient_validation_length"
    return "not_recorded_in_training_or_checkpoint_selection"


def _per_q_rows(
    *,
    total_length: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    q_values: np.ndarray,
    gradient_seen: bool,
    checkpoint_selection_seen: bool,
) -> list[dict[str, Any]]:
    """只约化保存每条 validation-Q 的当前长度 raw 指标。"""
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if prediction64.shape != truth64.shape or prediction64.shape[0] != q64.size:
        raise ValueError("Per-Q metric inputs must share canonical validation-Q shape.")
    mse = np.mean((prediction64 - truth64) ** 2, axis=(1, 2))
    relative = np.asarray(
        [formal._relative_l2(prediction64[index], truth64[index]) for index in range(q64.size)],
        dtype=np.float64,
    )
    return [
        {
            "total_length": int(total_length),
            "Q": float(q64[index]),
            "gradient_seen": bool(gradient_seen),
            "checkpoint_selection_seen": bool(checkpoint_selection_seen),
            "mse": float(mse[index]),
            "relative_l2": float(relative[index]),
        }
        for index in range(q64.size)
    ]


def evaluate_length_response(
    *,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    field: r1response.ValidationQField,
    provenance: r3formal.R3CheckpointProvenance,
    lengths: Sequence[int],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """对每个长度执行一次 R3 冻结前向，保持相同 validation-Q identities。"""
    requested = validate_requested_lengths(lengths, source_length=int(field.lambda_grid.size))
    training_lengths = set(int(value) for value in provenance.training_lengths)
    validation_lengths = set(int(value) for value in provenance.validation_lengths)
    model.eval()
    rows: list[dict[str, Any]] = []
    per_q_rows: list[dict[str, Any]] = []
    for total_length in requested:
        prefix = strict_prefix_field(field, total_length)
        prediction, runtime = r3formal.run_frozen_inference(
            model=model,
            checkpoint=checkpoint,
            field=prefix,
            provenance=provenance,
            device=device,
        )
        metrics = formal.compute_region_metrics(
            prediction,
            prefix.canonical_truth,
            prefix.canonical_q,
        )
        gradient_seen = total_length in training_lengths
        selection_seen = total_length in validation_lengths
        rows.append(
            {
                "total_length": int(total_length),
                "gradient_seen": bool(gradient_seen),
                "checkpoint_selection_seen": bool(selection_seen),
                "length_status": _length_status(
                    total_length=total_length,
                    training_lengths=training_lengths,
                    validation_lengths=validation_lengths,
                ),
                "mse": float(metrics["mse"]),
                "global_relative_l2": float(metrics["global_relative_l2"]),
                "mean_per_q_relative_l2": float(metrics["mean_per_q_relative_l2"]),
                "median_per_q_relative_l2": float(metrics["median_per_q_relative_l2"]),
                "p95_per_q_relative_l2": float(metrics["p95_per_q_relative_l2"]),
                "max_per_q_relative_l2": float(metrics["max_per_q_relative_l2"]),
                "worst_q_index": int(metrics["worst_q_index"]),
                "worst_q_value": float(metrics["worst_q_value"]),
                "component_metrics": metrics["components"],
                "r3_spectral_runtime": runtime,
                "frozen_forward_passes": 1,
            }
        )
        per_q_rows.extend(
            _per_q_rows(
                total_length=total_length,
                prediction=prediction,
                truth=prefix.canonical_truth,
                q_values=prefix.canonical_q,
                gradient_seen=gradient_seen,
                checkpoint_selection_seen=selection_seen,
            )
        )
    return rows, per_q_rows


def sawtooth_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按已定义的局部线性中点残差量化离散长度 sawtooth。"""
    values = {int(row["total_length"]): float(row["mean_per_q_relative_l2"]) for row in rows}
    residuals: dict[str, float | None] = {}
    for intermediate, left, right in ((700, 600, 800), (900, 800, 1000), (1100, 1000, 1200)):
        key = f"interpolation_residual_T{intermediate}"
        residuals[key] = (
            values[intermediate] - 0.5 * (values[left] + values[right])
            if {intermediate, left, right}.issubset(values)
            else None
        )
    residual_values = [abs(value) for value in residuals.values() if value is not None]
    ordered_lengths = sorted(values)
    adjacent = {
        f"abs_E{right}_minus_E{left}": abs(values[right] - values[left])
        for left, right in zip(ordered_lengths[:-1], ordered_lengths[1:])
    }
    adjacent_values = list(adjacent.values())
    gradient_values = [float(row["mean_per_q_relative_l2"]) for row in rows if bool(row["gradient_seen"])]
    non_gradient_values = [
        float(row["mean_per_q_relative_l2"])
        for row in rows
        if not bool(row["gradient_seen"]) and bool(row["checkpoint_selection_seen"])
    ]
    gradient_mean = float(np.mean(gradient_values)) if gradient_values else None
    non_gradient_mean = float(np.mean(non_gradient_values)) if non_gradient_values else None
    return {
        "primary_metric": "mean_per_q_relative_l2",
        "interpolation_residual_definitions": {
            "T700": "E700 - 0.5 * (E600 + E800)",
            "T900": "E900 - 0.5 * (E800 + E1000)",
            "T1100": "E1100 - 0.5 * (E1000 + E1200)",
        },
        **residuals,
        "mean_absolute_interpolation_residual": float(np.mean(residual_values)) if residual_values else None,
        "max_absolute_interpolation_residual": float(np.max(residual_values)) if residual_values else None,
        "gradient_seen_mean": gradient_mean,
        "non_gradient_validation_mean": non_gradient_mean,
        "non_gradient_minus_gradient_gap": (
            float(non_gradient_mean - gradient_mean)
            if gradient_mean is not None and non_gradient_mean is not None
            else None
        ),
        "adjacent_absolute_changes": adjacent,
        "mean_adjacent_absolute_change": float(np.mean(adjacent_values)) if adjacent_values else None,
        "max_adjacent_absolute_change": float(np.max(adjacent_values)) if adjacent_values else None,
        "interpretation_boundary": (
            "descriptive_development_summary; non_gradient_validation_lengths_were_used_for_checkpoint_selection; "
            "no_memorization_or_continuous_invariance_claim"
        ),
    }


def build_summary(
    *,
    task_name: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    field: r1response.ValidationQField,
    provenance: r3formal.R3CheckpointProvenance,
    requested_lengths: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, Any]:
    """构造不含 prediction、truth 或 hidden arrays 的紧凑开发诊断摘要。"""
    config = checkpoint["config"]
    return {
        "schema_version": "1.0",
        "evaluation_type": "r3_validation_length_response_diagnostic",
        "scientific_status": "development_diagnostic",
        "formal_test_evidence": False,
        "split": "val",
        "task_name": str(task_name),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_task": str(config["source_task"]),
        "run_name": config.get("run_name"),
        "validation_q_count": int(field.canonical_q.size),
        "canonical_q_order": "stable_ascending",
        "same_validation_q_across_lengths": True,
        "strict_prefixes_from_single_t1200_source": True,
        "lengths": [int(value) for value in requested_lengths],
        "primary_metric": "mean_per_q_relative_l2",
        "checkpoint_training_lengths": list(provenance.training_lengths),
        "checkpoint_validation_lengths": list(provenance.validation_lengths),
        "coordinate_representation": {
            "channel_order": list(provenance.input_channel_order),
            "absolute_lambda_input": False,
            "relative_coordinate_definition": "s = lambda / L",
            "domain_length_definition": "L = N * delta_lambda (DFT logical period)",
            "L_ref": float(provenance.reference_domain_length),
        },
        "spectral_parameterization": {
            "type": "physical_frequency_anchor_interpolation",
            "physical_frequency_formula": "k / (N * delta_lambda)",
            "training_delta_lambda": float(provenance.delta_lambda),
            "anchor_frequency_values": list(provenance.anchor_frequencies),
            "global_fft_structure_unchanged": True,
            "physical_bandwidth_shrinkage_repaired": False,
        },
        "normalization_provenance": {
            "source": "checkpoint_only_no_refit",
            "input_policy": dict(provenance.input_normalization_policy),
            "target_transform": provenance.target_transform,
        },
        "frozen_inference": {
            "one_direct_forward_per_total_length": True,
            "frozen_forward_passes": int(len(rows)),
            "optimizer": False,
            "scheduler": False,
            "backward": False,
            "training": False,
            "adaptation": False,
            "teacher_forcing": False,
            "autoregression": False,
            "prediction_feedback": False,
        },
        "results_by_length": {f"T{int(row['total_length'])}": dict(row) for row in rows},
        "sawtooth_diagnostic": sawtooth_summary(rows),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "git_commit": r1response._git_commit(),
        },
        "output_files": list(OUTPUT_FILENAMES),
    }


def _json_value(value: Any) -> Any:
    """递归转换 NumPy 值，确保 JSON 不写入数组文件。"""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    per_q_rows: Sequence[Mapping[str, Any]],
) -> None:
    """拒绝覆盖，且仅写三份 compact JSON/CSV artifact。"""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(dict(summary)), handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    for filename, fieldnames, source_rows in (
        (OUTPUT_FILENAMES[1], BY_LENGTH_FIELDS, rows),
        (OUTPUT_FILENAMES[2], PER_Q_FIELDS, per_q_rows),
    ):
        with (output_dir / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in source_rows:
                writer.writerow({name: row.get(name) for name in fieldnames})


def main(argv: Sequence[str] | None = None) -> None:
    """执行 validation-Q only R3 七长度冻结开发诊断。"""
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")
    field = r1response.load_validation_source(str(args.task_name))
    requested_lengths = validate_requested_lengths(args.lengths, source_length=int(field.lambda_grid.size))
    checkpoint = load_checkpoint_2d(checkpoint_path=args.checkpoint_path, device=str(args.device))
    provenance = verify_r3_validation_provenance(checkpoint=checkpoint, task_name=str(args.task_name))
    model = r3formal.build_r3_model(checkpoint, provenance, str(args.device))
    rows, per_q_rows = evaluate_length_response(
        model=model,
        checkpoint=checkpoint,
        field=field,
        provenance=provenance,
        lengths=requested_lengths,
        device=str(args.device),
    )
    summary = build_summary(
        task_name=str(args.task_name),
        checkpoint_path=args.checkpoint_path,
        checkpoint=checkpoint,
        field=field,
        provenance=provenance,
        requested_lengths=requested_lengths,
        rows=rows,
        device=str(args.device),
    )
    write_outputs(output_dir=args.output_dir, summary=summary, rows=rows, per_q_rows=per_q_rows)
    print("R3 validation-Q length-response diagnostic completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
