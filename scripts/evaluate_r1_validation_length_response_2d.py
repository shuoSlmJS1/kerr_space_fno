"""R1 validation-Q 七长度响应的冻结 checkpoint 开发诊断。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.evaluate_formal_length_extrapolation_2d import (  # noqa: E402
    compute_region_metrics,
)
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)
from src.common.paths import get_task_dataset_npz_path  # noqa: E402
from src.training.fno2d.dataset_loader_2d import (  # noqa: E402
    load_task_meta,
    load_task_vary_params_order,
    validate_single_param_task,
    validate_task_generation_meta,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    transform_output_field,
)


SOURCE_LENGTH = 1200
DEFAULT_LENGTHS = (600, 700, 800, 900, 1000, 1100, 1200)
OUTPUT_FILENAMES = (
    "r1_validation_length_response_summary.json",
    "r1_validation_length_response.csv",
)
CSV_FIELDS = (
    "total_length",
    "seen_during_training",
    "used_for_checkpoint_selection",
    "normalized_space_mse",
    "raw_global_relative_l2",
    "raw_mean_per_q_relative_l2",
)


@dataclass(frozen=True)
class ValidationQField:
    """只包含 validation split 且已 canonicalized 的 raw field。"""

    task_name: str
    canonical_q: np.ndarray
    canonical_truth: np.ndarray
    lambda_grid: np.ndarray
    canonical_to_source_index: np.ndarray


def _json_value(value: Any) -> Any:
    """将 NumPy 标量和数组递归转换为 JSON 安全值。"""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _git_commit() -> str | None:
    """读取本地 HEAD，不执行任何远程 Git 操作。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def validate_requested_lengths(
    lengths: Sequence[int],
    source_length: int = SOURCE_LENGTH,
) -> tuple[int, ...]:
    """检查诊断长度，且不允许超出唯一 T1200 source。"""
    normalized = tuple(int(value) for value in lengths)
    if not normalized:
        raise ValueError("lengths must not be empty.")
    if any(value <= 0 for value in normalized):
        raise ValueError("lengths must contain strictly positive values.")
    if any(value > source_length for value in normalized):
        raise ValueError(f"lengths cannot exceed source length {source_length}.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("lengths must be unique.")
    return tuple(sorted(normalized))


def build_validation_field(
    *,
    task_name: str,
    q_raw: np.ndarray,
    truth_raw: np.ndarray,
    lambda_grid: np.ndarray,
) -> ValidationQField:
    """稳定排序 validation Q，并保持 raw float64 truth 的相同置换。"""
    q = np.asarray(q_raw)
    truth = np.asarray(truth_raw)
    lambda_values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[1] != 1:
        raise ValueError(f"x_val must have shape [N,1], got {q.shape}.")
    if truth.ndim != 3 or truth.shape != (q.shape[0], lambda_values.size, 3):
        raise ValueError(
            "y_val must have shape [N,source_length,3] compatible with x_val and lambda_grid."
        )
    if truth.dtype != np.float64:
        raise ValueError(f"y_val must retain raw float64 truth, got {truth.dtype}.")
    if lambda_values.size != SOURCE_LENGTH:
        raise ValueError(
            f"R1 validation response requires a T{SOURCE_LENGTH} source, "
            f"got {lambda_values.size} lambda points."
        )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(truth)):
        raise ValueError("Validation Q and raw truth must be finite.")
    if not np.all(np.diff(lambda_values) > 0.0):
        raise ValueError("lambda_grid must be one-dimensional and strictly increasing.")
    canonical_to_source = np.argsort(q[:, 0], kind="stable").astype(np.int64)
    canonical_q = np.asarray(q[canonical_to_source, 0], dtype=np.float64)
    if canonical_q.size == 0 or not np.all(np.diff(canonical_q) > 0.0):
        raise ValueError("validation Q values must be unique and strictly ascending after canonicalization.")
    return ValidationQField(
        task_name=str(task_name),
        canonical_q=canonical_q,
        canonical_truth=np.asarray(truth[canonical_to_source], dtype=np.float64),
        lambda_grid=lambda_values,
        canonical_to_source_index=canonical_to_source,
    )


def load_validation_source(task_name: str) -> ValidationQField:
    """只读取并使用 source dataset 的 x_val、y_val 与 lambda_grid。"""
    meta = load_task_meta(task_name)
    validate_task_generation_meta(task_name=task_name, meta=meta)
    param_name = validate_single_param_task(load_task_vary_params_order(task_name))
    if param_name != "Q":
        raise ValueError(f"R1 validation response requires a Q-only task, got {param_name!r}.")
    dataset_path = get_task_dataset_npz_path(task_name)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {"x_val", "y_val", "lambda_grid"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Dataset is missing validation diagnostic fields: {missing}")
        q_val = np.asarray(archive["x_val"])
        truth_val = np.asarray(archive["y_val"])
        lambda_grid = np.asarray(archive["lambda_grid"])
    return build_validation_field(
        task_name=task_name,
        q_raw=q_val,
        truth_raw=truth_val,
        lambda_grid=lambda_grid,
    )


def strict_prefix(field: ValidationQField, total_length: int) -> tuple[np.ndarray, np.ndarray]:
    """返回严格 lambda/truth prefix；不插值、重采样、padding 或 masking。"""
    if total_length <= 0 or total_length > field.lambda_grid.size:
        raise ValueError(f"Invalid total_length={total_length} for source length={field.lambda_grid.size}.")
    return (
        np.asarray(field.lambda_grid[:total_length], dtype=np.float64),
        np.asarray(field.canonical_truth[:, :total_length, :], dtype=np.float64),
    )


def verify_r1_provenance(
    checkpoint: Mapping[str, Any],
    task_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """严格读取 R1 metadata，避免臆造训练或 selection exposure 标签。"""
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint has no config mapping required for R1 provenance.")
    if config.get("experiment_type") != "r1_variable_length_training":
        raise ValueError("Checkpoint experiment_type is not r1_variable_length_training.")
    if config.get("repair_class") != "TRAINING_PROTOCOL_REPAIR":
        raise ValueError("Checkpoint repair_class is not TRAINING_PROTOCOL_REPAIR.")
    if config.get("source_task") != task_name:
        raise ValueError(
            "Checkpoint source_task does not match the requested validation source task."
        )
    train_lengths = config.get("train_lengths")
    validation_lengths = config.get("validation_lengths")
    if not isinstance(train_lengths, list) or not isinstance(validation_lengths, list):
        raise ValueError("Checkpoint lacks train_lengths or validation_lengths provenance.")
    return (
        validate_requested_lengths(train_lengths),
        validate_requested_lengths(validation_lengths),
    )


def _model_space_arrays(
    *,
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    raw_truth: np.ndarray,
    normalization_stats: FieldNormalizationStats,
    target_transform: TargetTransformConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 checkpoint 的 input/target semantics 构造一个独立长度的 model-space field。"""
    q32 = np.asarray(q_values, dtype=np.float32).reshape(-1)
    lambda32 = np.asarray(lambda_grid, dtype=np.float32).reshape(-1)
    truth32 = np.asarray(raw_truth, dtype=np.float32)
    expected_shape = (q32.size, lambda32.size, 3)
    if truth32.shape != expected_shape:
        raise ValueError(f"Raw validation prefix shape={truth32.shape}, expected={expected_shape}.")
    q_channel = np.broadcast_to(q32[:, None], (q32.size, lambda32.size))
    lambda_channel = np.broadcast_to(lambda32[None, :], (q32.size, lambda32.size))
    x_raw = np.stack((q_channel, lambda_channel), axis=-1)[None, ...].astype(np.float32)
    y_raw = truth32[None, ...]
    y_transformed = transform_output_field(y=y_raw, config=target_transform)
    x_model = normalize_input_field(x=x_raw, stats=normalization_stats)
    y_model = normalize_output_field(y=y_transformed, stats=normalization_stats)
    return x_model, y_model, y_raw


@torch.no_grad()
def evaluate_one_length(
    *,
    model: torch.nn.Module,
    field: ValidationQField,
    total_length: int,
    normalization_stats: FieldNormalizationStats,
    target_transform: TargetTransformConfig,
    device: str,
) -> tuple[float, np.ndarray]:
    """一个长度只执行一次直接冻结 forward，并返回 normalized MSE 和 raw prediction。"""
    lambda_prefix, truth_prefix = strict_prefix(field, total_length)
    x_model, y_model, y_raw_model_precision = _model_space_arrays(
        q_values=field.canonical_q,
        lambda_grid=lambda_prefix,
        raw_truth=truth_prefix,
        normalization_stats=normalization_stats,
        target_transform=target_transform,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_model).float(), torch.from_numpy(y_model).float()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    predictions_model, targets_model = predict_2d_loader(model=model, loader=loader, device=device)
    normalized_space_mse = float(np.mean((predictions_model - targets_model) ** 2))
    predictions_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=predictions_model,
        targets_model_space=targets_model,
        raw_targets_reference=y_raw_model_precision,
        normalization_stats=normalization_stats,
        target_transform_config=target_transform,
    )
    prediction = np.asarray(predictions_raw[0], dtype=np.float32)
    if prediction.shape != truth_prefix.shape:
        raise ValueError(f"Frozen prediction shape={prediction.shape}, expected={truth_prefix.shape}.")
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Frozen inference produced non-finite predictions.")
    return normalized_space_mse, prediction


@torch.no_grad()
def evaluate_length_response(
    *,
    model: torch.nn.Module,
    field: ValidationQField,
    lengths: Sequence[int],
    training_lengths: Sequence[int],
    checkpoint_selection_lengths: Sequence[int],
    normalization_stats: FieldNormalizationStats,
    target_transform: TargetTransformConfig,
    device: str,
) -> list[dict[str, Any]]:
    """在相同 validation Q 上逐长度执行独立冻结前向。"""
    requested = validate_requested_lengths(lengths, source_length=field.lambda_grid.size)
    train_set = set(int(value) for value in training_lengths)
    selection_set = set(int(value) for value in checkpoint_selection_lengths)
    model.eval()
    rows: list[dict[str, Any]] = []
    for total_length in requested:
        normalized_mse, prediction = evaluate_one_length(
            model=model,
            field=field,
            total_length=total_length,
            normalization_stats=normalization_stats,
            target_transform=target_transform,
            device=device,
        )
        _, truth_prefix = strict_prefix(field, total_length)
        raw_metrics = compute_region_metrics(prediction, truth_prefix, field.canonical_q)
        rows.append(
            {
                "total_length": int(total_length),
                "seen_during_training": bool(total_length in train_set),
                "used_for_checkpoint_selection": bool(total_length in selection_set),
                "normalized_space_mse": normalized_mse,
                "raw_global_relative_l2": float(raw_metrics["global_relative_l2"]),
                "raw_mean_per_q_relative_l2": float(raw_metrics["mean_per_q_relative_l2"]),
                "frozen_forward_passes": 1,
            }
        )
    return rows


def build_summary(
    *,
    task_name: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    field: ValidationQField,
    requested_lengths: Sequence[int],
    training_lengths: Sequence[int],
    validation_lengths: Sequence[int],
    normalization_stats: FieldNormalizationStats,
    target_transform: TargetTransformConfig,
    rows: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, Any]:
    """构造不含预测、truth 或 per-Q 大数组的紧凑开发诊断摘要。"""
    seen_values = [
        float(row["raw_mean_per_q_relative_l2"])
        for row in rows
        if bool(row["seen_during_training"])
    ]
    unseen_values = [
        float(row["raw_mean_per_q_relative_l2"])
        for row in rows
        if not bool(row["seen_during_training"])
    ]
    return {
        "schema_version": "1.0",
        "diagnostic_type": "r1_validation_q_length_response",
        "status": "completed",
        "scientific_scope": "development_diagnostic",
        "formal_long_domain_extrapolation_result": False,
        "task_name": str(task_name),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_score": checkpoint.get("best_val_mse"),
        "validation_q_count": int(field.canonical_q.size),
        "canonical_q_order": "ascending",
        "requested_lengths": [int(value) for value in requested_lengths],
        "training_lengths_from_checkpoint": [int(value) for value in training_lengths],
        "validation_lengths_from_checkpoint": [int(value) for value in validation_lengths],
        "exposure_label_source": "checkpoint_provenance",
        "normalization_provenance": {
            "source": "checkpoint",
            "values": normalization_stats.to_dict(),
        },
        "target_transform": target_transform.to_dict(),
        "frozen_inference_only": True,
        "optimizer": False,
        "scheduler": False,
        "adaptation": False,
        "autoregression": False,
        "prediction_feedback": False,
        "frozen_forward_passes": int(len(rows)),
        "results_by_length": {f"T{row['total_length']}": dict(row) for row in rows},
        "neutral_pattern_summary": {
            "mean_raw_relative_l2_seen_training_lengths": (
                float(np.mean(seen_values)) if seen_values else None
            ),
            "mean_raw_relative_l2_not_seen_training_lengths": (
                float(np.mean(unseen_values)) if unseen_values else None
            ),
            "interpretation": "row_level_evidence_is_primary; no_memorization_claim_is_made",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        "git_commit": _git_commit(),
        "output_files": list(OUTPUT_FILENAMES),
    }


def write_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """拒绝覆盖，并且只写入两个紧凑结果文件。"""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / OUTPUT_FILENAMES[0]
    csv_path = output_dir / OUTPUT_FILENAMES[1]
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(dict(summary)), handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in CSV_FIELDS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 validation-Q development diagnostic CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen R1 checkpoint on validation-Q strict prefixes. "
            "This is a development diagnostic, not a formal long-domain result."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行 checkpoint-only validation-Q length response diagnostic。"""
    args = parse_args(argv)
    requested_lengths = validate_requested_lengths(args.lengths)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")
    field = load_validation_source(str(args.task_name))
    checkpoint = load_checkpoint_2d(args.checkpoint_path, device=str(args.device))
    training_lengths, validation_lengths = verify_r1_provenance(checkpoint, str(args.task_name))
    normalization_stats = load_normalization_stats_from_checkpoint(checkpoint)
    target_transform = load_target_transform_config_from_checkpoint(checkpoint)
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))
    rows = evaluate_length_response(
        model=model,
        field=field,
        lengths=requested_lengths,
        training_lengths=training_lengths,
        checkpoint_selection_lengths=validation_lengths,
        normalization_stats=normalization_stats,
        target_transform=target_transform,
        device=str(args.device),
    )
    summary = build_summary(
        task_name=str(args.task_name),
        checkpoint_path=args.checkpoint_path,
        checkpoint=checkpoint,
        field=field,
        requested_lengths=requested_lengths,
        training_lengths=training_lengths,
        validation_lengths=validation_lengths,
        normalization_stats=normalization_stats,
        target_transform=target_transform,
        rows=rows,
        device=str(args.device),
    )
    write_outputs(output_dir=args.output_dir, summary=summary, rows=rows)
    print("R1 validation-Q length-response diagnostic completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
