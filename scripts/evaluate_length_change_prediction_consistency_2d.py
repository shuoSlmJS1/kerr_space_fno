"""冻结 FNO2D 在输入 lambda 长度改变时的预测一致性诊断。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.io_utils import load_npz  # noqa: E402
from src.common.paths import get_task_dataset_npz_path  # noqa: E402
from src.training.fno2d.normalization_2d import (  # noqa: E402
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    transform_output_field,
)
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)


SPLITS = ("train", "val", "test")
COMPONENT_NAMES = ("x", "y", "z")
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    """解析正式诊断所需的最小命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen FNO2D short/long lambda-domain prediction "
            "consistency evaluation and write one JSON result."
        )
    )
    parser.add_argument("--training-task-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--short-task-name", required=True)
    parser.add_argument("--long-task-name", required=True)
    parser.add_argument(
        "--dataset-pair-validation-json",
        required=True,
        type=Path,
        help="Stage-2 strict prefix-identity validation JSON.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help=(
            "Evaluation scope. 'all' is the formal full-Q protocol: it "
            "combines train, val, and test source identities, then stably "
            "sorts Q ascending before model input. Individual splits are "
            "diagnostic-only and are also sorted internally."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path; otherwise project conventions are used.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _relative_path(path: Path) -> str:
    """尽可能记录相对项目根目录的路径。"""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _as_json_value(value: Any) -> Any:
    """将 NumPy 标量和数组转换为 JSON 可序列化对象。"""
    if isinstance(value, np.ndarray):
        return [_as_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    return value


def load_required_pair_validation(path: Path) -> dict[str, Any]:
    """读取并严格检查 Stage-2 的短到中长度配对结论。"""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset-pair validation JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    classifications = artifact.get("pair_classification")
    reuse = artifact.get("scientific_reuse")
    if not isinstance(classifications, dict) or not isinstance(reuse, dict):
        raise ValueError("Dataset-pair validation JSON has no required classification fields.")
    if classifications.get("short_to_medium") != "EXACT_PREFIX":
        raise ValueError(
            "Dataset-pair validation requires short_to_medium=EXACT_PREFIX."
        )
    if reuse.get("historical_t1800_reusable") is not True:
        raise ValueError(
            "Dataset-pair validation requires historical_t1800_reusable=true."
        )
    return artifact


@dataclass(frozen=True)
class CanonicalQField:
    """同时保留原始身份顺序与模型所需的 canonical Q 轴。"""
    source_q: np.ndarray
    source_truth: np.ndarray
    lambda_grid: np.ndarray
    canonical_q: np.ndarray
    canonical_truth: np.ndarray
    canonical_to_source_index: np.ndarray
    source_to_canonical_index: np.ndarray
    source_records: list[dict[str, Any]]
    selected_splits: tuple[str, ...]


def _load_split(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一个 split，同时保留原始数组顺序与 float64 真值。"""
    x_key = f"x_{split}"
    y_key = f"y_{split}"
    if x_key not in data or y_key not in data:
        raise KeyError(f"Dataset is missing {x_key} or {y_key}.")
    x = np.asarray(data[x_key])
    y = np.asarray(data[y_key])
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"{x_key} must have shape [N,1], got {x.shape}.")
    if y.ndim != 3 or y.shape[0] != x.shape[0] or y.shape[2] != 3:
        raise ValueError(f"{y_key} must have shape [N,T,3], got {y.shape}.")
    return x, y


def build_canonical_q_field(
    source_q: np.ndarray,
    source_truth: np.ndarray,
    lambda_grid: np.ndarray,
    source_records: list[dict[str, Any]],
    selected_splits: tuple[str, ...],
) -> CanonicalQField:
    """稳定排序 Q，并以同一置换重排真值与来源记录。"""
    q = np.asarray(source_q, dtype=np.float64).reshape(-1)
    truth = np.asarray(source_truth, dtype=np.float64)
    lambda_values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if truth.ndim != 3 or truth.shape[0] != q.size or truth.shape[2] != 3:
        raise ValueError("Source Q and raw truth arrays have incompatible shapes.")
    if truth.shape[1] != lambda_values.size:
        raise ValueError("Raw truth length does not match lambda grid length.")
    if len(source_records) != q.size:
        raise ValueError("Source identity records do not match Q count.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(truth)):
        raise ValueError("Selected dataset split contains non-finite values.")
    canonical_to_source = np.argsort(q, kind="stable").astype(np.int64)
    canonical_q = q[canonical_to_source]
    if canonical_q.size > 1 and not np.all(np.diff(canonical_q) > 0.0):
        raise ValueError("Canonical Q field requires unique strictly ascending Q values.")
    source_to_canonical = np.empty_like(canonical_to_source)
    source_to_canonical[canonical_to_source] = np.arange(q.size, dtype=np.int64)
    if not np.array_equal(canonical_to_source[source_to_canonical], np.arange(q.size)):
        raise RuntimeError("Q-order permutations are not inverse mappings.")
    records: list[dict[str, Any]] = []
    for canonical_index, source_index in enumerate(canonical_to_source):
        record = dict(source_records[int(source_index)])
        record["Q"] = float(canonical_q[canonical_index])
        record["canonical_model_index"] = int(canonical_index)
        records.append(record)
    return CanonicalQField(
        source_q=q,
        source_truth=truth,
        lambda_grid=lambda_values,
        canonical_q=canonical_q,
        canonical_truth=truth[canonical_to_source],
        canonical_to_source_index=canonical_to_source,
        source_to_canonical_index=source_to_canonical,
        source_records=records,
        selected_splits=selected_splits,
    )


def load_task_raw_field(task_name: str, split_policy: str) -> CanonicalQField:
    """加载原始 split 身份，并构造独立的升序 Q 模型输入场。"""
    dataset_path = get_task_dataset_npz_path(task_name)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    data = load_npz(dataset_path)
    if "lambda_grid" not in data:
        raise KeyError("Dataset is missing lambda_grid.")
    lambda_grid = np.asarray(data["lambda_grid"], dtype=np.float64).reshape(-1)
    selected_splits = SPLITS if split_policy == "all" else (split_policy,)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    source_offset = 0
    for split in selected_splits:
        x, y = _load_split(data, split)
        x_parts.append(x)
        y_parts.append(y)
        for index in range(x.shape[0]):
            source_records.append({
                "source_split": split,
                "source_index_within_split": int(index),
                "source_concatenated_index": int(source_offset + index),
            })
        source_offset += int(x.shape[0])
    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    if y_all.dtype != np.float64:
        raise ValueError(
            "Canonical shared truth requires raw dataset trajectories with dtype float64."
        )
    if y_all.shape[1] != lambda_grid.size:
        raise ValueError(
            f"Trajectory length={y_all.shape[1]} does not match lambda length={lambda_grid.size}."
        )
    return build_canonical_q_field(
        source_q=x_all[:, 0],
        source_truth=y_all,
        lambda_grid=lambda_grid,
        source_records=source_records,
        selected_splits=selected_splits,
    )


def validate_raw_pair(
    short_q: np.ndarray,
    short_truth: np.ndarray,
    short_lambda: np.ndarray,
    long_q: np.ndarray,
    long_truth: np.ndarray,
    long_lambda: np.ndarray,
) -> None:
    """在实际推理前复核当前原始配对数据，绝不排序或重配 Q。"""
    if long_lambda.size < short_lambda.size:
        raise ValueError("Long lambda grid is shorter than the short lambda grid.")
    if not np.array_equal(short_q, long_q):
        raise ValueError("Short and long Q arrays differ in selected split order.")
    if not np.array_equal(short_lambda, long_lambda[: short_lambda.size]):
        raise ValueError("Short and long lambda grids are not an exact raw prefix.")
    long_prefix = long_truth[:, : short_lambda.size, :]
    if not np.array_equal(short_truth, long_prefix):
        raise ValueError("Short and long raw trajectories are not an exact raw prefix.")


def build_raw_field(
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    y_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构造现有 FNO2D 推理管线需要的二维场。"""
    q_float32 = np.asarray(q_values, dtype=np.float32).reshape(-1)
    lambda_float32 = np.asarray(lambda_grid, dtype=np.float32).reshape(-1)
    y_float32 = np.asarray(y_raw, dtype=np.float32)
    expected_shape = (q_float32.size, lambda_float32.size, 3)
    if y_float32.shape != expected_shape:
        raise ValueError(f"Raw trajectory shape={y_float32.shape}, expected={expected_shape}.")
    q_channel = np.broadcast_to(q_float32[:, None], q_float32.shape[0:1] + lambda_float32.shape)
    lambda_channel = np.broadcast_to(lambda_float32[None, :], (q_float32.size, lambda_float32.size))
    x_raw = np.stack((q_channel, lambda_channel), axis=-1)[None, ...]
    return x_raw.astype(np.float32), y_float32[None, ...]


def run_frozen_inference(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    raw_truth: np.ndarray,
    device: str,
) -> np.ndarray:
    """使用同一冻结模型完成一次前向推理，并恢复 raw xyz 预测。"""
    x_raw, y_raw = build_raw_field(q_values, lambda_grid, raw_truth)
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform_config = load_target_transform_config_from_checkpoint(checkpoint)
    y_transformed = transform_output_field(y=y_raw, config=transform_config)
    x_model = normalize_input_field(x=x_raw, stats=stats)
    y_model = normalize_output_field(y=y_transformed, stats=stats)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_model).float(), torch.from_numpy(y_model).float()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    predictions_model_space, targets_model_space = predict_2d_loader(
        model=model,
        loader=loader,
        device=device,
    )
    predictions_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=predictions_model_space,
        targets_model_space=targets_model_space,
        raw_targets_reference=y_raw,
        normalization_stats=stats,
        target_transform_config=transform_config,
    )
    prediction = np.asarray(predictions_raw[0], dtype=np.float32)
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Frozen inference produced non-finite predictions.")
    return prediction


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """计算明确以 reference 为分母的 Relative L2。"""
    difference = prediction - reference
    return float(np.linalg.norm(difference.ravel()) / (np.linalg.norm(reference.ravel()) + EPSILON))


def compute_comparison_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    q_values: np.ndarray,
    *,
    denominator_description: str,
) -> dict[str, Any]:
    """计算全局、逐 Q 和 xyz 分量指标，全部在 float64 评估空间中完成。"""
    prediction64 = np.asarray(prediction, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if prediction64.shape != reference64.shape or prediction64.ndim != 3:
        raise ValueError(
            f"Prediction/reference must share shape [N,T,3], got {prediction64.shape} and {reference64.shape}."
        )
    if prediction64.shape[0] != q64.size or prediction64.shape[2] != 3:
        raise ValueError("Q values or xyz component dimension is inconsistent with predictions.")
    if not np.all(np.isfinite(prediction64)) or not np.all(np.isfinite(reference64)):
        raise ValueError("Metrics require finite predictions and references.")

    difference = prediction64 - reference64
    per_q_numerator = np.linalg.norm(difference.reshape(q64.size, -1), axis=1)
    per_q_denominator = np.linalg.norm(reference64.reshape(q64.size, -1), axis=1)
    per_q = per_q_numerator / (per_q_denominator + EPSILON)
    worst_index = int(np.argmax(per_q)) if per_q.size else None
    components: dict[str, dict[str, float]] = {}
    for index, name in enumerate(COMPONENT_NAMES):
        component_prediction = prediction64[:, :, index]
        component_reference = reference64[:, :, index]
        components[name] = {
            "mse": float(np.mean((component_prediction - component_reference) ** 2)),
            "relative_l2": _relative_l2(component_prediction, component_reference),
        }
    return {
        "reference_description": denominator_description,
        "mse": float(np.mean(difference**2)),
        "global_relative_l2": _relative_l2(prediction64, reference64),
        "per_q_relative_l2": {
            "count": int(per_q.size),
            "mean": float(np.mean(per_q)) if per_q.size else 0.0,
            "median": float(np.median(per_q)) if per_q.size else 0.0,
            "p95": float(np.percentile(per_q, 95)) if per_q.size else 0.0,
            "max": float(np.max(per_q)) if per_q.size else 0.0,
            "worst_q_index": worst_index,
            "worst_q_value": float(q64[worst_index]) if worst_index is not None else None,
        },
        "components": components,
    }


def _git_commit() -> str:
    """读取本地当前提交；失败时显式记录不可用状态。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _ordering_provenance(
    short_field: CanonicalQField,
    long_field: CanonicalQField,
    split_policy: str,
) -> dict[str, Any]:
    """记录来源身份与 canonical 模型输入顺序之间的可逆映射。"""
    formal_scope = split_policy == "all"
    source_identity_order = (
        "train_then_val_then_test_original_order"
        if formal_scope
        else f"{split_policy}_split_original_order"
    )
    model_input_q_order = (
        "ascending_Q_full_field"
        if formal_scope
        else "ascending_Q_within_selected_split_diagnostic_only"
    )
    return {
        "evaluation_scope": (
            "full_q400_canonical_field"
            if formal_scope
            else "diagnostic_only_split_field"
        ),
        "source_split_order": list(short_field.selected_splits),
        "source_identity_order": source_identity_order,
        "model_input_q_order": model_input_q_order,
        "stable_sort": True,
        "q_count": int(short_field.canonical_q.size),
        "canonical_q_exact_match_between_short_and_long": bool(
            np.array_equal(short_field.canonical_q, long_field.canonical_q)
        ),
        "short": {
            "canonical_to_source_index": short_field.canonical_to_source_index,
            "source_to_canonical_index": short_field.source_to_canonical_index,
            "source_records_in_canonical_order": short_field.source_records,
        },
        "long": {
            "canonical_to_source_index": long_field.canonical_to_source_index,
            "source_to_canonical_index": long_field.source_to_canonical_index,
            "source_records_in_canonical_order": long_field.source_records,
        },
    }


def build_result(
    *,
    training_task_name: str,
    model_name: str,
    short_task_name: str,
    long_task_name: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    pair_validation_path: Path,
    pair_validation: dict[str, Any],
    split_policy: str,
    device: str,
    short_field: CanonicalQField,
    long_field: CanonicalQField,
    short_prediction: np.ndarray,
    long_prediction: np.ndarray,
) -> dict[str, Any]:
    """构建紧凑、可审计且不包含大数组的正式结果。"""
    short_q = short_field.canonical_q
    short_truth = short_field.canonical_truth
    short_lambda = short_field.lambda_grid
    long_lambda = long_field.lambda_grid
    shared_length = int(short_lambda.size)
    long_prefix_prediction = np.asarray(long_prediction[:, :shared_length, :])
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform_config = load_target_transform_config_from_checkpoint(checkpoint)
    config = checkpoint.get("config", {})
    model_config = config.get("model_config", {})
    delta = np.diff(short_lambda)
    return _as_json_value({
        "schema_version": "1.0",
        "diagnostic_type": "frozen_length_change_prediction_consistency",
        "status": "completed",
        "training_task_name": training_task_name,
        "short_task_name": short_task_name,
        "long_task_name": long_task_name,
        "model_name": model_name,
        "checkpoint": {
            "path": _relative_path(checkpoint_path),
            "role": "frozen_best_model",
            "config_summary": {"model_config": model_config},
        },
        "dataset_pair_validation": {
            "artifact_path": _relative_path(pair_validation_path),
            "classification": pair_validation["pair_classification"]["short_to_medium"],
            "historical_t1800_reusable": pair_validation["scientific_reuse"]["historical_t1800_reusable"],
        },
        "ordering": _ordering_provenance(short_field, long_field, split_policy),
        "q_count": int(short_q.size),
        "short_length": shared_length,
        "long_length": int(long_lambda.size),
        "shared_prefix_length": shared_length,
        "lambda": {
            "short_min": float(np.min(short_lambda)),
            "short_max": float(np.max(short_lambda)),
            "long_min": float(np.min(long_lambda)),
            "long_max": float(np.max(long_lambda)),
            "short_delta_lambda_min": float(np.min(delta)) if delta.size else None,
            "short_delta_lambda_max": float(np.max(delta)) if delta.size else None,
        },
        "truth_reference": {
            "name": "shared_truth_raw_float64",
            "source": "short_dataset_original_raw_truth",
            "dtype": "float64",
            "used_for_primary_metrics": True,
        },
        "inference_pipeline_truth_representation_check": {
            "performed": False,
            "reason": "historical_recovered_target_artifacts_are_not_used",
        },
        "normalization": _as_json_value(stats.to_dict()),
        "target_transform": _as_json_value(transform_config.to_dict()),
        "frozen_inference_only": True,
        "autoregressive_rollout": False,
        "teacher_forcing": False,
        "adaptation": "none",
        "metrics": {
            "short_prediction_vs_shared_truth": compute_comparison_metrics(
                short_prediction,
                short_truth,
                short_q,
                denominator_description="shared_truth_raw_float64",
            ),
            "long_prefix_prediction_vs_shared_truth": compute_comparison_metrics(
                long_prefix_prediction,
                short_truth,
                short_q,
                denominator_description="shared_truth_raw_float64",
            ),
            "long_prefix_vs_short_prediction": compute_comparison_metrics(
                long_prefix_prediction,
                short_prediction,
                short_q,
                denominator_description="short_prediction_raw_xyz",
            ),
        },
        "runtime": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": device,
        },
        "output_policy": {
            "json_output_refuses_overwrite": True,
            "prediction_arrays_written": False,
            "target_arrays_written": False,
            "plots_written": False,
        },
    })


def write_json_exclusively(result: dict[str, Any], output_path: Path) -> None:
    """只以排他方式写一个结果 JSON，拒绝覆盖既有文件。"""
    output_path = output_path.resolve()
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    """执行一次新的短/长冻结前向推理并写出唯一 JSON。"""
    args = parse_args()
    pair_validation_path = Path(args.dataset_pair_validation_json)
    pair_validation = load_required_pair_validation(pair_validation_path)
    short_field = load_task_raw_field(args.short_task_name, args.split)
    long_field = load_task_raw_field(args.long_task_name, args.split)
    validate_raw_pair(
        short_field.canonical_q,
        short_field.canonical_truth,
        short_field.lambda_grid,
        long_field.canonical_q,
        long_field.canonical_truth,
        long_field.lambda_grid,
    )
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else PROJECT_ROOT / "outputs" / args.training_task_name / args.model_name / "checkpoints" / "best_model.pt"
    )
    checkpoint = load_checkpoint_2d(checkpoint_path=checkpoint_path, device=str(args.device))
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))
    short_prediction = run_frozen_inference(
        model,
        checkpoint,
        short_field.canonical_q,
        short_field.lambda_grid,
        short_field.canonical_truth,
        str(args.device),
    )
    long_prediction = run_frozen_inference(
        model,
        checkpoint,
        long_field.canonical_q,
        long_field.lambda_grid,
        long_field.canonical_truth,
        str(args.device),
    )
    result = build_result(
        training_task_name=args.training_task_name,
        model_name=args.model_name,
        short_task_name=args.short_task_name,
        long_task_name=args.long_task_name,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        pair_validation_path=pair_validation_path,
        pair_validation=pair_validation,
        split_policy=args.split,
        device=str(args.device),
        short_field=short_field,
        long_field=long_field,
        short_prediction=short_prediction,
        long_prediction=long_prediction,
    )
    write_json_exclusively(result, Path(args.output_json))
    print(f"Diagnostic result written: {Path(args.output_json).resolve()}")
    print(f"Q count: {result['q_count']}")
    print(f"Shared prefix length: {result['shared_prefix_length']}")


if __name__ == "__main__":
    main()
