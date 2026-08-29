"""Formal frozen FNO2D evaluation across three lambda-domain lengths."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.io_utils import load_npz  # noqa: E402
from src.common.paths import get_task_dataset_npz_path  # noqa: E402
from src.training.fno2d.normalization_2d import normalize_input_field, normalize_output_field  # noqa: E402
from src.training.fno2d.target_transform_2d import transform_output_field  # noqa: E402
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
OUTPUT_FILENAMES = (
    "a1_length_extrapolation_summary.json",
    "per_q_metrics.csv",
    "lambda_window_metrics.csv",
)


@dataclass(frozen=True)
class CanonicalQField:
    """保留来源身份，同时提供模型所需的升序 Q 场。"""

    task_name: str
    source_q: np.ndarray
    source_truth: np.ndarray
    lambda_grid: np.ndarray
    canonical_q: np.ndarray
    canonical_truth: np.ndarray
    canonical_to_source_index: np.ndarray
    source_to_canonical_index: np.ndarray
    source_records: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """解析正式 A1 评估所需的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen FNO2D forward pass for each of three canonical "
            "lambda-domain lengths and write compact formal A1 metrics."
        )
    )
    parser.add_argument("--training-task-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--short-task-name", required=True)
    parser.add_argument("--medium-task-name", required=True)
    parser.add_argument("--long-task-name", required=True)
    parser.add_argument(
        "--dataset-pair-validation-json",
        required=True,
        type=Path,
        help="Stage-2 strict T1200/T1800/T2400 prefix-identity JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory. The evaluator refuses to overwrite it.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path; otherwise project conventions are used.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--window-width",
        type=float,
        default=0.5,
        help="Physical lambda width for non-overlapping extrapolation windows.",
    )
    return parser.parse_args()


def _relative_path(path: Path) -> str:
    """优先将路径记录为项目根目录相对路径。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _as_json_value(value: Any) -> Any:
    """将 NumPy 值递归转换为 JSON 可序列化值。"""

    if isinstance(value, np.ndarray):
        return [_as_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    return value


def _git_commit() -> str | None:
    """读取本地 Git HEAD，不执行任何远程 Git 操作。"""

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


def load_required_pair_validation(path: Path) -> dict[str, Any]:
    """读取并严格检查 Stage-2 三个长度配对前提。"""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset-pair validation JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    classifications = artifact.get("pair_classification")
    reuse = artifact.get("scientific_reuse")
    if not isinstance(classifications, dict) or not isinstance(reuse, dict):
        raise ValueError("Dataset-pair validation JSON has no required fields.")

    required_classifications = (
        "short_to_medium",
        "short_to_long",
        "medium_to_long",
    )
    for key in required_classifications:
        if classifications.get(key) != "EXACT_PREFIX":
            raise ValueError(f"Dataset-pair validation requires {key}=EXACT_PREFIX.")

    required_reuse = (
        "historical_t1800_reusable",
        "t2400_ready_for_future_a1",
    )
    for key in required_reuse:
        if reuse.get(key) is not True:
            raise ValueError(f"Dataset-pair validation requires {key}=true.")
    return artifact


def _load_split(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一个 split，并保留其原始 float64 轨迹顺序。"""

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
    if y.dtype != np.float64:
        raise ValueError(f"{y_key} must retain raw float64 truth, got {y.dtype}.")
    return x, y


def build_canonical_q_field(
    *,
    task_name: str,
    source_q: np.ndarray,
    source_truth: np.ndarray,
    lambda_grid: np.ndarray,
    source_records: list[dict[str, Any]],
) -> CanonicalQField:
    """稳定排序 Q，并将同一置换应用于真值和来源记录。"""

    q = np.asarray(source_q, dtype=np.float64).reshape(-1)
    truth = np.asarray(source_truth, dtype=np.float64)
    lambda_values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if truth.ndim != 3 or truth.shape != (q.size, lambda_values.size, 3):
        raise ValueError("Source Q, truth, and lambda grid have incompatible shapes.")
    if len(source_records) != q.size:
        raise ValueError("Source identity records do not match Q count.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(truth)):
        raise ValueError("Canonical fields require finite Q and raw truth values.")
    if lambda_values.size == 0 or not np.all(np.isfinite(lambda_values)):
        raise ValueError("Canonical fields require a finite non-empty lambda grid.")

    canonical_to_source = np.argsort(q, kind="stable").astype(np.int64)
    canonical_q = q[canonical_to_source]
    if canonical_q.size > 1 and not np.all(np.diff(canonical_q) > 0.0):
        raise ValueError("Canonical Q field requires unique strictly ascending Q values.")
    source_to_canonical = np.empty_like(canonical_to_source)
    source_to_canonical[canonical_to_source] = np.arange(q.size, dtype=np.int64)
    if not np.array_equal(canonical_to_source[source_to_canonical], np.arange(q.size)):
        raise RuntimeError("Canonical/source Q mappings are not inverse permutations.")

    canonical_records: list[dict[str, Any]] = []
    for canonical_index, source_index in enumerate(canonical_to_source):
        record = dict(source_records[int(source_index)])
        record["Q"] = float(canonical_q[canonical_index])
        record["canonical_model_index"] = int(canonical_index)
        canonical_records.append(record)

    return CanonicalQField(
        task_name=task_name,
        source_q=q,
        source_truth=truth,
        lambda_grid=lambda_values,
        canonical_q=canonical_q,
        canonical_truth=truth[canonical_to_source],
        canonical_to_source_index=canonical_to_source,
        source_to_canonical_index=source_to_canonical,
        source_records=canonical_records,
    )


def load_task_raw_field(task_name: str) -> CanonicalQField:
    """合并三份来源 split，并构造完整 canonical Q400 模型输入场。"""

    dataset_path = get_task_dataset_npz_path(task_name)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    data = load_npz(dataset_path)
    if "lambda_grid" not in data:
        raise KeyError("Dataset is missing lambda_grid.")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    source_offset = 0
    for split in SPLITS:
        x, y = _load_split(data, split)
        x_parts.append(x)
        y_parts.append(y)
        for index in range(x.shape[0]):
            source_records.append(
                {
                    "source_split": split,
                    "source_index_within_split": int(index),
                    "source_concatenated_index": int(source_offset + index),
                }
            )
        source_offset += int(x.shape[0])

    return build_canonical_q_field(
        task_name=task_name,
        source_q=np.concatenate(x_parts, axis=0)[:, 0],
        source_truth=np.concatenate(y_parts, axis=0),
        lambda_grid=np.asarray(data["lambda_grid"], dtype=np.float64),
        source_records=source_records,
    )


def validate_triplet(
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
) -> None:
    """在任何 checkpoint 读取或推理前验证 canonical 三长度真值关系。"""

    fields = (short, medium, long)
    expected_q_count = int(short.canonical_q.size)
    for field in fields:
        if field.canonical_q.size != expected_q_count:
            raise ValueError("All formal A1 tasks must have the same canonical Q count.")
        if not np.array_equal(short.canonical_q, field.canonical_q):
            raise ValueError("Canonical Q arrays must match exactly across all three tasks.")

    if medium.lambda_grid.size < short.lambda_grid.size or long.lambda_grid.size < medium.lambda_grid.size:
        raise ValueError("Formal A1 lambda lengths must be non-decreasing short/medium/long.")
    if not np.array_equal(short.lambda_grid, medium.lambda_grid[: short.lambda_grid.size]):
        raise ValueError("Short and medium lambda grids are not exact prefixes.")
    if not np.array_equal(short.lambda_grid, long.lambda_grid[: short.lambda_grid.size]):
        raise ValueError("Short and long lambda grids are not exact prefixes.")
    if not np.array_equal(medium.lambda_grid, long.lambda_grid[: medium.lambda_grid.size]):
        raise ValueError("Medium and long lambda grids are not exact prefixes.")
    if not np.array_equal(short.canonical_truth, medium.canonical_truth[:, : short.lambda_grid.size, :]):
        raise ValueError("T1200 and T1800 canonical truths are not exact prefixes.")
    if not np.array_equal(short.canonical_truth, long.canonical_truth[:, : short.lambda_grid.size, :]):
        raise ValueError("T1200 and T2400 canonical truths are not exact prefixes.")
    if not np.array_equal(medium.canonical_truth, long.canonical_truth[:, : medium.lambda_grid.size, :]):
        raise ValueError("T1800 and T2400 canonical truths are not exact prefixes.")


def build_model_input(
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    raw_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """为既有 float32 FNO2D 推理管线构造 [1,H,W,C] 输入。"""

    q32 = np.asarray(q_values, dtype=np.float32).reshape(-1)
    lambda32 = np.asarray(lambda_grid, dtype=np.float32).reshape(-1)
    truth32 = np.asarray(raw_truth, dtype=np.float32)
    expected = (q32.size, lambda32.size, 3)
    if truth32.shape != expected:
        raise ValueError(f"Raw truth shape={truth32.shape}, expected={expected}.")
    q_channel = np.broadcast_to(q32[:, None], (q32.size, lambda32.size))
    lambda_channel = np.broadcast_to(lambda32[None, :], (q32.size, lambda32.size))
    x_raw = np.stack((q_channel, lambda_channel), axis=-1)[None, ...]
    return x_raw.astype(np.float32), truth32[None, ...]


def run_frozen_inference(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    field: CanonicalQField,
    device: str,
) -> np.ndarray:
    """对一个总长度执行一次冻结前向，并恢复 raw xyz prediction。"""

    x_raw, y_raw = build_model_input(
        field.canonical_q,
        field.lambda_grid,
        field.canonical_truth,
    )
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
    if prediction.shape != field.canonical_truth.shape:
        raise ValueError(
            f"Frozen prediction shape={prediction.shape}, expected={field.canonical_truth.shape}."
        )
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Frozen inference produced non-finite predictions.")
    return prediction


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """计算指定数组的 global Relative L2。"""

    return float(np.linalg.norm(prediction - reference) / (np.linalg.norm(reference) + EPSILON))


def compute_region_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    q_values: np.ndarray,
) -> dict[str, Any]:
    """在 float64 科学评估空间计算一个区域的最小正式指标。"""

    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if prediction64.shape != truth64.shape or prediction64.ndim != 3 or prediction64.shape[2] != 3:
        raise ValueError("Region prediction and truth must share shape [Q,lambda,3].")
    if q64.size != prediction64.shape[0] or prediction64.shape[1] == 0:
        raise ValueError("Region Q count or lambda length is invalid.")

    per_q_relative_l2 = np.asarray(
        [_relative_l2(prediction64[index], truth64[index]) for index in range(q64.size)],
        dtype=np.float64,
    )
    worst_q_index = int(np.argmax(per_q_relative_l2))
    components: dict[str, dict[str, float]] = {}
    for component_index, component_name in enumerate(COMPONENT_NAMES):
        prediction_component = prediction64[:, :, component_index]
        truth_component = truth64[:, :, component_index]
        components[component_name] = {
            "mse": float(np.mean((prediction_component - truth_component) ** 2)),
            "relative_l2": _relative_l2(prediction_component, truth_component),
        }

    return {
        "mse": float(np.mean((prediction64 - truth64) ** 2)),
        "global_relative_l2": _relative_l2(prediction64, truth64),
        "mean_per_q_relative_l2": float(np.mean(per_q_relative_l2)),
        "median_per_q_relative_l2": float(np.median(per_q_relative_l2)),
        "p95_per_q_relative_l2": float(np.percentile(per_q_relative_l2, 95.0)),
        "max_per_q_relative_l2": float(np.max(per_q_relative_l2)),
        "worst_q_index": worst_q_index,
        "worst_q_value": float(q64[worst_q_index]),
        "components": components,
    }


def _per_q_region_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """为紧凑 CSV 计算每条 Q 轨道的 MSE 与 Relative L2。"""

    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    mse = np.mean((prediction64 - truth64) ** 2, axis=(1, 2))
    relative_l2 = np.asarray(
        [_relative_l2(prediction64[index], truth64[index]) for index in range(prediction64.shape[0])],
        dtype=np.float64,
    )
    return mse, relative_l2


def compute_per_q_rows(
    *,
    total_length: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    q_values: np.ndarray,
    training_length: int,
) -> list[dict[str, Any]]:
    """生成一行对应一个 Q 和一个输入总长度的紧凑指标。"""

    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if prediction64.shape != truth64.shape or prediction64.shape[0] != q64.size:
        raise ValueError("Per-Q inputs have incompatible shapes.")
    if total_length != prediction64.shape[1] or training_length > total_length:
        raise ValueError("Per-Q total length or training length is invalid.")

    prefix_mse, prefix_relative_l2 = _per_q_region_metrics(
        prediction64[:, :training_length, :],
        truth64[:, :training_length, :],
    )
    full_mse, full_relative_l2 = _per_q_region_metrics(prediction64, truth64)
    has_extrapolation = total_length > training_length
    if has_extrapolation:
        extrapolation_mse, extrapolation_relative_l2 = _per_q_region_metrics(
            prediction64[:, training_length:, :],
            truth64[:, training_length:, :],
        )

    rows: list[dict[str, Any]] = []
    for index, q_value in enumerate(q64):
        rows.append(
            {
                "total_length": int(total_length),
                "Q": float(q_value),
                "prefix_mse": float(prefix_mse[index]),
                "prefix_relative_l2": float(prefix_relative_l2[index]),
                "extrapolation_mse": float(extrapolation_mse[index]) if has_extrapolation else None,
                "extrapolation_relative_l2": float(extrapolation_relative_l2[index]) if has_extrapolation else None,
                "full_mse": float(full_mse[index]),
                "full_relative_l2": float(full_relative_l2[index]),
            }
        )
    return rows


def build_extrapolation_windows(
    lambda_grid: np.ndarray,
    training_length: int,
    window_width: float,
) -> list[tuple[float, float, np.ndarray, bool]]:
    """构造不重叠且覆盖全部 extrapolation 点的物理 lambda 窗口。"""

    lambda64 = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if training_length <= 0 or training_length > lambda64.size:
        raise ValueError("Training length is outside the lambda grid.")
    if not np.isfinite(window_width) or window_width <= 0.0:
        raise ValueError("Window width must be finite and positive.")
    if training_length == lambda64.size:
        return []

    boundary = float(lambda64[training_length - 1])
    extrapolation_indices = np.arange(training_length, lambda64.size, dtype=np.int64)
    endpoint = float(lambda64[-1])
    windows: list[tuple[float, float, np.ndarray, bool]] = []
    start = boundary
    consumed = np.zeros(extrapolation_indices.size, dtype=bool)
    while start <= endpoint + EPSILON:
        nominal_end = start + window_width
        is_final = nominal_end >= endpoint - EPSILON
        end = endpoint if is_final else nominal_end
        values = lambda64[extrapolation_indices]
        if is_final:
            mask = (values >= start - EPSILON) & (values <= end + EPSILON)
        else:
            mask = (values >= start - EPSILON) & (values < end - EPSILON)
        selected_positions = np.flatnonzero(mask)
        if selected_positions.size:
            if np.any(consumed[selected_positions]):
                raise RuntimeError("Physical lambda windows overlap.")
            consumed[selected_positions] = True
            windows.append((float(start), float(end), extrapolation_indices[selected_positions], is_final))
        if is_final:
            break
        start = nominal_end
    if not np.all(consumed):
        raise RuntimeError("Physical lambda windows do not cover every extrapolation point.")
    return windows


def compute_window_rows(
    *,
    total_length: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    lambda_grid: np.ndarray,
    training_length: int,
    window_width: float,
) -> list[dict[str, Any]]:
    """计算 training boundary 之后固定物理窗口的正式指标。"""

    lambda64 = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    boundary = float(lambda64[training_length - 1])
    rows: list[dict[str, Any]] = []
    for start, end, indices, is_final in build_extrapolation_windows(
        lambda64,
        training_length,
        window_width,
    ):
        metrics = compute_region_metrics(prediction[:, indices, :], truth[:, indices, :], np.arange(prediction.shape[0]))
        rows.append(
            {
                "total_length": int(total_length),
                "lambda_start": float(start),
                "lambda_end": float(end),
                "distance_from_training_boundary_start": float(start - boundary),
                "distance_from_training_boundary_end": float(end - boundary),
                "point_count": int(indices.size),
                "interval_right_closed": bool(is_final),
                "mse": float(metrics["mse"]),
                "global_relative_l2": float(metrics["global_relative_l2"]),
                "mean_per_q_relative_l2": float(metrics["mean_per_q_relative_l2"]),
            }
        )
    return rows


def evaluate_three_lengths(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
    device: str,
    window_width: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """对三个总长度各运行一次冻结前向，并汇总正式指标。"""

    training_length = int(short.lambda_grid.size)
    fields = (short, medium, long)
    results: dict[str, dict[str, Any]] = {}
    per_q_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for field in fields:
        total_length = int(field.lambda_grid.size)
        start = perf_counter()
        prediction = run_frozen_inference(
            model=model,
            checkpoint=checkpoint,
            field=field,
            device=device,
        )
        seconds = perf_counter() - start
        prefix_prediction = prediction[:, :training_length, :]
        prefix_truth = field.canonical_truth[:, :training_length, :]
        has_extrapolation = total_length > training_length
        results[f"T{total_length}"] = {
            "task_name": field.task_name,
            "total_length": total_length,
            "lambda_min": float(field.lambda_grid[0]),
            "lambda_max": float(field.lambda_grid[-1]),
            "prefix": compute_region_metrics(prefix_prediction, prefix_truth, field.canonical_q),
            "extrapolation": (
                compute_region_metrics(
                    prediction[:, training_length:, :],
                    field.canonical_truth[:, training_length:, :],
                    field.canonical_q,
                )
                if has_extrapolation
                else None
            ),
            "full": compute_region_metrics(prediction, field.canonical_truth, field.canonical_q),
            "inference_seconds": float(seconds),
            "frozen_forward_passes": 1,
        }
        per_q_rows.extend(
            compute_per_q_rows(
                total_length=total_length,
                prediction=prediction,
                truth=field.canonical_truth,
                q_values=field.canonical_q,
                training_length=training_length,
            )
        )
        if has_extrapolation:
            window_rows.extend(
                compute_window_rows(
                    total_length=total_length,
                    prediction=prediction,
                    truth=field.canonical_truth,
                    lambda_grid=field.lambda_grid,
                    training_length=training_length,
                    window_width=window_width,
                )
            )
    return results, per_q_rows, window_rows


def _ordering_provenance(field: CanonicalQField) -> dict[str, Any]:
    """记录来源身份与 canonical 模型 Q 轴之间的可逆映射。"""

    return {
        "source_identity_order": "train_then_val_then_test_original_row_order",
        "canonical_model_input_q_order": "ascending_Q_full_field",
        "canonical_to_source_index": field.canonical_to_source_index,
        "source_to_canonical_index": field.source_to_canonical_index,
        "source_records_in_canonical_order": field.source_records,
    }


def _grid_summary(lambda_grid: np.ndarray) -> dict[str, Any]:
    """记录物理 lambda 网格范围与局部步长范围。"""

    values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    differences = np.diff(values)
    return {
        "count": int(values.size),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
        "delta_lambda_min": float(np.min(differences)) if differences.size else None,
        "delta_lambda_max": float(np.max(differences)) if differences.size else None,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    pair_validation_path: Path,
    pair_validation: dict[str, Any],
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """构造可复核但不包含大型预测数组的正式 summary JSON。"""

    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform_config = load_target_transform_config_from_checkpoint(checkpoint)
    training_length = int(short.lambda_grid.size)
    return {
        "schema_version": "1.0",
        "diagnostic_type": "formal_frozen_one_shot_lambda_domain_length_extrapolation",
        "status": "completed",
        "training_task_name": str(args.training_task_name),
        "tasks": {
            "short": short.task_name,
            "medium": medium.task_name,
            "long": long.task_name,
        },
        "model_name": str(args.model_name),
        "checkpoint_path": _relative_path(checkpoint_path),
        "model_config": _as_json_value(checkpoint.get("config", {}).get("model_config", {})),
        "normalization": {
            "statistics_source": "checkpoint_training_dataset",
            "values": _as_json_value(stats.to_dict()),
        },
        "target_transform": _as_json_value(transform_config.to_dict()),
        "stage2_prefix_validation": {
            "artifact_path": _relative_path(pair_validation_path),
            "required_pair_classification": {
                key: pair_validation["pair_classification"][key]
                for key in ("short_to_medium", "short_to_long", "medium_to_long")
            },
            "required_scientific_reuse": {
                key: pair_validation["scientific_reuse"][key]
                for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1")
            },
        },
        "ordering": {
            "short": _ordering_provenance(short),
            "medium": _ordering_provenance(medium),
            "long": _ordering_provenance(long),
            "canonical_q_exact_match_across_all_lengths": True,
        },
        "lengths": {
            "short": _grid_summary(short.lambda_grid),
            "medium": _grid_summary(medium.lambda_grid),
            "long": _grid_summary(long.lambda_grid),
        },
        "training_domain_boundary": {
            "short_length": training_length,
            "lambda_last_training_point": float(short.lambda_grid[-1]),
            "lambda_first_extrapolation_point": float(medium.lambda_grid[training_length]),
            "derived_from": "short_task_lambda_grid",
        },
        "window_width": float(args.window_width),
        "interval_semantics": "[lambda_start, lambda_end) except the clipped final window is right-closed",
        "frozen_inference": {
            "one_full_forward_pass_per_total_length": True,
            "total_forward_passes": 3,
            "model_input_channels": ["Q", "lambda"],
            "truth_used_as_model_input": False,
            "autoregressive_rollout": False,
            "prediction_feedback": False,
            "teacher_forcing": False,
            "adaptation": "none",
            "training": False,
        },
        "metric_aggregation": {
            "evaluation_space": "raw_physical_xyz_float64",
            "prediction_promotion": "float32_prediction_promoted_to_float64_before_metrics",
            "truth_source": "raw_dataset_truth_float64",
            "mse": "mean_over_Q_lambda_xyz",
            "global_relative_l2": "norm_over_all_Q_lambda_xyz_divided_by_truth_norm",
            "mean_per_q_relative_l2": "mean_of_per_Q_flattened_relative_l2; historical_T1800_relative_l2_comparable",
            "component_metrics": "per_component_mse_and_global_relative_l2",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": str(args.device),
        },
        "git_commit": _git_commit(),
        "results": results,
        "output_files": list(OUTPUT_FILENAMES),
    }


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """写入紧凑 CSV，并将 None 表示为显式空字段。"""

    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_output_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    per_q_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> None:
    """以独占目录写入正式 A1 的三份紧凑产物。"""

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_as_json_value(summary), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        write_csv(
            per_q_rows,
            output_dir / OUTPUT_FILENAMES[1],
            [
                "total_length",
                "Q",
                "prefix_mse",
                "prefix_relative_l2",
                "extrapolation_mse",
                "extrapolation_relative_l2",
                "full_mse",
                "full_relative_l2",
            ],
        )
        write_csv(
            window_rows,
            output_dir / OUTPUT_FILENAMES[2],
            [
                "total_length",
                "lambda_start",
                "lambda_end",
                "distance_from_training_boundary_start",
                "distance_from_training_boundary_end",
                "point_count",
                "interval_right_closed",
                "mse",
                "global_relative_l2",
                "mean_per_q_relative_l2",
            ],
        )
    except Exception:
        raise


def main() -> None:
    """执行正式 A1 三长度冻结评估。"""

    args = parse_args()
    if not np.isfinite(args.window_width) or args.window_width <= 0.0:
        raise ValueError("--window-width must be finite and positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Formal A1 output directory already exists: {args.output_dir}")

    pair_validation = load_required_pair_validation(args.dataset_pair_validation_json)
    short = load_task_raw_field(str(args.short_task_name))
    medium = load_task_raw_field(str(args.medium_task_name))
    long = load_task_raw_field(str(args.long_task_name))
    validate_triplet(short, medium, long)

    checkpoint_path = args.checkpoint_path or (
        PROJECT_ROOT
        / "outputs"
        / str(args.training_task_name)
        / str(args.model_name)
        / "checkpoints"
        / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = load_checkpoint_2d(checkpoint_path=checkpoint_path, device=str(args.device))
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))

    results, per_q_rows, window_rows = evaluate_three_lengths(
        model=model,
        checkpoint=checkpoint,
        short=short,
        medium=medium,
        long=long,
        device=str(args.device),
        window_width=float(args.window_width),
    )
    summary = build_summary(
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        pair_validation_path=args.dataset_pair_validation_json,
        pair_validation=pair_validation,
        short=short,
        medium=medium,
        long=long,
        results=results,
    )
    write_output_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        per_q_rows=per_q_rows,
        window_rows=window_rows,
    )
    print("Formal A1 length-extrapolation evaluation completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
