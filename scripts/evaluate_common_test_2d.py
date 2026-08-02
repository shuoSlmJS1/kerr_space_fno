# ==========================================================
# File: scripts/evaluate_common_test_2d.py
#
# 功能：
# 1. 读取一份独立生成的公共测试任务；
# 2. 将该任务的 train/val/test 重新拼接成完整公共测试集；
# 3. 对多个已有 FNO2D checkpoint 使用同一测试场评价；
# 4. 每个模型使用自己 checkpoint 中保存的训练归一化统计量；
# 5. 在 raw xyz 物理空间计算正式 MSE 和 Relative L2；
# 6. 保存整体指标和逐 Q 轨道指标。
#
# 重要说明：
# - 公共测试集绝不用于重新计算 normalization statistics；
# - 每个 checkpoint 使用其训练阶段保存的 statistics；
# - 原有模型目录和 summary.json 不会被修改；
# - 终端和输出文件文本使用英文，代码注释使用中文。
# ==========================================================

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.io_utils import load_json, load_npz  # noqa: E402
from src.common.paths import (  # noqa: E402
    get_task_dataset_npz_path,
    get_task_meta_json_path,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    transform_output_field,
)
from scripts.run_analysis_2d import (  # noqa: E402
    compute_metrics_2d,
    get_normalization_method_from_checkpoint,
    get_target_transform_method_from_checkpoint,
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)


DEFAULT_MODEL_TASK_NAMES = [
    "q_1p6-3_n500_t1200",
    "q_1p6-3_n1000_t1200",
    "q_1p6-3_n2000_t1200",
    "q_1p6-3_n5000_t1200",
]

DEFAULT_MODEL_NAME = "fno2d_m16x32_w64_d4_e300"


# ==========================================================
# 一、通用文件工具
# ==========================================================

def save_json(
    data: Any,
    path: Path,
) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def safe_nested_get(
    data: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    """安全读取嵌套字典字段。"""
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ==========================================================
# 二、公共测试原始数据
# ==========================================================

def load_complete_common_test_task(
    task_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    读取公共测试任务，并把 train/val/test 重新拼接。

    返回：
    - q_values:    [H]
    - y_raw:       [H,W,3]
    - lambda_grid: [W]
    - meta
    """
    dataset_path = get_task_dataset_npz_path(task_name)
    meta_path = get_task_meta_json_path(task_name)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Common-test dataset does not exist: {dataset_path}"
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Common-test metadata does not exist: {meta_path}"
        )

    data = load_npz(dataset_path)
    meta = load_json(meta_path)

    required_keys = [
        "x_train",
        "x_val",
        "x_test",
        "y_train",
        "y_val",
        "y_test",
        "lambda_grid",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            f"Common-test dataset is missing keys: {missing_keys}"
        )

    generation_status = meta.get("generation_status", {})

    if generation_status.get("completed") is not True:
        raise ValueError(
            "The common-test task is not marked as completely generated."
        )

    x_all = np.concatenate(
        [
            np.asarray(data["x_train"]),
            np.asarray(data["x_val"]),
            np.asarray(data["x_test"]),
        ],
        axis=0,
    )

    y_all = np.concatenate(
        [
            np.asarray(data["y_train"]),
            np.asarray(data["y_val"]),
            np.asarray(data["y_test"]),
        ],
        axis=0,
    )

    if x_all.ndim != 2 or x_all.shape[1] != 1:
        raise ValueError(
            "The current common-test evaluator supports one varying "
            f"parameter only; current x shape={x_all.shape}."
        )

    if y_all.ndim != 3 or y_all.shape[-1] != 3:
        raise ValueError(
            "Common-test y must have shape [H,W,3]; "
            f"current shape={y_all.shape}."
        )

    lambda_grid = np.asarray(
        data["lambda_grid"],
        dtype=np.float64,
    ).reshape(-1)

    if y_all.shape[1] != lambda_grid.shape[0]:
        raise ValueError(
            "The lambda dimension in y does not match lambda_grid; "
            f"y shape={y_all.shape}, lambda shape={lambda_grid.shape}."
        )

    q_values = x_all[:, 0].astype(np.float64)

    # 按 Q 从小到大排序，确保二维场参数轴顺序稳定。
    order = np.argsort(q_values)

    q_values = q_values[order]
    y_all = y_all[order]

    if np.unique(q_values).shape[0] != q_values.shape[0]:
        raise ValueError(
            "The common-test parameter values contain duplicates."
        )

    return (
        q_values,
        y_all.astype(np.float32),
        lambda_grid,
        meta,
    )


def build_raw_common_test_fields(
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    y_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造单配置 FNO2D raw field。

    输出：
    - x_raw: [1,H,W,2]，通道顺序为 [Q, lambda]
    - y_raw: [1,H,W,3]
    """
    q_values = np.asarray(
        q_values,
        dtype=np.float32,
    ).reshape(-1)

    lambda_grid = np.asarray(
        lambda_grid,
        dtype=np.float32,
    ).reshape(-1)

    y_raw = np.asarray(
        y_raw,
        dtype=np.float32,
    )

    num_param = int(q_values.shape[0])
    num_lambda = int(lambda_grid.shape[0])

    if y_raw.shape != (num_param, num_lambda, 3):
        raise ValueError(
            "Raw target shape does not match parameter and lambda axes; "
            f"current y={y_raw.shape}, expected="
            f"{(num_param, num_lambda, 3)}."
        )

    q_channel = np.broadcast_to(
        q_values[:, None],
        (num_param, num_lambda),
    )

    lambda_channel = np.broadcast_to(
        lambda_grid[None, :],
        (num_param, num_lambda),
    )

    x_raw = np.stack(
        [q_channel, lambda_channel],
        axis=-1,
    )[None, ...]

    y_raw_2d = y_raw[None, ...]

    return (
        x_raw.astype(np.float32),
        y_raw_2d.astype(np.float32),
    )


# ==========================================================
# 三、公共测试 DataLoader
# ==========================================================

def build_checkpoint_specific_test_loader(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    checkpoint: dict[str, Any],
) -> tuple[
    DataLoader,
    np.ndarray,
    Any,
    Any,
]:
    """
    使用当前 checkpoint 的训练统计量构造公共测试 DataLoader。

    注意：
    - 不从公共测试集重新计算 normalization statistics；
    - x 和 y 均使用 checkpoint 中的训练 statistics；
    - raw y 会额外保留，供最终恢复物理空间使用。
    """
    normalization_method = (
        get_normalization_method_from_checkpoint(checkpoint)
    )

    target_transform_method = (
        get_target_transform_method_from_checkpoint(checkpoint)
    )

    normalization_stats = (
        load_normalization_stats_from_checkpoint(checkpoint)
    )

    target_transform_config = (
        load_target_transform_config_from_checkpoint(checkpoint)
    )

    if normalization_stats.method != normalization_method:
        raise ValueError(
            "Checkpoint normalization method does not match stored "
            "normalization statistics: "
            f"method={normalization_method!r}, "
            f"stats.method={normalization_stats.method!r}."
        )

    if target_transform_config.mode != target_transform_method:
        raise ValueError(
            "Checkpoint target-transform method does not match stored "
            "target-transform configuration: "
            f"method={target_transform_method!r}, "
            f"config.mode={target_transform_config.mode!r}."
        )

    y_transformed = transform_output_field(
        y=y_raw,
        config=target_transform_config,
    )

    x_model = normalize_input_field(
        x=x_raw,
        stats=normalization_stats,
    )

    y_model = normalize_output_field(
        y=y_transformed,
        stats=normalization_stats,
    )

    dataset = TensorDataset(
        torch.from_numpy(x_model).float(),
        torch.from_numpy(y_model).float(),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return (
        loader,
        y_raw,
        normalization_stats,
        target_transform_config,
    )


# ==========================================================
# 四、指标
# ==========================================================

def compute_per_parameter_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    q_values: np.ndarray,
    eps: float = 1e-12,
) -> list[dict[str, float]]:
    """
    对每个 Q 对应的完整轨道计算 MSE 和 Relative L2。

    predictions / targets:
        [1,H,W,3]
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have identical shapes; "
            f"pred={predictions.shape}, target={targets.shape}."
        )

    if predictions.ndim != 4 or predictions.shape[0] != 1:
        raise ValueError(
            "The common-test evaluator expects one field with shape "
            f"[1,H,W,3]; current shape={predictions.shape}."
        )

    num_param = int(predictions.shape[1])

    if q_values.shape[0] != num_param:
        raise ValueError(
            "The number of Q values does not match the prediction field."
        )

    rows: list[dict[str, float]] = []

    for index in range(num_param):
        pred_i = predictions[0, index]
        target_i = targets[0, index]

        mse = float(
            np.mean((pred_i - target_i) ** 2)
        )

        relative_l2 = float(
            np.linalg.norm(
                (pred_i - target_i).reshape(-1)
            )
            / (
                np.linalg.norm(target_i.reshape(-1))
                + eps
            )
        )

        rows.append(
            {
                "index": int(index),
                "Q": float(q_values[index]),
                "mse": mse,
                "relative_l2": relative_l2,
            }
        )

    return rows


def summarize_per_parameter_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    """汇总逐轨道误差分布。"""
    mse_values = np.asarray(
        [row["mse"] for row in rows],
        dtype=np.float64,
    )

    rel_values = np.asarray(
        [row["relative_l2"] for row in rows],
        dtype=np.float64,
    )

    return {
        "mse_mean_over_q": float(np.mean(mse_values)),
        "mse_median_over_q": float(np.median(mse_values)),
        "mse_max_over_q": float(np.max(mse_values)),
        "relative_l2_mean_over_q": float(np.mean(rel_values)),
        "relative_l2_median_over_q": float(np.median(rel_values)),
        "relative_l2_max_over_q": float(np.max(rel_values)),
        "relative_l2_p90_over_q": float(
            np.percentile(rel_values, 90.0)
        ),
        "relative_l2_p95_over_q": float(
            np.percentile(rel_values, 95.0)
        ),
    }


# ==========================================================
# 五、单个模型评价
# ==========================================================

def evaluate_one_model(
    model_task_name: str,
    model_name: str,
    device: str,
    q_values: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    common_lambda_grid: np.ndarray,
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    """在公共测试场上评价一个已有模型。"""
    model_dir = (
        PROJECT_ROOT
        / "outputs"
        / model_task_name
        / model_name
    )

    checkpoint_path = (
        model_dir
        / "checkpoints"
        / "best_model.pt"
    )

    summary_path = model_dir / "summary.json"

    checkpoint = load_checkpoint_2d(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 检查模型训练时的 lambda 维长度。
    checkpoint_config = checkpoint.get("config", {})
    dataset_summary = checkpoint_config.get(
        "dataset_summary",
        {},
    )

    checkpoint_test_summary = dataset_summary.get(
        "test",
        {},
    )

    checkpoint_num_lambda = checkpoint_test_summary.get(
        "num_lambda",
        None,
    )

    if (
        checkpoint_num_lambda is not None
        and int(checkpoint_num_lambda)
        != int(common_lambda_grid.shape[0])
    ):
        raise ValueError(
            "The common-test lambda length does not match the model "
            f"training configuration: common={common_lambda_grid.shape[0]}, "
            f"checkpoint={checkpoint_num_lambda}."
        )

    loader, raw_reference, stats, transform_config = (
        build_checkpoint_specific_test_loader(
            x_raw=x_raw,
            y_raw=y_raw,
            checkpoint=checkpoint,
        )
    )

    model = load_fno2d_checkpoint_model(
        checkpoint=checkpoint,
        device=device,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    inference_start = perf_counter()

    predictions_model_space, targets_model_space = (
        predict_2d_loader(
            model=model,
            loader=loader,
            device=device,
        )
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    inference_seconds = perf_counter() - inference_start

    predictions_raw, targets_raw = (
        recover_predictions_and_targets_to_raw_xyz(
            predictions_model_space=predictions_model_space,
            targets_model_space=targets_model_space,
            raw_targets_reference=raw_reference,
            normalization_stats=stats,
            target_transform_config=transform_config,
        )
    )

    official_metrics = compute_metrics_2d(
        pred=predictions_raw,
        target=targets_raw,
    )

    per_q_rows = compute_per_parameter_metrics(
        predictions=predictions_raw,
        targets=targets_raw,
        q_values=q_values,
    )

    distribution_metrics = (
        summarize_per_parameter_metrics(per_q_rows)
    )

    original_summary: dict[str, Any] = {}

    if summary_path.exists():
        original_summary = load_json(summary_path)

    original_physical = safe_nested_get(
        original_summary,
        ["metrics", "physical_space"],
        {},
    )

    model_output_dir = (
        output_dir
        / "models"
        / model_task_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_q_csv_path = (
        model_output_dir
        / "per_q_metrics.csv"
    )

    with per_q_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "Q",
                "mse",
                "relative_l2",
            ],
        )

        writer.writeheader()
        writer.writerows(per_q_rows)

    if save_predictions:
        np.savez_compressed(
            model_output_dir / "predictions.npz",
            Q=q_values.astype(np.float64),
            lambda_grid=common_lambda_grid.astype(np.float64),
            predictions_raw=predictions_raw.astype(np.float32),
            targets_raw=targets_raw.astype(np.float32),
        )

    result = {
        "model_task_name": model_task_name,
        "model_name": model_name,
        "checkpoint_path": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "normalization": {
            "method": stats.method,
            "statistics_source": (
                "checkpoint_training_dataset"
            ),
            "x_mean": stats.x_mean,
            "x_std": stats.x_std,
            "y_mean": stats.y_mean,
            "y_std": stats.y_std,
        },
        "target_transform": transform_config.to_dict(),
        "common_test": {
            "num_parameter_points": int(
                q_values.shape[0]
            ),
            "num_lambda_points": int(
                common_lambda_grid.shape[0]
            ),
            "Q_min": float(q_values[0]),
            "Q_max": float(q_values[-1]),
        },
        "metrics": {
            "metric_space": "physical_space",
            "usage": "common_test_official_evaluation",
            "mse": float(official_metrics["mse"]),
            "relative_l2": float(
                official_metrics["relative_l2"]
            ),
            **distribution_metrics,
        },
        "original_task_test_metrics": {
            "mse": (
                original_physical.get("test_mse")
                if isinstance(original_physical, dict)
                else None
            ),
            "relative_l2": (
                original_physical.get("test_relative_l2")
                if isinstance(original_physical, dict)
                else None
            ),
        },
        "timing": {
            "model_inference_seconds": float(
                inference_seconds
            ),
        },
        "files": {
            "per_q_metrics_csv": str(
                per_q_csv_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "predictions_saved": bool(
                save_predictions
            ),
        },
    }

    save_json(
        result,
        model_output_dir / "result.json",
    )

    del model
    del checkpoint

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return result


# ==========================================================
# 六、总结果保存
# ==========================================================

def save_comparison_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """保存公共测试模型对比表。"""
    rows = []

    for result in results:
        metrics = result["metrics"]
        original = result["original_task_test_metrics"]

        rows.append(
            {
                "model_task_name": result[
                    "model_task_name"
                ],
                "model_name": result["model_name"],
                "common_test_mse": metrics["mse"],
                "common_test_relative_l2": metrics[
                    "relative_l2"
                ],
                "common_test_relative_l2_median_over_q": (
                    metrics[
                        "relative_l2_median_over_q"
                    ]
                ),
                "common_test_relative_l2_p95_over_q": (
                    metrics[
                        "relative_l2_p95_over_q"
                    ]
                ),
                "common_test_relative_l2_max_over_q": (
                    metrics[
                        "relative_l2_max_over_q"
                    ]
                ),
                "original_test_mse": original["mse"],
                "original_test_relative_l2": original[
                    "relative_l2"
                ],
                "model_inference_seconds": result[
                    "timing"
                ]["model_inference_seconds"],
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_comparison_table(
    results: list[dict[str, Any]],
) -> None:
    """在终端打印简洁对比表。"""
    print("=" * 112)
    print("FNO2D Common-Test Comparison")
    print("=" * 112)

    print(
        f"{'Training Task':<29s} "
        f"{'Common MSE':>13s} "
        f"{'Common RelL2':>14s} "
        f"{'Median RelL2':>14s} "
        f"{'P95 RelL2':>13s} "
        f"{'Original RelL2':>15s}"
    )

    print("-" * 112)

    for result in results:
        metrics = result["metrics"]
        original = result["original_task_test_metrics"]

        original_rel = original["relative_l2"]

        original_text = (
            f"{float(original_rel):.6e}"
            if original_rel is not None
            else "N/A"
        )

        print(
            f"{result['model_task_name']:<29s} "
            f"{metrics['mse']:13.6e} "
            f"{metrics['relative_l2']:14.6e} "
            f"{metrics['relative_l2_median_over_q']:14.6e} "
            f"{metrics['relative_l2_p95_over_q']:13.6e} "
            f"{original_text:>15s}"
        )

    print("=" * 112)


# ==========================================================
# 七、CLI
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing FNO2D checkpoints on one shared "
            "independent common-test field."
        )
    )

    parser.add_argument(
        "--common-task-name",
        required=True,
        help=(
            "Generated task used only as the shared common-test "
            "dataset."
        ),
    )

    parser.add_argument(
        "--model-task-names",
        nargs="+",
        default=DEFAULT_MODEL_TASK_NAMES,
        help=(
            "Training task names whose checkpoints will be "
            "evaluated."
        ),
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Shared model directory name.",
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="Evaluation device.",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional output directory name under "
            "outputs/comparison."
        ),
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save raw physical-space predictions for every model."
        ),
    )

    return parser


def main() -> None:
    """统一测试评价主流程。"""
    args = build_parser().parse_args()

    common_task_name = str(args.common_task_name)
    model_task_names = [
        str(name)
        for name in args.model_task_names
    ]
    model_name = str(args.model_name)
    device = str(args.device)

    output_name = (
        str(args.output_name)
        if args.output_name is not None
        else f"common_test__{common_task_name}"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "comparison"
        / output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        q_values,
        y_raw_flat,
        lambda_grid,
        common_meta,
    ) = load_complete_common_test_task(
        task_name=common_task_name,
    )

    x_raw, y_raw = build_raw_common_test_fields(
        q_values=q_values,
        lambda_grid=lambda_grid,
        y_raw=y_raw_flat,
    )

    # 保存公共测试数据快照，保证后续可复现。
    common_snapshot_path = (
        output_dir
        / "common_test_dataset.npz"
    )

    np.savez_compressed(
        common_snapshot_path,
        Q=q_values.astype(np.float64),
        lambda_grid=lambda_grid.astype(np.float64),
        y_raw=y_raw.astype(np.float32),
    )

    common_test_summary = {
        "schema_version": "1.0",
        "purpose": (
            "Shared independent physical-space evaluation "
            "for data-scale comparison."
        ),
        "common_task_name": common_task_name,
        "common_task_meta": common_meta,
        "construction": {
            "source_splits_combined": [
                "train",
                "val",
                "test",
            ],
            "sorted_by_parameter": True,
            "input_channel_order": [
                "Q",
                "lambda",
            ],
            "normalization_policy": (
                "Each model uses normalization statistics stored "
                "in its own training checkpoint. No statistics are "
                "computed from the common-test dataset."
            ),
        },
        "num_parameter_points": int(
            q_values.shape[0]
        ),
        "num_lambda_points": int(
            lambda_grid.shape[0]
        ),
        "Q_min": float(q_values[0]),
        "Q_max": float(q_values[-1]),
        "lambda_min": float(lambda_grid[0]),
        "lambda_max": float(lambda_grid[-1]),
        "dataset_snapshot": str(
            common_snapshot_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    save_json(
        common_test_summary,
        output_dir / "common_test_summary.json",
    )

    print("=" * 72)
    print("FNO2D Shared Common-Test Evaluation")
    print("=" * 72)
    print(f"Common task          : {common_task_name}")
    print(f"Parameter points     : {q_values.shape[0]}")
    print(f"Lambda points        : {lambda_grid.shape[0]}")
    print(
        "Q range              : "
        f"[{q_values[0]:.8f}, {q_values[-1]:.8f}]"
    )
    print(f"Models to evaluate   : {len(model_task_names)}")
    print(f"Device               : {device}")

    results = []

    for index, model_task_name in enumerate(
        model_task_names,
        start=1,
    ):
        print()
        print("=" * 72)
        print(
            f"Evaluating model {index}/{len(model_task_names)}"
        )
        print("=" * 72)
        print(f"Training task        : {model_task_name}")
        print(f"Model                : {model_name}")

        result = evaluate_one_model(
            model_task_name=model_task_name,
            model_name=model_name,
            device=device,
            q_values=q_values,
            x_raw=x_raw,
            y_raw=y_raw,
            common_lambda_grid=lambda_grid,
            output_dir=output_dir,
            save_predictions=bool(
                args.save_predictions
            ),
        )

        results.append(result)

        print(
            "Common-test MSE       : "
            f"{result['metrics']['mse']:.6e}"
        )

        print(
            "Common-test RelL2     : "
            f"{result['metrics']['relative_l2']:.6e}"
        )

    combined_result = {
        "schema_version": "1.0",
        "common_test": common_test_summary,
        "model_name": model_name,
        "device": device,
        "results": results,
    }

    save_json(
        combined_result,
        output_dir / "common_test_results.json",
    )

    save_comparison_csv(
        results=results,
        output_path=(
            output_dir
            / "common_test_results.csv"
        ),
    )

    print()
    print_comparison_table(results)

    print()
    print(f"Saved output directory: {output_dir}")


if __name__ == "__main__":
    main()
