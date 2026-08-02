# ==========================================================
# File: scripts/evaluate_length_extrapolation_2d.py
#
# 功能：
# 1. 加载一个在短 lambda 长度上训练好的 FNO2D checkpoint；
# 2. 在更长 lambda 网格上直接进行零样本长度外推；
# 3. 使用 checkpoint 中保存的训练归一化统计量；
# 4. 分别计算 seen、extrapolation 和 full 三个区间的物理指标；
# 5. 保存逐 Q、逐 lambda 指标以及后续 3D HTML 所需数组；
# 6. 自动选择 low/mid/high/best/median/worst 六条代表轨道。
#
# 重要说明：
# - 不重新训练模型；
# - 不从外推数据中重新计算 normalization statistics；
# - 原模型目录和原 summary.json 不会被修改；
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
    get_normalization_method_from_checkpoint,
    get_target_transform_method_from_checkpoint,
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)


# ==========================================================
# 一、文件工具
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


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """保存 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(
            f"No rows are available for CSV output: {path}"
        )

    with path.open(
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


# ==========================================================
# 二、长任务读取与二维场构造
# ==========================================================

def load_complete_long_task(
    task_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    将长任务的全部 split 拼接并按 Q 排序。

    返回：
    - q_values:    [H]
    - y_raw:       [H,W,3]
    - lambda_grid: [W]
    - meta
    """
    dataset_path = get_task_dataset_npz_path(
        task_name
    )

    meta_path = get_task_meta_json_path(
        task_name
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}"
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata does not exist: {meta_path}"
        )

    data = load_npz(dataset_path)
    meta = load_json(meta_path)

    generation_status = meta.get(
        "generation_status",
        {},
    )

    if generation_status.get("completed") is not True:
        raise ValueError(
            "The extrapolation dataset is not marked as complete."
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

    lambda_grid = np.asarray(
        data["lambda_grid"],
        dtype=np.float64,
    ).reshape(-1)

    if x_all.ndim != 2 or x_all.shape[1] != 1:
        raise ValueError(
            "The current evaluator supports one varying parameter "
            f"only; current x shape={x_all.shape}."
        )

    if y_all.ndim != 3 or y_all.shape[-1] != 3:
        raise ValueError(
            "Raw trajectories must have shape [H,W,3]; "
            f"current shape={y_all.shape}."
        )

    if y_all.shape[1] != lambda_grid.shape[0]:
        raise ValueError(
            "Trajectory length does not match lambda_grid."
        )

    q_values = x_all[:, 0].astype(np.float64)
    order = np.argsort(q_values)

    q_values = q_values[order]
    y_all = y_all[order]

    if np.unique(q_values).shape[0] != q_values.shape[0]:
        raise ValueError(
            "The extrapolation task contains duplicate Q values."
        )

    return (
        q_values,
        y_all.astype(np.float32),
        lambda_grid,
        meta,
    )


def select_validation_subset(
    q_values: np.ndarray,
    y_raw: np.ndarray,
    max_q_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    流程验证时，在整个 Q 区间均匀选择少量轨道。

    返回：
    - subset_q
    - subset_y
    - original_indices
    """
    num_q = int(q_values.shape[0])

    if max_q_samples is None:
        indices = np.arange(
            num_q,
            dtype=np.int64,
        )

        return q_values, y_raw, indices

    sample_count = int(max_q_samples)

    if sample_count <= 0:
        raise ValueError(
            "--max-q-samples must be positive."
        )

    if sample_count >= num_q:
        indices = np.arange(
            num_q,
            dtype=np.int64,
        )
    else:
        indices = np.unique(
            np.rint(
                np.linspace(
                    0,
                    num_q - 1,
                    sample_count,
                )
            ).astype(np.int64)
        )

    return (
        q_values[indices],
        y_raw[indices],
        indices,
    )


def build_raw_field(
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    y_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造二维输入输出场。

    x_raw:
        [1,H,W,2]，通道顺序为 [Q, lambda]

    y_raw_2d:
        [1,H,W,3]
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

    num_q = int(q_values.shape[0])
    num_lambda = int(lambda_grid.shape[0])

    expected_shape = (
        num_q,
        num_lambda,
        3,
    )

    if y_raw.shape != expected_shape:
        raise ValueError(
            f"Raw trajectory shape={y_raw.shape}, "
            f"expected={expected_shape}."
        )

    q_channel = np.broadcast_to(
        q_values[:, None],
        (num_q, num_lambda),
    )

    lambda_channel = np.broadcast_to(
        lambda_grid[None, :],
        (num_q, num_lambda),
    )

    x_raw = np.stack(
        [
            q_channel,
            lambda_channel,
        ],
        axis=-1,
    )[None, ...]

    y_raw_2d = y_raw[None, ...]

    return (
        x_raw.astype(np.float32),
        y_raw_2d.astype(np.float32),
    )


# ==========================================================
# 三、模型输入处理
# ==========================================================

def build_model_space_loader(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    checkpoint: dict[str, Any],
) -> tuple[
    DataLoader,
    Any,
    Any,
]:
    """
    使用 checkpoint 自己的训练统计量构造推理 loader。
    """
    normalization_method = (
        get_normalization_method_from_checkpoint(
            checkpoint
        )
    )

    transform_method = (
        get_target_transform_method_from_checkpoint(
            checkpoint
        )
    )

    stats = (
        load_normalization_stats_from_checkpoint(
            checkpoint
        )
    )

    transform_config = (
        load_target_transform_config_from_checkpoint(
            checkpoint
        )
    )

    if stats.method != normalization_method:
        raise ValueError(
            "Checkpoint normalization metadata is inconsistent."
        )

    if transform_config.mode != transform_method:
        raise ValueError(
            "Checkpoint target-transform metadata is inconsistent."
        )

    y_transformed = transform_output_field(
        y=y_raw,
        config=transform_config,
    )

    x_model = normalize_input_field(
        x=x_raw,
        stats=stats,
    )

    y_model = normalize_output_field(
        y=y_transformed,
        stats=stats,
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
        stats,
        transform_config,
    )


# ==========================================================
# 四、指标函数
# ==========================================================

def region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    start: int,
    end: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    计算指定 lambda 区间上的轨道级平均指标。

    pred / target:
        [1,H,W,3]
    """
    pred_region = pred[:, :, start:end, :]
    target_region = target[:, :, start:end, :]

    if pred_region.shape[2] <= 0:
        raise ValueError(
            f"Empty metric region: start={start}, end={end}."
        )

    mse = float(
        np.mean(
            (pred_region - target_region) ** 2
        )
    )

    num_q = int(pred_region.shape[1])

    pred_flat = pred_region.reshape(
        num_q,
        -1,
    )

    target_flat = target_region.reshape(
        num_q,
        -1,
    )

    diff_norm = np.linalg.norm(
        pred_flat - target_flat,
        axis=1,
    )

    target_norm = np.linalg.norm(
        target_flat,
        axis=1,
    )

    relative_l2_values = (
        diff_norm
        / (target_norm + eps)
    )

    return {
        "mse": mse,
        "relative_l2": float(
            np.mean(relative_l2_values)
        ),
        "relative_l2_median": float(
            np.median(relative_l2_values)
        ),
        "relative_l2_p95": float(
            np.percentile(
                relative_l2_values,
                95.0,
            )
        ),
        "relative_l2_max": float(
            np.max(relative_l2_values)
        ),
    }


def compute_per_q_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    q_values: np.ndarray,
    train_length: int,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """计算每条 Q 轨道的分段指标。"""
    num_q = int(q_values.shape[0])
    total_length = int(pred.shape[2])

    rows: list[dict[str, Any]] = []

    for q_index in range(num_q):
        pred_q = pred[
            0,
            q_index,
        ]

        target_q = target[
            0,
            q_index,
        ]

        def one_region(
            start: int,
            end: int,
        ) -> tuple[float, float]:
            pred_region = pred_q[start:end]
            target_region = target_q[start:end]

            mse_value = float(
                np.mean(
                    (pred_region - target_region) ** 2
                )
            )

            rel_value = float(
                np.linalg.norm(
                    (
                        pred_region
                        - target_region
                    ).reshape(-1)
                )
                / (
                    np.linalg.norm(
                        target_region.reshape(-1)
                    )
                    + eps
                )
            )

            return mse_value, rel_value

        seen_mse, seen_rel = one_region(
            0,
            train_length,
        )

        extrap_mse, extrap_rel = one_region(
            train_length,
            total_length,
        )

        full_mse, full_rel = one_region(
            0,
            total_length,
        )

        rows.append(
            {
                "local_index": int(q_index),
                "Q": float(q_values[q_index]),
                "seen_mse": seen_mse,
                "seen_relative_l2": seen_rel,
                "extrapolation_mse": extrap_mse,
                "extrapolation_relative_l2": extrap_rel,
                "full_mse": full_mse,
                "full_relative_l2": full_rel,
                "extrapolation_to_seen_ratio": float(
                    extrap_rel
                    / (seen_rel + eps)
                ),
            }
        )

    return rows


def compute_lambda_profile(
    pred: np.ndarray,
    target: np.ndarray,
    lambda_grid: np.ndarray,
    train_length: int,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    计算每个 lambda 点上跨全部 Q 的误差。
    """
    diff = pred - target

    squared_xyz_error = np.mean(
        diff ** 2,
        axis=-1,
    )[0]

    euclidean_xyz_error = np.linalg.norm(
        diff,
        axis=-1,
    )[0]

    target_xyz_norm = np.linalg.norm(
        target,
        axis=-1,
    )[0]

    point_relative_error = (
        euclidean_xyz_error
        / (target_xyz_norm + eps)
    )

    rows: list[dict[str, Any]] = []

    for lambda_index in range(
        lambda_grid.shape[0]
    ):
        rows.append(
            {
                "lambda_index": int(
                    lambda_index
                ),
                "lambda": float(
                    lambda_grid[lambda_index]
                ),
                "region": (
                    "seen"
                    if lambda_index < train_length
                    else "extrapolation"
                ),
                "mean_mse_over_q": float(
                    np.mean(
                        squared_xyz_error[
                            :,
                            lambda_index,
                        ]
                    )
                ),
                "mean_absolute_xyz_error_over_q": float(
                    np.mean(
                        euclidean_xyz_error[
                            :,
                            lambda_index,
                        ]
                    )
                ),
                "mean_point_relative_error_over_q": float(
                    np.mean(
                        point_relative_error[
                            :,
                            lambda_index,
                        ]
                    )
                ),
                "median_point_relative_error_over_q": float(
                    np.median(
                        point_relative_error[
                            :,
                            lambda_index,
                        ]
                    )
                ),
            }
        )

    return rows


def choose_representative_trajectories(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    选择六条代表轨道：

    1. low-Q
    2. mid-Q
    3. high-Q
    4. best extrapolation
    5. median extrapolation
    6. worst extrapolation
    """
    if len(rows) < 3:
        raise ValueError(
            "At least three Q trajectories are required."
        )

    num_q = len(rows)

    low_index = 0
    mid_index = int(
        round((num_q - 1) / 2)
    )
    high_index = num_q - 1

    sorted_by_error = sorted(
        rows,
        key=lambda row: row[
            "extrapolation_relative_l2"
        ],
    )

    best_row = sorted_by_error[0]
    median_row = sorted_by_error[
        len(sorted_by_error) // 2
    ]
    worst_row = sorted_by_error[-1]

    selections = [
        (
            "low_q",
            rows[low_index],
        ),
        (
            "mid_q",
            rows[mid_index],
        ),
        (
            "high_q",
            rows[high_index],
        ),
        (
            "best_extrapolation",
            best_row,
        ),
        (
            "median_extrapolation",
            median_row,
        ),
        (
            "worst_extrapolation",
            worst_row,
        ),
    ]

    result = []

    for role, row in selections:
        result.append(
            {
                "role": role,
                "local_index": int(
                    row["local_index"]
                ),
                "Q": float(row["Q"]),
                "extrapolation_relative_l2": float(
                    row[
                        "extrapolation_relative_l2"
                    ]
                ),
                "full_relative_l2": float(
                    row["full_relative_l2"]
                ),
            }
        )

    return result


# ==========================================================
# 五、CLI
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """构造命令行接口。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a short-length FNO2D checkpoint on a "
            "longer lambda grid."
        )
    )

    parser.add_argument(
        "--training-task-name",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--extrapolation-task-name",
        required=True,
    )

    parser.add_argument(
        "--train-length",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--max-q-samples",
        type=int,
        default=None,
        help=(
            "Use a uniformly spaced subset of Q trajectories "
            "for pipeline validation."
        ),
    )

    parser.add_argument(
        "--output-name",
        default=None,
    )

    return parser


# ==========================================================
# 六、主流程
# ==========================================================

def main() -> None:
    """长度外推评价主流程。"""
    args = build_parser().parse_args()

    training_task_name = str(
        args.training_task_name
    )

    model_name = str(
        args.model_name
    )

    extrapolation_task_name = str(
        args.extrapolation_task_name
    )

    train_length = int(
        args.train_length
    )

    device = str(args.device)

    model_dir = (
        PROJECT_ROOT
        / "outputs"
        / training_task_name
        / model_name
    )

    checkpoint_path = (
        model_dir
        / "checkpoints"
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    (
        q_all,
        y_all,
        lambda_grid,
        extrapolation_meta,
    ) = load_complete_long_task(
        extrapolation_task_name
    )

    (
        q_values,
        y_selected,
        source_indices,
    ) = select_validation_subset(
        q_values=q_all,
        y_raw=y_all,
        max_q_samples=args.max_q_samples,
    )

    total_length = int(
        lambda_grid.shape[0]
    )

    if train_length <= 0:
        raise ValueError(
            "--train-length must be positive."
        )

    if train_length >= total_length:
        raise ValueError(
            "The extrapolation task must be longer than the "
            f"training length: train={train_length}, "
            f"total={total_length}."
        )

    x_raw, y_raw = build_raw_field(
        q_values=q_values,
        lambda_grid=lambda_grid,
        y_raw=y_selected,
    )

    checkpoint = load_checkpoint_2d(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    loader, stats, transform_config = (
        build_model_space_loader(
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

    start_time = perf_counter()

    (
        predictions_model_space,
        targets_model_space,
    ) = predict_2d_loader(
        model=model,
        loader=loader,
        device=device,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    inference_seconds = (
        perf_counter()
        - start_time
    )

    predictions_raw, targets_raw = (
        recover_predictions_and_targets_to_raw_xyz(
            predictions_model_space=(
                predictions_model_space
            ),
            targets_model_space=(
                targets_model_space
            ),
            raw_targets_reference=y_raw,
            normalization_stats=stats,
            target_transform_config=transform_config,
        )
    )

    if not np.all(
        np.isfinite(predictions_raw)
    ):
        raise FloatingPointError(
            "Predictions contain NaN or infinity."
        )

    if predictions_raw.shape != targets_raw.shape:
        raise ValueError(
            "Prediction and target shapes are inconsistent: "
            f"pred={predictions_raw.shape}, "
            f"target={targets_raw.shape}."
        )

    seen_metrics = region_metrics(
        pred=predictions_raw,
        target=targets_raw,
        start=0,
        end=train_length,
    )

    extrapolation_metrics = region_metrics(
        pred=predictions_raw,
        target=targets_raw,
        start=train_length,
        end=total_length,
    )

    full_metrics = region_metrics(
        pred=predictions_raw,
        target=targets_raw,
        start=0,
        end=total_length,
    )

    per_q_rows = compute_per_q_metrics(
        pred=predictions_raw,
        target=targets_raw,
        q_values=q_values,
        train_length=train_length,
    )

    lambda_rows = compute_lambda_profile(
        pred=predictions_raw,
        target=targets_raw,
        lambda_grid=lambda_grid,
        train_length=train_length,
    )

    selected_trajectories = (
        choose_representative_trajectories(
            per_q_rows
        )
    )

    validation_suffix = (
        f"validation_q{q_values.shape[0]}"
        if args.max_q_samples is not None
        else f"full_q{q_values.shape[0]}"
    )

    output_name = (
        str(args.output_name)
        if args.output_name is not None
        else (
            f"{extrapolation_task_name}__"
            f"{validation_suffix}"
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "extrapolation_2d"
        / training_task_name
        / model_name
        / output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "Q.npy",
        q_values.astype(np.float64),
    )

    np.save(
        output_dir / "source_q_indices.npy",
        source_indices.astype(np.int64),
    )

    np.save(
        output_dir / "lambda_grid.npy",
        lambda_grid.astype(np.float64),
    )

    np.save(
        output_dir / "predictions_raw.npy",
        predictions_raw.astype(np.float32),
    )

    np.save(
        output_dir / "targets_raw.npy",
        targets_raw.astype(np.float32),
    )

    write_csv(
        per_q_rows,
        output_dir / "per_q_metrics.csv",
    )

    write_csv(
        lambda_rows,
        output_dir / "lambda_error_profile.csv",
    )

    save_json(
        {
            "schema_version": "1.0",
            "training_task_name": training_task_name,
            "model_name": model_name,
            "checkpoint_path": str(
                checkpoint_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "extrapolation_task_name": (
                extrapolation_task_name
            ),
            "extrapolation_task_meta": (
                extrapolation_meta
            ),
            "train_length": train_length,
            "total_length": total_length,
            "seen_length": train_length,
            "extrapolation_length": (
                total_length - train_length
            ),
            "num_q": int(
                q_values.shape[0]
            ),
            "source_num_q": int(
                q_all.shape[0]
            ),
            "subset_mode": (
                "validation_subset"
                if args.max_q_samples is not None
                else "full_dataset"
            ),
            "lambda_train_max": float(
                lambda_grid[
                    train_length - 1
                ]
            ),
            "lambda_total_max": float(
                lambda_grid[-1]
            ),
            "normalization": {
                "method": stats.method,
                "statistics_source": (
                    "checkpoint_training_dataset"
                ),
            },
            "target_transform": (
                transform_config.to_dict()
            ),
            "metrics": {
                "seen": seen_metrics,
                "extrapolation": (
                    extrapolation_metrics
                ),
                "full": full_metrics,
                "extrapolation_to_seen_relative_l2_ratio": float(
                    extrapolation_metrics[
                        "relative_l2"
                    ]
                    / (
                        seen_metrics[
                            "relative_l2"
                        ]
                        + 1e-12
                    )
                ),
            },
            "inference": {
                "device": device,
                "seconds": float(
                    inference_seconds
                ),
                "prediction_shape": list(
                    predictions_raw.shape
                ),
                "target_shape": list(
                    targets_raw.shape
                ),
                "finite_predictions": True,
            },
            "selected_trajectories": (
                selected_trajectories
            ),
            "saved_files": {
                "Q": "Q.npy",
                "source_q_indices": (
                    "source_q_indices.npy"
                ),
                "lambda_grid": (
                    "lambda_grid.npy"
                ),
                "predictions_raw": (
                    "predictions_raw.npy"
                ),
                "targets_raw": (
                    "targets_raw.npy"
                ),
                "per_q_metrics": (
                    "per_q_metrics.csv"
                ),
                "lambda_error_profile": (
                    "lambda_error_profile.csv"
                ),
            },
        },
        output_dir / "metrics.json",
    )

    print("=" * 82)
    print("FNO2D Length-Extrapolation Evaluation")
    print("=" * 82)
    print(f"Training task          : {training_task_name}")
    print(f"Model                  : {model_name}")
    print(f"Extrapolation task     : {extrapolation_task_name}")
    print(f"Q trajectories         : {q_values.shape[0]}")
    print(f"Training length        : {train_length}")
    print(f"Total length           : {total_length}")
    print(
        "Extrapolation length   : "
        f"{total_length - train_length}"
    )
    print(
        "Prediction shape       : "
        f"{tuple(predictions_raw.shape)}"
    )
    print("-" * 82)
    print(
        "Seen MSE               : "
        f"{seen_metrics['mse']:.6e}"
    )
    print(
        "Seen Relative L2       : "
        f"{seen_metrics['relative_l2']:.6e}"
    )
    print(
        "Extrapolation MSE      : "
        f"{extrapolation_metrics['mse']:.6e}"
    )
    print(
        "Extrapolation RelL2    : "
        f"{extrapolation_metrics['relative_l2']:.6e}"
    )
    print(
        "Full MSE               : "
        f"{full_metrics['mse']:.6e}"
    )
    print(
        "Full Relative L2       : "
        f"{full_metrics['relative_l2']:.6e}"
    )
    print(
        "Extrapolation/Seen     : "
        f"{extrapolation_metrics['relative_l2'] / (seen_metrics['relative_l2'] + 1e-12):.3f} x"
    )
    print("-" * 82)
    print("Selected trajectories:")

    for item in selected_trajectories:
        print(
            f"  {item['role']:<22s} "
            f"Q={item['Q']:.8f} "
            f"extrap_RelL2="
            f"{item['extrapolation_relative_l2']:.6e}"
        )

    print("=" * 82)
    print(f"Saved output directory: {output_dir}")


if __name__ == "__main__":
    main()
