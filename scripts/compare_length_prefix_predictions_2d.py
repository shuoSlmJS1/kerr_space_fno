# ==========================================================
# File: scripts/compare_length_prefix_predictions_2d.py
#
# 功能：
# 1. 读取 T=1200 公共测试结果；
# 2. 读取 T=1800 长度外推验证结果；
# 3. 选择完全相同的 Q；
# 4. 比较：
#    - T=1200 输入产生的预测；
#    - T=1800 输入产生的预测的前 1200 点；
# 5. 判断改变输入长度是否破坏原区间预测。
#
# 终端输出使用英文，代码注释使用中文。
# ==========================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def load_sorted_task(
    task_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载任务并按 Q 排序。"""
    data = load_npz(
        get_task_dataset_npz_path(task_name)
    )

    x_all = np.concatenate(
        [
            data["x_train"],
            data["x_val"],
            data["x_test"],
        ],
        axis=0,
    )

    y_all = np.concatenate(
        [
            data["y_train"],
            data["y_val"],
            data["y_test"],
        ],
        axis=0,
    )

    q_values = np.asarray(
        x_all[:, 0],
        dtype=np.float64,
    )

    order = np.argsort(q_values)

    return (
        q_values[order],
        np.asarray(y_all[order], dtype=np.float32),
        np.asarray(data["lambda_grid"], dtype=np.float64),
    )


def build_raw_field(
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    y_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 [1,H,W,2] 输入场和 [1,H,W,3] 输出场。"""
    q_values = np.asarray(
        q_values,
        dtype=np.float32,
    )

    lambda_grid = np.asarray(
        lambda_grid,
        dtype=np.float32,
    )

    num_q = int(q_values.shape[0])
    num_lambda = int(lambda_grid.shape[0])

    q_channel = np.broadcast_to(
        q_values[:, None],
        (num_q, num_lambda),
    )

    lambda_channel = np.broadcast_to(
        lambda_grid[None, :],
        (num_q, num_lambda),
    )

    x_raw = np.stack(
        [q_channel, lambda_channel],
        axis=-1,
    )[None, ...]

    return (
        x_raw.astype(np.float32),
        y_raw[None, ...].astype(np.float32),
    )


def relative_l2_per_q(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """按 Q 计算 Relative L2。"""
    num_q = int(a.shape[1])

    a_flat = a.reshape(num_q, -1)
    b_flat = b.reshape(num_q, -1)

    return (
        np.linalg.norm(a_flat - b_flat, axis=1)
        / (
            np.linalg.norm(b_flat, axis=1)
            + eps
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare T=1200 predictions with the first 1200 "
            "points of T=1800 predictions for identical Q values."
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
        "--short-task-name",
        required=True,
    )

    parser.add_argument(
        "--long-validation-dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    device = str(args.device)

    long_dir = Path(
        args.long_validation_dir
    )

    long_q = np.load(
        long_dir / "Q.npy"
    )

    long_pred = np.load(
        long_dir / "predictions_raw.npy"
    )

    long_target = np.load(
        long_dir / "targets_raw.npy"
    )

    short_q_all, short_y_all, short_lambda = (
        load_sorted_task(
            str(args.short_task_name)
        )
    )

    # 找到与长验证结果完全对应的 Q 索引。
    selected_indices = []

    for q_value in long_q:
        matches = np.where(
            np.isclose(
                short_q_all,
                q_value,
                atol=0.0,
                rtol=0.0,
            )
        )[0]

        if matches.shape[0] != 1:
            raise RuntimeError(
                f"Could not uniquely match Q={q_value:.16e}."
            )

        selected_indices.append(
            int(matches[0])
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    short_q = short_q_all[
        selected_indices
    ]

    short_y = short_y_all[
        selected_indices
    ]

    if not np.array_equal(
        short_q,
        long_q,
    ):
        raise RuntimeError(
            "Short and long Q values are not exactly identical."
        )

    train_length = int(
        short_lambda.shape[0]
    )

    if train_length != 1200:
        raise ValueError(
            f"Expected short length 1200, got {train_length}."
        )

    long_truth_prefix = long_target[
        0,
        :,
        :train_length,
        :,
    ]

    if short_y.shape != long_truth_prefix.shape:
        raise RuntimeError(
            "Short-task truth and long-task truth prefix have "
            f"different shapes: short={short_y.shape}, "
            f"long_prefix={long_truth_prefix.shape}."
        )

    truth_max_abs_diff = float(
        np.max(
            np.abs(
                short_y
                - long_truth_prefix
            )
        )
    )

    if not np.allclose(
        short_y,
        long_truth_prefix,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError(
            "Short-task truth does not match the long-task prefix "
            "within float32 tolerance. "
            f"Maximum absolute difference={truth_max_abs_diff:.12e}."
        )

    print(
        "Truth-prefix maximum absolute difference: "
        f"{truth_max_abs_diff:.12e}"
    )

    x_raw, y_raw = build_raw_field(
        q_values=short_q,
        lambda_grid=short_lambda,
        y_raw=short_y,
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / str(args.training_task_name)
        / str(args.model_name)
        / "checkpoints"
        / "best_model.pt"
    )

    checkpoint = load_checkpoint_2d(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    stats = load_normalization_stats_from_checkpoint(
        checkpoint
    )

    transform_config = (
        load_target_transform_config_from_checkpoint(
            checkpoint
        )
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

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_model).float(),
            torch.from_numpy(y_model).float(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = load_fno2d_checkpoint_model(
        checkpoint=checkpoint,
        device=device,
    )

    short_pred_model, short_target_model = (
        predict_2d_loader(
            model=model,
            loader=loader,
            device=device,
        )
    )

    short_pred_raw, short_target_raw = (
        recover_predictions_and_targets_to_raw_xyz(
            predictions_model_space=short_pred_model,
            targets_model_space=short_target_model,
            raw_targets_reference=y_raw,
            normalization_stats=stats,
            target_transform_config=transform_config,
        )
    )

    long_prefix_pred = long_pred[
        :,
        :,
        :train_length,
        :,
    ]

    long_prefix_target = long_target[
        :,
        :,
        :train_length,
        :,
    ]

    short_truth_error = relative_l2_per_q(
        short_pred_raw,
        short_target_raw,
    )

    long_prefix_truth_error = relative_l2_per_q(
        long_prefix_pred,
        long_prefix_target,
    )

    prediction_difference = relative_l2_per_q(
        long_prefix_pred,
        short_pred_raw,
    )

    print("=" * 84)
    print("Length-Change Prediction Consistency")
    print("=" * 84)
    print(f"Q trajectories                   : {long_q.shape[0]}")
    print(f"Compared prefix length           : {train_length}")
    print("-" * 84)
    print(
        "Short-input prediction vs truth : "
        f"{float(np.mean(short_truth_error)):.6e}"
    )
    print(
        "Long-input prefix vs truth       : "
        f"{float(np.mean(long_prefix_truth_error)):.6e}"
    )
    print(
        "Long prefix vs short prediction  : "
        f"{float(np.mean(prediction_difference)):.6e}"
    )
    print("-" * 84)

    for index, q_value in enumerate(long_q):
        print(
            f"Q={q_value:.8f} | "
            f"short_truth={short_truth_error[index]:.6e} | "
            f"long_truth={long_prefix_truth_error[index]:.6e} | "
            f"long_vs_short={prediction_difference[index]:.6e}"
        )

    print("=" * 84)


if __name__ == "__main__":
    main()
