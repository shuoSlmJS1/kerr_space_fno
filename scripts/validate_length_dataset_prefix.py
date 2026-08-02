# ==========================================================
# File: scripts/validate_length_dataset_prefix.py
#
# 功能：
# 1. 将两个不同长度任务的 train/val/test 重新拼接并按 Q 排序；
# 2. 检查两份数据是否使用完全相同的 Q 网格；
# 3. 检查长任务的前缀轨道是否与短任务完全一致；
# 4. 防止不同数据、补样或求解器变化混入外推比较。
#
# 终端输出使用英文，代码注释使用中文。
# ==========================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_sorted_task(
    task_name: str,
) -> dict[str, Any]:
    """加载任务，将全部 split 拼接后按 Q 排序。"""
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "tasks"
        / task_name
        / "dataset.npz"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}"
        )

    with np.load(dataset_path) as data:
        required_keys = [
            "x_train",
            "x_val",
            "x_test",
            "y_train",
            "y_val",
            "y_test",
            "lambda_grid",
        ]

        missing = [
            key
            for key in required_keys
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Dataset is missing required keys: {missing}"
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
            data["lambda_grid"]
        ).copy()

    if x_all.ndim != 2 or x_all.shape[1] != 1:
        raise ValueError(
            "Only one varying parameter is supported; "
            f"current x shape={x_all.shape}."
        )

    q_values = x_all[:, 0]
    order = np.argsort(q_values)

    return {
        "task_name": task_name,
        "Q": q_values[order],
        "Y": y_all[order],
        "lambda_grid": lambda_grid,
    }


def main() -> None:
    """主流程。"""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that one generated trajectory dataset is an "
            "exact temporal prefix of another dataset."
        )
    )

    parser.add_argument(
        "--short-task-name",
        required=True,
    )

    parser.add_argument(
        "--long-task-name",
        required=True,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    short = load_sorted_task(
        str(args.short_task_name)
    )

    long = load_sorted_task(
        str(args.long_task_name)
    )

    short_length = int(
        short["lambda_grid"].shape[0]
    )

    long_length = int(
        long["lambda_grid"].shape[0]
    )

    if long_length < short_length:
        raise ValueError(
            "The long task must contain at least as many lambda "
            f"points as the short task: short={short_length}, "
            f"long={long_length}."
        )

    if short["Q"].shape != long["Q"].shape:
        raise ValueError(
            "The two tasks contain different numbers of Q points: "
            f"short={short['Q'].shape}, long={long['Q'].shape}."
        )

    q_exact = np.array_equal(
        short["Q"],
        long["Q"],
    )

    lambda_exact = np.array_equal(
        short["lambda_grid"],
        long["lambda_grid"][:short_length],
    )

    trajectory_exact = np.array_equal(
        short["Y"],
        long["Y"][:, :short_length, :],
    )

    q_max_abs_diff = float(
        np.max(
            np.abs(short["Q"] - long["Q"])
        )
    )

    lambda_max_abs_diff = float(
        np.max(
            np.abs(
                short["lambda_grid"]
                - long["lambda_grid"][:short_length]
            )
        )
    )

    trajectory_max_abs_diff = float(
        np.max(
            np.abs(
                short["Y"]
                - long["Y"][:, :short_length, :]
            )
        )
    )

    q_close = np.allclose(
        short["Q"],
        long["Q"],
        atol=float(args.atol),
        rtol=float(args.rtol),
    )

    lambda_close = np.allclose(
        short["lambda_grid"],
        long["lambda_grid"][:short_length],
        atol=float(args.atol),
        rtol=float(args.rtol),
    )

    trajectory_close = np.allclose(
        short["Y"],
        long["Y"][:, :short_length, :],
        atol=float(args.atol),
        rtol=float(args.rtol),
    )

    print("=" * 78)
    print("Length-Dataset Prefix Validation")
    print("=" * 78)
    print(f"Short task                 : {short['task_name']}")
    print(f"Long task                  : {long['task_name']}")
    print(f"Q points                   : {short['Q'].shape[0]}")
    print(f"Short lambda points        : {short_length}")
    print(f"Long lambda points         : {long_length}")
    print("-" * 78)
    print(f"Q exactly equal            : {q_exact}")
    print(f"Lambda prefix exact        : {lambda_exact}")
    print(f"Trajectory prefix exact    : {trajectory_exact}")
    print(f"Q max absolute difference  : {q_max_abs_diff:.12e}")
    print(f"Lambda max absolute diff   : {lambda_max_abs_diff:.12e}")
    print(f"Trajectory max absolute diff: {trajectory_max_abs_diff:.12e}")
    print("-" * 78)
    print(f"Q within tolerance         : {q_close}")
    print(f"Lambda within tolerance    : {lambda_close}")
    print(f"Trajectory within tolerance: {trajectory_close}")
    print("=" * 78)

    if not q_close:
        raise RuntimeError(
            "The Q grids are inconsistent."
        )

    if not lambda_close:
        raise RuntimeError(
            "The lambda grids are inconsistent in the shared prefix."
        )

    if not trajectory_close:
        raise RuntimeError(
            "The trajectory values are inconsistent in the shared prefix."
        )

    print("Prefix validation passed.")


if __name__ == "__main__":
    main()
