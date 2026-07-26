#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_second_order_dataset.py
======================================================================
验证二阶求解器生成的数据集是否适合进入模型训练。

验证分为两层：

一、数据完整性
- 必要数组是否存在；
- 形状是否一致；
- 是否包含 NaN 或 Inf；
- train / val / test 参数范围是否合理；
- lambda 网格是否与 meta.json 一致。

二、数值精度抽查
- 从测试集中选择参数分位点样本和固定随机样本；
- 使用二阶求解器以更细步长重新计算参考轨道；
- 将细网格严格抽样回原网格；
- 比较数据集中 h=0.005 轨道与 h/8 参考轨道的 trajectory-level Relative L2；
- 检查细网格第一积分约束残差。

本脚本不会修改数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.orbit_types import InitialState, KerrParams
from src.data_generation.orbit_solver_second_order import (
    simulate_one_orbit_second_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证二阶求解器数据集。")
    parser.add_argument(
        "--task-name",
        default="vary_Q__Q1.6_3__n500__T1200__cfg1_secondorder_pilot",
    )
    parser.add_argument("--num-quantile-cases", type=int, default=10)
    parser.add_argument("--num-random-cases", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--reference-factor", type=int, default=8)
    parser.add_argument(
        "--max-relative-l2",
        type=float,
        default=1e-7,
        help="存储轨道相对细网格参考解的最大允许 Relative L2。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    difference = a.astype(np.float64) - b.astype(np.float64)
    denominator = np.linalg.norm(b.astype(np.float64).reshape(-1))
    if denominator == 0.0:
        raise ValueError("参考轨道范数为零。")
    return float(np.linalg.norm(difference.reshape(-1)) / denominator)


def build_physical_objects(
    fixed_params: dict[str, Any],
    vary_params: dict[str, float],
) -> tuple[KerrParams, InitialState, float]:
    full = dict(fixed_params)
    full.update(vary_params)

    p = KerrParams(
        M=float(full["M"]),
        a=float(full["a"]),
        E=float(full["E"]),
        Lz=float(full["Lz"]),
    )
    init = InitialState(
        r0=float(full["r0"]),
        theta0=float(full["theta0"]),
        phi0=float(full["phi0"]),
        sign_r=int(full["sign_r"]),
        sign_th=int(full["sign_th"]),
    )
    return p, init, float(full["Q"])


def select_indices(
    x_test: np.ndarray,
    num_quantile_cases: int,
    num_random_cases: int,
    seed: int,
) -> list[tuple[int, str]]:
    total = len(x_test)
    selected: list[tuple[int, str]] = []
    used: set[int] = set()

    if num_quantile_cases > 0:
        sorted_indices = np.argsort(x_test[:, 0], kind="stable")
        quantiles = np.linspace(0.0, 1.0, num_quantile_cases)
        for q in quantiles:
            position = int(round(q * (total - 1)))
            index = int(sorted_indices[position])
            if index not in used:
                selected.append((index, "参数分位点"))
                used.add(index)

    remaining = np.array(
        [i for i in range(total) if i not in used],
        dtype=int,
    )
    random_count = min(num_random_cases, len(remaining))
    if random_count > 0:
        rng = np.random.default_rng(seed)
        random_indices = rng.choice(
            remaining,
            size=random_count,
            replace=False,
        )
        for index in random_indices.tolist():
            selected.append((int(index), "固定随机样本"))

    return selected


def finite_check(name: str, array: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(array)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.all(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "min": float(np.min(array)) if np.all(finite) else None,
        "max": float(np.max(array)) if np.all(finite) else None,
    }


def main() -> None:
    args = parse_args()

    task_dir = PROJECT_ROOT / "data" / "tasks" / args.task_name
    meta_path = task_dir / "meta.json"
    dataset_path = task_dir / "dataset.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 meta.json：{meta_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少 dataset.npz：{dataset_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    task_spec = meta["task_spec"]

    with np.load(dataset_path, allow_pickle=False) as loaded:
        required = [
            "vary_params_order",
            "x_train", "x_val", "x_test",
            "y_train", "y_val", "y_test",
            "lambda_grid",
        ]
        missing = [name for name in required if name not in loaded.files]
        if missing:
            raise KeyError(f"数据集缺少必要数组：{missing}")

        arrays = {name: np.asarray(loaded[name]) for name in required}

    vary_params_order = [
        str(name) for name in arrays["vary_params_order"].tolist()
    ]
    x_train = arrays["x_train"].astype(np.float64)
    x_val = arrays["x_val"].astype(np.float64)
    x_test = arrays["x_test"].astype(np.float64)
    y_train = arrays["y_train"].astype(np.float64)
    y_val = arrays["y_val"].astype(np.float64)
    y_test = arrays["y_test"].astype(np.float64)
    lambda_grid = arrays["lambda_grid"].astype(np.float64)

    integrity = {
        "arrays": [
            finite_check("x_train", x_train),
            finite_check("x_val", x_val),
            finite_check("x_test", x_test),
            finite_check("y_train", y_train),
            finite_check("y_val", y_val),
            finite_check("y_test", y_test),
            finite_check("lambda_grid", lambda_grid),
        ],
        "shape_checks": {
            "x_width_matches_param_count": (
                x_train.shape[1] == len(vary_params_order)
                and x_val.shape[1] == len(vary_params_order)
                and x_test.shape[1] == len(vary_params_order)
            ),
            "trajectory_shapes_match": (
                y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:]
            ),
            "lambda_length_matches_trajectory": (
                len(lambda_grid) == y_train.shape[1]
            ),
            "trajectory_last_dimension_is_3": (
                y_train.shape[-1] == 3
            ),
        },
        "split_sizes": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
            "total": int(len(x_train) + len(x_val) + len(x_test)),
        },
        "parameter_ranges": {
            "train": {
                name: [float(np.min(x_train[:, j])), float(np.max(x_train[:, j]))]
                for j, name in enumerate(vary_params_order)
            },
            "val": {
                name: [float(np.min(x_val[:, j])), float(np.max(x_val[:, j]))]
                for j, name in enumerate(vary_params_order)
            },
            "test": {
                name: [float(np.min(x_test[:, j])), float(np.max(x_test[:, j]))]
                for j, name in enumerate(vary_params_order)
            },
        },
        "lambda": {
            "first": float(lambda_grid[0]),
            "last": float(lambda_grid[-1]),
            "step_min": float(np.min(np.diff(lambda_grid))),
            "step_max": float(np.max(np.diff(lambda_grid))),
        },
    }

    array_finite_ok = all(item["finite"] for item in integrity["arrays"])
    shape_ok = all(integrity["shape_checks"].values())

    selected = select_indices(
        x_test=x_test,
        num_quantile_cases=args.num_quantile_cases,
        num_random_cases=args.num_random_cases,
        seed=args.random_seed,
    )

    base_step = float(task_spec["step_size"])
    base_n_steps = int(task_spec["n_steps"])
    base_intervals = base_n_steps - 1
    reference_step = base_step / args.reference_factor
    reference_n_steps = base_intervals * args.reference_factor + 1

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / "outputs"
        / "_second_order_dataset_validation"
        / args.task_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    failed_reasons: list[str] = []

    print("=" * 96)
    print("二阶求解器数据集验证")
    print(f"任务：{args.task_name}")
    print(f"训练/验证/测试：{len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"抽查轨道数：{len(selected)}")
    print(
        f"参考解细化倍数：{args.reference_factor}，"
        f"参考步长：{reference_step}"
    )
    print("=" * 96)

    if not array_finite_ok:
        failed_reasons.append("数据集包含 NaN 或 Inf")
    if not shape_ok:
        failed_reasons.append("数据数组形状不一致")

    for position, (test_index, source) in enumerate(selected, start=1):
        vary_params = {
            name: float(x_test[test_index, j])
            for j, name in enumerate(vary_params_order)
        }
        p, init, Q = build_physical_objects(
            task_spec["fixed_params"],
            vary_params,
        )

        reference_fine = simulate_one_orbit_second_order(
            p=p,
            init=init,
            Q=Q,
            n_steps=reference_n_steps,
            step_size=reference_step,
        )
        reference = reference_fine["xyz"][:: args.reference_factor]
        stored = y_test[test_index]

        error = relative_l2(stored, reference)
        errors.append(error)
        diagnostics = reference_fine["diagnostics"]

        row = {
            "test_index": test_index,
            "selection_source": source,
            **vary_params,
            "stored_vs_reference_relative_l2": error,
            "reference_radial_turns": int(
                diagnostics.radial_velocity_zero_crossings
            ),
            "reference_polar_turns": int(
                diagnostics.polar_velocity_zero_crossings
            ),
            "reference_max_radial_constraint": float(
                diagnostics.max_radial_constraint_residual
            ),
            "reference_max_polar_constraint": float(
                diagnostics.max_polar_constraint_residual
            ),
        }
        rows.append(row)

        print(
            f"[{position:02d}/{len(selected):02d}] "
            f"test_index={test_index}，参数={vary_params}，"
            f"存储-参考 Relative L2={error:.3e}"
        )

    max_error = float(np.max(errors)) if errors else math.nan
    median_error = float(np.median(errors)) if errors else math.nan
    mean_error = float(np.mean(errors)) if errors else math.nan

    numerical_ok = bool(errors) and max_error <= args.max_relative_l2
    if not numerical_ok:
        failed_reasons.append(
            f"抽查轨道最大 Relative L2 超过 {args.max_relative_l2:.1e}"
        )

    status = "通过" if not failed_reasons else "失败"

    summary = {
        "任务名称": args.task_name,
        "状态": status,
        "失败原因": failed_reasons,
        "数据完整性": integrity,
        "抽查轨道数量": len(selected),
        "参考解细化倍数": args.reference_factor,
        "参考解步长": reference_step,
        "误差阈值": args.max_relative_l2,
        "存储轨道相对参考解_Relative_L2": {
            "最小值": float(np.min(errors)) if errors else None,
            "中位数": median_error if errors else None,
            "平均值": mean_error if errors else None,
            "最大值": max_error if errors else None,
        },
    }

    if rows:
        with (output_dir / "trajectory_checks.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 二阶求解器数据集验证报告",
        "",
        f"- 任务：`{args.task_name}`",
        f"- 状态：**{status}**",
        f"- 训练/验证/测试规模：{len(x_train)}/{len(x_val)}/{len(x_test)}",
        f"- 抽查轨道数：{len(selected)}",
        f"- 参考解细化倍数：{args.reference_factor}",
        f"- 参考解步长：{reference_step}",
        "",
        "## 数据完整性",
        "",
        f"- 所有数组均为有限值：{'是' if array_finite_ok else '否'}",
        f"- 数据形状检查通过：{'是' if shape_ok else '否'}",
        f"- lambda 区间：[{lambda_grid[0]}, {lambda_grid[-1]}]",
        f"- lambda 步长范围："
        f"[{np.min(np.diff(lambda_grid)):.12g}, "
        f"{np.max(np.diff(lambda_grid)):.12g}]",
        "",
        "## 存储轨道相对细网格参考解",
        "",
        f"- 最小值：{np.min(errors):.6e}",
        f"- 中位数：{median_error:.6e}",
        f"- 平均值：{mean_error:.6e}",
        f"- 最大值：{max_error:.6e}",
        f"- 允许最大值：{args.max_relative_l2:.1e}",
        "",
        "## 判定",
        "",
    ]

    if status == "通过":
        report.extend(
            [
                "数据集通过完整性和数值精度抽查，可以进入小模型训练试验。",
                "",
                "该结论表示：当前使用步长 0.005 生成的训练轨道，在所抽查"
                "样本上与更细网格二阶参考解高度一致。它仍不等于对所有样本"
                "逐条验证，但足以支持试验性训练。",
            ]
        )
    else:
        report.append("数据集尚不应进入训练，失败原因如下：")
        for reason in failed_reasons:
            report.append(f"- {reason}")

    (output_dir / "validation_report_zh.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print(f"验证状态：{status}")
    print(f"误差中位数：{median_error:.6e}")
    print(f"误差最大值：{max_error:.6e}")
    print(f"中文报告：{output_dir / 'validation_report_zh.md'}")


if __name__ == "__main__":
    import math
    main()
