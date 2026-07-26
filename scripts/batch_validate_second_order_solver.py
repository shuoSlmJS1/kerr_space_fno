#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_validate_second_order_solver.py
======================================================================
批量验证二阶 Kerr 轨道求解器。

研究问题
--------
三条代表性轨道已经表现出稳定四阶收敛，但这还不足以说明整个测试参数
范围都可靠。本脚本扩大样本覆盖，并对每条轨道统一检查：

1. 不同步长下的径向、极向转向次数是否稳定；
2. 第一积分约束残差是否随步长减小；
3. 相邻网格轨道误差是否单调下降；
4. 观测收敛阶是否接近 RK4 的四阶；
5. 最细两层轨道差异是否足够小；
6. 是否出现 NaN、Inf、视界越界或其他运行错误。

抽样策略
--------
默认将样本分成两部分：

- 参数分位点：均匀覆盖参数范围的低端、中间和高端；
- 固定随机样本：补充非规则位置，避免只验证整齐的分位点。

随机种子会写入结果文件，因此实验可以复现。

输出
----
outputs/_second_order_batch_validation/<任务名>/
    selected_cases.csv
    case_results.csv
    validation_summary.json
    validation_report_zh.md
    failed_cases.json

说明
----
这是“候选求解器的广泛数值验证”，还不是最终物理正确性证明。
通过本实验后，仍建议与独立高精度求解器交叉验证少量轨道。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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


@dataclass
class SelectedCase:
    case_id: int
    test_index: int
    selection_source: str
    vary_params: dict[str, float]


@dataclass
class ValidationThresholds:
    """
    批量判定阈值。

    min_observed_order:
        允许的最低观测收敛阶。默认 3.5，给理想四阶留出一定数值波动。

    max_finest_relative_l2:
        最细两层轨道之间允许的最大 Relative L2。

    max_finest_radial_constraint / max_finest_polar_constraint:
        最细网格上第一积分约束最大残差阈值。

    require_monotone_errors:
        是否要求相邻网格误差严格递减。
    """

    min_observed_order: float = 3.5
    max_finest_relative_l2: float = 1e-8
    max_finest_radial_constraint: float = 1e-8
    max_finest_polar_constraint: float = 1e-9
    require_monotone_errors: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量验证二阶 Kerr 轨道求解器。"
    )
    parser.add_argument(
        "--task-name",
        default="vary_Q__Q1.6_3__n5000__T1200__cfg1",
    )
    parser.add_argument(
        "--num-quantile-cases",
        type=int,
        default=20,
        help="按第一变化参数分位点选取的轨道数量。",
    )
    parser.add_argument(
        "--num-random-cases",
        type=int,
        default=20,
        help="额外随机抽取的测试轨道数量。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260722,
    )
    parser.add_argument(
        "--refinement-factors",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--min-observed-order",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--max-finest-relative-l2",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max-finest-radial-constraint",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max-finest-polar-constraint",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--allow-nonmonotone-errors",
        action="store_true",
        help="不把相邻网格误差非单调作为失败条件。",
    )
    return parser.parse_args()


def load_task(task_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    task_dir = PROJECT_ROOT / "data" / "tasks" / task_name
    meta_path = task_dir / "meta.json"
    dataset_path = task_dir / "dataset.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"缺少任务元数据：{meta_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少任务数据：{dataset_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    return meta, data


def build_selected_case(
    case_id: int,
    test_index: int,
    selection_source: str,
    x_test: np.ndarray,
    vary_params_order: list[str],
) -> SelectedCase:
    vary_params = {
        name: float(x_test[test_index, column])
        for column, name in enumerate(vary_params_order)
    }
    return SelectedCase(
        case_id=case_id,
        test_index=int(test_index),
        selection_source=selection_source,
        vary_params=vary_params,
    )


def select_cases(
    x_test: np.ndarray,
    vary_params_order: list[str],
    num_quantile_cases: int,
    num_random_cases: int,
    random_seed: int,
) -> list[SelectedCase]:
    """
    同时选取分位点样本与固定随机样本，并去除重复索引。

    当前任务通常只有一个变化参数，因此按第一列排序。如果将来用于多参数
    任务，这仍可保证第一参数范围覆盖，但还应增加多维覆盖策略。
    """
    total = len(x_test)
    if total == 0:
        raise ValueError("测试集为空。")

    selected: list[tuple[int, str]] = []
    used: set[int] = set()

    if num_quantile_cases > 0:
        sorted_indices = np.argsort(x_test[:, 0])
        quantiles = np.linspace(0.0, 1.0, num_quantile_cases)

        for quantile in quantiles:
            position = int(round(quantile * (total - 1)))
            test_index = int(sorted_indices[position])
            if test_index not in used:
                selected.append((test_index, "参数分位点"))
                used.add(test_index)

    remaining_indices = np.array(
        [index for index in range(total) if index not in used],
        dtype=int,
    )
    random_count = min(num_random_cases, len(remaining_indices))

    if random_count > 0:
        rng = np.random.default_rng(random_seed)
        random_indices = rng.choice(
            remaining_indices,
            size=random_count,
            replace=False,
        )
        for test_index in random_indices.tolist():
            selected.append((int(test_index), "固定随机样本"))
            used.add(int(test_index))

    return [
        build_selected_case(
            case_id=case_id,
            test_index=test_index,
            selection_source=source,
            x_test=x_test,
            vary_params_order=vary_params_order,
        )
        for case_id, (test_index, source) in enumerate(selected)
    ]


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


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(b.reshape(-1))
    if denominator == 0.0:
        raise ValueError("参考轨道范数为零。")
    return float(
        np.linalg.norm((a - b).reshape(-1)) / denominator
    )


def observed_orders(errors: list[float], factors: list[int]) -> list[float]:
    orders: list[float] = []
    for index in range(len(errors) - 1):
        coarse_factor = factors[index]
        next_coarse_factor = factors[index + 1]
        ratio = next_coarse_factor / coarse_factor

        previous_error = errors[index]
        current_error = errors[index + 1]

        if previous_error <= 0.0 or current_error <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(
                float(
                    math.log(previous_error / current_error)
                    / math.log(ratio)
                )
            )
    return orders


def classify_case(
    factors: list[int],
    run_records: dict[int, dict[str, Any]],
    errors: list[float],
    orders: list[float],
    thresholds: ValidationThresholds,
) -> tuple[str, list[str]]:
    """
    按统一标准将案例分为“通过”或“失败”，并返回具体原因。

    这里使用较严格的二元判定，便于批量筛查。后续人工审查时可以再将
    部分边界案例标成“警告”。
    """
    reasons: list[str] = []

    radial_turns = {
        run_records[factor]["radial_turns"] for factor in factors
    }
    polar_turns = {
        run_records[factor]["polar_turns"] for factor in factors
    }

    if len(radial_turns) != 1:
        reasons.append("径向转向次数随步长变化")
    if len(polar_turns) != 1:
        reasons.append("极向转向次数随步长变化")

    if thresholds.require_monotone_errors:
        for previous, current in zip(errors[:-1], errors[1:]):
            if not current < previous:
                reasons.append("相邻网格误差没有严格递减")
                break

    finite_orders = [value for value in orders if math.isfinite(value)]
    if len(finite_orders) != len(orders):
        reasons.append("观测收敛阶包含非有限值")
    elif any(
        value < thresholds.min_observed_order
        for value in finite_orders
    ):
        reasons.append(
            f"观测收敛阶低于 {thresholds.min_observed_order}"
        )

    finest_error = errors[-1]
    if finest_error > thresholds.max_finest_relative_l2:
        reasons.append(
            "最细两层 Relative L2 超过阈值 "
            f"{thresholds.max_finest_relative_l2:.1e}"
        )

    finest = run_records[factors[-1]]
    if (
        finest["max_radial_constraint"]
        > thresholds.max_finest_radial_constraint
    ):
        reasons.append(
            "最细网格径向约束残差超过阈值 "
            f"{thresholds.max_finest_radial_constraint:.1e}"
        )
    if (
        finest["max_polar_constraint"]
        > thresholds.max_finest_polar_constraint
    ):
        reasons.append(
            "最细网格极向约束残差超过阈值 "
            f"{thresholds.max_finest_polar_constraint:.1e}"
        )

    return ("通过" if not reasons else "失败"), reasons


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    factors = sorted(set(int(value) for value in args.refinement_factors))
    if len(factors) < 3:
        raise ValueError("至少需要三个步长层级才能计算观测收敛阶。")
    if factors[0] != 1:
        raise ValueError("细化倍数必须包含 1。")
    if any(value <= 0 for value in factors):
        raise ValueError("细化倍数必须为正整数。")
    if any(
        finer % coarser != 0
        for coarser, finer in zip(factors[:-1], factors[1:])
    ):
        raise ValueError("相邻细化倍数必须为整数倍关系。")

    thresholds = ValidationThresholds(
        min_observed_order=float(args.min_observed_order),
        max_finest_relative_l2=float(
            args.max_finest_relative_l2
        ),
        max_finest_radial_constraint=float(
            args.max_finest_radial_constraint
        ),
        max_finest_polar_constraint=float(
            args.max_finest_polar_constraint
        ),
        require_monotone_errors=not args.allow_nonmonotone_errors,
    )

    meta, data = load_task(args.task_name)
    task_spec = meta["task_spec"]
    x_test = np.asarray(data["x_test"], dtype=np.float64)
    vary_params_order = [
        str(name) for name in data["vary_params_order"].tolist()
    ]

    cases = select_cases(
        x_test=x_test,
        vary_params_order=vary_params_order,
        num_quantile_cases=int(args.num_quantile_cases),
        num_random_cases=int(args.num_random_cases),
        random_seed=int(args.random_seed),
    )

    base_n_steps = int(task_spec["n_steps"])
    base_step = float(task_spec["step_size"])
    base_intervals = base_n_steps - 1
    lambda_max = base_intervals * base_step

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / "outputs"
        / "_second_order_batch_validation"
        / args.task_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []

    for case in cases:
        selected_rows.append(
            {
                "case_id": case.case_id,
                "test_index": case.test_index,
                "selection_source": case.selection_source,
                **case.vary_params,
            }
        )

    print("=" * 100)
    print("二阶 Kerr 轨道求解器批量验证")
    print(f"任务：{args.task_name}")
    print(f"样本数：{len(cases)}")
    print(f"细化倍数：{factors}")
    print(f"共同 lambda_max：{lambda_max}")
    print("=" * 100)

    total_started = time.perf_counter()

    for position, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        print(
            f"[{position:02d}/{len(cases):02d}] "
            f"test_index={case.test_index}, "
            f"来源={case.selection_source}, "
            f"参数={case.vary_params}"
        )

        try:
            p, init, Q = build_physical_objects(
                fixed_params=task_spec["fixed_params"],
                vary_params=case.vary_params,
            )

            solutions: dict[int, dict[str, Any]] = {}
            run_records: dict[int, dict[str, Any]] = {}

            for factor in factors:
                h = base_step / factor
                n_steps = base_intervals * factor + 1

                started = time.perf_counter()
                result = simulate_one_orbit_second_order(
                    p=p,
                    init=init,
                    Q=Q,
                    n_steps=n_steps,
                    step_size=h,
                )
                elapsed = time.perf_counter() - started
                diagnostics = result["diagnostics"]

                solutions[factor] = result
                run_records[factor] = {
                    "factor": factor,
                    "step_size": h,
                    "n_steps": n_steps,
                    "elapsed_seconds": float(elapsed),
                    "radial_turns": int(
                        diagnostics.radial_velocity_zero_crossings
                    ),
                    "polar_turns": int(
                        diagnostics.polar_velocity_zero_crossings
                    ),
                    "max_radial_constraint": float(
                        diagnostics.max_radial_constraint_residual
                    ),
                    "max_polar_constraint": float(
                        diagnostics.max_polar_constraint_residual
                    ),
                    "mean_radial_constraint": float(
                        diagnostics.mean_radial_constraint_residual
                    ),
                    "mean_polar_constraint": float(
                        diagnostics.mean_polar_constraint_residual
                    ),
                    "min_radial_potential": float(
                        diagnostics.min_radial_potential
                    ),
                    "min_polar_potential": float(
                        diagnostics.min_polar_potential
                    ),
                }

            errors: list[float] = []
            for coarse, fine in zip(factors[:-1], factors[1:]):
                ratio = fine // coarse
                coarse_xyz = solutions[coarse]["xyz"]
                fine_xyz = solutions[fine]["xyz"][::ratio]

                if coarse_xyz.shape != fine_xyz.shape:
                    raise RuntimeError(
                        "细网格抽样后与粗网格形状不一致。"
                    )
                errors.append(relative_l2(coarse_xyz, fine_xyz))

            orders = observed_orders(errors, factors)

            status, reasons = classify_case(
                factors=factors,
                run_records=run_records,
                errors=errors,
                orders=orders,
                thresholds=thresholds,
            )

            finest = run_records[factors[-1]]
            case_elapsed = time.perf_counter() - case_started

            row: dict[str, Any] = {
                "case_id": case.case_id,
                "test_index": case.test_index,
                "selection_source": case.selection_source,
                **case.vary_params,
                "status": status,
                "failure_reasons": "；".join(reasons),
                "radial_turns": finest["radial_turns"],
                "polar_turns": finest["polar_turns"],
                "finest_relative_l2": errors[-1],
                "minimum_observed_order": min(orders),
                "maximum_observed_order": max(orders),
                "finest_max_radial_constraint": (
                    finest["max_radial_constraint"]
                ),
                "finest_max_polar_constraint": (
                    finest["max_polar_constraint"]
                ),
                "finest_min_radial_potential": (
                    finest["min_radial_potential"]
                ),
                "finest_min_polar_potential": (
                    finest["min_polar_potential"]
                ),
                "case_elapsed_seconds": float(case_elapsed),
            }

            for index, error in enumerate(errors):
                row[
                    f"relative_l2_factor_{factors[index]}_to_"
                    f"{factors[index + 1]}"
                ] = error
            for index, order in enumerate(orders):
                row[
                    f"observed_order_level_{index + 1}"
                ] = order

            case_rows.append(row)

            case_result = {
                **asdict(case),
                "status": status,
                "failure_reasons": reasons,
                "runs": {
                    str(factor): run_records[factor]
                    for factor in factors
                },
                "adjacent_grid_relative_l2": errors,
                "observed_orders": orders,
            }
            full_results.append(case_result)

            if status == "失败":
                failed_cases.append(case_result)

            print(
                f"    {status}；最细误差={errors[-1]:.3e}；"
                f"收敛阶={','.join(f'{value:.3f}' for value in orders)}；"
                f"转向=({finest['radial_turns']},"
                f"{finest['polar_turns']})"
            )
            if reasons:
                print(f"    原因：{'；'.join(reasons)}")

        except Exception as exc:  # 批量任务必须保留失败案例并继续
            case_elapsed = time.perf_counter() - case_started
            reason = f"{type(exc).__name__}: {exc}"

            row = {
                "case_id": case.case_id,
                "test_index": case.test_index,
                "selection_source": case.selection_source,
                **case.vary_params,
                "status": "运行失败",
                "failure_reasons": reason,
                "case_elapsed_seconds": float(case_elapsed),
            }
            case_rows.append(row)

            failed_result = {
                **asdict(case),
                "status": "运行失败",
                "failure_reasons": [reason],
            }
            failed_cases.append(failed_result)
            full_results.append(failed_result)

            print(f"    运行失败：{reason}")

    total_elapsed = time.perf_counter() - total_started

    passed_count = sum(
        row["status"] == "通过" for row in case_rows
    )
    failed_count = len(case_rows) - passed_count

    finest_errors = [
        float(row["finest_relative_l2"])
        for row in case_rows
        if row.get("status") in {"通过", "失败"}
        and "finest_relative_l2" in row
    ]
    minimum_orders = [
        float(row["minimum_observed_order"])
        for row in case_rows
        if row.get("status") in {"通过", "失败"}
        and "minimum_observed_order" in row
    ]

    summary = {
        "任务名称": args.task_name,
        "测试集大小": int(len(x_test)),
        "验证样本数": len(cases),
        "分位点样本数": int(args.num_quantile_cases),
        "随机样本数": int(args.num_random_cases),
        "随机种子": int(args.random_seed),
        "细化倍数": factors,
        "基础步长": base_step,
        "共同_lambda_max": lambda_max,
        "判定阈值": asdict(thresholds),
        "通过数量": int(passed_count),
        "失败或运行失败数量": int(failed_count),
        "通过率": float(passed_count / len(cases)) if cases else 0.0,
        "最细两层误差统计": {
            "最小值": min(finest_errors) if finest_errors else None,
            "中位数": (
                float(np.median(finest_errors))
                if finest_errors else None
            ),
            "最大值": max(finest_errors) if finest_errors else None,
        },
        "每条轨道最低观测阶统计": {
            "最小值": min(minimum_orders) if minimum_orders else None,
            "中位数": (
                float(np.median(minimum_orders))
                if minimum_orders else None
            ),
            "最大值": max(minimum_orders) if minimum_orders else None,
        },
        "总运行时间秒": float(total_elapsed),
        "案例结果": full_results,
    }

    write_csv(output_dir / "selected_cases.csv", selected_rows)
    write_csv(output_dir / "case_results.csv", case_rows)

    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failed_cases.json").write_text(
        json.dumps(failed_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 二阶 Kerr 轨道求解器批量验证报告",
        "",
        f"- 任务：`{args.task_name}`",
        f"- 测试集大小：{len(x_test)}",
        f"- 实际验证样本数：{len(cases)}",
        f"- 细化倍数：{factors}",
        f"- 随机种子：{args.random_seed}",
        f"- 通过数量：{passed_count}",
        f"- 失败或运行失败数量：{failed_count}",
        f"- 通过率：{summary['通过率']:.2%}",
        f"- 总运行时间：{total_elapsed:.2f} 秒",
        "",
        "## 判定标准",
        "",
        f"- 每条轨道最低观测阶不低于 "
        f"{thresholds.min_observed_order}；",
        f"- 最细两层 Relative L2 不超过 "
        f"{thresholds.max_finest_relative_l2:.1e}；",
        f"- 最细网格径向约束残差不超过 "
        f"{thresholds.max_finest_radial_constraint:.1e}；",
        f"- 最细网格极向约束残差不超过 "
        f"{thresholds.max_finest_polar_constraint:.1e}；",
        "- 转向次数在各细化层级保持一致；",
        "- 相邻网格误差严格递减。"
        if thresholds.require_monotone_errors
        else "- 本次不强制相邻网格误差严格递减。",
        "",
        "## 汇总统计",
        "",
    ]

    if finest_errors:
        report_lines.extend(
            [
                "- 最细两层误差：",
                f"  - 最小值：{min(finest_errors):.3e}",
                f"  - 中位数：{np.median(finest_errors):.3e}",
                f"  - 最大值：{max(finest_errors):.3e}",
            ]
        )
    if minimum_orders:
        report_lines.extend(
            [
                "- 每条轨道最低观测阶：",
                f"  - 最小值：{min(minimum_orders):.4f}",
                f"  - 中位数：{np.median(minimum_orders):.4f}",
                f"  - 最大值：{max(minimum_orders):.4f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## 失败案例",
            "",
        ]
    )

    if not failed_cases:
        report_lines.append("本次抽样验证没有发现失败案例。")
    else:
        for item in failed_cases:
            params = "，".join(
                f"{key}={value:.10g}"
                for key, value in item["vary_params"].items()
            )
            reasons = "；".join(item["failure_reasons"])
            report_lines.append(
                f"- test_index={item['test_index']}，"
                f"{params}：{reasons}"
            )

    report_lines.extend(
        [
            "",
            "## 科研解释",
            "",
            "本报告用于判断二阶求解器是否在所抽取的测试参数范围内表现出"
            "一致的数值收敛。即使全部通过，也只能说明当前离散实现具有良好"
            "的数值稳定性和四阶收敛特征；它不能单独替代与独立数值方法或"
            "已知解析性质的交叉验证。",
            "",
            "## 下一步",
            "",
            "1. 人工检查所有失败或边界案例；",
            "2. 若通过率足够高，使用独立高精度求解器交叉验证少量轨道；",
            "3. 生成旧求解器、二阶参考解与现有 FNO 预测的三方比较；",
            "4. 在确认新求解器可靠后，再决定是否重新生成训练数据。",
            "",
        ]
    )

    (output_dir / "validation_report_zh.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print("批量验证完成")
    print(f"通过：{passed_count}/{len(cases)}")
    print(f"输出目录：{output_dir}")
    print(f"中文报告：{output_dir / 'validation_report_zh.md'}")
    print(f"失败案例：{output_dir / 'failed_cases.json'}")


if __name__ == "__main__":
    main()
