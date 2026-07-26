#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_error_profiles_2d.py
======================================================================
比较 n500 与 n2000 等多个误差剖析结果。

输入是 analyze_fno2d_error_and_3d.py 生成的 error_summary.json。
输出一个中文比较报告和 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        nargs=2,
        action="append",
        metavar=("LABEL", "SUMMARY_JSON"),
        required=True,
        help="可重复提供，例如 --profile n500 path/error_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, path_text in args.profile:
        path = Path(path_text)
        data = json.loads(path.read_text(encoding="utf-8"))

        rel = data["trajectory_relative_l2"]
        max_dist = data["max_pointwise_distance"]
        final_dist = data["final_point_distance"]
        growth = data["后半段误差与前半段误差比值"]

        rows.append(
            {
                "实验": label,
                "测试轨道数": data["测试轨道数"],
                "总体Relative_L2": data["inference_metrics"]["relative_l2"],
                "逐轨道误差中位数": rel["中位数"],
                "逐轨道误差第95百分位": rel["第95百分位"],
                "逐轨道误差最大值": rel["最大值"],
                "最大点距离中位数": max_dist["中位数"],
                "最大点距离第95百分位": max_dist["第95百分位"],
                "终点距离中位数": final_dist["中位数"],
                "后半段前半段误差比中位数": growth["中位数"],
                "Q与误差相关系数": data["Q与Relative_L2相关系数"],
            }
        )

    csv_path = args.output_dir / "error_profile_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# n500 与 n2000 误差剖析比较",
        "",
        "| 实验 | 总体 Relative L2 | 逐轨道中位数 | 95%分位 | 最大值 | "
        "最大点距离中位数 | 终点距离中位数 | 后半段/前半段误差比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        report.append(
            f"| {row['实验']} "
            f"| {row['总体Relative_L2']:.6e} "
            f"| {row['逐轨道误差中位数']:.6e} "
            f"| {row['逐轨道误差第95百分位']:.6e} "
            f"| {row['逐轨道误差最大值']:.6e} "
            f"| {row['最大点距离中位数']:.6e} "
            f"| {row['终点距离中位数']:.6e} "
            f"| {row['后半段前半段误差比中位数']:.4f} |"
        )

    report.extend(
        [
            "",
            "## 判断原则",
            "",
            "- 不能只比较总体 Relative L2，还要检查 95% 分位和最大值，"
            "判断是否存在少量严重失败轨道。",
            "- 后半段/前半段误差比明显大于 1 时，应检查长时间相位漂移。",
            "- 最大点距离和终点距离决定模型是否适合精确轨道定位。",
            "",
        ]
    )

    (args.output_dir / "error_profile_comparison_zh.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"比较报告：{args.output_dir / 'error_profile_comparison_zh.md'}")
    print(f"比较表格：{csv_path}")


if __name__ == "__main__":
    main()
