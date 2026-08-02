# ==========================================================
# File: scripts/collect_queue_a_results_2d.py
#
# 功能：
# 1. 汇总 Queue A 的 width=48 数据量曲线；
# 2. 汇总 Queue A 的 n=2000 宽度曲线；
# 3. 合并已有的 n2000/w64/e500 正式结果；
# 4. 统一使用公共测试集 physical-space 指标；
# 5. 保存 CSV 和 JSON。
#
# 终端及输出文件使用英文，代码注释使用中文。
# ==========================================================

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

RESULTS_DIR = (
    OUTPUTS_ROOT
    / "comparison"
    / "queue_a_cross_scale_summary"
)


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def nested_get(
    data: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    """安全读取嵌套字段。"""
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def model_name_for_width(width: int) -> str:
    """构造本次实验的模型名。"""
    return f"fno2d_m16x32_w{width}_d4_e500"


def get_root_summary(
    task_name: str,
    model_name: str,
) -> dict[str, Any]:
    """读取模型根级 summary.json。"""
    path = (
        OUTPUTS_ROOT
        / task_name
        / model_name
        / "summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model summary: {path}"
        )

    return load_json(path)


def get_common_test_result_path(
    task_name: str,
    model_name: str,
) -> Path:
    """
    定位公共测试结果。

    Queue A 新结果使用：
        outputs/comparison/
        queue_a__<task>__<model>/models/<task>/result.json

    已有 n2000/w64/e500 使用：
        common_test_e500_w64_n2000_n5000
    """
    queue_a_path = (
        OUTPUTS_ROOT
        / "comparison"
        / f"queue_a__{task_name}__{model_name}"
        / "models"
        / task_name
        / "result.json"
    )

    if queue_a_path.exists():
        return queue_a_path

    legacy_common_path = (
        OUTPUTS_ROOT
        / "comparison"
        / "common_test_e500_w64_n2000_n5000"
        / "models"
        / task_name
        / "result.json"
    )

    if legacy_common_path.exists():
        return legacy_common_path

    # 在 comparison 下做一次保守搜索，避免目录名称变化。
    candidates = list(
        (
            OUTPUTS_ROOT
            / "comparison"
        ).glob(
            f"*/models/{task_name}/result.json"
        )
    )

    matching_candidates = []

    for candidate in candidates:
        result = load_json(candidate)

        if result.get("model_name") == model_name:
            matching_candidates.append(candidate)

    if len(matching_candidates) == 1:
        return matching_candidates[0]

    raise FileNotFoundError(
        "Could not uniquely locate common-test result for "
        f"task={task_name}, model={model_name}. "
        f"Candidates={matching_candidates}"
    )


def collect_one_run(
    task_name: str,
    n_samples: int,
    width: int,
) -> dict[str, Any]:
    """汇总一个正式运行。"""
    model_name = model_name_for_width(width)

    summary = get_root_summary(
        task_name=task_name,
        model_name=model_name,
    )

    common_result_path = get_common_test_result_path(
        task_name=task_name,
        model_name=model_name,
    )

    common_result = load_json(
        common_result_path
    )

    common_metrics = common_result["metrics"]

    physical = nested_get(
        summary,
        ["metrics", "physical_space"],
        {},
    )

    row = {
        "task_name": task_name,
        "model_name": model_name,
        "n_samples": int(n_samples),
        "width": int(width),
        "modes_param": 16,
        "modes_lambda": 32,
        "depth": 4,
        "epochs": 500,
        "num_parameters": nested_get(
            summary,
            ["model", "num_parameters"],
        ),
        "best_epoch": nested_get(
            summary,
            ["training", "best_epoch"],
        ),
        "train_total_seconds": nested_get(
            summary,
            ["timing", "train_total_seconds"],
        ),
        "original_test_mse": (
            physical.get("test_mse")
            if isinstance(physical, dict)
            else None
        ),
        "original_test_relative_l2": (
            physical.get("test_relative_l2")
            if isinstance(physical, dict)
            else None
        ),
        "common_test_mse": float(
            common_metrics["mse"]
        ),
        "common_test_relative_l2": float(
            common_metrics["relative_l2"]
        ),
        "common_test_relative_l2_median": float(
            common_metrics[
                "relative_l2_median_over_q"
            ]
        ),
        "common_test_relative_l2_p95": float(
            common_metrics[
                "relative_l2_p95_over_q"
            ]
        ),
        "common_test_relative_l2_max": float(
            common_metrics[
                "relative_l2_max_over_q"
            ]
        ),
        "summary_path": str(
            (
                OUTPUTS_ROOT
                / task_name
                / model_name
                / "summary.json"
            ).relative_to(PROJECT_ROOT)
        ),
        "common_result_path": str(
            common_result_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    return row


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """保存 CSV。"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def print_table(
    title: str,
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
) -> None:
    """打印对比表。"""
    print("=" * 116)
    print(title)
    print("=" * 116)

    print(
        f"{x_label:>8s} "
        f"{'Params':>12s} "
        f"{'BestEp':>8s} "
        f"{'Common MSE':>14s} "
        f"{'Common RelL2':>15s} "
        f"{'Median':>13s} "
        f"{'P95':>13s} "
        f"{'Train(s)':>11s}"
    )

    print("-" * 116)

    for row in rows:
        params = row["num_parameters"]
        best_epoch = row["best_epoch"]
        train_seconds = row["train_total_seconds"]

        params_text = (
            str(params)
            if params is not None
            else "N/A"
        )

        best_epoch_text = (
            str(best_epoch)
            if best_epoch is not None
            else "N/A"
        )

        train_text = (
            f"{float(train_seconds):.3f}"
            if train_seconds is not None
            else "N/A"
        )

        print(
            f"{int(row[x_key]):8d} "
            f"{params_text:>12s} "
            f"{best_epoch_text:>8s} "
            f"{row['common_test_mse']:14.6e} "
            f"{row['common_test_relative_l2']:15.6e} "
            f"{row['common_test_relative_l2_median']:13.6e} "
            f"{row['common_test_relative_l2_p95']:13.6e} "
            f"{train_text:>11s}"
        )

    print("=" * 116)


def print_relative_changes(
    title: str,
    rows: list[dict[str, Any]],
    x_key: str,
) -> None:
    """打印相邻配置的误差变化。"""
    print(title)

    for previous, current in zip(
        rows,
        rows[1:],
    ):
        old_error = previous[
            "common_test_relative_l2"
        ]

        new_error = current[
            "common_test_relative_l2"
        ]

        change_percent = (
            (new_error - old_error)
            / old_error
            * 100.0
        )

        print(
            f"{x_key}={previous[x_key]} -> "
            f"{x_key}={current[x_key]}: "
            f"{change_percent:+.2f}%"
        )


def main() -> None:
    """主流程。"""
    data_scale_specs = [
        ("q_1p6-3_n500_t1200", 500, 48),
        ("q_1p6-3_n1000_t1200", 1000, 48),
        ("q_1p6-3_n2000_t1200", 2000, 48),
        ("q_1p6-3_n5000_t1200", 5000, 48),
    ]

    width_scale_specs = [
        ("q_1p6-3_n2000_t1200", 2000, 16),
        ("q_1p6-3_n2000_t1200", 2000, 32),
        ("q_1p6-3_n2000_t1200", 2000, 48),
        ("q_1p6-3_n2000_t1200", 2000, 64),
        ("q_1p6-3_n2000_t1200", 2000, 80),
    ]

    data_scale_rows = [
        collect_one_run(*spec)
        for spec in data_scale_specs
    ]

    width_scale_rows = [
        collect_one_run(*spec)
        for spec in width_scale_specs
    ]

    data_scale_rows.sort(
        key=lambda row: row["n_samples"]
    )

    width_scale_rows.sort(
        key=lambda row: row["width"]
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        data_scale_rows,
        RESULTS_DIR
        / "width48_data_scale_results.csv",
    )

    write_csv(
        width_scale_rows,
        RESULTS_DIR
        / "n2000_width_scale_results.csv",
    )

    combined = {
        "schema_version": "1.0",
        "official_metric_space": (
            "common_test_physical_space"
        ),
        "fixed_settings": {
            "modes_param": 16,
            "modes_lambda": 32,
            "depth": 4,
            "epochs": 500,
            "training_seed": 27,
        },
        "width48_data_scale": data_scale_rows,
        "n2000_width_scale": width_scale_rows,
    }

    with (
        RESULTS_DIR
        / "queue_a_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_table(
        title=(
            "Queue A Data-Scale Curve: "
            "fixed width=48, modes=16x32, depth=4, epochs=500"
        ),
        rows=data_scale_rows,
        x_key="n_samples",
        x_label="N",
    )

    print()

    print_relative_changes(
        title="Data-Scale Relative Changes",
        rows=data_scale_rows,
        x_key="n_samples",
    )

    print()

    print_table(
        title=(
            "Queue A Width-Scale Curve: "
            "fixed n=2000, modes=16x32, depth=4, epochs=500"
        ),
        rows=width_scale_rows,
        x_key="width",
        x_label="Width",
    )

    print()

    print_relative_changes(
        title="Width-Scale Relative Changes",
        rows=width_scale_rows,
        x_key="width",
    )

    print()
    print(
        f"Saved results directory: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
