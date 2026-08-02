# ==========================================================
# File: scripts/collect_scale_results_2d.py
#
# 功能：
# 1. 汇总二维 FNO 数据量规模实验；
# 2. 汇总二维 FNO 模型宽度实验；
# 3. 从统一 summary.json 提取正式物理空间指标；
# 4. 保存 CSV 与 JSON，便于后续绘图和汇报。
#
# 说明：
# - 终端和保存文件中的文本使用英文；
# - 代码注释使用中文；
# - 正式比较统一使用 physical_space 指标。
# ==========================================================

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_ROOT / "comparison" / "scale_experiments_2d"

TASK_PATTERN = re.compile(
    r"^q_1p6-3_n(?P<n>\d+)_t(?P<t>\d+)$"
)

MODEL_PATTERN = re.compile(
    r"^fno2d_m(?P<modes_param>\d+)x(?P<modes_lambda>\d+)"
    r"_w(?P<width>\d+)_d(?P<depth>\d+)_e(?P<epochs>\d+)"
    r"(?:_.+)?$"
)


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_nested_get(
    data: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    """安全读取嵌套字典字段。"""
    current: Any = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def collect_runs() -> list[dict[str, Any]]:
    """扫描 outputs 目录并收集成功运行。"""
    rows: list[dict[str, Any]] = []

    if not OUTPUTS_ROOT.exists():
        raise FileNotFoundError(
            f"Outputs directory does not exist: {OUTPUTS_ROOT}"
        )

    for task_dir in sorted(OUTPUTS_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue

        task_match = TASK_PATTERN.match(task_dir.name)
        if task_match is None:
            continue

        n_samples = int(task_match.group("n"))
        n_steps = int(task_match.group("t"))

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_match = MODEL_PATTERN.match(model_dir.name)
            if model_match is None:
                continue

            summary_path = model_dir / "summary.json"
            if not summary_path.exists():
                continue

            summary = load_json(summary_path)

            physical = safe_nested_get(
                summary,
                ["metrics", "physical_space"],
            )
            model_space = safe_nested_get(
                summary,
                ["metrics", "model_space"],
            )

            # 只收集完成训练且完成正式评价的运行。
            if not isinstance(physical, dict):
                continue

            if not isinstance(model_space, dict):
                continue

            row = {
                "task_name": task_dir.name,
                "model_name": model_dir.name,
                "n_samples": n_samples,
                "n_steps": n_steps,
                "modes_param": int(
                    model_match.group("modes_param")
                ),
                "modes_lambda": int(
                    model_match.group("modes_lambda")
                ),
                "width": int(model_match.group("width")),
                "depth": int(model_match.group("depth")),
                "epochs": int(model_match.group("epochs")),
                "num_parameters": safe_nested_get(
                    summary,
                    ["model", "num_parameters"],
                ),
                "best_epoch": safe_nested_get(
                    summary,
                    ["training", "best_epoch"],
                ),
                "best_val_mse_model_space": model_space.get(
                    "best_val_mse"
                ),
                "test_mse_model_space": model_space.get(
                    "test_mse"
                ),
                "test_relative_l2_model_space": model_space.get(
                    "test_relative_l2"
                ),
                "test_mse_physical_space": physical.get(
                    "test_mse"
                ),
                "test_relative_l2_physical_space": physical.get(
                    "test_relative_l2"
                ),
                "train_total_seconds": safe_nested_get(
                    summary,
                    ["timing", "train_total_seconds"],
                ),
                "model_inference_total_seconds": safe_nested_get(
                    summary,
                    ["timing", "inference", "model_total_seconds"],
                ),
                "model_inference_avg_seconds": safe_nested_get(
                    summary,
                    [
                        "timing",
                        "inference",
                        "model_avg_seconds_per_sample",
                    ],
                ),
                "second_order_total_seconds": safe_nested_get(
                    summary,
                    [
                        "timing",
                        "inference",
                        "second_order_standard_total_seconds",
                    ],
                ),
                "speedup_standard": safe_nested_get(
                    summary,
                    ["timing", "inference", "speedup_standard"],
                ),
                "summary_path": str(
                    summary_path.relative_to(PROJECT_ROOT)
                ),
            }

            rows.append(row)

    return rows


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


def write_json(
    data: Any,
    path: Path,
) -> None:
    """保存 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def format_number(
    value: Any,
    scientific: bool = False,
) -> str:
    """格式化终端数值。"""
    if value is None:
        return "N/A"

    if scientific:
        return f"{float(value):.6e}"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def print_data_scale_table(
    rows: list[dict[str, Any]],
) -> None:
    """打印数据量规模实验表。"""
    print("=" * 118)
    print("Data-Scale Experiment: fixed width=64, depth=4, modes=16x32, epochs=300")
    print("=" * 118)
    print(
        f"{'N':>7s} "
        f"{'Params':>12s} "
        f"{'BestEp':>7s} "
        f"{'Phys MSE':>13s} "
        f"{'Phys RelL2':>13s} "
        f"{'Model RelL2':>13s} "
        f"{'Train(s)':>10s} "
        f"{'Speedup':>10s}"
    )
    print("-" * 118)

    for row in rows:
        print(
            f"{row['n_samples']:7d} "
            f"{format_number(row['num_parameters']):>12s} "
            f"{format_number(row['best_epoch']):>7s} "
            f"{format_number(row['test_mse_physical_space'], True):>13s} "
            f"{format_number(row['test_relative_l2_physical_space'], True):>13s} "
            f"{format_number(row['test_relative_l2_model_space'], True):>13s} "
            f"{format_number(row['train_total_seconds']):>10s} "
            f"{format_number(row['speedup_standard']):>10s}"
        )

    print("=" * 118)


def print_width_scale_table(
    rows: list[dict[str, Any]],
) -> None:
    """打印模型宽度规模实验表。"""
    print("=" * 118)
    print("Width-Scale Experiment: fixed n=5000, depth=4, modes=16x32, epochs=300")
    print("=" * 118)
    print(
        f"{'Width':>7s} "
        f"{'Params':>12s} "
        f"{'BestEp':>7s} "
        f"{'Phys MSE':>13s} "
        f"{'Phys RelL2':>13s} "
        f"{'Model RelL2':>13s} "
        f"{'Train(s)':>10s} "
        f"{'Speedup':>10s}"
    )
    print("-" * 118)

    for row in rows:
        print(
            f"{row['width']:7d} "
            f"{format_number(row['num_parameters']):>12s} "
            f"{format_number(row['best_epoch']):>7s} "
            f"{format_number(row['test_mse_physical_space'], True):>13s} "
            f"{format_number(row['test_relative_l2_physical_space'], True):>13s} "
            f"{format_number(row['test_relative_l2_model_space'], True):>13s} "
            f"{format_number(row['train_total_seconds']):>10s} "
            f"{format_number(row['speedup_standard']):>10s}"
        )

    print("=" * 118)


def main() -> None:
    """主入口。"""
    all_rows = collect_runs()

    data_scale_rows = sorted(
        [
            row
            for row in all_rows
            if (
                row["n_steps"] == 1200
                and row["width"] == 64
                and row["depth"] == 4
                and row["epochs"] == 300
                and row["modes_param"] == 16
                and row["modes_lambda"] == 32
            )
        ],
        key=lambda row: row["n_samples"],
    )

    width_scale_rows = sorted(
        [
            row
            for row in all_rows
            if (
                row["n_samples"] == 5000
                and row["n_steps"] == 1200
                and row["depth"] == 4
                and row["epochs"] == 300
                and row["modes_param"] == 16
                and row["modes_lambda"] == 32
            )
        ],
        key=lambda row: row["width"],
    )

    if not data_scale_rows:
        raise RuntimeError(
            "No completed data-scale experiments were found."
        )

    if not width_scale_rows:
        raise RuntimeError(
            "No completed width-scale experiments were found."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        data_scale_rows,
        RESULTS_DIR / "data_scale_results.csv",
    )
    write_csv(
        width_scale_rows,
        RESULTS_DIR / "width_scale_results.csv",
    )

    combined = {
        "schema_version": "1.0",
        "official_metric_space": "physical_space",
        "data_scale": data_scale_rows,
        "width_scale": width_scale_rows,
    }

    write_json(
        combined,
        RESULTS_DIR / "scale_results_summary.json",
    )

    print_data_scale_table(data_scale_rows)
    print()
    print_width_scale_table(width_scale_rows)
    print()
    print(f"Saved results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
