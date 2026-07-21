#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize experiments under outputs/.

Canonical sources:
- final accuracy: inference/metrics.json
- inference timing: inference/timing.json
- training metadata: logs/train_summary.json

Outputs:
- outputs/_summary/experiment_inventory.csv
- outputs/_summary/experiment_inventory.json
- outputs/_summary/experiment_inventory.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def discover_model_dirs(outputs_root: Path) -> Iterable[Path]:
    discovered = set()
    for path in outputs_root.rglob("inference/metrics.json"):
        discovered.add(path.parent.parent)
    for path in outputs_root.rglob("inference/timing.json"):
        discovered.add(path.parent.parent)
    for path in outputs_root.rglob("logs/train_summary.json"):
        discovered.add(path.parent.parent)
    return sorted(discovered, key=str)


def infer_model_type(model_name: str, summary: Dict[str, Any]) -> Optional[str]:
    trainer = summary.get("trainer_config", {})
    trainer = trainer if isinstance(trainer, dict) else {}
    model_cfg = trainer.get("model_config", {})
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}

    explicit = first_not_none(
        summary.get("model_type"),
        trainer.get("model_type"),
        model_cfg.get("model_type"),
    )
    if explicit is not None:
        return str(explicit)

    name = model_name.lower()
    if name.startswith("fno2d"):
        return "fno2d"
    if name.startswith("fno"):
        return "fno1d"
    if name.startswith("cnn"):
        return "cnn1d"
    return None


def build_record(model_dir: Path) -> Dict[str, Any]:
    task_name = model_dir.parent.name
    model_name = model_dir.name

    metrics_path = model_dir / "inference" / "metrics.json"
    timing_path = model_dir / "inference" / "timing.json"
    summary_path = model_dir / "logs" / "train_summary.json"

    metrics = read_json(metrics_path)
    timing = read_json(timing_path)
    summary = read_json(summary_path)

    metrics_data = metrics or {}
    timing_data = timing or {}
    summary_data = summary or {}

    trainer = summary_data.get("trainer_config", {})
    trainer = trainer if isinstance(trainer, dict) else {}
    model_cfg = trainer.get("model_config", {})
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    legacy_cfg = summary_data.get("config", {})
    legacy_cfg = legacy_cfg if isinstance(legacy_cfg, dict) else {}

    return {
        "task_name": task_name,
        "model_name": model_name,
        "model_type": infer_model_type(model_name, summary_data),
        "relative_l2": metrics_data.get("relative_l2"),
        "mse": metrics_data.get("mse"),
        "speedup_total": timing_data.get("speedup_total"),
        "model_total_seconds": timing_data.get("model_total_seconds"),
        "traditional_total_seconds": timing_data.get("traditional_total_seconds"),
        "model_avg_seconds_per_sample": timing_data.get(
            "model_avg_seconds_per_sample"
        ),
        "traditional_avg_seconds_per_sample": timing_data.get(
            "traditional_avg_seconds_per_sample"
        ),
        "num_samples": timing_data.get("num_samples"),
        "num_fields": timing_data.get("num_fields"),
        "epochs": first_not_none(trainer.get("epochs"), legacy_cfg.get("epochs")),
        "learning_rate": first_not_none(trainer.get("lr"), legacy_cfg.get("lr")),
        "batch_size": first_not_none(
            trainer.get("batch_size"), legacy_cfg.get("batch_size")
        ),
        "normalization": first_not_none(
            summary_data.get("normalization"),
            trainer.get("normalization"),
            legacy_cfg.get("normalization"),
        ),
        "target_transform": first_not_none(
            summary_data.get("target_transform"),
            trainer.get("target_transform"),
            legacy_cfg.get("target_transform"),
        ),
        "train_total_seconds": first_not_none(
            summary_data.get("train_total_seconds"),
            summary_data.get("train_seconds"),
        ),
        "best_epoch": summary_data.get("best_epoch"),
        "best_val_mse": summary_data.get("best_val_mse"),
        "modes": first_not_none(model_cfg.get("modes"), legacy_cfg.get("modes")),
        "modes1": first_not_none(
            model_cfg.get("modes1"),
            legacy_cfg.get("modes1"),
            legacy_cfg.get("modes_param"),
        ),
        "modes2": first_not_none(
            model_cfg.get("modes2"),
            legacy_cfg.get("modes2"),
            legacy_cfg.get("modes_lambda"),
        ),
        "width": first_not_none(model_cfg.get("width"), legacy_cfg.get("width")),
        "depth": first_not_none(model_cfg.get("depth"), legacy_cfg.get("depth")),
        "num_parameters": first_not_none(
            trainer.get("num_parameters"),
            summary_data.get("model_parameters"),
        ),
        "metrics_status": "ok" if metrics is not None else "missing",
        "timing_status": "ok" if timing is not None else "missing",
        "train_summary_status": "ok" if summary is not None else "missing",
        "metrics_path": str(metrics_path),
        "timing_path": str(timing_path),
        "train_summary_path": str(summary_path),
    }


def format_value(value: Any, mode: str = "plain") -> str:
    if value is None:
        return "NA"
    if mode == "fixed6":
        return f"{float(value):.6f}"
    if mode == "scientific":
        return f"{float(value):.6e}"
    if mode == "fixed2":
        return f"{float(value):.2f}"
    if mode == "fixed1":
        return f"{float(value):.1f}"
    return str(value)


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"num_experiments": len(records), "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_markdown(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment Inventory",
        "",
        "Final accuracy metrics are taken from `inference/metrics.json`.",
        "",
        "| Task | Model | Type | Relative L2 | MSE | Speedup | Samples | Fields | Epochs | LR | Width | Depth | Modes | Train time (s) | Files |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]

    for record in records:
        if record["modes"] is not None:
            modes_text = str(record["modes"])
        elif record["modes1"] is not None or record["modes2"] is not None:
            modes_text = f"({record['modes1'] or 'NA'}, {record['modes2'] or 'NA'})"
        else:
            modes_text = "NA"

        files_text = (
            f"M:{record['metrics_status']}, "
            f"T:{record['timing_status']}, "
            f"S:{record['train_summary_status']}"
        )

        values = [
            record["task_name"],
            record["model_name"],
            record["model_type"] or "NA",
            format_value(record["relative_l2"], "fixed6"),
            format_value(record["mse"], "scientific"),
            format_value(record["speedup_total"], "fixed2"),
            format_value(record["num_samples"]),
            format_value(record["num_fields"]),
            format_value(record["epochs"]),
            format_value(record["learning_rate"], "scientific"),
            format_value(record["width"]),
            format_value(record["depth"]),
            modes_text,
            format_value(record["train_total_seconds"], "fixed1"),
            files_text,
        ]
        lines.append("| " + " | ".join(values) + " |")

    incomplete = [
        record
        for record in records
        if "missing"
        in {
            record["metrics_status"],
            record["timing_status"],
            record["train_summary_status"],
        }
    ]

    lines.extend(
        [
            "",
            "## Completeness check",
            "",
            f"- Total experiments: {len(records)}",
            f"- Experiments with missing result files: {len(incomplete)}",
        ]
    )

    for record in incomplete:
        lines.append(
            f"- `{record['task_name']}/{record['model_name']}`: "
            f"metrics={record['metrics_status']}, "
            f"timing={record['timing_status']}, "
            f"train_summary={record['train_summary_status']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Experiment output root.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs/_summary"),
        help="Summary output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.outputs_root.exists():
        raise FileNotFoundError(
            f"outputs root does not exist: {args.outputs_root}"
        )

    model_dirs = list(discover_model_dirs(args.outputs_root))
    records = [build_record(model_dir) for model_dir in model_dirs]
    records.sort(key=lambda item: (item["task_name"], item["model_name"]))

    csv_path = args.summary_dir / "experiment_inventory.csv"
    json_path = args.summary_dir / "experiment_inventory.json"
    markdown_path = args.summary_dir / "experiment_inventory.md"

    write_csv(records, csv_path)
    write_json(records, json_path)
    write_markdown(records, markdown_path)

    incomplete_count = sum(
        1
        for record in records
        if "missing"
        in {
            record["metrics_status"],
            record["timing_status"],
            record["train_summary_status"],
        }
    )

    print("=" * 72)
    print("Experiment summary completed")
    print("=" * 72)
    print(f"Experiments : {len(records)}")
    print(f"Incomplete  : {incomplete_count}")
    print(f"CSV         : {csv_path}")
    print(f"JSON        : {json_path}")
    print(f"Markdown    : {markdown_path}")


if __name__ == "__main__":
    main()
