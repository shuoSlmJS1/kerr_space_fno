"""Validate Plan B Protocol v1 paired coarse/fine numerical truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许从项目根目录直接执行该脚本，同时不改变包导入路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.plan_b_paired import (
    validate_plan_b_ground_truth,
    write_consistency_result_exclusively,
)


def build_parser() -> argparse.ArgumentParser:
    """构造 Plan B truth qualification 的窄命令行接口。"""
    parser = argparse.ArgumentParser(
        description="Validate Plan B v1 paired coarse/fine ground truth before FNO inference."
    )
    parser.add_argument("--coarse-dataset-dir", type=Path, required=True)
    parser.add_argument("--fine-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_plan_b_ground_truth(
        coarse_dataset_dir=args.coarse_dataset_dir,
        fine_dataset_dir=args.fine_dataset_dir,
    )
    write_consistency_result_exclusively(result, args.output_json)
    print(f"Plan B qualification written: {args.output_json.resolve()}")
    print(f"Structural valid: {result['structural_valid']}")


if __name__ == "__main__":
    main()
