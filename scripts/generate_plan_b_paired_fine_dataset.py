"""Generate a Plan B Protocol v1 fine dataset by replaying source-Q identities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许从项目根目录直接执行该脚本，同时不改变包导入路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.plan_b_paired import (
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    generate_paired_fine_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """构造 Plan B paired replay 的窄命令行接口。"""
    parser = argparse.ArgumentParser(
        description="Generate Plan B v1 paired fine truth without Q resampling or replacement."
    )
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = generate_paired_fine_dataset(
        source_dataset_dir=args.source_dataset_dir,
        output_dir=args.output_dir,
        n_steps=FINE_N_STEPS,
        step_size=FINE_STEP_SIZE,
        progress_every=args.progress_every,
    )
    replay = metadata["paired_replay"]
    print(f"Plan B paired fine dataset written: {args.output_dir.resolve()}")
    print(f"Source Q count: {replay['source_q_count']}")
    print(f"Success count: {replay['success_count']}")
    print(f"Failure count: {replay['failure_count']}")
    print(f"Paired completeness: {replay['paired_completeness']}")


if __name__ == "__main__":
    main()
