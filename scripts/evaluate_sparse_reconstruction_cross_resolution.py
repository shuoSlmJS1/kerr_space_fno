from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.training.trajectory_reconstruction.cross_resolution import (  # noqa: E402
    evaluate_frozen_cross_resolution_run,
    load_frozen_reconstruction_run,
    save_cross_resolution_result,
)


def build_parser() -> argparse.ArgumentParser:
    """构造冻结跨分辨率稀疏重建评估命令行。"""
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen sparse reconstruction run at a new observation stride."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--evaluation-stride", type=int, default=32)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """加载冻结 checkpoint、改变观测 stride 并保存评估结果。"""
    frozen_run = load_frozen_reconstruction_run(
        run_dir=args.run_dir,
        device=args.device,
    )
    result = evaluate_frozen_cross_resolution_run(
        frozen_run=frozen_run,
        dataset_path=args.dataset_path,
        evaluation_stride=args.evaluation_stride,
        split=args.split,
        batch_size=args.batch_size,
    )
    output_path = save_cross_resolution_result(args.output_json, result)
    relative_l2 = result["cross_resolution_metrics"]["raw_hidden_only_metrics"]["overall"]["relative_l2"]
    print("Cross-resolution sparse reconstruction evaluation completed.")
    print(f"Model family: {result['model_family']}")
    print(f"Train stride: {result['train_stride']}")
    print(f"Evaluation stride: {result['evaluation_stride']}")
    print(f"Test raw hidden Relative L2: {float(relative_l2):.6e}")
    print(f"Output JSON: {output_path}")
    return result


def main() -> None:
    """命令行主入口。"""
    try:
        run_evaluation(build_parser().parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
