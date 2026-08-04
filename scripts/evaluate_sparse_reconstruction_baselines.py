from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.training.trajectory_reconstruction import (  # noqa: E402
    build_sparse_trajectory_data,
    compute_hidden_masked_metrics,
    reconstruct_linear,
    reconstruct_pchip,
)


BASELINE_FUNCTIONS = {
    "linear": reconstruct_linear,
    "pchip": reconstruct_pchip,
}


def build_parser() -> argparse.ArgumentParser:
    """构造独立 baseline 评价命令行。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sparse trajectory reconstruction with Linear and PCHIP "
            "interpolation baselines."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to an existing dataset.npz file.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Existing dataset split to evaluate.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="Regular sparse sampling stride. Must be at least 2.",
    )
    parser.add_argument(
        "--baseline",
        choices=("linear", "pchip", "both"),
        default="both",
        help="Interpolation baseline to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="New JSON output path. Existing files are never overwritten.",
    )
    return parser


def load_dataset_split(
    dataset_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """直接读取既有 split，不执行重新划分。"""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    target_key = f"y_{split}"
    with np.load(dataset_path, allow_pickle=False) as dataset:
        required_keys = (target_key, "lambda_grid")
        missing_keys = [key for key in required_keys if key not in dataset]
        if missing_keys:
            raise KeyError(
                f"Dataset is missing required arrays: {missing_keys}."
            )
        target_xyz = np.asarray(dataset[target_key])
        lambda_grid = np.asarray(dataset["lambda_grid"])

    return target_xyz, lambda_grid


def select_baselines(name: str) -> tuple[str, ...]:
    """解析需要运行的 baseline 集合。"""
    if name == "both":
        return ("linear", "pchip")
    return (name,)


def evaluate_baselines(args: argparse.Namespace) -> dict[str, Any]:
    """构造共享稀疏输入并运行无训练 baseline。"""
    dataset_path = args.dataset_path.resolve()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Output JSON already exists and will not be overwritten: {output_path}"
        )

    target_xyz, lambda_grid = load_dataset_split(
        dataset_path=dataset_path,
        split=str(args.split),
    )
    sparse_data = build_sparse_trajectory_data(
        target_xyz=target_xyz,
        lambda_grid=lambda_grid,
        stride=int(args.stride),
    )

    baseline_results: dict[str, Any] = {}
    for baseline_name in select_baselines(str(args.baseline)):
        prediction = BASELINE_FUNCTIONS[baseline_name](
            lambda_grid=sparse_data.lambda_grid,
            sparse_xyz=sparse_data.sparse_xyz,
            observed_mask=sparse_data.observed_mask,
        )
        baseline_results[baseline_name] = {
            "prediction_shape": list(prediction.shape),
            "metrics": compute_hidden_masked_metrics(
                prediction_xyz=prediction,
                target_xyz=sparse_data.target_xyz,
                hidden_mask=sparse_data.hidden_mask,
            ),
        }

    result = {
        "schema_version": "1.0",
        "experiment_type": "sparse_trajectory_reconstruction_baselines",
        "dataset_path": str(dataset_path),
        "split": str(args.split),
        "target_xyz_shape": list(sparse_data.target_xyz.shape),
        "sparse_xyz_shape": list(sparse_data.sparse_xyz.shape),
        "observed_mask_shape": list(sparse_data.observed_mask.shape),
        "hidden_mask_shape": list(sparse_data.hidden_mask.shape),
        "sampling": sparse_data.sampling.to_dict(),
        "baselines": baseline_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            result,
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_file.write("\n")

    print("Sparse reconstruction baseline evaluation completed.")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split}")
    print(f"Stride: {args.stride}")
    print(f"Baselines: {', '.join(baseline_results)}")
    print(f"Output JSON: {output_path}")
    return result


def main() -> None:
    """命令行主入口。"""
    args = build_parser().parse_args()
    evaluate_baselines(args)


if __name__ == "__main__":
    main()
