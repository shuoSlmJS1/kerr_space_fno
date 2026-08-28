#!/usr/bin/env python3
"""严格验证三个长度数据集的 split、网格与轨迹前缀身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
REQUIRED_KEYS = (
    "vary_params_order",
    "x_train", "x_val", "x_test",
    "y_train", "y_val", "y_test",
    "lambda_grid",
)
PAIR_SPECS = (
    ("short_to_medium", "short", "medium"),
    ("short_to_long", "short", "long"),
    ("medium_to_long", "medium", "long"),
)


def parse_args() -> argparse.Namespace:
    """解析显式数据集路径和预先声明的数值容差。"""
    parser = argparse.ArgumentParser(
        description="Validate strict short/long dataset prefix identity."
    )
    parser.add_argument("--short-dataset-dir", type=Path, required=True)
    parser.add_argument("--medium-dataset-dir", type=Path, required=True)
    parser.add_argument("--long-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--trajectory-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        parser.error("--atol and --rtol must be non-negative.")
    if args.trajectory_batch_size <= 0:
        parser.error("--trajectory-batch-size must be positive.")
    return args


def _json_value(value: Any) -> Any:
    """将数组标量递归转换为 JSON 原生类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _content_sha256(array: np.ndarray) -> str:
    """计算包含 dtype 与 shape 的确定性数组内容哈希。"""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _finite_summary(array: np.ndarray) -> dict[str, Any]:
    """报告数值数组中的 NaN、Inf 与有限值状态。"""
    if array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return {
            "finite": False,
            "nan_count": None,
            "inf_count": None,
            "nonfinite_count": int(array.size),
            "reason": "non_numeric_dtype",
        }
    finite = np.isfinite(array)
    nan_count = int(np.count_nonzero(np.isnan(array)))
    inf_count = int(np.count_nonzero(np.isinf(array)))
    return {
        "finite": bool(np.all(finite)),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
    }


def _load_dataset_info(label: str, directory: Path) -> dict[str, Any]:
    """读取轻量元数据、lambda 与 Q split，不保留完整 y 数组。"""
    directory = directory.resolve()
    dataset_path = directory / "dataset.npz"
    meta_path = directory / "meta.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata does not exist: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as loaded:
        missing = [key for key in REQUIRED_KEYS if key not in loaded.files]
        if missing:
            raise KeyError(f"Dataset is missing required keys: {missing}")
        x = {split: np.asarray(loaded[f"x_{split}"]).copy() for split in SPLITS}
        lambda_grid = np.asarray(loaded["lambda_grid"]).copy()
        vary_params_order = np.asarray(loaded["vary_params_order"]).copy()
    return {
        "label": label,
        "directory": directory,
        "dataset_path": dataset_path,
        "meta_path": meta_path,
        "meta": meta,
        "x": x,
        "lambda_grid": lambda_grid,
        "vary_params_order": vary_params_order,
        "finite_checks": {},
        "y_schema": {},
    }


def _load_y(info: dict[str, Any], split: str) -> np.ndarray:
    """只在需要时解压一个 split 的 trajectory 数组。"""
    with np.load(info["dataset_path"], allow_pickle=False) as loaded:
        return np.asarray(loaded[f"y_{split}"])


def _scan_finite_checks(info: dict[str, Any]) -> None:
    """逐个扫描 y split，避免三份完整轨迹同时驻留内存。"""
    checks: dict[str, Any] = {
        "lambda_grid": _finite_summary(info["lambda_grid"]),
    }
    for split in SPLITS:
        checks[f"x_{split}"] = _finite_summary(info["x"][split])
        y = _load_y(info, split)
        checks[f"y_{split}"] = _finite_summary(y)
        info["y_schema"][split] = {
            "shape": list(y.shape),
            "dtype": str(y.dtype),
        }
    info["finite_checks"] = checks


def _meta_value(meta: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = meta
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _metadata_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """比较当前 meta.json schema 中的关键身份与生成字段。"""
    fields = {
        "vary_params": ("task_spec", "vary_params"),
        "vary_ranges": ("task_spec", "vary_ranges"),
        "fixed_params": ("task_spec", "fixed_params"),
        "split_ratios": ("task_spec", "split_ratios"),
        "seed": ("task_spec", "seed"),
        "sampling_mode": ("task_spec", "sampling_mode"),
        "completion_policy": ("task_spec", "completion_policy"),
        "solver_name": ("task_spec", "metadata", "orbit_solver"),
        "solver_version": ("task_spec", "metadata", "orbit_solver_version"),
        "step_size": ("task_spec", "step_size"),
        "generation_completion_policy": ("generation_status", "completion_policy"),
        "successful_points_strictly_uniform": (
            "generation_status", "successful_points_strictly_uniform"
        ),
    }
    compared: dict[str, Any] = {}
    missing_critical: list[str] = []
    mismatches: list[str] = []
    for name, path in fields.items():
        left_present, left_value = _meta_value(left["meta"], path)
        right_present, right_value = _meta_value(right["meta"], path)
        equal = bool(left_present and right_present and left_value == right_value)
        compared[name] = {
            "path": list(path),
            "left_present": left_present,
            "right_present": right_present,
            "left": _json_value(left_value),
            "right": _json_value(right_value),
            "equal": equal,
            "critical": True,
        }
        if not left_present or not right_present:
            missing_critical.append(name)
        elif not equal:
            mismatches.append(name)

    vary_exact = np.array_equal(left["vary_params_order"], right["vary_params_order"])
    compared["vary_params_order"] = {
        "left": _json_value(left["vary_params_order"]),
        "right": _json_value(right["vary_params_order"]),
        "equal": bool(vary_exact),
        "critical": True,
    }
    if not vary_exact:
        mismatches.append("vary_params_order")

    target_shapes_match = all(
        len(left["y_schema"][split]["shape"]) == 3
        and len(right["y_schema"][split]["shape"]) == 3
        and left["y_schema"][split]["shape"][2] == 3
        and right["y_schema"][split]["shape"][2] == 3
        and left["y_schema"][split]["shape"][2:] == right["y_schema"][split]["shape"][2:]
        for split in SPLITS
    )
    compared["target_representation"] = {
        "source": "dataset_npz_schema",
        "expected_last_dimension": 3,
        "compatible": bool(target_shapes_match),
        "critical": True,
    }
    if not target_shapes_match:
        mismatches.append("target_representation")

    expected_differences = {}
    for name, path in {
        "n_steps": ("task_spec", "n_steps"),
        "lambda_max": ("task_spec", "lambda_max"),
    }.items():
        left_present, left_value = _meta_value(left["meta"], path)
        right_present, right_value = _meta_value(right["meta"], path)
        expected_differences[name] = {
            "left_present": left_present,
            "right_present": right_present,
            "left": _json_value(left_value),
            "right": _json_value(right_value),
            "equal": bool(left_present and right_present and left_value == right_value),
            "expected_to_differ": True,
        }
    return {
        "critical_compatible": not missing_critical and not mismatches,
        "missing_critical_fields": missing_critical,
        "critical_mismatches": mismatches,
        "fields": compared,
        "expected_to_differ": expected_differences,
    }


def _q_identity(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """在保留原 split 顺序的前提下检查 Q 身份。"""
    same_shape = left.shape == right.shape
    exact_equal = bool(same_shape and np.array_equal(left, right))
    same_values_different_order = False
    secondary_eligible = False
    if same_shape and not exact_equal and left.ndim == 2 and left.shape[1] == 1:
        left_q = left[:, 0]
        right_q = right[:, 0]
        if np.all(np.isfinite(left_q)) and np.all(np.isfinite(right_q)):
            same_values_different_order = bool(
                np.array_equal(np.sort(left_q), np.sort(right_q))
            )
            secondary_eligible = bool(
                same_values_different_order
                and np.unique(left_q).size == left_q.size
                and np.unique(right_q).size == right_q.size
            )
    return {
        "left": {
            "shape": list(left.shape), "dtype": str(left.dtype),
            "count": int(left.shape[0]) if left.ndim else int(left.size),
            "content_sha256": _content_sha256(left),
        },
        "right": {
            "shape": list(right.shape), "dtype": str(right.dtype),
            "count": int(right.shape[0]) if right.ndim else int(right.size),
            "content_sha256": _content_sha256(right),
        },
        "exact_equal": exact_equal,
        "same_values_different_order": same_values_different_order,
        "secondary_deterministic_q_match": {
            "eligible": secondary_eligible,
            "performed": False,
            "reason": "primary_validation_preserves_original_split_order",
        },
    }


def _numeric_prefix_check(
    left: np.ndarray,
    right: np.ndarray,
    atol: float,
    rtol: float,
    batch_size: int,
    q_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """比较同形状数组；轨迹数组按 batch 累积统计，避免全局差分副本。"""
    result: dict[str, Any] = {
        "compared_shapes": {"left": list(left.shape), "right": list(right.shape)},
        "exact_equal": False,
        "max_abs_difference": None,
        "mean_abs_difference": None,
        "tolerance_pass": False,
    }
    if left.shape != right.shape:
        result["reason"] = "shape_mismatch"
        return result
    if not _finite_summary(left)["finite"] or not _finite_summary(right)["finite"]:
        result["reason"] = "nonfinite_input"
        return result
    exact_equal = bool(np.array_equal(left, right))
    result["exact_equal"] = exact_equal
    if left.ndim == 1:
        diff = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
        absolute = np.abs(diff)
        result.update({
            "max_abs_difference": float(np.max(absolute)) if absolute.size else 0.0,
            "mean_abs_difference": float(np.mean(absolute)) if absolute.size else 0.0,
            "tolerance_pass": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        })
        return result
    if left.ndim != 3:
        result["reason"] = "trajectory_array_must_be_rank_3"
        return result

    total_abs = 0.0
    total_sq = 0.0
    total_target_sq = 0.0
    total_count = 0
    max_abs = 0.0
    tolerance_pass = True
    per_trajectory: list[float] = []
    for start in range(0, left.shape[0], batch_size):
        stop = min(start + batch_size, left.shape[0])
        left_batch = left[start:stop].astype(np.float64, copy=False)
        right_batch = right[start:stop].astype(np.float64, copy=False)
        diff = left_batch - right_batch
        absolute = np.abs(diff)
        total_abs += float(np.sum(absolute))
        total_sq += float(np.sum(diff * diff))
        total_target_sq += float(np.sum(right_batch * right_batch))
        total_count += int(diff.size)
        max_abs = max(max_abs, float(np.max(absolute)) if absolute.size else 0.0)
        tolerance_pass = tolerance_pass and bool(
            np.allclose(left_batch, right_batch, atol=atol, rtol=rtol)
        )
        diff_norm = np.linalg.norm(diff.reshape(stop - start, -1), axis=1)
        target_norm = np.linalg.norm(right_batch.reshape(stop - start, -1), axis=1)
        per_trajectory.extend((diff_norm / (target_norm + 1e-12)).tolist())
    per = np.asarray(per_trajectory, dtype=np.float64)
    worst_index = int(np.argmax(per)) if per.size else None
    result.update({
        "max_abs_difference": max_abs,
        "mean_abs_difference": total_abs / total_count if total_count else 0.0,
        "rmse": math.sqrt(total_sq / total_count) if total_count else 0.0,
        "relative_l2_overall": math.sqrt(total_sq) / (math.sqrt(total_target_sq) + 1e-12),
        "tolerance_pass": tolerance_pass,
        "per_trajectory_relative_l2": {
            "count": int(per.size),
            "mean": float(np.mean(per)) if per.size else 0.0,
            "median": float(np.median(per)) if per.size else 0.0,
            "p95": float(np.percentile(per, 95)) if per.size else 0.0,
            "max": float(np.max(per)) if per.size else 0.0,
            "worst_trajectory_index": worst_index,
            "worst_trajectory_q": (
                float(q_values[worst_index, 0])
                if worst_index is not None and q_values is not None
                and q_values.ndim == 2 and q_values.shape[1] == 1 else None
            ),
        },
    })
    return result


def _pair_finite(info: dict[str, Any]) -> bool:
    return all(check["finite"] for check in info["finite_checks"].values())


def _classify_pair(metadata: dict[str, Any], split_checks: dict[str, Any],
                   lambda_check: dict[str, Any], trajectory_checks: dict[str, Any],
                   finite_ok: bool) -> str:
    q_ok = all(check["exact_equal"] for check in split_checks.values())
    numeric_checks = [lambda_check, *trajectory_checks.values()]
    shapes_ok = all("reason" not in check for check in numeric_checks)
    exact_numeric = all(check["exact_equal"] for check in numeric_checks)
    tolerance_ok = all(check["tolerance_pass"] for check in numeric_checks)
    if metadata["critical_compatible"] and finite_ok and q_ok and shapes_ok and exact_numeric:
        return "EXACT_PREFIX"
    if (metadata["critical_compatible"] and finite_ok and q_ok and shapes_ok
            and not exact_numeric and tolerance_ok):
        return "NUMERICALLY_EQUIVALENT_PREFIX"
    return "NOT_PAIRED"


def validate_datasets(short_dir: Path, medium_dir: Path, long_dir: Path,
                      atol: float = 1e-12, rtol: float = 1e-12,
                      trajectory_batch_size: int = 16) -> dict[str, Any]:
    """执行全部三对严格身份检查，并返回 JSON 可序列化结果。"""
    if atol < 0.0 or rtol < 0.0 or trajectory_batch_size <= 0:
        raise ValueError("Invalid tolerance or trajectory batch size.")
    infos = {
        "short": _load_dataset_info("short", short_dir),
        "medium": _load_dataset_info("medium", medium_dir),
        "long": _load_dataset_info("long", long_dir),
    }
    for info in infos.values():
        _scan_finite_checks(info)

    metadata_results: dict[str, Any] = {}
    split_results: dict[str, Any] = {}
    lambda_results: dict[str, Any] = {}
    trajectory_results: dict[str, Any] = {}
    pair_classes: dict[str, str] = {}
    for pair_name, left_name, right_name in PAIR_SPECS:
        left, right = infos[left_name], infos[right_name]
        metadata = _metadata_comparison(left, right)
        split_checks = {split: _q_identity(left["x"][split], right["x"][split]) for split in SPLITS}
        prefix_length = int(left["lambda_grid"].shape[0])
        lambda_check = _numeric_prefix_check(
            right["lambda_grid"][:prefix_length], left["lambda_grid"], atol, rtol, trajectory_batch_size
        )
        trajectory_checks: dict[str, Any] = {}
        for split in SPLITS:
            left_y = _load_y(left, split)
            right_y = _load_y(right, split)
            trajectory_checks[split] = _numeric_prefix_check(
                right_y[:, :prefix_length, :], left_y, atol, rtol,
                trajectory_batch_size, left["x"][split]
            )
        finite_ok = _pair_finite(left) and _pair_finite(right)
        metadata_results[pair_name] = metadata
        split_results[pair_name] = split_checks
        lambda_results[pair_name] = lambda_check
        trajectory_results[pair_name] = trajectory_checks
        pair_classes[pair_name] = _classify_pair(metadata, split_checks, lambda_check, trajectory_checks, finite_ok)

    if all(value == "EXACT_PREFIX" for value in pair_classes.values()):
        classification = "EXACT_PREFIX"
    elif all(value != "NOT_PAIRED" for value in pair_classes.values()):
        classification = "NUMERICALLY_EQUIVALENT_PREFIX"
    else:
        classification = "NOT_PAIRED"
    return _json_value({
        "schema_version": "1.0",
        "experiment_type": "dataset_prefix_identity_validation",
        "datasets": {
            name: {
                "dataset_dir": str(info["directory"]),
                "dataset_path": str(info["dataset_path"]),
                "meta_path": str(info["meta_path"]),
            } for name, info in infos.items()
        },
        "tolerance": {
            "atol": atol, "rtol": rtol,
            "tolerance_source": "predeclared_cli_or_default",
            "q_identity_mode": "exact_only",
            "trajectory_batch_size": trajectory_batch_size,
        },
        "metadata_compatibility": metadata_results,
        "split_identity": split_results,
        "lambda_prefix_checks": lambda_results,
        "trajectory_prefix_checks": trajectory_results,
        "finite_checks": {name: info["finite_checks"] for name, info in infos.items()},
        "pair_classification": pair_classes,
        "classification": classification,
        "scientific_reuse": {
            "historical_t1800_reusable": pair_classes["short_to_medium"] != "NOT_PAIRED",
            "t2400_ready_for_future_a1": pair_classes["short_to_long"] != "NOT_PAIRED",
        },
        "notes": [
            "Primary split validation preserves original order and never reorders Q values.",
            "Sequence-length-specific n_steps and lambda_max are recorded as expected-to-differ metadata.",
        ],
    })


def write_result_exclusively(result: dict[str, Any], output_path: Path) -> None:
    """以排他模式写入明确请求的 JSON，拒绝任何覆盖。"""
    output_path = output_path.resolve()
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    result = validate_datasets(
        args.short_dataset_dir, args.medium_dataset_dir, args.long_dataset_dir,
        args.atol, args.rtol, args.trajectory_batch_size,
    )
    write_result_exclusively(result, args.output_json)
    print(f"Validation result written: {args.output_json.resolve()}")
    print(f"Classification: {result['classification']}")


if __name__ == "__main__":
    main()
