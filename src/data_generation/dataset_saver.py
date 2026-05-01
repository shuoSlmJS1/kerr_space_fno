# ==========================================================
# File: src/data_generation/dataset_saver.py
#
# 功能简介：
# 1. 将 DatasetBuildResult 保存到磁盘；
# 2. 自动执行 train / val / test 切分；
# 3. 保存：
#    - dataset.npz
#    - meta.json
#    - failed_samples.json
# 4. 打印数据生成摘要信息。
#
# 依赖关系：
# - 依赖 dataset_builder.py
# - 依赖 common/io_utils.py
# - 依赖 common/paths.py
#
# 重要说明：
# - 本文件是数据生成模块中的“保存层”；
# - 采用统一路径写入；
# - 同路径保存时默认覆盖旧文件；
# - 不负责轨道积分和样本检查。
# ==========================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.io_utils import print_kv, print_section, save_json, save_npz, split_indices
from src.common.paths import build_task_data_paths, ensure_task_data_dirs
from src.common.task_spec import TaskSpec
from src.data_generation.dataset_builder import DatasetBuildResult


# ==========================================================
# 一、对外主入口
# ==========================================================

def save_built_dataset(build_result: DatasetBuildResult) -> dict[str, Any]:
    """
    将内存中的 DatasetBuildResult 保存到磁盘。

    保存内容：
    1. dataset.npz
    2. meta.json
    3. failed_samples.json

    返回：
    - 一个摘要字典，便于入口脚本打印/记录
    """
    task_spec = build_result.task_spec
    task_name = task_spec.metadata.get("task_name", None)
    if task_name is None:
        raise ValueError(
            "task_spec.metadata 中缺少 task_name。"
            "请在入口脚本中先生成 task_name 并写入 task_spec.metadata['task_name']。"
        )

    paths = ensure_task_data_dirs(task_name)

    # ------------------------------------------------------
    # A. 切分索引
    # ------------------------------------------------------
    train_idx, val_idx, test_idx = split_indices(
        n=build_result.success_count,
        split_ratios=task_spec.split_ratios,
        seed=task_spec.seed,
    )

    # ------------------------------------------------------
    # B. 提取变化参数数组
    # ------------------------------------------------------
    vary_array = convert_vary_params_list_to_array(
        vary_params_list=build_result.successful_vary_params,
        vary_params_order=task_spec.vary_params,
    )

    # ------------------------------------------------------
    # C. 保存 npz
    # ------------------------------------------------------
    npz_payload = build_npz_payload(
        task_spec=task_spec,
        vary_array=vary_array,
        xyz_array=build_result.successful_outputs_xyz,
        lambda_grid=build_result.lambda_grid,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    save_npz(paths.dataset_npz, **npz_payload)

    # ------------------------------------------------------
    # D. 保存 meta.json
    # ------------------------------------------------------
    meta = build_meta_dict(
        build_result=build_result,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    save_json(meta, paths.meta_json)

    # ------------------------------------------------------
    # E. 保存 failed_samples.json
    # ------------------------------------------------------
    save_json(build_result.failed_samples, paths.failed_samples_json)

    # ------------------------------------------------------
    # F. 打印摘要
    # ------------------------------------------------------
    print_dataset_save_summary(
        task_name=task_name,
        dataset_npz_path=str(paths.dataset_npz),
        meta_json_path=str(paths.meta_json),
        failed_json_path=str(paths.failed_samples_json),
        requested_samples=build_result.requested_samples,
        success_count=build_result.success_count,
        fail_count=build_result.fail_count,
        train_size=len(train_idx),
        val_size=len(val_idx),
        test_size=len(test_idx),
    )

    return {
        "task_name": task_name,
        "dataset_npz_path": str(paths.dataset_npz),
        "meta_json_path": str(paths.meta_json),
        "failed_json_path": str(paths.failed_samples_json),
        "requested_samples": build_result.requested_samples,
        "success_count": build_result.success_count,
        "fail_count": build_result.fail_count,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }


# ==========================================================
# 二、变化参数列表转数组
# ==========================================================

def convert_vary_params_list_to_array(
    vary_params_list: list[dict[str, Any]],
    vary_params_order: list[str],
) -> np.ndarray:
    """
    将成功样本中的变化参数字典列表转成二维数组。

    输入：
    - vary_params_list:
        [
            {"Q": 1.6},
            {"Q": 1.7},
            ...
        ]
      或
        [
            {"a": 0.4, "Q": 1.6},
            {"a": 0.4, "Q": 1.7},
            ...
        ]

    输出：
    - np.ndarray, shape = [N, K]
      其中 K 是变化参数个数，列顺序由 vary_params_order 指定。
    """
    rows = []
    for item in vary_params_list:
        row = [float(item[p]) for p in vary_params_order]
        rows.append(row)

    return np.asarray(rows, dtype=np.float64)


# ==========================================================
# 三、构造 npz payload
# ==========================================================

def build_npz_payload(
    task_spec: TaskSpec,
    vary_array: np.ndarray,
    xyz_array: np.ndarray,
    lambda_grid: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """
    构造保存到 dataset.npz 的字典内容。

    统一规则：
    - 单参数任务：
        x_train / x_val / x_test : [N,1]
    - 双参数任务：
        x_train / x_val / x_test : [N,2]
    - 更高维同理扩展

    输出始终是：
        y_train / y_val / y_test : [N,T,3]
    """
    payload = {
        "vary_params_order": np.asarray(task_spec.vary_params, dtype=np.str_),
        "x_train": vary_array[train_idx],
        "x_val": vary_array[val_idx],
        "x_test": vary_array[test_idx],
        "y_train": xyz_array[train_idx],
        "y_val": xyz_array[val_idx],
        "y_test": xyz_array[test_idx],
        "lambda_grid": lambda_grid,
    }
    return payload


# ==========================================================
# 四、构造 meta.json
# ==========================================================

def build_meta_dict(
    build_result: DatasetBuildResult,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """
    构造 meta.json 内容。

    说明：
    - meta.json 是任务级别的说明文件；
    - 既包含 TaskSpec，也包含实际生成统计结果。
    """
    task_spec = build_result.task_spec
    task_name = task_spec.metadata.get("task_name", None)

    return {
        "task_name": task_name,
        "task_spec": task_spec.to_dict(),
        "config_tag": task_spec.config_tag,
        "requested_samples": build_result.requested_samples,
        "success_count": build_result.success_count,
        "fail_count": build_result.fail_count,
        "success_ratio": (
            build_result.success_count / build_result.requested_samples
            if build_result.requested_samples > 0 else 0.0
        ),
        "split": {
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "train_ratio": float(task_spec.split_ratios[0]),
            "val_ratio": float(task_spec.split_ratios[1]),
            "test_ratio": float(task_spec.split_ratios[2]),
        },
        "integration": {
            "n_steps": int(task_spec.n_steps),
            "step_size": float(task_spec.step_size),
            "lambda_max": float(task_spec.lambda_max),
        },
        "vary_params_order": list(task_spec.vary_params),
        "astrophysical_warnings": list(build_result.astrophysical_warnings),
    }


# ==========================================================
# 五、日志打印
# ==========================================================

def print_dataset_save_summary(
    task_name: str,
    dataset_npz_path: str,
    meta_json_path: str,
    failed_json_path: str,
    requested_samples: int,
    success_count: int,
    fail_count: int,
    train_size: int,
    val_size: int,
    test_size: int,
) -> None:
    """
    打印数据集保存摘要。
    """
    print_section("Dataset generation finished")
    print_kv("task_name", task_name)
    print_kv("requested_samples", requested_samples)
    print_kv("success_count", success_count)
    print_kv("fail_count", fail_count)
    print_kv("train / val / test", f"{train_size} / {val_size} / {test_size}")
    print_kv("dataset.npz", dataset_npz_path)
    print_kv("meta.json", meta_json_path)
    print_kv("failed_samples.json", failed_json_path)
