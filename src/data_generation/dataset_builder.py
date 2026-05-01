# ==========================================================
# File: src/data_generation/dataset_builder.py
#
# 功能简介：
# 1. 串联整个数据生成链条；
# 2. 执行：
#    TaskSpec -> 参数采样 -> 合法性检查 -> 轨道积分 -> 结果收集
# 3. 收集成功样本与失败样本；
# 4. 输出内存中的 DatasetBuildResult 对象；
# 5. 为 dataset_saver.py 提供待保存的数据集结果。
#
# 依赖关系：
# - 依赖 sampler.py
# - 依赖 orbit_solver.py
# - 依赖 validity.py
# - 依赖 astrophysical_checks.py
#
# 重要说明：
# - 本文件负责“构建结果”，但不直接保存到磁盘；
# - 它是数据生成模块中的“流程总控”；
# - 真正写入 dataset.npz / meta.json 的逻辑在 dataset_saver.py。
# ==========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.common.task_spec import TaskSpec
from src.data_generation.astrophysical_checks import warn_task_spec_astrophysical_ranges
from src.data_generation.orbit_solver import InitialState, KerrParams, simulate_one_orbit
from src.data_generation.sampler import build_parameter_samples, count_expected_samples
from src.data_generation.validity import (
    validate_single_sample_hard_constraints,
    validate_task_spec_hard_constraints,
)


# ==========================================================
# 一、数据集构建结果对象
# ==========================================================

@dataclass
class DatasetBuildResult:
    """
    数据集构建结果。

    说明：
    - 这是“内存中的结果对象”，还没有保存到磁盘；
    - 后续由 dataset_saver.py 负责写入 dataset.npz / meta.json / failed_samples.json。
    """

    task_spec: TaskSpec
    lambda_grid: np.ndarray

    # 成功样本
    successful_vary_params: list[dict[str, Any]]
    successful_outputs_xyz: np.ndarray
    successful_outputs_sph: np.ndarray

    # 失败样本
    failed_samples: list[dict[str, Any]]

    # 统计信息
    requested_samples: int
    success_count: int
    fail_count: int
    astrophysical_warnings: list[str]


# ==========================================================
# 二、主入口
# ==========================================================

def build_dataset(task_spec: TaskSpec) -> DatasetBuildResult:
    """
    根据 TaskSpec 构建完整数据集。

    总流程：
    1. 任务级硬约束检查
    2. 任务级现实范围 warning 检查
    3. 构造参数样本列表
    4. 逐个样本检查与数值积分
    5. 收集成功与失败结果
    6. 打包成 DatasetBuildResult 返回
    """
    # ------------------------------------------------------
    # A. 任务级检查
    # ------------------------------------------------------
    validate_task_spec_hard_constraints(task_spec)
    astrophysical_warnings = warn_task_spec_astrophysical_ranges(task_spec)

    # ------------------------------------------------------
    # B. 采样
    # ------------------------------------------------------
    parameter_samples = build_parameter_samples(task_spec)
    requested_samples = count_expected_samples(task_spec)

    # 理论上这里二者应一致，若不一致说明采样器逻辑有问题
    if len(parameter_samples) != requested_samples:
        raise RuntimeError(
            "参数采样数量与理论请求数量不一致："
            f"len(parameter_samples)={len(parameter_samples)}, "
            f"requested_samples={requested_samples}"
        )

    # ------------------------------------------------------
    # C. 构建结果容器
    # ------------------------------------------------------
    successful_vary_params: list[dict[str, Any]] = []
    successful_outputs_xyz: list[np.ndarray] = []
    successful_outputs_sph: list[np.ndarray] = []
    failed_samples: list[dict[str, Any]] = []

    lambda_grid_ref: np.ndarray | None = None

    # ------------------------------------------------------
    # D. 逐个样本处理
    # ------------------------------------------------------
    for sample_params in parameter_samples:
        try:
            # ---------- 单样本硬约束检查 ----------
            validate_single_sample_hard_constraints(
                sample_params=sample_params,
                fixed_params=task_spec.fixed_params,
            )

            # ---------- 合并完整参数 ----------
            full_params = merge_sample_and_fixed_params(
                sample_params=sample_params,
                fixed_params=task_spec.fixed_params,
            )

            # ---------- 构造轨道所需对象 ----------
            kerr_params = build_kerr_params(full_params)
            init_state = build_initial_state(full_params)
            Q_value = float(full_params["Q"])

            # ---------- 数值积分 ----------
            orbit_result = simulate_one_orbit(
                p=kerr_params,
                init=init_state,
                Q=Q_value,
                n_steps=task_spec.n_steps,
                step_size=task_spec.step_size,
            )

            # ---------- 统一 lambda_grid ----------
            current_lambda_grid = orbit_result["lambda_grid"]
            if lambda_grid_ref is None:
                lambda_grid_ref = current_lambda_grid
            else:
                if not np.allclose(lambda_grid_ref, current_lambda_grid):
                    raise RuntimeError("不同样本生成出的 lambda_grid 不一致。")

            # ---------- 收集成功样本 ----------
            successful_vary_params.append(sample_params)
            successful_outputs_xyz.append(orbit_result["xyz"])
            successful_outputs_sph.append(orbit_result["sph"])

        except Exception as e:
            failed_samples.append({
                "vary_params": dict(sample_params),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

    # ------------------------------------------------------
    # E. 构建最终结果对象
    # ------------------------------------------------------
    if lambda_grid_ref is None:
        raise RuntimeError("所有样本都生成失败，没有得到任何有效 lambda_grid。")

    if len(successful_outputs_xyz) == 0:
        raise RuntimeError("所有样本都生成失败，没有任何成功样本。")
    
    # [N,T,3]
    # 一共有 N 条轨道（成功的轨道数量）；
    # 每条轨道有 T 个离散点（λ方向上的采样点数）；
    # 每个点有 3 个坐标值 (x,y,z)。
    xyz_array = np.stack(successful_outputs_xyz, axis=0)   # [N,T,3]
    sph_array = np.stack(successful_outputs_sph, axis=0)   # [N,T,3]

    result = DatasetBuildResult(
        task_spec=task_spec,
        lambda_grid=lambda_grid_ref,
        successful_vary_params=successful_vary_params,
        successful_outputs_xyz=xyz_array,
        successful_outputs_sph=sph_array,
        failed_samples=failed_samples,
        requested_samples=requested_samples,
        success_count=len(successful_outputs_xyz),
        fail_count=len(failed_samples),
        astrophysical_warnings=astrophysical_warnings,
    )
    return result


# ==========================================================
# 三、参数拼装
# ==========================================================

def merge_sample_and_fixed_params(
    sample_params: dict[str, Any],
    fixed_params: dict[str, Any],
) -> dict[str, Any]:
    """
    合并单个样本的变化参数与任务固定参数，得到完整参数字典。
    """
    merged = dict(fixed_params)
    merged.update(sample_params)
    return merged


def build_kerr_params(full_params: dict[str, Any]) -> KerrParams:
    """
    从完整参数字典构造 KerrParams。
    """
    return KerrParams(
        M=float(full_params["M"]),
        a=float(full_params["a"]),
        E=float(full_params["E"]),
        Lz=float(full_params["Lz"]),
    )


def build_initial_state(full_params: dict[str, Any]) -> InitialState:
    """
    从完整参数字典构造 InitialState。
    """
    return InitialState(
        r0=float(full_params["r0"]),
        theta0=float(full_params["theta0"]),
        phi0=float(full_params["phi0"]),
        sign_r=int(full_params["sign_r"]),
        sign_th=int(full_params["sign_th"]),
    )


# ==========================================================
# 四、辅助统计
# ==========================================================

def summarize_build_result(build_result: DatasetBuildResult) -> dict[str, Any]:
    """
    将 DatasetBuildResult 中的关键信息整理成摘要字典。

    用途：
    - 日志打印
    - 保存到 meta.json 前做中间摘要
    """
    return {
        "requested_samples": build_result.requested_samples,
        "success_count": build_result.success_count,
        "fail_count": build_result.fail_count,
        "success_ratio": (
            build_result.success_count / build_result.requested_samples
            if build_result.requested_samples > 0 else 0.0
        ),
        "task_spec": build_result.task_spec.to_dict(),
        "num_astrophysical_warnings": len(build_result.astrophysical_warnings),
    }