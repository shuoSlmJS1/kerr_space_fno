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
# - 依赖 orbit_solver_second_order.py
# - 依赖 validity.py
# - 依赖 astrophysical_checks.py
#
# 重要说明：
# - 本文件负责“构建结果”，但不直接保存到磁盘；
# - 它是数据生成模块中的“流程总控”；
# - 真正写入 dataset.npz / meta.json 的逻辑在 dataset_saver.py。
# ==========================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.common.task_spec import TaskSpec
from src.data_generation.astrophysical_checks import warn_task_spec_astrophysical_ranges
from src.data_generation.orbit_types import InitialState, KerrParams
from src.data_generation.orbit_solver_second_order import (
    simulate_one_orbit_second_order,
)
from src.data_generation.sampler import (
    build_parameter_samples,
    build_random_completion_candidates,
    count_expected_samples,
    sample_key,
)
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

    # 生成统计
    target_success_count: int
    initial_candidate_count: int
    max_attempt_count: int
    attempt_count: int
    success_count: int
    fail_count: int
    generation_completed: bool
    used_completion_sampling: bool
    successful_points_strictly_uniform: bool
    astrophysical_warnings: list[str]


# ==========================================================
# 二、主入口
# ==========================================================

def build_dataset(
    task_spec: TaskSpec,
    progress_every: int = 100,
) -> DatasetBuildResult:
    """
    根据 TaskSpec 构建完整数据集。

    第二阶段生成规则：
    1. sample_shape 的乘积表示目标成功样本数；
    2. 第一批候选点使用 sampling_mode 指定的规则网格；
    3. 初始候选失败后，继续生成新的、不重复的随机候选点；
    4. 成功数达到目标后停止；
    5. 最大尝试数为
       ceil(target_success_count * max_attempt_factor)；
    6. 达到最大尝试数仍未完成时，返回 incomplete 结果，
       由保存层保存失败记录后再由入口脚本报错。
    """
    # ------------------------------------------------------
    # A. 任务级检查
    # ------------------------------------------------------
    validate_task_spec_hard_constraints(task_spec)
    astrophysical_warnings = warn_task_spec_astrophysical_ranges(
        task_spec
    )

    # ------------------------------------------------------
    # B. 生成目标和初始候选
    # ------------------------------------------------------
    target_success_count = count_expected_samples(task_spec)
    initial_samples = build_parameter_samples(task_spec)
    initial_candidate_count = len(initial_samples)

    if initial_candidate_count != target_success_count:
        raise RuntimeError(
            "初始参数采样数量与目标成功数不一致："
            f"initial_candidate_count={initial_candidate_count}, "
            f"target_success_count={target_success_count}"
        )

    max_attempt_count = int(
        math.ceil(
            target_success_count
            * float(task_spec.max_attempt_factor)
        )
    )

    candidate_queue: list[
        tuple[dict[str, Any], str]
    ] = [
        (dict(sample), "initial_grid")
        for sample in initial_samples
    ]

    attempted_keys = {
        sample_key(sample, task_spec.vary_params)
        for sample in initial_samples
    }

    completion_rng = np.random.default_rng(task_spec.seed)

    # ------------------------------------------------------
    # C. 构建结果容器
    # ------------------------------------------------------
    successful_vary_params: list[dict[str, Any]] = []
    successful_outputs_xyz: list[np.ndarray] = []
    successful_outputs_sph: list[np.ndarray] = []
    failed_samples: list[dict[str, Any]] = []

    lambda_grid_ref: np.ndarray | None = None

    max_radial_constraint = 0.0
    max_polar_constraint = 0.0

    attempt_count = 0
    used_completion_sampling = False

    if progress_every <= 0:
        raise ValueError(
            "progress_every must be a positive integer."
        )

    generation_start_time = perf_counter()

    # ------------------------------------------------------
    # D. 目标成功数驱动生成
    # ------------------------------------------------------
    while (
        len(successful_outputs_xyz) < target_success_count
        and attempt_count < max_attempt_count
    ):
        # 初始网格耗尽后，生成新的补充候选。
        if not candidate_queue:
            remaining_attempt_capacity = (
                max_attempt_count - attempt_count
            )
            remaining_success_needed = (
                target_success_count
                - len(successful_outputs_xyz)
            )

            candidate_count = min(
                remaining_attempt_capacity,
                remaining_success_needed,
            )

            if candidate_count <= 0:
                break

            completion_candidates = (
                build_random_completion_candidates(
                    task_spec=task_spec,
                    candidate_count=candidate_count,
                    attempted_keys=attempted_keys,
                    rng=completion_rng,
                )
            )

            candidate_queue.extend(
                (dict(sample), "completion_random")
                for sample in completion_candidates
            )
            used_completion_sampling = True

        sample_params, candidate_source = (
            candidate_queue.pop(0)
        )
        attempt_count += 1
        stage = "hard_constraint_validation"

        try:
            # ---------- 单样本硬约束检查 ----------
            validate_single_sample_hard_constraints(
                sample_params=sample_params,
                fixed_params=task_spec.fixed_params,
            )

            # ---------- 合并完整参数 ----------
            stage = "parameter_assembly"
            full_params = merge_sample_and_fixed_params(
                sample_params=sample_params,
                fixed_params=task_spec.fixed_params,
            )

            # ---------- 构造轨道所需对象 ----------
            stage = "state_construction"
            kerr_params = build_kerr_params(full_params)
            init_state = build_initial_state(full_params)
            Q_value = float(full_params["Q"])

            # ---------- 数值积分 ----------
            stage = "orbit_integration"
            orbit_result = simulate_one_orbit_second_order(
                p=kerr_params,
                init=init_state,
                Q=Q_value,
                n_steps=task_spec.n_steps,
                step_size=task_spec.step_size,
            )

            # ---------- 统一 lambda_grid ----------
            stage = "lambda_grid_validation"
            current_lambda_grid = orbit_result["lambda_grid"]

            if lambda_grid_ref is None:
                lambda_grid_ref = current_lambda_grid
            elif not np.allclose(
                lambda_grid_ref,
                current_lambda_grid,
            ):
                raise RuntimeError(
                    "不同样本生成出的 lambda_grid 不一致。"
                )

            # ---------- 更新求解器诊断 ----------
            stage = "solver_diagnostics"
            diagnostics = orbit_result["diagnostics"]

            max_radial_constraint = max(
                max_radial_constraint,
                float(
                    diagnostics
                    .max_radial_constraint_residual
                ),
            )
            max_polar_constraint = max(
                max_polar_constraint,
                float(
                    diagnostics
                    .max_polar_constraint_residual
                ),
            )

            # ---------- 收集成功样本 ----------
            stage = "success_collection"
            successful_vary_params.append(
                dict(sample_params)
            )
            successful_outputs_xyz.append(
                orbit_result["xyz"]
            )
            successful_outputs_sph.append(
                orbit_result["sph"]
            )

        except Exception as error:
            failed_samples.append({
                "attempt_index": int(attempt_count),
                "candidate_source": candidate_source,
                "stage": stage,
                "vary_params": dict(sample_params),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "solver": "second_order_rk4",
                "n_steps": int(task_spec.n_steps),
                "step_size": float(task_spec.step_size),
            })

        # 每隔固定次数刷新同一行，避免终端日志不断向上滚动。
        current_success_count = len(successful_outputs_xyz)
        current_fail_count = len(failed_samples)

        should_report_progress = (
            attempt_count % progress_every == 0
            or current_success_count >= target_success_count
            or attempt_count >= max_attempt_count
        )

        if should_report_progress:
            elapsed_seconds = (
                perf_counter() - generation_start_time
            )

            success_rate = (
                current_success_count / elapsed_seconds
                if elapsed_seconds > 0.0
                else 0.0
            )

            remaining_success_count = max(
                target_success_count
                - current_success_count,
                0,
            )

            eta_seconds = (
                remaining_success_count / success_rate
                if success_rate > 0.0
                else float("inf")
            )

            completion_percent = (
                100.0
                * current_success_count
                / target_success_count
            )

            progress_text = (
                "Generation progress"
                f" | success {current_success_count}/{target_success_count}"
                f" | attempts {attempt_count}"
                f" | failed {current_fail_count}"
                f" | {completion_percent:5.1f}%"
                f" | elapsed {format_duration(elapsed_seconds)}"
                f" | ETA {format_duration(eta_seconds)}"
            )

            print(
                "\r\033[K" + progress_text,
                end="",
                flush=True,
            )

    # 结束进度行，保证后续摘要从新行开始。
    print()

    # ------------------------------------------------------
    # E. 最终状态
    # ------------------------------------------------------
    success_count = len(successful_outputs_xyz)
    fail_count = len(failed_samples)

    generation_completed = (
        success_count == target_success_count
    )

    successful_points_strictly_uniform = (
        not used_completion_sampling
        and fail_count == 0
        and success_count == target_success_count
    )

    if success_count == 0 or lambda_grid_ref is None:
        raise RuntimeError(
            "所有参数候选都生成失败，没有得到有效轨道。"
        )

    xyz_array = np.stack(
        successful_outputs_xyz,
        axis=0,
    )
    sph_array = np.stack(
        successful_outputs_sph,
        axis=0,
    )

    # ------------------------------------------------------
    # F. 元数据
    # ------------------------------------------------------
    task_spec.metadata["orbit_solver"] = (
        "second_order_rk4"
    )
    task_spec.metadata["orbit_solver_version"] = "v1"

    task_spec.metadata[
        "dataset_max_radial_constraint_residual"
    ] = max_radial_constraint

    task_spec.metadata[
        "dataset_max_polar_constraint_residual"
    ] = max_polar_constraint

    task_spec.metadata["generation"] = {
        "completion_policy": (
            task_spec.completion_policy
        ),
        "target_success_count": int(
            target_success_count
        ),
        "initial_candidate_count": int(
            initial_candidate_count
        ),
        "max_attempt_factor": float(
            task_spec.max_attempt_factor
        ),
        "max_attempt_count": int(
            max_attempt_count
        ),
        "attempt_count": int(attempt_count),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "completed": bool(generation_completed),
        "used_completion_sampling": bool(
            used_completion_sampling
        ),
        "initial_sampling_mode": (
            task_spec.sampling_mode
        ),
        "completion_sampling_mode": (
            "uniform_random"
            if used_completion_sampling
            else None
        ),
        "successful_points_strictly_uniform": bool(
            successful_points_strictly_uniform
        ),
        "data_seed": int(task_spec.seed),
    }

    return DatasetBuildResult(
        task_spec=task_spec,
        lambda_grid=lambda_grid_ref,
        successful_vary_params=successful_vary_params,
        successful_outputs_xyz=xyz_array,
        successful_outputs_sph=sph_array,
        failed_samples=failed_samples,
        target_success_count=target_success_count,
        initial_candidate_count=initial_candidate_count,
        max_attempt_count=max_attempt_count,
        attempt_count=attempt_count,
        success_count=success_count,
        fail_count=fail_count,
        generation_completed=generation_completed,
        used_completion_sampling=used_completion_sampling,
        successful_points_strictly_uniform=(
            successful_points_strictly_uniform
        ),
        astrophysical_warnings=(
            astrophysical_warnings
        ),
    )



def format_duration(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS。

    说明：
    - 该函数只用于终端进度显示；
    - 无法估计 ETA 时返回 unknown。
    """
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_part:02d}"
    )


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
    - 区分目标成功数、实际尝试数与实际成功数
    """
    return {
        "target_success_count": build_result.target_success_count,
        "initial_candidate_count": build_result.initial_candidate_count,
        "max_attempt_count": build_result.max_attempt_count,
        "attempt_count": build_result.attempt_count,
        "success_count": build_result.success_count,
        "fail_count": build_result.fail_count,
        "generation_completed": build_result.generation_completed,
        "success_ratio_vs_attempts": (
            build_result.success_count / build_result.attempt_count
            if build_result.attempt_count > 0
            else 0.0
        ),
        "completion_ratio": (
            build_result.success_count
            / build_result.target_success_count
            if build_result.target_success_count > 0
            else 0.0
        ),
        "used_completion_sampling": (
            build_result.used_completion_sampling
        ),
        "successful_points_strictly_uniform": (
            build_result.successful_points_strictly_uniform
        ),
        "task_spec": build_result.task_spec.to_dict(),
        "num_astrophysical_warnings": len(
            build_result.astrophysical_warnings
        ),
    }
