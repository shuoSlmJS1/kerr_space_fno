# ==========================================================
# File: src/inference_analysis/timing.py
#
# 功能简介：
# 1. 统计模型推理耗时；
# 2. 统计二阶数值轨道生成耗时；
# 3. 只使用正式二阶求解器进行数值计时；
# 4. 提供统一的时间对比接口。
#
# 计时口径：
# - 模型计时：包含 DataLoader 遍历、CPU 到 device 传输、模型 forward、
#   每个 batch 的 CUDA synchronize；不包含反归一化、误差计算和文件保存。
# - 数值计时：只包含逐轨道积分；不包含文件保存、数据集拆分和训练。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_generation.dataset_builder import (
    build_initial_state,
    build_kerr_params,
    merge_sample_and_fixed_params,
)
from src.data_generation.orbit_solver_second_order import (
    simulate_one_orbit_second_order,
)


@dataclass
class TimingResult:
    total_seconds: float
    avg_seconds_per_sample: float
    num_samples: int
    solver_name: str = "unknown"
    n_steps: int | None = None
    step_size: float | None = None
    refinement_factor: int = 1


@torch.no_grad()
def time_model_inference_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    warmup: bool = True,
) -> TimingResult:
    """
    对整个 DataLoader 的模型推理计时。

    计入：
    - DataLoader 迭代；
    - x.to(device)；
    - model(x)；
    - CUDA 同步。

    不计入：
    - 反归一化；
    - 指标计算；
    - 保存预测文件。
    """
    model.eval()

    if warmup:
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            break

    total_samples = 0
    start = perf_counter()

    for x, _ in loader:
        x = x.to(device)
        _ = model(x)

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        total_samples += int(x.shape[0])

    total_seconds = perf_counter() - start

    if total_samples <= 0:
        raise RuntimeError("模型推理计时失败：DataLoader 中没有样本。")

    return TimingResult(
        total_seconds=float(total_seconds),
        avg_seconds_per_sample=float(total_seconds / total_samples),
        num_samples=int(total_samples),
        solver_name="fno",
    )


def build_full_param_dicts_for_timing(
    vary_params_array: np.ndarray,
    vary_params_order: list[str],
    fixed_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """把变化参数数组恢复成完整物理参数字典列表。"""
    if vary_params_array.ndim != 2:
        raise ValueError(
            f"vary_params_array 必须是二维数组 [N,K]，当前 shape={vary_params_array.shape}"
        )

    if vary_params_array.shape[1] != len(vary_params_order):
        raise ValueError(
            "vary_params_array 列数必须等于 vary_params_order 长度："
            f"{vary_params_array.shape} 与 {vary_params_order}"
        )

    full_param_dicts: list[dict[str, Any]] = []

    for row in vary_params_array:
        sample_params = {
            param_name: float(value)
            for param_name, value in zip(vary_params_order, row)
        }
        full_param_dicts.append(
            merge_sample_and_fixed_params(sample_params, fixed_params)
        )

    return full_param_dicts


def time_traditional_orbit_generation(
    vary_params_array: np.ndarray,
    vary_params_order: list[str],
    fixed_params: dict[str, Any],
    n_steps: int,
    step_size: float,
    refinement_factor: int = 1,
) -> TimingResult:
    full_param_dicts = build_full_param_dicts_for_timing(
        vary_params_array=vary_params_array,
        vary_params_order=vary_params_order,
        fixed_params=fixed_params,
    )

    return time_traditional_orbit_generation_from_param_dicts(
        full_param_dicts=full_param_dicts,
        n_steps=n_steps,
        step_size=step_size,
        refinement_factor=refinement_factor,
        warmup=True,
    )


def time_traditional_orbit_generation_from_param_dicts(
    full_param_dicts: list[dict[str, Any]],
    n_steps: int,
    step_size: float,
    refinement_factor: int = 1,
    warmup: bool = True,
) -> TimingResult:
    """
    数值轨道生成计时。

    refinement_factor:
    - 1：任务原始步长；
    - 8：步长缩小 8 倍，同时增加点数，保持 lambda_max 不变。
    """
    num_samples = len(full_param_dicts)
    if num_samples <= 0:
        raise ValueError("full_param_dicts 不能为空。")
    if refinement_factor < 1:
        raise ValueError("refinement_factor 必须为正整数。")
    actual_step_size = float(step_size) / int(refinement_factor)
    actual_n_steps = (int(n_steps) - 1) * int(refinement_factor) + 1

    def run_one(full_params: dict[str, Any]) -> None:
        kerr_params = build_kerr_params(full_params)
        init_state = build_initial_state(full_params)
        Q_value = float(full_params["Q"])

        _ = simulate_one_orbit_second_order(
            p=kerr_params,
            init=init_state,
            Q=Q_value,
            n_steps=actual_n_steps,
            step_size=actual_step_size,
        )

    if warmup:
        run_one(full_param_dicts[0])

    start = perf_counter()
    for full_params in full_param_dicts:
        run_one(full_params)
    total_seconds = perf_counter() - start

    return TimingResult(
        total_seconds=float(total_seconds),
        avg_seconds_per_sample=float(total_seconds / num_samples),
        num_samples=int(num_samples),
        solver_name="second_order_rk4",
        n_steps=int(actual_n_steps),
        step_size=float(actual_step_size),
        refinement_factor=int(refinement_factor),
    )


# ==========================================================
# 四、时间结果摘要与对比
# ==========================================================

def summarize_timing_result(result: TimingResult) -> dict[str, Any]:
    """将 TimingResult 整理成可保存的摘要字典。"""
    return {
        "total_seconds": float(result.total_seconds),
        "avg_seconds_per_sample": float(result.avg_seconds_per_sample),
        "num_samples": int(result.num_samples),
        "solver_name": str(result.solver_name),
        "n_steps": (
            int(result.n_steps)
            if result.n_steps is not None
            else None
        ),
        "step_size": (
            float(result.step_size)
            if result.step_size is not None
            else None
        ),
        "refinement_factor": int(result.refinement_factor),
    }


def compare_timing_results(
    model_timing: TimingResult,
    traditional_timing: TimingResult,
) -> dict[str, Any]:
    """对比模型推理时间与传统数值积分时间。"""
    if model_timing.num_samples != traditional_timing.num_samples:
        raise ValueError(
            "模型与传统积分的计时样本数必须一致，"
            f"当前分别为 {model_timing.num_samples} 和 "
            f"{traditional_timing.num_samples}。"
        )

    speedup_total = (
        traditional_timing.total_seconds / model_timing.total_seconds
        if model_timing.total_seconds > 0
        else float("inf")
    )

    speedup_per_sample = (
        traditional_timing.avg_seconds_per_sample
        / model_timing.avg_seconds_per_sample
        if model_timing.avg_seconds_per_sample > 0
        else float("inf")
    )

    return {
        "model_total_seconds": float(model_timing.total_seconds),
        "traditional_total_seconds": float(
            traditional_timing.total_seconds
        ),
        "model_avg_seconds_per_sample": float(
            model_timing.avg_seconds_per_sample
        ),
        "traditional_avg_seconds_per_sample": float(
            traditional_timing.avg_seconds_per_sample
        ),
        "speedup_total": float(speedup_total),
        "speedup_per_sample": float(speedup_per_sample),
        "num_samples": int(model_timing.num_samples),
        "model_solver_name": str(model_timing.solver_name),
        "traditional_solver_name": str(
            traditional_timing.solver_name
        ),
    }
