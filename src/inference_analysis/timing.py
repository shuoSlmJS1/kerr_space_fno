# ==========================================================
# File: src/inference_analysis/timing.py
#
# 功能简介：
# 1. 统计模型推理耗时；
# 2. 统计传统数值轨道生成耗时；
# 3. 提供统一的时间对比接口；
# 4. 为 comparison.py / run_analysis.py / result_saver.py 提供标准时间结果。
#
# 依赖关系：
# - 依赖 src/inference_analysis/inference.py（间接）
# - 可依赖 src/data_generation/orbit_solver.py 做传统数值时间统计
# - 被 scripts/run_analysis.py 调用
#
# 重要说明：
# - 本文件只负责“计时”；
# - 不负责误差计算；
# - 不负责画图；
# - 不负责结果保存。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_generation.dataset_builder import build_initial_state, build_kerr_params, merge_sample_and_fixed_params
from src.data_generation.orbit_solver import simulate_one_orbit


# ==========================================================
# 一、结果对象
# ==========================================================

@dataclass
class TimingResult:
    """
    时间统计结果。

    字段说明：
    - total_seconds           : 总耗时（秒）
    - avg_seconds_per_sample  : 平均每个样本耗时（秒）
    - num_samples             : 样本数
    """
    total_seconds: float
    avg_seconds_per_sample: float
    num_samples: int


# ==========================================================
# 二、模型推理计时
# ==========================================================

@torch.no_grad()
def time_model_inference_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    warmup: bool = True,
) -> TimingResult:
    """
    对整个 DataLoader 的模型推理做计时。

    参数：
    - model
    - loader
    - device
    - warmup: 是否先做一次热身，减少首次调用偏差

    返回：
    - TimingResult
    """
    model.eval()

    # ------------------------------
    # A. 热身
    # ------------------------------
    if warmup:
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            break

    # ------------------------------
    # B. 正式计时
    # ------------------------------
    total_samples = 0
    start = perf_counter()

    for x, _ in loader:
        x = x.to(device)
        _ = model(x)

        # CUDA 必须同步，否则计时会偏小
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
    )


# ==========================================================
# 三、传统数值方法计时
# ==========================================================

# 这个故意先 NotImplementedError ，因为单靠 vary_params_array 还不够，
# 需要结合 vary_params_order + fixed_params 才能恢复完整参数。
# 
# 这个恢复逻辑我已经单独写成了：
# build_full_param_dicts_for_timing(...)
# 所以后面在 run_analysis.py 里组合调用就行
def time_traditional_orbit_generation(
    vary_params_array: np.ndarray,
    vary_params_order: list[str],
    fixed_params: dict[str, Any],
    n_steps: int,
    step_size: float,
) -> TimingResult:
    """
    对传统数值轨道生成做计时（便捷接口）。

    参数：
    - vary_params_array:
        shape = [N, K]
        每行是一个样本的变化参数取值
    - vary_params_order:
        长度为 K 的参数名列表，例如：
        ["Q"] 或 ["a", "Q"]
    - fixed_params:
        当前任务的固定参数字典
    - n_steps
    - step_size

    返回：
    - TimingResult

    逻辑：
    1. 先把变化参数数组恢复为完整参数字典列表
    2. 再调用传统数值轨道生成计时函数
    """
    full_param_dicts = build_full_param_dicts_for_timing(
        vary_params_array=vary_params_array,
        vary_params_order=vary_params_order,
        fixed_params=fixed_params,
    )

    return time_traditional_orbit_generation_from_param_dicts(
        full_param_dicts=full_param_dicts,
        n_steps=n_steps,
        step_size=step_size,
    )


def time_traditional_orbit_generation_from_param_dicts(
    full_param_dicts: list[dict[str, Any]],
    n_steps: int,
    step_size: float,
) -> TimingResult:
    """
    对传统数值轨道生成做计时（推荐接口）。

    参数：
    - full_param_dicts:
        长度为 N 的列表，每个元素都是一个完整参数字典
    - n_steps
    - step_size

    返回：
    - TimingResult

    说明：
    - 每个参数字典应至少包含：
        M, a, E, Lz, Q, r0, theta0, phi0, sign_r, sign_th
    """
    num_samples = len(full_param_dicts)
    if num_samples <= 0:
        raise ValueError("full_param_dicts 不能为空。")

    start = perf_counter()

    for full_params in full_param_dicts:
        kerr_params = build_kerr_params(full_params)
        init_state = build_initial_state(full_params)
        Q_value = float(full_params["Q"])

        _ = simulate_one_orbit(
            p=kerr_params,
            init=init_state,
            Q=Q_value,
            n_steps=n_steps,
            step_size=step_size,
        )

    total_seconds = perf_counter() - start

    return TimingResult(
        total_seconds=float(total_seconds),
        avg_seconds_per_sample=float(total_seconds / num_samples),
        num_samples=int(num_samples),
    )


# ==========================================================
# 四、构造完整参数字典列表
# ==========================================================

def build_full_param_dicts_for_timing(
    vary_params_array: np.ndarray,
    vary_params_order: list[str],
    fixed_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    将原始变化参数数组恢复为完整参数字典列表。

    参数：
    - vary_params_array : [N, K]
    - vary_params_order : 长度为 K 的参数名列表
    - fixed_params      : 固定参数字典

    返回：
    - 长度为 N 的完整参数字典列表
    """
    if vary_params_array.ndim != 2:
        raise ValueError(
            f"vary_params_array 必须是二维数组 [N,K]，当前 shape={vary_params_array.shape}"
        )

    if vary_params_array.shape[1] != len(vary_params_order):
        raise ValueError(
            "vary_params_array 列数必须等于 vary_params_order 长度："
            f"vary_params_array.shape={vary_params_array.shape}, vary_params_order={vary_params_order}"
        )

    full_param_dicts: list[dict[str, Any]] = []

    for row in vary_params_array:
        sample_params = {
            param_name: float(value)
            for param_name, value in zip(vary_params_order, row)
        }
        full_params = merge_sample_and_fixed_params(sample_params, fixed_params)
        full_param_dicts.append(full_params)

    return full_param_dicts


# ==========================================================
# 五、时间对比摘要
# ==========================================================

def summarize_timing_result(result: TimingResult) -> dict[str, Any]:
    """
    将 TimingResult 整理成摘要字典。
    """
    return {
        "total_seconds": float(result.total_seconds),
        "avg_seconds_per_sample": float(result.avg_seconds_per_sample),
        "num_samples": int(result.num_samples),
    }


def compare_timing_results(
    model_timing: TimingResult,
    traditional_timing: TimingResult,
) -> dict[str, Any]:
    """
    对比模型推理时间与传统数值计算时间。

    返回：
    {
        "model_total_seconds": ...,
        "traditional_total_seconds": ...,
        "model_avg_seconds_per_sample": ...,
        "traditional_avg_seconds_per_sample": ...,
        "speedup_total": ...,
        "speedup_per_sample": ...,
    }
    """
    speedup_total = (
        traditional_timing.total_seconds / model_timing.total_seconds
        if model_timing.total_seconds > 0 else float("inf")
    )

    speedup_per_sample = (
        traditional_timing.avg_seconds_per_sample / model_timing.avg_seconds_per_sample
        if model_timing.avg_seconds_per_sample > 0 else float("inf")
    )

    return {
        "model_total_seconds": float(model_timing.total_seconds),
        "traditional_total_seconds": float(traditional_timing.total_seconds),
        "model_avg_seconds_per_sample": float(model_timing.avg_seconds_per_sample),
        "traditional_avg_seconds_per_sample": float(traditional_timing.avg_seconds_per_sample),
        "speedup_total": float(speedup_total),
        "speedup_per_sample": float(speedup_per_sample),
        "num_samples": int(model_timing.num_samples),
    }