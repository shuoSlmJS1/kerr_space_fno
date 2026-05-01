# ==========================================================
# File: src/inference_analysis/metrics.py
#
# 功能简介：
# 1. 对推理结果计算精度指标；
# 2. 当前主要提供：
#    - MSE
#    - Relative L2
# 3. 支持整体数据集指标与单样本指标；
# 4. 为 plotting.py / comparison.py / run_analysis.py 提供标准指标接口。
#
# 依赖关系：
# - 依赖 src/inference_analysis/inference.py 中的 InferenceResult
# - 被 scripts/run_analysis.py 调用
# - 被 comparison.py 间接调用
#
# 重要说明：
# - 本文件只负责“算指标”；
# - 不负责推理；
# - 不负责耗时统计；
# - 不负责画图和保存文件。
# ==========================================================

from __future__ import annotations

from typing import Any

import numpy as np

from src.inference_analysis.inference import InferenceResult


# ==========================================================
# 一、基础指标
# ==========================================================

def mse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算整体 MSE。

    输入：
    - pred   : 任意相同形状数组
    - target : 任意相同形状数组

    返回：
    - 标量 float
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={pred.shape}, target={target.shape}"
        )

    return float(np.mean((pred - target) ** 2))


def relative_l2_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    计算 batch / 数据集平均 Relative L2。

    约定：
    - 第 0 维是样本维 N
    - 后续维度全部展平后做每个样本的 L2 范数

    计算方式：
        rel_i = ||pred_i - target_i||_2 / (||target_i||_2 + eps)
        return mean_i(rel_i)
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={pred.shape}, target={target.shape}"
        )

    num_samples = pred.shape[0]

    pred_flat = pred.reshape(num_samples, -1)
    target_flat = target.reshape(num_samples, -1)

    diff_norm = np.linalg.norm(pred_flat - target_flat, ord=2, axis=1)
    target_norm = np.linalg.norm(target_flat, ord=2, axis=1)

    rel = diff_norm / (target_norm + eps)
    return float(np.mean(rel))


# ==========================================================
# 二、单样本指标
# ==========================================================

def per_sample_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    计算每个样本的 MSE。

    输入：
    - pred   : [N, ...]
    - target : [N, ...]

    返回：
    - shape = [N]
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={pred.shape}, target={target.shape}"
        )

    num_samples = pred.shape[0]
    diff2 = (pred - target) ** 2
    return diff2.reshape(num_samples, -1).mean(axis=1)


def per_sample_relative_l2(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    计算每个样本的 Relative L2。

    输入：
    - pred   : [N, ...]
    - target : [N, ...]

    返回：
    - shape = [N]
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={pred.shape}, target={target.shape}"
        )

    num_samples = pred.shape[0]

    pred_flat = pred.reshape(num_samples, -1)
    target_flat = target.reshape(num_samples, -1)

    diff_norm = np.linalg.norm(pred_flat - target_flat, ord=2, axis=1)
    target_norm = np.linalg.norm(target_flat, ord=2, axis=1)

    return diff_norm / (target_norm + eps)


# ==========================================================
# 三、基于 InferenceResult 的统一接口
# ==========================================================

def compute_inference_metrics(result: InferenceResult) -> dict[str, float]:
    """
    对单模型推理结果计算整体指标。

    返回：
    {
        "mse": ...,
        "relative_l2": ...
    }
    """
    mse = mse_numpy(result.predictions, result.targets)
    rel_l2 = relative_l2_numpy(result.predictions, result.targets)

    return {
        "mse": mse,
        "relative_l2": rel_l2,
    }


def compute_inference_metrics_with_details(result: InferenceResult) -> dict[str, Any]:
    """
    对推理结果计算更详细的指标。

    返回：
    {
        "mse": ...,
        "relative_l2": ...,
        "per_sample_mse_mean": ...,
        "per_sample_mse_std": ...,
        "per_sample_relative_l2_mean": ...,
        "per_sample_relative_l2_std": ...,
        "best_sample_index_by_mse": ...,
        "worst_sample_index_by_mse": ...,
    }
    """
    sample_mse = per_sample_mse(result.predictions, result.targets)
    sample_rel = per_sample_relative_l2(result.predictions, result.targets)

    return {
        "mse": float(np.mean((result.predictions - result.targets) ** 2)),
        "relative_l2": float(np.mean(sample_rel)),
        "per_sample_mse_mean": float(np.mean(sample_mse)),
        "per_sample_mse_std": float(np.std(sample_mse)),
        "per_sample_relative_l2_mean": float(np.mean(sample_rel)),
        "per_sample_relative_l2_std": float(np.std(sample_rel)),
        "best_sample_index_by_mse": int(np.argmin(sample_mse)),
        "worst_sample_index_by_mse": int(np.argmax(sample_mse)),
    }