# ==========================================================
# File: src/training/losses.py
#
# 功能简介：
# 1. 定义训练与验证使用的损失函数；
# 2. 定义常用误差指标；
# 3. 当前主要提供：
#    - MSE loss
#    - Relative L2 error
# 4. 为 trainer.py 提供统一接口。
#
# 依赖关系：
# - 被 src/training/trainer.py 调用
# - 被后续分析模块间接复用
#
# 重要说明：
# - MSE 主要作为优化目标；
# - Relative L2 主要作为更直观的误差评估指标；
# - 本文件不负责训练循环本身。
# ==========================================================

from __future__ import annotations

import torch
import torch.nn as nn


# ==========================================================
# 一、基础损失构造
# ==========================================================

def build_mse_loss() -> nn.Module:
    """
    返回标准均方误差损失函数。
    """
    return nn.MSELoss()


# ==========================================================
# 二、误差指标
# ==========================================================

def _reshape_trajectory_samples(array: torch.Tensor) -> torch.Tensor:
    """
    将轨道张量整理为 [N_traj, T * C]。

    - 1D: [B, T, C] -> B 条轨道；
    - 2D: [B_field, N_param, T, C] -> B_field * N_param 条轨道。
    """
    if array.ndim >= 3:
        return array.reshape(-1, array.shape[-2] * array.shape[-1])

    if array.ndim == 2:
        return array.reshape(array.shape[0], -1)

    raise ValueError(f"Relative L2 至少需要 2 维张量，当前 shape={tuple(array.shape)}")


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    计算 batch 平均 Relative L2 误差。

    输入：
    - pred   : [B, ...]
    - target : [B, ...]

    返回：
    - 标量张量，表示当前 batch 的平均相对 L2 误差

    计算方式：
        rel_i = ||pred_i - target_i||_2 / (||target_i||_2 + eps)
        return mean_i(rel_i)

    说明：
    - 这里默认 batch 维是第 0 维；
    - 后续所有其他维度都展平后参与 L2 范数计算。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    pred_flat = _reshape_trajectory_samples(pred)
    target_flat = _reshape_trajectory_samples(target)

    diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
    target_norm = torch.norm(target_flat, p=2, dim=1)

    rel = diff_norm / (target_norm + eps)
    return rel.mean()


def mse_value(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    直接计算当前 batch 的 MSE 标量值。

    说明：
    - 这个函数和 nn.MSELoss() 功能相近；
    - 提供这个函数主要是为了接口风格统一。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的形状必须一致，当前得到：pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    return torch.mean((pred - target) ** 2)


# ==========================================================
# 三、打包评估
# ==========================================================

def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    计算当前 batch 的常用指标，并转成 Python float。

    返回：
    {
        "mse": ...,
        "relative_l2": ...
    }
    """
    mse = mse_value(pred, target)
    rel_l2 = relative_l2_error(pred, target)

    return {
        "mse": float(mse.detach().cpu().item()),
        "relative_l2": float(rel_l2.detach().cpu().item()),
    }