# ==========================================================
# File: src/training/fno1d/input_builder_1d.py
#
# 功能简介：
# 1. 将 dataset.npz 中保存的原始参数输入 X 转换为模型真正使用的输入张量；
# 2. 当前主要服务于一维 FNO（FNO1d）；
# 3. 支持单参数、双参数以及后续更多参数输入的统一广播构造；
# 4. 将 lambda_grid 作为额外输入通道拼接到模型输入中。
#
# 依赖关系：
# - 被 src/training/fno1d/dataset_loader_1d.py 调用
# - 被后续训练模块间接调用
#
# 重要说明：
# - dataset.npz 中的 X 只是“样本参数值”，不是最终模型输入；
# - 本文件负责把 X 扩展成 [B, T, C] 形式；
# - Y 不在这里构造，Y 由 dataset_loader.py 一并读取。
# ==========================================================

# [B, T, C] 模型的输入张量
# B：batch size，也就是当前这一小批次里有多少个样本
# T：轨道长度，也就是 λ 方向上的采样点数
# C：输入通道数，也就是每个 λ 点喂给模型多少个输入特征

from __future__ import annotations

import numpy as np
import torch


# ==========================================================
# 一、基础输入构造
# ==========================================================

def build_fno1d_input_array(
    x_params: np.ndarray,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """
    将原始参数输入数组构造成 FNO1d 使用的输入张量。

    参数：
    - x_params:
        shape = [N, K]
        其中：
        - N 是样本数
        - K 是变化参数个数
        例如：
        - 单参数任务：K = 1
        - 双参数任务：K = 2

    - lambda_grid:
        shape = [T]
        轨道采样网格

    返回：
    - x_model:
        shape = [N, T, K+1]

    说明：
    - 前 K 个通道：每个参数沿 T 维广播
    - 最后 1 个通道：lambda_grid
    """
    if x_params.ndim != 2:
        raise ValueError(
            f"x_params 必须是二维数组 [N,K]，当前 shape={x_params.shape}"
        )

    if lambda_grid.ndim != 1:
        raise ValueError(
            f"lambda_grid 必须是一维数组 [T]，当前 shape={lambda_grid.shape}"
        )

    num_samples, num_params = x_params.shape
    num_steps = lambda_grid.shape[0]

    # ------------------------------------------------------
    # 1) 将参数沿时间/轨道维广播
    # [N,K] -> [N,1,K] -> [N,T,K]
    # ------------------------------------------------------
    param_broadcast = np.broadcast_to(
        x_params[:, None, :],
        (num_samples, num_steps, num_params),
    ).astype(np.float32)

    # ------------------------------------------------------
    # 2) 将 lambda_grid 广播到 batch 维
    # [T] -> [1,T,1] -> [N,T,1]
    # ------------------------------------------------------
    lambda_channel = np.broadcast_to(
        lambda_grid[None, :, None],
        (num_samples, num_steps, 1),
    ).astype(np.float32)

    # ------------------------------------------------------
    # 3) 拼接得到最终输入
    # [N,T,K] + [N,T,1] -> [N,T,K+1]
    # ------------------------------------------------------
    x_model = np.concatenate([param_broadcast, lambda_channel], axis=-1)
    return x_model


# ==========================================================
# 二、PyTorch 张量版本
# ==========================================================

def build_fno1d_input_tensor(
    x_params: torch.Tensor,
    lambda_grid: torch.Tensor,
) -> torch.Tensor:
    """
    将原始参数输入张量构造成 FNO1d 使用的输入张量。

    参数：
    - x_params:
        shape = [B, K]

    - lambda_grid:
        shape = [T]

    返回：
    - x_model:
        shape = [B, T, K+1]

    说明：
    - 这是 build_fno1d_input_array 的 torch 版本；
    - 便于后续如果想在 Dataset 内部直接构造 torch 张量时使用。
    """
    if x_params.ndim != 2:
        raise ValueError(
            f"x_params 必须是二维张量 [B,K]，当前 shape={tuple(x_params.shape)}"
        )

    if lambda_grid.ndim != 1:
        raise ValueError(
            f"lambda_grid 必须是一维张量 [T]，当前 shape={tuple(lambda_grid.shape)}"
        )

    batch_size, num_params = x_params.shape
    num_steps = lambda_grid.shape[0]

    param_broadcast = x_params[:, None, :].expand(batch_size, num_steps, num_params)
    lambda_channel = lambda_grid[None, :, None].expand(batch_size, num_steps, 1)

    x_model = torch.cat([param_broadcast, lambda_channel], dim=-1)
    return x_model


# ==========================================================
# 三、辅助查询函数
# ==========================================================

def infer_fno1d_input_dim(num_vary_params: int) -> int:
    """
    根据变化参数个数，推断 FNO1d 的输入维度。

    规则：
    - 输入维度 = 变化参数个数 + 1
    - 额外的 1 对应 lambda_grid 通道

    例如：
    - 单参数任务：1 + 1 = 2
    - 双参数任务：2 + 1 = 3
    """
    if num_vary_params <= 0:
        raise ValueError(f"num_vary_params 必须 > 0，当前得到：{num_vary_params}")

    return num_vary_params + 1


def describe_input_layout(vary_params_order: list[str]) -> list[str]:
    """
    返回模型输入通道的语义说明。

    例如：
    - vary_params_order = ["a"]
      -> ["a", "lambda"]

    - vary_params_order = ["a", "Q"]
      -> ["a", "Q", "lambda"]
    """
    return list(vary_params_order) + ["lambda"]