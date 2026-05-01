# ==========================================================
# File: src/models/fno1d/layers.py
#
# 功能简介：
# 1. 定义 fno1d 所需的底层层模块；
# 2. 实现一维频域卷积层 SpectralConv1d；
# 3. 实现 FNO 的基础 block：FNOBlock1d；
# 4. 为上层完整模型 fno1d.py 提供可复用组件。
#
# 依赖关系：
# - 被 src/models/fno1d/fno1d.py 调用
# - 被后续训练模块间接调用
#
# 重要说明：
# - 本文件只负责“层”的定义；
# - 不负责完整模型拼接；
# - 不负责训练逻辑；
# - 不负责数据输入输出格式组织。
# ==========================================================

from __future__ import annotations

import torch
import torch.nn as nn


# ==========================================================
# 一、一维频域卷积层
# ==========================================================

class SpectralConv1d(nn.Module):
    """
    一维频域卷积层。

    核心思想：
    1. 先对输入沿最后一维做 rFFT；
    2. 在低频模式上做可学习的复数线性变换；
    3. 再做 irFFT 回到实空间。

    输入形状：
    - x: [B, C_in, T]

    输出形状：
    - out: [B, C_out, T]

    参数：
    - in_channels   : 输入通道数
    - out_channels  : 输出通道数
    - modes         : 保留的低频模式数
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels 必须 > 0，当前得到：{in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels 必须 > 0，当前得到：{out_channels}")
        if modes <= 0:
            raise ValueError(f"modes 必须 > 0，当前得到：{modes}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 权重缩放，避免初始值过大
        scale = 1.0 / (in_channels * out_channels)

        # 复数权重拆成实部和虚部，便于显式管理参数
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def compl_mul1d(self, input_fft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        复数频域乘法。

        输入：
        - input_fft: [B, C_in, M]
        - weight   : [C_in, C_out, M]

        输出：
        - [B, C_out, M]
        """
        return torch.einsum("bim,iom->bom", input_fft, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B, C_in, T]

        返回：
        - out: [B, C_out, T]
        """
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d 输入必须是 3 维张量 [B,C,T]，当前 shape={tuple(x.shape)}")

        batch_size = x.shape[0]
        signal_length = x.shape[-1]

        # 1) 沿最后一维做实数 FFT
        x_ft = torch.fft.rfft(x, dim=-1)   # [B, C_in, T//2 + 1]

        # 2) 输出频域张量初始化为 0
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            signal_length // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # 3) 只保留低频 modes
        effective_modes = min(self.modes, signal_length // 2 + 1)

        weight = torch.complex(
            self.weight_real[:, :, :effective_modes],
            self.weight_imag[:, :, :effective_modes],
        )

        out_ft[:, :, :effective_modes] = self.compl_mul1d(
            x_ft[:, :, :effective_modes],
            weight,
        )

        # 4) 回到实空间
        out = torch.fft.irfft(out_ft, n=signal_length, dim=-1)
        return out


# ==========================================================
# 二、FNO 基础块
# ==========================================================

class FNOBlock1d(nn.Module):
    """
    一维 FNO 基础块。

    结构：
    - 一条频域支路：SpectralConv1d
    - 一条点卷积支路：1x1 Conv1d
    - 两者相加后经过激活函数

    输入输出形状：
    - 输入 : [B, width, T]
    - 输出 : [B, width, T]
    """

    def __init__(self, width: int, modes: int, activation: nn.Module | None = None) -> None:
        super().__init__()

        if width <= 0:
            raise ValueError(f"width 必须 > 0，当前得到：{width}")
        if modes <= 0:
            raise ValueError(f"modes 必须 > 0，当前得到：{modes}")

        self.width = width
        self.modes = modes

        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise_conv = nn.Conv1d(width, width, kernel_size=1)
        self.activation = activation if activation is not None else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B, width, T]

        返回：
        - [B, width, T]
        """
        if x.ndim != 3:
            raise ValueError(f"FNOBlock1d 输入必须是 3 维张量 [B,width,T]，当前 shape={tuple(x.shape)}")

        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return self.activation(x1 + x2)