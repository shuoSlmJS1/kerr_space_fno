# ==========================================================
# File: src/models/fno2d/layers2d.py
#
# 功能简介：
# 1. 定义 FNO2d 使用的二维谱卷积层 SpectralConv2d；
# 2. 定义 FNO2d 的基本模块 FNOBlock2d；
# 3. 二维算子维度为 (Q, lambda)，即在两个网格方向上做 FFT；
# 4. 被 src/models/fno2d/fno2d.py 调用。
#
# 重要张量约定：
# - 谱卷积输入: [B, C, H, W]
# - H 表示参数方向，例如 Q-grid
# - W 表示轨道方向，例如 lambda-grid
# ==========================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 一、二维谱卷积层
# ==========================================================

class SpectralConv2d(nn.Module):
    """
    二维谱卷积层。

    数学思想：
    1. 对输入在两个空间方向上做二维 FFT；
    2. 只保留低频 modes1 × modes2；
    3. 在频域中做可学习的复数线性变换；
    4. 再通过 inverse FFT 回到物理空间。

    输入：
    - x: [B, in_channels, H, W]

    输出：
    - y: [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels 必须 > 0，当前为 {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels 必须 > 0，当前为 {out_channels}")
        if modes1 <= 0:
            raise ValueError(f"modes1 必须 > 0，当前为 {modes1}")
        if modes2 <= 0:
            raise ValueError(f"modes2 必须 > 0，当前为 {modes2}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        # 复数权重缩放因子，避免初始化过大
        scale = 1.0 / (in_channels * out_channels)

        # 正频率部分权重
        self.weights_pos = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

        # 负频率部分权重
        # 对 rfft2 来说，最后一个维度只保留非负频率；
        # 但倒数第二个维度仍然有正负频率，所以这里单独处理底部 modes1。
        self.weights_neg = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(
        self,
        input_ft: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        复数频域乘法。

        input_ft:
            [B, in_channels, modes1, modes2]

        weights:
            [in_channels, out_channels, modes1, modes2]

        返回：
            [B, out_channels, modes1, modes2]
        """
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B, C, H, W]

        输出：
        - y: [B, C_out, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"SpectralConv2d 输入必须是 [B,C,H,W]，当前 shape={tuple(x.shape)}")

        batch_size, _, height, width = x.shape

        # --------------------------------------------------
        # 1. 二维实数 FFT
        # --------------------------------------------------
        # x_ft: [B, C, H, W//2 + 1]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # --------------------------------------------------
        # 2. 初始化输出频域张量
        # --------------------------------------------------
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # --------------------------------------------------
        # 3. 自动裁剪 modes，避免 modes 大于实际网格
        # --------------------------------------------------
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)

        # 正频率区域
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights_pos[:, :, :m1, :m2],
        )

        # 负频率区域
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights_neg[:, :, :m1, :m2],
        )

        # --------------------------------------------------
        # 4. inverse FFT 回到物理空间
        # --------------------------------------------------
        x = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))

        return x


# ==========================================================
# 二、FNO2d 基本块
# ==========================================================

class FNOBlock2d(nn.Module):
    """
    FNO2d 基本模块。

    结构：
        spectral convolution + pointwise convolution + activation

    输入：
    - x: [B, width, H, W]

    输出：
    - y: [B, width, H, W]
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if width <= 0:
            raise ValueError(f"width 必须 > 0，当前为 {width}")

        self.width = int(width)

        self.spectral_conv = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )

        # 1×1 pointwise convolution，相当于每个网格点上的通道混合
        self.pointwise_conv = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"不支持的 activation={activation!r}，当前支持 gelu/relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B, width, H, W]

        输出：
        - y: [B, width, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"FNOBlock2d 输入必须是 [B,width,H,W]，当前 shape={tuple(x.shape)}")

        spectral_part = self.spectral_conv(x)
        pointwise_part = self.pointwise_conv(x)

        x = spectral_part + pointwise_part
        x = self.activation(x)

        return x