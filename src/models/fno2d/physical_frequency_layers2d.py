"""R3-B1：按物理 lambda 频率插值的二维谱卷积层。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalFrequencySpectralConv2d(nn.Module):
    """保留离散 FFT bins，但以物理频率插值复数谱权重。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, *, delta_lambda: float, anchor_frequencies: torch.Tensor) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes1, modes2) <= 0:
            raise ValueError("Spectral dimensions must be positive.")
        if not float(delta_lambda) > 0.0:
            raise ValueError("delta_lambda must be positive.")
        anchors = torch.as_tensor(anchor_frequencies, dtype=torch.float64).reshape(-1)
        if anchors.numel() != modes2 or anchors.numel() < 2:
            raise ValueError("Anchor count must equal modes2 and be at least two.")
        if not torch.isclose(anchors[0], torch.zeros((), dtype=anchors.dtype)):
            raise ValueError("First physical-frequency anchor must be zero.")
        steps = anchors[1:] - anchors[:-1]
        if torch.any(steps <= 0) or not torch.allclose(steps, steps[0], rtol=1e-12, atol=1e-14):
            raise ValueError("Physical-frequency anchors must be uniformly increasing.")
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.modes1, self.modes2 = int(modes1), int(modes2)
        self.delta_lambda = float(delta_lambda)
        self.register_buffer("anchor_frequencies", anchors, persistent=True)
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights_pos_anchor = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_neg_anchor = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def runtime_frequencies(self, width: int, m2: int, *, delta_lambda: float | None = None) -> torch.Tensor:
        """按 DFT logical period N*delta_lambda 构造运行时 xi_k。"""
        spacing = self.delta_lambda if delta_lambda is None else float(delta_lambda)
        if width <= 0 or m2 <= 0 or spacing <= 0.0:
            raise ValueError("width, m2, and delta_lambda must be positive.")
        return torch.arange(m2, dtype=torch.float64, device=self.anchor_frequencies.device) / (float(width) * spacing)

    def _interpolation_indices(self, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回锚点下标和 Cartesian 线性插值系数，并拒绝越界频率。"""
        values = frequencies.to(dtype=torch.float64, device=self.anchor_frequencies.device)
        maximum = self.anchor_frequencies[-1]
        tolerance = torch.finfo(values.dtype).eps * torch.maximum(torch.ones_like(values), maximum) * 32.0
        if torch.any(values < -tolerance) or torch.any(values > maximum + tolerance):
            raise ValueError("Runtime physical frequency lies outside learned anchor support.")
        step = self.anchor_frequencies[1] - self.anchor_frequencies[0]
        scaled = values / step
        nearest = torch.round(scaled).to(dtype=torch.long).clamp(0, self.modes2 - 1)
        exact = torch.isclose(values, self.anchor_frequencies[nearest], rtol=1e-12, atol=1e-14)
        lower = torch.floor(scaled).to(dtype=torch.long).clamp(0, self.modes2 - 2)
        lower = torch.where(exact, nearest, lower)
        upper = torch.where(exact, nearest, lower + 1)
        lower_frequency = self.anchor_frequencies[lower]
        upper_frequency = self.anchor_frequencies[upper]
        denominator = torch.where(
            upper == lower,
            torch.ones_like(upper_frequency),
            upper_frequency - lower_frequency,
        )
        alpha = (values - lower_frequency) / denominator
        alpha = torch.where(exact, torch.zeros_like(alpha), alpha)
        return lower, upper, alpha.to(dtype=torch.float32)

    def interpolated_weights(self, branch: str, frequencies: torch.Tensor, *, m1: int | None = None) -> torch.Tensor:
        """以同一实系数对复数实部和虚部做 Cartesian 线性插值。"""
        if branch == "pos":
            anchors = self.weights_pos_anchor
        elif branch == "neg":
            anchors = self.weights_neg_anchor
        else:
            raise ValueError("branch must be 'pos' or 'neg'.")
        lower, upper, alpha = self._interpolation_indices(frequencies)
        selected_m1 = self.modes1 if m1 is None else int(m1)
        if selected_m1 <= 0 or selected_m1 > self.modes1:
            raise ValueError("m1 is outside configured anchor-Q support.")
        left = anchors[:, :, :selected_m1, lower]
        right = anchors[:, :, :selected_m1, upper]
        return left * (1.0 - alpha)[None, None, None, :] + right * alpha[None, None, None, :]

    @staticmethod
    def compl_mul2d(input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """保持 baseline SpectralConv2d 的复数通道乘法语义。"""
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def validate_runtime_support(self, width: int, *, delta_lambda: float | None = None) -> torch.Tensor:
        """显式验证固定离散 retained bins 的物理频率均落在锚点支持内。"""
        m2 = min(self.modes2, width // 2 + 1)
        frequencies = self.runtime_frequencies(width, m2, delta_lambda=delta_lambda)
        self._interpolation_indices(frequencies)
        return frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行不改变 rfft2/irfft2 约定的 R3-B1 谱分支。"""
        if x.ndim != 4:
            raise ValueError(f"PhysicalFrequencySpectralConv2d requires [B,C,H,W], got {tuple(x.shape)}.")
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batch_size, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, height), min(self.modes2, width // 2 + 1)
        frequencies = self.runtime_frequencies(width, m2).to(device=x.device)
        pos = self.interpolated_weights("pos", frequencies, m1=m1)
        neg = self.interpolated_weights("neg", frequencies, m1=m1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], pos)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], neg)
        return torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))


class PhysicalFrequencyFNOBlock2d(nn.Module):
    """R3 保留 pointwise branch、GELU 和 block 数量的 FNO block。"""

    def __init__(self, width: int, modes1: int, modes2: int, *, delta_lambda: float, anchor_frequencies: torch.Tensor, activation: str = "gelu") -> None:
        super().__init__()
        self.spectral_conv = PhysicalFrequencySpectralConv2d(width, width, modes1, modes2, delta_lambda=delta_lambda, anchor_frequencies=anchor_frequencies)
        self.pointwise_conv = nn.Conv2d(width, width, kernel_size=1)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("Unsupported activation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """合并物理频率谱分支与未改变的逐点分支。"""
        return self.activation(self.spectral_conv(x) + self.pointwise_conv(x))
