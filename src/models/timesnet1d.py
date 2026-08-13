from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional


DEFAULT_INCEPTION_KERNELS: tuple[int, ...] = (1, 3, 5)


def _validate_positive_integer(value: int, name: str) -> int:
    """验证正整数配置，且不接受 bool。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中 requires_grad 参数的总数。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def select_dominant_frequencies(
    x: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按时间维 FFT 选择共享频率，并返回每样本聚合权重。"""
    if x.ndim != 3:
        raise ValueError("x must have shape [B,T,C].")
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one batch sample.")
    sequence_length = int(x.shape[1])
    if sequence_length < 2:
        raise ValueError("TimesNet requires a sequence length of at least 2.")
    top_k = _validate_positive_integer(top_k, "top_k")
    selectable_count = sequence_length // 2
    if top_k > selectable_count:
        raise ValueError(
            "top_k exceeds the number of selectable nonzero rFFT frequency bins."
        )

    spectrum = torch.fft.rfft(x, dim=1)
    amplitude = spectrum.abs().mean(dim=-1)
    frequency_score = amplitude.mean(dim=0)
    # 从非零候选区间选择，避免修改参与 autograd 的张量。
    _, nonzero_indices = torch.topk(frequency_score[1:], k=top_k, dim=0)
    frequency_indices = nonzero_indices + 1
    periods = torch.div(
        sequence_length,
        frequency_indices,
        rounding_mode="floor",
    )
    period_weight = torch.softmax(amplitude[:, frequency_indices], dim=-1)
    return frequency_indices, periods, period_weight


class InceptionBlockV1(nn.Module):
    """使用并行奇数二维卷积核的 TimesNet 风格 Inception 块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = DEFAULT_INCEPTION_KERNELS,
    ) -> None:
        super().__init__()
        in_channels = _validate_positive_integer(in_channels, "in_channels")
        out_channels = _validate_positive_integer(out_channels, "out_channels")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one value.")
        self.kernel_sizes = tuple(
            _validate_positive_integer(kernel_size, "kernel_size")
            for kernel_size in kernel_sizes
        )
        if any(kernel_size % 2 == 0 for kernel_size in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain only odd values.")
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in self.kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """平均各并行卷积分支，并保持二维空间形状。"""
        if x.ndim != 4:
            raise ValueError("InceptionBlockV1 input must have shape [B,C,H,W].")
        branch_outputs = [convolution(x) for convolution in self.convolutions]
        return torch.stack(branch_outputs, dim=0).mean(dim=0)


class TimesBlock(nn.Module):
    """按共享 FFT 频率构造二维周期分支的 canonical TimesNet 块。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int,
        kernel_sizes: Sequence[int] = DEFAULT_INCEPTION_KERNELS,
    ) -> None:
        super().__init__()
        self.d_model = _validate_positive_integer(d_model, "d_model")
        self.d_ff = _validate_positive_integer(d_ff, "d_ff")
        self.top_k = _validate_positive_integer(top_k, "top_k")
        self.kernel_sizes = tuple(kernel_sizes)
        self.inception_in = InceptionBlockV1(
            self.d_model,
            self.d_ff,
            self.kernel_sizes,
        )
        self.activation = nn.GELU()
        self.inception_out = InceptionBlockV1(
            self.d_ff,
            self.d_model,
            self.kernel_sizes,
        )

    def frequency_diagnostics(self, x: torch.Tensor) -> dict[str, list[int]]:
        """返回当前输入的频率和整数周期诊断，不改变模型状态。"""
        frequency_indices, periods, _ = select_dominant_frequencies(x, self.top_k)
        return {
            "selected_frequency_indices": [int(value) for value in frequency_indices],
            "selected_periods": [int(value) for value in periods],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行频率选择、二维周期处理、加权聚合与残差连接。"""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                "TimesBlock input must have shape [B,T,d_model]; "
                f"received shape={tuple(x.shape)}."
            )
        sequence_length = int(x.shape[1])
        frequency_indices, periods, period_weight = select_dominant_frequencies(
            x,
            self.top_k,
        )
        branch_outputs: list[torch.Tensor] = []
        for period in periods.tolist():
            padded_length = (
                (sequence_length + period - 1) // period
            ) * period
            if padded_length == sequence_length:
                padded = x
            else:
                padded = functional.pad(x, (0, 0, 0, padded_length - sequence_length))
            height = padded_length // period
            branch = padded.reshape(x.shape[0], height, period, self.d_model)
            branch = branch.permute(0, 3, 1, 2).contiguous()
            branch = self.inception_in(branch)
            branch = self.activation(branch)
            branch = self.inception_out(branch)
            branch = branch.permute(0, 2, 3, 1).reshape(
                x.shape[0], padded_length, self.d_model
            )
            branch_outputs.append(branch[:, :sequence_length, :])

        stacked = torch.stack(branch_outputs, dim=-1)
        weighted = (stacked * period_weight[:, None, None, :]).sum(dim=-1)
        return weighted + x


class TimesNetReconstruction1D(nn.Module):
    """用于稀疏轨道完整重建的 canonical TimesNet 一维输入模型。"""

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 3,
        d_model: int = 80,
        d_ff: int = 96,
        num_blocks: int = 2,
        top_k: int = 2,
        kernel_sizes: Sequence[int] = DEFAULT_INCEPTION_KERNELS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = _validate_positive_integer(in_dim, "in_dim")
        self.out_dim = _validate_positive_integer(out_dim, "out_dim")
        self.d_model = _validate_positive_integer(d_model, "d_model")
        self.d_ff = _validate_positive_integer(d_ff, "d_ff")
        self.num_blocks = _validate_positive_integer(num_blocks, "num_blocks")
        self.top_k = _validate_positive_integer(top_k, "top_k")
        self.kernel_sizes = tuple(kernel_sizes)
        if self.kernel_sizes != DEFAULT_INCEPTION_KERNELS:
            raise ValueError("This baseline requires kernel_sizes=(1, 3, 5).")
        if float(dropout) != 0.0:
            raise ValueError("This baseline requires dropout=0.0.")
        self.dropout = float(dropout)
        self.input_projection = nn.Linear(self.in_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    top_k=self.top_k,
                    kernel_sizes=self.kernel_sizes,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.feature_normalizations = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.num_blocks)]
        )
        self.output_projection = nn.Linear(self.d_model, self.out_dim)

    def architecture_metadata(self) -> dict[str, object]:
        """返回可写入实验配置的固定架构与频率机制元数据。"""
        return {
            "family": "timesnet1d",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_times_blocks": len(self.blocks),
            "top_k": self.top_k,
            "inception_kernel_sizes": list(self.kernel_sizes),
            "activation": "GELU",
            "dropout": self.dropout,
            "layer_norm": "feature_dimension_only",
            "fft_axis": "time_dimension_dim_1",
            "zero_frequency": "excluded",
            "frequency_selection": "batch_shared_top_k",
            "aggregation_weighting": "per_sample_spectral_amplitude_softmax",
            "period_conversion": "T // frequency_index",
            "latent_padding": "right_zero_padding_then_crop",
            "trainable_parameter_count": count_trainable_parameters(self),
        }

    @torch.no_grad()
    def inspect_frequency_diagnostics(self, x: torch.Tensor) -> list[dict[str, list[int]]]:
        """返回每个 TimesBlock 的频率诊断，且不影响后续 forward 行为。"""
        self._validate_input(x)
        hidden = self.input_projection(x)
        diagnostics: list[dict[str, list[int]]] = []
        for block, normalization in zip(self.blocks, self.feature_normalizations):
            diagnostics.append(block.frequency_diagnostics(hidden))
            hidden = normalization(block(hidden))
        return diagnostics

    def _validate_input(self, x: torch.Tensor) -> None:
        """验证固定五通道重建输入的形状与时间长度。"""
        if x.ndim != 3:
            raise ValueError(
                "TimesNetReconstruction1D input must have shape [B,T,in_dim]; "
                f"received shape={tuple(x.shape)}."
            )
        if x.shape[0] == 0:
            raise ValueError("TimesNetReconstruction1D requires a non-empty batch.")
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"TimesNetReconstruction1D expected in_dim={self.in_dim}; "
                f"received shape={tuple(x.shape)}."
            )
        if x.shape[1] < 2:
            raise ValueError("TimesNetReconstruction1D requires at least two time steps.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [B,T,5] 映射为完整轨道的 [B,T,3] 预测。"""
        self._validate_input(x)
        hidden = self.input_projection(x)
        for block, normalization in zip(self.blocks, self.feature_normalizations):
            hidden = normalization(block(hidden))
        return self.output_projection(hidden)


def build_timesnet1d_model(
    in_dim: int = 5,
    out_dim: int = 3,
    d_model: int = 80,
    d_ff: int = 96,
    num_blocks: int = 2,
    top_k: int = 2,
) -> TimesNetReconstruction1D:
    """构造正式配置或显式缩小后的本地 smoke TimesNet 模型。"""
    return TimesNetReconstruction1D(
        in_dim=in_dim,
        out_dim=out_dim,
        d_model=d_model,
        d_ff=d_ff,
        num_blocks=num_blocks,
        top_k=top_k,
        kernel_sizes=DEFAULT_INCEPTION_KERNELS,
        dropout=0.0,
    )
