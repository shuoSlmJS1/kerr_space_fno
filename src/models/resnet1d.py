from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


DEFAULT_DILATION_SCHEDULE: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
)


def _validate_positive_integer(value: int, name: str) -> int:
    """验证正整数配置，且不接受 bool。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def calculate_theoretical_receptive_field(
    kernel_size: int,
    dilations: Sequence[int],
    local_dilation: int = 1,
) -> int:
    """计算每个残差块含一层膨胀卷积和一层局部卷积时的理论感受野。"""
    kernel_size = _validate_positive_integer(kernel_size, "kernel_size")
    local_dilation = _validate_positive_integer(local_dilation, "local_dilation")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    if not dilations:
        raise ValueError("dilations must contain at least one value.")

    validated_dilations = tuple(
        _validate_positive_integer(dilation, "dilation")
        for dilation in dilations
    )

    # 所有卷积 stride 均为 1，因此每层对原网格的跳步保持为 1。
    return 1 + (kernel_size - 1) * sum(
        dilation + local_dilation for dilation in validated_dilations
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中 requires_grad 参数的总数。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class DilatedResidualBlock(nn.Module):
    """由一层膨胀卷积和一层局部卷积组成的残差块。"""

    def __init__(
        self,
        width: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        width = _validate_positive_integer(width, "width")
        kernel_size = _validate_positive_integer(kernel_size, "kernel_size")
        dilation = _validate_positive_integer(dilation, "dilation")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.width = width
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dilated_padding_width = ((kernel_size - 1) // 2) * dilation
        self.local_padding_width = (kernel_size - 1) // 2

        self.activation_before_dilated = nn.GELU()
        self.dilated_padding = nn.ReplicationPad1d(self.dilated_padding_width)
        self.dilated_conv = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            stride=1,
        )
        self.activation_before_local = nn.GELU()
        self.local_padding = nn.ReplicationPad1d(self.local_padding_width)
        self.local_conv = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=1,
            padding=0,
            stride=1,
        )
        self.activation_after_residual = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行保持 [B,width,T] 形状不变的残差映射。"""
        residual = x
        x = self.activation_before_dilated(x)
        x = self.dilated_padding(x)
        x = self.dilated_conv(x)
        x = self.activation_before_local(x)
        x = self.local_padding(x)
        x = self.local_conv(x)
        return self.activation_after_residual(x + residual)


class DilatedResNet1D(nn.Module):
    """用于稀疏轨道重建的非循环膨胀一维残差网络。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 92,
        kernel_size: int = 7,
        dilations: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
    ) -> None:
        super().__init__()
        in_dim = _validate_positive_integer(in_dim, "in_dim")
        out_dim = _validate_positive_integer(out_dim, "out_dim")
        width = _validate_positive_integer(width, "width")
        kernel_size = _validate_positive_integer(kernel_size, "kernel_size")
        if kernel_size != 7:
            raise ValueError("This baseline requires kernel_size=7.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if not dilations:
            raise ValueError("dilations must contain at least one value.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.kernel_size = kernel_size
        self.dilations = tuple(
            _validate_positive_integer(dilation, "dilation")
            for dilation in dilations
        )
        self.padding_type = "replicate"
        self.theoretical_receptive_field = calculate_theoretical_receptive_field(
            kernel_size=self.kernel_size,
            dilations=self.dilations,
        )

        self.input_projection = nn.Conv1d(in_dim, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    width=width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.output_head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_dim, kernel_size=1),
        )

    def architecture_metadata(self) -> dict[str, object]:
        """返回可写入实验配置的固定架构元数据。"""
        return {
            "family": "dilated_resnet1d",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "width": self.width,
            "block_count": len(self.blocks),
            "kernel_size": self.kernel_size,
            "dilation_schedule": list(self.dilations),
            "local_dilation": 1,
            "padding_type": self.padding_type,
            "theoretical_receptive_field": self.theoretical_receptive_field,
            "trainable_parameter_count": count_trainable_parameters(self),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [B,T,in_dim] 映射为 [B,T,out_dim]。"""
        if x.ndim != 3:
            raise ValueError(
                "DilatedResNet1D input must have shape [B,T,in_dim]; "
                f"received shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"DilatedResNet1D expected in_dim={self.in_dim}; "
                f"received shape={tuple(x.shape)}."
            )
        if x.shape[1] <= 0:
            raise ValueError("DilatedResNet1D requires at least one time step.")

        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_head(x)
        return x.permute(0, 2, 1)


def build_dilated_resnet1d(
    in_dim: int,
    out_dim: int,
    width: int = 92,
    blocks: int = len(DEFAULT_DILATION_SCHEDULE),
) -> DilatedResNet1D:
    """构造默认正式配置或用于本地 smoke 的前缀残差块配置。"""
    blocks = _validate_positive_integer(blocks, "blocks")
    if blocks > len(DEFAULT_DILATION_SCHEDULE):
        raise ValueError(
            "blocks must not exceed the approved dilation schedule length."
        )
    return DilatedResNet1D(
        in_dim=in_dim,
        out_dim=out_dim,
        width=width,
        kernel_size=7,
        dilations=DEFAULT_DILATION_SCHEDULE[:blocks],
    )
