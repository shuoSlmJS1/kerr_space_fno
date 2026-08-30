"""R3-B1：仅替换 lambda 谱权重参数化的 FNO2D。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.models.fno2d.physical_frequency_layers2d import PhysicalFrequencyFNOBlock2d


@dataclass(frozen=True)
class PhysicalFrequencyFNO2dConfig:
    """R3-B1 的可审计模型配置。"""

    in_dim: int = 3
    out_dim: int = 3
    modes1: int = 16
    modes2: int = 32
    width: int = 64
    depth: int = 4
    hidden_dim: int = 128
    activation: str = "gelu"


class PhysicalFrequencyFNO2d(nn.Module):
    """输入 [Q,s,ell]，对 lambda 频率使用锚点插值权重的 FNO2D。"""

    def __init__(self, *, in_dim: int, out_dim: int, modes1: int, modes2: int, width: int, depth: int, hidden_dim: int, activation: str, delta_lambda: float, anchor_frequencies: torch.Tensor) -> None:
        super().__init__()
        if in_dim != 3:
            raise ValueError("R3 requires in_dim=3 for [Q, s, ell].")
        if min(out_dim, modes1, modes2, width, depth, hidden_dim) <= 0:
            raise ValueError("R3 model dimensions must be positive.")
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.modes1, self.modes2, self.width, self.depth = int(modes1), int(modes2), int(width), int(depth)
        self.hidden_dim, self.activation_name, self.delta_lambda = int(hidden_dim), str(activation), float(delta_lambda)
        anchors = torch.as_tensor(anchor_frequencies, dtype=torch.float64).reshape(-1)
        self.register_buffer("anchor_frequencies", anchors, persistent=True)
        self.input_projection = nn.Linear(self.in_dim, self.width)
        self.blocks = nn.ModuleList([PhysicalFrequencyFNOBlock2d(self.width, self.modes1, self.modes2, delta_lambda=self.delta_lambda, anchor_frequencies=anchors, activation=activation) for _ in range(self.depth)])
        self.output_projection = nn.Sequential(nn.Linear(self.width, self.hidden_dim), nn.GELU() if activation == "gelu" else nn.ReLU(), nn.Linear(self.hidden_dim, self.out_dim))

    def validate_runtime_support(self, width: int, delta_lambda: float) -> torch.Tensor:
        """确保正式运行网格的 retained physical frequencies 不越出训练锚点支持。"""
        if not torch.isclose(torch.tensor(float(delta_lambda), dtype=torch.float64), torch.tensor(self.delta_lambda, dtype=torch.float64), rtol=1e-10, atol=1e-12):
            raise ValueError("Runtime delta_lambda differs from the R3 training spacing.")
        return self.blocks[0].spectral_conv.validate_runtime_support(int(width), delta_lambda=float(delta_lambda))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """保持 R2 的点式投影、block 深度和输出头。"""
        if x.ndim != 4 or x.shape[-1] != self.in_dim:
            raise ValueError("R3 model requires [B,H,W,3] [Q,s,ell] input.")
        hidden = self.input_projection(x).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        return self.output_projection(hidden)


def build_physical_frequency_fno2d_model(**config: Any) -> PhysicalFrequencyFNO2d:
    """构造 R3 专用模型，避免修改 baseline registry。"""
    return PhysicalFrequencyFNO2d(**config)


def summarize_physical_frequency_fno2d_config(**config: Any) -> dict[str, Any]:
    """生成 checkpoint 可恢复的 R3 专用模型配置。"""
    result = {key: value for key, value in config.items() if key != "anchor_frequencies"}
    result["model_type"] = "fno2d_physical_frequency"
    result["anchor_frequencies"] = [float(value) for value in torch.as_tensor(config["anchor_frequencies"], dtype=torch.float64).reshape(-1).tolist()]
    return result


def count_parameters(model: nn.Module) -> int:
    """统计 R3 训练参数。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
