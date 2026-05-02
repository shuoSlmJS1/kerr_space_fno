# ==========================================================
# File: src/models/fno2d/fno2d.py
#
# 功能简介：
# 1. 定义完整 FNO2d 模型；
# 2. 输入二维网格场 [B,H,W,C]；
# 3. 在 (H,W) 两个方向上做 Fourier operator 学习；
# 4. 输出二维轨道场 [B,H,W,3]；
# 5. 第一版用于 (Q, lambda) -> (x,y,z)。
#
# 张量约定：
# - B: batch size
# - H: 参数方向网格点数，例如 Q-grid
# - W: lambda 方向网格点数
# - C: 输入通道数，通常为 2，即 Q 和 lambda
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.models.fno2d.layers2d import FNOBlock2d


# ==========================================================
# 一、模型配置
# ==========================================================

@dataclass
class FNO2dConfig:
    """
    FNO2d 模型配置。

    参数说明：
    - in_dim:
        输入通道数。
        对第一版 (Q, lambda) 任务，通常 in_dim=2。

    - out_dim:
        输出通道数。
        对 Kerr 轨道 xyz 输出，通常 out_dim=3。

    - modes1:
        第一维 Fourier modes 数量。
        对第一版任务，它对应 Q 方向。

    - modes2:
        第二维 Fourier modes 数量。
        对第一版任务，它对应 lambda 方向。

    - width:
        隐藏通道宽度。

    - depth:
        FNOBlock2d 的层数。

    - hidden_dim:
        输出头中间层宽度。
    """
    in_dim: int = 2
    out_dim: int = 3
    modes1: int = 16
    modes2: int = 32
    width: int = 64
    depth: int = 4
    hidden_dim: int = 128
    activation: str = "gelu"


# ==========================================================
# 二、完整 FNO2d 模型
# ==========================================================

class FNO2d(nn.Module):
    """
    完整 FNO2d 模型。

    输入：
        x: [B,H,W,in_dim]

    输出：
        y: [B,H,W,out_dim]

    第一版物理含义：
        输入每个网格点上的 (Q, lambda)，
        输出该网格点对应的 (x,y,z)。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        modes1: int,
        modes2: int,
        width: int,
        depth: int,
        hidden_dim: int = 128,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim 必须 > 0，当前为 {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim 必须 > 0，当前为 {out_dim}")
        if modes1 <= 0:
            raise ValueError(f"modes1 必须 > 0，当前为 {modes1}")
        if modes2 <= 0:
            raise ValueError(f"modes2 必须 > 0，当前为 {modes2}")
        if width <= 0:
            raise ValueError(f"width 必须 > 0，当前为 {width}")
        if depth <= 0:
            raise ValueError(f"depth 必须 > 0，当前为 {depth}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须 > 0，当前为 {hidden_dim}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.activation_name = activation

        # --------------------------------------------------
        # A. 输入升维
        # --------------------------------------------------
        # 每个网格点上的输入通道 in_dim -> width
        self.input_projection = nn.Linear(
            in_features=self.in_dim,
            out_features=self.width,
        )

        # --------------------------------------------------
        # B. FNO2d blocks
        # --------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    width=self.width,
                    modes1=self.modes1,
                    modes2=self.modes2,
                    activation=activation,
                )
                for _ in range(self.depth)
            ]
        )

        # --------------------------------------------------
        # C. 输出投影
        # --------------------------------------------------
        # 每个网格点上的 hidden width -> hidden_dim -> out_dim
        self.output_projection = nn.Sequential(
            nn.Linear(self.width, self.hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B,H,W,in_dim]

        输出：
        - y: [B,H,W,out_dim]
        """
        if x.ndim != 4:
            raise ValueError(
                f"FNO2d 输入必须是 [B,H,W,C]，当前 shape={tuple(x.shape)}"
            )

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"输入最后一维必须等于 in_dim={self.in_dim}，"
                f"当前 x.shape[-1]={x.shape[-1]}"
            )

        # --------------------------------------------------
        # 1. 输入通道升维
        # --------------------------------------------------
        # [B,H,W,in_dim] -> [B,H,W,width]
        x = self.input_projection(x)

        # --------------------------------------------------
        # 2. 换维度给 Conv2d / FFT 使用
        # --------------------------------------------------
        # [B,H,W,width] -> [B,width,H,W]
        x = x.permute(0, 3, 1, 2).contiguous()

        # --------------------------------------------------
        # 3. 多层 FNO2d block
        # --------------------------------------------------
        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------
        # 4. 换回点通道格式
        # --------------------------------------------------
        # [B,width,H,W] -> [B,H,W,width]
        x = x.permute(0, 2, 3, 1).contiguous()

        # --------------------------------------------------
        # 5. 输出投影
        # --------------------------------------------------
        # [B,H,W,width] -> [B,H,W,out_dim]
        x = self.output_projection(x)

        return x


# ==========================================================
# 三、构造函数
# ==========================================================

def build_fno2d_model(
    in_dim: int,
    out_dim: int,
    modes1: int,
    modes2: int,
    width: int,
    depth: int,
    hidden_dim: int = 128,
    activation: str = "gelu",
) -> FNO2d:
    """
    构造 FNO2d 模型。

    用途：
    - 给 train_model_2d.py 使用；
    - 不依赖 1D registry.py。
    """
    return FNO2d(
        in_dim=in_dim,
        out_dim=out_dim,
        modes1=modes1,
        modes2=modes2,
        width=width,
        depth=depth,
        hidden_dim=hidden_dim,
        activation=activation,
    )


# ==========================================================
# 四、模型信息摘要
# ==========================================================

def summarize_fno2d_config(
    in_dim: int,
    out_dim: int,
    modes1: int,
    modes2: int,
    width: int,
    depth: int,
    hidden_dim: int = 128,
    activation: str = "gelu",
) -> dict[str, Any]:
    """
    整理 FNO2d 模型配置摘要。
    """
    return {
        "model_type": "fno2d",
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "modes1": int(modes1),
        "modes2": int(modes2),
        "width": int(width),
        "depth": int(depth),
        "hidden_dim": int(hidden_dim),
        "activation": activation,
    }


def count_parameters(model: nn.Module) -> int:
    """
    统计模型可训练参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)