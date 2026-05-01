# ==========================================================
# File: src/models/fno1d/fno1d.py
#
# 功能简介：
# 1. 定义完整的一维 Fourier Neural Operator 模型 fno1d；
# 2. 调用 fno1d/layers.py 中的底层层组件；
# 3. 将输入特征提升到隐空间 width；
# 4. 经过若干个 FNOBlock1d 做算子学习；
# 5. 再投影回目标输出维度。
#
# 依赖关系：
# - 依赖 src/models/fno1d/layers.py
# - 被 src/models/registry.py 调用
# - 被后续训练模块间接调用
#
# 重要说明：
# - 当前这是沿一维网格（通常是 lambda 轴）工作的 FNO；
# - “1D” 指的是底层算子作用的网格维度，不是输出坐标维度；
# - 本文件只负责完整模型定义，不负责训练和数据加载。
# ==========================================================

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fno1d.layers1d import FNOBlock1d


# ==========================================================
# 一、完整 fno1d 模型
# ==========================================================

class fno1d(nn.Module):
    """
    一维 Fourier Neural Operator。

    输入张量形状：
    - x: [B, T, in_dim]

    输出张量形状：
    - y: [B, T, out_dim]

    说明：
    - 这里的 T 通常对应一维离散网格长度，例如 lambda 方向采样点数；
    - in_dim 由输入构造逻辑决定，例如：
        单参数任务可为 [param, lambda] -> in_dim = 2
        双参数任务可为 [param1, param2, lambda] -> in_dim = 3
    - out_dim 通常为 3，对应 x(lambda), y(lambda), z(lambda)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        modes: int,
        depth: int,
        activation: nn.Module | None = None,
    ) -> None:
        """
        参数：
        - in_dim       : 输入特征维度
        - out_dim      : 输出特征维度
        - width        : 隐空间通道宽度
        - modes        : 低频模式数
        - depth        : FNO block 层数
        - activation   : 激活函数，默认使用 GELU
        """
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim 必须 > 0，当前得到：{in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim 必须 > 0，当前得到：{out_dim}")
        if width <= 0:
            raise ValueError(f"width 必须 > 0，当前得到：{width}")
        if modes <= 0:
            raise ValueError(f"modes 必须 > 0，当前得到：{modes}")
        if depth <= 0:
            raise ValueError(f"depth 必须 > 0，当前得到：{depth}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.modes = modes
        self.depth = depth

        self.activation = activation if activation is not None else nn.GELU()

        # 1) 输入提升：把原始输入特征映射到 width 维隐空间
        self.input_proj = nn.Linear(in_dim, width)

        # 2) 若干个 FNO block
        self.blocks = nn.ModuleList(
            [FNOBlock1d(width=width, modes=modes, activation=self.activation) for _ in range(depth)]
        )

        # 3) 输出投影：从隐空间 width 映射到目标输出维度
        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            self.activation,
            nn.Conv1d(width, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        输入：
        - x: [B, T, in_dim]

        返回：
        - y: [B, T, out_dim]
        """
        if x.ndim != 3:
            raise ValueError(f"fno1d 输入必须是 3 维张量 [B,T,in_dim]，当前 shape={tuple(x.shape)}")

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"fno1d 输入最后一维应等于 in_dim={self.in_dim}，"
                f"当前输入 shape={tuple(x.shape)}"
            )

        # --------------------------------------------------
        # 1) 输入投影到隐空间
        # [B, T, in_dim] -> [B, T, width]
        # --------------------------------------------------
        x = self.input_proj(x)

        # --------------------------------------------------
        # 2) 转成 Conv/FNO block 使用的格式
        # [B, T, width] -> [B, width, T]
        # --------------------------------------------------
        x = x.permute(0, 2, 1)

        # --------------------------------------------------
        # 3) 依次通过多个 FNO block
        # --------------------------------------------------
        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------
        # 4) 输出投影
        # [B, width, T] -> [B, out_dim, T]
        # --------------------------------------------------
        x = self.output_proj(x)

        # --------------------------------------------------
        # 5) 转回统一输出格式
        # [B, out_dim, T] -> [B, T, out_dim]
        # --------------------------------------------------
        x = x.permute(0, 2, 1)

        return x


# ==========================================================
# 二、模型构造辅助函数
# ==========================================================

def build_fno1d_model(
    in_dim: int,
    out_dim: int,
    modes: int,
    width: int,
    depth: int,
) -> fno1d:
    """
    统一的 fno1d 构造函数。

    用途：
    - 给 registry.py 调用
    - 给训练入口调用
    - 减少外部直接写类初始化细节

    返回：
    - fno1d 实例
    """
    return fno1d(
        in_dim=in_dim,
        out_dim=out_dim,
        width=width,
        modes=modes,
        depth=depth,
    )