# ==========================================================
# File: src/models/registry_2d.py
#
# 功能简介：
# 1. 统一管理二维模型的构造；
# 2. 当前支持 fno2d；
# 3. 后续可以继续加入 cnn2d / unet2d / fno2d_large 等模型；
# 4. 供 scripts/train_model_2d.py 调用；
# 5. 使 2D 训练入口保持通用，而不是直接绑定某一个模型文件。
#
# 设计原则：
# - train_model_2d.py 不直接 import 具体模型；
# - train_model_2d.py 只调用 build_model_2d；
# - 新增模型时只修改 registry_2d.py。
# ==========================================================

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.fno2d.fno2d import (
    build_fno2d_model,
    summarize_fno2d_config,
)


# ==========================================================
# 一、支持的二维模型类型
# ==========================================================

SUPPORTED_2D_MODELS = {
    "fno2d",
}


def get_supported_2d_models() -> list[str]:
    """
    返回当前支持的二维模型类型列表。
    """
    return sorted(SUPPORTED_2D_MODELS)


def validate_2d_model_type(model_type: str) -> str:
    """
    检查二维模型类型是否合法。

    参数：
    - model_type:
        例如 "fno2d"

    返回：
    - 标准化后的 model_type
    """
    model_type = str(model_type).lower()

    if model_type not in SUPPORTED_2D_MODELS:
        raise ValueError(
            f"不支持的 2D model_type={model_type!r}。"
            f"当前支持：{get_supported_2d_models()}"
        )

    return model_type


# ==========================================================
# 二、二维模型名生成
# ==========================================================

def build_model_name_2d(
    model_type: str,
    modes1: int,
    modes2: int,
    width: int,
    depth: int,
) -> str:
    """
    构造二维模型名。

    说明：
    - modes1 表示第一个网格方向的 Fourier modes；
    - modes2 表示第二个网格方向的 Fourier modes；
    - 对当前 (p, lambda) 任务：
        modes1 = 参数方向 modes
        modes2 = lambda 方向 modes

    示例：
        fno2d_m1_16_m2_32_w64_d4
    """
    model_type = validate_2d_model_type(model_type)

    return (
        f"{model_type}"
        f"_m1_{int(modes1)}"
        f"_m2_{int(modes2)}"
        f"_w{int(width)}"
        f"_d{int(depth)}"
    )


# ==========================================================
# 三、二维模型构造
# ==========================================================

def build_model_2d(
    model_type: str,
    in_dim: int,
    out_dim: int,
    modes1: int,
    modes2: int,
    width: int,
    depth: int,
    hidden_dim: int = 128,
    activation: str = "gelu",
) -> nn.Module:
    """
    根据 model_type 构造二维模型。

    当前支持：
    - fno2d

    后续扩展：
    - cnn2d
    - unet2d
    - fno2d_large
    """
    model_type = validate_2d_model_type(model_type)

    if model_type == "fno2d":
        return build_fno2d_model(
            in_dim=in_dim,
            out_dim=out_dim,
            modes1=modes1,
            modes2=modes2,
            width=width,
            depth=depth,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    raise RuntimeError(f"未处理的 2D model_type={model_type!r}")


# ==========================================================
# 四、二维模型配置摘要
# ==========================================================

def summarize_model_config_2d(
    model_type: str,
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
    生成二维模型配置摘要。

    用于：
    - checkpoint config
    - train_summary.json
    - 后续结果追踪
    """
    model_type = validate_2d_model_type(model_type)

    if model_type == "fno2d":
        return summarize_fno2d_config(
            in_dim=in_dim,
            out_dim=out_dim,
            modes1=modes1,
            modes2=modes2,
            width=width,
            depth=depth,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    raise RuntimeError(f"未处理的 2D model_type={model_type!r}")


# ==========================================================
# 五、帮助信息
# ==========================================================

def get_model_help_text_2d() -> str:
    """
    返回二维模型帮助信息。
    """
    lines = [
        "Supported 2D models:",
        "",
        "1. fno2d",
        "   用途：二维 Fourier Neural Operator。",
        "   当前主要用于 (p, lambda) -> (x,y,z)。",
        "   参数：",
        "     --model fno2d",
        "     --modes1 / --modes2 或入口脚本中的 --modes-param / --modes-lambda",
        "     --width",
        "     --depth",
        "",
        "后续可扩展：cnn2d, unet2d, fno2d_large 等。",
    ]

    return "\n".join(lines)