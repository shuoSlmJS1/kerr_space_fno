# ==========================================================
# File: src/models/registry.py
#
# 功能简介：
# 1. 统一管理模型注册表；
# 2. 将模型类型字符串映射到具体的模型构造函数；
# 3. 为训练入口脚本提供统一的 build_model 接口；
# 4. 当前先支持 fno1d，后续可扩展到 cnn1d / mlp / fno2d 等。
#
# 依赖关系：
# - 依赖 src/models/fno1d/fno1d.py
# - 被 scripts/train_model.py 调用
# - 被训练模块间接调用
#
# 重要说明：
# - 本文件不负责训练；
# - 不负责模型参数命名；
# - 只负责“给定 model_type 和参数，构造对应模型”。
# ==========================================================

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from src.models.fno1d.fno1d import build_fno1d_model


# ==========================================================
# 一、模型构造函数注册表
# ==========================================================

MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "fno1d": build_fno1d_model,
}


# ==========================================================
# 二、查询接口
# ==========================================================

def get_supported_model_types() -> list[str]:
    """
    返回当前支持的模型类型列表。

    例如：
    - ["fno1d"]
    """
    return sorted(MODEL_BUILDERS.keys())


def get_model_help_text() -> str:
    """
    返回模型相关 help 文本。

    用途：
    - 给 scripts/train_model.py 的 --help 或额外帮助输出使用
    """
    lines = [
        "Supported model types:",
        "  " + ", ".join(get_supported_model_types()),
        "",
        "Current model parameter requirements:",
        "",
        "1) fno1d:",
        "   required args:",
        "     --model fno1d",
        "     --modes <int>",
        "     --width <int>",
        "     --depth <int>",
        "",
        "Example:",
        "   --model fno1d --modes 32 --width 64 --depth 4",
    ]
    return "\n".join(lines)


# ==========================================================
# 三、参数检查
# ==========================================================

def validate_model_build_kwargs(model_type: str, kwargs: dict[str, Any]) -> None:
    """
    检查某个模型构造所需参数是否齐全。

    当前规则：
    - fno1d 需要：
        modes, width, depth, in_dim, out_dim
    """
    model_type = model_type.lower()

    if model_type == "fno1d":
        required = ["modes", "width", "depth", "in_dim", "out_dim"]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(
                f"构造 fno1d 缺少必要参数：{missing}，当前 kwargs={kwargs}"
            )
        return

    raise ValueError(
        f"不支持的 model_type={model_type!r}，"
        f"当前支持：{get_supported_model_types()}"
    )


# ==========================================================
# 四、统一构造入口
# ==========================================================

def build_model(model_type: str, **kwargs: Any) -> nn.Module:
    """
    统一模型构造入口。

    参数：
    - model_type: 模型类型字符串，例如 "fno1d"
    - kwargs: 模型构造参数

    返回：
    - 对应的 nn.Module

    示例：
        model = build_model(
            model_type="fno1d",
            in_dim=2,
            out_dim=3,
            modes=32,
            width=64,
            depth=4,
        )
    """
    model_type = model_type.lower()

    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"不支持的 model_type={model_type!r}，"
            f"当前支持：{get_supported_model_types()}"
        )

    validate_model_build_kwargs(model_type, kwargs)

    builder = MODEL_BUILDERS[model_type]
    return builder(**kwargs)