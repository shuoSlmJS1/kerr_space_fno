# ==========================================================
# File: src/training/optimizer_builder.py
#
# 功能简介：
# 1. 统一构造训练使用的优化器；
# 2. 统一构造学习率调度器；
# 3. 当前默认支持：
#    - AdamW 优化器
#    - ExponentialLR 学习率调度器
# 4. 为 trainer.py 提供固定接口，减少训练循环中的重复代码。
#
# 依赖关系：
# - 被 src/training/trainer.py 调用
# - 被 scripts/train_model.py 间接调用
#
# 重要说明：
# - 本文件不负责训练循环；
# - 不负责保存 checkpoint；
# - 只负责“如何构造优化器与 scheduler”。
# ==========================================================

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ==========================================================
# 一、优化器构造
# ==========================================================

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    根据给定名称构造优化器。

    当前支持：
    - adamw

    参数：
    - model
    - optimizer_name
    - lr
    - weight_decay
    - kwargs: 预留扩展参数
    """
    optimizer_name = optimizer_name.lower()

    if lr <= 0:
        raise ValueError(f"学习率 lr 必须 > 0，当前得到：{lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay 不能为负，当前得到：{weight_decay}")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"不支持的 optimizer_name={optimizer_name!r}。"
        f"当前支持：['adamw']"
    )


# ==========================================================
# 二、学习率调度器构造
# ==========================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "exponential",
    scheduler_gamma: float = 0.995,
    **kwargs: Any,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    根据给定名称构造学习率调度器。

    当前支持：
    - exponential

    参数：
    - optimizer
    - scheduler_name
    - scheduler_gamma
    - kwargs: 预留扩展参数
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "exponential":
        if not (0.0 < scheduler_gamma <= 1.0):
            raise ValueError(
                f"ExponentialLR 的 gamma 应满足 0 < gamma <= 1，当前得到：{scheduler_gamma}"
            )
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_gamma,
        )

    raise ValueError(
        f"不支持的 scheduler_name={scheduler_name!r}。"
        f"当前支持：['exponential']"
    )


# ==========================================================
# 三、训练配置摘要
# ==========================================================

def summarize_optimizer_config(
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    scheduler_name: str,
    scheduler_gamma: float,
) -> dict[str, Any]:
    """
    将优化器与调度器配置整理成摘要字典。
    """
    return {
        "optimizer_name": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "scheduler_name": scheduler_name,
        "scheduler_gamma": float(scheduler_gamma),
    }