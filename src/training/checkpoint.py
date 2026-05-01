# ==========================================================
# File: src/training/checkpoint.py
#
# 功能简介：
# 1. 统一保存训练过程中的 checkpoint；
# 2. 保存最佳模型权重 best_model.pt；
# 3. 保存最后一轮模型权重 last_model.pt；
# 4. 保存训练历史 train_history.json；
# 5. 保存训练摘要 train_summary.json。
#
# 依赖关系：
# - 依赖 src/common/io_utils.py
# - 依赖 src/common/paths.py
# - 被 src/training/trainer.py 调用
#
# 重要说明：
# - 所有文件按统一路径保存；
# - 同一路径写入时默认覆盖旧文件；
# - 本文件只负责保存与读取，不负责训练循环本身。
# ==========================================================

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.common.io_utils import save_json
from src.common.paths import (
    ensure_model_output_dirs,
    get_best_checkpoint_path,
    get_last_checkpoint_path,
    get_train_history_json_path,
    get_train_summary_json_path,
)


# ==========================================================
# 一、保存模型权重
# ==========================================================

def save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_name: str,
    model_name: str,
    epoch: int,
    best_val_mse: float,
    config: dict[str, Any],
) -> str:
    """
    保存最佳模型 checkpoint。

    保存内容：
    - epoch
    - model_state_dict
    - optimizer_state_dict
    - best_val_mse
    - config
    """
    ensure_model_output_dirs(task_name, model_name)
    ckpt_path = get_best_checkpoint_path(task_name, model_name)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mse": float(best_val_mse),
            "config": config,
        },
        ckpt_path,
    )
    return str(ckpt_path)


def save_last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_name: str,
    model_name: str,
    epoch: int,
    best_val_mse: float,
    config: dict[str, Any],
) -> str:
    """
    保存最后一轮模型 checkpoint。
    """
    ensure_model_output_dirs(task_name, model_name)
    ckpt_path = get_last_checkpoint_path(task_name, model_name)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mse": float(best_val_mse),
            "config": config,
        },
        ckpt_path,
    )
    return str(ckpt_path)


# ==========================================================
# 二、读取模型权重
# ==========================================================

def load_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    读取 checkpoint 文件。
    """
    return torch.load(checkpoint_path, map_location=map_location)


# ==========================================================
# 三、保存训练日志
# ==========================================================

def save_train_history(
    history: dict[str, Any],
    task_name: str,
    model_name: str,
) -> str:
    """
    保存训练历史。

    history 通常包括：
    - train_mse
    - train_rel_l2
    - val_mse
    - val_rel_l2
    - lr
    """
    ensure_model_output_dirs(task_name, model_name)
    path = get_train_history_json_path(task_name, model_name)
    save_json(history, path)
    return str(path)


def save_train_summary(
    summary: dict[str, Any],
    task_name: str,
    model_name: str,
) -> str:
    """
    保存训练摘要。
    """
    ensure_model_output_dirs(task_name, model_name)
    path = get_train_summary_json_path(task_name, model_name)
    save_json(summary, path)
    return str(path)