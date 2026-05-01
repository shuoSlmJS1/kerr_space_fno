# ==========================================================
# File: src/training/trainer.py
#
# 功能简介：
# 1. 定义完整训练流程；
# 2. 实现单轮训练、验证、测试；
# 3. 管理 best model 选择；
# 4. 保存 checkpoint、训练历史和训练摘要；
# 5. 为 train_model.py 提供统一训练入口。
#
# 依赖关系：
# - 依赖 src/training/fno1d/dataset_loader_1d.py
# - 依赖 src/training/losses.py
# - 依赖 src/training/checkpoint.py
# - 依赖 src/training/optimizer_builder.py
# - 被 scripts/train_model.py 调用
#
# 重要说明：
# - 当前训练流程先服务于一维 FNO；
# - 本文件负责训练逻辑，不负责命令行解析；
# - 模型本身由外层先构造好再传进来。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from time import perf_counter

from src.training.checkpoint import (
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
    save_train_history,
    save_train_summary,
)
from src.training.losses import build_mse_loss, relative_l2_error
from src.training.optimizer_builder import (
    build_optimizer,
    build_scheduler,
    summarize_optimizer_config,
)


# ==========================================================
# 一、训练配置
# ==========================================================

@dataclass
class TrainerConfig:
    """
    训练器配置。

    说明：
    - 这里放训练逻辑真正关心的参数；
    - 命令行参数会在 train_model.py 中解析，再映射到这里。
    """
    task_name: str
    model_name: str
    device: str = "cpu"

    epochs: int = 300
    print_every: int = 1

    optimizer_name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4

    scheduler_name: str = "exponential"
    scheduler_gamma: float = 0.995


# ==========================================================
# 二、单轮 train / eval
# ==========================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    返回：
    - mean_mse
    - mean_rel_l2
    """
    model.train()

    total_mse = 0.0
    total_rel = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        rel = relative_l2_error(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_mse += float(loss.detach().cpu().item()) * batch_size
        total_rel += float(rel.detach().cpu().item()) * batch_size
        total_count += batch_size

    mean_mse = total_mse / total_count
    mean_rel_l2 = total_rel / total_count
    return mean_mse, mean_rel_l2


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    在验证集或测试集上评估一个 epoch。

    返回：
    - mean_mse
    - mean_rel_l2
    """
    model.eval()

    total_mse = 0.0
    total_rel = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        rel = relative_l2_error(pred, y)

        batch_size = x.shape[0]
        total_mse += float(loss.detach().cpu().item()) * batch_size
        total_rel += float(rel.detach().cpu().item()) * batch_size
        total_count += batch_size

    mean_mse = total_mse / total_count
    mean_rel_l2 = total_rel / total_count
    return mean_mse, mean_rel_l2


# ==========================================================
# 三、训练主流程
# ==========================================================

def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainerConfig,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """
    完整训练流程。

    参数：
    - model
    - train_loader / val_loader / test_loader
    - config       : TrainerConfig
    - model_config : 模型结构相关信息（用于保存摘要）

    返回：
    - summary: 训练摘要字典
    """
    device = config.device
    model = model.to(device)

    criterion = build_mse_loss()
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config.optimizer_name,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.scheduler_name,
        scheduler_gamma=config.scheduler_gamma,
    )

    history = {
        "train_mse": [],
        "train_rel_l2": [],
        "val_mse": [],
        "val_rel_l2": [],
        "lr": [],
    }

    best_val_mse = float("inf")
    best_epoch = -1

    train_start_time = perf_counter()

    # ------------------------------------------------------
    # A. 主训练循环
    # ------------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        train_mse, train_rel = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_mse, val_rel = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_mse"].append(train_mse)
        history["train_rel_l2"].append(train_rel)
        history["val_mse"].append(val_mse)
        history["val_rel_l2"].append(val_rel)
        history["lr"].append(current_lr)

        if epoch % config.print_every == 0:
            print(
                f"Epoch [{epoch:03d}/{config.epochs}] | "
                f"lr={current_lr:.6e} | "
                f"train_mse={train_mse:.6e} | "
                f"train_relL2={train_rel:.6e} | "
                f"val_mse={val_mse:.6e} | "
                f"val_relL2={val_rel:.6e}"
            )

        # ---------- best ----------
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                task_name=config.task_name,
                model_name=config.model_name,
                epoch=epoch,
                best_val_mse=best_val_mse,
                config=_build_checkpoint_config_dict(config, model_config),
            )

        # ---------- last ----------
        save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            task_name=config.task_name,
            model_name=config.model_name,
            epoch=epoch,
            best_val_mse=best_val_mse,
            config=_build_checkpoint_config_dict(config, model_config),
        )

        scheduler.step()

    # ------------------------------------------------------
    # B. 加载 best 并做最终 test
    # ------------------------------------------------------
    best_ckpt_path = _get_best_ckpt_path(config)
    checkpoint = load_checkpoint(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_mse, test_rel = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    train_total_seconds = perf_counter() - train_start_time

    # ------------------------------------------------------
    # C. 保存 history / summary
    # ------------------------------------------------------
    save_train_history(
        history=history,
        task_name=config.task_name,
        model_name=config.model_name,
    )

    summary = {
        "task_name": config.task_name,
        "model_name": config.model_name,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_relative_l2": test_rel,
        "train_total_seconds": float(train_total_seconds),
        "trainer_config": {
            "task_name": config.task_name,
            "model_name": config.model_name,
            "device": config.device,
            "epochs": config.epochs,
            "print_every": config.print_every,
        },
        "optimizer_config": summarize_optimizer_config(
            optimizer_name=config.optimizer_name,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scheduler_name=config.scheduler_name,
            scheduler_gamma=config.scheduler_gamma,
        ),
        "model_config": dict(model_config),
        "history_keys": list(history.keys()),
    }

    save_train_summary(
        summary=summary,
        task_name=config.task_name,
        model_name=config.model_name,
    )

    print("-" * 70)
    print(f"Training finished. Best epoch = {best_epoch}, best val MSE = {best_val_mse:.6e}")
    print(f"Test MSE         : {test_mse:.6e}")
    print(f"Test Relative L2 : {test_rel:.6e}")
    print("-" * 70)

    return summary


# ==========================================================
# 四、辅助函数
# ==========================================================

def _build_checkpoint_config_dict(
    trainer_config: TrainerConfig,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """
    将训练配置与模型配置合并成 checkpoint 中保存的 config 字典。
    """
    return {
        "trainer_config": {
            "task_name": trainer_config.task_name,
            "model_name": trainer_config.model_name,
            "device": trainer_config.device,
            "epochs": trainer_config.epochs,
            "print_every": trainer_config.print_every,
            "optimizer_name": trainer_config.optimizer_name,
            "lr": trainer_config.lr,
            "weight_decay": trainer_config.weight_decay,
            "scheduler_name": trainer_config.scheduler_name,
            "scheduler_gamma": trainer_config.scheduler_gamma,
        },
        "model_config": dict(model_config),
    }


def _get_best_ckpt_path(config: TrainerConfig) -> str:
    """
    获取 best checkpoint 路径字符串。

    这里单独写成函数，是为了让 fit_model 主流程更清楚。
    """
    from src.common.paths import get_best_checkpoint_path

    return str(get_best_checkpoint_path(config.task_name, config.model_name))
