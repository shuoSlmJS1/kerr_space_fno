# ==========================================================
# File: src/inference_analysis/inference.py
#
# 功能简介：
# 1. 统一执行模型推理；
# 2. 支持单 batch 推理与整数据集推理；
# 3. 收集模型预测结果与对应真值；
# 4. 为 metrics.py / timing.py / plotting.py 提供标准输入格式。
#
# 依赖关系：
# - 被 scripts/run_analysis.py 调用
# - 被 comparison.py 间接调用
#
# 重要说明：
# - 本文件只负责“推理”；
# - 不负责误差计算；
# - 不负责耗时统计；
# - 不负责画图和保存文件。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ==========================================================
# 一、推理结果打包对象
# ==========================================================

@dataclass
class InferenceResult:
    """
    单模型推理结果。

    字段说明：
    - predictions : 模型预测结果，形状 [N, T, 3]
    - targets     : 真值结果，形状 [N, T, 3]
    - num_samples : 样本数 N
    - num_steps   : 轨道长度 T
    - output_dim  : 输出维数，当前通常为 3
    """
    predictions: np.ndarray
    targets: np.ndarray
    num_samples: int
    num_steps: int
    output_dim: int


# ==========================================================
# 二、单 batch 推理
# ==========================================================

@torch.no_grad()
def predict_batch(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    对单个 batch 做推理。

    参数：
    - model
    - x      : [B, T, C]
    - device

    返回：
    - pred   : [B, T, 3]（或更一般的 [B, T, out_dim]）
    """
    model.eval()
    x = x.to(device)
    pred = model(x)
    return pred


# ==========================================================
# 三、整数据集推理
# ==========================================================

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> InferenceResult:
    """
    对整个 DataLoader 做推理，并收集所有预测与真值。

    参数：
    - model
    - loader
    - device

    返回：
    - InferenceResult
    """
    model.eval()

    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        pred_list.append(pred.detach().cpu().numpy())
        target_list.append(y.detach().cpu().numpy())

    if len(pred_list) == 0:
        raise RuntimeError("推理失败：DataLoader 为空，没有任何 batch。")

    predictions = np.concatenate(pred_list, axis=0)
    targets = np.concatenate(target_list, axis=0)

    if predictions.shape != targets.shape:
        raise RuntimeError(
            "推理结果与真值形状不一致："
            f"predictions.shape={predictions.shape}, targets.shape={targets.shape}"
        )

    return InferenceResult(
        predictions=predictions,
        targets=targets,
        num_samples=int(predictions.shape[0]),
        num_steps=int(predictions.shape[1]),
        output_dim=int(predictions.shape[2]),
    )


# ==========================================================
# 四、辅助摘要
# ==========================================================

def summarize_inference_result(result: InferenceResult) -> dict[str, Any]:
    """
    将推理结果整理成摘要字典。
    """
    return {
        "predictions_shape": tuple(result.predictions.shape),
        "targets_shape": tuple(result.targets.shape),
        "num_samples": result.num_samples,
        "num_steps": result.num_steps,
        "output_dim": result.output_dim,
    }