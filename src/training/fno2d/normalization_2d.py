# ==========================================================
# File: src/training/fno2d/normalization_2d.py
#
# 功能简介：
# 1. 为 FNO2d 二维场数据提供标准化工具；
# 2. 支持 input field x_2d 和 output field y_2d 的 standard normalization；
# 3. 统计量只从 train split 计算；
# 4. val/test 必须使用 train 统计量，避免数据泄露；
# 5. 推理分析时可以用同一组统计量把预测结果反标准化回物理量。
#
# 当前支持：
# - normalization = "none"
# - normalization = "standard"
#
# 张量 / 数组约定：
# - x_2d: [B,H,W,C]
# - y_2d: [B,H,W,3]
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


# ==========================================================
# 一、归一化统计量对象
# ==========================================================

@dataclass
class FieldNormalizationStats:
    """
    二维场标准化统计量。

    字段说明：
    - method:
        归一化方法。
        当前支持 "none" 和 "standard"。

    - x_mean:
        输入 x 的通道均值，shape = [C]

    - x_std:
        输入 x 的通道标准差，shape = [C]

    - y_mean:
        输出 y 的通道均值，shape = [3]

    - y_std:
        输出 y 的通道标准差，shape = [3]

    - eps:
        防止除零的小量。
    """
    method: str
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        """
        转成可 JSON 保存的字典。
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FieldNormalizationStats":
        """
        从 JSON 字典恢复统计量对象。
        """
        return FieldNormalizationStats(
            method=str(data["method"]),
            x_mean=[float(v) for v in data["x_mean"]],
            x_std=[float(v) for v in data["x_std"]],
            y_mean=[float(v) for v in data["y_mean"]],
            y_std=[float(v) for v in data["y_std"]],
            eps=float(data.get("eps", 1e-8)),
        )


# ==========================================================
# 二、基础检查
# ==========================================================

def validate_field_array(
    array: np.ndarray,
    name: str,
) -> None:
    """
    检查二维场数组是否合法。

    要求：
    - 必须是 numpy.ndarray
    - 必须是 4 维：[B,H,W,C]
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray，当前类型为 {type(array)}")

    if array.ndim != 4:
        raise ValueError(f"{name} 必须是 4 维数组 [B,H,W,C]，当前 shape={array.shape}")


def validate_normalization_method(method: str) -> str:
    """
    检查归一化方法是否合法。
    """
    method = str(method).lower()

    if method not in {"none", "standard"}:
        raise ValueError(
            f"不支持的 normalization method={method!r}，"
            "当前只支持 'none' 和 'standard'。"
        )

    return method


# ==========================================================
# 三、统计量计算
# ==========================================================

def compute_channel_mean_std(
    array: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    按通道计算均值和标准差。

    输入：
    - array: [B,H,W,C]

    输出：
    - mean: [C]
    - std : [C]

    说明：
    - 对 B,H,W 三个维度求统计量；
    - 最后一维 C 是通道维，不混在一起。
    """
    validate_field_array(array, name="array")

    mean = np.mean(array, axis=(0, 1, 2))
    std = np.std(array, axis=(0, 1, 2))

    std = np.maximum(std, eps)

    return mean.astype(np.float32), std.astype(np.float32)


def compute_field_normalization_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "standard",
    eps: float = 1e-8,
) -> FieldNormalizationStats:
    """
    从训练集二维场计算标准化统计量。

    输入：
    - x_train: [B,H,W,C]
    - y_train: [B,H,W,3]
    - method : "none" 或 "standard"

    返回：
    - FieldNormalizationStats
    """
    method = validate_normalization_method(method)

    validate_field_array(x_train, name="x_train")
    validate_field_array(y_train, name="y_train")

    if method == "none":
        x_channels = int(x_train.shape[-1])
        y_channels = int(y_train.shape[-1])

        return FieldNormalizationStats(
            method="none",
            x_mean=[0.0] * x_channels,
            x_std=[1.0] * x_channels,
            y_mean=[0.0] * y_channels,
            y_std=[1.0] * y_channels,
            eps=float(eps),
        )

    x_mean, x_std = compute_channel_mean_std(x_train, eps=eps)
    y_mean, y_std = compute_channel_mean_std(y_train, eps=eps)

    return FieldNormalizationStats(
        method="standard",
        x_mean=[float(v) for v in x_mean],
        x_std=[float(v) for v in x_std],
        y_mean=[float(v) for v in y_mean],
        y_std=[float(v) for v in y_std],
        eps=float(eps),
    )


# ==========================================================
# 四、应用归一化 / 反归一化
# ==========================================================

def normalize_input_field(
    x: np.ndarray,
    stats: FieldNormalizationStats,
) -> np.ndarray:
    """
    标准化输入二维场 x。

    输入：
    - x: [B,H,W,C]

    输出：
    - x_norm: [B,H,W,C]
    """
    validate_field_array(x, name="x")

    if stats.method == "none":
        return x.astype(np.float32)

    x_mean = np.asarray(stats.x_mean, dtype=np.float32)
    x_std = np.asarray(stats.x_std, dtype=np.float32)

    return ((x.astype(np.float32) - x_mean) / x_std).astype(np.float32)


def normalize_output_field(
    y: np.ndarray,
    stats: FieldNormalizationStats,
) -> np.ndarray:
    """
    标准化输出二维场 y。

    输入：
    - y: [B,H,W,3]

    输出：
    - y_norm: [B,H,W,3]
    """
    validate_field_array(y, name="y")

    if stats.method == "none":
        return y.astype(np.float32)

    y_mean = np.asarray(stats.y_mean, dtype=np.float32)
    y_std = np.asarray(stats.y_std, dtype=np.float32)

    return ((y.astype(np.float32) - y_mean) / y_std).astype(np.float32)


def denormalize_output_field(
    y_norm: np.ndarray,
    stats: FieldNormalizationStats,
) -> np.ndarray:
    """
    将模型输出从标准化空间还原到物理空间。

    输入：
    - y_norm: [B,H,W,3]

    输出：
    - y: [B,H,W,3]
    """
    validate_field_array(y_norm, name="y_norm")

    if stats.method == "none":
        return y_norm.astype(np.float32)

    y_mean = np.asarray(stats.y_mean, dtype=np.float32)
    y_std = np.asarray(stats.y_std, dtype=np.float32)

    return (y_norm.astype(np.float32) * y_std + y_mean).astype(np.float32)


def apply_normalization_to_field_pair(
    x: np.ndarray,
    y: np.ndarray,
    stats: FieldNormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    """
    同时标准化输入 x 和输出 y。

    输入：
    - x: [B,H,W,C]
    - y: [B,H,W,3]

    输出：
    - x_norm
    - y_norm
    """
    x_norm = normalize_input_field(x=x, stats=stats)
    y_norm = normalize_output_field(y=y, stats=stats)

    return x_norm, y_norm


# ==========================================================
# 五、摘要函数
# ==========================================================

def summarize_normalization_stats(
    stats: FieldNormalizationStats,
) -> dict[str, Any]:
    """
    输出归一化统计量摘要。
    """
    return {
        "method": stats.method,
        "x_mean": stats.x_mean,
        "x_std": stats.x_std,
        "y_mean": stats.y_mean,
        "y_std": stats.y_std,
        "eps": float(stats.eps),
    }