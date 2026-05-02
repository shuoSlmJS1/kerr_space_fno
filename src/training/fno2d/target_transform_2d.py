# ==========================================================
# File: src/training/fno2d/target_transform_2d.py
#
# 功能简介：
# 1. 为 FNO2d 输出目标提供通用 target transform；
# 2. 支持 raw / residual_initial 两种模式；
# 3. raw:
#       直接学习原始 xyz 轨道；
# 4. residual_initial:
#       学习相对初始点的轨道残差：
#           Y_res(p, lambda) = Y(p, lambda) - Y(p, lambda_0)
# 5. 分析时可把残差预测还原成物理空间 xyz：
#           Y_pred = Y_res_pred + Y(p, lambda_0)
#
# 张量 / 数组约定：
# - y_2d: [B, H, W, 3]
#   B: batch size
#   H: 参数方向，例如 Q/a/E/Lz
#   W: lambda 方向
#   3: xyz
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


# ==========================================================
# 一、target transform 配置
# ==========================================================

SUPPORTED_TARGET_TRANSFORMS = {
    "raw",
    "residual_initial",
}


@dataclass
class TargetTransformConfig:
    """
    target transform 配置。

    字段说明：
    - mode:
        目标变换模式。
        当前支持：
            raw
            residual_initial

    - lambda_reference_index:
        residual_initial 使用哪个 lambda 点作为参考点。
        默认 0，即轨道初始点 lambda_0。
    """
    mode: str = "raw"
    lambda_reference_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        转成可 JSON 保存的字典。
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TargetTransformConfig":
        """
        从字典恢复配置。
        """
        return TargetTransformConfig(
            mode=str(data.get("mode", "raw")),
            lambda_reference_index=int(data.get("lambda_reference_index", 0)),
        )


# ==========================================================
# 二、基础检查
# ==========================================================

def validate_target_transform_mode(mode: str) -> str:
    """
    检查 target transform 模式是否合法。
    """
    mode = str(mode).lower()

    if mode not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"不支持的 target transform mode={mode!r}，"
            f"当前支持：{sorted(SUPPORTED_TARGET_TRANSFORMS)}"
        )

    return mode


def validate_y_field(y: np.ndarray, name: str = "y") -> None:
    """
    检查 y field 是否是 [B,H,W,3]。
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray，当前类型为 {type(y)}")

    if y.ndim != 4:
        raise ValueError(f"{name} 必须是 4 维数组 [B,H,W,3]，当前 shape={y.shape}")

    if y.shape[-1] != 3:
        raise ValueError(f"{name} 最后一维必须是 3，对应 xyz，当前 shape={y.shape}")


def validate_reference_index(
    y: np.ndarray,
    lambda_reference_index: int,
) -> int:
    """
    检查 lambda_reference_index 是否合法。
    """
    validate_y_field(y, name="y")

    num_lambda = int(y.shape[2])
    idx = int(lambda_reference_index)

    if idx < 0 or idx >= num_lambda:
        raise ValueError(
            f"lambda_reference_index 必须在 [0, {num_lambda - 1}] 内，当前为 {idx}"
        )

    return idx


# ==========================================================
# 三、目标变换
# ==========================================================

def extract_initial_reference(
    y: np.ndarray,
    lambda_reference_index: int = 0,
) -> np.ndarray:
    """
    提取每条参数轨道的参考点。

    输入：
    - y: [B,H,W,3]

    输出：
    - y_ref: [B,H,1,3]

    说明：
    - 对 residual_initial，默认取 lambda_0 处的 xyz；
    - 后续通过广播与 [B,H,W,3] 相加或相减。
    """
    idx = validate_reference_index(
        y=y,
        lambda_reference_index=lambda_reference_index,
    )

    return y[:, :, idx:idx + 1, :].astype(np.float32)


def transform_output_field(
    y: np.ndarray,
    config: TargetTransformConfig,
) -> np.ndarray:
    """
    对输出目标 y 做 target transform。

    输入：
    - y:
        原始物理空间 xyz，shape=[B,H,W,3]

    - config:
        target transform 配置

    输出：
    - transformed_y:
        变换后的训练目标，shape=[B,H,W,3]
    """
    validate_y_field(y, name="y")

    mode = validate_target_transform_mode(config.mode)

    if mode == "raw":
        return y.astype(np.float32)

    if mode == "residual_initial":
        y_ref = extract_initial_reference(
            y=y,
            lambda_reference_index=config.lambda_reference_index,
        )

        return (y.astype(np.float32) - y_ref).astype(np.float32)

    raise RuntimeError(f"未处理的 target transform mode={mode!r}")


def inverse_transform_output_field(
    transformed_y: np.ndarray,
    reference_y_raw: np.ndarray,
    config: TargetTransformConfig,
) -> np.ndarray:
    """
    将模型输出从 target-transform 空间还原回物理空间 xyz。

    输入：
    - transformed_y:
        模型预测结果，shape=[B,H,W,3]
        如果 mode=raw，它本身就是物理空间；
        如果 mode=residual_initial，它是残差空间。

    - reference_y_raw:
        原始物理空间真值 y，shape=[B,H,W,3]
        用于提取每条轨道的初始点 y(p, lambda_0)。

    - config:
        target transform 配置

    输出：
    - y_physical:
        物理空间 xyz，shape=[B,H,W,3]
    """
    validate_y_field(transformed_y, name="transformed_y")
    validate_y_field(reference_y_raw, name="reference_y_raw")

    if transformed_y.shape != reference_y_raw.shape:
        raise ValueError(
            "transformed_y 和 reference_y_raw shape 必须一致，"
            f"当前 transformed_y={transformed_y.shape}, reference_y_raw={reference_y_raw.shape}"
        )

    mode = validate_target_transform_mode(config.mode)

    if mode == "raw":
        return transformed_y.astype(np.float32)

    if mode == "residual_initial":
        y_ref = extract_initial_reference(
            y=reference_y_raw,
            lambda_reference_index=config.lambda_reference_index,
        )

        return (transformed_y.astype(np.float32) + y_ref).astype(np.float32)

    raise RuntimeError(f"未处理的 target transform mode={mode!r}")


# ==========================================================
# 四、摘要函数
# ==========================================================

def summarize_target_transform_config(
    config: TargetTransformConfig,
) -> dict[str, Any]:
    """
    输出 target transform 配置摘要。
    """
    return config.to_dict()