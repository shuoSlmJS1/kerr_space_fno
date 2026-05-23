# ==========================================================
# File: src/training/fno2d/input_builder_2d.py
#
# 功能简介：
# 1. 将“单变化参数”的 1D 轨道数据转换成 FNO2d 所需的二维网格输入；
# 2. 支持 Q-only / a-only / E-only / Lz-only 等单参数任务；
# 3. 支持多 cfg 条件通道，例如把不同固定 a 的二维场堆叠为：
#       [B_cfg, N_param, T_lambda, C]
#    此时输入通道可以是 [Q, lambda, a]。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FNO2dFieldData:
    """
    FNO2d 二维场数据。

    形状约定：
    - x_2d: [B, N, T, C]
    - y_2d: [B, N, T, 3]

    单 cfg 旧模式下 B=1, C=2，对应 [p, lambda]。
    多 cfg 新模式下 B=B_cfg, C>=3，例如 [Q, lambda, a]。
    """
    x_2d: np.ndarray
    y_2d: np.ndarray
    param_grid: np.ndarray
    lambda_grid: np.ndarray
    param_name: str
    num_param: int
    num_lambda: int
    in_dim: int
    out_dim: int
    input_channel_names: list[str] | None = None
    cfg_param_names: list[str] | None = None
    cfg_values: np.ndarray | None = None


def validate_single_param_raw_arrays(
    x_raw: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    param_name: str,
) -> None:
    """检查单参数原始数组是否合法。"""
    if not isinstance(param_name, str) or len(param_name.strip()) == 0:
        raise ValueError(f"param_name 必须是非空字符串，当前为 {param_name!r}")

    if not isinstance(x_raw, np.ndarray):
        raise TypeError("x_raw 必须是 numpy.ndarray。")
    if not isinstance(y, np.ndarray):
        raise TypeError("y 必须是 numpy.ndarray。")
    if not isinstance(lambda_grid, np.ndarray):
        raise TypeError("lambda_grid 必须是 numpy.ndarray。")

    if x_raw.ndim != 2:
        raise ValueError(f"x_raw 必须是二维数组 [N,1]，当前 shape={x_raw.shape}")
    if x_raw.shape[1] != 1:
        raise ValueError(
            "当前 FNO2d 构造函数只支持单变化参数数据，因此 x_raw.shape[1] 必须为 1，"
            f"当前 shape={x_raw.shape}"
        )
    if y.ndim != 3:
        raise ValueError(f"y 必须是三维数组 [N,T,3]，当前 shape={y.shape}")
    if y.shape[2] != 3:
        raise ValueError(f"y 最后一维必须是 3，对应 xyz，当前 shape={y.shape}")
    if lambda_grid.ndim != 1:
        raise ValueError(f"lambda_grid 必须是一维数组 [T]，当前 shape={lambda_grid.shape}")
    if x_raw.shape[0] != y.shape[0]:
        raise ValueError(
            "x_raw 和 y 的样本数 N 必须一致，"
            f"当前 x_raw.shape={x_raw.shape}, y.shape={y.shape}"
        )
    if lambda_grid.shape[0] != y.shape[1]:
        raise ValueError(
            "lambda_grid 长度必须等于 y 的轨道长度 T，"
            f"当前 lambda_grid.shape={lambda_grid.shape}, y.shape={y.shape}"
        )


def ensure_param_grid_sorted(
    x_raw: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按单变化参数 p 从小到大排序。"""
    order = np.argsort(x_raw[:, 0])
    return x_raw[order], y[order]


def build_param_lambda_input_field(
    x_raw: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    param_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    extra_constant_channels: dict[str, float] | None = None,
) -> FNO2dFieldData:
    """
    将单参数原始数据构造成 FNO2d 二维场。

    默认输入通道：
        [param_name, lambda]

    如果 extra_constant_channels={"a": 0.5}，则输入通道变为：
        [param_name, lambda, a]

    输出：
    - x_2d: [1,N,T,C]
    - y_2d: [1,N,T,3]
    """
    validate_single_param_raw_arrays(
        x_raw=x_raw,
        y=y,
        lambda_grid=lambda_grid,
        param_name=param_name,
    )

    if sort_param:
        x_raw, y = ensure_param_grid_sorted(x_raw=x_raw, y=y)

    num_param = int(x_raw.shape[0])
    num_lambda = int(lambda_grid.shape[0])

    param_grid = x_raw[:, 0].astype(dtype)
    lambda_grid = lambda_grid.astype(dtype)

    p_mesh = np.repeat(param_grid[:, None], num_lambda, axis=1)
    l_mesh = np.repeat(lambda_grid[None, :], num_param, axis=0)

    channels = [p_mesh, l_mesh]
    input_channel_names = [str(param_name), "lambda"]

    cfg_param_names: list[str] = []
    cfg_values_list: list[float] = []

    if extra_constant_channels:
        for name, value in extra_constant_channels.items():
            value_float = float(value)
            const_mesh = np.full((num_param, num_lambda), value_float, dtype=dtype)
            channels.append(const_mesh)
            input_channel_names.append(str(name))
            cfg_param_names.append(str(name))
            cfg_values_list.append(value_float)

    x_2d_no_batch = np.stack(channels, axis=-1).astype(dtype)
    x_2d = x_2d_no_batch[None, ...]
    y_2d = y.astype(dtype)[None, ...]

    cfg_values = None
    if cfg_values_list:
        cfg_values = np.asarray([cfg_values_list], dtype=dtype)  # [1, num_cfg_channels]

    return FNO2dFieldData(
        x_2d=x_2d,
        y_2d=y_2d,
        param_grid=param_grid,
        lambda_grid=lambda_grid,
        param_name=str(param_name),
        num_param=num_param,
        num_lambda=num_lambda,
        in_dim=int(x_2d.shape[-1]),
        out_dim=3,
        input_channel_names=input_channel_names,
        cfg_param_names=cfg_param_names if cfg_param_names else None,
        cfg_values=cfg_values,
    )


def build_param_lambda_train_val_test_fields(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    lambda_grid: np.ndarray,
    param_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    extra_constant_channels: dict[str, float] | None = None,
) -> dict[str, FNO2dFieldData]:
    """分别构造 train / val / test 的二维场数据。"""
    train_field = build_param_lambda_input_field(
        x_raw=x_train_raw,
        y=y_train,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
        extra_constant_channels=extra_constant_channels,
    )

    val_field = build_param_lambda_input_field(
        x_raw=x_val_raw,
        y=y_val,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
        extra_constant_channels=extra_constant_channels,
    )

    test_field = build_param_lambda_input_field(
        x_raw=x_test_raw,
        y=y_test,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
        extra_constant_channels=extra_constant_channels,
    )

    return {"train": train_field, "val": val_field, "test": test_field}


def summarize_fno2d_field(field: FNO2dFieldData) -> dict[str, Any]:
    """输出单个 FNO2dFieldData 的摘要信息。"""
    summary = {
        "x_2d_shape": tuple(field.x_2d.shape),
        "y_2d_shape": tuple(field.y_2d.shape),
        "param_name": field.param_name,
        "param_grid_shape": tuple(field.param_grid.shape),
        "lambda_grid_shape": tuple(field.lambda_grid.shape),
        "num_fields": int(field.x_2d.shape[0]),
        "num_param": int(field.num_param),
        "num_lambda": int(field.num_lambda),
        "in_dim": int(field.in_dim),
        "out_dim": int(field.out_dim),
        "input_channel_names": field.input_channel_names,
        "cfg_param_names": field.cfg_param_names,
        "param_min": float(np.min(field.param_grid)),
        "param_max": float(np.max(field.param_grid)),
        "lambda_min": float(np.min(field.lambda_grid)),
        "lambda_max": float(np.max(field.lambda_grid)),
    }

    if field.cfg_values is not None:
        summary["cfg_values_shape"] = tuple(field.cfg_values.shape)
        summary["cfg_values"] = field.cfg_values.astype(float).tolist()

    return summary


def summarize_fno2d_fields(fields: dict[str, FNO2dFieldData]) -> dict[str, Any]:
    """输出 train / val / test 三个二维场的摘要。"""
    return {
        split_name: summarize_fno2d_field(field)
        for split_name, field in fields.items()
    }
