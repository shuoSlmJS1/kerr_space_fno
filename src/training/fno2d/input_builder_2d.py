# ==========================================================
# File: src/training/fno2d/input_builder_2d.py
#
# 功能简介：
# 1. 将“单变化参数”的 1D 轨道数据转换成 FNO2d 所需的二维网格输入；
# 2. 支持 Q-only / a-only / E-only / Lz-only 等单参数任务；
# 3. 原始数据形式：
#       x_raw       : [N, 1]
#       y           : [N, T, 3]
#       lambda_grid : [T]
# 4. 转换后形式：
#       x_2d : [1, N, T, 2]
#       y_2d : [1, N, T, 3]
# 5. 二维网格方向为：
#       第 1 个方向：单变化参数 p
#       第 2 个方向：lambda 轨道方向
#
# 数学含义：
#   把一批独立轨道：
#       p_i -> xyz(lambda)
#   重新组织成二维函数：
#       (p, lambda) -> (x, y, z)
#
#   这才是 FNO2d 的二维算子输入结构。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ==========================================================
# 一、二维数据打包对象
# ==========================================================

@dataclass
class FNO2dFieldData:
    """
    FNO2d 二维场数据。

    字段说明：
    - x_2d:
        FNO2d 输入，shape = [1, N, T, 2]
        最后一维两个通道分别是：
            channel 0: 单变化参数 p
            channel 1: lambda

    - y_2d:
        FNO2d 输出真值，shape = [1, N, T, 3]
        最后一维三个通道分别是：
            x, y, z

    - param_grid:
        单变化参数网格，shape = [N]

    - lambda_grid:
        lambda 网格，shape = [T]

    - param_name:
        参数名，例如 "Q"、"a"、"E"、"Lz"

    - num_param:
        参数方向网格点数 N

    - num_lambda:
        lambda 方向网格点数 T

    - in_dim:
        输入通道数，第一版固定为 2

    - out_dim:
        输出通道数，Kerr xyz 输出固定为 3
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


# ==========================================================
# 二、基础检查函数
# ==========================================================

def validate_single_param_raw_arrays(
    x_raw: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    param_name: str,
) -> None:
    """
    检查单参数原始数组是否合法。

    参数：
    - x_raw:
        shape = [N, 1]
        只允许一个变化参数

    - y:
        shape = [N, T, 3]
        每个参数值对应一条 xyz(lambda) 轨道

    - lambda_grid:
        shape = [T]
        轨道方向的一维网格

    - param_name:
        当前参数名，例如 "Q"、"a"、"E"、"Lz"
    """
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
            "当前 FNO2d 第一版只支持单变化参数数据，因此 x_raw.shape[1] 必须为 1，"
            f"当前 shape={x_raw.shape}"
        )

    if y.ndim != 3:
        raise ValueError(f"y 必须是三维数组 [N,T,3]，当前 shape={y.shape}")

    if y.shape[2] != 3:
        raise ValueError(f"y 最后一维必须是 3，对应 xyz，当前 shape={y.shape}")

    if lambda_grid.ndim != 1:
        raise ValueError(
            f"lambda_grid 必须是一维数组 [T]，当前 shape={lambda_grid.shape}"
        )

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
    """
    按单变化参数 p 从小到大排序。

    为什么要排序：
    - FNO2d 把 p 当作一个网格方向；
    - 网格方向最好按参数值递增排列；
    - 否则二维场的参数方向会乱序。

    输入：
    - x_raw : [N,1]
    - y     : [N,T,3]

    返回：
    - x_sorted : [N,1]
    - y_sorted : [N,T,3]
    """
    param_values = x_raw[:, 0]
    order = np.argsort(param_values)

    x_sorted = x_raw[order]
    y_sorted = y[order]

    return x_sorted, y_sorted


# ==========================================================
# 三、核心构造函数
# ==========================================================

def build_param_lambda_input_field(
    x_raw: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    param_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
) -> FNO2dFieldData:
    """
    将单参数原始数据构造成 FNO2d 二维场。

    输入：
    - x_raw:
        shape = [N, 1]
        每行是一个参数值 p

    - y:
        shape = [N, T, 3]
        对应 xyz(lambda)

    - lambda_grid:
        shape = [T]

    - param_name:
        参数名，例如 "Q"、"a"、"E"、"Lz"

    - sort_param:
        是否按照参数 p 从小到大排序。
        推荐 True。

    - dtype:
        输出数组的数据类型。
        默认 np.float32，适合 PyTorch 训练。

    输出：
    - FNO2dFieldData
        x_2d: [1,N,T,2]
        y_2d: [1,N,T,3]
    """
    validate_single_param_raw_arrays(
        x_raw=x_raw,
        y=y,
        lambda_grid=lambda_grid,
        param_name=param_name,
    )

    # ------------------------------------------------------
    # A. 可选：按参数值从小到大排序
    # ------------------------------------------------------
    if sort_param:
        x_raw, y = ensure_param_grid_sorted(x_raw=x_raw, y=y)

    # ------------------------------------------------------
    # B. 提取网格长度
    # ------------------------------------------------------
    num_param = int(x_raw.shape[0])
    num_lambda = int(lambda_grid.shape[0])

    # ------------------------------------------------------
    # C. 构造参数网格和 lambda 网格
    # ------------------------------------------------------
    # param_grid: [N]
    param_grid = x_raw[:, 0].astype(dtype)

    # lambda_grid: [T]
    lambda_grid = lambda_grid.astype(dtype)

    # P_mesh: [N,T]
    # 每一行是同一个参数值 p，对应不同 lambda
    P_mesh = np.repeat(param_grid[:, None], num_lambda, axis=1)

    # L_mesh: [N,T]
    # 每一列是同一个 lambda，对应不同参数 p
    L_mesh = np.repeat(lambda_grid[None, :], num_param, axis=0)

    # ------------------------------------------------------
    # D. 合并输入通道
    # ------------------------------------------------------
    # x_2d_no_batch: [N,T,2]
    # channel 0: parameter p
    # channel 1: lambda
    x_2d_no_batch = np.stack([P_mesh, L_mesh], axis=-1).astype(dtype)

    # ------------------------------------------------------
    # E. 添加 batch 维度
    # ------------------------------------------------------
    # FNO2d 输入约定是 [B,H,W,C]
    # 这里一个完整的 p-lambda field 就是一个样本，所以 B=1
    x_2d = x_2d_no_batch[None, ...]   # [1,N,T,2]

    # y: [N,T,3] -> [1,N,T,3]
    y_2d = y.astype(dtype)[None, ...]

    return FNO2dFieldData(
        x_2d=x_2d,
        y_2d=y_2d,
        param_grid=param_grid,
        lambda_grid=lambda_grid,
        param_name=str(param_name),
        num_param=num_param,
        num_lambda=num_lambda,
        in_dim=2,
        out_dim=3,
    )


# ==========================================================
# 四、训练 / 验证 / 测试二维场构造
# ==========================================================

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
) -> dict[str, FNO2dFieldData]:
    """
    分别构造 train / val / test 的二维场数据。

    输入：
    - x_train_raw, y_train
    - x_val_raw, y_val
    - x_test_raw, y_test
    - lambda_grid
    - param_name:
        当前单变化参数名，例如 "Q"、"a"、"E"、"Lz"

    输出：
    {
        "train": FNO2dFieldData,
        "val": FNO2dFieldData,
        "test": FNO2dFieldData,
    }

    注意：
    - 第一版为了最小改动，仍沿用原来的 train/val/test 划分；
    - 每个 split 会被构造成一个完整二维场；
    - 所以每个 split 的 batch size 实际是 1。
    """
    train_field = build_param_lambda_input_field(
        x_raw=x_train_raw,
        y=y_train,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
    )

    val_field = build_param_lambda_input_field(
        x_raw=x_val_raw,
        y=y_val,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
    )

    test_field = build_param_lambda_input_field(
        x_raw=x_test_raw,
        y=y_test,
        lambda_grid=lambda_grid,
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
    )

    return {
        "train": train_field,
        "val": val_field,
        "test": test_field,
    }


# ==========================================================
# 五、摘要函数
# ==========================================================

def summarize_fno2d_field(field: FNO2dFieldData) -> dict[str, Any]:
    """
    输出单个 FNO2dFieldData 的摘要信息。
    """
    return {
        "x_2d_shape": tuple(field.x_2d.shape),
        "y_2d_shape": tuple(field.y_2d.shape),
        "param_name": field.param_name,
        "param_grid_shape": tuple(field.param_grid.shape),
        "lambda_grid_shape": tuple(field.lambda_grid.shape),
        "num_param": int(field.num_param),
        "num_lambda": int(field.num_lambda),
        "in_dim": int(field.in_dim),
        "out_dim": int(field.out_dim),
        "param_min": float(np.min(field.param_grid)),
        "param_max": float(np.max(field.param_grid)),
        "lambda_min": float(np.min(field.lambda_grid)),
        "lambda_max": float(np.max(field.lambda_grid)),
    }


def summarize_fno2d_fields(fields: dict[str, FNO2dFieldData]) -> dict[str, Any]:
    """
    输出 train / val / test 三个二维场的摘要。
    """
    return {
        split_name: summarize_fno2d_field(field)
        for split_name, field in fields.items()
    }