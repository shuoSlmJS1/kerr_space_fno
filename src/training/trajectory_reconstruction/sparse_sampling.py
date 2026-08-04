from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SparseSamplingConfig:
    """规则稀疏采样配置。"""

    stride: int
    always_include_first: bool
    always_include_last: bool
    observed_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    observed_point_count: int
    hidden_point_count: int

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的配置。"""
        data = asdict(self)
        data["observed_indices"] = list(self.observed_indices)
        data["hidden_indices"] = list(self.hidden_indices)
        return data


@dataclass(frozen=True)
class SparseTrajectoryData:
    """共享稀疏轨道数据契约。"""

    target_xyz: np.ndarray
    sparse_xyz: np.ndarray
    observed_mask: np.ndarray
    hidden_mask: np.ndarray
    lambda_grid: np.ndarray
    sampling: SparseSamplingConfig


def _validate_integer(value: int, name: str) -> int:
    """验证不接受 bool 的整数参数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _validate_lambda_grid(lambda_grid: np.ndarray, time_steps: int) -> np.ndarray:
    """验证时间坐标为有限且严格递增的一维网格。"""
    grid = np.asarray(lambda_grid)
    if grid.ndim != 1:
        raise ValueError(
            f"lambda_grid must have shape [T]; received shape={grid.shape}."
        )
    if grid.shape[0] != time_steps:
        raise ValueError(
            "lambda_grid length must match target_xyz time dimension; "
            f"received {grid.shape[0]} and {time_steps}."
        )
    if not np.issubdtype(grid.dtype, np.number) or np.iscomplexobj(grid):
        raise TypeError("lambda_grid must have a real numeric dtype.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("lambda_grid must contain only finite values.")
    if np.any(np.diff(grid.astype(np.float64, copy=False)) <= 0.0):
        raise ValueError("lambda_grid must be strictly increasing.")
    return grid.copy()


def build_observed_indices(
    time_steps: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成包含首尾点的 observed 与 hidden 索引。"""
    time_steps = _validate_integer(time_steps, "time_steps")
    stride = _validate_integer(stride, "stride")

    if time_steps < 2:
        raise ValueError("time_steps must be at least 2.")
    if stride < 2:
        raise ValueError("stride must be at least 2.")

    observed = np.arange(0, time_steps, stride, dtype=np.int64)
    observed = np.unique(
        np.concatenate((observed, np.asarray([time_steps - 1], dtype=np.int64)))
    )

    all_indices = np.arange(time_steps, dtype=np.int64)
    hidden = np.setdiff1d(all_indices, observed, assume_unique=True)

    if hidden.size == 0:
        raise ValueError(
            "Sparse reconstruction requires at least one hidden point."
        )

    return observed, hidden


def build_sparse_trajectory_data(
    target_xyz: np.ndarray,
    lambda_grid: np.ndarray,
    stride: int,
) -> SparseTrajectoryData:
    """从完整轨道构造共享稀疏输入、互补 mask 和元数据。"""
    target = np.asarray(target_xyz)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(
            "target_xyz must have shape [N,T,3]; "
            f"received shape={target.shape}."
        )
    if target.shape[0] < 1:
        raise ValueError("target_xyz must contain at least one trajectory.")
    if not np.issubdtype(target.dtype, np.number) or np.iscomplexobj(target):
        raise TypeError("target_xyz must have a real numeric dtype.")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must contain only finite values.")

    time_steps = int(target.shape[1])
    grid = _validate_lambda_grid(lambda_grid, time_steps=time_steps)
    observed_indices, hidden_indices = build_observed_indices(
        time_steps=time_steps,
        stride=stride,
    )

    observed_mask = np.zeros(
        (target.shape[0], time_steps, 1),
        dtype=np.bool_,
    )
    observed_mask[:, observed_indices, 0] = True
    hidden_mask = np.logical_not(observed_mask)

    sparse_xyz = np.zeros_like(target)
    sparse_xyz[:, observed_indices, :] = target[:, observed_indices, :]

    sampling = SparseSamplingConfig(
        stride=int(stride),
        always_include_first=True,
        always_include_last=True,
        observed_indices=tuple(int(index) for index in observed_indices),
        hidden_indices=tuple(int(index) for index in hidden_indices),
        observed_point_count=int(observed_indices.size),
        hidden_point_count=int(hidden_indices.size),
    )

    # 返回只读视图，避免复制完整轨道，同时防止通过结果对象修改输入。
    target_view = target.view()
    target_view.flags.writeable = False

    return SparseTrajectoryData(
        target_xyz=target_view,
        sparse_xyz=sparse_xyz,
        observed_mask=observed_mask,
        hidden_mask=hidden_mask,
        lambda_grid=grid,
        sampling=sampling,
    )
