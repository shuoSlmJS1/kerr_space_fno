from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator


def _validate_baseline_inputs(
    lambda_grid: np.ndarray,
    sparse_xyz: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """验证插值 baseline 的共享输入契约。"""
    grid = np.asarray(lambda_grid)
    sparse = np.asarray(sparse_xyz)
    mask = np.asarray(observed_mask)

    if grid.ndim != 1:
        raise ValueError(
            f"lambda_grid must have shape [T]; received shape={grid.shape}."
        )
    if sparse.ndim != 3 or sparse.shape[-1] != 3:
        raise ValueError(
            "sparse_xyz must have shape [N,T,3]; "
            f"received shape={sparse.shape}."
        )
    if sparse.shape[0] < 1:
        raise ValueError("sparse_xyz must contain at least one trajectory.")
    expected_mask_shape = sparse.shape[:2] + (1,)
    if mask.shape != expected_mask_shape:
        raise ValueError(
            "observed_mask must have shape [N,T,1]; "
            f"expected {expected_mask_shape}, received {mask.shape}."
        )
    if mask.dtype != np.bool_:
        raise TypeError("observed_mask must use bool dtype.")
    if grid.shape[0] != sparse.shape[1]:
        raise ValueError(
            "lambda_grid length must match sparse_xyz time dimension; "
            f"received {grid.shape[0]} and {sparse.shape[1]}."
        )
    if not np.issubdtype(grid.dtype, np.number) or np.iscomplexobj(grid):
        raise TypeError("lambda_grid must have a real numeric dtype.")
    if not np.issubdtype(sparse.dtype, np.number) or np.iscomplexobj(sparse):
        raise TypeError("sparse_xyz must have a real numeric dtype.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("lambda_grid must contain only finite values.")
    if not np.all(np.isfinite(sparse)):
        raise ValueError("sparse_xyz must contain only finite values.")

    grid64 = grid.astype(np.float64, copy=False)
    if np.any(np.diff(grid64) <= 0.0):
        raise ValueError("lambda_grid must be strictly increasing.")

    observed_counts = np.sum(mask[..., 0], axis=1)
    insufficient = np.flatnonzero(observed_counts < 2)
    if insufficient.size > 0:
        raise ValueError(
            "Every trajectory must contain at least two observed points; "
            f"insufficient for trajectory indices {insufficient.tolist()}."
        )

    missing_endpoints = np.flatnonzero(
        np.logical_not(mask[:, 0, 0]) | np.logical_not(mask[:, -1, 0])
    )
    if missing_endpoints.size > 0:
        raise ValueError(
            "Every trajectory must observe the first and last time points; "
            f"missing for trajectory indices {missing_endpoints.tolist()}."
        )

    return grid64, sparse, mask


def _reconstruct(
    lambda_grid: np.ndarray,
    sparse_xyz: np.ndarray,
    observed_mask: np.ndarray,
    interpolator: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """逐轨道逐坐标执行只依赖 observed points 的插值。"""
    grid, sparse, mask = _validate_baseline_inputs(
        lambda_grid=lambda_grid,
        sparse_xyz=sparse_xyz,
        observed_mask=observed_mask,
    )
    prediction = np.empty_like(sparse, dtype=np.float64)

    for trajectory_index in range(sparse.shape[0]):
        observed = mask[trajectory_index, :, 0]
        observed_grid = grid[observed]
        for component_index in range(3):
            observed_values = np.asarray(
                sparse[
                    trajectory_index,
                    observed,
                    component_index,
                ],
                dtype=np.float64,
            )
            prediction[trajectory_index, :, component_index] = interpolator(
                observed_grid,
                observed_values,
                grid,
            )

        # 显式恢复观测点，保证节点值与输入完全一致。
        prediction[trajectory_index, observed, :] = sparse[
            trajectory_index,
            observed,
            :,
        ]

    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Interpolation produced non-finite predictions.")
    return prediction


def _linear_interpolator(
    observed_grid: np.ndarray,
    observed_values: np.ndarray,
    full_grid: np.ndarray,
) -> np.ndarray:
    """NumPy 线性插值适配器。"""
    return np.interp(full_grid, observed_grid, observed_values)


def _pchip_interpolator(
    observed_grid: np.ndarray,
    observed_values: np.ndarray,
    full_grid: np.ndarray,
) -> np.ndarray:
    """SciPy PCHIP 插值适配器。"""
    interpolator = PchipInterpolator(
        observed_grid,
        observed_values,
        extrapolate=False,
    )
    return np.asarray(interpolator(full_grid), dtype=np.float64)


def reconstruct_linear(
    lambda_grid: np.ndarray,
    sparse_xyz: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """仅使用 observed points 执行逐坐标线性插值。"""
    return _reconstruct(
        lambda_grid=lambda_grid,
        sparse_xyz=sparse_xyz,
        observed_mask=observed_mask,
        interpolator=_linear_interpolator,
    )


def reconstruct_pchip(
    lambda_grid: np.ndarray,
    sparse_xyz: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """仅使用 observed points 执行逐坐标 PCHIP 插值。"""
    return _reconstruct(
        lambda_grid=lambda_grid,
        sparse_xyz=sparse_xyz,
        observed_mask=observed_mask,
        interpolator=_pchip_interpolator,
    )
