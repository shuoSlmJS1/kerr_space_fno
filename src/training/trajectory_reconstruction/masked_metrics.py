from __future__ import annotations

from typing import Any

import numpy as np


COMPONENT_NAMES = ("x", "y", "z")


def _validate_metric_inputs(
    prediction_xyz: np.ndarray,
    target_xyz: np.ndarray,
    hidden_mask: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """验证 hidden-only 指标输入。"""
    prediction = np.asarray(prediction_xyz)
    target = np.asarray(target_xyz)
    mask = np.asarray(hidden_mask)

    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(
            "prediction_xyz must have shape [N,T,3]; "
            f"received shape={prediction.shape}."
        )
    if target.shape != prediction.shape:
        raise ValueError(
            "target_xyz must have the same shape as prediction_xyz; "
            f"received {target.shape} and {prediction.shape}."
        )
    if prediction.shape[0] < 1:
        raise ValueError("prediction_xyz must contain at least one trajectory.")
    expected_mask_shape = prediction.shape[:2] + (1,)
    if mask.shape != expected_mask_shape:
        raise ValueError(
            "hidden_mask must have shape [N,T,1]; "
            f"expected {expected_mask_shape}, received {mask.shape}."
        )
    if mask.dtype != np.bool_:
        raise TypeError("hidden_mask must use bool dtype.")
    if (
        not np.issubdtype(prediction.dtype, np.number)
        or np.iscomplexobj(prediction)
    ):
        raise TypeError("prediction_xyz must have a real numeric dtype.")
    if not np.issubdtype(target.dtype, np.number) or np.iscomplexobj(target):
        raise TypeError("target_xyz must have a real numeric dtype.")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction_xyz must contain only finite values.")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must contain only finite values.")

    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite value.")

    hidden_per_trajectory = np.sum(mask[..., 0], axis=1)
    if np.any(hidden_per_trajectory == 0):
        indices = np.flatnonzero(hidden_per_trajectory == 0).tolist()
        raise ValueError(
            "Every trajectory must contain at least one hidden point; "
            f"missing for trajectory indices {indices}."
        )

    return prediction, target, mask, eps


def _relative_l2(
    squared_error_sum: float,
    squared_target_sum: float,
    eps: float,
) -> float:
    """按统一稳定项定义计算 Relative L2。"""
    numerator = np.sqrt(squared_error_sum)
    denominator = np.sqrt(squared_target_sum) + eps
    return float(numerator / denominator)


def compute_hidden_masked_metrics(
    prediction_xyz: np.ndarray,
    target_xyz: np.ndarray,
    hidden_mask: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """仅在 hidden points 上计算 raw xyz 指标。"""
    prediction, target, mask, eps = _validate_metric_inputs(
        prediction_xyz=prediction_xyz,
        target_xyz=target_xyz,
        hidden_mask=hidden_mask,
        eps=eps,
    )

    hidden_point_count = int(np.sum(mask, dtype=np.int64))
    component_error_sums = np.zeros(3, dtype=np.float64)
    component_target_sums = np.zeros(3, dtype=np.float64)
    trajectory_relative_l2 = np.empty(prediction.shape[0], dtype=np.float64)

    # 按轨道提取 hidden 点，避免为整个批次创建 float64 中间数组。
    for trajectory_index in range(prediction.shape[0]):
        hidden = mask[trajectory_index, :, 0]
        prediction_hidden = np.asarray(
            prediction[trajectory_index, hidden, :],
            dtype=np.float64,
        )
        target_hidden = np.asarray(
            target[trajectory_index, hidden, :],
            dtype=np.float64,
        )
        difference = prediction_hidden - target_hidden
        trajectory_component_error = np.einsum(
            "ij,ij->j",
            difference,
            difference,
            dtype=np.float64,
        )
        trajectory_component_target = np.einsum(
            "ij,ij->j",
            target_hidden,
            target_hidden,
            dtype=np.float64,
        )
        component_error_sums += trajectory_component_error
        component_target_sums += trajectory_component_target
        trajectory_relative_l2[trajectory_index] = _relative_l2(
            squared_error_sum=float(np.sum(trajectory_component_error)),
            squared_target_sum=float(np.sum(trajectory_component_target)),
            eps=eps,
        )

    components: dict[str, dict[str, float]] = {}
    for component_index, component_name in enumerate(COMPONENT_NAMES):
        component_error_sum = float(component_error_sums[component_index])
        component_target_sum = float(component_target_sums[component_index])
        components[component_name] = {
            "mse": component_error_sum / hidden_point_count,
            "relative_l2": _relative_l2(
                squared_error_sum=component_error_sum,
                squared_target_sum=component_target_sum,
                eps=eps,
            ),
        }

    total_error_sum = float(np.sum(component_error_sums, dtype=np.float64))
    total_target_sum = float(np.sum(component_target_sums, dtype=np.float64))

    return {
        "metric_space": "raw_xyz_physical_space",
        "mask_scope": "hidden_points_only",
        "eps": eps,
        "hidden_point_count": hidden_point_count,
        "components": components,
        "overall": {
            "mse": total_error_sum / (3 * hidden_point_count),
            "relative_l2": _relative_l2(
                squared_error_sum=total_error_sum,
                squared_target_sum=total_target_sum,
                eps=eps,
            ),
        },
        "per_trajectory_relative_l2": {
            "values": [float(value) for value in trajectory_relative_l2],
            "mean": float(np.mean(trajectory_relative_l2)),
            "median": float(np.median(trajectory_relative_l2)),
            "p95": float(np.percentile(trajectory_relative_l2, 95.0)),
            "max": float(np.max(trajectory_relative_l2)),
        },
    }
