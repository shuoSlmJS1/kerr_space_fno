from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.trajectory_reconstruction.masked_metrics import (
    compute_hidden_masked_metrics,
)
from src.training.trajectory_reconstruction.sparse_sampling import (
    SparseTrajectoryData,
    build_sparse_trajectory_data,
)


@dataclass(frozen=True)
class ReconstructionNormalizationStats:
    """稀疏重建任务的训练集归一化统计量。"""

    input_xyz_mean: list[float]
    input_xyz_std: list[float]
    target_xyz_mean: list[float]
    target_xyz_std: list[float]
    lambda_min: float
    lambda_max: float
    eps: float

    def to_dict(self) -> dict[str, object]:
        """返回可写入 JSON 的统计量。"""
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionSplits:
    """三个既有数据划分对应的稀疏重建数据。"""

    train: SparseTrajectoryData
    val: SparseTrajectoryData
    test: SparseTrajectoryData
    dataset_path: Path


class SparseReconstructionDataset(Dataset):
    """为 FNO1D 提供稀疏重建所需的样本张量。"""

    def __init__(
        self,
        sparse_data: SparseTrajectoryData,
        normalization: ReconstructionNormalizationStats,
    ) -> None:
        model_input = build_reconstruction_model_input(
            sparse_data=sparse_data,
            normalization=normalization,
        )
        target_normalized = normalize_target_xyz(
            target_xyz=sparse_data.target_xyz,
            normalization=normalization,
        )

        self.model_input = torch.from_numpy(np.ascontiguousarray(model_input))
        self.target_normalized = torch.from_numpy(
            np.ascontiguousarray(target_normalized)
        )
        # 保留原始物理量供验证与测试指标使用，避免从归一化目标反推真值。
        self.target_raw = torch.from_numpy(
            np.array(sparse_data.target_xyz, copy=True)
        )
        self.sparse_raw = torch.from_numpy(
            np.array(sparse_data.sparse_xyz, copy=True)
        )
        self.observed_mask = torch.from_numpy(
            np.array(sparse_data.observed_mask, copy=True)
        )
        self.hidden_mask = torch.from_numpy(
            np.array(sparse_data.hidden_mask, copy=True)
        )

    def __len__(self) -> int:
        return int(self.model_input.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.model_input[index],
            self.target_normalized[index],
            self.target_raw[index],
            self.sparse_raw[index],
            self.observed_mask[index],
            self.hidden_mask[index],
        )


def _validate_finite_real_array(array: np.ndarray, name: str) -> np.ndarray:
    """验证实数数组的有限性。"""
    value = np.asarray(array)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise TypeError(f"{name} must have a real numeric dtype.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def load_reconstruction_splits(
    dataset_path: str | Path,
    stride: int,
) -> ReconstructionSplits:
    """读取既有划分并分别构造稀疏重建输入。"""
    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    with np.load(path, allow_pickle=False) as dataset:
        required_keys = ("y_train", "y_val", "y_test", "lambda_grid")
        missing_keys = [key for key in required_keys if key not in dataset]
        if missing_keys:
            raise KeyError(f"Dataset is missing required arrays: {missing_keys}.")
        y_train = np.asarray(dataset["y_train"])
        y_val = np.asarray(dataset["y_val"])
        y_test = np.asarray(dataset["y_test"])
        lambda_grid = np.asarray(dataset["lambda_grid"])

    return ReconstructionSplits(
        train=build_sparse_trajectory_data(y_train, lambda_grid, stride),
        val=build_sparse_trajectory_data(y_val, lambda_grid, stride),
        test=build_sparse_trajectory_data(y_test, lambda_grid, stride),
        dataset_path=path,
    )


def fit_reconstruction_normalization(
    train_data: SparseTrajectoryData,
    eps: float = 1e-8,
) -> ReconstructionNormalizationStats:
    """仅使用训练集拟合输入与目标的标准化统计量。"""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite value.")

    observed_values = train_data.sparse_xyz[
        train_data.observed_mask[..., 0]
    ]
    if observed_values.shape[0] == 0:
        raise ValueError("Training data must contain observed points.")

    input_mean = np.mean(observed_values, axis=0, dtype=np.float64)
    input_std = np.maximum(
        np.std(observed_values, axis=0, dtype=np.float64),
        eps,
    )
    target_mean = np.mean(train_data.target_xyz, axis=(0, 1), dtype=np.float64)
    target_std = np.maximum(
        np.std(train_data.target_xyz, axis=(0, 1), dtype=np.float64),
        eps,
    )
    lambda_min = float(np.min(train_data.lambda_grid))
    lambda_max = float(np.max(train_data.lambda_grid))
    if not lambda_max > lambda_min:
        raise ValueError("lambda_grid must span a positive interval.")

    return ReconstructionNormalizationStats(
        input_xyz_mean=[float(value) for value in input_mean],
        input_xyz_std=[float(value) for value in input_std],
        target_xyz_mean=[float(value) for value in target_mean],
        target_xyz_std=[float(value) for value in target_std],
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        eps=eps,
    )


def _normalization_arrays(
    normalization: ReconstructionNormalizationStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将保存的统计量转换为 float32 数组。"""
    input_mean = np.asarray(normalization.input_xyz_mean, dtype=np.float32)
    input_std = np.asarray(normalization.input_xyz_std, dtype=np.float32)
    target_mean = np.asarray(normalization.target_xyz_mean, dtype=np.float32)
    target_std = np.asarray(normalization.target_xyz_std, dtype=np.float32)
    for name, value in {
        "input_xyz_mean": input_mean,
        "input_xyz_std": input_std,
        "target_xyz_mean": target_mean,
        "target_xyz_std": target_std,
    }.items():
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain three finite values.")
    if np.any(input_std <= 0.0) or np.any(target_std <= 0.0):
        raise ValueError("Normalization standard deviations must be positive.")
    return input_mean, input_std, target_mean, target_std


def build_reconstruction_model_input(
    sparse_data: SparseTrajectoryData,
    normalization: ReconstructionNormalizationStats,
) -> np.ndarray:
    """构造不含 Q 的 [N,T,5] FNO1D 输入。"""
    sparse = _validate_finite_real_array(
        sparse_data.sparse_xyz,
        "sparse_xyz",
    )
    observed_mask = np.asarray(sparse_data.observed_mask)
    if observed_mask.shape != sparse.shape[:2] + (1,):
        raise ValueError("observed_mask must have shape [N,T,1].")
    if observed_mask.dtype != np.bool_:
        raise TypeError("observed_mask must use bool dtype.")

    input_mean, input_std, _, _ = _normalization_arrays(normalization)
    normalized_sparse = np.zeros(sparse.shape, dtype=np.float32)
    observed = observed_mask[..., 0]
    normalized_sparse[observed] = (
        sparse[observed].astype(np.float32) - input_mean
    ) / input_std

    lambda_scaled = (
        sparse_data.lambda_grid.astype(np.float32)
        - np.float32(normalization.lambda_min)
    ) / np.float32(normalization.lambda_max - normalization.lambda_min)
    if not np.all(np.isfinite(lambda_scaled)):
        raise ValueError("Scaled lambda_grid must contain only finite values.")
    lambda_channel = np.broadcast_to(
        lambda_scaled[None, :, None],
        sparse.shape[:2] + (1,),
    )

    return np.concatenate(
        (
            normalized_sparse,
            observed_mask.astype(np.float32),
            lambda_channel.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def normalize_target_xyz(
    target_xyz: np.ndarray,
    normalization: ReconstructionNormalizationStats,
) -> np.ndarray:
    """使用训练集目标统计量标准化完整轨道目标。"""
    target = _validate_finite_real_array(target_xyz, "target_xyz")
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target_xyz must have shape [N,T,3].")
    _, _, target_mean, target_std = _normalization_arrays(normalization)
    return ((target.astype(np.float32) - target_mean) / target_std).astype(
        np.float32
    )


def denormalize_prediction_tensor(
    prediction_normalized: torch.Tensor,
    normalization: ReconstructionNormalizationStats,
) -> torch.Tensor:
    """将模型输出从标准化目标空间还原到 raw xyz 空间。"""
    if prediction_normalized.ndim != 3 or prediction_normalized.shape[-1] != 3:
        raise ValueError("prediction_normalized must have shape [B,T,3].")
    _, _, target_mean, target_std = _normalization_arrays(normalization)
    mean = torch.as_tensor(
        target_mean,
        dtype=prediction_normalized.dtype,
        device=prediction_normalized.device,
    )
    std = torch.as_tensor(
        target_std,
        dtype=prediction_normalized.dtype,
        device=prediction_normalized.device,
    )
    return prediction_normalized * std + mean


def hidden_only_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> torch.Tensor:
    """计算标准化空间的 hidden-only MSE。"""
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [B,T,3].")
    if target.shape != prediction.shape:
        raise ValueError("target must have the same shape as prediction.")
    expected_mask_shape = prediction.shape[:2] + (1,)
    if tuple(hidden_mask.shape) != tuple(expected_mask_shape):
        raise ValueError("hidden_mask must have shape [B,T,1].")

    hidden_count = hidden_mask.sum()
    if int(hidden_count.detach().cpu()) == 0:
        raise ValueError("hidden_mask must contain at least one hidden point.")

    mask = hidden_mask.to(dtype=prediction.dtype)
    squared_error = (prediction - target).square()
    return (squared_error * mask).sum() / (3 * hidden_count.to(prediction.dtype))


def restore_observed_points_tensor(
    prediction_raw: torch.Tensor,
    sparse_raw: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    """用原始稀疏观测值精确覆盖预测中的 observed positions。"""
    if prediction_raw.ndim != 3 or prediction_raw.shape[-1] != 3:
        raise ValueError("prediction_raw must have shape [B,T,3].")
    if sparse_raw.shape != prediction_raw.shape:
        raise ValueError("sparse_raw must have the same shape as prediction_raw.")
    expected_mask_shape = prediction_raw.shape[:2] + (1,)
    if tuple(observed_mask.shape) != tuple(expected_mask_shape):
        raise ValueError("observed_mask must have shape [B,T,1].")
    return torch.where(observed_mask.bool(), sparse_raw, prediction_raw)


@torch.no_grad()
def evaluate_reconstruction_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str | torch.device,
    normalization: ReconstructionNormalizationStats,
) -> tuple[float, dict[str, object]]:
    """运行验证或测试并返回标准化损失与 raw-space 指标。"""
    model.eval()
    total_squared_error = 0.0
    total_hidden_scalar_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    hidden_masks: list[np.ndarray] = []

    for batch in loader:
        (
            model_input,
            target_normalized,
            target_raw,
            sparse_raw,
            observed_mask,
            hidden_mask,
        ) = batch
        model_input = model_input.to(device)
        target_normalized = target_normalized.to(device)
        hidden_mask_device = hidden_mask.to(device)

        prediction_normalized = model(model_input)
        batch_loss = hidden_only_mse_loss(
            prediction_normalized,
            target_normalized,
            hidden_mask_device,
        )
        hidden_scalar_count = int(hidden_mask_device.sum().detach().cpu()) * 3
        total_squared_error += float(batch_loss.detach().cpu()) * hidden_scalar_count
        total_hidden_scalar_count += hidden_scalar_count

        prediction_raw = denormalize_prediction_tensor(
            prediction_normalized,
            normalization,
        )
        prediction_restored = restore_observed_points_tensor(
            prediction_raw,
            sparse_raw.to(device),
            observed_mask.to(device),
        )
        predictions.append(prediction_restored.detach().cpu().numpy())
        targets.append(target_raw.numpy())
        hidden_masks.append(hidden_mask.numpy())

    if total_hidden_scalar_count == 0:
        raise RuntimeError("Evaluation loader contains no hidden points.")

    metrics = compute_hidden_masked_metrics(
        prediction_xyz=np.concatenate(predictions, axis=0),
        target_xyz=np.concatenate(targets, axis=0),
        hidden_mask=np.concatenate(hidden_masks, axis=0),
    )
    return total_squared_error / total_hidden_scalar_count, metrics
