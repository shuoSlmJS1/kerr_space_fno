# ==========================================================
# File: src/training/fno2d/dataset_loader_2d.py
#
# 功能简介：
# 1. 读取已有的单参数任务 dataset.npz；
# 2. 检查该任务是否为单变化参数任务；
# 3. 将原始 1D trajectory dataset 转换成 FNO2d 的二维场数据；
# 4. 支持 Q-only / a-only / E-only / Lz-only 等单参数任务；
# 5. 支持 FNO2d 输入/输出标准化：
#       normalization = "none"
#       normalization = "standard"
#
# 输入数据：
#   x_train: [N_train, 1]
#   y_train: [N_train, T, 3]
#   lambda_grid: [T]
#
# 输出给模型：
#   x_2d: [1, N_train, T, 2]
#   y_2d: [1, N_train, T, 3]
#
# 数学结构：
#   (p, lambda) -> (x, y, z)
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.io_utils import load_json, load_npz
from src.common.paths import get_task_dataset_npz_path, get_task_meta_json_path
from src.training.fno2d.input_builder_2d import (
    FNO2dFieldData,
    build_param_lambda_train_val_test_fields,
    summarize_fno2d_fields,
)
from src.training.fno2d.normalization_2d import (
    FieldNormalizationStats,
    apply_normalization_to_field_pair,
    compute_field_normalization_stats,
    summarize_normalization_stats,
)


# ==========================================================
# 一、数据包对象
# ==========================================================

@dataclass
class FNO2dDatasetBundle:
    """
    FNO2d 数据包。

    字段说明：
    - task_name:
        当前任务名

    - param_name:
        单变化参数名，例如 Q / a / E / Lz

    - train_field / val_field / test_field:
        train / val / test 对应的二维场数据

    - lambda_grid:
        lambda 网格

    - vary_params_order:
        原任务中的变化参数顺序

    - normalization:
        当前归一化方法："none" 或 "standard"

    - normalization_stats:
        从 train split 计算出的归一化统计量。
        val/test 必须使用同一组统计量。
    """
    task_name: str
    param_name: str
    train_field: FNO2dFieldData
    val_field: FNO2dFieldData
    test_field: FNO2dFieldData
    lambda_grid: np.ndarray
    vary_params_order: list[str]
    normalization: str
    normalization_stats: FieldNormalizationStats


# ==========================================================
# 二、PyTorch Dataset
# ==========================================================

class FNO2dFieldDataset(Dataset):
    """
    FNO2d 二维场 Dataset。

    注意：
    - 第一版中，一个完整的二维场就是一个样本；
    - 因此通常 len(dataset)=1；
    - DataLoader 的 batch_size 通常也应设置为 1。
    """

    def __init__(self, field: FNO2dFieldData) -> None:
        self.x = torch.from_numpy(field.x_2d).float()
        self.y = torch.from_numpy(field.y_2d).float()

        if self.x.ndim != 4:
            raise ValueError(f"x 必须是 [B,H,W,C]，当前 shape={tuple(self.x.shape)}")

        if self.y.ndim != 4:
            raise ValueError(f"y 必须是 [B,H,W,3]，当前 shape={tuple(self.y.shape)}")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"x 和 y 的 batch 维度必须一致，当前 x={tuple(self.x.shape)}, "
                f"y={tuple(self.y.shape)}"
            )

    def __len__(self) -> int:
        """
        返回样本数。

        第一版中通常为 1，因为一个完整二维场就是一个样本。
        """
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
        - x[idx]: [H,W,2]
        - y[idx]: [H,W,3]
        """
        return self.x[idx], self.y[idx]


# ==========================================================
# 三、原始数据读取与检查
# ==========================================================

def load_raw_task_arrays(task_name: str) -> dict[str, np.ndarray]:
    """
    读取 dataset.npz。
    """
    dataset_path = get_task_dataset_npz_path(task_name)
    return load_npz(dataset_path)


def load_task_vary_params_order(task_name: str) -> list[str]:
    """
    从 meta.json 中读取 vary_params_order。

    优先读取 meta.json 顶层 vary_params_order。
    如果没有，则尝试从 task_spec 中读取。
    """
    meta_path = get_task_meta_json_path(task_name)
    meta = load_json(meta_path)

    if "vary_params_order" in meta:
        return list(meta["vary_params_order"])

    task_spec = meta.get("task_spec", {})
    if "vary_params" in task_spec:
        return list(task_spec["vary_params"])

    raise KeyError(
        f"无法从 meta.json 中读取 vary_params_order 或 task_spec.vary_params: {meta_path}"
    )


def validate_single_param_task(vary_params_order: list[str]) -> str:
    """
    检查当前任务是否是单参数任务。

    返回：
    - param_name
    """
    if len(vary_params_order) != 1:
        raise ValueError(
            "FNO2d 第一版只支持单变化参数任务，"
            f"当前 vary_params_order={vary_params_order}"
        )

    param_name = str(vary_params_order[0])
    if len(param_name.strip()) == 0:
        raise ValueError(f"参数名为空：vary_params_order={vary_params_order}")

    return param_name


def validate_npz_keys(data: dict[str, np.ndarray]) -> None:
    """
    检查 dataset.npz 是否包含必要字段。
    """
    required_keys = [
        "x_train",
        "x_val",
        "x_test",
        "y_train",
        "y_val",
        "y_test",
        "lambda_grid",
    ]

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"dataset.npz 缺少字段：{missing}")


# ==========================================================
# 四、归一化后的 field 重建
# ==========================================================

def rebuild_field_with_normalized_arrays(
    field: FNO2dFieldData,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
) -> FNO2dFieldData:
    """
    用归一化后的 x/y 数组重建 FNO2dFieldData。

    其他网格信息保持不变。
    """
    return FNO2dFieldData(
        x_2d=x_norm,
        y_2d=y_norm,
        param_grid=field.param_grid,
        lambda_grid=field.lambda_grid,
        param_name=field.param_name,
        num_param=field.num_param,
        num_lambda=field.num_lambda,
        in_dim=field.in_dim,
        out_dim=field.out_dim,
    )


def apply_normalization_to_fields(
    fields: dict[str, FNO2dFieldData],
    normalization: str,
) -> tuple[dict[str, FNO2dFieldData], FieldNormalizationStats]:
    """
    对 train / val / test 三个 field 应用归一化。

    关键原则：
    - 统计量只从 train field 计算；
    - val/test 使用 train 统计量；
    - 不允许从 val/test 计算自己的统计量。
    """
    stats = compute_field_normalization_stats(
        x_train=fields["train"].x_2d,
        y_train=fields["train"].y_2d,
        method=normalization,
    )

    normalized_fields: dict[str, FNO2dFieldData] = {}

    for split_name, field in fields.items():
        x_norm, y_norm = apply_normalization_to_field_pair(
            x=field.x_2d,
            y=field.y_2d,
            stats=stats,
        )

        normalized_fields[split_name] = rebuild_field_with_normalized_arrays(
            field=field,
            x_norm=x_norm,
            y_norm=y_norm,
        )

    return normalized_fields, stats


# ==========================================================
# 五、构造 FNO2d 数据包
# ==========================================================

def load_fno2d_dataset_bundle(
    task_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    normalization: str = "none",
) -> FNO2dDatasetBundle:
    """
    读取单参数任务，并转换为 FNO2d 数据包。

    参数：
    - normalization:
        "none" 或 "standard"
    """
    data = load_raw_task_arrays(task_name)
    validate_npz_keys(data)

    vary_params_order = load_task_vary_params_order(task_name)
    param_name = validate_single_param_task(vary_params_order)

    fields = build_param_lambda_train_val_test_fields(
        x_train_raw=data["x_train"],
        y_train=data["y_train"],
        x_val_raw=data["x_val"],
        y_val=data["y_val"],
        x_test_raw=data["x_test"],
        y_test=data["y_test"],
        lambda_grid=data["lambda_grid"],
        param_name=param_name,
        sort_param=sort_param,
        dtype=dtype,
    )

    fields, normalization_stats = apply_normalization_to_fields(
        fields=fields,
        normalization=normalization,
    )

    return FNO2dDatasetBundle(
        task_name=task_name,
        param_name=param_name,
        train_field=fields["train"],
        val_field=fields["val"],
        test_field=fields["test"],
        lambda_grid=data["lambda_grid"],
        vary_params_order=vary_params_order,
        normalization=str(normalization),
        normalization_stats=normalization_stats,
    )


# ==========================================================
# 六、构造 Dataset / DataLoader
# ==========================================================

def build_fno2d_datasets(
    task_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    normalization: str = "none",
) -> tuple[FNO2dFieldDataset, FNO2dFieldDataset, FNO2dFieldDataset, FNO2dDatasetBundle]:
    """
    构造 train / val / test Dataset。
    """
    bundle = load_fno2d_dataset_bundle(
        task_name=task_name,
        sort_param=sort_param,
        dtype=dtype,
        normalization=normalization,
    )

    train_dataset = FNO2dFieldDataset(bundle.train_field)
    val_dataset = FNO2dFieldDataset(bundle.val_field)
    test_dataset = FNO2dFieldDataset(bundle.test_field)

    return train_dataset, val_dataset, test_dataset, bundle


def build_fno2d_dataloaders(
    task_name: str,
    batch_size: int = 1,
    num_workers: int = 0,
    sort_param: bool = True,
    normalization: str = "none",
) -> tuple[DataLoader, DataLoader, DataLoader, FNO2dDatasetBundle]:
    """
    构造 train / val / test DataLoader。

    注意：
    - 第一版 FNO2d 中，每个 split 只有一个二维场样本；
    - 推荐 batch_size=1；
    - 如果 batch_size > 1，通常也没有意义，因为 dataset 长度一般就是 1。
    """
    train_dataset, val_dataset, test_dataset, bundle = build_fno2d_datasets(
        task_name=task_name,
        sort_param=sort_param,
        normalization=normalization,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, bundle


# ==========================================================
# 七、摘要函数
# ==========================================================

def summarize_fno2d_bundle(bundle: FNO2dDatasetBundle) -> dict[str, Any]:
    """
    输出 FNO2d 数据包摘要。
    """
    field_summary = summarize_fno2d_fields(
        {
            "train": bundle.train_field,
            "val": bundle.val_field,
            "test": bundle.test_field,
        }
    )

    return {
        "task_name": bundle.task_name,
        "param_name": bundle.param_name,
        "vary_params_order": bundle.vary_params_order,
        "normalization": bundle.normalization,
        "normalization_stats": summarize_normalization_stats(bundle.normalization_stats),
        "train": field_summary["train"],
        "val": field_summary["val"],
        "test": field_summary["test"],
    }