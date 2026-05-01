# ==========================================================
# File: src/training/fno1d/dataset_loader_1d.py
#
# 功能简介：
# 1. 读取 dataset.npz；
# 2. 取出原始参数输入 X、监督目标 Y、lambda_grid 等内容；
# 3. 调用 input_builder_1d.py 将原始 X 转换为 FNO1d 真正使用的输入张量；
# 4. 提供 PyTorch Dataset / DataLoader；
# 5. 提供训练、验证、测试集的统一加载接口。
#
# 依赖关系：
# - 依赖 src/common/io_utils.py
# - 依赖 src/common/paths.py
# - 依赖 src/training/fno1d/input_builder_1d.py
# - 被后续 trainer.py 和 train_model.py 调用
#
# 重要说明：
# - dataset.npz 中保存的是原始参数输入 X 和目标输出 Y；
# - 本文件负责把 X 转换成 FNO1d 真正使用的 [N,T,C] 输入；
# - Y 保持为 [N,T,3]，作为监督目标。
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.io_utils import load_npz
from src.common.paths import get_task_dataset_npz_path
from src.training.fno1d.input_builder_1d import build_fno1d_input_array


# ==========================================================
# 一、数据集对象
# ==========================================================

class OrbitSequenceDataset1D(Dataset):
    """
    一维轨道序列数据集。

    输入：
    - x_model: [N, T, C]
    - y:       [N, T, 3]

    返回：
    - 单个样本:
        x: [T, C]
        y: [T, 3]
    """

    def __init__(self, x_model: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        if x_model.ndim != 3:
            raise ValueError(f"x_model 必须是三维数组 [N,T,C]，当前 shape={x_model.shape}")
        if y.ndim != 3:
            raise ValueError(f"y 必须是三维数组 [N,T,3]，当前 shape={y.shape}")

        if x_model.shape[0] != y.shape[0]:
            raise ValueError(
                "x_model 和 y 的样本数必须一致："
                f"x_model.shape[0]={x_model.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        if x_model.shape[1] != y.shape[1]:
            raise ValueError(
                "x_model 和 y 的序列长度 T 必须一致："
                f"x_model.shape[1]={x_model.shape[1]}, y.shape[1]={y.shape[1]}"
            )

        self.x_model = x_model.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.x_model.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x_model[idx])   # [T,C]
        y = torch.from_numpy(self.y[idx])         # [T,3]
        return x, y


# ==========================================================
# 二、打包读取结果
# ==========================================================

@dataclass
class LoadedDatasetBundle1D:
    """
    一维训练任务加载后的数据打包对象。

    用途：
    - 让训练入口一次性拿到 train/val/test 数据及其关键信息
    """
    vary_params_order: list[str]
    lambda_grid: np.ndarray

    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    x_train_model: np.ndarray
    x_val_model: np.ndarray
    x_test_model: np.ndarray


# ==========================================================
# 三、读取 dataset.npz
# ==========================================================

def load_task_dataset_1d(task_name: str) -> LoadedDatasetBundle1D:
    """
    根据 task_name 读取 dataset.npz，并构造 FNO1d 模型输入。

    返回：
    - LoadedDatasetBundle1D
    """
    dataset_path = get_task_dataset_npz_path(task_name)
    data = load_npz(dataset_path)

    vary_params_order = _parse_vary_params_order(data["vary_params_order"])
    lambda_grid = np.asarray(data["lambda_grid"], dtype=np.float32)

    x_train_raw = np.asarray(data["x_train"], dtype=np.float32)
    x_val_raw = np.asarray(data["x_val"], dtype=np.float32)
    x_test_raw = np.asarray(data["x_test"], dtype=np.float32)

    y_train = np.asarray(data["y_train"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)

    x_train_model = build_fno1d_input_array(x_train_raw, lambda_grid)
    x_val_model = build_fno1d_input_array(x_val_raw, lambda_grid)
    x_test_model = build_fno1d_input_array(x_test_raw, lambda_grid)

    return LoadedDatasetBundle1D(
        vary_params_order=vary_params_order,
        lambda_grid=lambda_grid,
        x_train_raw=x_train_raw,
        x_val_raw=x_val_raw,
        x_test_raw=x_test_raw,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        x_train_model=x_train_model,
        x_val_model=x_val_model,
        x_test_model=x_test_model,
    )


def _parse_vary_params_order(value: np.ndarray) -> list[str]:
    """
    将 npz 中读出的 vary_params_order 转成普通字符串列表。
    """
    return [str(x) for x in value.tolist()]


# ==========================================================
# 四、构造 Dataset
# ==========================================================

def build_datasets_1d(
    task_name: str,
) -> tuple[OrbitSequenceDataset1D, OrbitSequenceDataset1D, OrbitSequenceDataset1D, LoadedDatasetBundle1D]:
    """
    根据 task_name 构造 train / val / test 三个 Dataset。

    返回：
    - train_dataset
    - val_dataset
    - test_dataset
    - bundle
    """
    bundle = load_task_dataset_1d(task_name)

    train_dataset = OrbitSequenceDataset1D(bundle.x_train_model, bundle.y_train)
    val_dataset = OrbitSequenceDataset1D(bundle.x_val_model, bundle.y_val)
    test_dataset = OrbitSequenceDataset1D(bundle.x_test_model, bundle.y_test)

    return train_dataset, val_dataset, test_dataset, bundle


# ==========================================================
# 五、构造 DataLoader
# ==========================================================

def build_dataloaders_1d(
    task_name: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LoadedDatasetBundle1D]:
    """
    根据 task_name 构造 train / val / test 三个 DataLoader。

    参数：
    - task_name
    - batch_size
    - num_workers

    返回：
    - train_loader
    - val_loader
    - test_loader
    - bundle
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前得到：{batch_size}")

    train_dataset, val_dataset, test_dataset, bundle = build_datasets_1d(task_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, bundle


# ==========================================================
# 六、摘要信息
# ==========================================================

def summarize_loaded_bundle_1d(bundle: LoadedDatasetBundle1D) -> dict[str, Any]:
    """
    将加载后的数据集信息整理成摘要字典。
    """
    return {
        "vary_params_order": bundle.vary_params_order,
        "lambda_steps": int(bundle.lambda_grid.shape[0]),
        "x_train_raw_shape": tuple(bundle.x_train_raw.shape),
        "x_val_raw_shape": tuple(bundle.x_val_raw.shape),
        "x_test_raw_shape": tuple(bundle.x_test_raw.shape),
        "x_train_model_shape": tuple(bundle.x_train_model.shape),
        "x_val_model_shape": tuple(bundle.x_val_model.shape),
        "x_test_model_shape": tuple(bundle.x_test_model.shape),
        "y_train_shape": tuple(bundle.y_train.shape),
        "y_val_shape": tuple(bundle.y_val.shape),
        "y_test_shape": tuple(bundle.y_test.shape),
    }