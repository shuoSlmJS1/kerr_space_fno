# ==========================================================
# File: src/training/fno2d/dataset_loader_2d.py
#
# 功能简介：
# 1. 读取已有的单参数任务 dataset.npz；
# 2. 将原始 1D trajectory dataset 转换成 FNO2d 的二维场数据；
# 3. 支持旧模式：单 task -> [1,N,T,2]；
# 4. 支持新模式：多 cfg task -> [B_cfg,N,T,C]，例如 [Q,lambda,a]；
# 5. 支持 target transform 和 normalization。
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
from src.training.fno2d.target_transform_2d import (
    TargetTransformConfig,
    summarize_target_transform_config,
    transform_output_field,
)


@dataclass
class FNO2dDatasetBundle:
    """FNO2d 数据包。"""
    task_name: str
    param_name: str
    train_field: FNO2dFieldData
    val_field: FNO2dFieldData
    test_field: FNO2dFieldData
    lambda_grid: np.ndarray
    vary_params_order: list[str]
    normalization: str
    normalization_stats: FieldNormalizationStats
    target_transform: str
    target_transform_config: TargetTransformConfig
    task_names: list[str] | None = None
    cfg_param_name: str | None = None


class FNO2dFieldDataset(Dataset):
    """
    FNO2d 二维场 Dataset。

    单 cfg 时 len(dataset)=1。
    多 cfg 时 len(dataset)=B_cfg，每个 cfg field 是一个样本。
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
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ==========================================================
# 原始数据读取与检查
# ==========================================================

def load_raw_task_arrays(task_name: str) -> dict[str, np.ndarray]:
    dataset_path = get_task_dataset_npz_path(task_name)
    return load_npz(dataset_path)


def load_task_meta(task_name: str) -> dict[str, Any]:
    meta_path = get_task_meta_json_path(task_name)
    return load_json(meta_path)


def load_task_vary_params_order(task_name: str) -> list[str]:
    meta = load_task_meta(task_name)

    if "vary_params_order" in meta:
        return list(meta["vary_params_order"])

    task_spec = meta.get("task_spec", {})
    if "vary_params" in task_spec:
        return list(task_spec["vary_params"])

    raise KeyError(f"无法从 meta.json 中读取 vary_params_order 或 task_spec.vary_params: {task_name}")


def load_task_fixed_params(task_name: str) -> dict[str, Any]:
    """读取 task_spec.fixed_params，用于提取多 cfg 条件参数，例如 a。"""
    meta = load_task_meta(task_name)
    task_spec = meta.get("task_spec", {})

    if "fixed_params" in task_spec:
        return dict(task_spec["fixed_params"])

    if "fixed_kerr_params" in meta:
        fixed = dict(meta["fixed_kerr_params"])
        if "initial_state" in meta:
            fixed.update(dict(meta["initial_state"]))
        return fixed

    raise KeyError(f"无法从 meta.json 中读取 fixed_params: {task_name}")


def validate_single_param_task(vary_params_order: list[str]) -> str:
    if len(vary_params_order) != 1:
        raise ValueError(
            "FNO2d 当前多场版本仍只支持单变化参数任务，"
            f"当前 vary_params_order={vary_params_order}"
        )

    param_name = str(vary_params_order[0])
    if len(param_name.strip()) == 0:
        raise ValueError(f"参数名为空：vary_params_order={vary_params_order}")
    return param_name


def validate_npz_keys(data: dict[str, np.ndarray]) -> None:
    required_keys = [
        "x_train", "x_val", "x_test",
        "y_train", "y_val", "y_test",
        "lambda_grid",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"dataset.npz 缺少字段：{missing}")


# ==========================================================
# field 重建 / stack 工具
# ==========================================================

def rebuild_field_with_arrays(
    field: FNO2dFieldData,
    x_new: np.ndarray | None = None,
    y_new: np.ndarray | None = None,
) -> FNO2dFieldData:
    return FNO2dFieldData(
        x_2d=field.x_2d if x_new is None else x_new,
        y_2d=field.y_2d if y_new is None else y_new,
        param_grid=field.param_grid,
        lambda_grid=field.lambda_grid,
        param_name=field.param_name,
        num_param=field.num_param,
        num_lambda=field.num_lambda,
        in_dim=int((field.x_2d if x_new is None else x_new).shape[-1]),
        out_dim=field.out_dim,
        input_channel_names=field.input_channel_names,
        cfg_param_names=field.cfg_param_names,
        cfg_values=field.cfg_values,
    )


def stack_field_list(fields: list[FNO2dFieldData], split_name: str) -> FNO2dFieldData:
    """把多个 [1,N,T,C] field 堆成 [B_cfg,N,T,C]。"""
    if len(fields) == 0:
        raise ValueError(f"{split_name}: fields 为空，无法 stack。")

    ref = fields[0]
    x_list = []
    y_list = []
    cfg_values_list = []

    for i, field in enumerate(fields):
        if field.x_2d.shape[0] != 1 or field.y_2d.shape[0] != 1:
            raise ValueError(
                f"{split_name}: 第 {i} 个 field 不是单 cfg field，"
                f"x={field.x_2d.shape}, y={field.y_2d.shape}"
            )
        if field.x_2d.shape[1:] != ref.x_2d.shape[1:]:
            raise ValueError(
                f"{split_name}: 第 {i} 个 field x shape 不一致，"
                f"ref={ref.x_2d.shape}, current={field.x_2d.shape}"
            )
        if field.y_2d.shape[1:] != ref.y_2d.shape[1:]:
            raise ValueError(
                f"{split_name}: 第 {i} 个 field y shape 不一致，"
                f"ref={ref.y_2d.shape}, current={field.y_2d.shape}"
            )
        if not np.allclose(field.param_grid, ref.param_grid):
            raise ValueError(
                f"{split_name}: 第 {i} 个 field 的 param_grid 与第 0 个不一致。"
                "当前版本要求每个 cfg 的参数网格完全一致。"
            )
        if not np.allclose(field.lambda_grid, ref.lambda_grid):
            raise ValueError(
                f"{split_name}: 第 {i} 个 field 的 lambda_grid 与第 0 个不一致。"
            )

        x_list.append(field.x_2d)
        y_list.append(field.y_2d)
        if field.cfg_values is not None:
            cfg_values_list.append(field.cfg_values)

    x_stacked = np.concatenate(x_list, axis=0)
    y_stacked = np.concatenate(y_list, axis=0)

    cfg_values = None
    if cfg_values_list:
        cfg_values = np.concatenate(cfg_values_list, axis=0)

    return FNO2dFieldData(
        x_2d=x_stacked,
        y_2d=y_stacked,
        param_grid=ref.param_grid,
        lambda_grid=ref.lambda_grid,
        param_name=ref.param_name,
        num_param=ref.num_param,
        num_lambda=ref.num_lambda,
        in_dim=int(x_stacked.shape[-1]),
        out_dim=ref.out_dim,
        input_channel_names=ref.input_channel_names,
        cfg_param_names=ref.cfg_param_names,
        cfg_values=cfg_values,
    )


# ==========================================================
# target transform / normalization
# ==========================================================

def apply_target_transform_to_fields(
    fields: dict[str, FNO2dFieldData],
    target_transform: str,
    lambda_reference_index: int = 0,
) -> tuple[dict[str, FNO2dFieldData], TargetTransformConfig]:
    config = TargetTransformConfig(
        mode=str(target_transform),
        lambda_reference_index=int(lambda_reference_index),
    )

    transformed_fields: dict[str, FNO2dFieldData] = {}
    for split_name, field in fields.items():
        y_transformed = transform_output_field(y=field.y_2d, config=config)
        transformed_fields[split_name] = rebuild_field_with_arrays(field=field, y_new=y_transformed)

    return transformed_fields, config


def apply_normalization_to_fields(
    fields: dict[str, FNO2dFieldData],
    normalization: str,
) -> tuple[dict[str, FNO2dFieldData], FieldNormalizationStats]:
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
        normalized_fields[split_name] = rebuild_field_with_arrays(
            field=field,
            x_new=x_norm,
            y_new=y_norm,
        )

    return normalized_fields, stats


# ==========================================================
# 构造单 task / 多 cfg bundle
# ==========================================================

def load_fno2d_dataset_bundle(
    task_name: str,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    normalization: str = "none",
    target_transform: str = "raw",
    lambda_reference_index: int = 0,
) -> FNO2dDatasetBundle:
    """读取单参数任务，并转换为 FNO2d 数据包。"""
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

    fields, target_transform_config = apply_target_transform_to_fields(
        fields=fields,
        target_transform=target_transform,
        lambda_reference_index=lambda_reference_index,
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
        target_transform=str(target_transform),
        target_transform_config=target_transform_config,
        task_names=[task_name],
        cfg_param_name=None,
    )


def load_fno2d_multicfg_dataset_bundle(
    task_names: list[str],
    cfg_param_name: str,
    output_task_name: str | None = None,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    normalization: str = "none",
    target_transform: str = "raw",
    lambda_reference_index: int = 0,
) -> FNO2dDatasetBundle:
    """
    读取多个同结构单参数任务，并堆叠为多 cfg 2D field。

    例子：
    - task_names = [cfg_a042, cfg_a046, cfg_a050, ...]
    - cfg_param_name = "a"
    - 变化参数仍为 Q
    - 输入通道变成 [Q, lambda, a]
    """
    if len(task_names) < 2:
        raise ValueError("多 cfg 模式至少需要 2 个 task_names。")
    if not cfg_param_name or len(str(cfg_param_name).strip()) == 0:
        raise ValueError("多 cfg 模式必须提供 cfg_param_name，例如 a。")

    train_fields: list[FNO2dFieldData] = []
    val_fields: list[FNO2dFieldData] = []
    test_fields: list[FNO2dFieldData] = []

    ref_vary_params_order: list[str] | None = None
    ref_param_name: str | None = None
    ref_lambda_grid: np.ndarray | None = None

    for task_name in task_names:
        data = load_raw_task_arrays(task_name)
        validate_npz_keys(data)

        vary_params_order = load_task_vary_params_order(task_name)
        param_name = validate_single_param_task(vary_params_order)
        fixed_params = load_task_fixed_params(task_name)

        if cfg_param_name not in fixed_params:
            raise KeyError(
                f"task={task_name} 的 fixed_params 中没有 {cfg_param_name!r}，"
                f"可用 keys={list(fixed_params.keys())}"
            )

        if ref_vary_params_order is None:
            ref_vary_params_order = vary_params_order
            ref_param_name = param_name
            ref_lambda_grid = data["lambda_grid"]
        else:
            if vary_params_order != ref_vary_params_order:
                raise ValueError(
                    f"多 cfg 任务 vary_params_order 必须一致，"
                    f"ref={ref_vary_params_order}, current={vary_params_order}, task={task_name}"
                )
            if not np.allclose(data["lambda_grid"], ref_lambda_grid):
                raise ValueError(f"多 cfg 任务 lambda_grid 必须一致，当前 task={task_name}")

        cfg_value = float(fixed_params[cfg_param_name])
        extra_channels = {str(cfg_param_name): cfg_value}

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
            extra_constant_channels=extra_channels,
        )

        train_fields.append(fields["train"])
        val_fields.append(fields["val"])
        test_fields.append(fields["test"])

    raw_fields = {
        "train": stack_field_list(train_fields, split_name="train"),
        "val": stack_field_list(val_fields, split_name="val"),
        "test": stack_field_list(test_fields, split_name="test"),
    }

    raw_fields, target_transform_config = apply_target_transform_to_fields(
        fields=raw_fields,
        target_transform=target_transform,
        lambda_reference_index=lambda_reference_index,
    )
    raw_fields, normalization_stats = apply_normalization_to_fields(
        fields=raw_fields,
        normalization=normalization,
    )

    bundle_task_name = output_task_name or ("multi_cfg__" + "__".join(task_names))

    return FNO2dDatasetBundle(
        task_name=bundle_task_name,
        param_name=str(ref_param_name),
        train_field=raw_fields["train"],
        val_field=raw_fields["val"],
        test_field=raw_fields["test"],
        lambda_grid=ref_lambda_grid,
        vary_params_order=list(ref_vary_params_order),
        normalization=str(normalization),
        normalization_stats=normalization_stats,
        target_transform=str(target_transform),
        target_transform_config=target_transform_config,
        task_names=list(task_names),
        cfg_param_name=str(cfg_param_name),
    )


# ==========================================================
# 构造 Dataset / DataLoader
# ==========================================================

def build_fno2d_datasets(
    task_name: str | None = None,
    task_names: list[str] | None = None,
    cfg_param_name: str | None = None,
    output_task_name: str | None = None,
    sort_param: bool = True,
    dtype: np.dtype = np.float32,
    normalization: str = "none",
    target_transform: str = "raw",
    lambda_reference_index: int = 0,
) -> tuple[FNO2dFieldDataset, FNO2dFieldDataset, FNO2dFieldDataset, FNO2dDatasetBundle]:
    """构造 train / val / test Dataset。"""
    if task_names is not None and len(task_names) > 0:
        if cfg_param_name is None:
            raise ValueError("使用 --task-names 多 cfg 模式时，必须提供 --cfg-param-name。")
        bundle = load_fno2d_multicfg_dataset_bundle(
            task_names=list(task_names),
            cfg_param_name=str(cfg_param_name),
            output_task_name=output_task_name,
            sort_param=sort_param,
            dtype=dtype,
            normalization=normalization,
            target_transform=target_transform,
            lambda_reference_index=lambda_reference_index,
        )
    else:
        if task_name is None:
            raise ValueError("必须提供 task_name 或 task_names。")
        bundle = load_fno2d_dataset_bundle(
            task_name=str(task_name),
            sort_param=sort_param,
            dtype=dtype,
            normalization=normalization,
            target_transform=target_transform,
            lambda_reference_index=lambda_reference_index,
        )

    train_dataset = FNO2dFieldDataset(bundle.train_field)
    val_dataset = FNO2dFieldDataset(bundle.val_field)
    test_dataset = FNO2dFieldDataset(bundle.test_field)

    return train_dataset, val_dataset, test_dataset, bundle


def build_fno2d_dataloaders(
    task_name: str | None = None,
    task_names: list[str] | None = None,
    cfg_param_name: str | None = None,
    output_task_name: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    sort_param: bool = True,
    normalization: str = "none",
    target_transform: str = "raw",
    lambda_reference_index: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, FNO2dDatasetBundle]:
    """构造 train / val / test DataLoader。"""
    train_dataset, val_dataset, test_dataset, bundle = build_fno2d_datasets(
        task_name=task_name,
        task_names=task_names,
        cfg_param_name=cfg_param_name,
        output_task_name=output_task_name,
        sort_param=sort_param,
        normalization=normalization,
        target_transform=target_transform,
        lambda_reference_index=lambda_reference_index,
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
# 摘要函数
# ==========================================================

def summarize_fno2d_bundle(bundle: FNO2dDatasetBundle) -> dict[str, Any]:
    field_summary = summarize_fno2d_fields(
        {"train": bundle.train_field, "val": bundle.val_field, "test": bundle.test_field}
    )

    return {
        "task_name": bundle.task_name,
        "task_names": bundle.task_names,
        "cfg_param_name": bundle.cfg_param_name,
        "param_name": bundle.param_name,
        "vary_params_order": bundle.vary_params_order,
        "normalization": bundle.normalization,
        "normalization_stats": summarize_normalization_stats(bundle.normalization_stats),
        "target_transform": bundle.target_transform,
        "target_transform_config": summarize_target_transform_config(bundle.target_transform_config),
        "train": field_summary["train"],
        "val": field_summary["val"],
        "test": field_summary["test"],
    }
