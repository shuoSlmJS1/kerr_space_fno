"""R1：使用不变 FNO2D 架构进行多物理域长度前缀训练。"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_model_2d import (  # noqa: E402
    get_model_output_dirs,
    mse_loss,
    relative_l2_error,
    save_checkpoint_2d,
    save_json,
)
from src.common.paths import get_model_output_dir  # noqa: E402
from src.models.fno2d.fno2d import count_parameters  # noqa: E402
from src.models.registry_2d import build_model_2d, summarize_model_config_2d  # noqa: E402
from src.training.fno2d.dataset_loader_2d import (  # noqa: E402
    extract_data_seed,
    load_raw_task_arrays,
    load_task_meta,
    load_task_vary_params_order,
    validate_npz_keys,
    validate_single_param_task,
    validate_task_generation_meta,
)
from src.training.fno2d.input_builder_2d import build_param_lambda_input_field  # noqa: E402
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    apply_normalization_to_field_pair,
    compute_field_normalization_stats,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    transform_output_field,
)


EXPERIMENT_TYPE = "r1_variable_length_training"
REPAIR_CLASS = "TRAINING_PROTOCOL_REPAIR"
SOURCE_LENGTH = 1200
NORMALIZATION_METHOD = "standard"
TARGET_TRANSFORM_MODE = "raw"
INPUT_CHANNEL_NAMES = ["Q", "lambda"]
DEFAULT_TRAIN_LENGTHS = [600, 800, 1000, 1200]
DEFAULT_VALIDATION_LENGTHS = [700, 900, 1100, 1200]
BASELINE_MODEL_CONFIG: dict[str, Any] = {
    "model_type": "fno2d",
    "in_dim": 2,
    "out_dim": 3,
    "modes1": 16,
    "modes2": 32,
    "width": 64,
    "depth": 4,
    "hidden_dim": 128,
    "activation": "gelu",
}


@dataclass(frozen=True)
class CanonicalSplit:
    """按稳定升序 Q 重建的一个原始数据 split。"""

    q: np.ndarray
    truth: np.ndarray
    source_row_indices: np.ndarray


@dataclass(frozen=True)
class PrefixView:
    """一个同质长度、已标准化的完整 Q 场。"""

    length: int
    x: torch.Tensor
    y: torch.Tensor


def validate_safe_name(value: str, field_name: str) -> str:
    """阻止路径分隔符和歧义运行名。"""
    normalized = str(value).strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(f"{field_name} must contain only letters, digits, '.', '_' or '-'.")
    return normalized


def validate_length_protocol(
    train_lengths: Sequence[int],
    validation_lengths: Sequence[int],
    source_length: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """检查 R1 长度集合及 held-out interpolation 约束。"""

    def normalize(values: Sequence[int], name: str) -> tuple[int, ...]:
        result = tuple(int(value) for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty.")
        if any(value <= 0 for value in result):
            raise ValueError(f"{name} must contain strictly positive lengths.")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must contain unique lengths.")
        if any(value > source_length for value in result):
            raise ValueError(f"{name} cannot exceed source length {source_length}.")
        return tuple(sorted(result))

    if int(source_length) != SOURCE_LENGTH:
        raise ValueError(
            f"R1 requires one T{SOURCE_LENGTH} source dataset; got T{source_length}."
        )
    train = normalize(train_lengths, "train_lengths")
    validation = normalize(validation_lengths, "validation_lengths")
    if max(train) != source_length:
        raise ValueError("max(train_lengths) must equal the source length.")
    if source_length not in validation:
        raise ValueError("validation_lengths must include the full source length.")
    overlap = set(train).intersection(validation) - {source_length}
    if overlap:
        raise ValueError(
            "Held-out interpolation validation lengths must not occur in train_lengths: "
            f"{sorted(overlap)}"
        )
    return train, validation


def canonicalize_split(q_raw: np.ndarray, truth_raw: np.ndarray) -> CanonicalSplit:
    """以稳定排序保持 Q 身份，并将相同置换应用于 raw truth。"""
    q = np.asarray(q_raw)
    truth = np.asarray(truth_raw)
    if q.ndim != 2 or q.shape[1] != 1:
        raise ValueError(f"Q array must have shape [N,1], got {q.shape}.")
    if truth.ndim != 3 or truth.shape[0] != q.shape[0] or truth.shape[2] != 3:
        raise ValueError(f"Truth must have shape [N,T,3], got {truth.shape}.")
    order = np.argsort(q[:, 0], kind="stable")
    q_sorted = np.asarray(q[order], dtype=np.float64)
    truth_sorted = np.asarray(truth[order])
    if np.unique(q_sorted[:, 0]).size != q_sorted.shape[0]:
        raise ValueError("Q identities must be unique within each split.")
    return CanonicalSplit(q=q_sorted, truth=truth_sorted, source_row_indices=order)


def assert_disjoint_q_identities(splits: Mapping[str, CanonicalSplit]) -> None:
    """保证同一 Q 轨道不跨 train/val/test。"""
    names = list(splits)
    for index, left_name in enumerate(names):
        left = splits[left_name].q[:, 0]
        for right_name in names[index + 1 :]:
            right = splits[right_name].q[:, 0]
            overlap = np.intersect1d(left, right)
            if overlap.size:
                raise ValueError(
                    f"Q identity leakage between {left_name} and {right_name}: "
                    f"{overlap.astype(float).tolist()}"
                )


def construct_strict_prefix(array: np.ndarray, length: int, lambda_axis: int) -> np.ndarray:
    """只切取严格物理前缀，不插值、不重采样、不补零。"""
    if length <= 0 or length > array.shape[lambda_axis]:
        raise ValueError(f"Invalid prefix length {length} for shape {array.shape}.")
    slices = [slice(None)] * array.ndim
    slices[lambda_axis] = slice(0, int(length))
    return np.asarray(array[tuple(slices)])


def build_full_source_fields(
    splits: Mapping[str, CanonicalSplit],
    lambda_grid: np.ndarray,
    param_name: str,
    target_transform_config: TargetTransformConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """构造未归一化的完整 T1200 canonical fields。"""
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, split in splits.items():
        field = build_param_lambda_input_field(
            x_raw=split.q,
            y=split.truth,
            lambda_grid=lambda_grid,
            param_name=param_name,
            sort_param=False,
            dtype=np.float32,
        )
        if field.input_channel_names != INPUT_CHANNEL_NAMES:
            raise ValueError(
                f"R1 requires input channels {INPUT_CHANNEL_NAMES}, got {field.input_channel_names}."
            )
        y_transformed = transform_output_field(
            y=field.y_2d,
            config=target_transform_config,
        )
        result[split_name] = (field.x_2d, y_transformed)
    return result


def fit_full_source_normalization(
    train_full_field: tuple[np.ndarray, np.ndarray],
) -> FieldNormalizationStats:
    """只从完整 T1200 training-Q field 拟合 baseline standard statistics。"""
    x_train, y_train = train_full_field
    if x_train.shape[2] != SOURCE_LENGTH or y_train.shape[2] != SOURCE_LENGTH:
        raise ValueError("Normalization must be fitted from the full T1200 train field.")
    return compute_field_normalization_stats(
        x_train=x_train,
        y_train=y_train,
        method=NORMALIZATION_METHOD,
    )


def build_prefix_views(
    full_field: tuple[np.ndarray, np.ndarray],
    lengths: Sequence[int],
    stats: FieldNormalizationStats,
    device: torch.device,
) -> dict[int, PrefixView]:
    """所有短前缀复用同一组 T1200 statistics。"""
    x_full, y_full = full_field
    views: dict[int, PrefixView] = {}
    for length in lengths:
        x_prefix = construct_strict_prefix(x_full, length=length, lambda_axis=2)
        y_prefix = construct_strict_prefix(y_full, length=length, lambda_axis=2)
        x_norm, y_norm = apply_normalization_to_field_pair(
            x=x_prefix,
            y=y_prefix,
            stats=stats,
        )
        views[int(length)] = PrefixView(
            length=int(length),
            x=torch.from_numpy(x_norm).to(device=device, dtype=torch.float32),
            y=torch.from_numpy(y_norm).to(device=device, dtype=torch.float32),
        )
    return views


def equal_weight_loss(losses: Sequence[torch.Tensor]) -> torch.Tensor:
    """每个物理域长度等权，不按网格点数加权。"""
    if not losses:
        raise ValueError("At least one length loss is required.")
    return torch.stack(tuple(losses)).mean()


def accumulate_multi_length_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_views: Mapping[int, PrefixView],
) -> dict[str, Any]:
    """每个长度各前反传一次，全部梯度累积后只更新一次。"""
    if not train_views:
        raise ValueError("train_views must not be empty.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    length_count = len(train_views)
    per_length_mse: dict[int, float] = {}
    per_length_relative_l2: dict[int, float] = {}
    for length in sorted(train_views):
        view = train_views[length]
        prediction = model(view.x)
        loss = mse_loss(prediction, view.y)
        (loss / length_count).backward()
        per_length_mse[length] = float(loss.detach().item())
        per_length_relative_l2[length] = float(
            relative_l2_error(prediction.detach(), view.y).item()
        )
    optimizer.step()
    return {
        "mse_by_length": per_length_mse,
        "relative_l2_by_length": per_length_relative_l2,
        "equal_weight_mse": float(np.mean(list(per_length_mse.values()))),
        "forward_backward_passes": int(length_count),
        "optimizer_steps": 1,
    }


def run_training_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_views: Mapping[int, PrefixView],
) -> dict[str, Any]:
    """一次 epoch 等于一次累积 optimizer update 和一次 scheduler update。"""
    metrics = accumulate_multi_length_update(model, optimizer, train_views)
    scheduler.step()
    metrics["scheduler_steps"] = 1
    return metrics


@torch.no_grad()
def evaluate_views(
    model: nn.Module,
    views: Mapping[int, PrefixView],
) -> dict[str, Any]:
    """逐长度独立验证，避免混合不同 W。"""
    if not views:
        raise ValueError("validation views must not be empty.")
    model.eval()
    mse_by_length: dict[int, float] = {}
    relative_l2_by_length: dict[int, float] = {}
    for length in sorted(views):
        view = views[length]
        prediction = model(view.x)
        mse_by_length[length] = float(mse_loss(prediction, view.y).item())
        relative_l2_by_length[length] = float(
            relative_l2_error(prediction, view.y).item()
        )
    return {
        "mse_by_length": mse_by_length,
        "relative_l2_by_length": relative_l2_by_length,
        "selection_score": composite_validation_score(mse_by_length),
    }


def composite_validation_score(mse_by_length: Mapping[int, float]) -> float:
    """checkpoint selection 只使用各 validation length 的等权平均 MSE。"""
    if not mse_by_length:
        raise ValueError("mse_by_length must not be empty.")
    return float(np.mean([float(value) for value in mse_by_length.values()]))


def should_replace_best(current_score: float, best_score: float) -> bool:
    """使用 composite validation score 选择 best checkpoint。"""
    return float(current_score) < float(best_score)


def assert_architecture_unchanged(model_config: Mapping[str, Any]) -> None:
    """R1 不允许通过配置改变 baseline FNO2D。"""
    if dict(model_config) != BASELINE_MODEL_CONFIG:
        raise ValueError("R1 model_config must exactly match the baseline FNO2D architecture.")


def derive_default_run_name(epochs: int, train_lengths: Sequence[int]) -> str:
    """生成明确包含 R1 prefix lengths 的隔离 run name。"""
    lengths = "-".join(str(int(length)) for length in sorted(train_lengths))
    return f"fno2d_m16x32_w64_d4_e{int(epochs)}_r1_multilen_t{lengths}"


def refuse_existing_run(task_name: str, run_name: str) -> Path:
    """任何既有 run directory 都拒绝复用或覆盖。"""
    output_dir = get_model_output_dir(task_name=task_name, model_name=run_name)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    return output_dir


def get_git_commit() -> str:
    """读取本地 Git commit；失败时保留可审计标记。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_r1_config(
    *,
    task_name: str,
    run_name: str,
    epochs: int,
    train_lengths: Sequence[int],
    validation_lengths: Sequence[int],
    optimizer_config: Mapping[str, Any],
    scheduler_config: Mapping[str, Any],
    training_seed: int,
    device: torch.device,
    normalization_stats: FieldNormalizationStats,
    param_name: str,
    data_seed: int | None,
) -> dict[str, Any]:
    """构造与现有恢复 helper 兼容的 R1 checkpoint config。"""
    assert_architecture_unchanged(BASELINE_MODEL_CONFIG)
    target_config = TargetTransformConfig(mode=TARGET_TRANSFORM_MODE)
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "repair_class": REPAIR_CLASS,
        "task_name": task_name,
        "source_task": task_name,
        "model_name": run_name,
        "run_name": run_name,
        "model_type": "fno2d",
        "normalization": NORMALIZATION_METHOD,
        "target_transform": TARGET_TRANSFORM_MODE,
        "lambda_reference_index": 0,
        "model_config": dict(BASELINE_MODEL_CONFIG),
        "optimizer_config": dict(optimizer_config),
        "scheduler_config": dict(scheduler_config),
        "training_seed": int(training_seed),
        "epochs": int(epochs),
        "source_max_length": SOURCE_LENGTH,
        "train_lengths": [int(value) for value in train_lengths],
        "validation_lengths": [int(value) for value in validation_lengths],
        "max_training_length": int(max(train_lengths)),
        "normalization_fit_length": SOURCE_LENGTH,
        "normalization_source": "full_source_train_field",
        "architecture_unchanged": True,
        "coordinate_representation": list(INPUT_CHANNEL_NAMES),
        "augmentation_type": "variable-domain_prefix_augmentation",
        "prefix_views_are_independent_trajectories": False,
        "equal_length_loss_weighting": True,
        "optimizer_step_semantics": "one_step_after_all_train_lengths",
        "optimizer_steps": int(epochs),
        "forward_backward_passes_per_step": int(len(train_lengths)),
        "optimizer_update_matched": True,
        "compute_matched": False,
        "deterministic_length_order": "ascending",
        "mixed_width_batching": False,
        "zero_padding": False,
        "masking": False,
        "consistency_loss": False,
        "domain_length_channel": False,
        "relative_coordinate": False,
        "physical_frequency_conditioning": False,
        "formal_long_test_lengths_excluded_from_training": True,
        "unseen_domain_evaluation": "separate_formal_evaluator",
        "checkpoint_selection": {
            "split": "validation_Q_only",
            "metric": "equal_mean_normalized_space_mse",
            "lengths": [int(value) for value in validation_lengths],
            "formal_long_lengths_used": False,
        },
        "dataset_summary": {
            "source_task": task_name,
            "param_name": param_name,
            "input_channel_names": list(INPUT_CHANNEL_NAMES),
            "normalization_stats": normalization_stats.to_dict(),
            "target_transform_config": target_config.to_dict(),
            "canonical_q_order": "stable_ascending",
            "q_split_identity_unit": "trajectory",
            "data_seed": data_seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "git_commit": get_git_commit(),
        },
    }


def seed_everything(seed: int) -> None:
    """复用 baseline 的 Python、NumPy、PyTorch seed 语义。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """解析训练设备且不自动安装或改动环境。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    return device


def load_source_dataset(task_name: str) -> tuple[
    dict[str, CanonicalSplit], np.ndarray, str, dict[str, Any], int | None
]:
    """读取唯一 T1200 source，并核查 generation 与 Q split 身份。"""
    meta = load_task_meta(task_name)
    validate_task_generation_meta(task_name=task_name, meta=meta)
    data = load_raw_task_arrays(task_name)
    validate_npz_keys(data)
    lambda_grid = np.asarray(data["lambda_grid"], dtype=np.float64)
    if lambda_grid.ndim != 1 or lambda_grid.size != SOURCE_LENGTH:
        raise ValueError(
            f"R1 source lambda_grid must have exactly {SOURCE_LENGTH} points, "
            f"got shape {lambda_grid.shape}."
        )
    if not np.all(np.diff(lambda_grid) > 0.0):
        raise ValueError("lambda_grid must be strictly ascending.")
    param_name = validate_single_param_task(load_task_vary_params_order(task_name))
    if param_name != "Q":
        raise ValueError(f"R1 requires the Q-only source task, got param_name={param_name!r}.")
    splits = {
        name: canonicalize_split(data[f"x_{name}"], data[f"y_{name}"])
        for name in ("train", "val", "test")
    }
    assert_disjoint_q_identities(splits)
    return splits, lambda_grid, param_name, meta, extract_data_seed(meta)


def format_length_metrics(prefix: str, values: Mapping[int, float]) -> dict[str, float]:
    """将整数长度转换成稳定的英文 metric keys。"""
    return {f"{prefix}_T{length}": float(values[length]) for length in sorted(values)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 R1 单源训练 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Train the unchanged Q-only FNO2D on strict variable-domain prefixes "
            "from one T1200 source task. Formal long-domain evaluation is separate."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--train-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_LENGTHS),
    )
    parser.add_argument(
        "--validation-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_LENGTHS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.995)
    parser.add_argument("--training-seed", type=int, default=27)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行正式 R1 训练；本地测试只调用纯合成 helper。"""
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be strictly positive.")
    if args.print_every <= 0:
        raise ValueError("print_every must be strictly positive.")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("lr must be positive and weight_decay must be non-negative.")
    if not 0.0 < args.scheduler_gamma <= 1.0:
        raise ValueError("scheduler_gamma must be in (0,1].")

    task_name = validate_safe_name(args.task_name, "task_name")
    train_lengths, validation_lengths = validate_length_protocol(
        args.train_lengths,
        args.validation_lengths,
        SOURCE_LENGTH,
    )
    run_name = validate_safe_name(
        args.run_name or derive_default_run_name(args.epochs, train_lengths),
        "run_name",
    )
    refuse_existing_run(task_name, run_name)
    device = resolve_device(args.device)
    seed_everything(args.training_seed)

    splits, lambda_grid, param_name, _meta, data_seed = load_source_dataset(task_name)
    target_config = TargetTransformConfig(mode=TARGET_TRANSFORM_MODE)
    full_fields = build_full_source_fields(
        splits=splits,
        lambda_grid=lambda_grid,
        param_name=param_name,
        target_transform_config=target_config,
    )
    stats = fit_full_source_normalization(full_fields["train"])
    train_views = build_prefix_views(
        full_fields["train"], train_lengths, stats=stats, device=device
    )
    validation_views = build_prefix_views(
        full_fields["val"], validation_lengths, stats=stats, device=device
    )

    assert_architecture_unchanged(BASELINE_MODEL_CONFIG)
    model = build_model_2d(**BASELINE_MODEL_CONFIG).to(device)
    optimizer_config = {
        "name": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
    }
    scheduler_config = {
        "name": "ExponentialLR",
        "gamma": float(args.scheduler_gamma),
        "steps_per_optimizer_step": 1,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.scheduler_gamma
    )
    config = build_r1_config(
        task_name=task_name,
        run_name=run_name,
        epochs=args.epochs,
        train_lengths=train_lengths,
        validation_lengths=validation_lengths,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        training_seed=args.training_seed,
        device=device,
        normalization_stats=stats,
        param_name=param_name,
        data_seed=data_seed,
    )
    config["model_parameter_count"] = int(count_parameters(model))
    config["model_config"] = summarize_model_config_2d(**BASELINE_MODEL_CONFIG)

    output_dirs = get_model_output_dirs(task_name=task_name, model_name=run_name)
    save_json(config, output_dirs["run_config_json"])
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    started = perf_counter()

    print(f"R1 training task: {task_name}")
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Train lengths: {list(train_lengths)}")
    print(f"Validation lengths: {list(validation_lengths)}")
    print("Optimizer updates are matched; forward/backward compute is not matched.")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_training_epoch(model, optimizer, scheduler, train_views)
        validation_metrics = evaluate_views(model, validation_views)
        selection_score = float(validation_metrics["selection_score"])
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_steps": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_equal_weight_mse": float(train_metrics["equal_weight_mse"]),
            "val_selection_score": selection_score,
            **format_length_metrics("train_mse", train_metrics["mse_by_length"]),
            **format_length_metrics(
                "train_relative_l2", train_metrics["relative_l2_by_length"]
            ),
            **format_length_metrics("val_mse", validation_metrics["mse_by_length"]),
            **format_length_metrics(
                "val_relative_l2", validation_metrics["relative_l2_by_length"]
            ),
        }
        history.append(epoch_record)

        if should_replace_best(selection_score, best_score):
            best_score = selection_score
            best_epoch = epoch
            save_checkpoint_2d(
                output_dirs["checkpoints_dir"] / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mse=best_score,
                config=config,
            )
        save_checkpoint_2d(
            output_dirs["checkpoints_dir"] / "last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mse=best_score,
            config=config,
        )
        save_json({"epochs": history}, output_dirs["logs_dir"] / "train_history.json")

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            train_text = " ".join(
                f"train_mse_T{length}={train_metrics['mse_by_length'][length]:.6e}"
                for length in train_lengths
            )
            val_text = " ".join(
                f"val_mse_T{length}={validation_metrics['mse_by_length'][length]:.6e}"
                for length in validation_lengths
            )
            print(
                f"epoch={epoch}/{args.epochs} {train_text} {val_text} "
                f"val_selection_score={selection_score:.6e}"
            )

    elapsed = perf_counter() - started
    summary = {
        "experiment_type": EXPERIMENT_TYPE,
        "repair_class": REPAIR_CLASS,
        "task_name": task_name,
        "run_name": run_name,
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
        "epochs": int(args.epochs),
        "optimizer_steps": int(args.epochs),
        "forward_backward_passes_per_step": int(len(train_lengths)),
        "optimizer_update_matched": True,
        "compute_matched": False,
        "elapsed_seconds": float(elapsed),
        "final_epoch_metrics": history[-1],
        "formal_long_domain_evaluation_run": False,
    }
    save_json(summary, output_dirs["summary_json"])
    save_json(summary, output_dirs["logs_dir"] / "train_summary.json")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation selection score: {best_score:.6e}")
    print(f"Output directory: {output_dirs['model_dir']}")
    print("Formal T1800/T2400 evaluation must be run separately.")


if __name__ == "__main__":
    main()
