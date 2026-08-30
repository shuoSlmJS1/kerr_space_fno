"""R2：使用 [Q, s, ell] 域条件坐标进行多物理域长度前缀训练。"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts import train_fno2d_variable_length_r1 as r1  # noqa: E402
from scripts.train_model_2d import (  # noqa: E402
    get_model_output_dirs,
    save_checkpoint_2d,
    save_json,
)
from src.common.paths import get_model_output_dir  # noqa: E402
from src.models.fno2d.fno2d import count_parameters  # noqa: E402
from src.models.registry_2d import build_model_2d, summarize_model_config_2d  # noqa: E402
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    apply_normalization_to_field_pair,
    compute_field_normalization_stats,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    transform_output_field,
)


EXPERIMENT_TYPE = "r2_domain_conditioned_coordinate_training"
REPAIR_CLASS = "INPUT_REPRESENTATION_REPAIR"
SOURCE_LENGTH = r1.SOURCE_LENGTH
NORMALIZATION_METHOD = "standard"
TARGET_TRANSFORM_MODE = "raw"
INPUT_CHANNEL_NAMES = ["Q", "s", "ell"]
DEFAULT_TRAIN_LENGTHS = list(r1.DEFAULT_TRAIN_LENGTHS)
DEFAULT_VALIDATION_LENGTHS = list(r1.DEFAULT_VALIDATION_LENGTHS)
R2_MODEL_CONFIG: dict[str, Any] = {
    **r1.BASELINE_MODEL_CONFIG,
    "in_dim": 3,
}


@dataclass(frozen=True)
class DomainCoordinateSpec:
    """R2 的离散域长与相对坐标定义。"""

    delta_lambda: float
    reference_domain_length: float
    source_length: int


@dataclass(frozen=True)
class PrefixView:
    """一个同质长度、已标准化的 R2 完整 Q 场。"""

    length: int
    x: torch.Tensor
    y: torch.Tensor


def derive_uniform_delta_lambda(lambda_grid: np.ndarray) -> float:
    """从原始网格导出唯一的均匀物理步长。"""
    grid = np.asarray(lambda_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("lambda_grid must be one-dimensional with at least two points.")
    differences = np.diff(grid)
    if not np.all(differences > 0.0):
        raise ValueError("lambda_grid must be strictly increasing.")
    delta_lambda = float(differences[0])
    if not np.allclose(differences, delta_lambda, rtol=1e-10, atol=1e-12):
        raise ValueError("R2 requires a uniformly spaced lambda_grid.")
    return delta_lambda


def build_domain_coordinate_spec(lambda_grid: np.ndarray) -> DomainCoordinateSpec:
    """采用与既有 DFT 频率约定一致的 L=N*delta_lambda。"""
    grid = np.asarray(lambda_grid, dtype=np.float64)
    if grid.size != SOURCE_LENGTH:
        raise ValueError(
            f"R2 requires a T{SOURCE_LENGTH} source lambda_grid, got {grid.size} points."
        )
    delta_lambda = derive_uniform_delta_lambda(grid)
    return DomainCoordinateSpec(
        delta_lambda=delta_lambda,
        reference_domain_length=float(grid.size * delta_lambda),
        source_length=int(grid.size),
    )


def domain_length_for_prefix(length: int, spec: DomainCoordinateSpec) -> float:
    """返回该离散前缀的 DFT logical period。"""
    if length <= 0 or length > spec.source_length:
        raise ValueError(f"Invalid prefix length {length} for source length {spec.source_length}.")
    return float(int(length) * spec.delta_lambda)


def build_domain_conditioned_input_field(
    q: np.ndarray,
    lambda_prefix: np.ndarray,
    spec: DomainCoordinateSpec,
) -> np.ndarray:
    """构造 [Q, s, ell]，其中 s=lambda/L，ell=L/L_ref。"""
    q_values = np.asarray(q, dtype=np.float32)
    lambda_values = np.asarray(lambda_prefix, dtype=np.float64)
    if q_values.ndim != 2 or q_values.shape[1] != 1:
        raise ValueError(f"Q values must have shape [H,1], got {q_values.shape}.")
    if lambda_values.ndim != 1:
        raise ValueError(f"lambda_prefix must be one-dimensional, got {lambda_values.shape}.")
    length = int(lambda_values.size)
    domain_length = domain_length_for_prefix(length, spec)
    expected_grid = np.arange(length, dtype=np.float64) * spec.delta_lambda + lambda_values[0]
    if not np.allclose(lambda_values, expected_grid, rtol=1e-10, atol=1e-12):
        raise ValueError("R2 prefixes must preserve the source grid without resampling.")
    q_field = np.broadcast_to(q_values[:, 0][None, :, None], (1, q_values.shape[0], length))
    s_values = (lambda_values / domain_length).astype(np.float32)
    s_field = np.broadcast_to(s_values[None, None, :], q_field.shape)
    ell_value = np.float32(domain_length / spec.reference_domain_length)
    ell_field = np.full(q_field.shape, ell_value, dtype=np.float32)
    return np.stack((q_field, s_field, ell_field), axis=-1).astype(np.float32, copy=True)


def build_raw_prefix_field(
    split: r1.CanonicalSplit,
    lambda_grid: np.ndarray,
    length: int,
    spec: DomainCoordinateSpec,
    target_transform_config: TargetTransformConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """从同一 raw Q split 构造严格前缀，不插值、不填充。"""
    lambda_prefix = r1.construct_strict_prefix(lambda_grid, length=length, lambda_axis=0)
    truth_prefix = r1.construct_strict_prefix(split.truth, length=length, lambda_axis=1)
    x = build_domain_conditioned_input_field(split.q, lambda_prefix, spec)
    y = transform_output_field(
        truth_prefix[None, ...].astype(np.float32),
        config=target_transform_config,
    )
    if x.shape[:3] != y.shape[:3] or x.shape[-1] != 3 or y.shape[-1] != 3:
        raise ValueError(f"R2 field shape mismatch: x={x.shape}, y={y.shape}.")
    return x, y


def fit_r2_normalization(
    train_split: r1.CanonicalSplit,
    lambda_grid: np.ndarray,
    spec: DomainCoordinateSpec,
    target_transform_config: TargetTransformConfig,
) -> tuple[FieldNormalizationStats, dict[str, Any]]:
    """仅从完整 T1200 train field 拟合 Q 与 target；s、ell 保持固定无量纲缩放。"""
    x_full, y_full = build_raw_prefix_field(
        split=train_split,
        lambda_grid=lambda_grid,
        length=spec.source_length,
        spec=spec,
        target_transform_config=target_transform_config,
    )
    q_and_target_stats = compute_field_normalization_stats(
        x_train=x_full[..., :1],
        y_train=y_full,
        method=NORMALIZATION_METHOD,
    )
    stats = FieldNormalizationStats(
        method=NORMALIZATION_METHOD,
        x_mean=[float(q_and_target_stats.x_mean[0]), 0.0, 0.0],
        x_std=[float(q_and_target_stats.x_std[0]), 1.0, 1.0],
        y_mean=list(q_and_target_stats.y_mean),
        y_std=list(q_and_target_stats.y_std),
        eps=float(q_and_target_stats.eps),
    )
    policy = {
        "Q": "standard_full_source_train_field",
        "s": "identity_dimensionless",
        "ell": "identity_dimensionless_L_over_L_ref",
        "target": "standard_full_source_train_field",
        "fit_uses_validation_lengths": False,
        "fit_uses_formal_long_lengths": False,
    }
    return stats, policy


def build_prefix_views(
    split: r1.CanonicalSplit,
    lambda_grid: np.ndarray,
    lengths: Sequence[int],
    spec: DomainCoordinateSpec,
    stats: FieldNormalizationStats,
    target_transform_config: TargetTransformConfig,
    device: torch.device,
) -> dict[int, PrefixView]:
    """每个长度重建正确的 s 与 ell，再复用同一组 R2 statistics。"""
    views: dict[int, PrefixView] = {}
    for length in lengths:
        x_raw, y_raw = build_raw_prefix_field(
            split=split,
            lambda_grid=lambda_grid,
            length=int(length),
            spec=spec,
            target_transform_config=target_transform_config,
        )
        x_norm, y_norm = apply_normalization_to_field_pair(x=x_raw, y=y_raw, stats=stats)
        views[int(length)] = PrefixView(
            length=int(length),
            x=torch.from_numpy(x_norm).to(device=device, dtype=torch.float32),
            y=torch.from_numpy(y_norm).to(device=device, dtype=torch.float32),
        )
    return views


def assert_r2_architecture(model_config: Mapping[str, Any]) -> None:
    """确认 R2 仅将输入投影维度从 2 改为 3。"""
    if dict(model_config) != R2_MODEL_CONFIG:
        raise ValueError("R2 model_config must match the baseline except for in_dim=3.")
    for key, value in r1.BASELINE_MODEL_CONFIG.items():
        if key != "in_dim" and model_config[key] != value:
            raise ValueError(f"R2 must preserve baseline {key}={value!r}.")
    if int(model_config["in_dim"]) != 3:
        raise ValueError("R2 requires in_dim=3 for [Q, s, ell].")


def assert_r3_isolation(model_config: Mapping[str, Any]) -> None:
    """确认 R2 仍使用既有离散索引 SpectralConv2d 参数化。"""
    assert_r2_architecture(model_config)
    if model_config["model_type"] != "fno2d":
        raise ValueError("R2 must use the existing fno2d model type.")


def derive_default_run_name(epochs: int, train_lengths: Sequence[int]) -> str:
    """生成隔离的 R2 运行名。"""
    lengths = "-".join(str(int(length)) for length in sorted(train_lengths))
    return f"fno2d_m16x32_w64_d4_e{int(epochs)}_r2_q-s-ell_multilen_t{lengths}"


def refuse_existing_run(task_name: str, run_name: str) -> Path:
    """拒绝复用或覆盖既有 R2 输出目录。"""
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


def build_r2_config(
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
    normalization_policy: Mapping[str, Any],
    coordinate_spec: DomainCoordinateSpec,
    param_name: str,
    data_seed: int | None,
) -> dict[str, Any]:
    """构造可由既有通用 restore helper 恢复的 R2 checkpoint config。"""
    assert_r3_isolation(R2_MODEL_CONFIG)
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
        "model_config": dict(R2_MODEL_CONFIG),
        "optimizer_config": dict(optimizer_config),
        "scheduler_config": dict(scheduler_config),
        "training_seed": int(training_seed),
        "epochs": int(epochs),
        "source_max_length": int(coordinate_spec.source_length),
        "train_lengths": [int(value) for value in train_lengths],
        "validation_lengths": [int(value) for value in validation_lengths],
        "max_training_length": int(max(train_lengths)),
        "normalization_fit_length": int(coordinate_spec.source_length),
        "normalization_source": "full_source_train_field_for_Q_and_target",
        "input_normalization_policy": dict(normalization_policy),
        "output_normalization_policy": "standard_full_source_train_field",
        "coordinate_representation": list(INPUT_CHANNEL_NAMES),
        "input_channel_names": list(INPUT_CHANNEL_NAMES),
        "relative_coordinate_definition": "s = lambda / L",
        "domain_length_definition": "L = N * delta_lambda (DFT logical period)",
        "relative_coordinate_final_sample": "(N - 1) / N for lambda_grid[0] = 0",
        "L_ref": float(coordinate_spec.reference_domain_length),
        "ell_definition": "L / L_ref",
        "absolute_lambda_input": False,
        "architecture_base": "fno2d",
        "architecture_unchanged_after_input_projection": True,
        "modes1": int(R2_MODEL_CONFIG["modes1"]),
        "modes2": int(R2_MODEL_CONFIG["modes2"]),
        "width": int(R2_MODEL_CONFIG["width"]),
        "depth": int(R2_MODEL_CONFIG["depth"]),
        "spectral_parameterization_unchanged": True,
        "discrete_mode_index_weights": True,
        "physical_frequency_conditioning": False,
        "frequency_interpolation": False,
        "dynamic_spectral_weights": False,
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
        "formal_long_test_lengths_excluded_from_training": True,
        "unseen_domain_evaluation": "separate_r2_aware_formal_evaluator_required",
        "existing_formal_a1_evaluator_compatible": False,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析单源 R2 训练 CLI；不接受长域任务。"""
    parser = argparse.ArgumentParser(
        description=(
            "Train R2 domain-conditioned [Q, s, ell] FNO2D on strict prefixes "
            "from one T1200 source task. Formal long-domain evaluation is separate."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=list(DEFAULT_TRAIN_LENGTHS))
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
    """执行正式 R2 训练；本地测试只调用纯合成 helper。"""
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be strictly positive.")
    if args.print_every <= 0:
        raise ValueError("print_every must be strictly positive.")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("lr must be positive and weight_decay must be non-negative.")
    if not 0.0 < args.scheduler_gamma <= 1.0:
        raise ValueError("scheduler_gamma must be in (0,1].")

    task_name = r1.validate_safe_name(args.task_name, "task_name")
    train_lengths, validation_lengths = r1.validate_length_protocol(
        args.train_lengths,
        args.validation_lengths,
        SOURCE_LENGTH,
    )
    run_name = r1.validate_safe_name(
        args.run_name or derive_default_run_name(args.epochs, train_lengths),
        "run_name",
    )
    refuse_existing_run(task_name, run_name)
    device = r1.resolve_device(args.device)
    r1.seed_everything(args.training_seed)

    splits, lambda_grid, param_name, _meta, data_seed = r1.load_source_dataset(task_name)
    coordinate_spec = build_domain_coordinate_spec(lambda_grid)
    target_config = TargetTransformConfig(mode=TARGET_TRANSFORM_MODE)
    normalization_stats, normalization_policy = fit_r2_normalization(
        train_split=splits["train"],
        lambda_grid=lambda_grid,
        spec=coordinate_spec,
        target_transform_config=target_config,
    )
    train_views = build_prefix_views(
        split=splits["train"],
        lambda_grid=lambda_grid,
        lengths=train_lengths,
        spec=coordinate_spec,
        stats=normalization_stats,
        target_transform_config=target_config,
        device=device,
    )
    validation_views = build_prefix_views(
        split=splits["val"],
        lambda_grid=lambda_grid,
        lengths=validation_lengths,
        spec=coordinate_spec,
        stats=normalization_stats,
        target_transform_config=target_config,
        device=device,
    )

    assert_r3_isolation(R2_MODEL_CONFIG)
    model = build_model_2d(**R2_MODEL_CONFIG).to(device)
    optimizer_config = {"name": "AdamW", "lr": float(args.lr), "weight_decay": float(args.weight_decay)}
    scheduler_config = {"name": "ExponentialLR", "gamma": float(args.scheduler_gamma), "steps_per_optimizer_step": 1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    config = build_r2_config(
        task_name=task_name,
        run_name=run_name,
        epochs=args.epochs,
        train_lengths=train_lengths,
        validation_lengths=validation_lengths,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        training_seed=args.training_seed,
        device=device,
        normalization_stats=normalization_stats,
        normalization_policy=normalization_policy,
        coordinate_spec=coordinate_spec,
        param_name=param_name,
        data_seed=data_seed,
    )
    config["model_parameter_count"] = int(count_parameters(model))
    config["model_config"] = summarize_model_config_2d(**R2_MODEL_CONFIG)

    output_dirs = get_model_output_dirs(task_name=task_name, model_name=run_name)
    save_json(config, output_dirs["run_config_json"])
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    started = perf_counter()

    print(f"R2 training task: {task_name}")
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Train lengths: {list(train_lengths)}")
    print(f"Validation lengths: {list(validation_lengths)}")
    print("Optimizer updates are matched; forward/backward compute is not matched.")

    for epoch in range(1, args.epochs + 1):
        train_metrics = r1.run_training_epoch(model, optimizer, scheduler, train_views)
        validation_metrics = r1.evaluate_views(model, validation_views)
        selection_score = float(validation_metrics["selection_score"])
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_steps": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_equal_weight_mse": float(train_metrics["equal_weight_mse"]),
            "val_selection_score": selection_score,
            **r1.format_length_metrics("train_mse", train_metrics["mse_by_length"]),
            **r1.format_length_metrics("train_relative_l2", train_metrics["relative_l2_by_length"]),
            **r1.format_length_metrics("val_mse", validation_metrics["mse_by_length"]),
            **r1.format_length_metrics("val_relative_l2", validation_metrics["relative_l2_by_length"]),
        }
        history.append(epoch_record)
        if r1.should_replace_best(selection_score, best_score):
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
    print("Formal R2 long-domain evaluation requires a separate R2-aware evaluator.")


if __name__ == "__main__":
    main()
