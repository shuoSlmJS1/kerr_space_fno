"""R3-B1：以物理频率锚点插值谱权重的多长度 FNO2D 训练。"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_fno2d_domain_conditioned_r2 as r2  # noqa: E402
from scripts import train_fno2d_variable_length_r1 as r1  # noqa: E402
from scripts.train_model_2d import get_model_output_dirs, save_checkpoint_2d, save_json  # noqa: E402
from src.models.fno2d.fno2d_physical_frequency import (  # noqa: E402
    build_physical_frequency_fno2d_model,
    count_parameters,
    summarize_physical_frequency_fno2d_config,
)
from src.training.fno2d.target_transform_2d import TargetTransformConfig  # noqa: E402


EXPERIMENT_TYPE = "r3_physical_frequency_spectral_training"
REPAIR_CLASS = "SPECTRAL_PARAMETERIZATION_REDESIGN"
SPECTRAL_PARAMETERIZATION = "physical_frequency_anchor_interpolation"
SOURCE_LENGTH = r1.SOURCE_LENGTH
DEFAULT_TRAIN_LENGTHS = tuple(r1.DEFAULT_TRAIN_LENGTHS)
DEFAULT_VALIDATION_LENGTHS = tuple(r1.DEFAULT_VALIDATION_LENGTHS)
INPUT_CHANNEL_NAMES = ("Q", "s", "ell")


def build_anchor_frequencies(*, train_lengths: Sequence[int], delta_lambda: float, modes2: int, num_anchors: int | None = None) -> np.ndarray:
    """仅由训练长度构造均匀物理频率锚点，禁止使用长域测试信息。"""
    lengths = tuple(int(value) for value in train_lengths)
    if not lengths or min(lengths) <= 0 or len(set(lengths)) != len(lengths):
        raise ValueError("train_lengths must be non-empty, positive, and unique.")
    if not np.isfinite(delta_lambda) or delta_lambda <= 0.0:
        raise ValueError("delta_lambda must be finite and positive.")
    anchor_count = int(modes2 if num_anchors is None else num_anchors)
    if anchor_count != int(modes2) or anchor_count < 2:
        raise ValueError("R3-B1 requires num_lambda_frequency_anchors == modes2 >= 2.")
    maximum = float((int(modes2) - 1) / (min(lengths) * float(delta_lambda)))
    return np.linspace(0.0, maximum, anchor_count, dtype=np.float64)


def build_r3_model_config(*, delta_lambda: float, anchor_frequencies: Sequence[float]) -> dict[str, Any]:
    """构造仅替换谱参数化、其余保持 R2 的模型配置。"""
    config = dict(r2.R2_MODEL_CONFIG)
    config.update(
        {
            "model_type": "fno2d_physical_frequency",
            "delta_lambda": float(delta_lambda),
            "anchor_frequencies": [float(value) for value in anchor_frequencies],
        }
    )
    return config


def build_r3_model(model_config: Mapping[str, Any]) -> torch.nn.Module:
    """以 R3 专用构造器恢复模型，不修改共享 registry。"""
    config = dict(model_config)
    if config.pop("model_type", None) != "fno2d_physical_frequency":
        raise ValueError("R3 model_config requires model_type=fno2d_physical_frequency.")
    return build_physical_frequency_fno2d_model(**config)


def assert_r3_isolation(model_config: Mapping[str, Any], *, modes2: int) -> None:
    """确认 R3-B1 不混入 R3-B2、R4 或坐标重设计。"""
    if int(model_config.get("in_dim", -1)) != 3:
        raise ValueError("R3 requires R2 in_dim=3 [Q,s,ell] input.")
    for key in ("out_dim", "modes1", "modes2", "width", "depth", "hidden_dim", "activation"):
        if model_config.get(key) != r2.R2_MODEL_CONFIG[key]:
            raise ValueError(f"R3 must preserve R2 {key}.")
    anchors = model_config.get("anchor_frequencies")
    if not isinstance(anchors, list) or len(anchors) != int(modes2):
        raise ValueError("R3-B1 requires one physical-frequency anchor per modes2 index.")


def derive_default_run_name(epochs: int, train_lengths: Sequence[int]) -> str:
    """生成隔离且可审计的 R3-B1 运行名。"""
    suffix = "-".join(str(int(value)) for value in sorted(train_lengths))
    return f"fno2d_m16x32_w64_d4_e{int(epochs)}_r3_physfreq_anchor32_q-s-ell_multilen_t{suffix}"


def refuse_existing_run(task_name: str, run_name: str) -> Path:
    """拒绝覆盖已有 R3 输出目录。"""
    return r2.refuse_existing_run(task_name, run_name)


def build_r3_config(*, task_name: str, run_name: str, epochs: int, train_lengths: Sequence[int], validation_lengths: Sequence[int], optimizer_config: Mapping[str, Any], scheduler_config: Mapping[str, Any], training_seed: int, device: torch.device, normalization_stats: Any, normalization_policy: Mapping[str, Any], coordinate_spec: r2.DomainCoordinateSpec, param_name: str, data_seed: int | None, model_config: Mapping[str, Any]) -> dict[str, Any]:
    """构造包含物理频率锚点 provenance 的 R3 checkpoint 配置。"""
    assert_r3_isolation(model_config, modes2=int(model_config["modes2"]))
    anchors = [float(value) for value in model_config["anchor_frequencies"]]
    expected_maximum = (int(model_config["modes2"]) - 1) / (min(int(value) for value in train_lengths) * coordinate_spec.delta_lambda)
    if not np.isclose(anchors[-1], expected_maximum, rtol=1e-12, atol=1e-14):
        raise ValueError("Anchor support must derive from the shortest training length.")
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "repair_class": REPAIR_CLASS,
        "task_name": task_name,
        "source_task": task_name,
        "model_name": run_name,
        "run_name": run_name,
        "model_type": "fno2d_physical_frequency",
        "normalization": "standard",
        "target_transform": "raw",
        "lambda_reference_index": 0,
        "model_config": summarize_physical_frequency_fno2d_config(**dict(model_config)),
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
        "L_ref": float(coordinate_spec.reference_domain_length),
        "ell_definition": "L / L_ref",
        "absolute_lambda_input": False,
        "modes1": int(model_config["modes1"]),
        "modes2": int(model_config["modes2"]),
        "width": int(model_config["width"]),
        "depth": int(model_config["depth"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "activation": str(model_config["activation"]),
        "spectral_parameterization": SPECTRAL_PARAMETERIZATION,
        "physical_frequency_formula": "k / (N * delta_lambda)",
        "num_lambda_frequency_anchors": len(anchors),
        "anchor_frequency_min": anchors[0],
        "anchor_frequency_max": anchors[-1],
        "anchor_frequency_values": anchors,
        "anchor_support_source": "training_lengths_only",
        "complex_interpolation": "cartesian_linear",
        "runtime_retained_mode_policy": "fixed_discrete_indices_k_0_to_modes2_minus_1",
        "physical_frequency_conditioning": True,
        "discrete_mode_index_weights": False,
        "physical_cutoff_repair": False,
        "physical_bandwidth_shrinkage_repaired": False,
        "global_fft_structure_unchanged": True,
        "frequency_interpolation": True,
        "dynamic_spectral_weights": False,
        "hypernetwork": False,
        "augmentation_type": "variable-domain_prefix_augmentation",
        "prefix_views_are_independent_trajectories": False,
        "equal_length_loss_weighting": True,
        "optimizer_step_semantics": "one_step_after_all_train_lengths",
        "optimizer_steps": int(epochs),
        "forward_backward_passes_per_step": int(len(train_lengths)),
        "optimizer_update_matched": True,
        "compute_matched": False,
        "mixed_width_batching": False,
        "zero_padding": False,
        "masking": False,
        "consistency_loss": False,
        "formal_long_test_lengths_excluded_from_training": True,
        "checkpoint_selection": {"split": "validation_Q_only", "metric": "equal_mean_normalized_space_mse", "lengths": [int(value) for value in validation_lengths], "formal_long_lengths_used": False},
        "dataset_summary": {"source_task": task_name, "param_name": param_name, "input_channel_names": list(INPUT_CHANNEL_NAMES), "normalization_stats": normalization_stats.to_dict(), "target_transform_config": TargetTransformConfig(mode="raw").to_dict(), "canonical_q_order": "stable_ascending", "q_split_identity_unit": "trajectory", "data_seed": data_seed},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pytorch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "device": str(device), "git_commit": r2.get_git_commit()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析只接受 T1200 source task 的 R3-B1 训练参数。"""
    parser = argparse.ArgumentParser(description="Train R3 physical-frequency-anchor FNO2D on one T1200 source task; formal long-domain evaluation is separate.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=list(DEFAULT_TRAIN_LENGTHS))
    parser.add_argument("--validation-lengths", type=int, nargs="+", default=list(DEFAULT_VALIDATION_LENGTHS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.995)
    parser.add_argument("--training-seed", type=int, default=27)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行正式 R3-B1 训练；本地验证仅使用合成张量。"""
    args = parse_args(argv)
    if args.epochs <= 0 or args.print_every <= 0 or args.lr <= 0.0 or args.weight_decay < 0.0 or not 0.0 < args.scheduler_gamma <= 1.0:
        raise ValueError("Invalid R3 training hyperparameters.")
    task_name = r1.validate_safe_name(args.task_name, "task_name")
    train_lengths, validation_lengths = r1.validate_length_protocol(args.train_lengths, args.validation_lengths, SOURCE_LENGTH)
    run_name = r1.validate_safe_name(args.run_name or derive_default_run_name(args.epochs, train_lengths), "run_name")
    refuse_existing_run(task_name, run_name)
    device = r1.resolve_device(args.device)
    r1.seed_everything(args.training_seed)
    splits, lambda_grid, param_name, _meta, data_seed = r1.load_source_dataset(task_name)
    spec = r2.build_domain_coordinate_spec(lambda_grid)
    target_config = TargetTransformConfig(mode="raw")
    stats, policy = r2.fit_r2_normalization(splits["train"], lambda_grid, spec, target_config)
    train_views = r2.build_prefix_views(splits["train"], lambda_grid, train_lengths, spec, stats, target_config, device)
    validation_views = r2.build_prefix_views(splits["val"], lambda_grid, validation_lengths, spec, stats, target_config, device)
    anchors = build_anchor_frequencies(train_lengths=train_lengths, delta_lambda=spec.delta_lambda, modes2=int(r2.R2_MODEL_CONFIG["modes2"]))
    model_config = build_r3_model_config(delta_lambda=spec.delta_lambda, anchor_frequencies=anchors)
    model = build_r3_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    optimizer_config = {"name": "AdamW", "lr": float(args.lr), "weight_decay": float(args.weight_decay)}
    scheduler_config = {"name": "ExponentialLR", "gamma": float(args.scheduler_gamma), "steps_per_optimizer_step": 1}
    config = build_r3_config(task_name=task_name, run_name=run_name, epochs=args.epochs, train_lengths=train_lengths, validation_lengths=validation_lengths, optimizer_config=optimizer_config, scheduler_config=scheduler_config, training_seed=args.training_seed, device=device, normalization_stats=stats, normalization_policy=policy, coordinate_spec=spec, param_name=param_name, data_seed=data_seed, model_config=model_config)
    config["model_parameter_count"] = count_parameters(model)
    config["baseline_discrete_lambda_spectral_parameter_count"] = 2 * int(model_config["width"]) * int(model_config["width"]) * int(model_config["modes1"]) * int(model_config["modes2"])
    output_dirs = get_model_output_dirs(task_name=task_name, model_name=run_name)
    save_json(config, output_dirs["run_config_json"])
    history: list[dict[str, Any]] = []
    best_score, best_epoch = float("inf"), 0
    started = perf_counter()
    print(f"R3 training task: {task_name}")
    print(f"Run name: {run_name}")
    print(f"Train lengths: {list(train_lengths)}")
    print("Physical-frequency anchors are learned from training lengths only.")
    for epoch in range(1, args.epochs + 1):
        train_metrics = r1.run_training_epoch(model, optimizer, scheduler, train_views)
        validation_metrics = r1.evaluate_views(model, validation_views)
        score = float(validation_metrics["selection_score"])
        record = {"epoch": epoch, "optimizer_steps": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), "train_equal_weight_mse": float(train_metrics["equal_weight_mse"]), "val_selection_score": score, **r1.format_length_metrics("train_mse", train_metrics["mse_by_length"]), **r1.format_length_metrics("train_relative_l2", train_metrics["relative_l2_by_length"]), **r1.format_length_metrics("val_mse", validation_metrics["mse_by_length"]), **r1.format_length_metrics("val_relative_l2", validation_metrics["relative_l2_by_length"])}
        history.append(record)
        if r1.should_replace_best(score, best_score):
            best_score, best_epoch = score, epoch
            save_checkpoint_2d(output_dirs["checkpoints_dir"] / "best_model.pt", model, optimizer, epoch, best_score, config)
        save_checkpoint_2d(output_dirs["checkpoints_dir"] / "last_model.pt", model, optimizer, epoch, best_score, config)
        save_json({"epochs": history}, output_dirs["logs_dir"] / "train_history.json")
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} val_selection_score={score:.6e}")
    summary = {"experiment_type": EXPERIMENT_TYPE, "repair_class": REPAIR_CLASS, "task_name": task_name, "run_name": run_name, "best_epoch": int(best_epoch), "best_val_selection_score": float(best_score), "epochs": int(args.epochs), "optimizer_steps": int(args.epochs), "forward_backward_passes_per_step": int(len(train_lengths)), "optimizer_update_matched": True, "compute_matched": False, "elapsed_seconds": float(perf_counter() - started), "final_epoch_metrics": history[-1], "formal_long_domain_evaluation_run": False}
    save_json(summary, output_dirs["summary_json"])
    save_json(summary, output_dirs["logs_dir"] / "train_summary.json")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation selection score: {best_score:.6e}")
    print(f"Output directory: {output_dirs['model_dir']}")
    print("Formal R3 long-domain evaluation requires the separate R3-aware evaluator.")


if __name__ == "__main__":
    main()
