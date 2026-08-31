"""Unified server-facing workflow for the Plan B T2399-to-T1200 reverse direction.

本入口以已确认的 T1200 baseline contract 为唯一训练语义：先 replay matched
T2399 splits，再训练 theta_2399，并以同一冻结 checkpoint 评估 Q400 的 native
T2399 与 reverse T1200。它只读取已完成 theta_1200 指标，绝不重跑该 forward。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.evaluate_plan_b_resolution_generalization_2d import (  # noqa: E402
    FrozenCheckpointContract,
    build_canonical_field_from_artifact,
    compute_raw_metrics,
    run_frozen_forward,
    validate_plan_b_inputs,
)
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
)
from scripts.train_model_2d import (  # noqa: E402
    evaluate_one_epoch_2d,
    train_one_epoch_2d,
)
from src.data_generation.plan_b_matched_training import (  # noqa: E402
    HISTORICAL_T1200_TRAINING_CONTRACT,
    HistoricalT1200TrainingContract,
    replay_matched_t2399_training_dataset,
    source_split_q_hashes,
    validate_historical_t1200_source,
)
from src.data_generation.plan_b_paired import (  # noqa: E402
    FINE_N_STEPS,
    SPLITS,
    load_dataset_artifact,
    validate_plan_b_ground_truth,
)
from src.models.registry_2d import build_model_2d, summarize_model_config_2d  # noqa: E402
from src.training.fno2d.input_builder_2d import (  # noqa: E402
    FNO2dFieldData,
    build_param_lambda_train_val_test_fields,
    summarize_fno2d_fields,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    compute_field_normalization_stats,
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    transform_output_field,
)


EPSILON = 1e-12
EXPERIMENT_TYPE = "plan_b_bidirectional_fixed_domain_resolution"


@dataclass(frozen=True)
class TrainingRunSpec:
    """训练 helper 的显式参数；正式入口仅传入历史 confirmed contract。"""

    model_config: dict[str, Any]
    training_config: dict[str, Any]


def _json_value(value: Any) -> Any:
    """转换 NumPy 值，并拒绝机器输出中的非有限数。"""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Output contains a non-finite float.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _set_training_seed(seed: int) -> None:
    """保持历史 seed，同时不修改全局环境配置。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析单一 server workflow 所需的显式路径。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run the matched Plan B T2399 training and frozen T2399-to-T1200 "
            "reverse evaluation workflow."
        )
    )
    parser.add_argument("--source-t1200-dataset-dir", required=True, type=Path)
    parser.add_argument("--matched-t2399-dataset-dir", required=True, type=Path)
    parser.add_argument("--q400-t1200-dataset-dir", required=True, type=Path)
    parser.add_argument("--q400-t2399-dataset-dir", required=True, type=Path)
    parser.add_argument("--theta1200-metrics-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _training_spec_from_contract(
    contract: HistoricalT1200TrainingContract = HISTORICAL_T1200_TRAINING_CONTRACT,
) -> TrainingRunSpec:
    model_config = dict(contract.model_config)
    model_config.pop("num_parameters", None)
    return TrainingRunSpec(model_config=model_config, training_config=dict(contract.training_config))


def _require_matched_t2399_dataset(
    dataset_dir: str | Path,
    *,
    contract: HistoricalT1200TrainingContract = HISTORICAL_T1200_TRAINING_CONTRACT,
) -> dict[str, Any]:
    """训练前硬验证 replay identity、哈希、完整性与 Protocol v1 fine grid。"""

    artifact = load_dataset_artifact(dataset_dir)
    metadata = artifact.metadata
    if not np.array_equal(artifact.arrays["lambda_grid"], np.arange(FINE_N_STEPS, dtype=np.float64) * 0.0025):
        raise ValueError("Matched training dataset does not satisfy Protocol v1 T2399/h=0.0025.")
    actual_hashes = source_split_q_hashes(artifact)
    if actual_hashes != contract.split_q_sha256:
        raise ValueError("Matched training dataset split Q hashes do not equal the historical T1200 contract.")
    paired = metadata.get("paired_replay")
    if not isinstance(paired, dict):
        raise ValueError("Matched training dataset is missing paired_replay metadata.")
    if paired.get("replacement_policy") != "forbidden" or paired.get("target_success_replacement_used") is not False:
        raise ValueError("Matched training dataset replacement policy is invalid.")
    if paired.get("paired_completeness") is not True or int(paired.get("failure_count", -1)) != 0:
        raise ValueError("Matched training dataset is incomplete; training is forbidden.")
    for split in SPLITS:
        mask_name = f"success_mask_{split}"
        if mask_name in artifact.arrays and not np.all(artifact.arrays[mask_name]):
            raise ValueError(f"Matched training dataset has failed rows in {split}; training is forbidden.")
        if not np.all(np.isfinite(artifact.arrays[f"y_{split}"])):
            raise ValueError(f"Matched training dataset has non-finite trajectories in {split}; training is forbidden.")
    return {
        "split_q_sha256": actual_hashes,
        "paired_completeness": True,
        "failure_count": 0,
        "replacement_policy": "forbidden",
        "fine_grid": {"T": FINE_N_STEPS, "step_size": 0.0025, "lambda_max": 5.995},
    }


def build_matched_training_fields(dataset_dir: str | Path) -> tuple[dict[str, FNO2dFieldData], FieldNormalizationStats]:
    """从 T2399 train split 拟合新统计量，并对三个 split 复用同一份 stats。"""

    artifact = load_dataset_artifact(dataset_dir)
    arrays = artifact.arrays
    fields = build_param_lambda_train_val_test_fields(
        x_train_raw=arrays["x_train"],
        y_train=arrays["y_train"],
        x_val_raw=arrays["x_val"],
        y_val=arrays["y_val"],
        x_test_raw=arrays["x_test"],
        y_test=arrays["y_test"],
        lambda_grid=arrays["lambda_grid"],
        param_name="Q",
        sort_param=True,
    )
    transform = TargetTransformConfig(mode="raw", lambda_reference_index=0)
    train_transformed = transform_output_field(fields["train"].y_2d, transform)
    stats = compute_field_normalization_stats(
        x_train=fields["train"].x_2d,
        y_train=train_transformed,
        method="standard",
    )
    return fields, stats


def _normalized_loader(field: FNO2dFieldData, stats: FieldNormalizationStats) -> DataLoader:
    transform = TargetTransformConfig(mode="raw", lambda_reference_index=0)
    y_transformed = transform_output_field(field.y_2d, transform)
    x_model = normalize_input_field(field.x_2d, stats)
    y_model = normalize_output_field(y_transformed, stats)
    return DataLoader(
        TensorDataset(torch.from_numpy(x_model).float(), torch.from_numpy(y_model).float()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def _checkpoint_config(
    *,
    fields: dict[str, FNO2dFieldData],
    stats: FieldNormalizationStats,
    spec: TrainingRunSpec,
    source_validation: dict[str, Any],
) -> dict[str, Any]:
    """构建兼容既有 loader 的 checkpoint config，并暴露 matched provenance。"""

    model_config = summarize_model_config_2d(**spec.model_config)
    return {
        "schema_version": "1.0",
        "run_type": "training",
        "experiment_type": EXPERIMENT_TYPE,
        "model_type": "fno2d",
        "operator_axes": ["Q", "lambda"],
        "training_seed": int(spec.training_config["training_seed"]),
        "epochs": int(spec.training_config["epochs"]),
        "batch_size": int(spec.training_config["batch_size"]),
        "lr": float(spec.training_config["lr"]),
        "weight_decay": float(spec.training_config["weight_decay"]),
        "scheduler_gamma": float(spec.training_config["scheduler_gamma"]),
        "normalization": "standard",
        "target_transform": "raw",
        "lambda_reference_index": 0,
        "model_config": model_config,
        "dataset_summary": {
            "param_name": "Q",
            "vary_params_order": ["Q"],
            "normalization_stats": stats.to_dict(),
            "target_transform_config": TargetTransformConfig(mode="raw").to_dict(),
            **summarize_fno2d_fields(fields),
        },
        "plan_b_reverse_contract": {
            "source_task_name": HISTORICAL_T1200_TRAINING_CONTRACT.source_task_name,
            "source_split_q_sha256": source_validation["split_q_sha256"],
            "matched_training_grid": {"T": FINE_N_STEPS, "step_size": 0.0025, "lambda_max": 5.995},
            "normalization_fit_split": "train",
            "normalization_reused_from_t1200": False,
            "target_success_replacement_used": False,
        },
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int,
    best_val_mse: float,
    config: dict[str, Any],
) -> None:
    """保存可恢复训练状态，同时保持既有 FNO checkpoint 核心字段兼容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_val_mse),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def train_matched_t2399_model(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    device: str,
    source_validation: dict[str, Any],
    spec: TrainingRunSpec | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """训练或恢复 theta_2399；所有 normalized field 均来自同一新 fitted stats。"""

    run_spec = spec or _training_spec_from_contract()
    fields, stats = build_matched_training_fields(dataset_dir)
    train_loader = _normalized_loader(fields["train"], stats)
    val_loader = _normalized_loader(fields["val"], stats)
    test_loader = _normalized_loader(fields["test"], stats)
    output = Path(output_dir)
    checkpoints_dir = output / "checkpoints"
    history_path = output / "train_history.json"
    config_path = output / "run_config.json"
    last_path = checkpoints_dir / "last_model.pt"
    best_path = checkpoints_dir / "best_model.pt"
    output.mkdir(parents=True, exist_ok=True)
    config = _checkpoint_config(fields=fields, stats=stats, spec=run_spec, source_validation=source_validation)

    _set_training_seed(int(run_spec.training_config["training_seed"]))
    model = build_model_2d(**run_spec.model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_spec.training_config["lr"]),
        weight_decay=float(run_spec.training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(run_spec.training_config["scheduler_gamma"])
    )
    history = {"train_mse": [], "train_rel_l2": [], "val_mse": [], "val_rel_l2": [], "lr": []}
    start_epoch = 1
    best_epoch = -1
    best_val_mse = float("inf")
    if resume:
        if not last_path.is_file() or not history_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("--resume requires last_model.pt, train_history.json, and run_config.json.")
        previous_config = _read_json(config_path)
        if previous_config.get("plan_b_reverse_contract") != config["plan_b_reverse_contract"]:
            raise ValueError("Resume checkpoint provenance does not match the current matched-training contract.")
        checkpoint = load_checkpoint_2d(last_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = _read_json(history_path)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_val_mse = float(checkpoint["best_val_mse"])
    elif any(path.exists() for path in (last_path, best_path, history_path, config_path)):
        raise FileExistsError("Training output already contains state; use --resume or choose a new output directory.")
    else:
        _write_json(config_path, config)

    started = perf_counter()
    for epoch in range(start_epoch, int(run_spec.training_config["epochs"]) + 1):
        train_mse, train_rel = train_one_epoch_2d(model, train_loader, optimizer, device)
        val_mse, val_rel = evaluate_one_epoch_2d(model, val_loader, device)
        history["train_mse"].append(float(train_mse))
        history["train_rel_l2"].append(float(train_rel))
        history["val_mse"].append(float(val_mse))
        history["val_rel_l2"].append(float(val_rel))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            best_epoch = int(epoch)
            _save_training_checkpoint(
                best_path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                best_epoch=best_epoch, best_val_mse=best_val_mse, config=config,
            )
        scheduler.step()
        _save_training_checkpoint(
            last_path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            best_epoch=best_epoch, best_val_mse=best_val_mse, config=config,
        )
        _write_json(history_path, history)
        print(
            f"Epoch [{epoch:03d}/{run_spec.training_config['epochs']}] | "
            f"lr={history['lr'][-1]:.6e} | train_mse={train_mse:.6e} | val_mse={val_mse:.6e}"
        )
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written during matched T2399 training.")
    checkpoint = load_checkpoint_2d(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse, test_rel = evaluate_one_epoch_2d(model, test_loader, device)
    summary = {
        "status": "completed",
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(history_path),
        "best_epoch": int(best_epoch),
        "best_val_mse_model_space": float(best_val_mse),
        "test_mse_model_space": float(test_mse),
        "test_relative_l2_model_space": float(test_rel),
        "normalization_stats": stats.to_dict(),
        "normalization_fit_split": "train",
        "normalization_reused_from_t1200": False,
        "target_transform": "raw",
        "seconds_this_invocation": float(perf_counter() - started),
    }
    _write_json(output / "train_summary.json", summary)
    return _json_value(summary)


def validate_t2399_checkpoint_contract(checkpoint: dict[str, Any]) -> FrozenCheckpointContract:
    """验证 theta_2399 的训练 provenance、raw target 和冻结 stats。"""

    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("Checkpoint is not a Plan B matched T2399 training checkpoint.")
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint has no model_config.")
    expected_model = _training_spec_from_contract().model_config
    for key, expected in expected_model.items():
        if model_config.get(key) != expected:
            raise ValueError(f"T2399 checkpoint model contract mismatch for {key}.")
    dataset_summary = config.get("dataset_summary")
    if not isinstance(dataset_summary, dict) or int(dataset_summary.get("train", {}).get("num_lambda", -1)) != FINE_N_STEPS:
        raise ValueError("T2399 checkpoint does not record T=2399 training provenance.")
    reverse_contract = config.get("plan_b_reverse_contract")
    if not isinstance(reverse_contract, dict) or reverse_contract.get("normalization_reused_from_t1200") is not False:
        raise ValueError("T2399 checkpoint normalization provenance is invalid.")
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    target = load_target_transform_config_from_checkpoint(checkpoint)
    if stats.method != "standard" or target.mode != "raw" or target.lambda_reference_index != 0:
        raise ValueError("T2399 checkpoint must use standard normalization and raw target transform.")
    return FrozenCheckpointContract(
        normalization_stats=stats,
        target_transform_config=target,
        model_config=dict(model_config),
        provenance={
            "checkpoint_training_lambda_count": FINE_N_STEPS,
            "normalization_statistics_source": "checkpoint.config.dataset_summary.normalization_stats",
            "normalization_refit_during_evaluation": False,
            "normalization_fit_split": "T2399_train",
            "target_transform_source": "checkpoint.config.dataset_summary.target_transform_config",
        },
    )


def evaluate_theta2399_on_q400(
    *,
    checkpoint_path: str | Path,
    q400_t1200_dataset_dir: str | Path,
    q400_t2399_dataset_dir: str | Path,
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """先 qualification，再对 native fine 与 reverse coarse 各做一次 frozen forward。"""

    qualification, coarse, fine = validate_plan_b_inputs(q400_t1200_dataset_dir, q400_t2399_dataset_dir)
    checkpoint = load_checkpoint_2d(Path(checkpoint_path), device)
    frozen = validate_t2399_checkpoint_contract(checkpoint)
    model = load_fno2d_checkpoint_model(checkpoint, device)
    native_start = perf_counter()
    fine_prediction = run_frozen_forward(model=model, contract=frozen, field=fine, device=device)
    native_seconds = perf_counter() - native_start
    reverse_start = perf_counter()
    coarse_prediction = run_frozen_forward(model=model, contract=frozen, field=coarse, device=device)
    reverse_seconds = perf_counter() - reverse_start
    native_metrics = compute_raw_metrics(fine_prediction, fine.canonical_truth, fine.canonical_q)
    reverse_metrics = compute_raw_metrics(coarse_prediction, coarse.canonical_truth, coarse.canonical_q)
    reverse_gap: dict[str, dict[str, float | None]] = {}
    for name in ("global_relative_l2", "mean_per_q_relative_l2"):
        native_value = float(native_metrics[name])
        reverse_value = float(reverse_metrics[name])
        reverse_gap[name] = {
            "reverse_minus_native": reverse_value - native_value,
            "reverse_divided_by_native": reverse_value / native_value if native_value != 0.0 else None,
        }
    result = {
        "schema_version": "1.0",
        "experiment_type": EXPERIMENT_TYPE,
        "status": "completed",
        "checkpoint": {
            "path": _relative_path(Path(checkpoint_path)),
            "epoch": checkpoint.get("epoch"),
            "model_config": frozen.model_config,
            "provenance": frozen.provenance,
            "normalization": frozen.normalization_stats.to_dict(),
            "target_transform": frozen.target_transform_config.to_dict(),
        },
        "ground_truth_qualification": {
            "structural_valid": qualification["structural_valid"],
            "numerical_metrics": qualification["numerical_metrics"],
        },
        "frozen_inference": {
            "native_t2399_forward_passes": 1,
            "reverse_t1200_forward_passes": 1,
            "retraining": False,
            "fine_tuning": False,
            "normalization_refit": False,
            "native_t2399_seconds": float(native_seconds),
            "reverse_t1200_seconds": float(reverse_seconds),
        },
        "metrics": {
            "theta2399_native_t2399": native_metrics,
            "theta2399_reverse_t1200": reverse_metrics,
        },
        "reverse_generalization": reverse_gap,
    }
    return _json_value(result), coarse_prediction, fine_prediction


def load_completed_theta1200_metrics(metrics_path: str | Path) -> dict[str, Any]:
    """读取已完成 coarse-to-fine 结果；禁止调用任何 theta_1200 inference。"""

    value = _read_json(Path(metrics_path))
    required = {"coarse_t1200", "fine_t2399_full_grid_primary"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Completed theta_1200 metrics are missing keys: {missing}")
    return {
        "source": _relative_path(Path(metrics_path)),
        "reuse_completed_theta1200_inference": True,
        "theta1200_native_t1200": value["coarse_t1200"],
        "theta1200_coarse_to_fine_t2399": value["fine_t2399_full_grid_primary"],
    }


def assemble_bidirectional_matrix(theta1200: dict[str, Any], theta2399: dict[str, Any]) -> dict[str, Any]:
    """组合 2x2 原始 physical-xyz 指标矩阵，不重算 theta_1200。"""

    return {
        "training_rows": {
            "theta1200": {
                "test_t1200": theta1200["theta1200_native_t1200"],
                "test_t2399": theta1200["theta1200_coarse_to_fine_t2399"],
            },
            "theta2399": {
                "test_t1200": theta2399["metrics"]["theta2399_reverse_t1200"],
                "test_t2399": theta2399["metrics"]["theta2399_native_t2399"],
            },
        },
        "theta1200_source": theta1200["source"],
        "theta1200_inference_rerun": False,
        "evaluation_space": "raw_physical_xyz_float64",
    }


def _write_evaluation_outputs(output_dir: Path, result: dict[str, Any], coarse_prediction: np.ndarray, fine_prediction: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", result["metrics"])
    _write_json(output_dir / "experiment_summary.json", result)
    np.save(output_dir / "theta2399_reverse_t1200_prediction_raw_xyz.npy", np.asarray(coarse_prediction, dtype=np.float32))
    np.save(output_dir / "theta2399_native_t2399_prediction_raw_xyz.npy", np.asarray(fine_prediction, dtype=np.float32))


def run_bidirectional_workflow(
    *,
    source_t1200_dataset_dir: str | Path,
    matched_t2399_dataset_dir: str | Path,
    q400_t1200_dataset_dir: str | Path,
    q400_t2399_dataset_dir: str | Path,
    theta1200_metrics_json: str | Path,
    output_dir: str | Path,
    device: str,
    resume: bool = False,
) -> dict[str, Any]:
    """执行可恢复的完整 server workflow，步骤完成后以状态文件实现安全继续。"""

    output = Path(output_dir)
    if output.exists() and not resume:
        raise FileExistsError("Output directory exists; use --resume only for the same workflow state.")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "workflow_state.json"
    state = _read_json(state_path) if resume and state_path.is_file() else {"schema_version": "1.0", "completed_steps": []}
    source_artifact, source_validation = validate_historical_t1200_source(source_t1200_dataset_dir)
    matched_dir = Path(matched_t2399_dataset_dir)
    if not matched_dir.exists():
        replay = replay_matched_t2399_training_dataset(source_t1200_dataset_dir, matched_dir)
    else:
        replay = _require_matched_t2399_dataset(matched_dir)
    if replay.get("training_permitted") is False:
        raise RuntimeError("Matched T2399 replay is incomplete; training is forbidden.")
    qualification = validate_plan_b_ground_truth(source_artifact.directory, matched_dir)
    if qualification.get("structural_valid") is not True:
        raise RuntimeError("Matched T1200/T2399 truth qualification failed structurally; training is forbidden.")
    state["completed_steps"] = ["source_contract_validated", "matched_t2399_replay_qualified"]
    state["matched_training_replay"] = replay
    _write_json(state_path, state)

    training_dir = output / "training"
    train_summary_path = training_dir / "train_summary.json"
    if resume and train_summary_path.is_file() and (training_dir / "checkpoints" / "best_model.pt").is_file():
        training = _read_json(train_summary_path)
    else:
        training = train_matched_t2399_model(
            dataset_dir=matched_dir,
            output_dir=training_dir,
            device=device,
            source_validation=source_validation,
            resume=resume,
        )
    state["completed_steps"] = ["source_contract_validated", "matched_t2399_replay_qualified", "theta2399_training_completed"]
    state["training"] = training
    _write_json(state_path, state)

    evaluation_dir = output / "theta2399_q400_evaluation"
    if resume and (evaluation_dir / "experiment_summary.json").is_file():
        theta2399 = _read_json(evaluation_dir / "experiment_summary.json")
    else:
        theta2399, coarse_prediction, fine_prediction = evaluate_theta2399_on_q400(
            checkpoint_path=training["best_checkpoint"],
            q400_t1200_dataset_dir=q400_t1200_dataset_dir,
            q400_t2399_dataset_dir=q400_t2399_dataset_dir,
            device=device,
        )
        _write_evaluation_outputs(evaluation_dir, theta2399, coarse_prediction, fine_prediction)
    theta1200 = load_completed_theta1200_metrics(theta1200_metrics_json)
    matrix = assemble_bidirectional_matrix(theta1200, theta2399)
    _write_json(output / "bidirectional_resolution_matrix.json", matrix)
    final = {
        "schema_version": "1.0",
        "experiment_type": EXPERIMENT_TYPE,
        "status": "completed",
        "source_contract": asdict(HISTORICAL_T1200_TRAINING_CONTRACT),
        "matched_training_ground_truth_qualification": qualification,
        "theta2399_training": training,
        "theta2399_evaluation": theta2399,
        "theta1200_completed_metrics": theta1200,
        "bidirectional_matrix": matrix,
    }
    _write_json(output / "experiment_summary.json", final)
    state["completed_steps"] = [
        "source_contract_validated", "matched_t2399_replay_qualified", "theta2399_training_completed",
        "theta2399_native_and_reverse_evaluated", "bidirectional_matrix_assembled",
    ]
    _write_json(state_path, state)
    return _json_value(final)


def main() -> None:
    args = parse_args()
    result = run_bidirectional_workflow(
        source_t1200_dataset_dir=args.source_t1200_dataset_dir,
        matched_t2399_dataset_dir=args.matched_t2399_dataset_dir,
        q400_t1200_dataset_dir=args.q400_t1200_dataset_dir,
        q400_t2399_dataset_dir=args.q400_t2399_dataset_dir,
        theta1200_metrics_json=args.theta1200_metrics_json,
        output_dir=args.output_dir,
        device=str(args.device),
        resume=bool(args.resume),
    )
    print("Plan B bidirectional reverse workflow completed.")
    print(f"Output directory: {args.output_dir}")
    print(f"Completed steps: {', '.join(result['bidirectional_matrix']['training_rows'].keys())}")


if __name__ == "__main__":
    main()
