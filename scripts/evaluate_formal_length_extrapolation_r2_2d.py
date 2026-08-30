"""R2：使用 [Q, s, ell] 输入的正式三长度冻结 A1 评估。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts import evaluate_formal_length_extrapolation_2d as formal  # noqa: E402
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    transform_output_field,
)


EXPERIMENT_TYPE = "r2_domain_conditioned_coordinate_training"
REPAIR_CLASS = "INPUT_REPRESENTATION_REPAIR"
COORDINATE_REPRESENTATION = ("Q", "s", "ell")
DOMAIN_LENGTH_DEFINITION = "L = N * delta_lambda (DFT logical period)"
RELATIVE_COORDINATE_DEFINITION = "s = lambda / L"
ELL_DEFINITION = "L / L_ref"
OUTPUT_FILENAMES = (
    "r2_a1_length_extrapolation_summary.json",
    "r2_per_q_metrics.csv",
    "r2_lambda_window_metrics.csv",
)


@dataclass(frozen=True)
class R2CheckpointProvenance:
    """R2 checkpoint 中必须可复核的坐标与频谱参数化信息。"""

    model_config: dict[str, Any]
    input_channel_order: tuple[str, str, str]
    domain_length_definition: str
    relative_coordinate_definition: str
    ell_definition: str
    reference_domain_length: float
    training_lengths: tuple[int, ...]
    validation_lengths: tuple[int, ...]
    input_normalization_policy: dict[str, Any]
    output_normalization_policy: str
    target_transform: str
    spectral_parameterization_unchanged: bool
    discrete_mode_index_weights: bool
    physical_frequency_conditioning: bool


@dataclass(frozen=True)
class R2RuntimeCoordinates:
    """一个运行时长度的可复核 R2 物理域与坐标摘要。"""

    total_length: int
    delta_lambda: float
    domain_length: float
    reference_domain_length: float
    ell: float
    s_min: float
    s_max: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 R2 正式三长度冻结评估所需的明确输入。"""
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen R2 [Q, s, ell] FNO2D forward pass for each of three "
            "canonical lambda-domain lengths and write compact formal A1 metrics."
        )
    )
    parser.add_argument("--training-task-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--short-task-name", required=True)
    parser.add_argument("--medium-task-name", required=True)
    parser.add_argument("--long-task-name", required=True)
    parser.add_argument("--dataset-pair-validation-json", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-width", type=float, default=0.5)
    return parser.parse_args(argv)


def _relative_path(path: Path) -> str:
    """优先记录项目相对路径，外部路径则保留解析后的绝对路径。"""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _git_commit() -> str | None:
    """读取本地 Git HEAD，不执行任何远程 Git 操作。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """检查 checkpoint 的嵌套 provenance 字段。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"R2 checkpoint requires mapping field {name}.")
    return value


def _require_exact_sequence(value: Any, expected: tuple[str, ...], name: str) -> tuple[str, ...]:
    """检查有序输入通道字段，避免按集合猜测语义。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"R2 checkpoint requires ordered sequence field {name}.")
    actual = tuple(str(item) for item in value)
    if actual != expected:
        raise ValueError(f"R2 checkpoint requires {name}={list(expected)}, got {list(actual)}.")
    return actual


def _require_positive_float(value: Any, name: str) -> float:
    """检查物理域尺度字段。"""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"R2 checkpoint requires finite positive {name}.") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"R2 checkpoint requires finite positive {name}.")
    return result


def _require_length_list(value: Any, name: str) -> tuple[int, ...]:
    """检查 checkpoint 记录的训练或验证长度。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"R2 checkpoint requires non-empty {name}.")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"R2 checkpoint has invalid {name}.")
    return result


def validate_r2_checkpoint_provenance(checkpoint: Mapping[str, Any]) -> R2CheckpointProvenance:
    """在加载模型和推理前拒绝非 R2 或语义不完整的 checkpoint。"""
    config = _require_mapping(checkpoint.get("config"), "config")
    if config.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("Checkpoint is not an R2 domain-conditioned training checkpoint.")
    if config.get("repair_class") != REPAIR_CLASS:
        raise ValueError("Checkpoint repair_class is not INPUT_REPRESENTATION_REPAIR.")

    model_config = dict(_require_mapping(config.get("model_config"), "model_config"))
    if str(model_config.get("model_type")) != "fno2d" or int(model_config.get("in_dim", -1)) != 3:
        raise ValueError("R2 checkpoint requires fno2d model_config with in_dim=3.")
    if config.get("normalization") != "standard":
        raise ValueError("R2 checkpoint requires normalization=standard.")
    if config.get("target_transform") != "raw":
        raise ValueError("R2 checkpoint requires target_transform=raw.")

    channel_order = _require_exact_sequence(
        config.get("coordinate_representation"),
        COORDINATE_REPRESENTATION,
        "coordinate_representation",
    )
    _require_exact_sequence(
        config.get("input_channel_names"),
        COORDINATE_REPRESENTATION,
        "input_channel_names",
    )
    dataset_summary = _require_mapping(config.get("dataset_summary"), "dataset_summary")
    _require_exact_sequence(
        dataset_summary.get("input_channel_names"),
        COORDINATE_REPRESENTATION,
        "dataset_summary.input_channel_names",
    )
    if config.get("absolute_lambda_input") is not False:
        raise ValueError("R2 checkpoint must explicitly record absolute_lambda_input=false.")
    if config.get("relative_coordinate_definition") != RELATIVE_COORDINATE_DEFINITION:
        raise ValueError("R2 checkpoint relative-coordinate definition is unsupported or missing.")
    if config.get("domain_length_definition") != DOMAIN_LENGTH_DEFINITION:
        raise ValueError("R2 checkpoint domain-length definition is unsupported or missing.")
    if config.get("ell_definition") != ELL_DEFINITION:
        raise ValueError("R2 checkpoint ell definition is unsupported or missing.")

    spectral_unchanged = config.get("spectral_parameterization_unchanged")
    discrete_weights = config.get("discrete_mode_index_weights")
    physical_conditioning = config.get("physical_frequency_conditioning")
    if spectral_unchanged is not True or discrete_weights is not True or physical_conditioning is not False:
        raise ValueError("R2 checkpoint spectral-parameterization provenance is invalid or missing.")
    if config.get("frequency_interpolation") is not False or config.get("dynamic_spectral_weights") is not False:
        raise ValueError("R2 checkpoint must explicitly exclude R3 spectral mechanisms.")

    policy = dict(_require_mapping(config.get("input_normalization_policy"), "input_normalization_policy"))
    expected_policy = {
        "Q": "standard_full_source_train_field",
        "s": "identity_dimensionless",
        "ell": "identity_dimensionless_L_over_L_ref",
        "target": "standard_full_source_train_field",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"R2 checkpoint input_normalization_policy requires {key}={expected!r}.")
    if policy.get("fit_uses_validation_lengths") is not False or policy.get("fit_uses_formal_long_lengths") is not False:
        raise ValueError("R2 checkpoint normalization policy must exclude validation and formal long lengths.")
    output_policy = str(config.get("output_normalization_policy"))
    if output_policy != "standard_full_source_train_field":
        raise ValueError("R2 checkpoint output normalization policy is unsupported or missing.")

    return R2CheckpointProvenance(
        model_config=model_config,
        input_channel_order=channel_order,
        domain_length_definition=str(config["domain_length_definition"]),
        relative_coordinate_definition=str(config["relative_coordinate_definition"]),
        ell_definition=str(config["ell_definition"]),
        reference_domain_length=_require_positive_float(config.get("L_ref"), "L_ref"),
        training_lengths=_require_length_list(config.get("train_lengths"), "train_lengths"),
        validation_lengths=_require_length_list(config.get("validation_lengths"), "validation_lengths"),
        input_normalization_policy=policy,
        output_normalization_policy=output_policy,
        target_transform=str(config["target_transform"]),
        spectral_parameterization_unchanged=True,
        discrete_mode_index_weights=True,
        physical_frequency_conditioning=False,
    )


def derive_uniform_delta_lambda(lambda_grid: np.ndarray) -> float:
    """验证均匀物理 lambda 网格，并导出 DFT 使用的步长。"""
    values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("R2 formal evaluation requires at least two lambda points.")
    differences = np.diff(values)
    if not np.all(differences > 0.0):
        raise ValueError("R2 formal evaluation requires a strictly increasing lambda grid.")
    delta_lambda = float(differences[0])
    if not np.allclose(differences, delta_lambda, rtol=1e-10, atol=1e-12):
        raise ValueError("R2 formal evaluation requires a uniformly spaced lambda grid.")
    return delta_lambda


def build_r2_model_input(
    *,
    q_values: np.ndarray,
    lambda_grid: np.ndarray,
    raw_truth: np.ndarray,
    provenance: R2CheckpointProvenance,
) -> tuple[np.ndarray, np.ndarray, R2RuntimeCoordinates]:
    """按 checkpoint 已记录的 R2 规则构造 [Q, s, ell] float32 输入。"""
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    lambda64 = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    truth64 = np.asarray(raw_truth, dtype=np.float64)
    expected_truth_shape = (q64.size, lambda64.size, 3)
    if truth64.shape != expected_truth_shape:
        raise ValueError(f"Raw truth shape={truth64.shape}, expected={expected_truth_shape}.")
    if not np.all(np.isfinite(q64)) or not np.all(np.isfinite(truth64)):
        raise ValueError("R2 model input requires finite Q and raw truth values.")
    delta_lambda = derive_uniform_delta_lambda(lambda64)
    total_length = int(lambda64.size)
    domain_length = float(total_length * delta_lambda)
    ell = float(domain_length / provenance.reference_domain_length)
    s_values = lambda64 / domain_length
    q_channel = np.broadcast_to(q64[:, None], (q64.size, total_length))
    s_channel = np.broadcast_to(s_values[None, :], q_channel.shape)
    ell_channel = np.full(q_channel.shape, ell, dtype=np.float64)
    x_raw = np.stack((q_channel, s_channel, ell_channel), axis=-1)[None, ...].astype(np.float32)
    runtime = R2RuntimeCoordinates(
        total_length=total_length,
        delta_lambda=delta_lambda,
        domain_length=domain_length,
        reference_domain_length=float(provenance.reference_domain_length),
        ell=ell,
        s_min=float(np.min(s_values)),
        s_max=float(np.max(s_values)),
    )
    return x_raw, truth64[None, ...].astype(np.float32), runtime


def validate_r2_normalization_stats(
    stats: FieldNormalizationStats,
    provenance: R2CheckpointProvenance,
) -> None:
    """验证 R2 训练时的 Q standard 与 s、ell identity 缩放仍可精确复现。"""
    if stats.method != "standard":
        raise ValueError("R2 checkpoint input statistics require standard normalization.")
    if len(stats.x_mean) != 3 or len(stats.x_std) != 3:
        raise ValueError("R2 checkpoint normalization statistics require three input channels.")
    if not np.isclose(float(stats.x_mean[1]), 0.0) or not np.isclose(float(stats.x_mean[2]), 0.0):
        raise ValueError("R2 checkpoint s and ell means must be identity zero.")
    if not np.isclose(float(stats.x_std[1]), 1.0) or not np.isclose(float(stats.x_std[2]), 1.0):
        raise ValueError("R2 checkpoint s and ell standard deviations must be identity one.")
    if provenance.input_normalization_policy["Q"] != "standard_full_source_train_field":
        raise ValueError("R2 checkpoint Q normalization policy is unsupported.")


def run_frozen_inference(
    *,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    field: formal.CanonicalQField,
    provenance: R2CheckpointProvenance,
    device: str,
) -> tuple[np.ndarray, R2RuntimeCoordinates]:
    """对一个总长度执行一次 R2 冻结前向，并恢复 raw xyz prediction。"""
    x_raw, y_raw, runtime = build_r2_model_input(
        q_values=field.canonical_q,
        lambda_grid=field.lambda_grid,
        raw_truth=field.canonical_truth,
        provenance=provenance,
    )
    stats = load_normalization_stats_from_checkpoint(dict(checkpoint))
    transform_config = load_target_transform_config_from_checkpoint(dict(checkpoint))
    validate_r2_normalization_stats(stats, provenance)
    if transform_config.mode != provenance.target_transform:
        raise ValueError("R2 checkpoint target transform cannot be reconstructed consistently.")
    y_transformed = transform_output_field(y=y_raw, config=transform_config)
    x_model = normalize_input_field(x=x_raw, stats=stats)
    y_model = normalize_output_field(y=y_transformed, stats=stats)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_model).float(), torch.from_numpy(y_model).float()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    with torch.no_grad():
        predictions_model_space, targets_model_space = predict_2d_loader(
            model=model,
            loader=loader,
            device=device,
        )
    predictions_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=predictions_model_space,
        targets_model_space=targets_model_space,
        raw_targets_reference=y_raw,
        normalization_stats=stats,
        target_transform_config=transform_config,
    )
    prediction = np.asarray(predictions_raw[0], dtype=np.float32)
    if prediction.shape != field.canonical_truth.shape:
        raise ValueError(
            f"Frozen prediction shape={prediction.shape}, expected={field.canonical_truth.shape}."
        )
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Frozen inference produced non-finite predictions.")
    return prediction, runtime


def evaluate_three_lengths(
    *,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    provenance: R2CheckpointProvenance,
    short: formal.CanonicalQField,
    medium: formal.CanonicalQField,
    long: formal.CanonicalQField,
    device: str,
    window_width: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """对 T1200/T1800/T2400 分别执行一次直接冻结 R2 前向。"""
    training_length = int(short.lambda_grid.size)
    fields = (short, medium, long)
    results: dict[str, dict[str, Any]] = {}
    per_q_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for field in fields:
        total_length = int(field.lambda_grid.size)
        started = perf_counter()
        prediction, coordinates = run_frozen_inference(
            model=model,
            checkpoint=checkpoint,
            field=field,
            provenance=provenance,
            device=device,
        )
        seconds = perf_counter() - started
        has_extrapolation = total_length > training_length
        results[f"T{total_length}"] = {
            "task_name": field.task_name,
            "total_length": total_length,
            "lambda_min": float(field.lambda_grid[0]),
            "lambda_max": float(field.lambda_grid[-1]),
            "r2_coordinates": {
                "total_length": coordinates.total_length,
                "delta_lambda": coordinates.delta_lambda,
                "DFT_logical_domain_length_L": coordinates.domain_length,
                "L_ref": coordinates.reference_domain_length,
                "ell": coordinates.ell,
                "s_min": coordinates.s_min,
                "s_max": coordinates.s_max,
                "input_channel_order": list(provenance.input_channel_order),
            },
            "prefix": formal.compute_region_metrics(
                prediction[:, :training_length, :],
                field.canonical_truth[:, :training_length, :],
                field.canonical_q,
            ),
            "extrapolation": (
                formal.compute_region_metrics(
                    prediction[:, training_length:, :],
                    field.canonical_truth[:, training_length:, :],
                    field.canonical_q,
                )
                if has_extrapolation
                else None
            ),
            "full": formal.compute_region_metrics(prediction, field.canonical_truth, field.canonical_q),
            "inference_seconds": float(seconds),
            "frozen_forward_passes": 1,
        }
        per_q_rows.extend(
            formal.compute_per_q_rows(
                total_length=total_length,
                prediction=prediction,
                truth=field.canonical_truth,
                q_values=field.canonical_q,
                training_length=training_length,
            )
        )
        if has_extrapolation:
            window_rows.extend(
                formal.compute_window_rows(
                    total_length=total_length,
                    prediction=prediction,
                    truth=field.canonical_truth,
                    lambda_grid=field.lambda_grid,
                    training_length=training_length,
                    window_width=window_width,
                )
            )
    return results, per_q_rows, window_rows


def _ordering_provenance(field: formal.CanonicalQField) -> dict[str, Any]:
    """记录 source identity 与 canonical Q 模型轴之间的映射。"""
    return {
        "source_identity_order": "train_then_val_then_test_original_row_order",
        "canonical_model_input_q_order": "ascending_Q_full_field",
        "canonical_to_source_index": field.canonical_to_source_index,
        "source_to_canonical_index": field.source_to_canonical_index,
        "source_records_in_canonical_order": field.source_records,
    }


def _grid_summary(lambda_grid: np.ndarray) -> dict[str, Any]:
    """记录物理 lambda 网格与 DFT 逻辑域长度。"""
    values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    delta_lambda = derive_uniform_delta_lambda(values)
    return {
        "count": int(values.size),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
        "delta_lambda": delta_lambda,
        "DFT_logical_domain_length_L": float(values.size * delta_lambda),
    }


def build_summary(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    provenance: R2CheckpointProvenance,
    pair_validation_path: Path,
    pair_validation: Mapping[str, Any],
    short: formal.CanonicalQField,
    medium: formal.CanonicalQField,
    long: formal.CanonicalQField,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """构造不含预测数组的紧凑 R2 正式评估 summary。"""
    stats = load_normalization_stats_from_checkpoint(dict(checkpoint))
    transform_config = load_target_transform_config_from_checkpoint(dict(checkpoint))
    training_length = int(short.lambda_grid.size)
    return {
        "schema_version": "1.0",
        "experiment_type": EXPERIMENT_TYPE,
        "evaluation_type": "formal_frozen_one_shot_lambda_domain_length_extrapolation_r2",
        "status": "completed",
        "training_task_name": str(args.training_task_name),
        "tasks": {"short": short.task_name, "medium": medium.task_name, "long": long.task_name},
        "model_name": str(args.model_name),
        "checkpoint_path": _relative_path(checkpoint_path),
        "model_config": formal._as_json_value(provenance.model_config),
        "r2_coordinate_representation": {
            "channel_order": list(provenance.input_channel_order),
            "domain_length_definition": provenance.domain_length_definition,
            "relative_coordinate_definition": provenance.relative_coordinate_definition,
            "ell_definition": provenance.ell_definition,
            "L_ref": provenance.reference_domain_length,
            "absolute_lambda_input": False,
        },
        "checkpoint_training_lengths": list(provenance.training_lengths),
        "checkpoint_validation_lengths": list(provenance.validation_lengths),
        "normalization": {
            "input_policy": formal._as_json_value(provenance.input_normalization_policy),
            "output_policy": provenance.output_normalization_policy,
            "statistics_source": "checkpoint_training_dataset",
            "values": formal._as_json_value(stats.to_dict()),
        },
        "target_transform": formal._as_json_value(transform_config.to_dict()),
        "spectral_parameterization": {
            "unchanged": provenance.spectral_parameterization_unchanged,
            "discrete_mode_index_weights": provenance.discrete_mode_index_weights,
            "physical_frequency_aware_weights": False,
        },
        "stage2_prefix_validation": {
            "artifact_path": _relative_path(pair_validation_path),
            "required_pair_classification": {
                key: pair_validation["pair_classification"][key]
                for key in ("short_to_medium", "short_to_long", "medium_to_long")
            },
            "required_scientific_reuse": {
                key: pair_validation["scientific_reuse"][key]
                for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1")
            },
        },
        "ordering": {
            "short": _ordering_provenance(short),
            "medium": _ordering_provenance(medium),
            "long": _ordering_provenance(long),
            "canonical_q_exact_match_across_all_lengths": True,
        },
        "lengths": {"short": _grid_summary(short.lambda_grid), "medium": _grid_summary(medium.lambda_grid), "long": _grid_summary(long.lambda_grid)},
        "training_domain_boundary": {
            "short_length": training_length,
            "lambda_last_training_point": float(short.lambda_grid[-1]),
            "lambda_first_extrapolation_point": float(medium.lambda_grid[training_length]),
            "derived_from": "short_task_lambda_grid",
        },
        "window_width": float(args.window_width),
        "window_coordinate": "physical_lambda",
        "interval_semantics": "[lambda_start, lambda_end) except the clipped final window is right-closed",
        "frozen_inference": {
            "one_full_forward_pass_per_total_length": True,
            "total_forward_passes": 3,
            "truth_used_as_model_input": False,
            "autoregressive_rollout": False,
            "prediction_feedback": False,
            "teacher_forcing": False,
            "adaptation": False,
            "optimizer": False,
            "scheduler": False,
            "backward": False,
            "training": False,
        },
        "metric_aggregation": {
            "evaluation_space": "raw_physical_xyz_float64",
            "prediction_promotion": "float32_prediction_promoted_to_float64_before_metrics",
            "truth_source": "raw_dataset_truth_float64",
            "mse": "mean_over_Q_lambda_xyz",
            "global_relative_l2": "norm_over_all_Q_lambda_xyz_divided_by_truth_norm",
            "mean_per_q_relative_l2": "mean_of_per_Q_flattened_relative_l2; R0_R1_comparable",
            "component_metrics": "per_component_mse_and_global_relative_l2",
        },
        "scientific_interpretation_boundary": (
            "This evaluator reports R2 evidence neutrally and does not attribute any result "
            "to a sole mechanism or repair the discrete Fourier parameterization."
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": str(args.device),
        },
        "git_commit": _git_commit(),
        "results": formal._as_json_value(results),
        "output_files": list(OUTPUT_FILENAMES),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """写入紧凑 CSV，并以空字段表示不适用的 extrapolation 指标。"""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_output_artifacts(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    per_q_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> None:
    """以独占新目录写入且只写入三份 R2 正式紧凑产物。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(formal._as_json_value(dict(summary)), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_csv(
        per_q_rows,
        output_dir / OUTPUT_FILENAMES[1],
        [
            "total_length",
            "Q",
            "prefix_mse",
            "prefix_relative_l2",
            "extrapolation_mse",
            "extrapolation_relative_l2",
            "full_mse",
            "full_relative_l2",
        ],
    )
    _write_csv(
        window_rows,
        output_dir / OUTPUT_FILENAMES[2],
        [
            "total_length",
            "lambda_start",
            "lambda_end",
            "distance_from_training_boundary_start",
            "distance_from_training_boundary_end",
            "point_count",
            "interval_right_closed",
            "mse",
            "global_relative_l2",
            "mean_per_q_relative_l2",
        ],
    )


def main(argv: Sequence[str] | None = None) -> None:
    """执行 R2 的正式 T1200/T1800/T2400 冻结 A1 评估。"""
    args = parse_args(argv)
    if not np.isfinite(args.window_width) or args.window_width <= 0.0:
        raise ValueError("--window-width must be finite and positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"R2 formal output directory already exists: {args.output_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")

    pair_validation = formal.load_required_pair_validation(args.dataset_pair_validation_json)
    short = formal.load_task_raw_field(str(args.short_task_name))
    medium = formal.load_task_raw_field(str(args.medium_task_name))
    long = formal.load_task_raw_field(str(args.long_task_name))
    formal.validate_triplet(short, medium, long)
    checkpoint = load_checkpoint_2d(checkpoint_path=args.checkpoint_path, device=str(args.device))
    provenance = validate_r2_checkpoint_provenance(checkpoint)
    if str(checkpoint["config"].get("source_task")) != str(args.training_task_name):
        raise ValueError("R2 checkpoint source_task does not match --training-task-name.")
    if int(short.lambda_grid.size) not in provenance.training_lengths:
        raise ValueError("R2 checkpoint training_lengths must include the short formal length.")
    short_delta_lambda = derive_uniform_delta_lambda(short.lambda_grid)
    expected_reference = float(short.lambda_grid.size * short_delta_lambda)
    if not np.isclose(provenance.reference_domain_length, expected_reference, rtol=1e-10, atol=1e-12):
        raise ValueError("R2 checkpoint L_ref does not match the short-task DFT logical domain length.")
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))

    results, per_q_rows, window_rows = evaluate_three_lengths(
        model=model,
        checkpoint=checkpoint,
        provenance=provenance,
        short=short,
        medium=medium,
        long=long,
        device=str(args.device),
        window_width=float(args.window_width),
    )
    summary = build_summary(
        args=args,
        checkpoint_path=args.checkpoint_path,
        checkpoint=checkpoint,
        provenance=provenance,
        pair_validation_path=args.dataset_pair_validation_json,
        pair_validation=pair_validation,
        short=short,
        medium=medium,
        long=long,
        results=results,
    )
    write_output_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        per_q_rows=per_q_rows,
        window_rows=window_rows,
    )


if __name__ == "__main__":
    main()
