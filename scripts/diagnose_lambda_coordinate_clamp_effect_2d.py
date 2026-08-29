"""Diagnose a nonphysical frozen FNO2D appended-lambda coordinate clamp probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import diagnose_fno2d_internal_length_sensitivity as m3  # noqa: E402
from scripts.evaluate_formal_length_extrapolation_2d import (  # noqa: E402
    CanonicalQField,
    build_model_input,
    compute_region_metrics,
    load_required_pair_validation,
    load_task_raw_field,
    validate_triplet,
)
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    recover_predictions_and_targets_to_raw_xyz,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import transform_output_field  # noqa: E402


EPSILON = 1e-12
EXPECTED_INPUT_CHANNELS = ("Q", "lambda")
OUTPUT_FILENAMES = (
    "m4b_lambda_clamp_summary.json",
    "m4b_intervention_response.csv",
    "m4b_stage_comparison.csv",
)


@dataclass
class ArmCapture:
    """保存一个冻结前向 arm 的紧凑诊断信息。"""

    arm_name: str
    length_label: str
    intervention: str
    field: CanonicalQField
    normalized_input: np.ndarray
    normalized_input_prefix: torch.Tensor
    lifted_feature_prefix: torch.Tensor
    first_spectral_input_prefix: torch.Tensor
    spectral_branch_output_prefix: torch.Tensor
    first_block_output_prefix: torch.Tensor
    final_prediction_prefix: np.ndarray
    secondary_truth_reference_prefix_metrics: dict[str, Any]
    spectral: m3.SpectralCapture


def parse_args() -> argparse.Namespace:
    """解析 M4b 所需的正式输入。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run a nonphysical frozen same-length FNO2D appended-lambda coordinate "
            "clamp mechanism probe and write compact intervention-response diagnostics."
        )
    )
    parser.add_argument("--training-task-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--short-task-name", required=True)
    parser.add_argument("--medium-task-name", required=True)
    parser.add_argument("--long-task-name", required=True)
    parser.add_argument("--dataset-pair-validation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _as_json_value(value: Any) -> Any:
    """将 NumPy 和 Tensor 标量递归转换为 JSON 安全值。"""

    if isinstance(value, torch.Tensor):
        return _as_json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_as_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    return value


def _relative_path(path: Path) -> str:
    """优先记录项目根目录相对路径。"""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _git_commit() -> str | None:
    """读取本地 HEAD，不执行远程 Git 操作。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def require_checkpoint_input_contract(checkpoint: dict[str, Any]) -> tuple[FieldNormalizationStats, tuple[str, ...], int, int]:
    """严格验证 M4b 所需的 checkpoint 输入归一化与通道契约。"""

    config = checkpoint.get("config", {})
    if config.get("normalization") != "standard":
        raise ValueError("M4b requires checkpoint config normalization == 'standard'.")
    summary = config.get("dataset_summary", {})
    names = tuple(summary.get("train", {}).get("input_channel_names", ()))
    if names != EXPECTED_INPUT_CHANNELS:
        raise ValueError(
            "M4b requires checkpoint input channel order ['Q', 'lambda']; "
            f"received {list(names)!r}."
        )
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    if stats.method != "standard":
        raise ValueError("M4b requires standard checkpoint normalization statistics.")
    if len(stats.x_mean) != len(names) or len(stats.x_std) != len(names):
        raise ValueError("Checkpoint input normalization statistics do not match input channels.")
    model_config = config.get("model_config", {})
    if int(model_config.get("in_dim", -1)) != len(names):
        raise ValueError("Checkpoint model in_dim does not match the required two input channels.")
    return stats, names, names.index("Q"), names.index("lambda")


def derive_lambda_clamp_bound(short_lambda_grid: np.ndarray, stats: FieldNormalizationStats, lambda_channel_index: int) -> tuple[float, float]:
    """从短任务真实网格和 checkpoint 统计量导出归一化 clamp 上界。"""

    grid = np.asarray(short_lambda_grid, dtype=np.float64).reshape(-1)
    if grid.size < 1:
        raise ValueError("Short-task lambda grid must contain at least one value.")
    raw_max = float(grid[-1])
    mean = float(stats.x_mean[lambda_channel_index])
    std = float(stats.x_std[lambda_channel_index])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("Checkpoint lambda normalization statistics must be finite with positive std.")
    return raw_max, (raw_max - mean) / std


def clamp_appended_lambda_channel(
    normalized_input: np.ndarray,
    *,
    shared_length: int,
    q_channel_index: int,
    lambda_channel_index: int,
    clamp_upper_bound: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """仅在追加区对归一化 lambda 通道实施上界 clamp，并验证控制条件。"""

    original = np.asarray(normalized_input)
    if original.ndim != 4 or original.shape[0] != 1:
        raise ValueError("Normalized model input must have shape [1, H, W, C].")
    if not 0 <= shared_length <= original.shape[2]:
        raise ValueError("Shared prefix length is outside normalized input width.")
    if q_channel_index == lambda_channel_index:
        raise ValueError("Q and lambda channel indices must differ.")
    clamped = original.copy()
    lambda_original = original[..., lambda_channel_index]
    lambda_clamped = clamped[..., lambda_channel_index]
    lambda_clamped[:, :, shared_length:] = np.minimum(
        lambda_original[:, :, shared_length:],
        np.asarray(clamp_upper_bound, dtype=original.dtype),
    )
    difference = clamped != original
    allowed = np.zeros_like(difference, dtype=bool)
    allowed[:, :, shared_length:, lambda_channel_index] = True
    if np.any(difference & ~allowed):
        raise AssertionError("Clamp changed a non-appended or non-lambda input value.")
    if not np.array_equal(original[:, :, :shared_length, :], clamped[:, :, :shared_length, :]):
        raise AssertionError("Clamp changed the shared normalized-input prefix.")
    if not np.array_equal(original[..., q_channel_index], clamped[..., q_channel_index]):
        raise AssertionError("Clamp changed the Q channel.")
    position_mask = np.any(
        difference[:, :, :, lambda_channel_index], axis=(0, 1)
    )
    modified_positions = int(np.count_nonzero(position_mask))
    total_positions = int(original.shape[2])
    return clamped, {
        "number_of_modified_lambda_positions": modified_positions,
        "fraction_of_total_lambda_positions_modified": modified_positions / total_positions,
        "number_of_modified_lambda_tensor_values": int(np.count_nonzero(difference[..., lambda_channel_index])),
        "max_original_normalized_lambda": float(np.max(lambda_original)),
        "clamp_upper_bound": float(clamp_upper_bound),
        "max_clamped_normalized_lambda": float(np.max(lambda_clamped)),
        "shared_prefix_exactly_unchanged": True,
        "q_channel_exactly_unchanged": True,
        "only_appended_lambda_values_changed": True,
    }


def intervention_semantics() -> dict[str, Any]:
    """返回 M4b 非物理 mechanism-probe 的固定科学解释边界。"""

    return {
        "intervention_type": "nonphysical_appended_lambda_coordinate_clamp",
        "purpose": "mechanism_probe",
        "valid_for_production_prediction": False,
        "valid_as_formal_length_extrapolation_protocol": False,
        "shared_prefix_coordinates_modified": False,
        "appended_lambda_coordinates_modified": True,
        "tensor_length_changed": False,
        "fft_grid_changed_within_original_vs_clamped_pair": False,
        "model_weights_changed": False,
        "interpretation": "Measures sensitivity to appended out-of-training-range lambda-coordinate magnitude under a controlled frozen-forward intervention.",
    }

def _secondary_truth_reference_prefix_metrics(
    prediction: np.ndarray,
    field: CanonicalQField,
    shared_length: int,
) -> dict[str, Any]:
    """仅把未修改 physical shared prefix 的 raw truth 用作次级机制上下文参考。"""

    truth = np.asarray(field.canonical_truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if prediction64.shape != truth.shape:
        raise ValueError("Recovered prediction shape does not match raw truth.")
    return compute_region_metrics(
        prediction64[:, :shared_length, :], truth[:, :shared_length, :], field.canonical_q
    )

def _make_arm_capture(
    *,
    arm_name: str,
    length_label: str,
    intervention: str,
    field: CanonicalQField,
    normalized_input: np.ndarray,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    device: str,
    shared_length: int,
) -> ArmCapture:
    """执行一个正常冻结前向，并仅保留紧凑阶段信息。"""

    _, y_raw = build_model_input(field.canonical_q, field.lambda_grid, field.canonical_truth)
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform = load_target_transform_config_from_checkpoint(checkpoint)
    y_model = normalize_output_field(y=transform_output_field(y=y_raw, config=transform), stats=stats)
    x_tensor = torch.from_numpy(np.asarray(normalized_input)).float().to(device)
    prediction_model, hooks = m3.capture_one_forward(model, x_tensor)
    spectral, replicated_output = m3._replicate_first_spectral_layer(
        hooks.first_spectral_input, model.blocks[0].spectral_conv, device
    )
    spectral.replicated_output_metrics = m3._metrics(replicated_output, hooks.spectral_branch_output)
    prediction_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=prediction_model.detach().cpu().numpy(),
        targets_model_space=y_model,
        raw_targets_reference=y_raw,
        normalization_stats=stats,
        target_transform_config=transform,
    )
    prediction = np.asarray(prediction_raw[0], dtype=np.float64)
    return ArmCapture(
        arm_name=arm_name,
        length_label=length_label,
        intervention=intervention,
        field=field,
        normalized_input=np.asarray(normalized_input).copy(),
        normalized_input_prefix=m3._to_cpu_clone(x_tensor[:, :, :shared_length, :]),
        lifted_feature_prefix=hooks.lifted_feature[:, :, :shared_length, :].clone(),
        first_spectral_input_prefix=hooks.first_spectral_input[:, :, :, :shared_length].clone(),
        spectral_branch_output_prefix=hooks.spectral_branch_output[:, :, :, :shared_length].clone(),
        first_block_output_prefix=hooks.first_block_output[:, :, :, :shared_length].clone(),
        final_prediction_prefix=prediction[:, :shared_length, :].copy(),
        secondary_truth_reference_prefix_metrics=_secondary_truth_reference_prefix_metrics(prediction, field, shared_length),
        spectral=spectral,
    )


def _same_index_spectral_metrics(left: ArmCapture, right: ArmCapture, stage: str) -> dict[str, float]:
    """在相同长度 arm 的相同 retained 离散 lambda index 上比较频域系数。"""

    if left.field.lambda_grid.size != right.field.lambda_grid.size:
        raise ValueError("M4b same-index comparison requires identical lambda-axis lengths.")
    modes = min(left.spectral.modes_lambda, right.spectral.modes_lambda)
    if stage == "first_fft_retained":
        left_value = m3._concat_q_slices(left.spectral.pre_pos[:, :, :, :modes], left.spectral.pre_neg[:, :, :, :modes])
        right_value = m3._concat_q_slices(right.spectral.pre_pos[:, :, :, :modes], right.spectral.pre_neg[:, :, :, :modes])
    elif stage == "post_weight_spectral":
        left_value = m3._concat_q_slices(left.spectral.post_pos[:, :, :, :modes], left.spectral.post_neg[:, :, :, :modes])
        right_value = m3._concat_q_slices(right.spectral.post_pos[:, :, :, :modes], right.spectral.post_neg[:, :, :, :modes])
    else:
        raise ValueError(f"Unsupported M4b spectral stage: {stage}")
    return m3._metrics(left_value, right_value)


def _reference_distance_change(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """以 T1200 reference 距离的变化描述干预是否具有恢复方向。"""

    before_distance = float(before["relative_l2_difference"])
    after_distance = float(after["relative_l2_difference"])
    return {
        "original_to_t1200_reference_relative_l2_difference": before_distance,
        "clamped_to_t1200_reference_relative_l2_difference": after_distance,
        "intervention_reference_distance_reduction": before_distance - after_distance,
        "intervention_reference_distance_reduction_fraction": (
            (before_distance - after_distance) / (before_distance + EPSILON)
        ),
        "original_to_t1200_reference_normalized_rmse": float(before["normalized_rmse"]),
        "clamped_to_t1200_reference_normalized_rmse": float(after["normalized_rmse"]),
        "original_to_t1200_reference_cosine_similarity": float(before["cosine_similarity"]),
        "clamped_to_t1200_reference_cosine_similarity": float(after["cosine_similarity"]),
        "reference_cosine_similarity_change": (
            float(after["cosine_similarity"]) - float(before["cosine_similarity"])
        ),
    }


def build_stage_rows(captures: dict[str, ArmCapture]) -> list[dict[str, Any]]:
    """生成 M4b 主证据表：同长度干预响应及其相对 T1200 reference 的位置。"""

    rows: list[dict[str, Any]] = []
    reference = captures["T1200_original"]
    spatial_stages = (
        ("normalized_input_prefix", "normalized_input_prefix"),
        ("lifted_feature_prefix", "lifted_feature_prefix"),
        ("first_spectral_input_prefix", "first_spectral_input_prefix"),
        ("spectral_branch_output_prefix", "spectral_branch_output_prefix"),
        ("first_block_output_prefix", "first_block_output_prefix"),
        ("final_prediction_prefix", "final_prediction_prefix"),
    )
    for length_label in ("T1800", "T2400"):
        original = captures[f"{length_label}_original"]
        clamped = captures[f"{length_label}_lambda_clamped"]
        comparison = f"{original.arm_name}_vs_{clamped.arm_name}"
        for stage_name, attribute in spatial_stages:
            intervention_metrics = m3._metrics(
                getattr(original, attribute), getattr(clamped, attribute)
            )
            original_reference = m3._metrics(
                getattr(original, attribute), getattr(reference, attribute)
            )
            clamped_reference = m3._metrics(
                getattr(clamped, attribute), getattr(reference, attribute)
            )
            rows.append({
                "comparison": comparison,
                "total_length": int(original.field.lambda_grid.size),
                "stage": stage_name,
                "view": "shared_spatial_prefix",
                "original_vs_clamped_relative_l2_difference": intervention_metrics["relative_l2_difference"],
                "original_vs_clamped_normalized_rmse": intervention_metrics["normalized_rmse"],
                "original_vs_clamped_cosine_similarity": intervention_metrics["cosine_similarity"],
                **_reference_distance_change(original_reference, clamped_reference),
            })
        for stage_name in ("first_fft_retained", "post_weight_spectral"):
            intervention_metrics = _same_index_spectral_metrics(original, clamped, stage_name)
            rows.append({
                "comparison": comparison,
                "total_length": int(original.field.lambda_grid.size),
                "stage": stage_name,
                "view": "same_discrete_index_same_fft_grid",
                "original_vs_clamped_relative_l2_difference": intervention_metrics["relative_l2_difference"],
                "original_vs_clamped_normalized_rmse": intervention_metrics["normalized_rmse"],
                "original_vs_clamped_cosine_similarity": intervention_metrics["cosine_similarity"],
                "original_to_t1200_reference_relative_l2_difference": None,
                "clamped_to_t1200_reference_relative_l2_difference": None,
                "intervention_reference_distance_reduction": None,
                "intervention_reference_distance_reduction_fraction": None,
                "original_to_t1200_reference_normalized_rmse": None,
                "clamped_to_t1200_reference_normalized_rmse": None,
                "original_to_t1200_reference_cosine_similarity": None,
                "clamped_to_t1200_reference_cosine_similarity": None,
                "reference_cosine_similarity_change": None,
            })
    return rows


def _secondary_truth_reference_row(arm: ArmCapture) -> dict[str, Any]:
    """把 shared-prefix truth context 指标写为明确非性能语义的紧凑 CSV 行。"""

    metrics = arm.secondary_truth_reference_prefix_metrics
    return {
        "arm": arm.arm_name,
        "length_label": arm.length_label,
        "intervention": arm.intervention,
        "reference_type": "secondary_truth_reference",
        "scientific_scope": "mechanism_intervention_context_only",
        "not_formal_model_performance": True,
        "region": "shared_prefix_only",
        "mse": metrics["mse"],
        "global_relative_l2": metrics["global_relative_l2"],
        "mean_per_q_relative_l2": metrics["mean_per_q_relative_l2"],
        "median_per_q_relative_l2": metrics["median_per_q_relative_l2"],
        "p95_per_q_relative_l2": metrics["p95_per_q_relative_l2"],
        "max_per_q_relative_l2": metrics["max_per_q_relative_l2"],
    }


def build_secondary_truth_reference_rows(captures: dict[str, ArmCapture]) -> list[dict[str, Any]]:
    """仅输出 long arm 的 shared-prefix 次级 truth-reference 上下文。"""

    return [
        _secondary_truth_reference_row(captures[arm_name])
        for arm_name in (
            "T1800_original",
            "T1800_lambda_clamped",
            "T2400_original",
            "T2400_lambda_clamped",
        )
    ]


def secondary_truth_reference_change(original: ArmCapture, clamped: ArmCapture) -> dict[str, Any]:
    """中性记录同一 shared prefix 对 truth reference 的变化，不作性能归因。"""

    original_value = float(original.secondary_truth_reference_prefix_metrics["mean_per_q_relative_l2"])
    clamped_value = float(clamped.secondary_truth_reference_prefix_metrics["mean_per_q_relative_l2"])
    change = original_value - clamped_value
    return {
        "original_prefix_mean_per_q_relative_l2": original_value,
        "clamped_prefix_mean_per_q_relative_l2": clamped_value,
        "truth_reference_absolute_change_original_minus_clamped": change,
        "truth_reference_fractional_change_original_minus_clamped": change / (original_value + EPSILON),
        "not_formal_model_performance": True,
    }

def lifting_decomposition(
    normalized_input: np.ndarray,
    model: torch.nn.Module,
    *,
    q_channel_index: int,
    lambda_channel_index: int,
    shared_length: int,
) -> dict[str, dict[str, float]]:
    """只读分解 input projection 的 Q、lambda 与 bias 能量。"""

    projection = getattr(model, "input_projection", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError("M4b lifting decomposition requires a torch.nn.Linear input_projection.")
    values = torch.from_numpy(np.asarray(normalized_input)).float().cpu()
    weight = projection.weight.detach().cpu()
    bias = projection.bias.detach().cpu()
    q_term = values[..., q_channel_index : q_channel_index + 1] * weight[:, q_channel_index].view(1, 1, 1, -1)
    lambda_term = values[..., lambda_channel_index : lambda_channel_index + 1] * weight[:, lambda_channel_index].view(1, 1, 1, -1)
    bias_term = bias.view(1, 1, 1, -1).expand_as(q_term)

    def summarize(start: int, stop: int) -> dict[str, float]:
        q_energy = float(torch.sum(q_term[:, :, start:stop, :] ** 2).item())
        lambda_energy = float(torch.sum(lambda_term[:, :, start:stop, :] ** 2).item())
        bias_energy = float(torch.sum(bias_term[:, :, start:stop, :] ** 2).item())
        denominator = q_energy + lambda_energy + bias_energy
        return {
            "q_energy": q_energy,
            "lambda_energy": lambda_energy,
            "bias_energy": bias_energy,
            "lambda_fraction_of_decomposed_energy": lambda_energy / (denominator + EPSILON),
        }

    width = int(values.shape[2])
    result = {"prefix": summarize(0, shared_length), "full": summarize(0, width)}
    if shared_length < width:
        result["appended"] = summarize(shared_length, width)
    return result


def _arm_summary(arm: ArmCapture) -> dict[str, Any]:
    """生成不包含预测或隐藏张量的单 arm 摘要。"""

    return {
        "arm_name": arm.arm_name,
        "length_label": arm.length_label,
        "intervention": arm.intervention,
        "N": int(arm.field.lambda_grid.size),
        "lambda_min": float(arm.field.lambda_grid[0]),
        "lambda_max": float(arm.field.lambda_grid[-1]),
        "secondary_truth_reference_prefix_metrics": arm.secondary_truth_reference_prefix_metrics,
        "first_layer_spectral_energy": {
            "full_fft_energy": arm.spectral.full_fft_energy,
            "retained_input_energy": arm.spectral.retained_input_energy,
            "retained_output_energy": arm.spectral.retained_output_energy,
            "retained_lambda_indices": [0, int(arm.spectral.modes_lambda - 1)],
            "replicated_spectral_output_vs_hook": arm.spectral.replicated_output_metrics,
        },
    }


def build_summary(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    pair_validation: dict[str, Any],
    pair_path: Path,
    captures: dict[str, ArmCapture],
    clamp_details: dict[str, dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    model: torch.nn.Module,
    input_channel_names: tuple[str, ...],
    stats: FieldNormalizationStats,
    short_lambda_max_raw: float,
    clamp_upper_bound: float,
    q_channel_index: int,
    lambda_channel_index: int,
    lifting: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """生成可审计 M4b 摘要，不序列化大数组或中间张量。"""

    secondary_truth_changes = {
        label: secondary_truth_reference_change(
            captures[f"{label}_original"], captures[f"{label}_lambda_clamped"]
        )
        for label in ("T1800", "T2400")
    }
    return {
        "schema_version": "1.0",
        "diagnostic_type": "fno2d_lambda_coordinate_clamp_effect_m4b",
        "status": "completed",
        "tasks": {
            "training": str(args.training_task_name),
            "short": captures["T1200_original"].field.task_name,
            "medium": captures["T1800_original"].field.task_name,
            "long": captures["T2400_original"].field.task_name,
        },
        "model_name": str(args.model_name),
        "checkpoint_path": _relative_path(checkpoint_path),
        "model_config": _as_json_value(checkpoint.get("config", {}).get("model_config", {})),
        "model_architecture": {
            "modes1": int(model.modes1),
            "modes2": int(model.modes2),
            "width": int(model.width),
            "depth": int(model.depth),
        },
        "stage2_prefix_validation": {
            "artifact_path": _relative_path(pair_path),
            "pair_classification": {
                key: pair_validation["pair_classification"][key]
                for key in ("short_to_medium", "short_to_long", "medium_to_long")
            },
            "scientific_reuse": {
                key: pair_validation["scientific_reuse"][key]
                for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1")
            },
        },
        "canonical_q": {
            "source_identity_order": "train_then_val_then_test_original_row_order",
            "model_input_order": "stable_ascending_Q_full_field",
            "exact_match_across_lengths": True,
            "raw_truth_prefixes_exact": True,
        },
        "normalization": {
            "required_method": "standard",
            "statistics_source": "checkpoint_training_dataset",
            "values": _as_json_value(stats.to_dict()),
            "input_channel_order": list(input_channel_names),
            "q_channel_index": q_channel_index,
            "lambda_channel_index": lambda_channel_index,
        },
        "intervention_semantics": intervention_semantics(),
        "intervention": {
            "definition": "For T1800 and T2400 only, clamp appended normalized lambda values above the T1200 normalized upper bound; keep Q, shared prefix, tensor shape, checkpoint, truth, and FFT grid unchanged.",
            "short_lambda_train_max_raw": short_lambda_max_raw,
            "lambda_train_max_normalized": clamp_upper_bound,
            "same_length_control": "Original versus clamped comparisons hold N, FFT frequency grid, Q field, truth, checkpoint, and weights fixed within each length.",
            "clamp_details": clamp_details,
        },
        "arms": {name: _arm_summary(capture) for name, capture in captures.items()},
        "primary_intervention_response": {
            "primary_evidence_table": "m4b_stage_comparison.csv",
            "comparison": "same_length_original_vs_lambda_clamped_over_shared_T1200_prefix",
            "reference_recovery": "For spatial stages, compare both long arms to T1200_original and report intervention_reference_distance_reduction.",
            "evidence_classification": "PENDING_SCIENTIFIC_REVIEW_NO_FIXED_NUMERICAL_THRESHOLD",
            "classification_guidance": {
                "STRONG_CONTRIBUTION_SIGNAL": "Large intervention response and substantial movement toward T1200 reference.",
                "MODERATE_CONTRIBUTION_SIGNAL": "Meaningful but incomplete movement toward T1200 reference.",
                "WEAK_CONTRIBUTION_SIGNAL": "Only modest response while strong length sensitivity remains.",
                "NO_CLEAR_CONTRIBUTION_SIGNAL": "Negligible or inconsistent response.",
            },
        },
        "metric_definitions": {
            "primary_same_length_intervention_response": {
                "scope": "original_vs_lambda_clamped_shared_prefix_within_same_long_length",
                "metrics": ["relative_l2_difference", "normalized_rmse", "cosine_similarity"],
            },
            "t1200_reference_recovery": {
                "scope": "spatial_shared_prefix_stages_only",
                "metrics": [
                    "original_to_t1200_reference_relative_l2_difference",
                    "clamped_to_t1200_reference_relative_l2_difference",
                    "intervention_reference_distance_reduction",
                    "intervention_reference_distance_reduction_fraction",
                ],
            },
            "secondary_truth_reference": {
                "scope": "shared_prefix_only",
                "scientific_scope": "mechanism_intervention_context_only",
                "not_formal_model_performance": True,
            },
        },
        "secondary_truth_reference": {
            "scope": "shared_prefix_only",
            "scientific_scope": "mechanism_intervention_context_only",
            "not_formal_model_performance": True,
            "changes": secondary_truth_changes,
        },
        "captured_stages": [
            "normalized_input",
            "input_projection_output",
            "blocks[0].spectral_conv_input",
            "replicated_rfft2",
            "replicated_post_weight_coefficients",
            "blocks[0].spectral_conv_output",
            "blocks[0]_output",
            "final_prediction",
        ],
        "hooks_used": [
            "input_projection forward hook",
            "blocks[0].spectral_conv forward pre-hook",
            "blocks[0].spectral_conv forward hook",
            "blocks[0] forward hook",
        ],
        "spectral_comparison": {
            "view": "same_discrete_index_same_fft_grid",
            "reason": "Each original/clamped pair has identical N and lambda FFT grid; physical-frequency remapping is not part of this within-length intervention comparison.",
        },
        "lifting_decomposition": lifting,
        "frozen_protocol": {
            "model_eval": True,
            "torch_no_grad": True,
            "one_normal_model_forward_per_arm": True,
            "arms": [
                "T1200_original",
                "T1800_original",
                "T1800_lambda_clamped",
                "T2400_original",
                "T2400_lambda_clamped",
            ],
            "total_model_forwards": 5,
            "optimizer": False,
            "scheduler": False,
            "backward": False,
            "adaptation": False,
            "fine_tuning": False,
            "autoregression": False,
            "prediction_feedback": False,
            "teacher_forcing": False,
        },
        "interpretation_boundary": (
            "This is a controlled frozen-forward intervention, not a production fix. "
            "A clamp response measures the contribution of appended out-of-training-range "
            "lambda-coordinate magnitude under this intervention; it does not uniquely assign "
            "causality among coordinate, global Fourier, or discrete-mode mechanisms."
        ),
        "stage_comparison_row_count": len(stage_rows),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": str(args.device),
        },
        "git_commit": _git_commit(),
        "output_files": list(OUTPUT_FILENAMES),
    }


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """写入紧凑 CSV，None 写为空字段。"""

    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_output_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    secondary_truth_reference_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    """独占创建目录，并且只写入三个紧凑 M4b 产物。"""

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_as_json_value(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(
        secondary_truth_reference_rows,
        output_dir / OUTPUT_FILENAMES[1],
        [
            "arm",
            "length_label",
            "intervention",
            "reference_type",
            "scientific_scope",
            "not_formal_model_performance",
            "region",
            "mse",
            "global_relative_l2",
            "mean_per_q_relative_l2",
            "median_per_q_relative_l2",
            "p95_per_q_relative_l2",
            "max_per_q_relative_l2",
        ],
    )
    write_csv(
        stage_rows,
        output_dir / OUTPUT_FILENAMES[2],
        [
            "comparison",
            "total_length",
            "stage",
            "view",
            "original_vs_clamped_relative_l2_difference",
            "original_vs_clamped_normalized_rmse",
            "original_vs_clamped_cosine_similarity",
            "original_to_t1200_reference_relative_l2_difference",
            "clamped_to_t1200_reference_relative_l2_difference",
            "intervention_reference_distance_reduction",
            "intervention_reference_distance_reduction_fraction",
            "original_to_t1200_reference_normalized_rmse",
            "clamped_to_t1200_reference_normalized_rmse",
            "original_to_t1200_reference_cosine_similarity",
            "clamped_to_t1200_reference_cosine_similarity",
            "reference_cosine_similarity_change",
        ],
    )


def run_m4b(*, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """执行正式 M4b；调用方负责在新目录写入紧凑结果。"""

    if args.output_dir.exists():
        raise FileExistsError(f"M4b output directory already exists: {args.output_dir}")
    pair_validation = load_required_pair_validation(args.dataset_pair_validation_json)
    short = load_task_raw_field(str(args.short_task_name))
    medium = load_task_raw_field(str(args.medium_task_name))
    long = load_task_raw_field(str(args.long_task_name))
    validate_triplet(short, medium, long)
    checkpoint_path = args.checkpoint_path or (
        PROJECT_ROOT / "outputs" / str(args.training_task_name) / str(args.model_name) / "checkpoints" / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = load_checkpoint_2d(checkpoint_path=checkpoint_path, device=str(args.device))
    stats, input_channel_names, q_channel_index, lambda_channel_index = require_checkpoint_input_contract(checkpoint)
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))
    model.eval()
    shared_length = int(short.lambda_grid.size)
    raw_max, clamp_upper_bound = derive_lambda_clamp_bound(short.lambda_grid, stats, lambda_channel_index)

    fields = {"T1200": short, "T1800": medium, "T2400": long}
    original_inputs: dict[str, np.ndarray] = {}
    for label, field in fields.items():
        x_raw, _ = build_model_input(field.canonical_q, field.lambda_grid, field.canonical_truth)
        original_inputs[label] = normalize_input_field(x=x_raw, stats=stats)

    clamped_inputs: dict[str, np.ndarray] = {}
    clamp_details: dict[str, dict[str, Any]] = {}
    for label in ("T1800", "T2400"):
        clamped, details = clamp_appended_lambda_channel(
            original_inputs[label],
            shared_length=shared_length,
            q_channel_index=q_channel_index,
            lambda_channel_index=lambda_channel_index,
            clamp_upper_bound=clamp_upper_bound,
        )
        clamped_inputs[label] = clamped
        clamp_details[label] = details

    captures = {
        "T1200_original": _make_arm_capture(
            arm_name="T1200_original", length_label="T1200", intervention="original", field=short,
            normalized_input=original_inputs["T1200"], model=model, checkpoint=checkpoint,
            device=str(args.device), shared_length=shared_length,
        ),
        "T1800_original": _make_arm_capture(
            arm_name="T1800_original", length_label="T1800", intervention="original", field=medium,
            normalized_input=original_inputs["T1800"], model=model, checkpoint=checkpoint,
            device=str(args.device), shared_length=shared_length,
        ),
        "T1800_lambda_clamped": _make_arm_capture(
            arm_name="T1800_lambda_clamped", length_label="T1800", intervention="lambda_clamped", field=medium,
            normalized_input=clamped_inputs["T1800"], model=model, checkpoint=checkpoint,
            device=str(args.device), shared_length=shared_length,
        ),
        "T2400_original": _make_arm_capture(
            arm_name="T2400_original", length_label="T2400", intervention="original", field=long,
            normalized_input=original_inputs["T2400"], model=model, checkpoint=checkpoint,
            device=str(args.device), shared_length=shared_length,
        ),
        "T2400_lambda_clamped": _make_arm_capture(
            arm_name="T2400_lambda_clamped", length_label="T2400", intervention="lambda_clamped", field=long,
            normalized_input=clamped_inputs["T2400"], model=model, checkpoint=checkpoint,
            device=str(args.device), shared_length=shared_length,
        ),
    }
    stage_rows = build_stage_rows(captures)
    secondary_truth_reference_rows = build_secondary_truth_reference_rows(captures)
    lifting = {
        name: lifting_decomposition(
            capture.normalized_input,
            model,
            q_channel_index=q_channel_index,
            lambda_channel_index=lambda_channel_index,
            shared_length=shared_length,
        )
        for name, capture in captures.items()
        if capture.length_label in ("T1800", "T2400")
    }
    summary = build_summary(
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        pair_validation=pair_validation,
        pair_path=args.dataset_pair_validation_json,
        captures=captures,
        clamp_details=clamp_details,
        stage_rows=stage_rows,
        model=model,
        input_channel_names=input_channel_names,
        stats=stats,
        short_lambda_max_raw=raw_max,
        clamp_upper_bound=clamp_upper_bound,
        q_channel_index=q_channel_index,
        lambda_channel_index=lambda_channel_index,
        lifting=lifting,
    )
    return summary, secondary_truth_reference_rows, stage_rows


def main() -> None:
    """执行 M4b CLI。"""

    args = parse_args()
    summary, secondary_truth_reference_rows, stage_rows = run_m4b(args=args)
    write_output_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        secondary_truth_reference_rows=secondary_truth_reference_rows,
        stage_rows=stage_rows,
    )
    print("M4b lambda-coordinate clamp diagnostic completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
