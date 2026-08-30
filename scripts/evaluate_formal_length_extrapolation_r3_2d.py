"""R3-B1：物理频率锚点谱权重的正式三长度冻结 A1 评估。"""

from __future__ import annotations

import argparse
import csv
import json
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

from scripts import evaluate_formal_length_extrapolation_2d as formal  # noqa: E402
from scripts import evaluate_formal_length_extrapolation_r2_2d as r2eval  # noqa: E402
from scripts.run_analysis_2d import load_checkpoint_2d  # noqa: E402
from src.models.fno2d.fno2d_physical_frequency import build_physical_frequency_fno2d_model  # noqa: E402
from src.training.fno2d.normalization_2d import FieldNormalizationStats  # noqa: E402

EXPERIMENT_TYPE = "r3_physical_frequency_spectral_training"
REPAIR_CLASS = "SPECTRAL_PARAMETERIZATION_REDESIGN"
COORDINATE_REPRESENTATION = ("Q", "s", "ell")
OUTPUT_FILENAMES = ("r3_a1_length_extrapolation_summary.json", "r3_per_q_metrics.csv", "r3_lambda_window_metrics.csv")


@dataclass(frozen=True)
class R3CheckpointProvenance:
    """R3 checkpoint 中必须能重建的坐标、锚点和频谱语义。"""

    model_config: dict[str, Any]
    input_channel_order: tuple[str, str, str]
    reference_domain_length: float
    training_lengths: tuple[int, ...]
    validation_lengths: tuple[int, ...]
    input_normalization_policy: dict[str, Any]
    output_normalization_policy: str
    target_transform: str
    delta_lambda: float
    anchor_frequencies: tuple[float, ...]


class _R2CompatibleProvenance:
    """为 R2 坐标与 normalization helper 提供同形只读字段。"""

    def __init__(self, provenance: R3CheckpointProvenance) -> None:
        self.input_channel_order = provenance.input_channel_order
        self.reference_domain_length = provenance.reference_domain_length
        self.input_normalization_policy = provenance.input_normalization_policy
        self.target_transform = provenance.target_transform


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 R3 正式三长度冻结评估的明确输入。"""
    parser = argparse.ArgumentParser(description="Run formal frozen R3 physical-frequency-anchor FNO2D A1 evaluation.")
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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """拒绝缺失的 checkpoint 嵌套字段。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"R3 checkpoint requires mapping field {name}.")
    return value


def _require_lengths(value: Any, name: str) -> tuple[int, ...]:
    """验证训练和验证长度 provenance。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"R3 checkpoint requires non-empty {name}.")
    result = tuple(int(item) for item in value)
    if min(result) <= 0 or len(set(result)) != len(result):
        raise ValueError(f"R3 checkpoint has invalid {name}.")
    return result


def _anchor_lists_are_scientifically_equivalent(stored: Sequence[float], canonical: np.ndarray) -> bool:
    """允许历史 float32 model_config 列表，但要求其与 canonical provenance 等价。"""
    values = np.asarray(tuple(float(value) for value in stored), dtype=np.float64)
    if values.shape != canonical.shape:
        return False
    scale = max(1.0, float(np.max(np.abs(canonical))))
    precision = np.finfo(np.float32).eps
    return bool(np.allclose(values, canonical, rtol=16.0 * precision, atol=16.0 * precision * scale))


def validate_r3_checkpoint_provenance(checkpoint: Mapping[str, Any]) -> R3CheckpointProvenance:
    """在模型构造前严格验证 R3-B1 provenance，拒绝 R2、B2 和不完整 checkpoint。"""
    config = _require_mapping(checkpoint.get("config"), "config")
    if config.get("experiment_type") != EXPERIMENT_TYPE or config.get("repair_class") != REPAIR_CLASS:
        raise ValueError("Checkpoint is not an R3 physical-frequency spectral training checkpoint.")
    model_config = dict(_require_mapping(config.get("model_config"), "model_config"))
    if model_config.get("model_type") != "fno2d_physical_frequency" or int(model_config.get("in_dim", -1)) != 3:
        raise ValueError("R3 checkpoint requires fno2d_physical_frequency with in_dim=3.")
    if tuple(config.get("coordinate_representation", ())) != COORDINATE_REPRESENTATION or tuple(config.get("input_channel_names", ())) != COORDINATE_REPRESENTATION:
        raise ValueError("R3 checkpoint requires coordinate_representation=['Q', 's', 'ell'].")
    if config.get("absolute_lambda_input") is not False:
        raise ValueError("R3 checkpoint must explicitly exclude absolute lambda input.")
    if config.get("normalization") != "standard" or config.get("target_transform") != "raw":
        raise ValueError("R3 checkpoint must preserve R2 standard normalization and raw target transform.")
    if config.get("spectral_parameterization") != "physical_frequency_anchor_interpolation" or config.get("physical_frequency_formula") != "k / (N * delta_lambda)" or config.get("complex_interpolation") != "cartesian_linear":
        raise ValueError("R3 physical-frequency interpolation provenance is missing or unsupported.")
    if config.get("runtime_retained_mode_policy") != "fixed_discrete_indices_k_0_to_modes2_minus_1" or config.get("physical_cutoff_repair") is not False or config.get("physical_bandwidth_shrinkage_repaired") is not False:
        raise ValueError("R3-B1 must retain the fixed discrete-index policy and exclude bandwidth repair.")
    if config.get("global_fft_structure_unchanged") is not True or config.get("hypernetwork") is not False or config.get("dynamic_spectral_weights") is not False:
        raise ValueError("R3-B1 must preserve global FFT and exclude hypernetwork/dynamic-weight paths.")
    modes2 = int(model_config.get("modes2", -1))
    anchors = tuple(float(value) for value in config.get("anchor_frequency_values", ()))
    if modes2 <= 1 or len(anchors) != modes2 or int(config.get("num_lambda_frequency_anchors", -1)) != modes2:
        raise ValueError("R3 anchor count must equal modes2.")
    anchor_array = np.asarray(anchors, dtype=np.float64)
    if not np.isclose(anchor_array[0], 0.0) or not np.all(np.diff(anchor_array) > 0.0) or not np.allclose(np.diff(anchor_array), np.diff(anchor_array)[0], rtol=1e-12, atol=1e-14):
        raise ValueError("R3 anchor_frequency_values must be uniformly increasing from zero.")
    model_anchor_values = model_config.get("anchor_frequencies", ())
    if not isinstance(model_anchor_values, (list, tuple)) or not _anchor_lists_are_scientifically_equivalent(model_anchor_values, anchor_array):
        raise ValueError("R3 model_config anchor_frequencies are inconsistent with canonical provenance.")
    lengths = _require_lengths(config.get("train_lengths"), "train_lengths")
    validation = _require_lengths(config.get("validation_lengths"), "validation_lengths")
    delta = float(model_config.get("delta_lambda", np.nan))
    expected_maximum = (modes2 - 1) / (min(lengths) * delta)
    if not np.isfinite(delta) or delta <= 0.0 or not np.isclose(anchor_array[-1], expected_maximum, rtol=1e-12, atol=1e-14):
        raise ValueError("R3 anchor support must derive from shortest training length and delta_lambda.")
    policy = dict(_require_mapping(config.get("input_normalization_policy"), "input_normalization_policy"))
    expected_policy = {"Q": "standard_full_source_train_field", "s": "identity_dimensionless", "ell": "identity_dimensionless_L_over_L_ref", "target": "standard_full_source_train_field"}
    if any(policy.get(key) != value for key, value in expected_policy.items()) or policy.get("fit_uses_validation_lengths") is not False or policy.get("fit_uses_formal_long_lengths") is not False:
        raise ValueError("R3 checkpoint does not preserve R2 normalization policy.")
    if config.get("output_normalization_policy") != "standard_full_source_train_field":
        raise ValueError("R3 output normalization provenance is invalid.")
    return R3CheckpointProvenance(model_config=model_config, input_channel_order=COORDINATE_REPRESENTATION, reference_domain_length=float(config["L_ref"]), training_lengths=lengths, validation_lengths=validation, input_normalization_policy=policy, output_normalization_policy=str(config["output_normalization_policy"]), target_transform="raw", delta_lambda=delta, anchor_frequencies=anchors)


def build_r3_model(checkpoint: Mapping[str, Any], provenance: R3CheckpointProvenance, device: str) -> torch.nn.Module:
    """以 canonical checkpoint provenance 重建 R3 锚点，再加载保存的模型状态。"""
    config = dict(provenance.model_config)
    config.pop("model_type", None)
    config["anchor_frequencies"] = [float(value) for value in provenance.anchor_frequencies]
    model = build_physical_frequency_fno2d_model(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    expected = torch.as_tensor(provenance.anchor_frequencies, dtype=torch.float64, device=model.anchor_frequencies.device)
    if not torch.allclose(model.anchor_frequencies, expected, rtol=1e-12, atol=1e-14):
        raise ValueError("Saved R3 anchor buffer is inconsistent with canonical checkpoint provenance.")
    model.eval()
    return model


def runtime_physical_frequencies(*, total_length: int, delta_lambda: float, modes2: int) -> np.ndarray:
    """按 established DFT logical-period 约定生成固定 retained indices 的 xi_k。"""
    m2 = min(int(modes2), int(total_length) // 2 + 1)
    return np.arange(m2, dtype=np.float64) / (int(total_length) * float(delta_lambda))


def validate_runtime_anchor_support(*, total_length: int, delta_lambda: float, provenance: R3CheckpointProvenance) -> np.ndarray:
    """拒绝任何会要求锚点外推或静默 clamp 的 formal runtime。"""
    frequencies = runtime_physical_frequencies(total_length=total_length, delta_lambda=delta_lambda, modes2=int(provenance.model_config["modes2"]))
    if not np.isclose(delta_lambda, provenance.delta_lambda, rtol=1e-10, atol=1e-12):
        raise ValueError("Formal runtime delta_lambda differs from R3 training delta_lambda.")
    if np.any(frequencies < provenance.anchor_frequencies[0] - 1e-12) or np.any(frequencies > provenance.anchor_frequencies[-1] + 1e-12):
        raise ValueError("Formal runtime retained physical frequency lies outside R3 anchor support.")
    return frequencies


def run_frozen_inference(*, model: torch.nn.Module, checkpoint: Mapping[str, Any], field: formal.CanonicalQField, provenance: R3CheckpointProvenance, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    """对一个长度执行一次 R3 冻结直接前向并记录 runtime xi bins。"""
    delta = r2eval.derive_uniform_delta_lambda(field.lambda_grid)
    frequencies = validate_runtime_anchor_support(total_length=int(field.lambda_grid.size), delta_lambda=delta, provenance=provenance)
    model.validate_runtime_support(int(field.lambda_grid.size), delta)  # type: ignore[attr-defined]
    prediction, coordinates = r2eval.run_frozen_inference(model=model, checkpoint=checkpoint, field=field, provenance=_R2CompatibleProvenance(provenance), device=device)
    return prediction, {"delta_lambda": delta, "DFT_logical_domain_length_L": float(field.lambda_grid.size * delta), "retained_lambda_indices": list(range(int(frequencies.size))), "retained_physical_frequencies": [float(value) for value in frequencies], "anchor_frequency_min": provenance.anchor_frequencies[0], "anchor_frequency_max": provenance.anchor_frequencies[-1], "r2_coordinates": {"ell": coordinates.ell, "s_min": coordinates.s_min, "s_max": coordinates.s_max, "input_channel_order": list(provenance.input_channel_order)}}


def evaluate_three_lengths(*, model: torch.nn.Module, checkpoint: Mapping[str, Any], provenance: R3CheckpointProvenance, short: formal.CanonicalQField, medium: formal.CanonicalQField, long: formal.CanonicalQField, device: str, window_width: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """按 formal A1 regions 对三长度各执行一次 R3 直接冻结前向。"""
    training_length = int(short.lambda_grid.size)
    results: dict[str, Any] = {}
    per_q_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for field in (short, medium, long):
        started = perf_counter()
        prediction, spectral_runtime = run_frozen_inference(model=model, checkpoint=checkpoint, field=field, provenance=provenance, device=device)
        total_length = int(field.lambda_grid.size)
        has_extra = total_length > training_length
        results[f"T{total_length}"] = {"task_name": field.task_name, "total_length": total_length, "prefix": formal.compute_region_metrics(prediction[:, :training_length, :], field.canonical_truth[:, :training_length, :], field.canonical_q), "extrapolation": formal.compute_region_metrics(prediction[:, training_length:, :], field.canonical_truth[:, training_length:, :], field.canonical_q) if has_extra else None, "full": formal.compute_region_metrics(prediction, field.canonical_truth, field.canonical_q), "r3_spectral_runtime": spectral_runtime, "inference_seconds": float(perf_counter() - started), "frozen_forward_passes": 1}
        per_q_rows.extend(formal.compute_per_q_rows(total_length=total_length, prediction=prediction, truth=field.canonical_truth, q_values=field.canonical_q, training_length=training_length))
        if has_extra:
            window_rows.extend(formal.compute_window_rows(total_length=total_length, prediction=prediction, truth=field.canonical_truth, lambda_grid=field.lambda_grid, training_length=training_length, window_width=window_width))
    return results, per_q_rows, window_rows


def build_summary(*, args: argparse.Namespace, checkpoint_path: Path, checkpoint: Mapping[str, Any], provenance: R3CheckpointProvenance, pair_validation_path: Path, pair_validation: Mapping[str, Any], short: formal.CanonicalQField, medium: formal.CanonicalQField, long: formal.CanonicalQField, results: Mapping[str, Any]) -> dict[str, Any]:
    """构造不含预测、隐藏张量或 FFT 矩阵的紧凑 R3 summary。"""
    stats = r2eval.load_normalization_stats_from_checkpoint(dict(checkpoint))
    r2eval.validate_r2_normalization_stats(stats, _R2CompatibleProvenance(provenance))
    return {"schema_version": "1.0", "experiment_type": EXPERIMENT_TYPE, "evaluation_type": "formal_frozen_one_shot_lambda_domain_length_extrapolation_r3", "status": "completed", "scientific_scope": "formal_A1_repair_evaluation", "training_task_name": str(args.training_task_name), "tasks": {"short": short.task_name, "medium": medium.task_name, "long": long.task_name}, "model_name": str(args.model_name), "checkpoint_path": str(checkpoint_path), "model_config": formal._as_json_value(provenance.model_config), "coordinate_representation": {"channel_order": list(provenance.input_channel_order), "absolute_lambda_input": False, "relative_coordinate_definition": "s = lambda / L", "domain_length_definition": "L = N * delta_lambda (DFT logical period)", "L_ref": provenance.reference_domain_length}, "spectral_parameterization": {"type": "physical_frequency_anchor_interpolation", "physical_frequency_formula": "k / (N * delta_lambda)", "anchor_frequency_values": list(provenance.anchor_frequencies), "anchor_support_source": "training_lengths_only", "complex_interpolation": "cartesian_linear", "runtime_retained_mode_policy": "fixed_discrete_indices_k_0_to_modes2_minus_1", "physical_bandwidth_shrinkage_repaired": False, "global_fft_structure_unchanged": True}, "normalization": {"input_policy": formal._as_json_value(provenance.input_normalization_policy), "values": formal._as_json_value(stats.to_dict())}, "target_transform": "raw", "checkpoint_training_lengths": list(provenance.training_lengths), "checkpoint_validation_lengths": list(provenance.validation_lengths), "stage2_prefix_validation": formal._as_json_value(pair_validation), "frozen_inference": {"one_full_forward_pass_per_total_length": True, "total_forward_passes": 3, "optimizer": False, "scheduler": False, "backward": False, "training": False, "adaptation": False, "autoregressive_rollout": False, "teacher_forcing": False, "prediction_feedback": False}, "metric_aggregation": {"evaluation_space": "raw_physical_xyz_float64", "truth_source": "raw_dataset_truth_float64", "mean_per_q_relative_l2": "mean_of_per_Q_flattened_relative_l2; R0_R1_R2_comparable"}, "window_coordinate": "physical_lambda", "window_width": float(args.window_width), "git_commit": r2eval._git_commit(), "results": formal._as_json_value(results), "output_files": list(OUTPUT_FILENAMES)}


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """写入 compact CSV。"""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_output_artifacts(*, output_dir: Path, summary: Mapping[str, Any], per_q_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> None:
    """拒绝覆盖且只写三份 compact formal artifact。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(formal._as_json_value(dict(summary)), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_csv(per_q_rows, output_dir / OUTPUT_FILENAMES[1], ["total_length", "Q", "prefix_mse", "prefix_relative_l2", "extrapolation_mse", "extrapolation_relative_l2", "full_mse", "full_relative_l2"])
    _write_csv(window_rows, output_dir / OUTPUT_FILENAMES[2], ["total_length", "lambda_start", "lambda_end", "distance_from_training_boundary_start", "distance_from_training_boundary_end", "point_count", "interval_right_closed", "mse", "global_relative_l2", "mean_per_q_relative_l2"])


def main(argv: Sequence[str] | None = None) -> None:
    """执行 R3 formal A1，严格保持三次冻结前向和 Stage-2 前提。"""
    args = parse_args(argv)
    if args.output_dir.exists() or not np.isfinite(args.window_width) or args.window_width <= 0.0:
        raise ValueError("Output directory must not exist and window width must be positive.")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint does not exist.")
    pair_validation = formal.load_required_pair_validation(args.dataset_pair_validation_json)
    short, medium, long = (formal.load_task_raw_field(str(name)) for name in (args.short_task_name, args.medium_task_name, args.long_task_name))
    formal.validate_triplet(short, medium, long)
    checkpoint = load_checkpoint_2d(checkpoint_path=args.checkpoint_path, device=str(args.device))
    provenance = validate_r3_checkpoint_provenance(checkpoint)
    if str(checkpoint["config"].get("source_task")) != str(args.training_task_name):
        raise ValueError("R3 checkpoint source_task does not match --training-task-name.")
    short_delta = r2eval.derive_uniform_delta_lambda(short.lambda_grid)
    if not np.isclose(provenance.reference_domain_length, short.lambda_grid.size * short_delta, rtol=1e-10, atol=1e-12):
        raise ValueError("R3 checkpoint L_ref does not match short logical domain length.")
    model = build_r3_model(checkpoint, provenance, str(args.device))
    results, per_q_rows, window_rows = evaluate_three_lengths(model=model, checkpoint=checkpoint, provenance=provenance, short=short, medium=medium, long=long, device=str(args.device), window_width=float(args.window_width))
    summary = build_summary(args=args, checkpoint_path=args.checkpoint_path, checkpoint=checkpoint, provenance=provenance, pair_validation_path=args.dataset_pair_validation_json, pair_validation=pair_validation, short=short, medium=medium, long=long, results=results)
    write_output_artifacts(output_dir=args.output_dir, summary=summary, per_q_rows=per_q_rows, window_rows=window_rows)


if __name__ == "__main__":
    main()
