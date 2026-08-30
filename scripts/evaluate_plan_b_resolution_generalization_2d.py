"""Plan B fixed-domain frozen FNO2D resolution-generalization evaluator."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.evaluate_formal_length_extrapolation_2d import (  # noqa: E402
    CanonicalQField,
    build_canonical_q_field,
    build_model_input,
)
from scripts.run_analysis_2d import (  # noqa: E402
    load_checkpoint_2d,
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
    predict_2d_loader,
    recover_predictions_and_targets_to_raw_xyz,
)
from src.data_generation.plan_b_paired import (  # noqa: E402
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    SPLITS,
    load_dataset_artifact,
    validate_plan_b_ground_truth,
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


EPSILON = 1e-12
HISTORICAL_Q400_T1200_BASELINE = {
    "global_relative_l2": 0.0071953455,
    "mean_per_q_relative_l2": 0.0054274904,
}


@dataclass(frozen=True)
class FrozenCheckpointContract:
    """保存只可从 checkpoint 恢复、绝不在 evaluation 中拟合的状态。"""

    normalization_stats: FieldNormalizationStats
    target_transform_config: TargetTransformConfig
    model_config: dict[str, Any]
    provenance: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 Plan B 正式 evaluator 的最小命令行接口。"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one historical Q-only FNO2D checkpoint on paired Plan B "
            "T1200 and T2399 fields without normalization refit or adaptation."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--coarse-dataset-dir", required=True, type=Path)
    parser.add_argument("--fine-dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _json_value(value: Any) -> Any:
    """递归转换 NumPy 数值，并拒绝非有限 JSON 数值。"""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Evaluation output contains a non-finite float.")
        return value
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _relative_path(path: Path) -> str:
    """优先记录相对于项目根目录的路径。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _source_records(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """保留 dataset split 内原始 Q 身份与 canonical 映射所需来源记录。"""

    records: list[dict[str, Any]] = []
    offset = 0
    for split in SPLITS:
        q_values = np.asarray(arrays[f"x_{split}"], dtype=np.float64)
        for index in range(q_values.shape[0]):
            records.append(
                {
                    "source_split": split,
                    "source_index_within_split": int(index),
                    "source_concatenated_index": int(offset + index),
                }
            )
        offset += int(q_values.shape[0])
    return records


def build_canonical_field_from_artifact(directory: str | Path) -> CanonicalQField:
    """读取一个 Plan B artifact，并构造完整 Q400 的升序模型输入场。"""

    artifact = load_dataset_artifact(directory)
    arrays = artifact.arrays
    source_q = np.concatenate([np.asarray(arrays[f"x_{split}"]) for split in SPLITS], axis=0)[:, 0]
    source_truth = np.concatenate([np.asarray(arrays[f"y_{split}"]) for split in SPLITS], axis=0)
    task_name = str(artifact.metadata.get("task_name", artifact.directory.name))
    return build_canonical_q_field(
        task_name=task_name,
        source_q=source_q,
        source_truth=source_truth,
        lambda_grid=np.asarray(arrays["lambda_grid"], dtype=np.float64),
        source_records=_source_records(arrays),
    )


def validate_checkpoint_contract(checkpoint: dict[str, Any]) -> FrozenCheckpointContract:
    """在模型恢复前验证历史 T1200 Q-only FNO2D provenance 与冻结统计量。"""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no config dictionary.")
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint config has no model_config dictionary.")
    if str(model_config.get("model_type")) != "fno2d":
        raise ValueError("Plan B requires a historical fno2d checkpoint.")
    if int(model_config.get("in_dim", -1)) != 2 or int(model_config.get("out_dim", -1)) != 3:
        raise ValueError("Plan B requires the Q/lambda-to-xyz FNO2D channel contract (2 -> 3).")

    dataset_summary = config.get("dataset_summary")
    if not isinstance(dataset_summary, dict):
        raise ValueError("Checkpoint has no dataset_summary for Q-only provenance validation.")
    if dataset_summary.get("param_name") != "Q" or dataset_summary.get("vary_params_order") != ["Q"]:
        raise ValueError("Plan B requires a historical Q-only checkpoint with vary_params_order=['Q'].")
    train_summary = dataset_summary.get("train")
    if not isinstance(train_summary, dict) or int(train_summary.get("num_lambda", -1)) != COARSE_N_STEPS:
        raise ValueError("Plan B requires checkpoint training provenance at T=1200.")

    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform_config = load_target_transform_config_from_checkpoint(checkpoint)
    if len(stats.x_mean) != 2 or len(stats.y_mean) != 3:
        raise ValueError("Checkpoint normalization statistics do not match the Q/lambda-to-xyz contract.")
    if transform_config.mode not in {"raw", "residual_initial"}:
        raise ValueError("Checkpoint target transform is unsupported for Plan B evaluation.")

    return FrozenCheckpointContract(
        normalization_stats=stats,
        target_transform_config=transform_config,
        model_config=dict(model_config),
        provenance={
            "checkpoint_training_lambda_count": int(train_summary["num_lambda"]),
            "checkpoint_training_param_name": str(dataset_summary["param_name"]),
            "checkpoint_training_vary_params_order": list(dataset_summary["vary_params_order"]),
            "normalization_statistics_source": "checkpoint.config.dataset_summary.normalization_stats",
            "normalization_refit_during_evaluation": False,
            "target_transform_source": "checkpoint.config.dataset_summary.target_transform_config",
        },
    )


def validate_plan_b_inputs(
    coarse_dataset_dir: str | Path,
    fine_dataset_dir: str | Path,
) -> tuple[dict[str, Any], CanonicalQField, CanonicalQField]:
    """在 checkpoint 加载或 inference 之前执行全部 paired-dataset hard validation。"""

    qualification = validate_plan_b_ground_truth(coarse_dataset_dir, fine_dataset_dir)
    if qualification.get("structural_valid") is not True:
        raise ValueError("Plan B paired dataset structural validation failed; inference is forbidden.")
    coarse = build_canonical_field_from_artifact(coarse_dataset_dir)
    fine = build_canonical_field_from_artifact(fine_dataset_dir)
    if coarse.canonical_q.size != fine.canonical_q.size:
        raise ValueError("Plan B canonical Q counts differ.")
    if not np.array_equal(coarse.canonical_q, fine.canonical_q):
        raise ValueError("Plan B canonical Q identities differ.")
    if coarse.lambda_grid.size != COARSE_N_STEPS or fine.lambda_grid.size != FINE_N_STEPS:
        raise ValueError("Plan B canonical field lengths do not match Protocol v1.")
    return qualification, coarse, fine


def run_frozen_forward(
    *,
    model: torch.nn.Module,
    contract: FrozenCheckpointContract,
    field: CanonicalQField,
    device: str,
) -> np.ndarray:
    """用 checkpoint 冻结的变换和统计量执行一次 one-shot raw-xyz forward。"""

    x_raw, y_raw = build_model_input(
        field.canonical_q,
        field.lambda_grid,
        field.canonical_truth,
    )
    y_transformed = transform_output_field(y=y_raw, config=contract.target_transform_config)
    x_model = normalize_input_field(x=x_raw, stats=contract.normalization_stats)
    y_model = normalize_output_field(y=y_transformed, stats=contract.normalization_stats)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_model).float(), torch.from_numpy(y_model).float()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    predictions_model_space, targets_model_space = predict_2d_loader(
        model=model,
        loader=loader,
        device=device,
    )
    predictions_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=predictions_model_space,
        targets_model_space=targets_model_space,
        raw_targets_reference=y_raw,
        normalization_stats=contract.normalization_stats,
        target_transform_config=contract.target_transform_config,
    )
    prediction = np.asarray(predictions_raw[0], dtype=np.float32)
    if prediction.shape != field.canonical_truth.shape:
        raise ValueError(f"Frozen prediction shape={prediction.shape}, expected={field.canonical_truth.shape}.")
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Frozen inference produced non-finite predictions.")
    return prediction


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    """计算 raw physical xyz 空间的全局 Relative L2。"""

    return float(np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + EPSILON))


def compute_raw_metrics(prediction: np.ndarray, truth: np.ndarray, q_values: np.ndarray) -> dict[str, Any]:
    """计算 full field 或 common-node raw-physical 指标与 per-Q 汇总。"""

    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    q64 = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if prediction64.ndim != 3 or prediction64.shape[-1] != 3 or prediction64.shape != truth64.shape:
        raise ValueError("Raw metric inputs must share shape [Q,lambda,3].")
    if prediction64.shape[0] != q64.size or prediction64.shape[1] == 0:
        raise ValueError("Raw metric Q count or lambda length is invalid.")
    if not np.all(np.isfinite(prediction64)) or not np.all(np.isfinite(truth64)):
        raise ValueError("Raw metrics require finite predictions and truth.")

    per_q = np.asarray([_relative_l2(prediction64[index], truth64[index]) for index in range(q64.size)])
    worst_index = int(np.argmax(per_q))
    return {
        "evaluation_space": "raw_physical_xyz_float64",
        "point_count_per_q": int(prediction64.shape[1]),
        "global_mse": float(np.mean((prediction64 - truth64) ** 2)),
        "global_relative_l2": _relative_l2(prediction64, truth64),
        "mean_per_q_relative_l2": float(np.mean(per_q)),
        "per_q_relative_l2_summary": {
            "mean": float(np.mean(per_q)),
            "median": float(np.median(per_q)),
            "max": float(np.max(per_q)),
            "p95": float(np.percentile(per_q, 95.0)),
            "p99": float(np.percentile(per_q, 99.0)),
            "worst_q_index": worst_index,
            "worst_q_value": float(q64[worst_index]),
        },
    }


def compute_generalization_gap(fine_metrics: dict[str, Any], coarse_metrics: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """报告 fine 相对 coarse 的 ratio 与绝对差，不施加 pass/fail 阈值。"""

    result: dict[str, dict[str, float | None]] = {}
    for metric_name in ("global_relative_l2", "mean_per_q_relative_l2"):
        fine_value = float(fine_metrics[metric_name])
        coarse_value = float(coarse_metrics[metric_name])
        result[metric_name] = {
            "fine_minus_coarse": float(fine_value - coarse_value),
            "fine_divided_by_coarse": float(fine_value / coarse_value) if coarse_value != 0.0 else None,
        }
    return result


def _grid_summary(lambda_grid: np.ndarray) -> dict[str, Any]:
    """记录 Protocol v1 grid 的点数、端点和步长。"""

    values = np.asarray(lambda_grid, dtype=np.float64)
    delta = np.diff(values)
    return {
        "T": int(values.size),
        "lambda_min": float(values[0]),
        "lambda_max": float(values[-1]),
        "step_size": float(delta[0]) if delta.size else None,
    }


def _historical_baseline_cross_check(coarse_metrics: dict[str, Any]) -> dict[str, Any]:
    """记录历史 Q400/T1200 baseline，仅供 provenance 对照而非自动阈值判定。"""

    comparison: dict[str, Any] = {}
    for metric_name, historical_value in HISTORICAL_Q400_T1200_BASELINE.items():
        current_value = float(coarse_metrics[metric_name])
        comparison[metric_name] = {
            "historical_value": historical_value,
            "current_recomputed_value": current_value,
            "current_minus_historical": float(current_value - historical_value),
            "current_divided_by_historical": float(current_value / historical_value),
        }
    return {
        "historical_baseline": HISTORICAL_Q400_T1200_BASELINE,
        "recomputed_comparison": comparison,
        "threshold_defined_by_evaluator": False,
        "scientific_interpretation": "manual review required before drawing conclusions",
    }


def evaluate_plan_b(
    *,
    checkpoint_path: str | Path,
    coarse_dataset_dir: str | Path,
    fine_dataset_dir: str | Path,
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """依次验证 paired truth、冻结 checkpoint，再对 coarse/fine 各运行一次 forward。"""

    qualification, coarse, fine = validate_plan_b_inputs(coarse_dataset_dir, fine_dataset_dir)
    checkpoint = load_checkpoint_2d(Path(checkpoint_path), device=device)
    contract = validate_checkpoint_contract(checkpoint)
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=device)

    coarse_start = perf_counter()
    coarse_prediction = run_frozen_forward(model=model, contract=contract, field=coarse, device=device)
    coarse_seconds = perf_counter() - coarse_start
    fine_start = perf_counter()
    fine_prediction = run_frozen_forward(model=model, contract=contract, field=fine, device=device)
    fine_seconds = perf_counter() - fine_start

    coarse_metrics = compute_raw_metrics(coarse_prediction, coarse.canonical_truth, coarse.canonical_q)
    fine_full_metrics = compute_raw_metrics(fine_prediction, fine.canonical_truth, fine.canonical_q)
    fine_common_metrics = compute_raw_metrics(
        fine_prediction[:, ::2, :],
        fine.canonical_truth[:, ::2, :],
        fine.canonical_q,
    )
    result = {
        "schema_version": "1.0",
        "experiment_type": "plan_b_fixed_domain_lambda_discretization_resolution_generalization",
        "status": "completed",
        "checkpoint": {
            "path": _relative_path(Path(checkpoint_path)),
            "epoch": checkpoint.get("epoch"),
            "model_config": contract.model_config,
            "provenance": contract.provenance,
            "target_transform": contract.target_transform_config.to_dict(),
            "normalization": contract.normalization_stats.to_dict(),
        },
        "datasets": {
            "coarse_path": _relative_path(Path(coarse_dataset_dir)),
            "fine_path": _relative_path(Path(fine_dataset_dir)),
            "coarse_grid": _grid_summary(coarse.lambda_grid),
            "fine_grid": _grid_summary(fine.lambda_grid),
            "fixed_independent_q_field": "Q400 canonical ascending order",
        },
        "structural_checks": qualification["structural_checks"],
        "ground_truth_qualification": {
            "structural_valid": qualification["structural_valid"],
            "numerical_metrics": qualification["numerical_metrics"],
        },
        "frozen_inference": {
            "coarse_forward_passes": 1,
            "fine_forward_passes": 1,
            "truth_used_as_model_input": False,
            "autoregressive_rollout": False,
            "prediction_feedback": False,
            "teacher_forcing": False,
            "retraining": False,
            "fine_tuning": False,
            "adaptation": "none",
            "coarse_inference_seconds": float(coarse_seconds),
            "fine_inference_seconds": float(fine_seconds),
        },
        "metrics": {
            "coarse_t1200": coarse_metrics,
            "fine_t2399_full_grid_primary": fine_full_metrics,
            "fine_t2399_common_nodes_diagnostic": fine_common_metrics,
        },
        "resolution_generalization_gap": {
            "fine_full_vs_coarse": compute_generalization_gap(fine_full_metrics, coarse_metrics),
            "fine_common_vs_coarse": compute_generalization_gap(fine_common_metrics, coarse_metrics),
        },
        "coarse_historical_baseline_cross_check": _historical_baseline_cross_check(coarse_metrics),
    }
    return _json_value(result), coarse_prediction, fine_prediction


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    result: dict[str, Any],
    coarse_prediction: np.ndarray,
    fine_prediction: np.ndarray,
) -> None:
    """以新目录写入 compact JSON 与 raw-xyz prediction 供正式 server 复核。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "metrics.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(result["metrics"]), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (directory / "experiment_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(result), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    np.save(directory / "coarse_prediction_raw_xyz.npy", np.asarray(coarse_prediction, dtype=np.float32))
    np.save(directory / "fine_prediction_raw_xyz.npy", np.asarray(fine_prediction, dtype=np.float32))


def main() -> None:
    """执行正式 Plan B frozen-resolution evaluator。"""

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    result, coarse_prediction, fine_prediction = evaluate_plan_b(
        checkpoint_path=args.checkpoint,
        coarse_dataset_dir=args.coarse_dataset_dir,
        fine_dataset_dir=args.fine_dataset_dir,
        device=str(args.device),
    )
    write_evaluation_outputs(
        output_dir=args.output_dir,
        result=result,
        coarse_prediction=coarse_prediction,
        fine_prediction=fine_prediction,
    )
    print("Plan B frozen FNO2D evaluation completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
