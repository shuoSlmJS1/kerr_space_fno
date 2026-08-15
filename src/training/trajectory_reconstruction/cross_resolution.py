from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.fno1d.fno1d import build_fno1d_model
from src.models.resnet1d import build_dilated_resnet1d
from src.models.timesnet1d import build_timesnet1d_model
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    ReconstructionNormalizationStats,
    SparseReconstructionDataset,
    evaluate_reconstruction_model,
)
from src.training.trajectory_reconstruction.sparse_sampling import (
    SparseTrajectoryData,
    build_sparse_trajectory_data,
)


EXPECTED_INPUT_CHANNELS = (
    "sparse_x",
    "sparse_y",
    "sparse_z",
    "observed_mask",
    "lambda_coordinate",
)
SUPPORTED_MODEL_FAMILIES = (
    "fno1d",
    "dilated_resnet1d",
    "timesnet1d",
)


@dataclass(frozen=True)
class FrozenReconstructionRun:
    """保存已加载且冻结的稀疏重建运行信息。"""

    run_dir: Path
    run_config: dict[str, Any]
    model_family: str
    model: torch.nn.Module
    normalization: ReconstructionNormalizationStats
    train_stride: int
    recorded_batch_size: int
    checkpoint_path: Path
    parameter_count: int


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    """验证 JSON 或 checkpoint 中的映射字段。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object.")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    """验证不接受 bool 的正整数配置。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    """读取并验证一个 UTF-8 JSON 对象文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    return _require_mapping(value, name)


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    """统计已加载模型的可训练参数数量。"""
    return sum(parameter.numel() for parameter in model.parameters())


def _validate_common_run_config(run_config: dict[str, Any]) -> None:
    """验证三个模型共同的冻结实验契约。"""
    if run_config.get("q_input") != "excluded":
        raise ValueError("run_config must record q_input as excluded.")
    if tuple(run_config.get("input_channel_names", ())) != EXPECTED_INPUT_CHANNELS:
        raise ValueError("run_config input_channel_names do not match the five-channel contract.")

    input_shape = run_config.get("input_shape_per_sample")
    output_shape = run_config.get("output_shape_per_sample")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 2
        or not isinstance(output_shape, list)
        or len(output_shape) != 2
    ):
        raise ValueError("run_config must contain two-dimensional input and output sample shapes.")
    if input_shape[1] != 5 or output_shape[1] != 3:
        raise ValueError("run_config must record input shape [T,5] and output shape [T,3].")
    _require_positive_integer(input_shape[0], "input_shape_per_sample[0]")
    _require_positive_integer(output_shape[0], "output_shape_per_sample[0]")
    if input_shape[0] != output_shape[0]:
        raise ValueError("run_config input and output sequence lengths must match.")

    sampling = _require_mapping(run_config.get("sampling"), "run_config sampling")
    _require_positive_integer(sampling.get("stride"), "run_config sampling stride")
    normalization = _require_mapping(
        run_config.get("normalization"),
        "run_config normalization",
    )
    required_normalization = {
        "input_xyz_mean",
        "input_xyz_std",
        "target_xyz_mean",
        "target_xyz_std",
        "lambda_min",
        "lambda_max",
        "eps",
    }
    missing_normalization = sorted(required_normalization.difference(normalization))
    if missing_normalization:
        raise ValueError(
            "run_config normalization is missing fields: "
            f"{missing_normalization}."
        )

    training = _require_mapping(run_config.get("training"), "run_config training")
    _require_positive_integer(training.get("batch_size"), "run_config training batch_size")


def _build_model_from_config(run_config: dict[str, Any]) -> tuple[str, torch.nn.Module]:
    """按保存的模型配置重建三类冻结架构。"""
    model_config = _require_mapping(run_config.get("model"), "run_config model")
    family = model_config.get("family")
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported reconstruction model family: {family!r}.")

    expected_experiment_type = {
        "fno1d": "sparse_trajectory_reconstruction_fno1d",
        "dilated_resnet1d": "sparse_trajectory_reconstruction_resnet1d",
        "timesnet1d": "sparse_trajectory_reconstruction_timesnet1d",
    }[family]
    if run_config.get("experiment_type") != expected_experiment_type:
        raise ValueError(
            "run_config experiment_type does not match model family: "
            f"expected {expected_experiment_type}."
        )

    if model_config.get("in_dim") != 5 or model_config.get("out_dim") != 3:
        raise ValueError("run_config model must use in_dim=5 and out_dim=3.")

    if family == "fno1d":
        model = build_fno1d_model(
            in_dim=5,
            out_dim=3,
            modes=_require_positive_integer(model_config.get("modes"), "model modes"),
            width=_require_positive_integer(model_config.get("width"), "model width"),
            depth=_require_positive_integer(model_config.get("depth"), "model depth"),
        )
    elif family == "dilated_resnet1d":
        model = build_dilated_resnet1d(
            in_dim=5,
            out_dim=3,
            width=_require_positive_integer(model_config.get("width"), "model width"),
            blocks=_require_positive_integer(
                model_config.get("block_count"),
                "model block_count",
            ),
        )
        expected_schedule = model_config.get("dilation_schedule")
        if expected_schedule != list(model.dilations):
            raise ValueError("run_config dilation_schedule does not match the approved ResNet architecture.")
    else:
        model = build_timesnet1d_model(
            in_dim=5,
            out_dim=3,
            d_model=_require_positive_integer(model_config.get("d_model"), "model d_model"),
            d_ff=_require_positive_integer(model_config.get("d_ff"), "model d_ff"),
            num_blocks=_require_positive_integer(
                model_config.get("num_times_blocks"),
                "model num_times_blocks",
            ),
            top_k=_require_positive_integer(model_config.get("top_k"), "model top_k"),
        )

    expected_parameter_count = model_config.get("trainable_parameter_count")
    actual_parameter_count = _count_trainable_parameters(model)
    if expected_parameter_count is not None and expected_parameter_count != actual_parameter_count:
        raise ValueError(
            "Saved trainable_parameter_count does not match the reconstructed model: "
            f"saved={expected_parameter_count}, actual={actual_parameter_count}."
        )
    return str(family), model


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    """确认 checkpoint 属于同一份已保存的运行配置。"""
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict.")
    checkpoint_config = _require_mapping(
        checkpoint.get("run_config"),
        "checkpoint run_config",
    )
    contract_keys = (
        "experiment_type",
        "dataset_path",
        "q_input",
        "input_channel_names",
        "input_shape_per_sample",
        "output_shape_per_sample",
        "sampling",
        "normalization",
        "training",
        "model",
    )
    mismatched_keys = [
        key
        for key in contract_keys
        if checkpoint_config.get(key) != run_config.get(key)
    ]
    if mismatched_keys:
        raise ValueError(
            "Checkpoint run_config does not match run_config.json for keys: "
            f"{mismatched_keys}."
        )


def load_frozen_reconstruction_run(
    run_dir: str | Path,
    device: str | torch.device = "cpu",
) -> FrozenReconstructionRun:
    """加载、校验并冻结一个已完成的稀疏重建最佳 checkpoint。"""
    resolved_run_dir = Path(run_dir).resolve()
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_run_dir}")

    run_config = _load_json_object(resolved_run_dir / "run_config.json", "run_config.json")
    _validate_common_run_config(run_config)
    model_family, model = _build_model_from_config(run_config)

    checkpoint_path = resolved_run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {checkpoint_path}")
    checkpoint_value = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint = _require_mapping(checkpoint_value, "checkpoint")
    _validate_checkpoint_contract(checkpoint, run_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    normalization_config = _require_mapping(
        run_config["normalization"],
        "run_config normalization",
    )
    normalization = ReconstructionNormalizationStats(**normalization_config)
    sampling_config = _require_mapping(run_config["sampling"], "run_config sampling")
    training_config = _require_mapping(run_config["training"], "run_config training")
    return FrozenReconstructionRun(
        run_dir=resolved_run_dir,
        run_config=run_config,
        model_family=model_family,
        model=model,
        normalization=normalization,
        train_stride=_require_positive_integer(sampling_config["stride"], "train_stride"),
        recorded_batch_size=_require_positive_integer(
            training_config["batch_size"],
            "recorded_batch_size",
        ),
        checkpoint_path=checkpoint_path,
        parameter_count=_count_trainable_parameters(model),
    )


def load_evaluation_sparse_data(
    dataset_path: str | Path,
    split: str,
    evaluation_stride: int,
) -> tuple[Path, SparseTrajectoryData]:
    """只读取指定评估划分，并按新的 stride 重建稀疏观测模式。"""
    if split != "test":
        raise ValueError("Only the test split is supported for this cross-resolution evaluator.")
    resolved_dataset_path = Path(dataset_path).resolve()
    if not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved_dataset_path}")
    with np.load(resolved_dataset_path, allow_pickle=False) as dataset:
        required_keys = ("y_test", "lambda_grid")
        missing_keys = [key for key in required_keys if key not in dataset]
        if missing_keys:
            raise KeyError(f"Dataset is missing required arrays: {missing_keys}.")
        target_xyz = np.asarray(dataset["y_test"])
        lambda_grid = np.asarray(dataset["lambda_grid"])
    return resolved_dataset_path, build_sparse_trajectory_data(
        target_xyz=target_xyz,
        lambda_grid=lambda_grid,
        stride=evaluation_stride,
    )


def _load_same_resolution_reference(run_dir: Path) -> dict[str, object]:
    """从原始 test 指标文件读取同分辨率参考值，不可用时明确记录。"""
    reference_path = run_dir / "metrics" / "test_hidden_only_metrics.json"
    relative_path = str(reference_path.relative_to(run_dir))
    if not reference_path.is_file():
        return {
            "available": False,
            "source": relative_path,
            "reason": "same_resolution_metric_file_unavailable",
        }
    try:
        metrics_file = _load_json_object(reference_path, "same-resolution metrics")
        raw_metrics = _require_mapping(
            metrics_file.get("raw_hidden_only_metrics"),
            "same-resolution raw_hidden_only_metrics",
        )
        overall_metrics = _require_mapping(
            raw_metrics.get("overall"),
            "same-resolution overall metrics",
        )
        reference_value = float(overall_metrics["relative_l2"])
    except (KeyError, TypeError, ValueError):
        return {
            "available": False,
            "source": relative_path,
            "reason": "same_resolution_relative_l2_unavailable",
        }
    if not np.isfinite(reference_value) or reference_value < 0.0:
        return {
            "available": False,
            "source": relative_path,
            "reason": "same_resolution_relative_l2_unavailable",
        }
    return {
        "available": True,
        "source": relative_path,
        "hidden_relative_l2": reference_value,
    }


def evaluate_frozen_cross_resolution_run(
    frozen_run: FrozenReconstructionRun,
    dataset_path: str | Path,
    evaluation_stride: int,
    split: str = "test",
    batch_size: int | None = None,
) -> dict[str, object]:
    """在不适配参数或归一化的条件下评估新的稀疏观测 stride。"""
    if batch_size is None:
        effective_batch_size = frozen_run.recorded_batch_size
    else:
        effective_batch_size = _require_positive_integer(batch_size, "batch_size")
    if (
        frozen_run.model_family == "timesnet1d"
        and effective_batch_size != frozen_run.recorded_batch_size
    ):
        raise ValueError(
            "TimesNet evaluation batch_size must match the recorded training run batch_size."
        )

    resolved_dataset_path, evaluation_data = load_evaluation_sparse_data(
        dataset_path=dataset_path,
        split=split,
        evaluation_stride=evaluation_stride,
    )
    recorded_dataset_path = Path(str(frozen_run.run_config["dataset_path"])).resolve()
    if resolved_dataset_path != recorded_dataset_path:
        raise ValueError(
            "dataset_path must exactly match the dataset_path recorded by the frozen run."
        )

    expected_sequence_length = int(frozen_run.run_config["input_shape_per_sample"][0])
    if evaluation_data.target_xyz.shape[1] != expected_sequence_length:
        raise ValueError(
            "Dataset sequence length does not match the frozen model run: "
            f"dataset={evaluation_data.target_xyz.shape[1]}, run={expected_sequence_length}."
        )
    split_sizes = frozen_run.run_config.get("split_sizes")
    if isinstance(split_sizes, dict) and "test" in split_sizes:
        expected_test_size = _require_positive_integer(split_sizes["test"], "run_config split_sizes test")
        if evaluation_data.target_xyz.shape[0] != expected_test_size:
            raise ValueError(
                "Dataset test trajectory count does not match the frozen model run: "
                f"dataset={evaluation_data.target_xyz.shape[0]}, run={expected_test_size}."
            )

    evaluation_dataset = SparseReconstructionDataset(
        sparse_data=evaluation_data,
        normalization=frozen_run.normalization,
    )
    loader = DataLoader(
        evaluation_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
    )
    normalized_hidden_mse, raw_metrics = evaluate_reconstruction_model(
        model=frozen_run.model,
        loader=loader,
        device=next(frozen_run.model.parameters()).device,
        normalization=frozen_run.normalization,
    )
    if any(parameter.grad is not None for parameter in frozen_run.model.parameters()):
        raise RuntimeError("Frozen evaluation produced parameter gradients.")

    reference = _load_same_resolution_reference(frozen_run.run_dir)
    cross_relative_l2 = float(raw_metrics["overall"]["relative_l2"])
    if reference["available"] and float(reference["hidden_relative_l2"]) > 0.0:
        degradation_factor: float | None = (
            cross_relative_l2 / float(reference["hidden_relative_l2"])
        )
    else:
        degradation_factor = None
        if reference["available"]:
            reference = {
                "available": False,
                "source": reference["source"],
                "reason": "same_resolution_relative_l2_is_zero",
            }

    return {
        "experiment_type": "cross_resolution_sparse_reconstruction",
        "model_family": frozen_run.model_family,
        "run_dir": str(frozen_run.run_dir),
        "dataset_path": str(resolved_dataset_path),
        "split": split,
        "train_stride": frozen_run.train_stride,
        "evaluation_stride": int(evaluation_stride),
        "sequence_length": int(evaluation_data.target_xyz.shape[1]),
        "num_trajectories": int(evaluation_data.target_xyz.shape[0]),
        "checkpoint": str(frozen_run.checkpoint_path.relative_to(frozen_run.run_dir)),
        "parameter_count": frozen_run.parameter_count,
        "normalization_source": "original_training_run",
        "adaptation": "none",
        "evaluation_batch_size": effective_batch_size,
        "sampling": evaluation_data.sampling.to_dict(),
        "same_resolution_reference": reference,
        "cross_resolution_metrics": {
            "model_space_hidden_mse": float(normalized_hidden_mse),
            "raw_hidden_only_metrics": raw_metrics,
        },
        "degradation_factor_relative_l2": degradation_factor,
    }


def save_cross_resolution_result(path: str | Path, result: dict[str, object]) -> Path:
    """以排他创建写入完整且可序列化的评估 JSON。"""
    output_path = Path(path).resolve()
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {output_path.parent}"
        )
    if output_path.exists():
        raise FileExistsError(
            f"Output JSON already exists and will not be overwritten: {output_path}"
        )
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
        output_file.write("\n")
    return output_path
