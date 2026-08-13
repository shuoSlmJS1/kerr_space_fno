from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.timesnet1d import (  # noqa: E402
    TimesNetReconstruction1D,
    build_timesnet1d_model,
    count_trainable_parameters,
)
from src.training.trajectory_reconstruction.fno1d_reconstruction import (  # noqa: E402
    ReconstructionNormalizationStats,
    SparseReconstructionDataset,
    load_reconstruction_splits,
)


EXPECTED_CHANNELS = [
    "sparse_x",
    "sparse_y",
    "sparse_z",
    "observed_mask",
    "lambda_coordinate",
]


def build_parser() -> argparse.ArgumentParser:
    """构造 TimesNet 频率诊断命令行。"""
    parser = argparse.ArgumentParser(
        description="Inspect read-only TimesNet frequency and period diagnostics."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def _require_mapping(mapping: object, name: str) -> dict[str, Any]:
    """验证 JSON 对象字段为字典。"""
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return mapping


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """读取已完成 run 的 JSON 配置，不创建或修改任何文件。"""
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as input_file:
        return _require_mapping(json.load(input_file), "run_config")


def _normalization_from_config(config: dict[str, Any]) -> ReconstructionNormalizationStats:
    """从 run 配置恢复训练阶段保存的归一化统计量。"""
    raw = _require_mapping(config.get("normalization"), "normalization")
    required = (
        "input_xyz_mean",
        "input_xyz_std",
        "target_xyz_mean",
        "target_xyz_std",
        "lambda_min",
        "lambda_max",
        "eps",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise KeyError(f"run_config normalization is missing fields: {missing}")
    return ReconstructionNormalizationStats(
        input_xyz_mean=[float(value) for value in raw["input_xyz_mean"]],
        input_xyz_std=[float(value) for value in raw["input_xyz_std"]],
        target_xyz_mean=[float(value) for value in raw["target_xyz_mean"]],
        target_xyz_std=[float(value) for value in raw["target_xyz_std"]],
        lambda_min=float(raw["lambda_min"]),
        lambda_max=float(raw["lambda_max"]),
        eps=float(raw["eps"]),
    )


def _validate_run_config(config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """验证诊断能无猜测地复现正式模型与输入合同。"""
    if config.get("experiment_type") != "sparse_trajectory_reconstruction_timesnet1d":
        raise ValueError("run_config is not a TimesNet sparse reconstruction run.")
    if config.get("q_input") != "excluded":
        raise ValueError("run_config must record q_input as excluded.")
    if config.get("input_channel_names") != EXPECTED_CHANNELS:
        raise ValueError("run_config input channels do not match the five-channel contract.")
    model = _require_mapping(config.get("model"), "model")
    required_model = (
        "family",
        "in_dim",
        "out_dim",
        "d_model",
        "d_ff",
        "num_times_blocks",
        "top_k",
        "inception_kernel_sizes",
        "layer_norm",
        "fft_axis",
        "zero_frequency",
        "frequency_selection",
        "aggregation_weighting",
        "period_conversion",
        "latent_padding",
        "trainable_parameter_count",
    )
    missing = [name for name in required_model if name not in model]
    if missing:
        raise KeyError(f"run_config model is missing fields: {missing}")
    expected = {
        "family": "timesnet1d",
        "in_dim": 5,
        "out_dim": 3,
        "inception_kernel_sizes": [1, 3, 5],
        "layer_norm": "feature_dimension_only",
        "fft_axis": "time_dimension_dim_1",
        "zero_frequency": "excluded",
        "frequency_selection": "batch_shared_top_k",
        "aggregation_weighting": "per_sample_spectral_amplitude_softmax",
        "period_conversion": "T // frequency_index",
        "latent_padding": "right_zero_padding_then_crop",
    }
    for name, value in expected.items():
        if model[name] != value:
            raise ValueError(f"run_config model field {name} does not match canonical TimesNet.")
    training = _require_mapping(config.get("training"), "training")
    if "batch_size" not in training:
        raise KeyError("run_config training is missing batch_size.")
    recorded_batch_size = int(training["batch_size"])
    if recorded_batch_size <= 0:
        raise ValueError("run_config training batch_size must be positive.")
    _normalization_from_config(config)
    return model, recorded_batch_size


def load_model_from_run(
    run_dir: Path,
    device: str | torch.device,
) -> tuple[TimesNetReconstruction1D, dict[str, Any]]:
    """从 run 配置和最佳 checkpoint 重建处于 eval 模式的模型。"""
    resolved_run_dir = run_dir.resolve()
    config = _load_run_config(resolved_run_dir)
    model_config, _ = _validate_run_config(config)
    model = build_timesnet1d_model(
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        d_model=int(model_config["d_model"]),
        d_ff=int(model_config["d_ff"]),
        num_blocks=int(model_config["num_times_blocks"]),
        top_k=int(model_config["top_k"]),
    ).to(device)
    actual_parameter_count = count_trainable_parameters(model)
    if actual_parameter_count != int(model_config["trainable_parameter_count"]):
        raise ValueError(
            "Reconstructed model parameter count does not match run_config: "
            f"{actual_parameter_count} != {model_config['trainable_parameter_count']}."
        )
    checkpoint_path = resolved_run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("Best checkpoint is missing model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def build_diagnostic_loader(
    dataset_path: Path,
    config: dict[str, Any],
    split: str,
    batch_size_override: int | None,
) -> tuple[DataLoader, int, int]:
    """用保存的训练统计量重建无 shuffle 的验证或测试输入分组。"""
    model_config, recorded_batch_size = _validate_run_config(config)
    if batch_size_override is None:
        batch_size = recorded_batch_size
    else:
        batch_size = int(batch_size_override)
        if batch_size != recorded_batch_size:
            raise ValueError(
                "batch_size must match the run_config batch_size because "
                "frequency selection is batch-shared."
            )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    sampling = _require_mapping(config.get("sampling"), "sampling")
    if "stride" not in sampling:
        raise KeyError("run_config sampling is missing stride.")
    expected_dataset_path = config.get("dataset_path")
    if not isinstance(expected_dataset_path, str) or not expected_dataset_path:
        raise KeyError("run_config is missing dataset_path.")
    resolved_dataset_path = Path(dataset_path).resolve()
    if resolved_dataset_path != Path(expected_dataset_path).resolve():
        raise ValueError("dataset_path must match the dataset_path recorded in run_config.")
    splits = load_reconstruction_splits(
        resolved_dataset_path,
        stride=int(sampling["stride"]),
    )
    split_data = {"validation": splits.val, "test": splits.test}[split]
    if split_data.sampling.to_dict() != sampling:
        raise ValueError("Dataset sampling metadata does not match run_config.")
    expected_shape = config.get("input_shape_per_sample")
    if expected_shape != [int(split_data.target_xyz.shape[1]), 5]:
        raise ValueError("Dataset sequence length does not match run_config input shape.")
    if int(model_config["in_dim"]) != 5:
        raise ValueError("run_config model input dimension must be 5.")
    normalization = _normalization_from_config(config)
    dataset = SparseReconstructionDataset(split_data, normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if not isinstance(loader.sampler, SequentialSampler):
        raise RuntimeError("Diagnostic loader must use a sequential sampler.")
    return loader, int(split_data.sampling.stride), int(split_data.target_xyz.shape[1])


def _summarize_values(
    values_by_batch: list[list[int]],
    top_k: int,
) -> dict[str, Any]:
    """同时汇总选择槽位计数与包含该值的 batch 计数。"""
    slot_counts: Counter[int] = Counter()
    batch_counts: Counter[int] = Counter()
    for values in values_by_batch:
        slot_counts.update(int(value) for value in values)
        batch_counts.update(set(int(value) for value in values))
    total_batches = len(values_by_batch)
    total_slots = total_batches * top_k
    value_keys = sorted(set(slot_counts) | set(batch_counts))
    entries = {
        str(value): {
            "selection_slot_count": int(slot_counts[value]),
            "selection_slot_fraction": (
                float(slot_counts[value] / total_slots) if total_slots else 0.0
            ),
            "batch_presence_count": int(batch_counts[value]),
            "batch_presence_fraction": (
                float(batch_counts[value] / total_batches) if total_batches else 0.0
            ),
        }
        for value in value_keys
    }
    return {
        "total_batches": int(total_batches),
        "total_selection_slots": int(total_slots),
        "values": entries,
    }


@torch.no_grad()
def collect_model_diagnostics(
    model: TimesNetReconstruction1D,
    loader: DataLoader,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    """从 latent TimesBlock 诊断接口收集频率和 period，不执行 forward 或 backward。"""
    model.eval()
    block_batches: list[list[dict[str, Any]]] = [[] for _ in model.blocks]
    for batch_index, batch in enumerate(loader):
        model_input = batch[0].to(device)
        diagnostics = model.inspect_frequency_diagnostics(model_input)
        if len(diagnostics) != len(block_batches):
            raise RuntimeError("Model diagnostic block count is inconsistent.")
        for block_index, diagnostic in enumerate(diagnostics):
            frequencies = [int(value) for value in diagnostic["selected_frequency_indices"]]
            periods = [int(value) for value in diagnostic["selected_periods"]]
            if len(frequencies) != model.top_k or len(periods) != model.top_k:
                raise RuntimeError("Model diagnostic top_k result is inconsistent.")
            block_batches[block_index].append(
                {
                    "batch_index": int(batch_index),
                    "batch_size": int(model_input.shape[0]),
                    "selected_frequency_indices": frequencies,
                    "selected_periods": periods,
                }
            )
    if not block_batches or not block_batches[0]:
        raise ValueError("Diagnostic loader must contain at least one batch.")
    return [
        {
            "block_index": int(block_index),
            "batches": batches,
            "frequency_summary": _summarize_values(
                [batch["selected_frequency_indices"] for batch in batches],
                model.top_k,
            ),
            "period_summary": _summarize_values(
                [batch["selected_periods"] for batch in batches],
                model.top_k,
            ),
        }
        for block_index, batches in enumerate(block_batches)
    ]


def _lookup_relation(
    summary: dict[str, Any],
    values: list[int],
) -> dict[str, dict[str, float | int]]:
    """从汇总表中提取一组导出关系值，未选中时保持零计数。"""
    total_batches = int(summary["total_batches"])
    total_slots = int(summary["total_selection_slots"])
    entries = _require_mapping(summary["values"], "summary values")
    result: dict[str, dict[str, float | int]] = {}
    for value in values:
        entry = entries.get(str(value), {})
        slot_count = int(entry.get("selection_slot_count", 0))
        batch_count = int(entry.get("batch_presence_count", 0))
        result[str(value)] = {
            "selection_slot_count": slot_count,
            "selection_slot_fraction": float(slot_count / total_slots) if total_slots else 0.0,
            "batch_presence_count": batch_count,
            "batch_presence_fraction": float(batch_count / total_batches) if total_batches else 0.0,
        }
    return result


def build_sampling_stride_diagnostics(
    blocks: list[dict[str, Any]],
    stride: int,
    sequence_length: int,
) -> dict[str, Any]:
    """生成与采样 stride 的中性对应关系统计，不改变模型选择。"""
    reference_frequency = sequence_length // stride
    exact_period_frequencies = [
        frequency
        for frequency in range(1, sequence_length // 2 + 1)
        if sequence_length // frequency == stride
    ]
    related_periods = sorted(
        {
            value
            for value in (stride // 2, stride - 1, stride, stride + 1, 2 * stride)
            if value >= 1
        }
    )
    return {
        "sampling_stride": int(stride),
        "reference_frequency_index": int(reference_frequency),
        "reference_frequency_period": int(sequence_length // reference_frequency),
        "frequency_indices_mapping_to_exact_stride_period": exact_period_frequencies,
        "related_periods": {
            "half_stride": int(stride // 2),
            "near_stride": [int(stride - 1), int(stride), int(stride + 1)],
            "double_stride": int(2 * stride),
        },
        "blocks": [
            {
                "block_index": int(block["block_index"]),
                "reference_frequency": _lookup_relation(
                    _require_mapping(block["frequency_summary"], "frequency_summary"),
                    [reference_frequency],
                ),
                "frequencies_mapping_to_exact_stride_period": _lookup_relation(
                    _require_mapping(block["frequency_summary"], "frequency_summary"),
                    exact_period_frequencies,
                ),
                "sampling_stride_related_periods": _lookup_relation(
                    _require_mapping(block["period_summary"], "period_summary"),
                    related_periods,
                ),
            }
            for block in blocks
        ],
        "interpretation": (
            "These values describe latent TimesBlock selections only and do not "
            "establish a causal effect of sparse sampling."
        ),
    }


def _save_json_new(path: Path, data: dict[str, Any]) -> None:
    """以排他创建方式写入诊断 JSON，避免覆盖已有结果。"""
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """执行只读 TimesNet 频率与 period 诊断，并保存紧凑汇总 JSON。"""
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_path}")
    run_dir = Path(args.run_dir).resolve()
    model, config = load_model_from_run(run_dir, args.device)
    loader, stride, sequence_length = build_diagnostic_loader(
        dataset_path=Path(args.dataset_path),
        config=config,
        split=str(args.split),
        batch_size_override=args.batch_size,
    )
    blocks = collect_model_diagnostics(model, loader, args.device)
    model_config, recorded_batch_size = _validate_run_config(config)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "split": str(args.split),
        "stride": int(stride),
        "sequence_length": int(sequence_length),
        "batch_size": int(recorded_batch_size),
        "model": {
            "family": model_config["family"],
            "blocks": int(model_config["num_times_blocks"]),
            "top_k": int(model_config["top_k"]),
            "frequency_selection": model_config["frequency_selection"],
            "diagnostic_source": "latent_timesblock_hidden_representation",
        },
        "blocks": blocks,
        "sampling_stride_diagnostics": build_sampling_stride_diagnostics(
            blocks,
            stride=stride,
            sequence_length=sequence_length,
        ),
    }
    _save_json_new(output_path, result)
    print("TimesNet frequency diagnostics completed.")
    print(f"Output JSON: {output_path}")
    return result


def main() -> None:
    """命令行主入口。"""
    try:
        run_diagnostics(build_parser().parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
