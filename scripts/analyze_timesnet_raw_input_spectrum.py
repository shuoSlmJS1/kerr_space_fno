from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.analyze_timesnet_frequency_diagnostics import (  # noqa: E402
    EXPECTED_CHANNELS,
    build_diagnostic_loader,
    load_model_from_run,
)


DEFAULT_RAW_TOP_K = 10


def build_parser() -> argparse.ArgumentParser:
    """构造 TimesNet 原始输入频谱诊断命令行。"""
    parser = argparse.ArgumentParser(
        description="Inspect read-only raw TimesNet input spectra."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k-raw", type=int, default=DEFAULT_RAW_TOP_K)
    return parser


def _validate_model_input(model_input: torch.Tensor) -> None:
    """验证共享重建流程生成的固定五通道输入。"""
    if model_input.ndim != 3 or model_input.shape[-1] != len(EXPECTED_CHANNELS):
        raise ValueError("model_input must have shape [B,T,5].")
    if model_input.shape[0] == 0 or model_input.shape[1] < 2:
        raise ValueError("model_input must contain a non-empty sequence of length at least 2.")
    if not torch.isfinite(model_input).all():
        raise ValueError("model_input must contain only finite values.")


@torch.no_grad()
def aggregate_raw_input_spectrum(
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[torch.Tensor, int, int]:
    """按轨道沿时间维累积五个原始输入通道的 FFT 幅值。"""
    amplitude_sum: torch.Tensor | None = None
    trajectory_count = 0
    sequence_length: int | None = None
    for batch in loader:
        model_input = batch[0].to(device)
        _validate_model_input(model_input)
        current_length = int(model_input.shape[1])
        if sequence_length is None:
            sequence_length = current_length
        elif sequence_length != current_length:
            raise ValueError("All model_input batches must have the same sequence length.")
        amplitude = torch.fft.rfft(model_input, dim=1).abs().to(torch.float64)
        batch_sum = amplitude.sum(dim=0).detach().cpu()
        amplitude_sum = batch_sum if amplitude_sum is None else amplitude_sum + batch_sum
        trajectory_count += int(model_input.shape[0])
    if amplitude_sum is None or sequence_length is None or trajectory_count == 0:
        raise ValueError("Spectrum loader must contain at least one trajectory.")
    return amplitude_sum / trajectory_count, trajectory_count, sequence_length


@torch.no_grad()
def aggregate_input_projection_spectrum(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """只读取训练后输入投影的输出并汇总其时间频谱。"""
    model.eval()
    amplitude_sum: torch.Tensor | None = None
    trajectory_count = 0
    sequence_length: int | None = None
    feature_count: int | None = None
    for batch in loader:
        model_input = batch[0].to(device)
        _validate_model_input(model_input)
        projected = model.input_projection(model_input)
        current_length = int(projected.shape[1])
        if sequence_length is None:
            sequence_length = current_length
            feature_count = int(projected.shape[-1])
        elif sequence_length != current_length or feature_count != int(projected.shape[-1]):
            raise ValueError("All projected batches must have the same shape.")
        amplitude = torch.fft.rfft(projected, dim=1).abs().to(torch.float64)
        batch_sum = amplitude.sum(dim=(0, 2)).detach().cpu()
        amplitude_sum = batch_sum if amplitude_sum is None else amplitude_sum + batch_sum
        trajectory_count += int(projected.shape[0])
    if (
        amplitude_sum is None
        or sequence_length is None
        or feature_count is None
        or trajectory_count == 0
    ):
        raise ValueError("Projection spectrum loader must contain at least one trajectory.")
    return amplitude_sum / (trajectory_count * feature_count), trajectory_count, sequence_length, feature_count


def _ranked_frequency_records(
    mean_amplitude: torch.Tensor,
    sequence_length: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[int, int], float]:
    """排除直流分量后生成频率排序、整数周期和通道内相对幅值。"""
    if mean_amplitude.ndim != 1:
        raise ValueError("mean_amplitude must have shape [F].")
    selectable_count = int(mean_amplitude.shape[0] - 1)
    if selectable_count < 1:
        raise ValueError("At least one nonzero frequency bin is required.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k_raw must be a positive integer.")
    if top_k > selectable_count:
        raise ValueError("top_k_raw exceeds the number of nonzero rFFT frequency bins.")
    nonzero = mean_amplitude[1:]
    if not torch.isfinite(nonzero).all():
        raise ValueError("mean_amplitude must contain only finite values.")
    nonzero_total = float(nonzero.sum().item())
    ranked_offsets = torch.argsort(nonzero, descending=True)
    rank_by_frequency = {
        int(offset.item()) + 1: rank + 1
        for rank, offset in enumerate(ranked_offsets)
    }

    def record(frequency: int) -> dict[str, Any]:
        amplitude = float(mean_amplitude[frequency].item())
        return {
            "rank": int(rank_by_frequency[frequency]),
            "frequency_index": int(frequency),
            "integer_period": int(sequence_length // frequency),
            "mean_amplitude": amplitude,
            "within_channel_nonzero_spectral_fraction": (
                float(amplitude / nonzero_total) if nonzero_total > 0.0 else 0.0
            ),
        }

    top_frequencies = [int(offset.item()) + 1 for offset in ranked_offsets[:top_k]]
    return [record(frequency) for frequency in top_frequencies], rank_by_frequency, nonzero_total


def _records_for_frequencies(
    mean_amplitude: torch.Tensor,
    sequence_length: int,
    rank_by_frequency: dict[int, int],
    nonzero_total: float,
    frequencies: list[int],
) -> list[dict[str, Any]]:
    """为可选频率探针生成与 top 频率一致的记录格式。"""
    records: list[dict[str, Any]] = []
    for frequency in frequencies:
        if 1 <= frequency < mean_amplitude.shape[0]:
            amplitude = float(mean_amplitude[frequency].item())
            records.append(
                {
                    "rank": int(rank_by_frequency[frequency]),
                    "frequency_index": int(frequency),
                    "integer_period": int(sequence_length // frequency),
                    "mean_amplitude": amplitude,
                    "within_channel_nonzero_spectral_fraction": (
                        float(amplitude / nonzero_total) if nonzero_total > 0.0 else 0.0
                    ),
                }
            )
    return records


def _frequencies_for_period(sequence_length: int, period: int) -> list[int]:
    """返回所有满足现有 T // f period 规则的有效 rFFT 频率。"""
    if period < 1:
        return []
    return [
        frequency
        for frequency in range(1, sequence_length // 2 + 1)
        if sequence_length // frequency == period
    ]


def build_channel_spectrum_summary(
    mean_amplitude: torch.Tensor,
    sequence_length: int,
    stride: int,
    top_k_raw: int,
) -> dict[str, Any]:
    """为单个输入通道构建 top 频率与采样关系探针汇总。"""
    top_records, rank_by_frequency, nonzero_total = _ranked_frequency_records(
        mean_amplitude,
        sequence_length,
        top_k_raw,
    )
    ultra_low = _records_for_frequencies(
        mean_amplitude,
        sequence_length,
        rank_by_frequency,
        nonzero_total,
        [1, 2, 3, 4, 5],
    )
    period_probes = {
        "half_stride": stride // 2,
        "exact_stride": stride,
        "double_stride": 2 * stride,
    }
    stride_probes = {
        name: {
            "target_period": int(period),
            "frequencies_mapping_to_period": _records_for_frequencies(
                mean_amplitude,
                sequence_length,
                rank_by_frequency,
                nonzero_total,
                _frequencies_for_period(sequence_length, period),
            ),
        }
        for name, period in period_probes.items()
    }
    return {
        "top_nonzero_frequencies": top_records,
        "probe_frequencies": {
            "ultra_low": ultra_low,
            "sampling_stride_related": stride_probes,
        },
        "nonzero_frequency_amplitude_sum": float(nonzero_total),
    }


def build_projection_spectrum_summary(
    mean_amplitude: torch.Tensor,
    sequence_length: int,
    top_k_raw: int,
    feature_count: int,
) -> dict[str, Any]:
    """为输入投影后的特征平均频谱生成独立诊断汇总。"""
    top_records, _, nonzero_total = _ranked_frequency_records(
        mean_amplitude,
        sequence_length,
        top_k_raw,
    )
    return {
        "hidden_feature_count": int(feature_count),
        "top_nonzero_frequencies": top_records,
        "nonzero_frequency_amplitude_sum": float(nonzero_total),
        "diagnostic_source": "trained_input_projection_output_before_first_timesblock",
    }


def _save_json_new(path: Path, data: dict[str, Any]) -> None:
    """以排他创建方式写入 JSON，避免覆盖已有诊断结果。"""
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")


def run_raw_input_spectrum(args: argparse.Namespace) -> dict[str, Any]:
    """运行只读 raw-input 与 input-projection 频谱诊断。"""
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_path}")
    model, config = load_model_from_run(Path(args.run_dir), args.device)
    loader, stride, sequence_length = build_diagnostic_loader(
        dataset_path=Path(args.dataset_path),
        config=config,
        split=str(args.split),
        batch_size_override=args.batch_size,
    )
    raw_amplitude, trajectory_count, raw_length = aggregate_raw_input_spectrum(
        loader,
        args.device,
    )
    projection_amplitude, projection_count, projection_length, feature_count = (
        aggregate_input_projection_spectrum(model, loader, args.device)
    )
    if raw_length != sequence_length or projection_length != sequence_length:
        raise RuntimeError("Spectrum sequence lengths do not match reconstructed input.")
    if projection_count != trajectory_count:
        raise RuntimeError("Projection trajectory count does not match raw input count.")
    channels = {
        name: build_channel_spectrum_summary(
            raw_amplitude[:, channel_index],
            sequence_length=sequence_length,
            stride=stride,
            top_k_raw=int(args.top_k_raw),
        )
        for channel_index, name in enumerate(EXPECTED_CHANNELS)
    }
    result: dict[str, Any] = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "split": str(args.split),
        "stride": int(stride),
        "sequence_length": int(sequence_length),
        "num_trajectories": int(trajectory_count),
        "top_k_raw": int(args.top_k_raw),
        "fft_axis": "time_dimension_dim_1",
        "channels": channels,
        "input_projection_spectrum": build_projection_spectrum_summary(
            projection_amplitude,
            sequence_length=sequence_length,
            top_k_raw=int(args.top_k_raw),
            feature_count=feature_count,
        ),
        "interpretation_guardrail": (
            "These raw-input and input-projection spectra do not establish that any "
            "channel caused latent TimesBlock selections or reconstruction behavior."
        ),
    }
    _save_json_new(output_path, result)
    print("TimesNet raw input spectrum diagnostics completed.")
    print(f"Output JSON: {output_path}")
    return result


def main() -> None:
    """命令行主入口。"""
    try:
        run_raw_input_spectrum(build_parser().parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
