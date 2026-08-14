from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.analyze_timesnet_frequency_diagnostics import (  # noqa: E402
    EXPECTED_CHANNELS,
    build_diagnostic_loader,
    load_model_from_run,
)


DEFAULT_TOP_K = 10
PRIMARY_FREQUENCIES = (1, 2, 3, 4, 5)


def build_parser() -> argparse.ArgumentParser:
    """构造 TimesNet 输入投影频谱贡献诊断命令行。"""
    parser = argparse.ArgumentParser(
        description="Inspect read-only TimesNet input-projection spectral contributions."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--latent-diagnostics-json", type=Path, default=None)
    return parser


def _validate_model_input(model_input: torch.Tensor) -> None:
    """验证共享重建流程产生的固定五通道输入。"""
    if model_input.ndim != 3 or model_input.shape[-1] != len(EXPECTED_CHANNELS):
        raise ValueError("model_input must have shape [B,T,5].")
    if model_input.shape[0] == 0 or model_input.shape[1] < 2:
        raise ValueError("model_input must contain a non-empty sequence of length at least 2.")
    if not torch.isfinite(model_input).all():
        raise ValueError("model_input must contain only finite values.")


def _validate_projection(model: torch.nn.Module) -> torch.nn.Linear:
    """验证训练模型的输入投影仍是五通道线性层。"""
    projection = getattr(model, "input_projection", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError("model.input_projection must be torch.nn.Linear.")
    if projection.in_features != len(EXPECTED_CHANNELS):
        raise ValueError("model.input_projection must accept exactly five channels.")
    return projection


def _frequency_bins_for_period(sequence_length: int, period: int) -> list[int]:
    """返回满足现有整数周期转换规则的全部有效非零频率。"""
    if period < 1:
        return []
    return [
        frequency
        for frequency in range(1, sequence_length // 2 + 1)
        if sequence_length // frequency == period
    ]


def _selected_latent_frequencies(path: Path | None) -> set[int]:
    """从可选的既有 latent 诊断 JSON 提取实际选择的非零频率。"""
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Latent diagnostics JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        raise ValueError("Latent diagnostics JSON must contain a blocks list.")
    selected: set[int] = set()
    for block in data["blocks"]:
        if not isinstance(block, dict) or not isinstance(block.get("batches"), list):
            raise ValueError("Each latent diagnostics block must contain a batches list.")
        for batch in block["batches"]:
            if not isinstance(batch, dict) or not isinstance(
                batch.get("selected_frequency_indices"), list
            ):
                raise ValueError("Each latent diagnostics batch must list selected_frequency_indices.")
            for frequency in batch["selected_frequency_indices"]:
                if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
                    raise ValueError("Latent selected frequencies must be positive integers.")
                selected.add(frequency)
    return selected


@torch.no_grad()
def aggregate_projection_contributions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """按轨道聚合五个输入通道对线性投影非零频谱的精确贡献。"""
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative.")
    model.eval()
    projection = _validate_projection(model)
    weight = projection.weight
    if not torch.isfinite(weight).all() or (
        projection.bias is not None and not torch.isfinite(projection.bias).all()
    ):
        raise ValueError("input_projection parameters must contain only finite values.")

    component_magnitude_sum: torch.Tensor | None = None
    combined_magnitude_sum: torch.Tensor | None = None
    trajectory_count = 0
    sequence_length: int | None = None
    feature_count = projection.out_features
    maximum_reconstruction_error = 0.0
    maximum_manual_projection_error = 0.0
    maximum_manual_fft_error = 0.0

    for batch in loader:
        model_input = batch[0].to(device)
        _validate_model_input(model_input)
        current_length = int(model_input.shape[1])
        if sequence_length is None:
            sequence_length = current_length
        elif sequence_length != current_length:
            raise ValueError("All model_input batches must have the same sequence length.")

        projected = projection(model_input)
        manual_projected = functional.linear(model_input, weight, projection.bias)
        maximum_manual_projection_error = max(
            maximum_manual_projection_error,
            float((projected - manual_projected).abs().max().detach().cpu()),
        )
        input_fft = torch.fft.rfft(model_input, dim=1)
        projected_fft = torch.fft.rfft(projected, dim=1)
        manual_fft = torch.fft.rfft(manual_projected, dim=1)
        channel_fft = input_fft.unsqueeze(-1) * weight.transpose(0, 1).view(
            1, 1, len(EXPECTED_CHANNELS), feature_count
        )
        reconstructed_fft = channel_fft.sum(dim=2)
        nonzero_difference = reconstructed_fft[:, 1:, :] - projected_fft[:, 1:, :]
        maximum_reconstruction_error = max(
            maximum_reconstruction_error,
            float(nonzero_difference.abs().max().detach().cpu()),
        )
        maximum_manual_fft_error = max(
            maximum_manual_fft_error,
            float((manual_fft[:, 1:, :] - projected_fft[:, 1:, :]).abs().max().detach().cpu()),
        )
        if not torch.allclose(reconstructed_fft[:, 1:, :], projected_fft[:, 1:, :], atol=atol, rtol=rtol):
            raise RuntimeError("Channel contributions do not reconstruct the nonzero projected FFT.")

        component_magnitudes = torch.linalg.vector_norm(channel_fft, dim=-1).to(torch.float64)
        combined_magnitudes = torch.linalg.vector_norm(projected_fft, dim=-1).to(torch.float64)
        batch_component_sum = component_magnitudes.sum(dim=0).detach().cpu()
        batch_combined_sum = combined_magnitudes.sum(dim=0).detach().cpu()
        component_magnitude_sum = (
            batch_component_sum
            if component_magnitude_sum is None
            else component_magnitude_sum + batch_component_sum
        )
        combined_magnitude_sum = (
            batch_combined_sum
            if combined_magnitude_sum is None
            else combined_magnitude_sum + batch_combined_sum
        )
        trajectory_count += int(model_input.shape[0])

    if (
        component_magnitude_sum is None
        or combined_magnitude_sum is None
        or sequence_length is None
        or trajectory_count == 0
    ):
        raise ValueError("Contribution loader must contain at least one trajectory.")
    return {
        "component_mean_magnitudes": component_magnitude_sum / trajectory_count,
        "combined_mean_magnitudes": combined_magnitude_sum / trajectory_count,
        "num_trajectories": trajectory_count,
        "sequence_length": sequence_length,
        "feature_count": feature_count,
        "reconstruction_check": {
            "passed": True,
            "atol": float(atol),
            "rtol": float(rtol),
            "max_nonzero_complex_reconstruction_abs_error": maximum_reconstruction_error,
            "max_manual_time_domain_projection_abs_error": maximum_manual_projection_error,
            "max_manual_nonzero_fft_abs_error": maximum_manual_fft_error,
        },
    }


def _frequency_record(
    frequency: int,
    sequence_length: int,
    component_mean_magnitudes: torch.Tensor,
    combined_mean_magnitudes: torch.Tensor,
) -> dict[str, Any]:
    """构造单个频率的分量幅值和相位抵消敏感汇总。"""
    component_values = component_mean_magnitudes[frequency]
    component_sum = float(component_values.sum().item())
    combined = float(combined_mean_magnitudes[frequency].item())
    contributions = {
        channel: {
            "mean_projected_component_magnitude": float(component_values[index].item()),
            "relative_to_sum_component_magnitudes": (
                float(component_values[index].item() / component_sum)
                if component_sum > 0.0
                else 0.0
            ),
        }
        for index, channel in enumerate(EXPECTED_CHANNELS)
    }
    return {
        "integer_period": int(sequence_length // frequency),
        "combined_projection_mean_magnitude": combined,
        "sum_component_mean_magnitudes": component_sum,
        "channel_contributions": contributions,
    }


def _select_report_frequencies(
    combined_mean_magnitudes: torch.Tensor,
    sequence_length: int,
    stride: int,
    top_k: int,
    latent_frequencies: set[int],
) -> tuple[list[int], dict[str, list[int]]]:
    """合并主要、投影 top、采样相关和可选 latent 频率。"""
    available = int(combined_mean_magnitudes.shape[0] - 1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if top_k > available:
        raise ValueError("top_k exceeds the number of nonzero rFFT frequency bins.")
    top_offsets = torch.argsort(combined_mean_magnitudes[1:], descending=True)[:top_k]
    top_frequencies = {int(offset.item()) + 1 for offset in top_offsets}
    sampling_regions = {
        "half_stride": _frequency_bins_for_period(sequence_length, stride // 2),
        "exact_stride": _frequency_bins_for_period(sequence_length, stride),
        "double_stride": _frequency_bins_for_period(sequence_length, 2 * stride),
    }
    selected = {
        frequency
        for frequency in set(PRIMARY_FREQUENCIES) | top_frequencies | latent_frequencies
        if 1 <= frequency <= available
    }
    selected.update(frequency for values in sampling_regions.values() for frequency in values)
    return sorted(selected), sampling_regions


def _save_json_new(path: Path, data: dict[str, Any]) -> None:
    """以排他创建方式写入紧凑 JSON，避免覆盖既有诊断结果。"""
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")


def run_projection_spectral_contributions(args: argparse.Namespace) -> dict[str, Any]:
    """运行只读训练后输入投影频谱贡献分解。"""
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
    aggregate = aggregate_projection_contributions(model, loader, args.device)
    if int(aggregate["sequence_length"]) != sequence_length:
        raise RuntimeError("Contribution sequence length does not match reconstructed input.")
    latent_frequencies = _selected_latent_frequencies(args.latent_diagnostics_json)
    report_frequencies, sampling_regions = _select_report_frequencies(
        aggregate["combined_mean_magnitudes"],
        sequence_length,
        stride,
        int(args.top_k),
        latent_frequencies,
    )
    frequencies = {
        str(frequency): _frequency_record(
            frequency,
            sequence_length,
            aggregate["component_mean_magnitudes"],
            aggregate["combined_mean_magnitudes"],
        )
        for frequency in report_frequencies
    }
    result: dict[str, Any] = {
        "split": str(args.split),
        "stride": int(stride),
        "sequence_length": int(sequence_length),
        "num_trajectories": int(aggregate["num_trajectories"]),
        "channels": list(EXPECTED_CHANNELS),
        "checkpoint": "checkpoints/best_model.pt",
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "top_k_full_projection": int(args.top_k),
        "sampling_stride_related_frequency_bins": sampling_regions,
        "latent_selected_frequency_indices": sorted(latent_frequencies),
        "frequencies": frequencies,
        "reconstruction_check": aggregate["reconstruction_check"],
        "interpretation_guardrail": (
            "Component-magnitude fractions compare individual complex component sizes. "
            "They are not additive causal shares because |a+b| is generally not |a|+|b|. "
            "This diagnostic does not establish that any channel caused latent TimesBlock "
            "selections or reconstruction behavior."
        ),
    }
    _save_json_new(output_path, result)
    print("TimesNet projection spectral contribution diagnostics completed.")
    print(f"Output JSON: {output_path}")
    return result


def main() -> None:
    """命令行主入口。"""
    try:
        run_projection_spectral_contributions(build_parser().parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
