"""Analyze raw Kerr trajectory lambda-direction spectral energy without a model."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.io_utils import load_npz  # noqa: E402
from src.common.paths import get_task_dataset_npz_path  # noqa: E402


SPLITS = ("train", "val", "test")
COMPONENT_NAMES = ("x", "y", "z")
DEFAULT_RETAINED_MODES_LAMBDA = 32
DEFAULT_TOP_PEAKS = 10
EPSILON = 1e-12
OUTPUT_FILENAMES = (
    "raw_spectral_energy_summary.json",
    "spectral_band_energy.csv",
    "dominant_peaks.csv",
)


@dataclass(frozen=True)
class CanonicalQField:
    """保留来源身份，并提供升序 Q 的 raw float64 轨迹场。"""

    task_name: str
    source_q: np.ndarray
    source_truth: np.ndarray
    lambda_grid: np.ndarray
    canonical_q: np.ndarray
    canonical_truth: np.ndarray
    canonical_to_source_index: np.ndarray
    source_to_canonical_index: np.ndarray
    source_records: list[dict[str, Any]]


@dataclass(frozen=True)
class SpectralBand:
    """定义一个互不重叠的物理频率能量区间。"""

    label: str
    lower: float | None
    upper: float | None
    lower_inclusive: bool
    upper_inclusive: bool


@dataclass
class LengthSpectralAnalysis:
    """保存一个长度的紧凑汇总前所需的中间频谱信息。"""

    task_name: str
    total_length: int
    delta_lambda: float
    dft_period: float
    frequency_resolution: float
    frequencies: np.ndarray
    component_energy_spectrum: np.ndarray
    retained_modes: int
    highest_retained_index: int
    physical_cutoff: float
    component_summaries: dict[str, dict[str, float | int]]
    peak_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """解析 raw Kerr 频谱诊断命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw float64 Kerr trajectory lambda spectral energy across "
            "exact-prefix T1200/T1800/T2400 datasets without loading a model."
        )
    )
    parser.add_argument("--short-task-name", required=True)
    parser.add_argument("--medium-task-name", required=True)
    parser.add_argument("--long-task-name", required=True)
    parser.add_argument(
        "--dataset-pair-validation-json",
        required=True,
        type=Path,
        help="Stage-2 strict T1200/T1800/T2400 prefix-identity JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory. Existing directories are refused.",
    )
    parser.add_argument(
        "--retained-modes-lambda",
        type=int,
        default=DEFAULT_RETAINED_MODES_LAMBDA,
        help="Number of non-negative lambda rFFT indices retained by the spectral branch.",
    )
    parser.add_argument(
        "--detrend",
        choices=("none", "mean", "linear"),
        default="none",
        help="Optional raw-truth detrending sensitivity mode; primary analysis uses none.",
    )
    return parser.parse_args()


def _as_json_value(value: Any) -> Any:
    """递归转换 NumPy 标量和数组，避免保存大型中间频谱。"""

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
    """优先把路径记录为项目根目录相对路径。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _git_commit() -> str | None:
    """读取本地 HEAD，不执行远程 Git 操作。"""

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


def load_required_pair_validation(path: Path) -> dict[str, Any]:
    """加载并严格验证 Stage-2 三个长度配对前提。"""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset-pair validation JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    classifications = artifact.get("pair_classification")
    reuse = artifact.get("scientific_reuse")
    if not isinstance(classifications, dict) or not isinstance(reuse, dict):
        raise ValueError("Dataset-pair validation JSON has no required fields.")

    for key in ("short_to_medium", "short_to_long", "medium_to_long"):
        if classifications.get(key) != "EXACT_PREFIX":
            raise ValueError(f"Dataset-pair validation requires {key}=EXACT_PREFIX.")

    for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1"):
        if reuse.get(key) is not True:
            raise ValueError(f"Dataset-pair validation requires {key}=true.")
    return artifact


def _load_split(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一个来源 split，并拒绝降精度的 raw truth。"""

    x_key = f"x_{split}"
    y_key = f"y_{split}"
    if x_key not in data or y_key not in data:
        raise KeyError(f"Dataset is missing {x_key} or {y_key}.")
    x = np.asarray(data[x_key])
    y = np.asarray(data[y_key])
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"{x_key} must have shape [N,1], got {x.shape}.")
    if y.ndim != 3 or y.shape[0] != x.shape[0] or y.shape[2] != 3:
        raise ValueError(f"{y_key} must have shape [N,T,3], got {y.shape}.")
    if y.dtype != np.float64:
        raise ValueError(f"{y_key} must retain raw float64 truth, got {y.dtype}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{split} contains non-finite Q or raw truth values.")
    return x, y


def build_canonical_q_field(
    *,
    task_name: str,
    source_q: np.ndarray,
    source_truth: np.ndarray,
    lambda_grid: np.ndarray,
    source_records: list[dict[str, Any]],
) -> CanonicalQField:
    """稳定排序 Q，并把同一置换应用于 raw truth 与来源记录。"""

    q = np.asarray(source_q, dtype=np.float64).reshape(-1)
    truth = np.asarray(source_truth, dtype=np.float64)
    lambda_values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if truth.shape != (q.size, lambda_values.size, 3):
        raise ValueError("Source Q, raw truth, and lambda grid have incompatible shapes.")
    if len(source_records) != q.size:
        raise ValueError("Source identity records do not match Q count.")
    if lambda_values.size < 2 or not np.all(np.isfinite(lambda_values)):
        raise ValueError("Lambda grid must be finite and contain at least two points.")

    canonical_to_source = np.argsort(q, kind="stable").astype(np.int64)
    canonical_q = q[canonical_to_source]
    if canonical_q.size > 1 and not np.all(np.diff(canonical_q) > 0.0):
        raise ValueError("Canonical Q field requires unique strictly ascending Q values.")
    source_to_canonical = np.empty_like(canonical_to_source)
    source_to_canonical[canonical_to_source] = np.arange(q.size, dtype=np.int64)
    if not np.array_equal(canonical_to_source[source_to_canonical], np.arange(q.size)):
        raise RuntimeError("Canonical/source Q mappings are not inverse permutations.")

    canonical_records: list[dict[str, Any]] = []
    for canonical_index, source_index in enumerate(canonical_to_source):
        record = dict(source_records[int(source_index)])
        record["Q"] = float(canonical_q[canonical_index])
        record["canonical_model_index"] = int(canonical_index)
        canonical_records.append(record)

    return CanonicalQField(
        task_name=task_name,
        source_q=q,
        source_truth=truth,
        lambda_grid=lambda_values,
        canonical_q=canonical_q,
        canonical_truth=truth[canonical_to_source],
        canonical_to_source_index=canonical_to_source,
        source_to_canonical_index=source_to_canonical,
        source_records=canonical_records,
    )


def load_task_raw_field(task_name: str) -> CanonicalQField:
    """合并 train/val/test 来源并构造完整 canonical Q raw field。"""

    dataset_path = get_task_dataset_npz_path(task_name)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    data = load_npz(dataset_path)
    if "lambda_grid" not in data:
        raise KeyError("Dataset is missing lambda_grid.")

    q_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    source_offset = 0
    for split in SPLITS:
        q_values, truth = _load_split(data, split)
        q_parts.append(q_values)
        truth_parts.append(truth)
        for index in range(q_values.shape[0]):
            source_records.append(
                {
                    "source_split": split,
                    "source_index_within_split": int(index),
                    "source_concatenated_index": int(source_offset + index),
                }
            )
        source_offset += int(q_values.shape[0])

    return build_canonical_q_field(
        task_name=task_name,
        source_q=np.concatenate(q_parts, axis=0)[:, 0],
        source_truth=np.concatenate(truth_parts, axis=0),
        lambda_grid=np.asarray(data["lambda_grid"], dtype=np.float64),
        source_records=source_records,
    )


def validate_triplet(
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
) -> None:
    """在频谱分析前复核 canonical Q、网格和 raw truth 的 exact-prefix 身份。"""

    fields = (short, medium, long)
    for field in fields:
        if not np.array_equal(short.canonical_q, field.canonical_q):
            raise ValueError("Canonical Q arrays must match exactly across all three tasks.")

    if medium.lambda_grid.size < short.lambda_grid.size or long.lambda_grid.size < medium.lambda_grid.size:
        raise ValueError("Lambda lengths must be non-decreasing short/medium/long.")
    if not np.array_equal(short.lambda_grid, medium.lambda_grid[: short.lambda_grid.size]):
        raise ValueError("T1200 and T1800 lambda grids are not exact prefixes.")
    if not np.array_equal(short.lambda_grid, long.lambda_grid[: short.lambda_grid.size]):
        raise ValueError("T1200 and T2400 lambda grids are not exact prefixes.")
    if not np.array_equal(medium.lambda_grid, long.lambda_grid[: medium.lambda_grid.size]):
        raise ValueError("T1800 and T2400 lambda grids are not exact prefixes.")
    if not np.array_equal(short.canonical_truth, medium.canonical_truth[:, : short.lambda_grid.size, :]):
        raise ValueError("T1200 and T1800 raw truths are not exact prefixes.")
    if not np.array_equal(short.canonical_truth, long.canonical_truth[:, : short.lambda_grid.size, :]):
        raise ValueError("T1200 and T2400 raw truths are not exact prefixes.")
    if not np.array_equal(medium.canonical_truth, long.canonical_truth[:, : medium.lambda_grid.size, :]):
        raise ValueError("T1800 and T2400 raw truths are not exact prefixes.")


def infer_uniform_delta_lambda(lambda_grid: np.ndarray) -> float:
    """验证均匀 lambda 网格并返回 DFT 使用的相邻采样间距。"""

    values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Lambda grid must be finite and contain at least two points.")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("Lambda grid must be strictly increasing.")
    delta = float(differences[0])
    if not np.allclose(differences, delta, rtol=1e-12, atol=1e-15):
        raise ValueError("Raw spectral energy analysis requires a uniform lambda grid.")
    return delta


def detrend_truth(truth: np.ndarray, method: str) -> np.ndarray:
    """可选地在 lambda 方向去均值或去线性趋势，默认保留原始信号。"""

    values = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Raw truth must have shape [Q,lambda,3].")
    if method == "none":
        return values
    if method == "mean":
        return values - np.mean(values, axis=1, keepdims=True)
    if method == "linear":
        positions = np.arange(values.shape[1], dtype=np.float64)
        centered = positions - np.mean(positions)
        denominator = float(np.sum(centered**2))
        if denominator <= 0.0:
            raise ValueError("Linear detrending requires at least two lambda points.")
        mean = np.mean(values, axis=1, keepdims=True)
        slope = np.sum((values - mean) * centered[None, :, None], axis=1, keepdims=True) / denominator
        return values - (mean + slope * centered[None, :, None])
    raise ValueError(f"Unsupported detrend method: {method}")


def one_sided_rfft_energy(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算满足 Parseval 关系的一侧 rFFT 能量。"""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("Signals must have shape [Q,lambda,component] with lambda length >= 2.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signals contain non-finite values.")

    total_length = int(values.shape[1])
    coefficients = np.fft.rfft(values, axis=1)
    energy = (np.abs(coefficients) ** 2) / float(total_length)
    if total_length % 2 == 0:
        if energy.shape[1] > 2:
            energy[:, 1:-1, :] *= 2.0
    else:
        energy[:, 1:, :] *= 2.0
    return coefficients, energy


def validate_parseval_energy(signals: np.ndarray, one_sided_energy: np.ndarray) -> None:
    """在运行时确认一侧能量与时域平方和一致。"""

    time_energy = np.sum(np.asarray(signals, dtype=np.float64) ** 2, axis=1)
    spectral_energy = np.sum(np.asarray(one_sided_energy, dtype=np.float64), axis=1)
    if not np.allclose(time_energy, spectral_energy, rtol=1e-10, atol=1e-10):
        raise RuntimeError("One-sided rFFT energy failed the Parseval consistency check.")


def build_common_physical_bands(cutoffs: dict[str, float]) -> list[SpectralBand]:
    """由三个实际 retained cutoff 构造无间隙物理频率区间。"""

    if set(cutoffs) != {"T1200", "T1800", "T2400"}:
        raise ValueError("Cutoffs must contain T1200, T1800, and T2400.")
    short_cutoff = float(cutoffs["T1200"])
    medium_cutoff = float(cutoffs["T1800"])
    long_cutoff = float(cutoffs["T2400"])
    if not (0.0 < long_cutoff < medium_cutoff < short_cutoff):
        raise ValueError("Expected strictly decreasing T1200/T1800/T2400 physical cutoffs.")
    return [
        SpectralBand("0_to_t2400_cutoff", 0.0, long_cutoff, True, True),
        SpectralBand("above_t2400_to_t1800_cutoff", long_cutoff, medium_cutoff, False, True),
        SpectralBand("above_t1800_to_t1200_cutoff", medium_cutoff, short_cutoff, False, True),
        SpectralBand("above_t1200_cutoff", short_cutoff, None, False, False),
    ]


def spectral_band_mask(frequencies: np.ndarray, band: SpectralBand) -> np.ndarray:
    """按明确的左右端点语义选择物理频率 bin。"""

    values = np.asarray(frequencies, dtype=np.float64)
    mask = np.ones(values.shape, dtype=bool)
    if band.lower is not None:
        if band.lower_inclusive:
            mask &= values >= band.lower - EPSILON
        else:
            mask &= values > band.lower + EPSILON
    if band.upper is not None:
        if band.upper_inclusive:
            mask &= values <= band.upper + EPSILON
        else:
            mask &= values < band.upper - EPSILON
    return mask


def validate_band_partition(frequencies: np.ndarray, bands: list[SpectralBand]) -> None:
    """确保 common physical bands 对每个 rFFT bin 恰好覆盖一次。"""

    coverage = np.zeros(np.asarray(frequencies).shape, dtype=np.int64)
    for band in bands:
        coverage += spectral_band_mask(frequencies, band).astype(np.int64)
    if not np.all(coverage == 1):
        raise RuntimeError("Common physical-frequency bands are not gap-free and non-overlapping.")


def _component_spectra(component_energy_spectrum: np.ndarray) -> dict[str, np.ndarray]:
    """把 x/y/z 平均频谱扩展为三个分量和 xyz 汇总频谱。"""

    values = np.asarray(component_energy_spectrum, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Component energy spectrum must have shape [frequency,3].")
    result = {name: values[:, index] for index, name in enumerate(COMPONENT_NAMES)}
    result["xyz_aggregate"] = np.sum(values, axis=1)
    return result


def summarize_energy_partition(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    highest_retained_index: int,
) -> dict[str, float | int]:
    """汇总 retained index 与 physical cutoff 的能量分区。"""

    values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    frequency_values = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if values.shape != frequency_values.shape:
        raise ValueError("Spectrum and frequency axis must have equal shape.")
    if highest_retained_index < 0 or highest_retained_index >= values.size:
        raise ValueError("Highest retained lambda index is outside the rFFT spectrum.")

    total = float(np.sum(values))
    retained = float(np.sum(values[: highest_retained_index + 1]))
    above = float(np.sum(values[highest_retained_index + 1 :]))
    cutoff = float(frequency_values[highest_retained_index])
    physical_mask = frequency_values <= cutoff + EPSILON
    physical_retained = float(np.sum(values[physical_mask]))
    physical_above = float(np.sum(values[~physical_mask]))
    if not np.isclose(retained, physical_retained, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Retained-index and physical-cutoff partitions disagree.")
    if not np.isclose(above, physical_above, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Above-cutoff partitions disagree.")

    denominator = total + EPSILON
    return {
        "total_spectral_energy": total,
        "retained_energy": retained,
        "above_retained_energy": above,
        "retained_energy_fraction": retained / denominator,
        "above_retained_energy_fraction": above / denominator,
        "highest_retained_index": int(highest_retained_index),
        "physical_cutoff": cutoff,
    }


def extract_dominant_peaks(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    total_length: int,
    component: str,
    top_k: int = DEFAULT_TOP_PEAKS,
    exclude_dc: bool = True,
) -> list[dict[str, Any]]:
    """从 Q 平均能量谱提取少量非 DC 的主导峰。"""

    values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    frequency_values = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if values.shape != frequency_values.shape:
        raise ValueError("Spectrum and frequency axis must have equal shape.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    candidates = np.arange(values.size, dtype=np.int64)
    if exclude_dc:
        candidates = candidates[candidates != 0]
    candidates = candidates[values[candidates] > 0.0]
    if candidates.size == 0:
        return []
    ranked = candidates[np.argsort(values[candidates], kind="stable")[::-1]]
    total = float(np.sum(values)) + EPSILON
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked[:top_k], start=1):
        frequency = float(frequency_values[index])
        rows.append(
            {
                "total_length": int(total_length),
                "component": component,
                "rank": int(rank),
                "discrete_index": int(index),
                "physical_frequency": frequency,
                "period": float(1.0 / frequency) if frequency > 0.0 else None,
                "energy_fraction": float(values[index] / total),
            }
        )
    return rows


def analyze_length_field(
    field: CanonicalQField,
    retained_modes_lambda: int,
    detrend: str,
) -> LengthSpectralAnalysis:
    """分析一个 canonical raw field，不保留或写出完整 FFT 数组。"""

    if retained_modes_lambda <= 0:
        raise ValueError("retained_modes_lambda must be positive.")
    total_length = int(field.lambda_grid.size)
    delta_lambda = infer_uniform_delta_lambda(field.lambda_grid)
    signals = detrend_truth(field.canonical_truth, detrend)
    _, one_sided_energy = one_sided_rfft_energy(signals)
    validate_parseval_energy(signals, one_sided_energy)

    frequencies = np.fft.rfftfreq(total_length, d=delta_lambda).astype(np.float64)
    effective_retained_modes = min(int(retained_modes_lambda), int(frequencies.size))
    highest_retained_index = effective_retained_modes - 1
    component_energy_spectrum = np.mean(one_sided_energy, axis=0)
    spectra = _component_spectra(component_energy_spectrum)
    component_summaries = {
        component: summarize_energy_partition(spectrum, frequencies, highest_retained_index)
        for component, spectrum in spectra.items()
    }
    peak_rows: list[dict[str, Any]] = []
    for component, spectrum in spectra.items():
        peak_rows.extend(
            extract_dominant_peaks(
                spectrum=spectrum,
                frequencies=frequencies,
                total_length=total_length,
                component=component,
            )
        )

    return LengthSpectralAnalysis(
        task_name=field.task_name,
        total_length=total_length,
        delta_lambda=delta_lambda,
        dft_period=float(total_length * delta_lambda),
        frequency_resolution=float(1.0 / (total_length * delta_lambda)),
        frequencies=frequencies,
        component_energy_spectrum=component_energy_spectrum,
        retained_modes=effective_retained_modes,
        highest_retained_index=highest_retained_index,
        physical_cutoff=float(frequencies[highest_retained_index]),
        component_summaries=component_summaries,
        peak_rows=peak_rows,
    )


def compute_band_rows(
    analysis: LengthSpectralAnalysis,
    bands: list[SpectralBand],
) -> list[dict[str, Any]]:
    """汇总一个长度在共同物理频率区间内的分量能量。"""

    validate_band_partition(analysis.frequencies, bands)
    rows: list[dict[str, Any]] = []
    spectra = _component_spectra(analysis.component_energy_spectrum)
    for component, spectrum in spectra.items():
        total = float(np.sum(spectrum)) + EPSILON
        for band in bands:
            mask = spectral_band_mask(analysis.frequencies, band)
            energy = float(np.sum(spectrum[mask]))
            rows.append(
                {
                    "total_length": int(analysis.total_length),
                    "component": component,
                    "band_label": band.label,
                    "frequency_start": band.lower,
                    "frequency_end": band.upper,
                    "energy": energy,
                    "energy_fraction": energy / total,
                }
            )
    return rows


def compare_prefix_spectra(
    left_truth: np.ndarray,
    right_truth: np.ndarray,
    detrend: str,
) -> dict[str, Any]:
    """比较相同长度 raw prefix 的 rFFT，验证相同信号得到相同频谱。"""

    left = detrend_truth(left_truth, detrend)
    right = detrend_truth(right_truth, detrend)
    left_fft = np.fft.rfft(left, axis=1)
    right_fft = np.fft.rfft(right, axis=1)
    if left_fft.shape != right_fft.shape:
        raise ValueError("Prefix spectra have different shapes.")
    difference = np.abs(left_fft - right_fft)
    return {
        "exact_equal": bool(np.array_equal(left_fft, right_fft)),
        "allclose": bool(np.allclose(left_fft, right_fft, rtol=1e-12, atol=1e-12)),
        "max_abs_difference": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs_difference": float(np.mean(difference)) if difference.size else 0.0,
        "spectrum_shape": [int(value) for value in left_fft.shape],
    }


def build_prefix_spectral_consistency(
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
    detrend: str,
) -> dict[str, dict[str, Any]]:
    """生成 T1200 与 T1800 shared-prefix 频谱一致性控制结果。"""

    short_length = int(short.lambda_grid.size)
    medium_length = int(medium.lambda_grid.size)
    return {
        "short_vs_medium_t1200": compare_prefix_spectra(
            short.canonical_truth,
            medium.canonical_truth[:, :short_length, :],
            detrend,
        ),
        "short_vs_long_t1200": compare_prefix_spectra(
            short.canonical_truth,
            long.canonical_truth[:, :short_length, :],
            detrend,
        ),
        "medium_vs_long_t1800": compare_prefix_spectra(
            medium.canonical_truth,
            long.canonical_truth[:, :medium_length, :],
            detrend,
        ),
    }


def build_mode_shift_evidence(
    analyses: dict[str, LengthSpectralAnalysis],
) -> list[dict[str, Any]]:
    """以 T1200 xyz 汇总主峰为参考，记录相同物理频率的最近 bin 映射。"""

    short_analysis = analyses["T1200"]
    short_spectrum = _component_spectra(short_analysis.component_energy_spectrum)["xyz_aggregate"]
    references = extract_dominant_peaks(
        spectrum=short_spectrum,
        frequencies=short_analysis.frequencies,
        total_length=short_analysis.total_length,
        component="xyz_aggregate",
    )
    rows: list[dict[str, Any]] = []
    for reference in references:
        frequency = float(reference["physical_frequency"])
        row: dict[str, Any] = {
            "reference_source": "T1200_xyz_aggregate_dominant_peak",
            "reference_rank": int(reference["rank"]),
            "physical_peak_frequency": frequency,
        }
        for length_label, analysis in analyses.items():
            nearest = int(np.argmin(np.abs(analysis.frequencies - frequency)))
            aggregate = _component_spectra(analysis.component_energy_spectrum)["xyz_aggregate"]
            total = float(np.sum(aggregate)) + EPSILON
            suffix = length_label.lower()
            row[f"nearest_bin_{suffix}"] = nearest
            row[f"nearest_frequency_{suffix}"] = float(analysis.frequencies[nearest])
            row[f"continuous_index_{suffix}"] = float(frequency * analysis.dft_period)
            row[f"nearest_bin_energy_fraction_{suffix}"] = float(aggregate[nearest] / total)
        rows.append(row)
    return rows


def analyze_triplet(
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
    retained_modes_lambda: int,
    detrend: str,
) -> tuple[dict[str, LengthSpectralAnalysis], list[dict[str, Any]], dict[str, Any]]:
    """执行三长度 raw spectral analysis，并返回仅供紧凑输出使用的结果。"""

    validate_triplet(short, medium, long)
    fields = {"T1200": short, "T1800": medium, "T2400": long}
    analyses = {
        label: analyze_length_field(field, retained_modes_lambda, detrend)
        for label, field in fields.items()
    }
    cutoffs = {label: analysis.physical_cutoff for label, analysis in analyses.items()}
    bands = build_common_physical_bands(cutoffs)
    band_rows = [row for analysis in analyses.values() for row in compute_band_rows(analysis, bands)]
    controls = build_prefix_spectral_consistency(short, medium, long, detrend)
    return analyses, band_rows, {
        "bands": bands,
        "prefix_spectral_consistency": controls,
        "mode_shift_evidence": build_mode_shift_evidence(analyses),
    }


def _analysis_summary(analysis: LengthSpectralAnalysis) -> dict[str, Any]:
    """将单长度分析转换为不包含完整频谱数组的 JSON 汇总。"""

    return {
        "task_name": analysis.task_name,
        "N": int(analysis.total_length),
        "lambda_min": None,
        "lambda_max": None,
        "delta_lambda": float(analysis.delta_lambda),
        "dft_logical_period": float(analysis.dft_period),
        "frequency_resolution": float(analysis.frequency_resolution),
        "retained_modes_lambda_effective": int(analysis.retained_modes),
        "retained_k_range": [0, int(analysis.highest_retained_index)],
        "physical_cutoff": float(analysis.physical_cutoff),
        "component_energy_partitions": analysis.component_summaries,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    pair_validation_path: Path,
    pair_validation: dict[str, Any],
    short: CanonicalQField,
    medium: CanonicalQField,
    long: CanonicalQField,
    analyses: dict[str, LengthSpectralAnalysis],
    auxiliary: dict[str, Any],
) -> dict[str, Any]:
    """构建可复核且不保存完整 FFT 或 trajectory 的结果摘要。"""

    fields = {"T1200": short, "T1800": medium, "T2400": long}
    lengths: dict[str, dict[str, Any]] = {}
    for label, field in fields.items():
        item = _analysis_summary(analyses[label])
        item["lambda_min"] = float(field.lambda_grid[0])
        item["lambda_max"] = float(field.lambda_grid[-1])
        lengths[label] = item

    bands = auxiliary["bands"]
    return {
        "schema_version": "1.0",
        "diagnostic_type": "raw_kerr_lambda_spectral_energy",
        "status": "completed",
        "task_names": {
            "short": short.task_name,
            "medium": medium.task_name,
            "long": long.task_name,
        },
        "q_count": int(short.canonical_q.size),
        "source_identity_order": "train_then_val_then_test_original_row_order",
        "canonical_model_input_q_order": "ascending_Q_full_field",
        "canonical_q_exact_match_across_all_lengths": True,
        "stage2_prefix_validation": {
            "artifact_path": _relative_path(pair_validation_path),
            "pair_classification": {
                key: pair_validation["pair_classification"][key]
                for key in ("short_to_medium", "short_to_long", "medium_to_long")
            },
            "scientific_reuse": {
                key: pair_validation["scientific_reuse"][key]
                for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1")
            },
        },
        "input_truth": {
            "source": "raw_dataset_truth_float64",
            "model_or_checkpoint_loaded": False,
            "detrend": str(args.detrend),
        },
        "fft_convention": {
            "transform": "numpy.fft.rfft along lambda axis",
            "physical_frequency": "f_k = k / (N * delta_lambda)",
            "angular_frequency": "omega_k = 2*pi*k / (N * delta_lambda)",
            "dft_period_definition": "N * delta_lambda; not (N - 1) * delta_lambda",
            "energy_definition": (
                "abs(rfft)^2 / N, with factor 2 for non-DC/non-Nyquist positive "
                "bins; sums to time-domain squared amplitude by Parseval"
            ),
            "energy_aggregation": "mean one-sided energy over canonical Q, then component or xyz aggregation",
        },
        "retained_lambda_spectral_branch": {
            "requested_modes_lambda": int(args.retained_modes_lambda),
            "interpreted_retained_indices": "k=0..retained_modes_lambda-1, clipped only if rFFT is shorter",
        },
        "lengths": lengths,
        "common_physical_frequency_bands": [
            {
                "label": band.label,
                "frequency_start": band.lower,
                "frequency_end": band.upper,
                "interval_semantics": (
                    "[start,end]" if band.lower_inclusive and band.upper_inclusive
                    else "(start,end]" if not band.lower_inclusive and band.upper_inclusive
                    else "> start" if band.upper is None
                    else "implementation-defined"
                ),
            }
            for band in bands
        ],
        "exact_prefix_spectral_consistency": auxiliary["prefix_spectral_consistency"],
        "mode_shift_evidence": auxiliary["mode_shift_evidence"],
        "interpretation_boundary": (
            "Descriptive raw-trajectory spectral analysis only. It does not attribute "
            "A1 model failure to bandwidth shrinkage, mode-index shift, or any other mechanism."
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
        "output_files": list(OUTPUT_FILENAMES),
    }


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """写入紧凑 CSV，并保持空值显式为空字段。"""

    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_output_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    band_rows: list[dict[str, Any]],
    peak_rows: list[dict[str, Any]],
) -> None:
    """独占创建输出目录，并只写入三份紧凑产物。"""

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_as_json_value(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(
        band_rows,
        output_dir / OUTPUT_FILENAMES[1],
        [
            "total_length",
            "component",
            "band_label",
            "frequency_start",
            "frequency_end",
            "energy",
            "energy_fraction",
        ],
    )
    write_csv(
        peak_rows,
        output_dir / OUTPUT_FILENAMES[2],
        [
            "total_length",
            "component",
            "rank",
            "discrete_index",
            "physical_frequency",
            "period",
            "energy_fraction",
        ],
    )


def main() -> None:
    """执行正式 raw Kerr lambda spectral-energy diagnostic。"""

    args = parse_args()
    if args.retained_modes_lambda <= 0:
        raise ValueError("--retained-modes-lambda must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    pair_validation = load_required_pair_validation(args.dataset_pair_validation_json)
    short = load_task_raw_field(str(args.short_task_name))
    medium = load_task_raw_field(str(args.medium_task_name))
    long = load_task_raw_field(str(args.long_task_name))
    analyses, band_rows, auxiliary = analyze_triplet(
        short=short,
        medium=medium,
        long=long,
        retained_modes_lambda=int(args.retained_modes_lambda),
        detrend=str(args.detrend),
    )
    summary = build_summary(
        args=args,
        pair_validation_path=args.dataset_pair_validation_json,
        pair_validation=pair_validation,
        short=short,
        medium=medium,
        long=long,
        analyses=analyses,
        auxiliary=auxiliary,
    )
    peak_rows = [row for analysis in analyses.values() for row in analysis.peak_rows]
    write_output_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        band_rows=band_rows,
        peak_rows=peak_rows,
    )
    print("Raw Kerr spectral-energy analysis completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
