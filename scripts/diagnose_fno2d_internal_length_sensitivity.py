"""Diagnose frozen FNO2D internal sensitivity to lambda-domain length."""

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

from scripts.evaluate_formal_length_extrapolation_2d import (  # noqa: E402
    CanonicalQField,
    build_canonical_q_field,
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
    normalize_input_field,
    normalize_output_field,
)
from src.training.fno2d.target_transform_2d import transform_output_field  # noqa: E402


EPSILON = 1e-12
PAIR_LABELS = (("T1200", "T1800"), ("T1200", "T2400"), ("T1800", "T2400"))
OUTPUT_FILENAMES = (
    "m3_internal_length_sensitivity_summary.json",
    "m3_stage_comparison.csv",
    "m3_spectral_mode_comparison.csv",
)


@dataclass
class ForwardCapture:
    """保存一次正常前向中的第一层瞬时张量。"""

    lifted_feature: torch.Tensor | None = None
    first_spectral_input: torch.Tensor | None = None
    spectral_branch_output: torch.Tensor | None = None
    first_block_output: torch.Tensor | None = None


@dataclass
class SpectralCapture:
    """保存缩减后的第一层频域系数和复算一致性统计。"""

    frequencies: np.ndarray
    pre_pos: torch.Tensor
    pre_neg: torch.Tensor
    post_pos: torch.Tensor
    post_neg: torch.Tensor
    modes_q: int
    modes_lambda: int
    full_fft_energy: float
    retained_input_energy: float
    retained_output_energy: float
    replicated_output_metrics: dict[str, float]


@dataclass
class LengthCapture:
    """保存跨长度比较所需的前缀张量和紧凑频域信息。"""

    label: str
    field: CanonicalQField
    normalized_input_prefix: torch.Tensor
    lifted_feature_prefix: torch.Tensor
    first_spectral_input_prefix: torch.Tensor
    spectral_branch_output_prefix: torch.Tensor
    first_block_output_prefix: torch.Tensor
    final_prediction_prefix: np.ndarray
    final_prefix_metrics: dict[str, Any]
    spectral: SpectralCapture


def parse_args() -> argparse.Namespace:
    """解析 M3 所需的正式输入。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen FNO2D forward pass per exact-prefix lambda length and "
            "write compact first-layer internal-sensitivity diagnostics."
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


def _to_cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    """在 hook 中保存与模型计算图断开的 CPU 副本。"""

    return tensor.detach().to(device="cpu").clone()


def install_first_layer_hooks(model: torch.nn.Module, capture: ForwardCapture) -> list[Any]:
    """安装只读 hooks，且 hook 不返回任何替代输出。"""

    if not hasattr(model, "input_projection") or not hasattr(model, "blocks"):
        raise TypeError("Model does not expose the required FNO2D modules.")
    if len(model.blocks) < 1:
        raise ValueError("FNO2D model must contain at least one block.")
    first_block = model.blocks[0]

    def save_lifted(_: torch.nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
        capture.lifted_feature = _to_cpu_clone(output)

    def save_spectral_input(_: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        capture.first_spectral_input = _to_cpu_clone(inputs[0])

    def save_spectral_output(_: torch.nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
        capture.spectral_branch_output = _to_cpu_clone(output)

    def save_block_output(_: torch.nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
        capture.first_block_output = _to_cpu_clone(output)

    return [
        model.input_projection.register_forward_hook(save_lifted),
        first_block.spectral_conv.register_forward_pre_hook(save_spectral_input),
        first_block.spectral_conv.register_forward_hook(save_spectral_output),
        first_block.register_forward_hook(save_block_output),
    ]


def capture_one_forward(model: torch.nn.Module, normalized_input: torch.Tensor) -> tuple[torch.Tensor, ForwardCapture]:
    """以 hooks 运行一次正常前向，不改变模型输出。"""

    capture = ForwardCapture()
    handles = install_first_layer_hooks(model, capture)
    try:
        with torch.no_grad():
            prediction = model(normalized_input)
    finally:
        for handle in handles:
            handle.remove()
    if any(
        value is None
        for value in (
            capture.lifted_feature,
            capture.first_spectral_input,
            capture.spectral_branch_output,
            capture.first_block_output,
        )
    ):
        raise RuntimeError("Required first-layer hook did not capture a tensor.")
    return prediction, capture


def _metrics(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> dict[str, float]:
    """计算紧凑、对实数和复数均适用的差异指标。"""

    left_array = np.asarray(left.detach().cpu().numpy() if isinstance(left, torch.Tensor) else left)
    right_array = np.asarray(right.detach().cpu().numpy() if isinstance(right, torch.Tensor) else right)
    if left_array.shape != right_array.shape:
        raise ValueError(f"Comparison shapes differ: {left_array.shape} versus {right_array.shape}.")
    difference = left_array - right_array
    left_norm = float(np.linalg.norm(left_array.reshape(-1)))
    right_norm = float(np.linalg.norm(right_array.reshape(-1)))
    difference_norm = float(np.linalg.norm(difference.reshape(-1)))
    scale = left_norm + EPSILON
    dot = np.vdot(left_array.reshape(-1), right_array.reshape(-1))
    cosine = float(np.real(dot) / (left_norm * right_norm + EPSILON))
    return {
        "relative_l2_difference": difference_norm / scale,
        "normalized_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2)) / (np.sqrt(np.mean(np.abs(left_array) ** 2)) + EPSILON)),
        "cosine_similarity": cosine,
    }


def _replicate_first_spectral_layer(
    spectral_input_cpu: torch.Tensor,
    spectral_conv: torch.nn.Module,
    device: str,
) -> tuple[SpectralCapture, torch.Tensor]:
    """从捕获输入精确复算第一层 rFFT、乘权和 IFFT，不改动模型。"""

    x = spectral_input_cpu.to(device=device)
    batch, _, height, width = x.shape
    with torch.no_grad():
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        m1 = min(int(spectral_conv.modes1), height)
        m2 = min(int(spectral_conv.modes2), width // 2 + 1)
        out_ft = torch.zeros(
            batch,
            int(spectral_conv.out_channels),
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :m1, :m2] = spectral_conv.compl_mul2d(
            x_ft[:, :, :m1, :m2], spectral_conv.weights_pos[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = spectral_conv.compl_mul2d(
            x_ft[:, :, -m1:, :m2], spectral_conv.weights_neg[:, :, :m1, :m2]
        )
        replicated_output = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))
        full_energy = float(torch.sum(torch.abs(x_ft) ** 2).item())
        retained_input = float(
            (torch.sum(torch.abs(x_ft[:, :, :m1, :m2]) ** 2)
             + torch.sum(torch.abs(x_ft[:, :, -m1:, :m2]) ** 2)).item()
        )
        retained_output = float(
            (torch.sum(torch.abs(out_ft[:, :, :m1, :m2]) ** 2)
             + torch.sum(torch.abs(out_ft[:, :, -m1:, :m2]) ** 2)).item()
        )
        capture = SpectralCapture(
            frequencies=np.fft.rfftfreq(width, d=1.0).astype(np.float64),
            pre_pos=_to_cpu_clone(x_ft[:, :, :m1, :]),
            pre_neg=_to_cpu_clone(x_ft[:, :, -m1:, :]),
            post_pos=_to_cpu_clone(out_ft[:, :, :m1, :m2]),
            post_neg=_to_cpu_clone(out_ft[:, :, -m1:, :m2]),
            modes_q=m1,
            modes_lambda=m2,
            full_fft_energy=full_energy,
            retained_input_energy=retained_input,
            retained_output_energy=retained_output,
            replicated_output_metrics={},
        )
    return capture, _to_cpu_clone(replicated_output)


def _mode_energy(pos: torch.Tensor, neg: torch.Tensor) -> np.ndarray:
    """按 lambda mode 汇总正负 Q slices、batch 和 channels 的复系数能量。"""

    return (
        torch.sum(torch.abs(pos) ** 2, dim=(0, 1, 2)).numpy()
        + torch.sum(torch.abs(neg) ** 2, dim=(0, 1, 2)).numpy()
    ).astype(np.float64)


def _concat_q_slices(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """拼接两个实际参与谱卷积的 Q-mode slices。"""

    return torch.cat((pos, neg), dim=2)


def _grid_delta(lambda_grid: np.ndarray) -> float:
    """验证均匀 lambda 网格并返回采样间距。"""

    values = np.asarray(lambda_grid, dtype=np.float64).reshape(-1)
    differences = np.diff(values)
    if values.size < 2 or np.any(differences <= 0.0):
        raise ValueError("Lambda grid must be strictly increasing with at least two points.")
    delta = float(differences[0])
    if not np.allclose(differences, delta, rtol=1e-12, atol=1e-15):
        raise ValueError("M3 requires a uniform lambda grid.")
    return delta


def _physical_frequencies(field: CanonicalQField) -> np.ndarray:
    """使用实现对应的 N*delta_lambda DFT 频率约定。"""

    return np.fft.rfftfreq(field.lambda_grid.size, d=_grid_delta(field.lambda_grid)).astype(np.float64)


def _post_coefficient_at(capture: LengthCapture, index: int) -> torch.Tensor:
    """返回一个物理频率对应的 post-weight 系数，保留范围外则为零。"""

    template = _concat_q_slices(
        capture.spectral.pre_pos[:, :, :, 0:1],
        capture.spectral.pre_neg[:, :, :, 0:1],
    ).squeeze(-1)
    if index >= capture.spectral.modes_lambda:
        return torch.zeros_like(template)
    return _concat_q_slices(
        capture.spectral.post_pos[:, :, :, index : index + 1],
        capture.spectral.post_neg[:, :, :, index : index + 1],
    ).squeeze(-1)


def _pre_coefficient_at(capture: LengthCapture, index: int) -> torch.Tensor:
    """返回一个第一层输入 FFT lambda bin 的正负 Q slices。"""

    return _concat_q_slices(
        capture.spectral.pre_pos[:, :, :, index : index + 1],
        capture.spectral.pre_neg[:, :, :, index : index + 1],
    ).squeeze(-1)


def _make_length_capture(
    *,
    label: str,
    field: CanonicalQField,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    device: str,
    shared_length: int,
) -> LengthCapture:
    """构造一个长度的一次前向捕获结果。"""

    x_raw, y_raw = build_model_input(field.canonical_q, field.lambda_grid, field.canonical_truth)
    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform = load_target_transform_config_from_checkpoint(checkpoint)
    x_model = normalize_input_field(x=x_raw, stats=stats)
    y_model = normalize_output_field(y=transform_output_field(y=y_raw, config=transform), stats=stats)
    x_tensor = torch.from_numpy(x_model).float().to(device)
    prediction_model, hooks = capture_one_forward(model, x_tensor)
    spectral, replicated_output = _replicate_first_spectral_layer(
        hooks.first_spectral_input, model.blocks[0].spectral_conv, device
    )
    spectral.replicated_output_metrics = _metrics(replicated_output, hooks.spectral_branch_output)
    prediction_model_np = prediction_model.detach().cpu().numpy()
    prediction_raw, _ = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=prediction_model_np,
        targets_model_space=y_model,
        raw_targets_reference=y_raw,
        normalization_stats=stats,
        target_transform_config=transform,
    )
    prediction = np.asarray(prediction_raw[0], dtype=np.float64)
    if prediction.shape != field.canonical_truth.shape:
        raise ValueError("Recovered prediction shape does not match raw truth.")
    if shared_length > field.lambda_grid.size:
        raise ValueError("Shared prefix exceeds current lambda length.")
    prefix_metrics = compute_region_metrics(
        prediction[:, :shared_length, :], field.canonical_truth[:, :shared_length, :], field.canonical_q
    )
    return LengthCapture(
        label=label,
        field=field,
        normalized_input_prefix=_to_cpu_clone(x_tensor[:, :, :shared_length, :]),
        lifted_feature_prefix=hooks.lifted_feature[:, :, :shared_length, :].clone(),
        first_spectral_input_prefix=hooks.first_spectral_input[:, :, :, :shared_length].clone(),
        spectral_branch_output_prefix=hooks.spectral_branch_output[:, :, :, :shared_length].clone(),
        first_block_output_prefix=hooks.first_block_output[:, :, :, :shared_length].clone(),
        final_prediction_prefix=prediction[:, :shared_length, :].copy(),
        final_prefix_metrics=prefix_metrics,
        spectral=spectral,
    )


def _same_index_spectral_metrics(left: LengthCapture, right: LengthCapture, stage: str) -> dict[str, float]:
    """在模型实际接收的相同 retained lambda index 上比较系数。"""

    modes = min(left.spectral.modes_lambda, right.spectral.modes_lambda)
    if stage == "pre_weight":
        left_value = _concat_q_slices(left.spectral.pre_pos[:, :, :, :modes], left.spectral.pre_neg[:, :, :, :modes])
        right_value = _concat_q_slices(right.spectral.pre_pos[:, :, :, :modes], right.spectral.pre_neg[:, :, :, :modes])
    elif stage == "post_weight":
        left_value = _concat_q_slices(left.spectral.post_pos[:, :, :, :modes], left.spectral.post_neg[:, :, :, :modes])
        right_value = _concat_q_slices(right.spectral.post_pos[:, :, :, :modes], right.spectral.post_neg[:, :, :, :modes])
    else:
        raise ValueError(f"Unsupported spectral stage: {stage}")
    return _metrics(left_value, right_value)


def build_stage_rows(captures: dict[str, LengthCapture]) -> list[dict[str, Any]]:
    """建立主科学输出：各阶段 shared-prefix invariance 表。"""

    rows: list[dict[str, Any]] = []
    spatial_stages = (
        ("normalized_input_prefix", "normalized_input_prefix"),
        ("lifted_feature_prefix", "lifted_feature_prefix"),
        ("first_spectral_input_prefix", "first_spectral_input_prefix"),
        ("spectral_branch_output_prefix", "spectral_branch_output_prefix"),
        ("first_block_output_prefix", "first_block_output_prefix"),
        ("final_prediction_prefix", "final_prediction_prefix"),
    )
    for left_label, right_label in PAIR_LABELS:
        left, right = captures[left_label], captures[right_label]
        for stage_name, attribute in spatial_stages:
            rows.append({"comparison": f"{left_label}_vs_{right_label}", "stage": stage_name, "view": "shared_spatial_prefix", **_metrics(getattr(left, attribute), getattr(right, attribute))})
        for spectral_stage, name in (("pre_weight", "first_fft_retained_same_index"), ("post_weight", "post_weight_spectral_same_index")):
            rows.append({"comparison": f"{left_label}_vs_{right_label}", "stage": name, "view": "same_discrete_index", **_same_index_spectral_metrics(left, right, spectral_stage)})
    return rows


def build_spectral_rows(captures: dict[str, LengthCapture]) -> list[dict[str, Any]]:
    """建立 same-index 与物理频率对齐的紧凑 mode 级比较。"""

    rows: list[dict[str, Any]] = []
    for left_label, right_label in PAIR_LABELS:
        left, right = captures[left_label], captures[right_label]
        left_freq = _physical_frequencies(left.field)
        right_freq = _physical_frequencies(right.field)
        left_pre_energy = _mode_energy(left.spectral.pre_pos, left.spectral.pre_neg)
        right_pre_energy = _mode_energy(right.spectral.pre_pos, right.spectral.pre_neg)
        for index in range(min(left.spectral.modes_lambda, right.spectral.modes_lambda)):
            for stage in ("pre_weight", "post_weight"):
                if stage == "pre_weight":
                    metrics = _metrics(_pre_coefficient_at(left, index), _pre_coefficient_at(right, index))
                else:
                    metrics = _metrics(_post_coefficient_at(left, index), _post_coefficient_at(right, index))
                rows.append({"comparison": f"{left_label}_vs_{right_label}", "view": "same_discrete_index", "spectral_stage": stage, "lambda_index_left": index, "lambda_index_right": index, "physical_frequency_left": float(left_freq[index]), "physical_frequency_right": float(right_freq[index]), "right_index_retained": True, "left_energy_fraction": float(left_pre_energy[index] / (np.sum(left_pre_energy) + EPSILON)), "right_energy_fraction": float(right_pre_energy[index] / (np.sum(right_pre_energy) + EPSILON)), **metrics})
        top = np.argsort(left_pre_energy)[::-1][: min(10, left_pre_energy.size)]
        for index in top:
            nearest = int(np.argmin(np.abs(right_freq - left_freq[index])))
            for stage in ("pre_weight", "post_weight"):
                metrics = _metrics(_pre_coefficient_at(left, int(index)), _pre_coefficient_at(right, nearest)) if stage == "pre_weight" else _metrics(_post_coefficient_at(left, int(index)), _post_coefficient_at(right, nearest))
                rows.append({"comparison": f"{left_label}_vs_{right_label}", "view": "physical_frequency_aligned", "spectral_stage": stage, "lambda_index_left": int(index), "lambda_index_right": nearest, "physical_frequency_left": float(left_freq[index]), "physical_frequency_right": float(right_freq[nearest]), "right_index_retained": bool(nearest < right.spectral.modes_lambda), "left_energy_fraction": float(left_pre_energy[index] / (np.sum(left_pre_energy) + EPSILON)), "right_energy_fraction": float(right_pre_energy[nearest] / (np.sum(right_pre_energy) + EPSILON)), **metrics})
    return rows


def _length_summary(capture: LengthCapture) -> dict[str, Any]:
    """生成不含大型张量的单长度摘要。"""

    frequencies = _physical_frequencies(capture.field)
    energy = _mode_energy(capture.spectral.pre_pos, capture.spectral.pre_neg)
    top = np.argsort(energy)[::-1][: min(10, energy.size)]
    return {
        "task_name": capture.field.task_name,
        "N": int(capture.field.lambda_grid.size),
        "lambda_min": float(capture.field.lambda_grid[0]),
        "lambda_max": float(capture.field.lambda_grid[-1]),
        "delta_lambda": _grid_delta(capture.field.lambda_grid),
        "dft_logical_period": float(capture.field.lambda_grid.size * _grid_delta(capture.field.lambda_grid)),
        "retained_q_modes_per_sign": int(capture.spectral.modes_q),
        "retained_lambda_indices": [0, int(capture.spectral.modes_lambda - 1)],
        "full_fft_energy": capture.spectral.full_fft_energy,
        "retained_input_energy": capture.spectral.retained_input_energy,
        "retained_output_energy": capture.spectral.retained_output_energy,
        "replicated_spectral_output_vs_hook": capture.spectral.replicated_output_metrics,
        "final_prediction_prefix_metrics": capture.final_prefix_metrics,
        "top_input_fft_lambda_modes": [{"index": int(index), "physical_frequency": float(frequencies[index]), "energy_fraction": float(energy[index] / (np.sum(energy) + EPSILON))} for index in top],
    }


def build_summary(
    *, args: argparse.Namespace, checkpoint_path: Path, checkpoint: dict[str, Any], pair_validation: dict[str, Any], pair_path: Path, captures: dict[str, LengthCapture], stage_rows: list[dict[str, Any]], spectral_rows: list[dict[str, Any]], model: torch.nn.Module, shared_length: int,
) -> dict[str, Any]:
    """生成 M3 可审计摘要，不序列化任何中间张量。"""

    stats = load_normalization_stats_from_checkpoint(checkpoint)
    transform = load_target_transform_config_from_checkpoint(checkpoint)
    return {
        "schema_version": "1.0",
        "diagnostic_type": "fno2d_internal_lambda_domain_length_sensitivity_m3",
        "status": "completed",
        "tasks": {"training": str(args.training_task_name), "short": captures["T1200"].field.task_name, "medium": captures["T1800"].field.task_name, "long": captures["T2400"].field.task_name},
        "model_name": str(args.model_name),
        "checkpoint_path": _relative_path(checkpoint_path),
        "model_config": _as_json_value(checkpoint.get("config", {}).get("model_config", {})),
        "model_architecture": {"modes1": int(model.modes1), "modes2": int(model.modes2), "width": int(model.width), "depth": int(model.depth)},
        "normalization": {"statistics_source": "checkpoint_training_dataset", "values": _as_json_value(stats.to_dict())},
        "target_transform": _as_json_value(transform.to_dict()),
        "stage2_prefix_validation": {"artifact_path": _relative_path(pair_path), "pair_classification": {key: pair_validation["pair_classification"][key] for key in ("short_to_medium", "short_to_long", "medium_to_long")}, "scientific_reuse": {key: pair_validation["scientific_reuse"][key] for key in ("historical_t1800_reusable", "t2400_ready_for_future_a1")}},
        "canonical_q": {"source_identity_order": "train_then_val_then_test_original_row_order", "model_input_order": "stable_ascending_Q_full_field", "exact_match_across_lengths": True},
        "shared_prefix_length": int(shared_length),
        "captured_stages": ["normalized_input", "input_projection_output", "blocks[0].spectral_conv_input", "replicated_rfft2", "replicated_post_weight_coefficients", "blocks[0].spectral_conv_output", "blocks[0]_output", "final_prediction"],
        "hooks_used": ["input_projection forward hook", "blocks[0].spectral_conv forward pre-hook", "blocks[0].spectral_conv forward hook", "blocks[0] forward hook"],
        "replicated_fft": {"transform": "torch.fft.rfft2(dim=(-2,-1))", "inverse": "torch.fft.irfft2(s=(H,W), dim=(-2,-1))", "lambda_frequency": "f_k = k / (N * delta_lambda)", "same_index_view": "k=0..modes2-1", "physical_view": "nearest bin at the selected left physical frequency"},
        "frozen_protocol": {"one_model_forward_per_length": True, "total_model_forwards": 3, "model_eval": True, "torch_no_grad": True, "optimizer": False, "scheduler": False, "backward": False, "adaptation": False, "fine_tuning": False, "autoregression": False, "prediction_feedback": False, "teacher_forcing": False},
        "lengths": {label: _length_summary(capture) for label, capture in captures.items()},
        "stage_comparison_row_count": len(stage_rows),
        "spectral_mode_comparison_row_count": len(spectral_rows),
        "interpretation_boundary": "Observational internal diagnostic only; it does not identify a sole causal mechanism, eliminate coordinate effects or bandwidth effects, or prescribe an architectural fix.",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda, "device": str(args.device)},
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


def write_output_artifacts(*, output_dir: Path, summary: dict[str, Any], stage_rows: list[dict[str, Any]], spectral_rows: list[dict[str, Any]]) -> None:
    """独占创建目录，并且只写入三个紧凑 M3 产物。"""

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / OUTPUT_FILENAMES[0]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_as_json_value(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(stage_rows, output_dir / OUTPUT_FILENAMES[1], ["comparison", "stage", "view", "relative_l2_difference", "normalized_rmse", "cosine_similarity"])
    write_csv(spectral_rows, output_dir / OUTPUT_FILENAMES[2], ["comparison", "view", "spectral_stage", "lambda_index_left", "lambda_index_right", "physical_frequency_left", "physical_frequency_right", "right_index_retained", "left_energy_fraction", "right_energy_fraction", "relative_l2_difference", "normalized_rmse", "cosine_similarity"])


def run_m3(*, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """执行正式 M3；调用方负责在新目录写入紧凑结果。"""

    if args.output_dir.exists():
        raise FileExistsError(f"M3 output directory already exists: {args.output_dir}")
    pair_validation = load_required_pair_validation(args.dataset_pair_validation_json)
    short = load_task_raw_field(str(args.short_task_name))
    medium = load_task_raw_field(str(args.medium_task_name))
    long = load_task_raw_field(str(args.long_task_name))
    validate_triplet(short, medium, long)
    checkpoint_path = args.checkpoint_path or (PROJECT_ROOT / "outputs" / str(args.training_task_name) / str(args.model_name) / "checkpoints" / "best_model.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = load_checkpoint_2d(checkpoint_path=checkpoint_path, device=str(args.device))
    model = load_fno2d_checkpoint_model(checkpoint=checkpoint, device=str(args.device))
    model.eval()
    shared_length = int(short.lambda_grid.size)
    captures = {label: _make_length_capture(label=label, field=field, model=model, checkpoint=checkpoint, device=str(args.device), shared_length=shared_length) for label, field in (("T1200", short), ("T1800", medium), ("T2400", long))}
    stage_rows = build_stage_rows(captures)
    spectral_rows = build_spectral_rows(captures)
    summary = build_summary(args=args, checkpoint_path=checkpoint_path, checkpoint=checkpoint, pair_validation=pair_validation, pair_path=args.dataset_pair_validation_json, captures=captures, stage_rows=stage_rows, spectral_rows=spectral_rows, model=model, shared_length=shared_length)
    return summary, stage_rows, spectral_rows


def main() -> None:
    """执行 M3 CLI。"""

    args = parse_args()
    summary, stage_rows, spectral_rows = run_m3(args=args)
    write_output_artifacts(output_dir=args.output_dir, summary=summary, stage_rows=stage_rows, spectral_rows=spectral_rows)
    print("M3 internal FNO2D length-sensitivity diagnostic completed.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
