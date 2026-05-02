# ==========================================================
# File: scripts/run_analysis_2d.py
#
# 功能简介：
# 1. 二维模型的独立推理分析入口；
# 2. 不修改原来的 scripts/run_analysis.py；
# 3. 支持通过 registry_2d.py 恢复二维模型；
# 4. 支持 normalization = none / standard；
# 5. 支持 target transform = raw / residual_initial；
# 6. 如果训练时使用 standard normalization 和 residual_initial，
#    则分析时执行：
#
#       model output
#       -> denormalization
#       -> inverse target transform
#       -> raw xyz physical space
#
# 7. 最终在物理空间 raw xyz 上计算 MSE / Relative L2。
#
# 当前二维算子任务：
#
#       (p, lambda) -> (x, y, z)
#
# 其中：
# - p 可以是 Q / a / E / Lz；
# - lambda 是轨道参数方向；
# - 输出是 Kerr 轨道 xyz 坐标。
#
# 输入：
#   data/tasks/<task_name>/dataset.npz
#   outputs/<task_name>/<model_name>/checkpoints/best_model.pt
#
# 输出：
#   outputs/<task_name>/<model_name>/inference/
#   outputs/<task_name>/<model_name>/analysis/
# ==========================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ==========================================================
# 一、保证可以从 scripts/ 正确导入 src/
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.registry_2d import build_model_2d  # noqa: E402
from src.training.fno2d.dataset_loader_2d import (  # noqa: E402
    build_fno2d_dataloaders,
    summarize_fno2d_bundle,
)
from src.training.fno2d.normalization_2d import (  # noqa: E402
    FieldNormalizationStats,
    denormalize_output_field,
)
from src.training.fno2d.target_transform_2d import (  # noqa: E402
    TargetTransformConfig,
    inverse_transform_output_field,
)
from src.common.io_utils import load_json  # noqa: E402
from src.common.paths import get_task_meta_json_path  # noqa: E402
from src.inference_analysis.timing import (  # noqa: E402
    build_full_param_dicts_for_timing,
    time_traditional_orbit_generation_from_param_dicts,
)


# ==========================================================
# 二、路径工具
# ==========================================================

def get_2d_model_dirs(task_name: str, model_name: str) -> dict[str, Path]:
    """
    构造二维模型推理分析目录。

    返回：
    - model_dir
    - checkpoint_path
    - inference_dir
    - analysis_dir
    """
    model_dir = PROJECT_ROOT / "outputs" / task_name / model_name
    checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
    inference_dir = model_dir / "inference"
    analysis_dir = model_dir / "analysis"

    inference_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_dir": model_dir,
        "checkpoint_path": checkpoint_path,
        "inference_dir": inference_dir,
        "analysis_dir": analysis_dir,
    }


def save_json(obj: dict[str, Any], path: Path) -> None:
    """
    保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def save_npy(array: np.ndarray, path: Path) -> None:
    """
    保存 NPY 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


# ==========================================================
# 三、指标函数
# ==========================================================

def compute_mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算整体 MSE。

    pred / target:
        [B, H, W, 3]
    """
    return float(np.mean((pred - target) ** 2))


def compute_relative_l2_np(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    计算整体 Relative L2。

    对每个 batch 样本展平后计算，再取平均。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target shape 必须一致，当前 pred={pred.shape}, target={target.shape}"
        )

    batch_size = int(pred.shape[0])
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    diff_norm = np.linalg.norm(pred_flat - target_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    rel = diff_norm / (target_norm + eps)

    return float(np.mean(rel))


def compute_metrics_2d(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """
    计算二维场推理指标。

    注意：
    - pred 和 target 必须已经在 raw xyz 物理空间；
    - 如果训练时使用 normalization / target transform，
      必须先完成反归一化和 inverse transform 再传入。
    """
    return {
        "mse": compute_mse_np(pred, target),
        "relative_l2": compute_relative_l2_np(pred, target),
        "pred_shape": list(pred.shape),
        "target_shape": list(target.shape),
    }


# ==========================================================
# 四、checkpoint / 模型 / 变换配置加载
# ==========================================================

def load_checkpoint_2d(
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    """
    加载二维模型 checkpoint。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    return torch.load(checkpoint_path, map_location=device)


def load_fno2d_checkpoint_model(
    checkpoint: dict[str, Any],
    device: str,
) -> nn.Module:
    """
    从 checkpoint 恢复二维模型。

    checkpoint 中需要包含：
    - model_state_dict
    - config
    - config["model_config"]
    """
    config = checkpoint["config"]
    model_config = config["model_config"]

    model = build_model_2d(
        model_type=model_config["model_type"],
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        modes1=int(model_config["modes1"]),
        modes2=int(model_config["modes2"]),
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        activation=str(model_config.get("activation", "gelu")),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_normalization_method_from_checkpoint(
    checkpoint: dict[str, Any],
) -> str:
    """
    从 checkpoint config 中读取 normalization 方法。
    """
    config = checkpoint.get("config", {})
    return str(config.get("normalization", "none"))


def get_target_transform_method_from_checkpoint(
    checkpoint: dict[str, Any],
) -> str:
    """
    从 checkpoint config 中读取 target transform 方法。
    """
    config = checkpoint.get("config", {})
    return str(config.get("target_transform", "raw"))


def get_lambda_reference_index_from_checkpoint(
    checkpoint: dict[str, Any],
) -> int:
    """
    从 checkpoint config 中读取 residual_initial 的参考 lambda 索引。
    """
    config = checkpoint.get("config", {})
    return int(config.get("lambda_reference_index", 0))


def load_normalization_stats_from_checkpoint(
    checkpoint: dict[str, Any],
) -> FieldNormalizationStats:
    """
    从 checkpoint config 中读取 FNO2d normalization stats。

    训练时 train_config 里保存了 dataset_summary，
    dataset_summary 中包含 normalization_stats。
    """
    config = checkpoint["config"]

    dataset_summary = config.get("dataset_summary", None)
    if dataset_summary is None:
        raise KeyError(
            "checkpoint['config'] 中没有 dataset_summary，"
            "无法恢复 normalization_stats。"
        )

    stats_dict = dataset_summary.get("normalization_stats", None)
    if stats_dict is None:
        raise KeyError(
            "dataset_summary 中没有 normalization_stats，"
            "无法恢复 normalization stats。"
        )

    return FieldNormalizationStats.from_dict(stats_dict)


def load_target_transform_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> TargetTransformConfig:
    """
    从 checkpoint config 中读取 target transform config。

    优先读取 dataset_summary 中的 target_transform_config；
    如果没有，则从 config 顶层字段恢复。
    """
    config = checkpoint["config"]

    dataset_summary = config.get("dataset_summary", {})
    transform_dict = dataset_summary.get("target_transform_config", None)

    if transform_dict is not None:
        return TargetTransformConfig.from_dict(transform_dict)

    return TargetTransformConfig(
        mode=get_target_transform_method_from_checkpoint(checkpoint),
        lambda_reference_index=get_lambda_reference_index_from_checkpoint(checkpoint),
    )


# ==========================================================
# 五、模型推理与计时
# ==========================================================

@torch.no_grad()
def predict_2d_loader(
    model: nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对二维场 DataLoader 做推理。

    返回：
    - predictions: [B, H, W, 3]
    - targets:     [B, H, W, 3]

    注意：
    - 这里返回的是模型训练目标空间中的结果；
    - 如果使用 standard normalization，则是 normalized target space；
    - 如果使用 residual_initial，则是 residual target space；
    - 后续必须做 inverse pipeline 才能回到 raw xyz。
    """
    model.eval()

    preds = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    predictions = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets


@torch.no_grad()
def time_model_inference_2d(
    model: nn.Module,
    loader,
    device: str,
    warmup: bool = True,
) -> dict[str, Any]:
    """
    统计二维模型推理时间。

    注意：
    - 当前 FNO2d 一个测试 split 通常是一个二维场；
    - 这个二维场包含 H 条参数轨道；
    - 因此 num_param_samples 用 H 表示轨道数量。
    """
    model.eval()

    num_fields = 0
    num_param_samples = 0

    # ------------------------------------------------------
    # A. warmup，不计入正式时间
    # ------------------------------------------------------
    if warmup:
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            break

    # ------------------------------------------------------
    # B. 正式计时
    # ------------------------------------------------------
    start = perf_counter()

    for x, _ in loader:
        x = x.to(device)

        _ = model(x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        # x: [B, H, W, C]
        num_fields += int(x.shape[0])
        num_param_samples += int(x.shape[0] * x.shape[1])

    total_seconds = perf_counter() - start

    avg_seconds_per_param_sample = (
        total_seconds / num_param_samples
        if num_param_samples > 0
        else 0.0
    )

    return {
        "model_total_seconds": float(total_seconds),
        "model_avg_seconds_per_sample": float(avg_seconds_per_param_sample),
        "num_fields": int(num_fields),
        "num_samples": int(num_param_samples),
    }


# ==========================================================
# 六、raw target loader：用于 inverse target transform
# ==========================================================

def load_raw_test_targets_for_inverse(
    task_name: str,
    batch_size: int,
) -> tuple[np.ndarray, Any]:
    """
    构造 raw target test loader，用于 inverse target transform。

    为什么需要：
    - residual_initial 的 inverse transform 需要 raw y 的参考点 y(lambda_0)；
    - 训练/推理 loader 中的 y 可能已经经过 residual + normalization；
    - 因此这里单独加载 raw y_test。
    """
    _, _, raw_test_loader, raw_bundle = build_fno2d_dataloaders(
        task_name=task_name,
        batch_size=batch_size,
        num_workers=0,
        sort_param=True,
        normalization="none",
        target_transform="raw",
        lambda_reference_index=0,
    )

    raw_targets = []

    for _, y_raw in raw_test_loader:
        raw_targets.append(y_raw.detach().cpu().numpy())

    raw_targets_np = np.concatenate(raw_targets, axis=0)

    return raw_targets_np, raw_bundle


# ==========================================================
# 七、inverse pipeline
# ==========================================================

def recover_predictions_and_targets_to_raw_xyz(
    predictions_model_space: np.ndarray,
    targets_model_space: np.ndarray,
    raw_targets_reference: np.ndarray,
    normalization_stats: FieldNormalizationStats,
    target_transform_config: TargetTransformConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将模型输出和标签从训练目标空间恢复到 raw xyz 物理空间。

    恢复顺序：
    1. denormalize:
        normalized target -> transformed target

    2. inverse target transform:
        transformed target -> raw xyz

    输入：
    - predictions_model_space:
        模型输出，可能是 normalized residual / normalized raw / raw 等。

    - targets_model_space:
        loader 给出的标签，处于同样的训练目标空间。

    - raw_targets_reference:
        原始 raw xyz test target，用于 residual_initial 的参考点。

    - normalization_stats:
        训练时保存的 normalization stats。

    - target_transform_config:
        训练时保存的 target transform config。

    输出：
    - predictions_raw
    - targets_raw
    """
    # ------------------------------------------------------
    # A. 先从 normalized target space 还原到 transformed target space
    # ------------------------------------------------------
    predictions_transformed = denormalize_output_field(
        y_norm=predictions_model_space,
        stats=normalization_stats,
    )

    targets_transformed = denormalize_output_field(
        y_norm=targets_model_space,
        stats=normalization_stats,
    )

    # ------------------------------------------------------
    # B. 再从 transformed target space 还原到 raw xyz physical space
    # ------------------------------------------------------
    predictions_raw = inverse_transform_output_field(
        transformed_y=predictions_transformed,
        reference_y_raw=raw_targets_reference,
        config=target_transform_config,
    )

    targets_raw = inverse_transform_output_field(
        transformed_y=targets_transformed,
        reference_y_raw=raw_targets_reference,
        config=target_transform_config,
    )

    return predictions_raw, targets_raw


# ==========================================================
# 八、传统数值计算计时
# ==========================================================

def load_task_fixed_info(task_name: str) -> tuple[dict[str, Any], int, float]:
    """
    从 meta.json 中读取传统积分所需信息。

    返回：
    - fixed_params
    - n_steps
    - step_size
    """
    meta_path = get_task_meta_json_path(task_name)
    meta = load_json(meta_path)

    task_spec = meta["task_spec"]

    fixed_params = task_spec["fixed_params"]
    n_steps = int(task_spec["n_steps"])
    step_size = float(task_spec["step_size"])

    return fixed_params, n_steps, step_size


def time_traditional_for_2d_test_field(
    task_name: str,
    param_name: str,
    param_values: np.ndarray,
) -> Any:
    """
    对二维测试场中的每个参数值重新进行传统数值积分计时。

    param_values:
        [H]
    """
    fixed_params, n_steps, step_size = load_task_fixed_info(task_name)

    vary_params_array = np.asarray(param_values, dtype=np.float64).reshape(-1, 1)

    full_param_dicts = build_full_param_dicts_for_timing(
        vary_params_array=vary_params_array,
        vary_params_order=[param_name],
        fixed_params=fixed_params,
    )

    traditional_timing = time_traditional_orbit_generation_from_param_dicts(
        full_param_dicts=full_param_dicts,
        n_steps=n_steps,
        step_size=step_size,
    )

    return traditional_timing


def get_traditional_total_seconds(traditional_timing: Any) -> float:
    """
    兼容 dict 或 TimingResult 对象。
    """
    if isinstance(traditional_timing, dict):
        if "traditional_total_seconds" in traditional_timing:
            return float(traditional_timing["traditional_total_seconds"])
        if "total_seconds" in traditional_timing:
            return float(traditional_timing["total_seconds"])

    if hasattr(traditional_timing, "total_seconds"):
        return float(traditional_timing.total_seconds)

    raise TypeError(
        "无法从 traditional_timing 中读取 total_seconds，"
        f"当前类型为 {type(traditional_timing)}"
    )


def build_timing_comparison_2d(
    model_timing: dict[str, Any],
    traditional_timing: Any,
) -> dict[str, Any]:
    """
    生成二维模型与传统积分的时间对比结果。
    """
    model_total = float(model_timing["model_total_seconds"])
    traditional_total = get_traditional_total_seconds(traditional_timing)

    num_samples = int(model_timing["num_samples"])

    model_avg = model_total / num_samples if num_samples > 0 else 0.0
    traditional_avg = traditional_total / num_samples if num_samples > 0 else 0.0

    speedup_total = traditional_total / model_total if model_total > 0 else float("inf")
    speedup_per_sample = traditional_avg / model_avg if model_avg > 0 else float("inf")

    return {
        "model_total_seconds": float(model_total),
        "traditional_total_seconds": float(traditional_total),
        "model_avg_seconds_per_sample": float(model_avg),
        "traditional_avg_seconds_per_sample": float(traditional_avg),
        "speedup_total": float(speedup_total),
        "speedup_per_sample": float(speedup_per_sample),
        "num_samples": int(num_samples),
        "num_fields": int(model_timing["num_fields"]),
    }


# ==========================================================
# 九、命令行
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """
    构造命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="Run inference and analysis for 2D Kerr operator model."
    )

    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="任务名，例如 vary_Q__Q1.6_3__n2000__T1200__cfg1。",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "二维模型名，例如 "
            "fno2d_m1_16_m2_32_w64_d4_norm-standard_target-residual_initial_ref0。"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备：cuda 或 cpu。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="二维场推理 batch size，第一版通常保持 1。",
    )

    return parser


# ==========================================================
# 十、主流程
# ==========================================================

def main() -> None:
    """
    二维模型推理分析主流程。
    """
    parser = build_parser()
    args = parser.parse_args()

    task_name = str(args.task_name)
    model_name = str(args.model_name)
    device = str(args.device)
    batch_size = int(args.batch_size)

    dirs = get_2d_model_dirs(
        task_name=task_name,
        model_name=model_name,
    )

    # ------------------------------------------------------
    # A. 先加载 checkpoint
    # ------------------------------------------------------
    checkpoint = load_checkpoint_2d(
        checkpoint_path=dirs["checkpoint_path"],
        device=device,
    )

    normalization_method = get_normalization_method_from_checkpoint(checkpoint)
    normalization_stats = load_normalization_stats_from_checkpoint(checkpoint)
    target_transform_method = get_target_transform_method_from_checkpoint(checkpoint)
    target_transform_config = load_target_transform_config_from_checkpoint(checkpoint)

    # ------------------------------------------------------
    # B. 用训练时同样的 normalization / target transform 构造 test loader
    # ------------------------------------------------------
    _, _, test_loader, bundle = build_fno2d_dataloaders(
        task_name=task_name,
        batch_size=batch_size,
        num_workers=0,
        sort_param=True,
        normalization=normalization_method,
        target_transform=target_transform_method,
        lambda_reference_index=target_transform_config.lambda_reference_index,
    )

    bundle_summary = summarize_fno2d_bundle(bundle)

    # ------------------------------------------------------
    # C. 额外加载 raw y_test，供 inverse target transform 使用
    # ------------------------------------------------------
    raw_targets_reference, raw_bundle = load_raw_test_targets_for_inverse(
        task_name=task_name,
        batch_size=batch_size,
    )

    print("=" * 70)
    print("Loaded 2D test dataset summary")
    print("=" * 70)
    print(json.dumps(bundle_summary, indent=4, ensure_ascii=False))

    # ------------------------------------------------------
    # D. 恢复模型
    # ------------------------------------------------------
    model = load_fno2d_checkpoint_model(
        checkpoint=checkpoint,
        device=device,
    )

    # ------------------------------------------------------
    # E. 推理，得到训练目标空间中的预测和标签
    # ------------------------------------------------------
    predictions_model_space, targets_model_space = predict_2d_loader(
        model=model,
        loader=test_loader,
        device=device,
    )

    # ------------------------------------------------------
    # F. 恢复到 raw xyz 物理空间
    # ------------------------------------------------------
    predictions, targets = recover_predictions_and_targets_to_raw_xyz(
        predictions_model_space=predictions_model_space,
        targets_model_space=targets_model_space,
        raw_targets_reference=raw_targets_reference,
        normalization_stats=normalization_stats,
        target_transform_config=target_transform_config,
    )

    # ------------------------------------------------------
    # G. 在 raw xyz 物理空间计算误差
    # ------------------------------------------------------
    metrics = compute_metrics_2d(
        pred=predictions,
        target=targets,
    )

    # ------------------------------------------------------
    # H. 计时：模型推理
    # ------------------------------------------------------
    model_timing = time_model_inference_2d(
        model=model,
        loader=test_loader,
        device=device,
        warmup=True,
    )

    # ------------------------------------------------------
    # I. 计时：传统积分
    # ------------------------------------------------------
    param_values = bundle.test_field.param_grid

    traditional_timing = time_traditional_for_2d_test_field(
        task_name=task_name,
        param_name=bundle.param_name,
        param_values=param_values,
    )

    timing = build_timing_comparison_2d(
        model_timing=model_timing,
        traditional_timing=traditional_timing,
    )

    # ------------------------------------------------------
    # J. 保存结果
    # ------------------------------------------------------
    predictions_path = dirs["inference_dir"] / "predictions.npy"
    predictions_model_space_path = dirs["inference_dir"] / "predictions_model_space.npy"
    targets_path = dirs["inference_dir"] / "targets.npy"
    targets_model_space_path = dirs["inference_dir"] / "targets_model_space.npy"
    raw_targets_reference_path = dirs["inference_dir"] / "targets_raw_reference.npy"
    metrics_path = dirs["inference_dir"] / "metrics.json"
    timing_path = dirs["inference_dir"] / "timing.json"
    summary_path = dirs["analysis_dir"] / "analysis_summary.json"

    save_npy(predictions, predictions_path)
    save_npy(predictions_model_space, predictions_model_space_path)
    save_npy(targets, targets_path)
    save_npy(targets_model_space, targets_model_space_path)
    save_npy(raw_targets_reference, raw_targets_reference_path)
    save_json(metrics, metrics_path)
    save_json(timing, timing_path)

    analysis_summary = {
        "task_name": task_name,
        "model_name": model_name,
        "model_type": checkpoint["config"].get("model_type", "unknown"),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "normalization_method": normalization_method,
        "normalization": normalization_stats.to_dict(),
        "target_transform_method": target_transform_method,
        "target_transform": target_transform_config.to_dict(),
        "dataset_summary": bundle_summary,
        "raw_dataset_summary": summarize_fno2d_bundle(raw_bundle),
        "metrics": metrics,
        "timing": timing,
        "saved_files": {
            "predictions": str(predictions_path),
            "predictions_model_space": str(predictions_model_space_path),
            "targets": str(targets_path),
            "targets_model_space": str(targets_model_space_path),
            "raw_targets_reference": str(raw_targets_reference_path),
            "metrics": str(metrics_path),
            "timing": str(timing_path),
            "analysis_summary": str(summary_path),
        },
    }

    save_json(analysis_summary, summary_path)

    # ------------------------------------------------------
    # K. 打印摘要
    # ------------------------------------------------------
    print("-" * 70)
    print("2D inference analysis finished")
    print(f"Task name          : {task_name}")
    print(f"Model name         : {model_name}")
    print(f"Normalization      : {normalization_method}")
    print(f"Target transform   : {target_transform_method}")
    print(f"Test MSE           : {metrics['mse']:.6e}")
    print(f"Test Relative L2   : {metrics['relative_l2']:.6e}")
    print(f"Model total time   : {timing['model_total_seconds']:.6e} s")
    print(f"Traditional time   : {timing['traditional_total_seconds']:.6e} s")
    print(f"Speedup            : {timing['speedup_total']:.3f} x")
    print(f"Output dir         : {dirs['model_dir']}")
    print("-" * 70)


if __name__ == "__main__":
    main()