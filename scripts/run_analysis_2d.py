# ==========================================================
# File: scripts/run_analysis_2d.py
#
# 功能简介：
# 1. 二维模型的独立推理分析入口；
# 2. 不修改原来的 scripts/run_analysis.py；
# 3. 支持通过 registry_2d.py 恢复二维模型；
# 4. 当前主要用于单参数二维算子任务：
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

    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    diff_norm = np.linalg.norm(pred_flat - target_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    rel = diff_norm / (target_norm + eps)

    return float(np.mean(rel))


def compute_metrics_2d(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """
    计算二维场推理指标。
    """
    return {
        "mse": compute_mse_np(pred, target),
        "relative_l2": compute_relative_l2_np(pred, target),
        "pred_shape": list(pred.shape),
        "target_shape": list(target.shape),
    }


# ==========================================================
# 四、模型加载
# ==========================================================

def load_fno2d_checkpoint_model(
    checkpoint_path: Path,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    从 best_model.pt 恢复二维模型。

    checkpoint 中需要包含：
    - model_state_dict
    - config
    - config["model_config"]
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return model, checkpoint


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
# 六、传统数值计算计时
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
) -> dict[str, Any]:
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


def build_timing_comparison_2d(
    model_timing: dict[str, Any],
    traditional_timing: dict[str, Any],
) -> dict[str, Any]:
    """
    生成二维模型与传统积分的时间对比结果。
    """
    model_total = float(model_timing["model_total_seconds"])
    traditional_total = float(traditional_timing["traditional_total_seconds"])

    num_samples = int(model_timing["num_samples"])

    model_avg = (
        model_total / num_samples
        if num_samples > 0
        else 0.0
    )

    traditional_avg = (
        traditional_total / num_samples
        if num_samples > 0
        else 0.0
    )

    speedup_total = (
        traditional_total / model_total
        if model_total > 0
        else float("inf")
    )

    speedup_per_sample = (
        traditional_avg / model_avg
        if model_avg > 0
        else float("inf")
    )

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
# 七、命令行
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
        help="二维模型名，例如 fno2d_m1_16_m2_32_w64_d4。",
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
# 八、主流程
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

    dirs = get_2d_model_dirs(
        task_name=task_name,
        model_name=model_name,
    )

    # ------------------------------------------------------
    # A. 加载数据
    # ------------------------------------------------------
    _, _, test_loader, bundle = build_fno2d_dataloaders(
        task_name=task_name,
        batch_size=int(args.batch_size),
        num_workers=0,
        sort_param=True,
    )

    bundle_summary = summarize_fno2d_bundle(bundle)

    print("=" * 70)
    print("Loaded 2D test dataset summary")
    print("=" * 70)
    print(json.dumps(bundle_summary, indent=4, ensure_ascii=False))

    # ------------------------------------------------------
    # B. 加载模型
    # ------------------------------------------------------
    model, checkpoint = load_fno2d_checkpoint_model(
        checkpoint_path=dirs["checkpoint_path"],
        device=device,
    )

    # ------------------------------------------------------
    # C. 推理并计算误差
    # ------------------------------------------------------
    predictions, targets = predict_2d_loader(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = compute_metrics_2d(
        pred=predictions,
        target=targets,
    )

    # ------------------------------------------------------
    # D. 计时：模型推理
    # ------------------------------------------------------
    model_timing = time_model_inference_2d(
        model=model,
        loader=test_loader,
        device=device,
        warmup=True,
    )

    # ------------------------------------------------------
    # E. 计时：传统积分
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
    # F. 保存结果
    # ------------------------------------------------------
    predictions_path = dirs["inference_dir"] / "predictions.npy"
    metrics_path = dirs["inference_dir"] / "metrics.json"
    timing_path = dirs["inference_dir"] / "timing.json"
    summary_path = dirs["analysis_dir"] / "analysis_summary.json"

    save_npy(predictions, predictions_path)
    save_json(metrics, metrics_path)
    save_json(timing, timing_path)

    analysis_summary = {
        "task_name": task_name,
        "model_name": model_name,
        "model_type": checkpoint["config"].get("model_type", "unknown"),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "dataset_summary": bundle_summary,
        "metrics": metrics,
        "timing": timing,
        "saved_files": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "timing": str(timing_path),
            "analysis_summary": str(summary_path),
        },
    }

    save_json(analysis_summary, summary_path)

    # ------------------------------------------------------
    # G. 打印摘要
    # ------------------------------------------------------
    print("-" * 70)
    print("2D inference analysis finished")
    print(f"Task name          : {task_name}")
    print(f"Model name         : {model_name}")
    print(f"Test MSE           : {metrics['mse']:.6e}")
    print(f"Test Relative L2   : {metrics['relative_l2']:.6e}")
    print(f"Model total time   : {timing['model_total_seconds']:.6e} s")
    print(f"Traditional time   : {timing['traditional_total_seconds']:.6e} s")
    print(f"Speedup            : {timing['speedup_total']:.3f} x")
    print(f"Output dir         : {dirs['model_dir']}")
    print("-" * 70)


if __name__ == "__main__":
    main()