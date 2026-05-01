# ==========================================================
# File: src/inference_analysis/plotting.py
#
# 功能简介：
# 1. 绘制单模型推理分析图；
# 2. 当前主要提供：
#    - 3D 轨道对比图
#    - x/y/z 分量曲线对比图
#    - 单样本误差曲线图
#    - 多样本误差分布图
# 3. 先服务于单模型分析；
# 4. 多模型对比图后续放在 comparison.py 中处理。
#
# 依赖关系：
# - 依赖 src/inference_analysis/inference.py
# - 依赖 src/inference_analysis/metrics.py
# - 被 scripts/run_analysis.py 调用
#
# 重要说明：
# - 本文件只负责画图；
# - 不负责推理；
# - 不负责保存 JSON；
# - 图像保存采用统一路径写入，默认覆盖同名文件。
# ==========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common.io_utils import ensure_parent_dir
from src.inference_analysis.inference import InferenceResult
from src.inference_analysis.metrics import per_sample_mse, per_sample_relative_l2


# ==========================================================
# 一、基础工具
# ==========================================================

def _save_figure(fig: plt.Figure, save_path: str | Path) -> str:
    """
    保存图像到指定路径。

    说明：
    - 使用统一路径保存；
    - 若同名文件已存在，则默认覆盖。
    """
    save_path = ensure_parent_dir(save_path)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(save_path)


def _validate_sample_index(result: InferenceResult, sample_index: int) -> None:
    """
    检查 sample_index 是否越界。
    """
    if not (0 <= sample_index < result.num_samples):
        raise IndexError(
            f"sample_index 越界：sample_index={sample_index}, num_samples={result.num_samples}"
        )


def _build_pointwise_l2_error_curve(pred_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """
    构造单样本逐步点误差曲线。

    输入：
    - pred_xyz   : [T, 3]
    - target_xyz : [T, 3]

    返回：
    - error_curve: [T]
      每个时间步/轨道步上的三维欧氏误差
    """
    if pred_xyz.shape != target_xyz.shape:
        raise ValueError(
            f"pred_xyz 与 target_xyz 形状必须一致，当前得到：{pred_xyz.shape} vs {target_xyz.shape}"
        )

    return np.linalg.norm(pred_xyz - target_xyz, ord=2, axis=1)


# ==========================================================
# 二、单样本图
# ==========================================================

def plot_single_sample_3d_trajectory(
    result: InferenceResult,
    sample_index: int,
    save_path: str | Path,
    title: str | None = None,
) -> str:
    """
    绘制单样本 3D 轨道对比图。

    内容：
    - Ground Truth 轨道
    - Prediction 轨道
    """
    _validate_sample_index(result, sample_index)

    pred = result.predictions[sample_index]   # [T,3]
    target = result.targets[sample_index]     # [T,3]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(target[:, 0], target[:, 1], target[:, 2], label="Ground Truth")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="Prediction")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    if title is None:
        title = f"3D Trajectory Comparison (sample {sample_index})"
    ax.set_title(title)

    return _save_figure(fig, save_path)


def plot_single_sample_xyz_curves(
    result: InferenceResult,
    sample_index: int,
    save_path: str | Path,
    lambda_grid: np.ndarray | None = None,
    title: str | None = None,
) -> str:
    """
    绘制单样本 x/y/z 三个分量的真实值与预测值曲线。

    说明：
    - 若提供 lambda_grid，则横轴使用 lambda；
    - 否则使用离散索引 0,1,...,T-1。
    """
    _validate_sample_index(result, sample_index)

    pred = result.predictions[sample_index]   # [T,3]
    target = result.targets[sample_index]     # [T,3]
    T = pred.shape[0]

    if lambda_grid is None:
        x_axis = np.arange(T)
        x_label = "step index"
    else:
        if lambda_grid.shape[0] != T:
            raise ValueError(
                f"lambda_grid 长度必须等于轨道长度 T，当前得到：len(lambda_grid)={lambda_grid.shape[0]}, T={T}"
            )
        x_axis = lambda_grid
        x_label = "lambda"

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(x_axis, target[:, i], label="Ground Truth")
        ax.plot(x_axis, pred[:, i], label="Prediction")
        ax.set_ylabel(labels[i])
        ax.legend()

    axes[-1].set_xlabel(x_label)

    if title is None:
        title = f"XYZ Component Curves (sample {sample_index})"
    fig.suptitle(title)

    return _save_figure(fig, save_path)


def plot_single_sample_error_curve(
    result: InferenceResult,
    sample_index: int,
    save_path: str | Path,
    lambda_grid: np.ndarray | None = None,
    title: str | None = None,
) -> str:
    """
    绘制单样本逐点误差曲线。

    误差定义：
    - 每个步点上的 3D 欧氏距离误差
    """
    _validate_sample_index(result, sample_index)

    pred = result.predictions[sample_index]
    target = result.targets[sample_index]
    error_curve = _build_pointwise_l2_error_curve(pred, target)
    T = error_curve.shape[0]

    if lambda_grid is None:
        x_axis = np.arange(T)
        x_label = "step index"
    else:
        if lambda_grid.shape[0] != T:
            raise ValueError(
                f"lambda_grid 长度必须等于轨道长度 T，当前得到：len(lambda_grid)={lambda_grid.shape[0]}, T={T}"
            )
        x_axis = lambda_grid
        x_label = "lambda"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(x_axis, error_curve)
    ax.set_xlabel(x_label)
    ax.set_ylabel("pointwise L2 error")

    if title is None:
        title = f"Pointwise Error Curve (sample {sample_index})"
    ax.set_title(title)

    return _save_figure(fig, save_path)


# ==========================================================
# 三、多样本统计图
# ==========================================================

def plot_error_histogram(
    result: InferenceResult,
    save_path: str | Path,
    metric_name: str = "relative_l2",
    bins: int = 30,
    title: str | None = None,
) -> str:
    """
    绘制多样本误差分布直方图。

    参数：
    - metric_name: "relative_l2" 或 "mse"
    """
    if metric_name == "relative_l2":
        values = per_sample_relative_l2(result.predictions, result.targets)
    elif metric_name == "mse":
        values = per_sample_mse(result.predictions, result.targets)
    else:
        raise ValueError(f"不支持的 metric_name={metric_name!r}，当前仅支持 'relative_l2' / 'mse'")

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    ax.hist(values, bins=bins)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("count")

    if title is None:
        title = f"{metric_name} distribution"
    ax.set_title(title)

    return _save_figure(fig, save_path)


def plot_sample_error_curve_summary(
    result: InferenceResult,
    save_path: str | Path,
    metric_name: str = "relative_l2",
    title: str | None = None,
) -> str:
    """
    绘制“每个样本一个误差值”的摘要图。

    作用：
    - 看不同样本之间误差起伏情况
    """
    if metric_name == "relative_l2":
        values = per_sample_relative_l2(result.predictions, result.targets)
    elif metric_name == "mse":
        values = per_sample_mse(result.predictions, result.targets)
    else:
        raise ValueError(f"不支持的 metric_name={metric_name!r}，当前仅支持 'relative_l2' / 'mse'")

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(np.arange(values.shape[0]), values)
    ax.set_xlabel("sample index")
    ax.set_ylabel(metric_name)

    if title is None:
        title = f"Per-sample {metric_name}"
    ax.set_title(title)

    return _save_figure(fig, save_path)


# ==========================================================
# 四、单模型标准分析图打包
# ==========================================================

def generate_standard_single_model_plots(
    result: InferenceResult,
    output_dir: str | Path,
    lambda_grid: np.ndarray | None = None,
    representative_sample_index: int | None = None,
) -> dict[str, str]:
    """
    为单模型分析生成一组标准图。

    图像包括：
    - 3D 轨道对比图
    - xyz 曲线图
    - 单样本误差曲线图
    - Relative L2 分布图
    - MSE 分布图
    - 每样本 Relative L2 摘要图

    参数：
    - representative_sample_index:
        若为 None，则默认选第 0 个样本
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if representative_sample_index is None:
        representative_sample_index = 0

    paths = {}

    paths["trajectory_3d"] = plot_single_sample_3d_trajectory(
        result=result,
        sample_index=representative_sample_index,
        save_path=output_dir / "sample_trajectory_3d.png",
    )

    paths["xyz_curves"] = plot_single_sample_xyz_curves(
        result=result,
        sample_index=representative_sample_index,
        save_path=output_dir / "sample_xyz_curves.png",
        lambda_grid=lambda_grid,
    )

    paths["error_curve"] = plot_single_sample_error_curve(
        result=result,
        sample_index=representative_sample_index,
        save_path=output_dir / "sample_error_curve.png",
        lambda_grid=lambda_grid,
    )

    paths["relative_l2_hist"] = plot_error_histogram(
        result=result,
        save_path=output_dir / "relative_l2_hist.png",
        metric_name="relative_l2",
    )

    paths["mse_hist"] = plot_error_histogram(
        result=result,
        save_path=output_dir / "mse_hist.png",
        metric_name="mse",
    )

    paths["relative_l2_per_sample"] = plot_sample_error_curve_summary(
        result=result,
        save_path=output_dir / "relative_l2_per_sample.png",
        metric_name="relative_l2",
    )

    return paths