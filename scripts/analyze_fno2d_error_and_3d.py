#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fno2d_error_and_3d.py
======================================================================
对一个 Q-only 2D FNO 实验进行完整误差剖析和交互式三维轨道对比。

本脚本回答的问题
----------------
1. 整体 Relative L2 的平均值掩盖了怎样的逐轨道分布？
2. 模型误差是否随 Q 发生系统变化？
3. 最差轨道是整体形状错误，还是后半段相位逐渐漂移？
4. 平均点距离、最大点距离和终点距离分别有多大？
5. 最好、中位、最差以及低/中/高 Q 轨道的三维形态是否一致？

重要数据顺序
------------
2D 推理数组使用 sort_param=True，因此：
    predictions[0, output_position]
    targets_raw_reference[0, output_position]
按 Q 升序排列。

而 dataset.npz 中：
    x_test[test_index]
    y_test[test_index]
保持原测试集顺序。

本脚本会显式构造：
    sorted_test_indices[output_position] = test_index
避免再次发生索引错位。

主要输出
--------
outputs/_error_profile_2d/<任务名>/<模型名>/
    trajectory_metrics.csv
    error_summary.json
    error_report_zh.md
    error_vs_Q.png
    error_quantiles.png
    pointwise_error_selected.png
    interactive_3d_comparison.html
    selected_trajectories.json

交互式 HTML
-----------
可以旋转、缩放三维轨道，并通过下拉菜单切换代表性轨道。
每个案例同时显示：
    数值目标轨道
    FNO 预测轨道
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SelectedTrajectory:
    role: str
    output_position: int
    test_index: int
    Q: float
    relative_l2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 2D FNO 的逐轨道误差和三维预测形态。"
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--label",
        default=None,
        help="报告中使用的简短实验名称，例如 n500 或 n2000。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--html-max-points",
        type=int,
        default=1200,
        help="交互式 HTML 中每条轨道最多保留的点数。",
    )
    return parser.parse_args()


def trajectory_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """
    对单条 [T,3] 轨道计算物理空间指标。

    Relative L2:
        ||prediction-target||_2 / ||target||_2

    pointwise_distance:
        每个 lambda 网格点上的三维欧氏距离。
    """
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)

    difference = prediction - target
    target_norm = np.linalg.norm(target.reshape(-1))
    if target_norm <= 0.0:
        raise ValueError("目标轨道范数为零。")

    pointwise = np.linalg.norm(difference, axis=-1)

    return {
        "relative_l2": float(
            np.linalg.norm(difference.reshape(-1)) / target_norm
        ),
        "mse": float(np.mean(difference**2)),
        "mean_pointwise_distance": float(np.mean(pointwise)),
        "median_pointwise_distance": float(np.median(pointwise)),
        "max_pointwise_distance": float(np.max(pointwise)),
        "final_point_distance": float(pointwise[-1]),
        "first_half_mean_distance": float(
            np.mean(pointwise[: len(pointwise) // 2])
        ),
        "second_half_mean_distance": float(
            np.mean(pointwise[len(pointwise) // 2 :])
        ),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "最小值": float(np.min(values)),
        "第25百分位": float(np.percentile(values, 25)),
        "中位数": float(np.median(values)),
        "平均值": float(np.mean(values)),
        "第75百分位": float(np.percentile(values, 75)),
        "第90百分位": float(np.percentile(values, 90)),
        "第95百分位": float(np.percentile(values, 95)),
        "最大值": float(np.max(values)),
    }


def nearest_index(values: np.ndarray, target_value: float) -> int:
    return int(np.argmin(np.abs(values - target_value)))


def choose_representatives(
    Q_sorted: np.ndarray,
    relative_l2: np.ndarray,
    sorted_test_indices: np.ndarray,
) -> list[SelectedTrajectory]:
    """
    选择：
    - 最好轨道；
    - 中位误差轨道；
    - 最差轨道；
    - 低、中、高 Q 轨道。

    若角色落在同一轨道上，保留一个轨道但合并角色名称。
    """
    candidate_roles: list[tuple[str, int]] = [
        ("Best error", int(np.argmin(relative_l2))),
        (
            "Median error",
            int(
                np.argmin(
                    np.abs(relative_l2 - np.median(relative_l2))
                )
            ),
        ),
        ("Worst error", int(np.argmax(relative_l2))),
        ("Low Q", nearest_index(Q_sorted, np.percentile(Q_sorted, 10))),
        ("Mid Q", nearest_index(Q_sorted, np.percentile(Q_sorted, 50))),
        ("High Q", nearest_index(Q_sorted, np.percentile(Q_sorted, 90))),
    ]

    merged: dict[int, list[str]] = {}
    for role, output_position in candidate_roles:
        merged.setdefault(output_position, []).append(role)

    result: list[SelectedTrajectory] = []
    for output_position, roles in merged.items():
        result.append(
            SelectedTrajectory(
                role=" / ".join(roles),
                output_position=int(output_position),
                test_index=int(sorted_test_indices[output_position]),
                Q=float(Q_sorted[output_position]),
                relative_l2=float(relative_l2[output_position]),
            )
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_interactive_html(
    output_path: Path,
    selected: list[SelectedTrajectory],
    predictions: np.ndarray,
    targets: np.ndarray,
    max_points: int,
    label: str,
    task_name: str | None = None,
    model_name: str | None = None,
    parameter_name: str = "Q",
    parameter_range_text: str | None = None,
    experiment_config: dict[str, Any] | None = None,
) -> None:
    """
    使用 Plotly 生成一个 HTML 文件。

    设计目标：
    1. 保留单图 + 下拉菜单切换；
    2. 每次只展示一条轨道的 truth / prediction；
    3. 在图中显式展示任务名、模型名、参数信息与当前轨道信息；
    4. 图例和可见文字尽量使用英文，便于汇报展示。

    Plotly 只在生成 HTML 时使用；HTML 本身可独立打开。
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required. Please run: pip install plotly"
        ) from exc

    if len(selected) == 0:
        raise ValueError("selected must contain at least one trajectory.")

    figure = go.Figure()
    buttons = []

    def build_title(case: SelectedTrajectory) -> str:
        return (
            f"{label} | {case.role} | "
            f"{parameter_name}={case.Q:.8f} | "
            f"Relative L2={case.relative_l2:.6e}"
        )

    def build_meta_text(case: SelectedTrajectory) -> str:
        lines = []

        if task_name:
            lines.append(f"<b>Task:</b> {task_name}")
        if model_name:
            lines.append(f"<b>Model:</b> {model_name}")

        config = experiment_config or {}

        if config:
            lines.append(
                "<b>Architecture:</b> "
                f"modes={config.get('modes_param', 'N/A')}"
                f"x{config.get('modes_lambda', 'N/A')}, "
                f"width={config.get('width', 'N/A')}, "
                f"depth={config.get('depth', 'N/A')}, "
                f"hidden_dim={config.get('hidden_dim', 'N/A')}"
            )
            lines.append(
                "<b>Training:</b> "
                f"epochs={config.get('epochs', 'N/A')}, "
                f"best_epoch={config.get('best_epoch', 'N/A')}, "
                f"normalization={config.get('normalization', 'N/A')}, "
                f"target={config.get('target_transform', 'N/A')}"
            )
            lines.append(
                "<b>Dataset:</b> "
                f"total={config.get('total_samples', 'N/A')}, "
                f"train/val/test="
                f"{config.get('train_samples', 'N/A')}/"
                f"{config.get('val_samples', 'N/A')}/"
                f"{config.get('test_samples', 'N/A')}"
            )
            lines.append(
                "<b>Operator axes:</b> "
                f"{config.get('operator_axes', 'N/A')} | "
                f"varying parameter={parameter_name} "
                f"{parameter_range_text or 'N/A'} | "
                f"lambda={config.get('lambda_range', 'N/A')} "
                f"({config.get('lambda_points', 'N/A')} points)"
            )
            lines.append(
                "<b>Seeds:</b> "
                f"data={config.get('data_seed', 'N/A')}, "
                f"training={config.get('training_seed', 'N/A')}"
            )

        lines.append(
            f"<b>Selected trajectory:</b> {case.role} | "
            f"{parameter_name}={case.Q:.8f} | "
            f"Relative L2={case.relative_l2:.6e}"
        )

        return "<br>".join(lines)

    for case_index, case in enumerate(selected):
        target = targets[case.output_position]
        prediction = predictions[case.output_position]

        stride = max(1, int(math.ceil(len(target) / max_points)))
        target_plot = target[::stride]
        prediction_plot = prediction[::stride]

        visible = case_index == 0

        figure.add_trace(
            go.Scatter3d(
                x=target_plot[:, 0],
                y=target_plot[:, 1],
                z=target_plot[:, 2],
                mode="lines",
                name="Ground Truth",
                visible=visible,
                line={"width": 6, "color": "blue"},
                hovertemplate=(
                    "Ground Truth<br>x=%{x:.5f}<br>y=%{y:.5f}<br>"
                    "z=%{z:.5f}<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=prediction_plot[:, 0],
                y=prediction_plot[:, 1],
                z=prediction_plot[:, 2],
                mode="lines",
                name="FNO Prediction",
                visible=visible,
                line={"width": 5, "color": "red"},
                hovertemplate=(
                    "Prediction<br>x=%{x:.5f}<br>y=%{y:.5f}<br>"
                    "z=%{z:.5f}<extra></extra>"
                ),
            )
        )

        visibility = [False] * (2 * len(selected))
        visibility[2 * case_index] = True
        visibility[2 * case_index + 1] = True

        buttons.append(
            {
                "label": (
                    f"{case.role} | "
                    f"{parameter_name}={case.Q:.4f} | "
                    f"RelL2={case.relative_l2:.3e}"
                ),
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": {"text": build_title(case)},
                        "annotations": [
                            {
                                "text": build_meta_text(case),
                                "xref": "paper",
                                "yref": "paper",
                                "x": 0.55,
                                "y": 0.98,
                                "xanchor": "left",
                                "yanchor": "top",
                                "showarrow": False,
                                "align": "left",
                                "font": {"size": 14},
                                "bordercolor": "black",
                                "borderwidth": 1,
                                "borderpad": 6,
                                "bgcolor": "white",
                            }
                        ],
                    },
                ],
            }
        )

    first = selected[0]
    figure.update_layout(
        title={"text": build_title(first)},
        scene={
            "domain": {
                "x": [0.0, 0.67],
                "y": [0.0, 0.94],
            },
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.01,
                "xanchor": "left",
                "yanchor": "bottom",
            }
        ],
        annotations=[
            {
                "text": build_meta_text(first),
                "xref": "paper",
                "yref": "paper",
                "x": 0.55,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "font": {"size": 14},
                "bordercolor": "black",
                "borderwidth": 1,
                "borderpad": 6,
                "bgcolor": "white",
            }
        ],
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 20, "r": 30, "t": 110, "b": 20},
    )

    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )


def embed_png_sections_in_html(
    html_path: Path,
    image_specs: list[tuple[str, Path]],
) -> None:
    """
    将误差 PNG 以 Base64 形式嵌入 Plotly HTML。

    原始 PNG 文件仍然保留，便于单独下载和插入幻灯片。
    """
    import base64

    html = html_path.read_text(encoding="utf-8")

    sections: list[str] = [
        """
<section style="
    max-width: 1200px;
    margin: 32px auto;
    padding: 0 20px 40px 20px;
    font-family: Arial, sans-serif;
">
<h2 style="margin-bottom: 8px;">Error Analysis</h2>
<p style="color: #555; margin-top: 0;">
Static figures are embedded for convenient presentation.
The original PNG files are also preserved separately.
</p>
"""
    ]

    for title, image_path in image_specs:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for HTML embedding: {image_path}"
            )

        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        sections.append(
            f"""
<div style="margin-top: 28px;">
<h3>{title}</h3>
<img
    src="data:image/png;base64,{encoded}"
    alt="{title}"
    style="
        display: block;
        width: 100%;
        height: auto;
        border: 1px solid #ddd;
    "
>
</div>
"""
        )

    sections.append("</section>")

    embedded_section = "\n".join(sections)

    if "</body>" not in html:
        raise RuntimeError(
            f"HTML closing body tag not found: {html_path}"
        )

    html = html.replace(
        "</body>",
        embedded_section + "\n</body>",
        1,
    )

    html_path.write_text(
        html,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    label = args.label or args.task_name

    task_dir = PROJECT_ROOT / "data" / "tasks" / args.task_name
    inference_dir = (
        PROJECT_ROOT
        / "outputs"
        / args.task_name
        / args.model_name
        / "inference"
    )

    dataset_path = task_dir / "dataset.npz"
    predictions_path = inference_dir / "predictions.npy"
    targets_path = inference_dir / "targets_raw_reference.npy"
    metrics_path = inference_dir / "metrics.json"
    run_config_path = (
        PROJECT_ROOT
        / "outputs"
        / args.task_name
        / args.model_name
        / "run_config.json"
    )

    for path in [
        dataset_path,
        predictions_path,
        targets_path,
        metrics_path,
        run_config_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        x_test = np.asarray(data["x_test"], dtype=np.float64)
        y_test = np.asarray(data["y_test"], dtype=np.float64)
        lambda_grid = np.asarray(data["lambda_grid"], dtype=np.float64)

    predictions_raw = np.load(predictions_path)
    targets_raw = np.load(targets_path)

    if predictions_raw.shape[0] != 1 or targets_raw.shape[0] != 1:
        raise ValueError("当前脚本要求数组形状为 [1,N,T,3]。")

    predictions = np.asarray(predictions_raw[0], dtype=np.float64)
    targets = np.asarray(targets_raw[0], dtype=np.float64)

    sorted_test_indices = np.argsort(x_test[:, 0], kind="stable")
    Q_sorted = x_test[sorted_test_indices, 0]
    y_test_sorted = y_test[sorted_test_indices]

    target_alignment_error = float(
        np.max(np.abs(targets - y_test_sorted))
    )
    if target_alignment_error > 1e-5:
        raise RuntimeError(
            "推理目标与按 Q 排序后的 y_test 仍不一致："
            f"{target_alignment_error:.6e}"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / "outputs"
        / "_error_profile_2d"
        / args.task_name
        / args.model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    all_pointwise: list[np.ndarray] = []

    for output_position in range(len(Q_sorted)):
        prediction = predictions[output_position]
        target = targets[output_position]
        metrics = trajectory_metrics(prediction, target)
        pointwise = np.linalg.norm(prediction - target, axis=-1)
        all_pointwise.append(pointwise)

        rows.append(
            {
                "output_position": output_position,
                "test_index": int(sorted_test_indices[output_position]),
                "Q": float(Q_sorted[output_position]),
                **metrics,
                "second_half_to_first_half_ratio": float(
                    metrics["second_half_mean_distance"]
                    / max(metrics["first_half_mean_distance"], 1e-15)
                ),
            }
        )

    relative_l2 = np.array(
        [row["relative_l2"] for row in rows],
        dtype=np.float64,
    )
    mean_distance = np.array(
        [row["mean_pointwise_distance"] for row in rows],
        dtype=np.float64,
    )
    max_distance = np.array(
        [row["max_pointwise_distance"] for row in rows],
        dtype=np.float64,
    )
    final_distance = np.array(
        [row["final_point_distance"] for row in rows],
        dtype=np.float64,
    )
    phase_growth_ratio = np.array(
        [row["second_half_to_first_half_ratio"] for row in rows],
        dtype=np.float64,
    )

    selected = choose_representatives(
        Q_sorted=Q_sorted,
        relative_l2=relative_l2,
        sorted_test_indices=sorted_test_indices,
    )

    write_csv(output_dir / "trajectory_metrics.csv", rows)
    (output_dir / "selected_trajectories.json").write_text(
        json.dumps(
            [case.__dict__ for case in selected],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    correlation_Q_error = float(
        np.corrcoef(Q_sorted, relative_l2)[0, 1]
    )
    correlation_Q_max_distance = float(
        np.corrcoef(Q_sorted, max_distance)[0, 1]
    )

    summary = {
        "任务名称": args.task_name,
        "模型名称": args.model_name,
        "实验标签": label,
        "测试轨道数": int(len(Q_sorted)),
        "推理目标排序一致性最大绝对差": target_alignment_error,
        "inference_metrics": saved_metrics,
        "trajectory_relative_l2": distribution(relative_l2),
        "mean_pointwise_distance": distribution(mean_distance),
        "max_pointwise_distance": distribution(max_distance),
        "final_point_distance": distribution(final_distance),
        "后半段误差与前半段误差比值": distribution(
            phase_growth_ratio
        ),
        "Q与Relative_L2相关系数": correlation_Q_error,
        "Q与最大点距离相关系数": correlation_Q_max_distance,
        "代表轨道": [case.__dict__ for case in selected],
    }
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 误差随 Q 的变化
    figure = plt.figure(figsize=(9, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter(Q_sorted, relative_l2, s=18)
    axis.set_xlabel("Q")
    axis.set_ylabel("Trajectory Relative L2")
    axis.set_title(f"{label}: Error versus Q")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_dir / "error_vs_Q.png", dpi=200)
    plt.close(figure)

    # 误差分位数
    quantile_levels = np.array([0, 25, 50, 75, 90, 95, 100])
    quantile_values = np.percentile(relative_l2, quantile_levels)

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(quantile_levels, quantile_values, marker="o")
    axis.set_xlabel("Percentile")
    axis.set_ylabel("Trajectory Relative L2")
    axis.set_title(f"{label}: Trajectory Error Quantiles")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_dir / "error_quantiles.png", dpi=200)
    plt.close(figure)

    # 代表轨道逐点误差
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    for case in selected:
        pointwise = all_pointwise[case.output_position]
        axis.plot(
            lambda_grid,
            pointwise,
            label=(
                f"{case.role}, Q={case.Q:.4f}, "
                f"RelL2={case.relative_l2:.3e}"
            ),
        )
    axis.set_xlabel("Mino parameter lambda")
    axis.set_ylabel("Pointwise 3D Distance")
    axis.set_title(f"{label}: Pointwise Error of Selected Trajectories")
    axis.grid(True)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / "pointwise_error_selected.png",
        dpi=200,
    )
    plt.close(figure)

    run_config = json.loads(
        run_config_path.read_text(encoding="utf-8")
    )
    dataset_summary = run_config.get("dataset_summary", {})
    test_summary = dataset_summary.get("test", {})

    parameter_name = dataset_summary.get(
        "param_name",
        "Q",
    )
    parameter_min = test_summary.get("param_min")
    parameter_max = test_summary.get("param_max")

    parameter_range_text = None
    if parameter_min is not None and parameter_max is not None:
        parameter_range_text = (
            f"[{float(parameter_min):.6f}, "
            f"{float(parameter_max):.6f}]"
        )

    model_config = run_config.get("model_config", {})
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / args.task_name
        / args.model_name
        / "summary.json"
    )
    run_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    training_summary = run_summary.get("training", {})
    train_summary = dataset_summary.get("train", {})
    val_summary = dataset_summary.get("val", {})

    data_seed = run_config.get(
        "data_seeds",
        {},
    ).get(args.task_name)

    experiment_config = {
        "modes_param": model_config.get("modes1"),
        "modes_lambda": model_config.get("modes2"),
        "width": model_config.get("width"),
        "depth": model_config.get("depth"),
        "hidden_dim": model_config.get("hidden_dim"),
        "epochs": run_config.get("epochs"),
        "best_epoch": training_summary.get("best_epoch"),
        "normalization": run_config.get("normalization"),
        "target_transform": run_config.get("target_transform"),
        "total_samples": (
            int(train_summary.get("num_param", 0))
            + int(val_summary.get("num_param", 0))
            + int(test_summary.get("num_param", 0))
        ),
        "train_samples": train_summary.get("num_param"),
        "val_samples": val_summary.get("num_param"),
        "test_samples": test_summary.get("num_param"),
        "operator_axes": " x ".join(
            run_config.get("operator_axes", [])
        ),
        "lambda_points": test_summary.get("num_lambda"),
        "lambda_range": (
            f"[{float(test_summary['lambda_min']):.6f}, "
            f"{float(test_summary['lambda_max']):.6f}]"
            if (
                test_summary.get("lambda_min") is not None
                and test_summary.get("lambda_max") is not None
            )
            else "N/A"
        ),
        "data_seed": data_seed,
        "training_seed": run_config.get("training_seed"),
    }

    build_interactive_html(
        output_path=output_dir / "interactive_3d_comparison.html",
        selected=selected,
        predictions=predictions,
        targets=targets,
        max_points=int(args.html_max_points),
        label=label,
        task_name=args.task_name,
        model_name=args.model_name,
        parameter_name=str(parameter_name),
        parameter_range_text=parameter_range_text,
        experiment_config=experiment_config,
    )

    embed_png_sections_in_html(
        html_path=output_dir / "interactive_3d_comparison.html",
        image_specs=[
            (
                "Trajectory Error versus Q",
                output_dir / "error_vs_Q.png",
            ),
            (
                "Trajectory Error Quantiles",
                output_dir / "error_quantiles.png",
            ),
            (
                "Pointwise Error of Selected Trajectories",
                output_dir / "pointwise_error_selected.png",
            ),
        ],
    )

    report_lines = [
        f"# {label} 误差剖析与三维轨道对比",
        "",
        f"- 任务：`{args.task_name}`",
        f"- 模型：`{args.model_name}`",
        f"- 测试轨道数：{len(Q_sorted)}",
        f"- 最终统一 Relative L2："
        f"{saved_metrics.get('relative_l2', float('nan')):.6e}",
        "",
        "## 逐轨道 Relative L2 分布",
        "",
    ]
    for key, value in summary["trajectory_relative_l2"].items():
        report_lines.append(f"- {key}：{value:.6e}")

    report_lines.extend(
        [
            "",
            "## 逐点空间误差",
            "",
            f"- 最大点距离中位数："
            f"{summary['max_pointwise_distance']['中位数']:.6e}",
            f"- 最大点距离第95百分位："
            f"{summary['max_pointwise_distance']['第95百分位']:.6e}",
            f"- 终点距离中位数："
            f"{summary['final_point_distance']['中位数']:.6e}",
            "",
            "## 误差随轨道演化的变化",
            "",
            "- 后半段平均误差 / 前半段平均误差的中位数："
            f"{summary['后半段误差与前半段误差比值']['中位数']:.6f}",
            "- 若该比值明显大于 1，说明误差可能随轨道演化累积，"
            "需要进一步检查相位漂移。",
            "",
            "## 参数关系",
            "",
            f"- Q 与逐轨道 Relative L2 的相关系数："
            f"{correlation_Q_error:.6f}",
            f"- Q 与最大点距离的相关系数："
            f"{correlation_Q_max_distance:.6f}",
            "",
            "## 代表轨道",
            "",
        ]
    )
    for case in selected:
        report_lines.append(
            f"- {case.role}：test_index={case.test_index}，"
            f"Q={case.Q:.8f}，Relative L2={case.relative_l2:.6e}"
        )

    report_lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `error_vs_Q.png`：检查误差是否随 Q 系统变化。",
            "- `error_quantiles.png`：检查平均误差是否掩盖长尾。",
            "- `pointwise_error_selected.png`：检查误差是否随 λ 累积。",
            "- `interactive_3d_comparison.html`：交互式旋转、缩放并切换"
            "代表轨道。",
            "",
            "## 解释限制",
            "",
            "Relative L2 只反映整条轨道的总体误差。是否适合高精度物理"
            "用途，还需要结合最大点距离、终点误差、相位漂移和转向位置"
            "进行判断。",
            "",
        ]
    )
    (output_dir / "error_report_zh.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 88)
    print("误差剖析与三维轨道对比完成")
    print(f"任务：{args.task_name}")
    print(f"模型：{args.model_name}")
    print(
        "逐轨道 Relative L2："
        f"中位数={np.median(relative_l2):.6e}，"
        f"第95百分位={np.percentile(relative_l2,95):.6e}，"
        f"最大值={np.max(relative_l2):.6e}"
    )
    print(
        "后半段/前半段误差比值中位数："
        f"{np.median(phase_growth_ratio):.6f}"
    )
    print(f"输出目录：{output_dir}")
    print(
        "交互式三维文件："
        f"{output_dir / 'interactive_3d_comparison.html'}"
    )


if __name__ == "__main__":
    main()
