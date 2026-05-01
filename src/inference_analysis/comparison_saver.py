# ==========================================================
# File: src/inference_analysis/comparison_saver.py
#
# 功能简介：
# 1. 专门保存多模型比较结果；
# 2. 保存到任务级 comparison 目录；
# 3. 当前主要保存：
#    - metrics_summary.json
#    - timing_summary.json
#    - comparison_summary.json
# 4. 同一路径写入时默认覆盖旧文件。
#
# 依赖关系：
# - 依赖 src/common/io_utils.py
# - 依赖 src/common/paths.py
# - 被 scripts/run_analysis.py 调用
#
# 重要说明：
# - 本文件只负责“多模型比较结果保存”；
# - 不负责单模型结果保存；
# - 不负责推理；
# - 不负责画图。
# ==========================================================

from __future__ import annotations

from typing import Any

from src.common.io_utils import save_json
from src.common.paths import (
    ensure_task_comparison_dirs,
    get_task_comparison_dir,
    get_task_comparison_metrics_json_path,
    get_task_comparison_timing_json_path,
)


# ==========================================================
# 一、基础保存函数
# ==========================================================

def save_comparison_metrics_summary(
    task_name: str,
    metrics_summary: dict[str, Any],
) -> str:
    """
    保存多模型精度比较结果到 metrics_summary.json。
    """
    ensure_task_comparison_dirs(task_name)
    save_path = get_task_comparison_metrics_json_path(task_name)
    save_json(metrics_summary, save_path)
    return str(save_path)


def save_comparison_timing_summary(
    task_name: str,
    timing_summary: dict[str, Any],
) -> str:
    """
    保存多模型时间比较结果到 timing_summary.json。
    """
    ensure_task_comparison_dirs(task_name)
    save_path = get_task_comparison_timing_json_path(task_name)
    save_json(timing_summary, save_path)
    return str(save_path)


def save_comparison_summary(
    task_name: str,
    summary_dict: dict[str, Any],
) -> str:
    """
    保存整体 comparison summary。

    说明：
    - 这里单独保存一个 comparison_summary.json；
    - 方便以后统一查看：
        - 参与比较的模型列表
        - metrics 比较结果
        - timing 比较结果
        - 额外备注
    """
    ensure_task_comparison_dirs(task_name)
    save_path = get_task_comparison_dir(task_name) / "comparison_summary.json"
    save_json(summary_dict, save_path)
    return str(save_path)


# ==========================================================
# 二、统一打包保存
# ==========================================================

def save_full_comparison_outputs(
    task_name: str,
    metrics_comparison: dict[str, Any],
    timing_comparison: dict[str, Any] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    统一保存多模型比较结果。

    参数：
    - task_name
    - metrics_comparison
    - timing_comparison: 可为空
    - extra_summary: 可为空

    返回：
    {
        "metrics_summary_path": ...,
        "timing_summary_path": ...（若有）,
        "comparison_summary_path": ...
    }
    """
    metrics_path = save_comparison_metrics_summary(
        task_name=task_name,
        metrics_summary=metrics_comparison,
    )

    timing_path = None
    if timing_comparison is not None:
        timing_path = save_comparison_timing_summary(
            task_name=task_name,
            timing_summary=timing_comparison,
        )

    summary_dict = {
        "task_name": task_name,
        "metrics_comparison": metrics_comparison,
        "timing_comparison": timing_comparison,
    }

    if extra_summary is not None:
        summary_dict["extra_summary"] = extra_summary

    summary_path = save_comparison_summary(
        task_name=task_name,
        summary_dict=summary_dict,
    )

    result = {
        "metrics_summary_path": metrics_path,
        "comparison_summary_path": summary_path,
    }

    if timing_path is not None:
        result["timing_summary_path"] = timing_path

    return result