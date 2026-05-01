# ==========================================================
# File: src/inference_analysis/result_saver.py
#
# 功能简介：
# 1. 统一保存单模型推理分析结果；
# 2. 保存：
#    - predictions.npy
#    - metrics.json
#    - timing.json
#    - analysis_summary.json
# 3. 使用 common/paths.py 统一管理输出路径；
# 4. 同一路径写入时默认覆盖旧文件。
#
# 依赖关系：
# - 依赖 src/common/io_utils.py
# - 依赖 src/common/paths.py
# - 依赖 src/inference_analysis/inference.py
# - 被 scripts/run_analysis.py 调用
#
# 重要说明：
# - 本文件只负责“保存结果”；
# - 不负责推理；
# - 不负责画图；
# - 不负责多模型比较。
# ==========================================================

from __future__ import annotations

from typing import Any

from src.common.io_utils import save_json, save_npy
from src.common.paths import (
    ensure_model_output_dirs,
    get_analysis_summary_json_path,
    get_inference_metrics_json_path,
    get_inference_predictions_path,
    get_inference_timing_json_path,
)
from src.inference_analysis.inference import InferenceResult
from src.inference_analysis.timing import TimingResult


# ==========================================================
# 一、保存预测结果
# ==========================================================

def save_predictions(
    inference_result: InferenceResult,
    task_name: str,
    model_name: str,
) -> str:
    """
    保存模型预测结果 predictions.npy。

    说明：
    - 只保存 predictions；
    - targets 通常可从数据集重新读取，因此这里默认不重复保存 targets。
    """
    ensure_model_output_dirs(task_name, model_name)
    save_path = get_inference_predictions_path(task_name, model_name)
    save_npy(inference_result.predictions, save_path)
    return str(save_path)


# ==========================================================
# 二、保存精度指标
# ==========================================================

def save_metrics(
    metrics_dict: dict[str, Any],
    task_name: str,
    model_name: str,
) -> str:
    """
    保存单模型精度指标 metrics.json。
    """
    ensure_model_output_dirs(task_name, model_name)
    save_path = get_inference_metrics_json_path(task_name, model_name)
    save_json(metrics_dict, save_path)
    return str(save_path)


# ==========================================================
# 三、保存时间统计
# ==========================================================

def save_timing(
    timing_dict: dict[str, Any],
    task_name: str,
    model_name: str,
) -> str:
    """
    保存单模型时间统计 timing.json。
    """
    ensure_model_output_dirs(task_name, model_name)
    save_path = get_inference_timing_json_path(task_name, model_name)
    save_json(timing_dict, save_path)
    return str(save_path)


# ==========================================================
# 四、保存分析摘要
# ==========================================================

def save_analysis_summary(
    summary_dict: dict[str, Any],
    task_name: str,
    model_name: str,
) -> str:
    """
    保存单模型分析摘要 analysis_summary.json。
    """
    ensure_model_output_dirs(task_name, model_name)
    save_path = get_analysis_summary_json_path(task_name, model_name)
    save_json(summary_dict, save_path)
    return str(save_path)


# ==========================================================
# 五、统一打包保存
# ==========================================================

def save_single_model_analysis_outputs(
    inference_result: InferenceResult,
    metrics_dict: dict[str, Any],
    timing_dict: dict[str, Any],
    task_name: str,
    model_name: str,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    统一保存单模型分析输出。

    保存内容：
    - predictions.npy
    - metrics.json
    - timing.json
    - analysis_summary.json

    参数：
    - inference_result
    - metrics_dict
    - timing_dict
    - task_name
    - model_name
    - extra_summary:
        额外摘要信息，可为空

    返回：
    - 各文件路径字典
    """
    predictions_path = save_predictions(
        inference_result=inference_result,
        task_name=task_name,
        model_name=model_name,
    )

    metrics_path = save_metrics(
        metrics_dict=metrics_dict,
        task_name=task_name,
        model_name=model_name,
    )

    timing_path = save_timing(
        timing_dict=timing_dict,
        task_name=task_name,
        model_name=model_name,
    )

    summary_dict = {
        "task_name": task_name,
        "model_name": model_name,
        "predictions_shape": tuple(inference_result.predictions.shape),
        "targets_shape": tuple(inference_result.targets.shape),
        "metrics": metrics_dict,
        "timing": timing_dict,
    }

    if extra_summary is not None:
        summary_dict["extra_summary"] = extra_summary

    summary_path = save_analysis_summary(
        summary_dict=summary_dict,
        task_name=task_name,
        model_name=model_name,
    )

    return {
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "timing_path": timing_path,
        "analysis_summary_path": summary_path,
    }