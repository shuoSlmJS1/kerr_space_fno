# ==========================================================
# File: src/inference_analysis/comparison.py
#
# 功能简介：
# 1. 汇总多个模型的推理分析结果；
# 2. 对比多个模型的精度指标与时间指标；
# 3. 生成适合保存、展示和后续画图的比较结果字典；
# 4. 兼容单模型情况（此时只做汇总，不强调对比）。
#
# 依赖关系：
# - 被 scripts/run_analysis.py 调用
# - 与 metrics.py / timing.py / plotting.py / result_saver.py 配合使用
#
# 重要说明：
# - 本文件不负责单模型推理；
# - 不负责实际画图；
# - 不负责文件保存；
# - 只负责“多模型结果汇总与比较逻辑”。
# ==========================================================

from __future__ import annotations

from typing import Any


# ==========================================================
# 一、基础汇总
# ==========================================================

def summarize_model_metrics(
    model_name: str,
    metrics_dict: dict[str, Any],
    timing_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    将单模型的 metrics/timing 信息整理成统一格式。

    输入：
    - model_name
    - metrics_dict
    - timing_dict: 可为空

    返回：
    {
        "model_name": ...,
        "mse": ...,
        "relative_l2": ...,
        "timing": {...} 或 None,
    }
    """
    summary = {
        "model_name": model_name,
        "mse": metrics_dict.get("mse", None),
        "relative_l2": metrics_dict.get("relative_l2", None),
        "timing": timing_dict if timing_dict is not None else None,
    }
    return summary


# ==========================================================
# 二、多模型指标汇总
# ==========================================================

def build_metrics_comparison(
    model_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    构造多模型精度比较结果。

    输入：
    - model_metrics:
        {
            "fno_m32_w64_d4": {"mse": ..., "relative_l2": ...},
            "cnn_w64_d6_k9": {"mse": ..., "relative_l2": ...},
            ...
        }

    返回：
    {
        "models": [...],
        "metrics_table": [...],
        "best_by_mse": ...,
        "best_by_relative_l2": ...
    }
    """
    if not model_metrics:
        raise ValueError("model_metrics 不能为空。")

    model_names = list(model_metrics.keys())

    metrics_table = []
    for model_name in model_names:
        metrics_dict = model_metrics[model_name]
        metrics_table.append(
            {
                "model_name": model_name,
                "mse": metrics_dict.get("mse", None),
                "relative_l2": metrics_dict.get("relative_l2", None),
            }
        )

    # 过滤掉指标缺失的情况
    valid_mse_models = [
        row for row in metrics_table
        if row["mse"] is not None
    ]
    valid_rel_models = [
        row for row in metrics_table
        if row["relative_l2"] is not None
    ]

    best_by_mse = None
    if valid_mse_models:
        best_by_mse = min(valid_mse_models, key=lambda row: row["mse"])["model_name"]

    best_by_relative_l2 = None
    if valid_rel_models:
        best_by_relative_l2 = min(valid_rel_models, key=lambda row: row["relative_l2"])["model_name"]

    return {
        "models": model_names,
        "metrics_table": metrics_table,
        "best_by_mse": best_by_mse,
        "best_by_relative_l2": best_by_relative_l2,
    }


# ==========================================================
# 三、多模型时间汇总
# ==========================================================

def build_timing_comparison(
    model_timings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    构造多模型时间比较结果。

    输入：
    - model_timings:
        {
            "fno_m32_w64_d4": {
                "model_total_seconds": ...,
                "model_avg_seconds_per_sample": ...,
                ...
            },
            ...
        }

    返回：
    {
        "models": [...],
        "timing_table": [...],
        "fastest_by_total": ...,
        "fastest_by_per_sample": ...
    }
    """
    if not model_timings:
        raise ValueError("model_timings 不能为空。")

    model_names = list(model_timings.keys())

    timing_table = []
    for model_name in model_names:
        timing_dict = model_timings[model_name]
        timing_table.append(
            {
                "model_name": model_name,
                "model_total_seconds": timing_dict.get("model_total_seconds", None),
                "model_avg_seconds_per_sample": timing_dict.get("model_avg_seconds_per_sample", None),
                "traditional_total_seconds": timing_dict.get("traditional_total_seconds", None),
                "traditional_avg_seconds_per_sample": timing_dict.get("traditional_avg_seconds_per_sample", None),
                "speedup_total": timing_dict.get("speedup_total", None),
                "speedup_per_sample": timing_dict.get("speedup_per_sample", None),
            }
        )

    valid_total_models = [
        row for row in timing_table
        if row["model_total_seconds"] is not None
    ]
    valid_ps_models = [
        row for row in timing_table
        if row["model_avg_seconds_per_sample"] is not None
    ]

    fastest_by_total = None
    if valid_total_models:
        fastest_by_total = min(valid_total_models, key=lambda row: row["model_total_seconds"])["model_name"]

    fastest_by_per_sample = None
    if valid_ps_models:
        fastest_by_per_sample = min(valid_ps_models, key=lambda row: row["model_avg_seconds_per_sample"])["model_name"]

    return {
        "models": model_names,
        "timing_table": timing_table,
        "fastest_by_total": fastest_by_total,
        "fastest_by_per_sample": fastest_by_per_sample,
    }


# ==========================================================
# 四、综合比较
# ==========================================================

def build_full_comparison(
    model_metrics: dict[str, dict[str, Any]],
    model_timings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    构造完整的多模型比较结果。

    参数：
    - model_metrics
    - model_timings: 可为空

    返回：
    {
        "metrics_comparison": ...,
        "timing_comparison": ... 或 None,
    }
    """
    result = {
        "metrics_comparison": build_metrics_comparison(model_metrics),
        "timing_comparison": None,
    }

    if model_timings is not None and len(model_timings) > 0:
        result["timing_comparison"] = build_timing_comparison(model_timings)

    return result


# ==========================================================
# 五、单模型兼容接口
# ==========================================================

def build_single_model_summary(
    model_name: str,
    metrics_dict: dict[str, Any],
    timing_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    单模型兼容接口。

    用途：
    - 当当前只有一个模型时，也可以沿用 comparison.py 的汇总风格；
    - 这样外层脚本逻辑更统一。

    返回：
    {
        "num_models": 1,
        "models": [...],
        "metrics_comparison": ...,
        "timing_comparison": ...,
    }
    """
    model_metrics = {model_name: metrics_dict}
    model_timings = {model_name: timing_dict} if timing_dict is not None else None

    full = build_full_comparison(
        model_metrics=model_metrics,
        model_timings=model_timings,
    )

    return {
        "num_models": 1,
        "models": [model_name],
        **full,
    }