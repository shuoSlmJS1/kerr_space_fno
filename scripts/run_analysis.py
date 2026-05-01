# ==========================================================
# File: scripts/run_analysis.py
#
# 功能简介：
# 1. 推理分析模块的统一入口脚本；
# 2. 加载任务数据与已训练模型；
# 3. 执行单模型推理、误差分析、时间统计与画图；
# 4. 支持单模型分析模式；
# 5. 支持多模型比较模式；
# 6. 保存单模型结果与任务级 comparison 结果。
#
# 依赖关系：
# - 依赖 src/models/registry.py
# - 依赖 src/common/paths.py
# - 依赖 src/common/io_utils.py
# - 依赖 src/training/fno1d/dataset_loader_1d.py
# - 依赖 src/inference_analysis/ 下各模块
#
# 重要说明：
# - 单模型模式：保存到 outputs/<task_name>/<model_name>/
# - 多模型模式：额外保存到 outputs/<task_name>/comparison/
# - 同一路径写入时默认覆盖旧文件。
# ==========================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# 将项目根目录加入 sys.path，保证可以导入 src.xxx
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import load_json
from src.common.paths import (
    get_best_checkpoint_path,
    get_model_analysis_dir,
)
from src.inference_analysis.comparison import (
    build_full_comparison,
    build_single_model_summary,
)
from src.inference_analysis.comparison_saver import save_full_comparison_outputs
from src.inference_analysis.inference import predict_loader, summarize_inference_result
from src.inference_analysis.metrics import (
    compute_inference_metrics,
    compute_inference_metrics_with_details,
)
from src.inference_analysis.plotting import generate_standard_single_model_plots
from src.inference_analysis.single_model_result_saver import save_single_model_analysis_outputs
from src.inference_analysis.timing import (
    build_full_param_dicts_for_timing,
    compare_timing_results,
    summarize_timing_result,
    time_model_inference_loader,
    time_traditional_orbit_generation_from_param_dicts,
)
from src.models.registry import build_model, get_model_help_text
from src.training.fno1d.dataset_loader_1d import (
    build_dataloaders_1d,
    summarize_loaded_bundle_1d,
)


# ==========================================================
# 一、命令行解析
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inference and analysis entry for Kerr orbit prediction models."
    )

    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="任务名，对应 data/tasks/<task_name>/dataset.npz",
    )

    parser.add_argument(
        "--model-names",
        nargs="+",
        type=str,
        required=True,
        help="要分析的一个或多个 model_name。",
    )

    parser.add_argument(
        "--show-model-help",
        action="store_true",
        help="显示支持的模型类型及参数要求，然后退出。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cpu / cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="分析时 DataLoader 的 batch size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers",
    )

    parser.add_argument(
        "--representative-sample-index",
        type=int,
        default=0,
        help="用于单模型标准分析图的代表样本索引",
    )

    return parser


# ==========================================================
# 二、模型恢复
# ==========================================================

def load_model_from_best_checkpoint(
    task_name: str,
    model_name: str,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    从 best checkpoint 恢复模型。

    返回：
    - model
    - checkpoint
    """
    ckpt_path = get_best_checkpoint_path(task_name, model_name)
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint["config"]
    model_config = config["model_config"]

    model = build_model(
        model_type=model_config["model_type"],
        in_dim=model_config["in_dim"],
        out_dim=model_config["out_dim"],
        modes=model_config["modes"],
        width=model_config["width"],
        depth=model_config["depth"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_task_meta(task_name: str) -> dict[str, Any]:
    """
    读取任务 meta.json。
    """
    from src.common.paths import get_task_meta_json_path
    return load_json(get_task_meta_json_path(task_name))


# ==========================================================
# 三、单模型分析
# ==========================================================

def analyze_single_model(
    task_name: str,
    model_name: str,
    test_loader,
    bundle,
    task_meta: dict[str, Any],
    device: str,
    representative_sample_index: int,
) -> dict[str, Any]:
    """
    完成单模型完整分析流程。

    返回：
    {
        "metrics": ...,
        "metrics_detail": ...,
        "timing": ...,
        "saved_paths": ...,
        "plot_paths": ...,
        "inference_summary": ...
    }
    """
    # ------------------------------
    # A. 载入模型
    # ------------------------------
    model, checkpoint = load_model_from_best_checkpoint(
        task_name=task_name,
        model_name=model_name,
        device=device,
    )

    # ------------------------------
    # B. 推理
    # ------------------------------
    inference_result = predict_loader(
        model=model,
        loader=test_loader,
        device=device,
    )

    inference_summary = summarize_inference_result(inference_result)

    # ------------------------------
    # C. 精度指标
    # ------------------------------
    metrics_dict = compute_inference_metrics(inference_result)
    metrics_detail = compute_inference_metrics_with_details(inference_result)

    # ------------------------------
    # D. 时间统计
    # ------------------------------
    model_timing = time_model_inference_loader(
        model=model,
        loader=test_loader,
        device=device,
        warmup=True,
    )

    fixed_params = task_meta["task_spec"]["fixed_params"]
    n_steps = int(task_meta["task_spec"]["n_steps"])
    step_size = float(task_meta["task_spec"]["step_size"])

    full_param_dicts = build_full_param_dicts_for_timing(
        vary_params_array=bundle.x_test_raw,
        vary_params_order=bundle.vary_params_order,
        fixed_params=fixed_params,
    )

    traditional_timing = time_traditional_orbit_generation_from_param_dicts(
        full_param_dicts=full_param_dicts,
        n_steps=n_steps,
        step_size=step_size,
    )

    timing_dict = compare_timing_results(
        model_timing=model_timing,
        traditional_timing=traditional_timing,
    )

    # ------------------------------
    # E. 单模型画图
    # ------------------------------
    analysis_dir = get_model_analysis_dir(task_name, model_name)
    plot_paths = generate_standard_single_model_plots(
        result=inference_result,
        output_dir=analysis_dir,
        lambda_grid=bundle.lambda_grid,
        representative_sample_index=representative_sample_index,
    )

    # ------------------------------
    # F. 保存单模型结果
    # ------------------------------
    saved_paths = save_single_model_analysis_outputs(
        inference_result=inference_result,
        metrics_dict=metrics_dict,
        timing_dict=timing_dict,
        task_name=task_name,
        model_name=model_name,
        extra_summary={
            "metrics_detail": metrics_detail,
            "inference_summary": inference_summary,
            "plot_paths": plot_paths,
            "checkpoint_epoch": checkpoint.get("epoch", None),
        },
    )

    return {
        "metrics": metrics_dict,
        "metrics_detail": metrics_detail,
        "timing": timing_dict,
        "saved_paths": saved_paths,
        "plot_paths": plot_paths,
        "inference_summary": inference_summary,
    }


# ==========================================================
# 四、主流程
# ==========================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.show_model_help:
        print(get_model_help_text())
        return

    # ------------------------------
    # A. 加载测试集
    # ------------------------------
    _, _, test_loader, bundle = build_dataloaders_1d(
        task_name=args.task_name,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    print("=" * 70)
    print("Loaded dataset summary")
    print("=" * 70)
    for k, v in summarize_loaded_bundle_1d(bundle).items():
        print(f"{k:<24s} : {v}")

    task_meta = load_task_meta(args.task_name)

    # ------------------------------
    # B. 逐模型分析
    # ------------------------------
    all_metrics: dict[str, dict[str, Any]] = {}
    all_timings: dict[str, dict[str, Any]] = {}
    all_results: dict[str, dict[str, Any]] = {}

    for model_name in args.model_names:
        print("-" * 70)
        print(f"Running analysis for model: {model_name}")
        print("-" * 70)

        result = analyze_single_model(
            task_name=args.task_name,
            model_name=model_name,
            test_loader=test_loader,
            bundle=bundle,
            task_meta=task_meta,
            device=args.device,
            representative_sample_index=int(args.representative_sample_index),
        )

        all_results[model_name] = result
        all_metrics[model_name] = result["metrics"]
        all_timings[model_name] = result["timing"]

        print(f"{model_name} Test MSE         : {result['metrics']['mse']:.6e}")
        print(f"{model_name} Test Relative L2 : {result['metrics']['relative_l2']:.6e}")

    # ------------------------------
    # C. 单模型 / 多模型总结
    # ------------------------------
    print("-" * 70)

    if len(args.model_names) == 1:
        model_name = args.model_names[0]
        single_summary = build_single_model_summary(
            model_name=model_name,
            metrics_dict=all_metrics[model_name],
            timing_dict=all_timings[model_name],
        )
        print("Single-model summary:")
        print(single_summary)
    else:
        full_comparison = build_full_comparison(
            model_metrics=all_metrics,
            model_timings=all_timings,
        )

        save_paths = save_full_comparison_outputs(
            task_name=args.task_name,
            metrics_comparison=full_comparison["metrics_comparison"],
            timing_comparison=full_comparison["timing_comparison"],
            extra_summary={
                "model_names": list(args.model_names),
            },
        )

        print("Metrics summary:")
        print(full_comparison["metrics_comparison"])
        print("Timing summary:")
        print(full_comparison["timing_comparison"])
        print("Saved comparison files:")
        print(save_paths)

    print("-" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)