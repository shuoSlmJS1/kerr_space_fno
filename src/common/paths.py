# ==========================================================
# File: src/common/paths.py
#
# 功能简介：
# 1. 统一管理整个项目的数据路径与输出路径；
# 2. 根据 task_name 生成任务数据目录；
# 3. 根据 task_name + model_name 生成模型输出目录；
# 4. 提供 dataset/meta/checkpoint/log/analysis 等统一路径接口；
# 5. 提供目录创建辅助函数。
#
# 依赖关系：
# - 依赖 common/task_spec.py（间接）
# - 被 dataset_saver.py 使用
# - 被训练模块与分析模块使用
#
# 重要说明：
# - 所有路径都从项目根目录统一构造；
# - 避免在不同脚本中手写路径字符串；
# - 适合 Windows / Linux 跨平台迁移；
# - 本文件负责“路径规范”，不负责文件内容写入。
# ==========================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ==========================================================
# 一、项目根目录约定
# ==========================================================

def get_project_root() -> Path:
    """
    返回项目根目录。

    约定：
    - 当前文件位于：
        src/common/paths.py
    - 因此项目根目录就是它的上上上级目录

    这样做的好处：
    - 不依赖当前工作目录
    - Windows / Linux 都可用
    - 所有路径统一从项目根目录出发构造
    """
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    """
    返回数据根目录：
        <project_root>/data/tasks
    """
    return get_project_root() / "data" / "tasks"


def get_outputs_root() -> Path:
    """
    返回输出根目录：
        <project_root>/outputs
    """
    return get_project_root() / "outputs"


# ==========================================================
# 二、数据任务目录
# ==========================================================

def get_task_data_dir(task_name: str) -> Path:
    """
    返回某个任务的数据目录。

    例如：
        data/tasks/vary_a__a0.4_0.7__n240__T800/
    """
    return get_data_root() / task_name


def get_task_dataset_npz_path(task_name: str) -> Path:
    """
    返回某个任务的数据集文件路径。

    统一命名：
        dataset.npz
    """
    return get_task_data_dir(task_name) / "dataset.npz"


def get_task_meta_json_path(task_name: str) -> Path:
    """
    返回某个任务的元信息文件路径。

    统一命名：
        meta.json
    """
    return get_task_data_dir(task_name) / "meta.json"


def get_task_failed_samples_json_path(task_name: str) -> Path:
    """
    返回某个任务的失败样本日志路径。

    统一命名：
        failed_samples.json
    """
    return get_task_data_dir(task_name) / "failed_samples.json"


# ==========================================================
# 三、模型输出目录
# ==========================================================

def get_task_output_dir(task_name: str) -> Path:
    """
    返回某个任务的输出根目录。

    例如：
        outputs/vary_a__a0.4_0.7__n240__T800/
    """
    return get_outputs_root() / task_name


def get_model_output_dir(task_name: str, model_name: str) -> Path:
    """
    返回某个模型在某个任务下的总输出目录。

    例如：
        outputs/<task_name>/<model_name>/
    """
    return get_task_output_dir(task_name) / model_name


def get_model_checkpoints_dir(task_name: str, model_name: str) -> Path:
    """
    返回模型 checkpoint 目录。
    """
    return get_model_output_dir(task_name, model_name) / "checkpoints"


def get_model_logs_dir(task_name: str, model_name: str) -> Path:
    """
    返回模型日志目录。
    """
    return get_model_output_dir(task_name, model_name) / "logs"


def get_model_inference_dir(task_name: str, model_name: str) -> Path:
    """
    返回模型推理结果目录。
    """
    return get_model_output_dir(task_name, model_name) / "inference"


def get_model_analysis_dir(task_name: str, model_name: str) -> Path:
    """
    返回模型分析结果目录。
    """
    return get_model_output_dir(task_name, model_name) / "analysis"


# ==========================================================
# 四、统一文件名（训练阶段）
# ==========================================================

def get_best_checkpoint_path(task_name: str, model_name: str) -> Path:
    """
    返回最佳模型权重路径。

    统一命名：
        best_model.pt
    """
    return get_model_checkpoints_dir(task_name, model_name) / "best_model.pt"


def get_last_checkpoint_path(task_name: str, model_name: str) -> Path:
    """
    返回最后一轮模型权重路径。

    统一命名：
        last_model.pt
    """
    return get_model_checkpoints_dir(task_name, model_name) / "last_model.pt"


def get_train_history_json_path(task_name: str, model_name: str) -> Path:
    """
    返回训练历史记录路径。

    统一命名：
        train_history.json
    """
    return get_model_logs_dir(task_name, model_name) / "train_history.json"


def get_train_summary_json_path(task_name: str, model_name: str) -> Path:
    """
    返回训练摘要路径。

    统一命名：
        train_summary.json
    """
    return get_model_logs_dir(task_name, model_name) / "train_summary.json"


# ==========================================================
# 五、统一文件名（推理分析阶段）
# ==========================================================

def get_inference_predictions_path(task_name: str, model_name: str) -> Path:
    """
    返回单模型推理预测结果路径。

    统一命名：
        predictions.npy
    """
    return get_model_inference_dir(task_name, model_name) / "predictions.npy"


def get_inference_metrics_json_path(task_name: str, model_name: str) -> Path:
    """
    返回单模型精度指标路径。

    统一命名：
        metrics.json
    """
    return get_model_inference_dir(task_name, model_name) / "metrics.json"


def get_inference_timing_json_path(task_name: str, model_name: str) -> Path:
    """
    返回单模型耗时统计路径。

    统一命名：
        timing.json
    """
    return get_model_inference_dir(task_name, model_name) / "timing.json"


def get_analysis_summary_json_path(task_name: str, model_name: str) -> Path:
    """
    返回单模型分析摘要路径。

    统一命名：
        analysis_summary.json
    """
    return get_model_analysis_dir(task_name, model_name) / "analysis_summary.json"


# ==========================================================
# 六、任务级比较输出目录
# ==========================================================

def get_task_comparison_dir(task_name: str) -> Path:
    """
    返回任务级对比目录。

    用途：
    - 多模型对比图
    - 多模型对比指标
    - 多模型对比耗时
    """
    return get_task_output_dir(task_name) / "comparison"


def get_task_comparison_metrics_json_path(task_name: str) -> Path:
    """
    返回任务级多模型对比指标路径。

    统一命名：
        metrics_summary.json
    """
    return get_task_comparison_dir(task_name) / "metrics_summary.json"


def get_task_comparison_timing_json_path(task_name: str) -> Path:
    """
    返回任务级多模型耗时对比路径。

    统一命名：
        timing_summary.json
    """
    return get_task_comparison_dir(task_name) / "timing_summary.json"


def get_task_comparison_plots_dir(task_name: str) -> Path:
    """
    返回任务级比较图目录。
    """
    return get_task_comparison_dir(task_name) / "plots"


def get_task_comparison_report_dir(task_name: str) -> Path:
    """
    返回任务级文本报告目录。
    """
    return get_task_comparison_dir(task_name) / "report"


# ==========================================================
# 七、打包路径对象（便于外层调用）
# ==========================================================

@dataclass
class TaskDataPaths:
    """
    某个任务的数据相关路径集合。

    用途：
    - 让外层代码少写很多重复 get_xxx 调用
    - 一次性拿到该任务下的数据保存路径
    """
    task_dir: Path
    dataset_npz: Path
    meta_json: Path
    failed_samples_json: Path


@dataclass
class ModelOutputPaths:
    """
    某个任务 + 某个模型 的输出相关路径集合。

    用途：
    - 训练脚本、推理脚本、分析脚本都可以直接使用
    """
    model_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    inference_dir: Path
    analysis_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    train_history_json: Path
    train_summary_json: Path
    predictions_npy: Path
    metrics_json: Path
    timing_json: Path
    analysis_summary_json: Path


def build_task_data_paths(task_name: str) -> TaskDataPaths:
    """
    构建某个任务的数据路径集合对象。
    """
    return TaskDataPaths(
        task_dir=get_task_data_dir(task_name),
        dataset_npz=get_task_dataset_npz_path(task_name),
        meta_json=get_task_meta_json_path(task_name),
        failed_samples_json=get_task_failed_samples_json_path(task_name),
    )


def build_model_output_paths(task_name: str, model_name: str) -> ModelOutputPaths:
    """
    构建某个任务 + 某个模型 的输出路径集合对象。
    """
    return ModelOutputPaths(
        model_dir=get_model_output_dir(task_name, model_name),
        checkpoints_dir=get_model_checkpoints_dir(task_name, model_name),
        logs_dir=get_model_logs_dir(task_name, model_name),
        inference_dir=get_model_inference_dir(task_name, model_name),
        analysis_dir=get_model_analysis_dir(task_name, model_name),
        best_checkpoint=get_best_checkpoint_path(task_name, model_name),
        last_checkpoint=get_last_checkpoint_path(task_name, model_name),
        train_history_json=get_train_history_json_path(task_name, model_name),
        train_summary_json=get_train_summary_json_path(task_name, model_name),
        predictions_npy=get_inference_predictions_path(task_name, model_name),
        metrics_json=get_inference_metrics_json_path(task_name, model_name),
        timing_json=get_inference_timing_json_path(task_name, model_name),
        analysis_summary_json=get_analysis_summary_json_path(task_name, model_name),
    )


# ==========================================================
# 八、目录创建辅助
# ==========================================================

def ensure_task_data_dirs(task_name: str) -> TaskDataPaths:
    """
    创建任务数据目录，并返回对应路径对象。
    """
    paths = build_task_data_paths(task_name)
    paths.task_dir.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_model_output_dirs(task_name: str, model_name: str) -> ModelOutputPaths:
    """
    创建模型输出目录，并返回对应路径对象。
    """
    paths = build_model_output_paths(task_name, model_name)

    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.inference_dir.mkdir(parents=True, exist_ok=True)
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)

    return paths


def ensure_task_comparison_dirs(task_name: str) -> None:
    """
    创建任务级 comparison 目录。
    """
    get_task_comparison_dir(task_name).mkdir(parents=True, exist_ok=True)
    get_task_comparison_plots_dir(task_name).mkdir(parents=True, exist_ok=True)
    get_task_comparison_report_dir(task_name).mkdir(parents=True, exist_ok=True)