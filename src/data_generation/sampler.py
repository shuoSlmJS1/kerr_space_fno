# ==========================================================
# File: src/data_generation/sampler.py
#
# 功能简介：
# 1. 根据 TaskSpec 生成参数采样点；
# 2. 支持单参数一维均匀采样；
# 3. 支持双参数二维网格采样；
# 4. 后续可扩展到更高维参数组合；
# 5. 输出统一格式的样本参数字典列表。
#
# 依赖关系：
# - 依赖 common/task_spec.py
# - 被 dataset_builder.py 调用
#
# 重要说明：
# - 本文件只负责“采样”；
# - 不负责合法性检查；
# - 不负责数值积分；
# - 不负责将固定参数与变化参数合并。
# ==========================================================
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from src.common.task_spec import TaskSpec


# ==========================================================
# 一、对外统一入口
# ==========================================================

def build_parameter_samples(task_spec: TaskSpec) -> list[dict[str, Any]]:
    """
    根据 TaskSpec 构造参数样本列表。

    返回格式：
    - list[dict]
    - 每个元素表示一个具体样本的参数字典

    例如单参数任务：
        vary_params = ["Q"]
        ->
        [
            {"Q": 1.6},
            {"Q": 1.60585},
            ...
        ]

    例如双参数任务：
        vary_params = ["a", "Q"]
        ->
        [
            {"a": 0.4, "Q": 1.6},
            {"a": 0.4, "Q": 1.67},
            ...
        ]

    说明：
    - 这里只负责生成“变化参数”的取值样本；
    - fixed_params 不在这里合并；
    - fixed_params 的合并应由 dataset_builder.py 负责，
      因为它更适合在真正构造轨道样本时统一拼装。
    """
    if task_spec.sampling_mode == "grid":
        return _build_grid_samples(task_spec)

    raise ValueError(f"暂不支持的 sampling_mode: {task_spec.sampling_mode!r}")


# ==========================================================
# 二、网格采样
# ==========================================================

def _build_grid_samples(task_spec: TaskSpec) -> list[dict[str, Any]]:
    """
    按网格方式生成参数样本。

    逻辑：
    1. 对每个变化参数，先生成一维采样轴；
    2. 对多维轴做笛卡尔积；
    3. 将每个组合点转成字典样本。

    例如：
    - vary_params = ["Q"], sample_shape = [240]
      -> 生成长度为 240 的一维样本列表

    - vary_params = ["a", "Q"], sample_shape = [20, 20]
      -> 生成 400 个二维网格样本
    """
    axes = build_sampling_axes(task_spec)

    ordered_vary_params = task_spec.vary_params

    # product(*axes) 会生成所有网格点组合
    grid_points = product(*axes)

    samples: list[dict[str, Any]] = []
    for point in grid_points:
        sample = {
            param_name: float(value)
            for param_name, value in zip(ordered_vary_params, point)
        }
        samples.append(sample)

    return samples


# ==========================================================
# 三、采样轴生成
# ==========================================================

def build_sampling_axes(task_spec: TaskSpec) -> list[np.ndarray]:
    """
    为每个变化参数生成一个一维采样轴。

    返回：
    - list[np.ndarray]
    - 每个数组对应一个变化参数的采样点序列

    例如：
    - vary_params = ["Q"]
      -> [Q_axis]

    - vary_params = ["a", "Q"]
      -> [a_axis, Q_axis]

    注意：
    - 这里轴的顺序与 task_spec.vary_params 顺序一致；
    - 前面 parameter_parser.py 已经对 vary_params 做过规范排序，
      所以这里不再额外排序。
    """
    axes: list[np.ndarray] = []

    for param_name, axis_size in zip(task_spec.vary_params, task_spec.sample_shape):
        value_range = task_spec.vary_ranges[param_name]
        axis = build_1d_axis(
            param_name=param_name,
            value_range=value_range,
            axis_size=axis_size,
        )
        axes.append(axis)

    return axes


def build_1d_axis(
    param_name: str,
    value_range: tuple[float, float],
    axis_size: int,
) -> np.ndarray:
    """
    为单个参数生成一维均匀采样轴。

    参数：
    - param_name   : 参数名，仅用于调试或错误信息
    - value_range  : (min, max)
    - axis_size    : 采样点数

    返回：
    - np.ndarray，长度为 axis_size

    当前策略：
    - 使用均匀线性采样 np.linspace

    以后如有需要，可以扩展：
    - 对数采样
    - 分段采样
    - 自定义非均匀采样
    """
    v_min, v_max = value_range

    if axis_size < 2:
        raise ValueError(
            f"参数 {param_name} 的 axis_size 至少应为 2，当前得到：{axis_size}"
        )

    if v_max <= v_min:
        raise ValueError(
            f"参数 {param_name} 的采样范围必须满足 max > min，当前得到：{value_range}"
        )

    return np.linspace(v_min, v_max, axis_size, dtype=np.float64)


# ==========================================================
# 四、目标成功数补充候选
# ==========================================================

def sample_key(
    sample: dict[str, Any],
    param_order: list[str],
) -> tuple[str, ...]:
    """
    将参数样本转换为稳定、可比较的键。

    使用 float.hex() 保留浮点数的精确二进制值，
    用于避免重复尝试完全相同的候选点。
    """
    return tuple(
        float(sample[param_name]).hex()
        for param_name in param_order
    )


def build_random_completion_candidates(
    task_spec: TaskSpec,
    candidate_count: int,
    attempted_keys: set[tuple[str, ...]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    在原参数范围内生成新的均匀随机候选点。

    用途：
    - 第一批规则网格出现失败后补充候选；
    - 只返回尚未尝试过的参数点；
    - 随机性由外部传入的 rng 控制，保证可复现。

    说明：
    - 这些补充点通常会使最终成功样本不再是严格规则网格；
    - 是否发生补样应记录在 meta.json 中。
    """
    if candidate_count < 0:
        raise ValueError(
            f"candidate_count 不能为负，当前为 {candidate_count}"
        )

    if candidate_count == 0:
        return []

    param_order = list(task_spec.vary_params)
    candidates: list[dict[str, Any]] = []

    # 连续均匀分布下重复概率极低，但仍设置循环上限，
    # 防止未来离散采样规则造成意外死循环。
    max_draws = max(candidate_count * 20, 100)
    draw_count = 0

    while (
        len(candidates) < candidate_count
        and draw_count < max_draws
    ):
        draw_count += 1

        sample = {
            param_name: float(
                rng.uniform(
                    task_spec.vary_ranges[param_name][0],
                    task_spec.vary_ranges[param_name][1],
                )
            )
            for param_name in param_order
        }

        key = sample_key(sample, param_order)

        if key in attempted_keys:
            continue

        attempted_keys.add(key)
        candidates.append(sample)

    if len(candidates) != candidate_count:
        raise RuntimeError(
            "未能生成足够数量的不重复补充候选点："
            f"需要 {candidate_count}，实际得到 {len(candidates)}。"
        )

    return candidates


# ==========================================================
# 五、辅助函数
# ==========================================================

def count_expected_samples(task_spec: TaskSpec) -> int:
    """
    返回按当前 sample_shape 理论上应生成的总样本数。

    说明：
    - 这个值等价于 TaskSpec.total_requested_samples
    - 之所以在 sampler.py 里再提供一个函数，
      是为了让采样模块自身也能直接使用
    """
    return task_spec.total_requested_samples


def preview_sampling_axes(task_spec: TaskSpec, max_items: int = 5) -> dict[str, list[float]]:
    """
    返回采样轴的预览信息，便于调试和日志打印。

    参数：
    - max_items: 每个轴最多展示多少个采样点

    返回：
    - dict[param_name, list[float]]

    示例：
    {
        "Q": [1.6, 1.60585, 1.6117, ...],
        "a": [0.4, 0.4158, 0.4316, ...],
    }

    说明：
    - 只用于调试或打印日志；
    - 不参与正式数据生成逻辑。
    """
    axes = build_sampling_axes(task_spec)

    preview: dict[str, list[float]] = {}
    for param_name, axis in zip(task_spec.vary_params, axes):
        preview[param_name] = [float(x) for x in axis[:max_items]]

    return preview