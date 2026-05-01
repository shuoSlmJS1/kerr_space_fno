# ==========================================================
# File: src/data_generation/astrophysical_checks.py
#
# 功能简介：
# 1. 提供现实天体范围参考说明文本；
# 2. 对 TaskSpec 做现实范围 warning 检查；
# 3. 对高自旋、高能量、半径过近/过远等情况进行提醒；
# 4. 返回 warning 列表，供入口脚本决定是否需要用户确认后再继续。
#
# 依赖关系：
# - 依赖 common/task_spec.py
# - 依赖 validity.py 中的 warning 类型
# - 被 dataset_builder.py 使用
# - 被 scripts/generate_dataset.py 使用
#
# 重要说明：
# - 这里只做 warning，不报错终止；
# - 用户确认逻辑不写在这里，而写在入口脚本中；
# - 本文件负责“现实参考范围”，不负责硬约束。
# ==========================================================
from __future__ import annotations

import warnings

from src.common.task_spec import TaskSpec
from src.data_generation.validity import AstrophysicalRangeWarning


# ==========================================================
# 一、现实范围参考文本
# ==========================================================

def get_astrophysical_range_help_text() -> str:
    """
    返回“现实天体范围参考”的帮助文本。

    说明：
    - 这里只提供“建议范围 / 常见参考范围”；
    - 不代表硬约束；
    - 超出这些范围时，系统只做 warning，不会自动终止。
    """
    lines = [
        "Astrophysical reference ranges (warning only, not hard constraints):",
        "",
        "1) M:",
        "   - In this project, M is usually fixed to 1 as a dimensionless scale.",
        "   - So M is typically not used as a scan parameter.",
        "",
        "2) a:",
        "   - For Kerr black holes, usually 0 <= a <= M.",
        "   - In dimensionless experiments, a is commonly treated in [0, 1].",
        "   - a >= 0.95 is considered near-extremal and more sensitive.",
        "",
        "3) E:",
        "   - For typical bound timelike orbits, E < 1 is more common.",
        "   - E >= 1 may correspond to unbound / escape-like cases.",
        "",
        "4) Lz:",
        "   - No single universal observed range.",
        "   - In current project style, moderate positive values are easier to keep stable.",
        "   - Suggested first trial range: [2.0, 4.0].",
        "",
        "5) Q:",
        "   - No single universal observed range.",
        "   - In current project style, Q in about [1.4, 3.0] has been workable.",
        "",
        "6) r0:",
        "   - Too small: closer to horizon, more sensitive.",
        "   - Too large: more weak-field, less near-BH character.",
        "   - Suggested first trial range: [8, 15].",
        "",
        "7) theta0:",
        "   - Must satisfy 0 < theta0 < pi.",
        "   - Avoid being too close to 0 or pi.",
        "   - Suggested first trial range: [0.8, 2.3].",
        "",
        "Important:",
        "   - These are reference ranges for warning/help only.",
        "   - Hard constraints are checked in validity.py.",
    ]
    return "\n".join(lines)


# ==========================================================
# 二、任务级现实范围 warning 检查
# ==========================================================

def warn_task_spec_astrophysical_ranges(task_spec: TaskSpec) -> list[str]:
    """
    对 TaskSpec 做现实范围 warning 检查。

    返回：
    - warnings_list: list[str]

    说明：
    - 该函数不会报错终止；
    - 它一边收集 warning 文本，一边发出 warnings.warn(...)；
    - 后续入口脚本可以根据返回的 warning 列表决定是否提示用户确认再继续。
    """
    warnings_list: list[str] = []

    _warn_spin_near_extremal(task_spec, warnings_list)
    _warn_energy_range(task_spec, warnings_list)
    _warn_radius_range(task_spec, warnings_list)
    _warn_theta_near_poles(task_spec, warnings_list)
    _warn_lz_range(task_spec, warnings_list)
    _warn_q_range(task_spec, warnings_list)

    return warnings_list


# ==========================================================
# 三、具体 warning 规则
# ==========================================================

def _emit_warning(message: str, warnings_list: list[str]) -> None:
    """
    统一收集并发出 warning。
    """
    warnings_list.append(message)
    warnings.warn(message, AstrophysicalRangeWarning)


def _warn_spin_near_extremal(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对高自旋 a 做 warning。
    """
    if "a" in task_spec.fixed_params:
        a = float(task_spec.fixed_params["a"])
        if a >= 0.95:
            _emit_warning(
                f"[AstroRange] 固定自旋 a={a} 已接近极限 Kerr 自旋，数值与物理解释都可能更敏感。",
                warnings_list,
            )

    if "a" in task_spec.vary_params:
        a_min, a_max = task_spec.vary_ranges["a"]
        if a_max >= 0.95:
            _emit_warning(
                f"[AstroRange] a 的范围上界 a_max={a_max} 已接近极限 Kerr 自旋，属于高自旋敏感区。",
                warnings_list,
            )


def _warn_energy_range(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对 E 的范围做 warning。
    """
    if "E" in task_spec.fixed_params:
        E = float(task_spec.fixed_params["E"])
        if E >= 1.0:
            _emit_warning(
                f"[AstroRange] 固定能量 E={E} 已达到或超过 1，这通常不再对应典型束缚 timelike 轨道。",
                warnings_list,
            )

    if "E" in task_spec.vary_params:
        E_min, E_max = task_spec.vary_ranges["E"]
        if E_max >= 1.0:
            _emit_warning(
                f"[AstroRange] E 的范围上界 E_max={E_max} 已达到或超过 1，可能对应非束缚或更接近逃逸型轨道。",
                warnings_list,
            )


def _warn_radius_range(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对 r0 做经验 warning。
    """
    if "r0" in task_spec.fixed_params:
        r0 = float(task_spec.fixed_params["r0"])
        if r0 < 5:
            _emit_warning(
                f"[AstroRange] 固定初始半径 r0={r0} 较小，更接近视界区域，可能更敏感。",
                warnings_list,
            )
        if r0 > 50:
            _emit_warning(
                f"[AstroRange] 固定初始半径 r0={r0} 较大，更偏弱场区，近黑洞特征会减弱。",
                warnings_list,
            )

    if "r0" in task_spec.vary_params:
        r_min, r_max = task_spec.vary_ranges["r0"]
        if r_min < 5:
            _emit_warning(
                f"[AstroRange] r0 的范围下界 r_min={r_min} 较小，更容易进入近视界敏感区。",
                warnings_list,
            )
        if r_max > 50:
            _emit_warning(
                f"[AstroRange] r0 的范围上界 r_max={r_max} 较大，更偏弱场区。",
                warnings_list,
            )


def _warn_theta_near_poles(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对 theta0 靠近极点做 warning。
    """
    import math

    pole_eps = 0.1

    if "theta0" in task_spec.fixed_params:
        theta0 = float(task_spec.fixed_params["theta0"])
        if theta0 < pole_eps or theta0 > math.pi - pole_eps:
            _emit_warning(
                f"[AstroRange] 固定 theta0={theta0} 过于接近极点，可能带来数值敏感性。",
                warnings_list,
            )

    if "theta0" in task_spec.vary_params:
        t_min, t_max = task_spec.vary_ranges["theta0"]
        if t_min < pole_eps or t_max > math.pi - pole_eps:
            _emit_warning(
                f"[AstroRange] theta0 的范围 ({t_min}, {t_max}) 过于接近极点区域，可能带来数值敏感性。",
                warnings_list,
            )


def _warn_lz_range(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对 Lz 做经验 warning。

    这里只给项目当前风格下的建议范围提示，不做硬性限制。
    """
    ref_min, ref_max = 2.0, 4.0

    if "Lz" in task_spec.fixed_params:
        Lz = float(task_spec.fixed_params["Lz"])
        if Lz < ref_min or Lz > ref_max:
            _emit_warning(
                f"[AstroRange] 固定 Lz={Lz} 超出了当前项目建议首试范围 [{ref_min}, {ref_max}]，"
                "这不一定错误，但可能更难生成干净数据。",
                warnings_list,
            )

    if "Lz" in task_spec.vary_params:
        Lz_min, Lz_max = task_spec.vary_ranges["Lz"]
        if Lz_min < ref_min or Lz_max > ref_max:
            _emit_warning(
                f"[AstroRange] Lz 的范围 ({Lz_min}, {Lz_max}) 超出了当前项目建议首试范围 "
                f"[{ref_min}, {ref_max}]，可能更难保持数据集稳定。",
                warnings_list,
            )


def _warn_q_range(task_spec: TaskSpec, warnings_list: list[str]) -> None:
    """
    对 Q 做项目经验 warning。
    """
    ref_min, ref_max = 1.4, 3.0

    if "Q" in task_spec.fixed_params:
        Q = float(task_spec.fixed_params["Q"])
        if Q < ref_min or Q > ref_max:
            _emit_warning(
                f"[AstroRange] 固定 Q={Q} 超出了当前项目常用可行范围 [{ref_min}, {ref_max}]，"
                "这不一定错误，但可能更难得到稳定三维轨道。",
                warnings_list,
            )

    if "Q" in task_spec.vary_params:
        Q_min, Q_max = task_spec.vary_ranges["Q"]
        if Q_min < ref_min or Q_max > ref_max:
            _emit_warning(
                f"[AstroRange] Q 的范围 ({Q_min}, {Q_max}) 超出了当前项目常用可行范围 "
                f"[{ref_min}, {ref_max}]，可能降低成功样本率。",
                warnings_list,
            )