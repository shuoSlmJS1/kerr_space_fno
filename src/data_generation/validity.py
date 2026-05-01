# ==========================================================
# File: src/data_generation/validity.py
#
# 功能简介：
# 1. 提供数学/物理硬约束检查；
# 2. 检查 TaskSpec 级别的硬约束；
# 3. 检查单个具体样本的初值是否合法；
# 4. 检查：
#    - Kerr 自旋约束
#    - theta 合法范围
#    - 初始点是否在视界外
#    - 初始势函数是否为合法值
#
# 依赖关系：
# - 依赖 common/task_spec.py
# - 依赖 orbit_solver.py
# - 被 dataset_builder.py 调用
#
# 重要说明：
# - 这里的检查是“必须通过”的硬约束；
# - 不通过时应报错终止；
# - 现实天体范围 warning 不放在这里，而放在 astrophysical_checks.py。
# ==========================================================
from __future__ import annotations

import warnings
from typing import Any

from src.common.task_spec import TaskSpec
from src.data_generation.orbit_solver import (
    InitialState,
    KerrParams,
    clip_theta,
    outer_horizon_radius,
    polar_potential,
    radial_potential,
)


# ==========================================================
# 一、warning 类型
# ==========================================================

class AstrophysicalRangeWarning(UserWarning):
    """
    现实天体范围提醒。

    用途：
    - 提醒参数设置“可能偏离常见天体物理场景”
    - 不阻止程序运行
    """
    pass


# ==========================================================
# 二、TaskSpec 层面的硬约束检查
# ==========================================================

def validate_task_spec_hard_constraints(task_spec: TaskSpec) -> None:
    """
    对 TaskSpec 做数学/物理硬约束检查。

    这里检查的是“任务级别”的硬性条件，
    不依赖具体某个采样点。

    例如：
    - sample_shape 合法
    - n_steps / step_size 合法
    - a 的范围若作为变化参数，则必须满足 a <= M（当 M 固定时）
    """
    _validate_basic_numerics(task_spec)
    _validate_kerr_spin_bounds(task_spec)
    _validate_theta_ranges(task_spec)
    _validate_phi_ranges(task_spec)


def _validate_basic_numerics(task_spec: TaskSpec) -> None:
    """
    检查基本数值设置是否合法。
    """
    if task_spec.n_steps < 2:
        raise ValueError(f"n_steps 必须 >= 2，当前得到：{task_spec.n_steps}")

    if task_spec.step_size <= 0:
        raise ValueError(f"step_size 必须为正数，当前得到：{task_spec.step_size}")

    if len(task_spec.sample_shape) != len(task_spec.vary_params):
        raise ValueError(
            "sample_shape 维度数必须等于变化参数个数："
            f"vary_params={task_spec.vary_params}, sample_shape={task_spec.sample_shape}"
        )

    for n in task_spec.sample_shape:
        if n < 2:
            raise ValueError(f"sample_shape 中每一维至少应为 2，当前得到：{task_spec.sample_shape}")


def _validate_kerr_spin_bounds(task_spec: TaskSpec) -> None:
    """
    检查 Kerr 自旋参数的硬约束。

    标准 Kerr 黑洞要求：
        0 <= a <= M

    处理规则：
    - 如果 a 是固定参数，则直接检查 fixed_params["a"]
    - 如果 a 是变化参数，且 M 是固定参数，则检查 a_range 是否落在 [0, M] 内
    - 如果未来允许 M 也变化，则应进一步扩展联合检查逻辑
    """
    fixed_params = task_spec.fixed_params

    # ---------- M ----------
    if "M" in fixed_params:
        M = float(fixed_params["M"])
        if M <= 0:
            raise ValueError(f"黑洞质量 M 必须 > 0，当前得到：{M}")

    # ---------- a 固定 ----------
    if "a" in fixed_params:
        a = float(fixed_params["a"])
        M = float(fixed_params.get("M", 1.0))
        if a < 0:
            raise ValueError(f"Kerr 自旋参数 a 不能为负，当前得到：{a}")
        if a > M:
            raise ValueError(
                f"Kerr 黑洞要求 a <= M，当前得到：a={a}, M={M}"
            )

    # ---------- a 变化 ----------
    if "a" in task_spec.vary_params:
        a_min, a_max = task_spec.vary_ranges["a"]
        M = float(fixed_params.get("M", 1.0))

        if a_min < 0:
            raise ValueError(f"a 的范围下界不能为负，当前得到：a_min={a_min}")
        if a_max > M:
            raise ValueError(
                f"Kerr 黑洞要求 a <= M，当前得到 a_range=({a_min}, {a_max}), M={M}"
            )


def _validate_theta_ranges(task_spec: TaskSpec) -> None:
    """
    检查 theta0 的硬约束。

    数学上要求：
        0 < theta0 < pi

    这里对：
    - 固定 theta0
    - 变化 theta0
    都进行检查。
    """
    import math

    # 固定 theta0
    if "theta0" in task_spec.fixed_params:
        theta0 = float(task_spec.fixed_params["theta0"])
        if not (0.0 < theta0 < math.pi):
            raise ValueError(
                f"theta0 必须满足 0 < theta0 < pi，当前得到：theta0={theta0}"
            )

    # 变化 theta0
    if "theta0" in task_spec.vary_params:
        t_min, t_max = task_spec.vary_ranges["theta0"]
        if not (0.0 < t_min < t_max < math.pi):
            raise ValueError(
                "theta0 的范围必须满足 0 < theta0_min < theta0_max < pi，"
                f"当前得到：({t_min}, {t_max})"
            )


def _validate_phi_ranges(task_spec: TaskSpec) -> None:
    """
    检查 phi0 的硬约束。

    这里不强制要求一定在 [0, 2pi) 内，
    但建议固定在这一主值范围。
    所以这里只做“不是 NaN / inf 且范围有序”的基本检查。
    """
    # 固定 phi0
    if "phi0" in task_spec.fixed_params:
        phi0 = float(task_spec.fixed_params["phi0"])
        if not (-1e30 < phi0 < 1e30):
            raise ValueError(f"phi0 数值异常，当前得到：{phi0}")

    # 变化 phi0
    if "phi0" in task_spec.vary_params:
        p_min, p_max = task_spec.vary_ranges["phi0"]
        if p_max <= p_min:
            raise ValueError(f"phi0 范围必须满足 max > min，当前得到：({p_min}, {p_max})")


# ==========================================================
# 三、现实天体范围 warning 检查（任务级）
# ==========================================================

def warn_task_spec_astrophysical_ranges(task_spec: TaskSpec) -> None:
    """
    对 TaskSpec 做现实天体范围的 warning 检查。

    这些检查：
    - 不报错
    - 只发 warning
    - 主要帮助你快速识别“这个任务可能偏离常见天体物理场景”
    """
    _warn_spin_near_extremal(task_spec)
    _warn_energy_range(task_spec)
    _warn_radius_range(task_spec)
    _warn_theta_near_poles(task_spec)


def _warn_spin_near_extremal(task_spec: TaskSpec) -> None:
    """
    对高自旋做 warning。

    天体物理里虽然接近极限自旋不是绝对不可能，
    但太接近 1 往往更敏感。
    """
    if "a" in task_spec.fixed_params:
        a = float(task_spec.fixed_params["a"])
        if a >= 0.95:
            warnings.warn(
                f"固定自旋 a={a} 已经非常高，接近极限 Kerr 自旋，"
                "数值与物理解释上都可能更敏感。",
                AstrophysicalRangeWarning,
            )

    if "a" in task_spec.vary_params:
        a_min, a_max = task_spec.vary_ranges["a"]
        if a_max >= 0.95:
            warnings.warn(
                f"a 的范围上界 a_max={a_max} 已接近极限 Kerr 自旋，"
                "虽然允许，但更容易引入高自旋敏感性。",
                AstrophysicalRangeWarning,
            )


def _warn_energy_range(task_spec: TaskSpec) -> None:
    """
    对 E 的范围做 warning。

    对 timelike 束缚轨道，通常 E < 1 更常见。
    """
    if "E" in task_spec.fixed_params:
        E = float(task_spec.fixed_params["E"])
        if E >= 1.0:
            warnings.warn(
                f"固定能量 E={E} 已达到或超过 1，"
                "这通常不再对应典型束缚 timelike 轨道。",
                AstrophysicalRangeWarning,
            )

    if "E" in task_spec.vary_params:
        E_min, E_max = task_spec.vary_ranges["E"]
        if E_max >= 1.0:
            warnings.warn(
                f"E 的范围上界 E_max={E_max} 已达到或超过 1，"
                "这可能对应非束缚或更接近逃逸型轨道。",
                AstrophysicalRangeWarning,
            )


def _warn_radius_range(task_spec: TaskSpec) -> None:
    """
    对 r0 做 warning。

    这里只给经验提醒：
    - 太近：更容易数值不稳或掉进视界
    - 太远：轨道更接近弱场极限
    """
    if "r0" in task_spec.fixed_params:
        r0 = float(task_spec.fixed_params["r0"])
        if r0 < 5:
            warnings.warn(
                f"固定初始半径 r0={r0} 较小，可能更接近视界区域，"
                "更容易出现不稳定或非法轨道。",
                AstrophysicalRangeWarning,
            )
        if r0 > 50:
            warnings.warn(
                f"固定初始半径 r0={r0} 较大，轨道会更接近弱场区，"
                "与近黑洞强场研究场景可能偏离。",
                AstrophysicalRangeWarning,
            )

    if "r0" in task_spec.vary_params:
        r_min, r_max = task_spec.vary_ranges["r0"]
        if r_min < 5:
            warnings.warn(
                f"r0 的范围下界 r_min={r_min} 较小，可能更容易进入近视界敏感区。",
                AstrophysicalRangeWarning,
            )
        if r_max > 50:
            warnings.warn(
                f"r0 的范围上界 r_max={r_max} 较大，轨道可能更偏弱场区。",
                AstrophysicalRangeWarning,
            )


def _warn_theta_near_poles(task_spec: TaskSpec) -> None:
    """
    对 theta0 贴近极点做 warning。
    """
    import math

    pole_eps = 0.1

    if "theta0" in task_spec.fixed_params:
        theta0 = float(task_spec.fixed_params["theta0"])
        if theta0 < pole_eps or theta0 > math.pi - pole_eps:
            warnings.warn(
                f"固定 theta0={theta0} 过于接近极点，可能带来数值敏感性。",
                AstrophysicalRangeWarning,
            )

    if "theta0" in task_spec.vary_params:
        t_min, t_max = task_spec.vary_ranges["theta0"]
        if t_min < pole_eps or t_max > math.pi - pole_eps:
            warnings.warn(
                f"theta0 的范围 ({t_min}, {t_max}) 过于接近极点区域，"
                "可能带来数值敏感性。",
                AstrophysicalRangeWarning,
            )


# ==========================================================
# 四、单个样本级别的硬约束检查
# ==========================================================

def validate_single_sample_hard_constraints(
    sample_params: dict[str, Any],
    fixed_params: dict[str, Any],
) -> None:
    """
    对单个具体样本做硬约束检查。

    输入：
    - sample_params : 当前采样点中的变化参数值
    - fixed_params  : 当前任务的固定参数

    作用：
    - 将变化参数与固定参数合并成完整参数集
    - 对单个轨道样本的输入做合法性检查
    """
    full_params = _merge_params(sample_params, fixed_params)

    p = KerrParams(
        M=float(full_params["M"]),
        a=float(full_params["a"]),
        E=float(full_params["E"]),
        Lz=float(full_params["Lz"]),
    )

    init = InitialState(
        r0=float(full_params["r0"]),
        theta0=float(full_params["theta0"]),
        phi0=float(full_params["phi0"]),
        sign_r=int(full_params["sign_r"]),
        sign_th=int(full_params["sign_th"]),
    )

    Q = float(full_params["Q"])

    _validate_single_sample_initial_values(p, init, Q)
    _validate_single_sample_potentials(p, init, Q)


def _merge_params(
    sample_params: dict[str, Any],
    fixed_params: dict[str, Any],
) -> dict[str, Any]:
    """
    合并变化参数和固定参数，形成完整参数字典。
    """
    merged = dict(fixed_params)
    merged.update(sample_params)
    return merged


def _validate_single_sample_initial_values(
    p: KerrParams,
    init: InitialState,
    Q: float,
) -> None:
    """
    检查单样本的基础初值是否合法。
    """
    import math

    if p.M <= 0:
        raise ValueError(f"M 必须 > 0，当前得到：{p.M}")

    if p.a < 0:
        raise ValueError(f"a 不能为负，当前得到：{p.a}")

    if p.a > p.M:
        raise ValueError(f"Kerr 黑洞要求 a <= M，当前得到：a={p.a}, M={p.M}")

    if not (0.0 < init.theta0 < math.pi):
        raise ValueError(f"theta0 必须满足 0 < theta0 < pi，当前得到：{init.theta0}")

    if init.r0 <= 0:
        raise ValueError(f"r0 必须 > 0，当前得到：{init.r0}")

    r_plus = outer_horizon_radius(p.M, p.a)
    if init.r0 <= r_plus:
        raise ValueError(
            f"初始半径必须在外视界之外，当前得到 r0={init.r0}, r_plus={r_plus}"
        )

    if init.sign_r not in (-1, 1):
        raise ValueError(f"sign_r 必须取 ±1，当前得到：{init.sign_r}")

    if init.sign_th not in (-1, 1):
        raise ValueError(f"sign_th 必须取 ±1，当前得到：{init.sign_th}")

    if not (-1e30 < Q < 1e30):
        raise ValueError(f"Q 数值异常，当前得到：{Q}")


def _validate_single_sample_potentials(
    p: KerrParams,
    init: InitialState,
    Q: float,
) -> None:
    """
    检查初始点的势函数是否合法。

    若：
    - R(r0) < 0
    - Theta(theta0) < 0

    则说明当前样本不是合法初始点。
    """
    theta0 = clip_theta(init.theta0)

    R0 = radial_potential(init.r0, p, Q)
    TH0 = polar_potential(theta0, p, Q)

    if R0 < -1e-8:
        raise ValueError(
            f"初始 R(r0) < 0，不是合法初始点：R0={R0:.6e}, "
            f"r0={init.r0}, a={p.a}, E={p.E}, Lz={p.Lz}, Q={Q}"
        )

    if TH0 < -1e-8:
        raise ValueError(
            f"初始 Theta(theta0) < 0，不是合法初始点：TH0={TH0:.6e}, "
            f"theta0={theta0}, a={p.a}, E={p.E}, Lz={p.Lz}, Q={Q}"
        )