# ==========================================================
# File: src/data_generation/orbit_solver_second_order.py
#
# 研究目的
# --------
# 将原来的平方根一阶方程
#
#     (dr/dlambda)^2 = R(r)
#     (dtheta/dlambda)^2 = Theta(theta)
#
# 改写成二阶方程
#
#     d2r/dlambda2 = 1/2 * dR/dr
#     d2theta/dlambda2 = 1/2 * dTheta/dtheta
#
# 并将其写成一阶系统：
#
#     r'      = v_r
#     v_r'    = 1/2 R'(r)
#     theta'  = v_theta
#     v_theta'= 1/2 Theta'(theta)
#     phi'    = Phi(r, theta)
#
# 这样，转向点由速度 v_r 或 v_theta 自然穿过 0 实现，不再需要：
#
# - sign_r / sign_th 的人工翻转；
# - turning-point event detection；
# - coordinate_nudge。
#
# 重要限制
# --------
# 由平方关系求导得到二阶方程时，理论上依赖初始条件满足：
#
#     v_r(0)^2 = R(r0)
#     v_theta(0)^2 = Theta(theta0)
#
# 精确解会保持这些第一积分；数值解可能产生约束漂移。因此本模块
# 会记录：
#
#     |v_r^2 - R(r)|
#     |v_theta^2 - Theta(theta)|
#
# 这些残差是判断二阶求解器是否可信的核心诊断量。
#
# 本文件仍为实验版，不得直接覆盖正式数据生成器。
# ==========================================================
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.data_generation.orbit_types import InitialState, KerrParams
from src.data_generation.orbit_math import (
    clip_theta,
    delta,
    dphi_dlambda,
    outer_horizon_radius,
    polar_potential,
    radial_potential,
    spherical_to_cartesian,
)


@dataclass
class SecondOrderDiagnostics:
    """
    二阶系统数值诊断。

    max_radial_constraint_residual:
        max |v_r^2 - R(r)|

    max_polar_constraint_residual:
        max |v_theta^2 - Theta(theta)|

    radial_velocity_zero_crossings / polar_velocity_zero_crossings:
        速度符号改变次数，用作转向次数的近似计数。
    """

    max_radial_constraint_residual: float = 0.0
    max_polar_constraint_residual: float = 0.0
    mean_radial_constraint_residual: float = 0.0
    mean_polar_constraint_residual: float = 0.0
    radial_velocity_zero_crossings: int = 0
    polar_velocity_zero_crossings: int = 0
    min_radial_potential: float = math.inf
    min_polar_potential: float = math.inf


def radial_potential_derivative(
    r: float,
    p: KerrParams,
    Q: float,
) -> float:
    """
    计算 dR/dr。

    记：
        A(r) = E(r^2+a^2) - aLz
        B(r) = r^2 + (Lz-aE)^2 + Q
        Delta = r^2 - 2Mr + a^2

    则：
        R = A^2 - Delta B

        R' = 4Er A - (2r-2M)B - 2r Delta
    """
    M, a, E, Lz = p.M, p.a, p.E, p.Lz

    A = E * (r * r + a * a) - a * Lz
    B = r * r + (Lz - a * E) ** 2 + Q
    d_delta = 2.0 * r - 2.0 * M

    return (
        4.0 * E * r * A
        - d_delta * B
        - 2.0 * r * delta(r, M, a)
    )


def polar_potential_derivative(
    theta: float,
    p: KerrParams,
    Q: float,  # noqa: ARG001 -- 保留接口对称性
) -> float:
    """
    计算 dTheta/dtheta。

    Theta(theta) =
        Q - a^2(1-E^2) cos^2(theta) - Lz^2 cot^2(theta)

    因而：
        Theta' =
            2 a^2(1-E^2) sin(theta) cos(theta)
            + 2 Lz^2 cos(theta) / sin^3(theta)
    """
    theta = clip_theta(theta)
    s = math.sin(theta)
    c = math.cos(theta)

    a, E, Lz = p.a, p.E, p.Lz
    first = 2.0 * a * a * (1.0 - E * E) * s * c
    second = 2.0 * Lz * Lz * c / max(s**3, 1e-18)
    return first + second


def second_order_rhs(
    state: np.ndarray,
    p: KerrParams,
    Q: float,
) -> np.ndarray:
    """
    二阶系统的一阶化右端项。

    状态：
        [r, v_r, theta, v_theta, phi]
    """
    r = float(state[0])
    v_r = float(state[1])
    theta = clip_theta(float(state[2]))
    v_theta = float(state[3])

    acceleration_r = 0.5 * radial_potential_derivative(r, p, Q)
    acceleration_theta = 0.5 * polar_potential_derivative(theta, p, Q)
    phi_rate = dphi_dlambda(r, theta, p)

    return np.array(
        [
            v_r,
            acceleration_r,
            v_theta,
            acceleration_theta,
            phi_rate,
        ],
        dtype=np.float64,
    )


def rk4_step_second_order(
    state: np.ndarray,
    h: float,
    p: KerrParams,
    Q: float,
) -> np.ndarray:
    """对五维二阶系统做一个经典 RK4 步。"""
    k1 = second_order_rhs(state, p, Q)
    k2 = second_order_rhs(state + 0.5 * h * k1, p, Q)
    k3 = second_order_rhs(state + 0.5 * h * k2, p, Q)
    k4 = second_order_rhs(state + h * k3, p, Q)

    return state + (h / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


def _count_velocity_zero_crossings(
    previous_velocity: float,
    current_velocity: float,
    zero_tolerance: float = 1e-12,
) -> int:
    """
    判断速度是否跨过 0。

    这里不把单独落在极小零值上的点重复计数，只有前后符号明确相反时
    才计作一次转向。
    """
    if (
        abs(previous_velocity) <= zero_tolerance
        or abs(current_velocity) <= zero_tolerance
    ):
        return 0
    return int(previous_velocity * current_velocity < 0.0)


def simulate_one_orbit_second_order(
    p: KerrParams,
    init: InitialState,
    Q: float,
    n_steps: int,
    step_size: float,
) -> dict[str, object]:
    """
    用二阶动力系统模拟一条轨道。

    返回：
    {
        "lambda_grid": [T],
        "sph": [T,3],
        "velocities": [T,2],   # [v_r, v_theta]
        "xyz": [T,3],
        "diagnostics": SecondOrderDiagnostics
    }
    """
    R0 = radial_potential(init.r0, p, Q)
    TH0 = polar_potential(init.theta0, p, Q)

    if R0 < -1e-12:
        raise RuntimeError(f"初始径向势为负：R0={R0:.6e}")
    if TH0 < -1e-12:
        raise RuntimeError(f"初始极向势为负：Theta0={TH0:.6e}")

    initial_vr = (
        (1 if init.sign_r >= 0 else -1)
        * math.sqrt(max(R0, 0.0))
    )
    initial_vtheta = (
        (1 if init.sign_th >= 0 else -1)
        * math.sqrt(max(TH0, 0.0))
    )

    state = np.array(
        [
            init.r0,
            initial_vr,
            init.theta0,
            initial_vtheta,
            init.phi0,
        ],
        dtype=np.float64,
    )

    lambda_grid = np.arange(n_steps, dtype=np.float64) * step_size
    sph = np.zeros((n_steps, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, 2), dtype=np.float64)
    xyz = np.zeros((n_steps, 3), dtype=np.float64)

    radial_residuals = np.zeros(n_steps, dtype=np.float64)
    polar_residuals = np.zeros(n_steps, dtype=np.float64)

    diagnostics = SecondOrderDiagnostics()
    r_plus = outer_horizon_radius(p.M, p.a)

    previous_vr = float(state[1])
    previous_vtheta = float(state[3])

    for index in range(n_steps):
        r = float(state[0])
        v_r = float(state[1])
        theta = clip_theta(float(state[2]))
        v_theta = float(state[3])
        phi = float(state[4])

        R_value = radial_potential(r, p, Q)
        TH_value = polar_potential(theta, p, Q)

        radial_residual = abs(v_r * v_r - R_value)
        polar_residual = abs(v_theta * v_theta - TH_value)

        radial_residuals[index] = radial_residual
        polar_residuals[index] = polar_residual

        diagnostics.min_radial_potential = min(
            diagnostics.min_radial_potential,
            R_value,
        )
        diagnostics.min_polar_potential = min(
            diagnostics.min_polar_potential,
            TH_value,
        )

        if index > 0:
            diagnostics.radial_velocity_zero_crossings += (
                _count_velocity_zero_crossings(previous_vr, v_r)
            )
            diagnostics.polar_velocity_zero_crossings += (
                _count_velocity_zero_crossings(
                    previous_vtheta,
                    v_theta,
                )
            )

        sph[index] = np.array([r, theta, phi], dtype=np.float64)
        velocities[index] = np.array(
            [v_r, v_theta],
            dtype=np.float64,
        )
        xyz[index] = spherical_to_cartesian(r, theta, phi)

        if index == n_steps - 1:
            break

        if r <= r_plus + 1e-3:
            raise RuntimeError(
                f"轨道过于接近外视界：r={r:.6f}, "
                f"r_plus={r_plus:.6f}"
            )

        previous_vr = v_r
        previous_vtheta = v_theta

        state = rk4_step_second_order(
            state=state,
            h=step_size,
            p=p,
            Q=Q,
        )

        if not np.all(np.isfinite(state)):
            raise RuntimeError("二阶求解器出现 NaN 或 Inf。")

        state[2] = clip_theta(float(state[2]))

        if state[0] <= r_plus + 1e-4:
            raise RuntimeError("二阶求解器下一步进入视界安全边界。")

    diagnostics.max_radial_constraint_residual = float(
        np.max(radial_residuals)
    )
    diagnostics.max_polar_constraint_residual = float(
        np.max(polar_residuals)
    )
    diagnostics.mean_radial_constraint_residual = float(
        np.mean(radial_residuals)
    )
    diagnostics.mean_polar_constraint_residual = float(
        np.mean(polar_residuals)
    )

    return {
        "lambda_grid": lambda_grid,
        "sph": sph,
        "velocities": velocities,
        "xyz": xyz,
        "radial_constraint_residual": radial_residuals,
        "polar_constraint_residual": polar_residuals,
        "diagnostics": diagnostics,
    }
