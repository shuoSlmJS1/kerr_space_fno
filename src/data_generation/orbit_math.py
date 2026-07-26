# ==========================================================
# File: src/data_generation/orbit_math.py
#
# 功能：
# 1. 保存 Kerr 轨道计算中的公共数学公式；
# 2. 不包含任何具体积分算法；
# 3. 供二阶求解器、合法性检查和分析模块共同使用。
# ==========================================================
from __future__ import annotations

import math

import numpy as np

from src.data_generation.orbit_types import KerrParams


def sigma(r: float, theta: float, a: float) -> float:
    """
    计算：
        Σ = r^2 + a^2 cos^2(theta)
    """
    return r * r + a * a * (math.cos(theta) ** 2)


def delta(r: float, M: float, a: float) -> float:
    """
    计算：
        Δ = r^2 - 2Mr + a^2
    """
    return r * r - 2.0 * M * r + a * a


def outer_horizon_radius(M: float, a: float) -> float:
    """
    计算 Kerr 黑洞外视界半径：

        r_+ = M + sqrt(M^2 - a^2)

    若 a > M，这里仍使用数值保护；
    物理合法性由 validity.py 负责检查。
    """
    return M + math.sqrt(max(M * M - a * a, 0.0))


def radial_potential(r: float, p: KerrParams, Q: float) -> float:
    """
    计算径向势函数 R(r)：

        R(r) = [E(r^2+a^2)-aLz]^2
               - Δ[r^2+(Lz-aE)^2+Q]
    """
    M, a, E, Lz = p.M, p.a, p.E, p.Lz
    term1 = E * (r * r + a * a) - a * Lz
    term2 = r * r + (Lz - a * E) ** 2 + Q
    return term1 * term1 - delta(r, M, a) * term2


def polar_potential(theta: float, p: KerrParams, Q: float) -> float:
    """
    计算极向势函数 Θ(theta)：

        Θ(theta) = Q
                   - cos^2(theta)
                     [a^2(1-E^2)+Lz^2/sin^2(theta)]
    """
    a, E, Lz = p.a, p.E, p.Lz
    s = math.sin(theta)
    c = math.cos(theta)
    s2 = max(s * s, 1e-12)
    return Q - (c * c) * (
        a * a * (1.0 - E * E) + (Lz * Lz) / s2
    )


def dphi_dlambda(
    r: float,
    theta: float,
    p: KerrParams,
) -> float:
    """
    计算 Mino 参数下的 dphi/dlambda：

        dphi/dlambda =
            a/Δ [E(r^2+a^2)-aLz]
            + Lz/sin^2(theta) - aE
    """
    a, E, Lz = p.a, p.E, p.Lz
    dlt = delta(r, p.M, a)
    s = math.sin(theta)
    s2 = max(s * s, 1e-12)

    return (
        (a / dlt) * (E * (r * r + a * a) - a * Lz)
        + (Lz / s2)
        - a * E
    )


def clip_theta(theta: float, eps: float = 1e-4) -> float:
    """
    裁剪 theta，避免过于接近 0 或 pi。
    """
    return min(max(theta, eps), math.pi - eps)


def spherical_to_cartesian(
    r: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """
    将 (r, theta, phi) 转换为 (x, y, z)。
    """
    st = math.sin(theta)
    x = r * st * math.cos(phi)
    y = r * st * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z], dtype=np.float64)
