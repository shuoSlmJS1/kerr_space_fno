# ==========================================================
# File: src/data_generation/orbit_solver.py
#
# 功能简介：
# 1. 放置 Kerr 时空轨道积分的核心物理与数值函数；
# 2. 实现：
#    - Kerr 几何基础函数
#    - 势函数 R(r), Theta(theta)
#    - Mino 参数下的运动方程
#    - RK4 数值积分
#    - 轨道点到 xyz 的坐标转换
# 3. 提供统一接口 simulate_one_orbit()。
#
# 依赖关系：
# - 被 validity.py 使用
# - 被 dataset_builder.py 使用
#
# 重要说明：
# - 本文件是数据生成模块中的“物理核心”；
# - 只负责轨道数值积分与轨道输出；
# - 不负责现实范围 warning；
# - 不负责命令行解析；
# - 不负责数据集切分与保存。
# ==========================================================
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ==========================================================
# 一、基础参数数据结构
# ==========================================================

@dataclass
class KerrParams:
    """
    Kerr 黑洞背景参数 + 轨道守恒量。

    字段说明：
    - M   : 黑洞质量（当前项目里通常固定为 1）
    - a   : Kerr 自旋参数
    - E   : 单位质量能量守恒量
    - Lz  : 绕 z 轴的角动量守恒量

    说明：
    - 这里不包含 Q，因为在很多任务中 Q 是变化参数，
      更适合作为 simulate_one_orbit 的独立输入；
    - 这样后面单参数 / 双参数任务组合会更灵活。
    """
    M: float
    a: float
    E: float
    Lz: float


@dataclass
class InitialState:
    """
    轨道初始状态。

    字段说明：
    - r0       : 初始半径
    - theta0   : 初始极角（弧度）
    - phi0     : 初始方位角（弧度）
    - sign_r   : 初始径向方向，+1 向外，-1 向内
    - sign_th  : 初始极向方向，+1 / -1
    """
    r0: float
    theta0: float
    phi0: float
    sign_r: int
    sign_th: int


# ==========================================================
# 二、Kerr 几何相关基础函数
# ==========================================================

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

    说明：
    - 若 a > M，理论上不再是标准 Kerr 黑洞；
    - 这里仍使用 max(..., 0) 做数值保护，
      真正的物理合法性检查应在 validity.py 中完成。
    """
    return M + math.sqrt(max(M * M - a * a, 0.0))


# ==========================================================
# 三、势函数
# ==========================================================

def radial_potential(r: float, p: KerrParams, Q: float) -> float:
    """
    计算径向势函数 R(r)：

        R(r) = [ E(r^2+a^2) - aLz ]^2
               - Δ [ r^2 + (Lz-aE)^2 + Q ]
    """
    M, a, E, Lz = p.M, p.a, p.E, p.Lz
    term1 = E * (r * r + a * a) - a * Lz
    term2 = r * r + (Lz - a * E) ** 2 + Q
    return term1 * term1 - delta(r, M, a) * term2


def polar_potential(theta: float, p: KerrParams, Q: float) -> float:
    """
    计算极向势函数 Θ(theta)：

        Θ(theta) = Q
                   - cos^2(theta) [ a^2(1-E^2) + Lz^2 / sin^2(theta) ]
    """
    a, E, Lz = p.a, p.E, p.Lz
    s = math.sin(theta)
    c = math.cos(theta)
    s2 = max(s * s, 1e-12)
    return Q - (c * c) * (a * a * (1.0 - E * E) + (Lz * Lz) / s2)


def dphi_dlambda(r: float, theta: float, p: KerrParams) -> float:
    """
    计算 Mino 参数下的 dphi/dlambda：

        dphi/dlambda =
            a/Δ * [ E(r^2+a^2) - aLz ] + Lz/sin^2(theta) - aE
    """
    a, E, Lz = p.a, p.E, p.Lz
    dlt = delta(r, p.M, a)
    s = math.sin(theta)
    s2 = max(s * s, 1e-12)

    return (a / dlt) * (E * (r * r + a * a) - a * Lz) + (Lz / s2) - a * E


# ==========================================================
# 四、数值安全工具
# ==========================================================

def clip_theta(theta: float, eps: float = 1e-4) -> float:
    """
    对 theta 做安全裁剪，避免过于接近 0 或 pi，
    从而引发 sin(theta) 分母数值问题。
    """
    return min(max(theta, eps), math.pi - eps)


# ==========================================================
# 五、Mino 参数下的右端项
# ==========================================================

def geodesic_rhs(
    state: np.ndarray,
    p: KerrParams,
    Q: float,
    sign_r: int,
    sign_th: int,
) -> np.ndarray:
    """
    计算状态变量对 Mino 参数 lambda 的导数。

    状态向量定义：
        state = [r, theta, phi]

    返回：
        [dr/dlambda, dtheta/dlambda, dphi/dlambda]
    """
    r = float(state[0])
    theta = clip_theta(float(state[1]))
    phi = float(state[2])  # noqa: F841  # 当前不直接使用，但保留结构完整性

    R = radial_potential(r, p, Q)
    TH = polar_potential(theta, p, Q)

    dr = sign_r * math.sqrt(max(R, 0.0))
    dtheta = sign_th * math.sqrt(max(TH, 0.0))
    dphi = dphi_dlambda(r, theta, p)

    return np.array([dr, dtheta, dphi], dtype=np.float64)


# ==========================================================
# 六、RK4 积分
# ==========================================================

def rk4_step(
    state: np.ndarray,
    h: float,
    p: KerrParams,
    Q: float,
    sign_r: int,
    sign_th: int,
) -> np.ndarray:
    """
    用经典四阶 Runge-Kutta 方法做一步积分。
    """
    k1 = geodesic_rhs(state, p, Q, sign_r, sign_th)
    k2 = geodesic_rhs(state + 0.5 * h * k1, p, Q, sign_r, sign_th)
    k3 = geodesic_rhs(state + 0.5 * h * k2, p, Q, sign_r, sign_th)
    k4 = geodesic_rhs(state + h * k3, p, Q, sign_r, sign_th)

    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ==========================================================
# 七、坐标变换
# ==========================================================

def spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """
    将 (r, theta, phi) 转换为三维笛卡尔坐标 (x, y, z)。

    说明：
    - 当前仍沿用你项目前面的做法：
      直接用普通球坐标形式构造 xyz；
    - 这对应你当前机器学习任务里的输出定义。
    """
    st = math.sin(theta)
    x = r * st * math.cos(phi)
    y = r * st * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


# ==========================================================
# 八、单条轨道模拟
# ==========================================================

def simulate_one_orbit(
    p: KerrParams,
    init: InitialState,
    Q: float,
    n_steps: int,
    step_size: float,
    turn_tol_r: float = 1e-6,
    turn_tol_th: float = 1e-6,
) -> dict[str, np.ndarray]:
    """
    模拟单条 Kerr 轨道。

    输入：
    - p          : Kerr 参数与守恒量（不含 Q）
    - init       : 初始状态
    - Q          : Carter 常数
    - n_steps    : 轨道采样长度
    - step_size  : Mino 参数步长

    返回：
    {
        "lambda_grid": [T],
        "sph": [T,3],   # (r, theta, phi)
        "xyz": [T,3],   # (x, y, z)
    }

    说明：
    - 这里只负责数值积分与轨道输出；
    - 初始合法性检查、现实范围警告等，
      应在外层 validity.py / astrophysical_checks.py 中完成。
    """
    state = np.array([init.r0, init.theta0, init.phi0], dtype=np.float64)

    sign_r = int(np.sign(init.sign_r)) if init.sign_r != 0 else 1
    sign_th = int(np.sign(init.sign_th)) if init.sign_th != 0 else 1

    r_plus = outer_horizon_radius(p.M, p.a)

    lambda_grid = np.arange(n_steps, dtype=np.float64) * step_size
    sph = np.zeros((n_steps, 3), dtype=np.float64)
    xyz = np.zeros((n_steps, 3), dtype=np.float64)

    for i in range(n_steps):
        r = float(state[0])
        theta = clip_theta(float(state[1]))
        phi = float(state[2])

        # 1) 先存当前时刻轨道点
        sph[i] = np.array([r, theta, phi], dtype=np.float64)
        xyz[i] = spherical_to_cartesian(r, theta, phi)

        # 2) 最后一个点存完就结束
        if i == n_steps - 1:
            break

        # 3) 当前点若已经太接近视界，则判为失败
        if r <= r_plus + 1e-3:
            raise RuntimeError(
                f"轨道过于接近外视界：r={r:.6f}, r_plus={r_plus:.6f}, Q={Q:.6f}, a={p.a:.6f}"
            )

        # 4) 检查当前位置是否接近 turning point
        R_now = radial_potential(r, p, Q)
        TH_now = polar_potential(theta, p, Q)

        if R_now <= turn_tol_r:
            sign_r *= -1
        if TH_now <= turn_tol_th:
            sign_th *= -1

        # 5) 做一步 RK4
        next_state = rk4_step(state, step_size, p, Q, sign_r, sign_th)

        # 6) 基础数值检查
        if not np.all(np.isfinite(next_state)):
            raise RuntimeError(
                f"数值发散：出现 nan/inf, Q={Q:.6f}, a={p.a:.6f}"
            )

        # 7) 对 theta 做安全裁剪
        next_state[1] = clip_theta(float(next_state[1]))

        # 8) 如果下一步进入视界附近，则判失败
        if next_state[0] <= r_plus + 1e-4:
            raise RuntimeError(
                f"下一步进入视界附近，轨道不再视为安全束缚样本, Q={Q:.6f}, a={p.a:.6f}"
            )

        state = next_state

    return {
        "lambda_grid": lambda_grid,
        "sph": sph,
        "xyz": xyz,
    }