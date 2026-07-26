# ==========================================================
# File: src/data_generation/orbit_types.py
#
# 功能：
# 1. 定义 Kerr 轨道计算所需的公共参数数据结构；
# 2. 不包含任何具体数值积分算法；
# 3. 供二阶求解器、数据生成、合法性检查和验证脚本共同使用。
# ==========================================================
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KerrParams:
    """
    Kerr 黑洞背景参数与轨道守恒量。

    字段：
    - M  : 黑洞质量
    - a  : Kerr 自旋参数
    - E  : 单位质量能量守恒量
    - Lz : 绕 z 轴的角动量守恒量

    Q 单独作为求解器参数传入，因此不放在该数据类中。
    """

    M: float
    a: float
    E: float
    Lz: float


@dataclass
class InitialState:
    """
    轨道初始状态。

    字段：
    - r0      : 初始半径
    - theta0  : 初始极角
    - phi0    : 初始方位角
    - sign_r  : 初始径向运动方向
    - sign_th : 初始极向运动方向
    """

    r0: float
    theta0: float
    phi0: float
    sign_r: int
    sign_th: int
