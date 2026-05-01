# ==========================================================
# File: src/data_generation/parameter_parser.py
#
# 功能简介：
# 1. 为数据生成入口脚本注册命令行参数；
# 2. 解析用户输入的变化参数、固定参数、范围、样本形状等；
# 3. 校验输入格式是否合法；
# 4. 将 argparse 结果转换为 TaskSpec；
# 5. 提供数据生成脚本中的 help 说明文本。
#
# 依赖关系：
# - 依赖 common/task_spec.py
# - 依赖 common/naming.py 中的允许变化参数列表
# - 被 scripts/generate_dataset.py 调用
#
# 重要说明：
# - 本文件负责“输入解析与格式检查”；
# - 不负责现实范围 warning；
# - 不负责数学物理硬约束；
# - 不负责轨道积分与数据保存。
# ==========================================================
from __future__ import annotations

import argparse
from typing import Any

from src.common.naming import ALLOWED_VARY_PARAMS, normalize_param_order
from src.common.task_spec import TaskSpec


# ==========================================================
# 一、参数注册表
# ==========================================================

# 这里集中定义“当前系统允许作为固定参数或变化参数”的参数。
# 后面如果你想开放更多参数，只需要改这里和对应的物理逻辑即可。
ALLOWED_TASK_PARAMS: list[str] = [
    "M",
    "a",
    "E",
    "Lz",
    "Q",
    "r0",
    "theta0",
    "phi0",
    "sign_r",
    "sign_th",
]

# 这里定义哪些参数允许用作“扫描变化参数”。
# 当前版本不建议开放 M / sign_r / sign_th 作为变化参数，
# 因为它们要么更像尺度参数，要么更像离散标签参数。
DEFAULT_ALLOWED_VARY_PARAMS: list[str] = ALLOWED_VARY_PARAMS.copy()

# 每个参数的默认固定值。
# 这些值本身不代表“最佳物理设置”，而是你当前项目里常用、较稳定的一组基准值。
DEFAULT_FIXED_PARAMS: dict[str, Any] = {
    "M": 1.0,
    "a": 0.5,
    "E": 0.95,
    "Lz": 3.0,
    "Q": 2.0,
    "r0": 10.0,
    "theta0": 1.2,
    "phi0": 0.0,
    "sign_r": -1,
    "sign_th": 1,
}


# ==========================================================
# 二、帮助文本
# ==========================================================

def get_generate_dataset_help_text() -> str:
    """
    返回数据生成脚本的补充帮助信息。

    说明：
    - 这个函数不是 argparse 自动 help 的替代品；
    - 它用于在你想额外展示“当前允许变化参数有哪些、默认固定值是什么”时使用。
    """
    lines = [
        "Allowed vary params:",
        "  " + ", ".join(DEFAULT_ALLOWED_VARY_PARAMS),
        "",
        "Allowed task params:",
        "  " + ", ".join(ALLOWED_TASK_PARAMS),
        "",
        "Default fixed params:",
    ]

    for k, v in DEFAULT_FIXED_PARAMS.items():
        lines.append(f"  {k} = {v}")

    lines.extend([
        "",
        "Examples:",
        "  Single parameter:",
        "    --vary-params Q --Q-range 1.6 3.0 --sample-shape 240",
        "",
        "  Two parameters:",
        "    --vary-params Q a --Q-range 1.6 3.0 --a-range 0.4 0.7 --sample-shape 20 20",
    ])

    return "\n".join(lines)


# ==========================================================
# 三、给入口脚本构造 argparse 的辅助函数
# ==========================================================

def add_dataset_generation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    向 argparse parser 中添加数据生成相关参数。

    设计原则：
    - 这里只负责注册参数；
    - 不负责把参数转换成 TaskSpec；
    - TaskSpec 的构造在 parse_task_spec_from_args() 里进行。
    """
    # ------------------------------------------------------
    # A. 变化参数相关
    # ------------------------------------------------------
    parser.add_argument(
        "--vary-params",
        nargs="+",
        type=str,
        required=True,
        help=(
            "本次任务中要变化的参数列表。"
            f"允许值：{', '.join(DEFAULT_ALLOWED_VARY_PARAMS)}"
        ),
    )

    # ------------------------------------------------------
    # B. 各参数的范围输入
    # ------------------------------------------------------
    # 说明：
    # - 我们统一预注册所有可能用到的 range 参数；
    # - 真正是否需要，会在 parse_task_spec_from_args() 中根据 vary_params 再检查。
    parser.add_argument("--M-range", nargs=2, type=float, default=None, help="M 的变化范围：min max")
    parser.add_argument("--a-range", nargs=2, type=float, default=None, help="a 的变化范围：min max")
    parser.add_argument("--E-range", nargs=2, type=float, default=None, help="E 的变化范围：min max")
    parser.add_argument("--Lz-range", nargs=2, type=float, default=None, help="Lz 的变化范围：min max")
    parser.add_argument("--Q-range", nargs=2, type=float, default=None, help="Q 的变化范围：min max")
    parser.add_argument("--r0-range", nargs=2, type=float, default=None, help="r0 的变化范围：min max")
    parser.add_argument("--theta0-range", nargs=2, type=float, default=None, help="theta0 的变化范围：min max")
    parser.add_argument("--phi0-range", nargs=2, type=float, default=None, help="phi0 的变化范围：min max")

    # ------------------------------------------------------
    # C. 固定参数输入
    # ------------------------------------------------------
    # 说明：
    # - 所有固定参数都允许在命令行中覆写默认值；
    # - 如果某个参数出现在 vary_params 中，则它不能再作为 fixed 参数生效。
    parser.add_argument("--M", type=float, default=DEFAULT_FIXED_PARAMS["M"], help="固定的 M")
    parser.add_argument("--a", type=float, default=DEFAULT_FIXED_PARAMS["a"], help="固定的 a")
    parser.add_argument("--E", type=float, default=DEFAULT_FIXED_PARAMS["E"], help="固定的 E")
    parser.add_argument("--Lz", type=float, default=DEFAULT_FIXED_PARAMS["Lz"], help="固定的 Lz")
    parser.add_argument("--Q", type=float, default=DEFAULT_FIXED_PARAMS["Q"], help="固定的 Q")
    parser.add_argument("--r0", type=float, default=DEFAULT_FIXED_PARAMS["r0"], help="固定的 r0")
    parser.add_argument("--theta0", type=float, default=DEFAULT_FIXED_PARAMS["theta0"], help="固定的 theta0")
    parser.add_argument("--phi0", type=float, default=DEFAULT_FIXED_PARAMS["phi0"], help="固定的 phi0")
    parser.add_argument("--sign-r", type=int, default=DEFAULT_FIXED_PARAMS["sign_r"], help="固定的 sign_r")
    parser.add_argument("--sign-th", type=int, default=DEFAULT_FIXED_PARAMS["sign_th"], help="固定的 sign_th")
    
    # ------------------------------------------------------
    # D. 样本规模与积分设置
    # ------------------------------------------------------
    parser.add_argument(
        "--sample-shape",
        nargs="+",
        type=int,
        required=True,
        help=(
            "样本形状。"
            "单参数任务例如：240；双参数任务例如：20 20；"
            "其维度数必须与 vary_params 数量一致。"
        ),
    )

    parser.add_argument("--n-steps", type=int, default=800, help="轨道序列长度")
    parser.add_argument("--step-size", type=float, default=0.005, help="Mino 参数步长")

    # ------------------------------------------------------
    # E. 数据划分与随机种子
    # ------------------------------------------------------
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--config-tag", type=str, default="cfg1", help="固定参数/初始条件配置版本标签，例如 cfg1、cfg2、cfg3",)

    # ------------------------------------------------------
    # F. 采样方式
    # ------------------------------------------------------
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="grid",
        choices=["grid"],
        help="采样方式。当前版本先支持 grid。"
    )

    return parser


# ==========================================================
# 四、解析与校验
# ==========================================================

def parse_task_spec_from_args(args: argparse.Namespace) -> TaskSpec:
    """
    从 argparse 解析结果中构造 TaskSpec。

    这是本文件最核心的函数。
    它负责把：
        命令行参数  ->  规范化的任务描述对象 TaskSpec
    """
    vary_params = _parse_and_validate_vary_params(args.vary_params)
    vary_ranges = _extract_and_validate_vary_ranges(args, vary_params)
    fixed_params = _extract_fixed_params(args, vary_params)
    sample_shape = _parse_and_validate_sample_shape(args.sample_shape, len(vary_params))
    split_ratios = _parse_and_validate_split_ratios(args)

    task_spec = TaskSpec(
        vary_params=vary_params,
        vary_ranges=vary_ranges,
        fixed_params=fixed_params,
        sample_shape=sample_shape,
        n_steps=int(args.n_steps),
        step_size=float(args.step_size),
        split_ratios=split_ratios,
        seed=int(args.seed),
        config_tag=str(args.config_tag),
        sampling_mode=str(args.sampling_mode),
        metadata={},
    )
    return task_spec


def _parse_and_validate_vary_params(vary_params_raw: list[str]) -> list[str]:
    """
    解析并校验变化参数列表。

    校验内容：
    - 不能为空
    - 不能重复
    - 必须在允许变化参数列表中
    - 最终按规范顺序排序
    """
    if not vary_params_raw:
        raise ValueError("vary_params 不能为空。")

    if len(set(vary_params_raw)) != len(vary_params_raw):
        raise ValueError(f"变化参数列表中存在重复项：{vary_params_raw}")

    unknown = [p for p in vary_params_raw if p not in DEFAULT_ALLOWED_VARY_PARAMS]
    if unknown:
        raise ValueError(
            f"以下参数当前不允许作为变化参数：{unknown}。"
            f"允许值为：{DEFAULT_ALLOWED_VARY_PARAMS}"
        )

    return normalize_param_order(vary_params_raw)


def _extract_and_validate_vary_ranges(
    args: argparse.Namespace,
    vary_params: list[str],
) -> dict[str, tuple[float, float]]:
    """
    从 argparse 命令行结果中提取变化参数的范围，并做基础检查。

    例如：
    - vary_params = ["Q", "a"]
    则要求：
    - args.Q_range 不为空
    - args.a_range 不为空
    """
    vary_ranges: dict[str, tuple[float, float]] = {}

    for param_name in vary_params:
        attr_name = f"{param_name}_range".replace("-", "_")
        value = getattr(args, attr_name, None)

        if value is None:
            raise ValueError(
                f"变化参数 {param_name} 缺少范围输入。"
                f"你需要通过 --{param_name}-range min max 提供范围。"
            )

        if len(value) != 2:
            raise ValueError(
                f"参数 {param_name} 的范围输入必须是两个数，当前得到：{value}"
            )

        v_min, v_max = float(value[0]), float(value[1])
        if v_max <= v_min:
            raise ValueError(
                f"参数 {param_name} 的范围必须满足 max > min，当前得到：({v_min}, {v_max})"
            )

        vary_ranges[param_name] = (v_min, v_max)

    return vary_ranges


def _extract_fixed_params(
    args: argparse.Namespace,
    vary_params: list[str],
) -> dict[str, Any]:
    """
    提取固定参数。

    规则：
    - 所有 ALLOWED_TASK_PARAMS 中，不在 vary_params 里的参数，都视为 fixed_params；
    - sign_r / sign_th 来自命令行中的 sign-r / sign-th。
    """
    fixed_params: dict[str, Any] = {}

    for param_name in ALLOWED_TASK_PARAMS:
        if param_name in vary_params:
            continue

        if param_name == "sign_r":
            fixed_params["sign_r"] = int(args.sign_r)
        elif param_name == "sign_th":
            fixed_params["sign_th"] = int(args.sign_th)
        else:
            fixed_params[param_name] = getattr(args, param_name)

    return fixed_params


def _parse_and_validate_sample_shape(sample_shape_raw: list[int], num_vary_params: int) -> list[int]:
    """
    解析并校验 sample_shape。

    规则：
    - 单参数任务：sample_shape 长度应为 1，例如 [240]
    - 双参数任务：sample_shape 长度应为 2，例如 [20,20]
    - 维度数必须等于变化参数个数
    """
    if not sample_shape_raw:
        raise ValueError("sample_shape 不能为空。")

    sample_shape = [int(x) for x in sample_shape_raw]

    if len(sample_shape) != num_vary_params:
        raise ValueError(
            "sample_shape 的维度数必须等于变化参数个数："
            f"num_vary_params={num_vary_params}, sample_shape={sample_shape}"
        )

    for n in sample_shape:
        if n < 2:
            raise ValueError(
                f"sample_shape 中的每个维度至少应为 2，当前得到：{sample_shape}"
            )

    return sample_shape


def _parse_and_validate_split_ratios(args: argparse.Namespace) -> tuple[float, float, float]:
    """
    解析并校验 train / val / test 比例。
    """
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f"数据划分比例不能为负，当前得到："
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-12:
        raise ValueError(
            f"train/val/test 比例之和必须等于 1，当前得到："
            f"{train_ratio} + {val_ratio} + {test_ratio} = {s}"
        )

    return (train_ratio, val_ratio, test_ratio)
