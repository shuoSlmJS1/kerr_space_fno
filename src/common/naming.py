# ==========================================================
# File: src/common/naming.py
#
# 功能简介：
# 1. 统一生成 task_name；
# 2. 统一生成 model_name；
# 3. 提供参数顺序规范化逻辑，避免参数顺序不同导致命名混乱；
# 4. 提供允许变化参数列表；
# 5. 提供 help 文本中会用到的参数说明信息。
#
# 依赖关系：
# - 依赖 common/task_spec.py
# - 被 scripts/generate_dataset.py 使用
# - 被 scripts/train_model.py 使用
# - 被后续输出目录组织逻辑使用
#
# 重要说明：
# - task_name 必须体现：
#   变化参数 + 参数范围 + 样本形状 + 轨道长度
# - model_name 必须体现：
#   模型结构参数，而不是 simple small/large 这种模糊名字
# - 本文件只负责命名，不负责路径创建、不负责保存文件。
# ==========================================================
from __future__ import annotations

from typing import Any

from src.common.task_spec import TaskSpec


# ==========================================================
# 一、可公开给用户的参数注册表
# ==========================================================

# 这里定义“允许作为变化参数”的参数列表。
# 后续 generate_dataset.py 的 --help 可以直接引用这里。
ALLOWED_VARY_PARAMS: list[str] = [
    "a",
    "E",
    "Lz",
    "Q",
    "r0",
    "theta0",
    "phi0",
]

# 这里定义参数的全局规范顺序。
# 作用：
# - 保证 task_name 中参数顺序稳定；
# - 避免用户输入 ["Q", "a"] 和 ["a", "Q"] 时生成不同名字。
PARAM_ORDER: list[str] = [
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


# ==========================================================
# 二、基础工具函数
# ==========================================================

def normalize_param_order(param_names: list[str]) -> list[str]:
    """
    按 PARAM_ORDER 对参数名进行规范化排序。

    参数：
    - param_names: 用户给出的参数名列表

    返回：
    - 排序后的参数名列表

    说明：
    - 这样可以保证：
        ["Q", "a"] 和 ["a", "Q"]
      最终都会统一成同一个顺序。
    - 如果出现未在 PARAM_ORDER 中定义的参数，则报错，
      防止命名系统失控。
    """
    unknown = [p for p in param_names if p not in PARAM_ORDER]
    if unknown:
        raise ValueError(
            f"以下参数未在 PARAM_ORDER 中注册，无法生成规范命名：{unknown}"
        )

    order_index = {name: i for i, name in enumerate(PARAM_ORDER)}
    return sorted(param_names, key=lambda x: order_index[x])


def format_float_for_name(value: float, precision: int = 6) -> str:
    """
    将浮点数格式化为适合放进目录名/文件名的字符串。

    设计目标：
    - 尽量短
    - 保留足够信息
    - 避免目录名里出现多余空格
    - 保持跨平台文件名安全

    示例：
    - 0.4    -> "0.4"
    - 0.7000 -> "0.7"
    - 1.6000 -> "1.6"
    - 0.005  -> "0.005"

    注意：
    - 这里保留小数点，不替换成其他字符；
    - Windows/Linux 路径都允许普通小数点。
    """
    s = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def format_range_for_name(param_name: str, value_range: tuple[float, float]) -> str:
    """
    将单个参数范围格式化为命名片段。

    示例：
    - ("Q", (1.6, 3.0)) -> "Q1.6_3"
    - ("a", (0.4, 0.7)) -> "a0.4_0.7"
    """
    v_min, v_max = value_range
    return f"{param_name}{format_float_for_name(v_min)}_{format_float_for_name(v_max)}"


def format_sample_shape_for_name(sample_shape: list[int]) -> str:
    """
    将 sample_shape 格式化为命名片段。

    规则：
    - [240]      -> "n240"
    - [20, 20]   -> "n20x20"
    - [10,12,8]  -> "n10x12x8"

    说明：
    - 命名时必须保留“维度结构”，不能只写总样本数；
    - 因为 [240] 和 [15,16] 虽然总数相近，但任务结构完全不同。
    """
    return "n" + "x".join(str(x) for x in sample_shape)


# ==========================================================
# 三、TaskSpec 相关命名
# ==========================================================

def build_vary_part(task_spec: TaskSpec) -> str:
    """
    根据 TaskSpec 生成变化参数片段。

    示例：
    - vary_params = ["Q"]        -> "vary_Q"
    - vary_params = ["a"]        -> "vary_a"
    - vary_params = ["Q", "a"]   -> "vary_a_Q"   （按规范顺序）
    - vary_params = ["E", "Lz"]  -> "vary_E_Lz"
    """
    ordered = normalize_param_order(task_spec.vary_params)
    return "vary_" + "_".join(ordered)


def build_range_part(task_spec: TaskSpec) -> str:
    """
    根据 TaskSpec 生成范围片段。

    示例：
    - {"Q": (1.6, 3.0)} -> "Q1.6_3"
    - {"Q": (1.6, 3.0), "a": (0.4, 0.7)} -> "a0.4_0.7__Q1.6_3"

    说明：
    - 这里也按规范顺序输出；
    - 多个参数之间用双下划线 "__" 分开，增强可读性。
    """
    ordered = normalize_param_order(task_spec.vary_params)

    parts = []
    for p in ordered:
        parts.append(format_range_for_name(p, task_spec.vary_ranges[p]))

    return "__".join(parts)


def build_task_name(task_spec: TaskSpec) -> str:
    """
    根据 TaskSpec 生成完整 task_name。

    统一格式：
        vary_<vary_part>__<range_part>__n<sample_shape>__T<n_steps>__<config_tag>

    示例：
    - Q-only:
        vary_Q__Q1.6_3__n240__T800__cfg1

    - a-only:
        vary_a__a0.4_0.7__n240__T800__cfg1

    - QA:
        vary_a_Q__a0.4_0.7__Q1.6_3__n20x20__T800__cfg1

    说明：
    - task_name 用来唯一标识“数据任务”；
    - 只要变化参数、范围、样本规格、轨道长度有变化，
      task_name 就应当变化；
    - 这样才能保证不同任务的数据与输出不混淆。
    """
    vary_part = build_vary_part(task_spec)
    range_part = build_range_part(task_spec)
    sample_part = format_sample_shape_for_name(task_spec.sample_shape)
    time_part = f"T{task_spec.n_steps}"

    return f"{vary_part}__{range_part}__{sample_part}__{time_part}__{task_spec.config_tag}"


# ==========================================================
# 四、模型命名
# ==========================================================

def build_fno_model_name(modes: int, width: int, depth: int) -> str:
    """
    生成 FNO 模型名。

    格式：
        fno_m<modes>_w<width>_d<depth>

    示例：
    - modes=32, width=64, depth=4  -> fno_m32_w64_d4
    - modes=48, width=96, depth=5  -> fno_m48_w96_d5
    """
    return f"fno_m{modes}_w{width}_d{depth}"


def build_cnn_model_name(
    width: int,
    depth: int,
    kernel_size: int,
    lr: float | None = None,
) -> str:
    """
    生成 CNN 模型名。

    基本格式：
        cnn_w<width>_d<depth>_k<kernel_size>

    如果学习率需要体现在名字里，则扩展为：
        cnn_w<width>_d<depth>_k<kernel_size>_lr<...>

    示例：
    - width=64, depth=6, kernel_size=9
        -> cnn_w64_d6_k9

    - width=128, depth=10, kernel_size=9, lr=5e-4
        -> cnn_w128_d10_k9_lr5e-4
    """
    name = f"cnn_w{width}_d{depth}_k{kernel_size}"

    if lr is not None:
        name += f"_lr{lr:g}"

    return name


def build_generic_model_name(model_type: str, **kwargs: Any) -> str:
    """
    通用模型命名入口。

    用途：
    - 在 train_model.py 中统一调用；
    - 根据 model_type 自动路由到具体模型命名函数。

    当前支持：
    - "fno"
    - "cnn"

    后续可扩展：
    - "mlp"
    - "transformer"
    - 其他 operator learning 模型
    """
    model_type = model_type.lower()

    if model_type == "fno":
        required = ["modes", "width", "depth"]
        _check_required_kwargs(model_type, kwargs, required)
        return build_fno_model_name(
            modes=int(kwargs["modes"]),
            width=int(kwargs["width"]),
            depth=int(kwargs["depth"]),
        )

    if model_type == "cnn":
        required = ["width", "depth", "kernel_size"]
        _check_required_kwargs(model_type, kwargs, required)
        return build_cnn_model_name(
            width=int(kwargs["width"]),
            depth=int(kwargs["depth"]),
            kernel_size=int(kwargs["kernel_size"]),
            lr=kwargs.get("lr", None),
        )

    raise ValueError(f"暂不支持的 model_type: {model_type!r}")


def _check_required_kwargs(model_type: str, kwargs: dict[str, Any], required: list[str]) -> None:
    """
    检查某个模型命名函数所需参数是否齐全。
    """
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(
            f"生成 {model_type} 模型名时缺少必要参数：{missing}，当前 kwargs={kwargs}"
        )


# ==========================================================
# 五、用户帮助信息
# ==========================================================

def get_allowed_vary_params_help_text() -> str:
    """
    返回“可作为变化参数的参数列表”帮助文本。

    用途：
    - scripts/generate_dataset.py 中的 --help
    - 或者用户显式请求查看支持的变化参数时输出

    说明：
    - 这里不在平时自动打印；
    - 只在 help 场景使用，避免输出冗余。
    """
    lines = [
        "Allowed vary params:",
        "  " + ", ".join(ALLOWED_VARY_PARAMS),
        "",
        "Recommended current research params:",
        "  a, E, Lz, Q, r0, theta0",
    ]
    return "\n".join(lines)
