# ==========================================================
# File: src/common/task_spec.py
#
# 功能简介：
# 1. 定义 TaskSpec 任务描述对象；
# 2. 统一描述一次数据生成/训练/分析任务的核心信息；
# 3. 保存：
#    - 哪些参数变化（vary_params）
#    - 各变化参数范围（vary_ranges）
#    - 固定参数（fixed_params）
#    - 样本形状（sample_shape）
#    - 轨道长度与步长（n_steps, step_size）
# 4. 提供基础结构合法性检查；
# 5. 为后续 naming / paths / sampler / dataset_builder 提供统一输入。
#
# 依赖关系：
# - 被 common/naming.py 使用
# - 被 data_generation/sampler.py 使用
# - 被 data_generation/parameter_parser.py 构造
# - 被后续训练与分析模块复用
#
# 重要说明：
# - TaskSpec 是整个项目的“任务身份证”；
# - 它不负责现实物理范围 warning；
# - 它也不负责用户交互；
# - 它主要负责“任务描述结构是否自洽”。
# ==========================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    """
    任务描述对象。

    作用：
    1. 统一描述“这次到底在做什么数据生成任务”；
    2. 统一保存：
       - 哪些参数变化
       - 每个变化参数的范围
       - 其余固定参数
       - 样本规模
       - 积分长度与步长
    3. 后续会被：
       - naming.py 用来生成 task_name
       - paths.py 用来生成目录
       - sampler.py 用来采样
       - dataset_builder.py 用来构建数据集

    设计原则：
    - 任务类型不写死，不预设 Q-only / a-only / QA 这类固定任务；
    - 一切都由 vary_params + vary_ranges + fixed_params 决定；
    - 只要参数组合变了、范围变了、样本规模变了，就应该视为新任务。
    """

    # ==========================================================
    # 一、核心任务字段
    # ==========================================================

    vary_params: list[str]
    """
    变化参数列表。

    示例：
    - ["Q"]
    - ["a"]
    - ["Q", "a"]
    - ["E", "Lz"]

    说明：
    - 顺序建议在外部统一规范化后再传入；
    - 不建议在不同地方随意改变顺序，否则 task_name 会不一致。
    """

    vary_ranges: dict[str, tuple[float, float]]
    """
    变化参数范围字典。

    示例：
    {
        "Q": (1.6, 3.0),
        "a": (0.4, 0.7),
    }

    说明：
    - 这里只有“变化参数”才应该出现；
    - fixed_params 中的固定参数不应重复写到这里。
    """

    fixed_params: dict[str, Any]
    """
    固定参数字典。

    示例：
    {
        "M": 1.0,
        "E": 0.95,
        "Lz": 3.0,
        "r0": 10.0,
        "theta0": 1.2,
        "phi0": 0.0,
        "sign_r": -1,
        "sign_th": 1,
    }
    """

    sample_shape: list[int]
    """
    样本形状描述。

    约定：
    - 单参数任务：用 [N]
      例如 [240]
    - 双参数任务：用 [N1, N2]
      例如 [20, 20]
    - 三参数任务：以后可扩展为 [N1, N2, N3]

    说明：
    - 之所以不用单独的 num_samples，是为了兼容多维网格采样。
    """

    n_steps: int
    """
    每条轨道序列的长度。
    例如：800
    """

    step_size: float
    """
    Mino 参数 lambda 的积分步长。
    例如：0.005
    """

    # ==========================================================
    # 二、可选附加字段
    # ==========================================================

    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    """
    数据集划分比例：(train, val, test)
    默认使用 7:1.5:1.5
    """

    config_tag: str = "cfg1"
    """
    固定参数/初始条件配置版本标签。
    用于区分不同固定配置下生成的数据任务。
    例如：cfg1、cfg2、cfg3
    """

    seed: int = 42
    """
    随机种子。
    后续用于：
    - 数据切分
    - 随机采样（如果启用随机采样）
    """

    sampling_mode: str = "grid"
    """
    采样模式。

    当前先支持：
    - "grid"   : 网格采样
    后续可扩展：
    - "random" : 随机采样
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    额外元信息。

    用途：
    - 给将来扩展预留接口
    - 例如记录说明文字、版本号、标签等
    """

    # ==========================================================
    # 三、基础校验
    # ==========================================================

    def __post_init__(self) -> None:
        """
        对 TaskSpec 做基础合法性检查。
        这里只做“结构级别”的检查，不做物理合法性检查。
        物理合法性检查应在 validity.py 中完成。
        """
        self._validate_vary_params()
        self._validate_vary_ranges()
        self._validate_fixed_params()
        self._validate_sample_shape()
        self._validate_numerical_setup()
        self._validate_split_ratios()

    def _validate_vary_params(self) -> None:
        """
        检查变化参数列表是否合法。
        """
        if not self.vary_params:
            raise ValueError("vary_params 不能为空。")

        if len(set(self.vary_params)) != len(self.vary_params):
            raise ValueError(f"vary_params 中存在重复参数：{self.vary_params}")

        for name in self.vary_params:
            if not isinstance(name, str):
                raise TypeError(f"变化参数名必须是字符串，当前得到：{name!r}")

    def _validate_vary_ranges(self) -> None:
        """
        检查变化参数范围是否合法。
        """
        missing = [p for p in self.vary_params if p not in self.vary_ranges]
        if missing:
            raise ValueError(f"以下变化参数缺少范围定义：{missing}")

        extra = [p for p in self.vary_ranges if p not in self.vary_params]
        if extra:
            raise ValueError(f"vary_ranges 中存在未声明为变化参数的键：{extra}")

        for name, value in self.vary_ranges.items():
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    f"参数 {name} 的范围必须是长度为 2 的 tuple，例如 (min, max)，当前得到：{value!r}"
                )

            v_min, v_max = value
            if not isinstance(v_min, (int, float)) or not isinstance(v_max, (int, float)):
                raise TypeError(f"参数 {name} 的范围端点必须是数值，当前得到：{value!r}")

            if v_max <= v_min:
                raise ValueError(f"参数 {name} 的范围必须满足 max > min，当前得到：{value!r}")

    def _validate_fixed_params(self) -> None:
        """
        检查固定参数是否合法。
        """
        overlap = [p for p in self.vary_params if p in self.fixed_params]
        if overlap:
            raise ValueError(
                f"以下参数同时出现在 vary_params 和 fixed_params 中，这是不允许的：{overlap}"
            )

    def _validate_sample_shape(self) -> None:
        """
        检查样本形状描述是否合法。
        """
        if not self.sample_shape:
            raise ValueError("sample_shape 不能为空。")

        if len(self.sample_shape) != len(self.vary_params):
            raise ValueError(
                "sample_shape 的维度数必须与 vary_params 数量一致："
                f"当前 vary_params={self.vary_params}, sample_shape={self.sample_shape}"
            )

        for n in self.sample_shape:
            if not isinstance(n, int):
                raise TypeError(f"sample_shape 中的每个元素都必须是 int，当前得到：{n!r}")
            if n < 2:
                raise ValueError(f"sample_shape 中每个维度至少应为 2，当前得到：{self.sample_shape}")

    def _validate_numerical_setup(self) -> None:
        """
        检查积分相关数值设置是否合法。
        """
        if not isinstance(self.n_steps, int) or self.n_steps < 2:
            raise ValueError(f"n_steps 必须是 >= 2 的整数，当前得到：{self.n_steps!r}")

        if not isinstance(self.step_size, (int, float)) or self.step_size <= 0:
            raise ValueError(f"step_size 必须是正数，当前得到：{self.step_size!r}")

    def _validate_split_ratios(self) -> None:
        """
        检查 train/val/test 比例是否合法。
        """
        if len(self.split_ratios) != 3:
            raise ValueError(
                f"split_ratios 必须是长度为 3 的 tuple，当前得到：{self.split_ratios!r}"
            )

        train_ratio, val_ratio, test_ratio = self.split_ratios

        for x in self.split_ratios:
            if not isinstance(x, (int, float)):
                raise TypeError(f"split_ratios 中的元素必须是数值，当前得到：{self.split_ratios!r}")
            if x < 0:
                raise ValueError(f"split_ratios 中的元素不能为负，当前得到：{self.split_ratios!r}")

        s = train_ratio + val_ratio + test_ratio
        if abs(s - 1.0) > 1e-12:
            raise ValueError(
                f"split_ratios 三者之和必须等于 1，当前得到：{self.split_ratios!r}，和为 {s}"
            )

    # ==========================================================
    # 四、辅助属性
    # ==========================================================

    @property
    def num_vary_params(self) -> int:
        """
        返回变化参数个数。
        """
        return len(self.vary_params)

    @property
    def total_requested_samples(self) -> int:
        """
        返回总请求样本数。

        例如：
        - [240] -> 240
        - [20,20] -> 400
        - [10,12,8] -> 960
        """
        total = 1
        for n in self.sample_shape:
            total *= n
        return total

    @property
    def lambda_max(self) -> float:
        """
        返回积分区间末端近似值：
            (n_steps - 1) * step_size
        """
        return (self.n_steps - 1) * self.step_size

    # ==========================================================
    # 五、导出方法
    # ==========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        将 TaskSpec 转成普通字典。

        用途：
        - 保存到 meta.json
        - 记录任务配置
        """
        return {
            "vary_params": self.vary_params,
            "vary_ranges": self.vary_ranges,
            "fixed_params": self.fixed_params,
            "sample_shape": self.sample_shape,
            "n_steps": self.n_steps,
            "step_size": self.step_size,
            "split_ratios": self.split_ratios,
            "seed": self.seed,
            "sampling_mode": self.sampling_mode,
            "metadata": self.metadata,
        }
