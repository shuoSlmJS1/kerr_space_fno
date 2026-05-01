# ==========================================================
# File: scripts/generate_dataset.py
#
# 功能简介：
# 1. 数据生成模块的统一入口脚本；
# 2. 提供命令行接口；
# 3. 调用 parameter_parser.py 构造 TaskSpec；
# 4. 调用 naming.py 生成 task_name；
# 5. 在正式生成前执行现实范围 warning 检查；
# 6. 若存在 warning，则询问用户是否确认继续；
# 7. 调用 dataset_builder.py 构建数据集；
# 8. 调用 dataset_saver.py 保存结果。
#
# 依赖关系：
# - 依赖整个数据生成模块
# - 是用户真正直接运行的入口
#
# 重要说明：
# - 用户确认后再继续的逻辑写在这里；
# - help 信息也集中由这里对外展示；
# - 本脚本负责“入口调度”，不负责底层物理细节。
# ==========================================================
from __future__ import annotations

import argparse
import sys

from pathlib import Path
# 将项目根目录加入 sys.path，保证可以导入 src.xxx
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.naming import build_task_name
from src.common.paths import build_task_data_paths
from src.data_generation.astrophysical_checks import get_astrophysical_range_help_text
from src.data_generation.dataset_builder import build_dataset
from src.data_generation.dataset_saver import save_built_dataset
from src.data_generation.parameter_parser import (
    add_dataset_generation_arguments,
    get_generate_dataset_help_text,
    parse_task_spec_from_args,
)


# ==========================================================
# 一、命令行解析
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """
    构造数据生成入口脚本的 argparse 解析器。
    """
    parser = argparse.ArgumentParser(
        description="Unified dataset generation entry for Kerr orbit prediction tasks."
    )

    # 注册通用数据生成参数
    add_dataset_generation_arguments(parser)

    # 额外帮助选项
    parser.add_argument(
        "--show-param-help",
        action="store_true",
        help="显示可变化参数、默认固定参数与使用示例，然后退出。",
    )

    parser.add_argument(
        "--show-astro-help",
        action="store_true",
        help="显示现实天体范围参考说明，然后退出。",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "若出现现实范围 warning，默认自动继续，不再询问确认。"
            "适合批处理或服务器环境。"
        ),
    )

    return parser


# ==========================================================
# 二、用户确认逻辑
# ==========================================================

def ask_user_to_continue(warnings_list: list[str]) -> bool:
    """
    当存在现实范围 warning 时，询问用户是否继续。

    返回：
    - True  : 继续执行
    - False : 终止执行

    说明：
    - 该逻辑只放在入口脚本中；
    - 底层模块不负责与用户交互。
    """
    if not warnings_list:
        return True

    print("\n" + "=" * 70)
    print("Astrophysical range warnings detected")
    print("=" * 70)
    for i, w in enumerate(warnings_list, start=1):
        print(f"[{i}] {w}")

    print("\nThese are warnings only, not hard errors.")
    print("Do you want to continue dataset generation? [y/N]: ", end="", flush=True)

    answer = input().strip().lower()
    return answer in ("y", "yes")


# ==========================================================
# 三、主流程
# ==========================================================

def main() -> None:
    """
    主流程：
    1. 解析命令行
    2. 按需显示 help 文本
    3. 构造 TaskSpec
    4. 生成 task_name
    5. 先做 warning 检查并询问用户是否继续
    6. 正式生成数据集
    7. 保存到统一路径（同路径默认覆盖）
    """
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------
    # A. 显示帮助文本后退出
    # ------------------------------------------------------
    if args.show_param_help:
        print(get_generate_dataset_help_text())
        return

    if args.show_astro_help:
        print(get_astrophysical_range_help_text())
        return

    # ------------------------------------------------------
    # B. 解析 TaskSpec
    # ------------------------------------------------------
    task_spec = parse_task_spec_from_args(args)

    # 生成 task_name，并写入 metadata
    task_name = build_task_name(task_spec)
    task_spec.metadata["task_name"] = task_name

    # ------------------------------------------------------
    # C. 先构建一次数据集前检查 warning
    # ------------------------------------------------------
    # 注意：
    # 真正的 warning 列表是在 build_dataset(task_spec) 内部也会得到；
    # 但为了实现“用户确认后再继续”，我们先做一次轻量构建前预检查。
    from src.data_generation.astrophysical_checks import warn_task_spec_astrophysical_ranges
    from src.data_generation.validity import validate_task_spec_hard_constraints

    # 先做硬约束检查（失败就直接抛错）
    validate_task_spec_hard_constraints(task_spec)

    # 再做现实范围 warning 检查
    warnings_list = warn_task_spec_astrophysical_ranges(task_spec)

    # 如果没有 --yes，且有 warning，则要求用户确认
    if warnings_list and (not args.yes):
        should_continue = ask_user_to_continue(warnings_list)
        if not should_continue:
            print("Dataset generation aborted by user.")
            return

    # ------------------------------------------------------
    # D. 打印任务摘要
    # ------------------------------------------------------
    paths = build_task_data_paths(task_name)

    print("=" * 70)
    print("Dataset generation task summary")
    print("=" * 70)
    print(f"task_name           : {task_name}")
    print(f"vary_params         : {task_spec.vary_params}")
    print(f"vary_ranges         : {task_spec.vary_ranges}")
    print(f"sample_shape        : {task_spec.sample_shape}")
    print(f"requested_samples   : {task_spec.total_requested_samples}")
    print(f"n_steps             : {task_spec.n_steps}")
    print(f"step_size           : {task_spec.step_size}")
    print(f"dataset output dir  : {paths.task_dir}")
    print(
        "write mode          : overwrite same-path files if they already exist"
    )

    # ------------------------------------------------------
    # E. 正式生成数据集
    # ------------------------------------------------------
    build_result = build_dataset(task_spec)

    # ------------------------------------------------------
    # F. 保存结果
    # ------------------------------------------------------
    summary = save_built_dataset(build_result)

    print("=" * 70)
    print("Final summary")
    print("=" * 70)
    for k, v in summary.items():
        print(f"{k:<20s} : {v}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
        sys.exit(1)