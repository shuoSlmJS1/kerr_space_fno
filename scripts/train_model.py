# ==========================================================
# File: scripts/train_model.py
#
# 功能简介：
# 1. 训练模块的统一入口脚本；
# 2. 提供命令行接口；
# 3. 根据 task_name 读取对应数据集；
# 4. 根据 --model 选择并构造模型；
# 5. 根据模型参数自动生成 model_name；
# 6. 调用 trainer.py 完成训练、验证与测试；
# 7. 将模型权重、训练日志、训练摘要保存到统一目录。
#
# 依赖关系：
# - 依赖 src/models/registry.py
# - 依赖 src/common/naming.py
# - 依赖 src/training/fno1d/dataset_loader_1d.py
# - 依赖 src/training/trainer.py
#
# 重要说明：
# - 当前先支持 fno1d；
# - 后续若加入 cnn1d / fno2d，可继续复用该入口脚本；
# - 同一路径写入时默认覆盖旧文件。
# ==========================================================

from __future__ import annotations

import argparse
import sys

import torch

from pathlib import Path
# 将项目根目录加入 sys.path，保证可以导入 src.xxx
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.naming import build_generic_model_name
from src.models.registry import build_model, get_model_help_text
from src.training.fno1d.dataset_loader_1d import (
    build_dataloaders_1d,
    summarize_loaded_bundle_1d,
)
from src.training.fno1d.input_builder_1d import infer_fno1d_input_dim
from src.training.trainer import TrainerConfig, fit_model


# ==========================================================
# 一、命令行解析
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """
    构造训练入口脚本的 argparse 解析器。
    """
    parser = argparse.ArgumentParser(
        description="Unified training entry for Kerr orbit prediction models."
    )

    # ------------------------------
    # A. 基本任务参数
    # ------------------------------
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="要训练的数据任务名，对应 data/tasks/<task_name>/dataset.npz",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型类型，例如：fno1d",
    )

    parser.add_argument(
        "--show-model-help",
        action="store_true",
        help="显示支持的模型类型及其参数要求，然后退出。",
    )

    # ------------------------------
    # B. FNO1d 模型参数
    # ------------------------------
    parser.add_argument("--modes", type=int, default=32, help="FNO1d 低频模式数")
    parser.add_argument("--width", type=int, default=64, help="FNO1d 隐空间宽度")
    parser.add_argument("--depth", type=int, default=4, help="FNO1d block 层数")

    # ------------------------------
    # C. 训练参数
    # ------------------------------
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--print-every", type=int, default=1, help="日志打印频率")

    # ------------------------------
    # D. 优化器参数
    # ------------------------------
    parser.add_argument("--optimizer", type=str, default="adamw", help="优化器名称")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")

    parser.add_argument("--scheduler", type=str, default="exponential", help="学习率调度器名称")
    parser.add_argument("--scheduler-gamma", type=float, default=0.995, help="ExponentialLR 的 gamma")

    # ------------------------------
    # E. 运行设备
    # ------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备，例如 cpu / cuda",
    )

    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")

    return parser


# ==========================================================
# 二、模型配置构造
# ==========================================================

def build_model_config_from_args(args: argparse.Namespace, in_dim: int, out_dim: int) -> dict:
    """
    根据命令行参数构造模型配置字典。
    """
    model_type = args.model.lower()

    if model_type == "fno1d":
        return {
            "model_type": "fno1d",
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "modes": int(args.modes),
            "width": int(args.width),
            "depth": int(args.depth),
        }

    raise ValueError(f"当前不支持的模型类型：{model_type!r}")


def build_model_name_from_config(model_config: dict) -> str:
    """
    根据模型配置生成 model_name。
    """
    model_type = model_config["model_type"]

    if model_type == "fno1d":
        return build_generic_model_name(
            model_type="fno",
            modes=model_config["modes"],
            width=model_config["width"],
            depth=model_config["depth"],
        )

    raise ValueError(f"当前不支持的模型类型：{model_type!r}")


# ==========================================================
# 三、主流程
# ==========================================================

def main() -> None:
    """
    主流程：
    1. 解析命令行
    2. 根据 task_name 构造 DataLoader
    3. 根据模型参数构造模型
    4. 生成 model_name
    5. 调用训练器 fit_model()
    """
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------
    # A. 帮助信息
    # ------------------------------
    if args.show_model_help:
        print(get_model_help_text())
        return

    # ------------------------------
    # B. 当前先按一维任务加载数据
    # ------------------------------
    train_loader, val_loader, test_loader, bundle = build_dataloaders_1d(
        task_name=args.task_name,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    print("=" * 70)
    print("Loaded dataset summary")
    print("=" * 70)
    for k, v in summarize_loaded_bundle_1d(bundle).items():
        print(f"{k:<24s} : {v}")

    # ------------------------------
    # C. 推断输入输出维度
    # ------------------------------
    num_vary_params = len(bundle.vary_params_order)
    in_dim = infer_fno1d_input_dim(num_vary_params)
    out_dim = bundle.y_train.shape[-1]   # 通常是 3，对应 x,y,z

    # ------------------------------
    # D. 构造模型配置与模型名
    # ------------------------------
    model_config = build_model_config_from_args(args, in_dim=in_dim, out_dim=out_dim)
    model_name = build_model_name_from_config(model_config)

    # ------------------------------
    # E. 构造模型
    # ------------------------------
    model = build_model(
        model_type=model_config["model_type"],
        in_dim=model_config["in_dim"],
        out_dim=model_config["out_dim"],
        modes=model_config["modes"],
        width=model_config["width"],
        depth=model_config["depth"],
    )

    # ------------------------------
    # F. 训练配置
    # ------------------------------
    trainer_config = TrainerConfig(
        task_name=args.task_name,
        model_name=model_name,
        device=args.device,
        epochs=int(args.epochs),
        print_every=int(args.print_every),
        optimizer_name=args.optimizer,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        scheduler_name=args.scheduler,
        scheduler_gamma=float(args.scheduler_gamma),
    )

    print("=" * 70)
    print("Training task summary")
    print("=" * 70)
    print(f"task_name                : {args.task_name}")
    print(f"model_type               : {model_config['model_type']}")
    print(f"model_name               : {model_name}")
    print(f"in_dim                   : {model_config['in_dim']}")
    print(f"out_dim                  : {model_config['out_dim']}")
    print(f"modes                    : {model_config['modes']}")
    print(f"width                    : {model_config['width']}")
    print(f"depth                    : {model_config['depth']}")
    print(f"device                   : {trainer_config.device}")
    print(f"epochs                   : {trainer_config.epochs}")
    print(f"batch_size               : {args.batch_size}")
    print(f"optimizer                : {trainer_config.optimizer_name}")
    print(f"lr                       : {trainer_config.lr}")
    print(f"weight_decay             : {trainer_config.weight_decay}")
    print(f"scheduler                : {trainer_config.scheduler_name}")
    print(f"scheduler_gamma          : {trainer_config.scheduler_gamma}")
    print("write mode               : overwrite same-path files if they already exist")

    # ------------------------------
    # G. 开始训练
    # ------------------------------
    fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=trainer_config,
        model_config=model_config,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)