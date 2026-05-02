# ==========================================================
# File: scripts/train_model_2d.py
#
# 功能简介：
# 1. FNO2d / 其他二维模型的独立训练入口；
# 2. 不修改原来的 scripts/train_model.py；
# 3. 通过 src/models/registry_2d.py 构造二维模型；
# 4. 当前支持 fno2d；
# 5. 后续可以扩展 cnn2d / unet2d / fno2d_large 等模型；
#
# 当前二维算子任务：
#
#       (p, lambda) -> (x, y, z)
#
# 其中：
# - p 是单变化参数，例如 Q / a / E / Lz；
# - lambda 是轨道参数方向；
# - 输出是 Kerr 轨道的 xyz 坐标。
#
# 输入数据来自已有的一维任务数据集：
#
#       data/tasks/<task_name>/dataset.npz
#
# 输出目录：
#
#       outputs/<task_name>/<model_name>/
#
# model_name 示例：
#
#       fno2d_m1_16_m2_32_w64_d4
#
# 其中：
# - m1_16：第一个二维方向，即参数 p 方向 modes = 16；
# - m2_32：第二个二维方向，即 lambda 方向 modes = 32；
# - w64：hidden width = 64；
# - d4：FNO block depth = 4。
# ==========================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn


# ==========================================================
# 一、保证可以从 scripts/ 正确导入 src/
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.fno2d.fno2d import count_parameters  # noqa: E402
from src.models.registry_2d import (  # noqa: E402
    build_model_2d,
    build_model_name_2d,
    get_model_help_text_2d,
    summarize_model_config_2d,
)
from src.training.fno2d.dataset_loader_2d import (  # noqa: E402
    build_fno2d_dataloaders,
    summarize_fno2d_bundle,
)


# ==========================================================
# 二、路径工具
# ==========================================================

def get_model_output_dirs(task_name: str, model_name: str) -> dict[str, Path]:
    """
    构造二维模型输出目录。

    返回：
    - model_dir:
        outputs/<task_name>/<model_name>/

    - checkpoints_dir:
        outputs/<task_name>/<model_name>/checkpoints/

    - logs_dir:
        outputs/<task_name>/<model_name>/logs/
    """
    model_dir = PROJECT_ROOT / "outputs" / task_name / model_name
    checkpoints_dir = model_dir / "checkpoints"
    logs_dir = model_dir / "logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_dir": model_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
    }


def save_json(obj: dict[str, Any], path: Path) -> None:
    """
    保存 JSON 文件。

    说明：
    - 若路径已存在，会覆盖旧文件；
    - 这与当前项目统一的 same-path overwrite 逻辑一致。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# ==========================================================
# 三、指标函数
# ==========================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算 MSE loss。

    输入：
    - pred:
        [B, H, W, 3]

    - target:
        [B, H, W, 3]

    返回：
    - 标量 MSE
    """
    return torch.mean((pred - target) ** 2)


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    计算 Relative L2 error。

    公式：
        ||pred - target||_2 / ||target||_2

    说明：
    - 这里对每个 batch 样本整体展平后计算；
    - 对于当前 FNO2d 第一版，每个 split 通常只有一个二维场样本。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target shape 必须一致，当前 pred={tuple(pred.shape)}, "
            f"target={tuple(target.shape)}"
        )

    diff_norm = torch.linalg.norm(
        (pred - target).reshape(pred.shape[0], -1),
        dim=1,
    )

    target_norm = torch.linalg.norm(
        target.reshape(target.shape[0], -1),
        dim=1,
    )

    rel = diff_norm / (target_norm + eps)

    return torch.mean(rel)


# ==========================================================
# 四、训练与评估函数
# ==========================================================

def train_one_epoch_2d(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    返回：
    - train_mse
    - train_relative_l2
    """
    model.train()

    total_mse = 0.0
    total_rel = 0.0
    total_count = 0

    for x, y in loader:
        # x: [B, H, W, C]
        # y: [B, H, W, 3]
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)

        loss = mse_loss(pred, y)
        rel = relative_l2_error(pred, y)

        loss.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total_mse += float(loss.detach().cpu()) * batch_size
        total_rel += float(rel.detach().cpu()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("train_loader 中没有样本。")

    return total_mse / total_count, total_rel / total_count


@torch.no_grad()
def evaluate_one_epoch_2d(
    model: nn.Module,
    loader,
    device: str,
) -> tuple[float, float]:
    """
    验证或测试一个 epoch。

    返回：
    - mse
    - relative_l2
    """
    model.eval()

    total_mse = 0.0
    total_rel = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = mse_loss(pred, y)
        rel = relative_l2_error(pred, y)

        batch_size = int(x.shape[0])
        total_mse += float(loss.detach().cpu()) * batch_size
        total_rel += float(rel.detach().cpu()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("eval_loader 中没有样本。")

    return total_mse / total_count, total_rel / total_count


# ==========================================================
# 五、checkpoint 保存
# ==========================================================

def save_checkpoint_2d(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    config: dict[str, Any],
) -> None:
    """
    保存二维模型 checkpoint。

    保存内容：
    - epoch
    - best_val_mse
    - model_state_dict
    - optimizer_state_dict
    - config
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": int(epoch),
            "best_val_mse": float(best_val_mse),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


# ==========================================================
# 六、命令行参数
# ==========================================================

def build_parser() -> argparse.ArgumentParser:
    """
    构造命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a 2D model for single-parameter Kerr operator field: "
            "(p, lambda) -> (x,y,z)."
        )
    )

    # ------------------------------------------------------
    # A. 任务与模型类型
    # ------------------------------------------------------
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="已有单参数任务名，例如 vary_Q__Q1.6_3__n2000__T1200__cfg1。",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="fno2d",
        help="二维模型类型。当前支持 fno2d。",
    )

    parser.add_argument(
        "--show-model-help",
        action="store_true",
        help="显示支持的二维模型类型，然后退出。",
    )

    # ------------------------------------------------------
    # B. 二维 Fourier modes
    # ------------------------------------------------------
    parser.add_argument(
        "--modes-param",
        type=int,
        default=16,
        help="参数方向 Fourier modes 数量，例如 Q/a/E/Lz 方向。",
    )

    parser.add_argument(
        "--modes-lambda",
        type=int,
        default=32,
        help="lambda 方向 Fourier modes 数量。",
    )

    # ------------------------------------------------------
    # C. 模型结构参数
    # ------------------------------------------------------
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="二维模型 hidden width。",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="二维模型 block 层数。",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="输出 MLP hidden dimension。",
    )

    # ------------------------------------------------------
    # D. 训练参数
    # ------------------------------------------------------
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="FNO2d 第一版建议保持 1，因为一个完整二维场就是一个样本。",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮数。",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率。",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay。",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.995,
        help="ExponentialLR gamma。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备：cuda 或 cpu。",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="每多少个 epoch 打印一次日志。",
    )

    return parser


# ==========================================================
# 七、主函数
# ==========================================================

def main() -> None:
    """
    二维模型训练主流程。
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.show_model_help:
        print(get_model_help_text_2d())
        return

    device = str(args.device)

    # ------------------------------------------------------
    # A. 由 registry_2d 统一生成模型名
    # ------------------------------------------------------
    model_name = build_model_name_2d(
        model_type=str(args.model),
        modes1=int(args.modes_param),
        modes2=int(args.modes_lambda),
        width=int(args.width),
        depth=int(args.depth),
    )

    dirs = get_model_output_dirs(
        task_name=str(args.task_name),
        model_name=model_name,
    )

    # ------------------------------------------------------
    # B. 加载 FNO2d 数据
    # ------------------------------------------------------
    train_loader, val_loader, test_loader, bundle = build_fno2d_dataloaders(
        task_name=str(args.task_name),
        batch_size=int(args.batch_size),
        num_workers=0,
        sort_param=True,
    )

    bundle_summary = summarize_fno2d_bundle(bundle)

    print("=" * 70)
    print("Loaded 2D dataset summary")
    print("=" * 70)
    print(json.dumps(bundle_summary, indent=4, ensure_ascii=False))

    # ------------------------------------------------------
    # C. 通过 registry_2d 构造模型
    # ------------------------------------------------------
    model = build_model_2d(
        model_type=str(args.model),
        in_dim=2,
        out_dim=3,
        modes1=int(args.modes_param),
        modes2=int(args.modes_lambda),
        width=int(args.width),
        depth=int(args.depth),
        hidden_dim=int(args.hidden_dim),
        activation="gelu",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(args.scheduler_gamma),
    )

    model_config = summarize_model_config_2d(
        model_type=str(args.model),
        in_dim=2,
        out_dim=3,
        modes1=int(args.modes_param),
        modes2=int(args.modes_lambda),
        width=int(args.width),
        depth=int(args.depth),
        hidden_dim=int(args.hidden_dim),
        activation="gelu",
    )

    train_config = {
        "task_name": str(args.task_name),
        "model_name": model_name,
        "model_type": str(args.model),
        "device": device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler_gamma": float(args.scheduler_gamma),
        "print_every": int(args.print_every),
        "num_parameters": int(count_parameters(model)),
        "model_config": model_config,
    }

    print("=" * 70)
    print("2D training task summary")
    print("=" * 70)
    print(json.dumps(train_config, indent=4, ensure_ascii=False))
    print("write mode: overwrite same-path files if they already exist")

    # ------------------------------------------------------
    # D. 主训练循环
    # ------------------------------------------------------
    history = {
        "train_mse": [],
        "train_rel_l2": [],
        "val_mse": [],
        "val_rel_l2": [],
        "lr": [],
    }

    best_val_mse = float("inf")
    best_epoch = -1

    train_start_time = perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        train_mse, train_rel = train_one_epoch_2d(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_mse, val_rel = evaluate_one_epoch_2d(
            model=model,
            loader=val_loader,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_mse"].append(float(train_mse))
        history["train_rel_l2"].append(float(train_rel))
        history["val_mse"].append(float(val_mse))
        history["val_rel_l2"].append(float(val_rel))
        history["lr"].append(float(current_lr))

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            best_epoch = int(epoch)

            save_checkpoint_2d(
                path=dirs["checkpoints_dir"] / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mse=best_val_mse,
                config=train_config,
            )

        save_checkpoint_2d(
            path=dirs["checkpoints_dir"] / "last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mse=best_val_mse,
            config=train_config,
        )

        scheduler.step()

        if epoch % int(args.print_every) == 0:
            print(
                f"Epoch [{epoch:03d}/{args.epochs}] | "
                f"lr={current_lr:.6e} | "
                f"train_mse={train_mse:.6e} | "
                f"train_relL2={train_rel:.6e} | "
                f"val_mse={val_mse:.6e} | "
                f"val_relL2={val_rel:.6e}"
            )

    # ------------------------------------------------------
    # E. 加载 best model 并做 test
    # ------------------------------------------------------
    best_ckpt_path = dirs["checkpoints_dir"] / "best_model.pt"
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_mse, test_rel = evaluate_one_epoch_2d(
        model=model,
        loader=test_loader,
        device=device,
    )

    train_total_seconds = perf_counter() - train_start_time

    # ------------------------------------------------------
    # F. 保存 history / summary
    # ------------------------------------------------------
    save_json(
        history,
        dirs["logs_dir"] / "train_history.json",
    )

    summary = {
        "task_name": str(args.task_name),
        "model_name": model_name,
        "model_type": str(args.model),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val_mse),
        "test_mse": float(test_mse),
        "test_relative_l2": float(test_rel),
        "train_total_seconds": float(train_total_seconds),
        "trainer_config": train_config,
        "dataset_summary": bundle_summary,
        "history_keys": list(history.keys()),
    }

    save_json(
        summary,
        dirs["logs_dir"] / "train_summary.json",
    )

    print("-" * 70)
    print(f"2D training finished. Best epoch = {best_epoch}, best val MSE = {best_val_mse:.6e}")
    print(f"Test MSE         : {test_mse:.6e}")
    print(f"Test Relative L2 : {test_rel:.6e}")
    print(f"Train time       : {train_total_seconds:.2f} s")
    print(f"Output dir       : {dirs['model_dir']}")
    print("-" * 70)


if __name__ == "__main__":
    main()