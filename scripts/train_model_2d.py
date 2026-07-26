# ==========================================================
# File: scripts/train_model_2d.py
#
# 功能简介：
# 1. 二维模型的独立训练入口；
# 2. 不修改原来的 scripts/train_model.py；
# 3. 通过 src/models/registry_2d.py 构造二维模型；
# 4. 当前支持 fno2d；
# 5. 支持 normalization：
#       none
#       standard
# 6. 支持 target transform：
#       raw
#       residual_initial
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
# 数据处理顺序：
#   raw xyz
#   -> target transform
#   -> normalization
#
# 输出目录：
#
#   outputs/<task_name>/<model_name>/
#
# model_name 示例：
#
#   fno2d_m1_16_m2_32_w64_d4_norm-standard_target-residual_initial_ref0
# ==========================================================

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ==========================================================
# 一、保证可以从 scripts/ 正确导入 src/
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.io_utils import save_json as save_json_common  # noqa: E402
from src.common.paths import ensure_model_output_dirs  # noqa: E402
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

def get_model_output_dirs(
    task_name: str,
    model_name: str,
) -> dict[str, Path]:
    """
    构造统一模型输出目录。

    该接口暂时保留，避免大范围修改训练脚本；
    实际路径统一由 src.common.paths 管理。
    """
    paths = ensure_model_output_dirs(
        task_name=task_name,
        model_name=model_name,
    )

    return {
        "model_dir": paths.model_dir,
        "run_config_json": paths.run_config_json,
        "summary_json": paths.summary_json,
        "checkpoints_dir": paths.checkpoints_dir,
        "logs_dir": paths.logs_dir,
        "inference_dir": paths.inference_dir,
        "analysis_dir": paths.analysis_dir,
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
# 三、训练随机种子
# ==========================================================

def set_training_seed(seed: int) -> None:
    """
    固定训练阶段的随机种子。

    控制：
    - Python random；
    - NumPy；
    - PyTorch CPU；
    - PyTorch CUDA。

    注意：
    完全确定性还会受到 CUDA 算子和运行环境影响，但固定 seed
    可以显著提高实验可复现性。
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================================================
# 四、指标函数
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


def _reshape_trajectory_samples(array: torch.Tensor) -> torch.Tensor:
    """
    将轨道预测张量整理为 [N_traj, T * C]。

    统一约定：
    - 1D FNO 输出通常是 [B, T, C]，每个 B 是一条轨道；
    - 2D FNO 输出通常是 [B_field, N_param, T, C]，每个 (field, param) 是一条轨道；
    - C 通常为 3，对应 xyz。

    因此，对于 ndim >= 3 且最后一维是坐标通道的张量，
    所有倒数第二维 T 之前的维度都视为“轨道样本维”。
    """
    if array.ndim >= 3:
        return array.reshape(-1, array.shape[-2] * array.shape[-1])

    if array.ndim == 2:
        return array.reshape(array.shape[0], -1)

    raise ValueError(f"relative_l2_error requires at least 2 dimensions; current shape={tuple(array.shape)}")


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    计算统一口径的轨道级 Relative L2。

    计算方式：
        先把 pred/target 整理成 [N_traj, T*C]；
        对每条轨道分别计算 ||pred_i-target_i||_2 / ||target_i||_2；
        最后对所有轨道取平均。

    对 2D FNO：
        [B_field, N_param, T, 3] 会被视为 B_field * N_param 条轨道，
        不再把一个完整二维场当成一个样本来算 Relative L2。
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have identical shapes; current pred={tuple(pred.shape)}, "
            f"target={tuple(target.shape)}"
        )

    pred_flat = _reshape_trajectory_samples(pred)
    target_flat = _reshape_trajectory_samples(target)

    diff_norm = torch.linalg.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.linalg.norm(target_flat, dim=1)
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
        raise RuntimeError("train_loader contains no samples.")

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
        raise RuntimeError("eval_loader contains no samples.")

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
        default=None,
        help=(
            "Existing single-parameter task name, for example "
            "q_1p6-3_n2000_t1200. Use this option for single-task mode."
        ),
    )

    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multi-configuration mode: provide multiple "
            "single-parameter task names with matching dataset structures."
        ),
    )

    parser.add_argument(
        "--cfg-param-name",
        type=str,
        default=None,
        help=(
            "Fixed parameter used as an additional conditioning "
            "channel in multi-configuration mode, for example a. "
            "The input channels change from [Q, lambda] to [Q, lambda, a]."
        ),
    )

    parser.add_argument(
        "--output-task-name",
        type=str,
        default=None,
        help=(
            "Output task name used for multi-configuration runs. "
            "If omitted, a generated multi_cfg name is used."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="fno2d",
        help="Two-dimensional model type. Currently supported: fno2d.",
    )

    parser.add_argument(
        "--show-model-help",
        action="store_true",
        help="Show supported two-dimensional model types and exit.",
    )

    # ------------------------------------------------------
    # B. 二维 Fourier modes
    # ------------------------------------------------------
    parser.add_argument(
        "--modes-param",
        type=int,
        default=16,
        help=(
            "Number of Fourier modes along the parameter axis, "
            "such as Q, a, E, or Lz."
        ),
    )

    parser.add_argument(
        "--modes-lambda",
        type=int,
        default=32,
        help="Number of Fourier modes along the lambda axis.",
    )

    # ------------------------------------------------------
    # C. 模型结构参数
    # ------------------------------------------------------
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Hidden width of the two-dimensional model.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of blocks in the two-dimensional model.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of the output MLP.",
    )

    # ------------------------------------------------------
    # D. 数据变换参数
    # ------------------------------------------------------
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "standard"],
        help=(
            "Input and output normalization method. "
            "none disables normalization; standard uses statistics "
            "computed from the training split."
        ),
    )

    parser.add_argument(
        "--target-transform",
        type=str,
        default="raw",
        choices=["raw", "residual_initial"],
        help=(
            "Output target transform. raw learns xyz directly; "
            "residual_initial learns trajectory residuals relative to "
            "a reference lambda index."
        ),
    )

    parser.add_argument(
        "--lambda-reference-index",
        type=int,
        default=0,
        help=(
            "Reference lambda index used by residual_initial. "
            "The default is 0, corresponding to the initial point."
        ),
    )

    # ------------------------------------------------------
    # E. 训练参数
    # ------------------------------------------------------
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for two-dimensional fields. A value of 1 is "
            "recommended because one complete field is one dataset item."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.995,
        help="ExponentialLR gamma.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device: cuda or cpu.",
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=27,
        help="Random seed for training. Default: 27.",
    )

    parser.add_argument(
        "--name-tags",
        nargs="*",
        default=None,
        help=(
            "Optional tags to include in the model run directory "
            "name, for example --name-tags normalization seed. "
            "Full run configuration is stored in JSON."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print training metrics every N epochs.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print the complete dataset and run configuration. "
            "By default, only a concise human-readable summary is shown."
        ),
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

    if args.task_names is None and args.task_name is None:
        raise ValueError(
            "Either --task-name or --task-names must be provided."
        )

    if args.task_names is not None and len(args.task_names) > 0:
        if args.cfg_param_name is None:
            raise ValueError(
                "--cfg-param-name is required when using "
                "multi-configuration mode with --task-names."
            )
        effective_task_name = str(args.output_task_name or "multi_cfg_2d")
    else:
        effective_task_name = str(args.task_name)

    device = str(args.device)
    training_seed = int(args.training_seed)

    # 必须在模型构造和 DataLoader 使用之前设置。
    set_training_seed(training_seed)

    requested_name_tags = {
        str(tag).strip().lower()
        for tag in (args.name_tags or [])
        if str(tag).strip()
    }

    supported_name_tags = {
        "normalization",
        "target",
        "seed",
        "lr",
    }

    unknown_name_tags = (
        requested_name_tags - supported_name_tags
    )
    if unknown_name_tags:
        raise ValueError(
            "Unsupported --name-tags: "
            f"{sorted(unknown_name_tags)}; "
            f"supported values: {sorted(supported_name_tags)}"
        )

    extra_name_tags: list[str] = []

    if "normalization" in requested_name_tags:
        normalization_tag = {
            "standard": "std",
            "none": "none",
        }[str(args.normalization)]
        extra_name_tags.append(normalization_tag)

    if "target" in requested_name_tags:
        target_tag = {
            "raw": "raw",
            "residual_initial": "res",
        }[str(args.target_transform)]
        extra_name_tags.append(target_tag)

    if "seed" in requested_name_tags:
        extra_name_tags.append(f"s{training_seed}")

    if "lr" in requested_name_tags:
        lr_text = f"{float(args.lr):g}".replace(".", "p")
        extra_name_tags.append(f"lr{lr_text}")

    # ------------------------------------------------------
    # A. 由 registry_2d 统一生成模型运行名
    # ------------------------------------------------------
    model_name = build_model_name_2d(
        model_type=str(args.model),
        modes1=int(args.modes_param),
        modes2=int(args.modes_lambda),
        width=int(args.width),
        depth=int(args.depth),
        epochs=int(args.epochs),
        extra_tags=extra_name_tags,
    )

    dirs = get_model_output_dirs(
        task_name=effective_task_name,
        model_name=model_name,
    )

    # ------------------------------------------------------
    # C. 加载 FNO2d 数据
    # ------------------------------------------------------
    train_loader, val_loader, test_loader, bundle = build_fno2d_dataloaders(
        task_name=args.task_name,
        task_names=args.task_names,
        cfg_param_name=args.cfg_param_name,
        output_task_name=effective_task_name,
        batch_size=int(args.batch_size),
        num_workers=0,
        sort_param=True,
        normalization=str(args.normalization),
        target_transform=str(args.target_transform),
        lambda_reference_index=int(args.lambda_reference_index),
    )

    bundle_summary = summarize_fno2d_bundle(bundle)

    # 默认只显示适合人工阅读的短摘要；
    # 完整数据结构已经保存在 run_config.json 中。
    if args.verbose:
        print("=" * 70)
        print("Complete 2D Dataset Summary")
        print("=" * 70)
        print(json.dumps(bundle_summary, indent=4, ensure_ascii=False))
    else:
        print("=" * 70)
        print("2D Dataset")
        print("=" * 70)
        print(f"Task                 : {effective_task_name}")
        print(
            "Operator axes        : "
            f"{bundle.param_name}, lambda"
        )
        print(
            "Train / val / test   : "
            f"{bundle.train_field.num_param} / "
            f"{bundle.val_field.num_param} / "
            f"{bundle.test_field.num_param}"
        )
        print(
            "Lambda points        : "
            f"{bundle.train_field.num_lambda}"
        )
        print(
            "Input / output dims  : "
            f"{bundle.train_field.in_dim} / "
            f"{bundle.train_field.out_dim}"
        )
        print(f"Normalization        : {bundle.normalization}")
        print(f"Target transform     : {bundle.target_transform}")

    # ------------------------------------------------------
    # D. 通过 registry_2d 构造模型
    # ------------------------------------------------------
    model = build_model_2d(
        model_type=str(args.model),
        in_dim=int(bundle.train_field.in_dim),
        out_dim=int(bundle.train_field.out_dim),
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
        in_dim=int(bundle.train_field.in_dim),
        out_dim=int(bundle.train_field.out_dim),
        modes1=int(args.modes_param),
        modes2=int(args.modes_lambda),
        width=int(args.width),
        depth=int(args.depth),
        hidden_dim=int(args.hidden_dim),
        activation="gelu",
    )

    train_config = {
        "schema_version": "1.0",
        "run_type": "training",
        "experiment_family": "kerr_fno",
        "model_family": "fno",
        "task_name": effective_task_name,
        "task_names": args.task_names if args.task_names is not None else [str(args.task_name)],
        "cfg_param_name": args.cfg_param_name,
        "model_name": model_name,
        "model_type": str(args.model),
        "operator_axes": [
            bundle.param_name,
            "lambda",
        ],
        "spatial_dimension": 2,
        "device": device,
        "training_seed": int(training_seed),
        "data_seeds": bundle.data_seeds,
        "generation_statuses": bundle.generation_statuses,
        "name_tags": sorted(requested_name_tags),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler_gamma": float(args.scheduler_gamma),
        "print_every": int(args.print_every),
        "normalization": str(args.normalization),
        "target_transform": str(args.target_transform),
        "lambda_reference_index": int(args.lambda_reference_index),
        "num_parameters": int(count_parameters(model)),
        "model_config": model_config,
        "dataset_summary": bundle_summary,
    }

    # 保存运行开始前已经确定的统一配置。
    save_json_common(
        train_config,
        dirs["run_config_json"],
        indent=2,
    )

    if args.verbose:
        print("=" * 70)
        print("Complete Training Configuration")
        print("=" * 70)
        print(json.dumps(train_config, indent=4, ensure_ascii=False))
    else:
        data_seed_text = ", ".join(
            f"{name}={seed}"
            for name, seed in (bundle.data_seeds or {}).items()
        )

        print("=" * 70)
        print("Training Run")
        print("=" * 70)
        print(f"Model                : {model_name}")
        print(f"Model parameters     : {count_parameters(model):,}")
        print(
            "Fourier modes        : "
            f"{args.modes_param} x {args.modes_lambda}"
        )
        print(
            "Width / depth        : "
            f"{args.width} / {args.depth}"
        )
        print(f"Epochs               : {args.epochs}")
        print(f"Batch size           : {args.batch_size}")
        print(f"Device               : {device}")
        print(f"Data seed            : {data_seed_text or 'unknown'}")
        print(f"Training seed        : {training_seed}")
        print(f"Output directory     : {dirs['model_dir']}")

    print("Write mode           : overwrite existing files in this run directory")

    # ------------------------------------------------------
    # E. 主训练循环
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
    # F. 加载 best model 并做 test
    # ------------------------------------------------------
    best_ckpt_path = dirs["checkpoints_dir"] / "best_model.pt"
    checkpoint = torch.load(
        best_ckpt_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_mse, test_rel = evaluate_one_epoch_2d(
        model=model,
        loader=test_loader,
        device=device,
    )

    train_total_seconds = perf_counter() - train_start_time

    # ------------------------------------------------------
    # G. 保存 history / summary
    # ------------------------------------------------------
    save_json(
        history,
        dirs["logs_dir"] / "train_history.json",
    )

    summary = {
        "task_name": effective_task_name,
        "task_names": args.task_names if args.task_names is not None else [str(args.task_name)],
        "cfg_param_name": args.cfg_param_name,
        "model_name": model_name,
        "model_type": str(args.model),
        "operator_axes": [
            bundle.param_name,
            "lambda",
        ],
        "spatial_dimension": 2,
        "training_seed": int(training_seed),
        "data_seeds": bundle.data_seeds,
        "generation_statuses": bundle.generation_statuses,
        "name_tags": sorted(requested_name_tags),
        "normalization": str(args.normalization),
        "target_transform": str(args.target_transform),
        "lambda_reference_index": int(args.lambda_reference_index),
        "best_epoch": int(best_epoch),
        "metric_space": "model_space",
        "metric_usage": "training_monitoring_only",
        "metric_description": (
            "Metrics computed in model space after the configured "
            "normalization and target transform. These metrics are used "
            "only for training monitoring and model selection."
        ),
        "best_val_mse_model_space": float(best_val_mse),
        "test_mse_model_space": float(test_mse),
        "test_relative_l2_model_space": float(test_rel),
        "train_total_seconds": float(train_total_seconds),
        "trainer_config": train_config,
        "dataset_summary": bundle_summary,
        "history_keys": list(history.keys()),
    }

    save_json(
        summary,
        dirs["logs_dir"] / "train_summary.json",
    )

    root_summary = {
        "schema_version": "1.0",
        "run_completed": True,
        "run_type": "training",
        "experiment_family": "kerr_fno",
        "model_family": "fno",
        "task_name": effective_task_name,
        "model_name": model_name,
        "spatial_dimension": 2,
        "operator_axes": [
            bundle.param_name,
            "lambda",
        ],
        "vary_params": list(bundle.vary_params_order),
        "conditioning_params": (
            [str(bundle.cfg_param_name)]
            if bundle.cfg_param_name is not None
            else []
        ),
        "data_seeds": bundle.data_seeds,
        "training_seed": int(training_seed),
        "metrics": {
            "model_space": {
                "usage": "training_monitoring_only",
                "description": (
                    "Metrics computed in model space after the configured "
                    "normalization and target transform. These metrics are "
                    "used only for training monitoring and model selection."
                ),
                "normalization": str(args.normalization),
                "target_transform": str(args.target_transform),
                "best_val_mse": float(best_val_mse),
                "test_mse": float(test_mse),
                "test_relative_l2": float(test_rel),
            },
            "physical_space": None,
        },
        "timing": {
            "train_total_seconds": float(
                train_total_seconds
            ),
        },
        "model": {
            "model_type": str(args.model),
            "num_parameters": int(
                count_parameters(model)
            ),
            "model_config": model_config,
        },
        "training": {
            "epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler_gamma": float(
                args.scheduler_gamma
            ),
            "normalization": str(
                args.normalization
            ),
            "target_transform": str(
                args.target_transform
            ),
        },
        "artifacts": {
            "best_checkpoint": (
                "checkpoints/best_model.pt"
            ),
            "last_checkpoint": (
                "checkpoints/last_model.pt"
            ),
            "train_history": (
                "logs/train_history.json"
            ),
            "train_summary": (
                "logs/train_summary.json"
            ),
        },
    }

    save_json_common(
        root_summary,
        dirs["summary_json"],
        indent=2,
    )

    print("-" * 70)
    print("2D Training Completed")

    # 使用统一宽度打印键值，避免手工空格导致冒号不对齐。
    summary_key_width = 52
    summary_rows = [
        ("Best epoch", str(best_epoch)),
        (
            "Best validation MSE (model space, training only)",
            f"{best_val_mse:.6e}",
        ),
        (
            "Test MSE (model space, training only)",
            f"{test_mse:.6e}",
        ),
        (
            "Test Relative L2 (model space, training only)",
            f"{test_rel:.6e}",
        ),
        (
            "Training time",
            f"{train_total_seconds:.2f} s",
        ),
        (
            "Output directory",
            str(dirs["model_dir"]),
        ),
    ]

    for key, value in summary_rows:
        print(f"{key:<{summary_key_width}} : {value}")

    print("-" * 70)


if __name__ == "__main__":
    main()