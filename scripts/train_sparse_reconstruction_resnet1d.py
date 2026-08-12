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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.resnet1d import (  # noqa: E402
    DEFAULT_DILATION_SCHEDULE,
    build_dilated_resnet1d,
)
from src.training.trajectory_reconstruction.fno1d_reconstruction import (  # noqa: E402
    SparseReconstructionDataset,
    evaluate_reconstruction_model,
    fit_reconstruction_normalization,
    hidden_only_mse_loss,
    load_reconstruction_splits,
)


def build_parser() -> argparse.ArgumentParser:
    """构造稀疏 Dilated ResNet1D 重建训练命令行。"""
    parser = argparse.ArgumentParser(
        description="Train Dilated ResNet1D for sparse trajectory reconstruction."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sparse_reconstruction_resnet1d"),
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=92)
    parser.add_argument("--blocks", type=int, default=len(DEFAULT_DILATION_SCHEDULE))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def set_training_seed(seed: int) -> None:
    """设置本训练入口使用的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_args(args: argparse.Namespace) -> None:
    """验证训练入口参数。"""
    if not args.run_name.strip():
        raise ValueError("run_name must not be empty.")
    if Path(args.run_name).name != args.run_name:
        raise ValueError("run_name must be a single directory name.")
    for name in ("epochs", "batch_size", "width", "blocks"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(args.blocks) > len(DEFAULT_DILATION_SCHEDULE):
        raise ValueError("blocks exceeds the approved dilation schedule length.")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must not be negative.")
    if int(args.print_every) <= 0:
        raise ValueError("print_every must be positive.")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("lr must be positive and weight_decay must be non-negative.")
    if not 0.0 < float(args.scheduler_gamma) <= 1.0:
        raise ValueError("scheduler_gamma must be in (0, 1].")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")


def _save_json_new(path: Path, data: dict[str, Any]) -> None:
    """以排他方式写入训练 JSON。"""
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_relative_l2: float,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """构造 checkpoint 的可复现实验状态。"""
    return {
        "epoch": int(epoch),
        "best_val_relative_l2": float(best_val_relative_l2),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_config": run_config,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """训练一个 epoch 并按 hidden scalar count 汇总损失。"""
    model.train()
    total_squared_error = 0.0
    total_hidden_scalar_count = 0

    for batch in loader:
        model_input, target_normalized, _, _, _, hidden_mask = batch
        model_input = model_input.to(device)
        target_normalized = target_normalized.to(device)
        hidden_mask = hidden_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(model_input)
        loss = hidden_only_mse_loss(
            prediction,
            target_normalized,
            hidden_mask,
        )
        loss.backward()
        optimizer.step()

        hidden_scalar_count = int(hidden_mask.sum().detach().cpu()) * 3
        total_squared_error += float(loss.detach().cpu()) * hidden_scalar_count
        total_hidden_scalar_count += hidden_scalar_count

    if total_hidden_scalar_count == 0:
        raise RuntimeError("Training loader contains no hidden points.")
    return total_squared_error / total_hidden_scalar_count


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """执行独立的稀疏 Dilated ResNet1D 训练流程。"""
    _validate_args(args)
    set_training_seed(int(args.seed))

    output_dir = (Path(args.output_root) / str(args.run_name)).resolve()
    if output_dir.exists():
        raise FileExistsError(
            "Output run directory already exists and will not be overwritten: "
            f"{output_dir}"
        )

    splits = load_reconstruction_splits(args.dataset_path, stride=int(args.stride))
    normalization = fit_reconstruction_normalization(splits.train)
    train_dataset = SparseReconstructionDataset(splits.train, normalization)
    val_dataset = SparseReconstructionDataset(splits.val, normalization)
    test_dataset = SparseReconstructionDataset(splits.test, normalization)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = build_dilated_resnet1d(
        in_dim=5,
        out_dim=3,
        width=int(args.width),
        blocks=int(args.blocks),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(args.scheduler_gamma),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir()
    logs_dir.mkdir()
    metrics_dir.mkdir()

    run_config: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_type": "sparse_trajectory_reconstruction_resnet1d",
        "dataset_path": str(splits.dataset_path),
        "q_input": "excluded",
        "input_channel_names": [
            "sparse_x",
            "sparse_y",
            "sparse_z",
            "observed_mask",
            "lambda_coordinate",
        ],
        "input_shape_per_sample": list(train_dataset.model_input.shape[1:]),
        "output_shape_per_sample": [
            int(splits.train.target_xyz.shape[1]),
            3,
        ],
        "sampling": splits.train.sampling.to_dict(),
        "normalization": normalization.to_dict(),
        "checkpoint_selection": "validation_raw_hidden_only_overall_relative_l2",
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler": "ExponentialLR",
            "scheduler_gamma": float(args.scheduler_gamma),
            "device": str(args.device),
        },
        "model": model.architecture_metadata(),
        "split_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    }
    _save_json_new(output_dir / "run_config.json", run_config)

    history = {
        "train_normalized_hidden_mse": [],
        "val_normalized_hidden_mse": [],
        "val_raw_hidden_relative_l2": [],
        "lr": [],
    }
    best_val_relative_l2 = float("inf")
    best_epoch = -1
    start_time = perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_metrics = evaluate_reconstruction_model(
            model,
            val_loader,
            args.device,
            normalization,
        )
        val_relative_l2 = float(val_metrics["overall"]["relative_l2"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_normalized_hidden_mse"].append(float(train_loss))
        history["val_normalized_hidden_mse"].append(float(val_loss))
        history["val_raw_hidden_relative_l2"].append(val_relative_l2)
        history["lr"].append(current_lr)

        if val_relative_l2 < best_val_relative_l2:
            best_val_relative_l2 = val_relative_l2
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_relative_l2=best_val_relative_l2,
                    run_config=run_config,
                ),
                checkpoints_dir / "best_model.pt",
            )

        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_relative_l2=best_val_relative_l2,
                run_config=run_config,
            ),
            checkpoints_dir / "last_model.pt",
        )
        scheduler.step()

        if epoch == 1 or epoch % int(args.print_every) == 0:
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"train_hidden_mse={train_loss:.6e}, "
                f"val_hidden_mse={val_loss:.6e}, "
                f"val_hidden_relative_l2={val_relative_l2:.6e}, "
                f"lr={current_lr:.6e}"
            )

    best_checkpoint = torch.load(
        checkpoints_dir / "best_model.pt",
        map_location=args.device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate_reconstruction_model(
        model,
        test_loader,
        args.device,
        normalization,
    )
    elapsed_seconds = perf_counter() - start_time

    _save_json_new(logs_dir / "train_history.json", history)
    _save_json_new(
        logs_dir / "train_summary.json",
        {
            "best_epoch": int(best_epoch),
            "best_validation_raw_hidden_relative_l2": best_val_relative_l2,
            "test_normalized_hidden_mse": float(test_loss),
            "training_seconds": float(elapsed_seconds),
        },
    )
    _save_json_new(
        metrics_dir / "test_hidden_only_metrics.json",
        {
            "split": "test",
            "model_space_hidden_mse": float(test_loss),
            "raw_hidden_only_metrics": test_metrics,
        },
    )

    summary = {
        "output_dir": str(output_dir),
        "best_epoch": int(best_epoch),
        "best_validation_raw_hidden_relative_l2": best_val_relative_l2,
        "test_normalized_hidden_mse": float(test_loss),
        "test_raw_hidden_relative_l2": float(
            test_metrics["overall"]["relative_l2"]
        ),
    }
    print("Sparse Dilated ResNet1D reconstruction training completed.")
    print(f"Output directory: {output_dir}")
    print(f"Best validation hidden Relative L2: {best_val_relative_l2:.6e}")
    print(
        "Test raw hidden Relative L2: "
        f"{summary['test_raw_hidden_relative_l2']:.6e}"
    )
    return summary


def main() -> None:
    """命令行主入口。"""
    try:
        run_training(build_parser().parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
