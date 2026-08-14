from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_sparse_reconstruction_timesnet1d import (  # noqa: E402
    _checkpoint_payload,
    _save_json_new,
    set_training_seed,
    train_one_epoch,
)
from src.models.timesnet_lambda_isolated1d import (  # noqa: E402
    build_timesnet_lambda_isolated1d_model,
)
from src.training.trajectory_reconstruction.fno1d_reconstruction import (  # noqa: E402
    SparseReconstructionDataset,
    evaluate_reconstruction_model,
    fit_reconstruction_normalization,
    load_reconstruction_splits,
)


def build_parser() -> argparse.ArgumentParser:
    """构造 lambda-isolated TimesNet 稀疏重建训练命令行。"""
    parser = argparse.ArgumentParser(
        description="Train lambda-isolated period-selection TimesNet1D for sparse trajectory reconstruction."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sparse_reconstruction_timesnet_lambda_isolated1d"),
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-gamma", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """验证训练入口参数及固定公平比较配置。"""
    if not args.run_name.strip() or Path(args.run_name).name != args.run_name:
        raise ValueError("run_name must be a non-empty single directory name.")
    for name in ("stride", "epochs", "batch_size", "d_model", "d_ff", "blocks", "top_k"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(args.stride) < 2:
        raise ValueError("stride must be at least 2.")
    if int(args.num_workers) < 0 or int(args.print_every) <= 0:
        raise ValueError("num_workers must be non-negative and print_every must be positive.")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("lr must be positive and weight_decay must be non-negative.")
    if not 0.0 < float(args.scheduler_gamma) <= 1.0:
        raise ValueError("scheduler_gamma must be in (0, 1].")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")


@torch.no_grad()
def collect_period_selection_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> list[dict[str, object]]:
    """顺序收集测试集的 lambda-isolated period-selection 诊断。"""
    model.eval()
    block_batches: list[list[dict[str, object]]] | None = None
    for batch_index, batch in enumerate(loader):
        model_input = batch[0].to(device)
        diagnostics = model.inspect_frequency_diagnostics(model_input)
        if block_batches is None:
            block_batches = [[] for _ in diagnostics]
        if block_batches is None or len(diagnostics) != len(block_batches):
            raise RuntimeError("Period-selection diagnostic block count is inconsistent.")
        for block_index, diagnostic in enumerate(diagnostics):
            block_batches[block_index].append(
                {
                    "batch_index": int(batch_index),
                    "batch_size": int(model_input.shape[0]),
                    "selected_frequency_indices": [
                        int(value) for value in diagnostic["selected_frequency_indices"]
                    ],
                    "selected_periods": [int(value) for value in diagnostic["selected_periods"]],
                }
            )
    if block_batches is None or not block_batches[0]:
        raise ValueError("Diagnostic loader must contain at least one batch.")
    return [
        {"block_index": int(index), "batches": batches}
        for index, batches in enumerate(block_batches)
    ]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """执行与 canonical 协议一致的 lambda-isolated TimesNet 训练。"""
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
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    model = build_timesnet_lambda_isolated1d_model(
        in_dim=5, out_dim=3, d_model=int(args.d_model), d_ff=int(args.d_ff),
        num_blocks=int(args.blocks), top_k=int(args.top_k),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.scheduler_gamma))
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir, logs_dir, metrics_dir = output_dir / "checkpoints", output_dir / "logs", output_dir / "metrics"
    checkpoints_dir.mkdir(); logs_dir.mkdir(); metrics_dir.mkdir()
    run_config: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_type": "sparse_trajectory_reconstruction_timesnet_lambda_isolated_period_selection1d",
        "dataset_path": str(splits.dataset_path), "q_input": "excluded",
        "input_channel_names": ["sparse_x", "sparse_y", "sparse_z", "observed_mask", "lambda_coordinate"],
        "input_shape_per_sample": list(train_dataset.model_input.shape[1:]),
        "output_shape_per_sample": [int(splits.train.target_xyz.shape[1]), 3],
        "sampling": splits.train.sampling.to_dict(), "normalization": normalization.to_dict(),
        "checkpoint_selection": "validation_raw_hidden_only_overall_relative_l2",
        "training": {"epochs": int(args.epochs), "batch_size": int(args.batch_size), "seed": int(args.seed), "optimizer": "AdamW", "lr": float(args.lr), "weight_decay": float(args.weight_decay), "scheduler": "ExponentialLR", "scheduler_gamma": float(args.scheduler_gamma), "device": str(args.device)},
        "model": model.architecture_metadata(),
        "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
    }
    _save_json_new(output_dir / "run_config.json", run_config)
    history: dict[str, list[float]] = {"train_normalized_hidden_mse": [], "val_normalized_hidden_mse": [], "val_raw_hidden_relative_l2": [], "lr": []}
    best_val_relative_l2, best_epoch = float("inf"), -1
    start_time = perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_metrics = evaluate_reconstruction_model(model, val_loader, args.device, normalization)
        val_relative_l2 = float(val_metrics["overall"]["relative_l2"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        for name, value in (("train_normalized_hidden_mse", train_loss), ("val_normalized_hidden_mse", val_loss), ("val_raw_hidden_relative_l2", val_relative_l2), ("lr", current_lr)):
            history[name].append(float(value))
        if val_relative_l2 < best_val_relative_l2:
            best_val_relative_l2, best_epoch = val_relative_l2, epoch
            torch.save(_checkpoint_payload(model, optimizer, epoch, best_val_relative_l2, run_config), checkpoints_dir / "best_model.pt")
        torch.save(_checkpoint_payload(model, optimizer, epoch, best_val_relative_l2, run_config), checkpoints_dir / "last_model.pt")
        scheduler.step()
        if epoch == 1 or epoch % int(args.print_every) == 0:
            print(f"Epoch {epoch}/{args.epochs}: train_hidden_mse={train_loss:.6e}, val_hidden_mse={val_loss:.6e}, val_hidden_relative_l2={val_relative_l2:.6e}, lr={current_lr:.6e}")
    best_checkpoint = torch.load(checkpoints_dir / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate_reconstruction_model(model, test_loader, args.device, normalization)
    diagnostics = collect_period_selection_diagnostics(model, test_loader, args.device)
    elapsed_seconds = perf_counter() - start_time
    _save_json_new(logs_dir / "train_history.json", history)
    _save_json_new(logs_dir / "train_summary.json", {"best_epoch": int(best_epoch), "best_validation_raw_hidden_relative_l2": best_val_relative_l2, "test_normalized_hidden_mse": float(test_loss), "training_seconds": float(elapsed_seconds)})
    _save_json_new(metrics_dir / "test_hidden_only_metrics.json", {"split": "test", "model_space_hidden_mse": float(test_loss), "raw_hidden_only_metrics": test_metrics})
    _save_json_new(metrics_dir / "test_period_selection_diagnostics.json", {"split": "test", "period_selection_treatment": "lambda_isolated_shared_parameter_auxiliary_stream", "blocks": diagnostics})
    summary = {"output_dir": str(output_dir), "best_epoch": int(best_epoch), "best_validation_raw_hidden_relative_l2": best_val_relative_l2, "test_normalized_hidden_mse": float(test_loss), "test_raw_hidden_relative_l2": float(test_metrics["overall"]["relative_l2"])}
    print("Lambda-isolated TimesNet1D reconstruction training completed.")
    print(f"Output directory: {output_dir}")
    print(f"Best validation hidden Relative L2: {best_val_relative_l2:.6e}")
    print(f"Test raw hidden Relative L2: {summary['test_raw_hidden_relative_l2']:.6e}")
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
