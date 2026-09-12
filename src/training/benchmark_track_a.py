"""Unified Track A training, immutable checkpoints, and frozen resolution evaluation."""

import csv
import io
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter, time_ns

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.benchmark_track_a import MODEL_CONFIGS, build_model, model_config, parameter_counts
from src.training.benchmark_data import (
    PROJECT_ROOT, TRAIN_TASKS, EVAL_TASKS, load_training, load_evaluation,
    fit_normalization, restore_normalization, recover_xyz, sha256_file,
)
from src.training.optimizer_builder import build_optimizer, build_scheduler


SCHEMA = "benchmark_track_a_v1"
TRAINING = {"epochs": 500, "batch_size": 32, "optimizer": "AdamW", "lr": 1e-3,
            "weight_decay": 1e-4, "scheduler": "ExponentialLR", "gamma": 0.995,
            "seed": 27, "normalization": "standard", "target_transform": "raw",
            "loss": "normalized_mse", "selection": "validation_normalized_mse"}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class GPUPreflightError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"GPU preflight {status}: {message}")


def inspect_device(device, host_gpu=None):
    """只查询必要资源元数据；绝不驱逐任务或自动更换 GPU。"""
    if str(device) == "cpu":
        return {"device": "cpu", "host_gpu": None, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    mapping = os.environ.get("CUDA_VISIBLE_DEVICES")
    if str(device) != "cuda:0" or type(host_gpu) is not int or host_gpu not in range(4) or mapping != str(host_gpu):
        raise ValueError("Use one explicit host GPU and matching CUDA_VISIBLE_DEVICES; local device must be cuda:0.")
    try:
        def query(fields):
            output = subprocess.check_output([
                "nvidia-smi", f"--id={host_gpu}", fields, "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.PIPE, timeout=10)
            return [[value.strip() for value in row] for row in
                    csv.reader(io.StringIO(output), strict=True, skipinitialspace=True) if row]

        rows = query("--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu")
        if len(rows) != 1 or len(rows[0]) != 6:
            raise ValueError("Expected one complete selected-GPU inventory row.")
        selected = rows[0]
        if int(selected[0]) != host_gpu or not selected[1].startswith("GPU-") or not selected[2]:
            raise ValueError("Selected host GPU identity could not be established.")
        memory, free_memory, utilization = map(float, selected[3:])
        if (not all(math.isfinite(value) for value in (memory, free_memory, utilization))
                or min(memory, free_memory) < 0 or not 0 <= utilization <= 100):
            raise ValueError("Invalid selected-GPU diagnostic metadata.")
        occupancy = query("--query-compute-apps=gpu_uuid,pid")
        for row in occupancy:
            if len(row) != 2 or row[0] != selected[1] or int(row[1]) <= 0:
                raise ValueError("Compute-process metadata is ambiguous.")
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as error:
        raise GPUPreflightError("UNKNOWN", "GPU metadata could not be determined reliably. "
                                "A restricted-context query failure does not prove a broken GPU/driver. "
                                "Recheck through the established approved server execution context; "
                                "do not launch a workload until preflight succeeds.") from error
    # 所有既有计算任务均视为占用，包括本用户任务；无需读取其 UID 或私有进程数据。
    if occupancy:
        raise GPUPreflightError("OCCUPIED", "Selected GPU already has a compute workload; "
                                "this includes other jobs owned by the current user.")
    # 非零显存可能是图形/系统常驻分配；明确空的计算进程列表才是此处的空闲依据。
    try:
        cuda_visible = torch.cuda.is_available() and torch.cuda.device_count() == 1
    except RuntimeError as error:
        raise GPUPreflightError("UNKNOWN", "CUDA visibility query failed in this execution context; "
                                "recheck in the approved server context before launching.") from error
    if not cuda_visible:
        raise GPUPreflightError("UNKNOWN", "Expected exactly one visible CUDA device in this execution context; "
                                "recheck in the approved server context before launching.")
    return {"device": "cuda:0", "host_gpu": host_gpu, "CUDA_VISIBLE_DEVICES": mapping,
            "gpu_uuid": selected[1], "gpu_name": selected[2],
            "preflight_status": "AVAILABLE", "preflight_memory_used_mib": memory,
            "preflight_memory_free_mib": free_memory, "preflight_utilization_percent": utilization,
            "compute_process_count": 0, "own_compute_process_count": 0}


def source_provenance():
    # 保存实际执行源码的摘要，包括尚未提交但获准执行的实现。
    files = [Path(__file__), PROJECT_ROOT / "src/models/benchmark_track_a.py",
             PROJECT_ROOT / "src/training/benchmark_data.py",
             PROJECT_ROOT / "scripts/run_benchmark_track_a.py"]
    for directory in ("src/models/fno1d", "src/training/fno2d", "src/data_generation"):
        files.extend((PROJECT_ROOT / directory).glob("*.py"))
    files.extend(PROJECT_ROOT / name for name in (
        "src/models/resnet1d.py", "src/models/timesnet1d.py", "src/training/optimizer_builder.py",
        "src/training/fno1d/input_builder_1d.py", "scripts/evaluate_plan_b_resolution_generalization_2d.py",
        "scripts/evaluate_formal_length_extrapolation_2d.py"))
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(set(files))}}


def output_path(path):
    path = Path(path).resolve()
    root = (PROJECT_ROOT / "outputs").resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Benchmark artifacts must stay within a new directory under outputs/.")
    return path


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_checkpoint(path, payload):
    # 每个 epoch 使用唯一文件，恢复训练也不覆盖任何既有 checkpoint。
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def seed_training():
    random.seed(TRAINING["seed"])
    np.random.seed(TRAINING["seed"])
    torch.manual_seed(TRAINING["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(device):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def epoch_mse(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite normalized-space MSE; stop without changing protocol.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
    if count == 0:
        raise ValueError("Empty trajectory loader.")
    return total / count


def checkpoint_contract(payload):
    if payload.get("schema") != SCHEMA or payload.get("training") != TRAINING:
        raise ValueError("Checkpoint is not a locked Track A checkpoint; sparse checkpoints are not baselines.")
    if payload["train_resolution"] not in TRAIN_TASKS:
        raise ValueError("Checkpoint native resolution is invalid.")
    stats = restore_normalization(payload["normalization"])
    model = build_model(payload["model_name"], payload["model_config"])
    if parameter_counts(model) != payload["capacity"]:
        raise ValueError("Checkpoint capacity metadata mismatch.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, stats


def fit_epochs(model, datasets, stats, device, directory, metadata, *, epochs=500, resume=None):
    """公共训练内核；缩短 epochs 仅供合成单元测试，正式入口固定为 500。"""
    if not 1 <= epochs <= TRAINING["epochs"] or (resume is not None and resume["epoch"] > epochs):
        raise ValueError("Invalid training/recovery epoch boundary.")
    for dataset in datasets.values():
        dataset.stats = stats
    train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(datasets["val"], batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    optimizer = build_optimizer(model, lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    scheduler = build_scheduler(optimizer, scheduler_gamma=TRAINING["gamma"])
    start_epoch, history, best, best_epoch, prior_seconds = 0, [], float("inf"), 0, 0.0
    best_state = None
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch, history = resume["epoch"], resume["history"]
        best, best_epoch = resume["best_validation_mse"], resume["best_epoch"]
        prior_seconds = resume["training_wall_seconds"]
        best_state = resume["best_model_state_dict"]
        restore_rng(resume["rng"], device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = perf_counter()
    for epoch in range(start_epoch + 1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_mse = epoch_mse(model, train_loader, device, optimizer)
        val_mse = epoch_mse(model, val_loader, device)
        scheduler.step()
        if val_mse < best:
            best, best_epoch = val_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_normalized_mse": train_mse,
                        "validation_normalized_mse": val_mse, "learning_rate": lr})
        synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
        if resume is not None and peak is not None:
            peak = max(peak, resume.get("peak_gpu_memory_bytes") or 0)
        payload = {**metadata, "normalization": stats.to_dict(), "epoch": epoch,
                   "best_epoch": best_epoch, "best_validation_mse": best, "history": history,
                   "best_model_state_dict": best_state,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "rng": rng_state(device),
                   "training_wall_seconds": prior_seconds + perf_counter() - start,
                   "peak_gpu_memory_bytes": peak}
        # 固定间隔保留可恢复状态；最优权重随状态保存，绝不改变每轮选择策略。
        if epoch % 25 == 0 or epoch == epochs:
            save_checkpoint(directory / f"epoch_{epoch:04d}.pt", payload)
    synchronize(device)
    return {"best_epoch": best_epoch, "best_validation_mse": best,
            "best_checkpoint": "best_model.pt", "completed_epochs": epochs,
            "training_wall_seconds": prior_seconds + perf_counter() - start,
            "peak_gpu_memory_bytes": (max(torch.cuda.max_memory_allocated(device),
                                          (resume or {}).get("peak_gpu_memory_bytes") or 0)
                                      if torch.device(device).type == "cuda" else None),
            "history": history}


def train_run(name, resolution, directory, device="cpu", host_gpu=None, approved=False, resume_path=None):
    if not approved:
        raise PermissionError("A specific Phase I training run requires explicit execution approval.")
    runtime = inspect_device(device, host_gpu)
    directory = output_path(directory)
    if resume_path is None and directory.exists():
        raise FileExistsError("Run directory already exists; no overwrite is permitted.")
    datasets, provenance = load_training(resolution)
    seed_training()
    model = build_model(name)
    stats = fit_normalization(datasets["train"])
    metadata = {"schema": SCHEMA, "model_name": name, "model_config": model_config(name),
                "train_resolution": resolution, "training": dict(TRAINING),
                "capacity": parameter_counts(model), "dataset_provenance": provenance,
                "source_provenance": source_provenance(), "runtime": runtime,
                "environment": {"python_executable": sys.executable, "torch_version": str(torch.__version__),
                                "torch_cuda_build": torch.version.cuda},
                "checkpoint_interval_epochs": 25}
    resume = None
    if resume_path is not None:
        resume_path = Path(resume_path).resolve()
        if resume_path.parent != directory:
            raise ValueError("Resume must use an immutable checkpoint in the same run directory.")
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        checkpoint_contract(resume)
        for key in ("model_name", "model_config", "train_resolution", "training", "dataset_provenance", "source_provenance", "environment"):
            if resume[key] != metadata[key]:
                raise ValueError(f"Resume provenance mismatch: {key}.")
        if resume["normalization"] != stats.to_dict() or resume["runtime"]["device"] != runtime["device"]:
            raise ValueError("Resume normalization/device mismatch.")
        if resume["epoch"] not in range(1, 501) or resume_path.name != f"epoch_{resume['epoch']:04d}.pt":
            raise ValueError("Resume requires a valid epoch checkpoint, not the selected frozen model.")
        if (directory / "summary.json").exists() or any(
            (directory / f"epoch_{epoch:04d}.pt").exists() for epoch in range(resume["epoch"] + 1, 501)
        ):
            raise FileExistsError("Completed or later epoch assets already exist; no overwrite is permitted.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "run.json", metadata)
    try:
        model.to(device)
        summary = fit_epochs(model, datasets, stats, device, directory, metadata, resume=resume)
    except (RuntimeError, ValueError) as error:
        write_json(directory / f"failure_{time_ns()}.json", {
            "status": "STOPPED_NO_PROTOCOL_ADAPTATION", "error_type": type(error).__name__,
            "message": str(error), "runtime": runtime, "training": TRAINING})
        raise
    final_state = torch.load(directory / "epoch_0500.pt", map_location="cpu", weights_only=True)
    best_payload = {**{key: final_state[key] for key in metadata},
                    "normalization": stats.to_dict(), "epoch": summary["best_epoch"],
                    "model_state_dict": final_state["best_model_state_dict"], "completed_epochs": 500,
                    "selection": "minimum_validation_normalized_mse_over_500_epochs"}
    best_path = directory / "best_model.pt"
    if best_path.exists():
        existing = torch.load(best_path, map_location="cpu", weights_only=True)
        if (existing.keys() != best_payload.keys()
                or any(existing[key] != value for key, value in best_payload.items() if key != "model_state_dict")
                or existing["model_state_dict"].keys() != best_payload["model_state_dict"].keys()
                or any(not torch.equal(existing["model_state_dict"][key], value)
                       for key, value in best_payload["model_state_dict"].items())):
            raise ValueError("Existing selected checkpoint provenance mismatch; no overwrite is permitted.")
    else:
        save_checkpoint(best_path, best_payload)
    summary["best_checkpoint_sha256"] = sha256_file(best_path)
    write_json(directory / "summary.json", {**metadata, **summary})
    return summary


@torch.no_grad()
def predict_frozen(model, dataset, stats, device):
    model.eval()
    dataset.stats = stats
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions, seconds = [], 0.0
    # 固定 canonical Q 顺序和 batch=32；TimesNet 仍使用 batch-shared top-k。
    iterator = iter(loader)
    while True:
        synchronize(device)
        start = perf_counter()
        try:
            x, _ = next(iterator)
        except StopIteration:
            break
        y = model(x.to(device))
        synchronize(device)
        seconds += perf_counter() - start
        predictions.append(y.cpu().numpy())
    raw = recover_xyz(np.concatenate(predictions), stats)
    return raw, {"inference_wall_seconds": seconds, "evaluation_batch_size": 32,
                 "timing_scope": "loader_and_host_to_device_and_forward; no_warmup; excludes_inverse_normalization_metrics_output",
                 "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                           if torch.device(device).type == "cuda" else None)}


def resolution_routing(native):
    if native not in TRAIN_TASKS:
        raise ValueError("Unknown native training resolution.")
    return {t: ("native" if t == native else "reverse" if t < native else "cross") for t in EVAL_TASKS}


def robustness(metrics, native):
    baseline = metrics[native]
    return {t: {key: {"ratio_to_native": values[key] / baseline[key] if baseline[key] != 0 else None,
                     "delta_from_native": values[key] - baseline[key]}
                for key in ("global_relative_l2", "mean_per_q_relative_l2")}
            for t, values in metrics.items()}


def evaluate_run(checkpoint_path, directory, device="cpu", host_gpu=None, approved=False):
    from scripts.evaluate_plan_b_resolution_generalization_2d import compute_raw_metrics

    if not approved:
        raise PermissionError("Frozen formal evaluation requires specific execution approval.")
    runtime = inspect_device(device, host_gpu)
    checkpoint_path = output_path(checkpoint_path)
    directory = output_path(directory)
    if directory.exists():
        raise FileExistsError("Evaluation directory already exists; no overwrite is permitted.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, stats = checkpoint_contract(payload)
    current_source = source_provenance()
    if payload["source_provenance"]["source_sha256"] != current_source["source_sha256"]:
        raise ValueError("Frozen evaluation implementation hashes differ from the training checkpoint.")
    summary = json.loads((checkpoint_path.parent / "summary.json").read_text(encoding="utf-8"))
    if (summary["completed_epochs"] != 500 or summary["best_checkpoint"] != checkpoint_path.name
            or payload.get("completed_epochs") != 500 or payload["epoch"] != summary["best_epoch"]
            or summary["best_checkpoint_sha256"] != sha256_file(checkpoint_path)):
        raise ValueError("Frozen evaluation requires the selected best checkpoint from a completed 500-epoch run.")
    _, training_provenance = load_training(payload["train_resolution"])
    if training_provenance != payload["dataset_provenance"]:
        raise ValueError("Checkpoint training-data provenance mismatch.")
    model.to(device)
    model.requires_grad_(False)
    routing = resolution_routing(payload["train_resolution"])
    data = {t: load_evaluation(t) for t in EVAL_TASKS}
    reference_q = data[1200][0].q
    if any(not np.array_equal(dataset.q, reference_q) for dataset, _ in data.values()):
        raise ValueError("Protocol-impacting issue: evaluation Q identities/order differ across resolutions.")
    directory.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for t, (dataset, provenance) in data.items():
        try:
            prediction, compute = predict_frozen(model, dataset, stats, device)
        except (RuntimeError, ValueError) as error:
            write_json(directory / f"failure_t{t}.json", {
                "status": "STOPPED_NO_PROTOCOL_ADAPTATION", "resolution": t,
                "error_type": type(error).__name__, "message": str(error), "runtime": runtime})
            raise
        metrics[t] = {**compute_raw_metrics(prediction, dataset.xyz, dataset.q), **compute,
                      "route": routing[t], "dataset_provenance": provenance}
        with (directory / f"prediction_t{t}.npz").open("xb") as stream:
            np.savez_compressed(stream, prediction_xyz=prediction, Q=dataset.q, lambda_grid=dataset.lambda_grid)
        write_json(directory / f"metrics_t{t}.json", metrics[t])
    result = {"schema": SCHEMA, "checkpoint": str(checkpoint_path),
              "checkpoint_sha256": sha256_file(checkpoint_path), "model_name": payload["model_name"],
              "native_resolution": payload["train_resolution"], "capacity": payload["capacity"],
              "normalization": stats.to_dict(), "runtime": runtime,
              "source_provenance": current_source, "metrics": metrics,
              "resolution_robustness": robustness(metrics, payload["train_resolution"])}
    write_json(directory / "matrix.json", result)
    return result


def planned_matrix():
    return [{"model": name, "train_resolution": native,
             "evaluation_resolutions": list(EVAL_TASKS), "status": "PLANNED_NOT_EXECUTED"}
            for name in MODEL_CONFIGS for native in TRAIN_TASKS]
