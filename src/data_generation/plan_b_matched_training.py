"""Plan B matched T2399 training-data contract and replay validation.

该模块只定义反向实验的历史 T1200 训练任务约束，并复用 Protocol v1 的
paired replay 生成器。它不重建 Q grid、不改变 split，也不进行失败替换。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    FINE_N_STEPS,
    FINE_STEP_SIZE,
    SPLITS,
    DatasetArtifact,
    OrbitSimulator,
    generate_paired_fine_dataset,
    load_dataset_artifact,
    protocol_grid,
)


@dataclass(frozen=True)
class HistoricalT1200TrainingContract:
    """人工确认的历史 Q-only T1200 baseline 训练事实。"""

    source_task_name: str
    split_counts: dict[str, int]
    split_q_sha256: dict[str, str]
    fixed_params: dict[str, float | int]
    data_seed: int
    q_range: tuple[float, float]
    original_candidate_count: int
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    historical_checkpoint_path: str
    best_epoch: int
    best_val_mse: float


HISTORICAL_T1200_TRAINING_CONTRACT = HistoricalT1200TrainingContract(
    source_task_name="q_1p6-3_n2000_t1200",
    split_counts={"train": 1400, "val": 300, "test": 300},
    split_q_sha256={
        "train": "19568b3ebb25494faa0f769874d3aa02c0e32bae325e7c968308ffe297e964ea",
        "val": "065795c3ac45c8fbf52e8e3c115c0daeb57b476609f73499dc9f9cd420e54275",
        "test": "c67d057aba98eb080c374280524751deb44d752a58425f3dfff5b4dbcc377e80",
    },
    fixed_params={
        "M": 1.0,
        "a": 0.5,
        "E": 0.95,
        "Lz": 3.0,
        "r0": 10.0,
        "theta0": 1.2,
        "phi0": 0.0,
        "sign_r": -1,
        "sign_th": 1,
    },
    data_seed=10,
    q_range=(1.6, 3.0),
    original_candidate_count=2000,
    model_config={
        "model_type": "fno2d",
        "in_dim": 2,
        "out_dim": 3,
        "modes1": 16,
        "modes2": 32,
        "width": 64,
        "depth": 4,
        "hidden_dim": 128,
        "activation": "gelu",
        "num_parameters": 16802755,
    },
    training_config={
        "epochs": 500,
        "batch_size": 1,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "scheduler_gamma": 0.995,
        "training_seed": 27,
        "normalization": "standard",
        "target_transform": "raw",
        "lambda_reference_index": 0,
    },
    historical_checkpoint_path=(
        "outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/"
        "checkpoints/best_model.pt"
    ),
    best_epoch=500,
    best_val_mse=0.0004567307187244296,
)


def raw_float64_q_sha256(q_values: np.ndarray) -> str:
    """按正式 contract 定义哈希原始 split 行序的 float64 Q 值。"""

    values = np.asarray(q_values)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("Q values must have shape [N, 1].")
    canonical_dtype = np.ascontiguousarray(values[:, 0], dtype=np.float64)
    return hashlib.sha256(canonical_dtype.tobytes()).hexdigest()


def source_split_q_hashes(artifact: DatasetArtifact) -> dict[str, str]:
    """返回每个 source split 的 float64 row-order Q hash。"""

    return {
        split: raw_float64_q_sha256(artifact.arrays[f"x_{split}"])
        for split in SPLITS
    }


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"Historical T1200 contract mismatch for {name}: {actual!r} != {expected!r}.")


def validate_historical_t1200_source(
    source_dataset_dir: str | Path,
    *,
    contract: HistoricalT1200TrainingContract = HISTORICAL_T1200_TRAINING_CONTRACT,
) -> tuple[DatasetArtifact, dict[str, Any]]:
    """在 replay 前验证人工确认的历史 split、physics 和采样 provenance。"""

    artifact = load_dataset_artifact(source_dataset_dir)
    metadata = artifact.metadata
    task_spec = metadata.get("task_spec")
    if not isinstance(task_spec, dict):
        raise ValueError("Historical source dataset is missing task_spec metadata.")
    _require_equal("task_name", str(metadata.get("task_name")), contract.source_task_name)
    if not np.array_equal(artifact.arrays["lambda_grid"], protocol_grid("coarse")):
        raise ValueError("Historical source lambda_grid does not satisfy T1200/h=0.005.")
    _require_equal("task_spec.n_steps", int(task_spec.get("n_steps", -1)), COARSE_N_STEPS)
    if not math.isclose(float(task_spec.get("step_size", math.nan)), COARSE_STEP_SIZE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Historical source step_size does not satisfy 0.005.")
    _require_equal("task_spec.seed", int(task_spec.get("seed", -1)), contract.data_seed)
    _require_equal("task_spec.vary_params", task_spec.get("vary_params"), ["Q"])
    vary_ranges = task_spec.get("vary_ranges")
    if not isinstance(vary_ranges, dict) or list(vary_ranges.get("Q", [])) != list(contract.q_range):
        raise ValueError("Historical source Q range does not match [1.6, 3.0].")
    sample_shape = task_spec.get("sample_shape")
    if sample_shape is not None and int(np.prod(sample_shape)) != contract.original_candidate_count:
        raise ValueError("Historical source sample_shape does not match the original candidate count.")
    fixed_params = task_spec.get("fixed_params")
    if not isinstance(fixed_params, dict):
        raise ValueError("Historical source task_spec is missing fixed_params.")
    for key, expected in contract.fixed_params.items():
        if key not in fixed_params or float(fixed_params[key]) != float(expected):
            raise ValueError(f"Historical T1200 contract mismatch for fixed_params.{key}.")
    _require_equal("task_spec.sampling_mode", task_spec.get("sampling_mode"), "grid")

    status = metadata.get("generation_status", metadata.get("generation", {}))
    if not isinstance(status, dict):
        raise ValueError("Historical source dataset is missing generation-status metadata.")
    _require_equal("generation success_count", int(status.get("success_count", -1)), contract.original_candidate_count)
    _require_equal("generation initial_candidate_count", int(status.get("initial_candidate_count", -1)), contract.original_candidate_count)
    failures = status.get("failure_count", status.get("fail_count", -1))
    _require_equal("generation failure_count", int(failures), 0)
    _require_equal("generation used_completion_sampling", bool(status.get("used_completion_sampling")), False)
    _require_equal("generation successful_points_strictly_uniform", bool(status.get("successful_points_strictly_uniform")), True)
    if artifact.failed_samples:
        raise ValueError("Historical source records failed samples despite its zero-failure contract.")

    actual_hashes = source_split_q_hashes(artifact)
    for split in SPLITS:
        _require_equal(f"{split} count", int(artifact.arrays[f"x_{split}"].shape[0]), contract.split_counts[split])
        _require_equal(f"{split} Q SHA256", actual_hashes[split], contract.split_q_sha256[split])
    return artifact, {
        "source_task_name": contract.source_task_name,
        "split_q_sha256": actual_hashes,
        "all_reference_hashes_match": True,
        "source_grid": {"T": COARSE_N_STEPS, "step_size": COARSE_STEP_SIZE, "lambda_max": 5.995},
    }


def replay_matched_t2399_training_dataset(
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    simulator: OrbitSimulator | None = None,
    progress_every: int = 50,
    contract: HistoricalT1200TrainingContract = HISTORICAL_T1200_TRAINING_CONTRACT,
) -> dict[str, Any]:
    """验证历史 source 后严格 replay 三个 split；任何失败均保留并阻止完成态。"""

    _, source_validation = validate_historical_t1200_source(source_dataset_dir, contract=contract)
    kwargs: dict[str, Any] = {"n_steps": FINE_N_STEPS, "step_size": FINE_STEP_SIZE, "progress_every": progress_every}
    if simulator is not None:
        kwargs["simulator"] = simulator
    replay_metadata = generate_paired_fine_dataset(source_dataset_dir, output_dir, **kwargs)
    generated = load_dataset_artifact(output_dir)
    generated_hashes = source_split_q_hashes(generated)
    hash_match = generated_hashes == contract.split_q_sha256
    if not hash_match:
        raise ValueError("Generated matched T2399 dataset does not preserve the reference split Q hashes.")
    if not np.array_equal(generated.arrays["lambda_grid"], protocol_grid("fine")):
        raise ValueError("Generated matched T2399 lambda_grid is not Protocol v1 T2399/h=0.0025.")
    source = load_dataset_artifact(source_dataset_dir)
    for split in SPLITS:
        if not np.array_equal(source.arrays[f"x_{split}"], generated.arrays[f"x_{split}"]):
            raise ValueError(f"Generated matched T2399 dataset changed {split} Q row order or identities.")
    paired = replay_metadata.get("paired_replay", {})
    complete = bool(paired.get("paired_completeness")) and int(paired.get("failure_count", -1)) == 0
    return {
        "source_validation": source_validation,
        "generated_split_q_sha256": generated_hashes,
        "fine_grid": {"T": FINE_N_STEPS, "step_size": FINE_STEP_SIZE, "lambda_max": 5.995},
        "fine_common_nodes_equal_source_grid": bool(
            np.array_equal(generated.arrays["lambda_grid"][::2], source.arrays["lambda_grid"])
        ),
        "matched_dataset_complete": complete,
        "training_permitted": complete,
        "replacement_policy": "forbidden",
        "replay_metadata": replay_metadata,
    }
