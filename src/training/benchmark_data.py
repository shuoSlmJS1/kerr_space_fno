"""Registered Track A data, unchanged split identities, and train-only normalization."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_generation.plan_b_matched_training import (
    HISTORICAL_T1200_TRAINING_CONTRACT, source_split_q_hashes,
    validate_historical_t1200_source,
)
from src.data_generation.plan_b_paired import (
    SPLITS, load_dataset_artifact, _fixed_params, _solver_provenance,
)
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import (
    FieldNormalizationStats, validate_field_array,
    normalize_input_field, normalize_output_field, denormalize_output_field,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_TASKS = {1200: "q_1p6-3_n2000_t1200", 2399: "q_1p6-3_n2000_t2399_plan_b_matched_v1"}
EVAL_TASKS = {
    1200: "q_1p6007-2p9993_n400_t1200",
    2399: "q_1p6007-2p9993_n400_t2399_plan_b_v1",
    3598: "q_1p6007-2p9993_n400_t3598_plan_b_range_v1",
    4797: "q_1p6007-2p9993_n400_t4797_plan_b_range_v1",
}
STEPS = {1200: 0.005, 2399: 0.0025, 3598: 0.005 / 3, 4797: 0.00125}


def task_path(task):
    path = PROJECT_ROOT / "data" / "tasks" / task
    if not path.resolve().is_relative_to((PROJECT_ROOT / "data" / "tasks").resolve()):
        raise ValueError("Dataset path escapes the registered project scope.")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_provenance(artifact):
    return {
        "directory": str(artifact.directory),
        "dataset_sha256": sha256_file(artifact.directory / "dataset.npz"),
        "metadata_sha256": sha256_file(artifact.directory / "meta.json"),
        "split_q_sha256": source_split_q_hashes(artifact),
    }


def validate_artifact(artifact, resolution):
    grid = artifact.arrays["lambda_grid"]
    if grid.shape != (resolution,) or not np.allclose(
        grid, np.arange(resolution, dtype=np.float64) * STEPS[resolution], rtol=0, atol=1e-12
    ):
        raise ValueError("Protocol-impacting issue: physical lambda grid mismatch.")
    if artifact.failed_samples:
        raise ValueError("Protocol-impacting issue: dataset has failed samples.")
    for split in SPLITS:
        q, y = artifact.arrays[f"x_{split}"], artifact.arrays[f"y_{split}"]
        if y.shape != (len(q), resolution, 3) or y.dtype != np.float64:
            raise ValueError("Dataset must retain aligned raw float64 xyz truth.")
        if not np.isfinite(y).all() or not np.isfinite(q).all():
            raise ValueError("Dataset contains non-finite values.")
        mask = artifact.arrays.get(f"success_mask_{split}")
        if mask is not None and not np.all(mask):
            raise ValueError("Dataset contains unsuccessful rows.")


def validate_pair(source, target):
    for split in SPLITS:
        if not np.array_equal(source.arrays[f"x_{split}"], target.arrays[f"x_{split}"]):
            raise ValueError("Protocol-impacting issue: Q split identity or row order mismatch.")
    if _fixed_params(source.metadata) != _fixed_params(target.metadata):
        raise ValueError("Protocol-impacting issue: fixed physics mismatch.")
    if _solver_provenance(source.metadata) != _solver_provenance(target.metadata):
        raise ValueError("Protocol-impacting issue: solver provenance mismatch.")
    replay = target.metadata.get("paired_replay", {})
    if (replay.get("paired_completeness") is not True
            or replay.get("target_success_replacement_used") is not False
            or replay.get("replacement_policy") != "forbidden"
            or Path(replay.get("source_dataset_path", "")).resolve() != source.directory):
        raise ValueError("Protocol-impacting issue: paired dataset provenance mismatch.")


class Trajectories(Dataset):
    def __init__(self, q, lambda_grid, xyz, stats=None):
        self.q = np.array(q, dtype=np.float64, copy=True).reshape(-1, 1)
        self.lambda_grid = np.array(lambda_grid, dtype=np.float64, copy=True)
        self.xyz = np.asarray(xyz)
        if self.xyz.shape != (len(self.q), len(self.lambda_grid), 3):
            raise ValueError("Q/lambda/xyz shapes do not agree.")
        self.stats = stats

    def raw_input(self):
        return build_fno1d_input_array(self.q, self.lambda_grid)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, index):
        x = build_fno1d_input_array(self.q[index:index + 1], self.lambda_grid)[0]
        y = self.xyz[index].astype(np.float32)
        if self.stats is not None:
            x = normalize_input_field(x[None, None], self.stats)[0, 0]
            y = normalize_output_field(y[None, None], self.stats)[0, 0]
        return torch.from_numpy(x), torch.from_numpy(y)


def _benchmark_channel_mean_std(array, eps):
    # 仅新 benchmark 使用 float64 归约；历史 FNO2D 统计路径保持不变。
    validate_field_array(array, name="array")
    mean = np.mean(array, axis=(0, 1, 2), dtype=np.float64)
    std = np.std(array, axis=(0, 1, 2), dtype=np.float64, ddof=0)
    return mean.tolist(), np.maximum(std, eps).tolist()


def fit_normalization(train: Trajectories):
    # 保留既有 float32 样本及单例场维度、统计轴和 epsilon；只拟合 train。
    # Python float 列表保留 float64 统计精度，应用时才沿用 float32 运算。
    eps = 1e-8
    x_mean, x_std = _benchmark_channel_mean_std(train.raw_input()[None], eps)
    y_mean, y_std = _benchmark_channel_mean_std(train.xyz.astype(np.float32)[None], eps)
    return FieldNormalizationStats(
        method="standard", x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, eps=eps,
    )


def restore_normalization(payload):
    stats = FieldNormalizationStats.from_dict(payload)
    if stats.method != "standard" or len(stats.x_mean) != 2 or len(stats.y_mean) != 3:
        raise ValueError("Checkpoint normalization contract mismatch.")
    for values, count in ((stats.x_mean, 2), (stats.x_std, 2), (stats.y_mean, 3), (stats.y_std, 3)):
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError("Invalid checkpoint normalization statistics.")
    if min(stats.x_std + stats.y_std) <= 0 or stats.eps != 1e-8:
        raise ValueError("Invalid checkpoint normalization scale or epsilon.")
    return stats


def recover_xyz(prediction, stats):
    return denormalize_output_field(prediction[None], stats)[0]


def load_training(resolution):
    if resolution not in TRAIN_TASKS:
        raise ValueError("Training resolution must be T1200 or T2399.")
    source, _ = validate_historical_t1200_source(task_path(TRAIN_TASKS[1200]))
    artifact = source if resolution == 1200 else load_dataset_artifact(task_path(TRAIN_TASKS[resolution]))
    validate_artifact(artifact, resolution)
    if resolution != 1200:
        validate_pair(source, artifact)
    if source_split_q_hashes(artifact) != HISTORICAL_T1200_TRAINING_CONTRACT.split_q_sha256:
        raise ValueError("Protocol-impacting issue: historical split hash mismatch.")
    datasets = {split: Trajectories(artifact.arrays[f"x_{split}"], artifact.arrays["lambda_grid"],
                                    artifact.arrays[f"y_{split}"]) for split in SPLITS}
    return datasets, artifact_provenance(artifact)


def load_evaluation(resolution):
    from scripts.evaluate_plan_b_resolution_generalization_2d import build_canonical_field_from_artifact

    if resolution not in EVAL_TASKS:
        raise ValueError("Evaluation resolution is outside the locked matrix.")
    source = load_dataset_artifact(task_path(EVAL_TASKS[1200]))
    artifact = source if resolution == 1200 else load_dataset_artifact(task_path(EVAL_TASKS[resolution]))
    validate_artifact(source, 1200)
    validate_artifact(artifact, resolution)
    if resolution != 1200:
        validate_pair(source, artifact)
    field = build_canonical_field_from_artifact(artifact.directory)
    if len(field.canonical_q) != 400:
        raise ValueError("Protocol-impacting issue: evaluation must contain canonical Q400.")
    provenance = artifact_provenance(artifact)
    provenance["canonical_to_source_index"] = field.canonical_to_source_index.tolist()
    provenance["source_records"] = field.source_records
    return Trajectories(field.canonical_q, field.lambda_grid, field.canonical_truth), provenance
