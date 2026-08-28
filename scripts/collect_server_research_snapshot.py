#!/usr/bin/env python3
"""Collect a compact, read-only Stage-1 research snapshot to stdout."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0"
DEFAULT_EMBED_FILE_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_EMBED_TOTAL_SOFT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_PARAMETER_ARRAY_MAX_BYTES = 1 * 1024 * 1024
DEFAULT_NPY_HEADER_MAX_BYTES = 1 * 1024 * 1024
APPROVED_ROOTS = (Path("data") / "tasks", Path("outputs"))
REGISTRY_NAME = "SERVER_DATA_EXPERIMENT_REGISTRY.md"

JSON_EVIDENCE_NAMES = {
    "meta.json",
    "run_config.json",
    "summary.json",
    "train_summary.json",
    "train_history.json",
    "history.json",
    "metrics.json",
    "test_hidden_only_metrics.json",
    "analysis_summary.json",
    "result.json",
    "common_test_summary.json",
    "common_test_results.json",
    "validation_summary.json",
}
CSV_EVIDENCE_NAMES = {
    "per_q_metrics.csv",
    "lambda_error_profile.csv",
    "trajectory_metrics.csv",
    "trajectory_checks.csv",
}
RUN_MARKER_NAMES = {
    "run_config.json",
    "summary.json",
    "train_summary.json",
    "train_history.json",
    "history.json",
    "metrics.json",
    "test_hidden_only_metrics.json",
    "analysis_summary.json",
    "result.json",
}
RUN_SUBDIRECTORIES = {"logs", "metrics", "inference", "analysis", "checkpoints"}


@dataclass
class CollectorState:
    project_root: Path
    embed_file_max_bytes: int
    embed_total_soft_max_bytes: int
    parameter_array_max_bytes: int
    npy_header_max_bytes: int
    errors: list[dict[str, str]] = field(default_factory=list)
    embedded_bytes: int = 0
    registry_status_by_path: dict[str, str] = field(default_factory=dict)
    normal_files: dict[str, Path] = field(default_factory=dict)
    inventory: list[dict[str, Any]] = field(default_factory=list)

    def add_error(
        self,
        phase: str,
        relative_path: str,
        error_kind: str,
        message: object,
    ) -> None:
        text = str(message).replace(str(self.project_root), "<project_root>")
        self.errors.append(
            {
                "phase": phase,
                "relative_path": relative_path,
                "error_kind": error_kind,
                "message": text[:500],
            }
        )


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _relative_path(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _safe_lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError:
        return None


def _file_record(project_root: Path, path: Path, path_stat: os.stat_result) -> dict[str, Any]:
    mode = path_stat.st_mode
    if stat.S_ISLNK(mode):
        asset_type = "symlink"
    elif stat.S_ISDIR(mode):
        asset_type = "directory"
    elif stat.S_ISREG(mode):
        asset_type = "file"
    else:
        asset_type = "other"
    return {
        "relative_path": _relative_path(project_root, path),
        "asset_type": asset_type,
        "size": int(path_stat.st_size),
        "mtime_utc": _utc_timestamp(path_stat.st_mtime),
    }


def _is_within(project_root: Path, target: Path) -> bool:
    try:
        target.relative_to(project_root)
        return True
    except ValueError:
        return False


def _record_symlink(state: CollectorState, path: Path, path_stat: os.stat_result) -> None:
    record = _file_record(state.project_root, path, path_stat)
    try:
        target = path.resolve(strict=False)
        record["symlink_status"] = "not_followed" if _is_within(state.project_root, target) else "outside_allowlist"
    except OSError as error:
        record["symlink_status"] = "unresolved"
        state.add_error("inventory", record["relative_path"], "unknown", error)
    if record["symlink_status"] == "outside_allowlist":
        state.add_error(
            "inventory",
            record["relative_path"],
            "outside_allowlist",
            "Symlink target is outside the approved project root.",
        )
    state.inventory.append(record)


def _walk_approved_tree(state: CollectorState, directory: Path) -> None:
    try:
        children = sorted(directory.iterdir(), key=lambda value: value.name)
    except FileNotFoundError:
        return
    except PermissionError as error:
        state.add_error("inventory", _relative_path(state.project_root, directory), "permission_denied", error)
        return
    except OSError as error:
        state.add_error("inventory", _relative_path(state.project_root, directory), "unknown", error)
        return

    for child in children:
        child_stat = _safe_lstat(child)
        if child_stat is None:
            state.add_error("inventory", _relative_path(state.project_root, child), "missing_during_collection", "Path disappeared during collection.")
            continue
        if stat.S_ISLNK(child_stat.st_mode):
            _record_symlink(state, child, child_stat)
            continue
        record = _file_record(state.project_root, child, child_stat)
        state.inventory.append(record)
        if stat.S_ISDIR(child_stat.st_mode):
            _walk_approved_tree(state, child)
        elif stat.S_ISREG(child_stat.st_mode):
            state.normal_files[record["relative_path"]] = child


def _parse_registry_statuses(content: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in content.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 4:
            continue
        candidate = cells[1].strip("`")
        if candidate.startswith("data/tasks/"):
            statuses[candidate.rstrip("/")] = cells[2] or "unknown"
    return statuses


def _read_utf8_content(
    state: CollectorState,
    path: Path,
    evidence_kind: str,
    priority: int,
    required: bool = False,
) -> dict[str, Any] | None:
    path_stat = _safe_lstat(path)
    relative = _relative_path(state.project_root, path)
    if path_stat is None:
        state.add_error("read", relative, "missing_during_collection", "Path disappeared during collection.")
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        return None
    size = int(path_stat.st_size)
    record: dict[str, Any] = {
        "relative_path": relative,
        "size": size,
        "mtime_utc": _utc_timestamp(path_stat.st_mtime),
        "evidence_kind": evidence_kind,
        "priority": priority,
    }
    if size > state.embed_file_max_bytes:
        state.add_error("content", relative, "content_omitted_due_to_size", "File exceeds the per-file content threshold.")
        return record
    if state.embedded_bytes + size > state.embed_total_soft_max_bytes:
        state.add_error("content", relative, "content_omitted_due_to_total_budget", "The total embedded-content soft limit was reached.")
        return record
    try:
        raw = path.read_bytes()
    except PermissionError as error:
        state.add_error("read", relative, "permission_denied", error)
        return record
    except OSError as error:
        state.add_error("read", relative, "unknown", error)
        return record
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        state.add_error("decode", relative, "decode_error", error)
        return record

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            content: Any = json.loads(text)
        except json.JSONDecodeError as error:
            state.add_error("json_parse", relative, "malformed_json", error)
            return record
        content_format = "json"
    elif suffix == ".csv":
        try:
            list(csv.reader(io.StringIO(text), strict=True))
        except csv.Error as error:
            state.add_error("csv_parse", relative, "malformed_csv", error)
            return record
        content = text
        content_format = "csv_text"
    else:
        content = text
        content_format = "text"

    record.update(
        {
            "content_format": content_format,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "content": content,
        }
    )
    state.embedded_bytes += len(raw)
    return record


def _npy_member_name(member_name: str) -> str:
    return PurePosixPath(member_name).stem


def _read_exact(stream: Any, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("Unexpected end of NPY header.")
    return value


def _parse_npy_header(stream: Any, max_header_bytes: int) -> dict[str, Any]:
    magic = _read_exact(stream, 6)
    if magic != b"\x93NUMPY":
        raise ValueError("Missing NPY magic bytes.")
    major, minor = _read_exact(stream, 2)
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", _read_exact(stream, 2))[0]
    elif (major, minor) in {(2, 0), (3, 0)}:
        header_length = struct.unpack("<I", _read_exact(stream, 4))[0]
    else:
        raise ValueError(f"Unsupported NPY version {major}.{minor}.")
    if header_length > max_header_bytes:
        raise OverflowError(f"NPY header length {header_length} exceeds the configured limit.")
    header = _read_exact(stream, header_length)
    try:
        header_value = ast.literal_eval(header.decode("latin1"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("NPY header is not a valid dictionary.") from error
    if not isinstance(header_value, dict):
        raise ValueError("NPY header is not a dictionary.")
    if not {"descr", "fortran_order", "shape"}.issubset(header_value):
        raise ValueError("NPY header lacks required fields.")
    dtype = np.dtype(header_value["descr"])
    shape = tuple(int(value) for value in header_value["shape"])
    if any(value < 0 for value in shape):
        raise ValueError("NPY header contains a negative shape dimension.")
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(header_value["fortran_order"]),
    }


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    return str(value)


def _parameter_identity(raw_npy: bytes, header: dict[str, Any]) -> dict[str, Any]:
    array = np.load(io.BytesIO(raw_npy), allow_pickle=False)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(array.shape[0]) if array.ndim else int(array.size),
        "content_sha256": hashlib.sha256(raw_npy).hexdigest(),
    }
    numeric = array.dtype.kind in {"b", "i", "u", "f"}
    is_q_only = numeric and (array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1))
    if is_q_only:
        flattened = array.reshape(-1)
        result.update(
            {
                "kind": "q_values",
                "min": _safe_json_value(np.min(flattened)) if flattened.size else None,
                "max": _safe_json_value(np.max(flattened)) if flattened.size else None,
                "values": _safe_json_value(flattened.tolist()),
            }
        )
        return result
    result["kind"] = "parameter_array"
    if numeric and array.ndim >= 1:
        rows = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1, 1)
        result["per_column_min"] = _safe_json_value(np.min(rows, axis=0).tolist()) if rows.size else []
        result["per_column_max"] = _safe_json_value(np.max(rows, axis=0).tolist()) if rows.size else []
    if array.ndim == 0:
        result["first_rows"] = [_safe_json_value(array.item())]
        result["last_rows"] = [_safe_json_value(array.item())]
    else:
        result["first_rows"] = _safe_json_value(array[: min(5, array.shape[0])].tolist())
        result["last_rows"] = _safe_json_value(array[max(0, array.shape[0] - 5) :].tolist())
    return result


def _inspect_npz(state: CollectorState, path: Path) -> dict[str, Any]:
    path_stat = _safe_lstat(path)
    relative = _relative_path(state.project_root, path)
    result: dict[str, Any] = {
        "path": relative,
        "exists": path_stat is not None,
        "size": int(path_stat.st_size) if path_stat is not None else None,
        "mtime_utc": _utc_timestamp(path_stat.st_mtime) if path_stat is not None else None,
        "archive_status": "missing" if path_stat is None else "unknown",
        "arrays": [],
    }
    if path_stat is None:
        return result
    if not stat.S_ISREG(path_stat.st_mode):
        state.add_error("npz_header", relative, "unknown", "Dataset path is not a regular file.")
        return result
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [member for member in archive.infolist() if member.filename.endswith(".npy")]
            for member in members:
                name = _npy_member_name(member.filename)
                array_record: dict[str, Any] = {
                    "name": name,
                    "compressed_bytes": int(member.compress_size),
                    "uncompressed_bytes": int(member.file_size),
                }
                try:
                    with archive.open(member, "r") as stream:
                        header = _parse_npy_header(stream, state.npy_header_max_bytes)
                    array_record.update(header)
                    if name in {"x_train", "x_val", "x_test"} and member.file_size <= state.parameter_array_max_bytes:
                        with archive.open(member, "r") as stream:
                            raw_npy = stream.read()
                        array_record["identity"] = _parameter_identity(raw_npy, header)
                    elif name in {"x_train", "x_val", "x_test"}:
                        array_record["identity"] = {
                            "kind": "omitted_due_to_size",
                            "max_bytes": state.parameter_array_max_bytes,
                        }
                except OverflowError as error:
                    state.add_error("npz_header", relative, "npy_header_too_large", f"{member.filename}: {error}")
                except (OSError, ValueError, zipfile.BadZipFile) as error:
                    state.add_error("npz_header", relative, "invalid_npy_header", f"{member.filename}: {error}")
                result["arrays"].append(array_record)
            result["archive_status"] = "ok"
    except PermissionError as error:
        result["archive_status"] = "permission_denied"
        state.add_error("npz_header", relative, "permission_denied", error)
    except zipfile.BadZipFile as error:
        result["archive_status"] = "corrupt_archive"
        state.add_error("npz_header", relative, "corrupt_archive", error)
    except OSError as error:
        result["archive_status"] = "unknown"
        state.add_error("npz_header", relative, "unknown", error)
    return result


def _failed_sample_summary(content: Any) -> tuple[int | None, dict[str, int]]:
    if not isinstance(content, list):
        return None, {}
    reasons: Counter[str] = Counter()
    for item in content:
        if not isinstance(item, dict):
            continue
        for key in ("reason", "error", "error_type", "error_message"):
            value = item.get(key)
            if isinstance(value, str) and value:
                reasons[value] += 1
                break
    return len(content), dict(sorted(reasons.items()))


def _explicit_meta_status(meta_content: Any) -> str | None:
    if not isinstance(meta_content, dict):
        return None
    for container in (meta_content, meta_content.get("metadata")):
        if not isinstance(container, dict):
            continue
        for key in ("known_status", "status", "dataset_status"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _collect_dataset(state: CollectorState, directory: Path) -> dict[str, Any]:
    directory_stat = _safe_lstat(directory)
    relative = _relative_path(state.project_root, directory)
    record: dict[str, Any] = {
        "relative_path": relative,
        "exists": directory_stat is not None,
        "directory": _file_record(state.project_root, directory, directory_stat) if directory_stat is not None else None,
        "known_status": state.registry_status_by_path.get(relative, "unknown"),
        "meta_json": None,
        "dataset_npz": None,
        "failed_samples": None,
    }
    meta_path = directory / "meta.json"
    if meta_path.is_file() and not meta_path.is_symlink():
        meta_record = _read_utf8_content(state, meta_path, "dataset_meta", priority=0, required=True)
        record["meta_json"] = meta_record
        if meta_record is not None and "content" in meta_record:
            meta_status = _explicit_meta_status(meta_record["content"])
            if meta_status is not None:
                record["known_status"] = meta_status
    dataset_path = directory / "dataset.npz"
    if dataset_path.exists() and not dataset_path.is_symlink():
        record["dataset_npz"] = _inspect_npz(state, dataset_path)
    failed_path = directory / "failed_samples.json"
    if failed_path.is_file() and not failed_path.is_symlink():
        failed_record = _read_utf8_content(state, failed_path, "failed_samples", priority=1)
        if failed_record is not None:
            if isinstance(failed_record.get("content"), list):
                count, histogram = _failed_sample_summary(failed_record["content"])
                failed_record["count"] = count
                failed_record["reason_histogram"] = histogram
            else:
                failed_record.setdefault("count", None)
                failed_record.setdefault("reason_histogram", {})
        record["failed_samples"] = failed_record
    return record


def _is_json_evidence_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in JSON_EVIDENCE_NAMES or any(token in lowered for token in ("diagnostic", "diagnostics", "comparison", "evaluation"))


def _is_csv_evidence_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in CSV_EVIDENCE_NAMES or any(token in lowered for token in ("summary", "comparison", "validation", "metrics"))


def _is_text_evidence(path: Path) -> bool:
    lowered = path.name.lower()
    return path.suffix.lower() in {".md", ".txt"} and any(token in lowered for token in ("report", "summary", "validation", "analysis"))


def _evidence_priority(path: Path) -> int:
    name = path.name.lower()
    if name in {"run_config.json", "summary.json", "train_summary.json", "train_history.json", "history.json"}:
        return 1
    if name in {"metrics.json", "test_hidden_only_metrics.json", "analysis_summary.json", "result.json"}:
        return 2
    return 3


def _run_candidate_for_file(outputs_root: Path, path: Path) -> Path | None:
    name = path.name.lower()
    if name not in RUN_MARKER_NAMES and path.parent.name not in RUN_SUBDIRECTORIES:
        return None
    if path.parent.name in RUN_SUBDIRECTORIES:
        candidate = path.parent.parent
    else:
        candidate = path.parent
    try:
        candidate.relative_to(outputs_root)
    except ValueError:
        return None
    return candidate


def _find_dataset_paths(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_find_dataset_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_dataset_paths(item))
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        marker = "data/tasks/"
        index = normalized.find(marker)
        if index >= 0:
            found.add(normalized[index:])
    return found


def _status_from_record_content(content: Any) -> str | None:
    if not isinstance(content, dict):
        return None
    for key in ("status", "run_status", "state"):
        value = content.get(key)
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"interrupted", "failed", "stopped"}:
                return "interrupted"
            if lowered in {"complete", "completed", "finished", "success"}:
                return "complete"
    return None


def _collect_runs_and_evidence(state: CollectorState, outputs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_files = [(relative, path) for relative, path in state.normal_files.items() if relative.startswith("outputs/")]
    candidates: dict[str, Path] = {}
    checkpoints: list[dict[str, Any]] = []
    evidence_candidates: list[tuple[int, Path, str]] = []
    for relative, path in output_files:
        candidate = _run_candidate_for_file(outputs_root, path)
        if candidate is not None:
            candidates[_relative_path(state.project_root, candidate)] = candidate
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix in {".pt", ".pth"}:
            role = "best" if "best" in name else "last" if "last" in name else "epoch_specific" if "epoch" in name else "unknown"
            associated = _relative_path(state.project_root, path.parent.parent) if path.parent.name == "checkpoints" else None
            path_stat = path.lstat()
            checkpoints.append(
                {
                    "relative_path": relative,
                    "associated_run": associated,
                    "role": role,
                    "size": int(path_stat.st_size),
                    "mtime_utc": _utc_timestamp(path_stat.st_mtime),
                }
            )
        if suffix == ".json" and _is_json_evidence_name(name):
            evidence_candidates.append((_evidence_priority(path), path, "json"))
        elif suffix == ".csv" and _is_csv_evidence_name(name):
            evidence_candidates.append((_evidence_priority(path), path, "csv"))
        elif _is_text_evidence(path):
            evidence_candidates.append((_evidence_priority(path), path, "text"))

    evidence: list[dict[str, Any]] = []
    content_by_path: dict[str, Any] = {}
    for priority, path, evidence_kind in sorted(evidence_candidates, key=lambda item: (item[0], _relative_path(state.project_root, item[1]))):
        record = _read_utf8_content(state, path, evidence_kind, priority)
        if record is not None:
            evidence.append(record)
            if "content" in record:
                content_by_path[record["relative_path"]] = record["content"]

    runs: list[dict[str, Any]] = []
    for relative, directory in sorted(candidates.items()):
        child_records = [item for item in evidence if item["relative_path"].startswith(f"{relative}/")]
        checkpoint_paths = [item["relative_path"] for item in checkpoints if item.get("associated_run") == relative]
        basis = [item["relative_path"].removeprefix(f"{relative}/") for item in child_records] + [path.removeprefix(f"{relative}/") for path in checkpoint_paths]
        explicit_status = next((status for item in child_records if (status := _status_from_record_content(item.get("content"))) is not None), None)
        names = {Path(item["relative_path"]).name.lower() for item in child_records}
        final_names = {"summary.json", "metrics.json", "test_hidden_only_metrics.json", "result.json", "common_test_results.json", "validation_summary.json"}
        if explicit_status is not None:
            run_status = explicit_status
        elif names.intersection(final_names):
            run_status = "complete"
        elif any(name in names for name in {"run_config.json", "train_history.json", "history.json", "train_summary.json"}) or checkpoint_paths:
            run_status = "partial"
        else:
            run_status = "unknown"
        datasets: set[str] = set()
        for item in child_records:
            datasets.update(_find_dataset_paths(item.get("content")))
        runs.append(
            {
                "relative_path": relative,
                "run_status": run_status,
                "status_basis": sorted(set(basis)),
                "associated_dataset_paths": sorted(datasets),
                "embedded_records": [item["relative_path"] for item in child_records if "content" in item],
                "checkpoint_references": checkpoint_paths,
            }
        )
    return runs, checkpoints, evidence


def _summarize_history(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, dict):
        return None
    numeric_series: dict[str, list[float]] = {}
    for key, value in content.items():
        if isinstance(value, list) and all(isinstance(item, (int, float)) and np.isfinite(item) for item in value):
            numeric_series[key] = [float(item) for item in value]
    if not numeric_series:
        return None
    summary: dict[str, Any] = {"epoch_count": max(len(values) for values in numeric_series.values())}
    validation_key = next((key for key in numeric_series if "val" in key.lower()), None)
    train_key = next((key for key in numeric_series if "train" in key.lower()), None)
    if validation_key is not None and numeric_series[validation_key]:
        values = numeric_series[validation_key]
        best_index = int(np.argmin(values))
        summary["best_epoch"] = best_index + 1
        summary["best_validation_value"] = values[best_index]
        summary["final_validation_value"] = values[-1]
        summary["last_10_records"] = values[-10:]
        if len(values) >= 2:
            summary["final_segment_delta"] = values[-1] - values[max(0, len(values) - 10)]
    if train_key is not None and numeric_series[train_key]:
        summary["final_train_value"] = numeric_series[train_key][-1]
    return summary


def _attach_history_summaries(evidence: list[dict[str, Any]]) -> None:
    for record in evidence:
        if Path(record["relative_path"]).name.lower() in {"train_history.json", "history.json"}:
            summary = _summarize_history(record.get("content"))
            if summary is not None:
                record["history_summary"] = summary



def _attach_inventory_statuses(state: CollectorState, datasets: list[dict[str, Any]]) -> None:
    """将明确的数据集状态回填到对应的清单资产，未知状态保持 unknown。"""
    statuses = {str(item["relative_path"]): str(item["known_status"]) for item in datasets}
    for item in state.inventory:
        relative = str(item["relative_path"])
        parts = PurePosixPath(relative).parts
        if len(parts) >= 3 and parts[:2] == ("data", "tasks"):
            dataset_path = PurePosixPath(*parts[:3]).as_posix()
            item["known_status"] = statuses.get(dataset_path, "unknown")
        else:
            item["known_status"] = "unknown"
def _repository_metadata(state: CollectorState) -> dict[str, Any]:
    commands = {
        "head_commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short"],
        "head_subject": ["git", "log", "-1", "--format=%s"],
        "describe": ["git", "describe", "--always", "--dirty"],
    }
    metadata: dict[str, Any] = {}
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                cwd=state.project_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                shell=False,
            )
        except OSError as error:
            state.add_error("git", ".", "git_metadata_error", error)
            metadata[key] = None
            continue
        if completed.returncode != 0:
            state.add_error("git", ".", "git_metadata_error", completed.stderr.strip() or f"Git command failed for {key}.")
            metadata[key] = None
        elif key == "status_short":
            metadata[key] = [line for line in completed.stdout.splitlines() if line]
        else:
            metadata[key] = completed.stdout.strip() or None
    return metadata


def _scan_root_record(state: CollectorState, relative_root: Path) -> dict[str, Any]:
    path = state.project_root / relative_root
    path_stat = _safe_lstat(path)
    record = {
        "relative_path": relative_root.as_posix(),
        "exists": path_stat is not None,
        "scan_status": "missing" if path_stat is None else "ok",
    }
    if path_stat is None:
        return record
    if stat.S_ISLNK(path_stat.st_mode):
        record["scan_status"] = "outside_allowlist"
        _record_symlink(state, path, path_stat)
        return record
    if not stat.S_ISDIR(path_stat.st_mode):
        record["scan_status"] = "error"
        state.add_error("inventory", relative_root.as_posix(), "unknown", "Approved scan root is not a directory.")
        return record
    state.inventory.append(_file_record(state.project_root, path, path_stat))
    _walk_approved_tree(state, path)
    return record


def collect_snapshot(
    project_root: str | Path,
    *,
    embed_file_max_bytes: int = DEFAULT_EMBED_FILE_MAX_BYTES,
    embed_total_soft_max_bytes: int = DEFAULT_EMBED_TOTAL_SOFT_MAX_BYTES,
    parameter_array_max_bytes: int = DEFAULT_PARAMETER_ARRAY_MAX_BYTES,
    npy_header_max_bytes: int = DEFAULT_NPY_HEADER_MAX_BYTES,
) -> dict[str, Any]:
    """Return a read-only Stage-1 snapshot for one allowlisted project root."""
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    for value, name in (
        (embed_file_max_bytes, "embed_file_max_bytes"),
        (embed_total_soft_max_bytes, "embed_total_soft_max_bytes"),
        (parameter_array_max_bytes, "parameter_array_max_bytes"),
        (npy_header_max_bytes, "npy_header_max_bytes"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    state = CollectorState(
        project_root=root,
        embed_file_max_bytes=embed_file_max_bytes,
        embed_total_soft_max_bytes=embed_total_soft_max_bytes,
        parameter_array_max_bytes=parameter_array_max_bytes,
        npy_header_max_bytes=npy_header_max_bytes,
    )
    scan_roots = [_scan_root_record(state, relative_root) for relative_root in APPROVED_ROOTS]

    registry: dict[str, Any] = {"relative_path": REGISTRY_NAME, "exists": False}
    registry_path = root / REGISTRY_NAME
    if registry_path.is_file() and not registry_path.is_symlink():
        registry_record = _read_utf8_content(state, registry_path, "server_registry", priority=0, required=True)
        registry["exists"] = True
        registry["record"] = registry_record
        if registry_record is not None and isinstance(registry_record.get("content"), str):
            state.registry_status_by_path = _parse_registry_statuses(registry_record["content"])
    elif registry_path.exists():
        registry["exists"] = True
        state.add_error("registry", REGISTRY_NAME, "outside_allowlist", "Registry path is a symlink or not a regular file.")

    datasets: list[dict[str, Any]] = []
    tasks_root = root / "data" / "tasks"
    tasks_stat = _safe_lstat(tasks_root)
    if tasks_stat is not None and stat.S_ISDIR(tasks_stat.st_mode):
        try:
            for child in sorted(tasks_root.iterdir(), key=lambda value: value.name):
                child_stat = _safe_lstat(child)
                if child_stat is None:
                    state.add_error("dataset", _relative_path(root, child), "missing_during_collection", "Dataset directory disappeared during collection.")
                elif stat.S_ISLNK(child_stat.st_mode):
                    continue
                elif stat.S_ISDIR(child_stat.st_mode):
                    datasets.append(_collect_dataset(state, child))
        except (OSError, PermissionError) as error:
            state.add_error("dataset", "data/tasks", "permission_denied" if isinstance(error, PermissionError) else "unknown", error)

    outputs_root = root / "outputs"
    outputs_stat = _safe_lstat(outputs_root)
    if outputs_stat is not None and stat.S_ISDIR(outputs_stat.st_mode):
        runs, checkpoints, evidence_files = _collect_runs_and_evidence(state, outputs_root)
    else:
        runs, checkpoints, evidence_files = [], [], []
    _attach_history_summaries(evidence_files)
    _attach_inventory_statuses(state, datasets)

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "collection": {
            "mode": "stage1_research_snapshot",
            "created_utc": _utc_timestamp(datetime.now(tz=timezone.utc).timestamp()),
            "project_root_label": root.name,
            "scan_roots": [item.as_posix() for item in APPROVED_ROOTS],
            "singleton_files": [REGISTRY_NAME],
            "embed_file_max_bytes": embed_file_max_bytes,
            "embed_total_soft_max_bytes": embed_total_soft_max_bytes,
            "parameter_array_max_bytes": parameter_array_max_bytes,
            "npy_header_max_bytes": npy_header_max_bytes,
            "embedded_content_bytes": state.embedded_bytes,
        },
        "repository": _repository_metadata(state),
        "scan_roots": scan_roots,
        "registry": registry,
        "datasets": datasets,
        "runs": runs,
        "checkpoints": checkpoints,
        "evidence_files": evidence_files,
        "legacy_assets": state.inventory,
        "errors": state.errors,
    }
    return _safe_json_value(snapshot)


def build_parser() -> argparse.ArgumentParser:
    """构建仅输出 stdout 的命令行接口。"""
    parser = argparse.ArgumentParser(description="Collect a read-only Stage-1 server research snapshot to stdout.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--embed-file-max-bytes", type=int, default=DEFAULT_EMBED_FILE_MAX_BYTES)
    parser.add_argument("--embed-total-soft-max-bytes", type=int, default=DEFAULT_EMBED_TOTAL_SOFT_MAX_BYTES)
    parser.add_argument("--parameter-array-max-bytes", type=int, default=DEFAULT_PARAMETER_ARRAY_MAX_BYTES)
    parser.add_argument("--npy-header-max-bytes", type=int, default=DEFAULT_NPY_HEADER_MAX_BYTES)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """运行 collector，并只向 stdout 写入一个 JSON 文档。"""
    args = build_parser().parse_args(argv)
    snapshot = collect_snapshot(
        args.project_root,
        embed_file_max_bytes=args.embed_file_max_bytes,
        embed_total_soft_max_bytes=args.embed_total_soft_max_bytes,
        parameter_array_max_bytes=args.parameter_array_max_bytes,
        npy_header_max_bytes=args.npy_header_max_bytes,
    )
    json.dump(snapshot, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
