from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = PROJECT_ROOT / "scripts" / "collect_server_research_snapshot.py"
SPEC = importlib.util.spec_from_file_location("collect_server_research_snapshot", COLLECTOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load collector module.")
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


class TestCollectServerResearchSnapshot(unittest.TestCase):
    """Test the collector only against synthetic temporary project trees."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "project"
        (self.root / "data" / "tasks").mkdir(parents=True)
        (self.root / "outputs").mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def collect(self, **kwargs: object) -> dict[str, object]:
        return collector.collect_snapshot(self.root, **kwargs)

    def make_dataset(self, name: str = "task", include_large_y: bool = False) -> Path:
        task = self.root / "data" / "tasks" / name
        task.mkdir()
        (task / "meta.json").write_text(json.dumps({"task_name": name}), encoding="utf-8", newline="\n")
        payload: dict[str, np.ndarray] = {
            "x_train": np.array([[1.6], [1.8]], dtype=np.float64),
            "x_val": np.array([[2.0]], dtype=np.float64),
            "x_test": np.array([[2.2], [2.4]], dtype=np.float64),
            "lambda_grid": np.linspace(0.0, 1.0, 5),
            "y_train": np.zeros((2, 5, 3), dtype=np.float32),
            "y_val": np.zeros((1, 5, 3), dtype=np.float32),
            "y_test": np.zeros((2, 5, 3), dtype=np.float32),
        }
        if include_large_y:
            payload["y_train"] = np.zeros((8, 2048, 3), dtype=np.float32)
        np.savez_compressed(task / "dataset.npz", **payload)
        return task

    @staticmethod
    def find_dataset(snapshot: dict[str, object], suffix: str) -> dict[str, object]:
        return next(item for item in snapshot["datasets"] if item["relative_path"].endswith(suffix))

    @staticmethod
    def find_error(snapshot: dict[str, object], kind: str) -> dict[str, object]:
        return next(item for item in snapshot["errors"] if item["error_kind"] == kind)

    def test_approved_roots_only_and_unrelated_root_is_not_scanned(self) -> None:
        (self.root / "unrelated").mkdir()
        (self.root / "unrelated" / "secret.json").write_text('{"secret": true}', encoding="utf-8")
        self.make_dataset()
        snapshot = self.collect()
        serialized = json.dumps(snapshot)
        self.assertNotIn("unrelated/secret.json", serialized)
        self.assertEqual([item["relative_path"] for item in snapshot["scan_roots"]], ["data/tasks", "outputs"])

    def test_registry_and_meta_are_embedded(self) -> None:
        registry = "| `data/tasks/task` | CURRENT / VALIDATED |\n"
        (self.root / "SERVER_DATA_EXPERIMENT_REGISTRY.md").write_text(registry, encoding="utf-8", newline="\n")
        self.make_dataset()
        snapshot = self.collect()
        self.assertEqual(snapshot["registry"]["record"]["content"], registry)
        dataset = self.find_dataset(snapshot, "/task")
        self.assertEqual(dataset["known_status"], "CURRENT / VALIDATED")
        self.assertEqual(dataset["meta_json"]["content"]["task_name"], "task")

    def test_invalid_registry_is_recorded_without_aborting(self) -> None:
        (self.root / "SERVER_DATA_EXPERIMENT_REGISTRY.md").write_bytes(b"\xff\xfe")
        self.make_dataset()
        snapshot = self.collect()
        self.find_error(snapshot, "decode_error")
        self.assertEqual(len(snapshot["datasets"]), 1)

    def test_dataset_npz_metadata_and_small_q_identity(self) -> None:
        self.make_dataset()
        snapshot = self.collect()
        dataset = self.find_dataset(snapshot, "/task")
        arrays = {item["name"]: item for item in dataset["dataset_npz"]["arrays"]}
        self.assertEqual(arrays["y_train"]["shape"], [2, 5, 3])
        self.assertEqual(arrays["y_train"]["dtype"], "float32")
        self.assertEqual(arrays["x_test"]["identity"]["kind"], "q_values")
        self.assertEqual(arrays["x_test"]["identity"]["values"], [2.2, 2.4])

    def test_trajectory_member_is_not_loaded_with_numpy_load(self) -> None:
        self.make_dataset(include_large_y=True)
        with mock.patch.object(collector.np, "load", side_effect=AssertionError("full array loading is forbidden")):
            snapshot = self.collect(parameter_array_max_bytes=1)
        dataset = self.find_dataset(snapshot, "/task")
        arrays = {item["name"]: item for item in dataset["dataset_npz"]["arrays"]}
        self.assertEqual(arrays["y_train"]["shape"], [8, 2048, 3])
        self.assertNotIn("identity", arrays["y_train"])

    def test_parameter_identity_threshold_is_respected(self) -> None:
        self.make_dataset()
        snapshot = self.collect(parameter_array_max_bytes=1)
        dataset = self.find_dataset(snapshot, "/task")
        arrays = {item["name"]: item for item in dataset["dataset_npz"]["arrays"]}
        self.assertEqual(arrays["x_test"]["identity"]["kind"], "omitted_due_to_size")

    def test_failed_samples_summary(self) -> None:
        task = self.make_dataset()
        failures = [{"error_type": "ValueError"}, {"reason": "turning_point"}, {"reason": "turning_point"}]
        (task / "failed_samples.json").write_text(json.dumps(failures), encoding="utf-8")
        snapshot = self.collect()
        failed = self.find_dataset(snapshot, "/task")["failed_samples"]
        self.assertEqual(failed["count"], 3)
        self.assertEqual(failed["reason_histogram"], {"ValueError": 1, "turning_point": 2})

    def test_malformed_zip_and_npy_headers_are_safe(self) -> None:
        corrupt = self.root / "data" / "tasks" / "corrupt"
        corrupt.mkdir()
        (corrupt / "dataset.npz").write_bytes(b"not a zip")
        malformed = self.root / "data" / "tasks" / "malformed"
        malformed.mkdir()
        with zipfile.ZipFile(malformed / "dataset.npz", "w") as archive:
            archive.writestr("bad.npy", b"not-a-valid-npy")
        oversized = self.root / "data" / "tasks" / "oversized"
        oversized.mkdir()
        header_length = collector.DEFAULT_NPY_HEADER_MAX_BYTES + 1
        fake_header = b"\x93NUMPY" + bytes((2, 0)) + header_length.to_bytes(4, "little")
        with zipfile.ZipFile(oversized / "dataset.npz", "w") as archive:
            archive.writestr("large.npy", fake_header)
        snapshot = self.collect()
        self.find_error(snapshot, "corrupt_archive")
        self.find_error(snapshot, "invalid_npy_header")
        self.find_error(snapshot, "npy_header_too_large")

    def test_output_evidence_run_status_and_checkpoints(self) -> None:
        complete = self.root / "outputs" / "complete_run"
        (complete / "metrics").mkdir(parents=True)
        (complete / "checkpoints").mkdir()
        (complete / "run_config.json").write_text(json.dumps({"dataset_path": "data/tasks/task/dataset.npz"}), encoding="utf-8")
        (complete / "metrics" / "metrics.json").write_text(json.dumps({"relative_l2": 0.1}), encoding="utf-8")
        (complete / "checkpoints" / "best_model.pt").write_bytes(b"weights")
        (complete / "checkpoints" / "last_model.pt").write_bytes(b"weights")
        partial = self.root / "outputs" / "partial_run"
        partial.mkdir()
        (partial / "run_config.json").write_text("{}", encoding="utf-8")
        interrupted = self.root / "outputs" / "interrupted_run"
        interrupted.mkdir()
        (interrupted / "summary.json").write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
        snapshot = self.collect()
        runs = {item["relative_path"]: item for item in snapshot["runs"]}
        self.assertEqual(runs["outputs/complete_run"]["run_status"], "complete")
        self.assertEqual(runs["outputs/partial_run"]["run_status"], "partial")
        self.assertEqual(runs["outputs/interrupted_run"]["run_status"], "interrupted")
        roles = {item["relative_path"]: item["role"] for item in snapshot["checkpoints"]}
        self.assertEqual(roles["outputs/complete_run/checkpoints/best_model.pt"], "best")
        self.assertEqual(roles["outputs/complete_run/checkpoints/last_model.pt"], "last")
        embedded = {item["relative_path"] for item in snapshot["evidence_files"] if "content" in item}
        self.assertIn("outputs/complete_run/metrics/metrics.json", embedded)

    def test_unknown_run_status_remains_unknown(self) -> None:
        run = self.root / "outputs" / "unknown_run" / "metrics"
        run.mkdir(parents=True)
        (run / "unrecognized.json").write_text('{"note": "evidence only"}', encoding="utf-8")
        snapshot = self.collect()
        runs = {item["relative_path"]: item for item in snapshot["runs"]}
        self.assertEqual(runs["outputs/unknown_run"]["run_status"], "unknown")
    def test_history_summary_and_relevant_csv(self) -> None:
        run = self.root / "outputs" / "run"
        (run / "logs").mkdir(parents=True)
        (run / "analysis").mkdir()
        history = {"train_loss": [3.0, 2.0, 1.0], "val_loss": [4.0, 1.5, 2.0]}
        (run / "logs" / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
        (run / "analysis" / "per_q_metrics.csv").write_text("q,error\n1.6,0.1\n", encoding="utf-8")
        snapshot = self.collect()
        records = {item["relative_path"]: item for item in snapshot["evidence_files"]}
        history_record = records["outputs/run/logs/train_history.json"]
        self.assertEqual(history_record["history_summary"]["best_epoch"], 2)
        self.assertIn("outputs/run/analysis/per_q_metrics.csv", records)

    def test_unrelated_content_large_content_and_binary_are_not_embedded(self) -> None:
        run = self.root / "outputs" / "run"
        run.mkdir()
        (run / "random.json").write_text('{"unrelated": true}', encoding="utf-8")
        (run / "metrics.json").write_text("{" + '"x":"' + ("a" * 1024) + '"}', encoding="utf-8")
        (run / "predictions.npy").write_bytes(b"binary")
        (run / "figure.png").write_bytes(b"image")
        snapshot = self.collect(embed_file_max_bytes=128)
        embedded = {item["relative_path"] for item in snapshot["evidence_files"] if "content" in item}
        self.assertNotIn("outputs/run/random.json", embedded)
        self.assertNotIn("outputs/run/metrics.json", embedded)
        self.find_error(snapshot, "content_omitted_due_to_size")
        serialized = json.dumps(snapshot)
        self.assertNotIn("binary", serialized)

    def test_total_content_budget_continues_inventory(self) -> None:
        run = self.root / "outputs" / "run"
        run.mkdir()
        (run / "run_config.json").write_text('{"a": 1}', encoding="utf-8")
        (run / "metrics.json").write_text('{"b": 2}', encoding="utf-8")
        snapshot = self.collect(embed_total_soft_max_bytes=8)
        self.assertTrue(snapshot["legacy_assets"])
        self.find_error(snapshot, "content_omitted_due_to_total_budget")

    def test_symlinks_are_never_followed(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        (outside / "secret.json").write_text('{"secret": true}', encoding="utf-8")
        link = self.root / "outputs" / "escape"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable in this environment.")
        snapshot = self.collect()
        self.find_error(snapshot, "outside_allowlist")
        serialized = json.dumps(snapshot)
        self.assertNotIn("secret.json", serialized)
        self.assertNotIn(str(outside), serialized)

    def test_git_metadata_success_and_failure(self) -> None:
        subprocess.run(["git", "init"], cwd=self.root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=self.root, check=True)
        (self.root / "tracked.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "synthetic"], cwd=self.root, check=True, stdout=subprocess.DEVNULL)
        snapshot = self.collect()
        self.assertIsNotNone(snapshot["repository"]["head_commit"])
        with mock.patch.object(collector.subprocess, "run", side_effect=OSError("git unavailable")):
            failed = self.collect()
        self.find_error(failed, "git_metadata_error")

    def test_cli_stdout_valid_json_and_no_target_project_writes(self) -> None:
        self.make_dataset()
        before = {path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns for path in self.root.rglob("*")}
        command = [sys.executable, "-B", str(COLLECTOR_PATH), "--project-root", str(self.root)]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = json.loads(completed.stdout)
        after = {path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns for path in self.root.rglob("*")}
        self.assertEqual(output["schema_version"], "1.0")
        self.assertEqual(before, after)
        self.assertEqual(completed.stderr, "")

    def test_repeated_collection_does_not_modify_target_files(self) -> None:
        self.make_dataset()
        before = {path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns for path in self.root.rglob("*")}
        self.collect()
        self.collect()
        after = {path.relative_to(self.root).as_posix(): path.stat().st_mtime_ns for path in self.root.rglob("*")}
        self.assertEqual(before, after)
    def test_source_has_no_model_or_forbidden_git_operations(self) -> None:
        source = COLLECTOR_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("import torch", source)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("git pull", source)
        self.assertNotIn("git fetch", source)
        self.assertNotIn("git push", source)
        self.assertNotIn("git checkout", source)
        self.assertNotIn("git reset", source)
        self.assertNotIn("git clean", source)


if __name__ == "__main__":
    unittest.main()
