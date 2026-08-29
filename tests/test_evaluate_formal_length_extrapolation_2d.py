from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_formal_length_extrapolation_2d.py"
SPEC = importlib.util.spec_from_file_location("formal_length_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = formal
SPEC.loader.exec_module(formal)


def _pair_artifact(*, valid: bool = True) -> dict[str, object]:
    classification = "EXACT_PREFIX" if valid else "NOT_PAIRED"
    return {
        "pair_classification": {
            "short_to_medium": classification,
            "short_to_long": classification,
            "medium_to_long": classification,
        },
        "scientific_reuse": {
            "historical_t1800_reusable": valid,
            "t2400_ready_for_future_a1": valid,
        },
    }


def _field(
    *,
    task_name: str,
    q: np.ndarray,
    truth: np.ndarray,
    lambda_grid: np.ndarray,
) -> formal.CanonicalQField:
    records = [
        {
            "source_split": ("train", "val", "test")[index % 3],
            "source_index_within_split": index,
            "source_concatenated_index": index,
        }
        for index in range(q.size)
    ]
    return formal.build_canonical_q_field(
        task_name=task_name,
        source_q=q,
        source_truth=truth,
        lambda_grid=lambda_grid,
        source_records=records,
    )


def _triplet() -> tuple[formal.CanonicalQField, formal.CanonicalQField, formal.CanonicalQField]:
    q = np.array([2.0, 1.0], dtype=np.float64)
    short_truth = np.array(
        [
            [[20.0, 2.0, 3.0], [21.0, 2.0, 3.0]],
            [[10.0, 2.0, 3.0], [11.0, 2.0, 3.0]],
        ],
        dtype=np.float64,
    )
    medium_truth = np.concatenate((short_truth, short_truth[:, :1, :] + 5.0), axis=1)
    long_truth = np.concatenate((medium_truth, medium_truth[:, :1, :] + 9.0), axis=1)
    return (
        _field(task_name="short", q=q, truth=short_truth, lambda_grid=np.array([0.0, 0.5])),
        _field(task_name="medium", q=q, truth=medium_truth, lambda_grid=np.array([0.0, 0.5, 1.0])),
        _field(task_name="long", q=q, truth=long_truth, lambda_grid=np.array([0.0, 0.5, 1.0, 1.5])),
    )


class PrerequisiteAndCanonicalTests(unittest.TestCase):
    def test_pair_validation_rejects_missing_or_invalid_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(_pair_artifact(valid=False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "short_to_medium"):
                formal.load_required_pair_validation(path)
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required fields"):
                formal.load_required_pair_validation(path)

    def test_stable_canonical_q_applies_the_same_permutation_to_truth(self) -> None:
        field = _field(
            task_name="short",
            q=np.array([2.0, 1.0, 3.0]),
            truth=np.array(
                [
                    [[20.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0]],
                    [[30.0, 0.0, 0.0]],
                ],
                dtype=np.float64,
            ),
            lambda_grid=np.array([0.0]),
        )
        self.assertTrue(np.array_equal(field.canonical_q, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(field.canonical_truth[:, 0, 0], np.array([10.0, 20.0, 30.0])))
        self.assertTrue(
            np.array_equal(
                field.canonical_to_source_index[field.source_to_canonical_index],
                np.arange(3),
            )
        )
        self.assertEqual(field.source_records[0]["source_concatenated_index"], 1)

    def test_triplet_requires_exact_canonical_q_and_prefix_truth(self) -> None:
        short, medium, long = _triplet()
        formal.validate_triplet(short, medium, long)
        mismatched = _field(
            task_name="bad",
            q=np.array([2.0, 1.5]),
            truth=long.source_truth,
            lambda_grid=long.lambda_grid,
        )
        with self.assertRaisesRegex(ValueError, "Canonical Q arrays"):
            formal.validate_triplet(short, medium, mismatched)

    def test_raw_truth_remains_float64_while_model_input_is_float32(self) -> None:
        short, _, _ = _triplet()
        x_raw, y_raw = formal.build_model_input(short.canonical_q, short.lambda_grid, short.canonical_truth)
        self.assertEqual(short.canonical_truth.dtype, np.float64)
        self.assertEqual(x_raw.dtype, np.float32)
        self.assertEqual(y_raw.dtype, np.float32)


class MetricTests(unittest.TestCase):
    def test_global_and_mean_per_q_relative_l2_are_distinct_and_components_are_reported(self) -> None:
        truth = np.array(
            [
                [[1.0, 2.0, 3.0]],
                [[100.0, 200.0, 300.0]],
            ],
            dtype=np.float64,
        )
        prediction = truth.copy()
        prediction[0] *= 2.0
        metrics = formal.compute_region_metrics(prediction.astype(np.float32), truth, np.array([1.0, 2.0]))
        self.assertAlmostEqual(metrics["mean_per_q_relative_l2"], 0.5)
        self.assertLess(metrics["global_relative_l2"], 0.02)
        self.assertEqual(metrics["worst_q_index"], 0)
        self.assertEqual(metrics["worst_q_value"], 1.0)
        self.assertEqual(set(metrics["components"]), {"x", "y", "z"})
        self.assertGreater(metrics["components"]["x"]["mse"], 0.0)

    def test_per_q_rows_mark_t1200_extrapolation_as_not_applicable(self) -> None:
        prediction = np.ones((2, 2, 3), dtype=np.float32)
        truth = np.zeros((2, 2, 3), dtype=np.float64)
        rows = formal.compute_per_q_rows(
            total_length=2,
            prediction=prediction,
            truth=truth,
            q_values=np.array([1.0, 2.0]),
            training_length=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["extrapolation_mse"])
        self.assertIsNone(rows[0]["extrapolation_relative_l2"])

    def test_per_q_rows_slice_prefix_extrapolation_and_full_regions(self) -> None:
        truth = np.ones((1, 3, 3), dtype=np.float64)
        prediction = truth.copy()
        prediction[:, 2, :] += 2.0
        row = formal.compute_per_q_rows(
            total_length=3,
            prediction=prediction,
            truth=truth,
            q_values=np.array([1.0]),
            training_length=2,
        )[0]
        self.assertAlmostEqual(row["prefix_mse"], 0.0)
        self.assertAlmostEqual(row["extrapolation_mse"], 4.0)
        self.assertAlmostEqual(row["full_mse"], 4.0 / 3.0)


class WindowTests(unittest.TestCase):
    def test_half_width_windows_cover_every_extrapolation_point_without_overlap(self) -> None:
        lambda_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        windows = formal.build_extrapolation_windows(lambda_grid, training_length=2, window_width=0.5)
        selected = np.concatenate([indices for _, _, indices, _ in windows])
        self.assertTrue(np.array_equal(np.sort(selected), np.array([2, 3, 4])))
        self.assertEqual(len(np.unique(selected)), 3)
        self.assertTrue(windows[-1][3])
        self.assertEqual(windows[-1][1], 2.0)

    def test_window_metrics_do_not_exist_for_t1200_and_clip_final_window(self) -> None:
        truth = np.ones((1, 2, 3), dtype=np.float64)
        self.assertEqual(
            formal.compute_window_rows(
                total_length=2,
                prediction=truth,
                truth=truth,
                lambda_grid=np.array([0.0, 0.5]),
                training_length=2,
                window_width=0.5,
            ),
            [],
        )
        long_truth = np.ones((1, 4, 3), dtype=np.float64)
        rows = formal.compute_window_rows(
            total_length=4,
            prediction=long_truth,
            truth=long_truth,
            lambda_grid=np.array([0.0, 0.5, 1.0, 1.3]),
            training_length=2,
            window_width=0.5,
        )
        self.assertEqual(rows[-1]["lambda_end"], 1.3)
        self.assertTrue(rows[-1]["interval_right_closed"])


class OrchestrationAndOutputTests(unittest.TestCase):
    def test_evaluate_three_lengths_runs_one_mocked_forward_per_total_length(self) -> None:
        short, medium, long = _triplet()
        predictions = [
            short.canonical_truth.astype(np.float32),
            medium.canonical_truth.astype(np.float32) + 1.0,
            long.canonical_truth.astype(np.float32) + 2.0,
        ]
        with patch.object(formal, "run_frozen_inference", side_effect=predictions) as mocked:
            results, per_q_rows, window_rows = formal.evaluate_three_lengths(
                model=object(),
                checkpoint={},
                short=short,
                medium=medium,
                long=long,
                device="cpu",
                window_width=0.5,
            )
        self.assertEqual(mocked.call_count, 3)
        self.assertIsNone(results["T2"]["extrapolation"])
        self.assertIsNotNone(results["T3"]["extrapolation"])
        self.assertIsNotNone(results["T4"]["extrapolation"])
        self.assertEqual(len(per_q_rows), 6)
        self.assertGreater(len(window_rows), 0)

    def test_output_writer_refuses_overwrite_and_creates_only_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "formal_a1"
            formal.write_output_artifacts(
                output_dir=output_dir,
                summary={"schema_version": "1.0"},
                per_q_rows=[
                    {
                        "total_length": 2,
                        "Q": 1.0,
                        "prefix_mse": 0.0,
                        "prefix_relative_l2": 0.0,
                        "extrapolation_mse": None,
                        "extrapolation_relative_l2": None,
                        "full_mse": 0.0,
                        "full_relative_l2": 0.0,
                    }
                ],
                window_rows=[],
            )
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(formal.OUTPUT_FILENAMES))
            self.assertFalse(any(path.suffix == ".npy" for path in output_dir.iterdir()))
            with (output_dir / "per_q_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["extrapolation_mse"], "")
            with self.assertRaises(FileExistsError):
                formal.write_output_artifacts(
                    output_dir=output_dir,
                    summary={},
                    per_q_rows=[],
                    window_rows=[],
                )

    def test_summary_serializes_required_provenance(self) -> None:
        short, medium, long = _triplet()
        args = argparse.Namespace(
            training_task_name="training",
            model_name="model",
            window_width=0.5,
            device="cpu",
        )
        checkpoint = {
            "config": {
                "model_config": {"model_type": "fno2d"},
                "normalization": "none",
                "target_transform": "raw",
                "dataset_summary": {
                    "normalization_stats": {
                        "method": "none",
                        "x_mean": [0.0, 0.0],
                        "x_std": [1.0, 1.0],
                        "y_mean": [0.0, 0.0, 0.0],
                        "y_std": [1.0, 1.0, 1.0],
                    },
                    "target_transform_config": {"mode": "raw", "lambda_reference_index": 0},
                },
            },
        }
        summary = formal.build_summary(
            args=args,
            checkpoint_path=PROJECT_ROOT / "checkpoint.pt",
            checkpoint=checkpoint,
            pair_validation_path=PROJECT_ROOT / "pair.json",
            pair_validation=_pair_artifact(),
            short=short,
            medium=medium,
            long=long,
            results={},
        )
        self.assertTrue(summary["ordering"]["canonical_q_exact_match_across_all_lengths"])
        self.assertEqual(summary["frozen_inference"]["total_forward_passes"], 3)
        self.assertEqual(summary["metric_aggregation"]["truth_source"], "raw_dataset_truth_float64")
        json.dumps(formal._as_json_value(summary))

    def test_cli_help_and_static_no_training_paths(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--window-width", completed.stdout)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("scheduler.step", source)


if __name__ == "__main__":
    unittest.main()
