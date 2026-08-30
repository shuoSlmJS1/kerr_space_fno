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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_formal_length_extrapolation_r2_2d.py"
SPEC = importlib.util.spec_from_file_location("formal_r2_length_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
r2eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r2eval
SPEC.loader.exec_module(r2eval)


def pair_artifact(*, valid: bool = True) -> dict[str, object]:
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


def r2_checkpoint(*, valid: bool = True) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_type": r2eval.EXPERIMENT_TYPE if valid else "r1_variable_length_training",
        "repair_class": r2eval.REPAIR_CLASS,
        "source_task": "training",
        "normalization": "standard",
        "target_transform": "raw",
        "model_config": {
            "model_type": "fno2d",
            "in_dim": 3,
            "out_dim": 3,
            "modes1": 16,
            "modes2": 32,
            "width": 64,
            "depth": 4,
        },
        "coordinate_representation": ["Q", "s", "ell"],
        "input_channel_names": ["Q", "s", "ell"],
        "absolute_lambda_input": False,
        "relative_coordinate_definition": r2eval.RELATIVE_COORDINATE_DEFINITION,
        "domain_length_definition": r2eval.DOMAIN_LENGTH_DEFINITION,
        "ell_definition": r2eval.ELL_DEFINITION,
        "L_ref": 1.0,
        "train_lengths": [2, 3],
        "validation_lengths": [2, 3],
        "spectral_parameterization_unchanged": True,
        "discrete_mode_index_weights": True,
        "physical_frequency_conditioning": False,
        "frequency_interpolation": False,
        "dynamic_spectral_weights": False,
        "input_normalization_policy": {
            "Q": "standard_full_source_train_field",
            "s": "identity_dimensionless",
            "ell": "identity_dimensionless_L_over_L_ref",
            "target": "standard_full_source_train_field",
            "fit_uses_validation_lengths": False,
            "fit_uses_formal_long_lengths": False,
        },
        "output_normalization_policy": "standard_full_source_train_field",
        "dataset_summary": {
            "input_channel_names": ["Q", "s", "ell"],
            "normalization_stats": {
                "method": "standard",
                "x_mean": [2.0, 0.0, 0.0],
                "x_std": [0.5, 1.0, 1.0],
                "y_mean": [0.0, 0.0, 0.0],
                "y_std": [1.0, 1.0, 1.0],
                "eps": 1e-8,
            },
            "target_transform_config": {"mode": "raw", "lambda_reference_index": 0},
        },
    }
    return {"config": config}


def field(*, task_name: str, q: np.ndarray, truth: np.ndarray, grid: np.ndarray):
    records = [
        {
            "source_split": ("train", "val", "test")[index % 3],
            "source_index_within_split": index,
            "source_concatenated_index": index,
        }
        for index in range(q.size)
    ]
    return r2eval.formal.build_canonical_q_field(
        task_name=task_name,
        source_q=q,
        source_truth=truth,
        lambda_grid=grid,
        source_records=records,
    )


def triplet():
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
        field(task_name="short", q=q, truth=short_truth, grid=np.array([0.0, 0.5])),
        field(task_name="medium", q=q, truth=medium_truth, grid=np.array([0.0, 0.5, 1.0])),
        field(task_name="long", q=q, truth=long_truth, grid=np.array([0.0, 0.5, 1.0, 1.5])),
    )


class ProvenanceAndCoordinateTests(unittest.TestCase):
    def test_stage2_invalid_prerequisite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(pair_artifact(valid=False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "short_to_medium"):
                r2eval.formal.load_required_pair_validation(path)

    def test_non_r2_or_missing_coordinate_provenance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an R2"):
            r2eval.validate_r2_checkpoint_provenance(r2_checkpoint(valid=False))
        checkpoint = r2_checkpoint()
        del checkpoint["config"]["coordinate_representation"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "coordinate_representation"):
            r2eval.validate_r2_checkpoint_provenance(checkpoint)

    def test_in_dim_three_and_channel_order_are_required(self) -> None:
        checkpoint = r2_checkpoint()
        checkpoint["config"]["model_config"]["in_dim"] = 2  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "in_dim=3"):
            r2eval.validate_r2_checkpoint_provenance(checkpoint)
        checkpoint = r2_checkpoint()
        checkpoint["config"]["coordinate_representation"] = ["Q", "ell", "s"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "coordinate_representation"):
            r2eval.validate_r2_checkpoint_provenance(checkpoint)

    def test_spectral_parameterization_is_required_to_remain_discrete_index(self) -> None:
        checkpoint = r2_checkpoint()
        checkpoint["config"]["physical_frequency_conditioning"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "spectral-parameterization"):
            r2eval.validate_r2_checkpoint_provenance(checkpoint)

    def test_canonical_q_and_exact_truth_prefix_are_preserved(self) -> None:
        short, medium, long = triplet()
        r2eval.formal.validate_triplet(short, medium, long)
        self.assertTrue(np.array_equal(short.canonical_q, np.array([1.0, 2.0])))
        self.assertTrue(np.array_equal(short.canonical_truth, medium.canonical_truth[:, :2, :]))
        self.assertTrue(np.array_equal(medium.canonical_truth, long.canonical_truth[:, :3, :]))

    def test_uniform_grid_and_runtime_coordinates_follow_checkpoint_definitions(self) -> None:
        provenance = r2eval.validate_r2_checkpoint_provenance(r2_checkpoint())
        raw_truth = np.zeros((2, 3, 3), dtype=np.float64)
        x, y, runtime = r2eval.build_r2_model_input(
            q_values=np.array([1.0, 2.0]),
            lambda_grid=np.array([0.0, 0.5, 1.0]),
            raw_truth=raw_truth,
            provenance=provenance,
        )
        self.assertEqual(x.shape, (1, 2, 3, 3))
        self.assertEqual(y.shape, (1, 2, 3, 3))
        self.assertAlmostEqual(runtime.delta_lambda, 0.5)
        self.assertAlmostEqual(runtime.domain_length, 1.5)
        self.assertAlmostEqual(runtime.ell, 1.5)
        self.assertAlmostEqual(float(x[0, 0, -1, 1]), 1.0 / 1.5)
        self.assertTrue(np.allclose(x[..., 2], 1.5))

    def test_nonuniform_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniformly"):
            r2eval.derive_uniform_delta_lambda(np.array([0.0, 0.5, 1.1]))

    def test_no_absolute_lambda_channel_and_q_is_restored_before_normalization(self) -> None:
        provenance = r2eval.validate_r2_checkpoint_provenance(r2_checkpoint())
        x, _y, _runtime = r2eval.build_r2_model_input(
            q_values=np.array([1.0]),
            lambda_grid=np.array([0.0, 0.25, 0.5]),
            raw_truth=np.zeros((1, 3, 3), dtype=np.float64),
            provenance=provenance,
        )
        self.assertEqual(tuple(x[0, 0, :, 0]), (1.0, 1.0, 1.0))
        self.assertTrue(np.allclose(x[0, 0, :, 1], [0.0, 1.0 / 3.0, 2.0 / 3.0]))
        self.assertNotEqual(float(x[0, 0, -1, 1]), 0.5)
        self.assertTrue(np.allclose(x[..., 2], 0.75))

    def test_checkpoint_normalization_requires_q_standard_and_s_ell_identity(self) -> None:
        provenance = r2eval.validate_r2_checkpoint_provenance(r2_checkpoint())
        stats = r2eval.load_normalization_stats_from_checkpoint(r2_checkpoint())
        r2eval.validate_r2_normalization_stats(stats, provenance)
        bad_stats = r2eval.FieldNormalizationStats(
            method="standard",
            x_mean=[2.0, 0.1, 0.0],
            x_std=[0.5, 1.0, 1.0],
            y_mean=[0.0, 0.0, 0.0],
            y_std=[1.0, 1.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "identity zero"):
            r2eval.validate_r2_normalization_stats(bad_stats, provenance)


class MetricAndFrozenProtocolTests(unittest.TestCase):
    def test_one_frozen_forward_per_length_and_regions_are_sliced_correctly(self) -> None:
        short, medium, long = triplet()
        provenance = r2eval.validate_r2_checkpoint_provenance(r2_checkpoint())
        predictions = [
            short.canonical_truth.astype(np.float32),
            medium.canonical_truth.astype(np.float32) + 1.0,
            long.canonical_truth.astype(np.float32) + 2.0,
        ]
        coordinates = [
            r2eval.R2RuntimeCoordinates(2, 0.5, 1.0, 1.0, 1.0, 0.0, 0.5),
            r2eval.R2RuntimeCoordinates(3, 0.5, 1.5, 1.0, 1.5, 0.0, 2.0 / 3.0),
            r2eval.R2RuntimeCoordinates(4, 0.5, 2.0, 1.0, 2.0, 0.0, 0.75),
        ]
        with patch.object(r2eval, "run_frozen_inference", side_effect=list(zip(predictions, coordinates))) as mocked:
            results, per_q_rows, window_rows = r2eval.evaluate_three_lengths(
                model=object(),
                checkpoint=r2_checkpoint(),
                provenance=provenance,
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
        self.assertEqual(results["T3"]["r2_coordinates"]["input_channel_order"], ["Q", "s", "ell"])

    def test_formal_metrics_and_per_q_rows_reuse_a1_semantics(self) -> None:
        truth = np.array([[[1.0, 2.0, 3.0]], [[100.0, 200.0, 300.0]]], dtype=np.float64)
        prediction = truth.copy()
        prediction[0] *= 2.0
        metrics = r2eval.formal.compute_region_metrics(prediction, truth, np.array([1.0, 2.0]))
        self.assertAlmostEqual(metrics["mean_per_q_relative_l2"], 0.5)
        self.assertLess(metrics["global_relative_l2"], 0.02)
        rows = r2eval.formal.compute_per_q_rows(
            total_length=1,
            prediction=prediction,
            truth=truth,
            q_values=np.array([1.0, 2.0]),
            training_length=1,
        )
        self.assertIsNone(rows[0]["extrapolation_mse"])

    def test_windows_remain_on_physical_lambda_axis(self) -> None:
        truth = np.ones((1, 4, 3), dtype=np.float64)
        rows = r2eval.formal.compute_window_rows(
            total_length=4,
            prediction=truth,
            truth=truth,
            lambda_grid=np.array([0.0, 0.5, 1.0, 1.3]),
            training_length=2,
            window_width=0.5,
        )
        self.assertEqual(rows[-1]["lambda_end"], 1.3)
        self.assertTrue(rows[-1]["interval_right_closed"])


class OutputAndSummaryTests(unittest.TestCase):
    def test_output_writer_refuses_overwrite_and_creates_exactly_three_compact_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "r2_a1"
            r2eval.write_output_artifacts(
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
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(r2eval.OUTPUT_FILENAMES))
            self.assertFalse(any(path.suffix in {".npy", ".pt"} for path in output_dir.iterdir()))
            with (output_dir / "r2_per_q_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["extrapolation_mse"], "")
            with self.assertRaises(FileExistsError):
                r2eval.write_output_artifacts(
                    output_dir=output_dir,
                    summary={},
                    per_q_rows=[],
                    window_rows=[],
                )

    def test_summary_records_r2_coordinate_and_frozen_provenance(self) -> None:
        short, medium, long = triplet()
        args = argparse.Namespace(
            training_task_name="training",
            model_name="r2_model",
            window_width=0.5,
            device="cpu",
        )
        checkpoint = r2_checkpoint()
        provenance = r2eval.validate_r2_checkpoint_provenance(checkpoint)
        summary = r2eval.build_summary(
            args=args,
            checkpoint_path=PROJECT_ROOT / "checkpoint.pt",
            checkpoint=checkpoint,
            provenance=provenance,
            pair_validation_path=PROJECT_ROOT / "pair.json",
            pair_validation=pair_artifact(),
            short=short,
            medium=medium,
            long=long,
            results={},
        )
        self.assertEqual(summary["r2_coordinate_representation"]["channel_order"], ["Q", "s", "ell"])
        self.assertEqual(summary["frozen_inference"]["total_forward_passes"], 3)
        self.assertFalse(summary["spectral_parameterization"]["physical_frequency_aware_weights"])
        self.assertEqual(summary["metric_aggregation"]["truth_source"], "raw_dataset_truth_float64")
        json.dumps(r2eval.formal._as_json_value(summary))

    def test_cli_requires_explicit_checkpoint_and_has_no_training_paths(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--checkpoint-path", completed.stdout)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("scheduler.step", source)


if __name__ == "__main__":
    unittest.main()
