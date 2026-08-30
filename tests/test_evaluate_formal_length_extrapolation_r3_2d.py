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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_formal_length_extrapolation_r3_2d.py"
SPEC = importlib.util.spec_from_file_location("formal_r3_length_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
r3eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r3eval
SPEC.loader.exec_module(r3eval)


def checkpoint(valid: bool = True) -> dict[str, object]:
    anchors = np.linspace(0.0, 31 / (2 * 0.5), 32).tolist()
    config: dict[str, object] = {
        "experiment_type": r3eval.EXPERIMENT_TYPE if valid else "r2_domain_conditioned_coordinate_training",
        "repair_class": r3eval.REPAIR_CLASS,
        "source_task": "source",
        "normalization": "standard",
        "target_transform": "raw",
        "model_config": {"model_type": "fno2d_physical_frequency", "in_dim": 3, "out_dim": 3, "modes1": 16, "modes2": 32, "width": 64, "depth": 4, "hidden_dim": 128, "activation": "gelu", "delta_lambda": 0.5, "anchor_frequencies": anchors},
        "coordinate_representation": ["Q", "s", "ell"], "input_channel_names": ["Q", "s", "ell"], "absolute_lambda_input": False,
        "L_ref": 1.0, "train_lengths": [2, 3], "validation_lengths": [2, 3],
        "input_normalization_policy": {"Q": "standard_full_source_train_field", "s": "identity_dimensionless", "ell": "identity_dimensionless_L_over_L_ref", "target": "standard_full_source_train_field", "fit_uses_validation_lengths": False, "fit_uses_formal_long_lengths": False},
        "output_normalization_policy": "standard_full_source_train_field", "spectral_parameterization": "physical_frequency_anchor_interpolation", "physical_frequency_formula": "k / (N * delta_lambda)", "num_lambda_frequency_anchors": 32, "anchor_frequency_values": anchors, "complex_interpolation": "cartesian_linear", "runtime_retained_mode_policy": "fixed_discrete_indices_k_0_to_modes2_minus_1", "physical_cutoff_repair": False, "physical_bandwidth_shrinkage_repaired": False, "global_fft_structure_unchanged": True, "hypernetwork": False, "dynamic_spectral_weights": False,
        "dataset_summary": {"input_channel_names": ["Q", "s", "ell"], "normalization_stats": {"method": "standard", "x_mean": [2.0, 0.0, 0.0], "x_std": [0.5, 1.0, 1.0], "y_mean": [0.0, 0.0, 0.0], "y_std": [1.0, 1.0, 1.0], "eps": 1e-8}, "target_transform_config": {"mode": "raw", "lambda_reference_index": 0}},
    }
    return {"config": config}


def field(name: str, q: np.ndarray, truth: np.ndarray, grid: np.ndarray):
    records = [{"source_split": "val", "source_index_within_split": index, "source_concatenated_index": index} for index in range(q.size)]
    return r3eval.formal.build_canonical_q_field(task_name=name, source_q=q, source_truth=truth, lambda_grid=grid, source_records=records)


def triplet():
    q = np.array([2.0, 1.0])
    short_truth = np.ones((2, 2, 3), dtype=np.float64)
    medium_truth = np.concatenate((short_truth, np.ones((2, 1, 3))), axis=1)
    long_truth = np.concatenate((medium_truth, np.ones((2, 1, 3))), axis=1)
    return field("short", q, short_truth, np.array([0.0, 0.5])), field("medium", q, medium_truth, np.array([0.0, 0.5, 1.0])), field("long", q, long_truth, np.array([0.0, 0.5, 1.0, 1.5]))


class R3EvaluatorTests(unittest.TestCase):
    def test_provenance_requires_r3_anchor_interpolation(self) -> None:
        provenance = r3eval.validate_r3_checkpoint_provenance(checkpoint())
        self.assertEqual(provenance.input_channel_order, ("Q", "s", "ell"))
        self.assertEqual(len(provenance.anchor_frequencies), 32)
        with self.assertRaisesRegex(ValueError, "not an R3"):
            r3eval.validate_r3_checkpoint_provenance(checkpoint(valid=False))
        bad = checkpoint()
        bad["config"]["physical_cutoff_repair"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "fixed discrete-index"):
            r3eval.validate_r3_checkpoint_provenance(bad)

    def test_runtime_frequency_mapping_and_anchor_support(self) -> None:
        provenance = r3eval.validate_r3_checkpoint_provenance(checkpoint())
        matched = [2 / (4 * 0.5), 3 / (6 * 0.5), 4 / (8 * 0.5)]
        self.assertTrue(np.allclose(matched, matched[0]))
        values = r3eval.validate_runtime_anchor_support(total_length=4, delta_lambda=0.5, provenance=provenance)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values.size, 3)
        with self.assertRaisesRegex(ValueError, "delta_lambda"):
            r3eval.validate_runtime_anchor_support(total_length=4, delta_lambda=0.25, provenance=provenance)

    def test_three_frozen_forwards_and_formal_regions(self) -> None:
        short, medium, long = triplet()
        provenance = r3eval.validate_r3_checkpoint_provenance(checkpoint())
        runtime = {"delta_lambda": 0.5, "retained_lambda_indices": [0], "retained_physical_frequencies": [0.0]}
        with patch.object(r3eval, "run_frozen_inference", side_effect=[(short.canonical_truth.astype(np.float32), runtime), (medium.canonical_truth.astype(np.float32), runtime), (long.canonical_truth.astype(np.float32), runtime)]) as mocked:
            results, per_q, windows = r3eval.evaluate_three_lengths(model=object(), checkpoint=checkpoint(), provenance=provenance, short=short, medium=medium, long=long, device="cpu", window_width=0.5)
        self.assertEqual(mocked.call_count, 3)
        self.assertIsNone(results["T2"]["extrapolation"])
        self.assertIsNotNone(results["T3"]["extrapolation"])
        self.assertEqual(len(per_q), 6)
        self.assertGreater(len(windows), 0)

    def test_output_writer_refuses_overwrite_and_writes_only_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "r3"
            r3eval.write_output_artifacts(output_dir=output, summary={"schema_version": "1.0"}, per_q_rows=[{"total_length": 2, "Q": 1.0, "prefix_mse": 0.0, "prefix_relative_l2": 0.0, "extrapolation_mse": None, "extrapolation_relative_l2": None, "full_mse": 0.0, "full_relative_l2": 0.0}], window_rows=[])
            self.assertEqual({item.name for item in output.iterdir()}, set(r3eval.OUTPUT_FILENAMES))
            with (output / "r3_per_q_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["extrapolation_mse"], "")
            self.assertFalse(any(item.suffix in {".npy", ".pt"} for item in output.iterdir()))
            with self.assertRaises(FileExistsError):
                r3eval.write_output_artifacts(output_dir=output, summary={}, per_q_rows=[], window_rows=[])

    def test_summary_and_cli_exclude_adaptation_paths(self) -> None:
        short, medium, long = triplet()
        provenance = r3eval.validate_r3_checkpoint_provenance(checkpoint())
        args = argparse.Namespace(training_task_name="source", model_name="r3", window_width=0.5)
        summary = r3eval.build_summary(args=args, checkpoint_path=PROJECT_ROOT / "checkpoint.pt", checkpoint=checkpoint(), provenance=provenance, pair_validation_path=PROJECT_ROOT / "pair.json", pair_validation={"pair_classification": {}, "scientific_reuse": {}}, short=short, medium=medium, long=long, results={})
        self.assertTrue(summary["frozen_inference"]["one_full_forward_pass_per_total_length"])
        self.assertFalse(summary["spectral_parameterization"]["physical_bandwidth_shrinkage_repaired"])
        completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)


if __name__ == "__main__":
    unittest.main()
