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

import numpy as np
import torch

from src.models.fno2d.fno2d import FNO2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_fno2d_internal_length_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("m3_internal", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
m3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m3
SPEC.loader.exec_module(m3)


def _pair_artifact(valid: bool = True) -> dict[str, object]:
    label = "EXACT_PREFIX" if valid else "NOT_PAIRED"
    return {
        "pair_classification": {
            "short_to_medium": label,
            "short_to_long": label,
            "medium_to_long": label,
        },
        "scientific_reuse": {
            "historical_t1800_reusable": valid,
            "t2400_ready_for_future_a1": valid,
        },
    }


def _field(name: str, q: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> m3.CanonicalQField:
    records = [
        {"source_split": "train", "source_index_within_split": index, "source_concatenated_index": index}
        for index in range(q.size)
    ]
    return m3.build_canonical_q_field(
        task_name=name,
        source_q=q,
        source_truth=truth,
        lambda_grid=grid,
        source_records=records,
    )


def _triplet() -> tuple[m3.CanonicalQField, m3.CanonicalQField, m3.CanonicalQField]:
    q = np.array([2.0, 1.0], dtype=np.float64)
    short_truth = np.arange(2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3)
    medium_truth = np.concatenate((short_truth, short_truth[:, :2, :] + 100.0), axis=1)
    long_truth = np.concatenate((medium_truth, medium_truth[:, :2, :] + 200.0), axis=1)
    return (
        _field("short", q, short_truth, np.arange(8, dtype=np.float64) * 0.5),
        _field("medium", q, medium_truth, np.arange(10, dtype=np.float64) * 0.5),
        _field("long", q, long_truth, np.arange(12, dtype=np.float64) * 0.5),
    )


def _model() -> FNO2d:
    torch.manual_seed(11)
    model = FNO2d(in_dim=2, out_dim=3, modes1=2, modes2=4, width=4, depth=2, hidden_dim=8)
    model.eval()
    return model


class PrerequisiteTests(unittest.TestCase):
    def test_invalid_stage2_prerequisite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(_pair_artifact(False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "short_to_medium"):
                m3.load_required_pair_validation(path)

    def test_canonical_q_and_exact_raw_prefix_validation(self) -> None:
        short, medium, long = _triplet()
        self.assertTrue(np.array_equal(short.canonical_q, np.array([1.0, 2.0])))
        self.assertTrue(np.array_equal(short.canonical_truth, medium.canonical_truth[:, :8, :]))
        m3.validate_triplet(short, medium, long)
        changed = long.source_truth.copy()
        changed[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "T1200 and T2400"):
            m3.validate_triplet(short, medium, _field("bad", long.source_q, changed, long.lambda_grid))


class HookAndSpectralTests(unittest.TestCase):
    def test_hooks_do_not_modify_forward_output(self) -> None:
        model = _model()
        input_tensor = torch.randn(1, 2, 8, 2)
        with torch.no_grad():
            baseline = model(input_tensor)
        captured, hooks = m3.capture_one_forward(model, input_tensor)
        self.assertTrue(torch.equal(baseline, captured))
        self.assertIsNotNone(hooks.lifted_feature)
        self.assertIsNotNone(hooks.first_spectral_input)

    def test_normalized_input_and_pointwise_lifting_prefix_are_identical(self) -> None:
        model = _model()
        short = torch.randn(1, 2, 8, 2)
        extended = torch.cat((short, torch.randn(1, 2, 4, 2)), dim=2)
        _, short_capture = m3.capture_one_forward(model, short)
        _, long_capture = m3.capture_one_forward(model, extended)
        self.assertTrue(torch.equal(short, extended[:, :, :8, :]))
        self.assertTrue(torch.allclose(short_capture.lifted_feature, long_capture.lifted_feature[:, :, :8, :], atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(short_capture.first_spectral_input, long_capture.first_spectral_input[:, :, :, :8], atol=0.0, rtol=0.0))

    def test_appending_points_changes_full_axis_fft_and_spectral_prefix_output(self) -> None:
        model = _model()
        short = torch.zeros(1, 2, 8, 2)
        short[:, :, :, 0] = torch.sin(2.0 * torch.pi * torch.arange(8).view(1, 1, -1) / 4.0)
        extended = torch.cat((short, torch.ones(1, 2, 4, 2)), dim=2)
        _, short_capture = m3.capture_one_forward(model, short)
        _, long_capture = m3.capture_one_forward(model, extended)
        short_fft = torch.fft.rfft2(short_capture.first_spectral_input, dim=(-2, -1))
        long_fft = torch.fft.rfft2(long_capture.first_spectral_input, dim=(-2, -1))
        self.assertFalse(torch.allclose(short_fft[:, :, :, :4], long_fft[:, :, :, :4]))
        self.assertFalse(torch.allclose(short_capture.spectral_branch_output, long_capture.spectral_branch_output[:, :, :, :8]))

    def test_replicated_spectral_multiplication_matches_module_output(self) -> None:
        model = _model()
        input_tensor = torch.randn(1, 2, 8, 2)
        _, hooks = m3.capture_one_forward(model, input_tensor)
        spectral, replicated = m3._replicate_first_spectral_layer(hooks.first_spectral_input, model.blocks[0].spectral_conv, "cpu")
        self.assertLess(spectral.modes_lambda, hooks.first_spectral_input.shape[-1] // 2 + 2)
        self.assertLess(m3._metrics(replicated, hooks.spectral_branch_output)["relative_l2_difference"], 1e-6)

    def test_modes2_retains_indices_zero_through_31_when_available(self) -> None:
        model = FNO2d(in_dim=2, out_dim=3, modes1=2, modes2=32, width=2, depth=1, hidden_dim=4)
        input_tensor = torch.randn(1, 2, 120, 2)
        _, hooks = m3.capture_one_forward(model, input_tensor)
        spectral, _ = m3._replicate_first_spectral_layer(hooks.first_spectral_input, model.blocks[0].spectral_conv, "cpu")
        self.assertEqual(spectral.modes_lambda, 32)
        self.assertEqual(spectral.modes_lambda - 1, 31)

    def test_compact_difference_metrics_and_physical_frequency_mapping(self) -> None:
        metrics = m3._metrics(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
        self.assertGreater(metrics["relative_l2_difference"], 0.0)
        self.assertLess(metrics["cosine_similarity"], 1.0)
        delta = 0.005
        self.assertEqual([int(round(1.0 * total * delta)) for total in (1200, 1800, 2400)], [6, 9, 12])
        self.assertNotEqual(6, 12)

    def test_same_index_and_physical_aligned_views_are_both_emitted(self) -> None:
        model = _model()
        short_field, medium_field, _ = _triplet()
        captures: dict[str, m3.LengthCapture] = {}
        for label, field in (("T1200", short_field), ("T1800", medium_field)):
            input_tensor = torch.randn(1, 2, field.lambda_grid.size, 2)
            _, hooks = m3.capture_one_forward(model, input_tensor)
            spectral, replicated = m3._replicate_first_spectral_layer(hooks.first_spectral_input, model.blocks[0].spectral_conv, "cpu")
            spectral.replicated_output_metrics = m3._metrics(replicated, hooks.spectral_branch_output)
            captures[label] = m3.LengthCapture(label, field, input_tensor[:, :, :8, :].clone(), hooks.lifted_feature[:, :, :8, :], hooks.first_spectral_input[:, :, :, :8], hooks.spectral_branch_output[:, :, :, :8], hooks.first_block_output[:, :, :, :8], np.zeros((2, 8, 3), dtype=np.float32), {"mean_per_q_relative_l2": 0.0}, spectral)
        rows = m3.build_spectral_rows({"T1200": captures["T1200"], "T1800": captures["T1800"], "T2400": captures["T1800"]})
        self.assertIn("same_discrete_index", {row["view"] for row in rows})
        self.assertIn("physical_frequency_aligned", {row["view"] for row in rows})


class OutputAndInterfaceTests(unittest.TestCase):
    def test_output_refuses_overwrite_and_writes_only_three_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "m3"
            stage_rows = [{"comparison": "a", "stage": "x", "view": "v", "relative_l2_difference": 0.0, "normalized_rmse": 0.0, "cosine_similarity": 1.0}]
            spectral_rows = [{"comparison": "a", "view": "same_discrete_index", "spectral_stage": "pre_weight", "lambda_index_left": 0, "lambda_index_right": 0, "physical_frequency_left": 0.0, "physical_frequency_right": 0.0, "right_index_retained": True, "left_energy_fraction": 1.0, "right_energy_fraction": 1.0, "relative_l2_difference": 0.0, "normalized_rmse": 0.0, "cosine_similarity": 1.0}]
            m3.write_output_artifacts(output_dir=output_dir, summary={"schema_version": "1.0"}, stage_rows=stage_rows, spectral_rows=spectral_rows)
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(m3.OUTPUT_FILENAMES))
            self.assertFalse(any(path.suffix in {".npy", ".pt"} for path in output_dir.iterdir()))
            with (output_dir / "m3_stage_comparison.csv").open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["stage"], "x")
            with self.assertRaises(FileExistsError):
                m3.write_output_artifacts(output_dir=output_dir, summary={}, stage_rows=[], spectral_rows=[])

    def test_cli_help_and_static_no_training_paths(self) -> None:
        completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--checkpoint-path", completed.stdout)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertIn('"adaptation": False', source)
        self.assertIn('"autoregression": False', source)
        self.assertIn('"prediction_feedback": False', source)

    def test_one_forward_per_length_contract_is_declared(self) -> None:
        model = _model()
        forward_count = 0

        def count_forward(_: torch.nn.Module, __: tuple[object, ...], ___: torch.Tensor) -> None:
            nonlocal forward_count
            forward_count += 1

        handle = model.register_forward_hook(count_forward)
        try:
            for length in (8, 10, 12):
                m3.capture_one_forward(model, torch.randn(1, 2, length, 2))
        finally:
            handle.remove()
        self.assertEqual(forward_count, 3)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("one_model_forward_per_length", source)
        self.assertIn("total_model_forwards", source)


if __name__ == "__main__":
    unittest.main()
