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
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_r3_validation_length_response_2d.py"
SPEC = importlib.util.spec_from_file_location("r3_validation_response", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
response = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = response
SPEC.loader.exec_module(response)


class FakeModel:
    """用于验证每长度一次冻结 forward 的无训练替身。"""

    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self


def make_checkpoint(*, source_task: str = "synthetic_t1200") -> dict[str, object]:
    """构造满足 canonical R3 provenance 的最小 checkpoint。"""
    delta_lambda = 0.01
    anchors = np.linspace(0.0, 31.0 / (600 * delta_lambda), 32, dtype=np.float64)
    config: dict[str, object] = {
        "experiment_type": response.r3formal.EXPERIMENT_TYPE,
        "repair_class": response.r3formal.REPAIR_CLASS,
        "source_task": source_task,
        "run_name": "synthetic_r3",
        "normalization": "standard",
        "target_transform": "raw",
        "model_config": {
            "model_type": "fno2d_physical_frequency",
            "in_dim": 3,
            "out_dim": 3,
            "modes1": 16,
            "modes2": 32,
            "width": 64,
            "depth": 4,
            "hidden_dim": 128,
            "activation": "gelu",
            "delta_lambda": delta_lambda,
            "anchor_frequencies": anchors.astype(np.float32).astype(np.float64).tolist(),
        },
        "coordinate_representation": ["Q", "s", "ell"],
        "input_channel_names": ["Q", "s", "ell"],
        "absolute_lambda_input": False,
        "L_ref": 12.0,
        "train_lengths": [600, 800, 1000, 1200],
        "validation_lengths": [700, 900, 1100, 1200],
        "checkpoint_selection": {
            "split": "validation_Q_only",
            "lengths": [700, 900, 1100, 1200],
        },
        "input_normalization_policy": {
            "Q": "standard_full_source_train_field",
            "s": "identity_dimensionless",
            "ell": "identity_dimensionless_L_over_L_ref",
            "target": "standard_full_source_train_field",
            "fit_uses_validation_lengths": False,
            "fit_uses_formal_long_lengths": False,
        },
        "output_normalization_policy": "standard_full_source_train_field",
        "spectral_parameterization": "physical_frequency_anchor_interpolation",
        "physical_frequency_formula": "k / (N * delta_lambda)",
        "num_lambda_frequency_anchors": 32,
        "anchor_frequency_values": anchors.tolist(),
        "complex_interpolation": "cartesian_linear",
        "runtime_retained_mode_policy": "fixed_discrete_indices_k_0_to_modes2_minus_1",
        "physical_cutoff_repair": False,
        "physical_bandwidth_shrinkage_repaired": False,
        "global_fft_structure_unchanged": True,
        "hypernetwork": False,
        "dynamic_spectral_weights": False,
    }
    return {"config": config, "epoch": 77, "best_val_mse": 0.155}


def make_field() -> response.r1response.ValidationQField:
    """构造乱序 validation Q 与 raw float64 T1200 truth。"""
    q = np.array([[2.0], [1.0], [3.0]], dtype=np.float64)
    grid = np.arange(1200, dtype=np.float64) * 0.01
    truth = np.ones((3, 1200, 3), dtype=np.float64)
    truth[:, :, 0] *= q[:, 0, None]
    return response.r1response.build_validation_field(
        task_name="synthetic_t1200",
        q_raw=q,
        truth_raw=truth,
        lambda_grid=grid,
    )


def make_rows(values: dict[int, float]) -> list[dict[str, object]]:
    """构造只含 primary metric 和 exposure 的紧凑 synthetic 行。"""
    train = {600, 800, 1000, 1200}
    selection = {700, 900, 1100, 1200}
    return [
        {
            "total_length": length,
            "gradient_seen": length in train,
            "checkpoint_selection_seen": length in selection,
            "mean_per_q_relative_l2": value,
        }
        for length, value in sorted(values.items())
    ]


class R3ValidationLengthResponseTests(unittest.TestCase):
    """验证 R3 validation-Q 诊断的科学隔离、sawtooth 统计和紧凑输出。"""

    def test_default_lengths_are_exactly_the_required_seven(self) -> None:
        self.assertEqual(response.DEFAULT_LENGTHS, (600, 700, 800, 900, 1000, 1100, 1200))
        self.assertEqual(
            response.validate_requested_lengths(response.DEFAULT_LENGTHS, source_length=1200),
            response.DEFAULT_LENGTHS,
        )

    def test_invalid_or_duplicate_lengths_are_rejected(self) -> None:
        for lengths in ((0,), (1201,), (600, 600)):
            with self.assertRaises(ValueError):
                response.validate_requested_lengths(lengths, source_length=1200)

    def test_validation_q_is_stably_canonicalized_once(self) -> None:
        field = make_field()
        self.assertTrue(np.array_equal(field.canonical_q, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(field.canonical_truth[:, 0, 0], np.array([1.0, 2.0, 3.0])))
        prefix = response.strict_prefix_field(field, 600)
        self.assertTrue(np.array_equal(prefix.canonical_q, field.canonical_q))
        self.assertTrue(np.array_equal(prefix.canonical_to_source_index, field.canonical_to_source_index))
        self.assertEqual(prefix.canonical_truth.dtype, np.float64)

    def test_prefixes_are_strict_prefixes_of_one_validation_source(self) -> None:
        field = make_field()
        prefix = response.strict_prefix_field(field, 900)
        self.assertTrue(np.array_equal(prefix.lambda_grid, field.lambda_grid[:900]))
        self.assertTrue(np.array_equal(prefix.canonical_truth, field.canonical_truth[:, :900, :]))

    def test_r3_provenance_uses_canonical_anchor_reconstruction(self) -> None:
        checkpoint = make_checkpoint()
        provenance = response.verify_r3_validation_provenance(
            checkpoint=checkpoint,
            task_name="synthetic_t1200",
        )
        self.assertEqual(provenance.input_channel_order, ("Q", "s", "ell"))
        self.assertEqual(provenance.training_lengths, (600, 800, 1000, 1200))
        self.assertEqual(len(provenance.anchor_frequencies), 32)
        self.assertTrue(np.all(np.diff(np.asarray(provenance.anchor_frequencies)) > 0.0))

    def test_missing_checkpoint_selection_provenance_fails(self) -> None:
        checkpoint = make_checkpoint()
        del checkpoint["config"]["checkpoint_selection"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "checkpoint-selection provenance"):
            response.verify_r3_validation_provenance(checkpoint=checkpoint, task_name="synthetic_t1200")

    def test_r3_input_is_q_s_ell_not_absolute_lambda(self) -> None:
        field = make_field()
        provenance = response.verify_r3_validation_provenance(
            checkpoint=make_checkpoint(), task_name="synthetic_t1200"
        )
        prefix = response.strict_prefix_field(field, 600)
        compatible = response.r3formal._R2CompatibleProvenance(provenance)
        x_raw, _, runtime = response.r3formal.r2eval.build_r2_model_input(
            q_values=prefix.canonical_q,
            lambda_grid=prefix.lambda_grid,
            raw_truth=prefix.canonical_truth,
            provenance=compatible,
        )
        self.assertEqual(x_raw.shape, (1, 3, 600, 3))
        self.assertTrue(np.allclose(x_raw[0, :, :, 0], prefix.canonical_q[:, None]))
        expected_s = prefix.lambda_grid / (600 * 0.01)
        self.assertTrue(np.allclose(x_raw[0, 0, :, 1], expected_s))
        self.assertTrue(np.allclose(x_raw[0, :, :, 2], 0.5))
        self.assertFalse(np.allclose(x_raw[0, 0, :, 1], prefix.lambda_grid))
        self.assertAlmostEqual(runtime.domain_length, 6.0)

    def test_r3_wrapper_accepts_validation_prefix_with_canonical_anchor_support(self) -> None:
        field = make_field()
        checkpoint = make_checkpoint()
        provenance = response.verify_r3_validation_provenance(
            checkpoint=checkpoint, task_name="synthetic_t1200"
        )
        prefix = response.strict_prefix_field(field, 600)

        class RuntimeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[int, float]] = []

            def validate_runtime_support(self, total_length: int, delta_lambda: float) -> None:
                self.calls.append((total_length, delta_lambda))

        model = RuntimeModel()
        coordinates = SimpleNamespace(ell=0.5, s_min=0.0, s_max=0.9983333333333333)
        with patch.object(
            response.r3formal.r2eval,
            "run_frozen_inference",
            return_value=(prefix.canonical_truth.astype(np.float32), coordinates),
        ):
            prediction, runtime = response.r3formal.run_frozen_inference(
                model=model,
                checkpoint=checkpoint,
                field=prefix,
                provenance=provenance,
                device="cpu",
            )
        self.assertTrue(np.array_equal(prediction, prefix.canonical_truth.astype(np.float32)))
        self.assertEqual(model.calls, [(600, 0.01)])
        self.assertEqual(runtime["retained_lambda_indices"], list(range(32)))
        self.assertAlmostEqual(runtime["r2_coordinates"]["ell"], 0.5)
    def test_length_statuses_do_not_call_non_gradient_lengths_test(self) -> None:
        train = {600, 800, 1000, 1200}
        selection = {700, 900, 1100, 1200}
        self.assertEqual(response._length_status(total_length=700, training_lengths=train, validation_lengths=selection), "non_gradient_validation_length")
        self.assertEqual(response._length_status(total_length=1200, training_lengths=train, validation_lengths=selection), "gradient_seen_and_checkpoint_selection_seen")

    def test_evaluate_response_runs_one_frozen_forward_per_length(self) -> None:
        field = make_field()
        checkpoint = make_checkpoint()
        provenance = response.verify_r3_validation_provenance(checkpoint=checkpoint, task_name="synthetic_t1200")
        model = FakeModel()
        values = {600: 0.02, 700: 0.40, 800: 0.03, 900: 0.50, 1000: 0.04, 1100: 0.45, 1200: 0.05}

        def fake_forward(*, field, **_kwargs):
            value = values[int(field.lambda_grid.size)]
            return field.canonical_truth.astype(np.float32) * (1.0 + value), {"retained_lambda_indices": [0]}

        with patch.object(response.r3formal, "run_frozen_inference", side_effect=fake_forward) as mocked:
            rows, per_q_rows = response.evaluate_length_response(
                model=model,
                checkpoint=checkpoint,
                field=field,
                provenance=provenance,
                lengths=response.DEFAULT_LENGTHS,
                device="cpu",
            )
        self.assertEqual(mocked.call_count, 7)
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(per_q_rows), 21)
        self.assertEqual([row["total_length"] for row in rows], list(response.DEFAULT_LENGTHS))
        self.assertTrue(all(row["frozen_forward_passes"] == 1 for row in rows))
        self.assertTrue(np.isclose(rows[1]["mean_per_q_relative_l2"], 0.40, atol=1e-6))
        self.assertEqual(rows[1]["length_status"], "non_gradient_validation_length")

    def test_primary_metrics_include_full_raw_metric_set(self) -> None:
        field = make_field()
        checkpoint = make_checkpoint()
        provenance = response.verify_r3_validation_provenance(checkpoint=checkpoint, task_name="synthetic_t1200")

        def fake_forward(*, field, **_kwargs):
            return field.canonical_truth.astype(np.float32) * 1.1, {}

        with patch.object(response.r3formal, "run_frozen_inference", side_effect=fake_forward):
            rows, _ = response.evaluate_length_response(
                model=FakeModel(), checkpoint=checkpoint, field=field, provenance=provenance,
                lengths=(600,), device="cpu"
            )
        for key in ("mse", "global_relative_l2", "mean_per_q_relative_l2", "median_per_q_relative_l2", "p95_per_q_relative_l2", "max_per_q_relative_l2", "component_metrics"):
            self.assertIn(key, rows[0])
        self.assertTrue(np.isclose(rows[0]["mean_per_q_relative_l2"], 0.1, atol=1e-6))

    def test_sawtooth_residuals_are_positive_for_spikes(self) -> None:
        summary = response.sawtooth_summary(make_rows({600: 0.02, 700: 0.40, 800: 0.03, 900: 0.50, 1000: 0.04, 1100: 0.45, 1200: 0.05}))
        self.assertGreater(summary["interpolation_residual_T700"], 0.3)
        self.assertGreater(summary["interpolation_residual_T900"], 0.4)
        self.assertGreater(summary["interpolation_residual_T1100"], 0.4)
        self.assertGreater(summary["non_gradient_minus_gradient_gap"], 0.0)

    def test_sawtooth_residuals_are_zero_for_a_linear_curve(self) -> None:
        summary = response.sawtooth_summary(make_rows({600: 0.02, 700: 0.03, 800: 0.04, 900: 0.05, 1000: 0.06, 1100: 0.07, 1200: 0.08}))
        self.assertAlmostEqual(summary["interpolation_residual_T700"], 0.0)
        self.assertAlmostEqual(summary["interpolation_residual_T900"], 0.0)
        self.assertAlmostEqual(summary["interpolation_residual_T1100"], 0.0)
        self.assertAlmostEqual(summary["mean_absolute_interpolation_residual"], 0.0)

    def test_adjacent_change_summary_is_descriptive(self) -> None:
        summary = response.sawtooth_summary(make_rows({600: 0.0, 700: 0.1, 800: 0.2, 900: 0.3, 1000: 0.4, 1100: 0.5, 1200: 0.6}))
        self.assertEqual(len(summary["adjacent_absolute_changes"]), 6)
        self.assertAlmostEqual(summary["mean_adjacent_absolute_change"], 0.1)
        self.assertIn("no_memorization", summary["interpretation_boundary"])

    def test_summary_marks_development_scope_and_no_refit(self) -> None:
        field = make_field()
        checkpoint = make_checkpoint()
        provenance = response.verify_r3_validation_provenance(checkpoint=checkpoint, task_name="synthetic_t1200")
        rows = make_rows({600: 0.02, 700: 0.03, 800: 0.04, 900: 0.05, 1000: 0.06, 1100: 0.07, 1200: 0.08})
        summary = response.build_summary(
            task_name="synthetic_t1200", checkpoint_path=PROJECT_ROOT / "checkpoint.pt",
            checkpoint=checkpoint, field=field, provenance=provenance,
            requested_lengths=response.DEFAULT_LENGTHS, rows=rows, device="cpu"
        )
        self.assertEqual(summary["scientific_status"], "development_diagnostic")
        self.assertFalse(summary["formal_test_evidence"])
        self.assertTrue(summary["same_validation_q_across_lengths"])
        self.assertEqual(summary["normalization_provenance"]["source"], "checkpoint_only_no_refit")
        self.assertFalse(summary["frozen_inference"]["adaptation"])
        self.assertFalse(summary["coordinate_representation"]["absolute_lambda_input"])
        self.assertAlmostEqual(summary["spectral_parameterization"]["training_delta_lambda"], 0.01)

    def test_writer_creates_exactly_three_compact_artifacts_and_refuses_overwrite(self) -> None:
        rows = make_rows({600: 0.02})
        per_q_rows = [{"total_length": 600, "Q": 1.0, "gradient_seen": True, "checkpoint_selection_seen": False, "mse": 0.0, "relative_l2": 0.02}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            response.write_outputs(output_dir=output, summary={"scientific_status": "development_diagnostic"}, rows=rows, per_q_rows=per_q_rows)
            self.assertEqual({item.name for item in output.iterdir()}, set(response.OUTPUT_FILENAMES))
            self.assertFalse(any(item.suffix in {".npy", ".pt"} for item in output.iterdir()))
            with (output / response.OUTPUT_FILENAMES[1]).open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["total_length"], "600")
            with (output / response.OUTPUT_FILENAMES[0]).open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["scientific_status"], "development_diagnostic")
            with self.assertRaises(FileExistsError):
                response.write_outputs(output_dir=output, summary={}, rows=[], per_q_rows=[])

    def test_cli_help_and_source_excludes_training_paths(self) -> None:
        completed = subprocess.run([sys.executable, "-B", str(SCRIPT_PATH), "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("T1800", source)
        self.assertNotIn("T2400", source)


if __name__ == "__main__":
    unittest.main()
