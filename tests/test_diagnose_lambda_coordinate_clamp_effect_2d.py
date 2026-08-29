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
from src.training.fno2d.normalization_2d import FieldNormalizationStats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_lambda_coordinate_clamp_effect_2d.py"
SPEC = importlib.util.spec_from_file_location("m4b_clamp", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
m4b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m4b
SPEC.loader.exec_module(m4b)


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


def _field(name: str, q: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> m4b.CanonicalQField:
    records = [
        {"source_split": "train", "source_index_within_split": index, "source_concatenated_index": index}
        for index in range(q.size)
    ]
    return m4b.m3.build_canonical_q_field(
        task_name=name,
        source_q=q,
        source_truth=truth,
        lambda_grid=grid,
        source_records=records,
    )


def _triplet() -> tuple[m4b.CanonicalQField, m4b.CanonicalQField, m4b.CanonicalQField]:
    q = np.array([2.0, 1.0], dtype=np.float64)
    short_truth = np.arange(2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3)
    medium_truth = np.concatenate((short_truth, short_truth[:, :4, :] + 100.0), axis=1)
    long_truth = np.concatenate((medium_truth, medium_truth[:, :4, :] + 200.0), axis=1)
    return (
        _field("short", q, short_truth, np.arange(8, dtype=np.float64) * 0.5),
        _field("medium", q, medium_truth, np.arange(12, dtype=np.float64) * 0.5),
        _field("long", q, long_truth, np.arange(16, dtype=np.float64) * 0.5),
    )


def _model() -> FNO2d:
    torch.manual_seed(23)
    model = FNO2d(in_dim=2, out_dim=3, modes1=2, modes2=4, width=4, depth=2, hidden_dim=8)
    model.eval()
    return model


def _checkpoint() -> dict[str, object]:
    return {
        "config": {
            "normalization": "standard",
            "model_config": {"in_dim": 2},
            "dataset_summary": {
                "normalization_stats": {
                    "method": "standard",
                    "x_mean": [2.0, 1.75],
                    "x_std": [0.5, 1.25],
                    "y_mean": [0.0, 0.0, 0.0],
                    "y_std": [1.0, 1.0, 1.0],
                    "eps": 1e-8,
                },
                "train": {"input_channel_names": ["Q", "lambda"]},
            },
        }
    }


def _normalized_input(width: int) -> np.ndarray:
    values = np.zeros((1, 2, width, 2), dtype=np.float32)
    values[..., 0] = np.array([[-1.0], [1.0]], dtype=np.float32)
    values[..., 1] = np.arange(width, dtype=np.float32).reshape(1, 1, width)
    return values


class PrerequisiteAndClampTests(unittest.TestCase):
    def test_invalid_stage2_prerequisite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(_pair_artifact(False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "short_to_medium"):
                m4b.load_required_pair_validation(path)

    def test_canonical_q_and_exact_raw_prefix_validation(self) -> None:
        short, medium, long = _triplet()
        self.assertTrue(np.array_equal(short.canonical_q, np.array([1.0, 2.0])))
        self.assertTrue(np.array_equal(short.canonical_truth, medium.canonical_truth[:, :8, :]))
        m4b.validate_triplet(short, medium, long)
        changed = long.source_truth.copy()
        changed[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "T1200 and T2400"):
            m4b.validate_triplet(short, medium, _field("bad", long.source_q, changed, long.lambda_grid))

    def test_checkpoint_standard_normalization_and_channel_contract(self) -> None:
        stats, names, q_index, lambda_index = m4b.require_checkpoint_input_contract(_checkpoint())
        self.assertEqual(stats.method, "standard")
        self.assertEqual(names, ("Q", "lambda"))
        self.assertEqual((q_index, lambda_index), (0, 1))
        bad = _checkpoint()
        bad["config"]["dataset_summary"]["train"]["input_channel_names"] = ["lambda", "Q"]
        with self.assertRaisesRegex(ValueError, "input channel order"):
            m4b.require_checkpoint_input_contract(bad)

    def test_clamp_bound_is_derived_from_short_grid_and_checkpoint_stats(self) -> None:
        stats, _, _, lambda_index = m4b.require_checkpoint_input_contract(_checkpoint())
        raw_max, bound = m4b.derive_lambda_clamp_bound(np.arange(8, dtype=np.float64) * 0.5, stats, lambda_index)
        self.assertEqual(raw_max, 3.5)
        self.assertAlmostEqual(bound, (3.5 - 1.75) / 1.25)

    def test_clamp_changes_only_appended_lambda_channel_and_preserves_prefix(self) -> None:
        original = _normalized_input(12)
        clamped, details = m4b.clamp_appended_lambda_channel(
            original,
            shared_length=8,
            q_channel_index=0,
            lambda_channel_index=1,
            clamp_upper_bound=7.0,
        )
        self.assertTrue(np.array_equal(original[:, :, :8, :], clamped[:, :, :8, :]))
        self.assertTrue(np.array_equal(original[..., 0], clamped[..., 0]))
        self.assertTrue(np.array_equal(clamped[..., 1][:, :, 8:], np.full((1, 2, 4), 7.0, dtype=np.float32)))
        self.assertEqual(details["number_of_modified_lambda_positions"], 4)
        self.assertAlmostEqual(details["fraction_of_total_lambda_positions_modified"], 4.0 / 12.0)
        self.assertEqual(details["number_of_modified_lambda_tensor_values"], 8)

    def test_synthetic_t1800_t2400_style_modified_counts(self) -> None:
        for total, expected in ((18, 6), (24, 12)):
            original = _normalized_input(total)
            _, details = m4b.clamp_appended_lambda_channel(
                original,
                shared_length=12,
                q_channel_index=0,
                lambda_channel_index=1,
                clamp_upper_bound=11.0,
            )
            self.assertEqual(details["number_of_modified_lambda_positions"], expected)
            self.assertAlmostEqual(details["fraction_of_total_lambda_positions_modified"], expected / total)


class FrozenForwardAndMetricsTests(unittest.TestCase):
    def test_hooks_preserve_forward_output_and_one_forward_per_arm(self) -> None:
        model = _model()
        forward_count = 0

        def count_forward(_: torch.nn.Module, __: tuple[object, ...], ___: torch.Tensor) -> None:
            nonlocal forward_count
            forward_count += 1

        handle = model.register_forward_hook(count_forward)
        try:
            input_tensor = torch.randn(1, 2, 12, 2)
            with torch.no_grad():
                baseline = model(input_tensor)
            captured, hooks = m4b.m3.capture_one_forward(model, input_tensor)
            self.assertTrue(torch.equal(baseline, captured))
            self.assertIsNotNone(hooks.spectral_branch_output)
            for _ in range(5):
                m4b.m3.capture_one_forward(model, torch.randn(1, 2, 12, 2))
        finally:
            handle.remove()
        self.assertEqual(forward_count, 7)

    def test_same_length_spectral_comparison_uses_same_discrete_grid(self) -> None:
        model = _model()
        field = _triplet()[1]
        original = _normalized_input(12)
        clamped, _ = m4b.clamp_appended_lambda_channel(
            original, shared_length=8, q_channel_index=0, lambda_channel_index=1, clamp_upper_bound=7.0
        )
        captures: dict[str, m4b.ArmCapture] = {}
        for suffix, values in (("original", original), ("lambda_clamped", clamped)):
            _, hooks = m4b.m3.capture_one_forward(model, torch.from_numpy(values))
            spectral, replicated = m4b.m3._replicate_first_spectral_layer(hooks.first_spectral_input, model.blocks[0].spectral_conv, "cpu")
            spectral.replicated_output_metrics = m4b.m3._metrics(replicated, hooks.spectral_branch_output)
            captures[f"T1800_{suffix}"] = m4b.ArmCapture(
                arm_name=f"T1800_{suffix}", length_label="T1800", intervention=suffix,
                field=field, normalized_input=values, normalized_input_prefix=torch.from_numpy(values[:, :, :8, :]),
                lifted_feature_prefix=hooks.lifted_feature[:, :, :8, :],
                first_spectral_input_prefix=hooks.first_spectral_input[:, :, :, :8],
                spectral_branch_output_prefix=hooks.spectral_branch_output[:, :, :, :8],
                first_block_output_prefix=hooks.first_block_output[:, :, :, :8],
                final_prediction_prefix=np.zeros((2, 8, 3), dtype=np.float64),
                secondary_truth_reference_prefix_metrics={"mean_per_q_relative_l2": 1.0}, spectral=spectral,
            )
        metrics = m4b._same_index_spectral_metrics(captures["T1800_original"], captures["T1800_lambda_clamped"], "first_fft_retained")
        self.assertGreater(metrics["relative_l2_difference"], 0.0)
        self.assertEqual(captures["T1800_original"].field.lambda_grid.size, captures["T1800_lambda_clamped"].field.lambda_grid.size)

    def test_stage_rows_make_intervention_response_and_t1200_reference_primary(self) -> None:
        model = _model()
        short_field, medium_field, long_field = _triplet()

        def capture(arm_name: str, label: str, field: m4b.CanonicalQField, values: np.ndarray) -> m4b.ArmCapture:
            _, hooks = m4b.m3.capture_one_forward(model, torch.from_numpy(values))
            spectral, replicated = m4b.m3._replicate_first_spectral_layer(
                hooks.first_spectral_input, model.blocks[0].spectral_conv, "cpu"
            )
            spectral.replicated_output_metrics = m4b.m3._metrics(replicated, hooks.spectral_branch_output)
            return m4b.ArmCapture(
                arm_name=arm_name,
                length_label=label,
                intervention="lambda_clamped" if "clamped" in arm_name else "original",
                field=field,
                normalized_input=values,
                normalized_input_prefix=torch.from_numpy(values[:, :, :8, :]),
                lifted_feature_prefix=hooks.lifted_feature[:, :, :8, :],
                first_spectral_input_prefix=hooks.first_spectral_input[:, :, :, :8],
                spectral_branch_output_prefix=hooks.spectral_branch_output[:, :, :, :8],
                first_block_output_prefix=hooks.first_block_output[:, :, :, :8],
                final_prediction_prefix=hooks.first_block_output[0, :, :, :8].permute(1, 2, 0).numpy(),
                secondary_truth_reference_prefix_metrics={"mean_per_q_relative_l2": 1.0},
                spectral=spectral,
            )

        medium_original = _normalized_input(12)
        medium_clamped, _ = m4b.clamp_appended_lambda_channel(
            medium_original, shared_length=8, q_channel_index=0, lambda_channel_index=1, clamp_upper_bound=7.0
        )
        long_original = _normalized_input(16)
        long_clamped, _ = m4b.clamp_appended_lambda_channel(
            long_original, shared_length=8, q_channel_index=0, lambda_channel_index=1, clamp_upper_bound=7.0
        )
        captures = {
            "T1200_original": capture("T1200_original", "T1200", short_field, _normalized_input(8)),
            "T1800_original": capture("T1800_original", "T1800", medium_field, medium_original),
            "T1800_lambda_clamped": capture("T1800_lambda_clamped", "T1800", medium_field, medium_clamped),
            "T2400_original": capture("T2400_original", "T2400", long_field, long_original),
            "T2400_lambda_clamped": capture("T2400_lambda_clamped", "T2400", long_field, long_clamped),
        }
        rows = m4b.build_stage_rows(captures)
        spatial = next(row for row in rows if row["stage"] == "spectral_branch_output_prefix")
        self.assertIn("original_vs_clamped_relative_l2_difference", spatial)
        self.assertIn("original_to_t1200_reference_relative_l2_difference", spatial)
        self.assertIn("intervention_reference_distance_reduction", spatial)
        fft = next(row for row in rows if row["stage"] == "first_fft_retained")
        self.assertEqual(fft["view"], "same_discrete_index_same_fft_grid")
        self.assertIsNone(fft["original_to_t1200_reference_relative_l2_difference"])
    def test_t1200_reference_distance_recovery_and_secondary_truth_context(self) -> None:
        before = {"relative_l2_difference": 2.0, "normalized_rmse": 0.8, "cosine_similarity": 0.2}
        after = {"relative_l2_difference": 1.5, "normalized_rmse": 0.6, "cosine_similarity": 0.4}
        recovery = m4b._reference_distance_change(before, after)
        self.assertAlmostEqual(recovery["intervention_reference_distance_reduction"], 0.5)
        self.assertAlmostEqual(recovery["intervention_reference_distance_reduction_fraction"], 0.25)
        self.assertAlmostEqual(recovery["reference_cosine_similarity_change"], 0.2)
        original = type("Arm", (), {"secondary_truth_reference_prefix_metrics": {"mean_per_q_relative_l2": 2.0}})()
        clamped = type("Arm", (), {"secondary_truth_reference_prefix_metrics": {"mean_per_q_relative_l2": 1.5}})()
        truth_change = m4b.secondary_truth_reference_change(original, clamped)
        self.assertAlmostEqual(truth_change["truth_reference_absolute_change_original_minus_clamped"], 0.5)
        self.assertTrue(truth_change["not_formal_model_performance"])

    def test_nonphysical_mechanism_probe_semantics_are_explicit(self) -> None:
        semantics = m4b.intervention_semantics()
        self.assertEqual(semantics["intervention_type"], "nonphysical_appended_lambda_coordinate_clamp")
        self.assertEqual(semantics["purpose"], "mechanism_probe")
        self.assertFalse(semantics["valid_for_production_prediction"])
        self.assertFalse(semantics["valid_as_formal_length_extrapolation_protocol"])
        self.assertFalse(semantics["shared_prefix_coordinates_modified"])
        self.assertTrue(semantics["appended_lambda_coordinates_modified"])

    def test_summary_foregrounds_intervention_and_marks_truth_secondary(self) -> None:
        model = _model()
        short, medium, long = _triplet()
        stats, names, q_index, lambda_index = m4b.require_checkpoint_input_contract(_checkpoint())
        spectral = m4b.m3.SpectralCapture(
            frequencies=np.array([0.0]),
            pre_pos=torch.zeros((1, 1, 1, 1), dtype=torch.cfloat),
            pre_neg=torch.zeros((1, 1, 1, 1), dtype=torch.cfloat),
            post_pos=torch.zeros((1, 1, 1, 1), dtype=torch.cfloat),
            post_neg=torch.zeros((1, 1, 1, 1), dtype=torch.cfloat),
            modes_q=1,
            modes_lambda=1,
            full_fft_energy=0.0,
            retained_input_energy=0.0,
            retained_output_energy=0.0,
            replicated_output_metrics={"relative_l2_difference": 0.0},
        )

        def arm(name: str, label: str, field: m4b.CanonicalQField, intervention: str) -> m4b.ArmCapture:
            width = field.lambda_grid.size
            return m4b.ArmCapture(
                arm_name=name,
                length_label=label,
                intervention=intervention,
                field=field,
                normalized_input=np.zeros((1, 2, width, 2), dtype=np.float32),
                normalized_input_prefix=torch.zeros((1, 2, 8, 2)),
                lifted_feature_prefix=torch.zeros((1, 2, 8, 4)),
                first_spectral_input_prefix=torch.zeros((1, 4, 2, 8)),
                spectral_branch_output_prefix=torch.zeros((1, 4, 2, 8)),
                first_block_output_prefix=torch.zeros((1, 4, 2, 8)),
                final_prediction_prefix=np.zeros((2, 8, 3), dtype=np.float64),
                secondary_truth_reference_prefix_metrics={"mean_per_q_relative_l2": 1.0},
                spectral=spectral,
            )

        captures = {
            "T1200_original": arm("T1200_original", "T1200", short, "original"),
            "T1800_original": arm("T1800_original", "T1800", medium, "original"),
            "T1800_lambda_clamped": arm("T1800_lambda_clamped", "T1800", medium, "lambda_clamped"),
            "T2400_original": arm("T2400_original", "T2400", long, "original"),
            "T2400_lambda_clamped": arm("T2400_lambda_clamped", "T2400", long, "lambda_clamped"),
        }
        summary = m4b.build_summary(
            args=argparse.Namespace(training_task_name="train", model_name="model", device="cpu"),
            checkpoint_path=PROJECT_ROOT / "synthetic.pt",
            checkpoint=_checkpoint(),
            pair_validation=_pair_artifact(),
            pair_path=PROJECT_ROOT / "pair.json",
            captures=captures,
            clamp_details={"T1800": {}, "T2400": {}},
            stage_rows=[],
            model=model,
            input_channel_names=names,
            stats=stats,
            short_lambda_max_raw=3.5,
            clamp_upper_bound=1.4,
            q_channel_index=q_index,
            lambda_channel_index=lambda_index,
            lifting={},
        )
        self.assertEqual(summary["intervention_semantics"]["purpose"], "mechanism_probe")
        self.assertFalse(summary["intervention_semantics"]["valid_for_production_prediction"])
        self.assertEqual(summary["primary_intervention_response"]["primary_evidence_table"], "m4b_stage_comparison.csv")
        self.assertTrue(summary["secondary_truth_reference"]["not_formal_model_performance"])
        self.assertTrue(summary["metric_definitions"]["secondary_truth_reference"]["not_formal_model_performance"])
        self.assertNotIn("final_prediction_metrics", summary["arms"]["T1800_lambda_clamped"])
    def test_lifting_decomposition_is_compact_and_energy_based(self) -> None:
        model = _model()
        decomposition = m4b.lifting_decomposition(
            _normalized_input(12), model, q_channel_index=0, lambda_channel_index=1, shared_length=8
        )
        self.assertEqual(set(decomposition), {"prefix", "appended", "full"})
        self.assertGreaterEqual(decomposition["appended"]["lambda_fraction_of_decomposed_energy"], 0.0)
        self.assertLessEqual(decomposition["appended"]["lambda_fraction_of_decomposed_energy"], 1.0)


class OutputAndInterfaceTests(unittest.TestCase):
    def test_output_refuses_overwrite_and_writes_only_three_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "m4b"
            truth_rows = [{
                "arm": "T1800_original", "length_label": "T1800", "intervention": "original",
                "reference_type": "secondary_truth_reference", "scientific_scope": "mechanism_intervention_context_only",
                "not_formal_model_performance": True, "region": "shared_prefix_only", "mse": 1.0,
                "global_relative_l2": 2.0, "mean_per_q_relative_l2": 3.0,
                "median_per_q_relative_l2": 3.0, "p95_per_q_relative_l2": 4.0, "max_per_q_relative_l2": 5.0,
            }]
            stage_rows = [{
                "comparison": "a", "total_length": 12, "stage": "first_fft_retained",
                "view": "same_discrete_index_same_fft_grid",
                "original_vs_clamped_relative_l2_difference": 1.0,
                "original_vs_clamped_normalized_rmse": 1.0,
                "original_vs_clamped_cosine_similarity": 0.0,
                "original_to_t1200_reference_relative_l2_difference": None,
                "clamped_to_t1200_reference_relative_l2_difference": None,
                "intervention_reference_distance_reduction": None,
                "intervention_reference_distance_reduction_fraction": None,
                "original_to_t1200_reference_normalized_rmse": None,
                "clamped_to_t1200_reference_normalized_rmse": None,
                "original_to_t1200_reference_cosine_similarity": None,
                "clamped_to_t1200_reference_cosine_similarity": None,
                "reference_cosine_similarity_change": None,
            }]
            m4b.write_output_artifacts(output_dir=output_dir, summary={"schema_version": "1.0"}, secondary_truth_reference_rows=truth_rows, stage_rows=stage_rows)
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(m4b.OUTPUT_FILENAMES))
            self.assertFalse(any(path.suffix in {".npy", ".pt"} for path in output_dir.iterdir()))
            with (output_dir / "m4b_intervention_response.csv").open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row["reference_type"], "secondary_truth_reference")
                self.assertEqual(row["not_formal_model_performance"], "True")
            with self.assertRaises(FileExistsError):
                m4b.write_output_artifacts(output_dir=output_dir, summary={}, secondary_truth_reference_rows=[], stage_rows=[])

    def test_cli_help_and_static_no_training_paths(self) -> None:
        completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--checkpoint-path", completed.stdout)
        self.assertIn("nonphysical", completed.stdout)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertIn('"adaptation": False', source)
        self.assertIn('"autoregression": False', source)
        self.assertIn('"prediction_feedback": False', source)
        self.assertNotIn(".npy", source)
        self.assertIn("nonphysical_appended_lambda_coordinate_clamp", source)
        self.assertIn("not_formal_model_performance", source)
        self.assertNotIn("absolute_error_reduction", source)
        self.assertNotIn("fractional_error_reduction", source)
        self.assertNotIn("final_prediction_metrics", source)
        self.assertNotIn("prefix_error_response", source)


if __name__ == "__main__":
    unittest.main()
