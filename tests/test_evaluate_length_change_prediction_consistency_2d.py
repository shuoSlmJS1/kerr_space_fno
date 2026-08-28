"""冻结长度变化预测一致性诊断的合成数据测试。"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_length_change_prediction_consistency_2d.py"
SPEC = importlib.util.spec_from_file_location("length_change_consistency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def _pair_artifact(*, classification: str = "EXACT_PREFIX", reusable: bool = True) -> dict[str, object]:
    return {
        "pair_classification": {"short_to_medium": classification},
        "scientific_reuse": {"historical_t1800_reusable": reusable},
    }


def _stats() -> FieldNormalizationStats:
    return FieldNormalizationStats(
        method="none",
        x_mean=[0.0, 0.0],
        x_std=[1.0, 1.0],
        y_mean=[0.0, 0.0, 0.0],
        y_std=[1.0, 1.0, 1.0],
    )


class MetricTests(unittest.TestCase):
    def test_identical_predictions_have_zero_consistency_error(self) -> None:
        prediction = np.arange(18, dtype=np.float64).reshape(2, 3, 3) + 1.0
        metrics = diagnostic.compute_comparison_metrics(
            prediction,
            prediction.copy(),
            np.array([1.6, 1.7]),
            denominator_description="short_prediction_raw_xyz",
        )
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["global_relative_l2"], 0.0)
        self.assertEqual(metrics["per_q_relative_l2"]["mean"], 0.0)

    def test_global_and_mean_per_q_relative_l2_are_distinct(self) -> None:
        reference = np.array([[[1.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])
        prediction = np.array([[[2.0, 0.0, 0.0]], [[12.0, 0.0, 0.0]]])
        metrics = diagnostic.compute_comparison_metrics(
            prediction,
            reference,
            np.array([1.6, 1.7]),
            denominator_description="shared_truth_raw_float64",
        )
        self.assertAlmostEqual(metrics["global_relative_l2"], np.sqrt(5.0 / 101.0))
        self.assertAlmostEqual(metrics["per_q_relative_l2"]["mean"], 0.6)
        self.assertNotAlmostEqual(
            metrics["global_relative_l2"], metrics["per_q_relative_l2"]["mean"]
        )

    def test_component_metrics_and_short_prediction_denominator(self) -> None:
        short_prediction = np.array([[[2.0, 4.0, 8.0]]])
        long_prediction = np.array([[[4.0, 5.0, 10.0]]])
        metrics = diagnostic.compute_comparison_metrics(
            long_prediction,
            short_prediction,
            np.array([2.0]),
            denominator_description="short_prediction_raw_xyz",
        )
        self.assertEqual(metrics["reference_description"], "short_prediction_raw_xyz")
        self.assertAlmostEqual(metrics["components"]["x"]["mse"], 4.0)
        self.assertAlmostEqual(metrics["components"]["y"]["mse"], 1.0)
        self.assertAlmostEqual(metrics["components"]["z"]["mse"], 4.0)
        self.assertAlmostEqual(
            metrics["global_relative_l2"],
            np.linalg.norm(np.array([2.0, 1.0, 2.0])) / np.linalg.norm(short_prediction.ravel()),
        )

    def test_perturbed_long_prefix_is_detected(self) -> None:
        short_prediction = np.ones((2, 2, 3), dtype=np.float64)
        long_prediction = short_prediction.copy()
        long_prediction[1, :, :] += 3.0
        metrics = diagnostic.compute_comparison_metrics(
            long_prediction,
            short_prediction,
            np.array([1.6, 1.7]),
            denominator_description="short_prediction_raw_xyz",
        )
        self.assertGreater(metrics["global_relative_l2"], 0.0)
        self.assertEqual(metrics["per_q_relative_l2"]["worst_q_index"], 1)
        self.assertEqual(metrics["per_q_relative_l2"]["worst_q_value"], 1.7)


class PairValidationTests(unittest.TestCase):
    def test_missing_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                diagnostic.load_required_pair_validation(Path(temporary) / "missing.json")

    def test_valid_artifact_is_required_and_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair.json"
            path.write_text(json.dumps(_pair_artifact()), encoding="utf-8", newline="\n")
            loaded = diagnostic.load_required_pair_validation(path)
        self.assertEqual(loaded["pair_classification"]["short_to_medium"], "EXACT_PREFIX")

    def test_not_paired_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair.json"
            path.write_text(
                json.dumps(_pair_artifact(classification="NOT_PAIRED")),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(ValueError, "EXACT_PREFIX"):
                diagnostic.load_required_pair_validation(path)

    def test_missing_reuse_boolean_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair.json"
            path.write_text(
                json.dumps(_pair_artifact(reusable=False)), encoding="utf-8", newline="\n"
            )
            with self.assertRaisesRegex(ValueError, "historical_t1800_reusable"):
                diagnostic.load_required_pair_validation(path)

    def test_split_q_order_is_preserved_and_mismatch_rejected(self) -> None:
        short_q = np.array([1.8, 1.6], dtype=np.float64)
        long_q = np.array([1.6, 1.8], dtype=np.float64)
        short_truth = np.ones((2, 2, 3), dtype=np.float64)
        long_truth = np.ones((2, 3, 3), dtype=np.float64)
        short_lambda = np.array([0.0, 1.0], dtype=np.float64)
        long_lambda = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "selected split order"):
            diagnostic.validate_raw_pair(
                short_q, short_truth, short_lambda, long_q, long_truth, long_lambda
            )

    def test_raw_pair_requires_exact_trajectory_prefix(self) -> None:
        q = np.array([1.6], dtype=np.float64)
        short_truth = np.ones((1, 2, 3), dtype=np.float64)
        long_truth = np.ones((1, 3, 3), dtype=np.float64)
        long_truth[0, 1, 2] += 1e-12
        with self.assertRaisesRegex(ValueError, "trajectories"):
            diagnostic.validate_raw_pair(
                q,
                short_truth,
                np.array([0.0, 1.0]),
                q.copy(),
                long_truth,
                np.array([0.0, 1.0, 2.0]),
            )


class ResultAndOutputTests(unittest.TestCase):
    def test_result_uses_same_canonical_short_truth_for_both_truth_metrics(self) -> None:
        short_truth = np.array([[[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]])
        short_prediction = short_truth + 1.0
        long_prediction = np.concatenate((short_truth + 2.0, short_truth[:, :1, :]), axis=1)
        checkpoint = {"config": {"model_config": {"model_type": "fno2d"}}}
        with patch.object(diagnostic, "load_normalization_stats_from_checkpoint", return_value=_stats()), patch.object(
            diagnostic,
            "load_target_transform_config_from_checkpoint",
            return_value=TargetTransformConfig(mode="raw"),
        ):
            result = diagnostic.build_result(
                training_task_name="training",
                model_name="model",
                short_task_name="short",
                long_task_name="long",
                checkpoint_path=PROJECT_ROOT / "checkpoint.pt",
                checkpoint=checkpoint,
                pair_validation_path=PROJECT_ROOT / "pair.json",
                pair_validation=_pair_artifact(),
                split_policy="test",
                device="cpu",
                short_q=np.array([1.6]),
                short_truth=short_truth,
                short_lambda=np.array([0.0, 1.0]),
                long_lambda=np.array([0.0, 1.0, 2.0]),
                short_prediction=short_prediction,
                long_prediction=long_prediction,
            )
        metrics = result["metrics"]
        self.assertEqual(result["truth_reference"]["name"], "shared_truth_raw_float64")
        self.assertEqual(
            metrics["short_prediction_vs_shared_truth"]["reference_description"],
            "shared_truth_raw_float64",
        )
        self.assertEqual(
            metrics["long_prefix_prediction_vs_shared_truth"]["reference_description"],
            "shared_truth_raw_float64",
        )
        self.assertAlmostEqual(metrics["short_prediction_vs_shared_truth"]["mse"], 1.0)
        self.assertAlmostEqual(metrics["long_prefix_prediction_vs_shared_truth"]["mse"], 4.0)
        self.assertTrue(result["frozen_inference_only"])
        self.assertFalse(result["autoregressive_rollout"])
        self.assertEqual(result["adaptation"], "none")
        json.dumps(result, allow_nan=False)

    def test_output_refuses_overwrite_and_writes_only_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "result.json"
            diagnostic.write_json_exclusively({"status": "completed"}, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "completed")
            with self.assertRaises(FileExistsError):
                diagnostic.write_json_exclusively({"status": "completed"}, output)
            self.assertEqual([path.name for path in directory.iterdir()], ["result.json"])


class InterfaceSafetyTests(unittest.TestCase):
    def test_cli_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        self.assertIn("--dataset-pair-validation-json", completed.stdout)
        self.assertIn("--output-json", completed.stdout)
        self.assertIn("--split", completed.stdout)

    def test_source_has_no_training_or_large_artifact_output_path(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        prohibited = ("optimizer", ".backward(", "scheduler", "np.save(", "np.savez", "torch.save(")
        for token in prohibited:
            self.assertNotIn(token, source)
        self.assertIn("predict_2d_loader", source)
        self.assertIn("frozen_inference_only", source)
        self.assertIn("autoregressive_rollout", source)

    def test_frozen_inference_uses_the_existing_no_grad_predictor_without_feedback(self) -> None:
        inference_source = inspect.getsource(diagnostic.run_frozen_inference)
        predictor_source = inspect.getsource(diagnostic.predict_2d_loader)
        self.assertIn("predict_2d_loader", inference_source)
        self.assertEqual(inference_source.count("predict_2d_loader("), 1)
        self.assertNotIn("for ", inference_source)
        self.assertNotIn("model(", inference_source)
        self.assertIn("@torch.no_grad()", predictor_source)

    def test_model_input_builder_contains_only_q_and_lambda_channels(self) -> None:
        q = np.array([1.6, 1.7])
        lambda_grid = np.array([0.0, 1.0, 2.0])
        truth = np.ones((2, 3, 3), dtype=np.float64)
        x_raw, y_raw = diagnostic.build_raw_field(q, lambda_grid, truth)
        self.assertEqual(x_raw.shape, (1, 2, 3, 2))
        self.assertEqual(y_raw.shape, (1, 2, 3, 3))
        self.assertTrue(
            np.array_equal(
                x_raw[0, :, :, 0],
                np.broadcast_to(q.astype(np.float32)[:, None], (2, 3)),
            )
        )
        self.assertTrue(
            np.array_equal(
                x_raw[0, :, :, 1],
                np.broadcast_to(lambda_grid.astype(np.float32)[None, :], (2, 3)),
            )
        )


if __name__ == "__main__":
    unittest.main()
