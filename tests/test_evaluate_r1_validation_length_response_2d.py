"""R1 validation-Q 长度响应开发诊断的合成测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import evaluate_r1_validation_length_response_2d as diagnostic
from src.models.registry_2d import build_model_2d
from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


class ZeroModel(torch.nn.Module):
    """用于检查冻结前向次数和指标定义的确定性模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.forward_widths: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.forward_widths.append(int(x.shape[2]))
        return torch.zeros((x.shape[0], x.shape[1], x.shape[2], 3), dtype=x.dtype)


def make_stats() -> FieldNormalizationStats:
    """构造 raw-space identity 的 checkpoint 标准化统计量。"""

    return FieldNormalizationStats(
        method="standard",
        x_mean=np.array([0.0, 0.0], dtype=np.float32),
        x_std=np.array([1.0, 1.0], dtype=np.float32),
        y_mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        y_std=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def make_model_config() -> dict[str, object]:
    """构造与 restore helper 相同字段的最小模型配置。"""

    return {
        "model_type": "fno2d",
        "in_dim": 2,
        "out_dim": 3,
        "modes1": 2,
        "modes2": 2,
        "width": 4,
        "depth": 1,
        "hidden_dim": 8,
        "activation": "gelu",
    }


def make_checkpoint(
    *,
    source_task: str = "synthetic_t1200",
    train_lengths: list[int] | None = None,
    validation_lengths: list[int] | None = None,
) -> dict[str, object]:
    """构造最小的 R1 checkpoint metadata。"""

    stats = make_stats()
    target_transform = TargetTransformConfig(mode="raw")
    return {
        "epoch": 7,
        "best_val_mse": 0.125,
        "config": {
            "experiment_type": "r1_variable_length_training",
            "repair_class": "TRAINING_PROTOCOL_REPAIR",
            "source_task": source_task,
            "train_lengths": train_lengths or [600, 800, 1000, 1200],
            "validation_lengths": validation_lengths or [700, 900, 1100, 1200],
            "model_config": make_model_config(),
            "normalization": "standard",
            "target_transform": "raw",
            "dataset_summary": {
                "normalization_stats": stats.to_dict(),
                "target_transform_config": target_transform.to_dict(),
            },
        },
    }


def make_field(*, nonconstant_truth: bool = True) -> diagnostic.ValidationQField:
    """构造行顺序打乱、raw float64 的 T1200 validation 场。"""

    q_raw = np.array([[2.0], [1.0], [3.0]], dtype=np.float64)
    lambda_grid = np.arange(1200, dtype=np.float64) * 0.005
    truth = np.ones((3, 1200, 3), dtype=np.float64)
    if nonconstant_truth:
        truth[0, :, 0] = 2.0
        truth[1, :, 0] = 1.0
        truth[2, :, 0] = 3.0
    return diagnostic.build_validation_field(
        task_name="synthetic_t1200",
        q_raw=q_raw,
        truth_raw=truth,
        lambda_grid=lambda_grid,
    )


class TestR1ValidationLengthResponse(unittest.TestCase):
    """验证 validation-Q 隔离、provenance、冻结 forward 和紧凑输出。"""

    def test_canonical_q_sort_applies_the_same_truth_permutation(self) -> None:
        field = make_field()
        np.testing.assert_array_equal(field.canonical_q, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(field.canonical_truth[:, 0, 0], np.array([1.0, 2.0, 3.0]))
        self.assertEqual(field.lambda_grid.size, 1200)

    def test_source_contract_rejects_non_t1200_and_non_float64_truth(self) -> None:
        q_raw = np.array([[1.0]], dtype=np.float64)
        truth = np.ones((1, 1200, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "T1200"):
            diagnostic.build_validation_field(
                task_name="synthetic_t1200",
                q_raw=q_raw,
                truth_raw=truth[:, :1199, :],
                lambda_grid=np.arange(1199, dtype=np.float64),
            )
        with self.assertRaisesRegex(ValueError, "float64"):
            diagnostic.build_validation_field(
                task_name="synthetic_t1200",
                q_raw=q_raw,
                truth_raw=truth.astype(np.float32),
                lambda_grid=np.arange(1200, dtype=np.float64),
            )

    def test_load_validation_source_reads_only_validation_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.npz"
            np.savez(
                dataset_path,
                x_val=np.array([[2.0], [1.0]], dtype=np.float64),
                y_val=np.ones((2, 1200, 3), dtype=np.float64),
                lambda_grid=np.arange(1200, dtype=np.float64) * 0.005,
            )
            with (
                mock.patch.object(diagnostic, "get_task_dataset_npz_path", return_value=dataset_path),
                mock.patch.object(diagnostic, "load_task_meta", return_value={}),
                mock.patch.object(diagnostic, "validate_task_generation_meta"),
                mock.patch.object(diagnostic, "load_task_vary_params_order", return_value=["Q"]),
            ):
                field = diagnostic.load_validation_source("synthetic_t1200")
        np.testing.assert_array_equal(field.canonical_q, np.array([1.0, 2.0]))

    def test_strict_prefix_and_requested_lengths_are_validated(self) -> None:
        field = make_field()
        lambda_prefix, truth_prefix = diagnostic.strict_prefix(field, 600)
        self.assertEqual(lambda_prefix.shape, (600,))
        self.assertEqual(truth_prefix.shape, (3, 600, 3))
        np.testing.assert_array_equal(truth_prefix, field.canonical_truth[:, :600, :])
        self.assertEqual(diagnostic.validate_requested_lengths([800, 600]), (600, 800))
        for invalid in ([0], [1201], [600, 600], []):
            with self.assertRaises(ValueError):
                diagnostic.validate_requested_lengths(invalid)

    def test_r1_provenance_is_required_and_labels_are_metadata_derived(self) -> None:
        train_lengths, selection_lengths = diagnostic.verify_r1_provenance(
            make_checkpoint(), "synthetic_t1200"
        )
        self.assertEqual(train_lengths, (600, 800, 1000, 1200))
        self.assertEqual(selection_lengths, (700, 900, 1100, 1200))
        with self.assertRaisesRegex(ValueError, "experiment_type"):
            diagnostic.verify_r1_provenance({"config": {}}, "synthetic_t1200")
        with self.assertRaisesRegex(ValueError, "source_task"):
            diagnostic.verify_r1_provenance(make_checkpoint(source_task="other"), "synthetic_t1200")
        with self.assertRaisesRegex(ValueError, "unique"):
            diagnostic.verify_r1_provenance(
                make_checkpoint(train_lengths=[600, 600]), "synthetic_t1200"
            )

    def test_checkpoint_normalization_is_reused_for_all_prefixes(self) -> None:
        field = make_field(nonconstant_truth=False)
        stats = FieldNormalizationStats(
            method="standard",
            x_mean=np.array([2.0, 1.0], dtype=np.float32),
            x_std=np.array([2.0, 0.5], dtype=np.float32),
            y_mean=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            y_std=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        )
        target_transform = TargetTransformConfig(mode="raw")
        lambda_600, truth_600 = diagnostic.strict_prefix(field, 600)
        x_600, y_600, _ = diagnostic._model_space_arrays(
            q_values=field.canonical_q,
            lambda_grid=lambda_600,
            raw_truth=truth_600,
            normalization_stats=stats,
            target_transform=target_transform,
        )
        x_1200, y_1200, _ = diagnostic._model_space_arrays(
            q_values=field.canonical_q,
            lambda_grid=field.lambda_grid,
            raw_truth=field.canonical_truth,
            normalization_stats=stats,
            target_transform=target_transform,
        )
        np.testing.assert_array_equal(x_600, x_1200[:, :, :600, :])
        np.testing.assert_array_equal(y_600, y_1200[:, :, :600, :])
        self.assertAlmostEqual(float(x_1200[0, 0, 0, 1]), -2.0)

    def test_default_seven_lengths_use_one_frozen_forward_each(self) -> None:
        field = make_field(nonconstant_truth=False)
        model = ZeroModel()
        rows = diagnostic.evaluate_length_response(
            model=model,
            field=field,
            lengths=diagnostic.DEFAULT_LENGTHS,
            training_lengths=(600, 800, 1000, 1200),
            checkpoint_selection_lengths=(700, 900, 1100, 1200),
            normalization_stats=make_stats(),
            target_transform=TargetTransformConfig(mode="raw"),
            device="cpu",
        )
        self.assertEqual(model.forward_calls, 7)
        self.assertEqual(model.forward_widths, list(diagnostic.DEFAULT_LENGTHS))
        self.assertEqual([row["frozen_forward_passes"] for row in rows], [1] * 7)
        self.assertEqual(
            [(row["seen_during_training"], row["used_for_checkpoint_selection"]) for row in rows],
            [(True, False), (False, True), (True, False), (False, True),
             (True, False), (False, True), (True, True)],
        )

    def test_metrics_use_normalized_mse_and_raw_relative_l2(self) -> None:
        field = make_field(nonconstant_truth=False)
        rows = diagnostic.evaluate_length_response(
            model=ZeroModel(),
            field=field,
            lengths=(600,),
            training_lengths=(600,),
            checkpoint_selection_lengths=(700,),
            normalization_stats=make_stats(),
            target_transform=TargetTransformConfig(mode="raw"),
            device="cpu",
        )
        row = rows[0]
        self.assertAlmostEqual(float(row["normalized_space_mse"]), 1.0)
        self.assertAlmostEqual(float(row["raw_global_relative_l2"]), 1.0)
        self.assertAlmostEqual(float(row["raw_mean_per_q_relative_l2"]), 1.0)

    def test_summary_writes_exactly_two_compact_artifacts_and_refuses_overwrite(self) -> None:
        field = make_field(nonconstant_truth=False)
        rows = [{
            "total_length": 600,
            "seen_during_training": True,
            "used_for_checkpoint_selection": False,
            "normalized_space_mse": 0.1,
            "raw_global_relative_l2": 0.2,
            "raw_mean_per_q_relative_l2": 0.3,
            "frozen_forward_passes": 1,
        }]
        summary = diagnostic.build_summary(
            task_name="synthetic_t1200",
            checkpoint_path=Path("synthetic.pt"),
            checkpoint=make_checkpoint(),
            field=field,
            requested_lengths=(600,),
            training_lengths=(600, 800, 1000, 1200),
            validation_lengths=(700, 900, 1100, 1200),
            normalization_stats=make_stats(),
            target_transform=TargetTransformConfig(mode="raw"),
            rows=rows,
            device="cpu",
        )
        self.assertEqual(summary["scientific_scope"], "development_diagnostic")
        self.assertFalse(summary["formal_long_domain_extrapolation_result"])
        self.assertTrue(summary["frozen_inference_only"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "result"
            diagnostic.write_outputs(output_dir=output_directory, summary=summary, rows=rows)
            self.assertEqual({path.name for path in output_directory.iterdir()}, set(diagnostic.OUTPUT_FILENAMES))
            with (output_directory / diagnostic.OUTPUT_FILENAMES[0]).open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["status"], "completed")
            with (output_directory / diagnostic.OUTPUT_FILENAMES[1]).open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                diagnostic.write_outputs(output_dir=output_directory, summary=summary, rows=rows)
            self.assertFalse(any(path.suffix in {".npy", ".pt"} for path in output_directory.iterdir()))

    def test_r1_checkpoint_schema_restores_with_existing_helper(self) -> None:
        checkpoint = make_checkpoint()
        model = build_model_2d(**make_model_config())
        checkpoint["model_state_dict"] = model.state_dict()
        from scripts.run_analysis_2d import load_fno2d_checkpoint_model

        restored_model = load_fno2d_checkpoint_model(checkpoint, "cpu")
        self.assertEqual(restored_model.in_dim, 2)
        self.assertEqual(restored_model.out_dim, 3)

    def test_cli_and_source_contain_no_training_or_long_domain_paths(self) -> None:
        arguments = diagnostic.parse_args([
            "--task-name", "synthetic_t1200",
            "--checkpoint-path", "synthetic.pt",
            "--output-dir", "output",
        ])
        self.assertEqual(tuple(arguments.lengths), diagnostic.DEFAULT_LENGTHS)
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("scheduler.step", source)
        self.assertNotIn("--long-task-name", source)
        self.assertNotIn("--medium-task-name", source)
        self.assertIn("x_val", source)
        self.assertIn("y_val", source)


if __name__ == "__main__":
    unittest.main()
