from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

import scripts.train_fno2d_variable_length_r1 as r1
from scripts.run_analysis_2d import (
    load_fno2d_checkpoint_model,
    load_normalization_stats_from_checkpoint,
    load_target_transform_config_from_checkpoint,
)
from scripts.train_model_2d import save_checkpoint_2d
from src.models.registry_2d import build_model_2d
from src.training.fno2d.normalization_2d import FieldNormalizationStats
from src.training.fno2d.target_transform_2d import TargetTransformConfig


class ScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.scale * x


class CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def make_views(lengths: tuple[int, ...]) -> dict[int, r1.PrefixView]:
    views: dict[int, r1.PrefixView] = {}
    for length in lengths:
        x = torch.ones((1, 2, length, 1), dtype=torch.float32)
        y = 2.0 * x
        views[length] = r1.PrefixView(length=length, x=x, y=y)
    return views


def make_stats() -> FieldNormalizationStats:
    return FieldNormalizationStats(
        method="standard",
        x_mean=[2.0, 3.0],
        x_std=[0.5, 1.5],
        y_mean=[0.0, 0.0, 0.0],
        y_std=[1.0, 1.0, 1.0],
    )


class TestR1DataProtocol(unittest.TestCase):
    def test_strict_prefix_construction(self) -> None:
        source = np.arange(2 * 12 * 3).reshape(2, 12, 3)
        prefix = r1.construct_strict_prefix(source, length=7, lambda_axis=1)
        self.assertTrue(np.array_equal(prefix, source[:, :7, :]))
        self.assertEqual(prefix.shape, (2, 7, 3))

    def test_canonical_q_is_stable_and_truth_follows_permutation(self) -> None:
        q = np.array([[2.0], [1.0], [3.0]], dtype=np.float64)
        truth = np.stack(
            [np.full((5, 3), value, dtype=np.float64) for value in (20, 10, 30)]
        )
        split = r1.canonicalize_split(q, truth)
        self.assertEqual(split.q[:, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(split.source_row_indices.tolist(), [1, 0, 2])
        self.assertTrue(np.all(split.truth[:, 0, 0] == np.array([10, 20, 30])))

    def test_q_split_identity_is_trajectory_level(self) -> None:
        truth = np.zeros((2, 4, 3), dtype=np.float64)
        splits = {
            "train": r1.canonicalize_split(np.array([[1.0], [2.0]]), truth),
            "val": r1.canonicalize_split(np.array([[3.0], [4.0]]), truth),
            "test": r1.canonicalize_split(np.array([[5.0], [6.0]]), truth),
        }
        r1.assert_disjoint_q_identities(splits)
        leaking = dict(splits)
        leaking["val"] = r1.canonicalize_split(np.array([[2.0], [4.0]]), truth)
        with self.assertRaisesRegex(ValueError, "Q identity leakage"):
            r1.assert_disjoint_q_identities(leaking)

    def test_duplicate_q_within_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            r1.canonicalize_split(
                np.array([[1.0], [1.0]]),
                np.zeros((2, 4, 3), dtype=np.float64),
            )

    def test_default_length_protocol(self) -> None:
        train, validation = r1.validate_length_protocol(
            [1200, 600, 1000, 800], [1200, 700, 1100, 900], 1200
        )
        self.assertEqual(train, (600, 800, 1000, 1200))
        self.assertEqual(validation, (700, 900, 1100, 1200))
        self.assertEqual(set(train).intersection(validation), {1200})

    def test_invalid_duplicate_or_excess_lengths_are_rejected(self) -> None:
        cases = [
            ([600, 600, 1200], [700, 1200]),
            ([0, 1200], [700, 1200]),
            ([600, 1300], [700, 1200]),
            ([600, 1000], [700, 1200]),
            ([600, 700, 1200], [700, 1200]),
        ]
        for train, validation in cases:
            with self.subTest(train=train, validation=validation):
                with self.assertRaises(ValueError):
                    r1.validate_length_protocol(train, validation, 1200)

    def test_only_t1200_source_is_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires one T1200"):
            r1.validate_length_protocol([600, 1000], [700, 1000], 1000)

    def test_normalization_is_fitted_from_full_source_train_field(self) -> None:
        q = np.linspace(1.6, 3.0, 4, dtype=np.float32)
        lam = np.arange(r1.SOURCE_LENGTH, dtype=np.float32) * 0.005
        q_field = np.broadcast_to(q[None, :, None], (1, 4, r1.SOURCE_LENGTH))
        l_field = np.broadcast_to(lam[None, None, :], (1, 4, r1.SOURCE_LENGTH))
        x = np.stack((q_field, l_field), axis=-1).copy()
        y = np.stack((q_field, l_field, q_field + l_field), axis=-1).copy()
        stats = r1.fit_full_source_normalization((x, y))
        self.assertEqual(stats.method, "standard")
        self.assertAlmostEqual(stats.x_mean[1], float(lam.mean()), places=5)
        short_only_mean = float(lam[:600].mean())
        self.assertNotAlmostEqual(stats.x_mean[1], short_only_mean, places=2)

    def test_shorter_prefixes_reuse_same_normalization_stats(self) -> None:
        x = np.zeros((1, 2, 12, 2), dtype=np.float32)
        x[..., 0] = 2.0
        x[..., 1] = np.arange(12, dtype=np.float32)
        y = np.zeros((1, 2, 12, 3), dtype=np.float32)
        stats = FieldNormalizationStats(
            method="standard",
            x_mean=[2.0, 5.5],
            x_std=[1.0, 3.5],
            y_mean=[0.0, 0.0, 0.0],
            y_std=[1.0, 1.0, 1.0],
        )
        views = r1.build_prefix_views((x, y), [6, 12], stats, torch.device("cpu"))
        self.assertAlmostEqual(float(views[6].x[0, 0, 0, 1]), -5.5 / 3.5)
        self.assertAlmostEqual(float(views[12].x[0, 0, 0, 1]), -5.5 / 3.5)

    def test_raw_target_transform_preserves_truth(self) -> None:
        truth = np.random.default_rng(2).normal(size=(1, 3, 9, 3)).astype(np.float32)
        transformed = r1.transform_output_field(
            truth, TargetTransformConfig(mode=r1.TARGET_TRANSFORM_MODE)
        )
        self.assertTrue(np.array_equal(transformed, truth))


class TestR1OptimizationProtocol(unittest.TestCase):
    def test_equal_length_weighting_is_not_grid_point_weighting(self) -> None:
        losses = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(8.0)]
        self.assertEqual(float(r1.equal_weight_loss(losses)), 4.0)

    def test_one_optimizer_step_after_all_lengths(self) -> None:
        model = ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_step = optimizer.step
        optimizer.step = mock.Mock(wraps=original_step)  # type: ignore[method-assign]
        metrics = r1.accumulate_multi_length_update(model, optimizer, make_views((3, 5, 7)))
        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(optimizer.step.call_count, 1)  # type: ignore[attr-defined]
        self.assertEqual(metrics["forward_backward_passes"], 3)
        self.assertEqual(metrics["optimizer_steps"], 1)

    def test_scheduler_steps_once_per_epoch(self) -> None:
        model = ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CountingScheduler()
        metrics = r1.run_training_epoch(
            model, optimizer, scheduler, make_views((3, 5, 7, 9))
        )
        self.assertEqual(scheduler.step_calls, 1)
        self.assertEqual(metrics["scheduler_steps"], 1)

    def test_each_length_is_forwarded_at_its_own_width(self) -> None:
        class WidthModel(ScaleModel):
            def __init__(self) -> None:
                super().__init__()
                self.widths: list[int] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.widths.append(int(x.shape[-2]))
                return super().forward(x)

        model = WidthModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        r1.accumulate_multi_length_update(model, optimizer, make_views((4, 6, 8)))
        self.assertEqual(model.widths, [4, 6, 8])

    def test_optimizer_step_count_equals_epoch_count(self) -> None:
        model = ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CountingScheduler()
        original_step = optimizer.step
        optimizer.step = mock.Mock(wraps=original_step)  # type: ignore[method-assign]
        for _ in range(4):
            r1.run_training_epoch(model, optimizer, scheduler, make_views((3, 5)))
        self.assertEqual(optimizer.step.call_count, 4)  # type: ignore[attr-defined]
        self.assertEqual(scheduler.step_calls, 4)


class TestR1ValidationAndProvenance(unittest.TestCase):
    def test_validation_score_is_equal_mean(self) -> None:
        score = r1.composite_validation_score({700: 1.0, 900: 2.0, 1200: 6.0})
        self.assertEqual(score, 3.0)

    def test_best_selection_uses_composite_score(self) -> None:
        self.assertTrue(r1.should_replace_best(0.9, 1.0))
        self.assertFalse(r1.should_replace_best(1.0, 1.0))

    def test_t1200_validation_metric_is_recorded_separately(self) -> None:
        metrics = r1.format_length_metrics("val_mse", {700: 2.0, 1200: 3.0})
        self.assertEqual(metrics["val_mse_T1200"], 3.0)
        self.assertEqual(metrics["val_mse_T700"], 2.0)

    def test_architecture_is_exact_baseline(self) -> None:
        r1.assert_architecture_unchanged(r1.BASELINE_MODEL_CONFIG)
        changed = dict(r1.BASELINE_MODEL_CONFIG)
        changed["modes2"] = 48
        with self.assertRaisesRegex(ValueError, "exactly match"):
            r1.assert_architecture_unchanged(changed)
        self.assertEqual(
            r1.BASELINE_MODEL_CONFIG,
            {
                "model_type": "fno2d",
                "in_dim": 2,
                "out_dim": 3,
                "modes1": 16,
                "modes2": 32,
                "width": 64,
                "depth": 4,
                "hidden_dim": 128,
                "activation": "gelu",
            },
        )

    def test_r1_config_records_required_controls(self) -> None:
        config = r1.build_r1_config(
            task_name="q_source",
            run_name="r1_run",
            epochs=500,
            train_lengths=(600, 800, 1000, 1200),
            validation_lengths=(700, 900, 1100, 1200),
            optimizer_config={"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
            scheduler_config={"name": "ExponentialLR", "gamma": 0.995},
            training_seed=27,
            device=torch.device("cpu"),
            normalization_stats=make_stats(),
            param_name="Q",
            data_seed=11,
        )
        self.assertEqual(config["experiment_type"], r1.EXPERIMENT_TYPE)
        self.assertEqual(config["repair_class"], r1.REPAIR_CLASS)
        self.assertEqual(config["normalization_fit_length"], 1200)
        self.assertEqual(config["normalization_source"], "full_source_train_field")
        self.assertTrue(config["architecture_unchanged"])
        self.assertEqual(config["coordinate_representation"], ["Q", "lambda"])
        self.assertTrue(config["equal_length_loss_weighting"])
        self.assertEqual(
            config["optimizer_step_semantics"], "one_step_after_all_train_lengths"
        )
        self.assertEqual(config["optimizer_steps"], 500)
        self.assertEqual(config["forward_backward_passes_per_step"], 4)
        self.assertTrue(config["formal_long_test_lengths_excluded_from_training"])
        self.assertFalse(config["compute_matched"])
        self.assertFalse(config["consistency_loss"])
        self.assertFalse(config["domain_length_channel"])
        self.assertFalse(config["relative_coordinate"])
        self.assertFalse(config["physical_frequency_conditioning"])
        self.assertEqual(config["dataset_summary"]["normalization_stats"], make_stats().to_dict())

    def test_training_cli_has_no_long_domain_task_arguments(self) -> None:
        args = r1.parse_args(["--task-name", "q_source"])
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.train_lengths, [600, 800, 1000, 1200])
        self.assertEqual(args.validation_lengths, [700, 900, 1100, 1200])
        self.assertEqual(args.epochs, 500)
        self.assertFalse(hasattr(args, "medium_task_name"))
        self.assertFalse(hasattr(args, "long_task_name"))

    def test_output_directory_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir)
            with mock.patch.object(r1, "get_model_output_dir", return_value=existing):
                with self.assertRaises(FileExistsError):
                    r1.refuse_existing_run("task", "run")

    def test_default_run_name_is_isolated(self) -> None:
        name = r1.derive_default_run_name(500, (1200, 600, 1000, 800))
        self.assertEqual(
            name,
            "fno2d_m16x32_w64_d4_e500_r1_multilen_t600-800-1000-1200",
        )

    def test_checkpoint_schema_is_restore_helper_compatible(self) -> None:
        tiny_model_config = {
            "model_type": "fno2d",
            "in_dim": 2,
            "out_dim": 3,
            "modes1": 1,
            "modes2": 2,
            "width": 4,
            "depth": 1,
            "hidden_dim": 8,
            "activation": "gelu",
        }
        model = build_model_2d(**tiny_model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint_config = {
            "model_type": "fno2d",
            "normalization": "standard",
            "target_transform": "raw",
            "lambda_reference_index": 0,
            "model_config": tiny_model_config,
            "dataset_summary": {
                "normalization_stats": make_stats().to_dict(),
                "target_transform_config": TargetTransformConfig(mode="raw").to_dict(),
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint_2d(
                path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                best_val_mse=0.5,
                config=checkpoint_config,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        restored = load_fno2d_checkpoint_model(checkpoint, device=torch.device("cpu"))
        restored_stats = load_normalization_stats_from_checkpoint(checkpoint)
        restored_target = load_target_transform_config_from_checkpoint(checkpoint)
        self.assertEqual(type(restored), type(model))
        self.assertEqual(restored_stats.to_dict(), make_stats().to_dict())
        self.assertEqual(restored_target.mode, "raw")


if __name__ == "__main__":
    unittest.main()
