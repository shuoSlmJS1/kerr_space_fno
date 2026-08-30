from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

import scripts.train_fno2d_domain_conditioned_r2 as r2
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
        self.widths: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.widths.append(int(x.shape[-2]))
        return self.scale * x


class CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def make_lambda_grid(length: int = r2.SOURCE_LENGTH) -> np.ndarray:
    return np.arange(length, dtype=np.float64) * 0.005


def make_split(q_values: tuple[float, ...] = (3.0, 1.0, 2.0)) -> r1.CanonicalSplit:
    q = np.asarray(q_values, dtype=np.float64)[:, None]
    truth = np.empty((len(q_values), r2.SOURCE_LENGTH, 3), dtype=np.float32)
    for row, value in enumerate(q_values):
        truth[row, :, 0] = value
        truth[row, :, 1] = np.arange(r2.SOURCE_LENGTH, dtype=np.float32)
        truth[row, :, 2] = value + truth[row, :, 1]
    return r1.canonicalize_split(q, truth)


def make_stats() -> FieldNormalizationStats:
    return FieldNormalizationStats(
        method="standard",
        x_mean=[2.0, 0.0, 0.0],
        x_std=[0.5, 1.0, 1.0],
        y_mean=[0.0, 0.0, 0.0],
        y_std=[1.0, 1.0, 1.0],
    )


def make_views(lengths: tuple[int, ...]) -> dict[int, r2.PrefixView]:
    views: dict[int, r2.PrefixView] = {}
    for length in lengths:
        x = torch.ones((1, 2, length, 3), dtype=torch.float32)
        y = 2.0 * x
        views[length] = r2.PrefixView(length=length, x=x, y=y)
    return views


class TestR2CoordinatesAndDataProtocol(unittest.TestCase):
    def test_strict_prefix_construction_is_preserved(self) -> None:
        source = np.arange(2 * 12 * 3).reshape(2, 12, 3)
        prefix = r1.construct_strict_prefix(source, length=7, lambda_axis=1)
        self.assertTrue(np.array_equal(prefix, source[:, :7, :]))

    def test_canonical_q_and_split_identity_are_reused(self) -> None:
        split = make_split()
        self.assertEqual(split.q[:, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(split.source_row_indices.tolist(), [1, 2, 0])
        other = r1.canonicalize_split(
            np.array([[4.0], [5.0]]), np.zeros((2, r2.SOURCE_LENGTH, 3))
        )
        r1.assert_disjoint_q_identities({"train": split, "val": other})

    def test_dft_domain_length_and_reference_are_grid_derived(self) -> None:
        spec = r2.build_domain_coordinate_spec(make_lambda_grid())
        self.assertAlmostEqual(spec.delta_lambda, 0.005)
        self.assertAlmostEqual(spec.reference_domain_length, 6.0)
        self.assertAlmostEqual(r2.domain_length_for_prefix(600, spec), 3.0)
        self.assertAlmostEqual(r2.domain_length_for_prefix(1200, spec), 6.0)

    def test_nonuniform_grid_is_rejected(self) -> None:
        grid = make_lambda_grid()
        grid[100] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "uniformly"):
            r2.build_domain_coordinate_spec(grid)

    def test_relative_coordinate_and_broadcast_domain_channel(self) -> None:
        spec = r2.build_domain_coordinate_spec(make_lambda_grid())
        x = r2.build_domain_conditioned_input_field(
            np.array([[1.0], [2.0]], dtype=np.float64),
            make_lambda_grid()[:600],
            spec,
        )
        self.assertEqual(x.shape, (1, 2, 600, 3))
        self.assertTrue(np.allclose(x[0, :, 0, 0], [1.0, 2.0]))
        self.assertEqual(float(x[0, 0, 0, 1]), 0.0)
        self.assertAlmostEqual(float(x[0, 0, -1, 1]), 599.0 / 600.0, places=6)
        self.assertTrue(np.allclose(x[..., 2], 0.5))

    def test_primary_input_has_q_s_ell_and_no_absolute_lambda(self) -> None:
        spec = r2.build_domain_coordinate_spec(make_lambda_grid())
        x = r2.build_domain_conditioned_input_field(
            np.array([[1.0]], dtype=np.float64), make_lambda_grid()[:4], spec
        )
        self.assertEqual(r2.INPUT_CHANNEL_NAMES, ["Q", "s", "ell"])
        self.assertEqual(x.shape[-1], 3)
        self.assertNotEqual(float(x[0, 0, -1, 1]), float(make_lambda_grid()[3]))

    def test_raw_prefix_field_uses_strict_truth_prefix(self) -> None:
        split = make_split((1.0, 2.0))
        grid = make_lambda_grid()
        spec = r2.build_domain_coordinate_spec(grid)
        x, y = r2.build_raw_prefix_field(
            split, grid, 700, spec, TargetTransformConfig(mode="raw")
        )
        self.assertEqual(x.shape, (1, 2, 700, 3))
        self.assertEqual(y.shape, (1, 2, 700, 3))
        self.assertTrue(np.array_equal(y[0], split.truth[:, :700, :]))

    def test_length_protocol_is_r1_equivalent(self) -> None:
        train, validation = r1.validate_length_protocol(
            [600, 800, 1000, 1200], [700, 900, 1100, 1200], r2.SOURCE_LENGTH
        )
        self.assertEqual(train, (600, 800, 1000, 1200))
        self.assertEqual(validation, (700, 900, 1100, 1200))


class TestR2NormalizationAndArchitecture(unittest.TestCase):
    def test_q_and_target_stats_use_full_source_only(self) -> None:
        split = make_split()
        grid = make_lambda_grid()
        spec = r2.build_domain_coordinate_spec(grid)
        stats, policy = r2.fit_r2_normalization(
            split, grid, spec, TargetTransformConfig(mode="raw")
        )
        expected_q_mean = float(np.mean(split.q[:, 0]))
        self.assertAlmostEqual(stats.x_mean[0], expected_q_mean, places=6)
        self.assertEqual(stats.x_mean[1:], [0.0, 0.0])
        self.assertEqual(stats.x_std[1:], [1.0, 1.0])
        self.assertEqual(policy["s"], "identity_dimensionless")
        self.assertEqual(policy["ell"], "identity_dimensionless_L_over_L_ref")
        self.assertFalse(policy["fit_uses_validation_lengths"])

    def test_prefixes_reuse_same_stats_but_rebuild_s_and_ell(self) -> None:
        split = make_split((1.0, 2.0))
        grid = make_lambda_grid()
        spec = r2.build_domain_coordinate_spec(grid)
        stats = make_stats()
        views = r2.build_prefix_views(
            split,
            grid,
            [600, 1200],
            spec,
            stats,
            TargetTransformConfig(mode="raw"),
            torch.device("cpu"),
        )
        self.assertEqual(tuple(views[600].x.shape), (1, 2, 600, 3))
        self.assertEqual(tuple(views[1200].x.shape), (1, 2, 1200, 3))
        self.assertAlmostEqual(float(views[600].x[0, 0, -1, 1]), 599.0 / 600.0)
        self.assertAlmostEqual(float(views[1200].x[0, 0, -1, 1]), 1199.0 / 1200.0)
        self.assertAlmostEqual(float(views[600].x[0, 0, 0, 2]), 0.5)
        self.assertAlmostEqual(float(views[1200].x[0, 0, 0, 2]), 1.0)

    def test_target_transform_remains_raw(self) -> None:
        truth = np.random.default_rng(3).normal(size=(1, 2, 8, 3)).astype(np.float32)
        transformed = r2.transform_output_field(
            truth, TargetTransformConfig(mode=r2.TARGET_TRANSFORM_MODE)
        )
        self.assertTrue(np.array_equal(transformed, truth))

    def test_only_input_projection_dimension_changes(self) -> None:
        r2.assert_r2_architecture(r2.R2_MODEL_CONFIG)
        for key, value in r1.BASELINE_MODEL_CONFIG.items():
            if key != "in_dim":
                self.assertEqual(r2.R2_MODEL_CONFIG[key], value)
        self.assertEqual(r2.R2_MODEL_CONFIG["in_dim"], 3)
        changed = dict(r2.R2_MODEL_CONFIG)
        changed["modes2"] = 48
        with self.assertRaisesRegex(ValueError, "baseline"):
            r2.assert_r2_architecture(changed)

    def test_r3_path_is_explicitly_absent(self) -> None:
        r2.assert_r3_isolation(r2.R2_MODEL_CONFIG)
        config = r2.build_r2_config(
            task_name="q_source",
            run_name="r2_run",
            epochs=500,
            train_lengths=(600, 800, 1000, 1200),
            validation_lengths=(700, 900, 1100, 1200),
            optimizer_config={"name": "AdamW"},
            scheduler_config={"name": "ExponentialLR"},
            training_seed=27,
            device=torch.device("cpu"),
            normalization_stats=make_stats(),
            normalization_policy={"Q": "standard"},
            coordinate_spec=r2.build_domain_coordinate_spec(make_lambda_grid()),
            param_name="Q",
            data_seed=11,
        )
        self.assertTrue(config["spectral_parameterization_unchanged"])
        self.assertTrue(config["discrete_mode_index_weights"])
        self.assertFalse(config["physical_frequency_conditioning"])
        self.assertFalse(config["frequency_interpolation"])
        self.assertFalse(config["dynamic_spectral_weights"])


class TestR2OptimizationAndProvenance(unittest.TestCase):
    def test_equal_weight_loss_and_one_step_after_all_lengths(self) -> None:
        self.assertEqual(float(r1.equal_weight_loss([torch.tensor(1.0), torch.tensor(3.0)])), 2.0)
        model = ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_step = optimizer.step
        optimizer.step = mock.Mock(wraps=original_step)  # type: ignore[method-assign]
        metrics = r1.accumulate_multi_length_update(model, optimizer, make_views((3, 5, 7, 9)))
        self.assertEqual(model.forward_calls, 4)
        self.assertEqual(model.widths, [3, 5, 7, 9])
        self.assertEqual(optimizer.step.call_count, 1)  # type: ignore[attr-defined]
        self.assertEqual(metrics["forward_backward_passes"], 4)

    def test_scheduler_update_is_not_multiplied_by_lengths(self) -> None:
        model = ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CountingScheduler()
        metrics = r1.run_training_epoch(model, optimizer, scheduler, make_views((3, 5, 7, 9)))
        self.assertEqual(scheduler.step_calls, 1)
        self.assertEqual(metrics["scheduler_steps"], 1)

    def test_validation_selection_is_equal_mean_and_keeps_t1200_visible(self) -> None:
        score = r1.composite_validation_score({700: 1.0, 900: 2.0, 1100: 3.0, 1200: 6.0})
        self.assertEqual(score, 3.0)
        metrics = r1.format_length_metrics("val_mse", {700: 1.0, 1200: 2.0})
        self.assertEqual(metrics["val_mse_T1200"], 2.0)

    def test_config_records_coordinate_and_training_controls(self) -> None:
        spec = r2.build_domain_coordinate_spec(make_lambda_grid())
        config = r2.build_r2_config(
            task_name="q_source",
            run_name="r2_run",
            epochs=500,
            train_lengths=(600, 800, 1000, 1200),
            validation_lengths=(700, 900, 1100, 1200),
            optimizer_config={"name": "AdamW", "lr": 1e-3},
            scheduler_config={"name": "ExponentialLR", "gamma": 0.995},
            training_seed=27,
            device=torch.device("cpu"),
            normalization_stats=make_stats(),
            normalization_policy={"Q": "standard"},
            coordinate_spec=spec,
            param_name="Q",
            data_seed=11,
        )
        self.assertEqual(config["experiment_type"], r2.EXPERIMENT_TYPE)
        self.assertEqual(config["repair_class"], r2.REPAIR_CLASS)
        self.assertEqual(config["coordinate_representation"], ["Q", "s", "ell"])
        self.assertEqual(config["domain_length_definition"], "L = N * delta_lambda (DFT logical period)")
        self.assertAlmostEqual(config["L_ref"], 6.0)
        self.assertFalse(config["absolute_lambda_input"])
        self.assertTrue(config["architecture_unchanged_after_input_projection"])
        self.assertTrue(config["formal_long_test_lengths_excluded_from_training"])
        self.assertFalse(config["existing_formal_a1_evaluator_compatible"])
        self.assertEqual(config["forward_backward_passes_per_step"], 4)
        self.assertFalse(config["consistency_loss"])

    def test_cli_has_no_long_domain_arguments(self) -> None:
        args = r2.parse_args(["--task-name", "q_source"])
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.train_lengths, [600, 800, 1000, 1200])
        self.assertEqual(args.validation_lengths, [700, 900, 1100, 1200])
        self.assertFalse(hasattr(args, "medium_task_name"))
        self.assertFalse(hasattr(args, "long_task_name"))

    def test_output_directory_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(r2, "get_model_output_dir", return_value=Path(temp_dir)):
                with self.assertRaises(FileExistsError):
                    r2.refuse_existing_run("task", "run")

    def test_run_name_is_isolated(self) -> None:
        self.assertEqual(
            r2.derive_default_run_name(500, (1200, 600, 1000, 800)),
            "fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200",
        )

    def test_checkpoint_schema_restores_three_channel_model(self) -> None:
        tiny_model_config = {
            "model_type": "fno2d",
            "in_dim": 3,
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
            save_checkpoint_2d(path, model, optimizer, 1, 0.5, checkpoint_config)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        restored = load_fno2d_checkpoint_model(checkpoint, device=torch.device("cpu"))
        stats = load_normalization_stats_from_checkpoint(checkpoint)
        target = load_target_transform_config_from_checkpoint(checkpoint)
        self.assertEqual(type(restored), type(model))
        self.assertEqual(restored.in_dim, 3)
        self.assertEqual(stats.to_dict(), make_stats().to_dict())
        self.assertEqual(target.mode, "raw")


if __name__ == "__main__":
    unittest.main()
