from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.fno1d.fno1d import build_fno1d_model
from src.models.resnet1d import build_dilated_resnet1d
from src.models.timesnet1d import build_timesnet1d_model
from src.training.trajectory_reconstruction.cross_resolution import (
    evaluate_frozen_cross_resolution_run,
    load_evaluation_sparse_data,
    load_frozen_reconstruction_run,
    save_cross_resolution_result,
)
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    fit_reconstruction_normalization,
    restore_observed_points_tensor,
)
from src.training.trajectory_reconstruction.sparse_sampling import (
    build_sparse_trajectory_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "evaluate_sparse_reconstruction_cross_resolution.py"


class TestCrossResolutionSparseReconstruction(unittest.TestCase):
    """Test frozen stride-to-stride sparse reconstruction evaluation."""

    def _make_dataset(self, root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        root.mkdir(parents=True, exist_ok=True)
        lambda_grid = np.linspace(0.0, 1.0, 65, dtype=np.float64)
        sample_offsets = np.arange(3, dtype=np.float64)[:, None]
        x = np.sin(4.0 * np.pi * lambda_grid)[None, :] + sample_offsets
        y = np.square(lambda_grid)[None, :] + 0.1 * sample_offsets
        z = np.cos(3.0 * np.pi * lambda_grid)[None, :] - 0.2 * sample_offsets
        test_target = np.stack((x, y, z), axis=-1)
        train_target = test_target[:2] + 0.25
        val_target = test_target[:1] - 0.15
        dataset_path = root / "dataset.npz"
        np.savez(
            dataset_path,
            y_train=train_target,
            y_val=val_target,
            y_test=test_target,
            lambda_grid=lambda_grid,
        )
        return dataset_path, train_target, test_target

    def _make_model_and_config(
        self,
        family: str,
        sequence_length: int,
    ) -> tuple[torch.nn.Module, dict[str, object]]:
        if family == "fno1d":
            model = build_fno1d_model(
                in_dim=5,
                out_dim=3,
                modes=4,
                width=8,
                depth=2,
            )
            model_config: dict[str, object] = {
                "family": "fno1d",
                "in_dim": 5,
                "out_dim": 3,
                "modes": 4,
                "width": 8,
                "depth": 2,
            }
        elif family == "dilated_resnet1d":
            model = build_dilated_resnet1d(
                in_dim=5,
                out_dim=3,
                width=8,
                blocks=2,
            )
            model_config = model.architecture_metadata()
        elif family == "timesnet1d":
            model = build_timesnet1d_model(
                in_dim=5,
                out_dim=3,
                d_model=8,
                d_ff=8,
                num_blocks=1,
                top_k=2,
            )
            model_config = model.architecture_metadata()
        else:
            raise ValueError(f"Unsupported test family: {family}")
        self.assertEqual(model_config["in_dim"], 5)
        self.assertEqual(model_config["out_dim"], 3)
        return model, model_config

    def _make_run(self, root: Path, family: str) -> tuple[Path, Path, np.ndarray]:
        dataset_path, train_target, test_target = self._make_dataset(root)
        lambda_grid = np.linspace(0.0, 1.0, 65, dtype=np.float64)
        train_data = build_sparse_trajectory_data(train_target, lambda_grid, stride=16)
        normalization = fit_reconstruction_normalization(train_data)
        model, model_config = self._make_model_and_config(family, sequence_length=65)
        run_dir = root / family
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "metrics").mkdir()
        experiment_types = {
            "fno1d": "sparse_trajectory_reconstruction_fno1d",
            "dilated_resnet1d": "sparse_trajectory_reconstruction_resnet1d",
            "timesnet1d": "sparse_trajectory_reconstruction_timesnet1d",
        }
        run_config: dict[str, object] = {
            "schema_version": "1.0",
            "experiment_type": experiment_types[family],
            "dataset_path": str(dataset_path.resolve()),
            "q_input": "excluded",
            "input_channel_names": [
                "sparse_x",
                "sparse_y",
                "sparse_z",
                "observed_mask",
                "lambda_coordinate",
            ],
            "input_shape_per_sample": [65, 5],
            "output_shape_per_sample": [65, 3],
            "sampling": train_data.sampling.to_dict(),
            "normalization": normalization.to_dict(),
            "checkpoint_selection": "validation_raw_hidden_only_overall_relative_l2",
            "training": {"batch_size": 2},
            "model": model_config,
            "split_sizes": {"train": 2, "val": 1, "test": 3},
        }
        with (run_dir / "run_config.json").open("x", encoding="utf-8", newline="\n") as output_file:
            json.dump(run_config, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
        torch.save(
            {"model_state_dict": model.state_dict(), "run_config": run_config},
            run_dir / "checkpoints" / "best_model.pt",
        )
        with (run_dir / "metrics" / "test_hidden_only_metrics.json").open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            json.dump(
                {
                    "split": "test",
                    "raw_hidden_only_metrics": {
                        "overall": {"relative_l2": 0.5}
                    },
                },
                output_file,
                indent=2,
                allow_nan=False,
            )
            output_file.write("\n")
        return run_dir, dataset_path, test_target

    def test_frozen_evaluation_supports_all_three_model_families(self) -> None:
        """Evaluate FNO, ResNet, and TimesNet without training or parameter changes."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for family in ("fno1d", "dilated_resnet1d", "timesnet1d"):
                with self.subTest(model_family=family):
                    run_dir, dataset_path, _ = self._make_run(root / family, family)
                    frozen_run = load_frozen_reconstruction_run(run_dir, device="cpu")
                    checkpoint_hash_before = hashlib.sha256(
                        frozen_run.checkpoint_path.read_bytes()
                    ).hexdigest()
                    before = {
                        name: parameter.detach().clone()
                        for name, parameter in frozen_run.model.named_parameters()
                    }
                    result = evaluate_frozen_cross_resolution_run(
                        frozen_run=frozen_run,
                        dataset_path=dataset_path,
                        evaluation_stride=32,
                    )
                    self.assertEqual(result["model_family"], family)
                    self.assertEqual(result["train_stride"], 16)
                    self.assertEqual(result["evaluation_stride"], 32)
                    self.assertEqual(result["num_trajectories"], 3)
                    self.assertEqual(
                        result["cross_resolution_metrics"]["raw_hidden_only_metrics"]["hidden_point_count"],
                        3 * 62,
                    )
                    self.assertTrue(np.isfinite(result["degradation_factor_relative_l2"]))
                    self.assertTrue(
                        all(
                            torch.equal(before[name], parameter)
                            for name, parameter in frozen_run.model.named_parameters()
                        )
                    )
                    self.assertTrue(
                        all(
                            not parameter.requires_grad and parameter.grad is None
                            for parameter in frozen_run.model.parameters()
                        )
                    )
                    self.assertEqual(
                        checkpoint_hash_before,
                        hashlib.sha256(frozen_run.checkpoint_path.read_bytes()).hexdigest(),
                    )

    def test_evaluation_rebuilds_stride32_sparse_input_with_original_normalization(self) -> None:
        """Use the stride32 mask while keeping the original stride16 normalization values."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir, dataset_path, test_target = self._make_run(root, "fno1d")
            frozen_run = load_frozen_reconstruction_run(run_dir)
            _, stride16_data = load_evaluation_sparse_data(dataset_path, "test", 16)
            _, stride32_data = load_evaluation_sparse_data(dataset_path, "test", 32)
            self.assertFalse(np.array_equal(stride16_data.observed_mask, stride32_data.observed_mask))
            self.assertEqual(stride16_data.sampling.observed_indices, (0, 16, 32, 48, 64))
            self.assertEqual(stride32_data.sampling.observed_indices, (0, 32, 64))
            self.assertTrue(np.array_equal(stride32_data.target_xyz, test_target))
            self.assertTrue(np.all(stride32_data.sparse_xyz[~stride32_data.observed_mask[..., 0]] == 0.0))
            result = evaluate_frozen_cross_resolution_run(
                frozen_run,
                dataset_path,
                evaluation_stride=32,
            )
            self.assertEqual(result["normalization_source"], "original_training_run")
            self.assertEqual(result["sampling"]["stride"], 32)
            self.assertEqual(result["sampling"]["hidden_point_count"], 62)
            prediction = torch.full((1, 65, 3), -7.0)
            sparse = torch.from_numpy(stride32_data.sparse_xyz[:1])
            observed_mask = torch.from_numpy(stride32_data.observed_mask[:1])
            restored = restore_observed_points_tensor(
                prediction,
                sparse,
                observed_mask,
            )
            observed = stride32_data.observed_mask[0, :, 0]
            self.assertTrue(
                torch.equal(
                    restored[0, observed, :],
                    sparse[0, observed, :],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored[0, ~observed, :],
                    prediction[0, ~observed, :],
                )
            )

    def test_timesnet_rejects_a_batch_size_different_from_its_recorded_protocol(self) -> None:
        """Reject a TimesNet batch-size override that changes batch-shared selection groups."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir, dataset_path, _ = self._make_run(root, "timesnet1d")
            frozen_run = load_frozen_reconstruction_run(run_dir)
            with self.assertRaisesRegex(ValueError, "must match"):
                evaluate_frozen_cross_resolution_run(
                    frozen_run,
                    dataset_path,
                    evaluation_stride=32,
                    batch_size=1,
                )

    def test_reference_metric_and_json_overwrite_protection_are_safe(self) -> None:
        """Read the saved reference metric and reject replacement of an existing result JSON."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir, dataset_path, _ = self._make_run(root, "fno1d")
            result = evaluate_frozen_cross_resolution_run(
                load_frozen_reconstruction_run(run_dir),
                dataset_path,
                evaluation_stride=32,
            )
            self.assertEqual(result["same_resolution_reference"]["hidden_relative_l2"], 0.5)
            expected_factor = (
                result["cross_resolution_metrics"]["raw_hidden_only_metrics"]["overall"]["relative_l2"]
                / 0.5
            )
            self.assertAlmostEqual(result["degradation_factor_relative_l2"], expected_factor)
            output_path = root / "cross_resolution.json"
            save_cross_resolution_result(output_path, result)
            original_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            with output_path.open("r", encoding="utf-8") as input_file:
                parsed = json.load(input_file)
            self.assertEqual(parsed["train_stride"], 16)
            self.assertEqual(parsed["evaluation_stride"], 32)
            self.assertTrue(np.isfinite(parsed["cross_resolution_metrics"]["raw_hidden_only_metrics"]["overall"]["relative_l2"]))
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                save_cross_resolution_result(output_path, result)
            self.assertEqual(original_hash, hashlib.sha256(output_path.read_bytes()).hexdigest())

    def test_missing_same_resolution_reference_is_recorded_without_guessing(self) -> None:
        """Keep the cross-resolution metric while marking a missing reference as unavailable."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir, dataset_path, _ = self._make_run(root, "fno1d")
            (run_dir / "metrics" / "test_hidden_only_metrics.json").unlink()
            result = evaluate_frozen_cross_resolution_run(
                load_frozen_reconstruction_run(run_dir),
                dataset_path,
                evaluation_stride=32,
            )
            self.assertFalse(result["same_resolution_reference"]["available"])
            self.assertIsNone(result["degradation_factor_relative_l2"])

    def test_cli_runs_for_all_families_and_preserves_existing_json_on_retry(self) -> None:
        """Run the actual CLI for every supported family and verify overwrite rejection."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for family in ("fno1d", "dilated_resnet1d", "timesnet1d"):
                with self.subTest(model_family=family):
                    run_dir, dataset_path, _ = self._make_run(root / family, family)
                    output_path = root / f"{family}.json"
                    command = [
                        sys.executable,
                        "-B",
                        str(CLI_PATH),
                        "--run-dir",
                        str(run_dir),
                        "--dataset-path",
                        str(dataset_path),
                        "--evaluation-stride",
                        "32",
                        "--split",
                        "test",
                        "--output-json",
                        str(output_path),
                        "--device",
                        "cpu",
                    ]
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("Cross-resolution sparse reconstruction evaluation completed.", completed.stdout)
                    with output_path.open("r", encoding="utf-8") as input_file:
                        result = json.load(input_file)
                    self.assertEqual(result["model_family"], family)
                    self.assertEqual(result["evaluation_stride"], 32)
                    original_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
                    repeated = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(repeated.returncode, 0)
                    self.assertIn("will not be overwritten", repeated.stderr)
                    self.assertEqual(original_hash, hashlib.sha256(output_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
