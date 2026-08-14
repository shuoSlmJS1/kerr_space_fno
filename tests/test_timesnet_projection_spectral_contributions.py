from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.analyze_timesnet_frequency_diagnostics import build_diagnostic_loader, load_model_from_run
from scripts.analyze_timesnet_projection_spectral_contributions import (
    _select_report_frequencies,
    aggregate_projection_contributions,
    run_projection_spectral_contributions,
)
from src.models.timesnet1d import build_timesnet1d_model
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    fit_reconstruction_normalization,
    load_reconstruction_splits,
)


class TestTimesNetProjectionSpectralContributions(unittest.TestCase):
    """Test read-only TimesNet input-projection spectral contributions."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix="timesnet_projection_contributions_",
        )
        self.root = Path(self.temporary_directory.name)
        self.dataset_path = self.root / "synthetic_dataset.npz"
        lambda_grid = np.linspace(0.0, 1.0, 17, dtype=np.float32)

        def make_split(sample_count: int, offset: float) -> np.ndarray:
            return np.stack(
                [
                    np.stack(
                        (
                            np.sin(lambda_grid * (index + 1.0) + offset),
                            np.cos(2.0 * lambda_grid + offset + index),
                            np.square(lambda_grid) + offset + 0.1 * index,
                        ),
                        axis=-1,
                    )
                    for index in range(sample_count)
                ],
                axis=0,
            ).astype(np.float32)

        np.savez(
            self.dataset_path,
            y_train=make_split(4, 0.0),
            y_val=make_split(3, 0.2),
            y_test=make_split(3, 0.4),
            lambda_grid=lambda_grid,
        )
        self.run_dir = self.root / "run"
        (self.run_dir / "checkpoints").mkdir(parents=True)
        splits = load_reconstruction_splits(self.dataset_path, stride=4)
        normalization = fit_reconstruction_normalization(splits.train)
        self.model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=2, top_k=2)
        torch.save({"model_state_dict": self.model.state_dict()}, self.run_dir / "checkpoints" / "best_model.pt")
        self.run_config = {
            "experiment_type": "sparse_trajectory_reconstruction_timesnet1d",
            "dataset_path": str(self.dataset_path.resolve()),
            "q_input": "excluded",
            "input_channel_names": ["sparse_x", "sparse_y", "sparse_z", "observed_mask", "lambda_coordinate"],
            "input_shape_per_sample": [17, 5],
            "sampling": splits.train.sampling.to_dict(),
            "normalization": normalization.to_dict(),
            "training": {"batch_size": 2},
            "model": self.model.architecture_metadata(),
        }
        (self.run_dir / "run_config.json").write_text(json.dumps(self.run_config), encoding="utf-8", newline="\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _args(self, output_name: str = "contributions.json", **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "dataset_path": self.dataset_path,
            "run_dir": self.run_dir,
            "split": "test",
            "output_json": self.root / output_name,
            "batch_size": None,
            "device": "cpu",
            "top_k": 5,
            "latent_diagnostics_json": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _loader(inputs: torch.Tensor, batch_size: int) -> DataLoader:
        return DataLoader(TensorDataset(inputs, torch.zeros(inputs.shape[0])), batch_size=batch_size, shuffle=False)

    @staticmethod
    def _linear_model(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Module:
        model = torch.nn.Module()
        model.input_projection = torch.nn.Linear(5, weight.shape[0])
        with torch.no_grad():
            model.input_projection.weight.copy_(weight)
            model.input_projection.bias.copy_(bias)
        return model

    def test_exact_five_channel_input_reuses_saved_normalization_without_q(self) -> None:
        """Recreate the exact saved-normalization five-channel formal input without Q."""
        loader, _, _ = build_diagnostic_loader(self.dataset_path, self.run_config, "test", None)
        inputs = next(iter(loader))[0]
        self.assertEqual(tuple(inputs.shape[1:]), (17, 5))
        self.assertEqual(self.run_config["q_input"], "excluded")
        self.assertTrue(torch.isfinite(inputs).all())

    def test_controlled_linear_decomposition_reconstructs_nonzero_fft_and_excludes_bias(self) -> None:
        """Match analytically isolated channel contributions and exclude bias away from DC."""
        time = torch.arange(16, dtype=torch.float32)
        inputs = torch.zeros((1, 16, 5), dtype=torch.float32)
        inputs[..., 0] = torch.sin(2.0 * torch.pi * 3.0 * time / 16.0)
        weight = torch.zeros((2, 5), dtype=torch.float32)
        weight[0, 0] = 2.0
        bias = torch.tensor([7.0, -4.0], dtype=torch.float32)
        result = aggregate_projection_contributions(self._linear_model(weight, bias), self._loader(inputs, 1), "cpu")
        component = result["component_mean_magnitudes"]
        self.assertAlmostEqual(float(component[3, 0]), 16.0, places=5)
        self.assertEqual(float(component[:, 1:].sum()), 0.0)
        self.assertTrue(result["reconstruction_check"]["passed"])
        self.assertLess(result["reconstruction_check"]["max_nonzero_complex_reconstruction_abs_error"], 1e-5)

    def test_zero_channel_and_phase_cancellation_are_reported_without_forced_shares(self) -> None:
        """Keep a zero channel at zero and expose cancellation through separate magnitudes."""
        time = torch.arange(16, dtype=torch.float32)
        inputs = torch.zeros((1, 16, 5), dtype=torch.float32)
        signal = torch.sin(2.0 * torch.pi * 2.0 * time / 16.0)
        inputs[..., 0] = signal
        inputs[..., 1] = -signal
        weight = torch.zeros((1, 5), dtype=torch.float32)
        weight[0, 0] = 1.0
        weight[0, 1] = 1.0
        result = aggregate_projection_contributions(self._linear_model(weight, torch.zeros(1)), self._loader(inputs, 1), "cpu")
        component = result["component_mean_magnitudes"]
        combined = result["combined_mean_magnitudes"]
        self.assertGreater(float(component[2, 0]), 0.0)
        self.assertGreater(float(component[2, 1]), 0.0)
        self.assertEqual(float(component[2, 2]), 0.0)
        self.assertLess(float(combined[2]), 1e-5)

    def test_controlled_sinusoid_and_batch_partition_are_stable(self) -> None:
        """Locate a known sinusoid and keep aggregate diagnostics independent of batch partition."""
        time = torch.arange(32, dtype=torch.float32)
        inputs = torch.zeros((5, 32, 5), dtype=torch.float32)
        inputs[..., 4] = torch.sin(2.0 * torch.pi * 4.0 * time / 32.0)
        weight = torch.ones((3, 5), dtype=torch.float32)
        model = self._linear_model(weight, torch.zeros(3))
        first = aggregate_projection_contributions(model, self._loader(inputs, 2), "cpu")
        second = aggregate_projection_contributions(model, self._loader(inputs, 3), "cpu")
        self.assertEqual(int(torch.argmax(first["combined_mean_magnitudes"][1:]).item()) + 1, 4)
        self.assertTrue(torch.allclose(first["component_mean_magnitudes"], second["component_mean_magnitudes"], rtol=1e-12, atol=1e-12))
        self.assertTrue(torch.allclose(first["combined_mean_magnitudes"], second["combined_mean_magnitudes"], rtol=1e-12, atol=1e-12))

    def test_checkpoint_weights_are_used_and_parameters_remain_unchanged(self) -> None:
        """Use the saved checkpoint projection and leave every model parameter unchanged."""
        model, config = load_model_from_run(self.run_dir, "cpu")
        loader, _, _ = build_diagnostic_loader(self.dataset_path, config, "test", None)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = aggregate_projection_contributions(model, loader, "cpu")
        self.assertEqual(result["feature_count"], 8)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_report_frequencies_include_primary_sampling_and_latent_values(self) -> None:
        """Include primary frequencies, stride-related bins, and supplied latent selections."""
        magnitudes = torch.arange(17, dtype=torch.float64)
        frequencies, regions = _select_report_frequencies(magnitudes, 32, 4, 3, {7})
        self.assertTrue({1, 2, 3, 4, 5, 7}.issubset(frequencies))
        self.assertIn(8, regions["exact_stride"])
        self.assertIn(8, frequencies)

    def test_json_is_finite_compact_and_rejects_overwrite(self) -> None:
        """Write serializable diagnostics without targets and refuse a duplicate output path."""
        result = run_projection_spectral_contributions(self._args())
        output_path = self.root / "contributions.json"
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["num_trajectories"], 3)
        self.assertEqual(saved["channels"], ["sparse_x", "sparse_y", "sparse_z", "observed_mask", "lambda_coordinate"])
        serialized = json.dumps(saved).lower()
        self.assertNotIn("target_xyz", serialized)
        self.assertNotIn("prediction", serialized)
        self.assertTrue(np.isfinite(np.asarray(list(result["frequencies"].values())[0]["combined_projection_mean_magnitude"])))
        before = output_path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            run_projection_spectral_contributions(self._args())
        self.assertEqual(before, output_path.read_bytes())

    def test_cli_end_to_end_uses_checkpoint_and_writes_json(self) -> None:
        """Run the command line against a synthetic completed TimesNet run."""
        output_path = self.root / "cli_contributions.json"
        command = [
            sys.executable, "-B", "scripts/analyze_timesnet_projection_spectral_contributions.py",
            "--dataset-path", str(self.dataset_path), "--run-dir", str(self.run_dir),
            "--split", "test", "--output-json", str(output_path), "--top-k", "5",
        ]
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("completed", completed.stdout)
        self.assertTrue(json.loads(output_path.read_text(encoding="utf-8"))["reconstruction_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
