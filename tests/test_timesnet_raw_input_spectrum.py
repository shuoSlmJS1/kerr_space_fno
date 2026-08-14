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

from scripts.analyze_timesnet_frequency_diagnostics import build_diagnostic_loader
from scripts.analyze_timesnet_raw_input_spectrum import (
    aggregate_input_projection_spectrum,
    aggregate_raw_input_spectrum,
    build_channel_spectrum_summary,
    run_raw_input_spectrum,
)
from src.models.timesnet1d import build_timesnet1d_model
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    fit_reconstruction_normalization,
    load_reconstruction_splits,
)


class TestTimesNetRawInputSpectrum(unittest.TestCase):
    """Test read-only raw TimesNet input spectrum diagnostics."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix="timesnet_raw_input_spectrum_",
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
        model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=2, top_k=2)
        torch.save(
            {"model_state_dict": model.state_dict()},
            self.run_dir / "checkpoints" / "best_model.pt",
        )
        self.run_config = {
            "experiment_type": "sparse_trajectory_reconstruction_timesnet1d",
            "dataset_path": str(self.dataset_path.resolve()),
            "q_input": "excluded",
            "input_channel_names": [
                "sparse_x",
                "sparse_y",
                "sparse_z",
                "observed_mask",
                "lambda_coordinate",
            ],
            "input_shape_per_sample": [17, 5],
            "sampling": splits.train.sampling.to_dict(),
            "normalization": normalization.to_dict(),
            "training": {"batch_size": 2},
            "model": model.architecture_metadata(),
        }
        (self.run_dir / "run_config.json").write_text(
            json.dumps(self.run_config),
            encoding="utf-8",
            newline="\n",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _args(self, output_name: str = "raw_spectrum.json", **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "dataset_path": self.dataset_path,
            "run_dir": self.run_dir,
            "split": "test",
            "output_json": self.root / output_name,
            "batch_size": None,
            "device": "cpu",
            "top_k_raw": 5,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _loader_from_inputs(inputs: torch.Tensor, batch_size: int) -> DataLoader:
        return DataLoader(
            TensorDataset(inputs, torch.zeros(inputs.shape[0])),
            batch_size=batch_size,
            shuffle=False,
        )

    def test_exact_five_channel_input_reuses_saved_normalization(self) -> None:
        """Recreate normalized sparse inputs without Q or refitting statistics."""
        loader, _, _ = build_diagnostic_loader(
            self.dataset_path,
            self.run_config,
            "test",
            None,
        )
        model_input = next(iter(loader))[0].numpy()
        splits = load_reconstruction_splits(self.dataset_path, stride=4)
        observed = splits.test.observed_mask[..., 0]
        expected = (
            splits.test.sparse_xyz[observed].astype(np.float32)
            - np.asarray(self.run_config["normalization"]["input_xyz_mean"], dtype=np.float32)
        ) / np.asarray(self.run_config["normalization"]["input_xyz_std"], dtype=np.float32)
        np.testing.assert_allclose(model_input[..., :3][observed[:2]], expected[: np.count_nonzero(observed[:2])])
        self.assertEqual(model_input.shape[-1], 5)
        self.assertTrue(np.array_equal(model_input[..., 3], splits.test.observed_mask[:2, :, 0].astype(np.float32)))

    def test_time_fft_excludes_dc_and_ranks_controlled_sinusoid(self) -> None:
        """Rank a known temporal sinusoid while excluding the DC frequency bin."""
        time = torch.arange(32, dtype=torch.float32)
        inputs = torch.zeros((2, 32, 5), dtype=torch.float32)
        inputs[..., 0] = 7.0 + torch.sin(2.0 * torch.pi * 4.0 * time / 32.0)
        amplitude, count, sequence_length = aggregate_raw_input_spectrum(
            self._loader_from_inputs(inputs, batch_size=1),
            "cpu",
        )
        summary = build_channel_spectrum_summary(amplitude[:, 0], sequence_length, 4, 3)
        self.assertEqual(count, 2)
        self.assertEqual(summary["top_nonzero_frequencies"][0]["frequency_index"], 4)
        self.assertEqual(summary["top_nonzero_frequencies"][0]["integer_period"], 8)
        self.assertNotEqual(summary["top_nonzero_frequencies"][0]["frequency_index"], 0)

    def test_periodic_mask_has_sampling_related_peak(self) -> None:
        """Expose the expected nonzero spectral peak of a controlled periodic mask."""
        inputs = torch.zeros((1, 32, 5), dtype=torch.float32)
        inputs[:, ::4, 3] = 1.0
        amplitude, _, sequence_length = aggregate_raw_input_spectrum(
            self._loader_from_inputs(inputs, batch_size=1),
            "cpu",
        )
        summary = build_channel_spectrum_summary(amplitude[:, 3], sequence_length, 4, 3)
        probe = summary["probe_frequencies"]["sampling_stride_related"]["exact_stride"]
        expected = [record for record in probe["frequencies_mapping_to_period"] if record["frequency_index"] == 8]
        self.assertEqual(len(expected), 1)
        self.assertEqual(expected[0]["integer_period"], 4)
        self.assertAlmostEqual(expected[0]["mean_amplitude"], float(amplitude[1:, 3].max()))

    def test_lambda_ramp_spectrum_is_finite(self) -> None:
        """Keep the normalized lambda channel spectrum finite and serializable."""
        lambda_values = torch.linspace(0.0, 1.0, 17)
        inputs = torch.zeros((3, 17, 5), dtype=torch.float32)
        inputs[..., 4] = lambda_values
        amplitude, _, sequence_length = aggregate_raw_input_spectrum(
            self._loader_from_inputs(inputs, batch_size=2),
            "cpu",
        )
        summary = build_channel_spectrum_summary(amplitude[:, 4], sequence_length, 4, 5)
        self.assertTrue(torch.isfinite(amplitude).all())
        self.assertTrue(all(record["mean_amplitude"] >= 0.0 for record in summary["top_nonzero_frequencies"]))

    def test_full_split_aggregation_is_batch_partition_independent(self) -> None:
        """Aggregate per-trajectory amplitudes independently of loader partitioning."""
        inputs = torch.randn(5, 19, 5)
        first, first_count, first_length = aggregate_raw_input_spectrum(
            self._loader_from_inputs(inputs, batch_size=1),
            "cpu",
        )
        second, second_count, second_length = aggregate_raw_input_spectrum(
            self._loader_from_inputs(inputs, batch_size=3),
            "cpu",
        )
        self.assertEqual((first_count, first_length), (second_count, second_length))
        self.assertTrue(torch.allclose(first, second, rtol=0.0, atol=1e-12))

    def test_dataset_mismatch_and_output_overwrite_are_rejected(self) -> None:
        """Reject inconsistent dataset paths and pre-existing diagnostic JSON files."""
        other_dataset = self.root / "other_dataset.npz"
        other_dataset.write_bytes(self.dataset_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "must match the dataset_path"):
            run_raw_input_spectrum(self._args("mismatch.json", dataset_path=other_dataset))
        run_raw_input_spectrum(self._args())
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            run_raw_input_spectrum(self._args())

    def test_projection_spectrum_is_read_only(self) -> None:
        """Inspect only the trained input projection without changing model parameters."""
        from scripts.analyze_timesnet_frequency_diagnostics import load_model_from_run

        model, config = load_model_from_run(self.run_dir, "cpu")
        loader, _, _ = build_diagnostic_loader(self.dataset_path, config, "test", None)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        amplitude, count, sequence_length, feature_count = aggregate_input_projection_spectrum(model, loader, "cpu")
        self.assertEqual((count, sequence_length, feature_count), (3, 17, 8))
        self.assertTrue(torch.isfinite(amplitude).all())
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_cli_writes_compact_json_without_targets_and_rejects_overwrite(self) -> None:
        """Run the raw-spectrum command line with a synthetic completed run."""
        output_path = self.root / "cli_raw_spectrum.json"
        command = [
            sys.executable,
            "-B",
            "scripts/analyze_timesnet_raw_input_spectrum.py",
            "--dataset-path",
            str(self.dataset_path),
            "--run-dir",
            str(self.run_dir),
            "--split",
            "test",
            "--output-json",
            str(output_path),
            "--top-k-raw",
            "5",
        ]
        first = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(set(saved["channels"]), {"sparse_x", "sparse_y", "sparse_z", "observed_mask", "lambda_coordinate"})
        serialized = json.dumps(saved).lower()
        self.assertNotIn("target_xyz", serialized)
        self.assertNotIn("target_raw", serialized)
        self.assertNotIn("target_normalized", serialized)
        before = output_path.read_bytes()
        second = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(before, output_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
