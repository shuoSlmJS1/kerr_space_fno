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
from torch.utils.data import SequentialSampler

from scripts.analyze_timesnet_frequency_diagnostics import (
    _summarize_values,
    build_diagnostic_loader,
    collect_model_diagnostics,
    load_model_from_run,
    run_diagnostics,
)
from src.models.timesnet1d import build_timesnet1d_model
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    fit_reconstruction_normalization,
    load_reconstruction_splits,
)


class TestTimesNetFrequencyDiagnostics(unittest.TestCase):
    """Test read-only TimesNet frequency and period diagnostics."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix="timesnet_frequency_diagnostics_",
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

    def _args(self, output_name: str = "diagnostics.json", **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "dataset_path": self.dataset_path,
            "run_dir": self.run_dir,
            "split": "test",
            "output_json": self.root / output_name,
            "device": "cpu",
            "batch_size": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_run_reconstructs_checkpoint_and_writes_finite_json(self) -> None:
        """Reconstruct a synthetic run and write compact serializable diagnostics."""
        result = run_diagnostics(self._args())
        output_path = self.root / "diagnostics.json"
        self.assertTrue(output_path.is_file())
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["split"], "test")
        self.assertEqual(saved["stride"], 4)
        self.assertEqual(saved["sequence_length"], 17)
        self.assertEqual(saved["batch_size"], 2)
        self.assertEqual(len(saved["blocks"]), 2)
        self.assertEqual(result["model"]["diagnostic_source"], "latent_timesblock_hidden_representation")
        self.assertNotIn("target", json.dumps(saved).lower())
        self.assertTrue(
            all(
                all(value > 0 for value in batch["selected_frequency_indices"])
                for block in saved["blocks"]
                for batch in block["batches"]
            )
        )

    def test_diagnostics_do_not_change_parameters_or_create_gradients(self) -> None:
        """Keep parameter values and gradients unchanged during diagnostic collection."""
        model, config = load_model_from_run(self.run_dir, "cpu")
        loader, _, _ = build_diagnostic_loader(
            self.dataset_path,
            config,
            "validation",
            None,
        )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        diagnostics = collect_model_diagnostics(model, loader, "cpu")
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(
            all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_aggregation_counts_cover_selection_slots_and_batch_presence(self) -> None:
        """Separate repeated selection slots from batches containing a value."""
        summary = _summarize_values([[3, 3], [3, 5], [5, 7]], top_k=2)
        self.assertEqual(summary["total_batches"], 3)
        self.assertEqual(summary["total_selection_slots"], 6)
        self.assertEqual(summary["values"]["3"]["selection_slot_count"], 3)
        self.assertEqual(summary["values"]["3"]["batch_presence_count"], 2)
        self.assertEqual(summary["values"]["5"]["batch_presence_count"], 2)

    def test_batch_mismatch_is_rejected_and_loader_is_sequential(self) -> None:
        """Require formal batch grouping and disable validation or test shuffling."""
        _, config = load_model_from_run(self.run_dir, "cpu")
        loader, _, _ = build_diagnostic_loader(
            self.dataset_path,
            config,
            "test",
            None,
        )
        self.assertIsInstance(loader.sampler, SequentialSampler)
        with self.assertRaisesRegex(ValueError, "must match the run_config batch_size"):
            build_diagnostic_loader(self.dataset_path, config, "test", 1)

    def test_dataset_path_mismatch_is_rejected(self) -> None:
        """Require the exact dataset path recorded by the completed run."""
        _, config = load_model_from_run(self.run_dir, "cpu")
        other_dataset_path = self.root / "other_dataset.npz"
        other_dataset_path.write_bytes(self.dataset_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "must match the dataset_path"):
            build_diagnostic_loader(other_dataset_path, config, "test", None)

    def test_output_overwrite_is_rejected(self) -> None:
        """Refuse a second diagnostic result at the same output path."""
        run_diagnostics(self._args())
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            run_diagnostics(self._args())

    def test_cli_writes_json_and_rejects_overwrite(self) -> None:
        """Run the diagnostics command line with a synthetic completed run."""
        output_path = self.root / "cli_diagnostics.json"
        command = [
            sys.executable,
            "-B",
            "scripts/analyze_timesnet_frequency_diagnostics.py",
            "--dataset-path",
            str(self.dataset_path),
            "--run-dir",
            str(self.run_dir),
            "--split",
            "test",
            "--output-json",
            str(output_path),
        ]
        project_root = Path(__file__).resolve().parents[1]
        first = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        before = output_path.read_bytes()
        second = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(before, output_path.read_bytes())

    def test_q_input_must_be_excluded(self) -> None:
        """Reject a run configuration that does not preserve the five-channel contract."""
        self.run_config["q_input"] = "included"
        (self.run_dir / "run_config.json").write_text(
            json.dumps(self.run_config),
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaisesRegex(ValueError, "q_input as excluded"):
            run_diagnostics(self._args("invalid_q.json"))


if __name__ == "__main__":
    unittest.main()
