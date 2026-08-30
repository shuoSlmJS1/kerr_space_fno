from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import train_fno2d_domain_conditioned_r2 as r2
from scripts import train_fno2d_physical_frequency_r3 as r3
from scripts import train_fno2d_variable_length_r1 as r1
from src.models.fno2d.fno2d import build_fno2d_model, count_parameters as baseline_count
from src.models.fno2d.fno2d_physical_frequency import build_physical_frequency_fno2d_model, count_parameters
from src.models.fno2d.physical_frequency_layers2d import PhysicalFrequencySpectralConv2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_fno2d_physical_frequency_r3.py"


def anchors() -> np.ndarray:
    return r3.build_anchor_frequencies(train_lengths=(600, 800, 1000, 1200), delta_lambda=0.005, modes2=32)


class PhysicalFrequencyLayerTests(unittest.TestCase):
    def test_frequency_formula_and_matched_bins_share_weight(self) -> None:
        layer = PhysicalFrequencySpectralConv2d(1, 1, 1, 32, delta_lambda=0.005, anchor_frequencies=torch.from_numpy(anchors()))
        frequencies = [2 / (1200 * 0.005), 3 / (1800 * 0.005), 4 / (2400 * 0.005)]
        self.assertTrue(np.allclose(frequencies, frequencies[0]))
        interpolated = [layer.interpolated_weights("pos", torch.tensor([value], dtype=torch.float64)) for value in frequencies]
        self.assertTrue(torch.equal(interpolated[0], interpolated[1]))
        self.assertTrue(torch.equal(interpolated[1], interpolated[2]))
        self.assertEqual(float(layer.runtime_frequencies(1200, 1)[0]), 0.0)

    def test_exact_and_midpoint_complex_cartesian_interpolation(self) -> None:
        layer = PhysicalFrequencySpectralConv2d(1, 1, 1, 2, delta_lambda=0.5, anchor_frequencies=torch.tensor([0.0, 1.0]))
        with torch.no_grad():
            layer.weights_pos_anchor.copy_(torch.tensor([[[[1.0 + 2.0j, 5.0 + 6.0j]]]]))
            layer.weights_neg_anchor.copy_(torch.tensor([[[[7.0 + 8.0j, 9.0 + 10.0j]]]]))
        exact = layer.interpolated_weights("pos", torch.tensor([1.0], dtype=torch.float64))
        midpoint = layer.interpolated_weights("pos", torch.tensor([0.5], dtype=torch.float64))
        negative = layer.interpolated_weights("neg", torch.tensor([0.5], dtype=torch.float64))
        self.assertEqual(complex(exact.item()), complex(5.0 + 6.0j))
        self.assertEqual(complex(midpoint.item()), complex(3.0 + 4.0j))
        self.assertEqual(complex(negative.item()), complex(8.0 + 9.0j))

    def test_anchor_validator_accepts_float32_roundtrip_and_rejects_malformed_grids(self) -> None:
        canonical = np.linspace(0.0, 31 / (600 * 0.005), 32, dtype=np.float64)
        float32_roundtrip = canonical.astype(np.float32).tolist()
        PhysicalFrequencySpectralConv2d(1, 1, 1, 32, delta_lambda=0.005, anchor_frequencies=float32_roundtrip)
        duplicate = canonical.copy()
        duplicate[8] = duplicate[7]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PhysicalFrequencySpectralConv2d(1, 1, 1, 32, delta_lambda=0.005, anchor_frequencies=duplicate)
        nonmonotonic = canonical.copy()
        nonmonotonic[8] = nonmonotonic[7] - 0.1
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PhysicalFrequencySpectralConv2d(1, 1, 1, 32, delta_lambda=0.005, anchor_frequencies=nonmonotonic)
        nonuniform = canonical.copy()
        nonuniform[16] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            PhysicalFrequencySpectralConv2d(1, 1, 1, 32, delta_lambda=0.005, anchor_frequencies=nonuniform)

    def test_fixed_discrete_policy_support_and_gradients(self) -> None:
        layer = PhysicalFrequencySpectralConv2d(2, 2, 2, 4, delta_lambda=0.5, anchor_frequencies=torch.linspace(0.0, 0.5, 4, dtype=torch.float64))
        self.assertEqual(layer.validate_runtime_support(16).numel(), 4)
        with self.assertRaisesRegex(ValueError, "outside learned anchor support"):
            layer.validate_runtime_support(2)
        output = layer(torch.randn(1, 2, 3, 16))
        output.square().mean().backward()
        self.assertIsNotNone(layer.weights_pos_anchor.grad)
        self.assertIsNotNone(layer.weights_neg_anchor.grad)

    def test_shape_and_parameter_count_match_discrete_lambda_table(self) -> None:
        config = {"in_dim": 3, "out_dim": 3, "modes1": 2, "modes2": 4, "width": 6, "depth": 2, "hidden_dim": 8, "activation": "gelu"}
        physical = build_physical_frequency_fno2d_model(**config, delta_lambda=0.5, anchor_frequencies=torch.linspace(0.0, 3.0, 4))
        baseline = build_fno2d_model(**config)
        self.assertEqual(count_parameters(physical), baseline_count(baseline))
        self.assertEqual(tuple(physical(torch.randn(1, 3, 8, 3)).shape), (1, 3, 8, 3))


class R3TrainingProtocolTests(unittest.TestCase):
    def test_anchor_support_comes_only_from_shortest_training_length(self) -> None:
        values = anchors()
        self.assertEqual(values.size, 32)
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 31 / (600 * 0.005))
        self.assertTrue(np.allclose(np.diff(values), np.diff(values)[0]))

    def test_r2_coordinates_and_normalization_policy_are_preserved(self) -> None:
        grid = np.arange(1200, dtype=np.float64) * 0.005
        spec = r2.build_domain_coordinate_spec(grid)
        field = r2.build_domain_conditioned_input_field(np.array([[1.0], [2.0]]), grid[:600], spec)
        self.assertEqual(field.shape[-1], 3)
        self.assertTrue(np.allclose(field[0, :, :, 2], 0.5))
        config = r3.build_r3_model_config(delta_lambda=0.005, anchor_frequencies=anchors())
        self.assertEqual(config["in_dim"], 3)
        self.assertEqual(config["modes2"], 32)
        self.assertEqual(config["model_type"], "fno2d_physical_frequency")

    def test_one_update_after_all_length_forwards(self) -> None:
        model = build_physical_frequency_fno2d_model(in_dim=3, out_dim=3, modes1=1, modes2=2, width=4, depth=1, hidden_dim=8, activation="gelu", delta_lambda=0.5, anchor_frequencies=torch.tensor([0.0, 1.0]))
        views = {length: r1.PrefixView(length=length, x=torch.randn(1, 2, length, 3), y=torch.randn(1, 2, length, 3)) for length in (2, 3)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as stepped, mock.patch.object(scheduler, "step", wraps=scheduler.step) as scheduled:
            result = r1.run_training_epoch(model, optimizer, scheduler, views)
        self.assertEqual(result["forward_backward_passes"], 2)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(stepped.call_count, 1)
        self.assertEqual(scheduled.call_count, 1)

    def test_checkpoint_config_excludes_bandwidth_and_global_fft_repairs(self) -> None:
        grid = np.arange(1200, dtype=np.float64) * 0.005
        spec = r2.build_domain_coordinate_spec(grid)
        stats = r2.fit_r2_normalization(r1.CanonicalSplit(q=np.array([[1.0]]), truth=np.zeros((1, 1200, 3)), source_row_indices=np.array([0], dtype=np.int64)), grid, spec, r2.TargetTransformConfig(mode="raw"))[0]
        config = r3.build_r3_config(task_name="q_source", run_name="r3_run", epochs=2, train_lengths=(600, 800, 1000, 1200), validation_lengths=(700, 900, 1100, 1200), optimizer_config={}, scheduler_config={}, training_seed=1, device=torch.device("cpu"), normalization_stats=stats, normalization_policy={"Q": "standard_full_source_train_field", "s": "identity_dimensionless", "ell": "identity_dimensionless_L_over_L_ref", "target": "standard_full_source_train_field", "fit_uses_validation_lengths": False, "fit_uses_formal_long_lengths": False}, coordinate_spec=spec, param_name="Q", data_seed=1, model_config=r3.build_r3_model_config(delta_lambda=0.005, anchor_frequencies=anchors()))
        self.assertTrue(config["physical_frequency_conditioning"])
        self.assertFalse(config["physical_cutoff_repair"])
        self.assertFalse(config["physical_bandwidth_shrinkage_repaired"])
        self.assertTrue(config["global_fft_structure_unchanged"])
        self.assertFalse(config["hypernetwork"])
        self.assertTrue(np.array_equal(np.asarray(config["model_config"]["anchor_frequencies"], dtype=np.float64), anchors()))

    def test_cli_has_no_long_domain_arguments_and_help(self) -> None:
        args = r3.parse_args(["--task-name", "q_source"])
        self.assertEqual(args.train_lengths, [600, 800, 1000, 1200])
        self.assertFalse(hasattr(args, "long_task_name"))
        completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--train-lengths", completed.stdout)


if __name__ == "__main__":
    unittest.main()
