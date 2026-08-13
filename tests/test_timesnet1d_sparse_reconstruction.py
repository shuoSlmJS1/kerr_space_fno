from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.models.timesnet1d import (
    DEFAULT_INCEPTION_KERNELS,
    TimesNetReconstruction1D,
    build_timesnet1d_model,
    count_trainable_parameters,
    select_dominant_frequencies,
)
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    build_reconstruction_model_input,
    fit_reconstruction_normalization,
    hidden_only_mse_loss,
    restore_observed_points_tensor,
)
from src.training.trajectory_reconstruction.sparse_sampling import (
    build_sparse_trajectory_data,
)


class TestTimesNet1DSparseReconstruction(unittest.TestCase):
    """Test the canonical TimesNet sparse reconstruction contract."""

    def setUp(self) -> None:
        lambda_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        base = np.stack((np.sin(lambda_grid), np.square(lambda_grid), np.cos(lambda_grid)), axis=-1)
        target = np.stack((base, base + 2.0), axis=0)
        self.sparse_data = build_sparse_trajectory_data(target, lambda_grid, stride=2)
        self.normalization = fit_reconstruction_normalization(self.sparse_data)

    def test_model_input_has_exact_five_channels_without_q(self) -> None:
        """Build the shared five-channel input without Q."""
        model_input = build_reconstruction_model_input(self.sparse_data, self.normalization)
        self.assertEqual(model_input.shape, (2, 9, 5))
        self.assertEqual(model_input.shape[-1], 5)

    def test_formal_model_has_expected_shape_and_parameter_count(self) -> None:
        """Match the frozen formal dimensions and parameter-count contract."""
        model = build_timesnet1d_model()
        self.assertEqual((model.d_model, model.d_ff, len(model.blocks)), (80, 96, 2))
        self.assertEqual(count_trainable_parameters(model), 1_077_299)
        self.assertEqual(model.architecture_metadata()["trainable_parameter_count"], 1_077_299)
        prediction = model(torch.randn(1, 1200, 5))
        self.assertEqual(tuple(prediction.shape), (1, 1200, 3))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_model_rejects_six_channel_input_and_short_sequence(self) -> None:
        """Reject an extra Q-like channel and sequences too short for rFFT selection."""
        model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=1)
        with self.assertRaisesRegex(ValueError, "expected in_dim=5"):
            model(torch.randn(1, 9, 6))
        with self.assertRaisesRegex(ValueError, "at least two time steps"):
            model(torch.randn(1, 1, 5))

    def test_backward_gradients_are_finite(self) -> None:
        """Produce finite gradients through the TimesNet blocks."""
        model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=2)
        loss = model(torch.randn(2, 17, 5)).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_fft_selection_uses_time_axis_and_excludes_zero_frequency(self) -> None:
        """Select nonzero temporal bins from a known temporal sinusoid."""
        time = torch.arange(32, dtype=torch.float32)
        signal = torch.sin(2.0 * torch.pi * 4.0 * time / 32.0)
        x = signal[None, :, None].repeat(2, 1, 3)
        frequencies, periods, weights = select_dominant_frequencies(x, top_k=1)
        self.assertEqual(frequencies.tolist(), [4])
        self.assertEqual(periods.tolist(), [8])
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

    def test_invalid_top_k_is_rejected(self) -> None:
        """Reject requests exceeding the available nonzero rFFT bins."""
        with self.assertRaisesRegex(ValueError, "selectable nonzero"):
            select_dominant_frequencies(torch.randn(1, 8, 3), top_k=5)

    def test_padding_crop_duplicate_periods_and_weights_are_valid(self) -> None:
        """Keep non-divisible outputs, duplicate periods, and weights valid."""
        model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=1, top_k=2)
        x = torch.randn(2, 17, 5)
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (2, 17, 3))
        hidden = model.input_projection(x)
        _, periods, weights = select_dominant_frequencies(hidden, top_k=2)
        self.assertTrue(torch.all(periods >= 1))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2)))
        time = torch.arange(10, dtype=torch.float32)
        duplicate_signal = (
            8.0 * torch.sin(2.0 * torch.pi * 4.0 * time / 10.0)
            + torch.cos(torch.pi * time)
        )
        duplicate_input = duplicate_signal[None, :, None].repeat(1, 1, 3)
        frequencies, duplicate_periods, _ = select_dominant_frequencies(
            duplicate_input,
            top_k=2,
        )
        self.assertEqual(frequencies.tolist(), [4, 5])
        self.assertEqual(duplicate_periods.tolist(), [2, 2])

    def test_inception_blocks_and_feature_layer_norm_are_exact(self) -> None:
        """Use the frozen kernels and feature-only LayerNorm without batch norms."""
        model = build_timesnet1d_model()
        self.assertEqual(DEFAULT_INCEPTION_KERNELS, (1, 3, 5))
        self.assertTrue(all(block.inception_in.kernel_sizes == (1, 3, 5) for block in model.blocks))
        self.assertTrue(all(normalization.normalized_shape == (80,) for normalization in model.feature_normalizations))
        self.assertFalse(any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) for module in model.modules()))

    def test_shared_hidden_loss_and_observed_restoration_are_reused(self) -> None:
        """Keep observed values out of loss and restore them through shared utilities."""
        prediction = torch.zeros((1, 4, 3))
        target = torch.zeros_like(prediction)
        hidden_mask = torch.tensor([[[False], [True], [True], [False]]])
        baseline = hidden_only_mse_loss(prediction, target, hidden_mask)
        prediction[:, 0, :] = 100.0
        self.assertEqual(float(baseline), float(hidden_only_mse_loss(prediction, target, hidden_mask)))
        sparse = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]])
        observed_mask = ~hidden_mask
        restored = restore_observed_points_tensor(prediction, sparse, observed_mask)
        self.assertTrue(torch.equal(restored[:, [0, 3], :], sparse[:, [0, 3], :]))

    def test_frequency_diagnostics_are_inspectable_without_changing_output(self) -> None:
        """Expose selected frequencies and periods without mutating forward behavior."""
        torch.manual_seed(7)
        model = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=2)
        model.eval()
        x = torch.randn(2, 19, 5)
        before = model(x)
        diagnostics = model.inspect_frequency_diagnostics(x)
        after = model(x)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(len(diagnostics), 2)
        for block_diagnostics in diagnostics:
            self.assertTrue(all(frequency > 0 for frequency in block_diagnostics["selected_frequency_indices"]))
            self.assertTrue(all(period >= 1 for period in block_diagnostics["selected_periods"]))

    def test_fixed_seed_eval_repeatability(self) -> None:
        """Keep evaluation deterministic for a fixed model and input."""
        torch.manual_seed(42)
        model = TimesNetReconstruction1D(d_model=8, d_ff=8, num_blocks=1)
        model.eval()
        x = torch.randn(2, 13, 5)
        self.assertTrue(torch.equal(model(x), model(x)))


if __name__ == "__main__":
    unittest.main()
