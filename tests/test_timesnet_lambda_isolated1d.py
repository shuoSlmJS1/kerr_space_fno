from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from scripts.train_sparse_reconstruction_timesnet_lambda_isolated1d import build_parser
from src.models.timesnet1d import build_timesnet1d_model, count_trainable_parameters
from src.models.timesnet_lambda_isolated1d import (
    TimesNetLambdaIsolatedPeriodSelection1D,
    build_timesnet_lambda_isolated1d_model,
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


class TestTimesNetLambdaIsolated1D(unittest.TestCase):
    """Test the lambda-isolated TimesNet period-selection ablation contract."""

    def setUp(self) -> None:
        lambda_grid = np.linspace(0.0, 1.0, 17, dtype=np.float64)
        base = np.stack((np.sin(lambda_grid), np.square(lambda_grid), np.cos(lambda_grid)), axis=-1)
        target = np.stack((base, base + 2.0), axis=0)
        self.sparse_data = build_sparse_trajectory_data(target, lambda_grid, stride=2)
        self.normalization = fit_reconstruction_normalization(self.sparse_data)

    def test_input_output_and_parameter_contract(self) -> None:
        """Keep the five-channel input, three-channel output, and canonical parameter count."""
        model_input = build_reconstruction_model_input(self.sparse_data, self.normalization)
        model = build_timesnet_lambda_isolated1d_model()
        self.assertEqual(model_input.shape, (2, 17, 5))
        self.assertEqual(tuple(model(torch.from_numpy(model_input)).shape), (2, 17, 3))
        self.assertEqual(count_trainable_parameters(model), 1_077_299)
        self.assertEqual(count_trainable_parameters(model), count_trainable_parameters(build_timesnet1d_model()))

    def test_lambda_component_is_exactly_removed_only_from_initial_selection_signal(self) -> None:
        """Separate the non-bias lambda projection component without changing full latent values."""
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8)
        x = torch.randn(2, 19, 5)
        full = model.input_projection(x)
        component = model.lambda_projection_component(x)
        selection = full - component
        expected = x[..., 4:5] * model.input_projection.weight[:, 4].view(1, 1, -1)
        self.assertTrue(torch.allclose(component, expected))
        self.assertTrue(torch.allclose(selection + component, full))
        self.assertFalse(torch.equal(full, selection))

    def test_lambda_still_changes_full_prediction_path(self) -> None:
        """Retain lambda in the full input projection and output prediction path."""
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8)
        with torch.no_grad():
            model.input_projection.weight[:, 4].fill_(1.0)
        x = torch.zeros(1, 19, 5)
        changed = x.clone(); changed[..., 4] = 1.0
        self.assertFalse(torch.equal(model.input_projection(x), model.input_projection(changed)))

    def test_frequency_selection_allows_low_and_sampling_related_frequencies(self) -> None:
        """Do not hard-code suppression of f=1, f=2, or sampling-related frequencies."""
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8, num_blocks=1, top_k=1)
        with torch.no_grad():
            model.input_projection.weight.zero_(); model.input_projection.bias.zero_()
            model.input_projection.weight[0, 0] = 1.0
        time = torch.arange(32, dtype=torch.float32)
        low = torch.zeros(1, 32, 5); low[..., 0] = torch.sin(2.0 * torch.pi * time / 32.0)
        low_two = torch.zeros(1, 32, 5); low_two[..., 0] = torch.sin(2.0 * torch.pi * 2.0 * time / 32.0)
        sampled = torch.zeros(1, 32, 5); sampled[..., 0] = torch.sin(2.0 * torch.pi * 4.0 * time / 32.0)
        self.assertEqual(model.inspect_frequency_diagnostics(low)[0]["selected_frequency_indices"], [1])
        self.assertEqual(model.inspect_frequency_diagnostics(low_two)[0]["selected_frequency_indices"], [2])
        self.assertEqual(model.inspect_frequency_diagnostics(sampled)[0]["selected_frequency_indices"], [4])

    def test_full_period_branches_receive_full_not_selection_latent(self) -> None:
        """Apply selection periods to full latent branches while auxiliary flow remains separate."""
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8, num_blocks=1)
        x = torch.randn(2, 19, 5)
        full = model.input_projection(x)
        selection = full - model.lambda_projection_component(x)
        calls: list[torch.Tensor] = []
        original = model._apply_period_branches

        def capture(block, latent, periods, weights):
            calls.append(latent.detach().clone())
            return original(block, latent, periods, weights)

        with mock.patch.object(model, "_apply_period_branches", side_effect=capture):
            diagnostics = model.inspect_frequency_diagnostics(x)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.allclose(calls[0], full))
        self.assertTrue(torch.allclose(calls[1], selection))
        self.assertTrue(all(period > 0 for period in diagnostics[0]["selected_periods"]))

    def test_two_block_diagnostics_are_read_only_and_reproducible(self) -> None:
        """Expose both blocks' selected frequencies without changing prediction behavior."""
        torch.manual_seed(42)
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8, num_blocks=2)
        model.eval(); x = torch.randn(2, 23, 5)
        before = model(x); diagnostics = model.inspect_frequency_diagnostics(x); after = model(x)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(all(value > 0 for item in diagnostics for value in item["selected_frequency_indices"]))

    def test_t1200_forward_and_finite_backward(self) -> None:
        """Support a length-1200 sequence and finite gradients with the shared architecture."""
        model = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8, num_blocks=2)
        output = model(torch.randn(1, 1200, 5))
        self.assertEqual(tuple(output.shape), (1, 1200, 3))
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_shared_hidden_loss_and_observed_restoration(self) -> None:
        """Reuse the existing hidden-only loss and observed-point restoration utilities."""
        prediction = torch.zeros((1, 4, 3)); target = torch.zeros_like(prediction)
        hidden = torch.tensor([[[False], [True], [True], [False]]])
        baseline = hidden_only_mse_loss(prediction, target, hidden)
        prediction[:, 0, :] = 100.0
        self.assertEqual(float(baseline), float(hidden_only_mse_loss(prediction, target, hidden)))
        sparse = torch.ones_like(prediction); restored = restore_observed_points_tensor(prediction, sparse, ~hidden)
        self.assertTrue(torch.equal(restored[:, [0, 3]], sparse[:, [0, 3]]))

    def test_formal_defaults_and_metadata_are_explicit(self) -> None:
        """Keep formal training defaults and record the auxiliary-stream treatment."""
        args = build_parser().parse_args(["--dataset-path", "dataset.npz", "--run-name", "formal"])
        self.assertEqual((args.stride, args.epochs, args.batch_size, args.seed), (16, 600, 32, 42))
        model = build_timesnet_lambda_isolated1d_model()
        metadata = model.architecture_metadata()
        self.assertEqual(metadata["period_selection_treatment"], "lambda_isolated_shared_parameter_auxiliary_stream")
        self.assertEqual(metadata["selection_stream_trainable_parameters"], 0)

    def test_canonical_model_source_behavior_remains_independent(self) -> None:
        """Keep canonical TimesNet instantiation and behavior separate from the ablation class."""
        canonical = build_timesnet1d_model(d_model=8, d_ff=8, num_blocks=1)
        ablation = build_timesnet_lambda_isolated1d_model(d_model=8, d_ff=8, num_blocks=1)
        self.assertIsInstance(ablation, TimesNetLambdaIsolatedPeriodSelection1D)
        self.assertNotEqual(canonical.architecture_metadata()["family"], ablation.architecture_metadata()["family"])


if __name__ == "__main__":
    unittest.main()
