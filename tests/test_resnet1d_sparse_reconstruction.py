from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.models.resnet1d import (
    DEFAULT_DILATION_SCHEDULE,
    DilatedResNet1D,
    build_dilated_resnet1d,
    calculate_theoretical_receptive_field,
    count_trainable_parameters,
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


class TestDilatedResNet1DSparseReconstruction(unittest.TestCase):
    """Test the Dilated ResNet1D sparse reconstruction contract."""

    def setUp(self) -> None:
        lambda_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        base = np.stack(
            (
                np.sin(lambda_grid),
                np.square(lambda_grid),
                np.cos(lambda_grid),
            ),
            axis=-1,
        )
        target = np.stack((base, base + 2.0), axis=0)
        self.sparse_data = build_sparse_trajectory_data(
            target_xyz=target,
            lambda_grid=lambda_grid,
            stride=2,
        )
        self.normalization = fit_reconstruction_normalization(self.sparse_data)

    def test_model_input_has_exact_five_channels_without_q(self) -> None:
        """Build the shared five-channel input without a Q channel."""
        model_input = build_reconstruction_model_input(
            self.sparse_data,
            self.normalization,
        )
        self.assertEqual(model_input.shape, (2, 9, 5))
        np.testing.assert_array_equal(
            model_input[..., 3],
            self.sparse_data.observed_mask[..., 0].astype(np.float32),
        )
        self.assertEqual(model_input.shape[-1], 5)

    def test_default_dilation_schedule_is_exact(self) -> None:
        """Expose the approved nine-block dilation schedule exactly."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        self.assertEqual(model.dilations, DEFAULT_DILATION_SCHEDULE)
        self.assertEqual(len(model.blocks), 9)
        self.assertEqual(
            tuple(block.dilated_conv.dilation[0] for block in model.blocks),
            DEFAULT_DILATION_SCHEDULE,
        )
        self.assertTrue(
            all(block.local_conv.dilation == (1,) for block in model.blocks)
        )

    def test_replicate_padding_preserves_length_without_circular_padding(self) -> None:
        """Use explicit replicate padding and preserve temporal length."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        self.assertEqual(model.padding_type, "replicate")
        self.assertTrue(
            all(
                isinstance(block.dilated_padding, nn.ReplicationPad1d)
                for block in model.blocks
            )
        )
        self.assertTrue(
            all(
                isinstance(block.local_padding, nn.ReplicationPad1d)
                for block in model.blocks
            )
        )
        self.assertTrue(
            all(block.dilated_conv.padding == (0,) for block in model.blocks)
        )
        self.assertTrue(
            all(block.local_conv.padding == (0,) for block in model.blocks)
        )
        self.assertTrue(
            all(block.dilated_conv.padding_mode == "zeros" for block in model.blocks)
        )
        model_input = torch.randn(1, 17, 5)
        self.assertEqual(tuple(model(model_input).shape), (1, 17, 3))

    def test_model_does_not_contain_time_normalization_layers(self) -> None:
        """Avoid BatchNorm, GroupNorm, and LayerNorm in the first comparison model."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        forbidden_types = (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)
        self.assertFalse(
            any(isinstance(module, forbidden_types) for module in model.modules())
        )

    def test_theoretical_receptive_field_matches_independent_formula(self) -> None:
        """Independently verify the approved receptive field of 3121."""
        expected = 1 + 6 * (sum(DEFAULT_DILATION_SCHEDULE) + 9)
        calculated = calculate_theoretical_receptive_field(
            kernel_size=7,
            dilations=DEFAULT_DILATION_SCHEDULE,
        )
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        self.assertEqual(expected, 3121)
        self.assertEqual(calculated, expected)
        self.assertEqual(model.theoretical_receptive_field, expected)
        self.assertEqual(
            model.architecture_metadata()["theoretical_receptive_field"],
            expected,
        )

    def test_default_parameter_count_matches_fairness_contract(self) -> None:
        """Match the approved trainable parameter count for the formal model."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        self.assertEqual(count_trainable_parameters(model), 1_077_507)
        self.assertEqual(
            model.architecture_metadata()["trainable_parameter_count"],
            1_077_507,
        )

    def test_forward_pass_at_formal_time_length(self) -> None:
        """Run a full-architecture forward pass at T=1200."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        model_input = torch.randn(1, 1200, 5)
        prediction = model(model_input)
        self.assertEqual(tuple(prediction.shape), (1, 1200, 3))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_backward_pass_produces_finite_gradients(self) -> None:
        """Produce finite gradients through the full dilated architecture."""
        model = build_dilated_resnet1d(in_dim=5, out_dim=3)
        model_input = torch.randn(1, 64, 5)
        loss = model(model_input).square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_hidden_only_loss_ignores_observed_predictions(self) -> None:
        """Keep observed positions out of the shared hidden-only loss."""
        target = torch.zeros((1, 4, 3), dtype=torch.float32)
        prediction = torch.zeros_like(target)
        hidden_mask = torch.tensor([[[False], [True], [True], [False]]])
        baseline = hidden_only_mse_loss(prediction, target, hidden_mask)
        prediction[:, 0, :] = 100.0
        prediction[:, 3, :] = -100.0
        self.assertEqual(
            float(baseline),
            float(hidden_only_mse_loss(prediction, target, hidden_mask)),
        )

    def test_shared_observed_restoration_is_exact(self) -> None:
        """Restore observed raw values with the shared reconstruction utility."""
        prediction = torch.full((1, 4, 3), -2.0, dtype=torch.float64)
        sparse = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        observed_mask = torch.tensor([[[True], [False], [True], [False]]])
        restored = restore_observed_points_tensor(prediction, sparse, observed_mask)
        self.assertTrue(
            torch.equal(restored[:, [0, 2], :], sparse[:, [0, 2], :])
        )
        self.assertTrue(
            torch.equal(restored[:, [1, 3], :], prediction[:, [1, 3], :])
        )

    def test_build_function_supports_explicit_smoke_overrides(self) -> None:
        """Allow smaller width and block count without changing formal defaults."""
        smoke_model = build_dilated_resnet1d(
            in_dim=5,
            out_dim=3,
            width=8,
            blocks=2,
        )
        self.assertEqual(smoke_model.width, 8)
        self.assertEqual(smoke_model.dilations, (1, 2))
        self.assertEqual(smoke_model.theoretical_receptive_field, 31)

    def test_model_rejects_non_five_channel_input(self) -> None:
        """Reject an input that could introduce an extra Q-like channel."""
        model = DilatedResNet1D(in_dim=5, out_dim=3)
        with self.assertRaisesRegex(ValueError, "expected in_dim=5"):
            model(torch.randn(1, 9, 6))


if __name__ == "__main__":
    unittest.main()
