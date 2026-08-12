from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.fno1d.fno1d import build_fno1d_model
from src.training.trajectory_reconstruction.fno1d_reconstruction import (
    build_reconstruction_model_input,
    denormalize_prediction_tensor,
    fit_reconstruction_normalization,
    hidden_only_mse_loss,
    normalize_target_xyz,
    restore_observed_points_tensor,
)
from src.training.trajectory_reconstruction.sparse_sampling import (
    build_sparse_trajectory_data,
)


class TestFNO1DSparseReconstruction(unittest.TestCase):
    """Test the FNO1D sparse reconstruction data contract."""

    def setUp(self) -> None:
        lambda_grid = np.linspace(2.0, 6.0, 5, dtype=np.float64)
        base = np.stack(
            (
                lambda_grid,
                np.square(lambda_grid),
                np.sin(lambda_grid),
            ),
            axis=-1,
        )
        target = np.stack((base, base + 10.0), axis=0)
        self.train_data = build_sparse_trajectory_data(
            target_xyz=target,
            lambda_grid=lambda_grid,
            stride=2,
        )
        self.normalization = fit_reconstruction_normalization(self.train_data)

    def test_model_input_has_exact_five_channels_without_q(self) -> None:
        """Build a five-channel input without any Q channel."""
        model_input = build_reconstruction_model_input(
            self.train_data,
            self.normalization,
        )
        self.assertEqual(model_input.shape, (2, 5, 5))
        np.testing.assert_array_equal(
            model_input[..., 3],
            self.train_data.observed_mask[..., 0].astype(np.float32),
        )
        np.testing.assert_allclose(model_input[0, :, 4], np.linspace(0.0, 1.0, 5))
        np.testing.assert_array_equal(
            model_input[0, ~self.train_data.observed_mask[0, :, 0], :3],
            np.zeros((2, 3), dtype=np.float32),
        )

    def test_input_statistics_use_training_observed_values_only(self) -> None:
        """Exclude hidden placeholders and hidden target values from input stats."""
        observed = self.train_data.observed_mask[..., 0]
        expected_mean = np.mean(self.train_data.sparse_xyz[observed], axis=0)
        expected_std = np.std(self.train_data.sparse_xyz[observed], axis=0)
        np.testing.assert_allclose(
            self.normalization.input_xyz_mean,
            expected_mean,
        )
        np.testing.assert_allclose(
            self.normalization.input_xyz_std,
            expected_std,
        )

    def test_target_statistics_use_full_training_target_only(self) -> None:
        """Fit output statistics from full training trajectories only."""
        expected_mean = np.mean(self.train_data.target_xyz, axis=(0, 1))
        expected_std = np.std(self.train_data.target_xyz, axis=(0, 1))
        np.testing.assert_allclose(
            self.normalization.target_xyz_mean,
            expected_mean,
        )
        np.testing.assert_allclose(
            self.normalization.target_xyz_std,
            expected_std,
        )

    def test_validation_values_use_fixed_training_statistics(self) -> None:
        """Apply already fitted training statistics to a distinct validation split."""
        validation_target = np.full((1, 5, 3), 1000.0, dtype=np.float64)
        validation_data = build_sparse_trajectory_data(
            target_xyz=validation_target,
            lambda_grid=self.train_data.lambda_grid,
            stride=2,
        )
        model_input = build_reconstruction_model_input(
            validation_data,
            self.normalization,
        )
        observed = validation_data.observed_mask[..., 0]
        expected = (
            validation_data.sparse_xyz[observed]
            - np.asarray(self.normalization.input_xyz_mean)
        ) / np.asarray(self.normalization.input_xyz_std)
        np.testing.assert_allclose(model_input[..., :3][observed], expected)

    def test_target_normalization_is_invertible(self) -> None:
        """Recover raw targets after normalized-space conversion."""
        normalized = normalize_target_xyz(
            self.train_data.target_xyz,
            self.normalization,
        )
        recovered = denormalize_prediction_tensor(
            torch.from_numpy(normalized),
            self.normalization,
        ).numpy()
        np.testing.assert_allclose(recovered, self.train_data.target_xyz, rtol=1e-6)

    def test_hidden_only_loss_ignores_observed_positions(self) -> None:
        """Changing only observed predictions leaves hidden-only loss unchanged."""
        target = torch.zeros((1, 4, 3), dtype=torch.float32)
        prediction = torch.zeros_like(target)
        hidden_mask = torch.tensor([[[False], [True], [True], [False]]])
        baseline = hidden_only_mse_loss(prediction, target, hidden_mask)
        prediction[:, 0, :] = 100.0
        prediction[:, 3, :] = -100.0
        self.assertEqual(float(baseline), float(hidden_only_mse_loss(prediction, target, hidden_mask)))

    def test_hidden_only_loss_matches_hand_computation(self) -> None:
        """Use only one hidden scalar error in the manual MSE check."""
        target = torch.zeros((1, 2, 3), dtype=torch.float32)
        prediction = torch.zeros_like(target)
        prediction[0, 1, 2] = 6.0
        hidden_mask = torch.tensor([[[False], [True]]])
        self.assertAlmostEqual(
            float(hidden_only_mse_loss(prediction, target, hidden_mask)),
            12.0,
        )

    def test_hidden_only_loss_rejects_zero_hidden_points(self) -> None:
        """Raise a clear error when a batch has no hidden point."""
        with self.assertRaisesRegex(ValueError, "at least one hidden point"):
            hidden_only_mse_loss(
                torch.zeros((1, 2, 3)),
                torch.zeros((1, 2, 3)),
                torch.zeros((1, 2, 1), dtype=torch.bool),
            )

    def test_observed_restoration_is_exact(self) -> None:
        """Restore raw observed values without changing hidden predictions."""
        prediction = torch.full((1, 4, 3), -3.0, dtype=torch.float64)
        sparse = torch.tensor(
            [[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        observed_mask = torch.tensor([[[True], [False], [True], [False]]])
        restored = restore_observed_points_tensor(prediction, sparse, observed_mask)
        self.assertTrue(torch.equal(restored[:, [0, 2], :], sparse[:, [0, 2], :]))
        self.assertTrue(torch.equal(restored[:, [1, 3], :], prediction[:, [1, 3], :]))

    def test_fno1d_forward_pass_has_reconstruction_shape(self) -> None:
        """Run one FNO1D forward pass on the five-channel reconstruction input."""
        model = build_fno1d_model(
            in_dim=5,
            out_dim=3,
            modes=2,
            width=8,
            depth=2,
        )
        model_input = torch.from_numpy(
            build_reconstruction_model_input(self.train_data, self.normalization)
        )
        prediction = model(model_input)
        self.assertEqual(tuple(prediction.shape), (2, 5, 3))


if __name__ == "__main__":
    unittest.main()
