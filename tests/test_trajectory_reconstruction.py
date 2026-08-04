from __future__ import annotations

import unittest

import numpy as np

from src.training.trajectory_reconstruction import (
    build_observed_indices,
    build_sparse_trajectory_data,
    compute_hidden_masked_metrics,
    reconstruct_linear,
    reconstruct_pchip,
)


class TrajectoryReconstructionTestCase(unittest.TestCase):
    """Tests for sparse sampling, interpolation baselines, and masked metrics."""

    def setUp(self) -> None:
        """Create two nine-point linear synthetic trajectories."""
        self.lambda_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        base = np.stack(
            (
                self.lambda_grid,
                2.0 * self.lambda_grid + 1.0,
                -0.5 * self.lambda_grid + 3.0,
            ),
            axis=-1,
        )
        self.target_xyz = np.stack((base, 1.5 * base - 0.25), axis=0)

    def test_stride_two_indices_masks_and_shapes(self) -> None:
        """Verify stride=2 indices, endpoints, complementary masks, and shapes."""
        target_before = self.target_xyz.copy()
        data = build_sparse_trajectory_data(
            target_xyz=self.target_xyz,
            lambda_grid=self.lambda_grid,
            stride=2,
        )

        self.assertEqual(data.sampling.observed_indices, (0, 2, 4, 6, 8))
        self.assertEqual(data.sampling.hidden_indices, (1, 3, 5, 7))
        self.assertTrue(data.observed_mask[:, 0, 0].all())
        self.assertTrue(data.observed_mask[:, -1, 0].all())
        self.assertEqual(data.sparse_xyz.shape, (2, 9, 3))
        self.assertEqual(data.observed_mask.shape, (2, 9, 1))
        self.assertEqual(data.hidden_mask.shape, (2, 9, 1))
        self.assertEqual(data.target_xyz.shape, (2, 9, 3))
        self.assertEqual(data.observed_mask.dtype, np.bool_)
        self.assertEqual(data.hidden_mask.dtype, np.bool_)
        np.testing.assert_array_equal(
            np.logical_xor(data.observed_mask, data.hidden_mask),
            np.ones_like(data.observed_mask),
        )
        np.testing.assert_array_equal(
            np.logical_and(data.observed_mask, data.hidden_mask),
            np.zeros_like(data.observed_mask),
        )
        np.testing.assert_array_equal(self.target_xyz, target_before)
        np.testing.assert_array_equal(
            data.sparse_xyz[:, data.sampling.hidden_indices, :],
            0.0,
        )

    def test_linear_recovers_linear_hidden_points(self) -> None:
        """Verify Linear reconstruction of hidden points on linear trajectories."""
        data = build_sparse_trajectory_data(
            self.target_xyz,
            self.lambda_grid,
            stride=2,
        )
        prediction = reconstruct_linear(
            data.lambda_grid,
            data.sparse_xyz,
            data.observed_mask,
        )
        np.testing.assert_allclose(
            prediction[data.hidden_mask[..., 0]],
            data.target_xyz[data.hidden_mask[..., 0]],
            rtol=0.0,
            atol=1e-14,
        )

    def test_pchip_recovers_linear_hidden_points(self) -> None:
        """Verify PCHIP reconstruction of hidden points on linear trajectories."""
        data = build_sparse_trajectory_data(
            self.target_xyz,
            self.lambda_grid,
            stride=2,
        )
        prediction = reconstruct_pchip(
            data.lambda_grid,
            data.sparse_xyz,
            data.observed_mask,
        )
        np.testing.assert_allclose(
            prediction[data.hidden_mask[..., 0]],
            data.target_xyz[data.hidden_mask[..., 0]],
            rtol=0.0,
            atol=1e-14,
        )

    def test_baselines_restore_nonlinear_observed_points_exactly(self) -> None:
        """Verify exact restoration of observed points on nonlinear trajectories."""
        nonlinear_base = np.stack(
            (
                self.lambda_grid**2,
                np.sin(self.lambda_grid),
                np.exp(self.lambda_grid),
            ),
            axis=-1,
        )
        nonlinear_target = np.stack(
            (nonlinear_base, nonlinear_base + np.asarray([1.0, -2.0, 0.5])),
            axis=0,
        )
        data = build_sparse_trajectory_data(
            nonlinear_target,
            self.lambda_grid,
            stride=2,
        )

        observed = data.observed_mask[..., 0]
        for baseline in (reconstruct_linear, reconstruct_pchip):
            with self.subTest(baseline=baseline.__name__):
                prediction = baseline(
                    data.lambda_grid,
                    data.sparse_xyz,
                    data.observed_mask,
                )
                np.testing.assert_array_equal(
                    prediction[observed],
                    data.sparse_xyz[observed],
                )

    def test_masked_metrics_match_manual_values(self) -> None:
        """Verify component, overall, and per-trajectory hidden-only metrics."""
        eps = 1e-12
        target = np.asarray(
            [
                [[8.0, 8.0, 8.0], [1.0, 2.0, 4.0], [9.0, 9.0, 9.0]],
                [[7.0, 7.0, 7.0], [3.0, 4.0, 12.0], [6.0, 6.0, 6.0]],
            ]
        )
        prediction = target.copy()
        prediction[:, 1, :] = np.asarray([2.0, 4.0, 8.0])
        prediction[1, 1, :] = np.asarray([3.0, 8.0, 12.0])
        hidden_mask = np.asarray(
            [
                [[False], [True], [False]],
                [[False], [True], [False]],
            ]
        )

        metrics = compute_hidden_masked_metrics(
            prediction_xyz=prediction,
            target_xyz=target,
            hidden_mask=hidden_mask,
            eps=eps,
        )

        self.assertEqual(metrics["hidden_point_count"], 2)
        expected_component_relative_l2 = {
            "x": 1.0 / (np.sqrt(10.0) + eps),
            "y": np.sqrt(20.0) / (np.sqrt(20.0) + eps),
            "z": 4.0 / (np.sqrt(160.0) + eps),
        }
        self.assertAlmostEqual(metrics["components"]["x"]["mse"], 0.5)
        self.assertAlmostEqual(metrics["components"]["y"]["mse"], 10.0)
        self.assertAlmostEqual(metrics["components"]["z"]["mse"], 8.0)
        for component_name, expected_value in expected_component_relative_l2.items():
            with self.subTest(component=component_name):
                self.assertAlmostEqual(
                    metrics["components"][component_name]["relative_l2"],
                    expected_value,
                )

        self.assertAlmostEqual(metrics["overall"]["mse"], 37.0 / 6.0)
        expected_relative_l2 = np.sqrt(37.0) / (np.sqrt(190.0) + eps)
        self.assertAlmostEqual(
            metrics["overall"]["relative_l2"],
            expected_relative_l2,
        )
        expected_trajectory_values = np.asarray(
            [
                np.sqrt(21.0) / (np.sqrt(21.0) + eps),
                4.0 / (13.0 + eps),
            ]
        )
        np.testing.assert_allclose(
            metrics["per_trajectory_relative_l2"]["values"],
            expected_trajectory_values,
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(
            metrics["per_trajectory_relative_l2"]["mean"],
            float(np.mean(expected_trajectory_values)),
        )
        self.assertAlmostEqual(
            metrics["per_trajectory_relative_l2"]["median"],
            float(np.mean(expected_trajectory_values)),
        )
        expected_p95 = (
            expected_trajectory_values[1]
            + 0.95
            * (expected_trajectory_values[0] - expected_trajectory_values[1])
        )
        self.assertAlmostEqual(
            metrics["per_trajectory_relative_l2"]["p95"],
            float(expected_p95),
        )
        self.assertAlmostEqual(
            metrics["per_trajectory_relative_l2"]["max"],
            float(expected_trajectory_values[0]),
        )

    def test_stride_one_is_rejected(self) -> None:
        """Verify that stride=1 is rejected."""
        with self.assertRaisesRegex(ValueError, "stride must be at least 2"):
            build_observed_indices(time_steps=9, stride=1)

    def test_single_time_point_is_rejected(self) -> None:
        """Verify that a single time point is rejected."""
        with self.assertRaisesRegex(ValueError, "time_steps must be at least 2"):
            build_sparse_trajectory_data(
                target_xyz=np.zeros((2, 1, 3)),
                lambda_grid=np.asarray([0.0]),
                stride=2,
            )

    def test_fewer_than_two_observed_points_are_rejected(self) -> None:
        """Verify that both baselines reject insufficient observed points."""
        sparse = np.zeros((1, 3, 3), dtype=np.float64)
        mask = np.asarray([[[True], [False], [False]]])
        grid = np.asarray([0.0, 0.5, 1.0])

        for baseline in (reconstruct_linear, reconstruct_pchip):
            with self.subTest(baseline=baseline.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "at least two observed points",
                ):
                    baseline(grid, sparse, mask)

    def test_non_increasing_lambda_grid_is_rejected(self) -> None:
        """Verify that both baselines reject non-increasing grids."""
        sparse = np.zeros((1, 3, 3), dtype=np.float64)
        mask = np.ones((1, 3, 1), dtype=np.bool_)
        grid = np.asarray([0.0, 0.5, 0.5])

        for baseline in (reconstruct_linear, reconstruct_pchip):
            with self.subTest(baseline=baseline.__name__):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    baseline(grid, sparse, mask)

    def test_non_finite_inputs_are_rejected(self) -> None:
        """Verify that sampling and metrics reject non-finite inputs."""
        invalid_target = self.target_xyz.copy()
        invalid_target[0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "only finite values"):
            build_sparse_trajectory_data(
                invalid_target,
                self.lambda_grid,
                stride=2,
            )

        data = build_sparse_trajectory_data(
            self.target_xyz,
            self.lambda_grid,
            stride=2,
        )
        invalid_prediction = data.target_xyz.copy()
        invalid_prediction[0, 1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "only finite values"):
            compute_hidden_masked_metrics(
                invalid_prediction,
                data.target_xyz,
                data.hidden_mask,
            )

    def test_metrics_reject_trajectory_without_hidden_points(self) -> None:
        """Verify that metrics reject a trajectory without hidden points."""
        target = np.zeros((1, 3, 3), dtype=np.float64)
        hidden_mask = np.zeros((1, 3, 1), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "at least one hidden point"):
            compute_hidden_masked_metrics(target, target, hidden_mask)

    def test_empty_batches_are_rejected(self) -> None:
        """Verify that sampling, metrics, and baselines reject empty batches."""
        empty_target = np.empty((0, 3, 3), dtype=np.float64)
        grid = np.asarray([0.0, 0.5, 1.0])
        empty_mask = np.empty((0, 3, 1), dtype=np.bool_)

        with self.assertRaisesRegex(ValueError, "at least one trajectory"):
            build_sparse_trajectory_data(empty_target, grid, stride=2)
        with self.assertRaisesRegex(ValueError, "at least one trajectory"):
            compute_hidden_masked_metrics(
                empty_target,
                empty_target,
                empty_mask,
            )
        for baseline in (reconstruct_linear, reconstruct_pchip):
            with self.subTest(baseline=baseline.__name__):
                with self.assertRaisesRegex(ValueError, "at least one trajectory"):
                    baseline(grid, empty_target, empty_mask)

    def test_complex_dtypes_are_rejected(self) -> None:
        """Verify that sampling, metrics, and baselines reject complex dtypes."""
        complex_target = self.target_xyz.astype(np.complex128)
        with self.assertRaisesRegex(TypeError, "real numeric dtype"):
            build_sparse_trajectory_data(
                complex_target,
                self.lambda_grid,
                stride=2,
            )

        target = np.zeros((1, 3, 3), dtype=np.float64)
        hidden_mask = np.ones((1, 3, 1), dtype=np.bool_)
        with self.assertRaisesRegex(TypeError, "real numeric dtype"):
            compute_hidden_masked_metrics(
                target.astype(np.complex128),
                target,
                hidden_mask,
            )

        sparse = np.zeros((1, 3, 3), dtype=np.complex128)
        observed_mask = np.ones((1, 3, 1), dtype=np.bool_)
        grid = np.asarray([0.0, 0.5, 1.0])
        for baseline in (reconstruct_linear, reconstruct_pchip):
            with self.subTest(baseline=baseline.__name__):
                with self.assertRaisesRegex(TypeError, "real numeric dtype"):
                    baseline(grid, sparse, observed_mask)


if __name__ == "__main__":
    unittest.main()
