"""CPU precision regressions for new benchmark statistics, isolated from Plan B."""

import io
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.benchmark_track_a import build_model, model_config, parameter_counts
from src.training.benchmark_data import Trajectories, fit_normalization, restore_normalization
from src.training.benchmark_track_a import SCHEMA, TRAINING, checkpoint_contract
from src.training.fno2d import normalization_2d as historical


def sample(t=17):
    q = np.linspace(1.6007, 2.9993, 31)[:, None]
    grid = np.linspace(0, 5.995, t)
    xyz = np.stack([q + grid, q - grid, np.broadcast_to(q, (len(q), t))], axis=-1)
    return Trajectories(q, grid, xyz)


def reference(array):
    # 独立逐通道 fsum 参考，避免复用被测 NumPy 多轴归约。
    means, stds = [], []
    for channel in range(array.shape[-1]):
        values = [float(v) for v in array[..., channel].ravel()]
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        means.append(mean)
        stds.append(max(std, 1e-8))
    return means, stds


class BenchmarkNormalizationPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_channels_explicit_float64_reductions_and_storage(self):
        with patch("numpy.mean", wraps=np.mean) as mean, patch("numpy.std", wraps=np.std) as std:
            stats = fit_normalization(sample())
        self.assertEqual(mean.call_count, 2)
        self.assertEqual(std.call_count, 2)
        for call in mean.call_args_list + std.call_args_list:
            self.assertEqual(call.kwargs["dtype"], np.float64)
            self.assertEqual(call.kwargs["axis"], (0, 1, 2))
            self.assertEqual(call.args[0].dtype, np.float32)
        for call in std.call_args_list:
            self.assertEqual(call.kwargs["ddof"], 0)
        values = stats.x_mean + stats.x_std + stats.y_mean + stats.y_std
        self.assertTrue(all(type(v) is float for v in values))
        self.assertTrue(any(v != float(np.float32(v)) for v in values))

    def test_broadcast_q_invariant_under_resolution_change(self):
        coarse, fine = sample(1200), sample(2399)
        np.testing.assert_array_equal(coarse.q, fine.q)
        a, b = fit_normalization(coarse), fit_normalization(fine)
        np.testing.assert_allclose([a.x_mean[0], a.x_std[0]],
                                   [b.x_mean[0], b.x_std[0]], rtol=1e-12, atol=1e-12)
        self.assertNotEqual(a.x_std[1], b.x_std[1])

    def test_matches_independent_float64_reference(self):
        data = sample(2399)
        stats = fit_normalization(data)
        for array, mean, std in ((data.raw_input(), stats.x_mean, stats.x_std),
                                 (data.xyz.astype(np.float32), stats.y_mean, stats.y_std)):
            expected_mean, expected_std = reference(array)
            np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(std, expected_std, rtol=1e-12, atol=1e-12)

    def test_population_std_and_epsilon_floor(self):
        q = np.array([[2.0], [2.0]])
        grid = np.array([0.0, 2.0])
        xyz = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (2, 2, 3)).copy()
        xyz[:, 1, 1] = 1e-10
        xyz[:, 1, 2] = 4e-8
        stats = fit_normalization(Trajectories(q, grid, xyz))
        self.assertEqual(stats.eps, 1e-8)
        self.assertEqual(stats.x_std, [1e-8, 1.0])
        self.assertEqual(stats.y_std[:2], [1e-8, 1e-8])
        self.assertEqual(stats.y_std[2], float(np.float32(4e-8)) / 2)

    def test_train_only_application_keeps_float32_and_stored_precision(self):
        train, val = sample(), sample()
        val.q += 100
        val.xyz += 100
        stats = fit_normalization(train)
        stored = stats.to_dict()
        train.stats = val.stats = stats
        vx, _ = val[0]
        self.assertGreater(float(vx[:, 0].mean()), 100)
        self.assertEqual(fit_normalization(train).to_dict(), stored)
        x, y = train[0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        expected = (train.raw_input()[0] - np.asarray(stats.x_mean, dtype=np.float32)) / np.asarray(stats.x_std, dtype=np.float32)
        np.testing.assert_array_equal(x.numpy(), expected)
        model = build_model("fno1d")
        prediction = model(x[None])
        loss = torch.nn.functional.mse_loss(prediction, y[None])
        loss.backward()
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(all(p.dtype == torch.float32 and p.grad.dtype == torch.float32
                            for p in model.parameters()))
        self.assertEqual(stats.to_dict(), stored)

    def test_json_and_checkpoint_roundtrip_preserve_exact_statistics(self):
        stats = fit_normalization(sample())
        stored = stats.to_dict()
        self.assertEqual(restore_normalization(json.loads(json.dumps(stored))).to_dict(), stored)
        model = build_model("fno1d")
        payload = {"schema": SCHEMA, "training": dict(TRAINING), "train_resolution": 1200,
                   "model_name": "fno1d", "model_config": model_config("fno1d"),
                   "capacity": parameter_counts(model), "model_state_dict": model.state_dict(),
                   "normalization": stored}
        with io.BytesIO() as buffer:
            torch.save(payload, buffer)
            buffer.seek(0)
            _, restored = checkpoint_contract(torch.load(buffer, weights_only=True))
        self.assertEqual(restored.to_dict(), stored)
        data = sample()
        data.stats = stats
        before = data[0]
        data.stats = restored
        after = data[0]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_historical_statistics_stay_on_original_path(self):
        data = sample(2399)
        array = data.raw_input()[None]
        expected_mean = np.mean(array, axis=(0, 1, 2)).astype(np.float32)
        expected_std = np.maximum(np.std(array, axis=(0, 1, 2)), 1e-8).astype(np.float32)
        mean, std = historical.compute_channel_mean_std(array)
        np.testing.assert_array_equal(mean, expected_mean)
        np.testing.assert_array_equal(std, expected_std)
        old = historical.compute_field_normalization_stats(array, data.xyz.astype(np.float32)[None])
        with patch.object(historical, "compute_channel_mean_std", side_effect=AssertionError("Historical fitter must not be used")):
            corrected = fit_normalization(data)
        self.assertNotEqual(corrected.x_mean, old.x_mean)
        self.assertEqual(historical.FieldNormalizationStats.from_dict(old.to_dict()).to_dict(), old.to_dict())


if __name__ == "__main__":
    unittest.main()
