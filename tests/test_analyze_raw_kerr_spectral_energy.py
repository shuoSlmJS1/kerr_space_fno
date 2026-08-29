from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_raw_kerr_spectral_energy.py"
SPEC = importlib.util.spec_from_file_location("raw_kerr_spectral_energy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
spectral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spectral
SPEC.loader.exec_module(spectral)


def _pair_artifact(*, valid: bool = True) -> dict[str, object]:
    classification = "EXACT_PREFIX" if valid else "NOT_PAIRED"
    return {
        "pair_classification": {
            "short_to_medium": classification,
            "short_to_long": classification,
            "medium_to_long": classification,
        },
        "scientific_reuse": {
            "historical_t1800_reusable": valid,
            "t2400_ready_for_future_a1": valid,
        },
    }


def _field(
    *,
    task_name: str,
    q: np.ndarray,
    truth: np.ndarray,
    lambda_grid: np.ndarray,
) -> spectral.CanonicalQField:
    records = [
        {
            "source_split": ("train", "val", "test")[index % 3],
            "source_index_within_split": index,
            "source_concatenated_index": index,
        }
        for index in range(q.size)
    ]
    return spectral.build_canonical_q_field(
        task_name=task_name,
        source_q=q,
        source_truth=truth,
        lambda_grid=lambda_grid,
        source_records=records,
    )


def _triplet() -> tuple[spectral.CanonicalQField, spectral.CanonicalQField, spectral.CanonicalQField]:
    q = np.array([2.0, 1.0], dtype=np.float64)
    short_truth = np.array(
        [
            [[20.0, 2.0, 3.0], [21.0, 3.0, 5.0], [22.0, 5.0, 8.0], [23.0, 7.0, 13.0]],
            [[10.0, 1.0, 1.0], [11.0, 1.0, 2.0], [12.0, 2.0, 3.0], [13.0, 3.0, 5.0]],
        ],
        dtype=np.float64,
    )
    medium_truth = np.concatenate((short_truth, short_truth[:, :2, :] + 10.0), axis=1)
    long_truth = np.concatenate((medium_truth, medium_truth[:, :2, :] + 20.0), axis=1)
    return (
        _field(task_name="short", q=q, truth=short_truth, lambda_grid=np.arange(4, dtype=np.float64) * 0.5),
        _field(task_name="medium", q=q, truth=medium_truth, lambda_grid=np.arange(6, dtype=np.float64) * 0.5),
        _field(task_name="long", q=q, truth=long_truth, lambda_grid=np.arange(8, dtype=np.float64) * 0.5),
    )


class PrerequisiteAndCanonicalTests(unittest.TestCase):
    def test_invalid_stage2_prerequisite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(_pair_artifact(valid=False)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "short_to_medium"):
                spectral.load_required_pair_validation(path)
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required fields"):
                spectral.load_required_pair_validation(path)

    def test_stable_canonical_q_reorders_truth_with_source_identity(self) -> None:
        field = _field(
            task_name="short",
            q=np.array([2.0, 1.0, 3.0]),
            truth=np.array(
                [
                    [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
                    [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
                ],
                dtype=np.float64,
            ),
            lambda_grid=np.array([0.0, 0.5]),
        )
        self.assertTrue(np.array_equal(field.canonical_q, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(field.canonical_truth[:, 0, 0], np.array([10.0, 20.0, 30.0])))
        self.assertEqual(field.source_records[0]["source_concatenated_index"], 1)

    def test_exact_prefix_identity_is_required_before_analysis(self) -> None:
        short, medium, long = _triplet()
        spectral.validate_triplet(short, medium, long)
        changed_truth = long.source_truth.copy()
        changed_truth[0, 0, 0] += 1.0
        invalid_long = _field(
            task_name="invalid",
            q=long.source_q,
            truth=changed_truth,
            lambda_grid=long.lambda_grid,
        )
        with self.assertRaisesRegex(ValueError, "T1200 and T2400 raw truths"):
            spectral.validate_triplet(short, medium, invalid_long)


class FrequencyAndEnergyTests(unittest.TestCase):
    def test_rfftfreq_uses_n_times_delta_lambda(self) -> None:
        frequencies = np.fft.rfftfreq(1200, d=0.005)
        self.assertAlmostEqual(frequencies[1], 1.0 / 6.0)
        self.assertAlmostEqual(frequencies[31], 31.0 / 6.0)
        self.assertNotAlmostEqual(frequencies[1], 1.0 / 5.995)

    def test_same_physical_sinusoid_shifts_discrete_index_with_length(self) -> None:
        delta = 0.005
        physical_frequency = 1.0
        indices = []
        for total_length in (1200, 1800, 2400):
            positions = np.arange(total_length, dtype=np.float64) * delta
            signal = np.sin(2.0 * np.pi * physical_frequency * positions)
            coefficients = np.abs(np.fft.rfft(signal))
            indices.append(int(np.argmax(coefficients[1:]) + 1))
        self.assertEqual(indices, [6, 9, 12])

    def test_modes2_32_has_highest_retained_index_31(self) -> None:
        short, _, _ = _triplet()
        analysis = spectral.analyze_length_field(short, retained_modes_lambda=32, detrend="none")
        self.assertEqual(analysis.highest_retained_index, 2)
        long_positions = np.arange(1200, dtype=np.float64) * 0.005
        long_truth = np.stack(
            [np.sin(2.0 * np.pi * long_positions)] * 3,
            axis=-1,
        )[None, ...]
        long_field = _field(
            task_name="t1200",
            q=np.array([1.0]),
            truth=long_truth,
            lambda_grid=long_positions,
        )
        long_analysis = spectral.analyze_length_field(long_field, retained_modes_lambda=32, detrend="none")
        self.assertEqual(long_analysis.highest_retained_index, 31)

    def test_physical_cutoff_scales_one_two_thirds_one_half(self) -> None:
        delta = 0.005
        cutoffs = [31.0 / (total_length * delta) for total_length in (1200, 1800, 2400)]
        self.assertAlmostEqual(cutoffs[1] / cutoffs[0], 2.0 / 3.0)
        self.assertAlmostEqual(cutoffs[2] / cutoffs[0], 1.0 / 2.0)

    def test_one_sided_energy_is_parseval_consistent(self) -> None:
        generator = np.random.default_rng(7)
        signals = generator.normal(size=(3, 10, 3))
        _, energy = spectral.one_sided_rfft_energy(signals)
        spectral.validate_parseval_energy(signals, energy)

    def test_dc_and_nyquist_bins_are_not_doubled(self) -> None:
        positions = np.arange(8, dtype=np.float64)
        signal = 3.0 + 2.0 * ((-1.0) ** positions)
        signals = np.repeat(signal[None, :, None], repeats=3, axis=2)
        _, energy = spectral.one_sided_rfft_energy(signals)
        self.assertAlmostEqual(float(energy[0, 0, 0]), 72.0)
        self.assertAlmostEqual(float(energy[0, -1, 0]), 32.0)
        self.assertAlmostEqual(float(np.sum(energy)), 312.0)

    def test_energy_fractions_and_retained_partition_sum_to_one(self) -> None:
        spectrum = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
        frequencies = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
        summary = spectral.summarize_energy_partition(spectrum, frequencies, highest_retained_index=2)
        self.assertAlmostEqual(float(summary["retained_energy_fraction"]) + float(summary["above_retained_energy_fraction"]), 1.0)
        self.assertAlmostEqual(float(summary["retained_energy"]), 10.0)
        self.assertAlmostEqual(float(summary["above_retained_energy"]), 7.0)


class BandAndControlTests(unittest.TestCase):
    def test_common_bands_are_gap_free_and_non_overlapping(self) -> None:
        bands = spectral.build_common_physical_bands(
            {"T1200": 31.0 / 6.0, "T1800": 31.0 / 9.0, "T2400": 31.0 / 12.0}
        )
        for total_length in (1200, 1800, 2400):
            spectral.validate_band_partition(np.fft.rfftfreq(total_length, d=0.005), bands)

    def test_retained_partition_matches_physical_cutoff_partition(self) -> None:
        short, _, _ = _triplet()
        analysis = spectral.analyze_length_field(short, retained_modes_lambda=2, detrend="none")
        aggregate = spectral._component_spectra(analysis.component_energy_spectrum)["xyz_aggregate"]
        summary = spectral.summarize_energy_partition(
            aggregate,
            analysis.frequencies,
            analysis.highest_retained_index,
        )
        self.assertAlmostEqual(float(summary["retained_energy"]) + float(summary["above_retained_energy"]), float(summary["total_spectral_energy"]))

    def test_exact_shared_prefix_spectra_are_identical(self) -> None:
        short, medium, long = _triplet()
        controls = spectral.build_prefix_spectral_consistency(short, medium, long, detrend="none")
        for result in controls.values():
            self.assertTrue(result["exact_equal"])
            self.assertEqual(result["max_abs_difference"], 0.0)

    def test_dominant_peaks_exclude_dc(self) -> None:
        spectrum = np.array([100.0, 1.0, 10.0, 3.0], dtype=np.float64)
        rows = spectral.extract_dominant_peaks(
            spectrum=spectrum,
            frequencies=np.array([0.0, 0.25, 0.5, 0.75]),
            total_length=8,
            component="x",
            top_k=3,
        )
        self.assertEqual(rows[0]["discrete_index"], 2)
        self.assertTrue(all(row["discrete_index"] != 0 for row in rows))

    def test_triplet_analysis_returns_compact_mode_shift_evidence(self) -> None:
        short, medium, long = _triplet()
        analyses, rows, auxiliary = spectral.analyze_triplet(short, medium, long, retained_modes_lambda=2, detrend="none")
        self.assertEqual(set(analyses), {"T1200", "T1800", "T2400"})
        self.assertEqual(len(rows), 3 * 4 * 4)
        self.assertIn("mode_shift_evidence", auxiliary)


class OutputAndInterfaceTests(unittest.TestCase):
    def test_output_writer_refuses_overwrite_and_creates_only_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "spectral"
            spectral.write_output_artifacts(
                output_dir=output_dir,
                summary={"schema_version": "1.0"},
                band_rows=[
                    {
                        "total_length": 4,
                        "component": "x",
                        "band_label": "0_to_t2400_cutoff",
                        "frequency_start": 0.0,
                        "frequency_end": 0.25,
                        "energy": 1.0,
                        "energy_fraction": 1.0,
                    }
                ],
                peak_rows=[],
            )
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(spectral.OUTPUT_FILENAMES))
            self.assertFalse(any(path.suffix == ".npy" for path in output_dir.iterdir()))
            with (output_dir / "spectral_band_energy.csv").open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["component"], "x")
            with self.assertRaises(FileExistsError):
                spectral.write_output_artifacts(
                    output_dir=output_dir,
                    summary={},
                    band_rows=[],
                    peak_rows=[],
                )

    def test_cli_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--retained-modes-lambda", completed.stdout)
        self.assertIn("--detrend", completed.stdout)


if __name__ == "__main__":
    unittest.main()
