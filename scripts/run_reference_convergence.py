#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_reference_convergence.py
============================================================
Purpose
-------
Test whether the current Kerr RK4 orbit solver is numerically converged
under step-size refinement.

This script does NOT modify the training dataset or the production solver.
It mirrors the current integration loop only to add diagnostics:

1. Select representative trajectories from the test split.
2. Integrate each trajectory with refinement factors 1, 2, 4, 8.
3. Keep the same final Mino-parameter interval for all refinements.
4. Downsample refined trajectories to the original grid exactly.
5. Compare adjacent resolutions.
6. Estimate the observed convergence order.
7. Record radial/polar turning-point direction flips.
8. Save JSON, CSV, NPZ and diagnostic plots.

Important interpretation
------------------------
A fine-grid trajectory is not assumed to be exact. It is accepted as a
reference candidate only if adjacent refined solutions converge and the
finest-grid discrepancy is much smaller than the model error of interest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Make project imports available when the script is run from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.orbit_solver import (
    InitialState,
    KerrParams,
    clip_theta,
    outer_horizon_radius,
    polar_potential,
    radial_potential,
    rk4_step,
    spherical_to_cartesian,
)


@dataclass
class OrbitDiagnostics:
    """
    Diagnostics for one numerical integration.

    radial_flips / polar_flips:
        Number of sign reversals triggered near radial or polar turning points.

    min_radial_potential / min_polar_potential:
        Smallest potential values seen at stored grid points. Slightly negative
        values can appear from floating-point or event-location error; large
        negative values indicate a problematic trajectory.
    """

    radial_flips: int
    polar_flips: int
    min_radial_potential: float
    min_polar_potential: float
    elapsed_seconds: float


@dataclass
class SelectedCase:
    case_id: int
    test_index: int
    quantile: float
    vary_params: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RK4 step-refinement convergence tests."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="vary_Q__Q1.6_3__n5000__T1200__cfg1",
        help="Existing task under data/tasks/.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=3,
        help="Number of representative test trajectories.",
    )
    parser.add_argument(
        "--refinement-factors",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Integer refinement factors relative to the task step size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--turn-tol-r",
        type=float,
        default=1e-6,
        help="Radial turning-point threshold; matches the production solver.",
    )
    parser.add_argument(
        "--turn-tol-th",
        type=float,
        default=1e-6,
        help="Polar turning-point threshold; matches the production solver.",
    )
    return parser.parse_args()


def load_task(task_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    task_dir = PROJECT_ROOT / "data" / "tasks" / task_name
    meta_path = task_dir / "meta.json"
    dataset_path = task_dir / "dataset.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing task metadata: {meta_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing task dataset: {dataset_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    return meta, data


def choose_representative_cases(
    x_test: np.ndarray,
    vary_params_order: list[str],
    num_cases: int,
) -> list[SelectedCase]:
    """
    Choose cases at evenly spaced quantiles of the first varying parameter.

    For Q-only:
        num_cases=3 approximately selects low, middle and high Q.

    The cases are selected from the ACTUAL test split, which will later allow
    direct comparison with saved model predictions for the same trajectories.
    """
    if x_test.ndim != 2:
        raise ValueError(f"x_test must have shape [N,P], got {x_test.shape}")
    if x_test.shape[1] != len(vary_params_order):
        raise ValueError("x_test width and vary_params_order do not match.")
    if num_cases < 1 or num_cases > len(x_test):
        raise ValueError("num_cases must be between 1 and len(x_test).")

    first_param = x_test[:, 0]
    sorted_indices = np.argsort(first_param)
    quantiles = np.linspace(0.05, 0.95, num_cases)

    cases: list[SelectedCase] = []
    used: set[int] = set()

    for case_id, quantile in enumerate(quantiles):
        sorted_position = int(round(quantile * (len(sorted_indices) - 1)))
        test_index = int(sorted_indices[sorted_position])

        # Avoid accidental duplicates when the test set is small.
        while test_index in used and sorted_position + 1 < len(sorted_indices):
            sorted_position += 1
            test_index = int(sorted_indices[sorted_position])
        used.add(test_index)

        vary_params = {
            name: float(x_test[test_index, column])
            for column, name in enumerate(vary_params_order)
        }
        cases.append(
            SelectedCase(
                case_id=case_id,
                test_index=test_index,
                quantile=float(quantile),
                vary_params=vary_params,
            )
        )

    return cases


def build_physical_objects(
    fixed_params: dict[str, Any],
    vary_params: dict[str, float],
) -> tuple[KerrParams, InitialState, float]:
    full = dict(fixed_params)
    full.update(vary_params)

    p = KerrParams(
        M=float(full["M"]),
        a=float(full["a"]),
        E=float(full["E"]),
        Lz=float(full["Lz"]),
    )
    init = InitialState(
        r0=float(full["r0"]),
        theta0=float(full["theta0"]),
        phi0=float(full["phi0"]),
        sign_r=int(full["sign_r"]),
        sign_th=int(full["sign_th"]),
    )
    Q = float(full["Q"])
    return p, init, Q


def simulate_one_orbit_with_diagnostics(
    p: KerrParams,
    init: InitialState,
    Q: float,
    n_steps: int,
    step_size: float,
    turn_tol_r: float,
    turn_tol_th: float,
) -> tuple[dict[str, np.ndarray], OrbitDiagnostics]:
    """
    Mirror the production simulate_one_orbit() algorithm while recording
    turning-point events and minimum potential values.

    Why mirror instead of editing the production solver?
    ----------------------------------------------------
    A convergence experiment should not alter the code that generated the
    training data. Keeping diagnostics in a separate analysis script preserves
    reproducibility of previous experiments.
    """
    state = np.array([init.r0, init.theta0, init.phi0], dtype=np.float64)

    sign_r = int(np.sign(init.sign_r)) if init.sign_r != 0 else 1
    sign_th = int(np.sign(init.sign_th)) if init.sign_th != 0 else 1

    r_plus = outer_horizon_radius(p.M, p.a)
    lambda_grid = np.arange(n_steps, dtype=np.float64) * step_size
    sph = np.zeros((n_steps, 3), dtype=np.float64)
    xyz = np.zeros((n_steps, 3), dtype=np.float64)

    radial_flips = 0
    polar_flips = 0
    min_R = math.inf
    min_TH = math.inf

    start = time.perf_counter()

    for i in range(n_steps):
        r = float(state[0])
        theta = clip_theta(float(state[1]))
        phi = float(state[2])

        sph[i] = np.array([r, theta, phi], dtype=np.float64)
        xyz[i] = spherical_to_cartesian(r, theta, phi)

        R_now = radial_potential(r, p, Q)
        TH_now = polar_potential(theta, p, Q)
        min_R = min(min_R, float(R_now))
        min_TH = min(min_TH, float(TH_now))

        if i == n_steps - 1:
            break

        if r <= r_plus + 1e-3:
            raise RuntimeError(
                f"Orbit too close to horizon: r={r:.8f}, r_plus={r_plus:.8f}"
            )

        # These direction flips reproduce the current production logic.
        # They are event-like, nonsmooth operations; therefore their locations
        # must be inspected when interpreting the observed convergence order.
        if R_now <= turn_tol_r:
            sign_r *= -1
            radial_flips += 1

        if TH_now <= turn_tol_th:
            sign_th *= -1
            polar_flips += 1

        next_state = rk4_step(
            state=state,
            h=step_size,
            p=p,
            Q=Q,
            sign_r=sign_r,
            sign_th=sign_th,
        )

        if not np.all(np.isfinite(next_state)):
            raise RuntimeError("Numerical divergence: NaN or Inf encountered.")

        next_state[1] = clip_theta(float(next_state[1]))

        if next_state[0] <= r_plus + 1e-4:
            raise RuntimeError("Next RK4 step enters the horizon safety region.")

        state = next_state

    elapsed = time.perf_counter() - start

    return (
        {"lambda_grid": lambda_grid, "sph": sph, "xyz": xyz},
        OrbitDiagnostics(
            radial_flips=radial_flips,
            polar_flips=polar_flips,
            min_radial_potential=float(min_R),
            min_polar_potential=float(min_TH),
            elapsed_seconds=float(elapsed),
        ),
    )


def trajectory_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """
    Compare two trajectories sampled on the same lambda grid.

    relative_l2:
        ||approximation-reference||_2 / ||reference||_2
        with each trajectory flattened from [T,3].

    pointwise_distance:
        Euclidean distance in xyz at every lambda point.
    """
    if approximation.shape != reference.shape:
        raise ValueError(
            f"Trajectory shapes differ: {approximation.shape} vs {reference.shape}"
        )

    difference = approximation - reference
    denominator = np.linalg.norm(reference.reshape(-1))
    if denominator <= 0.0:
        raise ValueError("Reference trajectory has zero norm.")

    pointwise = np.linalg.norm(difference, axis=-1)

    return {
        "relative_l2": float(
            np.linalg.norm(difference.reshape(-1)) / denominator
        ),
        "mean_pointwise_distance": float(np.mean(pointwise)),
        "max_pointwise_distance": float(np.max(pointwise)),
        "final_point_distance": float(pointwise[-1]),
    }


def main() -> None:
    args = parse_args()

    factors = sorted(set(int(value) for value in args.refinement_factors))
    if factors[0] != 1:
        raise ValueError("refinement_factors must include 1.")
    if any(value <= 0 for value in factors):
        raise ValueError("refinement_factors must be positive.")
    if any(b % a != 0 for a, b in zip(factors[:-1], factors[1:])):
        raise ValueError(
            "Each refinement factor must be an integer multiple of the previous."
        )

    meta, data = load_task(args.task_name)
    task_spec = meta["task_spec"]

    vary_params_order = [str(name) for name in data["vary_params_order"].tolist()]
    x_test = np.asarray(data["x_test"], dtype=np.float64)

    cases = choose_representative_cases(
        x_test=x_test,
        vary_params_order=vary_params_order,
        num_cases=int(args.num_cases),
    )

    base_n_steps = int(task_spec["n_steps"])
    base_step = float(task_spec["step_size"])
    base_intervals = base_n_steps - 1
    lambda_max = base_intervals * base_step

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / "outputs"
        / "_reference_convergence"
        / args.task_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}

    print("=" * 78)
    print("RK4 REFERENCE CONVERGENCE EXPERIMENT")
    print("=" * 78)
    print(f"Task              : {args.task_name}")
    print(f"Base n_steps      : {base_n_steps}")
    print(f"Base step size    : {base_step}")
    print(f"Common lambda_max : {lambda_max}")
    print(f"Refinement factors: {factors}")
    print(f"Selected cases    : {len(cases)}")

    for case in cases:
        print("-" * 78)
        print(
            f"Case {case.case_id}: test_index={case.test_index}, "
            f"params={case.vary_params}"
        )

        p, init, Q = build_physical_objects(
            fixed_params=task_spec["fixed_params"],
            vary_params=case.vary_params,
        )

        # Check initial admissibility explicitly for a clear diagnostic.
        R0 = radial_potential(init.r0, p, Q)
        TH0 = polar_potential(init.theta0, p, Q)
        if R0 < -1e-10 or TH0 < -1e-10:
            raise RuntimeError(
                f"Initial state is inadmissible: R0={R0}, Theta0={TH0}"
            )

        solutions: dict[int, dict[str, np.ndarray]] = {}
        diagnostics: dict[int, OrbitDiagnostics] = {}

        for factor in factors:
            refined_step = base_step / factor
            refined_n_steps = base_intervals * factor + 1

            result, diag = simulate_one_orbit_with_diagnostics(
                p=p,
                init=init,
                Q=Q,
                n_steps=refined_n_steps,
                step_size=refined_step,
                turn_tol_r=float(args.turn_tol_r),
                turn_tol_th=float(args.turn_tol_th),
            )

            # Exact endpoint check: all resolutions must cover the same interval.
            actual_lambda_max = float(result["lambda_grid"][-1])
            if not np.isclose(actual_lambda_max, lambda_max, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"lambda_max mismatch: {actual_lambda_max} vs {lambda_max}"
                )

            solutions[factor] = result
            diagnostics[factor] = diag

            npz_payload[
                f"case_{case.case_id}_factor_{factor}_xyz"
            ] = result["xyz"]
            npz_payload[
                f"case_{case.case_id}_factor_{factor}_lambda"
            ] = result["lambda_grid"]

            print(
                f"  factor={factor:<2d} h={refined_step:.8f} "
                f"steps={refined_n_steps:<6d} "
                f"r_flips={diag.radial_flips:<3d} "
                f"th_flips={diag.polar_flips:<3d} "
                f"time={diag.elapsed_seconds:.3f}s"
            )

        case_pairwise: list[dict[str, Any]] = []

        for coarse_factor, fine_factor in zip(factors[:-1], factors[1:]):
            ratio = fine_factor // coarse_factor
            coarse_xyz = solutions[coarse_factor]["xyz"]
            fine_xyz_aligned = solutions[fine_factor]["xyz"][::ratio]

            if fine_xyz_aligned.shape != coarse_xyz.shape:
                raise RuntimeError(
                    "Downsampled fine trajectory does not match coarse shape."
                )

            metrics = trajectory_metrics(
                approximation=coarse_xyz,
                reference=fine_xyz_aligned,
            )

            row = {
                "case_id": case.case_id,
                "test_index": case.test_index,
                **case.vary_params,
                "coarse_factor": coarse_factor,
                "fine_factor": fine_factor,
                "coarse_step": base_step / coarse_factor,
                "fine_step": base_step / fine_factor,
                **metrics,
                "coarse_radial_flips": diagnostics[coarse_factor].radial_flips,
                "fine_radial_flips": diagnostics[fine_factor].radial_flips,
                "coarse_polar_flips": diagnostics[coarse_factor].polar_flips,
                "fine_polar_flips": diagnostics[fine_factor].polar_flips,
            }
            pairwise_rows.append(row)
            case_pairwise.append(row)

        # Observed order from adjacent pairwise relative errors.
        observed_orders: list[float | None] = []
        for previous, current in zip(case_pairwise[:-1], case_pairwise[1:]):
            previous_error = float(previous["relative_l2"])
            current_error = float(current["relative_l2"])
            refinement_ratio = (
                float(current["coarse_factor"])
                / float(previous["coarse_factor"])
            )
            if previous_error > 0.0 and current_error > 0.0:
                observed_orders.append(
                    float(
                        math.log(previous_error / current_error)
                        / math.log(refinement_ratio)
                    )
                )
            else:
                observed_orders.append(None)

        finest_pair_error = float(case_pairwise[-1]["relative_l2"])
        assumed_rk4_reference_error = finest_pair_error / (2.0**4 - 1.0)

        case_summary = {
            **asdict(case),
            "base_step": base_step,
            "lambda_max": lambda_max,
            "pairwise_metrics": case_pairwise,
            "observed_orders": observed_orders,
            "assumed_rk4_reference_relative_error": float(
                assumed_rk4_reference_error
            ),
            "diagnostics": {
                str(factor): asdict(diagnostics[factor]) for factor in factors
            },
        }
        case_summaries.append(case_summary)

    # Save numerical arrays for later comparison with model predictions.
    np.savez_compressed(output_dir / "convergence_trajectories.npz", **npz_payload)

    # Save selected test cases.
    (output_dir / "selected_cases.json").write_text(
        json.dumps([asdict(case) for case in cases], indent=2),
        encoding="utf-8",
    )

    # Save pairwise metrics as CSV.
    csv_path = output_dir / "pairwise_metrics.csv"
    if pairwise_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(pairwise_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(pairwise_rows)

    summary = {
        "task_name": args.task_name,
        "base_n_steps": base_n_steps,
        "base_step_size": base_step,
        "lambda_max": lambda_max,
        "refinement_factors": factors,
        "turn_tol_r": float(args.turn_tol_r),
        "turn_tol_th": float(args.turn_tol_th),
        "cases": case_summaries,
        "interpretation_note": (
            "The assumed RK4 reference-error estimate uses fourth-order "
            "Richardson scaling and is valid only if observed convergence "
            "and turning-point diagnostics support that assumption."
        ),
    }
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Plot relative differences between adjacent refinement levels.
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)

    for case in case_summaries:
        x_values = [
            item["fine_step"] for item in case["pairwise_metrics"]
        ]
        y_values = [
            item["relative_l2"] for item in case["pairwise_metrics"]
        ]
        label_params = ", ".join(
            f"{key}={value:.6g}"
            for key, value in case["vary_params"].items()
        )
        axis.loglog(
            x_values,
            y_values,
            marker="o",
            label=f"case {case['case_id']}: {label_params}",
        )

    axis.set_xlabel("Fine RK4 step size")
    axis.set_ylabel("Adjacent-grid trajectory Relative L2")
    axis.set_title("RK4 step-refinement convergence")
    axis.grid(True, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "convergence_relative_l2.png", dpi=200)
    plt.close(figure)

    print("-" * 78)
    print("Saved:")
    print(f"  {output_dir / 'selected_cases.json'}")
    print(f"  {output_dir / 'pairwise_metrics.csv'}")
    print(f"  {output_dir / 'convergence_summary.json'}")
    print(f"  {output_dir / 'convergence_trajectories.npz'}")
    print(f"  {output_dir / 'convergence_relative_l2.png'}")


if __name__ == "__main__":
    main()