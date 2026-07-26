#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_second_order_solver.py
============================================================
验证二阶 Kerr 轨道求解器：

1. 不需要 event / sign flip / coordinate nudge；
2. 检查速度自然跨零次数；
3. 检查第一积分约束漂移；
4. 比较 h, h/2, h/4, h/8 的轨道收敛。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.orbit_types import InitialState, KerrParams
from src.data_generation.orbit_solver_second_order import (
    simulate_one_orbit_second_order,
)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm((a - b).reshape(-1))
        / np.linalg.norm(b.reshape(-1))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        default="vary_Q__Q1.6_3__n5000__T1200__cfg1",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=103,
    )
    parser.add_argument(
        "--refinement-factors",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    args = parser.parse_args()

    task_dir = PROJECT_ROOT / "data" / "tasks" / args.task_name
    meta = json.loads(
        (task_dir / "meta.json").read_text(encoding="utf-8")
    )

    with np.load(
        task_dir / "dataset.npz",
        allow_pickle=False,
    ) as loaded:
        x_test = np.asarray(loaded["x_test"], dtype=np.float64)
        names = [
            str(x) for x in loaded["vary_params_order"].tolist()
        ]

    spec = meta["task_spec"]
    vary = {
        name: float(x_test[args.test_index, column])
        for column, name in enumerate(names)
    }
    full = dict(spec["fixed_params"])
    full.update(vary)

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

    base_steps = int(spec["n_steps"])
    base_h = float(spec["step_size"])
    base_intervals = base_steps - 1

    factors = sorted(set(args.refinement_factors))
    solutions = {}

    print("=" * 92)
    print("SECOND-ORDER SOLVER CONVERGENCE")
    print(f"task={args.task_name}")
    print(f"test_index={args.test_index}, params={vary}")

    for factor in factors:
        h = base_h / factor
        n_steps = base_intervals * factor + 1

        result = simulate_one_orbit_second_order(
            p=p,
            init=init,
            Q=Q,
            n_steps=n_steps,
            step_size=h,
        )
        solutions[factor] = result
        diag = result["diagnostics"]

        print(
            f"factor={factor:<2d} h={h:.8f} "
            f"turns=({diag.radial_velocity_zero_crossings},"
            f"{diag.polar_velocity_zero_crossings}) "
            f"max_constraint=("
            f"{diag.max_radial_constraint_residual:.3e},"
            f"{diag.max_polar_constraint_residual:.3e}) "
            f"min_potential=("
            f"{diag.min_radial_potential:.3e},"
            f"{diag.min_polar_potential:.3e})"
        )

    pairwise_errors = []

    print("adjacent-grid Relative L2:")
    for coarse, fine in zip(factors[:-1], factors[1:]):
        ratio = fine // coarse
        coarse_xyz = solutions[coarse]["xyz"]
        fine_xyz = solutions[fine]["xyz"][::ratio]

        error = relative_l2(coarse_xyz, fine_xyz)
        pairwise_errors.append(error)

        print(f"  factor {coarse} -> {fine}: {error:.8e}")

    observed_orders = []
    for previous, current in zip(
        pairwise_errors[:-1],
        pairwise_errors[1:],
    ):
        if previous > 0.0 and current > 0.0:
            observed_orders.append(
                math.log(previous / current, 2.0)
            )

    print(f"observed orders: {observed_orders}")


if __name__ == "__main__":
    main()
