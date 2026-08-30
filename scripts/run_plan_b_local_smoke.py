"""Run a tiny real-solver Plan B Protocol v1 smoke test outside formal asset paths."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


# 允许从项目根目录直接执行该脚本，同时不改变包导入路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.task_spec import TaskSpec
from src.data_generation.dataset_builder import build_initial_state, build_kerr_params
from src.data_generation.orbit_solver_second_order import simulate_one_orbit_second_order
from src.data_generation.plan_b_paired import (
    COARSE_N_STEPS,
    COARSE_STEP_SIZE,
    generate_paired_fine_dataset,
    protocol_grid,
    validate_plan_b_ground_truth,
    write_consistency_result_exclusively,
)


FIXED_PARAMS = {
    "M": 1.0,
    "a": 0.5,
    "E": 0.95,
    "Lz": 3.0,
    "r0": 10.0,
    "theta0": 1.2,
    "phi0": 0.0,
    "sign_r": -1,
    "sign_th": 1,
}


def build_parser() -> argparse.ArgumentParser:
    """构造 tiny real-solver smoke test 的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run a tiny Plan B v1 real-solver smoke test without formal Q400 generation."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--q-values", type=float, nargs="+", default=[1.8, 2.4])
    return parser


def _source_metadata(q_count: int) -> dict[str, object]:
    task_spec = TaskSpec(
        vary_params=["Q"],
        vary_ranges={"Q": (1.6, 3.0)},
        fixed_params=dict(FIXED_PARAMS),
        sample_shape=[max(2, q_count)],
        n_steps=COARSE_N_STEPS,
        step_size=COARSE_STEP_SIZE,
    )
    task_spec.metadata.update({
        "orbit_solver": "second_order_rk4",
        "orbit_solver_version": "v1",
        "task_name": "local_plan_b_real_solver_smoke_coarse",
    })
    return {
        "task_name": "local_plan_b_real_solver_smoke_coarse",
        "task_spec": task_spec.to_dict(),
        "integration": {
            "n_steps": COARSE_N_STEPS,
            "step_size": COARSE_STEP_SIZE,
            "lambda_min": 0.0,
            "lambda_max": 5.995,
        },
    }


def _write_real_coarse_source(directory: Path, q_values: list[float]) -> None:
    """只为本地 smoke 构造极小 coarse source；不进入正式任务资产路径。"""
    if not q_values:
        raise ValueError("At least one Q value is required.")
    directory.mkdir(parents=True, exist_ok=False)
    kerr_params = build_kerr_params(FIXED_PARAMS)
    initial_state = build_initial_state(FIXED_PARAMS)
    split_q = {
        "train": np.asarray(q_values[:1], dtype=np.float64).reshape(-1, 1),
        "val": np.asarray(q_values[1:2], dtype=np.float64).reshape(-1, 1),
        "test": np.asarray(q_values[2:], dtype=np.float64).reshape(-1, 1),
    }
    arrays: dict[str, np.ndarray] = {
        "vary_params_order": np.asarray(["Q"]),
        "lambda_grid": protocol_grid("coarse"),
    }
    diagnostics: list[dict[str, object]] = []
    for split, q_array in split_q.items():
        trajectories: list[np.ndarray] = []
        arrays[f"x_{split}"] = q_array
        for row_index, q_row in enumerate(q_array):
            q_value = float(q_row[0])
            orbit = simulate_one_orbit_second_order(
                p=kerr_params,
                init=initial_state,
                Q=q_value,
                n_steps=COARSE_N_STEPS,
                step_size=COARSE_STEP_SIZE,
            )
            trajectories.append(np.asarray(orbit["xyz"], dtype=np.float64))
            diagnostics.append({
                "split": split,
                "source_row_index": row_index,
                "Q": q_value,
                "success": True,
                "diagnostics": asdict(orbit["diagnostics"]),
            })
        arrays[f"y_{split}"] = (
            np.stack(trajectories, axis=0)
            if trajectories else np.empty((0, COARSE_N_STEPS, 3), dtype=np.float64)
        )
    np.savez_compressed(directory / "dataset.npz", **arrays)
    (directory / "meta.json").write_text(
        json.dumps(_source_metadata(len(q_values)), indent=2), encoding="utf-8", newline="\n"
    )
    (directory / "failed_samples.json").write_text("[]\n", encoding="utf-8", newline="\n")
    (directory / "solver_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8", newline="\n"
    )


def main() -> None:
    args = build_parser().parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke directory: {work_dir}")
    source_dir = work_dir / "source_coarse"
    fine_dir = work_dir / "fine_replay"
    work_dir.mkdir(parents=True, exist_ok=False)
    _write_real_coarse_source(source_dir, list(args.q_values))
    generate_paired_fine_dataset(source_dir, fine_dir, progress_every=1)
    result = validate_plan_b_ground_truth(source_dir, fine_dir)
    output_path = work_dir / "ground_truth_consistency.json"
    write_consistency_result_exclusively(result, output_path)
    print(f"Smoke source Q values: {list(args.q_values)}")
    print("Smoke coarse grid: T=1200, step_size=0.005")
    print("Smoke fine grid: T=2399, step_size=0.0025")
    print(f"Structural valid: {result['structural_valid']}")
    if result["numerical_metrics"] is not None:
        print(
            "Mean per-Q Relative L2: "
            f"{result['numerical_metrics']['relative_l2_summary']['mean']:.12e}"
        )
    print(f"Smoke result written: {output_path}")


if __name__ == "__main__":
    main()
