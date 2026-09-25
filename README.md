# FNO–Kerr

## Project overview

This project studies learning Kerr spacetime geodesic trajectories with Fourier Neural Operators (FNOs) and other neural architectures. The completed experiments learn the mapping from physical parameter Q and coordinate lambda to the trajectory xyz(lambda), with the remaining physical parameters and initial conditions fixed.

The research distinguishes **physical-domain length extrapolation**—extending the lambda domain at a fixed sampling step—from **fixed-domain cross-resolution generalization**—changing the sampling density within the same physical interval.

## Current research status

- **Plan A — physical-domain length extrapolation:** baseline evaluation, M1–M4 diagnostics, and R1–R3 redesign experiments are complete. The tested global FNO2D shows strong length sensitivity; no unique cause or complete repair has been established. Plan A is paused at the unstarted R4 redesign decision.
- **Plan B — FNO-only fixed-domain cross-resolution:** the initial coarse-to-fine comparison, bidirectional experiment, and resolution range through T4797 are complete. Frozen FNO2D transfer retains low error on the tested fixed-domain grids, without establishing exact resolution invariance.
- **Track A — six-model cross-resolution benchmark:** Phase I is complete, with **6 models / 12 training runs / 48 evaluation cells**. Native accuracy and resolution robustness show distinct, architecture-dependent behavior under this single-seed protocol.

## Main results

**If you only want the scientific results, start here.**

[Formal scientific results package — 2026-09-25](reports/fno_kerr_results_20260925/README.md)

The package connects the three research stages and links the dataset definitions, experiment protocols, formal results, conclusion boundaries, and provenance. It separates supported findings from unresolved mechanisms and avoids treating the benchmark as a universal architecture ranking.

## Repository structure

| Directory | Purpose |
| --- | --- |
| [src/](src/) | Reusable implementations for data generation, models, training, evaluation, and analysis |
| [scripts/](scripts/) | Experiment, analysis, and workflow entry points |
| [tests/](tests/) | Validation and regression tests |
| [reports/](reports/) | Formal result summaries and provenance/asset-inventory reports |
| [docs/](docs/) | Supporting documentation, including the research snapshot schema |

## Research records

Four project-level records preserve the scientific context beyond individual outputs:

| Record | Responsibility |
| --- | --- |
| [FNO_KERR_CURRENT_STATE.md](FNO_KERR_CURRENT_STATE.md) | Current research state, completed evidence, conclusions, limitations, and where work should resume |
| [FNO_KERR_EXPERIMENT_PLAN.md](FNO_KERR_EXPERIMENT_PLAN.md) | Authoritative experiment protocols, locked controls, evaluation definitions, and evidence requirements |
| [FNO_KERR_REASONING_LOG.md](FNO_KERR_REASONING_LOG.md) | How observations led to candidate explanations, discriminating diagnostics, and changes in interpretation |
| [SERVER_DATA_EXPERIMENT_REGISTRY.md](SERVER_DATA_EXPERIMENT_REGISTRY.md) | Dataset, checkpoint, and output identities, locations, status, and provenance; distinguishes existing assets from planned work |

The records complement the concise results package. The reasoning log preserves the research process; the registry indexes formal assets rather than replacing their contents.

## Results and audit reports

- [reports/fno_kerr_results_20260925/](reports/fno_kerr_results_20260925/README.md): mentor-facing formal results and dataset definitions; the recommended scientific entry point.
- [reports/plan_a_inventory_20260921/](reports/plan_a_inventory_20260921/README.md): Plan A provenance and asset audit, including evidence availability and asset gaps.
- [reports/cross_resolution_inventory_20260920/](reports/cross_resolution_inventory_20260920/README.md): cross-resolution provenance and asset audit, with dataset/run inventories and package manifests.

Audit reports establish what assets exist and how they relate; they serve a different purpose from the scientific result summaries.

## Main execution entry points

| Entry point | Purpose |
| --- | --- |
| [scripts/generate_dataset.py](scripts/generate_dataset.py) | Unified trajectory dataset generation |
| [scripts/evaluate_formal_length_extrapolation_2d.py](scripts/evaluate_formal_length_extrapolation_2d.py) | Formal FNO2D physical-domain length-extrapolation evaluation |
| [scripts/run_plan_b_bidirectional_reverse_2d.py](scripts/run_plan_b_bidirectional_reverse_2d.py) | Plan B matched fine-training and reverse-transfer workflow |
| [scripts/run_plan_b_resolution_range_sweep_2d.py](scripts/run_plan_b_resolution_range_sweep_2d.py) | Plan B fixed-domain resolution-range workflow |
| [scripts/run_benchmark_track_a.py](scripts/run_benchmark_track_a.py) | Track A inspection, individual training runs, and frozen evaluation |

Formal execution depends on the registered server assets and the corresponding locked protocol. Consult the experiment plan and each entry point's arguments before reproducing a run.

## Environment

The recorded server environment is documented by [requirements_fno_srv_2026-07-23.txt](requirements_fno_srv_2026-07-23.txt) and [environment_fno_srv_2026-07-23.yml](environment_fno_srv_2026-07-23.yml).

[LOCAL_ENVIRONMENT.md](LOCAL_ENVIRONMENT.md) describes the separate lightweight local development and validation environment. Local checks and formal server experiments have different runtime and resource requirements.

## Data policy

Raw datasets, predictions, checkpoints, and large outputs are intentionally not tracked in Git. They remain in registered server locations, principally `data/tasks/` and `outputs/`; [.gitignore](.gitignore) excludes data/output directories and model/array files.

The GitHub repository retains source code, tests, protocols, summaries, and provenance so that the research can be reviewed without downloading large artifacts. A referenced server asset path does not imply that the asset is included in a Git clone. Consult the registry and audit reports for asset identities and availability.

## Recommended reading order

1. This README.
2. [Formal scientific results package](reports/fno_kerr_results_20260925/README.md), then its linked data definitions and three result summaries.
3. [FNO_KERR_CURRENT_STATE.md](FNO_KERR_CURRENT_STATE.md).
4. [FNO_KERR_EXPERIMENT_PLAN.md](FNO_KERR_EXPERIMENT_PLAN.md).
5. [FNO_KERR_REASONING_LOG.md](FNO_KERR_REASONING_LOG.md).
6. [SERVER_DATA_EXPERIMENT_REGISTRY.md](SERVER_DATA_EXPERIMENT_REGISTRY.md).
