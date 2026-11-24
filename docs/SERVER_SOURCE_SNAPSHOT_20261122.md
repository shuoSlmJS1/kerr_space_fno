# School-server source snapshot — November 22, 2026

Revalidated on November 24 against the replacement compute container: all 214
freshly downloaded files match the original source hashes, with no added or missing
files. See the [revalidation receipt](SERVER_REVALIDATION_20261124.json).

## Scope and provenance

This publication synchronizes the school-server core code to the existing GitHub
repository, preserving its previous commits and historical reports. It also adds
the detailed November research reports prepared from server evidence.

The source collection contains **214 files**, approximately **3.02 MB** before
compression: core source, entry points, tests, research documents, environment
records, and small experiment configuration/provenance files. It includes
**166 Python files**. It does not include datasets, trained weights, prediction
arrays, training histories, or complete experiment output directories.

[server_source_manifest_20261122.json](server_source_manifest_20261122.json)
records the original path, size, and SHA-256 of every downloaded server file,
alongside its published repository path and hash. Server files were read only.
The existing model, solver, training, and metric implementations were not redesigned.

Publication preparation converts CRLF to LF according to `.gitattributes`, removes
trailing blank lines in newly added files, makes Markdown hard breaks explicit,
retains the executable modes already tracked by Git, and adds navigation and ignore rules.
The manifest distinguishes exact source bytes from published bytes. A snapshot hash
proves which file was downloaded; it does not retroactively prove which historical
version produced every experiment result.

## Reading map

| Area | Entry point | What to inspect |
| --- | --- | --- |
| Latest research narrative | [Reports](../reports/README.md) | Timeline, protocols, results, limitations and migration history |
| Current experimental notes | [Paper experiments and plan](FNO_KERR_论文实验记录与计划.md) | Recent experiment record retained from the server |
| Kerr ground truth | [Second-order solver](../src/data_generation/orbit_solver_second_order.py), [dataset generation](../scripts/generate_dataset.py) | Dynamics, sampling, numerical integration and saved dataset structure |
| FNO implementations | [FNO1D](../src/models/fno1d/fno1d.py), [FNO2D](../src/models/fno2d/fno2d.py), [physical-frequency FNO2D](../src/models/fno2d/fno2d_physical_frequency.py) | Spectral layers, channel dimensions and coordinate handling |
| Six-model benchmark | [Models](../src/models/benchmark_track_a.py), [runner](../scripts/run_benchmark_track_a.py), [training](../src/training/benchmark_track_a.py) | Architecture comparison, normalization, training and evaluation |
| Sparse reconstruction | [Implementation directory](../src/training/trajectory_reconstruction/) | Sparse sampling, interpolation baselines, masked metrics and reconstruction |
| P1 spline in Q | [Spline implementation](../experiments/server_snapshot/external_outputs/priority1_spline_t1200_val_20261011T0341Z/spline_in_q_priority1.py) | Interpolation across the physical parameter Q; distinct from interpolation across sparse trajectory samples |
| P1 plain MLP | [MLP runner](../experiments/server_snapshot/external_outputs/priority1_mlp_t1200_seed27_20261024T014339Z/code/p1_mlp_runner.py) | Pointwise network, normalization, optimizer and resumable checkpoint logic |
| P1 evaluation and timing | [Q400 evaluation](../experiments/server_snapshot/external_outputs/priority1_q400_comparison_t1200/evaluate_p1_q400.py), [timing](../experiments/server_snapshot/external_outputs/priority1_timing_t1200_20261024T0643Z/benchmark_p1_t1200.py) | Evaluation contract and measured execution stages |
| P2 Track A multi-seed | [Stage 1 control](../experiments/server_snapshot/project_outputs/p2_stage1_20261024T0948Z/control/) | Frozen protocol, runner, supervisors and statistics |
| P2 FNO2D replay and completion | [Replay](../experiments/server_snapshot/project_outputs/p2_fno2d_replay_20261107/), [remaining runs](../experiments/server_snapshot/project_outputs/p2_fno2d_remaining_20261107_v2/), [C completion](../experiments/server_snapshot/project_outputs/p2_c_completion_20261121/) | Replay verification, added seeds and final completion wrapper |

## Original locations and migration

| Repository location | Source location |
| --- | --- |
| `src/`, `scripts/`, `tests/`, `docs/`, root research records | `/workspace/shanjinshuo/projects/fno_kerr/kerr_project/` |
| `experiments/server_snapshot/project_outputs/` | `/workspace/shanjinshuo/projects/fno_kerr/kerr_project/outputs/` |
| `experiments/server_snapshot/external_outputs/` | `/workspace/shanjinshuo/outputs/fno_kerr/` |

Historical records also reference `/home/shanjinshuo/fno_kerr/kerr_project` and
the old `fno_srv` environment. The user confirmed that the research moved between
servers. These historical paths are preserved as provenance, not rewritten as
though all experiments ran on the current server.

## Reading versus reproducing

Cloning the repository is sufficient to read the implementations, tests, protocols,
small configurations and reports:

```bash
git clone https://github.com/shuoSlmJS1/kerr_space_fno.git
cd kerr_space_fno
```

The recorded Linux dependency exports are
[requirements_fno_srv_2026-07-23.txt](../requirements_fno_srv_2026-07-23.txt) and
[environment_fno_srv_2026-07-23.yml](../environment_fno_srv_2026-07-23.yml).
They are historical server snapshots, not a verified cross-platform installer.
[LOCAL_ENVIRONMENT.md](../LOCAL_ENVIRONMENT.md) describes the historical Windows
development machine, not the Mac used for this publication.

After preparing a compatible environment, CLI documentation can be inspected with:

```bash
python scripts/generate_dataset.py --help
python scripts/train_model_2d.py --help
python scripts/run_benchmark_track_a.py --help
```

These commands are reading aids and were not used as reproduction validation in
this publication. Formal reproduction additionally requires the registered datasets,
their metadata and split definitions, and, for frozen evaluation, the correct BEST
checkpoints. Use the experiment plan and the detailed configuration report to select
an exact run; generating an arbitrary replacement dataset would change the experiment.

The files under `experiments/server_snapshot/` are **archived execution sources**.
They retain original absolute paths, parent-directory assumptions, deadlines,
provenance checks and dependencies on runtime status files. Moving them into this
archive does not make their launch commands portable. Some modules read configuration
files at import time. Do not treat them as a ready-to-run experiment queue.

In particular, the historical `transfer_p1_assets.py` depends on an external helper
named `transfer_yolo_d1`; that unrelated transport helper is not bundled. It is
retained only to document the P1 asset preparation history. No SSH credentials or
authentication material are included.

## Validation performed

- Every one of the 214 downloaded files was checked against its source SHA-256.
- All 166 Python files passed AST syntax parsing on the server without executing
  experiment code or importing model modules.
- The publication checks JSON syntax, tracked file types/sizes, source hashes,
  documentation links, and Git whitespace errors.
- Full unit tests, model forward passes, training, and scientific reproduction were
  not run for this transfer. No local `fno_codex_local` environment was found in the
  usual installation locations; no environment or dependency was installed or changed.

The server remains the location of the full research assets. The detailed reports
explain which historical results have direct artifacts and which rely on retained
summaries or metadata.
