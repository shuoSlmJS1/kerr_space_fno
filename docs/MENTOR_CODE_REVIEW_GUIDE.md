# Mentor code review guide and persistent-storage audit

Audit date: 2026-11-24. GitHub baseline inspected: `084b6943ea3fb748c69667d094a5b3b5435b07a7`.

## What this audit establishes

The compute container is a temporary access endpoint to persistent storage. A new
container is not evidence of a new code version. This audit used file contents and
SHA-256 hashes, not container identity or modification dates.

The earlier 214-file publication covered the main project and P1/P2 experiment
directories. Its revalidation was correct **within that selection**, but it did not
establish coverage of the recovery, numerical-diagnostic, and environment directories.
The expanded audit found and now includes 10 additional Python scripts from those
locations. This is a correction to the earlier collection scope, not a new scientific
experiment or a model implementation change.

| Inspected location under `/workspace/shanjinshuo/` | File instances inventoried |
| --- | ---: |
| `projects/fno_kerr/kerr_project/` | 1,362 |
| `outputs/fno_kerr/` | 1,661 |
| `datasets/fno_kerr/` | 7 |
| `logs/fno-recovery-20261011T0134Z/` | 193 |
| `logs/fno-numeric-diagnostic-20261011T014713Z/` | 26 |
| `environment-locks/fno-phase4-20261010T0755Z/` | 38 |
| **Total** | **3,287** |

These counts include dataset/checkpoint filenames and sizes; their binary contents
were not downloaded. Package caches, installed environments, unrelated projects,
and cache directories were excluded. The environment-lock audit inspected its top-level
records and scripts, not downloaded package internals. A historical Markdown `.bak`
file is not a separate executable implementation and was not republished.

The inventory identified **331 code file instances**: **317 Python files and 14 shell
scripts**. Of these, 321 match already published source content, including the 148
code files in the recovery staging snapshot. The remaining 10 are now archived.
All 211 previously collected files eligible for this expanded text scan matched
their previous source hashes; the three root Git/editor dotfiles were outside this
scan's text-extension selection and were already covered by the earlier 214-file
revalidation. No scientific code changes were found in the main project.

The [machine-readable audit](PERSISTENT_STORAGE_CODE_AUDIT_20261124.json) maps every
one of the 331 code instances to a repository file and records both source and
published hashes. Identical recovery copies are mapped to existing files instead
of duplicating the entire source tree. The `added_files` list records the 70 copied
files: 10 scripts and 60 supporting text/configuration/result files. Whitespace-only
publication transformations are reflected in separate source and published hashes.

## Start with the research, then trace the code

The detailed [research reports](../reports/README.md) explain the full available
timeline, experimental settings, result matrices and evidence limitations. Their
evidence index contains original server paths; those paths are provenance references,
not promises that every referenced asset is included in this repository.

| Research stage | Code to read | Main review questions |
| --- | --- | --- |
| Physical reference and initial trajectory learning | [Second-order solver](../src/data_generation/orbit_solver_second_order.py), [dataset builder](../src/data_generation/dataset_builder.py), [generation CLI](../scripts/generate_dataset.py) | Initial conditions, turning points, integration step, validity tests, coordinate conversion and train/test identity |
| FNO2D capacity and dataset-scale exploration | [FNO2D](../src/models/fno2d/fno2d.py), [training CLI](../scripts/train_model_2d.py), [width sweep](../scripts/run_width_scale_experiments_2d.sh), [depth sweep](../scripts/run_depth_experiments_2d.sh) | Which comparisons hold data and evaluation Q fixed; actual run configurations versus CLI defaults |
| Sparse trajectory reconstruction | [Reconstruction utilities](../src/training/trajectory_reconstruction/), [baseline evaluation](../scripts/evaluate_sparse_reconstruction_baselines.py), [cross-resolution evaluation](../scripts/evaluate_sparse_reconstruction_cross_resolution.py) | Observed/hidden masks, no Q input, observed-point restoration, masked metrics and possible information leakage |
| TimesNet representation diagnostics | [TimesNet](../src/models/timesnet1d.py), [frequency diagnostics](../scripts/analyze_timesnet_frequency_diagnostics.py), [isolated lambda model](../src/models/timesnet_lambda_isolated1d.py) | Batch-dependent period selection, coordinate channels and the meaning of ablations |
| Plan A: extend physical domain at fixed step | [Formal evaluation](../scripts/evaluate_formal_length_extrapolation_2d.py), [prefix identity](../scripts/validate_length_dataset_prefix_identity.py), [internal sensitivity](../scripts/diagnose_fno2d_internal_length_sensitivity.py), [coordinate clamp](../scripts/diagnose_lambda_coordinate_clamp_effect_2d.py) | Shared-prefix truth identity; distinguish model length sensitivity from changed truth or Q order |
| Plan A redesigns R1/R2/R3 | [R1](../scripts/train_fno2d_variable_length_r1.py), [R2](../scripts/train_fno2d_domain_conditioned_r2.py), [R3](../scripts/train_fno2d_physical_frequency_r3.py), [physical-frequency layers](../src/models/fno2d/physical_frequency_layers2d.py) | Training-length selection, domain conditioning, frequency anchors and precision, validation/BEST policy |
| Plan B: fixed-domain resolution transfer | [Paired data](../src/data_generation/plan_b_paired.py), [evaluation](../scripts/evaluate_plan_b_resolution_generalization_2d.py), [reverse transfer](../scripts/run_plan_b_bidirectional_reverse_2d.py), [range sweep](../scripts/run_plan_b_resolution_range_sweep_2d.py) | Fixed endpoints, shared-node correspondence, matched Q splits and frozen normalization |
| Track A: six-model benchmark | [Architectures](../src/models/benchmark_track_a.py), [data](../src/training/benchmark_data.py), [training/evaluation](../src/training/benchmark_track_a.py), [CLI](../scripts/run_benchmark_track_a.py) | Information available to each architecture, parameter matching, float64 statistics, tensor layout, batch semantics and resolution adaptation |
| P1: simple baselines and cost | [Spline in Q](../experiments/server_snapshot/external_outputs/priority1_spline_t1200_val_20261011T0341Z/), [plain MLP](../experiments/server_snapshot/external_outputs/priority1_mlp_t1200_seed27_20261024T014339Z/code/), [Q400 comparison](../experiments/server_snapshot/external_outputs/priority1_q400_comparison_t1200/), [timing](../experiments/server_snapshot/external_outputs/priority1_timing_t1200_20261024T0643Z/) | Distinguish Q interpolation from lambda interpolation; raw versus normalized metrics; same test set; timed stages and synchronization |
| P2: multi-seed completion | [Track A control](../experiments/server_snapshot/project_outputs/p2_stage1_20261024T0948Z/control/), [FNO2D replay](../experiments/server_snapshot/project_outputs/p2_fno2d_replay_20261107/), [final C completion](../experiments/server_snapshot/project_outputs/p2_c_completion_20261121/) | Historical seed27 versus new seeds41/73, protocol matching, recovery, statistics and evidence availability |
| Migration and numerical reproducibility | [Recovery](../experiments/server_snapshot/recovery/), [numerical diagnosis](../experiments/server_snapshot/numeric_diagnostic/), [environment](../experiments/server_snapshot/environment/) | Same source/BEST/input, matching evaluation batch, device/kernel effects and explicit limits of reproduced predictions |

## Newly recovered code and what it did

| Added script | Historical purpose |
| --- | --- |
| `recovery/verify_fno_checkpoint.py` | Verify a restored FNO checkpoint and source hashes, reconstruct its input path, and compare its prediction with retained historical output |
| `recovery/smoke_fno_forward.py` | Bounded forward smoke and source checks after migration |
| `numeric_diagnostic/diagnose_fno_numerics.py` | Run an isolated CPU/GPU and batch-size diagnostic with fixed BEST and canonical inputs |
| `numeric_diagnostic/run_fno_numeric_condition.py` | Launch one bounded condition and collect its process/metric evidence |
| `numeric_diagnostic/summarize_fno_numerics.py` | Compare conditions and summarize numerical agreement without redefining the tolerance |
| `environment/install_fno_phase4.py` | Historical FNO environment installation with pinned packages, hashes and bounded downloads |
| `environment/installer_common.py` | Shared installer utilities used by the FNO installation entry point |
| `environment/resume_fno_phase4.py` | Resume a stopped installation using retained state |
| `environment/smoke_fno_cpu.py` | CPU package, tensor, FFT and import checks |
| `environment/test_fno_download.py` | Tests for downloader behavior and artifact validation |

All paths above are relative to `experiments/server_snapshot/`. The installer helper
retains defaults referring to another historical environment; the FNO entry point
overrides its `RUN`, `CACHE`, `WHEELS`, `PREFIX` and arguments. It is included because
the FNO scripts import it. **Do not execute the helper as an independent installer.**
These scripts are historical execution records and retain absolute paths, deadlines,
state prerequisites and platform dependencies. They have not been converted into a
portable installation command, and this audit did not execute them.

## A migration diagnostic missing from the earlier source package

The retained [diagnostic report](../experiments/server_snapshot/numeric_diagnostic/diagnostic-report.json)
documents three forwards on 2026-10-11 with the same saved BEST and canonical input:

| Condition | Index-0 relative L2 versus historical prediction | Allclose at locked tolerance |
| --- | ---: | --- |
| CPU, batch 1 | 9.909573517610805e-5 | false |
| GPU, batch 1 | 9.795989899153952e-5 | false |
| GPU, batch 32 | 0 | true |

The GPU batch32 output matched all 32 selected historical rows exactly in that
record (115,200 scalars). Tolerances remained `atol=1e-5`, `rtol=1e-4`; they were not
relaxed to declare success. The comparison is against **historical model predictions**,
not geodesic ground truth. It therefore measures repeatability, not physical accuracy.
It does not prove full Q400 reproduction, behavior of the final short batch, or a
universal hardware-independent numerical equivalence. Historical runtime flags were
not all recorded, so a unique low-level cause is not established by this audit.

This investigation is distinct from the much larger spline/MLP/FNO accuracy gap in
P1. The earlier small forward discrepancy should not be described as an unresolved
global corruption of the migrated model, nor should the 32-row match be generalized
to every historical experiment.

Supporting files now also include P1 metrics and timing summaries, P2 aggregate
statistics, the statistics-field alias correction record (`global_mse` to reported
`mse`), final C acceptance statistics, dataset metadata and the migrated environment
pins. They supplement the reports without publishing datasets or predictions.

## Completeness and reproduction boundaries

- All identified executable source files in the six inspected locations have a
  content match in the publication after these additions. This is not a claim that
  every historical file on an unavailable previous server has been recovered.
- Binary bundles, datasets, weights, predictions, package caches, transient locks,
  partial files and full training histories are excluded. Archive contents were not
  recursively treated as independent projects.
- Original Git history remains available for earlier committed implementations;
  the current snapshot is not evidence that every historical run used today's code.
- `transfer_p1_assets.py` still refers to the external `transfer_yolo_d1` transport
  helper. That helper was not found in these FNO locations and is not bundled. It is
  an asset-transfer dependency, not a model/solver implementation. No unrelated
  project was searched to obtain it.
- Archived wrappers depend on original data/checkpoint paths and sometimes prior
  status files. Their presence makes the historical orchestration reviewable; it
  does not make the clone a one-command reproduction environment.
- Running formal experiments requires restoring the exact registered datasets and
  BEST assets or explicitly regenerating their specified inputs, then checking
  hashes, splits, normalization, metric conventions and runtime settings.
- This was a coverage/provenance audit, not a finding that the research code is
  bug-free. The questions in the reading map are review prompts, not confirmed bugs.

## Checks performed for this publication

The remote inventory parsed all 317 Python source instances with `ast.parse`, without
importing or executing experiment modules. Downloaded text files were hash-verified.
Publication checks verify code-to-repository mappings, JSON syntax, documentation
links, whitespace and excluded binary/credential patterns. Full unit tests, new model
forwards, training and scientific reproduction were not run. Historical smoke/diagnostic
JSON files describe their historical execution dates, not tests performed today.
