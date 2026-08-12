# Server Data and Experiment Registry

## 1. Purpose and Maintenance Rules

This registry is a concise index for cross-session project handoff. It helps
prevent duplicate data generation and accidental use of obsolete data. Full
logs, metrics, and detailed result files remain in their original dataset and
output paths.

Update this registry after each completed data-generation or experiment stage.
Entries distinguish metadata facts from project recommendations: metadata facts
describe recorded assets, while recommendations are provisional choices for
current project work and may change as validation progresses.

## 2. Status Labels

| Label | Meaning |
| --- | --- |
| CURRENT / VALIDATED | Preferred recorded asset for new work under its documented configuration. |
| COMPARISON-ONLY | Recorded asset intended for controlled comparison or evaluation, not primary training. |
| LEGACY / EXPERIMENTAL | Retained for provenance and historical comparison; not the default choice for new work. |
| DEPRECATED | Superseded asset retained only for traceability. No current entries. |
| PENDING VERIFICATION | Asset or output directory whose detailed result-file status has not been verified here. |

## 3. Current Recommended Dataset Selection

The following are current project recommendations, not immutable rules.

| Current use | Recommended path | Reason |
| --- | --- | --- |
| First real-data sparse reconstruction baseline smoke test | `data/tasks/q_1p6-3_n500_t1200/dataset.npz` | Small validated dataset for fast real-data validation. |
| Default later development dataset | `data/tasks/q_1p6-3_n2000_t1200/dataset.npz` | Validated development-scale dataset. |

## 4. Dataset Registry

### Current validated datasets

Shared metadata facts for this group:

- Varying parameter: `Q` in `[1.6, 3.0]`.
- Fixed parameters: `M=1.0`, `a=0.5`, `E=0.95`, `Lz=3.0`, `r0=10.0`,
  `theta0=1.2`, `phi0=0.0`, `sign_r=-1`, and `sign_th=1`.
- `n_steps=1200`, `step_size=0.005`, and `lambda_max=5.995`.
- `sampling_mode=grid`, `solver=second_order_rk4`, `solver_version=v1`, and
  `completion_policy=target_success`.
- Successful points are strictly uniform and no failed samples are recorded.

| Path | Status | Samples | Split | Q range | T | Step | Solver version | Sampling / completion | Intended use | Regeneration needed |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `data/tasks/q_1p6-3_n500_t1200` | CURRENT / VALIDATED | 500 | 350 / 75 / 75 | [1.6, 3.0] | 1200 | 0.005 | v1 | grid / target_success | Project recommendation: smoke tests and small real-data validation. | No |
| `data/tasks/q_1p6-3_n1000_t1200` | CURRENT / VALIDATED | 1000 | 700 / 150 / 150 | [1.6, 3.0] | 1200 | 0.005 | v1 | grid / target_success | Project recommendation: small-scale experiments. | No |
| `data/tasks/q_1p6-3_n2000_t1200` | CURRENT / VALIDATED | 2000 | 1400 / 300 / 300 | [1.6, 3.0] | 1200 | 0.005 | v1 | grid / target_success | Project recommendation: default development-scale dataset. | No |
| `data/tasks/q_1p6-3_n5000_t1200` | CURRENT / VALIDATED | 5000 | 3500 / 750 / 750 | [1.6, 3.0] | 1200 | 0.005 | v1 | grid / target_success | Project recommendation: scaling studies and large formal experiments. | No |
| `data/tasks/q_1p6-3_n8000_t1200` | CURRENT / VALIDATED | 8000 | 5600 / 1200 / 1200 | [1.6, 3.0] | 1200 | 0.005 | v1 | grid / target_success | Project recommendation: scaling studies and large formal experiments. | No |

### Comparison-only datasets

Shared metadata facts for this group:

- Varying parameter: `Q` in `[1.6007, 2.9993]`.
- Samples and split: 400 total; 280 / 60 / 60.
- `sampling_mode=grid`, `solver=second_order_rk4`, `solver_version=v1`, and
  `completion_policy=target_success`.
- Successful points are strictly uniform.

| Path | Status | Samples | Split | Q range | T | Step | Solver version | Sampling / completion | Intended use | Regeneration needed |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `data/tasks/q_1p6007-2p9993_n400_t1200` | COMPARISON-ONLY | 400 | 280 / 60 / 60 | [1.6007, 2.9993] | 1200 | 0.005 | v1 | grid / target_success | Shared offset-grid common test for fair model comparison. | No, unless experiment design changes |
| `data/tasks/q_1p6007-2p9993_n400_t1800` | COMPARISON-ONLY | 400 | 280 / 60 / 60 | [1.6007, 2.9993] | 1800 | 0.005 | v1 | grid / target_success | Length-extrapolation evaluation. | No, unless experiment design changes |
| `data/tasks/q_1p6007-2p9993_n400_t2400` | COMPARISON-ONLY | 400 | 280 / 60 / 60 | [1.6007, 2.9993] | 2400 | 0.005 | v1 | grid / target_success | Longer length-extrapolation evaluation. | No, unless experiment design changes |

These are comparison assets and must not be described as primary training
datasets.

### Legacy experimental datasets

| Path | Status | Samples | Split | Q range | T | Step | Solver version | Sampling / completion | Intended use | Regeneration needed |
| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |
| `data/tasks/vary_Q__Q1.6_3__n500__T1200__cfg1_secondorder_pilot` | LEGACY / EXPERIMENTAL | 500 | 350 / 75 / 75 | [1.6, 3.0] | 1200 | Not recorded in this registry | experimental_v1 | Config: `cfg1_secondorder_pilot` | Provenance and historical comparison. | No for provenance; do not use by default for new experiments |
| `data/tasks/vary_Q__Q1.6_3__n2000__T1200__cfg1_secondorder` | LEGACY / EXPERIMENTAL | 2000 | 1400 / 300 / 300 | [1.6, 3.0] | 1200 | Not recorded in this registry | experimental_v1 | Config: `cfg1_secondorder` | Provenance and historical comparison. | No for provenance; do not use by default for new experiments |

Legacy experimental datasets are retained for provenance and historical
comparison. Prefer solver-v1 target-success datasets for new experiments. These
assets are not described here as invalid or corrupted.

## 5. Experiment Output Registry

Detailed metrics remain in the JSON or CSV files inside each output directory.
This registry does not reproduce numerical conclusions.

| Path or path pattern | Experiment role | Dataset basis | Model directories | Main result files | Current status |
| --- | --- | --- | --- | --- | --- |
| `outputs/comparison/common_test__q_1p6007-2p9993_n400_t1200` | Shared common-test comparison | Offset-grid common test, n400 t1200 | 4 | `common_test_results.json`, `common_test_summary.json`, `common_test_dataset.npz` | COMPLETED |
| `outputs/comparison/common_test_e500_w64_n2000_n5000` | Selected cross-dataset comparison | n2000 and n5000 selection | 2 | `common_test_results.json`, `common_test_summary.json`, `common_test_dataset.npz` | COMPLETED |
| `outputs/comparison/depth_scale__fno2d_m16x32_w64_d{2,3,5,6}_e500` | Depth scaling study | Common-test dataset snapshot in each directory | 4 separate directories; 1 each | Each directory contains common-test JSON files and one dataset snapshot | COMPLETED |
| `outputs/comparison/queue_a__...` | Width and data-scale Queue A experiments | n500, n1000, n2000, and n5000; n2000 widths 16, 32, 48, and 80; generally d4 and e500 | One per listed directory | Each directory contains common-test JSON files and one dataset snapshot | COMPLETED |
| `outputs/comparison/queue_a_cross_scale_summary` | Aggregate cross-scale summary directory | Not recorded in this registry | Not recorded in this registry | Not verified here | PENDING VERIFICATION |
| `outputs/comparison/scale_experiments_2d` | Scale-experiment output directory | Not recorded in this registry | Not recorded in this registry | Not verified here | PENDING VERIFICATION |

## 6. Current Sparse Reconstruction Stage

The first implementation stage is complete:

- Configurable stride-based sparse sampling is available.
- Observed and hidden masks are available.
- Linear and PCHIP reconstruction baselines are available.
- Hidden-only MSE and Relative L2 metrics are available.
- A baseline evaluation CLI and unit tests are available.

Latest implementation commit: `dbbb6ac`.

Based on the recorded stage handoff, local, GitHub, and server are synchronized
at `dbbb6ac`. Server lightweight validation passed: 13 / 13 tests and CLI help.
Formal real-data Linear/PCHIP baseline sweep completed on q_1p6-3_n500_t1200 test split for strides 2, 4, 8, 16, and 32.

### Completed Linear/PCHIP baseline sweep

Dataset: `data/tasks/q_1p6-3_n500_t1200/dataset.npz`<br>
Split: `test`

| Stride | Observed points | Hidden points | Linear Relative L2 | PCHIP Relative L2 | Linear MSE | PCHIP MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 601 | 599 | 1.635318e-04 | 1.706526e-05 | 6.544541e-07 | 7.126903e-09 |
| 4 | 301 | 899 | 5.499782e-04 | 9.142176e-05 | 7.403059e-06 | 2.045594e-07 |
| 8 | 151 | 1049 | 2.038428e-03 | 4.779948e-04 | 1.017007e-04 | 5.592154e-06 |
| 16 | 76 | 1124 | 7.861464e-03 | 2.501225e-03 | 1.512669e-03 | 1.531237e-04 |
| 32 | 39 | 1161 | 3.054264e-02 | 1.493277e-02 | 2.283004e-02 | 5.457255e-03 |

### Current project recommendations for neural reconstruction

The following are current project decisions and recommendations, not immutable facts:

- Strides 2 and 4 are easy interpolation regimes.
- Stride 8 is a moderate regime.
- Stride 16 is the primary planned neural reconstruction comparison regime and the main comparison point.
- Stride 32 is the high-difficulty comparison regime.
- Future FNO and ResNet reconstruction experiments should prioritize strides 16 and 32.
- Stride 32 is retained so that a neural-model advantage can still be evaluated if stride 16 proves too easy.

## 7. Update Checklist

- Register each new dataset path and its provenance.
- Record solver version and sampling mode.
- Record success and failure counts.
- Record each experiment output path.
- Record status and the key conclusion.
- Update project-recommended datasets when appropriate.
- Mark superseded assets without deleting provenance.
- Record the relevant Git commit.

## 8. Evidence Used for This Version

Version 1 was built on 2026-08-04 from:

- `data/tasks/*/meta.json`
- `outputs/comparison` directory inventory
- Existing experiment JSON file presence
- Current Git history and server validation

This version does not claim that every numerical result has been re-audited.

PASS — registry draft ready for user review
