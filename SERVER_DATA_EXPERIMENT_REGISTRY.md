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

### Completed Dilated ResNet1D same-resolution runs

Both runs used `data/tasks/q_1p6-3_n500_t1200/dataset.npz` with the existing
train / validation / test split, no Q input, train-only reconstruction
normalization, normalized-space hidden-only MSE training loss, validation raw
hidden-only overall Relative L2 checkpoint selection, and raw observed-point
restoration before hidden-only test metrics.

| Stride | Output directory | Best validation hidden Relative L2 | Test raw hidden Relative L2 | Status |
|---:|---|---:|---:|---|
| 16 | `outputs/sparse_reconstruction_resnet1d/q500_t1200_stride16_resnet1d_w92_b9_e600_seed42` | 1.931149e-03 | 1.870324e-03 | COMPLETED |
| 32 | `outputs/sparse_reconstruction_resnet1d/q500_t1200_stride32_resnet1d_w92_b9_e600_seed42` | 1.752349e-03 | 1.693836e-03 | COMPLETED |

Formal configuration shared by both runs, except for stride:

- Model: Dilated ResNet1D with full-trajectory theoretical receptive field.
- Epochs: 600; batch size: 32; seed: 42.
- Width: 92; residual blocks: 9; kernel size: 7.
- Dilation schedule: `[1, 2, 4, 8, 16, 32, 64, 128, 256]`.
- Theoretical receptive field: 3121; trainable parameters: 1,077,507.
- Optimizer: AdamW with learning rate `1e-3` and weight decay `1e-4`.
- Scheduler: ExponentialLR with gamma `0.995`.
- Checkpoint criterion: validation raw hidden-only overall Relative L2.
- Q input: excluded.

The epoch-600 train / validation hidden MSE and validation hidden Relative L2
were `4.016453e-06` / `4.023694e-06` / `1.971923e-03` for stride 16, and
`3.509473e-06` / `3.539195e-06` / `1.855690e-03` for stride 32.

### Completed canonical TimesNet1D same-resolution runs

Both runs used `data/tasks/q_1p6-3_n500_t1200/dataset.npz` with the existing
train / validation / test split, the five-channel input
`[sparse_x, sparse_y, sparse_z, observed_mask, lambda_coordinate]`, Q excluded,
train-only reconstruction normalization, normalized-space hidden-only MSE,
validation raw hidden-only overall Relative L2 checkpoint selection, best
checkpoint reload before test, and raw observed-point restoration before
hidden-only metrics.

| Stride | Output directory | Best validation hidden Relative L2 | Test raw hidden Relative L2 | Status |
|---:|---|---:|---:|---|
| 16 | `outputs/sparse_reconstruction_timesnet1d/q500_t1200_stride16_timesnet1d_dm80_df96_b2_k2_e600_seed42` | 1.697399e-02 | 1.657611e-02 | COMPLETED |
| 32 | `outputs/sparse_reconstruction_timesnet1d/q500_t1200_stride32_timesnet1d_dm80_df96_b2_k2_e600_seed42` | 1.542742e-01 | 1.510641e-01 | COMPLETED |

Formal configuration shared by both runs, except for stride:

- Model: canonical TimesNet1D; `d_model=80`, `d_ff=96`, blocks `2`, top-k `2`,
  and Inception kernels `[1, 3, 5]`.
- Trainable parameters: `1,077,299`; epochs: 600; batch size: 32; seed: 42.
- Optimizer: AdamW with learning rate `1e-3` and weight decay `1e-4`.
- Scheduler: ExponentialLR with gamma `0.995`.

### Canonical TimesNet frequency and spectrum diagnostics

Formal latent-frequency diagnostic outputs are
`outputs/timesnet_frequency_diagnostics/stride16_test.json` and
`outputs/timesnet_frequency_diagnostics/stride32_test.json`.

- At stride 16, both TimesBlocks selected `f=1` (period 1200) and `f=3`
  (period 400) in every test batch.
- At stride 32, block 0 selected `f=1` (period 1200) and `f=3` (period 400)
  in every test batch; block 1 selected `f=1` (period 1200) and `f=2`
  (period 600) in every test batch.
- Sampling-stride-related periods 16 and 32 were not selected in these
  canonical latent top-k diagnostics. This does not prove that the observed
  mask has no influence.

Raw-input spectrum outputs are
`outputs/timesnet_raw_input_spectrum/stride16_test.json` and
`outputs/timesnet_raw_input_spectrum/stride32_test.json`.

- The observed-mask channel has strong sampling-related peaks; for stride 16,
  its dominant mask frequency is `f=75` (period 16). This frequency was not a
  canonical latent top-k selection.
- For both strides, normalized lambda has `f=1`, `f=2`, and `f=3` ranked 1,
  2, and 3, respectively, showing strong ultra-low-frequency spectral content.
- The trained input projection ranks include `f=1` first and `f=3` second at
  stride 16; at stride 32 they include `f=1` first, `f=3` second, and `f=2`
  third. These are diagnostic observations, not causal conclusions.

Projection spectral-contribution outputs are
`outputs/timesnet_projection_spectral_contributions/stride16_test.json` and
`outputs/timesnet_projection_spectral_contributions/stride32_test.json`.

| Stride | Frequency | Main projected component-magnitude findings |
|---:|---:|---|
| 16 | 1 | lambda `0.796250` |
| 16 | 2 | lambda `0.728299` |
| 16 | 3 | lambda `0.277017`; sparse_z `0.256299`; sparse_y `0.240903`; sparse_x `0.220899` |
| 32 | 1 | lambda `0.845219` |
| 32 | 2 | lambda `0.780432` |
| 32 | 3 | lambda `0.352425`; sparse_z `0.268450`; sparse_y `0.218701`; sparse_x `0.149578` |

Lambda strongly dominates projected component magnitude at `f=1` and `f=2`,
while `f=3` is a mixed lambda plus sparse-xyz component. These fractions compare
individual complex component magnitudes; they are not additive causal shares of
the final FFT amplitude.

### Completed lambda-isolated TimesNet period-selection runs

The model is `TimesNetLambdaIsolatedPeriodSelection1D`. Lambda remains one of
the five inputs and remains in the full prediction latent, Conv2d period
branches, residual path, and final output. Only period selection changes: the
first block removes the exact non-bias lambda input-projection contribution from
its selection signal, and later selection uses a shared-parameter auxiliary
counterfactual stream. No frequency is hard-coded for suppression and no new
trainable parameters are introduced. The parameter count remains `1,077,299`.
All other formal training and evaluation settings are identical to canonical
TimesNet.

| Stride | Output directory | Best validation hidden Relative L2 | Test raw hidden Relative L2 | Status |
|---:|---|---:|---:|---|
| 16 | `outputs/sparse_reconstruction_timesnet_lambda_isolated1d/q500_t1200_stride16_timesnet_lambda_isolated_dm80_df96_b2_k2_e600_seed42` | 8.807591e-03 | 9.231237e-03 | COMPLETED |
| 32 | `outputs/sparse_reconstruction_timesnet_lambda_isolated1d/q500_t1200_stride32_timesnet_lambda_isolated_dm80_df96_b2_k2_e600_seed42` | 1.716161e-02 | 1.628297e-02 | COMPLETED |

Formal test diagnostics are stored as
`metrics/test_period_selection_diagnostics.json` inside each run.

- At stride 16, block 0 primarily selected `f=222` (period 5) and `f=147`
  (period 8); the final smaller batch used `f=297` (period 4). Block 1 selected
  `f=3` (period 400) and `f=4` (period 300).
- At stride 32, block 0 selected `f=484` and `f=559`, both mapping to integer
  period 2. Block 1 selected `f=600` (period 2) and `f=3` (period 400).

These diagnostics establish that lambda isolation materially changed period
selection. The selected periods are not identified here as physical Kerr periods.

### Current same-resolution sparse reconstruction comparison

Dataset: `data/tasks/q_1p6-3_n500_t1200/dataset.npz`<br>
Split: `test`<br>
Metric: raw hidden-only overall Relative L2

| Model | Stride 16 test hidden RelL2 | Stride 32 test hidden RelL2 |
|---|---:|---:|
| Linear | 7.861464e-03 | 3.054264e-02 |
| PCHIP | 2.501225e-03 | 1.493277e-02 |
| FNO1D | 1.324286e-03 | 1.829136e-03 |
| Dilated ResNet1D | 1.870324e-03 | 1.693836e-03 |
| Canonical TimesNet1D | 1.657611e-02 | 1.510641e-01 |
| Lambda-isolated TimesNet1D | 9.231237e-03 | 1.628297e-02 |

On this dataset and frozen single-seed configuration, canonical TimesNet is much
more sensitive to increased sparsity than FNO1D and Dilated ResNet1D. Canonical
TimesNet strongly selects ultra-low-frequency periods, lambda contributes
strongly to projected `f=1` and `f=2` components, and isolating lambda from
period selection changes selected frequencies and periods while substantially
improving TimesNet performance. Despite this improvement, lambda-isolated
TimesNet does not match FNO1D or Dilated ResNet1D in these same-resolution
experiments.

These results are limited to one dataset, one seed, one frozen neural
configuration per model, and same-resolution training and evaluation. The
lambda-isolated second-block selection stream is a shared-parameter
counterfactual auxiliary stream, not an exact nonlinear lambda decomposition.
They do not establish that lambda is the sole cause of canonical TimesNet
failure, that the ablation proves a universal TimesNet flaw, that FNO is
universally superior, that selected periods are physical Kerr periods, or that
cross-resolution superiority has been demonstrated.

Same-resolution benchmarking is sufficiently complete to move to
cross-resolution/generalization experiment design. Further TimesNet tuning is
deferred unless later evidence specifically requires it.

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
