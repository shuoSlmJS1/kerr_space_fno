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

### Completed Plan B paired fine-resolution dataset

| Existing asset | Source coarse asset | Q field | T | step_size | Sampled physical interval | Status | Pairing / solver result |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `data/tasks/q_1p6007-2p9993_n400_t2399_plan_b_v1` | `data/tasks/q_1p6007-2p9993_n400_t1200` | Same independent offset-grid Q400 identities and canonical ordering | 2399 | 0.0025 | `[0, 5.995]` | COMPLETED — EXISTING SERVER ASSET | 400/400 success; 0 failures; paired completeness `True`; endpoint-fixed `fine_lambda[::2] == coarse_lambda`. |

The fine grid is `lambda_grid[j] = j * 0.0025`, endpoint-fixed to Q400/T1200. All
Protocol-v1 physics, initial conditions, solver equations, and turning-point logic were
held fixed. Ground-truth qualification is
`outputs/plan_b_q400_t1200_to_t2399/ground_truth_consistency.json`: `structural_valid=true`,
zero coarse/fine failed samples, shared-node Relative L2 mean
`5.218413681635546e-09`, median `5.095788550601654e-09`, max
`6.894048665372391e-09`, p95 `6.692889855076337e-09`, and p99
`6.8532082659278876e-09`.

### Completed Plan B endpoint-fixed resolution-range datasets

| Existing asset | Source coarse asset | Q field | T | step_size | Sampled physical interval | Status | Pairing / workflow result |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `data/tasks/q_1p6007-2p9993_n400_t3598_plan_b_range_v1` | `data/tasks/q_1p6007-2p9993_n400_t1200` | Same independent offset-grid Q400 identities and canonical ordering | 3598 | `0.005 / 3` | `[0, 5.995]` | COMPLETED — EXISTING SERVER ASSET | Endpoint-fixed R3; paired refinement replay reached `400/400`; unified qualification and frozen-evaluation workflow completed without a hard-stop. |
| `data/tasks/q_1p6007-2p9993_n400_t4797_plan_b_range_v1` | `data/tasks/q_1p6007-2p9993_n400_t1200` | Same independent offset-grid Q400 identities and canonical ordering | 4797 | 0.00125 | `[0, 5.995]` | COMPLETED — EXISTING SERVER ASSET | Endpoint-fixed R4; paired refinement replay reached `400/400`; unified qualification and frozen-evaluation workflow completed without a hard-stop. |

The server range workflow completed the ordered stages generation -> structural/numerical
qualification -> frozen evaluation -> matrix assembly for R1/R2/R3/R4. This registry has
no directly captured per-resolution numerical-truth qualification summary beyond that
confirmed completion status and therefore does not manufacture mean or percentile values.

### Completed Plan B matched fine-training dataset

| Existing asset | Source asset | Alignment requirement | T | step_size | Sampled physical interval | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| `data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1` | `data/tasks/q_1p6-3_n2000_t1200` | Exact original-n2000 Q values, split membership, and row ordering; fixed Kerr physics/initial conditions and solver provenance; Q400 remains evaluation-only | 2399 | 0.0025 | `[0, 5.995]` | COMPLETED — EXISTING SERVER ASSET |

The matched T2399 training task was completed for the reverse arm by replaying the source
T1200 n2000 split identities. `matched_dataset_complete = True`; source identifier is
`q_1p6-3_n2000_t1200`. Confirmed server absolute paths are
`/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6-3_n2000_t1200` and
`/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1`.

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
| `outputs/length_dataset_identity_validation/q400_t1200_t1800_t2400_prefix_identity.json` | Strict length-dataset prefix identity validation | q400 T1200 / T1800 / T2400 comparison-only datasets | 0 | `q400_t1200_t1800_t2400_prefix_identity.json` | COMPLETED — EXACT_PREFIX |
| `outputs/length_change_prediction_consistency/q400_t1200_t1800_all_canonical_q.json` | Frozen length-change prediction consistency diagnostic | q400 T1200/T1800 exact-prefix comparison-only datasets | 1 frozen FNO2D checkpoint | `q400_t1200_t1800_all_canonical_q.json` | COMPLETED — FORMAL DIAGNOSTIC |
| `outputs/plan_b_q400_t1200_to_t2399/ground_truth_consistency.json` | Plan B endpoint-fixed paired truth qualification | Q400 T1200/T2399 paired fixed-domain fields | 0 | `ground_truth_consistency.json` | COMPLETED — structural_valid=true; shared-node truth mismatch at about 1e-9 Relative L2 |
| `outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization` | Plan B historical-T1200 frozen coarse-to-fine FNO2D evaluation | Independent Q400 T1200/T2399 fixed-domain paired fields | Historical baseline checkpoint only | `metrics.json`, `experiment_summary.json`, raw prediction `.npy` | COMPLETED — modest coarse-to-fine degradation; contributes the theta1200 matrix row |
| `outputs/plan_b_bidirectional_t1200_t2399` | Plan B bidirectional fixed-domain resolution core | Independent Q400 T1200/T2399 paired fields plus matched original-n2000 T2399 training task | Historical theta1200 and theta2399 `training/checkpoints/best_model.pt` | `bidirectional_resolution_matrix.json`, `experiment_summary.json`, `workflow_state.json` | COMPLETED — 2x2 bidirectional matrix assembled |
| `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation` | Theta2399 native-fine and frozen reverse Q400 evaluation | Same independent Q400 identities/order at T2399 and T1200 | Frozen theta2399 `training/checkpoints/best_model.pt` | `metrics.json`, `experiment_summary.json`, native/reverse raw-xyz prediction `.npy` | COMPLETED — native T2399 and frozen T2399->T1200 arms |
| `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797` | Plan B endpoint-fixed 1x--4x FNO-only range sweep | Q400 T1200/T2399/T3598/T4797 paired fixed-domain fields | Frozen theta1200 and theta2399 checkpoints | `resolution_range_matrix.json`, `experiment_summary.json`, `workflow_state.json`, per-cell metrics/predictions | COMPLETED — R1/R2/R3/R4 matrix assembled; T3598 and T4797 truth/evaluation cells completed |
| `outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200` | R2 domain-conditioned coordinate training repair | `q_1p6-3_n2000_t1200` strict prefixes only | 1 R2 FNO2D run | `run_config.json`, `summary.json`, `checkpoints/best_model.pt` | COMPLETED — R2 PARTIAL POSITIVE REPAIR SIGNAL |
| `outputs/formal_a1_length_extrapolation/fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200_best_q400_t1200_t1800_t2400` | Formal R2 A1 frozen length-extrapolation evaluation | q400 exact-prefix T1200/T1800/T2400 comparison-only datasets | 1 frozen R2 checkpoint | `r2_a1_length_extrapolation_summary.json`, `r2_per_q_metrics.csv`, `r2_lambda_window_metrics.csv` | COMPLETED — FORMAL R2 REPAIR EVALUATION |
| `outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200` | R3-B1 physical-frequency spectral training repair | `q_1p6-3_n2000_t1200` strict prefixes only | 1 R3 FNO2D run | `run_config.json`, `summary.json`, `checkpoints/best_model.pt` | COMPLETED — STRONG LONG-DOMAIN SIGNAL; severe T1200 trade-off |
| `outputs/formal_a1_length_extrapolation/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_fixed_q400_t1200_t1800_t2400` | Formal R3-B1 A1 frozen length-extrapolation evaluation | q400 exact-prefix T1200/T1800/T2400 comparison-only datasets | 1 frozen R3 checkpoint | `r3_a1_length_extrapolation_summary.json`, `r3_per_q_metrics.csv`, `r3_lambda_window_metrics.csv` | COMPLETED — FORMAL R3 REPAIR EVALUATION |
| `outputs/r3_validation_length_response/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_epoch77_val_t600-1200` | R3 seven-length validation-response development diagnostic | `q_1p6-3_n2000_t1200` validation Q strict prefixes T600–T1200 | 1 frozen epoch-77 R3 checkpoint | `r3_validation_length_response_summary.json`, `r3_validation_length_response_by_length.csv`, `r3_validation_length_response_per_q.csv` | COMPLETED — DEVELOPMENT DIAGNOSTIC; formal_test_evidence=false |
### Completed Plan B theta2399 training artifacts

- Training directory: `outputs/plan_b_bidirectional_t1200_t2399/training`.
- Best checkpoint: `outputs/plan_b_bidirectional_t1200_t2399/training/checkpoints/best_model.pt`.
- Last checkpoint: `outputs/plan_b_bidirectional_t1200_t2399/training/checkpoints/last_model.pt`.
- Run configuration: `outputs/plan_b_bidirectional_t1200_t2399/training/run_config.json`.
- Training history: `outputs/plan_b_bidirectional_t1200_t2399/training/train_history.json`.
- Training summary: `outputs/plan_b_bidirectional_t1200_t2399/training/train_summary.json`.

The theta2399 training task selected best epoch `500`. Its model-space best validation MSE,
test MSE, and test Relative L2 are `0.00045836385106667876`, `0.000446074060164392`, and
`0.020454108715057373`, respectively. These are training-task model-space metrics and are
not interchangeable with the independent-Q400 raw-physical-xyz native/reverse metrics.

### Completed theta2399 Q400 and bidirectional provenance

- Bidirectional matrix: `outputs/plan_b_bidirectional_t1200_t2399/bidirectional_resolution_matrix.json`.
- Experiment summary and workflow state: `outputs/plan_b_bidirectional_t1200_t2399/experiment_summary.json` and
  `outputs/plan_b_bidirectional_t1200_t2399/workflow_state.json`.
- Q400 evaluation directory and metrics: `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation` and
  `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation/metrics.json`.
- Raw-physical-xyz predictions: `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation/theta2399_native_t2399_prediction_raw_xyz.npy` and
  `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation/theta2399_reverse_t1200_prediction_raw_xyz.npy`.

### Completed strict length-dataset identity validation

- Output: `outputs/length_dataset_identity_validation/`
  `q400_t1200_t1800_t2400_prefix_identity.json`.
- Validator: commit `8a6ee0b` (`add strict length dataset identity validator`).
- Datasets: `data/tasks/q_1p6007-2p9993_n400_t1200`,
  `data/tasks/q_1p6007-2p9993_n400_t1800`, and
  `data/tasks/q_1p6007-2p9993_n400_t2400`.
- Overall and pair results: `EXACT_PREFIX` for `short_to_medium`,
  `short_to_long`, and `medium_to_long`.
- Lambda prefixes and all train/val/test trajectory prefixes are exact; the
  recorded max/mean absolute differences, RMSE, and Relative L2 values are 0.
- `historical_t1800_reusable=true`; `t2400_ready_for_future_a1=true`.
- This validates dataset pairing only. It does not provide T2400 inference
  performance or the short-input versus long-input-prefix prediction comparison.

### Completed formal length-change prediction consistency diagnostic

- Output: `outputs/length_change_prediction_consistency/`
  `q400_t1200_t1800_all_canonical_q.json`.
- Diagnostic type: frozen one-shot short-input versus long-input-prefix
  prediction consistency; evidence: `formal diagnostic`.
- Training task: `q_1p6-3_n2000_t1200`; model:
  `fno2d_m16x32_w64_d4_e500`; same frozen `best_model.pt` for both inputs.
- Short/long tasks: `q_1p6007-2p9993_n400_t1200` and
  `q_1p6007-2p9993_n400_t1800`; prerequisite artifact:
  `outputs/length_dataset_identity_validation/`
  `q400_t1200_t1800_t2400_prefix_identity.json` (`EXACT_PREFIX`).
- Protocol: full 400-Q canonical ascending-Q field; matching Q/y permutation;
  shared short raw float64 truth for both truth comparisons; no adaptation,
  autoregression, teacher forcing, fine-tuning, or checkpoint change.
- Primary mean-per-Q Relative L2: short vs truth `0.0054274904`; long prefix vs
  truth `1.7016413755`; long prefix vs short prediction `1.7021071446`.
- Global Relative L2 in the same order: `0.0071953455`, `1.6991658080`, and
  `1.6996710240`.
- Status: `COMPLETED`; evidence: `formal diagnostic`.
- An earlier scrambled-Q output was superseded as `protocol-debug` because of
  invalid Q-axis ordering; its metrics are not registered as scientific results.

### Completed R2 domain-conditioned coordinate repair

- Training run: `outputs/q_1p6-3_n2000_t1200/`
  `fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200/`.
- Selected asset: `checkpoints/best_model.pt`; `status = completed`,
  `experiment_type = r2_domain_conditioned_coordinate_training`, and
  `repair_class = INPUT_REPRESENTATION_REPAIR`.
- Source task: `q_1p6-3_n2000_t1200`; strict prefix training lengths
  `600, 800, 1000, 1200`; validation/checkpoint-selection lengths
  `700, 900, 1100, 1200`. T1800/T2400 were excluded from training, normalization,
  validation, and checkpoint selection.
- Model input: `[Q, s, ell]`, with `s = lambda / L`, `ell = L / L_ref`,
  `L = N * delta_lambda`, and `L_ref` the T1200 logical domain length. The standard
  discrete-index spectral parameterization was unchanged.
- Training record: `epochs = 500`, `optimizer_steps = 500`,
  `forward_backward_passes_per_step = 4`, `best_epoch = 118`, and
  `best_val_selection_score = 0.13257645582780242`.
- Formal output: `outputs/formal_a1_length_extrapolation/`
  `fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200_best_q400_t1200_t1800_t2400/`
  with `r2_a1_length_extrapolation_summary.json`, `r2_per_q_metrics.csv`, and
  `r2_lambda_window_metrics.csv`.
- Formal evaluation used the frozen R2 `best_model.pt`, Stage-2 `EXACT_PREFIX`,
  canonical Q400, raw float64 truth, and one direct forward per T1200/T1800/T2400.
  Evidence: `server-result-verified`. It is a formal repair evaluation, not evidence
  that coordinate representation is the sole mechanism or a complete solution.

### Completed R3-B1 physical-frequency-aware spectral repair

- Training run: `outputs/q_1p6-3_n2000_t1200/`
  `fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200/`.
  Selected `checkpoints/best_model.pt`: `status = completed`,
  `experiment_type = r3_physical_frequency_spectral_training`,
  `repair_class = SPECTRAL_PARAMETERIZATION_REDESIGN`, and `best_epoch = 77`.
- Source task and prefix protocol match R2: `q_1p6-3_n2000_t1200`; gradient-training
  lengths `600, 800, 1000, 1200`; validation/checkpoint-selection lengths
  `700, 900, 1100, 1200`; no T1800/T2400 training, normalization, validation, or
  checkpoint-selection leakage.
- R3-B1 retains `[Q, s, ell]` and the global 32-bin FFT branch, but uses
  `physical_frequency_anchor_interpolation` for `R_k -> R(xi_k)`,
  `xi_k = k / (N * delta_lambda)`. It does not repair the retained physical bandwidth.
- Formal output: `outputs/formal_a1_length_extrapolation/`
  `fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_fixed_q400_t1200_t1800_t2400/`
  with `r3_a1_length_extrapolation_summary.json`, `r3_per_q_metrics.csv`, and
  `r3_lambda_window_metrics.csv`. It used Stage-2 `EXACT_PREFIX`, canonical Q400,
  raw float64 truth, and one frozen forward per T1200/T1800/T2400; evidence is
  `server-result-verified`.
- Primary `mean_per_q_relative_l2`: T1200 prefix/full `0.30423661134631347`; T1800
  prefix/extrapolation/full `0.8345073047219703` / `1.3467645409082092` /
  `1.0320417147537395`; T2400 prefix/extrapolation/full `0.8909406006838355` /
  `1.3570231040536689` / `1.1463882703177917`.
- Provenance note: a legacy `model_config` anchor float32-precision issue was corrected
  in reconstruction. Canonical float64 `anchor_frequency_values` and state-dict buffers
  confirmed the existing checkpoint was valid; no retraining was required.
### Completed R3 seven-length validation-response diagnostic

- Output: `outputs/r3_validation_length_response/`
  `fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_epoch77_val_t600-1200/`.
- Frozen asset: the R3-B1 epoch-77 `checkpoints/best_model.pt` from
  `q_1p6-3_n2000_t1200`; `split = val`, `validation_q_count = 300`, stable ascending-Q,
  and the same canonical validation-Q identities for every strict prefix T600–T1200.
- Status: `completed`; `scientific_status = development_diagnostic` and
  `formal_test_evidence = false`. T700/T900/T1100 were checkpoint-selection validation
  lengths, not untouched tests. The protocol used seven frozen forwards; no training,
  adaptation, normalization refit, or autoregression.
- Model provenance: `[Q, s, ell]`, `s = lambda / L`, `ell = L / L_ref`,
  `L = N * delta_lambda`, `L_ref = 6.0`, canonical float64 physical-frequency anchors, and
  unchanged global FFT. `physical_bandwidth_shrinkage_repaired = false`.
- Primary `mean_per_q_relative_l2`: T600 `0.0723158813794609`, T700
  `0.4498312050898403`, T800 `0.16855927048223368`, T900 `0.42519888310195797`, T1000
  `0.32226570803539056`, T1100 `0.44829662581877244`, and T1200
  `0.3024272828096965`.
- Descriptive diagnostics: mean/max local interpolation residual `0.2150433844661226` /
  `0.329393629158993`; gradient-seen/non-gradient-validation means
  `0.2163920356766954` / `0.4411089046701902`; gap `0.2247168689934948`.
  Compared with R1, the sawtooth amplitude is smaller, but primarily because
  gradient-seen lengths degraded; intermediate non-gradient validation accuracy did not
  materially improve. This does not establish smooth within-range interpolation.
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

## 9. Planned cross-model Benchmark Protocol v1

The initial planned-work index below is retained with current status. Completed formal
Wave-1 and Wave-2 assets are registered in sections 9.1 and 9.2; no new dataset was generated.

| Planned work | Reused recorded basis | Status | Asset boundary |
| --- | --- | --- | --- |
| Phase-I Track-A task-aligned models: BiLSTM, Dilated ResNet, canonical TimesNet, Transformer encoder, FNO1D, DeepONet | `data/tasks/q_1p6-3_n2000_t1200`; matched `data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1`; independent Q400 endpoint-fixed evaluation family | IMPLEMENTED; FNO1D WAVE 1 and ResNet WAVE 2 COMPLETED; other four models NOT TRAINED | Formal assets are indexed in sections 9.1 and 9.2; Phase I remains incomplete. |
| Phase-II FNO formulation/capacity study | Phase-I FNO1D-small; existing FNO2D-large Plan B assets; future FNO2D-small matched data | PLANNED / NOT TRAINED | Existing FNO2D-large outputs are reusable evidence; FNO2D-small is not yet an asset. |
| Phase-III selective scaling | Phase-I/II results only after review | CONDITIONAL FUTURE / NOT STARTED | No selected model, capacity-scaled checkpoint, or expanded n5000/n10000 run is registered. |

All planned Phase-I models must reuse the original-n2000 T1200/T2399 matched training
identities and the independent canonical Q400 evaluation identities. The first evaluation
set is endpoint-fixed R1/R2/R3/R4 at T1200/T2399/T3598/T4797. The existing FNO range assets
for R3 and R4 are `data/tasks/q_1p6007-2p9993_n400_t3598_plan_b_range_v1` and
`data/tasks/q_1p6007-2p9993_n400_t4797_plan_b_range_v1`; their confirmed status does not
register any future baseline checkpoint or evaluation as existing. No future baseline may be
described as generated, trained, or evaluated before its own provenance and output paths are
registered.

### 9.1 Completed Phase I Wave 1 — FNO1D (2026-09-12)

These are existing formal assets, not planned or smoke outputs. Source HEAD:
`3335b3b197cf8b7d7a302d900a51af37a25d2144`; branch
`codex/clean-research-history-20260905`; clean Git preflight. Exact source/data/meta hashes,
environment, capacity and GPU mapping are embedded in the run/evaluation metadata.

| Existing path | Status | Contents / identity |
| --- | --- | --- |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200` | COMPLETED — 500 epochs | FNO1D modes32/width64/depth4/GELU; n2000/T1200 original split; best epoch499; host GPU1/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t2399` | COMPLETED — 500 epochs | Same locked FNO1D; matched n2000/T2399 split; best epoch459; host GPU2/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/eval_t1200` | COMPLETED — 4 frozen cells | T1200 best checkpoint evaluated on canonical Q400 T1200/T2399/T3598/T4797 |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/eval_t2399` | COMPLETED — 4 frozen cells | T2399 best checkpoint evaluated on the same four canonical Q400 datasets |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/wave_summary.json` | COMPLETED — verified aggregate | Both training summaries, eight metric cells, native robustness, normalization, paths and verification status |
| `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/wave_execution.json` | EXISTING — execution manifest | Authorized scope, clean source HEAD, independent single-GPU mapping; train/evaluation logs are alongside it |

Each training directory preserves `run.json`, `summary.json`, `best_model.pt` and twenty
immutable `epoch_0025.pt` through `epoch_0500.pt` recovery checkpoints. Normalization
statistics remain binary64 in checkpoint state; actual training is float32. History and
optimizer/scheduler/RNG state are retained in recovery checkpoints. Each evaluation
directory preserves `matrix.json`, `metrics_t<T>.json` and `prediction_t<T>.npz` for all
four T values. No historical dataset/checkpoint/output was overwritten or regenerated.

Best checkpoint SHA256:

- T1200: `11ccbebf1d0fd68c57d0895685adeb7d1386da723242ee8271b18286ebbb7ff1`.
- T2399: `1b226b5f62d2f931401a8aefe3794d0b95234acbed0c4b20a6b9c483852eb1f1`.

| Train T | Host GPU / local CUDA | Epochs | Best epoch | Best validation normalized MSE | Training seconds | Peak allocated bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1200 | 1 / cuda:0 | 500 | 499 | 1.067155274843875e-07 | 518.085670201 | 194066944 |
| 2399 | 2 / cuda:0 | 500 | 459 | 9.817391306417752e-08 | 594.374222904 | 357028352 |

Measured global raw-xyz Relative L2:

| Train / evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 0.000304676590517 | 0.00260773956864 | 0.00346591077718 | 0.00389546647459 |
| 2399 | 0.00262195772641 | 0.000295562198528 | 0.000891773304135 | 0.00130651011794 |

Full raw MSE, per-Q mean/median/p95/p99/max/worst-Q, native ratios/deltas, inference
wall times and peak allocated memory are in the per-cell metrics/matrices and wave
summary. All eight prediction metrics were recomputed exactly on CPU after completion;
Q/lambda ordering, normalization and selected checkpoint hashes were verified. No
interruption/resume or Protocol-impacting issue occurred. This registers only FNO1D:
the other five Phase-I models remain untrained, and Phase II/III remain unexecuted.

### 9.2 Completed Phase I Wave 2 — Dilated ResNet (2026-09-12)

These are existing formal outputs, verified during completed-experiment analysis.
Source HEAD: `29170c9c20813306abb465d10b86031fa5853d23`;
branch `codex/clean-research-history-20260905`. Initial execution and analysis Git
preflights were clean. No new training, inference or output generation occurred during
analysis. Wave-1's 70 files were verified unchanged against the saved preflight hashes.

| Existing path | Status | Identity / contents |
| --- | --- | --- |
| `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/train_t1200` | COMPLETED — epoch500 | Locked ResNet, original n2000/T1200 train split, best epoch490, host GPU1/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/train_t2399` | COMPLETED — epoch500 | Locked ResNet, matched n2000/T2399 train split, best epoch487, host GPU2/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/eval_t1200` | COMPLETED — four frozen cells | T1200 BEST checkpoint on canonical Q400 T1200/T2399/T3598/T4797 |
| `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/eval_t2399` | COMPLETED — four frozen cells | T2399 BEST checkpoint on the same four canonical Q400 assets |
| `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/wave_execution.json` | EXISTING — execution provenance | Scope, source HEAD, CPU/GPU scheduling, Wave-1 preflight file hashes |

Each training directory contains `run.json`, `summary.json`, `best_model.pt` and twenty
immutable `epoch_0025.pt` through `epoch_0500.pt` recovery checkpoints. Statistics,
optimizer/scheduler/RNG, history and best-so-far weights are retained in checkpoints.
Each evaluation directory contains `matrix.json`, four `metrics_t<T>.json` and four
`prediction_t<T>.npz`. Root `train_t<T>.log`, `evaluate_t<T>.log` and each corresponding
`*_start.json` / `*_exit.json` record execution. Both training and both evaluation exits
are successful; no failure or training-resume event is recorded. There are 77 retained
Wave-2 files; no standalone wave-summary JSON is claimed.

Architecture: input2/output3, width84, kernel7, 11 blocks, dilation powers of two through
1024, RF12349; both parameter counts are 1,096,119. All locked training defaults were
retained, including train-only float64 statistics and float32 training. No dataset,
Wave-1 asset or historical result was overwritten.

Best checkpoint SHA256:

- T1200: `aeee49cf033a304e2fd07a8ef5e45297cbc634d6904cf00787af7c1410f83270`.
- T2399: `12e0cf0c8e8294652bf2a011d605aaf37dd3ff6d2024a196d00371f06007d712`.

| Train T | Final epoch | Best epoch | Best validation normalized MSE | Final validation normalized MSE | Training seconds | Peak allocated bytes | Host GPU / CUDA_VISIBLE_DEVICES / local CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1200 | 500 | 490 | 8.08959028594e-7 | 9.28123791558e-7 | 8828.08704151 | 1216906240 | 1 / 1 / cuda:0 |
| 2399 | 500 | 487 | 3.85748219666e-7 | 4.38222237259e-7 | 10006.3546170 | 2259363328 | 2 / 2 / cuda:0 |

Measured global raw-xyz Relative L2:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 0.000875793115919 | 1.23641878184 | 1.15185604926 | 1.10103131103 |
| 2399 | 1.08073931383 | 0.000624316666496 | 1.03557683005 | 1.03118762782 |

Native-relative global-error ratios:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 1.00000000000 | 1411.77038203 | 1315.21477884 | 1257.18196572 |
| 2399 | 1731.07554520 | 1.00000000000 | 1658.73648041 | 1651.70607026 |

Raw MSE, all per-Q summaries/worst Q, ratios/deltas and inference resources are retained
in the per-cell metrics/matrices; Current State 15.6 includes the full secondary table.
Independent CPU recomputation of all eight saved prediction metrics passed at
`rtol=1e-12, atol=1e-14`; max absolute discrepancy 2.842170943040401e-14.
Normalization, best selection, source/data provenance and Q/lambda arrays were verified.

Provisional comparison only: ResNet has higher native errors and much larger off-native
errors/ratios than FNO1D in this single locked seed27 comparison. Training time and
peak allocated memory were also higher (see Current State 15.6); this is not a six-model
ranking or architecture-only causal claim. Four other Track A models remain untrained.

### 9.3 Completed Phase I Wave 3 — canonical TimesNet (2026-09-13)

Existing formal assets verified after independent background completion. Execution and
analysis HEAD: `d8b190ed982e09b42784ccd7178e1c8b001d3a39`; branch
`codex/clean-research-history-20260905`. Both Git preflights were clean.

| Existing path | Status / contents |
| --- | --- |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/train_t1200` | COMPLETED: 500 epochs; best epoch492; host GPU1/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/train_t2399` | COMPLETED: 500 epochs; best epoch486; host GPU2/local cuda:0 |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/eval_t1200` | COMPLETED: frozen BEST at Q400 T1200/T2399/T3598/T4797 |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/eval_t2399` | COMPLETED: frozen BEST at Q400 T1200/T2399/T3598/T4797 |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/wave_execution.json` | EXISTING: scope, configuration, source hashes and prior-wave file hashes |
| `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/run_workflow.py` | EXISTING: detached train-to-evaluation workflow, restricted to authorized TimesNet pair |

Each training directory preserves run.json, summary.json, best_model.pt and twenty
immutable epoch_0025.pt through epoch_0500.pt recovery checkpoints, including fitted
normalization, optimizer/scheduler/RNG, full history and best-so-far weights.
Each evaluation directory preserves matrix.json plus metrics_t<T>.json and
prediction_t<T>.npz for all four canonical resolutions. Root launch_t<T>.json,
workflow_t<T>_start.json, train/evaluate start/exit records, workflow_t<T>_complete.json
and corresponding logs establish successful independent execution. There are 86 retained
files; no separate wave-summary JSON is claimed.

Verified configuration: input2/output3, d_model80, d_ff96, two TimesNet blocks, top-k2,
kernels(1,3,5), dropout0, and real_scalar_parameter_count == tensor_numel == 1,077,059.
The saved-source implementation uses canonical runtime FFT discovery, batch-shared top-k,
1D-to-2D period folding and Inception Conv2d processing, not the lambda-isolated variant.
Locked AdamW/ExponentialLR defaults, batch32, seed27 and 500 epochs were unchanged.
Train-only float64 statistics match fresh fitting and an independent float64 reference;
checkpoint restoration and float32 model/sample application were verified. Frozen evaluation
restored the same statistics without refitting and retained canonical Q400 order with
batch32 (final batch16).

Best checkpoint SHA256:

- T1200: `3b69a9408dc7f3a60b9746a26ecf8c53ba0f9a91bb21839ca793abd897a73ea4`.
- T2399: `9beec9fdf4b3359619039dc8004b667e3e416036db978fcbe62c92f27e656794`.

| Train T | Final epoch | Best epoch | Best validation normalized MSE | Final validation normalized MSE | Training seconds | Peak allocated bytes | Host GPU / CUDA_VISIBLE_DEVICES / local CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1200 | 500 | 492 | 0.00000267102036256 | 0.00000311175996406 | 710.514718014 | 484014592 | 1 / 1 / cuda:0 |
| 2399 | 500 | 486 | 0.00000290461068592 | 0.0000207157540475 | 1404.66065574 | 1064739328 | 2 / 2 / cuda:0 |

Measured global raw-xyz Relative L2:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 0.00157254951507 | 0.00383845963700 | 0.00491844356093 | 0.00546934598239 |
| 2399 | 0.00326309006262 | 0.00165634091327 | 0.00194255174312 | 0.00222717317039 |

Native-relative global-error ratios:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 1 | 2.44091496020 | 3.12768756328 | 3.47801193537 |
| 2399 | 1.97005944638 | 1 | 1.17279705377 | 1.34463452091 |

Both training histories contain epochs 1--500, both train exits and both automatic
evaluation exits have returncode 0, and both workflow completion markers exist. All eight
frozen cells and twenty immutable recovery checkpoints per run are present; no failure or
resume event is recorded. Logs contain successful result JSON without extra error text.
All 70 Wave-1 and 77 Wave-2 file hashes match the Wave-3 launch manifest.
The 86 Wave-3 files were retained unchanged during this read-only result audit.

Single-thread CPU recomputation from saved predictions and authoritative float64 truth
used independent sum-of-squares formulas, metric epsilon 1e-12, and unchanged per-Q
summaries. All 64 scalar comparisons across eight cells passed rtol=1e-12, atol=1e-14;
maximum absolute discrepancy was 6.938893903907228e-18. Saved Q/lambda arrays, dataset/
split provenance, source hashes, per-cell versus matrix metrics, native ratios/deltas,
checkpoint history prefixes, and minimum-validation best-weight selection were verified.
No predictions were regenerated and no output asset was created by this analysis.

Full secondary metrics, worst Q, deltas and provisional three-model/compute comparison
are in Current State 15.7 and the saved per-cell metrics/matrices. TimesNet has higher
absolute error than FNO1D in all eight cells but smaller off-native ratios; compared with
ResNet it has higher native error but much lower cross-resolution error. These observations
are provisional and do not rank all six models or identify FFT period-selection causality.
Selected-period diagnostics were not recorded. Phase I is 6/12 trainings and 24/48 cells
complete; no BiLSTM/Transformer/DeepONet formal asset is registered here.
