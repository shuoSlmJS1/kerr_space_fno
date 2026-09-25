# Plan A Phase-1 server asset inventory

Inventory date: 2026-09-21. Repository: /home/shanjinshuo/fno_kerr/kerr_project. Source HEAD: 594f0eaf331b0c3a8b4c6342f5c554ab64c3a2a1.

This directory contains small metadata reports only. It is not a copied mentor package. No Phase-2 download manifest, archive or download script was generated.

## Main findings

- Four unique authoritative datasets, four unique BEST checkpoints, and 14 scientific map nodes were verified.
- R0 has reusable Q400 T1200 and T1800 raw predictions. T1200 is a verified shared native baseline from the Plan B output location, not a fine-grid Plan B asset.
- All four formal A1 directories intentionally contain only summary JSON and per-Q/window CSV. R0 T2400 and all R1/R2/R3 formal lengths have prediction_asset_gap=true.
- Existing compact results are sufficient for the reported scientific conclusions and metric plots. They do not support arbitrary orbit replotting where prediction arrays are absent.
- M1 and original-design audit are code/provenance stages. M4a is supported by the actual R0 checkpoint and the final M4b normalization record; no extra checkpoint is invented.

## Scientific story and evidence boundaries

Plan A changes the physical lambda domain at approximately fixed delta_lambda=0.005. Sampled spans are [0,5.995], [0,8.995] and [0,11.995]; DFT logical periods N*delta_lambda are 6, 9 and 12. Plan B holds the sampled physical interval fixed and changes discretization. Plan B-only data/checkpoints/results are outside this candidate set. Shared n2000 training source, Q400 T1200 truth, R0 checkpoint and native T1200 prediction are marked shared_asset=true.

The corrected R0 protocol is one direct full-field frozen forward per length, with canonical ascending Q400 and raw float64 truth. It is neither autoregressive nor adapted. A0 validates all three raw-truth prefix pairs as EXACT_PREFIX. The original source split row order is preserved in data provenance; a separate canonical permutation is used for the FNO2D operator Q axis.

M1 establishes the discrete/physical Fourier mapping. M2 weakens simple raw-truth high-frequency energy loss as the primary explanation; it does not eliminate hidden-feature bandwidth effects. M3 finds zero prefix differences through normalization/lifting/first spectral input and substantial differences after the first global spectral operation. This supports Candidates 1 and 3 as real pathways, not sole causal explanations. M4a verifies original lambda mean 2.996041774749756 and std 1.7314223051071167. M4b changes only appended normalized lambda within each fixed-length pair; it is an intentionally nonphysical probe, not a valid physical repair.

R1 changes training exposure to strict prefixes T600/800/1000/1200 while retaining architecture and absolute lambda. It is a completed negative repair. R2 uses [Q,s,ell], with s=lambda/L and ell=L/L_ref, improving shared-prefix stability but not extrapolated-region accuracy relative to R0 and degrading T1200 fidelity. R3 retains these coordinates but changes R_k to R(xi_k), keeping global FFT and k=0..31; it improves both long-domain prefix and extrapolation metrics while severely degrading T1200 accuracy.

The R3 seven-length curve has a smaller sawtooth mainly because gradient-seen errors increased; non-gradient validation accuracy barely improved. T700/T900/T1100 participated in checkpoint selection and are not untouched test lengths. T1800/T2400 have been repeatedly inspected during repair development: development_benchmark=true and untouched_final_confirmation=false. Candidate 3 remains unresolved because R3 left global FFT unchanged.

R4 status is not_started. Plan A is paused pending advisor report/decision; conditional global/local redesign is not an executed experiment. Fresh independent confirmation after design freeze is a later scientific decision, not an asset claimed by this inventory.

## Verified formal comparison

Primary metric: mean_per_q_relative_l2 in raw physical xyz. Global Relative L2, MSE, component metrics and existing window definitions remain preserved in original summaries and METRICS_INVENTORY.json.

| Stage | T1200 | T1800 prefix | T1800 extrapolation | T2400 prefix | T2400 extrapolation |
|---|---:|---:|---:|---:|---:|
| R0 | 0.0054274904 | 1.7016413755 | 2.1751712909 | 2.2687772980 | 1.8272940063 |
| R1 | 0.0071467186 | 2.5466008576 | 4.4813180542 | 3.6880944268 | 4.9750195680 |
| R2 | 0.0704619116 | 1.4114604874 | 2.4476288108 | 1.4996846482 | 2.0038780057 |
| R3 | 0.3042366113 | 0.8345073047 | 1.3467645409 | 0.8909406007 | 1.3570231041 |

Every formal per-Q CSV contains 400 unique Q values per length (1200 rows total); means agree with JSON to 1e-12. User-supplied five-decimal anchors agree at their rounding precision. Saved R0 T1200/T1800 predictions reproduce the same existing primary metrics to 1e-12 using authoritative raw truth; no inference was run.

## Dataset and checkpoint navigation

| Dataset | Q count / split | T | Payload bytes | Role |
|---|---|---:|---:|---|
| q_1p6-3_n2000_t1200 | 2000 / 1400,300,300 | 1200 | 55679506 | original_training_and_validation_source |
| q_1p6007-2p9993_n400_t1200 | 400 / 280,60,60 | 1200 | 11139587 | canonical_Q400_exact_prefix_evaluation_truth |
| q_1p6007-2p9993_n400_t1800 | 400 / 280,60,60 | 1800 | 16719729 | canonical_Q400_exact_prefix_evaluation_truth |
| q_1p6007-2p9993_n400_t2400 | 400 / 280,60,60 | 2400 | 22288645 | canonical_Q400_exact_prefix_evaluation_truth |

All raw xyz arrays are float64; dataset headers, metadata, split hashes, solver v1 and generator references are in DATASET_INVENTORY.json. Historical experimental_v1 datasets are not the R0/R1/R2/R3 training source. Smaller training/validation length views are not distinct datasets. Exact-prefix files remain distinct length-specific formal assets; no truncation, repacking or removal is proposed.

| BEST | Epoch | Bytes | SHA256 |
|---|---:|---:|---|
| R0_BEST | 500 | 402989986 | db5cb795af32a91a24ef61c0b87e07fa7352be62201dfa8ca834dd31b5d63b33 |
| R1_BEST | 500 | 402990114 | 58cfd1343523b161eac100cc3336f585538ea9550cb09dcae3eb57fea95b91b8 |
| R2_BEST | 118 | 402991906 | 29891a82963e4a78abd0eeb7e4420f9a1d665df2f34aa1529b08693782afcfa1 |
| R3_BEST | 77 | 402996002 | 418da911cc1324b8182b924df13f37a5e747209f6ff92a6357e3af8eb0e4a728 |

BEST paths and full provenance are in CHECKPOINT_INVENTORY.json. LAST files are excluded even when they share the selected epoch: R0/R1 BEST and LAST have different file hashes, so no byte-identity claim is made. M3/M4 reuse R0; validation diagnostics reuse their repair BEST. R3 pre/post reconstruction fix refers to the same existing epoch77 binary; canonical float64 buffers match configuration, with no retraining or duplicate checkpoint entry.

## Prediction completeness

| Stage / population | Existing reusable arrays | Gap / interpretation |
|---|---|---|
| R0 Q400 T1200 | outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/coarse_prediction_raw_xyz.npy | Shared native baseline; Q/grid reconstructed from canonical dataset; fine prediction excluded |
| R0 Q400 T1800 | outputs/extrapolation_2d/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/t1800_full_q400/predictions_raw.npy | Historical final prediction; Q.npy/lambda_grid.npy/source_q_indices.npy and metrics identify it |
| R0 Q400 T2400 | None located | prediction_asset_gap=true |
| R1/R2/R3 Q400 all formal lengths | None located | prediction_asset_gap=true; compact metrics only |
| R1/R3 validation Q300 seven lengths | None located | Compact diagnostics only; full arrays needed only for direct trajectory replotting |
| R0 original task test300 | Raw and model-space arrays exist | DO_NOT_UPLOAD for minimal Q400 Plan A package; different population |

Historical T1800 targets_raw.npy is a float32 recovered representation, not authoritative float64 truth. It is excluded; use the canonical sorted T1800 dataset. source_q_indices.npy indexes the already canonical source field (0..399), not train/val/test concatenation order. The Q and lambda arrays were verified exactly.

## Classification and size budget

All classifications are proposals, not permission to copy, delete or publish. DO_NOT_UPLOAD does not mean DELETE. Hashes are in SHA256SUMS.txt; source paths and commits are indexed in SOURCE_PROVENANCE.json.

| Class | Files | Bytes | MiB | GiB |
|---|---:|---:|---:|---:|
| MUST_UPLOAD | 45 | 122461001 | 116.787911 | 0.114050695 |
| RECOMMENDED_UPLOAD | 17 | 1613779599 | 1539.020156 | 1.502949371 |
| INDEX_ONLY | 37 | 933168 | 0.889938 | 0.000869080 |
| DO_NOT_UPLOAD | 29 | 1653173481 | 1576.589089 | 1.539637783 |
| REVIEW_REQUIRED | 0 | 0 | 0.000000 | 0.000000000 |
| MUST + RECOMMENDED | 62 | 1736240600 | 1655.808067 | 1.617000066 |

MUST_UPLOAD comprises authoritative datasets/provenance, exact-prefix and final compact evidence, plus the two existing reusable native/historical prediction sets. RECOMMENDED_UPLOAD adds four BEST files and nonduplicate training configuration/summary/history for checkpoint-level verification. INDEX_ONLY comprises Git-tracked sources and four research records. Exclusions include LAST, original test300 arrays, float32 derived truth, duplicate summaries/snapshots and superseded historical metric tables. No periodic checkpoints were present in the four selected Plan A training directories; unrelated periodic server history was not crawled.

Category totals use the same file-level accounting as the manifest:

| Category (upload candidates) | Files | Bytes | MiB | GiB |
|---|---:|---:|---:|---:|
| datasets | 9 | 105836586 | 100.933634 | 0.098568002 |
| predictions | 8 | 14518044 | 13.845486 | 0.013520982 |
| BEST_checkpoints | 4 | 1611968008 | 1537.292488 | 1.501262195 |
| metrics_summaries_provenance | 41 | 3917962 | 3.736460 | 0.003648886 |

## Top 10 largest scientifically useful upload candidates

| Path (repository relative) | Bytes | MiB | Type / stage | Class | Why |
|---|---:|---:|---|---|---|
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200/checkpoints/best_model.pt | 402996002 | 384.326937 | best_checkpoint / R3 | RECOMMENDED_UPLOAD | The unique selected BEST required for future controlled inference and checkpoint-level verification. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200/checkpoints/best_model.pt | 402991906 | 384.323030 | best_checkpoint / R2 | RECOMMENDED_UPLOAD | The unique selected BEST required for future controlled inference and checkpoint-level verification. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r1_multilen_t600-800-1000-1200/checkpoints/best_model.pt | 402990114 | 384.321321 | best_checkpoint / R1 | RECOMMENDED_UPLOAD | The unique selected BEST required for future controlled inference and checkpoint-level verification. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/best_model.pt | 402989986 | 384.321199 | best_checkpoint / R0 | RECOMMENDED_UPLOAD | The unique selected BEST required for future controlled inference and checkpoint-level verification. |
| data/tasks/q_1p6-3_n2000_t1200/dataset.npz | 55679506 | 53.100115 | dataset_payload / shared | MUST_UPLOAD | Authoritative raw truth or original training/validation split needed for independent analysis and provenance. |
| data/tasks/q_1p6007-2p9993_n400_t2400/dataset.npz | 22288645 | 21.256108 | dataset_payload / shared | MUST_UPLOAD | Authoritative raw truth or original training/validation split needed for independent analysis and provenance. |
| data/tasks/q_1p6007-2p9993_n400_t1800/dataset.npz | 16719729 | 15.945176 | dataset_payload / shared | MUST_UPLOAD | Authoritative raw truth or original training/validation split needed for independent analysis and provenance. |
| data/tasks/q_1p6007-2p9993_n400_t1200/dataset.npz | 11139587 | 10.623538 | dataset_payload / shared | MUST_UPLOAD | Authoritative raw truth or original training/validation split needed for independent analysis and provenance. |
| outputs/extrapolation_2d/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/t1800_full_q400/predictions_raw.npy | 8640128 | 8.239868 | prediction / R0 | MUST_UPLOAD | Existing reusable historical R0 T1800 prediction with explicit Q/grid/checkpoint provenance. |
| outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/coarse_prediction_raw_xyz.npy | 5760128 | 5.493286 | prediction / R0 | MUST_UPLOAD | Only the shared native T1200 arm is reused: same R0 checkpoint and canonical Q400; fine-grid predictions and Plan B-only assets are excluded. |

## Top 10 largest excluded/intermediate assets

| Path (repository relative) | Bytes | MiB | Type / stage | Class | Why |
|---|---:|---:|---|---|---|
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200/checkpoints/last_model.pt | 402996002 | 384.326937 | last_checkpoint / R3 | DO_NOT_UPLOAD | BEST exists and reproduces the reported result; LAST is not needed for the mentor package. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200/checkpoints/last_model.pt | 402991906 | 384.323030 | last_checkpoint / R2 | DO_NOT_UPLOAD | BEST exists and reproduces the reported result; LAST is not needed for the mentor package. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500_r1_multilen_t600-800-1000-1200/checkpoints/last_model.pt | 402990114 | 384.321321 | last_checkpoint / R1 | DO_NOT_UPLOAD | BEST exists and reproduces the reported result; LAST is not needed for the mentor package. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/last_model.pt | 402989986 | 384.321199 | last_checkpoint / R0 | DO_NOT_UPLOAD | BEST exists and reproduces the reported result; LAST is not needed for the mentor package. |
| outputs/extrapolation_2d/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/t1800_full_q400/targets_raw.npy | 8640128 | 8.239868 | derived_truth / R0 | DO_NOT_UPLOAD | Float32 recovered-target representation is not authoritative raw float64 truth; use the registered T1800 dataset. |
| outputs/comparison/common_test__q_1p6007-2p9993_n400_t1200/common_test_dataset.npz | 5324833 | 5.078156 | derived_truth / R0 | DO_NOT_UPLOAD | Older common-test snapshot/result; final Plan A A1 and authoritative raw datasets are preferred. |
| outputs/comparison/common_test_e500_w64_n2000_n5000/common_test_dataset.npz | 5324833 | 5.078156 | derived_truth / R0 | DO_NOT_UPLOAD | Byte-identical file verified by size and SHA256; reference the canonical file once. Preserve all server copies. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/inference/predictions.npy | 4320128 | 4.119995 | prediction_auxiliary / R0 | DO_NOT_UPLOAD | Original training-task test300 output, not the independent canonical Q400 Plan A comparison. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/inference/predictions_model_space.npy | 4320128 | 4.119995 | prediction_auxiliary / R0 | DO_NOT_UPLOAD | Original training-task test300 output, not the independent canonical Q400 Plan A comparison. |
| outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/inference/targets.npy | 4320128 | 4.119995 | prediction_auxiliary / R0 | DO_NOT_UPLOAD | Original training-task test300 output, not the independent canonical Q400 Plan A comparison. |

## Duplicate relationships

Five byte-identical groups were found: four empty failed_samples.json files; two historical common-test snapshots; and root/log training summary copies for each of R1/R2/R3. These are not five new scientific datasets/runs. Common-test snapshots are float32 derived truth, distinct from raw float64 authoritative datasets. DUPLICATES.json separates actual byte-copy savings from hypothetical per-experiment repetition; the budget already selects canonical candidates once.

## Research-story completeness

Yes: the candidate scientific files plus indexed Git source/records let the advisor reconstruct the Plan A story without browsing the full server history. The qualification is historical audit provenance, not absent formal R0-R3 results.

| Required topic | Coverage | Evidence |
|---|---|---|
| original fixed-domain baseline | covered | R0 run_config, training summary, BEST metadata, source audit |
| exact-prefix protocol | covered | A0 recorded three-pair EXACT_PREFIX; current Q split identity verified |
| baseline failure | covered | R0 formal A1 plus corrected consistency diagnostic |
| M1-M4 evidence | covered | M1 source; M2/M3 compact tables; M4a checkpoint/M4b normalization; M4b nonphysical probe |
| original-design audit | covered_with_historical_provenance_gap | Current code/config-derived audit is explicit; no historical standalone audit file was located |
| R1 motivation/result | covered | Research records, training/BEST/formal A1 and same-validation-Q response |
| R2 motivation/result | covered | Research records, q-s-ell run configuration, BEST and formal A1 |
| R3 motivation/result | covered | Physical-frequency config, epoch77 BEST, canonical float64 anchors and formal A1 |
| R3 seven-length caveat | covered | Validation-response summary and by-length/per-Q CSV |
| unresolved Candidate 3 | covered | M3 first spectral divergence; R3 global FFT unchanged; reasoning limits retained |
| why R4 is not started | covered | Conditional design decision, no authorized/completed R4; Plan B R4 excluded |
| why Plan A is paused | covered | Experiment Plan section5 and Current State sections12/14: pending advisor report |

MISSING_EVIDENCE: no standalone historical original-design audit file was located. SOURCE_PROVENANCE.json explicitly supplies current Git/code-derived evidence and does not invent the historical audit timing or authorship. M4a likewise has no distinct output file but its actual checkpoint statistics are preserved in the final M4b result. The earlier invalid scrambled-Q artifact was not found in the recorded diagnostic directory; its invalidity is documented in Current State 7.7, Reasoning Log Episode3 and commit a68dc2c. No missing path or byte size is invented. If located later, it belongs to DO_NOT_UPLOAD, not scientific negative evidence.

## Proposed final mentor README outline (Phase 2 only)

1. What Plan A asks
2. Datasets/protocol
3. R0 anomaly
4. M1-M4
5. Original-design audit
6. R1
7. R2
8. R3
9. Strongest current conclusions
10. What remains unproved
11. Relationship to successful Plan B
12. Questions for advisor
13. Package navigation

The final PLAN_A_MENTOR_README.md has not been created.

## Phase-2 decision points

1. Approve the candidate scope: MUST only, or MUST plus the four BEST/training provenance.
2. Decide whether orbit replotting requires a later controlled frozen inference run for missing arrays; metrics-only conclusions do not require it.
3. Decide whether to recover a historical original-design audit note or accept the explicit current source audit as provenance.
4. Only after scope approval, prepare the final mentor README and any transfer manifest/script. Phase 2 was not started.

## Generated files and validation

The report directory contains only UTF-8 LF text metadata. ASSET_MANIFEST is file-level; dataset directory sizes in DATASET_INVENTORY are descriptive and never double-counted. All candidate payloads and BEST files have SHA256. Git records/source are indexed with current hashes and commits; untracked-source status was checked. No untracked Plan A source was found among relevant indexed scripts.

No network, Git write, GPU workload, model construction, training, inference, binary copy/packing, dataset generation or existing-asset mutation was performed. Read-only CPU work comprised header/config inspection, one-at-a-time BEST loading, hashes and recomputation of already defined metrics from existing arrays. See SAFETY_VALIDATION.json for final stat/hash and Git checks.

- README.md
- ASSET_MANIFEST.csv
- ASSET_MANIFEST.json
- DATASET_INVENTORY.json
- EXPERIMENT_RUNS.json
- PLAN_A_EXPERIMENT_MAP.json
- DUPLICATES.json
- SIZE_BUDGET.json
- SHA256SUMS.txt
- PREDICTION_INVENTORY.json
- CHECKPOINT_INVENTORY.json
- METRICS_INVENTORY.json
- INSPECTION_EVIDENCE.json
- SOURCE_PROVENANCE.json
- CONSISTENCY_CHECKS.json
- STORY_COMPLETENESS.json
- SAFETY_VALIDATION.json

Report directory byte count is provided in SIZE_BUDGET.json and SAFETY_VALIDATION.json after final validation.
