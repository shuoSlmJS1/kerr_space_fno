# FNO-Kerr Cross-Resolution Delivery Inventory

Snapshot: 2026-09-20T19:44:35.504474+08:00. Repository: `/home/shanjinshuo/fno_kerr/kerr_project`.
Branch: `codex/clean-research-history-20260905`; source HEAD: `1ebc16696727aec2a556f5a856dfd1b36f4817a0`.

This is an inventory and proposed Windows mapping, not a transfer or archive. No scientific assets were copied, moved, deleted, compressed or modified. No experiment was launched. The three pre-existing uncommitted research-record edits were preserved byte-for-byte.

## Scope and reading order

- Plan B: fixed physical domain, coarse/fine ground-truth qualification, frozen FNO2D bidirectional evaluation and range through T4797. Eight unique predictions; two historical BEST checkpoints.
- Track A Phase I: FNO1D, Dilated ResNet, canonical TimesNet, BiLSTM, encoder-only Transformer and DeepONet; 12 completed 500-epoch runs, 12 BEST checkpoints and 48 frozen evaluation cells.
- Plan A physical-domain extrapolation is background only. No Plan A large assets, older training-size sweeps, unrelated datasets or speculative follow-up experiments are selected.
- Start with this README, SIZE_BUDGET.json and DATASET_INVENTORY.json. Use ASSET_MANIFEST.csv for the file-level download list; JSON contains the same entries plus snapshot/check limits. TRACK_A_RUNS.json groups the 12 runs and all cells.
- ASSET_MANIFEST fields give absolute server path, project-relative path, logical category, byte size, validation status, hash basis, classification and Windows destination. Empty destinations mean no proposed package copy.
- DUPLICATES.json records reuse and content comparisons. SHA256SUMS.txt addresses destination-relative paths, not original server paths.

## Scientific context (index, not a new protocol)

Both branches keep the physical lambda interval [0, 5.995] fixed. T is 1200, 2399, 3598 or 4797. Track A trains at T1200 and T2399 on identical Q split identities. It evaluates all 400 canonical Q values in ascending Q order at each resolution. The primary accuracy metric is global raw-xyz Relative L2. Native-relative ratios must not be confused with absolute accuracy.

Track A task: [Q_broadcast, lambda] -> xyz, with the locked DeepONet branch-Q/trunk-lambda formulation. The small-model capacity band is 0.9M-1.3M nominal registered trainable real scalar coordinates; native complex elements count twice. Shared training: 500 epochs, batch 32, AdamW, lr 1e-3, weight decay 1e-4, ExponentialLR gamma 0.995, seed 27, normalized MSE and BEST by validation MSE. Standard-normalization statistics are fitted only on each training-resolution train split, accumulated/stored in float64 and applied in the float32 training path. Frozen evaluation does not refit.

Historical Plan B FNO2D assets retain their original normalization and configuration. Do not substitute the newer Track A normalization or treat FNO2D-large as capacity matched to Track A: its historical tensor_numel is 16,802,755 and real_scalar_parameter_count is 33,579,971. The authoritative experiment plan is included under 05_docs; longer state/registry/reasoning documents remain repository indexes.

## Dataset inventory

All six are formal ground truth, shared rather than duplicated per model. Fine data are paired solver replays on the same Q identities/split order; different T arrays are distinct datasets, not duplicate copies. Split seed is 10. All principal Q, lambda and xyz arrays are float64. `x_<split>` is [N,1], `y_<split>` is [N,T,3], and `lambda_grid` is [T]. Q400 files retain their original split containers, but frozen evaluation uses all 400 rows after canonical sorting. Exact array schemas, generators, available SHA256, metadata paths and relationships are in DATASET_INVENTORY.json.

| Dataset | Q range / count | T / delta lambda | Train / val / test | Files | Directory bytes |
|---|---|---|---|---:|---:|
| training_n2000_T1200 | [1.6, 3.0] / 2000 | 1200 / 0.005 | 1400 / 300 / 300 | 3 | 55,681,768 |
| training_n2000_T2399 | [1.6, 3.0] / 2000 | 2399 / 0.0025 | 1400 / 300 / 300 | 4 | 112,393,203 |
| evaluation_Q400_T1200 | [1.6007, 2.9993] / 400 | 1200 / 0.005 | 280 / 60 / 60 | 3 | 11,141,867 |
| evaluation_Q400_T2399 | [1.6007, 2.9993] / 400 | 2399 / 0.0025 | 280 / 60 / 60 | 4 | 22,487,456 |
| evaluation_Q400_T3598 | [1.6007, 2.9993] / 400 | 3598 / 0.00166666666667 | 280 / 60 / 60 | 4 | 33,522,764 |
| evaluation_Q400_T4797 | [1.6007, 2.9993] / 400 | 4797 / 0.00125 | 280 / 60 / 60 | 4 | 44,521,468 |

Each dataset directory maps from its absolute path below to `01_datasets/<directory-name>/`:

- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6-3_n2000_t1200` -> `01_datasets/q_1p6-3_n2000_t1200/`
- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1` -> `01_datasets/q_1p6-3_n2000_t2399_plan_b_matched_v1/`
- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6007-2p9993_n400_t1200` -> `01_datasets/q_1p6007-2p9993_n400_t1200/`
- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6007-2p9993_n400_t2399_plan_b_v1` -> `01_datasets/q_1p6007-2p9993_n400_t2399_plan_b_v1/`
- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6007-2p9993_n400_t3598_plan_b_range_v1` -> `01_datasets/q_1p6007-2p9993_n400_t3598_plan_b_range_v1/`
- `/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/q_1p6007-2p9993_n400_t4797_plan_b_range_v1` -> `01_datasets/q_1p6007-2p9993_n400_t4797_plan_b_range_v1/`

Metadata caveat: paired replay metadata can retain the coarse source task_name. Use the actual directory, n_steps and lambda grid to identify the fine dataset. Original coarse generator invocation is not separately recorded; the inventory names the existing generator/source, not an invented command.

## Track A run-level inventory

All 12 BEST archives and summaries are present/readable. Each run has four finite/readable prediction arrays, four readable metrics JSONs and a complete matrix. Individual file sizes/statuses are in ASSET_MANIFEST. The table aggregates classes per run; root-level launch/workflow/recovery metadata are listed separately in the manifest.

| Model / train T | BEST | run.json + summary.json | matrix + 4 metrics | 4 predictions | Periodic checkpoints |
|---|---:|---:|---:|---:|---:|
| fno1d / 1200 | 4.093 MiB | 0.101 MiB | 0.714 MiB | 50.598 MiB | 20 / 327.728 MiB |
| fno1d / 2399 | 4.093 MiB | 0.101 MiB | 0.714 MiB | 50.605 MiB | 20 / 327.729 MiB |
| resnet / 1200 | 4.201 MiB | 0.102 MiB | 0.714 MiB | 49.982 MiB | 20 / 336.615 MiB |
| resnet / 2399 | 4.201 MiB | 0.101 MiB | 0.714 MiB | 50.538 MiB | 20 / 336.615 MiB |
| timesnet / 1200 | 4.124 MiB | 0.101 MiB | 0.714 MiB | 50.598 MiB | 20 / 330.271 MiB |
| timesnet / 2399 | 4.124 MiB | 0.101 MiB | 0.714 MiB | 50.606 MiB | 20 / 330.271 MiB |
| bilstm / 1200 | 4.182 MiB | 0.101 MiB | 0.714 MiB | 50.513 MiB | 20 / 334.729 MiB |
| bilstm / 2399 | 4.182 MiB | 0.101 MiB | 0.714 MiB | 50.610 MiB | 20 / 334.730 MiB |
| transformer / 1200 | 4.164 MiB | 0.101 MiB | 0.714 MiB | 50.601 MiB | 20 / 333.467 MiB |
| transformer / 2399 | 4.164 MiB | 0.101 MiB | 0.714 MiB | 50.611 MiB | 20 / 333.457 MiB |
| deeponet / 1200 | 4.153 MiB | 0.101 MiB | 0.714 MiB | 50.600 MiB | 20 / 332.543 MiB |
| deeponet / 2399 | 4.153 MiB | 0.101 MiB | 0.714 MiB | 50.601 MiB | 20 / 332.543 MiB |

Periodic checkpoints: **240 files**, 3.8972 GiB. Retain on server / archive only; NOT primary mentor package. This inventory does not authorize deletion.

## Plan B reuse and provenance

- `outputs/plan_b_q400_t1200_to_t2399`: initial truth consistency report and theta1200 coarse/fine predictions/metrics.
- `outputs/plan_b_bidirectional_t1200_t2399`: matched-training qualification (embedded in experiment_summary), theta2399 BEST, reverse/native predictions and bidirectional matrix.
- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797`: T3598/T4797 truth qualification, four new predictions, resolution matrix and curve. Original T1200/T2399 cells are referenced, not regenerated or recopied.
- Historical theta1200 BEST source: `outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/best_model.pt`. Only its training/provenance materials are recommended; the sibling 300-row native-test predictions/targets are not Plan B Q400 cells and are excluded.
- Plan B NPY predictions do not embed Q/lambda; use the corresponding canonical Q400 dataset and its ascending-Q mapping. Track A NPZ predictions include prediction_xyz, Q and lambda_grid.

## Deduplication and failure evidence

- Six unique dataset files, each selected once. All 56 final prediction files (48 Track A + 8 Plan B) have distinct SHA256 values in the inspected scope.
- Twelve Track A evaluation stdout logs are byte-identical to their matrix.json; package the JSON only.
- Historical theta1200 BEST and LAST have different whole-file hashes but identical normalized archive-entry payloads. Only BEST is selected. Theta2399 BEST/LAST differ both as files and archive payloads; equal size is not evidence of equality. LAST is excluded by role, not falsely declared duplicate.
- Repeated empty failure lists are tiny contextual metadata belonging to different datasets; preserve their context. An identical empty-file hash does not make unrelated files the same logical asset.
- Transformer T1200 resumed deterministically from epoch 225. The preserved zero-byte epoch250 failure artifact and original zero-byte train exit record remain on the server; recovery manifests/status records are recommended as documentation_only. The recovery workflow completed; these original zero-byte files are not newly corrupted final assets.
- Disk pressure was a strong suspicion for the interruption, not a proven ENOSPC cause.
- Only registry-derived directories were inspected. No claim is made about unregistered duplicates elsewhere, all periodic checkpoint payloads, unrelated old versions or smoke directories.

## Classification and size budget

Logical file lengths are used below (GiB = 2^30 bytes). No compression savings are assumed. Tiny context-specific metadata may repeat byte content; datasets, final predictions and BEST models are selected once per logical asset.

| Selection | Bytes | GiB |
|---|---:|---:|
| MUST_UPLOAD | 1,044,125,815 | 0.9724 |
| RECOMMENDED_UPLOAD | 860,988,042 | 0.8019 |
| INDEX_ONLY | 767,168 | 0.0007 |
| DO_NOT_UPLOAD | 5,018,121,983 | 4.6735 |
| MUST + 12 Track A BEST only | 1,096,381,053 | 1.0211 |
| MUST + all 14 BEST only | 1,902,360,961 | 1.7717 |
| MUST + all RECOMMENDED | 1,905,113,857 | 1.7743 |

- MUST_UPLOAD: six datasets and their metadata/solver diagnostics, final predictions, metrics/matrices, truth qualification and the authoritative protocol.
- RECOMMENDED_UPLOAD: 14 BEST files (12 Track A + 2 Plan B), training summaries/configurations, launch/runtime provenance and selected small recovery documentation.
- INDEX_ONLY: repository source/workflow code, long historical records and early launch diagnostics; keep source commit/hash references. Output-local workflow scripts are server indexes, not assumed Git-tracked.
- DO_NOT_UPLOAD: periodic/LAST checkpoints, duplicate/temporary runtime logs, retained failure payloads and unrelated historical native-test artifacts. This classification NEVER means delete.

| Selected content category | Bytes | MiB |
|---|---:|---:|
| Dataset NPZ files | 277,796,529 | 264.927 |
| Dataset directories including metadata | 279,748,526 | 266.789 |
| Final predictions | 751,066,423 | 716.273 |
| BEST checkpoints (14) | 858,235,146 | 818.477 |
| Metrics + metadata + protocol + summaries | 18,015,759 | 17.181 |

The dataset-directory row includes metadata already included in the last row; use dataset NPZ + final predictions + BEST + metrics/metadata for an additive total. Inventory-file overhead is separate (about 1 MiB); exact bytes are available from the generated file lengths.

## Largest 10 selected assets

| Asset (project-relative source) | Class | MiB |
|---|---|---:|
| `outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/best_model.pt` | RECOMMENDED_UPLOAD | 384.321 |
| `outputs/plan_b_bidirectional_t1200_t2399/training/checkpoints/best_model.pt` | RECOMMENDED_UPLOAD | 384.321 |
| `data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1/dataset.npz` | MUST_UPLOAD | 106.030 |
| `data/tasks/q_1p6-3_n2000_t1200/dataset.npz` | MUST_UPLOAD | 53.100 |
| `data/tasks/q_1p6007-2p9993_n400_t4797_plan_b_range_v1/dataset.npz` | MUST_UPLOAD | 42.225 |
| `data/tasks/q_1p6007-2p9993_n400_t3598_plan_b_range_v1/dataset.npz` | MUST_UPLOAD | 31.736 |
| `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/new_evaluations/theta1200_t4797/theta1200_t4797_prediction_raw_xyz.npy` | MUST_UPLOAD | 21.959 |
| `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/new_evaluations/theta2399_t4797/theta2399_t4797_prediction_raw_xyz.npy` | MUST_UPLOAD | 21.959 |
| `data/tasks/q_1p6007-2p9993_n400_t2399_plan_b_v1/dataset.npz` | MUST_UPLOAD | 21.212 |
| `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913/eval_t2399/prediction_t4797.npz` | MUST_UPLOAD | 20.207 |

Largest excluded files are the two historical Plan B LAST checkpoints, each about 384.32 MiB. The largest Track A periodic files are ResNet epoch0500 (about 16.84 MiB each); the aggregate periodic footprint matters more than a single file. Total intermediate checkpoints: 4.6478 GiB. Total DO_NOT_UPLOAD: 4.6735 GiB.

## Proposed Windows package layout

```text
FNO_Kerr_CrossResolution/
  README.md
  ASSET_MANIFEST.csv
  ASSET_MANIFEST.json
  SHA256SUMS.txt
  01_datasets/
    <six original dataset-directory names>/
  02_plan_b_fno/
    <original experiment root>/<selected original relative paths>
  03_track_a_benchmark/
    <wave root>/<selected original relative paths>
  04_best_checkpoints/
    plan_b/theta1200/best_model.pt
    plan_b/theta2399/best_model.pt
    track_a/<wave root>/train_t1200/best_model.pt
    track_a/<wave root>/train_t2399/best_model.pt
  05_docs/
    FNO_KERR_EXPERIMENT_PLAN.md
    DATASET_INVENTORY.json
    TRACK_A_RUNS.json
    DUPLICATES.json
    SIZE_BUDGET.json
```

For every selected existing asset, ASSET_MANIFEST is the exact server-source -> package-relative mapping. The package root is prepended on Windows. Transfer only selected files, not whole wave directories. New inventory files map as follows:

- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/README.md` -> `README.md`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/ASSET_MANIFEST.csv` -> `ASSET_MANIFEST.csv`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/ASSET_MANIFEST.json` -> `ASSET_MANIFEST.json`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/SHA256SUMS.txt` -> `SHA256SUMS.txt`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/DATASET_INVENTORY.json` -> `05_docs/DATASET_INVENTORY.json`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/TRACK_A_RUNS.json` -> `05_docs/TRACK_A_RUNS.json`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/DUPLICATES.json` -> `05_docs/DUPLICATES.json`
- `/home/shanjinshuo/fno_kerr/kerr_project/reports/cross_resolution_inventory_20260920/SIZE_BUDGET.json` -> `05_docs/SIZE_BUDGET.json`

Existing JSONs retain their original absolute server paths for provenance. Do not rewrite formal JSONs; resolve those paths through the manifest. This is an analysis package, not a self-relocating training installation. Source and exact dependencies remain repository references.

## Integrity, limitations and next transfer recommendation

- Selected JSON files parsed successfully. Selected datasets, predictions and BEST checkpoint ZIP archives passed CRC. Prediction arrays were readable and finite; Track A Q order was checked. Twelve training histories cover epochs 1..500. No model inference was run and no predictions were regenerated.
- Existing SHA256 values were reused for the six dataset files; large dataset hashes were NOT recomputed. CRC success is not a new cryptographic identity verification. Metadata hashes and selected result/BEST hashes were computed or compared with existing records. See sha256_basis per asset.
- No missing or unreadable selected formal result was found. BEST archive readability is not a new inference/load compatibility experiment. Periodic checkpoint bodies were not exhaustively verified.
- Hash sums cover all selected source files, not the inventory files themselves. SHA256SUMS intentionally includes optional BEST/provenance files; a MUST-only download will not contain those optional paths.
- Download the small inventory files and protocol first. Then select MUST_UPLOAD rows. Add the 12 compact Track A BEST files; add the two larger Plan B BEST files and other RECOMMENDED rows when historical re-evaluation is desired. No transfer was performed in this task.

Disk snapshot: /home available 27.2932 GiB; Track A allocated 4.5557 GiB; Wave 6 allocated 0.7596 GiB. These are transient measurements, not permanent allocation guarantees.
