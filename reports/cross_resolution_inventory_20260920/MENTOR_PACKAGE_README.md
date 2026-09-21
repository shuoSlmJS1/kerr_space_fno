# FNO-Kerr Cross-Resolution Research Package

## Research question and scope

Keep the physical domain fixed, change discretization resolution, and study resolution generalization.
Plan A physical-domain length extrapolation is background only; its large assets are excluded.

Plan B is the FNO-only paired-resolution experiment: historical joint-field FNO2D models trained at T1200/T2399,
ground-truth consistency qualification, and eight frozen evaluation cells through T4797.
Its range matrix reuses original coarse/fine cells. Fine truth arrays are solver replays, not upsampled predictions.

Track A Phase I is complete: **6 models, 2 training resolutions, 4 evaluation resolutions, 12 training runs and 48 cells**.
Models: FNO1D, Dilated ResNet, canonical TimesNet, BiLSTM, encoder-only Transformer and DeepONet.
The task is [Q_broadcast, lambda] -> xyz(lambda); DeepONet receives the same information through branch Q and trunk lambda.

## Six shared datasets and paired grids

| Family | Q range | Count | T | Train / val / test |
|---|---|---:|---|---|
| Training | [1.6, 3.0] | 2000 | 1200, 2399 | 1400 / 300 / 300 |
| Evaluation truth | [1.6007, 2.9993] | 400 | 1200, 2399, 3598, 4797 | 280 / 60 / 60 containers; evaluate all 400 |

All grids span lambda = [0, 5.995].

| T | Delta lambda | Coarse-node relationship |
|---:|---:|---|
| 1200 | 0.005 | Reference |
| 2399 | 0.0025 | Every second node matches T1200 |
| 3598 | 0.005 / 3 | Every third node matches T1200 |
| 4797 | 0.00125 | Every fourth node matches T1200; every second matches T2399 |

Paired resolutions preserve Q identities and split row order (data seed 10).
Q400 evaluation sorts all Q ascending, rather than using only its 60-row test container.
Fine metadata may retain the coarse source task_name; identify resolution by the directory, n_steps and grid.
05_docs/DATASET_INVENTORY.json supplies exact schemas, generators, metadata paths and recorded hashes.

## Locked comparison and interpretation

Track A: 500 epochs, batch 32, AdamW (lr 1e-3, weight decay 1e-4), ExponentialLR gamma 0.995, seed 27.
Train-only standard-normalization statistics are accumulated/stored in float64; models/training use float32.
Loss is normalized-space MSE; BEST is selected by validation MSE. Frozen evaluation does not refit statistics.
Capacity is 0.9M-1.3M nominal registered trainable real scalar parameters (target about 1.1M);
native complex elements count twice. This is not exact functional degrees of freedom.
Historical Plan B retains its original normalization and larger capacity; it is not a capacity-matched Track A baseline.

**Native accuracy != resolution robustness.** Native accuracy is the error at the training T;
robustness is assessed relative to that error, E(T_eval)/E(T_native).
Small variation across T does not imply low absolute error.
Observations apply only to this locked task, data, single seed, capacity range and training protocol,
not a universal model ranking or proof of architecture-only causality.

## Track A final results

Global raw-xyz Relative L2 (dimensionless; lower absolute error is better).
Rounded from saved formal metrics; JSON retains full precision and secondary metrics.

| Model / train T | Eval T1200 | Eval T2399 | Eval T3598 | Eval T4797 |
|---|---:|---:|---:|---:|
| FNO1D / 1200 | 0.0003046765905 | 0.002607739569 | 0.003465910777 | 0.003895466475 |
| FNO1D / 2399 | 0.002621957726 | 0.0002955621985 | 0.0008917733041 | 0.001306510118 |
| Dilated ResNet / 1200 | 0.0008757931159 | 1.236418782 | 1.151856049 | 1.101031311 |
| Dilated ResNet / 2399 | 1.080739314 | 0.0006243166665 | 1.03557683 | 1.031187628 |
| canonical TimesNet / 1200 | 0.001572549515 | 0.003838459637 | 0.004918443561 | 0.005469345982 |
| canonical TimesNet / 2399 | 0.003263090063 | 0.001656340913 | 0.001942551743 | 0.00222717317 |
| BiLSTM / 1200 | 0.0008166361945 | 0.3011073657 | 0.4072118416 | 0.4616841189 |
| BiLSTM / 2399 | 0.2028267915 | 0.0009904206664 | 0.08406687197 | 0.1254437207 |
| encoder-only Transformer / 1200 | 0.02604341679 | 0.02588182938 | 0.02582833357 | 0.02580165672 |
| encoder-only Transformer / 2399 | 0.03260342288 | 0.03225285992 | 0.03213670277 | 0.03207880227 |
| DeepONet / 1200 | 0.005720162442 | 0.005674431721 | 0.005659569871 | 0.005652211683 |
| DeepONet / 2399 | 0.009442024361 | 0.009396096556 | 0.009381145515 | 0.00937373523 |

FNO1D has low absolute errors across these grids. ResNet/BiLSTM have accurate native predictions but large off-grid increases.
TimesNet retains relatively low cross-resolution absolute errors. Transformer/DeepONet change little across resolution,
but have higher absolute errors than FNO1D here. These patterns are consistent with differing architectural biases;
the experiment does not prove a mechanism. T2399 training does not uniformly improve every model.

Transformer T1200 completed after deterministic resume from epoch225. Zero-byte evidence and recovery runtime files remain
on the server, outside this package. Disk pressure was suspected, not proven as the cause. Final summaries and results are included.

## Contents and binary formats

- 01_datasets/: six ground-truth NPZs and metadata/solver diagnostics, once per dataset.
  Q x_<split> is [N,1], xyz y_<split> [N,T,3], lambda_grid [T]; principal arrays are float64.
- 02_plan_b_fno/: metrics, eight raw-xyz NPY predictions, truth qualification, matrices and formal provenance.
  Predictions are float32 [400,T,3]; match them to the corresponding ascending-Q Q400 grid.
- 03_track_a_benchmark/: 12 matrices, 48 metrics JSONs, 48 NPZ predictions, run configs, summaries and wave provenance.
  Prediction NPZs contain float32 prediction_xyz [400,T,3], float64 Q [400,1] and lambda_grid [T].
  Summaries contain history, BEST identity and normalization/source provenance.
- 04_best_checkpoints/: all 14 unchanged BEST files (12 Track A + 2 Plan B).
  They support frozen re-evaluation with the associated source/configuration. Historical Plan B files are larger
  and may contain additional training state.
- 05_docs/: authoritative experiment plan and dataset inventory.
- ASSET_MANIFEST.csv/json: selected assets and original-server -> Windows-relative mapping.
- SHA256SUMS.txt: hashes for selected assets. Catalog files and the checksum list exclude themselves to avoid circular hashes;
  the separate download TSV provides their sizes and hashes.

Secondary metrics include raw MSE, mean per-Q Relative L2, median, p95, p99, max and worst Q.
No source code, periodic/LAST checkpoints, smoke outputs, temporary/duplicate logs, recovery runtime files or obsolete outputs are included.
Small formal summaries/configuration/provenance formerly classified RECOMMENDED are included as explicitly required for this final package.
This adds only a few MiB to MUST + 14 BEST and keeps the package about 1.77 GiB.
The download TSV is the complete transfer list. Download controls and failure logs sit outside the mentor package.

## Code and provenance

Use the project's GitHub repository supplied by the project owner; code is not copied.
No GitHub URL or remote publication state was verified in this inventory.
Branch: codex/clean-research-history-20260905.
Inventory source HEAD: 1ebc16696727aec2a556f5a856dfd1b36f4817a0.
Recorded Track A source commits:

| Model | Source commit |
|---|---|
| FNO1D | 3335b3b197cf8b7d7a302d900a51af37a25d2144 |
| Dilated ResNet | 29170c9c20813306abb465d10b86031fa5853d23 |
| canonical TimesNet | d8b190ed982e09b42784ccd7178e1c8b001d3a39 |
| BiLSTM | ff7ea434c718f3545c134b831462acd2574b5010 |
| encoder-only Transformer | 1a6f244e7914aaa3e029da991b1da111dd7f2c32 |
| DeepONet | 1ebc16696727aec2a556f5a856dfd1b36f4817a0 |

Per-run source_sha256 complements Git commits. Historical Plan B provenance is in its supplied summaries/configs.
Formal JSONs retain original server paths: resolve them through the manifest, without rewriting the formal files.
This is an analysis package, not an automatically relocatable training installation.

## Download behavior

Target: D:\AAA_MyNote\FNO_backup\FNO_Kerr_CrossResolution.
Review DOWNLOAD_MENTOR_PACKAGE.ps1 and its sibling MENTOR_PACKAGE_DOWNLOAD_MANIFEST.tsv before running.
Windows OpenSSH uses the existing server-school alias and normal configured authentication.
No password, private key or host-key bypass is embedded. The alias must already have a verified known-host entry.

The script validates the locked TSV, downloads each file directly, verifies size and known SHA256, then promotes its local
.download.partial file to the final name. Verified existing files are skipped. Existing final files with mismatching size/hash
are reported and left untouched. Each transfer gets at most three attempts; failed files are logged, remaining files are attempted,
and any failure produces a nonzero exit. Re-running retries missing/failed files.
This is file-level restart, not byte-range resume within a file. Partial files/failure CSVs are local only.
Nothing is removed or modified on the server.

Large-file hashes are reused from the completed inventory, not recomputed on the server during package preparation.
Windows local hashing verifies downloaded bytes against those recorded values.
