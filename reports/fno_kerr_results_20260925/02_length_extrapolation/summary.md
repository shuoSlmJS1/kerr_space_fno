# Plan A: Physical-domain length extrapolation

This summary consolidates the reviewed Plan A evidence as of 2026-09-25. It distinguishes direct measurements and interventions, indirect support from later experiments, and claims that remain unproven.

## 1. Research question

Plan A asks whether a frozen global Q-only FNO2D can extrapolate to a longer physical domain while the sampling step remains fixed.

| Role | Samples per trajectory | Step size | Sampled physical interval |
| --- | ---: | ---: | --- |
| Training and native evaluation | 1200 | 0.005 | $\lambda\in[0,5.995]$ |
| Longer-domain evaluation | 1800 | 0.005 | $\lambda\in[0,8.995]$ |
| Longer-domain evaluation | 2400 | 0.005 | $\lambda\in[0,11.995]$ |

The sampled endpoint is $(T-1)\Delta\lambda$. Each evaluation uses one full-field forward pass with frozen weights. This is physical-domain extension, distinct from later cross-resolution experiments that keep the sampled physical domain fixed and reduce $\Delta\lambda$.

## 2. Baseline phenomenon

Formal A1 evaluated the same frozen baseline checkpoint on the canonical ascending-Q400 field. The metric below is **mean-per-Q Relative L2 in raw physical xyz space**. The seen-prefix is the original first 1200 samples; extrapolation starts at $\lambda=6.0$. No observed xyz trajectory segment is supplied as model input.

| Evaluation | Seen-prefix | Extrapolation | Full-domain |
| --- | ---: | ---: | ---: |
| T1200 | 0.00542749 | — | 0.00542749 |
| T1800 | 1.70164138 | 2.17517129 | 1.87208136 |
| T2400 | 2.26877730 | 1.82729401 | 2.06209242 |

Native T1200 prediction is accurate. Extending the input domain damages not only the appended region but also the original shared prefix, whose ground truth is exactly identical across the three datasets. The T2400 shared prefix is even less accurate than the T1800 prefix.

The phenomenon therefore cannot be described simply as accurate prediction up to $\lambda=5.995$ followed by failure outside the training domain. Formal fixed-physical-window metrics also show that error does not increase monotonically with distance beyond the training boundary. For example, the first three T2400 extrapolation windows have mean-per-Q Relative L2 of approximately 2.5120, 1.5929, and 0.8444, before error rises again.

Strict truth-prefix validation and the corrected canonical-Q frozen diagnostic rule out ground-truth prefix mismatch and Q-axis scrambling as explanations of this formal result. The T1800 long-input prefix differs from the T1200 short-input prediction by mean-per-Q Relative L2 of 1.70210714. There is no autoregressive rollout, prediction feedback, or teacher forcing. These conclusions rely on formal numerical records; no unverified trajectory-image interpretation is added.

## 3. Candidate explanations

**The four Candidates do not correspond one-to-one to M1–M4.** They are coupled explanations examined through overlapping audits, diagnostics, and later repairs.

| Candidate | Mechanism | Evidence status |
| --- | --- | --- |
| **1. Fourier mode physical-frequency shift** | Changing domain length moves the same physical frequency to a different discrete Fourier index and therefore potentially a different learned mode weight. | Strong support from M1/M2/M3 and the R3 repair signal; its contribution to final prediction error has not been isolated. |
| **2. Fixed-mode physical bandwidth shrinkage** | A fixed number of retained modes covers a smaller physical-frequency band as the domain grows. | The shrinkage is real. However, about 98.7%–99.9% of raw xyz energy remains below the strictest cutoff, weakening the simple explanation based on loss of a large amount of raw high-frequency energy. Hidden-feature bandwidth effects remain possible. |
| **3. Global spectral representation changes the shared prefix** | The appended domain changes the full-field Fourier representation and can thereby change predictions on an unchanged prefix. | M3 strongly supports this as the earliest observed entry point of representation divergence. It does not establish a unique cause. |
| **4. Lambda coordinate / domain representation** | Appended lambda values exceed the training coordinate range and influence lifting and global spectral processing. | M4a/M4b establish a measurable coupled pathway; R2 supplies a partial positive repair signal. Dominance has not been established. |

## 4. Diagnostic chain

- **M1:** mathematical/code audit.
- **M2:** raw spectral-energy diagnostic.
- **M3:** internal FNO representation diagnostic.
- **M4:** coordinate audit plus controlled clamp intervention.

### M1: mathematical/code audit

The audit established the relevant global FFT, retained-mode, and coordinate-normalization semantics. Physical Fourier frequency follows

$$
f_k=\frac{k}{N\Delta\lambda}.
$$

Here $N\Delta\lambda$ is the DFT logical period, distinct from the sampled endpoint $(N-1)\Delta\lambda$. This audit makes frequency remapping and cutoff changes explicit; it does not by itself identify the cause of prediction failure.

### M2: raw spectral-energy diagnostic

M2 analyzed canonical-Q, raw float64 xyz truth without loading a model. For the retained lambda indices $k=0,\ldots,31$, the cutoff frequencies are:

$$
5.1667\ \longrightarrow\ 3.4444\ \longrightarrow\ 2.5833
\qquad (T1200\to T1800\to T2400).
$$

Using the same strict T2400 cutoff for all three datasets, the retained raw xyz energy fractions are:

| Dataset | Energy at or below the T2400 cutoff |
| --- | ---: |
| T1200 | 98.70% |
| T1800 | 99.93% |
| T2400 | 99.43% |

Loss of a large fraction of raw high-frequency xyz energy is therefore insufficient as a simple explanation of the collapse. At the same time, similar physical-frequency content occupies different discrete indices as the domain grows. This directly supports frequency-to-mode migration as a representation fact, not its causal share of model error. Raw xyz spectra do not determine hidden-feature spectra.

### M3: internal FNO representation diagnostic

M3 observed the same frozen checkpoint, normalization, and canonical Q400 field, with one normal forward per length. The shared-prefix differences remained zero through normalized input, lifting output, and first spectral-layer input.

| Stage: comparison with T1200 | T1800 relative difference | T2400 relative difference |
| --- | ---: | ---: |
| Normalized input prefix | 0 | 0 |
| Lifting output prefix | 0 | 0 |
| First spectral-layer input prefix | 0 | 0 |
| First spectral-branch output prefix | 0.8571 | 1.2817 |
| Final prediction prefix | 1.6997 | 2.2690 |

These are representation/prediction differences, not the mean-per-Q truth errors in Section 2. The first substantial spatial-prefix divergence appears after the first global spectral operation.

A representative aligned frequency is $f=1/3$, which maps to $k=2\to3\to4$ across T1200/T1800/T2400. The same physical frequency consequently encounters different learned discrete-mode weights. Internal comparisons show that this weighting can substantially change hidden-vector relationships. M3 supports Candidates 1 and 3 as real internal pathways, while leaving domain/basis effects and appended-coordinate effects coupled.

### M4: coordinate audit and controlled clamp

M4a verified the actual checkpoint's standard normalization and explicit $[Q,\lambda]$ input. Approximately 33.3% of T1800 coordinates and 50% of T2400 coordinates lie beyond the T1200 training range.

M4b changed only appended normalized lambda values above the training maximum. Within each original-versus-clamped pair, tensor length, FFT grid, Q, weights, shared-prefix coordinates, and truth stayed fixed.

| Original versus clamped | T1800 | T2400 |
| --- | ---: | ---: |
| First retained FFT representation difference | 0.2227 | 0.4223 |
| Final shared-prefix prediction difference | 0.2571 | 0.4336 |

Lambda-domain representation is therefore a measurable coupled pathway. Some early representations move toward the T1200 reference, but this does not produce consistent recovery through to the final prediction. **The clamp is an intentionally nonphysical mechanism probe, not a repair or a valid physical extrapolation protocol.** Its response is not a percentage attribution of final error.

## 5. R1 / R2 / R3 redesign

**R1/R2/R3 are repair experiments motivated by the diagnostics, not three new candidate hypotheses.** Their primary controlled comparisons are R1 versus R0, R2 versus R1, and R3 versus R2; comparison with R0 shows the overall practical trade-off.

All three repairs trained only on strict prefixes of the original T1200 source: gradient-training lengths 600/800/1000/1200 and validation/checkpoint-selection lengths 700/900/1100/1200. T1800/T2400 truth was excluded from training, normalization fitting, validation, and checkpoint selection. The selected checkpoints were then evaluated under the formal frozen A1 protocol.

All entries below are **mean-per-Q raw-xyz Relative L2**.

| Method | T1200 | T1800 prefix | T1800 extrap. | T2400 prefix | T2400 extrap. |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.00543 | 1.70164 | 2.17517 | 2.26878 | 1.82729 |
| R1 | 0.00715 | 2.54660 | 4.48132 | 3.68809 | 4.97502 |
| R2 | 0.07046 | 1.41146 | 2.44763 | 1.49968 | 2.00388 |
| R3-B1 | 0.30424 | 0.83451 | 1.34676 | 0.89094 | 1.35702 |

### R1 — Training protocol repair

R1 introduced multi-length prefix training while retaining the original architecture, absolute lambda input, and discrete-index spectral weights. It tested whether exposure to only one training length was the main limitation.

**Result:** native T1200 accuracy remained good, but longer-domain performance became worse. This multi-length training protocol did not solve the problem. It also did not produce smooth interpolation between training lengths in the same-validation-Q development diagnostic.

### R2 — Input representation repair

R2 retained the multi-length protocol and discrete-mode spectral weights, replacing absolute lambda with

$$
[Q,s,\ell],\qquad s=\frac{\lambda}{L},\qquad
\ell=\frac{L}{L_{\mathrm{ref}}},\qquad L=N\Delta\lambda,\quad L_{\mathrm{ref}}=6.0.
$$

**Result:** shared-prefix stability improved relative to R0 and R1, but extrapolation-region accuracy did not improve relative to R0; both reported extrapolation errors were higher. Native accuracy also declined. This is a partial positive coordinate-repair signal, not a full length-extrapolation repair.

### R3-B1 — Physical-frequency spectral parameterization

R3-B1 retained the R2 coordinates and replaced discrete-index-bound multipliers with physical-frequency-dependent weights:

$$
R_k\ \longrightarrow\ R(\xi_k),\qquad
\xi_k=\frac{k}{N\Delta\lambda}.
$$

**Result:** both long-domain prefixes and extrapolation regions improved substantially relative to R0. However, native T1200 error increased from about 0.0054 to 0.304. R3 provides important repair-stage support for the frequency/weight-mapping mechanism, but it is not a successful final solution.

R3-B1 retained the global FFT and the same 32 runtime modes: it did not repair physical-bandwidth shrinkage or implement a local/windowed model. A later same-validation-Q diagnostic also found that the reduced apparent length-response sawtooth mainly reflected worse accuracy at gradient-seen lengths, rather than meaningful improvement at intermediate lengths. Smooth within-range length generalization remains unresolved.

## 6. R4 status

**Candidate 4 and M4 already have experimental evidence.** The unstarted item is:

> R4 — global/local or local/windowed spectral redesign.

R4 primarily follows the global spectral representation issue highlighted by Candidate 3. Its recorded status is **candidate next stage; not started; Plan A paused at this decision point**. No local/windowed R4 experiment establishes whether that redesign would solve the problem.

The user's current retrospective view is that moving further toward local/windowed spectral structures would depart from this stage's focus on global FNO. This is explicitly a later interpretation of the scope decision, not a stopping rationale already documented in the contemporaneous formal experiment records.

## 7. Main conclusions

### Supported

1. The tested frozen global Q-only FNO2D is highly sensitive to **physical-domain length**.
2. Longer inputs change predictions on an exactly identical shared truth prefix; the failure is not confined to appended coordinates.
3. This is not autoregressive rollout error accumulation.
4. Loss of a large amount of raw xyz high-frequency energy is insufficient as a simple explanation of the collapse.
5. The first substantial observed prefix-representation difference appears after the first global spectral operation.
6. Physical-frequency-to-discrete-mode-weight remapping is supported by internal diagnostics and the R3 repair result.
7. Lambda coordinate/domain representation is another directly measurable factor, with a partial positive R2 repair signal.
8. The evidence favors several interacting global spectral representation factors over a single isolated explanation. This is an evidence-based synthesis, not proof of unique causality.

### Not established

1. No unique or dominant cause has been proven.
2. The causal contribution of each mechanism to final error has not been quantified.
3. Hidden-feature bandwidth and nonlinear coupling have not been excluded.
4. The results do not show that all FNOs are incapable of length extrapolation.
5. R3 does not jointly solve native accuracy, long-domain accuracy, and smooth length interpolation.
6. R4 has not been experimentally validated.
7. T1800/T2400 have been repeatedly used for diagnosis and redesign. They are development benchmarks, not an independent final confirmation set collected after design freeze; paper-level confirmation still requires fresh unseen Q and/or unseen long-domain lengths.

## 8. Later cross-resolution evidence

Later fixed-domain evaluation reused the **same historical T1200 FNO2D checkpoint** while keeping $\lambda\in[0,5.995]$ and increasing resolution. The metric is again mean-per-Q raw-xyz Relative L2.

| Test T | Relative L2 |
| --- | ---: |
| 1200 | 0.005427 |
| 2399 | 0.006257 |
| 3598 | 0.006736 |
| 4797 | 0.007000 |

In contrast, Plan A T2400 physical-domain extension has full-domain Relative L2 approximately 2.0621. This strongly weakens the explanation that the historical model necessarily fails whenever T changes. Changing tensor/grid resolution and extending the physical domain are different problems.

This is **indirect support**, not a new causal isolation experiment: fixed-domain refinement avoids the large changes in coordinate range and physical-frequency mapping introduced by domain extension, so it does not distinguish Candidates 1, 3, and 4 from one another.

The later Track A benchmark supplies architecture-dependent context. FNO1D and TimesNet retained relatively low off-native absolute errors; DeepONet and Transformer showed comparatively flat resolution responses, with different native accuracy levels. ResNet and BiLSTM still exhibited substantial resolution sensitivity. Thus changing T is not a universal model obstacle, nor is fixed-domain transfer automatically successful. These cross-model results do not prove a Plan A mechanism or establish length extrapolation for those models.

## 9. Sources / provenance

Paths below are relative to the project root. This summary creates no copies of datasets, predictions, checkpoints, or formal outputs.

**Long-lived research records**

- `FNO_KERR_CURRENT_STATE.md` — Sections 7.2–7.12, Plan B status, and Section 15.10.
- `FNO_KERR_REASONING_LOG.md` — mechanism, repair, Plan B, and completed benchmark evidence boundaries.
- `FNO_KERR_EXPERIMENT_PLAN.md` — Plan A definition and repair status; distinct fixed-domain protocol.
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` — formal asset identities and provenance.

**Baseline and validity checks**

- `outputs/formal_a1_length_extrapolation/q400_t1200_t1800_t2400_w0p5/a1_length_extrapolation_summary.json`
- `outputs/formal_a1_length_extrapolation/q400_t1200_t1800_t2400_w0p5/lambda_window_metrics.csv`
- `outputs/length_dataset_identity_validation/q400_t1200_t1800_t2400_prefix_identity.json`
- `outputs/length_change_prediction_consistency/q400_t1200_t1800_all_canonical_q.json`

**Mechanism diagnostics**

- `outputs/raw_kerr_spectral_energy/q400_t1200_t1800_t2400_m32_raw/raw_spectral_energy_summary.json`
- `outputs/raw_kerr_spectral_energy/q400_t1200_t1800_t2400_m32_raw/spectral_band_energy.csv`
- `outputs/raw_kerr_spectral_energy/q400_t1200_t1800_t2400_m32_raw/dominant_peaks.csv`
- `outputs/m3_internal_length_sensitivity/q400_t1200_t1800_t2400/m3_internal_length_sensitivity_summary.json`
- `outputs/m3_internal_length_sensitivity/q400_t1200_t1800_t2400/m3_stage_comparison.csv`
- `outputs/m3_internal_length_sensitivity/q400_t1200_t1800_t2400/m3_spectral_mode_comparison.csv`
- `outputs/m4b_lambda_clamp/q400_t1200_t1800_t2400/m4b_lambda_clamp_summary.json`
- `outputs/m4b_lambda_clamp/q400_t1200_t1800_t2400/m4b_stage_comparison.csv`

**Formal repairs and development boundary**

- `outputs/formal_a1_length_extrapolation/fno2d_m16x32_w64_d4_e500_r1_multilen_t600-800-1000-1200_q400_t1200_t1800_t2400/a1_length_extrapolation_summary.json`
- `outputs/formal_a1_length_extrapolation/fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200_best_q400_t1200_t1800_t2400/r2_a1_length_extrapolation_summary.json`
- `outputs/formal_a1_length_extrapolation/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_fixed_q400_t1200_t1800_t2400/r3_a1_length_extrapolation_summary.json`
- `outputs/r3_validation_length_response/fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_epoch77_val_t600-1200/r3_validation_length_response_summary.json`

**Indirect cross-resolution evidence**

- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/resolution_range_matrix.json`
- Track A six-model synthesis: `FNO_KERR_CURRENT_STATE.md`, Section 15.10; formal per-model evaluation matrices and per-cell metric paths are indexed in `SERVER_DATA_EXPERIMENT_REGISTRY.md`, Sections 9.1–9.6.
