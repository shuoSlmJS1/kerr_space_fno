# Plan B: FNO-only fixed-domain cross-resolution generalization

This summary consolidates the reviewed Plan B evidence as of 2026-09-25. All model-error metrics below are evaluated in raw physical xyz space on the same independent canonical Q400 field. Global Relative L2 aggregates all Q, lambda, and xyz entries before taking the norm ratio; mean-per-Q Relative L2 averages the separate trajectory-level ratios. MSE averages squared error over all entries.

## 1. Research question

Plan A held $\Delta\lambda=0.005$ fixed and increased T, extending the physical domain. Plan B instead fixes the sampled physical domain at $\lambda\in[0,5.995]$, decreases $\Delta\lambda$, and increases T and sampling density.

The central question is whether the historical T1200 FNO2D still fails severely when only discretization resolution changes, without extending the physical domain. This is a key contrast with Plan A: different tensor lengths need not represent the same scientific change.

The initial comparison uses T1200 at $\Delta\lambda=0.005$ and T2399 at $\Delta\lambda=0.0025$. Both have sampled endpoint $(T-1)\Delta\lambda=5.995$, and `fine_lambda[::2] == coarse_lambda`. Physics, initial conditions, solver provenance, Q400 identities and ordering, and each frozen model's weights and stored normalization remain controlled.

## 2. Initial T1200 → T2399 experiment

| Evaluation | MSE | Global Relative L2 | Mean-per-Q Relative L2 |
| --- | ---: | ---: | ---: |
| T1200 native | 0.001259059869 | 0.007195345501 | 0.005427490395 |
| T2399 fine | 0.001463006496 | 0.007756728428 | 0.006257048793 |
| fine_common | 0.001666906402 | 0.008279117934 | 0.006793341748 |

The coarse and fine evaluations use the **same frozen historical T1200 BEST checkpoint**, with no retraining, fine-tuning, adaptation, or normalization refit. Each grid receives one complete forward pass. The T2399 full grid is the primary fine-resolution result.

`fine_common` is a diagnostic obtained by taking `prediction_fine[:, ::2, :]` after the complete T2399 forward and comparing it with `truth_fine[:, ::2, :]`. It is not another forward using a T1200 input, and it is not the prediction-to-prediction comparison below.

T1200→T2399 does **not** reproduce the order-one error collapse seen in Plan A. Full-grid Global Relative L2 increases by 7.80%, and mean-per-Q Relative L2 by 15.28%, while absolute errors remain low.

The saved-prediction comparison `pred_fine[:, ::2, :]` versus `pred_coarse` nevertheless has a global normalized difference of **0.00300882**, defined using the coarse truth norm:

$$
D=\frac{\|\mathrm{pred}_{\mathrm{fine,common}}-\mathrm{pred}_{\mathrm{coarse}}\|_2}
{\|\mathrm{truth}_{\mathrm{coarse}}\|_2}.
$$

Thus the model exhibits practical resolution robustness, not exact finite-grid output invariance. This difference is not a causal percentage of model error.

## 3. Bidirectional experiment

The reverse arm trained one matched T2399 model and then froze it for native T2399 and reverse T1200 evaluation. The historical T1200 model was reused.

| Train T | Test T | MSE | Global Relative L2 | Mean-per-Q Relative L2 |
| --- | --- | ---: | ---: | ---: |
| 1200 | 1200 | 0.001259059869 | 0.007195345501 | 0.005427490395 |
| 1200 | 2399 | 0.001463006496 | 0.007756728428 | 0.006257048793 |
| 2399 | 1200 | 0.001411940100 | 0.007619677633 | 0.006087018418 |
| 2399 | 2399 | 0.001280229599 | 0.007256035374 | 0.005506914675 |

The two training datasets are the original n2000/T1200 task and its matched T2399 replay. Q identities, train/validation/test membership, and row ordering are preserved, with splits of 1400/300/300. Q400 remains evaluation-only.

Both models follow the same FNO2D architecture and training protocol: modes 16×32, width 64, depth 4, 500 epochs, batch size 1, AdamW with the same learning-rate schedule, seed 27, and raw targets. Each model fits normalization only on its own training-resolution train split. The T2399 statistics are newly fitted training statistics, not a copy of the T1200 values; all frozen evaluations restore the respective checkpoint statistics without refitting.

Relative to each model's native Global Relative L2:

- Coarse→fine degradation: **+7.80%**.
- Fine→coarse degradation: **+5.01%**.

Both directions achieve frozen resolution transfer with low absolute error in this matched experiment. The measured difference between directions is not a general claim that fine→coarse is superior.

## 4. Resolution-range experiment

The range experiment retains $\lambda\in[0,5.995]$ at T1200, T2399, T3598, and T4797, using step sizes 0.005, 0.0025, $0.005/3$, and 0.00125. These correspond to 1×, 2×, 3×, and 4× subdivision of the coarse intervals.

| Train model / metric | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| Train T1200 Global RelL2 | 0.00719535 | 0.00775673 | 0.00809813 | 0.00829232 |
| Train T1200 Mean-per-Q RelL2 | 0.00542749 | 0.00625705 | 0.00673632 | 0.00699967 |
| Train T2399 Global RelL2 | 0.00761968 | 0.00725604 | 0.00731539 | 0.00737815 |
| Train T2399 Mean-per-Q RelL2 | 0.00608702 | 0.00550691 | 0.00560126 | 0.00570148 |

The range stage reused the existing T1200/T2399 results and evaluated the same frozen models at the two additional grids; it introduced no new model training.

### Train T1200

Error increases across the measured resolution levels, but remains low in absolute terms. At T4797, Global Relative L2 is **15.25%** above native, and mean-per-Q Relative L2 is **28.97%** above native. There is no Plan A-like catastrophic failure through the tested 4× refinement.

### Train T2399

The four-point response is not monotonically increasing: error falls from T1200 evaluation to native T2399, then rises slightly at T3598 and T4797. Relative to native, Global Relative L2 increases by only **0.82%** at T3598 and **1.68%** at T4797.

Matched T2399 training yields lower errors than T1200 training on the two still-finer evaluation grids. It does not improve every resolution: the T1200-trained model remains more accurate at T1200. These observations describe the tested models and aggregate metrics, not a universal monotonicity or training-resolution law.

## 5. Ground-truth qualification

The coarse and fine evaluation data preserve the same Q400 identities and sampled endpoint 5.995. Common-grid coordinates correspond, with every second T2399 node matching T1200. Fine truth was generated by paired reintegration under the same physics and solver, not interpolation.

The formal T1200/T2399 qualification reports structural validity, complete pairing, 400/400 successful fine trajectories, and no failed samples. Shared-node truth differences have mean-per-Q Relative L2 approximately $5.22\times10^{-9}$ and maximum approximately $6.89\times10^{-9}$. These are far below the model errors in this comparison, so numerical ground-truth mismatch is not a major confounder here. This is an experiment-specific qualification, not a universal solver guarantee.

## 6. Relation to Plan A

The key contrast uses the same historical T1200 checkpoint and the same mean-per-Q raw-xyz Relative L2 convention:

- **Plan A, T2400 longer physical domain:** full-domain error approximately **2.06209**.
- **Plan B, T2399 fixed physical domain:** error approximately **0.006257**.
- **Plan B, T4797 fixed physical domain:** error approximately **0.00699967**.

This strongly weakens the explanation that the FNO must fail whenever input T differs from training T. **Resolution change and physical-domain extension are different problems.**

Plan B adds no lambda coordinates outside the training physical interval and does not introduce the large domain-length and physical-frequency remapping changes present in Plan A. The sampled interval is exactly fixed; the DFT logical period $T\Delta\lambda$ changes slightly, from 6.0 at T1200 to 5.9975 at T2399, rather than undergoing the large change associated with domain extension.

The result is compatible with the coupled Plan A explanations involving physical-frequency/discrete-mode mapping (Candidate 1), global spectral representation (Candidate 3), and lambda coordinate/domain representation (Candidate 4). It does not isolate any one of them. In particular, successful fixed-domain refinement does not rule out global-representation or hidden-feature bandwidth sensitivity when the physical domain itself expands. This is **indirect evidence about Plan A**, not proof of its unique causal mechanism.

## 7. Main conclusions

### Supported

1. Under this fixed-Q400, fixed-physical-domain protocol, changing resolution is insufficient to reproduce the Plan A collapse.
2. The historical T1200 FNO2D can be evaluated with frozen weights through the tested 4× refinement, retaining low absolute error.
3. Both coarse→fine and fine→coarse have formal low-error results.
4. Training resolution affects the observed error level; matched T2399 training is more favorable on the still-finer tested grids in this experiment.
5. Practical resolution robustness does not imply exact invariance: the common-node predictions change measurably.

### Not established

1. There is no arbitrary-resolution guarantee or established final resolution boundary.
2. Resolutions beyond T4797 have not been validated by these experiments.
3. No unique Plan A mechanism or quantitative causal error decomposition has been established.
4. These results do not generalize automatically to all FNO architectures, Kerr parameter families, or tasks.
5. Finer training has not been shown to be universally better, nor has a universal advantage of one transfer direction been established.
6. Exact continuous-operator invariance has not been demonstrated.

## 8. Sources / provenance

All paths are relative to the project root. This summary refers to existing formal assets without copying datasets, predictions, or checkpoints.

**Research records**

- `FNO_KERR_CURRENT_STATE.md` — Plan B status and its relation to the Plan A observations.
- `FNO_KERR_EXPERIMENT_PLAN.md` — Sections 3.1–3.7: locked controls and completed FNO-only experiments.
- `FNO_KERR_REASONING_LOG.md` — Episodes 12, 14, 15, and 17: protocol rationale, evidence, and interpretation boundaries.
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` — Plan B asset identities and provenance.

**Initial paired-truth and frozen evaluation**

- `outputs/plan_b_q400_t1200_to_t2399/ground_truth_consistency.json`
- `outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/metrics.json`
- `outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/experiment_summary.json`
- The saved-prediction common-node shift is recorded in `FNO_KERR_EXPERIMENT_PLAN.md`, Section 3.5, and `FNO_KERR_REASONING_LOG.md`, Episode 14.

**Matched bidirectional experiment**

- `outputs/plan_b_bidirectional_t1200_t2399/training/run_config.json`
- `outputs/plan_b_bidirectional_t1200_t2399/experiment_summary.json`
- `outputs/plan_b_bidirectional_t1200_t2399/bidirectional_resolution_matrix.json`
- `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation/metrics.json`
- `outputs/plan_b_bidirectional_t1200_t2399/theta2399_q400_evaluation/experiment_summary.json`

**Resolution-range experiment**

- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/resolution_range_matrix.json`
- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/experiment_summary.json`
- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/ground_truth_qualification/t3598.json`
- `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/ground_truth_qualification/t4797.json`

**Plan A comparison**

- `outputs/formal_a1_length_extrapolation/q400_t1200_t1800_t2400_w0p5/a1_length_extrapolation_summary.json`
