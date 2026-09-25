# FNO–Kerr results package — 2026-09-25

## 1. Project scope

This package summarizes the completed work on **physical-domain length extrapolation**, **FNO-only fixed-domain cross-resolution**, and the **six-model cross-resolution benchmark**. It provides dataset definitions, experiment protocols, formal results, scientific conclusions, and provenance links for research review.

The code resides in the project Git repository. Raw datasets, predictions, and checkpoints are excluded from this package; the linked summaries identify their formal project-relative sources.

## 2. Research progression

```text
Plan A: fixed delta_lambda, extend the physical domain
  -> severe length sensitivity in the historical global FNO2D
M1–M4 diagnostics
  -> evidence for multiple coupled global spectral representation factors
R1 / R2 / R3 redesign
  -> partial improvements in R2/R3, no complete repair; R4 not started
Plan B: fix the physical domain, change resolution only
  -> the historical FNO remains accurate on denser grids
Track A: extend the fixed-domain test to six architectures
  -> native accuracy and resolution robustness show distinct behaviors
```

The four candidate explanations are not in one-to-one correspondence with M1–M4. **Candidate 4 and M4 have formal results.** R1–R3 are subsequent redesign experiments; the unstarted item is R4, the proposed global/local or local/windowed spectral redesign.

## 3. Dataset guide

### Length-extrapolation data

[Length-extrapolation data definition](01_dataset_description/length_extrapolation_data.md): fixed $\Delta\lambda=0.005$, with T1200 → T1800/T2400 extending the sampled physical domain from [0,5.995] to [0,8.995]/[0,11.995].

### Cross-resolution data

[Cross-resolution data definition](01_dataset_description/cross_resolution_data.md): fixed physical domain [0,5.995], with T1200/T2399/T3598/T4797 using progressively smaller $\Delta\lambda$ and denser sampling.

These documents distinguish the matched n2000 training sources from the canonical Q400 evaluation datasets. **Domain extension and resolution refinement must not be conflated.**

## 4. Main result 1 — Length extrapolation

See [Plan A summary](02_length_extrapolation/summary.md) for the protocol, diagnostics, redesign results, and sources.

The historical FNO2D has native T1200 **mean-per-Q raw-xyz Relative L2 ≈ 0.00543**, but full-domain errors rise to **≈ 1.872 at T1800** and **≈ 2.062 at T2400**. Prediction changes severely even on the exactly identical shared truth prefix; failure is not confined to the appended extrapolation tail.

M2/M3/M4 support a coupled account involving physical-frequency/mode mapping, global spectral representation, and lambda-domain representation. Large losses of raw xyz high-frequency energy alone are insufficient to explain the collapse. The diagnostics do not isolate a unique cause.

- **R1:** the tested multi-length training repair failed and worsened long-domain errors.
- **R2:** shared-prefix prediction improved, without an extrapolation-region improvement over the baseline.
- **R3:** long-domain prediction improved and supplied strong mechanism evidence, at a substantial cost to native accuracy.
- **R4:** not started; Plan A paused at this decision point.

**Plan A supports strong physical-domain length sensitivity in the tested global FNO2D, but does not establish a single unique mechanism or a complete repair.**

## 5. Main result 2 — FNO-only cross-resolution

See [Plan B summary](03_fno_cross_resolution/summary.md) for the initial, bidirectional, and resolution-range experiments.

Using the **same historical T1200 FNO2D checkpoint**, mean-per-Q raw-xyz Relative L2 is **≈ 2.062 for Plan A long-domain T2400**, versus **≈ 0.00700 for Plan B fixed-domain T4797**. Changing T alone is therefore insufficient to reproduce the observed Plan A collapse under this protocol.

Both coarse→fine and fine→coarse have low-error frozen-transfer results, and the range experiment reaches **4× finer sampling**. This is practical resolution robustness, not exact invariance or an arbitrary-resolution guarantee. The contrast provides **indirect evidence** about Plan A; it does not establish its unique causal mechanism.

## 6. Main result 3 — Six-model benchmark

See [Track A Phase I summary](04_cross_model_benchmark/summary.md) for the matched-capacity protocol and full result matrix. Its metric is **Global raw-xyz Relative L2**, distinct from the mean-per-Q values quoted above.

| Model | Observed behavior under this protocol |
| --- | --- |
| FNO1D | Very high native accuracy and low absolute cross-resolution error |
| TimesNet | Low absolute error and milder relative degradation than FNO1D |
| ResNet / BiLSTM | High native accuracy and strong resolution sensitivity |
| Transformer / DeepONet | Nearly flat cross-resolution curves with higher absolute error baselines |

**Native accuracy and resolution robustness are different evaluation axes.** Phase I is complete: **6/6 models, 12/12 training runs, and 48/48 evaluation cells**, spanning two training resolutions and four evaluation resolutions on the fixed domain. Track A uses FNO1D; it is not the historical FNO2D checkpoint studied in Plans A/B.

## 7. What the results support

1. Physical-domain extension and resolution refinement are different generalization problems.
2. The tested historical FNO fails badly under domain extension but remains usable under fixed-domain resolution refinement.
3. Native accuracy does not imply resolution robustness.
4. Different architectures exhibit qualitatively different resolution-transfer behavior in the tested implementations and protocol.
5. FNO1D and TimesNet provide low absolute cross-resolution error in this protocol.
6. Flat resolution curves alone do not imply high accuracy.

## 8. What remains unresolved

- Plan A mechanisms remain coupled; their individual causal contributions have not been quantified.
- R4 local/global spectral redesign has not been tested, and the R3 native-accuracy tradeoff remains unresolved.
- Plan A long-domain datasets were used for diagnosis and redesign; independent confirmation after design freeze remains outstanding.
- Track A uses one seed (27). It does not establish statistical robustness or a universal architecture ranking; matched parameter counts do not guarantee optimal training for every model.
- Track A does not test long-domain extrapolation, so no such conclusion follows for its six models.
- Broader Kerr parameter families and resolution ranges beyond those reported here have not been verified.

## 9. Reading order

1. This README for the research progression and conclusion boundaries.
2. The two [length-extrapolation](01_dataset_description/length_extrapolation_data.md) and [cross-resolution](01_dataset_description/cross_resolution_data.md) data definitions.
3. [Plan A results](02_length_extrapolation/summary.md).
4. [Plan B results](03_fno_cross_resolution/summary.md).
5. [Track A results](04_cross_model_benchmark/summary.md).
6. Project-level [CURRENT_STATE](../../FNO_KERR_CURRENT_STATE.md), [EXPERIMENT_PLAN](../../FNO_KERR_EXPERIMENT_PLAN.md), [REASONING_LOG](../../FNO_KERR_REASONING_LOG.md), and [SERVER_DATA_EXPERIMENT_REGISTRY](../../SERVER_DATA_EXPERIMENT_REGISTRY.md) for deeper provenance.

## 10. Repository / provenance

Repository state at README creation:

```text
branch: codex/clean-research-history-20260905
HEAD: 594f0eaf331b0c3a8b4c6342f5c554ab64c3a2a1
```

This HEAD identifies the repository baseline at preparation time, not a commit containing this newly generated package or the generation commit of every historical artifact. Each detailed summary provides project-relative links or paths to its formal evidence.

**This results package contains summaries and metadata only; raw datasets, predictions, and checkpoints are intentionally excluded from the mentor-facing package.**
