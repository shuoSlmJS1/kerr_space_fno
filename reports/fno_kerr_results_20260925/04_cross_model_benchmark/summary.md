# Cross-model Resolution Benchmark Protocol v1 — Track A Phase I

This summary consolidates the reviewed six-model Phase I evidence as of 2026-09-25. It separates absolute accuracy, relative resolution robustness, and interpretations that remain unproven.

## 1. Research question

Plan B established practical fixed-domain cross-resolution capability for the historical FNO2D. Track A asks whether different model architectures exhibit different resolution-generalization behavior under the same task, input information, data, and comparable parameter capacity.

The formal design is **6 models × 2 training resolutions × 4 evaluation resolutions = 48 evaluation cells**. Each model trains separately at T1200 and T2399, producing 12 training runs. Each selected BEST checkpoint is then frozen and evaluated at T1200, T2399, T3598, and T4797.

Every grid preserves $\lambda\in[0,5.995]$. **Track A tests fixed-domain resolution transfer, not Plan A physical-domain length extrapolation.** All models perform trajectory-wise direct regression with the information $[Q,\lambda]\to xyz$; historical sparse-reconstruction baselines are not substituted for this task.

## 2. Models and capacity

| Model | Locked configuration | Real scalar parameters |
| --- | --- | ---: |
| FNO1D | modes=32, width=64, depth=4, GELU | 1,069,763 |
| Dilated ResNet | width=84, kernel=7, 11 residual blocks, dilations 1,2,4,…,1024 | 1,096,119 |
| canonical TimesNet | d_model=80, d_ff=96, 2 blocks, top-k=2, kernels=1/3/5, dropout=0 | 1,077,059 |
| BiLSTM | 2-layer bidirectional LSTM, hidden=184, dropout=0, linear xyz head | 1,093,331 |
| Transformer | Encoder-only; d_model=192, 6 heads, 2 layers, FF=1024, ReLU/post-norm, dropout=0.1 | 1,088,003 |
| DeepONet | Q branch / lambda trunk; four tanh hidden layers each, width=384, latent=128 | 1,085,699 |

All six models fall within the accepted **0.9M–1.3M real scalar parameter** band. The primary count measures registered trainable real scalar parameter coordinates, not effective functional degrees of freedom. In these implementations, `tensor_numel` equals the reported real scalar count.

Matched parameter scale is a capacity control; it does not imply equal functional freedom, computational complexity, or optimal training state. Configurations were selected within the locked capacity constraints, not by tuning against formal evaluation accuracy.

DeepONet organizes the same information as a Q-only branch and lambda-only query trunk; it receives no additional physical information. The Transformer uses full attention without a learned positional embedding tied to a fixed T. Canonical TimesNet retains runtime FFT-based period selection and folding.

## 3. Unified protocol

| Item | Value |
| --- | --- |
| epochs | 500 |
| batch | 32 |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-4 |
| scheduler | ExponentialLR, gamma=0.995 |
| seed | 27 |
| training objective | normalized-space MSE |
| BEST selection | minimum validation normalized MSE |
| target transform | raw |

Train T1200 uses `q_1p6-3_n2000_t1200`; Train T2399 uses `q_1p6-3_n2000_t2399_plan_b_matched_v1`. Both retain the same Q identities, split membership, and row ordering, with **1400/300/300 train/validation/test trajectories**.

Normalization is fitted only on the corresponding training-resolution train split. Mean/std reductions and stored statistics preserve float64 precision; model training and input/target application remain float32. This is not float64 model training.

Each model receives only the same trajectory information, normally organized as `[Q_broadcast, lambda]` with shape `(B,T,2)` and xyz target `(B,T,3)`. Frozen evaluation uses the same BEST checkpoint across all four resolutions, with no adaptation or normalization refit.

All models evaluate the same independent Q400 identities, physical truth definition, and canonical ascending-Q order. All 400 trajectories are used; the evaluation field is not used for training. Evaluation batch size remains 32, with a final batch of 16, including for canonical TimesNet's batch-shared period-selection behavior.

## 4. Full 48-cell result matrix

This section consistently reports **Global raw-xyz Relative L2**: the prediction-error norm over all Q, lambda, and xyz entries divided by the corresponding truth norm. It must not be mixed with the mean-per-Q values emphasized in the Plan A and Plan B comparisons.

All 48 entries were checked against the formal per-cell metrics and the 12 evaluation matrices. Displayed values are rounded; subsequent ratios and percentage comparisons use the unrounded metrics.

| Model / Train T | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| FNO1D / 1200 | 0.000305 | 0.002608 | 0.003466 | 0.003895 |
| FNO1D / 2399 | 0.002622 | 0.000296 | 0.000892 | 0.001307 |
| ResNet / 1200 | 0.000876 | 1.236419 | 1.151856 | 1.101031 |
| ResNet / 2399 | 1.080739 | 0.000624 | 1.035577 | 1.031188 |
| TimesNet / 1200 | 0.001573 | 0.003838 | 0.004918 | 0.005469 |
| TimesNet / 2399 | 0.003263 | 0.001656 | 0.001943 | 0.002227 |
| BiLSTM / 1200 | 0.000817 | 0.301107 | 0.407212 | 0.461684 |
| BiLSTM / 2399 | 0.202827 | 0.000990 | 0.084067 | 0.125444 |
| Transformer / 1200 | 0.02604342 | 0.02588183 | 0.02582833 | 0.02580166 |
| Transformer / 2399 | 0.03260342 | 0.03225286 | 0.03213670 | 0.03207880 |
| DeepONet / 1200 | 0.00572016 | 0.00567443 | 0.00565957 | 0.00565221 |
| DeepONet / 2399 | 0.00944202 | 0.00939610 | 0.00938115 | 0.00937374 |

## 5. Two evaluation axes

### Native accuracy

Native accuracy is Train T1200→Test T1200 or Train T2399→Test T2399. It measures how accurately each model predicts at its own training resolution on the independent Q400 field.

| Model | T1200 native | T2399 native |
| --- | ---: | ---: |
| FNO1D | 0.00030468 | 0.00029556 |
| ResNet | 0.00087579 | 0.00062432 |
| TimesNet | 0.00157255 | 0.00165634 |
| BiLSTM | 0.00081664 | 0.00099042 |
| Transformer | 0.02604342 | 0.03225286 |
| DeepONet | 0.00572016 | 0.00939610 |

### Resolution robustness

For one fixed BEST checkpoint, let $E(T)$ denote its Global raw-xyz Relative L2 at evaluation resolution T. Relative robustness is described by

$$
R(T)=\frac{E(T)}{E(T_{\mathrm{native}})}.
$$

The following range covers the six non-native cells per model: three for each training resolution.

| Model | Non-native/native ratio range |
| --- | ---: |
| FNO1D | 3.02–12.79 |
| TimesNet | 1.17–3.48 |
| ResNet | 1257–1731 |
| BiLSTM | 84.88–565.35 |
| Transformer | 0.9907–1.0109 |
| DeepONet | 0.9881–1.0049 |

**Ratios describe relative change and cannot replace absolute error.** FNO1D's larger ratios partly reflect its exceptionally low native baseline; its off-native absolute errors remain the lowest among the six models in every corresponding cell. Conversely, the flat Transformer and DeepONet curves do not make those models the most accurate.

Native accuracy and resolution robustness are distinct evaluation axes, not a single combined ranking. A ratio below one indicates a slightly lower sampled-grid aggregate error, not automatically an improved continuous predictor.

## 6. Per-model behavior

| Model | Behavior supported by this experiment |
| --- | --- |
| **FNO1D** | Lowest errors in both native cells and all corresponding off-native cells. Relative degradation exceeds that of TimesNet, Transformer, and DeepONet, but cross-resolution absolute error remains low. T2399 training substantially improves still-finer-grid evaluation. This is strong native accuracy and low transfer error under the present protocol, not universal architectural superiority. |
| **TimesNet** | Native errors are clearly higher than FNO1D's. Off-native errors remain approximately 0.00194–0.00547, with smaller relative degradation than FNO1D. It exhibits another accuracy/robustness balance; FFT and period folding do not prove invariance. |
| **ResNet** | Accurate native fits coexist with order-one errors at every off-native cell. It is among the most resolution-sensitive patterns. The theoretical receptive field is 12349, covering the tested lengths, so failure cannot simply be attributed to a receptive field shorter than the sequence; the mechanism remains unresolved. |
| **BiLSTM** | Native errors are around $10^{-3}$, but off-native errors rise to approximately 0.0841–0.4617. T2399 training improves finer-grid transfer substantially, while residual errors remain far above native levels. |
| **Transformer** | Each checkpoint has a nearly flat four-grid curve, with an absolute baseline around 0.026–0.033. A stable high-error baseline is observed. An approximation/error floor is a candidate explanation, not a verified optimization or architectural mechanism. |
| **DeepONet** | The four-grid curves are very flat. A Q-branch/lambda-query formulation is consistent with weak grid dependence, but has not been isolated as its cause. The T2399-trained model is worse than the T1200-trained model on all four grids. |

## 7. Training-resolution effect

The percentages below compare the two separately trained models on the **same evaluation grid**:

$$
100\left(\frac{E_{\mathrm{Train2399}}(T)}{E_{\mathrm{Train1200}}(T)}-1\right).
$$

Negative values indicate lower error after T2399 training. These are not the native-normalized ratios in Section 5.

| Model | Change at T3598 | Change at T4797 | Interpretation |
| --- | ---: | ---: | --- |
| FNO1D | -74.27% | -66.46% | Finer training strongly helps these finer grids |
| TimesNet | -60.50% | -59.28% | Finer-grid evaluation improves |
| BiLSTM | -79.36% | -72.83% | Large improvement, but substantial transfer error remains |
| ResNet | -10.09% | -6.34% | Small improvement, still order-one error |
| Transformer | +24.42% | +24.33% | T2399 training is worse |
| DeepONet | +65.76% | +65.84% | T2399 training is worse |

For FNO1D, TimesNet, BiLSTM, and ResNet, the T1200-trained model remains better at Test T1200, while the T2399-trained model is better at Test T2399. Transformer and DeepONet instead have worse T2399-trained results at all four evaluation grids.

**Finer training is not universally better.** These comparisons establish observed differences between the trained models, without assigning them to an unverified optimization, normalization, or representation mechanism.

## 8. Main findings

### Supported

1. Native accuracy and resolution robustness are distinct evaluation axes.
2. The six architectures exhibit markedly different resolution-transfer behavior under the same task and comparable nominal capacity.
3. FNO1D and TimesNet retain relatively low absolute error together with useful cross-resolution transfer, although their relative stability and absolute accuracy differ.
4. ResNet and BiLSTM exhibit high native accuracy combined with strong resolution sensitivity.
5. Transformer and DeepONet exhibit flatter cross-grid curves but higher absolute error baselines.
6. Fixed-domain resolution transfer shows clear architecture-dependent behavior in this controlled set of implementations and training runs.
7. Low-error resolution generalization is not exclusive to FNO in this experiment, but neither is it a capability naturally demonstrated by every tested model.

## 9. Not established

1. These results do not define a universal architecture ranking.
2. They do not prove that FNO outperforms alternatives on every task, capacity, data scale, or training budget.
3. Parameter matching and a unified training protocol do not establish that every architecture reached its optimal training state.
4. Transformer or DeepONet stability has not been causally attributed to a specific architectural mechanism.
5. A single seed, 27, does not establish statistical robustness or significance across seeds.
6. A flat error curve is not equivalent to high accuracy or exact resolution invariance.
7. Track A does not test Plan A long-domain extrapolation.
8. The matrix cannot establish the tested models' capabilities outside the training physical domain.

## 10. Phase I completion

| Formal completion item | Confirmed count |
| --- | ---: |
| Models | 6/6 |
| Training runs completed through epoch500 | 12/12 |
| Evaluation cells | 48/48 |
| BEST checkpoints | 12/12 |
| Training summaries | 12/12 |
| Evaluation matrices | 12/12 |
| Per-cell metrics | 48/48 |
| Final prediction files | 48/48 |

The reviewed histories cover epochs 1–500, and selected BEST epochs match minimum validation normalized MSE. Matrix and per-cell results agree; the recorded BEST identities and evaluation dataset/Q provenance are consistent. The latest read-only audit checked checkpoint/prediction presence and nonzero size without reloading all large artifacts or rerunning inference; earlier integrity audits are recorded in the research records.

**Transformer / Train T1200 was interrupted after its last valid epoch225 checkpoint and then continued through epoch500 using the formal deterministic resume.** This interruption/recovery incident belongs to the original 12 training runs, not a thirteenth run, and did not change the scientific protocol. The specific infrastructure failure cause remains unconfirmed; disk pressure was recorded as a suspect, not a proven cause. The study must not be described as twelve uninterrupted trainings.

## 11. Remaining uncertainty

- The specific sources of ResNet/BiLSTM resolution sensitivity have not been isolated.
- The cause of Transformer's flat but higher-error baseline remains unresolved.
- DeepONet's weak grid dependence is consistent with its query formulation, but its mechanism has not been causally separated from approximation effects.
- Why T2399 training helps some models but worsens Transformer and DeepONet remains an open question.
- Multi-seed robustness has not been established.
- Changing sampling density also changes the discrete quadrature underlying global metrics; small cross-grid error changes need not imply changes in the underlying continuous approximation.

## 12. Sources / provenance

All paths below are relative to the project root. This summary references existing assets without copying datasets, predictions, or checkpoints.

**Research records**

- `FNO_KERR_CURRENT_STATE.md` — Sections 15.1–15.10, including completion audits and the six-model synthesis.
- `FNO_KERR_EXPERIMENT_PLAN.md` — Sections 7.2–7.5: data, task alignment, capacity counting, model configurations, and training/evaluation protocol.
- `FNO_KERR_REASONING_LOG.md` — benchmark design, precision controls, and Waves 1–6 interpretation boundaries.
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` — Sections 9.1–9.6: formal assets and recovery provenance.

**Wave 1 — FNO1D**

- `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/eval_t2399/matrix.json`

**Wave 2 — Dilated ResNet**

- `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave2_resnet_20260912/eval_t2399/matrix.json`

**Wave 3 — canonical TimesNet**

- `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913/eval_t2399/matrix.json`

**Wave 4 — BiLSTM**

- `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913/eval_t2399/matrix.json`

**Wave 5 — encoder-only Transformer**

- `outputs/benchmark_track_a_v1/phase_i_wave5_transformer_20260913/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave5_transformer_20260913/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave5_transformer_20260913/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave5_transformer_20260913/eval_t2399/matrix.json`

**Wave 6 — DeepONet**

- `outputs/benchmark_track_a_v1/phase_i_wave6_deeponet_20260913/train_t1200/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave6_deeponet_20260913/train_t2399/summary.json`
- `outputs/benchmark_track_a_v1/phase_i_wave6_deeponet_20260913/eval_t1200/matrix.json`
- `outputs/benchmark_track_a_v1/phase_i_wave6_deeponet_20260913/eval_t2399/matrix.json`

Each evaluation matrix contains the four formal cells and their native-relative comparisons; the corresponding per-cell metrics are retained alongside it. Wave 6's directory retains the requested 20260913 suffix; its actual formal execution was on 2026-09-16.
