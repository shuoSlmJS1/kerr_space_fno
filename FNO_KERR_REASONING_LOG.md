# FNO Kerr Research Reasoning Log

## Purpose and scope

`FNO_KERR_CURRENT_STATE.md` is the compact record of formal evidence and the current
scientific conclusions.

`FNO_KERR_REASONING_LOG.md` is a research-reasoning, cause-tracing, and learning record.
It preserves how questions arose, why alternatives were considered, which checks could
discriminate among them, and how the evidence changed the next decision. It is not an
experiment registry, a replacement for formal artifacts, or a chronological dump of
commands and JSON output.

Many early diagnostic ideas and mechanism hypotheses were proposed with AI assistance.
The purpose of this log is not to retroactively claim independent discovery, but to
preserve the reasoning pattern so it can be studied, reproduced, and gradually
internalized by the researcher.

## Reusable episode template

## Episode / Question

### Observation

What did we observe?

### Why this was surprising

Why could the result not be accepted at face value?

### Candidate explanations

What mechanisms or errors could explain it?

### Why these candidates were considered

What mathematical, numerical, ML, or software principle suggested them?

### Discriminating check / experiment

What test separates the explanations?

### Result

What actually happened?

### Evidence update

Which explanations became stronger or weaker?

### What this did NOT prove

What conclusions remain unjustified?

### Next question

Why did the next experiment naturally follow?

## Episode 1 — Historical length-extrapolation failure looked suspicious

### Observation

A historical frozen T1800 input produced extremely poor prediction on the shared T1200
prefix even though the same model was accurate on T1200.

### Why this was surprising

A long-input failure can look like a model generalization failure, but it can also be an
invalid comparison caused by different data, solver behavior, target representations, or
input-axis semantics. Interpreting the number before checking those possibilities would
be unsafe.

### Candidate explanations

The initial alternatives were a short/long data-pairing mismatch, a trajectory-solver or
truth mismatch, a numerical representation artifact, an invalid model-input convention,
or genuine frozen-model length sensitivity.

### Why these candidates were considered

Scientific evaluation requires identical quantities to be compared under the intended
protocol. In numerical ML, a large error is not automatically evidence about the model if
the data or tensor contract may be wrong.

### Discriminating check / experiment

First validate raw paired datasets and their prefixes directly; then audit the evaluation
input construction rather than treating a historical saved target as ground truth.

### Result

The subsequent Stage-2 validator and the Q-axis audit showed that both data identity and
model-input semantics needed explicit controls.

### Evidence update

The historical result became a question to diagnose rather than a standalone negative
scientific conclusion.

### What this did NOT prove

It did not yet prove that the data were valid, that the model was length-sensitive, or
that any specific FNO mechanism caused the failure.

### Next question

Are the raw T1200, T1800, and T2400 trajectories exact-prefix companions?

## Episode 2 — Exact-prefix identity

### Observation

Temporary truth-prefix comparisons once reported a maximum absolute difference of
`9.536743164062e-07`.

### Why this was surprising

Even a very small apparent discrepancy could undermine an exact-prefix comparison if it
came from the raw datasets themselves. But it could also arise after transformations that
are not part of trajectory generation.

### Candidate explanations

The alternatives were a raw dataset mismatch, solver non-reproducibility, or a float32 /
model-space normalization and recovered-target roundtrip.

### Why these candidates were considered

Raw data identity and numerical representation identity are different claims. Converting
raw float64 values through float32 and later recovering raw-space values can introduce an
error around a float32 ULP scale without changing the underlying trajectory.

### Discriminating check / experiment

Stage-2 compared the underlying raw train/val/test datasets directly, preserved source
identity, and tested all three prefix pairs before model inference.

### Result

`short_to_medium`, `short_to_long`, and `medium_to_long` were all `EXACT_PREFIX`; raw
trajectory-prefix differences were exactly zero. The `9.536743164062e-07` observation
came from comparing different representation pipelines: raw short truth versus a long
saved target that had passed through float32/model-space normalization and/or recovery.

### Evidence update

Ground-truth mismatch was ruled out. Current formal diagnostics use raw short-dataset
float64 truth as the primary scientific reference and promote predictions to float64 for
metrics.

### What this did NOT prove

Exact data pairing does not by itself explain why a frozen FNO prediction changes with
length.

### Next question

Is the FNO2D input field being constructed with the correct Q-axis semantics?

## Episode 3 — Q-axis ordering bug

### Observation

The first formal diagnostic unexpectedly produced a poor T1200 short baseline, contrary
to known historical same-domain performance.

### Why this was surprising

The tensor was shape-valid, so a superficial tensor check would not reveal the problem.
Yet a correct same-domain baseline should not collapse solely because examples were
concatenated from dataset splits.

### Candidate explanations

Possible causes included a checkpoint mismatch, an incorrect normalization/target path,
or an invalid ordering of the Q dimension.

### Why these candidates were considered

In FNO2D, Q is not merely a batch-row label: it is the `H` operator/grid axis in the
input field, and the spectral convolution applies an FFT over both Q and lambda. Split
source-row order is therefore not a scientifically valid substitute for the model's
canonical Q field.

### Discriminating check / experiment

Audit the tensor convention and rebuild the evaluation field using a stable ascending-Q
permutation applied identically to Q and y.

### Result

The earlier execution used scrambled split order on the Q axis. Reconstructing canonical
ascending Q restored the accurate T1200 baseline. The scrambled run is retained only as a
protocol-debug artifact, not as scientific model-performance evidence.

### Evidence update

The evaluation protocol now requires canonical Q400 construction after source identity is
preserved and validated.

### What this did NOT prove

Correcting Q order did not make long-domain prediction accurate and did not identify the
mechanism of long-domain sensitivity.

### Next question

Under the corrected, one-shot frozen protocol, how does performance change jointly across
T1200, T1800, and T2400?

## Episode 4 — Formal A1 length sensitivity

### Observation

The formal A1 evaluator measured prefix `mean_per_q_relative_l2` of about `0.0054` for
T1200, `1.70` for T1800, and `2.27` for T2400. Extrapolation-window errors were also
non-monotonic with physical distance from the training boundary.

### Why this was surprising

The raw T1200 truth prefix is exactly identical in the three companions, and the model is
frozen and evaluated with one full-field forward pass. A simple story of accumulating
step-by-step rollout error cannot describe this result.

### Candidate explanations

The next candidates concerned physical-frequency/discrete-mode remapping, fixed retained
physical bandwidth, global spectral dependence on the whole lambda field, and explicit
lambda-coordinate extrapolation after normalization.

### Why these candidates were considered

Changing physical length at approximately fixed `delta_lambda` changes both the DFT basis
and the values of an explicit coordinate channel. These effects are properties of the
representation, not merely of the extrapolation target.

### Discriminating check / experiment

Separate descriptive checks of raw spectra from internal representation measurements and
coordinate interventions.

### Result

The A1 phenomenon was established, but no single causal mechanism was established at this
stage.

### Evidence update

The project moved from performance measurement to a bounded mechanism study (M1–M4).

### What this did NOT prove

It did not imply that FNO length extrapolation is impossible, nor that the problem is
standard fixed-domain grid-resolution generalization.

### Next question

Which representation changes are mathematically forced by changing lambda-domain length?

## Episode 5 — Why Fourier mechanisms were proposed

### Observation

FNO2D applies global Fourier operations over the Q/lambda field while retaining a fixed
number of discrete lambda modes.

### Why this was surprising

A fixed mode index may look like a stable feature identifier, but its physical meaning is
not fixed when the DFT period changes.

### Candidate explanations

Candidates 1–3 were physical-frequency/discrete-index remapping, fixed retained
physical-bandwidth shrinkage, and global spectral representation changes to the shared
prefix. Candidate 4 was lambda-coordinate extrapolation / normalization shift.

### Why these candidates were considered

For `N` samples at spacing `delta_lambda`, the DFT physical frequency is
`f_k = k / (N * delta_lambda)`. Increasing N at fixed spacing changes the physical
frequency represented by a fixed k and lowers the physical cutoff of a fixed retained
mode count. Separately, lambda is an explicit normalized input channel, so appended
values can affect pointwise lifting and later global operations.

### Discriminating check / experiment

M1 audited the actual FFT and normalization implementation. M2 measured raw truth
spectral energy; M3 localized internal divergence; M4 examined checkpoint coordinate
range and used a controlled clamp intervention.

### Result

These were AI-assisted mechanism hypotheses, not conclusions. Each was converted into a
specific observable or controlled test.

### Evidence update

The investigation became testable without modifying the FNO architecture or retraining.

### What this did NOT prove

Generic FNO theory alone cannot establish what caused the observed Kerr error.

### Next question

Before blaming the fixed 32-mode cutoff, does it actually exclude a large amount of raw
Kerr trajectory energy?

## Episode 6 — M2 raw spectral energy

### Observation

The lambda spectral branch retains `modes2 = 32`, i.e. indices `k=0..31`, and its
physical cutoff shrinks when domain length grows.

### Why this was surprising

A shrinking cutoff is mathematically real, but it is not automatically a practically
important loss of information for the raw Kerr trajectories.

### Candidate explanations

The immediate hypothesis was that large amounts of important raw high-frequency energy
fall outside the T1800/T2400 retained physical band.

### Why these candidates were considered

If most trajectory energy were beyond the retained cutoff, simple bandwidth loss would be
a plausible primary explanation for poor long-domain prediction.

### Discriminating check / experiment

M2 used canonical-Q, raw float64 xyz truth only, a one-sided lambda FFT, and physical
frequency `f_k = k / (N * delta_lambda)`. It compared energy fractions across the
length-specific cutoffs and common physical-frequency bands without loading a checkpoint.

### Result

About 99% of raw xyz spectral energy was below even the strict T2400 cutoff. Dominant
content around similar physical frequencies nevertheless occupied different discrete mode
indices as total length changed.

### Evidence update

Candidate 2 was weakened as a simple raw-truth energy-loss explanation. Candidate 1
received descriptive real-data support for index remapping.

### What this did NOT prove

Raw xyz spectra are not the same as spectra of hidden features, and a small raw
above-cutoff fraction does not make pointwise or nonlinear FNO paths irrelevant.

### Next question

If simple raw bandwidth loss is insufficient, where inside the frozen FNO does the
identical shared prefix first diverge?

## Episode 7 — M3 internal representation localization

### Observation

T1200/T1800/T2400 have an identical shared raw prefix, but their final prefix predictions
differ greatly.

### Why this was surprising

If normalized input and pointwise lifting preserve the shared prefix, a later global
operation must be introducing or transmitting the length sensitivity.

### Candidate explanations

The alternatives were divergence before the spectral layer, divergence at the first
FFT/spectral operation, or amplification only in deeper nonlinear blocks.

### Why these candidates were considered

Pointwise operations operate independently at each grid location, whereas FFT-based
spectral operations mix information across the full lambda axis. Physically aligned
frequency bins can also map to different learned discrete weights after length changes.

### Discriminating check / experiment

M3 reused one frozen checkpoint, executed one normal forward per length, and used
observational hooks plus read-only replicated FFT/spectral multiplication. It reduced all
captured tensors to compact shared-prefix statistics.

### Result

Normalized input, lifted feature, and first spectral input prefix differences were zero.
Strong divergence appeared immediately after the first global spectral operation. At
aligned physical frequencies, different discrete indices and learned `R_k` weights could
substantially change hidden-vector relationships.

### Evidence update

Candidates 1 and 3 became strongly supported as real internal mechanism pathways;
candidate 3 was localized as the earliest observed entry point.

### What this did NOT prove

The first spectral operation was not shown to be the sole cause, and the measurement did
not isolate coordinate-range influence from domain/basis effects.

### Next question

How much measurable influence comes from the appended lambda coordinates that lie outside
the training coordinate range?

## Episode 8 — M4 coordinate-range hypothesis

### Observation

Lambda is explicitly supplied as an input coordinate. T1800 and T2400 append values
outside the T1200 training range, and those values participate in global FFT processing.

### Why this was surprising

The coordinate effect is coupled to global spectral effects: changing appended lambda
values may alter the entire Fourier representation even when the shared prefix itself is
unchanged.

### Candidate explanations

Candidate 4 predicted a measurable contribution from out-of-training-range normalized
lambda magnitude; alternatives were that the coordinate channel was negligible or that
all observed effects came only from length/basis changes.

### Why these candidates were considered

M4a verified the actual checkpoint's standard normalization and `[Q, lambda]` channel
order. The lambda term has a direct trainable route into each input-projection channel,
then contributes to the full-field spectral transform.

### Discriminating check / experiment

M4b deliberately clamped only appended normalized lambda values above the T1200 training
upper bound, while holding each long arm's length, FFT grid, Q, truth, checkpoint,
weights, and shared-prefix coordinates fixed.

### Result

This intentionally nonphysical probe measurably changed early spectral and hidden
representations, with larger response for T2400. Early representations moved partly
toward T1200, but final outputs moved farther from the T1200 reference.

### Evidence update

Candidate 4 has a clear measurable contribution under this intervention, but it is
coupled and non-dominant under the tested clamp. The clamp is not a production fix or a
valid Kerr prediction protocol.

### What this did NOT prove

It did not assign a percentage of final error to candidate 4, show that another
coordinate representation would solve the problem, or establish candidate 4 as unique.

### Next question

Can mechanism-driven repair experiments improve direct one-shot length extrapolation
under a clean frozen/retrained comparison protocol?

## Research habits learned

- Verify raw data identity before interpreting a model failure.
- Distinguish floating-point representation artifacts from scientific data mismatches.
- Preserve axis semantics: a technically valid tensor can still be scientifically malformed.
- Separate an observed failure from a causal explanation for it.
- Distinguish descriptive diagnostics from controlled interventions.
- Hold variables fixed whenever a mechanism is being isolated.
- Treat a failed hypothesis test as useful evidence, not as wasted work.
- Do not convert a metric change directly into a causal-contribution percentage.
- Record negative results and protocol mistakes differently: the former may be evidence,
  while the latter must not be promoted as scientific performance.
- Keep physical-domain length extrapolation distinct from fixed-domain grid-resolution
  generalization.

## Provenance and authorship boundary

This reasoning log reconstructs the scientific decision path from repository evidence,
experiment outputs, user decisions, and AI-assisted analysis. It should not be presented
as a contemporaneous independent human lab notebook when that was not the case. Its role
is truthful learning provenance: to make the reasoning inspectable, reproducible, and
useful for future independent work.
## Episode 9 — R1 failure to R2 coordinate/domain-conditioned repair

### Observation

R1 exposed the unchanged `[Q, lambda]` FNO2D to T600/T800/T1000/T1200 gradient
training, but same-validation-Q evaluation showed a strong alternating response: the
gradient-seen lengths were accurate while T700/T900/T1100 were poor. Its formal
T1800/T2400 results were also worse than R0.

### Why this was surprising

Multiple domain lengths were expected to be the lowest-change way to train robustness to
changing FFT length. Instead, multi-length exposure alone did not produce smooth
within-range interpolation or useful longer-domain extrapolation.

### Candidate explanations

The remaining alternatives included the original absolute-lambda coordinate distribution,
which grows beyond the T1200 training range, and structural discrete-index/global-spectral
effects that R1 deliberately left unchanged.

### Why these candidates were considered

M4 had already shown that appended absolute lambda magnitude measurably changes early
spectral representations, although its nonphysical clamp was not a valid prediction fix.
The next clean intervention was therefore to alter only coordinate representation while
keeping FNO2D spectral parameterization, training-length protocol, loss, and optimizer
update count fixed. This AI-assisted reasoning path was then tested empirically rather
than treated as a conclusion.

### Discriminating check / experiment

R2 trained the same discrete-index FNO2D with `[Q, s, ell]`, where `s = lambda / L`
represents relative position and `ell = L / L_ref` preserves domain-scale information.
Together they avoid an ever-growing absolute-lambda ramp while retaining enough
information to recover physical scale conceptually. The formal evaluator then tested a
frozen R2 best checkpoint at T1200/T1800/T2400 under the same exact-prefix and canonical-Q
controls.

### Result

R2 improved long-input shared-prefix stability relative to R0 and R1: the R0/R2 prefix
values changed from about `1.70 -> 1.41` at T1800 and `2.27 -> 1.50` at T2400. It also
improved full-domain metrics relative to R1. However, T1200 precision degraded from about
`0.0054` (R0) to `0.0705`, and T1800/T2400 extrapolated-region accuracy did not improve
relative to R0.

### Evidence update

Candidate 4 is strengthened as a real contributor by a physically meaningful retrained
repair, not merely a clamp response. Coordinate repair alone is insufficient. Candidates
1 and 3 remain unresolved structural mechanisms because the same physical frequency is
still routed through different discrete `R_k` weights as domain length changes.

### What this did NOT prove

R2 did not show that candidate 4 was the sole cause, that relative coordinates are an
optimal formulation, or that discrete spectral remapping has been repaired. The result
also does not establish that FNO length extrapolation is impossible.

### Next question

With a cleaner coordinate distribution but unchanged discrete spectral weights, can a
physical-frequency-aware multiplier `R(xi_k)`, with
`xi_k = k / (N * delta_lambda)`, reduce the remaining domain-length dependence?

## Episode 10 — R3 physical-frequency-aware spectral repair

### Observation

R2 improved long-input shared-prefix stability but did not improve the true
extrapolated region relative to R0, and it degraded T1200 accuracy. R1 had already shown
that multi-length exposure alone produced an alternating dependence on gradient-seen
lengths and made T1800/T2400 worse than R0.

### Why this was surprising

Cleaning the coordinate distribution was a meaningful intervention, yet it left a large
long-domain failure and introduced an in-domain trade-off. This indicated that coordinate
range was only part of the mechanism rather than a complete explanation.

### Candidate explanations

The central remaining explanation was candidate 1: at fixed `delta_lambda`, the same
physical frequency is represented by different discrete FFT indices as total length
changes. M3 had also localized the first observed divergence to the first global spectral
operation.

### Why these candidates were considered

For example, the same physical frequency can map conceptually as T1200 `k=2`, T1800
`k=3`, and T2400 `k=4`. Under the original operator, those indices use different learned
`R_k` weights. The hypothesis and repair design were developed with AI assistance, then
tested empirically under an explicitly limited intervention rather than treated as a
conclusion.

### Discriminating check / experiment

R3-B1 retained R2 `[Q, s, ell]`, 32 retained discrete lambda bins, global FFT, and the
same depth/width. It changed only `R_k -> R(xi_k)`, using Cartesian-linear interpolation
of 32 physical-frequency anchors with `xi_k = k / (N * delta_lambda)`. It did not change
the retained physical bandwidth or replace the global FFT.

### Result

R3-B1 substantially improved T1800/T2400 prefix and extrapolation metrics relative to
R0: for example, T1800 prefix changed about `1.70 -> 0.83` and T2400 extrapolation about
`1.83 -> 1.36`. But T1200 mean-per-Q Relative L2 degraded sharply from about `0.0054` to
`0.3042`.

### Evidence update

Candidate 1 receives strong positive repair evidence: binding spectral multipliers to
physical frequency rather than discrete index materially changes long-domain behavior.
However, original-domain fidelity remains unresolved, candidate 3's global FFT pathway
was left unchanged, and the result proves neither unique causality nor a complete model.

### What this did NOT prove

R3-B1 did not show that physical-frequency parameterization alone solves length
extrapolation, that the global FFT is harmless, or that the observed trade-off is
acceptable. The anchor-reconstruction precision incident was engineering/provenance only;
the existing epoch-77 checkpoint was valid and required no retraining.

### Next question

Before considering R4, did R3 smooth the response across the same validation-Q lengths
`600, 700, 800, 900, 1000, 1100, 1200`, or does a discrete-length sawtooth remain? This
motivates a seven-length validation-response diagnostic, not an immediate architecture
claim.

## Episode 11 — R3 seven-length response: smaller sawtooth, but not genuine interpolation repair

### Observation

Formal R3-B1 improved T1800/T2400 prefix and extrapolation metrics relative to R0, but its
T1200 fidelity was much worse. The remaining question was whether this long-domain result
also made the model respond smoothly between the fixed training lengths.

### Why this was surprising

A lower error at long endpoints does not imply a smooth response at intermediate domain
lengths. R1 had shown a severe alternating length dependence on exactly this kind of
same-validation-Q comparison, so endpoint evidence alone could not resolve the issue.

### Candidate explanations

R3 may have reduced the discrete-index/physical-frequency mismatch enough to improve
long-domain behavior, while leaving a residual whole-domain global-FFT dependency. Another
possibility was merely that the apparent response curve became flatter because the formerly
low-error, gradient-seen lengths became worse.

### Why these candidates were considered

R3 changed `R_k -> R(xi_k)` but retained the global FFT, 32 retained bins, `[Q, s, ell]`,
and the same multi-length protocol. The diagnostic design and interpretation were
AI-assisted, then empirically checked rather than assumed.

### Discriminating check / experiment

The epoch-77 frozen R3 checkpoint was evaluated on exactly the same stable ascending
validation-Q set (`n = 300`) at strict T600/T700/T800/T900/T1000/T1100/T1200 prefixes.
T700/T900/T1100 had participated in checkpoint selection but not training gradients, so
this was development evidence rather than an untouched test. Seven lengths meant seven
frozen direct forwards, without training, adaptation, or normalization refitting.

### Result

R3 local interpolation residuals were `0.3294`, `0.1798`, and `0.1360`, with mean
absolute residual about `0.2150`, below R1's about `0.4215`. The gradient-seen versus
non-gradient gap also decreased from about `0.4225` (R1) to `0.2247` (R3). However, the
non-gradient validation mean was essentially unchanged (`0.4449 -> 0.4411`), whereas the
gradient-seen mean degraded sharply (`0.0224 -> 0.2164`).

### Evidence update

R3 retains strong positive long-domain repair evidence, but it does not solve within-range
length interpolation. The visibly smaller sawtooth is mainly caused by the low-error valleys
rising, not by material improvement at the high-error intermediate lengths. The result adds
a fidelity/generalization trade-off to the R3 assessment. Candidate 1 remains strongly
supported but bounded; candidate 3 remains structurally relevant because the global FFT was
unchanged.

### What this did NOT prove

This did not prove continuous length invariance, statistical memorization, candidate-3
causality, or complete R3 failure. It also did not negate the separate formal evidence that
R3 improves long-domain T1800/T2400 behavior.

### Next question

With physical-frequency weight remapping already repaired, does reducing whole-domain/global
FFT coupling improve fidelity and length robustness simultaneously? This is the motivation to
consider R4; R4 is a candidate next stage and has not started.
## Episode 12 — Plan B Protocol v1: fixed-domain lambda discretization-resolution generalization

### Observation

Human-context / user report: the requested next scientific question is resolution
generalization with a frozen Q-only FNO2D, not another physical-length extrapolation.
The question is whether the same finite Kerr trajectory interval can be represented on a
finer lambda grid while all other experimental controls remain fixed.

### User decision

The user froze **Plan B — Fixed-domain lambda discretization-resolution generalization**
as Protocol v1. It fixes the independent offset-grid Q400 evaluation field and changes
only `delta_lambda = 0.005 -> 0.0025`, with the corresponding endpoint-fixed grid change
`T = 1200 -> 2399`. The later evaluation uses the existing FNO2D checkpoint with frozen
weights; retraining, fine-tuning, Q resampling, replacement of failed Q values,
fine-grid normalization refitting, and architecture changes are prohibited.

### Why Plan B is distinct from prior work

Plan A T1200 -> T1800/T2400 retained `step_size=0.005` while extending the physical
lambda trajectory domain; it is length extrapolation, not discretization-resolution
generalization. Sparse stride16 -> stride32 changes observed-point density and masks
while retaining the underlying numerical trajectory grid; it is sparse
observation-density generalization, not Plan B.

### Evidence and audit boundary

The Q400 source field is the existing comparison-only offset grid: 400 uniform Q values
approximately in `[1.6007, 2.9993]`, with recorded Q400/T1200 frozen baseline global
Relative L2 `0.0071953455` and mean-per-Q Relative L2 `0.0054274904`. The Q400 field is
independent of the original `[1.6, 3.0]` training-grid construction; it was not selected
because test-300 was too small. At a fixed independent Q400 evaluation field, only the
lambda-axis discretization is changed. This must not be paraphrased as saying that only
lambda changes relative to the training field, because the training Q field and Q400
field differ.

The protocol decisions in this episode are user decisions. The comparison of existing
records and code conventions was AI-assisted. The Q400 metadata and frozen T1200 metrics
are record evidence; no new data generation, solver run, training, inference, or
numerical experiment was performed to create this documentation lock.

### T2399 versus T2400 convention analysis

Two existing, non-identical conventions matter:

1. The sampled Kerr trajectory / dataset interval uses
   `lambda_max = (N - 1) * delta_lambda`. Thus T1200 at `0.005` has last sampled
   coordinate `5.995`.
2. The DFT logical period used for physical-frequency interpretation is
   `L_DFT = N * delta_lambda`. Thus the coarse DFT logical period is `6.0`.

Protocol v1 prioritizes the sampled physical Kerr interval and chooses T2399 at `0.0025`.
It has 2398 fine intervals versus 1199 coarse intervals, exactly bisects every coarse
interval, preserves both sampled endpoints at `[0, 5.995]`, and satisfies
`fine_lambda[::2] == coarse_lambda`. No fine coordinate exceeds the coarse maximum. This
most directly tests one finite Kerr trajectory interval at higher numerical resolution.

The selection preserves a representation caveat rather than hiding it: the fine DFT
logical period is `5.9975`, versus coarse `6.0`, a relative difference of approximately
`0.0417%`. This is a DFT/discretization-representation difference, not a Plan A-style
physical-domain extension. The existing `N * delta_lambda` DFT convention remains valid
for its frequency interpretation and is not overwritten by the endpoint convention.

### Fixed controls and required qualification

Coarse and fine must retain the same Q400 identities and canonical ordering; `M`, `a`,
`E`, `Lz`, `r0`, `theta0`, `phi0`, `sign_r`, `sign_th`; solver equations and
turning-point logic; FNO architecture and checkpoint; training normalization statistics;
and target transform. The later numerical qualification must precede frozen FNO
inference: it compares `fine_xyz[:, ::2, :]` against `coarse_xyz` and reports per-Q
Relative L2 and MSE, mean, median, maximum, p95, p99, failures/anomalies, and available
turning-point diagnostics.

### What this did NOT prove

Protocol lock does not prove that the paired T2399 numerical truth already exists, that
coarse and fine trajectories are numerically consistent, that the frozen checkpoint runs
at T2399, or that Plan B will succeed. It establishes no arbitrary numerical acceptance
threshold. Structural validity, numerical-consistency metrics, and anomaly reporting are
predefined; any hard acceptance threshold awaits observed paired solver-consistency
results.

### Next question

The formal order is local implementation, unit tests, and a tiny local smoke test; then
server bundling, paired Q400/T2399 truth generation, coarse/fine ground-truth
qualification, and only then frozen T2399 inference and T1200-versus-T2399 metric
comparison. At the time of this record, only Protocol v1 lock is complete.
