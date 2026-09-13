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

## Episode 13 — Plan B local paired-replay implementation and tiny solver qualification

### Observation

Protocol v1 required a generation path that replays actual source-Q identities rather
than using the existing target-success builder, because its random completion policy could
silently replace a failed Q and break pairing.

### User decision

The user authorized local implementation, focused unit tests, and a tiny real-solver
smoke test only. Full Q400/T2399 generation, frozen FNO inference, training, server
execution, and bundling were explicitly out of scope.

### Discriminating implementation

The new narrow interface reads source `x_train/x_val/x_test` Q values without rebuilding a
range grid or resampling. It preserves split-row order, records raw and stable-sorted-Q
identity hashes, retains failed Q positions with no replacement, and marks paired
completeness false if any replay fails. Its checker first requires structural equality of
Q identities/order, fixed physics and initial conditions, solver provenance, Protocol-v1
grids, endpoints, and `fine_lambda[::2] == coarse_lambda`; only then does it compute
shared-node Relative L2 and MSE summaries.

### Result

Seven synthetic unit tests passed: endpoint-fixed grid refinement, Q identity/hash replay,
Q-mismatch rejection, nonfinite-trajectory rejection before metrics, T2400 rejection,
common-node metric extraction, and failed-Q non-replacement. A two-Q CPU smoke test with `Q = 1.8, 2.4` generated real second-order
solver coarse T1200/h=0.005 and replay fine T2399/h=0.0025 truth in a local test-artifact
directory. All structural checks passed, no Q failed, and mean per-Q Relative L2 at shared
nodes was `4.7593570648201685e-09`.

### Evidence update

This is local implementation and smoke-test evidence, not formal Q400 solver
qualification or Plan B model-performance evidence. It supports that the replay and
structural-first qualification contracts are executable on a small real-solver case.

### What this did NOT prove

The smoke result does not establish Q400-wide numerical consistency, an acceptance
threshold, absence of sensitivity elsewhere in the Q400 field, or frozen FNO
cross-resolution performance. The planned Q400/T2399 server asset remains `NOT GENERATED`.

### Next question

Bundle the reviewed code to the server, generate the full paired Q400/T2399 truth there,
and run the prescribed ground-truth qualification before any frozen FNO inference.

## Episode 14 — Plan B completed frozen coarse-to-fine resolution generalization

### Observation

User-provided formal server-result evidence reports completion of the endpoint-fixed
independent-Q400 Plan B arm. The local checkout used for this documentation update does
not contain the referenced server JSON files, so their paths and exact values are recorded
as user-reported formal result provenance rather than re-derived locally:
`outputs/plan_b_q400_t1200_to_t2399/ground_truth_consistency.json` and
`outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/metrics.json`.

The completed paired fine asset is
`data/tasks/q_1p6007-2p9993_n400_t2399_plan_b_v1`, paired to
`data/tasks/q_1p6007-2p9993_n400_t1200`. It retains the Q400 identities/order, sampled
interval `[0, 5.995]`, and endpoint relation `fine_lambda[::2] == coarse_lambda` while
changing only `T=1200, h=0.005` to `T=2399, h=0.0025`.

### User decision

The user reported that paired truth generation completed with 400/400 successes, zero
failures, paired completeness `True`, and `structural_valid=True`. The user further froze
the interpretation boundary: record the completed T1200-model -> T2399-evaluation arm,
but do not start a broader resolution sweep, multi-parameter QA, T2399 training, or the
reverse T2399-model -> T1200 experiment in this stage.

The user selected the next discriminating design: construct a matched T2399 training
dataset aligned to the original n2000 training Q candidates and split semantics, never use
Q400 for training, retain the baseline FNO2D architecture (`modes1=16`, `modes2=32`,
`width=64`, `depth=4`) and protocol wherever code facts permit, then evaluate native
T2399 and frozen reverse T2399->T1200 on the same independent Q400 field.

### Measured evidence

Paired ground truth compared `fine_xyz[:, ::2, :]` against `coarse_xyz`. Relative L2
mean/median/max/p95/p99 were `5.218413681635546e-09`,
`5.095788550601654e-09`, `6.894048665372391e-09`,
`6.692889855076337e-09`, and `6.8532082659278876e-09`; MSE
mean/median/max/p95/p99 were `6.951265864772155e-16`,
`6.458674596505792e-16`, `1.2733836202732828e-15`,
`1.1889406467703567e-15`, and `1.255802623132453e-15`. The distribution is concentrated,
with no hidden large-Q anomaly reported.

The frozen checkpoint was the original baseline only:
`outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/best_model.pt`.
No Plan A R1/R2/R3 repaired checkpoint, retraining, fine-tuning, normalization refit, or
adaptation was used. Native coarse raw-xyz global MSE / global Relative L2 /
mean-per-Q Relative L2 were `0.0012590598691126999` /
`0.007195345500573474` / `0.005427490395002388`. Fine full-grid primary values were
`0.0014630064962514512` / `0.007756728427546919` / `0.006257048792878679`; fine
common-node diagnostic values were `0.0016669064017779927` /
`0.008279117934318694` / `0.006793341748433332`.

For per-Q Relative L2, coarse median/p95/p99/max were
`0.00423478303876504`, `0.01093169026389112`, `0.022334344948613975`, and
`0.06534649935265367`; fine full-grid values were `0.005189180096032416`,
`0.011404593362521268`, `0.022006157951293282`, and `0.0662060075938115`; fine
common-node values were `0.0055552606977529355`, `0.012537290722076497`,
`0.023640856005393975`, and `0.0670052711640239`. Worst Q for all three views was
`1.6007`.

The saved-prediction discretization diagnostic used
`D_i = ||pred_fine_common - pred_coarse||_2 / ||truth_coarse||_2`. Its global/mean/
median/p95/p99/max were `3.008818547630e-03`, `3.003436179404e-03`,
`2.909109598286e-03`, `3.702832683988e-03`, `3.961805595476e-03`, and
`4.044745297005e-03`; worst Q was `1.62173157895`.

### AI-assisted interpretation

The truth mismatch is about the `1e-9` Relative L2 scale, many orders below the FNO
prediction errors in this Q400/T1200<->T2399 comparison. It is therefore a negligible
numerical-truth confounder for this experiment, while not establishing a general theorem
about the solver at arbitrary Kerr parameters or resolutions.

The full-grid global Relative L2 changes from `0.0071953455` to `0.0077567284` (about
+7.8%), and mean-per-Q Relative L2 from `0.0054274904` to `0.0062570488` (about +15.3%).
P95 rises only slightly; P99, maximum, and the worst-Q location remain broadly stable.
Thus the appropriate conclusion is: **prediction accuracy remains in the same error scale
under 2x endpoint-preserving grid refinement, with moderate degradation rather than
catastrophic failure.** This is strong coarse-to-fine discretization-resolution
generalization for this historical Q-only FNO2D checkpoint, fixed Q400 field, and fixed
sampled lambda interval.

The approximately `3e-3` prediction shift proves that the finite-grid output is not
exactly resolution invariant. It does not, however, justify treating `3e-3 / 7.2e-3` as a
literal fraction of model error caused by resolution: norms alone do not provide an
additive decomposition of error vectors.

### Why this is interesting

The result sharply contrasts with Plan A direct physical-domain length extrapolation:
Plan A changed the sampled lambda domain and showed large length sensitivity, whereas
Plan B holds `[0, 5.995]` fixed and has only modest frozen coarse-to-fine degradation.
This contrast supports a bounded robustness statement about discretization change, not a
claim that the architecture is universally invariant or that the Plan A mechanisms have
been disproved.

### What this did NOT prove

The completed arm does not demonstrate exact finite-grid invariance, universal
resolution invariance, all-Kerr-task robustness, bidirectional resolution generalization,
or a causal percentage attribution of resolution to model error. It also does not validate
the reverse T2399-trained model, its native T2399 performance, or any wider resolution
sweep.

### Next discriminating experiment

The required next scientific comparison is the matched bidirectional matrix:

| Training resolution | Test T1200 | Test T2399 |
| --- | --- | --- |
| T1200 model | Native coarse baseline completed | Coarse-to-fine completed |
| T2399 model | Frozen reverse fine-to-coarse pending | Native fine pending |

Only after the T2399 model has been trained on an original-n2000-aligned fine training
field and frozen evaluation has completed on both independent Q400 resolutions can the
project claim evidence about both directions. Broader resolution sweeps and multi-
parameter QA remain deferred pending advisor discussion.

## Episode 15 — Plan B bidirectional fixed-domain resolution generalization

### Observation

User-provided formal server-result evidence completes the second arm of Protocol v1. A
matched theta2399 model, trained on the original-n2000 T2399 replay and frozen thereafter,
was evaluated on the same independent paired Q400/T2399 and Q400/T1200 fields. The confirmed
server matched dataset is `data/tasks/q_1p6-3_n2000_t2399_plan_b_matched_v1`, and the best
checkpoint is `outputs/plan_b_bidirectional_t1200_t2399/training/checkpoints/best_model.pt`.

### User decision and matched-training controls

The user required exact replay of source `q_1p6-3_n2000_t1200`: 1400/300/300 train/val/test
Q rows, source seed 10, Q range `[1.6, 3.0]`, 2000/2000 successful uniform grid points, and
no completion sampling. The original float64 split SHA256 values are train
`19568b3ebb25494faa0f769874d3aa02c0e32bae325e7c968308ffe297e964ea`, validation
`065795c3ac45c8fbf52e8e3c115c0daeb57b476609f73499dc9f9cd420e54275`, and test
`c67d057aba98eb080c374280524751deb44d752a58425f3dfff5b4dbcc377e80`.

The matched T2399 task preserves those exact Q values, split membership, row ordering,
physics, initial conditions, solver provenance, and sampled interval `[0, 5.995]`. Only
T1200/h=0.005 changes to T2399/h=0.0025. Q400 is evaluation-only, never a substitute
training field. The user also fixed the current boundary: no additional resolution sweep or
QA work starts before advisor feedback.

### Measured evidence

The historical theta1200 row is native T1200 global/mean-per-Q Relative L2
`0.007195345500573474` / `0.005427490395002388` and frozen T2399
`0.007756728427546919` / `0.006257048792878679`. Its coarse-to-fine global and mean-per-Q
degradation are about +7.8% and +15.3% respectively.

Theta2399 used the same `fno2d_m16x32_w64_d4_e500` FNO2D/training protocol as the historical
baseline and selected best epoch 500, but fitted new standard normalization statistics from
its T2399 training split. Its native T2399 global MSE / global RelL2 / mean-per-Q RelL2 are
`0.0012802295993175293` / `0.007256035373512494` / `0.005506914674547638`; frozen reverse
T1200 values are `0.00141194010010074` / `0.007619677632647374` /
`0.006087018417501273`. Its fine-to-coarse global and mean-per-Q degradation are about
+5.0% and +10.5%. The worst Q remains `1.6007` in both theta2399 views.

The full global Relative L2 matrix is:

| Training resolution | Test T1200 | Test T2399 |
| --- | ---: | ---: |
| theta1200 | `0.007195345500573474` | `0.007756728427546919` |
| theta2399 | `0.007619677632647374` | `0.007256035373512494` |

The full mean-per-Q Relative L2 matrix is:

| Training resolution | Test T1200 | Test T2399 |
| --- | ---: | ---: |
| theta1200 | `0.005427490395002388` | `0.006257048792878679` |
| theta2399 | `0.006087018417501273` | `0.005506914674547638` |

The two native global RelL2 values differ by only about 0.84%, providing an important
native-quality control for the directional transfer comparison. The pre-existing Q400
paired-truth mismatch is about `1e-9` Relative L2, whereas the prediction-only common-node
discretization shift is about `3e-3` Relative L2. The latter is measurable but does not
permit a causal decomposition of model-error norms.

### AI-assisted interpretation

Both frozen directions remain in the same raw-xyz error scale with moderate rather than
catastrophic degradation. The evidence therefore supports strong practical bidirectional
discretization-resolution generalization for this Q-only Kerr task, fixed independent Q400
field, and endpoint-fixed `[0, 5.995]` interval. Fine-to-coarse is slightly more stable than
coarse-to-fine for this measured pair, but that is descriptive rather than a general
directional law. The prediction shift shows that the finite-grid model output is not exactly
resolution invariant.

### Why this is important

Plan A kept h=0.005 while increasing T and thereby extending the physical lambda domain; its
frozen predictions failed severely. Plan B instead keeps the sampled interval fixed and
changes only discretization. The sharp contrast demonstrates that changing T alone is not
the scientific cause of the Plan A failure: physical-domain extension and fixed-domain
resolution change have materially different behavior in the current experiment.

### What this did NOT prove

The result does not establish exact or universal resolution invariance, arbitrary-resolution
generalization, identical behavior for all Kerr parameter families, a complete resolution
range, QA/multi-parameter generalization, or a universal theory contrasting all FNO domain
extensions with all discretization changes.

### Next question

Plan B bidirectional core is complete at T1200/T2399. The remaining scientific decision is
whether advisor feedback warrants an additional resolution-range sweep or a later
multi-parameter extension; neither starts automatically.

## Episode 16 — Cross-model resolution Benchmark Protocol v1

Historical terminology note (2026-09-12): this episode preserves the original design
reasoning. Its FNO2D-large `16.8M` and approximately `15.6x` parameter-gap statements use
ordinary PyTorch `tensor_numel`, not the primary matching unit now approved in Episode 18.
The small-model band is now interpreted in `real_scalar_parameter_count` units.

### Observation

The completed FNO2D Plan B evidence establishes useful bidirectional fixed-domain
resolution behavior for one architecture, but cannot by itself answer whether FNO is more
resolution-robust than other model families or whether the Kerr task is simply easy for all
models under endpoint-preserving refinement. The preceding architecture audit found that
existing Dilated ResNet and TimesNet checkpoints are sparse-trajectory reconstruction
models with `[sparse_xyz, observed_mask, lambda]` inputs and excluded Q, not Q-only
`[Q, lambda] -> xyz` models. Their historical results therefore cannot be reused as native
Plan B baselines.

### User decisions

The user froze Benchmark Protocol v1 as a new post-Plan-B comparison stage. Track A uses
trajectory-wise direct regression with exactly `[Q_broadcast, lambda] -> xyz`, T1200/T2399
matched n2000 training tasks, and the independent canonical Q400 field for all evaluation.
The initial evaluation levels are endpoint-fixed T1200/T2399/T3598/T4797; no automatic 6x,
8x, QA, data scaling, training, or server run is authorized.

The Phase-I model set is BiLSTM, Dilated ResNet, canonical TimesNet, encoder-only
Transformer, FNO1D, and DeepONet. Phase I is a controlled approximately-1.1M-parameter
comparison with an accepted `0.9M--1.3M` band, not an exact-parameter contest. Each model
will have exactly two formal training runs, at T1200 and T2399, followed by frozen 2x4
Q400 evaluation. The user further fixed a separate Track B: FNO1D-small versus
FNO2D-small for formulation, and FNO2D-small versus existing 16.8M FNO2D-large for
capacity. Lambda-isolated TimesNet remains a diagnostic ablation, not the main TimesNet
baseline.

### AI-assisted design rationale

Same task and same information are the primary fairness controls. This is why Track A
uses Q plus the actual lambda coordinate for every trajectory model and restricts DeepONet
branch/trunk inputs to the equivalent Q/lambda information. Comparable small-model capacity
is the next control; it reduces the approximately 15.6x parameter gap between historical
1.08M sparse ResNet/TimesNet variants and 16.8M FNO2D-large without forcing unnatural exact
architectures.

FNO1D bridges Track A and Track B because it is trajectory-wise like the direct baselines
but operator-like in formulation. FNO2D-small is needed because comparison with
FNO2D-large alone would confound joint Q-field learning with capacity. Selective Phase-III
scaling is preferred to immediately enlarging every model: first establish controlled
small-model evidence, then scale only the strongest two or three non-FNO candidates if the
scientific question remains open.

The existing Dilated ResNet receptive-field calculation motivates a controlled envelope.
With kernel 7, 11 residual blocks, and dilations through 1024, the current block formula
produces `RF=12349`, greater than the predeclared 8x `T_max=9593`. Thus Phase-I's 1x--4x
comparison will not be confounded by an intentionally too-short ResNet view, while finite
receptive field remains a future explanatory hypothesis. Canonical TimesNet is retained
because runtime FFT selection, period folding, and period-grid convolutions may have a
resolution-sensitive inductive bias; the lambda-isolated model answers a different
mechanism question. Transformer must report quadratic attention memory/time scaling rather
than treating feasibility as an unmeasured implementation detail.

### Pre-registered hypotheses

The following are hypotheses, not results:

- operator/spectral models may show smaller frozen resolution degradation than conventional sequence models;
- the controlled Dilated ResNet may still reveal finite-receptive-field effects at wider ranges;
- TimesNet runtime FFT bins and period folding may be more discretization-sensitive than FNO;
- Transformer may reach `O(T^2)` compute limits before accuracy is limiting;
- FNO1D and DeepONet may be more robust than non-operator trajectory models;
- FNO2D may gain from joint Q-axis field learning relative to FNO1D; and
- some FNO2D-large performance may be capacity-driven rather than formulation-driven.

### What this does NOT prove

This protocol lock does not prove FNO superiority, TimesNet or Transformer weakness,
ResNet receptive-field causality, an operator-learning advantage, any cross-model ranking,
or a valid resolution range. It creates no task-aligned baseline implementation, parameter
count, checkpoint, data asset, training result, inference result, or compute measurement.
Historical sparse reconstruction metrics remain valid for their original task but are not
cross-model Plan B evidence.

### Next question

The next exact work item is task-aligned model implementation plus capacity matching, then
local unit and smoke tests. Only after those contracts are verified may a unified server
workflow execute the Phase-I two-training-run-per-model matrix. Phase II, selective scaling,
QA, and wider resolution levels remain later decisions rather than automatic work.

## Episode 17 — Plan B completed endpoint-fixed 1x--4x resolution range

### Observation

**Measured server evidence supplied by the user:** the unified FNO-only range workflow
completed on the independent canonical Q400 field at endpoint-fixed T1200/T2399/T3598/T4797.
It replayed the paired Q400 field through `400/400`, completed generation, structural and
numerical qualification, frozen evaluation, and matrix assembly without a hard-stop. The
new truth assets are `data/tasks/q_1p6007-2p9993_n400_t3598_plan_b_range_v1` and
`data/tasks/q_1p6007-2p9993_n400_t4797_plan_b_range_v1`; the output root is
`outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797` and the matrix is
`resolution_range_matrix.json` beneath that root. No additional numerical-truth summary is
inferred beyond the confirmed workflow completion.

The measured global Relative L2 matrix is:

| Training model | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| theta1200 | `0.007195345500573474` | `0.007756728427546919` | `0.008098130243693246` | `0.008292316031400246` |
| theta2399 | `0.007619677632647374` | `0.007256035373512494` | `0.0073153850467772156` | `0.007378147022109385` |

The corresponding mean-per-Q Relative L2 matrix is:

| Training model | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| theta1200 | `0.005427490395002388` | `0.006257048792878679` | `0.006736322137146906` | `0.006999674680921897` |
| theta2399 | `0.006087018417501273` | `0.005506914674547638` | `0.005601255905072378` | `0.005701483116388768` |

### User decisions and controls

The user retained the fixed independent Q400 identities/order, Kerr physics, initial
conditions, and sampled interval `[0, 5.995]`. R3 uses `T=3598`, `h=0.005/3`; R4 uses
`T=4797`, `h=0.00125`. The only intended variable is lambda-axis discretization. The user
also fixed that 6x/8x are conditional future experiments rather than an automatic extension,
and that the next active stage is the already locked Cross-model Benchmark Protocol v1.

### AI-assisted interpretation

Relative to native T1200, theta1200 global / mean-per-Q Relative L2 changes by approximately
`+7.8% / +15.3%` at T2399, `+12.5% / +24.1%` at T3598, and `+15.2% / +29.0%` at T4797.
The observed curve is smooth through the tested 4x refinement and has no threshold-like
collapse. Relative to native T2399, theta2399 changes only approximately `+0.82% / +1.71%`
at T3598 and `+1.68% / +3.53%` at T4797. Theta2399 has lower raw global Relative L2 than
theta1200 at both T3598 and T4797. These measurements support the bounded interpretation
that finer-resolution training improves robustness to the still-finer tested grids in this
Q-only Kerr setting.

This extends the completed bidirectional pair into a tested range. It also sharpens the
Plan A contrast: Plan A kept step size fixed while extending the physical lambda domain and
showed severe frozen length-extrapolation failure; Plan B holds the sampled interval fixed
and remains stable through the tested discretization refinements. Thus, in this experiment,
FNO2D is substantially more robust to fixed-domain discretization change than to
physical-domain-length extension.

### What this did NOT prove

The evidence does not establish exact finite-grid invariance, arbitrary-resolution
generalization, T4797 as a final boundary, universal Kerr behavior, QA/multi-parameter
robustness, or causal proof that training resolution alone explains every observed
improvement. The earlier common-node prediction discretization shift of
`3.008818547630e-03` remains relevant: practical robustness is not exact output invariance.

### Next question

The FNO-only Plan B core is complete through the tested R1--R4 range. The next discriminating
question is cross-model: under locked Benchmark Protocol v1, do task-aligned trajectory
models and operator models exhibit comparable native accuracy, frozen resolution robustness,
and computational cost? The next work item remains task-aligned model implementation plus
capacity matching; no benchmark model training or wider sweep is implied by this record.

## Episode 18 — Approved cross-model capacity-counting clarification (2026-09-12)

### Observation and measured audit facts

The accepted read-only CPU audit found that the same ordinary PyTorch `numel()` reporting
convention counts different storage representations differently: FNO2D uses native complex
Parameters, whereas FNO1D stores real/imaginary components as separate real Parameters.
The audit used existing model classes with only the specified configurations, filtered
`requires_grad=True`, and verified shared-Parameter deduplication and frozen exclusion.
It performed no model forward/backward, training, checkpoint read/write, or capacity search.

With input2/output3, FNO2D-large (modes1=16, modes2=32, width=64, depth=4, hidden_dim=128)
has `16,777,216` native complex elements plus `25,539` real elements: `tensor_numel` is
`16,802,755`, while counting complex elements twice gives `33,579,971` real scalar
parameter coordinates. The audited FNO1D modes32/width64/depth4, ResNet width84/11-block,
and TimesNet dm80/df96/2-block candidates have equal counts under both measures:
`1,069,763`, `1,096,119`, and `1,077,059`, respectively. Exact supporting configurations
and counts are recorded in Current State section 15.1.

### Why the original terminology was insufficient

A native complex element contains real and imaginary coordinates; splitting it into two
real Parameters should not change a model's reported matching scale. Conversely, multiplying
the existing FNO1D `numel()` by two would count its already-separated components twice.
TimesNet's FFT creates complex intermediates, not additional trainable Parameters.

The audit also identified a semantic limitation: inverse real FFTs ignore certain imaginary
coordinates, including DC components. Thus the proposed audit name `real_scalar_dof` could
be mistaken for exact effective functional degrees of freedom, which the counting formula
does not establish. This is a distinction between nominal parameter coordinates and
functional redundancy, not a reason to alter the existing models or their checkpoints.

### User-approved scientific clarification

The user approved `real_scalar_parameter_count` as the primary cross-model matching metric:
count each registered trainable real element as 1 and each native complex element as 2,
using only `requires_grad=True` Parameters. Its meaning is the nominal number of registered
trainable real scalar parameter coordinates, not exact effective functional degrees of
freedom. Do not subtract FFT/DC-coordinate redundancies or any other architecture-specific
functional redundancies. Report `tensor_numel` secondarily; storage bytes are diagnostic only.

The Track-A and Track-B small-model target remains approximately 1.1M with the accepted
0.9M--1.3M band, now explicitly in `real_scalar_parameter_count` units. The three audited
small candidates remain valid candidates and must not be changed solely because of this
clarification; this does not formally lock previously provisional configurations.

Historical FNO2D-large retains both `16,802,755 tensor_numel` (approximately 16.80M) and
`33,579,971 real_scalar_parameter_count` (approximately 33.58M). Historical 16.8M terminology
is preserved as the old tensor-element count, not used as the primary matching quantity.
The former approximately-15.6x comparison in Episode 16 is likewise historical tensor-count
arithmetic, not a real-scalar capacity ratio.

If Phase III is later executed specifically to capacity-match existing FNO2D-large, the
selected non-FNO target is approximately `33.58M real_scalar_parameter_count`, not 16.8M.
The user explicitly approved this target clarification; it is not an autonomous AI scaling
decision. Phase III is considered only after Phase-I results, only the strongest selected
two or three non-FNO models are candidates, and no large-model training starts without
explicit execution approval.

### Interpretation and limits

The approved convention removes the native-complex versus split-real storage discrepancy
from nominal capacity matching. It does not prove equal effective function-space dimension,
expressivity, optimizer behavior, compute cost, or scientific performance across models.
Same task/information and reasonable architecture remain higher fairness priorities.
The counting-unit, DOF-terminology, and conditional Phase-III target ambiguities are resolved
by the user's decisions, not by new performance evidence. No historical asset is altered.

### Next authorized boundary

This stage updates protocol records only. Task-aligned implementation and subsequent
capacity matching remain the next execution work; no model-code change, missing-model
capacity search, training, or inference is authorized by this record update.

## Episode 19 — Track A task alignment and unified workflow implementation (2026-09-12)

### Authorized implementation and observed facts

The user subsequently authorized Track A implementation, nominal capacity matching and
unit/synthetic smoke tests, explicitly excluding formal Phase I training. All six
builders now use the locked direct Q/actual-lambda task and satisfy the real-scalar band.
Measured counts and full configurations are recorded in Experiment Plan 7.3.2 and Current
State 15.2. Existing FNO1D/ResNet/TimesNet candidate counts were reproduced exactly.
No configuration was selected using formal accuracy, training outcomes or extra data.

The implementation passed 16 focused CPU tests and 42 existing regression tests. Each
model completed one small synthetic epoch/checkpoint roundtrip; the Transformer resume
test reproduced continuous two-epoch synthetic training exactly. A separate test confirmed
that best weights survive a worse final epoch. Frozen evaluation tests restored checkpoint
statistics without refitting and retained Q identities and native-resolution routing.
Real registered asset reads confirmed matched historical training split hashes and all
four canonical Q400 grids. No new formal artifact or experiment result was produced.

### Engineering choices within the accepted protocol

- Reuse the actual ResNet class with the explicit 11-block schedule, avoiding the older
  sparse factory's nine-block limit. The core and RF formula remain unchanged.
- Reuse canonical TimesNet with input 2/output 3, including batch-shared runtime top-k.
  Fixed canonical Q order and evaluation batch 32 are explicit because batch composition
  affects its frequency discovery; no lambda-isolated substitution or period redesign.
- Reuse existing field-normalization routines through a singleton dimension, preserving
  channel statistics, epsilon and float32 model input behavior. Keep float64 source Q,
  lambda and raw truth separate from normalized training tensors.
- Use a conventional two-layer bidirectional LSTM, standard encoder-only Transformer,
  and Q-branch/lambda-trunk DeepONet; dimensions instantiate nominal capacity constraints.
  No masks, trajectory observations, autoregression or extra features are introduced.
- Existing generic checkpoint helpers overwrite fixed paths. A dedicated benchmark
  schema uses exclusive writes, immutable recovery checkpoints, source/data hashes and
  stored normalization/RNG/optimizer/scheduler state. Recovery cadence 25 does not change
  every-epoch validation selection; the best model is exported only after completion.
- Reuse the Plan B raw-xyz metric implementation, with generic ratios/differences measured
  against the frozen checkpoint's own native resolution. CUDA timing synchronization and
  peak allocated-memory reporting are prepared; no formal cost numbers are measured.

### Interpretation and remaining gate

CPU tests establish implementation contracts and small synthetic workflow consistency,
not model quality, cross-resolution generalization or full-batch GPU feasibility.
Transformer retains standard quadratic attention, so resource feasibility at formal T
remains an empirical preflight question. A failure requires reporting, not an unapproved
attention, batch-size or budget change. No Protocol-impacting issue was discovered in the
completed checks. The next step is review of this implementation for explicit Phase I
execution approval, with required GPU/resource checks before any authorized launch.

## Episode 20 — Benchmark normalization reduction precision (2026-09-12)

### Observed preflight facts

The first authorized FNO1D wave stopped before formal training. Both authoritative
training assets retained identical Q train identities/order, but the initial shared
float32 reduction gave Q means 2.3031082153320312 (T1200) and 2.2984538078308105 (T2399).
Because Q is repeated equally along lambda, changing only T cannot change its population
mean or std mathematically. On the same float32 samples, float64 accumulation gave
Q mean 2.2996668343884603 at both resolutions. This isolated accumulation precision
from dataset/split identity; no model-performance evidence was involved.

### User-approved correction and implementation reasoning

The user approved float64 statistics for the new Benchmark Protocol v1 normalization,
without changing the definition of standard normalization. The original Episode 19
reuse decision is superseded only for benchmark statistics fitting: a benchmark-local
helper handles all channels with explicit float64 mean/std reductions and preserves
binary64 statistics in Python float lists and serialized checkpoint/JSON state. No
Q-specific formula or special case is introduced. The old float32 fitting samples,
reduction axes, train-only boundary, population std and epsilon semantics stay fixed.
Only application casts statistics to the existing float32 model/training path.

Changing the shared historical fitter would alter old workflows, so that utility and
all historical assets/results remain untouched. Frozen evaluation restores statistics;
it never estimates new statistics from evaluation data. This is a numerical-accuracy
correction before any formal benchmark checkpoint, not a new normalization method or
an adjustment selected in response to model performance.

### Validation evidence and limits

Both real train splits passed independent float64 reference checks and exact JSON/
checkpoint statistics roundtrips. Corrected Q std values are 0.4036297106003588 and
0.4036297106003346 (difference about 2.42e-14); all corrected channel values are recorded
in Current State 15.4. Different lambda std values are legitimate for distinct discrete
grids with shared endpoints. The xyz statistics also need not be equal across grids.

All 35 focused/benchmark regression tests passed on single-thread CPU in `fno_srv`,
including float32 forward/backward/loss application, train-only fitting, epsilon and
population std, historical fitting isolation, checkpoint/resume and frozen no-refit
evaluation. Temporary smoke artifacts were automatically cleaned. These establish
numerical/workflow correctness, not accuracy or a formal benchmark result. No formal
run, dataset regeneration or registry update occurred. The correction resolves the
identified blocker; the next formal task remains only FNO1D Wave 1 after required
execution preflight, with no architecture, optimizer, budget or evaluation change.

## Episode 21 — First formal Track A wave: FNO1D (2026-09-12)

### Authorization and measured facts

After accepting the benchmark precision correction, the user explicitly resumed only
FNO1D T1200/T2399 and their four-resolution frozen evaluations. Both runs completed all
500 epochs at source HEAD `3335b3b197cf8b7d7a302d900a51af37a25d2144` with a clean
preflight, float64 train-only statistics and the unchanged float32 training path. They
ran concurrently on independent host GPU1/GPU2, without DDP or parameter/budget changes.
No interruption, resume or compute conflict occurred. Best epochs were 499 and 459;
validation normalized MSE was 1.0671552748438747e-7 and 9.817391306417752e-8, respectively.

| Train / evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 0.000304676590517 | 0.00260773956864 | 0.00346591077718 | 0.00389546647459 |
| 2399 | 0.00262195772641 | 0.000295562198528 | 0.000891773304135 | 0.00130651011794 |

The table is measured global raw-xyz Relative L2 on the same canonical Q400. Native
normalization was restored, never refitted on evaluation data. Each training process
was followed immediately by its own frozen best-checkpoint evaluation. All eight
prediction arrays reproduced the stored raw metrics exactly in a read-only CPU check.
Artifact paths, hashes and full distributions are in Registry 9.1 and Current State 15.5.

### Interpretation, separated from measurements

Native errors are similar in magnitude (about 3e-4). Cross-resolution transfer increases
error relative to each model's small native baseline: T1200-to-T2399 is about 8.56x,
while T2399-to-T1200 is about 8.87x. T1200-to-T4797 is about 12.79x, and
T2399-to-T4797 about 4.42x. Relative degradation and absolute error remain distinct:
these ratios alone do not establish poor absolute accuracy or cross-model superiority.
At higher evaluation resolutions the T2399-trained model has lower absolute error in
this seed27 pair. No causal mechanism or seed robustness is established by this wave.

Worst-Q locations were Q=2.9993 except the T1200-trained model at T3598/T4797, where
Q=2.9256894736842103 was worst. These distribution facts do not authorize extra data,
retuning, longer training or a different model. No six-model conclusion is available:
only FNO1D has formal results. Historical Plan B results were not altered.

### Next recommendation and execution boundary

The next suggested wave is the already locked Dilated ResNet at T1200/T2399, each
followed by all four frozen Q400 evaluations, to add a conventional convolutional
comparison under the same task and budget. This is a recommendation requiring explicit
user authorization, not a ranking, new architecture choice or permission to launch.
No other model or Phase II/III experiment was executed in this task.

## Episode 22 — Completed ResNet Wave 2: native fit and resolution transfer (2026-09-12)

### Measured facts and completed-experiment audit

The user confirmed background completion and authorized analysis only. Both ResNet
trainings reached 500 epochs; both frozen BEST evaluations produced all four Q400 cells.
Start/exit records show successful training and evaluation with no recorded failure or
training resume. The earlier Codex conversation interruption did not stop either job.
Analysis used source HEAD `29170c9c20813306abb465d10b86031fa5853d23` and a clean
working tree. No training, prediction regeneration or protocol change was performed.

Best epochs were 490 (T1200) and 487 (T2399); validation normalized MSE was
8.08959028593866e-7 and 3.8574821966600816e-7. Final-epoch validation MSE was
9.281237915577852e-7 and 4.3822223725934844e-7, respectively. Full best/final selection,
checkpoint histories, locked RF12349/capacity/configuration, exact float64 train statistics,
float32 checkpoint/model path, data hashes and Q/lambda ordering passed audit. All eight
saved prediction metrics passed independent sum-of-squares CPU recomputation with
`rtol=1e-12, atol=1e-14`; max absolute discrepancy was 2.842170943040401e-14.
Wave-1's 70 file hashes remained unchanged. No numerical or provenance mismatch was found.

Measured global raw-xyz Relative L2:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 0.000875793115919 | 1.23641878184 | 1.15185604926 | 1.10103131103 |
| 2399 | 1.08073931383 | 0.000624316666496 | 1.03557683005 | 1.03118762782 |

Ratios to each checkpoint's own native resolution:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | ---: | ---: | ---: | ---: |
| 1200 | 1.00000000000 | 1411.77038203 | 1315.21477884 | 1257.18196572 |
| 2399 | 1731.07554520 | 1.00000000000 | 1658.73648041 | 1651.70607026 |

Native errors are 8.757931159186983e-4 and 6.243166664963908e-4. All six off-native errors
are between approximately 1.03 and 1.24, with an abrupt native/off-native gap rather than
a monotonic rise across the finer grids. T2399 training lowers absolute T3598/T4797 errors
relative to T1200 training, but leaves them near 1; its smaller native baseline produces
larger relative degradation ratios. Thus improved absolute error and improved
native-relative robustness must not be treated as the same claim.

Worst Q is 1.6007 for both native cells. T1200 training shifts to Q=1.6392578947368421 at
T2399, back to 1.6007 at T3598, and to 2.9256894736842103 at T4797. T2399 training shifts
to 2.9993 at T1200, stays at 1.6007 at T3598, and shifts to 1.6778157894736843 at T4797.
Training took 8828.0870415112 / 10006.35461697448 seconds with peak allocated memory
1216906240 / 2259363328 bytes on independent host GPU1/GPU2, each mapped to local cuda:0.

### Provisional interpretation, not a final model ranking

Compared with the same Wave-1 cells, ResNet native errors are approximately 2.87x / 2.11x
larger. Its non-native ratios are approximately 1257--1731x, versus approximately
3.02--12.79x for FNO1D. ResNet measured training time is about 17.04x / 16.84x and peak
allocated memory about 6.27x / 6.33x FNO1D. Both waves used the locked seed27/protocol
and concurrent independent single-GPU scheduling. These are descriptive observations;
they do not isolate architecture as the sole cause, establish seed robustness, or
rank all six models. Full matrices/distributions and asset hashes are in Current State
15.6 and Registry 9.2.

A provisional reading is that accurate native function regression can coexist with
strong discretization sensitivity in this ResNet configuration. RF12349 is a nominal
architecture property, not a guarantee of resolution invariance; these results do not
identify effective-RF, padding, learned filters or another mechanism as the cause.
No failed-transfer result is discarded or used to justify architecture, budget,
normalization or data changes. Poor accuracy is a measured result, not by itself a
Protocol-impacting issue.

### Hypothesis boundary and next step

The preregistered six-model hypotheses remain unchanged. The hypothesis that operator/
spectral models may transfer resolution more reliably remains for the complete benchmark
to assess; this two-model comparison does not establish a universal explanation.
Phase I is incomplete: 4/12 trainings and 16/48 evaluation cells are complete.
The next recommended wave is locked canonical TimesNet at T1200/T2399, each followed
by four frozen Q400 evaluations, only after explicit user authorization. No next wave
or other model was launched by this analysis.

## Episode 23 — Completed canonical TimesNet Wave 3: accuracy versus relative degradation (2026-09-13)

### User authorization and measured audit facts

The user authorized completed-experiment analysis only after both independent TimesNet
workflows finished. Training and automatic frozen evaluation exited successfully, with
500 epochs per training and eight saved evaluation cells. No interruption/resume is
recorded. Execution and clean analysis HEAD were
`d8b190ed982e09b42784ccd7178e1c8b001d3a39`.

Verified configuration: input2/output3, d_model80, d_ff96, two TimesNet blocks, top-k2,
kernels(1,3,5), dropout0, and real_scalar_parameter_count == tensor_numel == 1,077,059.
The saved-source implementation uses canonical runtime FFT discovery, batch-shared top-k,
1D-to-2D period folding and Inception Conv2d processing, not the lambda-isolated variant.
Locked AdamW/ExponentialLR defaults, batch32, seed27 and 500 epochs were unchanged.
Train-only float64 statistics match fresh fitting and an independent float64 reference;
checkpoint restoration and float32 model/sample application were verified. Frozen evaluation
restored the same statistics without refitting and retained canonical Q400 order with
batch32 (final batch16).

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

| Train T | Final epoch | Best epoch | Best validation normalized MSE | Final validation normalized MSE | Training seconds | Peak allocated bytes | Host GPU / CUDA_VISIBLE_DEVICES / local CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1200 | 500 | 492 | 0.00000267102036256 | 0.00000311175996406 | 710.514718014 | 484014592 | 1 / 1 / cuda:0 |
| 2399 | 500 | 486 | 0.00000290461068592 | 0.0000207157540475 | 1404.66065574 | 1064739328 | 2 / 2 / cuda:0 |

Measured global raw-xyz Relative L2:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 0.00157254951507 | 0.00383845963700 | 0.00491844356093 | 0.00546934598239 |
| 2399 | 0.00326309006262 | 0.00165634091327 | 0.00194255174312 | 0.00222717317039 |

Ratios to each checkpoint's own native resolution:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 1 | 2.44091496020 | 3.12768756328 | 3.47801193537 |
| 2399 | 1.97005944638 | 1 | 1.17279705377 | 1.34463452091 |

### Provisional comparison and interpretation

| Model / Train T | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| FNO1D / 1200 | 0.000304676590517 | 0.00260773956864 | 0.00346591077718 | 0.00389546647459 |
| ResNet / 1200 | 0.000875793115919 | 1.23641878184 | 1.15185604926 | 1.10103131103 |
| TimesNet / 1200 | 0.00157254951507 | 0.00383845963700 | 0.00491844356093 | 0.00546934598239 |
| FNO1D / 2399 | 0.00262195772641 | 0.000295562198528 | 0.000891773304135 | 0.00130651011794 |
| ResNet / 2399 | 1.08073931383 | 0.000624316666496 | 1.03557683005 | 1.03118762782 |
| TimesNet / 2399 | 0.00326309006262 | 0.00165634091327 | 0.00194255174312 | 0.00222717317039 |

**Measured facts:** TimesNet T1200-to-finer errors increase monotonically over the three
sampled finer grids, from 0.00383846 to 0.00546935 (2.44--3.48x native), without the
order-one off-native errors seen in ResNet. T2399-to-T1200 gives 0.00326309 (1.97x native);
T2399-to-T3598/T4797 gives 0.00194255/0.00222717 (1.17x/1.34x native).
T2399 training lowers finer-grid absolute errors and native-relative ratios compared with
T1200 training, although its native error is slightly higher. Worst Q for T1200 training
moves from 1.6007 at native T to 2.9993 at all finer grids; T2399 training has worst Q
1.6007 at all four grids. Final T2399 validation MSE exceeds its best validation MSE;
frozen results correctly use epoch486, not epoch500.

**Provisional three-model interpretation:** FNO1D has lower absolute global errors than
TimesNet in all eight corresponding cells. TimesNet has lower native-relative ratios
than FNO1D at all six off-native cells, partly reflecting its larger native baseline;
this must not be reported as better absolute accuracy. ResNet has lower native error
than TimesNet but substantially higher off-native error and degradation ratios.
TimesNet measured training time and peak allocated memory lie above FNO1D and below
ResNet at each training resolution. These are descriptive single-seed27/protocol facts,
not a final six-model ranking or an architecture-only causal explanation.

**TimesNet mechanism/hypothesis boundary:** the sampled curves show gradual, directionally
asymmetric degradation rather than the abrupt large native/off-native gap seen in ResNet.
This does not establish stability at unmeasured resolutions. Selected-period/frequency
diagnostics were not recorded in the formal workflow. The source exposes a diagnostic
method, but it was not invoked here; no causal claim about FFT period selection is
supported by these saved outputs. Preregistered hypotheses remain unchanged and await
the complete six-model comparison.

Measured training resources (same locked single-GPU workflow; not a controlled
architecture-only timing attribution):

| Model | Train T | Training seconds | Peak allocated bytes |
| --- | --- | --- | --- |
| FNO1D | 1200 | 518.085670201 | 194066944 |
| FNO1D | 2399 | 594.374222904 | 357028352 |
| ResNet | 1200 | 8828.08704151 | 1216906240 |
| ResNet | 2399 | 10006.3546170 | 2259363328 |
| TimesNet | 1200 | 710.514718014 | 484014592 |
| TimesNet | 2399 | 1404.66065574 | 1064739328 |

The completed formal root is `outputs/benchmark_track_a_v1/phase_i_wave3_timesnet_20260913`.
Current State 15.7 contains full secondary metrics/deltas and Registry 9.3 indexes all
asset classes and selected checkpoint hashes. No historical result or preregistered
hypothesis was rewritten. No model, data, normalization or budget adaptation was made.

### Next execution gate

Phase I is not complete: three of six models, 6/12 trainings and 24/48 evaluation cells
are complete. No Protocol-impacting issue or ordinary execution error was found.
The exact next recommended action is Phase I Wave 4: the locked two-layer BiLSTM
(hidden size 184, dropout 0, 1,093,331 real scalar parameters), trained separately at
T1200/T2399 with the unchanged 500-epoch protocol, each followed by the same four frozen
Q400 evaluations. This requires fresh explicit user execution approval and GPU/data/Git
preflight; it is not authorized or launched by this analysis.

## Episode 24 — Completed BiLSTM Wave 4: native fit versus resolution sensitivity (2026-09-13)

### User authorization and measured facts

Execution and clean analysis HEAD: `ff7ea434c718f3545c134b831462acd2574b5010`;
branch `codex/clean-research-history-20260905`. The user authorized completed-experiment
analysis after both independent background workflows reported COMPLETE. This analysis
launched no training or inference and regenerated no predictions.

Verified BiLSTM: input2/output3, two bidirectional LSTM layers, hidden size184, dropout0,
linear xyz output head, no autoregressive decoding or teacher forcing; both
real_scalar_parameter_count and tensor_numel are 1,093,331. Locked batch32/500 epochs,
AdamW lr1e-3/weight_decay1e-4, ExponentialLR gamma0.995 and seed27 were unchanged.
Normalization was fitted only to the corresponding training-resolution train split.
Stored float64 statistics match fresh fitting and an independent float64 reference;
restoration and float32 sample/model application were verified. Frozen evaluation restores
these statistics without refitting, preserves canonical Q400 order and uses batch32
(final batch16). BEST is selected by minimum validation normalized MSE over 500 epochs.

Both histories span epochs 1--500; both train exits and both frozen-evaluation exits
have returncode 0, and both workflow completion markers exist. All eight frozen cells
and twenty recovery checkpoints per training are present. No failure or resume event is
recorded; successful train/evaluation logs contain result JSON without extra error text.
The 70/77/86 files of Waves 1/2/3 match the Wave-4 preflight hashes. All 86 Wave-4 files
were retained unchanged during this audit.

Single-thread fno_srv CPU recomputation used saved predictions and authoritative float64
truth with independent sum-of-squares formulas, unchanged epsilon 1e-12 and per-Q
summaries. All 64 scalar comparisons across eight cells passed rtol=1e-12, atol=1e-14;
maximum absolute discrepancy was 2.6645352591003757e-15. Q/lambda arrays, dataset/split
provenance, source hashes, per-cell versus matrix metrics, native ratios/deltas, checkpoint
history prefixes, minimum-validation selection and selected best weights were verified.

| Train T | Final epoch | Best epoch | Best validation normalized MSE | Final validation normalized MSE | Training seconds | Peak allocated bytes | Host GPU / CUDA_VISIBLE_DEVICES / local CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1200 | 500 | 485 | 7.13191132794e-7 | 0.00000143270845153 | 3967.26292164 | 1598493696 | 1 / 1 / cuda:0 |
| 2399 | 500 | 477 | 0.00000106106260167 | 0.00000125360207676 | 6333.22000560 | 3126256128 | 2 / 2 / cuda:0 |

Measured global raw-xyz Relative L2:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 0.000816636194451 | 0.301107365739 | 0.407211841622 | 0.461684118895 |
| 2399 | 0.202826791462 | 0.000990420666357 | 0.0840668719669 | 0.125443720656 |

Ratios to each checkpoint's own native resolution:

| Train / Evaluate | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| 1200 | 1 | 368.716654717 | 498.645350756 | 565.348587329 |
| 2399 | 204.788529108 | 1 | 84.8799654758 | 126.657010417 |

**Measured BiLSTM behavior:** T1200-to-T2399/T3598/T4797 errors are
0.301107/0.407212/0.461684, versus native 0.000816636: a large native/off-native gap,
followed by monotonic growth over the three sampled finer grids. T2399-to-T1200 gives
0.202827 (204.79x native), while T2399-to-T3598/T4797 gives 0.0840669/0.125444
(84.88x/126.66x native). Coarse/fine transfer is asymmetric. T2399 training improves
finer-grid absolute errors and native-relative ratios relative to T1200 training,
although its native error is higher (0.000990421). Thus native accuracy and resolution
robustness move separately in this pair; better transfer does not imply better native fit.

Worst Q for T1200 training is 1.6007/2.9993/2.971257894736842/2.985278947368421 across
T1200/T2399/T3598/T4797. For T2399 training it is 1.6007/2.9993/1.6007/1.6007.
These are saved distribution facts, not authorization to change data or retune.

### Provisional four-model comparison and hypothesis boundary

| Model / Train T | T1200 | T2399 | T3598 | T4797 |
| --- | --- | --- | --- | --- |
| FNO1D / 1200 | 0.000304676590517 | 0.00260773956864 | 0.00346591077718 | 0.00389546647459 |
| ResNet / 1200 | 0.000875793115919 | 1.23641878184 | 1.15185604926 | 1.10103131103 |
| TimesNet / 1200 | 0.00157254951507 | 0.00383845963700 | 0.00491844356093 | 0.00546934598239 |
| BiLSTM / 1200 | 0.000816636194451 | 0.301107365739 | 0.407211841622 | 0.461684118895 |
| FNO1D / 2399 | 0.00262195772641 | 0.000295562198528 | 0.000891773304135 | 0.00130651011794 |
| ResNet / 2399 | 1.08073931383 | 0.000624316666496 | 1.03557683005 | 1.03118762782 |
| TimesNet / 2399 | 0.00326309006262 | 0.00165634091327 | 0.00194255174312 | 0.00222717317039 |
| BiLSTM / 2399 | 0.202826791462 | 0.000990420666357 | 0.0840668719669 | 0.125443720656 |

**Provisional four-model comparison:** FNO1D has lower absolute errors than BiLSTM in
all eight corresponding cells. BiLSTM has lower native errors than TimesNet at both
training resolutions but much higher off-native errors and ratios. Relative to ResNet,
BiLSTM has slightly lower native error at T1200 and higher native error at T2399;
all six BiLSTM off-native errors and ratios are lower than ResNet's. BiLSTM's off-native
ratios exceed those of both FNO1D and TimesNet. TimesNet's smaller ratios than FNO1D
continue to require its larger native baseline to be reported alongside absolute errors.
BiLSTM training time is above FNO1D/TimesNet and below ResNet at each T, while its peak
allocated training memory exceeds all three previously measured models at each T.

**Interpretation and hypothesis boundary:** accurate native regression coexists with
substantial resolution sensitivity in this BiLSTM pair. The sampled errors show a large
native/off-native gap, not uniformly smooth low-error transfer. This does not establish
continuity between grids or a causal role for recurrence or bidirectionality; no such
mechanism is isolated by these saved outputs. All comparisons are descriptive under one
locked seed27/protocol, not a final model ranking, an architecture-only causal claim or
evidence of seed robustness. Preregistered hypotheses remain unchanged and will be
revisited after all six models complete.

Measured training resources:

| Model | Train T | Training seconds | Peak allocated bytes |
| --- | --- | --- | --- |
| FNO1D | 1200 | 518.085670201 | 194066944 |
| FNO1D | 2399 | 594.374222904 | 357028352 |
| ResNet | 1200 | 8828.08704151 | 1216906240 |
| ResNet | 2399 | 10006.3546170 | 2259363328 |
| TimesNet | 1200 | 710.514718014 | 484014592 |
| TimesNet | 2399 | 1404.66065574 | 1064739328 |
| BiLSTM | 1200 | 3967.26292164 | 1598493696 |
| BiLSTM | 2399 | 6333.22000560 | 3126256128 |

Formal root: `outputs/benchmark_track_a_v1/phase_i_wave4_bilstm_20260913`.
Current State 15.8 contains complete secondary metrics/deltas and Registry 9.4 records
asset classes and best checkpoint hashes. No Protocol-impacting issue or ordinary
execution error was found; poor transfer is retained as a result, not used to change
architecture, data, normalization, budget or the preregistered hypotheses.

### Next execution gate

Phase I is incomplete: four of six models, 8/12 formal trainings and 32/48 frozen
evaluation cells are complete. Transformer and DeepONet remain untrained in Phase I.
The exact next recommendation is Phase I Wave 5: the locked encoder-only full-attention
Transformer (d_model192, 6 heads, 2 layers, feedforward1024, dropout0.1;
1,088,003 real scalar parameters), trained separately at T1200/T2399 for 500 epochs,
each followed by four frozen Q400 evaluations. Fresh explicit execution approval and
Git/data/GPU preflight are required. This analysis does not authorize or launch Wave 5.
