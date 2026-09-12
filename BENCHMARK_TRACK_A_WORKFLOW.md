# Track A Benchmark Workflow

Implementation of locked Cross-model Resolution Benchmark Protocol v1. This document
does not authorize Phase I training, formal frozen inference, or Phase II/III work.

## Entry point and environment

Use `/home/shanjinshuo/miniconda3/envs/fno_srv/bin/python -B` explicitly, from the
repository root. No package or environment changes are required.

```bash
/home/shanjinshuo/miniconda3/envs/fno_srv/bin/python -B scripts/run_benchmark_track_a.py inspect
```

`inspect` only builds CPU models for counting and prints the planned 12-run / 48-cell
matrix. It writes no artifacts and launches no training or inference. The other two
subcommands are `train` and `evaluate`; each requires a specific execution approval.
`--execution-approved` acknowledges that approval and does not confer it.

## Implemented configurations

Every model accepts `[B,T,2]` Q-broadcast / actual physical lambda inputs and predicts
`[B,T,3]` xyz. Standard normalization is the same train-fitted affine coordinate
transformation for every model, not replacement of physical lambda with an index grid.

| Model key | Configuration | real_scalar_parameter_count | tensor_numel |
| --- | --- | ---: | ---: |
| bilstm | 2 bidirectional layers, hidden184, dropout 0, linear xyz head | 1,093,331 | 1,093,331 |
| resnet | width84, kernel7, 11 blocks, dilations powers of two through1024, RF12349 | 1,096,119 | 1,096,119 |
| timesnet | dm80, df96, 2 blocks, top-k2, kernels1/3/5, dropout 0 | 1,077,059 | 1,077,059 |
| transformer | dm192, 6 heads, 2 encoder layers, feedforward 1024, ReLU, post-norm, dropout 0.1 | 1,088,003 | 1,088,003 |
| fno1d | modes32, width64, depth4, existing GELU core | 1,069,763 | 1,069,763 |
| deeponet | branch/trunk each four hidden layers of width384 with tanh; latent128 | 1,085,699 | 1,085,699 |

DeepONet has a Q-only branch producing three sets of 128 coefficients and a lambda-only
trunk producing 128 basis values, with xyz component biases. It receives no observations,
masks, or extra physics. Transformer uses independently initialized conventional
`TransformerEncoderLayer` layers with standard full attention, without fixed-length
positional embeddings, causal masks, or sparse/linear attention. Quadratic attention
scaling remains part of the benchmark; GPU feasibility/cost is not established by CPU tests.

The three existing ResNet, canonical TimesNet, and FNO1D cores are reused unchanged.
TimesNet retains runtime FFT, batch-shared top-k, period folding and Conv2d processing.
Its output can depend on batch composition: validation preserves source split order;
Q400 evaluation uses canonical ascending Q order, batch 32, and an unchanged final batch 16.
Do not change evaluation batching in response to memory pressure or observed results.

## Data and normalization

The loader uses only registered n2000/T1200, matched n2000/T2399, and Q400
T1200/T2399/T3598/T4797 assets. Training retains original 1400/300/300 split membership
and row order, with historical split hashes checked before execution. Replay data must
match source Q identities, fixed physics, solver provenance, replacement policy and
physical lambda grid. No dataset is generated or interpolated.

Q400 uses the existing Plan B canonical mapping, also recorded in output provenance.
Original float64 Q, lambda, and xyz are retained; float32 arrays feed the models.
Normalization reuses `compute_field_normalization_stats` through a singleton field
dimension. Per-channel population mean/std is fitted on the current-resolution train
split only, with the existing epsilon 1e-8 standard-deviation floor. Validation and
evaluation never fit statistics. Raw target transformation is identity; loss is
normalized-space MSE and frozen accuracy is measured against raw float64 xyz truth.

## Individually authorized execution

These are command templates for later approved work, not instructions to launch now:

```text
python -B scripts/run_benchmark_track_a.py train
  --model MODEL --train-resolution 1200|2399 --output NEW_OUTPUT_DIRECTORY
  --device cpu|cuda:0 [--host-gpu HOST_INDEX] --execution-approved

python -B scripts/run_benchmark_track_a.py evaluate
  --checkpoint RUN_DIRECTORY/best_model.pt --output NEW_EVALUATION_DIRECTORY
  --device cpu|cuda:0 [--host-gpu HOST_INDEX] --execution-approved
```

Use the explicit `fno_srv` Python above. On GPU, the environment must contain exactly
`CUDA_VISIBLE_DEVICES=HOST_INDEX`; the process-local device is `cuda:0`. Follow AGENTS.md
before execution. The entry point additionally checks current visibility, occupancy
and minimally necessary ownership metadata. Conflicts or ambiguity cause a stop;
there is no GPU eviction, fallback batch size, or multi-GPU conversion.

Training has no CLI optimizer, epoch, seed, batch-size or architecture tuning switches.
It uses 500 epochs, batch 32, AdamW(lr 1e-3, weight_decay 1e-4), ExponentialLR(gamma 0.995),
seed 27, normalized MSE, and the lowest validation normalized MSE over all epochs.
Training batches shuffle; validation batches do not; neither drops the final batch.
Failure stops the run and records the failure without adapting the protocol.

## Checkpoints, provenance, and recovery

All formal artifacts are under a fresh directory within `outputs/`. Existing run and
evaluation directories are refused. Dataset/meta SHA256, exact Q hashes, implementation
file hashes, Git HEAD, configuration, capacity, environment and GPU mapping are recorded.
Legacy sparse-reconstruction checkpoints are rejected by the benchmark schema.

Every 25 epochs and at the final epoch, an immutable `epoch_NNNN.pt` contains current
model/optimizer/scheduler, RNG state, history, fitted statistics, and best-so-far weights.
Best selection still occurs every epoch. At completion, `best_model.pt` and `summary.json`
are written once. The summary records the selected checkpoint hash. No historical
checkpoint is overwritten or deleted. Checkpoint cadence affects recovery granularity,
not optimizer updates or selection. Expect roughly 20 recovery checkpoints per formal run.

After a separately approved continuation, `train --resume RUN_DIRECTORY/epoch_NNNN.pt`
uses the same run directory. Dataset, code, environment, normalization and configuration
must match. RNG restoration preserves the next shuffle/dropout sequence. Existing later
epochs or a completed summary cause a stop; partial/corrupt existing files are not silently
removed. Resume replays work after the last complete recovery point; it adds no epochs
beyond the original 500-epoch budget. A source-code/Git provenance change requires review
rather than an automatic resume exception.

## Frozen evaluation and compute

One evaluator verifies a selected checkpoint from a completed 500-epoch run, restores its
normalization, freezes its weights, and evaluates the four registered Q400 grids.
It saves per-resolution physical xyz predictions with Q/lambda, metrics and provenance,
plus `matrix.json`. Each frozen evaluation uses a new output directory; interrupted
evaluations retain completed outputs and require review before repeating work.
The evaluator verifies both the selected checkpoint hash and the implementation hashes
recorded during training; a different model/evaluation implementation requires review.

The unchanged Plan B raw metrics provide global Relative L2 (primary), global raw MSE,
mean per-Q Relative L2, median/p95/p99/max and worst Q. Ratios and differences use the
checkpoint's own native resolution, including T2399 for T2399-trained models. A zero
native error yields a null ratio, not an invented finite value.

Capacity reports both required counts and optional storage bytes. CUDA timing uses
explicit synchronization. Training wall time covers the training/validation loop and
intermediate checkpoint I/O, excluding data loading/model setup and final artifact
publication. For resumed runs, cumulative time starts from the saved elapsed value;
it excludes downtime and the recovery checkpoint's own final write time. GPU peak
memory is PyTorch peak allocated bytes, not total driver-reserved memory.

Inference timing has no warmup and covers loader access, host-to-device transfer and
forward execution; it excludes prediction transfer back to CPU, inverse normalization,
metric computation and file writes. Record this scope when comparing compute results.
No formal timing result exists from this implementation task.

## Validation boundary

The focused tests use single-thread CPU synthetic data and automatically cleaned
temporary directories. They cover six-model forward/backward and a one-epoch synthetic
training/checkpoint roundtrip for each model, exact counting, dynamic T, information
contracts, train-only normalization, deterministic resume, best-weight retention,
frozen evaluator routing, Q-order preservation and refusal gates. FNO1D additionally
has single-trajectory forwards at all four formal T values; these are shape tests,
not Q400 accuracy or formal compute measurements.

Formal GPU memory/time, Transformer feasibility at batch 32 and full T, and all Phase I
accuracy results remain unmeasured. No smoke artifact belongs in the asset registry.
