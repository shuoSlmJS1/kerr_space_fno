# Plan A: Length-extrapolation data definition

This document describes the reviewed training and evaluation data used for Plan A. Dataset names below identify directories under `data/tasks/`, each containing `dataset.npz` and `meta.json`.

## 1. Task definition

- **Input physical parameter:** Q.
- **Input coordinate:** lambda, $\lambda$.
- **Target:** the trajectory $xyz(\lambda)$.

The raw dataset stores each split as:

```text
x_split:     (N, 1)       Q values
lambda_grid: (T,)         shared lambda coordinates
y_split:     (N, T, 3)    raw xyz ground truth
```

For the baseline Q-only FNO2D, these arrays are organized into a Q–lambda field:

```text
input:  (1, N, T, 2) = [Q, lambda]
target: (1, N, T, 3) = xyz
```

N is the number of Q trajectories in the selected split or evaluation field. The leading singleton dimension denotes one field, not one trajectory. Q and the corresponding xyz rows are sorted together when constructing the canonical model-input field.

## 2. Training dataset

The formal training source is **`q_1p6-3_n2000_t1200`**, at `data/tasks/q_1p6-3_n2000_t1200/dataset.npz`.

| Property | Definition |
| --- | --- |
| Q range | $[1.6,3.0]$ |
| Total trajectories | 2000, on a uniform Q grid |
| Train / validation / internal test | 1400 / 300 / 300 |
| Split seed | 10 |
| Samples per trajectory, T | 1200 |
| $\Delta\lambda$ | 0.005 |
| Sampled lambda interval | $[0,5.995]$ |
| Single-trajectory xyz shape | `(1200,3)` |
| Ground-truth storage dtype | `float64` |
| Model input and target dtype | `float32` |

The split is a seeded shuffle of whole Q trajectories followed by a 70%/15%/15% partition; it is not a partition along lambda within each trajectory.

**Only the 1400 train trajectories are used for training.** The 300 validation trajectories support model selection. The 300 internal test trajectories are separate from the later canonical Q400 length-extrapolation evaluation field.

## 3. Canonical length-extrapolation evaluation datasets

| Dataset | Q count | T | $\Delta\lambda$ | Lambda range | Role |
| --- | ---: | ---: | ---: | --- | --- |
| `q_1p6007-2p9993_n400_t1200` | 400 | 1200 | 0.005 | [0,5.995] | Same-domain evaluation |
| `q_1p6007-2p9993_n400_t1800` | 400 | 1800 | 0.005 | [0,8.995] | Length extrapolation |
| `q_1p6007-2p9993_n400_t2400` | 400 | 2400 | 0.005 | [0,11.995] | Length extrapolation |

All three datasets share:

- Q range $[1.6007,2.9993]$ and the same uniform 400-Q offset grid.
- Stored train/validation/test sizes of **280/60/60**, with split seed **20260728**.
- Identical Q identities, split membership, and original row ordering across lengths.
- Raw ground truth stored as `float64`; model input and target tensors constructed as `float32`.

Formal evaluation merges **all 400 trajectories** and applies the same canonical ascending-Q order to Q and truth. The stored split names are storage partitions here: the 280 rows named `train` do not train the model, and evaluation is not restricted to the 60 rows named `test`.

For each dataset, a single xyz trajectory has shape `(T,3)`, each stored xyz split has shape `(N,T,3)` with N=280/60/60, and the merged evaluation truth has shape `(400,T,3)`. Float32 model tensors do not replace the authoritative float64 ground truth.

## 4. Relationship between T1200, T1800, and T2400

The coordinate convention is

$$
\lambda_j=j\Delta\lambda,\qquad j=0,\ldots,T-1,
\qquad \lambda_{\mathrm{endpoint}}=(T-1)\Delta\lambda.
$$

With $\Delta\lambda=0.005$ fixed:

```text
T1200 -> endpoint 5.995
T1800 -> endpoint 8.995
T2400 -> endpoint 11.995
```

T1800 and T2400 extend the physical lambda domain for exactly the same Q400 identities. They do not refine the sampling density within a fixed domain.

Formal prefix validation classified all three comparisons as **EXACT_PREFIX**: T1200→T1800, T1200→T2400, and T1800→T2400. For the same Q, the first 1200 lambda coordinates and xyz truth samples match exactly across all three lengths. T1800 is also an exact truth prefix of T2400. This is a measured property of the stored arrays, not an inference from dataset names or equal step sizes.

## 5. Training grid versus evaluation grid

The training-source Q2000 grid and canonical evaluation Q400 grid are **different grids**. Q400 lies inside the training Q range, and direct comparison of the stored Q arrays found no exactly equal Q values between the two grids.

Consequently, Plan A tests **lambda-domain extrapolation**, not Q-range extrapolation. Exact-prefix pairing applies among the three Q400 evaluation datasets; it does not mean that Q400 trajectories are row-matched to the Q2000 training source.

## 6. Fixed physical parameters

Q is the only varying physical parameter. The training and evaluation datasets share:

```ini
M = 1.0
a = 0.5
E = 0.95
Lz = 3.0
r0 = 10.0
theta0 = 1.2
phi0 = 0.0
sign_r = -1
sign_th = 1
```

## 7. Data generation

The unified generation entry point is `scripts/generate_dataset.py`. The dataset metadata records solver **`second_order_rk4`**, version **`v1`**.

The T1800/T2400 queue is defined in `scripts/generate_length_extrapolation_datasets.sh`. The four datasets record completed generation, zero failures, and no completion sampling.

The full historical launch commands and exact generation commits have not been verified and are not reconstructed here from the current scripts.

## 8. Important distinction

**Plan A = fixed $\Delta\lambda$ + increased T + longer physical domain.**

Later cross-resolution experiments instead keep the physical domain fixed and use smaller $\Delta\lambda$ to increase sampling density. The larger T values in Plan A must not be described as fixed-domain resolution refinement.

## 9. Sources

All paths are relative to the project root.

- `data/tasks/q_1p6-3_n2000_t1200/meta.json`
- `data/tasks/q_1p6007-2p9993_n400_t1200/meta.json`
- `data/tasks/q_1p6007-2p9993_n400_t1800/meta.json`
- `data/tasks/q_1p6007-2p9993_n400_t2400/meta.json`
- The corresponding `dataset.npz` files in those four directories: array headers establish shapes/dtypes, and stored Q arrays establish grid identities.
- `outputs/length_dataset_identity_validation/q400_t1200_t1800_t2400_prefix_identity.json`
- `scripts/generate_dataset.py`
- `scripts/generate_length_extrapolation_datasets.sh`
- `src/common/io_utils.py` — seeded trajectory split implementation.
- `src/data_generation/dataset_saver.py` — dataset storage and split construction.
- `src/training/fno2d/input_builder_2d.py` — Q/lambda field construction and model-data dtype.
- `FNO_KERR_CURRENT_STATE.md` — dataset lineage and verified Plan A protocol.
- `FNO_KERR_EXPERIMENT_PLAN.md` — distinction between physical-domain extension and fixed-domain resolution changes.
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` — training and comparison-only asset roles.
