# Kerr FNO 当前研究状态

## 1. 文档目的与证据政策

本文件是 Kerr FNO 项目的可持续研究记忆：说明项目如何到达当前状态、哪些
结论由何种证据支持、哪些结论仍不成立，以及后续工作应从哪里恢复。它不以
目录名或会话记忆替代结果证据。

重要证据标签：

| 标签 | 含义 |
|---|---|
| `snapshot-verified` | Stage-1 snapshot 嵌入了对应元数据或结果内容，可直接核对。 |
| `registry-only` | 数值或结论记录在 `SERVER_DATA_EXPERIMENT_REGISTRY.md`，但快照未嵌入原始结果内容。 |
| `asset-only` | 快照只确认文件或目录存在，未保存其内容。 |
| `Git/code-derived` | 可由提交历史或当前代码接口确定，不能替代服务器结果。 |
| `server-result-verified` | 用户提供的正式服务器结果 artifact 可直接核对其记录的验证结论。 |
| `human-context` | 来自研究协作历史，不能由服务器资产机械恢复。 |
| `unknown` | 当前证据不足，不能可靠断言。 |

负结果不是自动无效或过时。`superseded` 只表示正式新工作应优先采用另一个
已记录资产或协议；它不自动否定保留的历史证据。

## 2. 当前 repository / server 状态

| 项目 | 记录 |
|---|---|
| Snapshot | `server_research_snapshot_20260828_021340.json` |
| `schema_version` | `1.0` |
| Snapshot mode | `stage1_research_snapshot` |
| Server / local HEAD | `a725f688ca88c1a5a5fd2d92a63fd7e6fcf402fe` |
| Branch | `fix/turning-point-solver` |
| HEAD subject | `add read-only server research snapshot collector` |
| Server / local worktree | clean (`status_short` empty) |
| Scan roots | `data/tasks`、`outputs` |
| Inventory | 10 datasets、50 runs、64 checkpoints、264 evidence files、866 legacy assets、0 errors |

上述 HEAD、分支和工作树状态在快照与本地检查中一致，属于
`snapshot-verified`。Stage-1 仅做资产与小型证据清单，不证明大型数组之间的
轨迹身份或科学有效性。

## 3. Dataset lineage

所有目前盘点的数据集只变化 `Q`。共享固定参数为：

```text
M=1.0, a=0.5, E=0.95, Lz=3.0, r0=10.0,
theta0=1.2, phi0=0.0, sign_r=-1, sign_th=1
```

Stage-1 的 NPZ ZIP/NPY-header 检查均为 `ok`，每份 `failed_samples.json`
记录的失败数为 0。以下状态来自 registry 与嵌入 `meta.json`，属于
`snapshot-verified`。

### 3.1 Current / validated

| 数据集 | 样本 / split | `T` | `delta_lambda` | 求解器与用途 |
|---|---:|---:|---:|---|
| `q_1p6-3_n500_t1200` | 500；350/75/75 | 1200 | 0.005 | `second_order_rk4 v1`；小型真实数据验证与 sparse 基线 |
| `q_1p6-3_n1000_t1200` | 1000；700/150/150 | 1200 | 0.005 | `second_order_rk4 v1`；小规模实验 |
| `q_1p6-3_n2000_t1200` | 2000；1400/300/300 | 1200 | 0.005 | `second_order_rk4 v1`；默认开发规模 |
| `q_1p6-3_n5000_t1200` | 5000；3500/750/750 | 1200 | 0.005 | `second_order_rk4 v1`；标度与大型正式实验 |
| `q_1p6-3_n8000_t1200` | 8000；5600/1200/1200 | 1200 | 0.005 | `second_order_rk4 v1`；标度与大型正式实验 |

这些数据集的 `Q` 范围均为 `[1.6, 3.0]`，`lambda_max=5.995`，并使用
`sampling_mode=grid`、`completion_policy=target_success`。

### 3.2 Comparison-only

| 数据集 | 样本 / split | `T` | `delta_lambda` | 物理 λ 域 | 用途 |
|---|---:|---:|---:|---|---|
| `q_1p6007-2p9993_n400_t1200` | 400；280/60/60 | 1200 | 0.005 | [0, 5.995] | 独立 offset-grid common test |
| `q_1p6007-2p9993_n400_t1800` | 400；280/60/60 | 1800 | 0.005 | [0, 8.995] | 历史长度外推评估 |
| `q_1p6007-2p9993_n400_t2400` | 400；280/60/60 | 2400 | 0.005 | [0, 11.995] | 更长长度外推候选 |

它们都使用 `second_order_rk4 v1`，`Q` 范围为 `[1.6007, 2.9993]`。三个
数据集的 train/val/test `x` split 身份哈希相同，支持 Q 样本配对设计。
然而，Stage-1 未对 `y` 做全量比较，因此 **不能** 声称长数据集的轨迹是真值
严格前缀；该结论需要 Stage-2 验证。

### 3.3 Legacy / experimental

| 数据集 | 样本 / split | `T` | 求解器版本 | 当前角色 |
|---|---:|---:|---|---|
| `vary_Q__Q1.6_3__n500__T1200__cfg1_secondorder_pilot` | 500；350/75/75 | 1200 | `experimental_v1` | 保留的 pilot/provenance 资产 |
| `vary_Q__Q1.6_3__n2000__T1200__cfg1_secondorder` | 2000；1400/300/300 | 1200 | `experimental_v1` | 保留的历史比较资产 |

这两份数据集也记录为二阶 RK4，`delta_lambda=0.005`。它们已经被 current
`v1 + target_success` 数据集取代用于新的正式工作，但 registry 并未称其为
损坏或科学无效。

## 4. Solver evolution

```text
first-order sqrt/sign-flip solver
    ↓
turning-point / convergence investigation
    ↓
second-order RK4
    ↓
validated v1 datasets
```

旧求解器以平方根一阶方程推进，并使用 `sign_r` / `sign_th` 人工翻转、
turning-point 阈值检测与 coordinate nudge。该机制将转向点处理与离散阈值、
符号状态耦合。

提交 `d1e7c97` 加入 RK4 收敛与 turning-point 诊断；`5c2e550` 采用二阶
Kerr 求解器。新系统以速度穿过零自然表示转向点，避免人工符号翻转、事件
检测与 nudge，并显式监控：

```text
|v_r^2 - R(r)|
|v_theta^2 - Theta(theta)|
```

这部分动因和实现属于 `Git/code-derived`。

直接验证证据为 `snapshot-verified`：

- n5000 批量验证 20/20 通过；最细两层相对误差中位数
  `1.9133e-11`，每轨最低观测阶中位数 `4.0175`；
- legacy n500 与 n2000 数据集各抽查 20 条，以步长 `0.000625` 作为参考；
  存储轨迹相对参考解 `Relative L2` 中位数分别为 `5.6133e-09` 与
  `5.5979e-09`；
- current v1 数据集在元数据中记录 target-success、严格均匀的成功点和
  零失败样本。

因此旧数据应保留为历史/探索证据，current v1 数据应作为未来正式 A/B 的
优先基础。

## 5. Early trajectory-modelling lineage

- `9227631`（2026-05-01）建立 Q-only FNO1D 基础管线；当前快照没有该早期
  管线的服务器数值结果，故其历史结果为 `unknown`。
- `dc7964c`（2026-05-02）加入 FNO2D 训练/分析；`a272ece` 和 `505a64c`
  分别加入归一化、target transform 支持；`bb6f3a5` 增加 multi-config
  能力。这些是 `Git/code-derived` 的技术演进。
- 快照中保留的 current FNO2D 运行均使用 `normalization=standard` 与
  `target_transform=raw`。没有足够成对服务器结果可重建全部 normalization/
  target-transform 数值比较，故不能由代码存在推导其优劣。

## 6. FNO2D scaling and common-test experiments

独立 common test 的价值在于：每个模型保持自己的训练归一化统计量，但在同一
`q_1p6007-2p9993_n400_t1200` offset-grid 上以 raw physical-space `xyz`
统一评估。它避免将不同训练任务各自 test split 的误差直接混为可比较结论。

### 6.1 数据量与宽度

| 比较 | 模型 / 训练数据 | Common-test MSE | Common-test Relative L2 | 证据 |
|---|---|---:|---:|---|
| 数据量 | n500, w48,d4,e500 | 3.834e-03 | 1.2556e-02 | `snapshot-verified` |
| 数据量 | n1000, w48,d4,e500 | 2.034e-03 | 9.1445e-03 | `snapshot-verified` |
| 数据量 | n2000, w48,d4,e500 | 1.641e-03 | 8.2146e-03 | `snapshot-verified` |
| 数据量 | n5000, w48,d4,e500 | 1.752e-03 | 8.4889e-03 | `snapshot-verified` |
| 宽度 | n2000, w16,d4,e500 | 1.917e-02 | 2.8075e-02 | `snapshot-verified` |
| 宽度 | n2000, w32,d4,e500 | 4.247e-03 | 1.3214e-02 | `snapshot-verified` |
| 宽度 | n2000, w48,d4,e500 | 1.641e-03 | 8.2146e-03 | `snapshot-verified` |
| 宽度 | n2000, w80,d4,e500 | 9.517e-04 | 6.2558e-03 | `snapshot-verified` |

在这个选择的 w48/d4/e500 比较中，n500 到 n2000 改善；但 n5000 并未严格
优于 n2000。宽度也在所测试的 n2000 设置中显著影响结果。这些是受限于
数据、模型、seed 和 common-test 协议的结论，而非一般单调缩放定律。

### 6.2 深度

| 深度配置（n2000, w64,e500） | Common-test MSE | Common-test Relative L2 | 证据 |
|---|---:|---:|---|
| d2 | 1.813e-03 | 8.6353e-03 | `snapshot-verified` |
| d3 | 1.314e-03 | 7.3514e-03 | `snapshot-verified` |
| d5 | 9.655e-04 | 6.3008e-03 | `snapshot-verified` |
| d6 | 1.085e-03 | 6.6790e-03 | `snapshot-verified` |

深度在此受控比较中有影响，d5 是这四项中的最低 common-test `Relative L2`；
这不证明任意更深模型都会继续改善。

### 6.3 遗留 FNO2D

保留的 legacy `experimental_v1` FNO2D 直接测试结果为：n500 pilot
`Relative L2=6.0293e-02`，n2000 `Relative L2=1.6755e-02`
（`snapshot-verified`）。它们是有价值的历史基线，但不应代替 current v1
资产上的正式比较。

## 7. Historical direct length extrapolation

### 7.1 Motivation

`human-context`：导师曾鼓励研究外推能力；项目因此探索直接长度外推。历史
长域结果看起来不理想后，用户向导师报告该方向似乎不具前景；导师随后建议
考察 FNO 的 cross-resolution/generalization 价值。这不是可由资产证明的
唯一因果链，只是已明确提供的研究协作背景。

### 7.2 Historical protocol

已完成的历史运行使用：

```text
Q-only FNO2D
training task: q_1p6-3_n2000_t1200
model: fno2d_m16x32_w64_d4_e500
checkpoint: best_model.pt
T_train = 1200
T_eval = 1800
delta_lambda = 0.005
Q samples = 400, full_dataset
input = whole long (Q, lambda) field
```

评估脚本在完整长输入上执行一次 FNO2D 前向；没有 autoregressive rollout、
teacher forcing、预测 `xyz` 回馈、未来真值 `xyz` 输入、fine-tuning 或
test-time adaptation。这一协议属于 `snapshot-verified` 加
`Git/code-derived`。

### 7.3 Verified result

| 区域 | MSE | Relative L2 | 证据 |
|---|---:|---:|---|
| 原训练域前缀 `0:1200` | 70.2125 | 1.70164 | `snapshot-verified` |
| 延长区间 `1200:1800` | 114.0648 | 2.17517 | `snapshot-verified` |
| 全长 | 84.8299 | 1.87208 | `snapshot-verified` |

正确表述是：**该特定冻结 FNO2D/domain-extension 协议表现很差，且在长输入
协议下，原见域前缀相对真值的误差已达到 `RelL2 ≈ 1.70`。**

表中 `1.70164` 是 long-input prediction 的共享 T1200 前缀相对真值的
mean-per-Q Relative L2；它不是 short-input prediction 与 long-input-prefix
prediction 的数值差异。该预测差异现已由下述正式诊断独立给出。

Stage-2 strict dataset identity validation 已在服务器完成，结果 artifact 为
`outputs/length_dataset_identity_validation/`
`q400_t1200_t1800_t2400_prefix_identity.json`，验证器提交为 `8a6ee0b`，证据为
`server-result-verified`。三个 pair 均为 `EXACT_PREFIX`：

```text
short_to_medium = EXACT_PREFIX
short_to_long   = EXACT_PREFIX
medium_to_long  = EXACT_PREFIX
```

对每一个 `train`、`val`、`test` split，Q 身份和排序均一致；三个 lambda-prefix
检查的 `exact_equal=true`，`max_abs_difference=0.0`，
`mean_abs_difference=0.0`。全部 trajectory-prefix 比较也严格相等，且
`tolerance_pass=true`、max/mean absolute difference、RMSE、overall `Relative L2`
与各轨迹 `Relative L2` 统计均为零。全零并列时的
`worst_trajectory_index` 只是首个被选中的索引，没有额外科学意义。

因此 T1200、T1800、T2400 是经直接验证的 exact-prefix companion datasets；
`historical_t1800_reusable=true`，`t2400_ready_for_future_a1=true`。历史 T1800
冻结单次前向协议在原 seen domain 上的差表现不能由 short/long ground-truth
prefix mismatch 解释。

#### Historical float32 representation-roundtrip clarification

An early temporary comparison reported:

```text
Truth-prefix maximum absolute difference: 9.536743164062e-07
```

This was a representation-roundtrip discrepancy, not a raw T1200/T1800 dataset
mismatch. The short-side truth came from the raw dataset truth path, whereas the
historical long saved-target representation had passed through float32/model-space
normalization and/or target recovery before being saved or compared. A float32
conversion followed by recovery to raw space can produce a difference at approximately
one float32 ULP scale; the observed value is consistent with that numerical roundtrip,
not with instability in the underlying trajectory solver.

Subsequent Stage-2 validation compared the underlying raw datasets directly and found
`short_to_medium = EXACT_PREFIX`, `short_to_long = EXACT_PREFIX`, and
`medium_to_long = EXACT_PREFIX`, with raw trajectory-prefix differences exactly zero.
Raw T1200/T1800/T2400 datasets therefore remain exact-prefix companion datasets. This
representation-roundtrip discrepancy neither weakens `EXACT_PREFIX`, explains the
historical length-extrapolation failure, nor constitutes trajectory-generation error.
Current formal diagnostics avoid it by using shared short-dataset raw float64 truth as
the primary scientific reference and promoting predictions to float64 for metric
computation.

### 7.4 Corrected formal frozen length-change prediction consistency diagnostic

用户提供的正式服务器结果 artifact 为
`outputs/length_change_prediction_consistency/`
`q400_t1200_t1800_all_canonical_q.json`。其角色为 `formal diagnostic`，结果
证据为 `server-result-verified`：在 Stage-2 `EXACT_PREFIX` 前提成立后，同一冻结
`best_model.pt` 分别对 T1200 short input 和 T1800 long input 重新执行单次
前向；Q 使用完整 400 点 canonical ascending-Q field，Q 与 y 使用同一置换。两个
truth comparison 都使用 short dataset 的 shared raw float64 truth。没有 adaptation、
autoregressive rollout、teacher forcing 或 fine-tuning。

| 比较 | Global Relative L2 | Mean-per-Q Relative L2 |
|---|---:|---:|
| short prediction vs shared truth | 0.0071953455 | 0.0054274904 |
| long-input prefix prediction vs shared truth | 1.6991658080 | 1.7016413755 |
| long-input prefix prediction vs short-input prediction | 1.6996710240 | 1.7021071446 |

该正式结果在 mean-per-Q 口径上实质复现了历史临时值约 `0.00542749`、
`1.701641` 和 `1.702107`。因此 short-input prediction 与 long-input-prefix
prediction consistency 已不再是 `unknown`：在这一历史冻结 Q-only FNO2D 协议下，
将 lambda 输入域从 T=1200 延长到 T=1800，会显著改变模型在原共享 T=1200 前缀上
的预测。由于 Stage-2 truth pairing、canonical Q ordering、同一 checkpoint 的
fresh inference 与 shared truth 均已满足，这一现象不能由 short/long ground-truth
prefix mismatch 或 Q-axis scrambling 解释。

该诊断建立现象，不建立机制。它不支持“FNO cannot perform Kerr length
extrapolation in general”、不支持将原因确定为 FFT frequency-grid change，也不支持
推广到所有 FNO architectures。该诊断本身只比较 T1200/T1800；三长度 formal A1
结果记录在下一节。

### 7.5 Formal A1 three-length frozen evaluation

用户提供的 formal A1 结果在同一冻结 Q-only FNO2D、full canonical ascending-Q
field、raw dataset float64 truth 和一次完整前向/长度协议下，已完成 T1200、T1800
和 T2400 的评估。T1200 是准确的 same-domain baseline；T1800/T2400 都在共享
T1200 前缀上显著退化：

| 输入长度 | Prefix mean-per-Q Relative L2 | Extrapolation mean-per-Q Relative L2 | Full mean-per-Q Relative L2 |
|---|---:|---:|---:|
| T1200 | 0.005427490395002388 | N/A | 0.005427490395002388 |
| T1800 | 1.7016413755435957 | 2.175171290898468 | 1.8720813646855663 |
| T2400 | 2.2687772980088625 | 1.827294006317696 | 2.062092417362002 |

T2400 的 shared-prefix 误差高于 T1800。固定物理 lambda-window 指标也显示，误差
相对于训练边界之外物理距离并不单调。结果建立的是当前冻结协议的 domain-length
sensitivity 现象，不建立其机制。

### 7.6 受支持与不受支持的结论

受支持：该一次冻结模型、同 `delta_lambda`、更大物理 λ 域的 T1800 协议失败，
并且失败并非只体现在新增尾部。

不受支持：

```text
FNO cannot perform Kerr length extrapolation in general.
```

formal A1 已完成 T2400 的冻结单次推理并确认上述 domain-length sensitivity；它不
证明机制，也不应被推广为所有 FNO architecture 或一般 Kerr length extrapolation 的
结论。

### 7.7 FNO2D Q-axis ordering methodological boundary

`Git/code-derived`: in the Q-only FNO2D tensor convention `[B, H, W, C]`, the `H` dimension is the Q parameter-grid/operator axis, the `W` dimension is the lambda axis, and the field channels are `[Q, lambda]`. Spectral convolutions apply the FFT over both spatial/operator dimensions (`H` and `W`); consequently, Q ordering is part of model-input semantics rather than a cosmetic row-order choice. Random NPZ split row order must not be supplied directly as the FNO2D `H` axis.

The required model-input convention is a stable canonical ascending-Q field: apply one permutation to Q and the corresponding y rows, and use the same canonical convention for training, common-test, and length-change evaluation. This does not contradict Stage-2 dataset identity validation. Dataset identity evidence retains each original train/val/test source-row order and does not sort Q, whereas FNO evaluation constructs a separate canonical model-input field after source identity has been preserved and checked.

An early formal consistency execution with invalid scrambled Q-axis ordering is a `protocol-debug artifact`, not scientific model-performance evidence and not a negative scientific result. Its numeric output is intentionally not retained in this document or the registry. The local correction is recorded in Git commit `a68dc2c`; it corrects the formal diagnostic introduced in `97de23d` without changing the Stage-2 raw dataset identity semantics.

### 7.8 Current mechanism hypotheses for FNO2D domain-length sensitivity

#### Established observation

- T1200 same-domain baseline is accurate (`mean_per_q_relative_l2 = 0.00542749`), whereas T1800 and T2400 long inputs strongly degrade the exactly shared T1200 truth prefix; T2400 prefix error is higher than T1800 prefix error.
- T1800/T2400 lambda-window errors are non-monotonic with physical distance beyond the training boundary.
- This is a single full-field frozen evaluation, not autoregressive rollout. The observed whole-field and shared-prefix changes therefore do not resemble simple stepwise/autoregressive error accumulation.

#### Overall working hypothesis — not yet experimentally verified

FNO2D performs global spectral operations over the parameter/lambda field. Changing the total physical lambda-domain length changes the global Fourier representation even when the original shared prefix is unchanged. This may alter the physical meaning of retained Fourier mode indices, the physical spectral bandwidth represented by a fixed number of modes, and the coordinate distribution seen by the network. This is a working hypothesis, not an experimentally verified cause.

#### Mechanism candidates — not results

1. **Fourier mode physical-frequency shift.** For physical domain length `L`, mode index `k` corresponds approximately to frequency `k / L`, or angular frequency `omega_k ~ 2*pi*k/L`. When `L ≈ 6 -> 9 -> 12`, the same discrete `k` no longer denotes the same physical frequency. A physical oscillation represented near one mode index during training may therefore shift to another index on a longer domain. M2 gives direct descriptive raw-spectrum support, and M3 gives direct real-model internal evidence that physically aligned content can be routed through different learned discrete-mode weights. Candidate 1 is strongly supported as a real internal mechanism pathway, but its quantitative contribution to final A1 error remains unresolved.
2. **Fixed-mode physical bandwidth shrinkage.** The FNO2D lambda direction retains a fixed number of Fourier modes. If its maximum retained index is approximately fixed at `K`, then the represented maximum physical frequency behaves approximately as `f_max ~ K / L`. Increasing `L` at fixed `K` reduces the physical-frequency bandwidth of those modes. M2 confirms the mathematical shrinkage but weakens the simplest raw-truth energy-loss explanation; it does not determine the role of internal features or nonlinear/pointwise paths.
3. **Global spectral representation changes the shared prefix.** Spectral-layer Fourier coefficients depend on the entire input field, not only its local prefix. Changing `T1200 -> T1800 -> T2400` changes both the complete lambda-domain signal and the Fourier basis; hence the representation used for an exactly identical raw T1200 prefix may change substantially. M3 strongly supports this as the earliest observed entry point of length sensitivity in the frozen FNO2D, but does not establish causal sufficiency or isolate it fully from candidates 1, 2, and 4.
4. **Lambda coordinate / domain representation.** In the original model the input channels are `[Q, lambda]`. Longer domains place raw lambda values beyond the T1200 training range; checkpoint normalization also moves their normalized values beyond the T1200 coordinate range. M4a/M4b established a real, measurable coupled pathway. The later physically meaningful R2 retraining result supplies a partial positive repair signal, strengthening this candidate beyond the nonphysical clamp alone, while neither result establishes dominance or unique causality.

#### Formal M2 raw Kerr spectral-energy evidence — intermediate, not causal

M2 used the exact-prefix T1200/T1800/T2400 companion datasets, canonical ascending Q,
raw float64 xyz truth, a one-sided lambda FFT, `f_k = k / (N * delta_lambda)`, and
`modes2 = 32` retained lambda indices `k=0..31`. It loaded no model or checkpoint and
performed no inference.

For the strictest retained physical cutoff, `f_cutoff ≈ 2.583333` (T2400), the
xyz-aggregated raw spectral-energy fraction at or below that frequency was:

| Dataset length | Energy fraction at `f <= 2.583333` |
|---|---:|
| T1200 | 0.9869508789 (about 98.70%) |
| T1800 | 0.9993208433 (about 99.93%) |
| T2400 | 0.9943153572 (about 99.43%) |

The fraction above each dataset's own retained cutoff is also small: about 0.62%
above T1200's `f ≈ 5.1667`, 0.049% above T1800's `f ≈ 3.4444`, and 0.57% above
T2400's `f ≈ 2.5833`.

This weakens, but does not eliminate, the simplest version of candidate 2. Raw Kerr
xyz trajectories do not put a large fraction of energy above the T1800/T2400 retained
physical cutoffs, so the statement that failure occurs because most important raw
high-frequency trajectory energy lies outside the retained 32 modes is poorly
supported. It is not disproven as a broader model mechanism: internal hidden features
may have a different spectrum from raw xyz truth, and pointwise/nonlinear FNO paths
complicate a direct raw-spectrum-to-model-capacity interpretation.

M2 also gives direct descriptive support for candidate 1. The xyz-aggregate spectra
show T1200 dominant `k=3 -> f=0.5`; T1800 `k=5 -> f≈0.5556` with nearby
`k=4 -> f≈0.4444`; and T2400 `k=6 -> f=0.5` with dominant `k=7 -> f≈0.5833`.
The point is not an exact forced peak match: similar physical-frequency content around
`f ~ 0.5-0.6` maps to different discrete indices as total domain length changes. This
supports a real representation remapping, not a causal role in the FNO collapse.

Formal A1 prefix mean-per-Q Relative L2 changes from about 0.0054 (T1200) to 1.70
(T1800) and 2.27 (T2400), while raw truth energy outside the retained cutoff remains
below about 1%. The large A1 degradation is therefore not naturally explained by a
simple loss of a large fraction of raw high-frequency energy. This scale mismatch
motivates M3; it is not causal attribution.

#### Formal M3 internal representation evidence — intermediate, not causal

M3 used the same frozen checkpoint for exact-prefix T1200/T1800/T2400 companion
datasets, the full canonical Q400 field, identical shared T1200 raw prefix, the same
checkpoint normalization, and one normal frozen forward per length. Observational hooks
and read-only replicated FFT/spectral calculations did not modify model behavior.

For T1200 versus T1800, shared-prefix relative differences were zero through
`normalized_input_prefix`, `lifted_feature_prefix`, and `first_spectral_input_prefix`,
then about 0.8571 at `spectral_branch_output_prefix`, 0.5657 at
`first_block_output_prefix`, and 1.6997 at `final_prediction_prefix`. For T1200 versus
T2400, the same three pre-spectral stages were zero, followed by about 1.2817, 0.8658,
and 2.2690 respectively. Thus the identical shared prefix remains unchanged through
normalization, pointwise lifting, and the input to the first spectral convolution;
substantial divergence first appears when the first global Fourier/spectral operation
acts on the full lambda domain. This is observational evidence, not a statement of sole
causality.

The retained first-layer FFT representation already differs strongly at the same discrete
lambda indices: `first_fft_retained_same_index` is about 1.1585 (T1200 vs T1800) and
2.8768 (T1200 vs T2400). After learned spectral multiplication the corresponding
aggregate differences are about 0.7984 and 1.8293. Hence the length change alters the
retained FFT representation before learned spectral weights act; the aggregate
same-index reduction does not imply that learned weights are harmless.

Physical-frequency-aligned comparisons provide the more specific candidate-1 evidence.
At exact `f = 1/3`, T1200 uses `k=2` while T1800 uses `k=3`. Pre-weight relative
difference is about 0.5000 with cosine similarity about 1.0000; after their different
learned mode weights, it is about 1.6592 with cosine similarity about 0.5607. For
T1200 to T2400, the same `f = 1/3` maps from `k=2` to `k=4`: pre-weight relative
difference is about 1.0000 with cosine similarity about 1.0000, while post-weight it is
about 3.1221 with cosine similarity about -0.3288. At exact `f = 1/6`, T1200 `k=1`
and T2400 `k=2` similarly change from about 1.0000 / 1.0000 pre-weight to about
3.0411 / 0.6925 post-weight (relative difference / cosine similarity).

These examples directly show that the same physical frequency can map to different
discrete indices and therefore different learned spectral weights when lambda-domain
length changes. In selected aligned examples learned multiplication substantially
increases relative difference and changes hidden-vector direction. This strongly
supports candidate 1 as a real internal mechanism pathway, but does not show that it
alone explains the final prediction collapse.

T1800/T2400 also append lambda coordinates outside the T1200 training range, and those
coordinate values participate in the same global Fourier transform. M3 alone could not
separate pure domain-length/basis effects from coordinate-range effects; M4 therefore
tested that contribution directly.

#### M4a checkpoint and coordinate/lifting audit — verified metadata

Read-only inspection of the actual server checkpoint verified `normalization = standard`
and `input_channel_order = ['Q', 'lambda']`, with:

```text
x_mean = [2.299506187438965, 2.996041774749756]
x_std  = [0.40356874465942383, 1.7314223051071167]
```

Thus the checkpoint-verified lambda statistics are
`lambda_mean = 2.996041774749756` and
`lambda_std = 1.7314223051071167`. Earlier uniform-grid estimates were close but are
not retained as the formal statistics.

T1200 remains within its training lambda-coordinate range. T1800 appends 600 lambda
positions beyond the T1200 raw training maximum (about 33.3% of its lambda points), and
T2400 appends 1200 positions (about 50%). After checkpoint standard normalization, the
long-domain lambda channel therefore reaches substantially beyond the T1200 normalized
training range. Candidate 4 is consequently architecturally real and testable: appended
out-of-training-range lambda values enter pointwise lifting and then the global spectral
operation.

For each lifted channel `j`, the input projection has the direct coordinate path

```text
h_j = W[j,Q] * Q_normalized + W[j,lambda] * lambda_normalized + b_j
```

Lambda therefore has a trainable route into every lifted hidden channel. M3 established
that the shared prefix is identical through normalization, pointwise lifting, and the
input to the first spectral convolution. Candidate 4 can therefore influence that prefix
only indirectly, through the appended coordinate field entering the global spectral
transform.

#### M4b nonphysical appended-lambda clamp probe — intermediate, not causal

M4b compared `T1800_original` with `T1800_lambda_clamped`, and `T2400_original` with
`T2400_lambda_clamped`. Within each same-length pair it preserved total tensor length,
the FFT grid, Q, checkpoint, weights, shared T1200 prefix coordinates, truth, and every
non-lambda input. It changed only appended normalized lambda values above the T1200
training upper bound.

**The clamped arm is intentionally nonphysical. It is a mechanism probe, not a valid
Kerr length-extrapolation protocol. Its appended-region trajectory error is not
model-performance evidence.**

The original-versus-clamped shared-prefix internal relative differences were:

| Pair | `first_fft_retained` | `post_weight_spectral` | `spectral_branch_output_prefix` | `first_block_output_prefix` | `final_prediction_prefix` |
|---|---:|---:|---:|---:|---:|
| T1800 original vs clamped | 0.2227 | 0.2296 | 0.2469 | 0.1894 | 0.2571 |
| T2400 original vs clamped | 0.4223 | 0.3993 | 0.4445 | 0.3676 | 0.4336 |

Changing only the appended out-of-range lambda-coordinate magnitude therefore produces
a substantial measurable response in the frozen model, especially for T2400. Candidate
4 is not negligible; these values are internal intervention responses, not physical
trajectory-performance improvements.

The response relative to the T1200 reference is non-monotonic. For T1800, the spectral
branch distance changes from `0.7561 -> 0.7032` (about 7% closer) and the first-block
distance from `0.5335 -> 0.4728` (about 11% closer), but final-prediction distance moves
from `1.2049 -> 1.3507` (farther). For T2400, the corresponding spectral-branch change
is `0.8621 -> 0.7730` (about 10% closer), the first-block change is
`0.6605 -> 0.5218` (about 21% closer), and final-prediction distance moves from
`1.2108 -> 1.4644` (farther).

Suppressing the out-of-training-range lambda magnitude partially restores early hidden
representations toward the T1200 reference, but this recovery does not propagate
monotonically to the final output. This demonstrates coupling with later spectral and
nonlinear processing; it is not a production fix or a causal-contribution percentage.

#### Mechanism evidence status after M4a/M4b and R2

1. **Candidate 1 — physical-frequency / discrete-index remapping.**
   `STRONGLY SUPPORTED as a real internal mechanism pathway.` Mathematical
   `f_k = k / (N * delta_lambda)` remapping, raw Kerr dominant-frequency mode shifts,
   and exact `f = 1/3` mappings (`k=2 -> 3 -> 4` for T1200 -> T1800 -> T2400) show that
   physically aligned content reaches different learned discrete-mode weights. Its
   quantitative share of final A1 error is not isolated.
2. **Candidate 2 — fixed retained-mode physical-bandwidth shrinkage.**
   `WEAKENED as a simple primary explanation.` The cutoff shrinkage is mathematically
   real, but roughly 99% of raw xyz spectral energy lies below even the strict T2400
   cutoff. Hidden-feature bandwidth may differ from raw truth, so this mechanism is not
   eliminated.
3. **Candidate 3 — global spectral representation changes the shared prefix.**
   `STRONGLY SUPPORTED as the earliest observed entry point.` Normalized-input, lifted-
   feature, and first-spectral-input prefix differences are all zero, followed by strong
   divergence immediately after the first global spectral operation. This establishes
   the entry pathway, not sole causal sufficiency.
4. **Candidate 4 — lambda coordinate / domain representation.**
   `MEASURABLE AND SUPPORTED BY A PARTIAL POSITIVE REPAIR SIGNAL.` About 33.3% / 50% of
   T1800/T2400 lambda positions exceed the original training coordinate range; verified
   checkpoint normalization confirms substantial normalized extrapolation; and M4b
   measurably changes first-layer spectral representations while partly moving early
   features toward T1200. R2 then replaces absolute lambda by `[s, ell]` in a physically
   meaningful retrained formulation and improves long-input shared-prefix stability.
   Its final prediction and extrapolated-region results remain insufficient, so the effect
   is strongly coupled with global spectral and later nonlinear processing. Candidate 4
   is not established as the dominant cause.

#### M2 scientific boundary

M2 establishes raw spectral-energy distribution, retained physical-frequency cutoff
comparison, and real-data mode-index remapping. It does not establish which mechanism
causes A1 failure, that candidate 2 has zero contribution, that candidate 1 is the
dominant cause, or that internal hidden-feature spectra behave like raw xyz spectra.

#### M3 scientific boundary

M3 establishes where divergence first appears, that the first spectral operation is
globally length-sensitive, and that physical-frequency remapping interacts with different
learned mode weights. It does not establish that candidate 1 or 3 is the sole cause,
that candidate 2 has zero influence, that candidate 4 is negligible, or that a specific
model modification would fix the problem.

#### M4 scientific boundary

M4 establishes checkpoint-verified coordinate extrapolation and a measurable frozen-model
response when only appended out-of-training-range lambda magnitude is changed. It does
not establish a causal-contribution percentage, that candidate 4 is the unique or dominant
cause, that coordinate representation is irrelevant in every alternative formulation, or
that clamping is a valid physical prediction method.

#### Current evidence synthesis — not a mathematical proof of unique causality

The A1 length-sensitivity failure is not well described by a single isolated defect.
Current evidence most strongly supports a coupled global-spectral mechanism: changing
domain length changes the Fourier representation and physical-frequency-to-discrete-mode
mapping; appended out-of-training-range lambda coordinates additionally perturb this
global representation. Simple loss of raw high-frequency content from the fixed 32-mode
cutoff appears insufficient to explain the observed collapse.

#### Mechanism diagnostic questions

##### Question 1 — Physical frequency mapping

When `T` changes `1200 -> 1800 -> 2400` at approximately fixed `delta_lambda`, how do the physical frequencies represented by the retained lambda Fourier mode indices change? This is primarily a mathematical/code audit question.

##### Question 2 — Kerr trajectory spectral energy

Where is raw Kerr trajectory spectral energy concentrated along lambda? Determine whether important trajectory energy shifts across discrete mode indices as domain length changes, and whether significant energy lies outside the physical bandwidth represented by the retained modes.

##### Question 3 — Internal spectral representation sensitivity

M3 established that substantial divergence first appears at the first global spectral operation. Later internal diagnostics may refine how that divergence develops through deeper blocks, while preserving M3 as observational evidence rather than causal proof.

##### Question 4 — Coordinate-range contribution

M4a/M4b established that normalized lambda-coordinate extrapolation has a measurable, coupled frozen-model effect, but did not isolate a unique causal share. Future work may compare alternative coordinate representations only within a clean repair protocol; no conclusion about physical prediction quality may be drawn from the nonphysical clamp arm.

#### A1 versus Plan B conceptual boundary

The hypotheses above concern current A1 domain-length sensitivity:

```text
physical domain-length change
approximately fixed delta_lambda
L changes
```

They do not claim that this A1 failure contradicts canonical FNO grid-resolution/discretization generalization. Future Plan B instead requires:

```text
fixed physical lambda domain
grid density / delta_lambda changes
L fixed
```

#### Resume point

```text
M1 mathematical/code audit: completed
M2 raw Kerr spectral-energy diagnostic: completed
M3 internal FNO2D representation diagnostic: completed
M4a lambda-coordinate normalization/lifting audit: completed
M4b nonphysical lambda-coordinate clamp mechanism probe: completed
```

Mechanism diagnosis led to the R1 and R2 repair experiments recorded below. The project
objective remains to attempt useful direct one-shot lambda-domain length extrapolation
with scientifically motivated modifications; this record does not declare length
extrapolation impossible.

### 7.9 R1 — variable-length training repair (completed; negative result)

#### Question and controlled protocol

R1 asked whether the original failure was mainly caused by exposing the unchanged FNO2D
to only one physical lambda-domain length during gradient training. It is a
`TRAINING_PROTOCOL_REPAIR`: the only intended change was
`single-length training -> multi-length prefix training`.

The architecture and representation remained unchanged:

```text
modes1 = 16
modes2 = 32
width = 64
depth = 4
input channels = [Q, lambda]
absolute lambda coordinate
normalization = standard
target_transform = raw
discrete-index spectral parameterization
```

The sole source task was `q_1p6-3_n2000_t1200`. Original train/val/test Q identities
were preserved. Each length is a strict prefix of the same T1200 trajectory, not an
independent trajectory. Gradient-training lengths were `T600`, `T800`, `T1000`, and
`T1200`; validation/checkpoint-selection lengths were `T700`, `T900`, `T1100`, and
`T1200`. T1800/T2400 truth was excluded from training, normalization, validation, and
checkpoint selection.

Normalization was fitted only from the complete T1200 train field and reused for every
prefix. Each epoch made one optimizer update after equal-weight gradient accumulation
over all four gradient-training lengths. Thus R1 matched the original baseline in
optimizer-update count, not in total forward/backward compute.

#### R1 training outcome and initial comparison limitation

R1 completed 500 epochs with `best_epoch = 500`,
`best_val_selection_score ≈ 0.1517609645`, `optimizer_steps = 500`, and
`forward_backward_passes_per_step = 4`.

Training-Q Relative L2 at gradient-seen lengths was approximately `0.00615` (T600),
`0.00540` (T800), `0.00578` (T1000), and `0.00661` (T1200). The initial validation
summary reported approximately `0.4557` (T700), `0.3446` (T900), `0.5227` (T1100),
and `0.0235` (T1200). That initial seen-versus-held-out comparison was confounded:
the first group used train Q while the second used validation Q. It could not by itself
identify a discrete-length response, so a dedicated same-validation-Q diagnostic was
required.

#### Seven-length validation-Q development diagnostic

The dedicated `r1_validation_q_length_response` diagnostic used one frozen R1
`best_model.pt`, the exact same canonical ascending validation-Q field at every length,
checkpoint-restored normalization and target transform, and one direct forward per
length. It is a `development_diagnostic`, not a formal long-domain extrapolation
result.

| T | Gradient-seen? | Used for checkpoint selection? | raw mean-per-Q RelL2 |
|---:|:---:|:---:|---:|
| 600 | yes | no | 0.0165737402 |
| 700 | no | yes | 0.4727689703 |
| 800 | yes | no | 0.0306300353 |
| 900 | no | yes | 0.3411092655 |
| 1000 | yes | no | 0.0198591204 |
| 1100 | no | yes | 0.5207644813 |
| 1200 | yes | yes | 0.0225884748 |

On the same validation-Q trajectories, R1 performs well at the four lengths that
participated in gradient training and poorly at the intermediate lengths that did not.
This is a strong alternating/discrete-length response: R1 exhibits strong dependence on
specific gradient-seen domain/FFT lengths and does not learn smooth within-range length
interpolation in this experiment. It is not a strict proof of statistical memorization.
T700/T900/T1100 participated in validation/checkpoint selection but not in training
gradients, so they are development lengths rather than untouched final test lengths.

#### Formal R1 A1 evaluation and comparison with R0

Formal A1 evaluation of the R1 `best_model.pt` used the same formal frozen protocol as
R0. The following values are `mean_per_q_relative_l2`:

| Length / region | R0 | R1 |
|---|---:|---:|
| T1200 prefix | 0.00543 | 0.00715 |
| T1800 prefix | 1.70164 | 2.54660 |
| T1800 extrapolation | 2.17517 | 4.48132 |
| T1800 full | 1.87208 | 3.31387 |
| T2400 prefix | 2.26878 | 3.68809 |
| T2400 extrapolation | 1.82729 | 4.97502 |
| T2400 full | 2.06209 | 4.37503 |

The full R1 values are: T1200 prefix/full `0.007146718611955653`; T1800 prefix
`2.546600857634415`, extrapolation `4.481318054165476`, full
`3.313866623129112`; T2400 prefix `3.688094426783673`, extrapolation
`4.97501956795887`, full `4.375028495248427`.

Thus R1 preserved good T1200 in-domain performance but did not improve true
longer-domain performance; T1800/T2400 were substantially worse than R0.

#### R1 assessment and development boundary

R1 is completed as a negative repair result. With the current FNO2D architecture,
absolute `[Q, lambda]` representation, standard normalization, and discrete-index
spectral parameterization unchanged, exposure to several fixed prefix lengths was
insufficient to produce smooth variable-domain generalization. The result supports that
the original failure is not explained solely by single-length training exposure.

A mechanism-consistent interpretation, not a proven explanation, is that multi-length
training imposes conflicting demands on discrete-index spectral weights because the
same `k` corresponds to different physical frequencies at different domain lengths. R1
does not show that FNO cannot perform length extrapolation, nor does it prove the exact
cause of its degradation.

T1800/T2400 have now been used repeatedly during mechanism and repair development.
They remain valid development benchmarks, but after R1–R4 model design is frozen, a
fresh confirmation set with unseen Q and/or unseen long-domain lengths should be
generated for final paper-level confirmation.

### 7.10 R2 — domain-conditioned coordinate representation repair (completed; partial positive repair signal)

#### Question and controlled protocol

R2 asked whether replacing the absolute lambda coordinate with relative position plus
explicit domain-length conditioning can reduce variable-domain sensitivity while keeping
the FNO spectral parameterization unchanged. It is an
`INPUT_REPRESENTATION_REPAIR`:

```text
[Q, lambda] -> [Q, s, ell]
s = lambda / L
ell = L / L_ref
L = N * delta_lambda
L_ref = T1200 logical domain length
```

Here `s` is relative position within the current domain and `ell` is the current logical
domain length relative to the T1200 reference. Absolute lambda is deliberately absent
from the primary R2 input. The architecture remains `modes1 = 16`, `modes2 = 32`,
`width = 64`, `depth = 4`, `hidden_dim = 128`, and `activation = gelu`; only the input
projection changes `in_dim: 2 -> 3`. Standard discrete-index spectral weights remain
unchanged: R2 contains no `R(xi_k)`, physical-frequency interpolation, dynamic weights,
or other R3 path.

The input/target normalization policy is intentionally controlled:

```text
Q:   standard normalization fitted from the full T1200 training-Q field
s:   identity_dimensionless
ell: identity_dimensionless_L_over_L_ref
target_transform = raw
output normalization: T1200-training-derived standard policy
```

R2 inherits the R1 strict-prefix protocol on source task `q_1p6-3_n2000_t1200`:
gradient-training lengths `600, 800, 1000, 1200`; validation/checkpoint-selection
lengths `700, 900, 1100, 1200`; no T1800/T2400 training, normalization, validation, or
checkpoint-selection leakage. There is no padding, mixed-width stacking, consistency
loss, or adaptation. Each epoch performs four forward/backward passes, one optimizer
step, and one scheduler step, preserving the R1/R0-style optimizer-update count.

#### Formal training asset and selection boundary

The completed training run is
`outputs/q_1p6-3_n2000_t1200/`
`fno2d_m16x32_w64_d4_e500_r2_q-s-ell_multilen_t600-800-1000-1200/`, with selected
checkpoint `checkpoints/best_model.pt`. Its recorded metadata is
`experiment_type = r2_domain_conditioned_coordinate_training` and
`repair_class = INPUT_REPRESENTATION_REPAIR`.

Training completed with `epochs = 500`, `optimizer_steps = 500`,
`forward_backward_passes_per_step = 4`, `best_epoch = 118`,
`best_val_selection_score = 0.13257645582780242`, and elapsed time about `1197.13 s`.
The `final_epoch_metrics` are epoch-500 diagnostics, not metrics of the selected best
checkpoint. They still show held-out-length sensitivity at the final epoch: T700 RelL2
about `0.4767`, T900 `0.2892`, T1100 `0.4793`, and T1200 `0.0331`.

#### Formal R2 A1 evaluation and comparison

The R2-aware formal evaluator used the R2 `best_model.pt`, the Stage-2
`EXACT_PREFIX` companion datasets, canonical Q400, raw float64 truth, and one frozen
forward per T1200/T1800/T2400 length. The user-provided successful formal result is
`server-result-verified`. The primary values below are
`mean_per_q_relative_l2`:

| Length / region | R0 | R1 | R2 |
|---|---:|---:|---:|
| T1200 prefix/full | 0.00543 | 0.00715 | 0.07046 |
| T1800 prefix | 1.70164 | 2.54660 | 1.41146 |
| T1800 extrapolation | 2.17517 | 4.48132 | 2.44763 |
| T1800 full | 1.87208 | 3.31387 | 1.82099 |
| T2400 prefix | 2.26878 | 3.68809 | 1.49968 |
| T2400 extrapolation | 1.82729 | 4.97502 | 2.00388 |
| T2400 full | 2.06209 | 4.37503 | 1.76865 |

The full-precision R2 values are T1200 prefix/full `0.07046191157897368`; T1800
prefix `1.4114604873682708`, extrapolation `2.4476288108222564`, full
`1.8209896139615445`; and T2400 prefix `1.4996846482451458`, extrapolation
`2.003878005670123`, full `1.7686532591227653`.

#### R2 assessment and next repair

R2 is a `PARTIAL POSITIVE REPAIR SIGNAL`; coordinate repair alone is insufficient. It
substantially improves shared-prefix stability relative to R1 and also relative to R0
(T1800 prefix `1.7016 -> 1.4115`; T2400 prefix `2.2688 -> 1.4997`). This is direct
repair-stage evidence that coordinate/domain representation matters. However, it
sacrifices substantial T1200 fixed-domain precision (`0.00543 -> 0.07046`) and does not
improve the true extrapolated region relative to R0 (T1800 `2.1752 -> 2.4476`; T2400
`1.8273 -> 2.0039`).

Thus R2 partially repairs long-input shared-prefix stability and improves full-domain
metrics relative to R1, confirming that absolute-coordinate/domain-conditioning effects
contribute to length sensitivity. It does not improve actual extrapolated-region accuracy
relative to R0. This is stronger candidate-4 evidence than the nonphysical M4b clamp
because R2 is a physically meaningful retrained coordinate formulation, but it does not
show that candidate 4 is the sole cause or repair the full failure. Candidates 1 and 3
therefore remain central motivations for R3; candidate 2 remains weakened as the simple
raw-truth bandwidth-loss explanation.

```text
R1 — multi-length training with unchanged architecture
completed; negative repair result

R2 — domain-conditioned coordinate representation
completed; partial positive repair signal; insufficient as a full length-extrapolation repair

Next:
R3 — physical-frequency-aware spectral parameterization
R_k -> R(xi_k), xi_k = k / (N * delta_lambda)

Later and conditional:
R4 — local/windowed spectral redesign only if earlier repairs remain insufficient
```

T1800/T2400 remain development benchmarks because they have been repeatedly inspected
during mechanism and repair development. After the R1–R4 design sequence is frozen, a
fresh confirmation set with unseen Q and/or unseen long-domain lengths is needed for
paper-level confirmation. Plan B remains separate: it is fixed-domain cross-resolution
generalization and is not started here.

### 7.11 R3-B1 — physical-frequency-aware spectral parameterization repair (completed; strong long-domain signal with severe in-domain trade-off)

#### Definition, controlled scope, and training record

R3-B1 is `experiment_type = r3_physical_frequency_spectral_training` and
`repair_class = SPECTRAL_PARAMETERIZATION_REDESIGN`. It retains the R2 coordinate
representation `[Q, s, ell]`, where `s = lambda / L`, `ell = L / L_ref`, and
`L = N * delta_lambda`, but changes the lambda-direction spectral multiplier from
`R_k` to `R(xi_k)`, with `xi_k = k / (N * delta_lambda)`.

The implementation is `physical_frequency_anchor_interpolation` with
`num_lambda_frequency_anchors = 32` and `complex_interpolation = cartesian_linear`.
The runtime retained bins remain `k = 0..31`. Consequently,
`physical_bandwidth_shrinkage_repaired = false` and
`global_fft_structure_unchanged = true`: R3-B1 does not implement a dynamic physical
cutoff, a bandwidth repair, local/windowed FFT, a hypernetwork, or a dynamic retained-mode
count. The R2 architecture and normalization controls remain in force: `in_dim = 3`,
`modes1 = 16`, `modes2 = 32`, `width = 64`, `depth = 4`, `hidden_dim = 128`,
`activation = gelu`, Q standard normalization, identity-dimensionless `s` and `ell`,
and `target_transform = raw`.

The completed training asset is
`outputs/q_1p6-3_n2000_t1200/`
`fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200/`, trained only
from `q_1p6-3_n2000_t1200` strict prefixes. It used gradient-training lengths
`600, 800, 1000, 1200`, validation/checkpoint-selection lengths `700, 900, 1100, 1200`,
`epochs = 500`, `optimizer_steps = 500`, and `forward_backward_passes_per_step = 4`.
T1800/T2400 were excluded from training, normalization, validation, and checkpoint
selection. The selected checkpoint is `checkpoints/best_model.pt`, with `best_epoch = 77`,
`best_val_selection_score = 0.15512157417833805`, and elapsed time about `1214.41 s`.
The epoch-500 diagnostics are not best-checkpoint metrics; they still showed held-out
length sensitivity at T700 RelL2 about `0.38195`, T900 `0.59972`, T1100 `0.40859`, and
T1200 `0.02639`.

#### Checkpoint reconstruction provenance incident

An initial formal evaluation stopped before inference with `ValueError: Physical-frequency
anchors must be uniformly increasing.` This was an engineering/provenance incident,
`CHECKPOINT_SERIALIZATION_PRECISION`, not a scientific R3 failure. Training used valid
float64 anchors; the old `model_config.anchor_frequencies` path had implicitly
round-tripped them through float32, whereas canonical `config["anchor_frequency_values"]`
and the state-dict anchor buffers preserved the scientifically correct float64 values.

The reconstruction fix now preserves float64 future serialization, validates monotonicity
separately from uniform spacing, reconstructs from canonical `anchor_frequency_values`,
and verifies state-dict anchor buffers against that provenance. Legacy float32-rounded
model-config anchors are accepted only after scientific-equivalence validation. The
existing `best_model.pt` remained valid; no retraining was required.

#### Formal R3 A1 evaluation and R0–R3 comparison

The manually verified formal asset is
`outputs/formal_a1_length_extrapolation/`
`fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200_best_fixed_q400_t1200_t1800_t2400/`,
using the epoch-77 checkpoint above. Its configuration confirms
`coordinate_representation = ["Q", "s", "ell"]`,
`spectral_parameterization = physical_frequency_anchor_interpolation`, and
`physical_frequency_formula = k / (N * delta_lambda)`. The compact formal artifacts are
`r3_a1_length_extrapolation_summary.json`, `r3_per_q_metrics.csv`, and
`r3_lambda_window_metrics.csv`.

The frozen formal protocol retained Stage-2 `EXACT_PREFIX`, canonical Q400, raw float64
truth, and one direct forward per T1200/T1800/T2400. The following primary values are
`mean_per_q_relative_l2`:

| Region | R0 | R1 | R2 | R3-B1 |
|---|---:|---:|---:|---:|
| T1200 prefix/full | 0.00543 | 0.00715 | 0.07046 | 0.30424 |
| T1800 prefix | 1.70164 | 2.54660 | 1.41146 | 0.83451 |
| T1800 extrapolation | 2.17517 | 4.48132 | 2.44763 | 1.34676 |
| T1800 full | 1.87208 | 3.31387 | 1.82099 | 1.03204 |
| T2400 prefix | 2.26878 | 3.68809 | 1.49968 | 0.89094 |
| T2400 extrapolation | 1.82729 | 4.97502 | 2.00388 | 1.35702 |
| T2400 full | 2.06209 | 4.37503 | 1.76865 | 1.14639 |

The full-precision R3-B1 values are T1200 prefix/full `0.30423661134631347`; T1800
prefix `0.8345073047219703`, extrapolation `1.3467645409082092`, full
`1.0320417147537395`; and T2400 prefix `0.8909406006838355`, extrapolation
`1.3570231040536689`, full `1.1463882703177917`.

#### Assessment, mechanism evidence, and next repair question

R3-B1 is a `STRONG POSITIVE LONG-DOMAIN REPAIR SIGNAL WITH SEVERE IN-DOMAIN ACCURACY
TRADE-OFF`, not a complete successful model. Relative to R0, it improves T1800/T2400
shared-prefix stability (`1.7016 -> 0.8345`; `2.2688 -> 0.8909`) and also improves the
true extrapolation region (`2.1752 -> 1.3468`; `1.8273 -> 1.3570`). This is stronger
than R2's partial signal because R2 did not improve extrapolation-region accuracy relative
to R0. However, R3-B1 severely degrades original T1200 accuracy (`0.00543 -> 0.30424`),
so its robustness/generalization gain is coupled to an unresolved fidelity trade-off.

Candidate 1 — physical-frequency / discrete-index remapping — is therefore strongly
strengthened: R3-B1 provides positive repair evidence that discrete-index-bound spectral
weights contribute materially to domain-length sensitivity. It does not prove that
candidate 1 is the sole cause. Candidate 3 remains strongly supported but un-repaired because
`global_fft_structure_unchanged = true`; candidate 4 remains supported by R2; and
candidate 2 remains weakened as a simple raw-truth bandwidth-loss explanation.

```text
R1 — completed; negative repair result
R2 — completed; partial positive repair signal; insufficient
R3-B1 — completed; strong positive long-domain repair signal with severe T1200 trade-off

R3 seven-length validation-response diagnostic — completed; apparent sawtooth reduced, but within-range interpolation remains unresolved
Decision point: evaluate whether to proceed to R4 — global/local spectral redesign
R4 — candidate next stage; not started
```

### 7.12 R3 seven-length validation-response diagnostic (completed; development diagnostic)

This completed `development_diagnostic` has `formal_test_evidence = false`. Its only
question was whether the completed R3-B1 physical-frequency-aware spectral
parameterization reduced the R1-style discrete-length sawtooth on exactly the same
validation-Q trajectories across T600–T1200. It is not new training, formal
T1800/T2400 A1 evidence, R3-B2, R4, or an untouched-test evaluation.

The source task was `q_1p6-3_n2000_t1200`; the frozen asset was
`outputs/q_1p6-3_n2000_t1200/`
`fno2d_m16x32_w64_d4_e500_r3_physicalfreq_multilen_t600-800-1000-1200/`
`checkpoints/best_model.pt` at `checkpoint_epoch = 77`. The diagnostic used `split = val`,
`validation_q_count = 300`, `canonical_q_order = stable_ascending`,
`same_validation_q_across_lengths = true`, and
`strict_prefixes_from_single_t1200_source = true`. It evaluated the seven strict prefixes
T600/T700/T800/T900/T1000/T1100/T1200 with seven frozen forwards and no training,
adaptation, normalization refit, or autoregression. T600/T800/T1000 were
`gradient_seen`; T700/T900/T1100 were `non_gradient_validation_length`; and T1200 was
`gradient_seen_and_checkpoint_selection_seen`. The non-gradient lengths participated in
checkpoint selection, so they are development validation lengths, not untouched tests.

The exact completed R3-B1 semantics were preserved: `[Q, s, ell]`,
`s = lambda / L`, `ell = L / L_ref`, `L = N * delta_lambda`, and `L_ref = 6.0`, with
`physical_frequency_anchor_interpolation`, `xi_k = k / (N * delta_lambda)`, and canonical
float64 anchor provenance. `global_fft_structure_unchanged = true` and
`physical_bandwidth_shrinkage_repaired = false` remain true.

The primary metric is `mean_per_q_relative_l2` on the same validation-Q set:

| Length | Exposure status | R3 mean-per-Q RelL2 |
|---|---|---:|
| T600 | gradient seen | 0.07232 |
| T700 | non-gradient validation | 0.44983 |
| T800 | gradient seen | 0.16856 |
| T900 | non-gradient validation | 0.42520 |
| T1000 | gradient seen | 0.32227 |
| T1100 | non-gradient validation | 0.44830 |
| T1200 | gradient + checkpoint selection seen | 0.30243 |

The local interpolation residuals were `r_700 = 0.329393629158993`,
`r_900 = 0.17978639384314585`, and `r_1100 = 0.13595013039622889`; their mean absolute
value was `0.2150433844661226` and maximum was `0.329393629158993`. The
`gradient_seen_mean = 0.2163920356766954`,
`non_gradient_validation_mean = 0.4411089046701902`, and their gap was
`0.2247168689934948`. The adjacent-response summaries were
`mean_adjacent_absolute_change = 0.2150433844661226` and
`max_adjacent_absolute_change = 0.37751532371037944`.

Against the previously recorded R1 same-validation-Q diagnostic, the apparent sawtooth
amplitude decreases, but not for the naive reason that intermediate lengths become good:

| Development diagnostic summary | R1 | R3-B1 |
|---|---:|---:|
| mean absolute interpolation residual | 0.42152 | 0.21504 |
| max absolute interpolation residual | 0.49954 | 0.32939 |
| gradient-seen mean | 0.02241 | 0.21639 |
| non-gradient validation mean | 0.44488 | 0.44111 |
| non-gradient minus gradient gap | 0.42247 | 0.22472 |

R3 is therefore less sharply sawtoothed numerically, but the reduction is driven primarily
by severe degradation at gradient-seen lengths rather than substantial improvement at the
intermediate non-gradient validation lengths. It does not solve within-range length
interpolation or establish smooth generalization to unseen intermediate lengths. The R3
T1200 validation value `0.30243` is also close to the independent formal-Q400 R3 T1200
value `0.30424`; this is a metric-consistency check across evaluation populations, not a
claim that the datasets are identical.

This development result coexists with, and does not contradict, the formal R3 long-domain
signal: R3-B1 improved R0 T1800 prefix `1.7016 -> 0.8345`, T1800 extrapolation
`2.1752 -> 1.3468`, T2400 prefix `2.2688 -> 0.8909`, and T2400 extrapolation
`1.8273 -> 1.3570`, while degrading R0 T1200 `0.00543 -> 0.30424`. The appropriate
synthesis is that R3-B1 materially changes long-domain behavior and improves
long-domain robustness/extrapolation, but introduces a severe fidelity trade-off and leaves
substantial within-range discrete-length sensitivity unresolved.

Candidate 1 remains strongly supported by its formal positive repair evidence, with the
necessary boundary that physical-frequency-aware weights do not by themselves yield
uniformly accurate or smoothly interpolating behavior across lengths. Candidate 3 remains
an unresolved structural mechanism because the global FFT was unchanged; the residual
sensitivity keeps that pathway scientifically relevant for a later repair, but this diagnostic
does not prove candidate-3 causality. Candidate 4 remains supported by M4/R2, and
candidate 2 remains weakened as the simple raw-spectrum bandwidth-loss explanation.

R3 seven-length validation-response diagnostic is completed. The next Plan A decision point
is whether to proceed to R4 global/local spectral redesign; R4 is a candidate stage and is
not started. T1800/T2400 remain development benchmarks because they have been repeatedly
inspected during mechanism and repair development. The same is true of this diagnostic's
checkpoint-selection validation lengths. After design freeze, paper-level confirmation
requires fresh unseen Q and/or unseen long-domain lengths.

## 8. Sparse reconstruction lineage

```text
sparse reconstruction formulation
    ↓
Linear / PCHIP
    ↓
FNO1D
    ↓
Dilated ResNet1D
    ↓
canonical TimesNet
    ↓
TimesNet diagnostics
    ↓
lambda-isolated TimesNet
```

`dbbb6ac` 加入可配置 sparse sampling、observed/hidden masks、Linear/PCHIP
以及 hidden-only 指标。随后 FNO1D、Dilated ResNet1D、canonical TimesNet 和
lambda-isolated TimesNet 依次进入项目；这一顺序由 Git 与服务器资产共同
支持。

以下为 n500/T1200 test、raw hidden-only overall `Relative L2`：

| 模型 | stride 16 | stride 32 | 证据 |
|---|---:|---:|---|
| Linear | 7.8615e-03 | 3.0543e-02 | `registry-only` |
| PCHIP | 2.5012e-03 | 1.4933e-02 | `registry-only` |
| FNO1D | 1.3243e-03 | 1.8291e-03 | `snapshot-verified` |
| Dilated ResNet1D | 1.8703e-03 | 1.6938e-03 | `snapshot-verified` |
| canonical TimesNet1D | 1.6576e-02 | 1.5106e-01 | `snapshot-verified` |
| lambda-isolated TimesNet1D | 9.2312e-03 | 1.6283e-02 | `snapshot-verified` |

在该单数据集、单 seed、固定配置的同分辨率比较中：

- FNO1D 与 Dilated ResNet 都获得低误差；FNO1D 在 stride 16 更低，ResNet 在
  stride 32 略低；
- canonical TimesNet 表现明显较差，特别是 stride 32；
- lambda-isolated 消融显著改善 TimesNet，但未超过 FNO1D 或 ResNet；
- TimesNet 的负结果仍是有用的、受限条件下的实验事实，而不是无效实验。

这些结论不能扩展为普遍的模型优劣排序。

## 9. Sparse observation-density generalization

快照确认三份输出资产存在：

```text
fno1d_train16_test32.json
resnet1d_train16_test32.json
timesnet1d_train16_test32.json
```

本地冻结评估器要求原始数据集路径、序列长度和 test 样本数与训练运行完全
一致，只在评估时重建 stride 32 的 sparse observation pattern，并禁止参数或
归一化适配。因此这项工作准确分类为：

```text
sparse observation-density generalization
```

底层 `T=1200` 和 `lambda_grid` 没有改变；它不是 classical FNO
grid/discretization-resolution generalization。三份 JSON 内容未嵌入
Stage-1 snapshot，当前为 `asset-only`，不能报告具体数值或退化因子。

## 10. TimesNet diagnostics

已存在的诊断链包括：frequency-selection、raw-input spectrum、projection
spectral contribution 和 lambda-isolated period-selection。

- canonical TimesNet 的频率、输入谱和投影贡献文件均存在；注册表的频率和
  分量叙述为 `registry-only`，不得上升为直接验证的因果结论；
- lambda-isolated 两份运行的 period-selection JSON 被直接嵌入，属于
  `snapshot-verified`：stride 16 的 block 0 主要选择周期 5/8、block 1 为
  400/300；stride 32 的 block 0 为周期 2、block 1 为周期 2/400；
- 这些结果证明消融改变了 period selection，不证明 lambda 是 canonical
  TimesNet 失败的唯一原因，也不把选出的周期解释为物理 Kerr 周期。

## 11. Current evidence classification

| 家族 | 分类 | 证据等级 | 置信度 | 理由 |
|---|---|---|---|---|
| current v1 FNO2D common-test 标度 | formal | `snapshot-verified` | 高 | 训练、检查点、独立 common-test 和物理指标完整。 |
| legacy `experimental_v1` FNO2D | superseded | `snapshot-verified` | 高 | 新正式工作优先 current v1；未被认定为损坏。 |
| normalization / target-transform 比较 | unknown | `Git/code-derived` | 中 | 有代码演进，缺少完整成对服务器结果。 |
| T1800 长度外推 | formal | `snapshot-verified` + `server-result-verified` | 高 | exact truth-prefix、canonical-Q diagnostic 与 formal A1 冻结评估均已记录；机制仍未确定。 |
| T2400 长度外推 | formal | `server-result-verified` | 高 | exact-prefix long-domain truth 与 formal A1 冻结评估已记录；机制仍未确定。 |
| R3-B1 physical-frequency spectral repair | formal repair | `server-result-verified` | 高 | 长域 prefix/extrapolation 均较 R0 改善，但 T1200 精度严重退化；为 development evidence，非完整修复。 |
| R3 seven-length validation response | development diagnostic | `server-result-verified` | 高 | 同一 validation-Q 的离散长度响应仍显著；锯齿幅度变小主要来自 gradient-seen 长度退化，而非中间长度显著改善。 |
| Plan B T1200/T2399 bidirectional resolution core | formal | `server-result-verified` + `human-context` provenance | 高 | Fixed-Q400 paired truth is qualified at about `1e-9` Relative L2; both frozen transfer directions retain native-scale accuracy with modest degradation. |
| Linear/PCHIP sparse sweep | formal | `registry-only` | 中 | 注册表称正式扫掠，结果文件存在但内容未嵌入。 |
| sparse FNO1D / ResNet | formal | `snapshot-verified` | 高 | 配置、训练摘要、检查点和隐藏点指标均存在。 |
| canonical TimesNet | formal | `snapshot-verified` | 高 | 已完成的受限配置比较；负结果不等于无效。 |
| TimesNet spectrum/projection | diagnostic | `registry-only` + `asset-only` | 中 | 资产存在，canonical 原始诊断内容未嵌入。 |
| lambda-isolated TimesNet | diagnostic | `snapshot-verified` | 高 | 消融指标和 period-selection 结果直接存在。 |
| sparse `16 -> 32` | exploratory | `asset-only` | 中 | 冻结协议和三份结果文件存在，数值未捕获。 |
| second-order solver validation | formal | `snapshot-verified` | 高 | 收敛、批量和数据集参考解验证直接存在。 |
| 原始 Q-only FNO1D 数值结果 | unknown | `Git/code-derived` | 高 | 代码历史存在，服务器数值结果未恢复。 |

## 12. Relation to future Plan A / Plan B

| 既有工作 | Plan A | Plan B | 正确解释 |
|---|---|---|---|
| 常规 FNO2D T1200 拟合/标度 | 不测试 | 不测试 | 固定域、固定网格的 surrogate 比较。 |
| T1800 长输入 | 覆盖核心形式 | 不测试 | 冻结、同 `delta_lambda`、更大物理域、一次前向；truth-prefix、canonical-Q diagnostic 与 formal A1 评估均已完成，机制仍未确定。 |
| T2400 长输入 | 覆盖核心形式 | 不测试 | exact-prefix 配对与 formal A1 冻结单次评估均已完成；机制仍未确定。 |
| sparse 同分辨率重建 | 不测试 | 不测试 | 观测掩码重建。 |
| sparse stride16 -> stride32 | 不测试 | 不测试 | 观测密度/掩码分布变化，不是网格密度变化。 |
| 求解器验证 | A/B 前提 | A/B 前提 | 数值数据可信性，不是泛化实验。 |

Plan A 的历史 T1800 工作应被复用为知识和资产，而不是盲目重做或丢弃；Stage-2
exact-prefix identity、canonical-Q formal prediction consistency、formal A1
T1200/T1800/T2400 evaluation、§7.8 的 M1–M4 机制诊断，以及 R1/R2 repair
评估均已完成。R1 是负修复结果；R2 是 partial positive repair signal，但不足以成为完整
长度外推修复；R3-B1 则给出强的长域正向修复信号，同时伴随严重 T1200 精度代价。R3
seven-length validation-response diagnostic 已完成：表观锯齿幅度减小主要来自
gradient-seen 长度的 fidelity 退化，而非中间 non-gradient validation lengths 的实质改善。
当前 A1 决策点是评估是否进入 R4 global/local spectral redesign；R4 仍为候选性后续
方案，尚未开始。本文件不预先宣告长度外推不可能。

### Plan B Protocol v1 current status

- Plan A is paused pending the advisor report; its historical results and current R4
  decision point are retained above.
- Plan B coarse-to-fine is complete on the fixed independent Q400 field: the paired
  Q400/T2399 truth has 400/400 successes, zero failures, paired completeness `True`, and
  structural qualification `True` against Q400/T1200 at shared nodes.
- Shared-node numerical-truth Relative L2 mean/median/max are
  `5.218413681635546e-09` / `5.095788550601654e-09` /
  `6.894048665372391e-09`; this is negligible relative to model error in this experiment.
- The historical T1200 Q-only checkpoint produced native coarse global/mean-per-Q
  Relative L2 `0.007195345500573474` / `0.005427490395002388` and fine full-grid values
  `0.007756728427546919` / `0.006257048792878679`: moderate +7.8% / +15.3% degradation,
  not catastrophic failure.
- Fine common-node global/mean-per-Q Relative L2 are `0.008279117934318694` /
  `0.006793341748433332`. The corresponding prediction discretization shift has global
  Relative L2 `3.008818547630e-03` and worst Q `1.62173157895`.
- The matched T2399 workflow and theta2399 training are complete under the original-n2000
  split/physics contract; theta2399 selected best epoch 500 and used newly fitted T2399
  train-split standard normalization, not numerical reuse of T1200 statistics.
- Theta2399 native T2399 global/mean-per-Q Relative L2 are `0.007256035373512494` /
  `0.005506914674547638`; frozen reverse T1200 values are `0.007619677632647374` /
  `0.006087018417501273`, i.e. about +5.0% / +10.5% relative degradation without a new
  catastrophic Q region.
- The complete 2x2 matrix supports bidirectional practical fixed-domain resolution
  generalization between T1200 and T2399. This contrasts with Plan A, where increasing T
  extended the physical lambda domain and frozen length extrapolation failed severely.
- The completed endpoint-fixed R1--R4 range matrix is recorded at
  `outputs/plan_b_resolution_range_t1200_t2399_t3598_t4797/resolution_range_matrix.json`.
  Theta1200 global / mean-per-Q Relative L2 changes smoothly from its native T1200 values
  to `0.008292316031400246` / `0.006999674680921897` at T4797 (+15.2% / +29.0%).
  Theta2399 is nearly flat toward finer grids: T3598 and T4797 global changes relative to
  native T2399 are about +0.82% and +1.68%, respectively. Neither observation establishes
  exact or arbitrary-resolution invariance.
- The FNO-only Plan B protocol, bidirectional core, and tested 1x--4x range stage are
  complete. Cross-model Benchmark Protocol v1 is the locked next stage; no multi-parameter
  QA is currently authorized.

## 13. Current unresolved questions

- Plan B FNO range evidence beyond the bidirectional-core summary must retain verified asset provenance; Benchmark Protocol v1 fixes the next cross-model comparison at R1--R4, while 6x/8x remain conditional rather than automatic;
- 在 R3-B1 已完成的七长度开发诊断之后，R4 所针对的 global FFT / whole-domain coupling 是否能在同时保持 T1200 fidelity 与长域 robustness 的条件下减弱残余长度敏感性；
- 在该冻结历史 Q-only FNO2D 协议下，候选机制 1–4 的量化相对贡献，以及何种 mechanism-driven repair 能带来可靠的 direct one-shot length extrapolation；
- 原始 Q-only FNO1D 的服务器数值结果；
- normalization/target-transform 研究的完整可比证据；
- 三份 sparse cross-stride JSON 的原始数值；
- canonical TimesNet 诊断 JSON 的原始统计；
- 研究协作中未记录在资产或 Git 的人类动机与导师决策。

## 14. Resume point

A1 的以下前提、正式评估与机制诊断已完成：

```text
✓ Stage-2 dataset exact-prefix identity
✓ formal short-vs-long prediction consistency diagnostic
✓ formal A1 T1200/T1800/T2400 frozen evaluation
✓ M1 mathematical/code audit
✓ M2 raw Kerr spectral-energy diagnostic
✓ M3 internal FNO2D representation diagnostic
✓ M4a coordinate normalization/lifting audit
✓ M4b nonphysical coordinate-clamp mechanism probe
```

R1（unchanged-architecture multi-length training）已完成，并是负修复结果：它产生强烈的
离散 gradient-seen length 依赖并使 T1800/T2400 恶化。R2（domain-conditioned
coordinate representation）已完成，是 partial positive repair signal：它改善了长输入
shared-prefix stability，却牺牲 T1200 精度，且未改善相对 R0 的真实 extrapolation
region。R3-B1（physical-frequency-aware spectral parameterization）也已完成：它显著改善
T1800/T2400 prefix 与 extrapolation，却以严重 T1200 accuracy 代价为交换。R3 seven-length
validation-response diagnostic 现已完成：表观锯齿减小主要是 gradient-seen length 的低误差
谷值上升，within-range interpolation 仍未解决。下一 A1 决策点是评估是否进入 R4 global/local spectral redesign；R4 尚未开始，且所有阶段都必须保持干净的冻结/重训比较协议。
只有合理修复路径反复失败后，才能在已测试条件下讨论当前架构/训练
表述是否缺乏可靠 Kerr 长度外推；当前不作这种结论。

Plan A is paused pending the advisor report. Plan B FNO-only work is complete on fixed Q400
and endpoint-fixed `[0, 5.995]`: the bidirectional T1200/T2399 core and the R1--R4
T1200/T2399/T3598/T4797 range matrix retain native-scale raw-xyz accuracy without a
threshold-like collapse through the tested 4x refinement. This supports practical tested-range
resolution generalization, not exact, universal, or arbitrary-resolution invariance and not
QA/multi-parameter generalization. The next project execution step is task-aligned cross-model
implementation plus capacity matching; broader 6x/8x expansion and QA remain conditional later
decisions.

## 15. Cross-model Benchmark Protocol v1 current status

The Plan B FNO bidirectional and resolution-range stage is complete as the recorded
predecessor to the next benchmark decision. Its completed FNO evidence remains separate
from Plan A physical-domain extension and from sparse observation-density experiments.

**Cross-model Benchmark Protocol v1 is now locked.** No task-aligned traditional or
operator baseline implementation, capacity-matching search, baseline checkpoint, dataset
regeneration, training run, frozen inference, or server benchmark execution has started.

The protocol fixes the original n2000 T1200/T2399 matched training data and independent
Q400 endpoint-fixed evaluation field. Phase I will compare trajectory-wise BiLSTM, Dilated
ResNet, canonical TimesNet, encoder-only Transformer, FNO1D, and DeepONet at approximately
1.1M parameters. Track B separately studies FNO1D-small, FNO2D-small, and existing
16.8M FNO2D-large; it is a formulation/capacity study rather than a general leaderboard.

The exact next step is **task-aligned model implementation plus capacity matching**, followed
by local unit and smoke tests. No Phase-I or Phase-II experiment is authorized by this
status update alone.
