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

1. **Fourier mode physical-frequency shift.** For physical domain length `L`, mode index `k` corresponds approximately to frequency `k / L`, or angular frequency `omega_k ~ 2*pi*k/L`. When `L ≈ 6 -> 9 -> 12`, the same discrete `k` no longer denotes the same physical frequency. A physical oscillation represented near one mode index during training may therefore shift to another index on a longer domain. This is a mechanism candidate, not a result.
2. **Fixed-mode physical bandwidth shrinkage.** The FNO2D lambda direction retains a fixed number of Fourier modes. If its maximum retained index is approximately fixed at `K`, then the represented maximum physical frequency behaves approximately as `f_max ~ K / L`. Increasing `L` at fixed `K` reduces the physical-frequency bandwidth of those modes. It is not yet known whether important Kerr trajectory energy lies outside that retained physical bandwidth.
3. **Global spectral representation changes the shared prefix.** Spectral-layer Fourier coefficients depend on the entire input field, not only its local prefix. Changing `T1200 -> T1800 -> T2400` changes both the complete lambda-domain signal and the Fourier basis; hence the representation used for an exactly identical raw T1200 prefix may change substantially. This directly motivates internal spectral/feature diagnostics, but is not yet isolated from candidates 1 and 2.
4. **Lambda coordinate extrapolation / normalization shift.** The input channels are `[Q, lambda]`. Longer domains place raw lambda values beyond the T1200 training range; checkpoint normalization may also place normalized lambda outside its training distribution. This coordinate shift may contribute independently or jointly with Fourier effects, but normalization is not established as the cause.

#### Four future diagnostic questions

##### Question 1 — Physical frequency mapping

When `T` changes `1200 -> 1800 -> 2400` at approximately fixed `delta_lambda`, how do the physical frequencies represented by the retained lambda Fourier mode indices change? This is primarily a mathematical/code audit question.

##### Question 2 — Kerr trajectory spectral energy

Where is raw Kerr trajectory spectral energy concentrated along lambda? Determine whether important trajectory energy shifts across discrete mode indices as domain length changes, and whether significant energy lies outside the physical bandwidth represented by the retained modes.

##### Question 3 — Internal spectral representation sensitivity

For the exact same shared T1200 truth prefix, how much do early/internal FNO spectral representations change when the full input field is T1200, T1800, or T2400? The objective is to determine whether strong length sensitivity appears in the first spectral layer or develops deeper in the network.

##### Question 4 — Coordinate-range contribution

How much of the failure is associated with lambda coordinate extrapolation / checkpoint normalization rather than Fourier-domain-length effects? Potential future controlled interventions may compare alternative coordinate representations, but no model modification should begin before the diagnostic audit.

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

Next:

1. perform a read-only mathematical/code audit of lambda-axis Fourier mode scaling, retained physical bandwidth, coordinate normalization, and available internal hooks;
2. then design minimal mechanism diagnostics;
3. do not modify or retrain the model before the audit.

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
exact-prefix identity、canonical-Q formal prediction consistency 与 formal A1
T1200/T1800/T2400 evaluation 均已完成。下一 A1 决策点是先进行狭义
failure-mechanism audit，再设计最小诊断；本文件不选择机制结论或架构改动。Plan B
尚未完成：当前没有在相同物理 λ 域上改变 `T` / `delta_lambda` / 离散密度并使用
冻结模型评估的实验。

```text
Plan B has not yet been performed.
```

## 13. Current unresolved questions

- 在该冻结历史 Q-only FNO2D 协议下，lambda-domain 长度改变为何会造成如此大的 shared-prefix prediction change；该现象已确认，但机制尚未确定；
- 原始 Q-only FNO1D 的服务器数值结果；
- normalization/target-transform 研究的完整可比证据；
- 三份 sparse cross-stride JSON 的原始数值；
- canonical TimesNet 诊断 JSON 的原始统计；
- 研究协作中未记录在资产或 Git 的人类动机与导师决策。

## 14. Resume point

**当前不应启动新的科学实验，直到本状态文档和实验计划被冻结并经审阅。**

A1 的以下前提和诊断已完成：

```text
✓ Stage-2 dataset exact-prefix identity
✓ formal short-vs-long prediction consistency diagnostic
✓ formal A1 T1200/T1800/T2400 frozen evaluation
```

下一 A1 决策点是执行 §7.8 的只读 mathematical/code audit，先检查 lambda-axis
Fourier mode scaling、retained physical bandwidth、coordinate normalization 与可用的
internal hooks；之后才设计最小 mechanism diagnostics，且在 audit 前不修改或重训模型。
Plan B 仍未开始；它只在 A1 review 完成后开始，并必须保持“同物理域、不同离散网格”
的定义，不能与长度外推或 sparse observation-density generalization 混淆。
