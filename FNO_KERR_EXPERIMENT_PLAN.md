# Kerr FNO 实验计划

## 1. 目的与研究纪律

本文件定义下一阶段实验应回答的问题、协议边界与证据要求。它不是
历史结果汇总；历史事实和当前已知限制见
`FNO_KERR_CURRENT_STATE.md`。

本项目保留负结果。一个负结果可以排除特定协议、帮助缩小问题范围，
但不会因结果不理想而被删除或自动视为无效。

- 每次只推进一个主要研究阶段；新想法先作为 follow-up，不中断正在
  收敛的阶段。
- 正式评估前冻结数据集、检查点、指标、拆分、随机种子和输出命名。
- 完成一个阶段后，先更新项目状态与 server registry，再决定下一阶段。
- 本地轻量检查、服务器正式实验、以及后续解释必须明确区分。
- 任何新结论都应标记其证据来源；不能用文件名代替结果证据。

当前计划分为两个不应混淆的主线：直接单次长度外推（Plan A）与真正的
网格/离散分辨率泛化（Plan B）。

## 2. Main Line A — Direct One-Shot Length Extrapolation

### 2.1 定义

Plan A 研究冻结的 Q-only FNO2D 在更大物理 `lambda` 域上的直接预测：

```text
same approximate delta_lambda
larger physical lambda domain
T_eval > T_train
frozen checkpoint
single full-sequence forward pass
```

初始算子形式为：

```text
Q-only FNO2D
(Q, lambda) -> xyz
```

候选长度为：

```text
T_train = 1200
T_eval = 1800
T_eval = 2400
delta_lambda = 0.005
```

这些候选值来自已盘点资产，不代表已经选定正式 A1。A1 开始前必须确认
所选 current dataset、冻结 checkpoint、训练归一化与 target transform。

### 2.2 明确排除项

Plan A 不是以下任一种协议：

- autoregressive rollout；
- teacher forcing；
- 将先前预测的 `xyz` 作为后续输入；
- 向模型输入未来真值 `xyz`；
- fine-tuning；
- test-time adaptation；
- 使用评估数据重新计算训练归一化统计量。

模型在完整的长 `(Q, lambda)` 输入上只执行一次前向计算。因而任何远离
训练边界的误差变化都不应被称为“自回归误差累积”。

### 2.3 A1 必需的身份与协议验证

在正式推理前，A1 必须完成 Stage-2 的针对性验证：

1. 确认 short 与 long 数据集的 Q 样本身份、排序与 split 对齐；
2. 验证共享区间的 `lambda_grid` 严格相容；
3. 对选定 short/long 数据集验证轨迹真值前缀身份；
4. 确认 checkpoint 的训练任务、模型结构、归一化和 target transform；
5. 固定 evaluation 的 Q 集、输出目录和指标定义；
6. 记录历史 `T=1800` 资产哪些可直接复用、哪些只能作为对照。

如果任何一项不能成立，应将其记录为协议限制，而不是通过静默重采样或
更换资产规避。

### 2.4 A1 指标

正式 A1 至少报告：

- short-input same-domain reference；
- extended-input 下原训练域的 seen-domain 指标；
- 训练域以外的 extrapolated-domain 指标；
- full-domain 指标；
- `x/y/z` 分量指标；
- 每轨迹 `Relative L2` 的均值、中位数、P95 与最大值；
- 明确的 Q 范围、评估样本数和物理指标空间。

所有主指标使用 raw physical-space `xyz`；训练监控的 model-space 指标
只能作为辅助信息，不能替代最终评估。

### 2.5 距离训练边界的退化诊断

将训练边界后的区域按物理 `lambda` 距离切为连续窗口，例如：

```text
[6.0, 6.5]
[6.5, 7.0]
[7.0, 7.5]
...
```

每个窗口单独计算误差。该诊断回答：随着 `lambda` 离开原训练域，单次
预测是否逐渐变差？它不是 autoregressive error accumulation：后一个点
不使用前一个预测点作为输入。

### 2.6 Prefix consistency diagnostic

除了对真值的 seen-domain 误差，还必须独立比较：

```text
short-input prediction
```

与：

```text
long-input prediction restricted to original domain
```

这项检查衡量输入长度改变是否改变原训练域上的预测；它不等同于
seen-domain error against truth。应先报告 short/long 真值前缀是否严格
配对，再解释预测差异。

### 2.7 可视化语义

长度外推主图固定使用：

```text
blue   = model prediction inside training domain
orange = same model prediction in extrapolated domain
green  = validated numerical-solver truth over full extended domain
```

图中必须标明训练边界。颜色表示域角色，不表示不同模型或不同求解器。

### 2.8 Plan A repair-stage status

本节只更新 Plan A 的 repair-stage 状态，不改变 Plan B 的定义或启动条件。

```text
R1 — multi-length training with unchanged architecture
completed; negative repair result

R2 — domain-conditioned coordinate representation
completed; partial positive repair signal; insufficient as a full length-extrapolation repair

R3-B1 — physical-frequency-aware spectral parameterization
completed; strong positive long-domain repair signal; severe T1200 accuracy trade-off

R3 seven-length validation-response diagnostic
completed; apparent sawtooth reduced mainly because gradient-seen lengths degraded;
within-range interpolation remains unresolved

decision point: evaluate whether to proceed to R4 — global/local spectral redesign
R4 — candidate next stage; not started
```

R1 表明仅增加若干固定训练长度不足以获得平滑的可变物理域泛化。R2 的
`[Q, s, ell]` 表示改善长输入共享前缀稳定性，但未改善相对 R0 的真实外推区间精度，
因而不能替代 R3。R3-B1 以实际物理频率而非离散索引参数化 spectral multiplier，
并改善了相对 R0 的长域 prefix 与 extrapolation 指标，但严重牺牲 T1200 accuracy；它不是
完整修复。已完成的同 validation-Q 七长度诊断显示表观离散锯齿幅度变小，但主要原因是
R3 的 gradient-seen length fidelity 退化；T700/T900/T1100 的 non-gradient validation
accuracy 没有实质改善。因此 within-range interpolation 仍未解决。下一决策点是评估是否
进入 R4 global/local spectral redesign；R4 只在此前修复与诊断仍不足时考虑，当前未启动。

T1800/T2400 已在机制与 repair 开发中重复使用，故为 development benchmarks。R1–R4
设计冻结后，需以未见 Q 和/或未见长域长度的独立 confirmation set 作 paper-level
confirmation。Plan B 仍是固定物理域、改变离散网格的独立主线；其 coarse-to-fine arm 已完成，reverse arm 仍待进行。

## 3. Plan B — Fixed-domain λ discretization-resolution generalization

### 3.1 Protocol v1 lock and scientific question

**Plan B Protocol v1 coarse-to-fine arm is complete; the reverse fine-to-coarse arm is not yet started.**

Plan B tests a Q-only FNO2D checkpoint trained on the coarse lambda grid and later
used with completely frozen weights on a finer lambda grid. Kerr physics, initial
conditions, the independent Q400 evaluation field, and the sampled physical lambda
interval remain fixed. The only active experimental change is lambda-axis
discretization:

```text
delta_lambda = 0.005 -> 0.0025
T            = 1200  -> 2399
```

Plan B does not include Plan A length extrapolation, sparse observation-stride
experiments, Q resampling, retraining, or fine-tuning.

### 3.2 Fixed independent Q400 evaluation field and coarse anchor

Plan B uses the existing independent offset-grid Q400 comparison field: 400 uniform
Q-only Kerr trajectories with Q approximately in `[1.6007, 2.9993]`. This grid was
constructed by a small offset near the original `[1.6, 3.0]` range; it is an
independent comparison field, not an expansion because the original test-300 set was
insufficient. Coarse and fine datasets must use exactly the same Q400 identities and
canonical ordering.

**At a fixed independent Q400 evaluation field, only the lambda-axis discretization is changed.** This does not mean that only lambda changes relative to
the training field: the FNO2D training Q field and the Q400 evaluation field differ.

The existing coarse anchor is Q400/T1200:

- `T=1200`, `step_size=0.005`, and `lambda_grid[j] = j * 0.005`;
- sampled physical interval `[0, 5.995]`;
- historical frozen baseline global Relative L2 `0.0071953455` and mean-per-Q
  Relative L2 `0.0054274904`.

These baseline values are existing recorded results and are not recomputed by Protocol
v1.

### 3.3 Endpoint-fixed fine Plan B grid

Protocol v1 fixes the fine grid as:

- `T=2399`, `step_size=0.0025`, and `lambda_grid[j] = j * 0.0025`;
- sampled physical interval `[0, 5.995]`;
- `fine_lambda[::2] == coarse_lambda`.

The coarse grid has 1199 intervals and the fine grid has 2398 intervals, so every
coarse interval is bisected exactly and the sampled endpoints remain identical.

### 3.4 Controls, exclusions, and truth qualification

The following remain fixed: Q400 identities and ordering; `M`, `a`, `E`, `Lz`, `r0`,
`theta0`, `phi0`, `sign_r`, `sign_th`; solver equations and turning-point logic; FNO
architecture and checkpoint; training normalization statistics; and target transform.

Retraining, fine-tuning, fine-grid normalization refitting, Q resampling, replacement
of failed Q values, and architecture changes are prohibited. `delta_lambda` is the
sole actively changed variable; the T change follows from the refinement.

Before formal frozen FNO inference, paired numerical truth must be qualified by
comparing `fine_xyz[:, ::2, :]` with `coarse_xyz`. The later qualification must report
per-Q Relative L2 and MSE; mean, median, maximum, p95, and p99; failures or anomalies;
and available turning-point diagnostics. Protocol v1 defines structural validity,
numerical-consistency metrics, and anomaly reporting, but no numerical pass threshold.

### 3.5 Completed coarse-to-fine evidence

The formal server assets are `data/tasks/q_1p6007-2p9993_n400_t1200` and the paired
`data/tasks/q_1p6007-2p9993_n400_t2399_plan_b_v1`. The fine field preserves the same
400 independent offset-grid Q identities/order, has 400/400 solver successes, zero
failures, paired completeness `True`, and `fine_lambda[::2] == coarse_lambda`.

The paired qualification at
`outputs/plan_b_q400_t1200_to_t2399/ground_truth_consistency.json` is structurally
valid with zero coarse/fine failures. Its shared-node Relative L2 mean/median/max/p95/p99
are `5.218413681635546e-09`, `5.095788550601654e-09`,
`6.894048665372391e-09`, `6.692889855076337e-09`, and
`6.8532082659278876e-09`; corresponding MSE summaries are
`6.951265864772155e-16`, `6.458674596505792e-16`,
`1.2733836202732828e-15`, `1.1889406467703567e-15`, and
`1.255802623132453e-15`. This supports treating numerical-truth inconsistency as a
negligible confounder for this fixed Q400 comparison, not as a universal solver theorem.

The frozen evaluator used only the historical T1200 Q-only baseline checkpoint
`outputs/q_1p6-3_n2000_t1200/fno2d_m16x32_w64_d4_e500/checkpoints/best_model.pt`;
no Plan A R1/R2/R3 repaired checkpoint, retraining, fine-tuning, normalization refit, or
adaptation was used. Raw-xyz results are recorded at
`outputs/plan_b_q400_t1200_to_t2399/frozen_fno_resolution_generalization/metrics.json`:

| Evaluation arm | Global MSE | Global Relative L2 | Mean-per-Q Relative L2 | Median / P95 / P99 / max | Worst Q |
| --- | ---: | ---: | ---: | --- | ---: |
| Native coarse T1200 | 0.0012590598691126999 | 0.007195345500573474 | 0.005427490395002388 | 0.00423478303876504 / 0.01093169026389112 / 0.022334344948613975 / 0.06534649935265367 | 1.6007 |
| Fine T2399 full grid (primary) | 0.0014630064962514512 | 0.007756728427546919 | 0.006257048792878679 | 0.005189180096032416 / 0.011404593362521268 / 0.022006157951293282 / 0.0662060075938115 | 1.6007 |
| Fine T2399 common nodes (diagnostic) | 0.0016669064017779927 | 0.008279117934318694 | 0.006793341748433332 | 0.0055552606977529355 / 0.012537290722076497 / 0.023640856005393975 / 0.0670052711640239 | 1.6007 |

Fine full-grid error remains in the same scale under endpoint-preserving 2x refinement,
with moderate rather than catastrophic degradation: global Relative L2 changes by about
+7.8% and mean-per-Q Relative L2 by about +15.3%. P95 changes only slightly, P99 and
maximum remain broadly stable, and no new catastrophic Q region appears.

The saved-prediction diagnostic `pred_fine[:, ::2, :]` versus `pred_coarse`, using
`D_i = ||pred_fine_common - pred_coarse||_2 / ||truth_coarse||_2`, has global/mean/
median/p95/p99/max `3.008818547630e-03`, `3.003436179404e-03`,
`2.909109598286e-03`, `3.702832683988e-03`, `3.961805595476e-03`, and
`4.044745297005e-03`; worst Q is `1.62173157895`. This is a measurable finite-grid
prediction shift, not exact resolution invariance. It must not be divided by model-error
norms and interpreted as a literal causal error fraction.

### 3.6 Formal execution order and bidirectional next stage

The completed arm establishes: **the historical T1200 Q-only FNO2D checkpoint has strong
coarse-to-fine discretization-resolution generalization from T1200 to T2399 on this fixed
Q400 field and sampled lambda interval, with modest degradation.** It does not establish
universal or exact resolution invariance.

| Training resolution | Test T1200 | Test T2399 |
| --- | --- | --- |
| T1200 model | Native coarse baseline | Completed coarse-to-fine arm |
| T2399 model | Reverse fine-to-coarse pending | Native fine baseline pending |

Next, generate a new T2399 training dataset aligned with the original n2000 training Q
candidate identities and split semantics, fixed Kerr physics/initial conditions and sampled
interval `[0, 5.995]`; only `T=1200, h=0.005 -> T=2399, h=0.0025` may change. The Q400
assets remain independent evaluation fields and must not be used for training. Train a
matched FNO2D with `modes1=16`, `modes2=32`, `width=64`, and `depth=4`, retaining the
original training protocol wherever code facts permit; train-only normalization fitting is
valid for this new training run. Then evaluate frozen native T2399 and reverse T2399->T1200
on the same independent Q400 field. Do not begin a broader resolution sweep or multi-
parameter QA before this bidirectional matrix is complete.

## 4. 仍有价值、但不属于 Plan A/B 的现有工作

以下工作被保留，不因不直接回答 A/B 而降级或删除：

| 工作 | 当前角色 |
|---|---|
| FNO2D 数据量、宽度、深度和 common-test 比较 | 固定网格上的 surrogate 标度证据 |
| sparse same-resolution reconstruction | 稀疏观测下的重建基线与模型比较 |
| sparse observation-density generalization | 掩码/观测密度分布变化证据 |
| TimesNet diagnostics 与 lambda-isolated ablation | 模型行为诊断，不是因果定论 |
| second-order RK4 validation | 数据生成与未来 A/B 的数值可信性前提 |

## 5. Stage ordering

Plan A is paused pending the advisor report. Plan B coarse-to-fine Protocol v1 is
complete without rewriting Plan A history. The next Plan B stage is the matched-T2399
training and frozen reverse T2399->T1200 evaluation defined in Section 3.6.

Plan A and Plan B remain scientifically distinct. Plan B must not reuse Plan A
length-extension assets as if they were fixed-domain refinement data, and it must not
be conflated with sparse observation-density generalization.

## 6. 证据与记录要求

每个完成阶段至少更新：

- `FNO_KERR_CURRENT_STATE.md` 中的结论和未决问题；
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` 中的资产路径、状态和主要结果；
- 可追溯的 run configuration、checkpoint 身份、数据集身份和结果文件；
- 证据标签：`snapshot-verified`、`registry-only`、`asset-only`、
  `Git/code-derived`、`human-context` 或 `unknown`。

没有直接结果文件的叙述必须降低证据等级，而不能通过重复引用注册表文本
提升为正式数值结论。
