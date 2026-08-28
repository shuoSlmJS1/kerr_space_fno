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

## 3. Main Line B — True Grid-Resolution Generalization

### 3.1 定义

Plan B 研究同一连续 Kerr 轨迹在不同离散网格表示上的泛化：

```text
same physical lambda domain
different T
different delta_lambda
different discretization density
frozen checkpoint
```

概念上：

```text
coarse grid: same [0, L], fewer points
fine grid:   same [0, L], more points
```

目标是检验：无需重训练，学习到的 surrogate 是否仍能在同一物理 `lambda`
区间的更细表示上保持准确。

### 3.2 非 Plan B 的情况

下列情况不构成真正的 Plan B：

- 物理 `lambda` 域变长的长度外推；
- sparse interpolation；
- 保持 `T=1200` 和原 `lambda_grid` 不变、仅把观测 stride 从 16 改为 32；
- 重新训练、微调或评估时适配归一化。

因此 `train stride16 -> evaluate stride32` 应称为
`sparse observation-density generalization`，而不是 grid-resolution
generalization。

### 3.3 B1 必需条件

正式 B1 前必须具备：

1. coarse/fine 数据对应相同轨迹身份与固定物理域；
2. 明确的 `T`、`delta_lambda`、网格包含关系或可审计映射；
3. 训练 checkpoint 与训练归一化冻结；
4. 对相同物理位置的 consistency diagnostic；
5. 明确的 interpolation/restriction 规则，且不能借此泄漏 fine truth；
6. 后续引入一个合理的 non-Fourier baseline，用于避免把任何差异都归因于
   Fourier operator。

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

阶段顺序固定为：

```text
Documentation / state reconstruction
    ↓
A1
    ↓
A1 review
    ↓
A2 only if justified
    ↓
B1
    ↓
B1 review
    ↓
B2 only if justified
```

Plan A 和 Plan B 不应同时扩张。A1 review 必须先判断历史 `T=1800` 资产、
前缀身份验证和新的正式协议是否已足以回答问题；只有在该判断完成后，才
决定 A2 或转入 B1。

## 6. 证据与记录要求

每个完成阶段至少更新：

- `FNO_KERR_CURRENT_STATE.md` 中的结论和未决问题；
- `SERVER_DATA_EXPERIMENT_REGISTRY.md` 中的资产路径、状态和主要结果；
- 可追溯的 run configuration、checkpoint 身份、数据集身份和结果文件；
- 证据标签：`snapshot-verified`、`registry-only`、`asset-only`、
  `Git/code-derived`、`human-context` 或 `unknown`。

没有直接结果文件的叙述必须降低证据等级，而不能通过重复引用注册表文本
提升为正式数值结论。
