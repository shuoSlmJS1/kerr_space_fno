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

不能把 1.70 解释为 short-input 与 long-input 前缀预测之间的数值差异。当前
没有已存活的直接结果来比较：

```text
prediction from short T=1200 input
```

与：

```text
first 1200 points of prediction from long T=1800 input
```

`compare_length_prefix_predictions_2d.py` 和
`validate_length_dataset_prefix.py` 存在，但快照未找到其输出。故 short/long
预测一致性及严格轨迹真值前缀都仍为 `unknown`。

### 7.4 受支持与不受支持的结论

受支持：该一次冻结模型、同 `delta_lambda`、更大物理 λ 域的 T1800 协议失败，
并且失败并非只体现在新增尾部。

不受支持：

```text
FNO cannot perform Kerr length extrapolation in general.
```

`q_1p6007-2p9993_n400_t2400` 数据集存在，但没有已验证的 T2400 推理结果；
它不能被写成完成的实验。

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
| T1800 长度外推 | exploratory | `snapshot-verified` | 高 | 一份完整负结果存在，但前缀配对与一致性缺失。 |
| T2400 长度外推 | unknown | `asset-only` | 高 | 数据存在，结果不存在。 |
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
| 历史 T1800 长输入 | 覆盖核心形式 | 不测试 | 冻结、同 `delta_lambda`、更大物理域、一次前向；但不是闭合正式 A1。 |
| T2400 数据集 | A 的候选资产 | 不测试 | 尚无结果。 |
| sparse 同分辨率重建 | 不测试 | 不测试 | 观测掩码重建。 |
| sparse stride16 -> stride32 | 不测试 | 不测试 | 观测密度/掩码分布变化，不是网格密度变化。 |
| 求解器验证 | A/B 前提 | A/B 前提 | 数值数据可信性，不是泛化实验。 |

Plan A 的历史 T1800 工作应被复用为知识和资产，而不是盲目重做或丢弃；新的
正式 A1 仍需补齐严格轨迹前缀身份、prefix prediction consistency 和完整输出
防护。Plan B 尚未完成：当前没有在相同物理 λ 域上改变 `T` / `delta_lambda` /
离散密度并使用冻结模型评估的实验。

```text
Plan B has not yet been performed.
```

## 13. Current unresolved questions

- T1200/T1800/T2400 的严格 trajectory-prefix identity；
- short-input 预测与 long-input 前缀预测的一致性数值；
- T2400 是否有未保留的推理结果；
- 原始 Q-only FNO1D 的服务器数值结果；
- normalization/target-transform 研究的完整可比证据；
- 三份 sparse cross-stride JSON 的原始数值；
- canonical TimesNet 诊断 JSON 的原始统计；
- 研究协作中未记录在资产或 Git 的人类动机与导师决策。

## 14. Resume point

**当前不应启动新的科学实验，直到本状态文档和实验计划被冻结并经审阅。**

之后先进行 Stage A1：

1. 识别唯一的正式 Q-only FNO2D checkpoint 与 current/selected comparison
   dataset；
2. 对 short/long 数据集执行 Stage-2 严格 identity validation；
3. 判断历史 T1800 资产哪些可复用，哪些缺少必要验证；
4. 只实现和运行修正后 A1 协议仍缺失的部分；
5. 审阅 A1 后，才决定 A2 或转入 B1。

Plan B 只在 A1 review 完成后开始，并必须保持“同物理域、不同离散网格”的
定义，不能与长度外推或 sparse observation-density generalization 混淆。
