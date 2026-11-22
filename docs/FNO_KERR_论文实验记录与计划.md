# FNO–Kerr 论文实验记录与后续计划

> **文件定位**
>
> 本文件是 FNO–Kerr 项目当前“论文补强 / reviewer-response 实验”的长期维护记录。
> 后续不同对话优先读取并更新本文件，而不是依赖分散的聊天上下文。
>
> **更新时间**：2026-10-24（Priority 1 结果核验与合并；原始研究计划保留）<br>
> **当前阶段**：已有 Plan A / Plan B / Track A 结果基础上，根据导师转发的 Claude 审阅意见补强论文实验。<br>
> **历史算力状态（2026-10-01）**：旧服务器空间紧张，已向老师申请新服务器；服务器容量问题暂不纳入科研方向决策。<br>
> **当前进度**：P1 已在学校平台完成限定范围验收；P2–P5 待完成，P6–P7 后置。本次仅更新文档，不启动新实验。

---

# 1. 来源与背景

导师使用 Claude 对当前 GitHub 仓库及已有结果包进行了审阅。审阅覆盖：

- 顶层 README；
- results package README；
- 数据集 / 定义文档；
- Plan A、Plan B、Track A 三组结果总结。

Claude 没有读取服务器端输出和未进入仓库的源码资产，因此其判断主要基于仓库中已经整理出的实验定义、结果和文档。

导师随后进一步明确：

> Claude 在 `In priority order` 下提出的 7 条建议中，**前 5 条必须完成**；第 6、7 条暂时后置。

因此，本文件后续所有实验优先级均以导师这一确认作为最高约束。

---

# 2. Claude 对当前工作的正面评价

当前工作中较强的部分主要有：

1. **实验溯源和证据管理较严格**
   - 区分已验证事实与未验证解释；
   - 检查了 ground truth 的 shared-prefix identity；
   - 验证 paired-truth consistency，共享节点误差约为 $5\times 10^{-9}$；
   - 区分不同 metric convention；
   - 公开记录 Transformer resume incident。

2. **Plan A 与 Plan B 的对照很清晰**
   - 同一 checkpoint：
     - domain extension 下 mean-per-Q error 约为 $2.06$；
     - $4\times$ refinement 下约为 $0.007$。
   - 说明“定义域延长”和“固定物理域的分辨率变化”是两个不同问题。

3. **M3 诊断具有价值**
   - shared-prefix representation 在进入第一层 spectral layer 之前保持一致；
   - 进入 spectral layer 后开始发生差异。

4. **Track A 控制较好**
   - 模型参数量大致匹配在约 $1.07$–$1.10$M；
   - 使用相同 Q400 field；
   - 使用 frozen BEST checkpoint；
   - evaluation 时不重新拟合 normalization。

---

# 3. Claude 指出的主要问题

## 3.1 Motivation / novelty 是当前最大风险

当前 Q-only 主任务中只有 Carter 常数 $Q$ 变化，而 $M,a,E,L_z$ 和初始条件固定。

因此任务可以写成：

$$
(Q,\lambda)\mapsto (x,y,z),
$$

而不是典型的函数空间到函数空间映射：

$$
f(\cdot)\mapsto u(\cdot).
$$

审阅意见指出：

- Kerr 束缚测地线已有成熟数值方法，且 Mino time 下存在解析结构；
- RK4 生成 1200 步轨道本身计算代价并不高；
- 因此如果把论文定位为“FNO 是一个更实用的 Kerr 轨道 surrogate”，动机容易受到质疑；
- ML 审稿人还会问：这个问题除了重复已知的 FNO discretization / aliasing 现象之外，还提供了什么新认识？

当前需要持续回答的核心问题是：

> **这个实验体系真正教会了我们什么此前不明确的东西？**

---

## 3.2 很多现象可以由模型结构预期，不应写成完全未知现象

### DeepONet

形式近似：

$$
u(Q,\lambda)=\sum_{k=1}^{p} b_k(Q)t_k(\lambda).
$$

某个 query point 的输出并不依赖同一 grid 上还有多少其他 query point，因此 resolution independence 很大程度上是结构决定的。

### Transformer

在当前无 positional embedding、直接把 $\lambda$ 作为 token feature 的结构下，softmax attention 是归一化加权平均。

均匀 refinement 时，分子与分母同时随采样密度变化，因此接近同一连续积分；near-flat behavior 是可以预期的。

### ResNet / BiLSTM

卷积核、dilation、递归步数在 index space 中定义。

例如卷积感受野覆盖 $r$ 个采样点，则对应物理尺度约为：

$$
r\Delta\lambda.
$$

当 $\Delta\lambda$ 改变时，实际物理感受野也改变，因此跨分辨率敏感是有结构原因的。

### FNO / Plan A

spectral weight 按离散 Fourier mode index $k$ 参数化。

当定义域长度变化时，相同离散 mode index 对应的 physical frequency 会变化；同时绝对 $\lambda$ 输入超出了训练范围。

因此 domain extrapolation 失败并不完全是神秘现象。

### 当前可转化的论文逻辑

弱点可以改写为：

> **architecture / inductive-bias prediction → controlled measurement → unexplained quantitative surprise**

真正更值得分析的“surprise”包括：

- 为什么 Transformer 在更高分辨率训练后反而变差；
- 为什么 DeepONet 在更高分辨率训练后反而变差；
- R3 physical-frequency redesign 为什么付出很高 native-accuracy cost。

---

# 4. 缺失的 baseline

Claude 在原始审阅时（2026-10-01 记录）明确指出缺少：

1. **spline-in-Q interpolation baseline**；
2. **plain MLP baseline**；
3. **solver wall-clock vs neural inference timing**。

对于 Q-only 平滑单参数轨道族，spline 有可能非常强。

如果 spline 显著优于所有神经网络，则不能再把论文主要卖点写成“高效 surrogate”，而应更偏向：

> **architecture / discretization / extrapolation behavior study**

这不会自动否定已有实验，但会改变论文定位。

**2026-10-24 更新**：这三类补实验已在当前单 seed、T1200、Q-only 范围完成，详见第 9 节 P1。以上保留为原始审阅问题，不再表示当前仍未执行。

---

# 5. FNO2D 的 Q-axis 结构问题

原 FNO2D 把整个数据组织成类似：

$$
(1,N_Q,T,2)
$$

的 field，并同时在：

- $Q$ 轴；
- $\lambda$ 轴

上做 FFT。

因此某一条 $Q$ 轨迹的预测会依赖同一个 field 中其他 $Q$ 轨迹。

当前存在以下问题：

- training field 中约 1400 个 $Q$，且是从 uniform grid 随机抽取后的非完整网格；
- evaluation 使用 Q400 uniform offset grid；
- 因此测试时不仅改变了 $Q$ identity，也改变了 Q-axis resolution / spacing；
- $Q$ 轴本身又不是天然周期轴；
- 之前的 Q-axis scrambling check 已经证明 trajectory coupling 确实存在。

同时：

- Plan A / Plan B 主要使用 FNO2D；
- Track A 使用 per-trajectory FNO1D。

这会削弱整篇论文模型叙事的一致性。

该问题已经被导师列入**必须解决的前五项**。

---

# 6. Plan A 当前的 clean-test 问题

已有 T1800、T2400 等长度曾用于：

- 诊断；
- redesign；
- R0/R1/R2/R3 对比。

因此这些长度不再适合作为完全独立、一次性的 final test。

Claude 还指出：

- Relative L2 = 1 大致对应“全零预测”；
- 现有部分 Plan A error 达到 $1.3$–$2.5$；
- 即使 R3 约为 $0.83$–$1.36$，也仍不能称为成功 extrapolation。

因此“R3 比原模型有所改善”只能是诊断性结论，不能直接作为强性能结论。

---

# 7. Single-seed 问题

当前主结果大量依赖 seed = 27。

Claude 的判断是：

- 大数量级差异可能对 seed 不敏感；
- 但以下细粒度比较没有多 seed 很难令人信服：
  - FNO1D vs TimesNet；
  - Transformer 的性能变化；
  - DeepONet 的性能变化；
  - 其他接近模型之间的差异。

最终正式表格需要至少报告：

$$
\text{mean}\pm\text{std}.
$$

导师已将 multi-seed 列为**必须完成的第 2 项**。

---

# 8. 论文表达层面的问题

当前结果报告更像内部科研日志，而不是 manuscript。

需要注意：

- 删除 “the user's current retrospective view”；
- 删除 “mentor-facing” 等内部协作措辞；
- 明确 $\lambda$ 的物理 / 数学参数化；
- 增加 trajectory figure；
- 增加 error-vs-$\lambda$ figure；
- 如果 AI 参与了文稿撰写，投稿前检查目标期刊 AI disclosure policy。

此部分属于后续论文整理任务，不抢当前前五项实验优先级。

---

# 9. 导师确认的正式实验优先级

> **以下前 5 项必须完成。顺序按导师确认后的原始 priority order 保留。**

### 七项任务进度（2026-10-24）

| 编号 | 任务 | 导师优先级约束 | 状态 | 验收范围或后续入口 |
| --- | --- | --- | --- | --- |
| P1 | Simple baselines and timing | 必须完成 | **Completed** | 限定单 seed=27、T1200、固定物理参数 Q-only；事实和限制见下 |
| P2 | Multi-seed statistics | 必须完成 | **In progress** | Stage 1：Track A 已验收；FNO2D 三个 seed 27 BEST 已冻结复现，新增种子按配置分别验收。Stage 2 未启动。 |
| P3 | Fresh frozen Plan A test | 必须完成 | **Pending** | 新数据身份与未用于设计的测试域，模型和协议先冻结 |
| P4 | FNO2D Q-axis issue | 必须完成 | **Pending** | 保留并论证 FNO2D，或对关键 cells 改用 FNO1D |
| P5 | Another physical configuration | 必须完成 | **Pending** | 至少另一个固定物理配置的正式对照 |
| P6 | Learning-rate sweep | 导师允许后置 | **Deferred** | 前五项之后按需回应优化公平性 |
| P7 | Inductive-bias narrative | 导师允许后置 | **Deferred** | 后续以实验支持机制叙事，不以叙述替代验证 |

每项任务均维护研究目的、实验事实、结果、对应审阅问题、科研结论、证据限制与后续待办；完成结论必须附结果目录及源码/数据 provenance。Pending 和 Deferred 不表示已经有正式补强实验结论；P6–P7 后续仍计划按需完成。

## Priority 1 — Simple baselines

必须补：

1. spline-in-$Q$ interpolation；
2. plain MLP；
3. RK4 / numerical solver vs neural inference timing。

### 目的

回答：

- 神经网络是否真的优于简单函数逼近；
- 当前任务是否有 surrogate 的实际价值；
- 论文更应定位为性能研究，还是 architecture-behavior study。

### 当前状态

- **Completed（2026-10-24）**：Spline 验证、MLP BEST、三模型 Q400 冻结评价及正式计算成本评估均满足本次验收。
- **完成范围**：单训练 seed=27、T1200、固定物理参数的 Q-only 任务；不代表多 seed、全新测试域、FNO2D Q-axis 或其他物理配置已经解决。
- 原始 2026-10-01 状态为“未开始正式补实验”，保留此时间线；本次仅合并已存在的结果，不重新训练或评价。

### P1 实验事实与精度结果

- Track A n2000、split seed=10：1400 train、300 validation；Spline 只使用全部训练 Q，稳定升序并保留原行映射。固定每个原始 λ 节点和 xyz 分量沿 Q 作 Natural Cubic Spline；训练凸包外由最近两个训练节点割线线性延伸，未用验证/Q400 重新设计规则。
- 原始 T1200 网格 λ=0…5.995、step=0.005；统一 raw xyz、float64 归约、历史 ε=1e-12。Global Relative L2 是整体误差范数/整体真值范数；Mean-per-Q 是各轨道相对误差的算术平均，不能混用。
- Spline 的 300 validation Mean-per-Q Relative L2 = **3.366217496832826e-8**。
- MLP：2→576→576→576→576→3，GELU，1,000,515 参数；seed=27、500 epochs，BEST epoch=494。
- FNO：Track A 历史 seed=27、T1200 **FNO1D** BEST epoch=499，modes=32、width=64、depth=4；不是 Plan A/B 的 FNO2D。
- Q400 统一同一 canonical Q identity 和 raw xyz truth；神经模型各自恢复训练时 normalization，不重新拟合；batch=32，最后 batch=16。FNO 全部 1,440,000 个预测标量与历史保存结果逐元素一致。

| Q400 方法 | Mean-per-Q Relative L2 | Global Relative L2 | MSE |
| --- | ---: | ---: | ---: |
| Spline-in-Q | **1.07716379e-8** | 1.91680947e-7 | 8.93513195e-13 |
| MLP1D | **2.85380292e-4** | 3.00764510e-4 | 2.19986613e-6 |
| FNO1D | **2.84387273e-4** | 3.04676591e-4 | 2.25746617e-6 |

**边界效应**：实际训练 Q 凸包为 [1.6021010505252626, 3.0]；Q400 中 399 条在内，只有 Q=1.6007 在外。该外部轨道的 Spline Relative L2 为 4.30736921e-6，贡献总平方误差和约 **99.99999954%**；内部 Mean-per-Q 为 3.22288910e-12。外部结果有限，但边界误差主导总体误差；不能把内插表现推广为宽范围外推能力。

### P1 正式计算成本

设备为 Xeon Platinum 8480+ 与 L40S/HAMI；CPU 固定两个物理核心、CPU/BLAS 线程=2（RK4 实现本身串行）。预热后，单轨道和 batch32 各 30 次、完整 Q400 5 次；GPU 计时前后同步。共 703 条样本全部保留，统计复核 PASS。

下表为 **end-to-end median**：驻内存物理 Q 输入到完整 [1200,3] raw xyz，包含相应归一化、传输和输出转换；不包含磁盘读取、初始化或拟合。单轨道为预先选定的 canonical index 200；Q400 神经模型 batch=32/最后16，Spline 向量化、RK4 串行。

| 方法 / 设备 | 单轨道延迟 ms | 完整 Q400 秒 | Q400 轨道/秒 |
| --- | ---: | ---: | ---: |
| RK4 CPU | 26.123638 | **10.719504** | 37.32 |
| Spline CPU | 0.035117 | **0.007356** | 54378.07 |
| MLP CPU | 11.866200 | 4.662412 | 85.79 |
| FNO CPU | 3.699050 | 0.488811 | 818.31 |
| MLP GPU | 0.264440 | **0.057407** | 6967.76 |
| FNO GPU | 0.970957 | **0.020923** | 19117.91 |

完整 mean/std/p95、model-only、batch32 与原始样本见正式计时目录中的 `timing-results.json`、`timing-raw.csv` 和 `benchmark-report.md`。model-only 含设备上的 raw xyz 还原，与端到端 CPU NumPy 转换并非严格嵌套，不能相减解释为纯传输开销。

### P1 准备、加载和存储成本

| 项目 | Spline | MLP | FNO1D |
| --- | --- | --- | --- |
| 拟合或历史训练 | 完整拟合 median 0.486048 s，训练数组已在内存 | 历史 370.107 s，L40S | 历史 518.086 s，RTX 4090 |
| 暖缓存加载 median | 系数 0.247624 s | CPU 0.037077 s / GPU 0.033770 s | CPU 0.009148 s / GPU 0.009156 s |
| 持久化实物大小 | 展开系数 161,313,730 bytes（153.84 MiB） | BEST 16,172,730 bytes（15.42 MiB） | BEST 4,291,732 bytes（4.09 MiB） |
| 补充说明 | train raw xyz 为 40,320,000 bytes（38.45 MiB） | 纯参数 4,002,060 bytes（3.82 MiB） | 纯参数 4,279,052 bytes（4.08 MiB） |

训练时长来自可靠历史记录，均涉及验证/checkpoint，但具体计时范围和硬件不同，**不能直接排名训练效率**。训练真值生成成本无统一可靠总记录，不视为零成本。暖缓存加载不是公平 cold-start 比较。Spline 可以保留训练节点后重拟合，当前系数存储量不是理论下限；该替代部署策略尚未计时。

GPU allocator 峰值为 MLP 196.12 MiB、FNO 84.91 MiB；它不是 HAMI 物理显存/配额测量。准备成本与 steady-state 推理分开报告，不混入单次延迟。

### P1 科研结论与证据限制

**实验支持**：

1. 当前密集 Q、固定物理参数和原始 T1200 网格上，Spline 精度与稳态延迟均优于本次 MLP/FNO。按 Mean-per-Q，神经误差约为 Spline 的 2.64–2.65 万倍。
2. MLP 与 FNO 精度同量级；Mean-per-Q 略偏 FNO、Global 略偏 MLP，单 seed 不支持稳健排序或显著性结论。
3. 同 CPU 设置下，Q400 历史 RK4/方法延迟比为 Spline 1457.26、MLP 2.30、FNO 21.93。RK4 包含历史诊断及约束检查，未代表编译/向量化优化求解器。
4. CPU RK4 与 GPU MLP/FNO 的部署比为 186.73/512.34，**属于跨硬件对比，不是纯算法加速**。神经 GPU 内部 MLP 单轨道更快，FNO Q400 吞吐量约为 MLP 的 2.74 倍；Spline CPU 在本次测量中仍最快。
5. 神经参数化较本次展开 Spline 系数紧凑，但当前 Q-only 场景没有充分证据证明神经网络相对 Spline 具有全面实用 surrogate 优势。

**合理解释与论文影响**：密集单参数轨道族适合沿 Q 插值；单凭“替代求解器”不足以支持 FNO 必要性。论文应收窄普遍精度/速度优越的叙述，以简单基线为参照，更强调受控 architecture / discretization / extrapolation behavior；这不等于已证明某个 inductive-bias 机制。

**尚未验证/必须保留的限制**：

- Q400 是历史已用于研究的既定 evaluation field，**不是全新 untouched held-out test**；RK4 truth 是数值参考，不是解析精确解。
- 单 seed、单共享容器；完整 Q400 仅 5 次重复，p95 仅描述性。Spline Q400 E2E CV 约 12.85%，未剔除慢值、未作显著性声明或跨容器计时复现。
- 稀疏 Q、多参数、另一物理配置、未见 λ/长度/分辨率、跨 seed 稳健性和对优化求解器的优势，均不能由 P1 直接推出。
- 正式计时退出码 0；7 条 `cuCtxSetCurrent111 failed res=4` 均在结果保存及 Python atexit 之后，属于仍未解决的 HAMI 退出清理警告，不能改写为“无异常”。
- 首轮计时包装层的 Q 形状适配失败已留档，正式轮次仅修正输入形状适配；旧轮次未混入正式统计，冻结科学规则未改变。

### P1 证据目录与 provenance

以下均为实际 NFS 绝对路径；远端不是完整 Git checkout，按实际源码快照 SHA-256 溯源，不伪造 Git revision。

- Spline validation：`/workspace/shanjinshuo/outputs/fno_kerr/priority1_spline_t1200_val_20261011T0341Z/`
- MLP training：`/workspace/shanjinshuo/outputs/fno_kerr/priority1_mlp_t1200_seed27_20261024T014339Z/`
- Q400 comparison：`/workspace/shanjinshuo/outputs/fno_kerr/priority1_q400_comparison_t1200/`
- Formal timing：`/workspace/shanjinshuo/outputs/fno_kerr/priority1_timing_t1200_20261024T0643Z/`
- Preserved failed timing attempt：`/workspace/shanjinshuo/outputs/fno_kerr/priority1_timing_t1200_20261024T0640Z/`
- FNO BEST：`/workspace/shanjinshuo/projects/fno_kerr/kerr_project/outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200/best_model.pt`

| 证据 | SHA-256 |
| --- | --- |
| timing-protocol.json | `ffcef5fe5427cc26191693f0d7091bee15a380c9ce879f46733378a89d873aaf` |
| timing-results.json | `1c3969cbf0075cc521ffd35e5b2a6fbf1dafd02c8ebd0f18f13c798f538781c6` |
| timing-raw.csv | `37b977e78508821c325e03dba8cfeefa8ee0ceef20f0e131ca3aa26919cbfb29` |
| benchmark_p1_t1200.py | `2a257b89e120c2b78208576584aada7dc2c8e13b8d1faea1e202cab6b0bc6384` |
| timing manifest.json | `6d4d2c1aeff2b672151a24e1fe91114350062d37c1d11c5fdee1769c97257b50` |
| Q400 manifest.json | `889a415dead43cd48af9763818b470f6c2786d25f8a3b84727c9b11fddf10da5` |
| MLP BEST | `91503281bd80e5106ab7fb051b5956c88e269d49b20b5749681b66824f78b4c0` |
| FNO BEST | `11ccbebf1d0fd68c57d0895685adeb7d1386da723242ee8271b18286ebbb7ff1` |

完整数据身份、输入/源码 hash 与运行设置见各目录的冻结协议和 manifest；正式计时 `verification.json` 为 PASS，`completion.json` 为 COMPLETED_WITH_EXIT_WARNING。先前“待合并增补”已在本次核对后整合于本节，原增补作为历史交接证据保留，不作为第二份主记录。

**下一步决定**：进入 P2 的协议冻结与预算审查；仅在另行明确批准后启动新训练，不用 Q400 结果挑 seed 或调整模型。

---

## Priority 2 — Multi-seed statistics

需要对：

- Track A；
- 关键 FNO2D cells

补 3–5 个 seeds，并报告 mean ± std。

### 注意

导师明确把这一项排在 learning-rate sweep 之前。

因此当前不应擅自把 LR sweep 提到它前面。

### 当前状态

- **Pending；导师要求必须完成。**
- **目的/实验事实**：检验接近模型排序及跨分辨率现象的 seed 稳健性；Track A 已有 seed=27，本轮补齐的 MLP 也仅 seed=27，其余 seeds 待补。
- **结果与结论/限制**：尚无正式 3–5 seeds 的 mean ± std；P1 不能代替多 seed 证据，不能据此声称接近模型排序稳定。
- **后续待办**：冻结共同 seed 列表（明确包含已有 27）、Track A 与关键 FNO2D cells 的最小矩阵、固定 split/normalization/BEST/指标和预算；在新结果出现前确定汇总规则，逐 seed 留痕。原生 T1200 可作启动入口，但不能把 P2 整体范围缩成只测 MLP/FNO 两个模型。FNO2D 关键 cells 与 P4 路线衔接，须明确批准后执行。

---

## Priority 3 — Fresh, frozen Plan A test

需要建立真正没有参与过诊断和 redesign 的 test。

核心要求：

- 新的 $Q$ values；
- 和 / 或此前未使用的新 domain length，例如 T3000；
- 模型、预处理、redesign 全部先冻结；
- test 只做一次正式 evaluation；
- 不能再利用该 test 继续设计 R4、R5 等方案。

### 关键思想

重点不是单纯“把长度再延长一点”，而是：

> **fresh + held-out + frozen evaluation**

### 当前状态

- **Pending；导师要求必须完成。**
- **目的/实验事实**：建立未参与诊断和 redesign 的冻结测试；已有 T1800/T2400 和 Q400 的历史使用事实不变。
- **结果与结论/限制**：无新的 fresh-test 结果；P1 的 Q400 不能作为本项完成证据。
- **后续待办**：先冻结新 Q/域长度、数据 provenance、模型及预处理和一次性评价规则，再经批准生成/评价；T3000 仍只是候选，不视为已选定。

---

## Priority 4 — 解决 FNO2D 的 Q-axis 问题

两条可接受路线：

### 路线 A：保留 FNO2D

则需要充分解释和论证：

- 为什么 $Q$ 轴可以这样进行 spectral operation；
- trajectory coupling 如何理解；
- train / test Q-axis spacing change 是否会污染结论；
- 非周期 $Q$ 轴上的 FFT 假设如何处理。

### 路线 B：改用 per-trajectory FNO1D

把 Plan A / Plan B 的关键实验迁移到 FNO1D，使正文形成一致的 model family。

### 当前工作假设

暂时更倾向：

- Track A / FNO1D 作为正文核心候选；
- 原 FNO2D 结果保留为 historical diagnostic evidence；
- 是否重跑全部 Plan A/B 暂不决定；
- 优先只考虑复现最关键 cells，而不是整套推倒重做。

> 这是当前讨论形成的工作假设，不是导师原话。

### 当前状态

- **Pending；导师要求必须完成。**
- **目的/实验事实**：澄清 Q-axis coupling 与 spacing/非周期假设对结论的影响；已有 scrambling 证明 trajectory coupling，P1 使用 FNO1D，不解决历史 FNO2D 的结构问题。
- **结果与结论/限制**：尚无本项正式补强验收；不能由 FNO1D 本次运行成功反推 FNO2D 问题已排除。
- **后续待办**：明确选择路线 A 或 B、冻结关键 cells 和身份/spacing 对照，保留 Plan A/B 历史结果；与 P2 的 FNO2D seed 矩阵协调，不自行重跑全部实验。

---

## Priority 5 — 增加至少一个新的物理配置

Claude 原始要求是：

> test at least one more physical configuration

不等价于一次性把 $a,E,L_z$ 全部扩成模型输入。

当前更保守的设计是：

- 保持原来的 Q-family 任务结构；
- 只改变一个物理配置；
- 优先考虑改变 spin $a$；
- 在新的固定 $a=a_2$ 下重新生成一套 Q-family；
- 重跑最关键的 resolution / extrapolation comparison。

### 目的

回答：

> 已观察到的 architecture / resolution behavior 是否只发生在当前单一 Kerr setting？

### 当前已有资产

项目里已有 $(Q,a)\rightarrow xyz(\lambda)$ 的数据生成和 FNO/CNN 训练代码。

但现有 QA 数据集采用规则 $(Q,a)$ 网格后随机打乱 train / val / test，因此主要反映二维参数域内插值，不直接等价于“跨物理配置 held-out robustness”。

### 当前状态

- **Pending；导师要求必须完成。**
- **实验事实**：有代码和历史 QA 实验资产；正式 reviewer-response 设计尚未执行。
- **结果与结论/限制**：没有另一固定物理配置的正式补强结果；随机划分 QA 插值不能证明跨物理配置稳健性。
- **后续待办**：冻结至少一个新配置（改变 spin 为候选）、数据划分与同口径基线/关键 resolution 和 extrapolation cells；先核验预算和授权再执行。

---

# 10. 暂缓项

导师确认可以暂时后置：

## Priority 6 — Learning-rate sweep

对各模型做小规模 LR sweep，以回应 optimization fairness。

当前不抢前五项优先级。

- **状态**：Deferred，导师允许后置，后续仍计划按需完成。
- **目的/事实与结果**：回应 optimization fairness；本阶段没有新的 LR sweep 结果，P1 沿用冻结训练配置。
- **结论与限制**：近似参数量与统一评价不等于已证明优化公平。
- **后续待办**：前五项之后按需冻结候选 LR、仅用 train/validation 选择的规则与预算，不在 Q400 上调参。

---

## Priority 7 — 按 inductive bias 重构论文叙事

最终论文应逐步转向：

> architecture prediction → experiment → quantitative agreement / surprise

并补充：

- FNO discretization literature；
- representation-equivalent / alias-free neural operator literature；
- Kerr geodesic machine-learning literature。

当前作为论文写作与理论整理任务保留，但不优先消耗训练资源。

- **状态**：Deferred，导师允许后置，后续仍计划按需完成。
- **目的/事实与结果**：将架构预期与受控测量相联系；P1 已要求收窄神经 surrogate 的性能叙述，但机制性综述和论证尚未完成。
- **结论与限制**：P1 不证明新的 inductive-bias 因果机制，工作版论文主问题仍不是导师最终定稿。
- **后续待办**：在 P2–P5 证据基础上补文献、预期与反例，逐条区分已验证事实、合理解释与待验证假设。

---

# 11. 当前论文主问题的工作版本

> **以下是讨论形成的工作版本，不是导师最终定稿。**

不再优先强调：

> “FNO 能否高效求 Kerr 测地线？”

更适合尝试围绕：

> **在具有明确物理结构且可以严格控制采样网格的 Kerr 轨道族上，不同神经架构的 inductive bias 如何影响 discretization change 与 domain extrapolation？**

当前已有实验可以组织成两个正交轴。

## Axis A — 固定物理域，改变离散化

保持物理区间不变，例如：

$$
\lambda\in[0,6],
$$

改变：

$$
\Delta\lambda,\qquad T.
$$

对应：

- Plan B；
- Track A；
- resolution transfer；
- discretization generalization。

## Axis B — 固定步长，扩展物理域

保持：

$$
\Delta\lambda=0.005,
$$

但改变：

$$
[0,6]\rightarrow[0,9]\rightarrow[0,12]\rightarrow\cdots
$$

对应：

- Plan A；
- domain extrapolation。

这两个轴必须在论文中严格区分。

---

# 12. 现有实验资产的当前地位

暂时冻结以下已有结果，不因 reviewer-response 阶段而直接覆盖：

- Plan A R0–R3；
- M1–M4；
- Plan B 已有结果；
- Track A seed = 27；
- Track A 现有 frozen evaluation matrix；
- FNO2D Q-axis scrambling 等诊断；
- QA 双参数历史实验与代码。

它们属于：

> **historical evidence / diagnostic assets**

后续是否进入正文，由前五项补强实验完成后的结果决定。

---

# 13. 当前明确不做的事情

在前五项完成前，原则上不主动扩张为：

- 同时把 $Q,a,E,L_z$ 全部作为高维输入；
- 全参数空间大规模扫点；
- 把所有旧模型和所有旧 redesign 全部从头重训；
- 为了“多做实验”而继续增加新的架构；
- 把 T3000、T3600、T4800 等连续长度都作为新的调参集合；
- 因为某个 test 结果不理想而继续在 fresh test 上迭代 redesign。

---

# 14. 当前执行顺序

按导师确认的硬优先级：

1. **Simple baselines and timing — Completed（限 P1 已定义范围）**
2. **Multi-seed statistics — Pending，当前下一入口**
3. **Fresh frozen Plan A test — Pending**
4. **Resolve FNO2D Q-axis issue — Pending**
5. **One additional physical configuration — Pending**
6. Learning-rate sweep — Deferred（导师允许后置）
7. Inductive-bias narrative / literature positioning — Deferred（导师允许后置）

如果某一步的结果会改变后续论文定位，可以在完成该步后重新评估后续具体实现，但不得随意改变导师规定的前五项必须完成这一事实。

---

# 15. 后续每次更新本文件时应记录什么

每完成一项实验，只更新以下四类信息；同时保留研究目的、证据限制和明确后续待办，并附实验结果目录与源码/数据 provenance。不能将未完成任务提前标记 Completed：

## 15.1 实验事实

记录：

- 数据定义；
- train / val / test；
- 模型；
- seed；
- metric；
- 最终数值；
- 是否 frozen evaluation。

不要把猜测写成事实。

## 15.2 对 reviewer concern 的回答

明确该实验对应解决哪一个问题。

## 15.3 论文结论是否改变

例如：

- spline 是否击败 neural models；
- multi-seed 后模型排序是否稳定；
- fresh Plan A 是否仍失败；
- FNO1D 是否复现原 FNO2D 的关键趋势；
- 新物理配置是否复现同样的 resolution behavior。

## 15.4 下一步决策

只记录由当前结果直接支持的下一步。

避免自由发散新的实验分支。

---

# 16. 当前最近一步

2026-10-01 的最近一步为 Priority 1；现已完成其单 seed、T1200、Q-only 范围的验收并合并到第 9 节，历史资产保持冻结。

> **Priority 2 — Multi-seed statistics：先准备协议，不自动启动。**

1. 盘点已有 seed=27 的可比 cells，拟定 Track A 与关键 FNO2D cells 的 3–5 seeds 矩阵；先明确是否包含已有 27，不能按新结果挑选 seed。
2. 固定数据 split seed=10、训练集拟合的 normalization、目标变换、BEST 选择、训练预算和评价规则；在结果出现前确定逐 seed 值、mean ± std 的汇总方法。
3. 原生 T1200 可作为最小启动单元；需与 P4 的 FNO2D 路线协调，不以一个小单元替代 P2 全部验收。Spline 是确定性基线，不伪造训练 seed 重复。
4. 继续明确 Q400 的历史 evaluation field 身份；P3 的全新冻结测试单独定义，不借 P2 调参或 redesign。
5. 经用户批准计算资源、训练矩阵和时间预算后才启动；本次文档更新没有执行 P2、任何新训练或评价。


### 2026-11-07｜P2 执行与证据增补

Track A 的 36 个训练单元和 144 个逐种子评价已通过核验。FNO2D 的 R0/T1200、R0/T2399、R3-B1/T1200 历史 BEST 已按 SHA-256 恢复；用现存源码与登记数据重新做冻结前向，分别有 36、12、24 个主/辅助指标与历史值通过预设数值一致性检查。这建立了本次复现的 checkpoint/source/data 关联，不反推旧进程的文件字节。本批次登记 R0/T1200 的 seeds 41、73，训练或评价完成以各自实际产物为准；尚不标记 FNO2D 或 P2 Completed。两组结果的科研解释留待统一分析。

证据：同一项目下 `outputs/p2_stage1_20261024T0948Z/resume_20261105T1430Z/queue-status.json`，本批次 `outputs/p2_fno2d_replay_20261107/{frozen.json,launch.json,queue.json}` 与各 `*_27_verification.json`。


### 2026-11-07｜P2 新增运行记录（暂不作科研解释）

R0/T1200 的 seeds 41、73 均已完成训练、Plan A 三长度及 Plan B 四网格冻结评价；证据见 outputs/p2_fno2d_replay_20261107/queue.json 及逐 seed 产物。剩余 R0/T2399、R3-B1/T1200 的 seeds 41、73 已登记到 outputs/p2_fno2d_remaining_20261107/，单卡按剩余时间准入，截止 14:33 UTC+8。已登记不等于已完成，最终以各单元 BEST、训练历史、冻结评价核验为准；P2 仍为 In progress，Stage 2 不启动。


### 2026-11-21 | P2 C41/C73 authorized continuation

B41/B73 passed checkpoint/hash, 500-epoch history, frozen-contract and existing four-grid evaluation-record verification; no retraining. C41 failure was traced to float32-rounded anchor values in historical model_config versus exact float64 checkpoint buffers. A comparison-only canonicalization passed a CPU preflight against all other frozen fields. C41/C73 were launched in outputs/p2_c_completion_20261121/ with the frozen protocol and cutoff 2026-11-21T08:30:00Z, based on user-reported remaining container time. See b_acceptance.json, launch_protocol.json, preflight_config.json and launch.json. These are running units, not accepted completed results. P2 remains In progress; Stage 2 is not authorized. Scientific interpretation awaits complete results.


### 2026-11-21 | P2 C41/C73 acceptance and three-seed Plan A summary

C41/C73 completed 500 epochs; BEST epochs 78/76 match minimum validation selection scores. Strict CPU loading, finite states, frozen source/data hashes, BEST-to-evaluation records and per-Q CSV-to-summary agreement passed. Evidence: outputs/p2_c_completion_20261121/acceptance_and_statistics.json. Each evaluation log retains six HAMI cuCtxSetCurrent111 errors at shutdown; completion is accepted with runtime warnings, not error-free execution. No new forward evaluation was performed during acceptance.

The three-seed full-trajectory mean-per-Q Relative L2 for R0 is 0.005363 +/- 0.00005685 (T1200), 1.78652 +/- 0.07416 (T1800), 1.87983 +/- 0.15824 (T2400); R3-B1 is 0.271959 +/- 0.032698, 1.036404 +/- 0.009760, 1.140428 +/- 0.005163 respectively. R3-B1 has lower long-domain error but worse native accuracy in all three seeds; long-domain prediction remains inaccurate. This is a repeated descriptive tradeoff, not proof of a causal mechanism or successful extrapolation. Fixed historical Q400 and three seeds limit scope.

All registered Stage 1 seed/config execution units are now complete (Track A 36 and FNO2D 9), but whole-P2 final scientific acceptance and five-seed necessity remain separate decisions. P2 remains In progress pending integrated review; Stage 2 is not authorized. Correction to prior local handoff wording: actual frozen training lengths are [600,800,1000,1200], validation lengths [700,900,1100,1200]; actual training preserved the correct protocol.
