# Kerr 研究报告阅读入口

整理日期：2026 年 11 月 22 日。报告覆盖最初可追溯探索、服务器迁移及截至11月21日的新增实验。所有文件均为Markdown。

November 24 addition: the [mentor code review and completeness guide](../docs/MENTOR_CODE_REVIEW_GUIDE.md) maps research stages to their code and adds previously unarchived migration, numerical-diagnostic and environment scripts. It also explains the October 11 batch1/batch32 repeatability diagnostic and the limits of its conclusions.

## 建议阅读顺序

1. [研究历程与实验全景报告](01_研究历程与实验全景报告.md)：中文正文，包含历史时间线、研究动机、模型与数据定义、成功和失败实验、最新三种子结果、结论与当前状态。
2. [实验总目录与逐运行配置](02_实验总目录与逐运行配置.md)：查某个具体实验用什么数据、模型、归一化、训练预算、BEST、运行环境和路径；包含历史探索逐项卡片。
3. [完整结果矩阵与统计](03_完整结果矩阵与统计.md)：查144个Track A评价单元、FNO2D的24个Plan B单元与18次Plan A前向的分区结果，以及重算的三seed统计。
4. [证据索引与迁移说明](04_证据索引与迁移说明.md)：查原服务器路径、文件哈希、迁移映射和证据边界。

## 如何发送给导师

主报告可单独阅读；建议将整个reports目录一起发送，保证配套Markdown链接可用。正文负责解释研究，附录负责查参数和核对数字。核心源码现已收录于同一仓库，参见[源码快照与阅读指南](../docs/SERVER_SOURCE_SNAPSHOT_20261122.md)；逐项讲解可在阅读报告后继续。

## 结果口径

三seed为27、41、73；±是样本标准差，不是置信区间。历史seed27有一部分依赖迁移保留的结果汇总，新增seed41/73有现存独立评价文件。报告不把路径迁移当作实验无效，也不把未找到的原始文件补成已核验。

本轮只读采集、读取历史BEST配置并整理已有结果，没有重新训练、生成新数据或执行新模型前向；没有修改服务器科研文件。
