# P12 Problem Definition: Guarded Candidate Admission

## 冻结事实与当前边界

本文件在任何 P11 production bridge 或 P12 实现之前冻结问题。它不把历史文档、队友 patch
或公开集结果当作新的授权，也不把本地派生集称作主办方私有集代理。

当前正式服务路径仍是 R08：`coverage/off/fast`。它使用 SQLite FTS5/BM25 的 Broad-120
与 Strict-80 两路召回、weighted RRF，以及 query-term coverage cascade。正式公开结果是
HR@10 `0.945`、MRR `0.606175`、MTTC `3.335`、TechnicalScore `0.807652`，运行时网络和
token 均为零。

P11 `p11.top10-linear.v3` 已在三份新的 product-disjoint split 上正式得到
`promote_p11_r01`，但尚未 production served，也从未运行 released public。它只允许重排
R08 当前 Top-10，严格保持 Top-10 集合和 rank 11 后的 tail，因此能够改善 MRR，无法改变
HR。任何 sidecar、身份、边界或运行时异常都必须完整回退到 R08。

最终 route 的 best-eligible-turn recall 为 R@10 `0.945`、R@20 `0.970`、R@50 `0.995`。
11 个公开 miss 中有 10 个曾位于 Final rank 11–50，只有 1 个在 Final Top-50 外。因此当前
主要瓶颈不是“再找更多候选”，而是如何安全跨越 Top-10 边界。

## 1. 简化问题：分离信号与噪声

### 有决策价值的信号

- Final Top-50 已覆盖 199/200，而 Final Top-10 覆盖 189/200。
- 10 个 miss 在可救的 rank 11–50 区间；它们不是单纯的深度召回失败。
- P11 在 primary、uniform-tail、confirmation 都保持 HR/MTTC，并提高整体 MRR；这证明
  IDF、field coverage、hard clause、subtype、正负约束等可见证据具有跨 fresh split 的
  排序价值。
- P11 不能改变 Top-10 成员，所以 HR 上限没有变化。
- P7 本地 BGE-small embedding dense shadow 对 sparse union 的新增 rescue 为 0，同时
  wall 与 RSS gate 失败；当前没有证据支持直接换 embedding 模型。
- 队友 v1 的 canonical-public 改善提示 category rescue、typed budget 和 information gain
  值得拆开验证，但其 phrase/product-disjoint 严重回归，说明 exact-template parser、公开集
  派生 prior 与全局 popularity 不能进入主路径。

### 当前应排除的噪声

- 没有截断诊断就扩大 Broad/Strict 深度。
- 仅更换 embedding checkpoint，或把 dense 本身当作创新结论。
- 在已消费 P11 结果上继续调 Top-10 权重、阈值、词表或 near-tie 定义。
- 围绕 11 个公开 miss、sample ID、ASIN 或固定 simulator 句式写规则。
- 让 raw/global popularity 覆盖当前用户显式要求。
- 同时改 parser、召回、admission、P11 和 clarification，导致无法归因。

核心问题压缩为：

> 如何只使用当前可见对话和 catalog-only 证据，以极低误伤率把 rank 11–50 中更可信的
> challenger 提升进 Top-10，同时保护已有 hit、override 语义、资源边界和完整 R08 fallback？

## 2. 类比：带噪信道中的列表解码

把系统看作一个带噪信道的列表解码器：

| 系统元素 | 类比角色 | 对设计的约束 |
| --- | --- | --- |
| 用户自然语言 | 含噪信号 | 不依赖 exact simulator 模板 |
| Broad / Strict / RRF | 候选解码列表 | 保留强 sparse control，不盲目扩大深度 |
| Typed QueryPlan | 已解析的校验位 | 每条约束带 polarity、hardness、source turn、version |
| catalog sidecar | 只读码本证据 | label-free、离线、不可由 target/outcome 构造 |
| hard clause / subtype / budget / positive-negative evidence | 独立校验信息 | 未知是 unknown，不是 violation |
| guarded admission | 纠错步骤 | 多个独立证据一致才允许跨 Top-10 边界 |
| P11 | 已选集合内部排序 | admission 后只做集合内精排 |
| clarification | 主动请求新校验位 | 只有预期收益超过一轮 MTTC 成本才提问 |

这个类比给出的实现原则不是“最高分者必进”，而是 challenger 必须同时通过可审计的
资格、证据、冲突、margin 和 displacement guards。证据不足时，正确行为是 no-op。

## 3. 重新提问：把分数愿望改为可证伪问题

不问“怎样把公开 HR 强行做到 99%”，而问：

> 在 Final Top-50 recall 已经是 99.5% 的条件下，哪一种冻结 admission rule 能在两份新的
> product-disjoint split 上产生稳定 miss-to-hit、零 hit-to-miss，并满足资源与重复性 gate？

不问“怎样让首轮命中达到 80%”，而问：

> 当前轮披露的证据是否足以区分 Top-10 与 challenger；如果不足，继续推荐与询问一个可执行
> 属性相比，哪一个动作的预期官方贡献更高？

不问“是否应该用 embedding”，而问：

> 现有 sparse Top-50 没有覆盖 target 的频率是否足以证明语义召回缺口；新的 dense route 能否
> 在新的 target-disjoint 数据上产生 sparse-only 之外的净 rescue，并同时通过 wall/RSS/license/
> offline gate？

按照现有证据，第三个问题当前答案是否定的，因此 embedding 不进入 P12 第一版。

## 4. 结构分析：单责、可关闭、可回退

```text
visible dialogue + safe aggregate profile
  -> Typed QueryPlan
  -> existing R08 Candidate Recall
  -> Guarded Top50-to-Top10 Candidate Admission
  -> frozen P11 Within-set Ranking
  -> Clarification Policy (Phase C only)
  -> contract-valid Top10 Response
```

### Typed QueryPlan

QueryPlan 是 SlotLedger 当前活动记录的不可变投影，而不是另一套竞争状态。至少包含：

- goal/version 与 source turn；
- category/subtype；
- positive/negative polarity；
- hard/soft hardness；
- budget interval 与 missing/known 语义；
- exact visible terms 与 latest hard clause；
- override/no-preference 后已经 superseded/deleted 的记录不得继续执行。

### Candidate Recall

保持 R08 Broad-120、Strict-80、weighted RRF 和 coverage 不变。P12 第一版只读取冻结的
Final Top-50，不增加新的召回模型，不改变 question policy。

### Guarded Candidate Admission

第一版最多从 rank 11–50 接纳一个 challenger，并最多替换一个 incumbent。候选必须：

1. 与 QueryPlan 当前 goal/version 一致；
2. 无显式 hard negative 或 subtype 冲突；
3. 对 latest hard clause 与可靠 typed constraints 有足够覆盖；
4. 相对被替换项具有预注册的、可解释的严格证据优势；
5. 不由全局 popularity 单独触发；
6. 输出仍是唯一、合法 catalog ID，异常时逐字节回退到 P11/R08 baseline。

### P11 Within-set Ranking

P11 的 scorer、weights、registry、semantics 和 sidecar identity 不得因 P12 结果改变。P12 决定
Top-10 成员之后，P11 只在最终成员集合内排序；组合顺序必须在 source freeze 前固定并做
ablation，不能事后挑选。

### Clarification

Phase B 期间固定 `fast`。只有 P12 selection 与 confirmation 都通过，Phase C 才可单独测试
candidate-aware clarification，避免把 HR、MRR 与 MTTC 变化混在一个实验里。

## 5. 反向推导：由目标计算必要条件

### 从 HR@10 99% 反推

- 200 条中至少要命中 198 条。
- 当前命中 189 条，所以需要至少 9 个净 miss-to-hit。
- Final Top-50 一共只覆盖 199 条，因此公开集理论上最多有 10 个边界 rescue。
- 若已有 hit 发生 1 个 hit-to-miss，则需要救回 10 个 miss 才达到 198；容错几乎为零。
- 若 rank 11–50 永远不能进入 Top-10，则 P11 无论如何调权重都无法提高 HR。

结论：99% 是用于理解边界的数学压力测试，不是 P12 的本地硬承诺。真正 promotion gate 必须
是 fresh selection 与 unopened confirmation 上的净收益、零 hit-to-miss、场景非回归和资源合格。

### 从 overall turn-1 80% 反推

- 200 条需要 160 个首轮命中。
- 30 个 Intent Override 在新意图披露前不能命中，因此最多只有 170 条首轮 eligible。
- 这要求 eligible 首轮命中率至少 `160/170 = 94.1%`。
- Browsing 与 Boundary 首轮通常只给粗类别，目标商品在信息论上并不总可辨识。

结论：overall turn-1 80% 不是合理的通用 hard gate。应报告 eligible-turn recall、first-hit turn
分布和 ask-vs-return 决策，而不能靠提前猜测 target 来换取表面 MTTC。

## 6. 归纳与推广：只推广被多份证据支持的机制

### 可以归纳的结论

- P11 的 fresh primary/confirmation 正向 CI 支持“结构化可见证据能改善集合内顺序”。
- sparse Final Top-50 的 99.5% recall 支持优先研究 Top-10 boundary，而非泛化声称“召回已解决”。
- 队友 v1 支持 bounded category rescue、typed budget 与 information gain 作为独立假设。
- 队友 v1 的 phrase/product-disjoint 失败反证 exact-template 和 public-derived priors 的泛化性。
- P7 dense 的零新增 rescue 与资源失败反证“换 embedding 自然会更好”。

### 允许的推广

- 将 P11 已冻结的 hard clause、subtype、positive/negative、observed/inferred/unknown 语义扩展为
  challenger 与 incumbent 的比较特征。
- 使用 catalog-only sidecar 将运行复杂度限制在 Top-50，而不是扫描 50,000 商品或完整类别。
- 使用新的、与 public 和 P1/P5–P11 全部 2,720 个 opened targets 隔离的 selection 与
  confirmation 验证。

### 禁止的推广

- 不从 released public 的 200 条直接推断 organizer-private 800 条成绩。
- 不从一份派生集推断真实用户自然语言分布。
- 不把 raw popularity、target ID、sample ID、scenario label、evaluator state 或历史 outcome
  作为 Agent feature。
- 不把 P7 的一个失败模型解释成“所有 embedding 永远无效”；它只说明当前没有激活依据。

## 可证伪假设与执行门槛

P12 的主假设是：

> 一个只允许单 challenger、要求多证据一致、保留严格冲突与 fallback 的 Top50→Top10
> admission 层，能相对 frozen integrated P11 在两份 fresh product-disjoint split 上提高
> TechnicalScore 和 HR，同时保持零 hit-to-miss、MRR/MTTC/场景 HR 非回归及资源合格。

该假设若在 selection 失败，就冻结为 Reject/Retain，不调整同一语料的权重或门槛；若 selection
通过，candidate、sidecar、registry、weights、source hashes 和 commit 必须先冻结，再由 runner
首次解析 confirmation。只有 confirmation 也通过，才可考虑 Phase C 与一次最终 released-public
比较。Public 结果之后不允许改代码或阈值重跑。

## 当前完成与未完成

已完成：官方资产与远端身份、R08 served checkpoint、P11 正式 promote 决策、Phase 0 baseline
manifest、pre-integration checkpoint 与隔离任务分支。

尚未完成：P11 production bridge、production fallback/resource/Observer gate、P12 QueryPlan/
sidecar/admission、fresh P12 selection/confirmation、可选 Phase C、最终 public 事件，以及 submission
default-branch/report/demo 闭环。

下一步只能先完成 Phase A。Phase A 未通过时保留 R08，不进入 Phase B。
