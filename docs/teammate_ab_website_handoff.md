# 队友交接：Fusion A/B 与 Agent Workbench

## 一句话说明

这个网站是本地、只绑定 loopback 的 **Agent Layer Observer**：首页的 Fusion Studio 用冻结证据解释队友 T0、Fusion A、Fusion B 的计算路径与结果；原有页面继续提供真实会话 trace、50k catalog/FTS5 浏览、受限实验任务、手动对话 Lab 和项目文档。网站是演示与开发工具，不会进入官方 Agent 的检索、排序或评分路径。

## 如何启动

```powershell
conda activate tiktok
cd D:\tiktok\techjam-v3-evaluator-parallel
python -m observer.launcher
```

Windows 也可双击 `Start Observer.vbs`。本地必须已有官方冻结 `data/catalog.jsonl`；网站不会下载或修改 catalog。启动后浏览器打开的第一页是 **Fusion A/B**。

## 网站现在能看什么

- **Fusion A/B**：切换 T0、A、B，查看每层职责、源码边界、问询策略、Public200/2k 指标与风险。
- **Compute Walkthrough**：选择 intent page 1、page 2、page 3+ 或 override，再逐步播放该轮的状态、召回、排序、路由、去重和问询过程。这是冻结架构说明，不会触发 evaluator。
- **会话诊断**：运行当前 `starter.Agent` 的真实逐轮 trace，并把 Agent 实际事件与 post-hoc target 诊断明确分开。
- **商品与索引**：浏览 50,000 商品及本机 FTS5 检索结果。
- **运行与实验**：仅允许仓库预定义动作；不能执行任意 shell。此次更新没有点击或运行任何 evaluator。
- **交互 Lab / 资料库**：目标盲手动对话，以及代码、规则、结果和本交接文档的只读查看。

首页数据来自 `docs/teammate_ab_website.json`，原始严格 A/B 证据仍在 `docs/deadline_v3_4_strict_fusion_ab.json`。网页不自行计算或改写指标。

## 三个版本的架构

### T0：队友 V1

```text
visible profile/message
  → intent/category/constraint session state
  → field-weighted FTS5 Top1000 + category union
  → ProductScorer
     (lexical/category/constraint/department/budget/rating/popularity)
  → exact-category boundary guard
  → exclude shown products
  → Top10 + candidate-aware specific attribute question
```

T0 的优点是第一页强、候选池宽、状态与 no-repeat 简单可靠；问询只使用 evaluator 能回答的具体属性，不使用 `other`。

### A：Fusion Core（严格禁止 `ask_attribute=other`）

A 同时运行 T0 与 v2.12 rank expert，但把风险放到较晚页面：

1. 每个 intent version 的前两页逐响应返回 T0，并记录 served ledger；
2. v2.12 expert 在 shadow 中计算 R08/P11 + 冻结 fold-safe small-ranker 的 final order；
3. 第三页以后从该 order 取 unseen candidates，不足时用 unseen T0 补齐；
4. visible intent override 清空页码与 served ledger，新 intent 重新得到两页 T0 grace；
5. shadow question 被取消，真实问题仍来自 T0；若 A 出现 `other` 则 fail closed；
6. 任何 malformed/duplicate/empty expert 输出都完整回退 T0。

### B：A + bounded evaluator-aligned `other`

B 从 A 的冻结提交派生，**检索、排序、两页 grace、tail、served ledger 和异常回退均不变**。唯一算法差异是问询 lifecycle：

- 消费上一轮 `other` 的 informative / exhausted / boundary reply；
- 每个 visible intent version 通常最多询问两次，turn 10 不问；
- override 重置该问询预算；
- 当 B 替换 A 尚未真正展示给用户的 specific question 时，会先撤销 A 的 question bookkeeping；
- adapter 异常时逐响应回退 A。

## 已冻结结果（此次没有重跑）

| Benchmark | Version | HR@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Public200 | Teammate T0 | 0.9950 | 0.703766 | 2.1100 | 0.886430 |
| Public200 | Fusion A | 0.9950 | 0.671333 | 2.1500 | 0.875900 |
| Public200 | **Fusion B** | **0.9950** | **0.706280** | **1.8550** | **0.892284** |
| local 2k OOF | Fusion A | 0.9905 | 0.609388 | 2.4145 | 0.849776 |
| local 2k OOF | **Fusion B** | **0.9915** | **0.626409** | **2.0285** | **0.863103** |
| local 2k OOF | v2.12 incumbent | 0.9910 | **0.695795** | 2.8690 | **0.866858** |

Public200 上，A 与 T0 的 HR 相同，但 MRR、MTTC、TechnicalScore 都较差，因此 **A 没有超过队友 T0**。B 相对 A 保持 HR，MRR `+0.034947`、MTTC `-0.295`、Score `+0.016384`；B 也比 T0 的 Public Score 高 `0.005854`。

本地 2k OOF 上，B 相对 A 为 `4 miss→hit / 2 hit→miss / net +2`；但它不是零伤害：fold 0 HR 从 `1.0000` 降到 `0.9975`，fold 0 MRR 下降，intent-override 少 2 个 hit。B 比 v2.12 多 1 个 aggregate hit、MTTC 快 `0.8405`，但 MRR 低 `0.069386`、Score 低 `0.003755`。因此综合分安全默认仍是 v2.12；只有明确优先 aggregate HR/MTTC 并接受 fold/override 风险时才选择 B。

`2k OOF` 是本地 `train_explore` out-of-fold benchmark，不是官方 private 800，也不能把它写成 private 成绩。T0 没有直接、可比较的完整 2k artifact，因此表格明确显示为 unavailable，而不是推测数值。Public 与 2k A/B 都记录为 exact-repeat。

## 官方边界与这次 Git 选择

- 官方要求仍是 `reset/respond`、最多 10 轮、只评分前 10 个合法唯一 `parent_asin`、冻结 50k catalog、离线可复现。
- 网站不会向 Agent 注入 target、sample label、scenario 或 evaluator state；label 只用于 post-hoc 诊断。
- 此次没有修改 `evaluator/local_evaluator.py`、官方数据、API contract、evaluation config、submission rules 或 `starter/agent.py`。
- 严格 A 在 Public200 没有超过 T0，所以按照约定 **不修改、不推送 main**；完整 A/B + website 放在 `deadline-v3.5-ab-website` 分支。

关键提交：A `d6c96ac`，B `5482948`。网站展示读取的是上述冻结结果，不会为了打开页面重新评测。
