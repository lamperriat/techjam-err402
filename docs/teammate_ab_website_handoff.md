# 队友交接：Fusion A/B 与 Agent Workbench

## 一句话说明

这个网站是本地、只绑定 loopback 的 **Agent Layer Observer**：首页的 Fusion Studio 用冻结证据解释队友 T0、Fusion A、Fusion B 的计算路径与结果；原有页面继续提供真实会话 trace、50k catalog/FTS5 浏览、受限实验任务、手动对话 Lab 和项目文档。网站是演示与开发工具，不会进入官方 Agent 的检索、排序或评分路径。

## 如何启动

先拉取完整交付分支（不要从 `main` 期待这套 A/B 网站）：

```powershell
git fetch origin
git switch deadline-v3.5-ab-website
git pull --ff-only origin deadline-v3.5-ab-website
```

网站源码、T0/A/B adapters、冻结指标 JSON、交接文档、P11 sidecar 和
fold-safe ranker artifact 都在 Git 中；官方 catalog 由 organizer Release 单独
分发，因此不会随普通 pull 进入工作树。

```powershell
conda activate tiktok
cd <你的仓库目录>
python -m observer.launcher
```

Windows 也可双击 `Start Observer.vbs`。fresh pull 即使尚未安装 `data/catalog.jsonl`，也会进入只读 **Showcase mode**，正常打开 Fusion A/B 架构、逐层播放器、冻结指标和资料库；它不会在缺少 catalog 时假装运行检索。放入官方冻结 catalog 并重启后，网站切换为完整本地模式，启用真实 T0/A/B Live Lab、商品索引、会话诊断和固定运行工具。网站不会下载、生成或修改 catalog。

官方 catalog 应解压到：

```text
data/catalog.jsonl
```

应有 50,000 行，SHA-256 为
`da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`。`results.json` 和被忽略的完整 2k artifact 都不是打开网站或查看冻结结果的必要文件。

## 网站现在能看什么

- **Fusion A/B**：切换 T0、A、B，查看每层职责、源码边界、问询策略、Public200/2k 指标与风险。
- **Compute Walkthrough**：选择 intent page 1、page 2、page 3+ 或 override，再逐步播放该轮的状态、召回、排序、路由、去重和问询过程。这是冻结架构说明，不会触发 evaluator。
- **会话诊断**：运行当前 `starter.Agent` 的真实逐轮 trace，并把 Agent 实际事件与 post-hoc target 诊断明确分开。
- **商品与索引**：浏览 50,000 商品及本机 FTS5 检索结果。
- **运行与实验**：仅允许仓库预定义动作；不能执行任意 shell。此次更新没有点击或运行任何 evaluator。
- **交互 Lab / 资料库**：目标盲手动对话，以及代码、规则、结果和本交接文档的只读查看。

首页数据来自 `docs/teammate_ab_website.json`，原始严格 A/B 证据仍在 `docs/deadline_v3_4_strict_fusion_ab.json`。网页不自行计算或改写指标。

## 真实 T0/A/B Live Lab

安装官方 catalog 后，Fusion 首页可以直接选择 T0、A 或 B，点击“开始 / 重置会话”，再使用 Buying、Browsing、Clarification、Override 快捷场景或自行输入 evaluator 形式的可见消息。每轮实际调用对应的 tracked Agent factory；切换版本时关闭旧索引，并创建新的 opaque session。

页面刻意区分三种资料：

- **真实运行结果**：Agent message、`ask_attribute` 和实际有序 Top10；
- **observer-derived 解释**：在相同可见 state 上确定性重放候选查询，并读取 session/page/ledger/question lifecycle，用来解释候选数、推断页面路由、fallback 与各层职责；它不是伪装成 Agent 原生发出的 trace；
- **冻结评测证据**：从 tracked JSON 读取的 Public200/本地 2k OOF 汇总，不会因启动网站或使用 Lab 而重跑。

Live Lab 只接收 profile、message、turn 和 `top_k=10`，不接收 `sample_id`、target、scenario、evaluator state 或既往结果。它没有评分 join，不能通过录屏输入泄漏 target。若无 catalog，页面会明确提示 Live Lab 不可用，但 Showcase 仍可演示。

## 建议录屏流程（约 5–8 分钟）

1. 打开首页，先说明这是本地 loopback Observer；无 catalog 也能展示，完整模式则读取官方 50k catalog。
2. 依次切换 T0、A、B，说明 T0 的 FTS1000 + ProductScorer、A 的两页 T0 grace + 第三页 unseen expert tail、B 只增加 bounded `other` lifecycle。
3. 播放 Page 1、Page 2、Page 3+ 与 Override，指着高亮节点说明每轮经历的状态、召回、排序、去重、路由和问询。
4. 切换 Public200 / 2k OOF 结果；明确 2k 是本地 `train_explore` OOF，不是官方 private 800，T0 也没有可比较的完整 2k artifact。
5. 在 Live Lab 分别新建 T0、A、B 会话，使用相同 Buying/Browsing prompt 展示真实 Top10；继续发送 Clarification，再用 Override 展示状态和 question lifecycle 重置。
6. 说明右侧 Top10/response 来自真实 Agent；页面路由、候选数量和八层说明为 observer-derived；两者都不含 target。
7. 最后回到冻结表格：A 的 Public HR 与 T0 持平但 MRR、MTTC、Score 更差，所以完整网站与 A/B 发布在新分支而不是 `main`。

仅录制展示时不要点击“运行 200 会话评测”。使用架构播放器或 Live Lab 本身都不会启动 evaluator。

## 交付验收记录（2026-09-01）

本轮没有重跑 Public200 或 2k evaluator；下列是网站与手动 Live Lab 的 contract smoke：

- 无 `data/catalog.jsonl` 实际启动成功：`/`、CSS、JS、health、overview、Fusion evidence、资料库均返回成功，Live Lab 明确返回 `available=false` 和安装说明；
- 使用 SHA-256 与上文一致的官方 50,000 商品 catalog 实际启动成功；
- T0、A、B 各自真实 reset/respond 均返回 8 个解释层和 10 个唯一 catalog-valid ID；
- 相同首轮 Buying 输入下，`T0 == A == B` 的 Top10 顺序逐项一致；T0/A 保留具体属性问询，B 唯一改为 `ask_attribute=other`；
- A 连续三轮路由依次为 T0 grace 1/2、T0 grace 2/2、v2.12 unseen expert tail；
- B override 后 page 回到 grace 1/2，`other.version` 从 1 增至 2，asks 从 2 重置为 1；
- Workbench 的 73 项可移植核心测试全部通过；其中 32 项直接覆盖 HTTP/token、Showcase、静态页面 marker、T0/A/B Live Lab、官方风格 Top10 normalization、variant 切换清理、10-turn、A/B grace/tail/override 与 response contract。需要 ignored P6/P8/P9 历史资产或可选 `pytest` 的研究测试仍由各自 CLI 环境运行，不会让网站按钮在 fresh pull 中误报失败；
- Python compile、JavaScript syntax、`git diff --check` 通过。

这些 smoke 只验证可启动、可操作、算法边界和输出 contract，不替代冻结 benchmark，也不会产生新的成绩。

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
| local 2k OOF | Teammate T0 | — | — | — | 未运行，无可比较 artifact |
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
