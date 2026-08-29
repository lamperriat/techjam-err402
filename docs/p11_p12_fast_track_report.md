# P11 / P12 Fast-track Report

## Phase 1 — P11 served integration

- **状态：** frozen P11 reranker 已接入 served path，默认 `active`；在固定的
  `coverage/off/fast` served preset 下设置 `TECHJAM_P11_MODE=off`，可完整恢复 R08
  response/ranking（调试 trace 仍会标注 P11 disabled）。
- **边界：** P11 只重排 R08 已召回的 Top 10，集合与 rank 11+ 不变；它可能改善
  MRR / Technical Score，但**不能改善 Hit Rate@10**。
- **冻结身份：** sidecar 为 32,501,760 bytes，SHA-256
  `83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473`。
  scorer 是 sparse / field / constraint linear scorer，不是 embedding 检索。
- **既有证据：** primary / uniform / confirmation 的 HR 与 MTTC 均不变、hit→miss
  为 0，Technical Score 分别变化 `+0.012685 / +0.003309 / +0.014279`；冻结决策为
  `promote_p11_r01`。
- **回退：** sidecar 缺失或不匹配，以及初始化、fetch、subtype、score、adapter、
  boundary 异常，均不得改变 R08 排名；失败会进入 P11 diagnostics。close 异常会
  向调用方抛出，避免把未完成的 shutdown 误报为成功。
- **验证：** 68 项 P11/core 测试、完整 580 项仓库测试、14 项官方资产检查与
  `compileall` 均通过；覆盖 Top-10 membership、exact-score tie 保序、故障注入、
  生命周期与 off-equivalence。
- **资源：** 既有 active/control 比值为 wall `1.0513`、P95 `1.0455`、RSS
  `1.0261`；本轮不另建大型 P11 evaluator、corpus 或 gate。
- **公开集纪律：** 本轮没有重跑官方 public 200。最后已发布 R08 checkpoint 仍为
  HR@10 `0.945`、MRR `0.606175`、MTTC `3.335`、Score `0.807652`。
- **残余风险：** uniform-tail override 与 confirmation-boundary 有局部 MRR 回退；
  P11 不改善 HR，private 800 泛化未知。

## Phase 2 — Amazon validation proxy

- 固定来源为 Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` 5-core
  **validation-only**；原始文件 345,027,412 bytes，SHA-256 `94b00815…a21ba`，
  `test_rows_read=0`。原始与派生行均 gitignored、不可随 release/ZIP 分发。
- builder 在解析前后校验 source/catalog/header，并 fail-closed 固定官方 revision、URL、
  byte/hash、50k catalog、配置身份、public 与 P1–P11 共 16 份已消费语料；2,720 个
  target 的排除 union 正确且 pairwise overlap=0。CLI 不接受 fixture 配置，Python API
  也不能通过伪造 `production_pinned` 绕过固定配置。
- 2,524,981 个 source rows 中，35,717 rows / 2,986 unique targets 能 inner join
  frozen 50k catalog 并通过排除。因 unique target 少于 8k，target 只可在自己的
  split 内重复；四 split 的 target overlap 均为 0。
- 已生成 train/explore、calibration、selection、sealed confirmation 各 2,000 行，
  场景均严格为 800/800/300/100；四 split 合计覆盖全部 2,986 eligible targets。
- source-frequency >=10% 的 1 个 outcome-independent validation-source outlier 在看
  结果前固定进入 train/explore，
  防止单个商品支配 held-out；selection 最大 source-weight share 为 15.03%。所有结论
  仍须同时报告 source-weighted、target-uniform 与 taxonomy-balanced 三个视图。
- raw user ID/rating/timestamp/history 不落盘；target rating 不作 prior rating。
  purchase-frequency 只使用 validation 前 history 长度的离散区间；prior item 对 frozen
  catalog 的 join rate 仅 2.4736%，因此许多 preference-tag aggregate 为 neutral。这是
  proxy 与 organizer-private 分布未知之间的明确限制。
- source-freeze commit 为 `d8f6805c3edf42bb8d21d81bcdf0b3527e928e87`；13 项
  fixture/security 测试通过，覆盖 source/config 漂移、test 路径提前拒绝、输出
  冲突、fresh-checkout aggregate evidence、production marker 伪造与中途发布回滚。
  六文件发布可在普通异常时回滚，但不是断电/强杀下的集合级原子事务；中断后必须先
  审核残留文件。完整仓库为 593/593 tests、官方资产 14/14、`compileall` 通过。
- 从 source-freeze commit 做两次完整构建，四 split、manifest 与 audit 全部
  byte-identical；随后只保留 aggregate evidence 的 fresh-checkout 模拟也逐字节补齐四
  split。manifest SHA-256 为
  `8058973426bbc76ea856a5c48a61e91ed9e35ae44988a21a6d7b2195e88a7193`，audit 为
  `1e6b084bf16fbd0000ec0bceca8057265390f3fde03c1936aced39f6f50537e8`。
- confirmation 只物化为 sealed 文件；本阶段没有运行 evaluator 或读取其结果。

## Phase 3 — P12 action oracle（旧 matrix train/calibration 已完成并停止）

- **代码身份：** action-oracle source 先后冻结于 `64c26e2` 与网络审计修正版
  `d1013b3`；四 worker 并行、RSS 口径与 ASIN-shape 安全修复后的 v5 commit 为
  `761b9a3`。P11 served integration commit 为 `7d79e31`；P12 proxy source/evidence
  commits 分别为 `d8f6805` / `6d0d28b`。
- **固定 action matrix：** 同一盲态 turn 比较 `KEEP_R08`、`KEEP_P11`、C50
  structured rerank、只在 C50 内打分的 frozen semantic rerank、真实 P5 R01
  result-aware rewrite，以及 observed `ASK`。`ASK` 不是独立反事实，排除在 oracle
  admission gate 之外；semantic action 不允许做 50k 全库向量搜索。
- **盲态边界：** parent 持有 target、label 与确定性 customer reply；worker 只接收
  投影后的 profile、当前可见 message、局部 ordinal、turn 与 top-k。所有 worker
  `finalize`、receipt 校验和干净退出后，parent 才可读取 trace 并 join label。
  confirmation 在 runner 首行即被拒绝，当前仍 sealed；source、split、catalog、P11
  sidecar 与 semantic assets 均做前后身份检查。
- **并行语义：** v5 将 split 连续均衡切为四个 shard，分别使用 nonce、局部 ordinal
  与 trace；全部 shard 成功后才做全局验证与 ordinal 映射，任一 worker 失败即整次
  fail-closed。合并统计使用整数计数，不能平均已四舍五入的 recall rate。

### 非决策 smoke 与安全修复

- 10-session single v2 为 `24.466635s`；parallel v3 为 `12.661261s`；RSS 修正后的
  v4 为 `15.867314s`。三者的 action oracle 与 normalized combined-trace digest
  完全等价（摘要 `54543902…ef23`），C10/C20/C50/C100 candidate recall 均为 `1.0`。
  这些仅验证并行等价与执行路径，不是质量选择证据。
- v4 的 10-session run 最大单 worker lifetime peak RSS 为 `462,536,704` bytes；四个
  worker peak 的保守求和上界为 `1,847,644,160` bytes。该上界不是并发时刻的实测
  总峰值，且不含 parent RSS。
- 第一次 2,000-session full attempt 约 `2.5 min` 后因
  `worker respond reply shape mismatch` 停止；增强安全异常分类后，第二次在同一
  shard 附近约 `2.5 min` 后因 `worker error reply identity is invalid` 停止。两次均
  未留下 aggregate 或 trace artifact，证明错误路径没有发布半成品。
- 根因不是 catalog/target 泄漏，而是确定性 customer reply 中出现 8 个“长得像
  ASIN”的非 catalog、非 target token：2 个来自 initial reply、6 个来自 size reply，
  共影响 train/explore 的 6 个 session；calibration 与 selection 均为 0。v5 先对 raw
  payload 严格拒绝任何真实 target/catalog/sample identifier，再把剩余纯 shape 命中
  redact 为 `[identifier omitted]`，之后二次校验；worker 端 strict guard 保留。
- v5 的 200-prefix smoke 用时 `103.241977s`，sanitization 计数为 session/message/token
  `1/1/1`；integrity 为 true，worker failure、network attempt 与 full-catalog semantic
  search 均为 0。最大单 worker lifetime peak RSS 为 `494,694,400` bytes，worker
  peak 保守求和上界为 `1,972,527,104` bytes，同样不含 parent。该非决策 prefix 的
  oracle HR delta 为 `+0.005`、Score delta 为 `+0.023338`，不得外推为完整 split
  或 private-800 效果。
- 最新 P12 定向测试为 `71/71`；并行与 RSS 回归后的完整仓库测试分别为
  `660/660`、`662/662`。独立 source review 未发现 blocker/high 问题。

### 公开集审计例外

Phase 3 早期曾误运行一次 `verify_official_assets.py`；此后为并行与 RSS 回归运行的
两次 full suite 也包含既有 public-integrity 测试，因此读取了 released-public 的
hash/schema/row integrity。这违反了“Phase 3 完全不读 public 文件”的字面纪律，必须
如实保留。没有运行 public evaluator、没有用 public 做模型评分或调参；P12 runner
记录的 `public_rows_read=0` 仍然成立，public target 也早已从 proxy union 中排除。

### 完整结果与暂停状态

统一身份：commit `761b9a3`；config canonical SHA-256
`492b42c19708b0e528755cb00374b368afaf037ce2c8b1f5d33f52685de3638c`；旧 action IDs 为
`KEEP_R08`、`KEEP_P11`、`CANDIDATE_RERANK`、`FROZEN_SEMANTIC_RERANK`、
`RESULT_AWARE_REWRITE_RETRIEVE`、`ASK`（observed，gate-excluded）。

| Split | Baseline HR@10 | C50 recall | Oracle HR@10（delta） | Oracle Score delta | Cluster-CI lower | Candidate m→h / h→m / net | Semantic m→h / h→m / net | Rewrite m→h / h→m / net | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train/explore 2,000 | 0.947500 | 0.991000 | 0.956000 (+0.008500) | +0.025785 | 0.019859270169 | 14 / 21 / -7 | 5 / 48 / -43 | 0 / 0 / 0 | 947.950242s |
| calibration 2,000 | 0.932000 | 0.987000 | 0.950000 (+0.018000) | +0.032477 | 0.023202301436 | 16 / 13 / +3 | 22 / 36 / -14 | 0 / 0 / 0 | 1076.397994s |

Calibration candidate 虽为净 +3（positive-net span 2 scenario / 2 taxonomy），但与 train
的净 -7 不一致；semantic 两个 split 均净退化，rewrite 均无 HR rescue。Oracle 是
hindsight upper bound，不是 deployable policy，因此旧 matrix 决策为 **STOP**，不授权
CAGE。train/explore 的 source-weighted baseline 被预先分配的高频 outlier 强烈影响
（HR `0.209173`、weight sum `21,585`；calibration 为 `0.930249/4,301`），不能单独拿来
跨 split 或外推 private 800；必须同时看 row/target/taxonomy 三个视图。

被中止的任务确认为 `selection`：运行约 8 分钟后按用户指令 Ctrl+C，parent 与四个 worker
均退出，未生成 `selection-full` artifact。由于 split 已被打开，它不能再作为正式 one-shot
selection；未来只有新 action 通过 train/calibration 阶梯后，才可生成从未打开、
target/product-family-disjoint 的 fresh selection。Sealed confirmation 未授权、未读取。

`COMPACT_NEGATIVE_C50` 与 `GUARDED_COMPACT_SLOT10` 尚未实现。下一步只把两者加入同一个
matrix，运行一次相关 targeted tests，再运行一次 train/explore `limit=100`；本轮不运行
limit 10/200、完整 split、calibration、selection 或 confirmation。
