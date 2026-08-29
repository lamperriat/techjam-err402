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

## Phase 3 — action oracle next

只在未封存的 explore/calibration/selection 上复用一个通用 runner，比较 KEEP_R08、
KEEP_P11、C50 structured rerank、candidate-only semantic、result-aware rewrite 与 ASK。
Oracle HR 至少 `+1.5pp`（或等价明确 Score 上界）且跨 scenario/taxonomy 有净 rescue
后，才实现最小 guarded Top50→Top10 admission；否则保留 P11/R08 并记录 No-Go。
