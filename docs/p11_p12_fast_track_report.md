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

## Phase 2 — next checkpoint

只在 Amazon Reviews 2023 Clothing_Shoes_and_Jewelry 5-core **validation** proxy
（至少 8k、target-group-disjoint、排除 public targets、绝不读取 test）的 action
oracle 达到预设 Go 条件后，才实现最小 guarded Top50→Top10 admission；否则保留
P11/R08 served path 并记录 No-Go。
