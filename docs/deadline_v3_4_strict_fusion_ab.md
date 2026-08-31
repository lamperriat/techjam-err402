# Deadline v3.4：严格融合 A/B（Public200 + 2k）

这次的 A/B 定义已经修正：A 的任何路径都不会输出 `ask_attribute=other`；B 从 A 的冻结提交派生，唯一新增的是 evaluator-aligned bounded `other`。A 前两页逐响应保留队友 T0，第三页以后才使用既有 P11 + fold-safe small-ranker 的 unseen tail，因此没有重新加入已证明有害的 hard-conflict 或直接 Top10 P11 重排。

| Benchmark | Version | HR@10 | MRR | MTTC | Score |
|---|---|---:|---:|---:|---:|
| Public200 | A | 0.9950 | 0.671333 | 2.1500 | 0.875900 |
| Public200 | **B** | **0.9950** | **0.706280** | **1.8550** | **0.892284** |
| 2k OOF | A | 0.9905 | 0.609388 | 2.4145 | 0.849776 |
| 2k OOF | **B** | **0.9915** | **0.626409** | **2.0285** | **0.863103** |
| 2k OOF | v2.12 reference | 0.9910 | **0.695795** | 2.8690 | **0.866858** |

Public200 上 B 保持 A 的 HR，并增加 MRR `+0.034947`、缩短 MTTC `0.295`、提高 Score `+0.016384`。完整 2k 上，B 相对 A 为 4 miss→hit、2 hit→miss、净 +2；HR `+0.0010`、MRR `+0.017021`、MTTC `-0.3860`、Score `+0.013327`。

B 不是零伤害方案：2k fold 0 的 HR 从 `1.0000` 降到 `0.9975`，MRR 也下降；intent-override 场景少 2 个 hit。总体上 B 的 HR 比 v2.12 高 1 个会话、MTTC 快 `0.8405` 轮，但 MRR 低 `0.069386`，TechnicalScore 仍低 `0.003755`。因此按综合分安全性仍应保留 v2.12 默认；若明确优先 aggregate HR 与 MTTC，可以选择 B，并接受上述风险。

A commit：`d6c96ac`。B commit：`5482948`，相对 A 只增加 35 行问询适配与测试。Public 与 2k 均使用 4-worker、两 replica exact-repeat；定向测试 15/15 通过。2k 是本地 `train_explore` OOF benchmark，不是比赛 private 成绩。完整机器可读结果见 `docs/deadline_v3_4_strict_fusion_ab.json`。
