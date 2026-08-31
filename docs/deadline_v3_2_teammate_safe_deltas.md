# Deadline v3.2 teammate-safe deltas

结论：两个限定增量在固定 200-session paired panel 上都未激活，均未产生净提升。按预设快速淘汰门槛，S1、S2 都是 No-Go；未运行组合，也未消费完整 2k。

## 可复现边界

- 起点：`42d73a5`；实现 commit：`776758b`；队友 V1：`5df5d51e7578e80616f45fcaa89ec977347845fa`。
- 固定 panel：每 fold 40 个，场景配额为 buying 16 / browsing 16 / override 6 / boundary 2。
- 10-session contract smoke：9/9 targeted tests 通过，串行与 4-worker ledger hash 一致；第一页 identity、去重、reset 和异常回退通过。
- Panel 为单 replica；没有把它表述为 exact repeat 或完整 2k 成绩。

## 结果

| Variant | HR@10 | MRR | MTTC | Score | m→h | h→m | Net | Activation | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 teammate V1 | 0.995 | 0.574593 | 2.335 | 0.843178 | 0 | 0 | 0 | 0 | 12.975s |
| T0+S1 | 0.995 | 0.574593 | 2.335 | 0.843178 | 0 | 0 | 0 | 0 | 12.909s |
| T0+S2 | 0.995 | 0.574593 | 2.335 | 0.843178 | 0 | 0 | 0 | 0 | 13.059s |

三者 fold HR 都是 `[1.0, 1.0, 1.0, 0.975, 1.0]`，ledger SHA-256 都是 `a3f899dccdb05cb44f17d7946c6817df133d1c7e255ed92e3e1ecab199dc3630`。S1 在官方规范 override 上与 V1 原生 reset 等价；S2 的“后续页不足才补位”条件在 466 个 turn 中从未出现。

## 最终选择

队友路线保留原始 T0，不提交 S1/S2。只读参照的 v2.12 完整 2k 为 HR 0.991 / MRR 0.695795 / MTTC 2.869 / Score 0.866858；因本次只是 200-panel，不能用 T0 的 0.995 宣称超过 v2.12。整体提交仍推荐已有完整证据的 v2.12。
