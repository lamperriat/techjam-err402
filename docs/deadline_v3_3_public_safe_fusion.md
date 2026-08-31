# Deadline v3.3：Public200 安全融合结果

本轮只使用官方 Public200，没有读取或运行 2k。当前 Public200 综合分最佳是 `create_intent_routed_forced_other_fusion`：HR@10 保持 0.995，同时比队友 T0 提高 MRR、降低 MTTC，并把 TechnicalScore 从 0.886430 提高到 0.897353。

| 版本 | HR@10 | MRR | MTTC | Score | 结论 |
|---|---:|---:|---:|---:|---|
| 队友 T0 | 0.995 | 0.703766 | 2.110 | 0.886430 | 基线 |
| Fusion-B | 0.995 | 0.714518 | 1.930 | 0.893255 | 有效组件 |
| T0 + forced bounded `other` | 0.995 | 0.714899 | **1.850** | 0.894970 | 有效组件 |
| 可见意图路由 | 0.995 | **0.728530** | 2.065 | 0.894759 | 有效组件 |
| **最终融合** | **0.995** | 0.724177 | 1.870 | **0.897353** | Public200 winner |

最终融合只在第 1 轮根据用户原文选择一次后端，之后不切换状态机：含 `but I'm still exploring.` 的开放请求走队友 T0 检索，并用 bounded `other` 尽早取得多个约束；明确购买与 intent-override 走 Fusion-B。90 个开放会话和 110 个明确会话分别路由，两个分支合计触发 338 次 `other` 问询。

失败消融也已收口：QRESET 使 MTTC 变差；冻结首轮旧队列使 HR 降至 0.885；直接 P11 重排使 MRR 降至 0.529752；逐页拼接 T0 与 Fusion-B 虽把 MTTC 降到 1.910，却把 MRR 降到 0.693093。它们均未进入提交版本。

最终 Public200 使用 4-worker exact repeat，两次 ledger 完全一致（`9f02c43d…68c34`），定向测试 22/22 通过。实现提交是 `bd89273`。完整机器可读结果见 `docs/deadline_v3_3_public_safe_fusion.json`。

注意：这是在 Public200 上选择出的版本，不能当作 private/2k 成绩；按用户要求，本轮没有运行 2k。
