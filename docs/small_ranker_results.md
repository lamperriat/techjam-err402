# Small Ranker v1 results

This ledger covers only the frozen `train_explore` research path. Calibration,
selection, confirmation, public, and full-Agent evaluation remain unopened.

| Stage | Commit | Feature schema | Model config | OOF HR delta | miss→hit | hit→miss | Fold coverage | Wall | Decision |
|---|---|---|---|---:|---:|---:|---|---:|---|
| Rich C100 cache | pending checkpoint | `92795134` (133 features) | n/a | n/a | n/a | n/a | 400 sessions × 5; 3,703–3,794 positive query groups/fold | 1,623.098 s feature phase | proceed to one preregistered LambdaMART batch |
| Grouped OOF | pending checkpoint | `92795134` | `ndcg_d4_lr003` | +0.0280 | 56 | 0 | 5/5 | 66.07 s train | pass OOF; export/runtime smoke allowed |
| Grouped OOF | pending checkpoint | `92795134` | `ndcg_d4_lr006` | +0.0270 | 55 | 1 | 5/5 | 60.32 s train | reject: one hit→miss |
| Grouped OOF | pending checkpoint | `92795134` | `ndcg_d6_lr003` | +0.0205 | 43 | 2 | 5/5 | 34.47 s train | reject: two hit→miss |
| Grouped OOF | pending checkpoint | `92795134` | `ndcg_d6_lr006` | +0.0235 | 47 | 0 | 5/5 | 49.30 s train | pass OOF, rank 2 |
| Grouped OOF | pending checkpoint | `92795134` | `ndcg_d4_regularized` | +0.0190 | 38 | 0 | 5/5 | 35.72 s train | pass OOF, rank 3 |
| Grouped OOF | pending checkpoint | `92795134` | `pairwise_d4_control` | +0.0270 | 55 | 1 | 5/5 | 78.67 s train | reject: one hit→miss |

Cache facts: 2,000,000 float32 rows, 20,000 query groups, 1,064,000,128
bytes, feature SHA-256
`2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089`,
label SHA-256
`9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb`.
The official baseline join is 1,895 hits and 105 misses. Numeric and ephemeral
visible-context scans found zero identifier-shaped values.

The catalog-metadata family signature produced 747 groups for 747 unique
labels (no cross-parent merges), so this run is exact-parent grouped in effect.
That is an explicitly optimistic evidence boundary, not proof of generalization
to unseen near-duplicate product families.

The one preregistered batch used the only locally installed offline trainer,
XGBoost 1.7.6 CPU. It supports `rank:ndcg`, qid, `hist`, and early stopping, but
does not expose the requested XGBoost 3.2 `lambdarank_pair_method`; no package
was downloaded. Batch wall was 342.87 seconds. The leader moves official-style
session HR@10 from 0.9475 to 0.9755 in grouped OOF, with per-fold net rescues
7/15/14/7/13, paired bootstrap interval [0.021, 0.035], family-uniform delta
+0.021419, 16 rescued exact-parent groups, and byte-identical repeat scores.
Rescues are concentrated in head-popularity clothing (53 clothing, 3 shoes),
so untouched calibration remains essential despite the zero observed OOF harm.

The complete ignored result is
`experiments/fast_track/small_ranker_v1/oof_batch_v1/oof_results.json`
(40,001 bytes, SHA-256
`00d7b67d2fe47361894d826d7908416469ddbd5d2c1efc08cc136f98e2f2b08f`).
No calibration, selection, confirmation, public, or full-Agent evaluator was
opened. The current decision is to export only `ndcg_d4_lr003`, implement
lightweight parity/fallback, and run the single allowed limit-100 smoke before
freezing for calibration.
