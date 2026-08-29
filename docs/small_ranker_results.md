# Small Ranker v1 results

This ledger covers the frozen `train_explore` research path, its one allowed
limit-100 runtime smoke, and its one untouched `calibration` run. Selection,
confirmation, public, and the official full-Agent evaluator remain unopened.

| Stage | Commit | Feature schema | Model/config | HR@10 delta | miss->hit | hit->miss | Coverage | Wall | Decision |
|---|---|---|---|---:|---:|---:|---|---:|---|
| Rich C100 cache | `fa61223` | `92795134` (133 features) | n/a | n/a | n/a | n/a | 2,000 sessions; 20,000 query groups | 1,623.098 s feature phase | freeze cache |
| Grouped OOF | `6903387` | `92795134` | `ndcg_d4_lr003` | +0.0280 | 56 | 0 | rescues in 5/5 folds | 66.07 s train | leader; runtime allowed |
| Grouped OOF | `6903387` | `92795134` | `ndcg_d4_lr006` | +0.0270 | 55 | 1 | 5/5 folds | 60.32 s train | reject: one harm |
| Grouped OOF | `6903387` | `92795134` | `ndcg_d6_lr003` | +0.0205 | 43 | 2 | 5/5 folds | 34.47 s train | reject: two harms |
| Grouped OOF | `6903387` | `92795134` | `ndcg_d6_lr006` | +0.0235 | 47 | 0 | 5/5 folds | 49.30 s train | pass; rank 2 |
| Grouped OOF | `6903387` | `92795134` | `ndcg_d4_regularized` | +0.0190 | 38 | 0 | 5/5 folds | 35.72 s train | pass; rank 3 |
| Grouped OOF | `6903387` | `92795134` | `pairwise_d4_control` | +0.0270 | 55 | 1 | 5/5 folds | 78.67 s train | reject: one harm |
| Research runtime export | `d281807` | `92795134`, semantic route absent | 273-tree `ndcg_d4_lr003` | +0.0280 OOF | 56 | 0 | exact C100 order on 1,000 parity rows | 1.645 s export | smoke allowed; artifact remains ignored |
| Runtime smoke | `d281807` | frozen runtime | active, limit 100 | n/a | n/a | n/a | 271 turns, 129 changes, 0 fallback | 56.840 s | authorize one calibration |
| Untouched calibration | `d281807` | frozen runtime | active, run 1/1 | +0.0330 | 66 | 0 | 4/4 scenarios positive | 897.433 s | **do not promote** |
| v1.1 runtime fix | `95b50cc` | unchanged | same frozen model/gate | not evaluated | n/a | n/a | synthetic cache eviction plus P95/RSS helpers | no Agent run | research only; needs new untouched evidence |

## Frozen training evidence

The cache contains 2,000,000 float32 rows with shape
`(2000, 10, 100, 133)`, occupies 1,064,000,128 bytes, and has feature SHA-256
`2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089`.
The separately joined label arrays have SHA-256
`9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb`.
Numeric and ephemeral visible-context scans found no identifier-shaped feature.

The catalog-family signature produced 747 groups for 747 unique labels, so the
run is exact-parent grouped in effect. This is an optimistic evidence boundary,
not proof of generalization to unseen near-duplicate families. The leader moves
official-style session HR@10 from 0.9475 to 0.9755 with fold net rescues
`7/15/14/7/13`, paired bootstrap interval `[0.021, 0.035]`, family-uniform
delta `+0.021419`, and 0 observed harms. Rescues are concentrated in clothing
(53 clothing, 3 shoes), which is another generalization caveat.

The one preregistered six-model batch used local XGBoost 1.7.6 CPU because no
package download was allowed. Batch wall time was 342.871 seconds. The served
runtime has no XGBoost dependency. Its ignored research JSON is 257,380 bytes
(SHA-256 `090f1dfe5a07b922f4b4663dc1bd093892530cc717b45cb31658adb800c5c96e`);
no model was copied to `starter/assets/`.

## Untouched calibration decision

The single frozen run improved aggregate HR@10 from 0.932 to 0.965:
`66 miss->hit`, `0 hit->miss`, net `+66`. Every scenario was positive and
zero-harm: boundary `+2`, browsing `+18`, buying `+35`, and intent override
`+11`. MTTC improved from 3.247 to 2.834 and TechnicalScore from 0.820510 to
0.826795, but MRR regressed from 0.664834 to 0.603249 (`-0.061585`).

Promotion is rejected. Across 5,598 turns, the runtime changed 2,988 outputs
but safely fell back to exact P11 111 times on `runtime_error:KeyError`.
Static diagnosis found a frozen implementation error in `_fetch_evidence`: at
the 16,384-row cache limit, inserting missing evidence can evict a current C100
row that was considered present before insertion. The limit-100 smoke read only
11,428 evidence rows, so it did not reach this boundary. Fixing it after seeing
calibration would create a new, uncalibrated runtime; this version therefore
remains unchanged and calibration is not rerun.

Calibration wall time was 897.433 seconds. Mean/max per-turn latency was
159.769/964.895 ms. P95 and peak RSS were not captured, so the resource gate is
also incomplete. There were no rank-1-to-9, slot-10, shape, or served-mode
invariant failures; no forbidden training library, token use, or target feature
entered runtime.

The immutable aggregate result is gitignored at
`experiments/fast_track/small_ranker_v1/oof_batch_v1/untouched_calibration.json`
(3,743 bytes, SHA-256
`ff6cbd59889ca2533088c4c94f58836846a0a68038884d5f3313474e3b062ffd`).
The tracked audit record is
`configs/small_ranker_v1.calibration.manifest.json`. The served default remains
`off`; post-calibration tuning and promotion are forbidden for this candidate.

## v1.1 implementation-only checkpoint

Branch `small-ranker-v1.1-runtime-fix` corrects the diagnosed cache failure
without changing the 133 features, ranker, gate, threshold, Agent integration,
or ignored artifact. Cached members of the current C100 are now touched before
missing rows are inserted, so the 16,384-row LRU evicts unrelated historical
rows instead of a candidate needed by the same turn. New guards reject an
oversized or duplicate candidate set rather than serving a partial result.

The future smoke runner now records deterministic nearest-rank P95 latency and
the OS process-lifetime peak RSS using only the Python standard library. Missing
either measurement makes the smoke fail closed. A focused synthetic regression
test forces eviction at a three-row cache boundary and verifies the current row
remains available; the resource helper test verifies P95 and a positive RSS.

No Agent, evaluator, smoke, calibration, selection, confirmation, or public run
was started for v1.1. The previous calibration cannot be reused as untouched
evidence for modified runtime code. Consequently v1.1 is not promoted, the
artifact remains ignored, and the served default remains `off`. Its source
hashes and exact test evidence are recorded in
`configs/small_ranker_v1_1.runtime_fix.manifest.json`.
