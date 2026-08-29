# Fast Algorithm Results

Last updated: 2026-08-29 SGT. This aggregate-only registry prevents exact-repeat runs; it
contains no blind trace, public-session row, target/ASIN, or confirmation result.

## Old matrix identity

- Frozen checkpoint: `761b9a3`
- Config canonical SHA-256: `492b42c19708b0e528755cb00374b368afaf037ce2c8b1f5d33f52685de3638c`
- Actions: `KEEP_R08`, `KEEP_P11`, `CANDIDATE_RERANK`,
  `FROZEN_SEMANTIC_RERANK`, `RESULT_AWARE_REWRITE_RETRIEVE`, `ASK`
- Parameters: four workers, fixed ten-turn replay, Top-10 response, candidate-only C50
  structured/semantic actions. `ASK` is observed and gate-excluded.

## Successful-run reuse ledger

`m→h / h→m / net` is reported for Candidate; Semantic follows in the next column. The v1
aggregate schema did not emit a separate activation count, so that field is recorded as
`not emitted` rather than inferred.

| Split | Limit | Runner source SHA prefix | Output | Wall | Baseline / C50 / Oracle HR | Candidate m→h / h→m / net | Semantic m→h / h→m / net | Rewrite net | Activation | Decision |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| train/explore | 10 | `eb01b2504f20` | `train_explore-smoke-10-aggregate.json` | 15.867314s | 1.000 / 1.000 / 1.000 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | not emitted | REUSE; protocol-only |
| train/explore | 100 | `910d5229b95b` | `train_explore-smoke-100-aggregate.json` | 76.357433s | 0.940 / 0.970 / 0.940 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | not emitted | STOP; no rescue |
| train/explore | 200 | `327b72b41988` | `train_explore-smoke-200-aggregate.json` | 103.241977s | 0.935 / 0.985 / 0.940 | 1 / 1 / 0 | 0 / 5 / -5 | 0 | not emitted | STOP; no net rescue |
| train/explore | full 2,000 | `327b72b41988` | `train_explore-full-aggregate.json` | 947.950242s | 0.9475 / 0.991 / 0.956 | 14 / 21 / -7 | 5 / 48 / -43 | 0 | not emitted | STOP old matrix |
| calibration | full 2,000 | `327b72b41988` | `calibration-full-aggregate.json` | 1076.397994s | 0.932 / 0.987 / 0.950 | 16 / 13 / +3 | 22 / 36 / -14 | 0 | not emitted | STOP; inconsistent with train |

The 10/100 rows are legacy successful aggregate artifacts with earlier runner source
closures; they remain non-decision evidence and must not be rerun merely to normalize the
runner hash. Before any future run, compare commit/config/action IDs/split/limit/parameters
and output path with this ledger. An exact match must reuse the existing aggregate.

## Interrupted selection

The old-matrix `selection` run was stopped by Ctrl+C after about eight minutes. Parent and
all four workers exited, and no `selection-full` artifact was produced. The opened split is
not valid formal one-shot evidence; any future formal selection must be newly generated,
previously unopened, and target/product-family-disjoint. Confirmation remains sealed.

## Compact-action batch 1

| Commit / config | Actions added | Split / limit | Wall | HR delta | m→h | h→m | Net rescue | Activation | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `87f2657` / `be4d72c77f94…` | `COMPACT_NEGATIVE_C50` | train/explore / 100 | 78.858889s | 0.000 | 0 | 0 | 0 | 0 turns / 0 sessions | ITERATE |
| `87f2657` / `be4d72c77f94…` | `GUARDED_COMPACT_SLOT10` | train/explore / 100 | 78.858889s | 0.000 | 0 | 0 | 0 | 0 turns / 0 sessions | ITERATE |

The shared run used four workers, fixed ten-turn replay, and the new eight-action registry.
`KEEP_P11` HR@10 was `0.940000`, C50 recall was `0.970000`, and oracle HR@10 remained
`0.940000`. Both new actions were exact KEEP_P11 no-ops on every observed turn: no Top10
order change, membership activation, rescue, or harm. The aggregate is
`experiments/fast_track/action_oracle_v1/train_explore-smoke-100-aggregate.json`; its
canonical config SHA-256 is
`be4d72c77f9424716abfa45580bd676140aa29f471eb7c7da0375dbc24d241a4`.

`COMPACT_NEGATIVE_C50` is the P11-based C50 stable
`compatible -> unknown -> explicit_violation` partition. `GUARDED_COMPACT_SLOT10` permits
only one rank-10 replacement, requires ranks 1-9 to be non-violating, and admits only the
first compatible rank-11-to-50 challenger; unknown evidence is never admitted. The sole
targeted batch passed `38/38`. Network attempts, full-catalog searches, semantic failures,
rewrite failures, and P11 invariant failures were all zero.

One earlier command attempt failed in config preflight before split loading, worker launch,
or artifact creation because the runner held a stale canonical-hash constant. Commit
`87f2657` corrected only that identity constant; the successful replay above is the sole
data-bearing run for this action matrix.

Neither individual action satisfies the Stage 1 expansion gate, so `limit=200` is forbidden.
The decision is `ITERATE`: diagnose why the frozen proxy prefix emits no executable compact
effect, then change the algorithm family or activation path before any further experiment.

## Next iteration plan: guarded positive admission

Source-only review explains the zero activation without opening blind traces. The frozen
proxy dialogue templates express positive requirements, overrides, browsing intent, and
generic rejection, but do not normally emit an explicit product-attribute exclusion that
survives the conservative negative compiler. The result therefore means that this proxy
prefix provides no usable signal for the compact-negative family; it does **not** establish
that compact ordering is generally ineffective. Keep both actions as diagnostic controls,
but do not spend another data run loosening the negative parser or tuning their weights.

The next matrix will test one protected rank-10 replacement family with three target-blind
variants:

1. `GUARDED_POSITIVE_SLOT10`: require a current, active, full-confidence hard positive
   constraint; replace an explicitly mismatching P11 rank 10 only with a catalog-observed
   matching challenger from ranks 11-50.
2. `HARD_CLAUSE_SLOT10_STRICT`: use normalized field-local phrase/n-gram coverage of the
   latest visible hard clause; admit a unique rank-11-to-50 challenger only when its fixed
   coverage margin over rank 10 clears the preregistered threshold.
3. `BUDGET_AROUND_SLOT10_STRICT`: for a currently active visible `around` budget, admit a
   unique challenger only when its catalog price has a preregistered material alignment
   advantage and the remaining structured evidence does not materially regress.

All variants must consume only the already fetched P11 C50 and visible dialogue/catalog
fields. They preserve Top 1-9 byte-for-byte, may perform exactly one membership swap, keep
the rest of the order stable, and fail closed on missing, ambiguous, stale, or unknown
evidence. They may not add an asset, LLM, network call, full-catalog search, target/label
input, trace field, evaluator change, or runner-protocol change.

Before the next replay, add aggregate-only worker counters for the eligibility funnel
(usable constraint, catalog support, rank-10 mismatch, eligible outsider, guard rejection,
and final membership change). Counters must contain no message text, attribute value,
product identifier, target, or row ordinal. This distinguishes “no executable evidence”
from “evidence exists but the guard rejects it” while preserving the blind boundary.

The three variants are to share one action/config commit, one targeted test batch, and one
fresh `train_explore --limit 100` replay. Exact-repeat identity must be checked first. An
individual action alone must have activation > 0, miss-to-hit > hit-to-miss, and HR delta >
0 before `limit=200` is allowed. Otherwise record `ITERATE` or `STOP` and change family.
Candidate-only variants whose outsiders must already be in the old structured or semantic
Top 10 are intentionally deprioritized: those two actions rescued zero misses on this same
100-session prefix, so that restriction cannot rescue the observed C50-only misses.
