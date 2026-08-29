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
order change, membership activation, rescue, or harm. The aggregate was archived as
`experiments/fast_track/action_oracle_v1/train_explore-smoke-100-compact-batch1-87f2657-aggregate.json`; its
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

## Deployable individual-action leaderboard

Rows are ordered by the strongest available evidence stage, then net rescue, then lower
hit-to-miss. `NE` means the artifact schema did not emit the field; it is never inferred as
zero. A positive result on only one split is not deployable evidence.

| Action | Family | Evidence | HR@10 delta | m->h / h->m / net | MRR | MTTC | Shared wall | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `CANDIDATE_RERANK` | old structured | train full / calibration full | -0.0035 / +0.0015 | 14/21/-7; 16/13/+3 | 0.619831 / 0.609205 | 3.187 / 3.207 | 947.950242s / 1076.397994s | REJECT: split-inconsistent |
| `RESULT_AWARE_REWRITE_RETRIEVE` | old rewrite | train full / calibration full | 0 / 0 | 0/0/0; 0/0/0 | 0.626667 / 0.619330 | 3.1625 / 3.245 | 947.950242s / 1076.397994s | REJECT: no HR signal |
| `FROZEN_SEMANTIC_RERANK` | old semantic | train full / calibration full | -0.0215 / -0.0070 | 5/48/-43; 22/36/-14 | 0.518078 / 0.509279 | 3.3035 / 3.2795 | 947.950242s / 1076.397994s | REJECT: harmful |
| `COMPACT_NEGATIVE_C50` | compact negative | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 77.508421s | ADVANCE_FAMILY: unsupported signal |
| `GUARDED_COMPACT_SLOT10` | compact negative | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 77.508421s | ADVANCE_FAMILY: unsupported signal |
| `GUARDED_COMPACT_SLOT10_STRICT` | compact negative | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 77.508421s | ADVANCE_FAMILY: unsupported signal |
| `P11_EVIDENCE_NOVEL_SLOT10` | risk-controlled admission | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 80.753492s | ADVANCE_FAMILY: guarded no-op |
| `HARD_CLAUSE_NOVEL_SLOT10` | risk-controlled admission | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 80.753492s | ADVANCE_FAMILY: guarded no-op |
| `TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10` | risk-controlled admission | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 80.753492s | ADVANCE_FAMILY: guarded no-op |
| `VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | CLOSE_ROUTE: zero activation |
| `DUAL_BOUNDARY_CONSENSUS_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | CLOSE_ROUTE: zero activation |
| `RECENT_OVERRIDE_RANK_FUSION_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | CLOSE_ROUTE: zero activation |

## Family 1 completion: compact-negative admission

Falsifiable hypothesis: an explicit negative compatibility improvement at the Top-10
boundary can rescue a C50 miss while preserving P11 ranks 1-9. Commit `232d686`, config
SHA-256 `661c69b70b385ef0f3591b38f844ad23ea0387e20bd6d9c071030b8a443cefb2`,
and one shared `train_explore --limit 100` replay tested all three family variants.
The aggregate is `experiments/fast_track/action_oracle_v1/train_explore-smoke-100-family1-232d686-aggregate.json`.

| Action | Definition | HR@10 / delta | m->h | h->m | Net | Activation turns / sessions | Scenario / taxonomy span | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `COMPACT_NEGATIVE_C50` | stable C50 compatible -> unknown -> violation partition | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |
| `GUARDED_COMPACT_SLOT10` | first rank11-50 challenger in a strictly better compact class | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |
| `GUARDED_COMPACT_SLOT10_STRICT` | rank10 violation <-> adjacent compatible rank11 only | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |

The baseline was HR@10 `0.940000`, MRR `0.676040`, MTTC `3.2`, and TechnicalScore
`0.828812`. Candidate recall was C10/C20/C50/C100 = `0.940/0.950/0.970/0.980`.
The hindsight oracle retained HR@10 `0.940000`; its MRR gain does not authorize expansion.
Wall time was `77.508421s`; maximum single-worker lifetime RSS was `480,882,688` bytes
with parent RSS excluded. Network attempts, full-catalog searches, semantic/rewrite
failures, P11 invariant failures, and diagnostic fail-closed turns were all zero. The
targeted batch passed `39/39` exactly once.

The aggregate-only funnel examined `13,968` ledger records across `1,000` turns but
compiled `0` executable negative constraints. Rejections were `10,353 not_negative`,
`2,827 not_active`, `698 stale_goal_version`, and `90 slot_not_allowed`. Consequently there
were zero C50 violations, better-class outsiders, adjacent compatible outsiders, partition
changes, or action activations. This confirms a proxy-signal boundary rather than a wiring
failure. Family 1 is closed after three materially distinct variants; `limit=200` is
forbidden. Next action: implement one shared Family 2 risk-controlled admission matrix
over C50 using visible positive/hard-clause/budget evidence.

## Family 2 completion: risk-controlled Top-10 admission

Falsifiable hypothesis: a target-blind challenger unique to P11 ranks 11-50, and absent
from the existing structured and semantic Top10, can safely replace P11 rank 10 when
independent frozen catalog signals provide a material margin. Commit `03ed674`, config
SHA-256 `69da9c40aa6ec32448490e8c454508c3f1d1aa4fa45139d47f49b22e4d327bda`,
and one shared `train_explore --limit 100` replay tested the three preregistered variants.
The aggregate is
`experiments/fast_track/action_oracle_v1/train_explore-smoke-100-family2-03ed674-aggregate.json`.

| Action | Admission evidence | HR@10 / delta | m->h | h->m | Net | Activation turns / sessions | Scenario / taxonomy span | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `P11_EVIDENCE_NOVEL_SLOT10` | weighted P11 evidence + two support groups | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |
| `HARD_CLAUSE_NOVEL_SLOT10` | unique long hard-clause field match | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |
| `TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10` | same unique lexical and constraint argmax | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | 0 / 0 | ADVANCE_FAMILY |

The baseline was HR@10 `0.940000`, MRR `0.676040`, MTTC `3.2`, and TechnicalScore
`0.828812`. Candidate recall remained C10/C20/C50/C100 =
`0.940/0.950/0.970/0.980`. Wall time was `80.753492s`; maximum single-worker lifetime
RSS was `484,474,880` bytes with parent RSS excluded. Network attempts, full-catalog
searches, semantic/rewrite failures, P11 invariant failures, and Family 2 score
fail-closed turns were all zero.

The aggregate-only scoring funnel covered all `1,000` turns and computed `50,000`
candidate breakdowns through `5,000` sidecar fetches, with at most `10` rows per fetch.
It observed a non-category visible preference on `939` turns and a four-or-more-term hard
clause on `68` turns. Despite complete inputs, all three admission guards returned exact
KEEP_P11 on every turn. The artifact does not emit per-guard rejection counts, so it
supports the bounded statement that the jointly preregistered novelty, safety, uniqueness,
and margin requirements never completed; it does not identify one threshold as the sole
bottleneck.

The source-only budget variant was rejected before execution: only `3/50,000` catalog
rows contain a price field, so it could not form a meaningful admission experiment. It
was replaced before the shared replay by the two-signal consensus action. No label, target,
blind trace row, network call, or full-catalog semantic search was used for this decision.

The one allowed targeted invocation ran `81` tests: `80` passed and one runner negative
test failed only because the correct earlier C50 guard emitted a different error string
than the assertion expected. The assertion was corrected without a second test invocation,
preserving the one-test-per-batch rule. This is recorded as a verification limitation,
not represented as an all-green run.

No individual action satisfies activation > 0 or any HR gate, so `limit=200` is forbidden.
Family 2 is closed after three materially distinct variants. Decision: `ADVANCE_FAMILY`;
next implement a lightweight target-blind router over already-computed safe actions without
opening selection or changing the frozen evaluator protocol.

## Family 3 completion: lightweight target-blind C50 router

Falsifiable hypothesis: visible constraint state, dual auxiliary boundary agreement, or a
recent override can route one candidate from the complete P11/structured/semantic C50
rankings into slot 10 without inheriting the known harms of a full rerank. All challengers
were required to be absent from both auxiliary Top10 lists, preserve P11 ranks 1-9, and
perform exactly one safe membership swap. Commit `842fa3b`, config SHA-256
`e58eb99d558ccb57352be05fd1a933d8d0b7fc7de30586f75ee1e6af285f14b9`, and
the sole shared `train_explore --limit 100` replay tested all three frozen actions. The
aggregate is
`experiments/fast_track/action_oracle_v1/train_explore-smoke-100-family3-842fa3b-aggregate.json`.

| Action | Route context | HR@10 / delta | m->h | h->m | Net | Activation turns / sessions | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10` | >=2 visible non-category preferences or >=4 hard terms | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | CLOSE_ROUTE |
| `DUAL_BOUNDARY_CONSENSUS_SLOT10` | novel candidate at both auxiliary ranks 11-15 | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | CLOSE_ROUTE |
| `RECENT_OVERRIDE_RANK_FUSION_SLOT10` | goal-version age 0-1 turns after override | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | CLOSE_ROUTE |

The baseline remained HR@10 `0.940000`, MRR `0.676040`, MTTC `3.2`, and
TechnicalScore `0.828812`; candidate recall remained C10/C20/C50/C100 =
`0.940/0.950/0.970/0.980`. Wall time was `59.270069s`, and maximum single-worker
lifetime RSS was `485,851,136` bytes with parent RSS excluded. The one allowed targeted
test invocation passed `94/94`. Network attempts, full-catalog searches, semantic/rewrite
failures, P11 invariant failures, Family 2 score failures, and Family 3 unexpected compute
failures were all zero.

All three routers returned `invalid_context` on all `1,000` turns, while the shared scorer
still produced all `50,000/50,000` C50 candidate breakdowns through `5,000` bounded
sidecar fetches. Therefore the replay establishes zero activation, not a comparison of the
three rank margins. Source review suggests the exact twelve-decimal
`total == relevance + tie_bonus` revalidation is a plausible fail-closed bottleneck because
the three values are rounded separately, but the identifier-free aggregate cannot prove
which context clause rejected each turn. Per the post-run decision rule, this is not grounds
for a repaired rerun or another hand-written variant.

`limit=200` is forbidden because every individual action has activation `0`, m->h `0`, and
HR delta `0`. The hand-written guarded-router route is closed. The next stage is a learned,
target-cluster cross-fitted rescue-vs-harm router: target membership may define training
labels but can never be a runtime feature. Its objective is `miss_to_hit - lambda *
hit_to_miss`; evaluation must keep HR@10 membership separate from HR@1/MRR ordering.
After that, prioritize `GUARDED_C100_SLOT10`; embedding may expand/fuse the candidate pool
but must not directly rerank Top10. No full split is authorized at this stage.
