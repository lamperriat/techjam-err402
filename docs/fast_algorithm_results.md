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
| `VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | IMPLEMENTATION_FAILURE: unexecuted |
| `DUAL_BOUNDARY_CONSENSUS_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | IMPLEMENTATION_FAILURE: unexecuted |
| `RECENT_OVERRIDE_RANK_FUSION_SLOT10` | hand-written C50 router | train prefix 100 | 0 | 0/0/0 | 0.676040 | 3.2 | 59.270069s | IMPLEMENTATION_FAILURE: unexecuted |

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
| `VISIBLE_CONSTRAINT_RANK_FUSION_SLOT10` | >=2 visible non-category preferences or >=4 hard terms | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | IMPLEMENTATION_FAILURE |
| `DUAL_BOUNDARY_CONSENSUS_SLOT10` | novel candidate at both auxiliary ranks 11-15 | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | IMPLEMENTATION_FAILURE |
| `RECENT_OVERRIDE_RANK_FUSION_SLOT10` | goal-version age 0-1 turns after override | 0.940 / 0 | 0 | 0 | 0 | 0 / 0 | IMPLEMENTATION_FAILURE |

The baseline remained HR@10 `0.940000`, MRR `0.676040`, MTTC `3.2`, and
TechnicalScore `0.828812`; candidate recall remained C10/C20/C50/C100 =
`0.940/0.950/0.970/0.980`. Wall time was `59.270069s`, and maximum single-worker
lifetime RSS was `485,851,136` bytes with parent RSS excluded. The one allowed targeted
test invocation passed `94/94`. Network attempts, full-catalog searches, semantic/rewrite
failures, P11 invariant failures, Family 2 score failures, and Family 3 unexpected compute
failures were all zero.

All three routers returned `invalid_context` on all `1,000` turns, while the shared scorer
still produced all `50,000/50,000` C50 candidate breakdowns through `5,000` bounded
sidecar fetches. Therefore the replay establishes an implementation failure and zero
executed action, not an algorithmic No-Go and not a comparison of the three rank margins.
Source review suggests the exact twelve-decimal
`total == relevance + tie_bonus` revalidation is a plausible fail-closed bottleneck because
the three values are rounded separately, but the identifier-free aggregate cannot prove
which context clause rejected each turn. Per the post-run decision rule, this is not grounds
for a repaired rerun or another hand-written variant.

`limit=200` is forbidden because every individual action has activation `0`, m->h `0`, and
HR delta `0`. This closes those three broken implementations under the frozen run protocol;
it does not reject the underlying rank-fusion hypothesis. The next stage is a learned,
target-cluster cross-fitted rescue-vs-harm router: target membership may define training
labels but can never be a runtime feature. Its objective is `miss_to_hit - lambda *
hit_to_miss`; evaluation must keep HR@10 membership separate from HR@1/MRR ordering.
After that, prioritize `GUARDED_C100_SLOT10`; embedding may expand/fuse the candidate pool
but must not directly rerank Top10. No full split is authorized at this stage.

## Family 4A: target-cluster cross-fitted C50 risk router

The first learned-router stage consumed only the already-closed historical
`train_explore-full` aggregate and its four verified blind traces. It did not launch an
Agent worker or open calibration, selection, or confirmation. The target-blind feature
table was built and hashed before the proxy label file was loaded. Product identity was
used only after that boundary to create atomic rescue/harm labels and deterministic
five-fold product-cluster groups; no identity, scenario, taxonomy, difficulty, ordinal,
or future-turn value enters the 19 runtime features.

The proposal universe was the complete R08/P11-preserved C50 tail, ranks 11-50. A
two-head cluster-weighted ridge model estimated rescue and harm scores for a single slot-10
swap. Nested group cross-fitting selected its regularization and gate using the integer
objective `miss_to_hit - 2 * hit_to_miss`. The membership policy always preserved P11
ranks 1-9; MRR was not an optimization feature and was rebuilt separately with the frozen
official formulas.

| Evidence | Baseline | Cross-fitted policy | Delta / transition |
| --- | ---: | ---: | ---: |
| HR@10 | 0.947500 | 0.947500 | 0; 0 m->h, 0 h->m |
| MRR | 0.674928 | 0.674928 | 0 |
| MTTC | 3.160500 | 3.160500 | 0 |
| TechnicalScore | 0.833018 | 0.833018 | 0 |
| Activation | - | 137 turns / 49 sessions | neutral swaps only |

The closed feature table contained `649` atomic rescue-positive rows and `0` atomic
single-turn harm-positive rows. Four outer folds selected a zero-rescue conservative gate;
one outer-training partition observed five inner rescues, but none transferred to its
held-out product clusters. The deployable conclusion is therefore not "the model never
ran": it activated, but its candidate choice did not generalize to a missed target.

The successful identifier-free research artifact is local-only at
`experiments/fast_track/counterfactual_router_v1.json`, SHA-256
`c46ec1cf81fd08d02f70ce2e8be8a2b6abdb560841c707313452d11e5de8b23d`, size
`11,788` bytes. Its feature-table SHA-256 is
`a9c83ea2252f1860eb7ad3610ab3810e07e0aef8e4de0f89325378fb91c18fde` and it is
explicitly marked `OOF_RESEARCH_ONLY_NOT_RUNTIME_DEPLOYABLE`. One preceding execution
completed fitting but failed before artifact creation because the output privacy audit
correctly rejected a provenance key containing the word `identifier`; renaming only that
key produced the successful deterministic replay.

Decision: do not tune rank, score, or margin thresholds and do not integrate this C50
model into runtime. Advance to the preregistered C100-only candidate-expansion test over
ranks 51-100 using the same target-cluster risk protocol. This is not a fourth hand-written
guard. The existing embedding may be used only as a candidate-local feature/fusion source;
it may not replace or reorder P11 Top10. No new full-split Agent run is authorized.

## Family 4B: guarded learned C100-only expansion

The trainer was generalized without changing the C50 checkpoint semantics so that one
second preregistered band could be evaluated: exact sparse C100 ranks 51-100. This is the
`GUARDED_C100_SLOT10` design boundary: P11 ranks 1-9 stay exact, at most one learned-risk
challenger can occupy slot 10, stable R08 order breaks ties, and every non-activation is an
exact KEEP_P11 fallback. It is a learned counterfactual route, not a fourth hand-written
threshold rule.

The historical blind trace contains complete C100 order but does not contain semantic or
P11 candidate-score rows for ranks 51-100. The experiment therefore used only signals that
can be reconstructed honestly: C100 rank/depth, previous-turn C100 presence/rank, visible
turn, incumbent expert support, and expert-set agreement. Missing C100 embedding scores
were not imputed. The same BGE was not run over the full 50,000-item catalog: frozen P7
already showed sparse recall `198/200`, Dense@120 `115/200`, sparse-plus-dense still
`198/200`, wall ratio `1.5521`, and RSS ratio `2.7595`.

| Evidence | Baseline | C100-only cross-fitted policy | Delta / transition |
| --- | ---: | ---: | ---: |
| HR@10 | 0.947500 | 0.947500 | 0; 0 m->h, 0 h->m |
| MRR | 0.674928 | 0.674928 | 0 |
| MTTC | 3.160500 | 3.160500 | 0 |
| TechnicalScore | 0.833018 | 0.833018 | 0 |
| Activation | - | 0 turns / 0 sessions | fail-closed |

The identifier-free C100 feature table contained `1,000,000` proposal rows, only `67`
atomic rescue-positive rows, and `0` atomic single-turn harm-positive rows. Every outer
fold and the final inner cross-fit selected objective `0`, rescue `0`, harm `0`, and zero
activation. Historical session recall bounds explain the scarcity: C50/C100 are only
`0.991/0.993` on the 2,000-session train split, so ranks 51-100 add four reachable
sessions beyond C50 before routing error is considered.

The local-only artifact is
`experiments/fast_track/counterfactual_router_c100_only_v1.json`, SHA-256
`2969971ae2e4efff33bc44a2c1bb8c06f6fa4d5a5bd7f95f368c8372daf83ea6`, size
`11,528` bytes. Its feature-table SHA-256 is
`42fe67a936b806ed15915259b644094890c59fda2693a741e6b1e02aefede500`; trainer
SHA-256 is `50aa23d45d313cb31a0358119170a4ace7d4f6e5d745522d5faee7d238c96248`.
It is also marked `OOF_RESEARCH_ONLY_NOT_RUNTIME_DEPLOYABLE`.

One earlier C100 execution was invalidated before evidence closure because the candidate
slice was ranks 51-100 while its numeric rank feature still began at 11. That artifact was
renamed with `invalid_rank-origin` and retained locally for audit. After changing only the
rank origin to `spec.rank_start + offset`, an independent static review found no remaining
C100-band correctness blocker and the corrected OOF replay produced the result above.

## Cached hard-case triage checkpoint

The closed 2,000-session trace was converted once into the gitignored numeric cache
`experiments/fast_track/p12_counterfactual_cache_v1.npz`: `142,611,522` bytes,
SHA-256 `bc4985e3c5f84e7512163c00cb7f49fbe9f1e45752dba42dfa71b1394639d3af`.
Feature construction, label/fold joining, sufficient-statistic aggregation, writing, and
hashing took `16.166547s`. The tracked manifest is
`configs/p12_counterfactual_cache_v1.manifest.json`; no product, session, text, or
reversible target value is serialized in the cache.

Hard-case triage contained all `105` baseline misses plus `5` target-cluster-matched hit
controls. C50 activated on `46` sessions / `284` turns in `1.309093s`, but produced
`0` miss-to-hit and `0` hit-to-miss. C100-only completed in `1.393990s` with zero
activation and the same `0/0` transition. Both iterations met the one-minute target, but
neither had positive net rescue, so the full 2,000-session cache-only OOF gate remained
closed. No Agent, calibration, selection, confirmation, or public evaluator was run.

## Family 1-4 evidence boundary

No individual action qualifies for `limit=200`: the three broken Family 3 implementations
executed no valid action; the learned C50 policy activated but produced no transition; and
the learned C100-only policy again had zero activation. Calibration was not used for
fitting, gating, or evaluation in Family 4, and selection/confirmation were not opened.
There is therefore no model to freeze for calibration and no basis for runtime integration.

The bottleneck is now localized. Sparse recall provides candidates for most misses, but
the historical target-blind feature contract does not identify which tail member is the
target across held-out product clusters. Increasing candidate depth alone adds very little
ceiling, and the already-tested full-catalog BGE does not add recall. Continuing to adjust
rank, utility, margin, ridge, or gate thresholds on these labels would be post-hoc fitting
and is closed.

A materially new future stage requires new authority and new evidence, not another rule:

1. capture complete target-blind CandidateScore components and candidate-only C100 cosine
   values before label join, using at most 10 sidecar rows per fetch and exactly 100
   advanced-index embedding rows per turn;
2. keep embedding as candidate expansion/risk evidence only, never as a direct Top10
   reranker;
3. repeat product-cluster OOF on train/explore, freezing membership independently from any
   later HR@1/MRR ordering model;
4. touch calibration only after a frozen OOF policy has positive net rescue, and authorize
   `limit=200` only when one individual action has activation and m->h > h->m.

Under the current instruction not to add a full split or another rule variant, Family 1-4
is complete with a reproducible negative boundary rather than a deployable HR action.

## SR-v1.9 fold-safe deployable artifact

The exact semantic-off projection repeat closed the reproducibility gap, after which the
frozen `ndcg_d4_lr003` ranker was paired with rescue and reciprocal-rank-regret admission
heads. RR multiplier is fixed at `1.0`; direct hit-loss is absent because its isolated
training label has no positives. Fold-specific raw probabilities are not reused. Each
outer fold selects an inner-safe activation quantile, and the final median quantile is
mapped to the unlabeled full-model utility distribution.

The deployment-form nested OOF result retains HR@10 `0.9715`: `48` miss-to-hit, `0`
hit-to-miss, fold net `7/13/8/7/13`, MRR `+0.001933`, MTTC `-0.1045`, and
TechnicalScore `+0.014670`. It activates `6,573` turns in `1,251` sessions. This equals
the current candidate HR and is a deployment checkpoint, not a new algorithm promotion
above `0.9715`.

The tracked artifact is `starter/assets/small_ranker_fold_safe_v1.json`, `256,639` bytes,
SHA-256 `f8d0b6c0e402edeb34b1e35119c5295449888bc1be713607e88337fa874d16dc`.
Its independent render is byte-identical; identity-shaped token count and forbidden target
key count are both zero. XGBoost versus pure-Python parity is exact for all sampled C100
orders (maximum score error `3.5214e-6`), and serialized admission decisions match the
vectorized reference on all `20,000` turns. The runtime remains off by default and falls
back completely to P11/R08 on missing/corrupt artifacts, invariant failures, exceptions,
or budget failures.

A target-free handwritten runtime benchmark imported no evaluator or training library and
opened no dataset/label. All `110` turns completed with zero fallback: cold Agent init
`2.628s`, one 10-turn session `3.141s`, 10-session batch `18.340s`, turn P50/P95
`185.83/277.76ms`, and process peak RSS `166,883,328` bytes. This benchmark is deployment
evidence only and is not promotion evidence. Full hashes and failure-attempt provenance
are in `configs/small_ranker_v1_9.deployable_artifact.manifest.json`.

## SR-v2.0 remaining-miss attribution

The frozen activation and chosen-candidate hashes were reproduced exactly before target
labels were joined in a posthoc-only analyzer. Of the 57 remaining misses, 14 (`24.56%`)
are absent from C100, 35 (`61.40%`) are ranker failures, 8 (`14.04%`) are admission
failures, and 0 meet the deliberately strict exact-evidence ambiguity lower bound. No
session or product identifier is serialized in the aggregate result.

Candidate recall is C10 `0.9475`, C20 `0.9715`, C50 `0.9910`, and C100 `0.9930`.
C200 is not present in the frozen trace and was not inferred. Thus candidate reachability
does not mathematically rule out 0.99, but the policy must rescue 37 of only 43 actionable
misses (`86.05%`) without one hit-to-miss. Candidate expansion is not the first bottleneck:
31 of 35 ranker failures are already within ranks 11-50, while only four are at 51-100.

Ranker failures concentrate in clothing (`17`) and shoes (`15`), browsing (`16`) and
buying (`13`), with fold counts `16/8/1/8/2`. The target already has fused support in all
35 cases, strict support in 29, broad support in 26, and semantic support in only one;
13 have a first-reachable category-conflict feature. This makes another embedding-first
candidate expansion unlikely to address the dominant error. The next single-variable
test is the existing pairwise ranker under the identical semantic-off projection and the
unchanged fold-safe two-head admission protocol.

## SR-v2.1 pairwise semantic-off projection

The already-trained `pairwise_d4_control` fold models were scored twice on the exact
semantic-off tensor. Both 8 MB OOF score files are byte-identical (SHA-256
`1765f60c3f111f751e8d0c133bbbd93d2a1e174db24b2c4c64c80aea66a4778b`), and a
1000-row reference audit exactly reproduced the original early-stopping-aware scores and
C100 orders. Nested admission activation also repeated exactly.

Against P11, pairwise reaches HR `0.9735`, `52/0` rescue/harm, MRR `0.677328`, MTTC
`3.057`, and TechnicalScore `0.848808`. That aggregate is not promotable. Relative to the
current `0.9715` policy it gains nine misses but loses five existing hits; fold net is
`5/-1/3/-1/-2`, folds 1 and 3 regress MRR, folds 1/3/4 regress MTTC, and aggregate MTTC
regresses `+0.001`. Pairwise alone is therefore closed without weight or threshold tuning.

The useful evidence is proposal complementarity: it proposes a correct candidate in 13
of the current 57 misses. The next materially different mechanism is a hard-case residual
ranker trained from OOF-derived target-versus-current-chosen pairs under full outer/inner
product-family cross-fitting; the current fold-safe admission protocol remains unchanged.

## SR-v2.2 hard-case residual ranker

The preregistered residual model used OOF-derived target-versus-current-choice pairs,
full product-family outer/inner cross-fitting, fixed hard/control weights `10/1`, and four
fixed extra hard negatives. Its local pair cache contains `7,098` rows, `134` target-blind
features, `1,019` sessions, and `35` current ranker-failure sessions (`3,898,802` bytes).
The complete nested run repeated exactly: activation SHA-256
`bb745e4f7e83f7e7393ced83c1a51f012e3b9f6ca834dd081a82beed785e8931` and chosen-order
SHA-256 `f9ff3ce77f2a58ed0a0165066bd946677ddce7df7e43b31faae4ffeb45052694`.

The formulation is a clear No-Go. It proposes the correct candidate in only one of the
57 current misses. Against P11 it activates `1,949` turns in `380` sessions but changes
no session outcome: HR `0.9475`, `0/0` rescue/harm, MRR `0.674928`, and MTTC `3.1605`.
Relative to the current `0.9715` policy it removes all 48 existing rescues, yielding
HR `-0.024`, MRR `-0.001933`, MTTC `+0.1045`, and fold net
`-7/-13/-8/-7/-13`. The run took `16.516s`, including two exact nested-OOF passes of
`5.349s` and `5.478s`; no Agent, evaluator, held-out split, sweep, or deployable artifact
was opened.

The result is retained locally at
`experiments/fast_track/small_ranker_v2_2/hardcase_residual_20260830T1615.json`, SHA-256
`08796b01cada715e2339f1eecfb34c066dd43379ad1869cf0af4f5a9a514b818`; the tracked
aggregate manifest is `configs/small_ranker_v2_2.hardcase_residual.manifest.json`.
This residual formulation is closed without tuning. The next mechanism must preserve the
current policy by default and treat the complementary pairwise proposal only as a
supplemental action behind a session-aware rescue-versus-regret gate.

## SR-v2.3 supplemental pairwise disagreement gate

Before opening metrics, an independent static review reduced the gate from 31 features
to eight bounded disagreement features, removed cross-objective raw scores, fixed both
cross-ranks as stable rank/91 over the valid slot-10 action set, and normalized each
head's training weight per session. The two fixed logistic heads learn isolated rescue
and composite RR/MTTC regret relative to the complete current `0.9715` trace. KEEP is an
exact current-policy fallback, and all threshold candidates are scored by simulating the
combined ten-turn policy.

The exact repeated nested OOF result is a No-Go. Only fold 1 selects a finite threshold,
activating `124` turns in `32` sessions; all activations are metric-neutral. Relative to
the current policy the result is `0/0` rescue/harm, HR `0.9715`, MRR delta `0`, MTTC
delta `0`, and fold net `0/0/0/0/0`. Four folds select KEEP. Both passes reproduce the
same decisions and selection hash, and the full run takes `8.657s`. No Agent, evaluator,
held-out split, full pairwise model, or runtime artifact was opened.

The posthoc action-family oracle is decisive for scope: pairwise offers an eligible,
non-P11 supplemental target choice in only 13 of the 57 current misses, distributed
`5/0/6/1/1` by fold. Even a perfect target-informed zero-harm router therefore tops out
at HR `0.9780`; it cannot reach `0.99`. This closes feature, weight, multiplier, and
threshold tuning for this proposal family. The next proposal mechanism must address the
broader set of 43 current misses whose targets are already reachable within C100.

The local result is
`experiments/fast_track/small_ranker_v2_3/supplemental_pairwise_20260830T1700.json`,
SHA-256 `d2ed3e8dc8d66bf11e41403c2919aed2ceaa4bdeaa6dd10896da1f715126eaee`;
the tracked aggregate evidence is
`configs/small_ranker_v2_3.supplemental_pairwise.manifest.json`.

## SR-v2.4 fixed semantic-off RRF-3

Three previously zero-harm rankers were fused with fixed equal-weight reciprocal rank
fusion over stable C100 ranks (`k=60`). The two missing semantic-off surfaces were each
projected twice with their original held-fold models and early-stopping limits. Both raw
reference audits have exact scores/orders across all five folds, and both projected score
pairs are byte-identical. A pre-metric audit superseded the first target-free projection
attempt and strengthened the result-to-member provenance binding before any labels or
metrics were opened.

RRF-3 is a strict No-Go. Every outer fold chooses KEEP, so there are zero supplemental
activations and no change from HR `0.9715`, MRR `0.676861`, or MTTC `3.056`. The full
admission pipeline and both probability surfaces repeat exactly. Its target-informed
proposal ceiling is also worse than pairwise: only 11 of 57 current misses are reachable,
distributed `2/2/6/1/0`, for maximum zero-harm HR `0.9770` versus pairwise `0.9780`.

This closes RRF-3 and RRF-6/member/`k`/weight/gate tuning. Three-ranker deployment or
distillation is not justified. The corrected projection and evaluation results are
SHA-256 `4cf40c4148081ec5cde4c252f86898c31611c6eac35e915a4a1f32d6a0a2ab95`
and `c5907946bbd662503bbfb8534f73cc55f3ea0dc541c694fb38e54a288c9acc26`;
aggregate evidence is tracked in `configs/small_ranker_v2_4.rrf3.manifest.json`.

## SR-v2.5 focused allowed-91 LambdaMART

The fixed depth-3 LambdaMART used all `43` C100-reachable current misses and
behaviorally matched rank-10 controls.  The one-shot cache has `631` complete
91-candidate query groups (`331` hard, `300` control), `57,421` rows, and 133
semantic-off target-blind features.  Targets appear only in the separate one-hot
relevance labels.  Cache rows were checked exactly against the projected feature
tensor; relevance, session/turn uniqueness, outer/inner folds, numeric dtypes, and
zero identifier matches were independently revalidated before training.

Five product-family outer-fold models were trained twice from scratch with fixed
XGBoost 1.7.6 parameters.  Both score tensors are byte-identical (SHA-256
`a77bde22c17a57b44a737ef397615e59012fbe999e46e7cbc6e4ba9d1159b776`), all
five first/repeat model hashes match, and reloaded-model sample projections have
zero score error and exact C100 order.  The complete first/repeat passes took
`30.868s` and `28.062s`; scoring P50/P95 was `5.64/6.28ms` per session and the
observed peak working set was `1,338,449,920` bytes.

Stage A is a preregistered No-Go.  The focused proposal can rescue only `11` of
the current 57 misses, distributed `3/0/7/1/0` by outer fold.  Its target-informed
zero-harm ceiling is therefore HR `0.9770`, below the fixed requirement of at
least 14 reachable sessions across at least three folds.  Stage B's 30-model
nested admission was not started.  Ungated application confirms that admission
is not the main missing capability: it yields `11` miss-to-hit but `19`
hit-to-miss, HR `0.9675`, MRR delta `-0.050486`, and TechnicalScore delta
`-0.013746` relative to the current policy.

This closes the exact focused cohort, weighting, tree parameters, and gate without
tuning.  Cache build, training, and posthoc evaluation took `7.102s`, `60.784s`,
and `4.786s`.  No Agent/full evaluator, calibration, selection, confirmation,
public split, external download, full model, or runtime artifact was opened.  The
aggregate evidence is tracked in
`configs/small_ranker_v2_5.focused_lambdamart.manifest.json`.

## SR-v2.6 frozen proposal-overlap diagnostic

A target-informed posthoc diagnostic compared only the already-frozen pairwise,
RRF-3, and focused LambdaMART proposal decisions under the identical supplemental
action definition.  It trained no ranker or gate and serialized no session or
product identifier.  Each individual oracle was reproduced exactly at `13`, `11`,
and `11` current misses.

The families are materially complementary.  Pairwise, RRF-3, and focused have
`5`, `3`, and `4` unique rescue sessions; seven sessions are shared by all three
and only one is shared by pairwise/RRF without focused.  Their union reaches `20`
misses across folds `9/2/7/1/1`, raising the target-informed zero-harm ceiling to
HR `0.9815`.  This exceeds the preregistered 14-session/three-fold direction gate,
so a discrete proposal selector is justified even though score fusion was closed.

The result repeated canonically and completed in `6.603s`.  The next single
mechanism is a target-blind nested rescue-versus-regret selector over deduplicated
proposal actions, with at most one supplemental action per session.  It is not a
claim that `0.9815` is achievable, and this existing proposal portfolio still
cannot reach `0.99` even with a perfect target-informed selector.  Aggregate
evidence is tracked in
`configs/small_ranker_v2_6.proposal_overlap.manifest.json`.

## SR-v2.7 causal portfolio selector

The three frozen proposal surfaces were expanded into `41,437` raw family actions
and deduplicated to `30,647` `(session, turn, candidate)` rows over all 2,000
sessions.  Runtime construction is structurally label-free: the 19 features and
family-support masks are frozen before isolated rescue/regret labels are attached.
Within each turn, ties resolve by utility, family-support count, and lower candidate
ordinal.  The deployed decision simulation scans turns chronologically and locks the
session after the first passing action, so it cannot use a future-turn maximum and
cannot take more than one supplement per session.

The fixed two-head L2 logistic selector passes its conditional direction gate.  Relative
to the current policy it rescues `2` sessions and harms `0`, moving HR@10 from `0.9715`
to `0.9725`, MRR from `0.676861` to `0.676961`, MTTC from `3.056` to `3.052`, and
TechnicalScore from `0.847688` to `0.848298`.  Fold net hits are `0/0/0/1/1`; every
fold has zero hit-to-miss, nonnegative rounded and unrounded MRR change, and nonpositive
MTTC change.  Folds 0 and 2 fail the preregistered independent-positive readiness check
and correctly KEEP.  Folds 1, 3, and 4 select quantiles `0.8125/0.453125/0.46875`,
producing exactly one supplemental action in each of `934` sessions.

Both complete nested passes are exact (`3.309s` and `3.330s`), and the total cached run
takes `14.502s`.  The result is
`experiments/fast_track/small_ranker_v2_7/portfolio_selector_20260830T1625.json`,
SHA-256 `e63008a1305e6b35555d8f9658c19f198c137d24a487da7b50d651688bab47bd`;
tracked evidence is `configs/small_ranker_v2_7.portfolio_selector.manifest.json`.

This is not yet a deployable gain.  The selector is nested only conditional on three
already-frozen OOF proposal surfaces; meta-training rows can contain upstream models
that saw a meta-held family.  Therefore no full model, artifact, or evaluator run is
authorized.  The next experiment must regenerate all upstream proposals inside each
meta outer/inner training domain and apply the frozen v2.7 selector without tuning.
The proposal union's target-informed ceiling remains HR `0.9815`, so even a successful
strict restack will not by itself reach the project-level `0.99` objective.
