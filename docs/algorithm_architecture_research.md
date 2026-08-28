# P4 Target-Blind Architecture Search

Last updated: 2026-08-28 SGT.

## Decision summary

P4 selected `R08.coverage_cascade` on the frozen public-target-disjoint corpus and has
now promoted it into the served Agent. The explicit `control` mode remains weighted RRF;
the default `coverage` mode orders that same candidate set by distinct visible-query-term
coverage and preserves fused rank on ties. The existing sparse control had fused recall
`0.995` at 50 on the released-public audit, and 11 of its 12 public misses entered that
Top-50 pool. That local evidence motivated shortlist discrimination first; it does not
prove that semantic retrieval is unnecessary or predict performance on the organizer-
private 800 sessions.

The research basis and official sources are frozen in `report-source.md`. RRF, MMR,
conversational context selection, dense retrieval, learned sparse retrieval,
late-interaction, and cross-encoder work are used only as design evidence. Every local
claim still requires a target-blind evaluator artifact.

## Frozen experimental protocol

- Selection corpus: 200 deterministic catalog-derived sessions with 200 unique targets,
  zero overlap with all released-public targets, and scenario mix 80/80/30/10.
- Fixed policy: `question_policy=fast`, `rerank_mode=off`, Top-10, at most 10 turns.
- Agent inputs: profile plus visible messages only; never target, sample ID, scenario,
  intent card, simulator behavior, prior result, or target rank.
- Control: `C00.control_rrf`, required to be response-equal to an explicit
  `Agent(retrieval_mode="control")`. The default Agent changed only after selection and
  promotion gates completed.
- A design counts only when it is non-control, contract-clean, fully evaluated, has at
  least one activation, and changes at least one Top-10 output.
- Promotion requires no HR, MRR, TechnicalScore, MTTC, scenario-HR, or hit-to-miss
  regression versus control. Aggregate score alone cannot override a failed gate.
- The raw-score winner, eligible winner, and deterministic confirmation are reported
  separately. The released public set is not used to choose among variants.

Runner:

```powershell
python scripts/evaluate_architectures.py `
  --derived-count 200 `
  --seed track4-p1-product-disjoint-v1 `
  --variants all `
  --confirm-top 3 `
  --output experiments/p4_architecture_search.json
```

## Frozen architecture registry

| ID | Mechanism | Hypothesis and boundary |
| --- | --- | --- |
| C00 | Control weighted RRF | Exact pre-promotion/explicit-control sparse ranking; not counted as a new experiment |
| R01 | Field RRF | Independent title/category, feature/detail, and description/store routes can recover field-specific evidence |
| R02 | Category guard | A strict category-field route can suppress cross-category lexical matches, with deterministic fallback |
| R03 | Turn RRF | Versioned per-turn routes can preserve useful evidence without flattening all terms into one query |
| R04 | Phrase route | Exact multiword slot and visible n-gram evidence can distinguish products sharing individual tokens |
| R05 | Alias expansion | A low-trust catalog-domain alias route can reduce lexical mismatch while preserving the original query |
| R06 | Rare anchor | The lowest-document-frequency visible term can anchor precision when frequent attributes dominate |
| R07 | CombSUM BM25 | Normalized raw route scores can discriminate differently from reciprocal rank |
| R08 | Coverage cascade | Distinct visible-term coverage can prioritize candidates satisfying more of the expressed need |
| R09 | Slot filter/relax | Known negative conflicts are never backfilled; positive hard constraints relax deterministically, missing metadata remains unknown, and fewer than ten results are permitted |
| R10 | Candidate carry-over | A decayed prior shortlist can stabilize refinement, but only inside the current goal version |
| R11 | Browsing MMR | Visibly open browsing can benefit from target-blind catalog-aspect diversity; buying/hard constraints bypass it |
| R12 | Numeric budget | Visible under/over/around constraints can rank known prices while reserving space for unknown prices; activation is measured rather than inferred from intent-card fields |
| R13 | Intent router | Visible hard constraints, visible open browsing, and ordinary refinement can use different experts without hidden scenario labels |
| R14 | Borda fusion | Normalized route-relative rank votes provide a genuinely distinct aggregation control from RRF and raw-score CombSUM |

Changing only a depth, weight, synonym list, or MMR penalty does not create another
architecture. Unique mechanism and stage-graph fingerprints are enforced by tests.

## Safety and reproducibility gates

The runner refuses to compare a control with contract errors or an incomplete session
set. It validates response keys, allowed question attributes, catalog membership,
uniqueness, Top-10 size, finite optional scores, and usage shape. Contract-invalid
variants cannot count toward the ten-experiment requirement or enter confirmation.

Each artifact records Git branch/commit/dirty state, Python/platform/SQLite versions,
catalog/public/derived hashes, all direct source hashes, timing, activation/fallback
statistics, complete session results, scenario deltas, hit-to-miss changes, and repeated
functional hashes. The full matrix must be run from a clean committed tree.

## Next architecture wave

R08 passed the isolated local gate, so the next wave must treat it as the served baseline
and must not tune against repeated public-session outcomes. First split target-blind
derived failures into candidate-truncation misses versus in-pool Top-10 discrimination.
If in-pool errors dominate, test one bounded structured/shortlist mechanism; if genuine
sparse truncation dominates, test an offline, legally distributable semantic route such
as small E5 dense retrieval or SPLADE-style learned sparse expansion with bounded hybrid
fusion. A shortlist cross-encoder or late-interaction reranker follows only when the
measured bucket justifies it. Every model path must declare model/version/license, asset
hash, disk/RAM, latency, token/network behavior, and an offline fallback before public
gating.

## P5 pre-registered guarded session PRF

P5 is isolated from the frozen P4 matrix and treats the promoted R08 coverage route as
its control. The selection corpus is a second deterministic 200-session catalog-derived
set with sample SHA-256
`0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c`. Its 200 unique
targets overlap neither the 200 released-public targets nor the 200 P1-derived targets;
its 80/80/30/10 scenario mix is fixed. This is local stress data, not organizer-private
data or a private-distribution proxy.

The following registry and parameters were frozen before reading any P5 metric:

| ID | Role | Output behavior |
| --- | --- | --- |
| `P5.C00.r08_coverage` | control | Exact served `coverage/off` Agent behavior |
| `P5.S00.prf_shadow` | diagnostic | Computes the full proposal but must return C00 output exactly |
| `P5.R01.guarded_session_prf` | sole active candidate | May replace only the Top-10 tail under every guard below |

For each turn, R01 takes at most five current coverage-ranked seeds whose original-query
coverage is at least `max(2, maximum_seed_coverage - 1)`. At least three seeds and route
agreement are required. Feedback is extracted only from title, categories, features,
and details; store tokens dynamically suppress likely brand terms, and store/description
never supply feedback. A term must be ASCII alphabetic, at least three characters, absent
from the original and excluded terms, supported by at least three seeds and 60% of the
available seeds, present across at least two field groups, have catalog document frequency
at most 2%, and occur in at least three non-seed documents. At most four terms survive.
Term score is BM25-style IDF times normalized reciprocal-log seed-rank support:

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
feedback(t) = idf(t) * sum_support(1/log2(seed_rank+1))
                         / sum_all_seeds(1/log2(seed_rank+1))
```

The second FTS route is exactly `(Q1 OR ...) AND (F1 OR ...)`, uses the existing field
weights and a fixed depth of 120, and therefore cannot retrieve a product from feedback
terms alone. Different-query BM25 values are not combined. The target-blind proposal over
the union instead uses ranks:

```text
C(d) = distinct original-query terms matched by d
B(d) = 1 / (60 + fused_rank(d))
P(d) = 0.15 / (60 + prf_rank(d))
proposal key = (-C(d), -(B(d)+P(d)), fused_rank, prf_rank, parent_asin)
```

The served proposal preserves the original first nine results and admits at most one
newcomer at rank 10. That candidate must rank ahead of the incumbent in the proposal,
match at least two selected feedback terms, match no excluded term, and meet the
incumbent's original-query coverage. A PRF-only candidate requires strictly higher
coverage. A same-coverage candidate already in the base pool requires both broad and
strict evidence when strict is available; otherwise it must be in broad Top-30.

The frozen runner first evaluates the actual served `Agent(coverage/off)` and requires
C00's complete 200-session evaluator hash and ordered response-trace hash to match. S00
must also match both C00 hashes exactly and cannot win. R01 is eligible only with an
effective output change, clean and complete
contract, non-decreasing HR/MRR and every scenario HR, non-increasing MTTC, strictly
higher TechnicalScore, zero hit-to-miss changes, and evaluation time no greater than
1.30x C00. An eligible R01 must repeat with an identical complete functional hash. The
released public set is read only to prove target exclusion and is not evaluated by the
P5 selection runner. If R01 fails, C00 remains served and the parameters will not be
tuned on this same P5 corpus.

### P5 frozen result

The selection ran once from clean commit `ac8e2217f2558a2e9fe0792e7f77ce1f2adff7e1`.
The actual served reference and C00 matched in both complete evaluator hash
`000654e6459f58a2483785daf3bcbeedbbe9dcb058284aa0da5d23d52cebc420` and ordered
response-trace hash
`f8727978dbaa8bffaf944e93684e76169fe6aefa371759fa5f7232bcfdb16525`.
S00 matched those outputs exactly. The frozen metrics were HR `0.940000`, MRR
`0.593937`, MTTC `3.080000`, and TechnicalScore `0.806581`.

R01 computed 48 PRF routes over 604 turns and made 21 guarded Top-10 tail changes, but
all 200 per-session official contributions were unchanged: no hit/miss, hit turn, or
target-rank improvement or regression occurred. Its aggregate metrics therefore tied
C00 exactly, failing the strict TechnicalScore-improvement gate. It also took
`30.961991s` evaluation time versus C00's `19.672008s` (`1.574x`), failing the `1.30x`
time gate; P95 response latency was `85.9451ms` versus `62.5151ms`. The active response
trace changed, as expected, but the evaluator-result hash remained equal because none of
those tail replacements changed target outcomes.

The decision is `retain_control_active_rejected`; no repeat confirmation or released-
public evaluation was run, and the served Agent remains R08 coverage. The ignored full
artifact is `experiments/p5_prf_selection.json`, SHA-256
`d0fce8879cd19f0853aeb632b56195a7496b690939581e8f4c731d4a0795d90f`.

## Frozen 200-session result

The full matrix ran from clean commit `e5d0d4966d01da9932d835cb3a754475b6fa13e2`.
The input sample hash was
`38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae`,
released-public target overlap was zero, and the ignored complete artifact SHA-256 is
`bedf4c8048186a9ca9d64a64fb9a8ee7184c5810ff13e5e69e138f15faa5e177`.

| ID | HR@10 | MRR | MTTC | Score | Hit→miss | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C00 | 0.935 | 0.630183 | 3.185 | 0.812855 | 0 | control |
| R01 | 0.870 | 0.549284 | 3.745 | 0.744885 | 13 | reject |
| R02 | 0.445 | 0.288804 | 7.070 | 0.387741 | 98 | reject |
| R03 | 0.810 | 0.480129 | 4.300 | 0.683039 | 25 | reject |
| R04 | 0.890 | 0.583438 | 3.595 | 0.768131 | 10 | reject |
| R05 | 0.940 | 0.642480 | 3.165 | 0.819444 | 1 | reject |
| R06 | 0.915 | 0.624448 | 3.400 | 0.796834 | 4 | reject |
| R07 | 0.930 | 0.646994 | 3.260 | 0.813898 | 1 | reject |
| R08 | 0.945 | 0.643516 | 3.115 | 0.823255 | 0 | **eligible** |
| R09 | 0.870 | 0.568663 | 3.775 | 0.750099 | 15 | reject |
| R10 | 0.935 | 0.511970 | 3.255 | 0.775991 | 0 | reject |
| R11 | 0.935 | 0.630813 | 3.200 | 0.812744 | 0 | reject |
| R12 | 0.935 | 0.625738 | 3.185 | 0.811521 | 0 | reject |
| R13 | 0.780 | 0.502887 | 4.645 | 0.667966 | 32 | reject |
| R14 | 0.845 | 0.565756 | 3.975 | 0.732727 | 18 | reject |

The raw runner mechanically reported all 14 non-control variants as effective and
contract-clean. A subsequent semantic activation audit found that R12's only activation
parsed `21.25inch-25inch` head circumference as an "around 21.25" price. That is a
measurement-to-price regex false positive, not a genuine numeric-budget activation. The
regex was corrected to require price context and reject measurement units. The honest
count is therefore 13 semantically independent effective designs, which still exceeds
the required minimum of ten. A hygiene-only rerun on the same 200 product-disjoint
sessions produced `activations=0`, `output_changes=0`, and metrics exactly equal to
control. Its ignored artifact `experiments/p4_r12_hygiene.json` has SHA-256
`6428a2f4049f0b17dc7d9d6287716803aee596ff2e6d383ed625d86e84a7324f`.
This rerun confirms the classification; it does not rerun selection or choose a new
winner. The raw matrix artifact remains unchanged as historical evidence.

R08 is the sole eligible winner. Against control it produced zero per-session official-
score regressions, five improvements, zero hit-to-miss changes, two miss-to-hit changes,
one earlier hit, and three rank improvements. Scenario HR was non-regressive: Boundary
`0.8→0.9`, Browsing `0.925→0.925`, Buying `0.9375→0.95`, and Intent Override
`1.0→1.0`. Its repeated complete functional hash matched exactly:
`3e0b1211179748c9b0581c840d8ad23973045d863f3311f70738b9cd28e71ba7`.

The matrix timing for R08 was 22.369 seconds versus 26.111 seconds for control in that
single sequential process, but this is not a controlled resource claim.

## Promotion and served-implementation verification

After selection was frozen, R08 passed the released-public, all-phrase, strict-contract,
two-run determinism/resource-measurement, no-key, and target-blind reference-bridge
checks. No organizer numeric resource threshold is published; this proves measurement
completeness and reproducibility, not compliance with an unknown limit. The
served `Agent(retrieval_mode="coverage", rerank_mode="off")` exactly reproduces the
frozen winner across canonical plus eight registered phrase suites:

| Evidence | Control | Promoted coverage | Delta |
| --- | ---: | ---: | ---: |
| HR@10 | 0.940000 | 0.945000 | +0.005000 |
| MRR | 0.605258 | 0.606175 | +0.000917 |
| MTTC | 3.375000 | 3.335000 | -0.040000 |
| TechnicalScore | 0.804077 | 0.807652 | +0.003575 |

The paired released-public comparison has zero hit-to-miss and one miss-to-hit change.
All nine suite-result hashes match the frozen winner, the complete response trace and
the broad/strict/fused/final routes are exact, the strict response contract is clean,
and the promoted two-run trace is deterministic. The verification artifact is
`experiments/p4_promoted_verification.json`, SHA-256
`8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.

The pre-promotion architecture artifacts remain the immutable selection evidence. The
post-promotion `architecture_lab.py` necessarily changed to pin its control mode and use
the shared coverage helper, so an old gate that requires that working-tree file to be
byte-identical to the selection commit is now a legacy/frozen-evidence check, not proof
of the served implementation. `scripts/verify_promoted_agent.py` supplies the explicit
bridge: it compares frozen/reference artifacts with independent runs of the actual
served Agent and checks response, route, contract, provenance, determinism, resource,
and no-key invariants. These are public/local verification claims only; no private-800
result is available.
