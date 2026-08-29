# Official Requirements and Compliance Audit

Last verified: 2026-08-29 SGT.

This document is the rules-first boundary for implementation and experiments. It
separates organizer-published requirements from local safeguards, inferred risks, and
still-unpublished final-run details.

## Source precedence

1. [TikTok TechJam 2026 Official Rules](https://tiktoktechjam2026.devpost.com/rules)
   control eligibility, timing, submission, judging, and conflict resolution.
2. The Track 4 [Agent contract](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/agent_api_contract.json),
   [evaluation config](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/evaluation_config.json),
   and [official evaluator](https://github.com/TechJam2026/techjam-conversational-search/blob/main/evaluator/local_evaluator.py)
   control the executable protocol and objective metrics.
3. The Track 4 [competition specification](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/competition_specification.md),
   [submission rules](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/submission_rules.md),
   and [official repository README](https://github.com/TechJam2026/techjam-conversational-search)
   define track scope and participant deliverables.
4. Local plans, research reports, historical Agent conversations, and derived corpora
   cannot override those sources.

The Official Rules state that they prevail when challenge materials conflict. Any
unresolved conflict must be raised with the organizer rather than silently interpreted.

## Frozen Track 4 contract

| Area | Requirement | Repository policy |
| --- | --- | --- |
| Data | Frozen 50,000-product catalog, 200 public sessions, 800 organizer-private sessions | Never edit catalog/public labels; never claim access to the private 800 |
| Entry point | Export `Agent`; helper Python modules, small configs, and lightweight required assets are allowed | The project is not restricted to editing only `starter/agent.py` |
| Calls | `reset(session_id, user_profile)`, then `respond(session_id, user_message, turn, top_k)` with turns 1--10 and official `top_k=10` | Validate the strict contract even where the public evaluator is permissive |
| Output | String `message`; allowed `ask_attribute` or `null`; ordered recommendation objects; optional non-negative usage | Return at most 10 catalog-backed unique `parent_asin` objects and no debug fields |
| Scoring | Exact `parent_asin`; first 10 valid unique IDs; miss MTTC is 11 | Do not rely on invalid-ID padding, duplicate IDs, or evaluator coercions |
| Metrics | `0.50*HR@10 + 0.30*MRR + 0.20*Efficiency` | Treat TechnicalScore as an objective input to Technical Execution, not the whole judging result |
| Private boundary | Target, scenario, intent card, simulator state, sample ID, and previous scores are not Agent inputs | All production and experimental decision features must be target-blind |
| Runtime | Organizer may impose CPU, RAM, timeout, network, and environment restrictions | Default path must be offline, deterministic, documented, and reproducible |
| Models | Legally accessible APIs/local models are allowed during development; final network may be disabled | Declare model, license, assets, tokens, latency, cost, network, and fallback |
| Scope | Keyword/dense/hybrid retrieval, routing, rewriting, reranking, state, clarification, and aggregate profile use are in scope | Full-model training, multimodal work, heavy vector-DB infrastructure, catalog mutation, and private-label reconstruction are out of scope |

SQLite `:memory:` is the official starter's implementation choice, not a rule that the
entire solution must run in memory. Conversely, the absence of a published numeric
resource limit is not permission to ignore latency or memory.

## Public evaluator versus strict contract

The public evaluator is intentionally useful for local development but is looser than
the JSON contract in several places: it can normalize bare string IDs, treats an invalid
question attribute like `other`, ignores malformed usage fields, and does not enforce a
hard wall-clock timeout. No production architecture may depend on those conveniences.

The JSON schema permits up to 100 recommendation objects while the README recommends at
most 10 and the evaluator scores only the first 10 valid unique IDs. This repository
uses the stricter and simpler invariant: return no more than `min(top_k, 10)` valid
catalog objects.

## Current local integrity evidence

- `python scripts/verify_official_assets.py` passes all `14/14` offline checks.
- Official upstream `main`: `34078351e1c3615e5505a2e829600b56a542e462`.
- Participant tag: `2a6cc8e776da66ce69b1cbd237838fbc43f32587`.
- Current evaluator Git blob: `7c808347b31ef3121a9cbc4810ac3eb325f950ba`,
  identical to upstream.
- Current public-set Git blob: `121dbec9c1368c81cd887d6959e62507512139c0`,
  identical to upstream after Git line-ending normalization.
- Local decompressed catalog: 50,000 rows and 50,000 unique non-empty IDs, SHA-256
  `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.
- Retained official compressed catalog: 19,235,996 bytes, SHA-256
  `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`,
  equal to the [official checksum](https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS).

See `docs/data_inventory.md` for row-level checks and the organizer/local data split.

## Current local architecture evidence (not an organizer result)

The repository now serves the locally selected `R08.coverage_cascade` when
`retrieval_mode=coverage` and `rerank_mode=off`; this is the normal no-override Agent
path. On the 200 organizer-released development sessions it records HR@10 `0.945000`,
MRR `0.606175`, MTTC `3.335000`, and TechnicalScore `0.807652`. Against the explicit
weighted-RRF control, the paired change is `+0.005000` HR, `+0.000917` MRR, `-0.040000`
MTTC, and `+0.003575` TechnicalScore, with zero hit-to-miss and one miss-to-hit.

The actual served Agent exactly matches the frozen winner on canonical plus eight phrase
suites and passes the local strict-contract, deterministic-trace, repeated resource-
measurement completeness, no-key, and reference-bridge checks. The combined
verification artifact SHA-256 is
`8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.
These facts show reproducibility on released/local inputs only. The organizer has not
endorsed this architecture, and no participant can verify or claim its performance on
the private 800 sessions before organizer evaluation.

P8 is separate local stress evidence, not an organizer result. Its one frozen 200-session
catalog-derived selection improved R01 versus C00 quality metrics but failed the
pre-registered wall, response-P95, and peak-RSS gates. The protocol retained C00, did not
run repeat, did not open confirmation, and did not rerun the released public set. Therefore
P8 cannot change the official-evidence table above or support a private-800 claim.

P9 is also separate local stress evidence. It used two additional target-disjoint
catalog-derived splits and a 1.42 MiB catalog-only sidecar. R01 passed its quality and
resource gates on both opened splits and selection exact repeat passed, but confirmation
B00/C00/S00 failed only a one-millionth TechnicalScore bridge check caused by different
rounding order. Confirmation repeat was not attempted, so the frozen protocol retained
C00 and made no released-public run or production change. This is neither organizer
validation nor evidence about the private 800. The staged Python audit/read boundary is
not an OS sandbox against hostile native code.

P11 is a third, separately preregistered local evidence layer, not an organizer result. It
used 600 formal sessions across primary, uniform-tail, and confirmation splits whose
targets are mutually disjoint and excluded every previously opened public/local target.
The fixed candidate only permuted the exact served R08 Top 10. It improved TechnicalScore
by `0.012685`, `0.003309`, and `0.014279` on the three splits while preserving HR, MTTC,
every scenario HR, and zero hit-to-miss; all frozen repeat, bootstrap, resource, identity,
contract, target-blind, network, token, and exception gates passed. The formal decision is
`promote_p11_r01`, but released public was not run and the served/default Agent has not
changed. Result SHA-256 is
`fe0f8820b22c07136db44fb3739809d22b8edc5d1125707c5b0523dec312b912`.
This claim does not imply scenario-MRR non-regression: the frozen gate covered scenario HR,
and two small scenario-MRR regressions remain integration risks.

## Competition-level timing and submission obligations

The Official Rules define the Submission Period as 2026-08-29 12:00 SGT through
2026-09-01 12:00 SGT and require an existing project to receive a significant update
after the period starts. The main P11 feature checkpoint `4f27ee8` was created at 06:08 SGT
and remains pre-competition evidence. The protocol-hash correction `639cf78` (13:26 SGT)
and lock-only commit `c6efa5f` (13:51 SGT) are post-start and make the formal result
auditable, but the repository does not overstate those narrowly scoped freeze operations
as the required significant product implementation. The planned reversible P11 served
integration must be a separate substantive post-start code commit with before/after tests
and public/default-branch evidence before the submission obligation is claimed complete.

The public Devpost overview additionally requests a written project description and
tech stack, a public code repository with a comprehensive README, and a public
three-minute end-to-end YouTube demonstration. These remain submission work, not Agent
runtime features.

There is a judging-description conflict that must not be hidden:

- the Official Rules describe four equally weighted Stage Two criteria: Technical
  Execution, Innovation & Problem Insight, Feasibility & Practicality, and Impact &
  Relevance;
- the Devpost overview shows those four plus Presentation & Communication;
- the Track 4 local TechnicalScore is only an objective input to Technical Execution.

Until the organizer clarifies the mismatch, engineering decisions use the Official
Rules as controlling while still preparing a strong presentation deliverable.

## Unpublished final-run details

The participant kit does not publish:

- concrete CPU, RAM, per-turn, or whole-run limits;
- whether final network access will actually be disabled;
- the final harness import/path details or archive size limit;
- whether the private simulator is byte-for-byte identical to the public evaluator;
- a formula mapping TechnicalScore into qualitative judging scores.

These are explicit unknowns. Experiments must record resource use and preserve an
offline fallback instead of inventing organizer limits.

## Promotion gate for every architecture experiment

1. Leave evaluator, catalog, and public labels byte-identical to official artifacts.
2. Record code/config/data/model hashes, dependency/license/network requirements, seed,
   dirty state, timing, RSS, and complete metrics.
3. Reject any target/scenario/sample-ID/result/ASIN-specific decision path.
4. Select designs on the frozen public-target-disjoint corpus; use the released public
   set only as the final gate for a frozen survivor.
5. Report hit-to-miss, miss-to-hit, rank, turn, and scenario changes rather than only an
   aggregate score.
6. Require strict repeated output equality, contract tests, no-key execution, and
   bounded resource use before changing the default Agent.
7. Describe the winner only as the best eligible local design; private-800 superiority
   cannot be claimed without organizer results.
