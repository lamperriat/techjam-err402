# Implementation Log

This tracked document records only code and behavior that exist in the repository and have been verified. Planning, hypotheses, and unvalidated designs belong in ignored `docs/internal_plan.md`; the current working-tree architecture belongs in ignored `docs/current_architecture.md`.

Last updated: 2026-08-28 SGT.

## Current verified implementation

- Branch: `p4-architecture-search`
- P4 served/reference bridge: `1f8fd3c` (`test: bridge frozen and promoted response traces`)
- P4 served promotion: `97bc89c` (`feat: promote coverage cascade into served agent`)
- P4 frozen matrix: `e5d0d49` (`feat: add target-blind architecture search lab`)
- P4 Workbench alignment: `04b6e21` (`feat: align observer with promoted retrieval`)
- P4 R12 hygiene fix: `eb626bc` (`fix: reject measurement-only budget signals`)
- P3 implementation: `9cc9262` (`feat: add auditable clarification shadows`)
- P3 verification and data inventory: `87447fb` (`docs: record p3 gates and official data inventory`)
- Frozen P1 head: `02f0741` on `p1-generalization`
- P2 core implementation: `586f3dd` (`feat: add target-blind shortlist reranker`)
- P2 Workbench/tooling: `4610480` (`feat: expose rerank experiments in workbench`)
- P2 v1 gate record: `f91b547` (`docs: record p2 rerank gate results`)
- Optional dependency isolation: `71383b5` (`build: isolate optional LLM dependencies`)
- Resource/route benchmark: `38ca016` (`test: add resource and route recall benchmark`)
- P1 implementation commit: `abae926` (`feat: add generalization gate and robust intent state`)
- P1 parent checkpoint: `66cb1cf` (`docs: finalize integration verification`)
- Stateful Agent integration: `5fed7a7` (`feat: integrate stateful sparse shopping agent`)
- Workbench baseline: `f4e435b` (`feat: add agent layer workbench`)
- Official upstream main rechecked on 2026-08-28: `34078351e1c3615e5505a2e829600b56a542e462`
- Runtime: Python 3.11.16 in the existing `tiktok` Conda environment
- Catalog: 50,000 parseable rows and 50,000 unique non-empty `parent_asin` values
- Public set: 200 sessions and 200 unique targets, split 80 Buying / 80 Browsing / 30 Intent Override / 10 Boundary
- Official catalog release SHA-256: `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`; local compressed asset is identical
- Official public-set Git blob: `121dbec9c1368c81cd887d6959e62507512139c0`; local Git-normalized content is identical
- Default execution: offline, no LLM object, no API key, no network call, zero reported tokens
- Direct `Agent()` defaults are `TECHJAM_RETRIEVAL_MODE=coverage`,
  `TECHJAM_RERANK_MODE=off`, and `TECHJAM_QUESTION_POLICY=fast`; clear inherited values
  or set them explicitly for a production run. `retrieval_mode=control` preserves the
  pre-P4 weighted-RRF output for paired experiments.

## Current public evaluation

The integrated Agent was run against the complete released 200-session evaluator after all code changes in this entry.

| Scope | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 0.925000 | 0.586057 | 3.162500 |
| Browsing | 80 | 0.987500 | 0.603224 | 2.925000 |
| Intent Override | 30 | 0.900000 | 0.655265 | 4.666667 |
| Boundary | 10 | 0.900000 | 0.643452 | 4.000000 |
| Overall | 200 | 0.945000 | 0.606175 | 3.335000 |

Overall Efficiency is `0.766500`; recommended TechnicalScore is `0.807652`; prompt and
completion token usage are both zero.

Compared with the explicit current-tree weighted-RRF control, promoted coverage changes
HR@10 by `+0.005000`, MRR by `+0.000917`, MTTC by `-0.040000`, and TechnicalScore by
`+0.003575`. The paired result has zero hit-to-miss and one miss-to-hit change.

The pre-integration Workbench checkpoint reproduced the official weak baseline at HR@10 `0.125`, MRR `0.068034`, MTTC `9.81`, Efficiency `0.119`, and TechnicalScore `0.10671`. The current gain therefore comes from the stateful sparse Agent integration, not from the browser observer.

These are public-development metrics, not a claim about the private 800 sessions.

## P4 coverage-cascade promotion

The frozen product-disjoint matrix recorded 14 raw non-control variants. A semantic
activation audit found R12's only apparent activation was caused by parsing a head-
circumference range (`21.25inch-25inch`) as a price. After rejecting that false
activation, 13 genuinely independent effective designs remain, still above the required
minimum of ten. A hygiene-only rerun on the same frozen corpus records zero R12
activations, zero output changes, and exact control metrics; its ignored artifact SHA-256
is `6428a2f4049f0b17dc7d9d6287716803aee596ff2e6d383ed625d86e84a7324f`.
This confirms classification without rerunning winner selection. The raw matrix artifact
is retained unchanged.

R08 was the sole eligible selection winner and is now implemented in
`starter/coverage.py` and served by `starter.agent.Agent` in the default
`coverage + rerank off` configuration. It counts distinct visible query terms matched
across the same catalog fields used by the frozen experiment, sorts by descending
coverage, and preserves weighted-RRF fused rank on ties.

The actual served Agent was independently run on canonical plus all eight registered
phrase suites. Every complete result hash equals the frozen winner; canonical HR@10 is
`0.945000`, MRR `0.606175`, MTTC `3.335000`, and TechnicalScore `0.807652`. The strict
response contract, no-key execution, two-run functional determinism, public/phrase
robustness, and resource measurement checks pass. A reference bridge additionally
proves exact complete response traces and broad/strict/fused/final route equality. The
combined artifact `experiments/p4_promoted_verification.json` has SHA-256
`8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.

The original architecture artifacts remain the selection evidence. Because promotion
changed `architecture_lab.py` to pin the old control and share the coverage helper, its
post-promotion working-tree bytes are no longer identical to the selection commit. The
old unchanged-source gate is therefore legacy/frozen evidence; the served implementation
is validated by `scripts/verify_promoted_agent.py`, not by pretending the promoted file
never changed. None of this is evidence about the organizer-private 800 sessions.

## P2 shortlist-reranker evaluation

P2 adds an explicitly gated `off / shadow / active` rerank mode. The rerank default remains
`off`; neither the released evaluator nor the normal Agent path activates an unproven
scorer.

- `off`: the complete 200-session result strictly matches frozen P1, including every
  session row and list order.
- `shadow`: computes normalized attributes and Top-50 scores but serves the original
  fused order. Its complete evaluator JSON strictly matches `off`.
- `active`: serves the experimental reranked order. The first frozen-weight run was
  rejected by the public gate.

| Mode | HR@10 | MRR | MTTC | TechnicalScore | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| off / frozen P1 | 0.940000 | 0.605258 | 3.375000 | 0.804077 | retained default |
| shadow | 0.940000 | 0.605258 | 3.375000 | 0.804077 | diagnostic only |
| active v1 | 0.930000 | 0.599974 | 3.430000 | 0.796392 | rejected |

Active v1 caused two baseline hit-to-miss regressions and reduced Buying HR from
`0.925` to `0.900`; it produced no compensating overall gain. Post-hoc diagnosis showed
that incomplete attribute coverage can incorrectly promote products with explicit
metadata above otherwise strong sparse matches. For example, one target's cotton/color
evidence existed only in the catalog description, which the conservative v1 extractor
does not treat as normalized attribute evidence. This is recorded as a failed
experiment, not presented as an improvement. Because the public activation gate failed,
active v1 was not advanced to the more expensive generalization and resource gates.

The preliminary P2 observation already suggested a time regression. The later controlled
P3 two-run artifact, recorded below against the final current source, confirms that
shadow exceeds the planned `1.5x` time gate.

## P2 v2 Top-10-member-safe control

The next target-blind control computes the same Top-50 score diagnostics but permits
movement only inside the original Fused Top 10. It preserves the Top-10 member set and
the complete order below rank 10. Adjacent candidates may cross only when both expose
the same requested-slot coverage signature. This removes the v1 hit-to-miss failure mode
by construction, but it does not bound rank displacement or guarantee an MRR gain.

The strategy was selected on the fixed 200-session product-disjoint corpus before any
new public gate:

| Derived canonical | HR@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Frozen P1 | 0.935000 | 0.630183 | 3.185000 | 0.812855 |
| P2 v2 control | 0.935000 | 0.624960 | 3.185000 | 0.811288 |

The control produced 6 best-rank improvements, 7 regressions, and zero hit-to-miss or
miss-to-hit changes. MRR fell by `0.005223` and TechnicalScore by `0.001567`, so v2 was
rejected without using the public set as a tuning loop and without running an active
resource gate. Reranking `off` remains the default.

## P3 slot and clarification shadows

`starter/slot_ledger.py` adds an auditable, target-blind normalized constraint history.
Each immutable record contains slot, normalized value, polarity, hardness, source,
confidence, source turn, state version, and an `active`, `superseded`, or `deleted`
lifecycle. Selective changes retire only the removed constraint; a no-preference event
deletes the named slot in the shadow view; explicit later evidence can reopen it. A
later, locally scoped hard restatement supersedes the earlier soft record without
upgrading unrelated values in a contrast clause. The ledger is diagnostic and does not
yet compile retrieval queries.

`starter/clarification.py` ranks unanswered attributes over the Fused Top-50 normalized
product views using:

```text
normalized information gain * catalog coverage * answerability - turn cost
```

Known, asked, exhausted, pending, category, and active-ledger attributes are omitted.
Each candidate contributes one primary value so multi-value combinations do not create
artificial entropy. Brand and feature are cardinality-penalized rather than rewarded for
raw long-tail entropy. Catalog prices are preserved in a shadow-only metadata side table,
so budget buckets can be diagnosed when price coverage exists. Turn 10 can expose the
evidence but never selects another question. The selected QuestionValue is exposed in
trace and Workbench beside the actual fixed-order question, but it does not change
`ask_attribute` or recommendations.

After the final current-tree source freeze, a fresh `off` run and a fresh `shadow` run both produced
HR@10 `0.940000`, MRR `0.605258`, MTTC `3.375000`, and TechnicalScore `0.804077`.
Their complete evaluator JSONs strictly match each other and frozen P1. The run manifests
include Agent, attributes, reranker, slot-ledger, clarification, evaluator, catalog, and
public-set hashes.

The same final source produced identical canonical session results on the frozen
public-target-disjoint corpus: HR@10 `0.935000`, MRR `0.630183`, MTTC `3.185000`, and
TechnicalScore `0.812855`. Both corpora therefore pass the non-interference gate; this
does not establish that the shadow question policy is better.

The controlled two-run resource audit passed determinism in both modes but failed the
planned `1.5x` activation budget:

| Mode | Mean total | Mean evaluator | Mean respond P95 | Mean peak RSS |
| --- | ---: | ---: | ---: | ---: |
| off | 25.436 s | 22.090 s | 71.156 ms | 369.7 MiB |
| shadow | 51.145 s | 47.742 s | 133.037 ms | 434.6 MiB |

Shadow is `2.01x` the off total wall time, `1.87x` the respond P95, and `1.18x` the
mean peak RSS. It remains a development diagnostic. The in-memory ranking-diagnostic
cache is bounded to 128 sessions to prevent unbounded Top-50 breakdown accumulation.

## Implemented Agent behavior

### Session and lifecycle

`starter/agent.py` now implements a per-session `SessionState` containing:

- aggregate profile copy;
- active category;
- active and excluded retrieval terms;
- known, asked, and exhausted attribute classes;
- pending clarification attribute and originating turn;
- per-turn terms and attribute classes;
- version, version anchor, and override count;
- the fast-policy `prefer_other_next` event.
- an auditable normalized shadow slot ledger.

`reset` replaces prior state for that session. `respond` validates turn 1–10 and positive `top_k`, updates state, ranks products, selects at most one allowed clarification attribute, and returns at most 10 catalog-backed recommendation objects. `drop_session` releases development replay/Lab state.

The SQLite connection uses `check_same_thread=False`; state and query operations are protected by an `RLock` for the multi-threaded local Workbench.

### Parsing and state transitions

Implemented deterministic parsing now produces a target-blind `ParsedTurn` and includes:

- anchored natural shopping openers such as looking/searching/shopping, need/want, show/find, and help-find;
- separation of category text, vague browsing suffixes, and actual constraint fragments;
- material, color, size, style, use-case, and budget class detection;
- conservative `not`, `no`, and `without` negative-term extraction, including `not too X` while excluding false negations such as `not only`, `not quite`, and `not sure`;
- explicit and pending-context no-preference/exhausted attribute handling;
- explicit ignore/disregard/forget, change-mind, no-longer, switch/replace, and context-bound `instead` detection;
- explicit `old -> new` spans are replaced selectively while vague `ignore earlier` events retain the auditable version-anchor behavior;
- loose `I need/want/show me` category openers establish the first goal only; later short color/material replies remain constraints, while product-head spans support explicit category switches;
- retry detection separated from negative constraints;
- repeated override version-anchor movement;
- category-goal changes that clear the previous goal's term and question lifecycle.

A plain sentence such as `Actually, cotton sounds fine` no longer triggers an override solely because it contains `actually`.

A selected clarification remains pending until the next ordinary user response. If an evaluator Override interrupts the expected answer, the pending attribute is released instead of being permanently marked asked; an interrupted `other` fallback also restores its disclosure preference.

### Sparse retrieval and fusion

The Agent builds one in-memory SQLite FTS5 catalog index over title, categories, features, details, store, and description.

Each turn compiles the current active state into two retrieval routes:

- Broad: quoted terms joined by OR, field-weighted BM25, Top 120.
- Strict: up to 16 quoted terms joined by AND, field-weighted BM25, Top 80.

The routes are fused deterministically with:

```text
score(d) = I_b(d) / (60 + broad_rank)
         + 1.8 * I_s(d) / (20 + strict_rank)
```

Ties use broad rank and then `parent_asin`. The promoted `coverage` retrieval mode loads
title, categories, features, details, store, and description for the fused candidate set,
counts distinct active query terms found in those visible catalog fields, sorts by
descending coverage, and preserves fused rank on ties. The response returns the first
`min(top_k, 10)` IDs from the explicit `final` route. With default `coverage + off`,
`fused` remains the control order, `reranked` remains equal to fused, and `final` is the
coverage order. Explicit `control + off` leaves final equal to fused.

### Normalized attributes and shortlist reranking

`starter/attributes.py` builds immutable, target-blind product and visible-conversation
views. The first frozen schema normalizes category, audience, material, color, closure,
style, use case, size, width, brand, price, and atomic feature phrases; records source,
confidence, and raw evidence; filters numeric/generic catalog noise; and uses no public
labels, sample IDs, target IDs, profile priors, network calls, or evaluator imports. Its
registry SHA-256 is
`1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca`.

`starter/reranker.py` is a deterministic pure scorer over the fused Top 50. It exposes
RRF prior, category consistency, positive slot match, exact feature match, negative
violation, total score, and matched evidence. Missing values remain unknown rather than
violations. Scoring diagnostics cover Top 50; active v2 can move only members of the
original Top 10 within equal requested-slot coverage groups and preserves the full order
from rank 11 onward. The mode is rejected and off remains the default.

The Agent exposes five auditable routes: `broad`, `strict`, `fused`, `reranked`, and
`final`. In explicit control mode, `off` skips attribute scoring, `shadow` computes it
without changing output, and `active` uses it only when explicitly requested. Coverage
is restricted to rerank off to prevent an ungated composition. A bounded 10,000-view LRU
cache avoids re-extracting common shortlist products. Target-blind debug diagnostics
expose component and coverage evidence; the Observer joins public targets only after
`respond` returns.

### Clarification policy

Three explicit policies are supported:

- `fast` (default): after a no-preference reply, prefer `other` to obtain the remaining disclosed constraints quickly;
- `boundary`: use that shortcut only for a direct Boundary-style reply;
- `conservative`: continue through the fixed allowed-attribute order.

Known, asked, and exhausted attributes are not re-asked. Turn 10 always returns `ask_attribute=null`.

The selected policy can be passed to `Agent` or set with `TECHJAM_QUESTION_POLICY`. Workbench experiment manifests now record it because it directly changes results.

### Optional LLM boundary

The default Agent does not import, construct, or call an LLM client. An explicitly injected compatible client is used only to consume and report token usage; it does not currently parse intent, retrieve, rerank, or write response prose.

`utils/llm_client.py` remains available for measured future experiments and is covered by
configuration, JSON-response, error, and usage tests. Core execution uses the stdlib-only
`requirements.txt`; optional OpenAI and dotenv packages are isolated in
`requirements-llm.txt`. Agent/evaluator imports were verified under `python -S` without
site packages.

## Implemented Agent Workbench

The local Workbench is a loopback-only development control plane, not part of official scoring.

### Startup and pages

- `Start Observer.vbs`: hidden `pythonw.exe` launch using the existing `tiktok` environment.
- `Start Observer.cmd` and `python -m observer.launcher`: troubleshooting fallbacks.
- Overview: runtime, Git, source fingerprint, data/hash/index health, metrics, and truthful algorithm registry.
- Session Diagnostics: deterministic public replay, actual Agent events, output validation, and post-hoc score diagnosis.
- Catalog & Index: 50k catalog browsing, field-weighted BM25 search, and raw product JSON.
- Runs & Experiments: fixed test/evaluator/generalization jobs, progress, logs, cancellation, metrics, versioned manifests, and target-blind cross-session shadow-policy summaries.
- Lab: target-free calls to the real `reset/respond` interface with opaque session IDs.
- Documents: read-only allowlisted project documentation and source.

### Trace integration

The Agent can emit optional versioned, target-blind events for:

```text
session -> parse -> retrieval -> state -> policy -> output
```

Retrieval events expose broad/strict/fused/reranked/final counts, the weighted-RRF
formula, retrieval/rerank modes, coverage schema and matched-term evidence, and
raw-fused/reranked/final Top-10 evidence. State events expose only information derived
from the profile and conversation.

For a public replay, the Observer calls `Agent.respond` with a random UUID session. Only
after the response does it compare the target with target-blind route IDs and component
diagnostics. It records broad, strict, fused, reranked, and final ranks; `final` is the
actual output route. Target, scenario, intent card, behavior, prior result, and public
sample ID are never passed into Agent decision features.

Completed replays and evicted Lab sessions release Agent and recorder state.

### Local control-plane safety

- loopback bind only;
- project fingerprint check before reusing port 8765;
- per-process API control token;
- Host, Origin, and browser-site checks;
- JSON-only mutation bodies;
- CSP and frame protections;
- fixed allowlisted test/evaluation/Lab/shutdown controls;
- no arbitrary shell or filesystem browser;
- loaded-vs-disk Agent/coverage/attributes/reranker/slot-ledger/clarification/
  shadow-analysis/evaluator/generalization sources plus catalog/public-set fingerprints,
  with stale-runtime blocking for every replay, evaluation, generalization, and Lab call;
- evaluation provenance captured before the background job and rechecked before artifact finalization, so a manifest cannot mix loaded code/data with later disk hashes.

The Workbench must not be publicly deployed or connected to private final labels.

## P1 generalization and reliability gate

`scripts/evaluate_generalization.py` adds a deterministic, target-blind robustness runner without changing the released evaluator. Its `PerturbedAgent` wrapper transforms only the visible `user_message` before delegating to `Agent.respond`; it is never given sample ID, scenario, target ID, intent card, or prior result.

The frozen phrase registry contains independent development, challenge, and audit wording for shopping openers, requirement disclosures, no-preference responses, overrides, and retry feedback. It records a hash over every regex/replacement and suite composition, applied-rule coverage/counts, examples, per-suite metrics, deltas, and paired session changes using the official per-session score contribution. A released-public suite fails instead of reporting robustness if any selected rule transforms zero messages. The current registry SHA-256 is `7ebc55c38f3389da2d2d01f549763c2c6b39908f89b60095fafb2c1964cc940b`.

Released-public phrase results before and after P1:

| Phrase suite | Before HR@10 | Before Score | After HR@10 | After Score |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 0.940000 | 0.803977 | 0.940000 | 0.804077 |
| Combined development | 0.845000 | 0.700243 | 0.940000 | 0.804077 |
| Combined challenge | 0.835000 | 0.687278 | 0.940000 | 0.804077 |
| Combined audit | not used for the initial baseline | not used for the initial baseline | 0.940000 | 0.804077 |

After P1, every individual suite plus the combined development, challenge, and audit suites has the same released-public HR@10, MRR, MTTC, and TechnicalScore as canonical. The all-suite robust hit count is 188/200 (`0.940000`). This demonstrates resilience to these frozen equivalent phrasings; it does not prove unrestricted natural-language understanding.

The same runner can build 200 deterministic catalog-derived sessions after excluding every released-public target. The fixed seed `track4-p1-product-disjoint-v1` produces SHA-256 `38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae`, 200 unique targets, zero public-target overlap, and an 80 Buying / 80 Browsing / 30 Intent Override / 10 Boundary split.

| Derived suite | Before HR@10 | Before Score | After HR@10 | After Score |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 0.935000 | 0.813096 | 0.935000 | 0.812855 |
| Combined development | 0.855000 | 0.740916 | 0.935000 | 0.812855 |
| Combined challenge | 0.835000 | 0.727386 | 0.935000 | 0.812855 |
| Combined audit | not used for the initial baseline | not used for the initial baseline | 0.935000 | 0.812855 |

The pending-question change slightly lowers derived canonical Score by `0.000241`: derived Buying HR changes from `0.950000` to `0.937500`, while Intent Override HR improves from `0.966667` to `1.000000` and its MTTC from `3.966667` to `3.800000`. This mixed scenario result is recorded rather than hidden. The derived corpus is a local metadata-based stress set, not organizer private data, and is not a prediction of private evaluation performance.

The Workbench exposes this fixed run as **运行泛化压力测试** and `POST /api/jobs/generalization`. Evaluation and generalization jobs are mutually exclusive because both build repeated 50,000-product in-memory indexes. The versioned artifact records Git state, source/input hashes, corpus metadata, phrase transforms, metrics, and robustness summaries.

## Strict result verification

`scripts/compare_results.py` supports both metric reporting and strict verification.

```powershell
python scripts/compare_results.py run_a.json run_b.json
python scripts/compare_results.py --assert-equal run_a.json run_b.json
```

Strict mode recursively compares the complete parsed objects, including key presence, list order, scenario metrics, usage, and every session row. Any semantic difference exits with code 1 and prints the first JSON paths that differ. Whitespace, indentation, CRLF, and LF formatting differences do not fail.

This replaces the handoff comparator behavior that printed aggregate deltas but always exited successfully.

## Verification completed

- The complete current suite passes `153/153` Python unit/integration tests after adding
  the P4 architecture lab, official-asset integrity, contract, lifecycle, budget,
  promotion bridge, R12 hygiene, and Workbench retrieval-mode coverage tests.
- Agent tests cover accumulation, natural openers/requirements/no-preference, pending-question interruption, category changes, negative phrases and false negations, false override prevention, first/repeated/selective overrides, Boundary exhaustion, question policies, five ranking routes, mode safety, catalog-price shadow ingestion, bounded diagnostic memory, output cap/final turn, optional usage, and target-blind trace/component diagnostics.
- Attribute/reranker/ledger/QuestionValue tests cover normalization boundaries, immutable provenance, unknown values, source confidence, noise removal, scorer arithmetic, negative penalties, deterministic ties, immutable fused input, Top-10 member and tail safety, lifecycle retirement/hard restatement, multi-value entropy control, final-turn suppression, and candidate-price coverage.
- Generalization tests cover phrase payload preservation, adapter input isolation, deterministic stratified public-target-disjoint generation, and rerank-mode propagation.
- Comparator tests cover formatting/line-ending equality, session-level mismatch, missing keys/list order, and invalid JSON.
- Existing evaluator, LLM client, Workbench replay, catalog/Lab/background evaluation,
  HTTP token/cross-site, and exclusive-listener tests pass. Observer tests also cover
  retrieval-mode propagation, coverage evidence/provenance, fused-versus-final route
  semantics, source fingerprints/schema metadata, cross-session target-blind shadow
  artifacts, visible ledger/QuestionValue components, and stale-source rejection.
- `node --check observer/static/app.js` passes.
- The complete 200-session evaluator completed successfully with no LLM environment variables.
- The final direct public evaluator result strictly matches the canonical result produced through the target-blind robustness wrapper.
- A headless Chrome smoke test rendered the live loopback Workbench against 50,000 indexed products and 200 sessions; Overview, state/fusion pipeline, public Trace, two-turn Lab state, and background Tests were exercised successfully with `restart_required=false`.

## Provenance and audit corrections

The v0.6 release material was independently audited before integration:

- outer ZIP hash and 73/73 declared file hashes matched;
- source.zip, bundle target tree, and the shared project source files matched byte-for-byte;
- bundle history is complete and contains official `3407835` as an ancestor;
- the patch exactly represents `367f1bf -> 89ef66c`, not `3407835 -> 89ef66c`;
- historical participant commit `914879c` added an `AgentBase` Protocol/type annotation only; the current file has been restored and has no diff from official upstream blob `7c808347b31ef3121a9cbc4810ac3eb325f950ba`;
- the packaged `original_baseline_reference/starter_agent.py` is a participant optional-client baseline, not the pristine official starter.

The current repository may describe the evaluator as restored to the official upstream
file. Historical audit reports must still distinguish the earlier type-only wrapper.

## Current limitations

1. The retrieval source of truth remains the term/turn state. A normalized slot ledger now exists in shadow, but it does not yet compile structured filters or retrieval queries. Explicit old→new spans are selective, while a vague `ignore earlier` override can still remove unrelated preferences introduced in the same anchor turn.
2. The parser now passes three frozen equivalent-phrase families, but it remains deterministic and English-pattern based; this is not unrestricted semantic parsing.
3. The default fast policy benefits from the public simulator's `other` disclosure behavior. This is protocol adaptation, not direct label leakage, but it creates public-strategy overfitting risk.
4. The product-disjoint corpus is derived from the same frozen catalog and official simulator, so it tests target overlap and wording robustness but is not an independent approximation of the private distribution.
5. Served clarification still uses a fixed order. Candidate-aware QuestionValue exists only in shadow; its Top-50 candidates are equally weighted, missing values are not modeled as a bucket, and its constants have not passed an activation gate.
6. No explicit Buying/Browsing router is implemented; hidden scenario labels are never available to the Agent.
7. Profile data is stored but not used for personalization.
8. There is no structured hard filter/relaxation execution, dense retrieval, learned reranker, or semantic reranker. The deterministic constraint scorer does not yet enforce negative constraints as a veto or bound Top-10 rank displacement; active v1 and v2 both failed their gates and remain disabled.
9. Budget buckets are visible in QuestionValue shadow, but a user budget such as `under $50` does not yet become a numeric range filter or ranking constraint. Budget questioning cannot be activated until that downstream path exists.
10. The controlled P3 shadow resource audit is deterministic but fails the planned time gate at `2.01x` off total wall time, so shadow remains development-only.

The current served implementation should be described as **versioned stateful sparse
retrieval with weighted-RRF candidate fusion, promoted visible-query-term coverage
ordering, and heuristic clarification, plus normalized slot/attribute/rerank/
QuestionValue diagnostics**, not as the complete IntentGraph target architecture.

## Change history

### 2026-08-28 - P5 independent selection corpus and pre-registered guarded PRF

- Added a deterministic P5 corpus builder that excludes both released-public and frozen
  P1-derived targets, validates the 50,000/200/200 inputs, preserves the official scenario
  mix, and writes cross-platform canonical hashes. The ignored frozen P5 corpus contains
  200 unique targets with both overlaps equal to zero; SHA-256 is
  `0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c`.
- Added isolated target-blind PRF primitives and a P5-only control/shadow/active Agent
  registry without changing the served Agent or the frozen P4 architecture evidence.
  Feedback uses cross-seed catalog-IDF evidence, an original-query-conjoined second FTS
  route, low-weight rank fusion, and a protected Top-9/one-newcomer safety boundary.
- Added a frozen P5 runner that hard-checks corpus identity and both exclusion sets,
  compares C00 with evaluator and ordered-response hashes from a complete run of the
  actual served coverage Agent, proves shadow output equality, enforces metric/scenario/
  hit-to-miss/runtime gates, and repeats only an
  eligible active candidate. Released-public rows are not evaluated by this runner.
- The protocol, constants, formulas, and rejection gates were committed before opening
  P5 metrics. Until that run is complete, this entry records implemented experiment
  infrastructure rather than an effectiveness claim or promotion decision.

### 2026-08-28 — P4 target-blind architecture search and promotion

- Added an experiment-only `ArchitectureAgent` registry with one exact control and 14
  materially different retrieval, fusion, constraint, state, diversification, budget,
  and routing candidates. Selection initially left `starter.agent.Agent` unchanged.
- Added a frozen product-disjoint matrix runner with strict response-contract validation,
  control-integrity failure, activation/output-change accounting, session/scenario gates,
  hit-to-miss rejection, separate raw/eligible winners, and deterministic confirmation.
- Contract-invalid or incomplete variants cannot count toward the ten-experiment
  requirement. R09 never backfills a known negative conflict; R11/R13 scope browsing
  evidence to the current goal version. The raw runner counted R12 after one output
  change, but semantic audit proved that change came from a measurement-to-price regex
  false positive. The corrected semantic-effective count is 13, not 14.
- Added preflight and postflight Git/source/input snapshots. A long run is discarded if
  any direct source, input, Git branch/commit, dirty state, or derived-path state changes.
  The artifact records all parsed invocation values and hashes direct Agent dependencies.
- Added the complete offline official-asset verifier and a rules-first audit. The local
  catalog/public/evaluator assets pass all row, schema, uniqueness, membership, scenario,
  Git-blob, and release-hash checks.
- The frozen clean-tree 200-session matrix recorded 14 mechanically effective,
  contract-clean non-control variants; 13 remain genuinely effective after the R12
  audit. `R08.coverage_cascade` is the sole eligible selection winner:
  HR `0.935→0.945`, MRR `0.630183→0.643516`, MTTC `3.185→3.115`, Score
  `0.812855→0.823255`, zero score regressions, and exact repeated functional output.
  That table is selection-corpus evidence only.
- Promoted R08 into the served Agent and independently verified the actual default Agent
  against the frozen/reference winner on all nine public phrase suites, complete response
  traces, broad/strict/fused/final routes, strict contract, determinism, resources, and
  no-key execution. The combined verification artifact SHA-256 is
  `8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.

### 2026-08-28 — P3 auditable slot and clarification shadows

- Added immutable normalized slot history with active/superseded/deleted lifecycles,
  scoped hard restatements, selective override retirement, and no-preference deletion.
- Added candidate-aware QuestionValue diagnostics, candidate-price ingestion, active-slot
  blocking, final-turn suppression, and bounded ranking-diagnostic memory without changing
  served questions or recommendations.
- Added full Workbench ledger/QuestionValue cards, cross-session target-blind policy
  artifacts, schema/source provenance, and stale-runtime coverage.
- Restored `evaluator/local_evaluator.py` to the official upstream Git blob and verified
  public and product-disjoint off/shadow functional equality.
- Expanded the suite from 94 to 114 tests. The two-run resource audit failed the shadow
  time gate, so P3 remains diagnostic and reranking remains off by default.

### 2026-08-28 — P2 normalized attributes and gated rerank (`586f3dd`, `4610480`)

- Added catalog-only normalized attribute views and visible-dialogue-only constraint
  evidence with frozen registries and provenance.
- Added deterministic Top-50 component scoring with immutable fused order, untouched
  tail, stable ties, and explicit off/shadow/active/final route semantics.
- Added target-blind component diagnostics, bounded attribute caching, experiment runners,
  five-route recall/resource audit, and Workbench visualization/source-stale protection.
- Expanded the suite from 63 to 94 tests and preserved the P1 result exactly in both off
  and shadow modes.
- Rejected active v1 after HR, MRR, MTTC, TechnicalScore, Buying HR, and preliminary
  resource evidence failed the activation gates; reranking remains off by default.

### 2026-08-28 — P1 generalization and intent-state reliability (`abae926`)

- Added frozen target-blind development/challenge/audit phrase suites and deterministic public-target-disjoint derived sessions.
- Recorded pre-change failures before expanding parser recognition, preserving a causal before/after comparison.
- Added `ParsedTurn`, broader but conservative phrase recognition, and category/constraint separation without changing retrieval weights.
- Added a pending-question lifecycle so Override messages do not silently consume unanswered clarification opportunities.
- Added a fixed Workbench robustness action, provenance manifests, progress/logs, experiment comparison, and source-stale protection.
- Expanded the test suite from 32 to 55 tests; public HR/MRR remain unchanged while overall MTTC improves by `0.005`.
- Rejected immediate activation of feature-first/candidate-aware clarification because a read-only experiment improved overall score but reduced Boundary HR from `0.9` to `0.8`; candidate evidence will be developed in shadow mode first.

### 2026-08-27 — v0.6 integration into Workbench (`5fed7a7`)

- Preserved the Workbench checkpoint in commit `f4e435b` on a new integration branch.
- Replaced the stateless current-message-only Agent with the audited versioned state, broad/strict sparse routes, weighted RRF, and question policies.
- Preserved and extended thread-safe target-blind trace events.
- Replaced the Observer's old current-message BM25 diagnosis with post-response broad/strict/fused route diagnosis.
- Added state lifecycle cleanup and policy fingerprints to experiment manifests.
- Updated Workbench pipeline/UI labels to reflect state and fusion without claiming dense/reranking layers.
- Added strict complete-result comparison and expanded the test suite from 16 to 32 tests.
- Reproduced the complete v0.6 public result exactly.

### 2026-08-27 — Agent Workbench checkpoint (`f4e435b`)

- Added the loopback browser control plane, one-click Windows launcher, target-free Lab, public replay diagnostics, catalog/index explorer, fixed background jobs, versioned experiment manifests, document viewer, source-stale guard, API security checks, and tests.
- The checkpoint intentionally retained the stateless weak BM25 Agent and reproduced TechnicalScore `0.10671` before the v0.6 integration.

### Earlier participant history

- `367f1bf`: recorded official upstream alignment on `pre`.
- `1496fec`: merged official `3407835` into the participant history.
- `8f9e64d`: reliable no-credential baseline and first Layer Observer.
- `914879c`: optional OpenAI-compatible client, usage tests, challenge notes, and the type-only evaluator Protocol wrapper.
- `2a6cc8e` and `9a35be5`: official participant-kit history shared byte-for-byte with upstream.

## Competition boundary

The Devpost Rules and event pages were checked on 2026-08-27 SGT. The submission window begins on 2026-08-29 at 12:00 SGT. This integration is a pre-window technical baseline and does not by itself prove the required significant post-start update. After the window opens, substantial code work must have clear commits, tests, results, and documentation.

TechnicalScore remains an objective input to Technical Execution rather than the complete judging result. Public metrics, architecture quality, feasibility, limitations, impact, and communication evidence all remain necessary.
