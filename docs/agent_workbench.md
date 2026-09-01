# Agent Workbench Guide

The Agent Workbench is a local, offline-first development control plane for the Shopping Copilot. Its default Fusion Studio explains the tracked teammate/Fusion A/B implementations and frozen evidence; the remaining pages make the legacy `starter.Agent` runtime observable without presenting planned layers as completed code.

## Current truth

The served Agent is a versioned stateful sparse system. It accumulates active
conversation terms, handles explicit Override and Boundary/no-preference events,
retrieves with broad and strict SQLite FTS5/BM25 routes, fuses them with weighted RRF,
then applies the promoted R08 visible-query-term coverage cascade. The explicit `fused`
route remains the control; `final` is coverage ordered and preserves fused rank on ties.
The default is `retrieval_mode=coverage`, `rerank_mode=off`. A normalized product-
attribute reranker remains behind `off / shadow / active`; active v1/v2 failed their
gates. A normalized slot ledger and candidate-aware QuestionValue policy remain
diagnostics only.

Implemented:

- no-credential Agent startup;
- versioned multi-turn state with category, active/excluded terms, answered/exhausted/pending attribute lifecycle, and override anchor;
- target-blind `ParsedTurn` events with broader opener, requirement, no-preference, retry, and conservative override recognition;
- broad OR Top-120 and strict AND Top-80 field-weighted BM25 routes;
- deterministic weighted RRF plus promoted query-term coverage ordering and
  catalog-backed Top 10 output;
- target-blind normalized product/constraint evidence and an explainable Top-50 scorer;
- explicit broad/strict/fused/reranked/final routes, with fused control separated from
  the served coverage final route;
- normalized slot records with polarity, hardness, source turn, version, and active/superseded/deleted history;
- candidate-aware clarification shadow scores based on normalized information gain, catalog coverage, answerability, and turn cost;
- catalog-price ingestion for budget evidence in shadow, without claiming numeric budget filtering;
- target-blind cross-session actual-vs-shadow question analysis with scenario slices and blocked-selection checks;
- fast, boundary, and conservative clarification policies;
- complete released-public result HR@10 `0.945`, MRR `0.606175`, MTTC `3.335`, and
  TechnicalScore `0.807652`;
- optional, versioned development trace events;
- public-session post-hoc diagnostics;
- catalog and in-memory index inspection;
- browser-controlled unit tests, 200-session evaluation, and fixed target-blind
  generalization stress run;
- versioned local experiment artifacts;
- a manual Agent playground;
- a read-only project document library.

Not implemented in the Agent:

- structured hard filters or safe relaxation;
- Buying/Browsing routing;
- a slot ledger that is the retrieval source of truth, conflict-aware hard filtering, or deterministic relaxation;
- dense retrieval or semantic reranking;
- active candidate-aware clarification;
- numeric budget range filtering/reranking;
- profile-based ranking.

Retrieval still compiles from the term/turn state. Coverage is a target-blind ordering
stage over the sparse candidate set, not a structured constraint engine. The normalized
slot ledger is an auditable shadow representation, not yet the retrieval source of
truth, so the system must not be described as a complete IntentGraph.

## Start without a terminal

On this Windows development machine, double-click:

```text
Start Observer.vbs
```

It discovers any available Python 3.10+ (preferring `OBSERVER_PYTHON` or the currently
active Conda environment), keeps the console hidden, and opens
`http://127.0.0.1:8765`. No environment name is hard-coded. The P4-aligned launcher explicitly
uses `retrieval_mode=coverage` and `rerank_mode=off`, matching the served Agent. The UI
shows the weighted-RRF fused control separately from the coverage-ordered final route;
it does not silently substitute the older shadow/control configuration.

Fallbacks:

- double-click `Start Observer.cmd` if Windows Script Host is disabled;
- run `python -m observer.launcher` for troubleshooting;
- inspect ignored `observer_startup_error.log` if the hidden launch fails.

### Two startup modes

The launcher deliberately supports a clean checkout that does not yet contain the
separately distributed catalog:

- **Showcase mode (catalog absent):** the server and browser still open. Fusion A/B
  diagrams, the step-through architecture player, frozen metrics, source map, and
  tracked documents work. Catalog search, session replay, evaluation controls, the
  legacy Lab, and real T0/A/B execution remain unavailable.
- **Full local mode (catalog installed):** extract the organizer's official catalog to
  `data/catalog.jsonl` and restart. The runtime builds its in-memory indexes and enables
  catalog/session tools plus the real T0/A/B Live Lab. Expected catalog: 50,000 rows,
  SHA-256 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`.

The Workbench does not download, generate, or modify the official catalog. A local
`results.json` is optional, and ignored full-result artifacts are not required to view
the frozen result tables. After adding or replacing the catalog, stop and restart the
Workbench; do not continue with a process that loaded a different input fingerprint.

If a Workbench instance is already healthy, the launcher opens that instance instead of starting another one. Use the red **停止** button in the page to stop it safely.

The launcher verifies a project-root fingerprint before reusing port 8765. If another clone owns that port, startup stops with an explicit error instead of silently opening the wrong project.

## Pages

### Fusion A/B

- switch between teammate T0, strict A, and bounded-`other` B architecture graphs;
- step through intent pages 1/2/3+ and visible override computation without running an evaluator;
- compare tracked Public200 and local `train_explore` 2k OOF metrics;
- keep missing T0 2k evidence as unavailable and label 2k OOF as non-private;
- link every layer to its implementation boundary and surface B's fold/override risk.

With the official catalog installed, the same page also contains a **real T0/A/B Live
Lab**. Selecting a variant starts a fresh opaque local session and invokes the tracked
factory for teammate T0, strict Fusion A, or Fusion B. Only one variant index remains
active at a time. The supplied prompt buttons use evaluator-shaped Buying, Browsing,
Clarification, and Override messages so the state transitions can be demonstrated
reliably without running an evaluator.

The Live Lab distinguishes three evidence types:

- **actual runtime output:** Agent message, `ask_attribute`, and ordered catalog-valid
  Top 10 returned by T0/A/B;
- **observer-derived inspection:** a deterministic replay of the candidate query and a
  read-only snapshot of visible state, candidate counts, inferred page/route, ledger,
  fallback diagnostics, and question lifecycle;
- **frozen benchmark evidence:** aggregate metrics loaded from tracked JSON, never
  recomputed merely by opening or using the page.

Observer-derived values explain a real response but are not claimed as native trace
events. The Lab has no target or scoring join, while the separate public-session page
may add explicitly labelled post-hoc diagnostics after `Agent.respond`.

The Fusion Studio reads `docs/teammate_ab_website.json`; it does not infer metrics from
the currently loaded `results.json`. See `docs/teammate_ab_website_handoff.md` for the
architecture and evidence interpretation.

### 总览

- latest HR@10, MRR, MTTC, Efficiency, and TechnicalScore;
- official-reference comparison;
- Python, SQLite, Git branch/commit, and dirty-state facts;
- catalog, public-set, and result health, sizes, row counts, and hashes;
- actual FTS5 index fields, tokenizer, BM25 weights, and row count;
- an algorithm registry that labels every layer as implemented, baseline-only, or not implemented.

### 会话诊断

- all 200 public sessions with scenario/result filtering;
- deterministic replay for one public session;
- actual Agent trace events for session, parse, retrieval, state, policy, and output;
- per-turn elapsed time, token usage, five-route counts, fusion evidence, retrieval mode,
  coverage histogram and per-result matched-term counts, normalized attribute components, slot
  history, actual question, candidate-aware shadow question, and final Top 10;
- visible active and retired ledger records plus QuestionValue information-gain, coverage, answerability, turn-cost, blocker, and candidate-split details;
- post-hoc public target broad/strict/fused/reranked/final ranks, hit eligibility, and derived score contribution;
- output malformed/invalid/duplicate diagnostics and a route/fusion-aware miss code;
- trace refresh and JSON export.

The UI intentionally separates **actual** Agent events from **post-hoc** public-label annotations.

### 商品与索引

- browse all 50,000 frozen catalog products;
- run the same tokenization and field-weighted FTS5/BM25 search used by the current Agent;
- inspect match counts, BM25 scores, catalog metadata, and complete raw product JSON.

### 运行与实验

- start the fixed, portable 73-test release suite (Agent/evaluator contract,
  T0/A/B, Live Lab, and Observer). Historical experiment tests that require ignored
  P6/P8/P9 assets or optional `pytest` are intentionally left to their CLI environments;
- start the released public evaluator, which preserves the official scoring behavior, with a fresh Agent index;
- start the fixed target-blind robustness gate over the released public corpus and a deterministic public-target-disjoint derived corpus;
- see current session, progress, elapsed time, and captured logs;
- request cancellation;
- compare the official baseline, current `results.json`, and timestamped local experiments.

The Workbench does not accept arbitrary shell commands. A successful evaluation writes ignored local artifacts under:

```text
experiments/<timestamp>_public_eval/
  manifest.json
  results.json
  shadow_policy.json

experiments/<timestamp>_generalization/
  manifest.json
  results.json
```

The manifests record Git state, retrieval and rerank modes, coverage schema/source hash,
clarification diagnostic mode, other relevant source hashes, catalog/dataset hashes,
runtime settings, functional elapsed time, and metrics so stale results are easier to
detect. A public Workbench evaluation also writes `shadow_policy.json`, which aggregates
actual-vs-shadow disagreements, attribute counts, selected-value components, blocked-
selection violations, and scenario slices without reading targets, intent cards,
behavior, or target ranks. The generalization result records the frozen suite registry
hash, transformation counts/examples, paired session changes, and derived-corpus
seed/sample hash/overlap audit. Elapsed time in a normal run is a functional observation;
use the dedicated resource artifact for controlled repeated RSS/P95 evidence.

The derived corpus is generated from catalog metadata after excluding every released-public target. It is useful for product-disjoint stress testing, but it is not organizer private data and must not be presented as a hidden-leaderboard estimate.

### 交互 Lab

This is a target-free manual playground. It calls the same `Agent.reset` and `Agent.respond` methods with an opaque lab session ID and shows recommendations plus actual state and retrieval events. It can demonstrate constraint accumulation, clarification, explicit override, category goal changes, and the current parser/slot limitations without exposing a target label.

This legacy `starter.Agent` playground is separate from the T0/A/B Live Lab on the
Fusion page. Use the latter when recording or comparing strict A and B; use this page
when diagnosing R08/P11 and the legacy trace schema.

## Suggested recording sequence

1. Open **Fusion A/B** and show that the status is Showcase or Full local mode.
2. Switch T0 → A → B and explain the highlighted layers and question-policy delta.
3. Switch Public200 → 2k OOF. State explicitly that 2k is local `train_explore` OOF,
   not the organizer's private 800, and that no comparable T0 2k artifact exists.
4. Play Page 1, Page 2, Page 3+, and Override with the arrow/play controls.
5. In full local mode, start a new T0 Live Lab session and send a supplied Buying or
   Browsing prompt; point out ProductScorer, unseen pagination and specific questions.
6. Start A and repeat the same visible prompt: pages 1–2 retain T0 exploitation, while
   page 3+ can use the unseen v2.12 expert tail with T0 fallback.
7. Start B and show that ranking is inherited from A while only the bounded `other`
   question lifecycle changes; use Override to show lifecycle reset.
8. End on the frozen result table and the branch decision: A did not beat T0 on the
   full Public metric set, so this package is published on
   `deadline-v3.5-ab-website` rather than `main`.

Do not click **运行 200 会话评测** for a presentation that is intended only to reuse
the frozen evidence. Starting or using either manual Lab does not invoke the evaluator.

### 资料库

The page provides read-only access to an allowlist of project documents, official-kit material, and the current Agent/evaluator/Workbench source files. It does not expose arbitrary filesystem paths and does not provide a browser source editor.

## Ground-truth isolation

Allowed data flow:

```text
profile + generated user message + opaque session ID + turn + top_k
                                |
                                v
                              Agent
                                |
                                v
                         Agent response/events
                                |
                                v
public target ----------------> post-hoc annotation and scoring
```

The Agent never receives:

- `sample_id`;
- `scenario_type` or difficulty;
- ground-truth ASIN/title/features;
- intent card or simulator behavior;
- prior evaluation result.

The Observer uses a fresh random session ID for every replay. The target-rank probe executes only after `Agent.respond`; it cannot change retrieval, ranking, policy, or stopping behavior.

## Local-only safety boundary

- The server refuses non-loopback bind addresses.
- API requests with a non-loopback Host/Origin or cross-site browser context are rejected.
- The browser obtains a fresh in-memory control token at startup; all protected API calls require it.
- Public labels and derived intent cards are development diagnostics only.
- The Workbench must not be deployed as a public service or connected to private final labels.
- Production Agent code does not import the Observer, evaluator, public dataset, or results.
- The optional trace callback is absent during normal evaluator runs, so diagnostics do not become ranking features.

The runtime records loaded and on-disk hashes for `starter/agent.py`,
`starter/coverage.py`, `starter/attributes.py`, `starter/reranker.py`,
`starter/slot_ledger.py`, `starter/clarification.py`,
`observer/shadow_analysis.py`, the evaluator, the generalization runner, the catalog,
and the public set. If any monitored file changes after startup, the page shows a restart
warning and blocks evaluation, generalization, every replay, and Lab execution.
Background runs recheck those fingerprints before artifact finalization, and manifests
use a captured start-of-run provenance snapshot. This prevents the long-running server
from mixing imported old code or cached data with later disk files and hashes.

## HTTP API

Read-only endpoints:

```text
GET /api/health
GET /api/token
GET /api/overview
GET /api/sessions
GET /api/trace?sample_id=...&refresh=1
GET /api/catalog?q=...&offset=...&limit=...
GET /api/product?parent_asin=...
GET /api/jobs
GET /api/experiments
GET /api/documents
GET /api/document?id=...
```

Fixed control endpoints:

```text
POST /api/jobs/evaluation
POST /api/jobs/generalization
POST /api/jobs/tests
POST /api/jobs/<job_id>/cancel
POST /api/lab/reset
POST /api/lab/respond
POST /api/fusion-lab/reset
POST /api/fusion-lab/respond
POST /api/shutdown
```

The page handles the control token automatically. A manual API client must first read `/api/token` from the same loopback instance and then send its value as `X-Observer-Token`. Mutation requests must use `Content-Type: application/json`.

## Maintenance contract

- `starter.Agent` may emit target-blind session/parse/retrieval/state/policy/output events through its optional `trace_sink` callback.
- Events use `schema_version=2.1`; future Agent layers should add events rather than making the UI reimplement private algorithm logic.
- Every pipeline card must reflect current code, not roadmap intent.
- Evaluator/scoring semantics stay in `evaluator/local_evaluator.py`; the Workbench imports and invokes them rather than changing that file.
- New control actions must remain fixed and allowlisted. Do not add an arbitrary command executor.
- Add API and behavior tests whenever a control or trace field changes.
