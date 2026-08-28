# Agent Workbench Guide

The Agent Workbench is a local, offline-first development control plane for the Shopping Copilot. It makes the current implementation observable without presenting planned IntentGraph layers as completed code.

## Current truth

The served Agent remains a versioned stateful sparse baseline. It accumulates active conversation terms, handles explicit Override and Boundary/no-preference events, retrieves with broad and strict SQLite FTS5/BM25 routes, fuses them with weighted RRF, and uses an auditable heuristic clarification policy. A normalized product-attribute reranker is implemented behind `off / shadow / active` modes; `off` is the Agent default, active v1 failed its public gate, and the Top-10-safe v2 control failed its product-disjoint MRR gate. A normalized slot ledger and candidate-aware QuestionValue policy now run as diagnostics only.

Implemented:

- no-credential Agent startup;
- versioned multi-turn state with category, active/excluded terms, answered/exhausted/pending attribute lifecycle, and override anchor;
- target-blind `ParsedTurn` events with broader opener, requirement, no-preference, retry, and conservative override recognition;
- broad OR Top-120 and strict AND Top-80 field-weighted BM25 routes;
- deterministic weighted RRF and catalog-backed Top 10 output;
- target-blind normalized product/constraint evidence and an explainable Top-50 scorer;
- explicit broad/strict/fused/reranked/final routes with output-safe shadow mode;
- normalized slot records with polarity, hardness, source turn, version, and active/superseded/deleted history;
- candidate-aware clarification shadow scores based on normalized information gain, catalog coverage, answerability, and turn cost;
- catalog-price ingestion for budget evidence in shadow, without claiming numeric budget filtering;
- target-blind cross-session actual-vs-shadow question analysis with scenario slices and blocked-selection checks;
- fast, boundary, and conservative clarification policies;
- complete public result HR@10 `0.94`, MRR `0.605258`, MTTC `3.375`, and TechnicalScore `0.804077`;
- optional, versioned development trace events;
- public-session post-hoc diagnostics;
- catalog and in-memory index inspection;
- browser-controlled unit tests, 200-session evaluation, and fixed P1 generalization stress run;
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

Retrieval still compiles from the term/turn state. The normalized slot ledger is an auditable shadow representation, not yet the retrieval source of truth, so the system must not be described as a complete IntentGraph.

## Start without a terminal

On this Windows development machine, double-click:

```text
Start Observer.vbs
```

It launches the existing `tiktok` Conda environment through `pythonw.exe`, keeps the console hidden, and opens `http://127.0.0.1:8765`. The one-click development launcher defaults the Workbench to `shadow`, so rerank evidence is visible while the recommendations remain the P1 fused order. This does not change the Agent or evaluator default, which remains `off`.

Fallbacks:

- double-click `Start Observer.cmd` if Windows Script Host is disabled;
- run `python -m observer.launcher` for troubleshooting;
- inspect ignored `observer_startup_error.log` if the hidden launch fails.

If a Workbench instance is already healthy, the launcher opens that instance instead of starting another one. Use the red **停止** button in the page to stop it safely.

The launcher verifies a project-root fingerprint before reusing port 8765. If another clone owns that port, startup stops with an explicit error instead of silently opening the wrong project.

## Pages

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
- per-turn elapsed time, token usage, five-route counts, fusion evidence, normalized attribute/rerank components, slot history, actual question, candidate-aware shadow question, and final Top 10;
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

- start the fixed repository unit-test command;
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

The manifests record Git state, selected rerank and clarification diagnostic modes, schema versions, catalog/dataset and relevant Agent/attributes/reranker/slot-ledger/clarification/shadow-analysis/evaluator/generalization source hashes, runtime settings, functional elapsed time, and metrics so stale results are easier to detect. A public Workbench evaluation also writes `shadow_policy.json`, which aggregates actual-vs-shadow disagreements, attribute counts, selected-value components, blocked-selection violations, and scenario slices without reading targets, intent cards, behavior, or target ranks. The generalization result records the frozen suite registry hash, transformation counts/examples, paired session changes, and derived-corpus seed/sample hash/overlap audit. Elapsed time in a normal run is a functional observation; use the dedicated resource artifact for controlled repeated RSS/P95 evidence.

The derived corpus is generated from catalog metadata after excluding every released-public target. It is useful for product-disjoint stress testing, but it is not organizer private data and must not be presented as a hidden-leaderboard estimate.

### 交互 Lab

This is a target-free manual playground. It calls the same `Agent.reset` and `Agent.respond` methods with an opaque lab session ID and shows recommendations plus actual state and retrieval events. It can demonstrate constraint accumulation, clarification, explicit override, category goal changes, and the current parser/slot limitations without exposing a target label.

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

The runtime records loaded and on-disk hashes for `starter/agent.py`, `starter/attributes.py`, `starter/reranker.py`, `starter/slot_ledger.py`, `starter/clarification.py`, `observer/shadow_analysis.py`, the evaluator, the generalization runner, the catalog, and the public set. If any monitored file changes after startup, the page shows a restart warning and blocks evaluation, generalization, every replay, and Lab execution. Background runs recheck those fingerprints before artifact finalization, and manifests use a captured start-of-run provenance snapshot. This prevents the long-running server from mixing imported old code or cached data with later disk files and hashes.

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
POST /api/shutdown
```

The page handles the control token automatically. A manual API client must first read `/api/token` from the same loopback instance and then send its value as `X-Observer-Token`. Mutation requests must use `Content-Type: application/json`.

## Maintenance contract

- `starter.Agent` may emit target-blind session/parse/retrieval/state/policy/output events through its optional `trace_sink` callback.
- Events use `schema_version=2.0`; future Agent layers should add events rather than making the UI reimplement private algorithm logic.
- Every pipeline card must reflect current code, not roadmap intent.
- Evaluator/scoring semantics stay in `evaluator/local_evaluator.py`; the Workbench imports and invokes them rather than changing that file.
- New control actions must remain fixed and allowlisted. Do not add an arbitrary command executor.
- Add API and behavior tests whenever a control or trace field changes.
