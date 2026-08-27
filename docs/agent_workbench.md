# Agent Workbench Guide

The Agent Workbench is a local, offline-first development control plane for the Shopping Copilot. It makes the current implementation observable without presenting planned IntentGraph layers as completed code.

## Current truth

The current Agent is still a stateless, current-message-only, field-weighted SQLite FTS5/BM25 baseline.

Implemented:

- no-credential Agent startup;
- offline, rule-based BM25 retrieval that reproduces the official baseline in the verified environment;
- exact reproduction of the official public baseline;
- optional, versioned development trace events;
- public-session post-hoc diagnostics;
- catalog and in-memory index inspection;
- browser-controlled unit tests and 200-session evaluation;
- versioned local experiment artifacts;
- a manual Agent playground;
- a read-only project document library.

Not implemented in the Agent:

- accumulated multi-turn slots;
- intent-override ledger or conflict replacement;
- no-preference/exhausted-attribute memory;
- structured hard filters or safe relaxation;
- Buying/Browsing routing;
- dense retrieval, RRF, or semantic reranking;
- candidate-aware clarification;
- profile-based ranking.

An external planning note that says a Stateful BM25 phase and seven state tests were already implemented refers to a separate, unavailable sandbox artifact. Those claims are not properties of this repository and must not be used in demos or experiment reports.

## Start without a terminal

On this Windows development machine, double-click:

```text
Start Observer.vbs
```

It launches the existing `tiktok` Conda environment through `pythonw.exe`, keeps the console hidden, and opens `http://127.0.0.1:8765`.

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
- actual Agent trace events for session, parse, retrieval, policy, and output;
- per-turn elapsed time, token usage, candidate count, BM25 scores, and normalized Top 10;
- post-hoc public target rank, hit eligibility, and derived score contribution;
- output malformed/invalid/duplicate diagnostics and a baseline-aware miss code;
- trace refresh and JSON export.

The UI intentionally separates **actual** Agent events from **post-hoc** public-label annotations.

### 商品与索引

- browse all 50,000 frozen catalog products;
- run the same tokenization and field-weighted FTS5/BM25 search used by the current Agent;
- inspect match counts, BM25 scores, catalog metadata, and complete raw product JSON.

### 运行与实验

- start the fixed repository unit-test command;
- start the official public evaluator with a fresh Agent index;
- see current session, progress, elapsed time, and captured logs;
- request cancellation;
- compare the official baseline, current `results.json`, and timestamped local experiments.

The Workbench does not accept arbitrary shell commands. A successful evaluation writes ignored local artifacts under:

```text
experiments/<timestamp>_public_eval/
  manifest.json
  results.json
```

The manifest records Git state, catalog/dataset and Agent/evaluator source hashes, runtime/evaluation settings, functional elapsed time, and metrics so stale results are easier to detect. Elapsed time is a single functional-run observation, not a controlled benchmark.

### 交互 Lab

This is a target-free manual playground. It calls the same `Agent.reset` and `Agent.respond` methods with an opaque lab session ID and shows recommendations plus actual trace events. It is useful for feeling the current limitation: follow-up messages do not preserve earlier constraints because state is not implemented yet.

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

The runtime records loaded and on-disk hashes for `starter/agent.py` and the evaluator. If either file changes after startup, the page shows a restart warning and blocks evaluation, refreshed replay, and Lab execution. This prevents tests from exercising disk-new code while the long-running server evaluates an imported old class.

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
POST /api/jobs/tests
POST /api/jobs/<job_id>/cancel
POST /api/lab/reset
POST /api/lab/respond
POST /api/shutdown
```

The page handles the control token automatically. A manual API client must first read `/api/token` from the same loopback instance and then send its value as `X-Observer-Token`. Mutation requests must use `Content-Type: application/json`.

## Maintenance contract

- `starter.Agent` may emit target-blind trace events through its optional `trace_sink` callback.
- Events use `schema_version=1.0`; future Agent layers should add events rather than making the UI reimplement private algorithm logic.
- Every pipeline card must reflect current code, not roadmap intent.
- Official evaluator semantics stay in `evaluator/local_evaluator.py`; the Workbench imports and invokes them rather than changing that file.
- New control actions must remain fixed and allowlisted. Do not add an arbitrary command executor.
- Add API and behavior tests whenever a control or trace field changes.
