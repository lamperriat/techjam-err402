# Implementation Status and Change Log

This tracked document records what the repository actually implements, how it relates to the official participant kit, what has been verified, and what remains unimplemented. Proposed designs are not implementation claims.

## Maintenance rules

- Keep the current-version section accurate before adding new roadmap claims.
- Add dated entries newest first and link each claim to code, tests, or a reproducible evaluation artifact.
- Label organizer-published metrics separately from metrics reproduced on this project commit.
- Record model, dependencies, data hash, latency, memory, token use, estimated cost, and fallback behavior when applicable.
- Never include credentials, private evaluation data, copied organizer-only material, or rules based on public target IDs.

## Current version at a glance

| Item | Current state |
| --- | --- |
| Implementation commit audited | `8f9e64d` (`feat: add reliable baseline and layer observer`) |
| Official baseline included | `34078351e1c3615e5505a2e829600b56a542e462` via merge `1496fec` |
| Base | Official TechJam conversational-search participant kit |
| Executable approach | Stateless, current-message-only SQLite FTS5/BM25 |
| Participant extensions | OpenAI-compatible JSON client/token accounting, optional injection, and local Layer Observer |
| LLM used for retrieval/ranking | No |
| Multi-turn intent state | Not implemented |
| Clarification policy | Not implemented; `ask_attribute` is always `null` |
| Hybrid/dense retrieval | Not implemented |
| Reranking | Not implemented |
| Local catalog | Present and verified: 50,000 unique `parent_asin` values |
| Current local evaluator result | Reproduced exactly: HR@10 `0.125`, MRR `0.068034`, MTTC `9.81`, TechnicalScore `0.10671` |
| Offline/no-credential startup | Implemented for default BM25 Agent; optional client may be injected explicitly |

## Repository provenance

This repository is a participant-derived version of the official [TechJam conversational search kit](https://github.com/TechJam2026/techjam-conversational-search), not a pristine copy. The `pre` branch now incorporates the current official `main` commit as an ancestor while retaining participant work.

The provenance is confirmed at Git object level:

```text
2a6cc8e776da66ce69b1cbd237838fbc43f32587
  Publish conversational search challenge
  Official participant-kit tag and this repository's root commit

9a35be51780ff1caf89eceaabca34259e946f40f
  Clarify participant model API costs
  Same official commit object in both histories

914879c354395b2da908411ecfc09c0ab293650e
  feat: add llm client
  Earlier participant extension

34078351e1c3615e5505a2e829600b56a542e462
  Clarify TechnicalScore judging role
  Current official upstream/main at verification time

8f9e64d
  feat: add reliable baseline and layer observer
  Current implementation commit

1496fec
  Merge remote-tracking branch 'upstream/main' into pre
  Establishes official 3407835 as an ancestor of pre
```

The official repository later added `34078351e1c3615e5505a2e829600b56a542e462`, clarifying that TechnicalScore is an objective input to Technical Execution rather than a separate judging criterion or the whole Technical Execution score. That commit is now merged into `pre`; it does not change the evaluator.

The participant remote is `origin=https://github.com/lamperriat/techjam-err402.git`, and the official source is configured as `upstream=https://github.com/TechJam2026/techjam-conversational-search.git`. The GitHub UI fork relationship was not independently established, but identical commit objects and the successful upstream merge prove source-tree ancestry.

The official repository content matches the Shopping Copilot task through its 50,000-product catalog, 200/800 sessions, four scenarios, 10-turn protocol, exact `parent_asin` scoring, Agent API, and evaluator. The official kit itself does not contain the literal numeric label “Task 4,” so that number should be cross-checked against the event page.

## Implemented functionality

### Agent construction and catalog indexing

File: `starter/agent.py`

- Accepts a catalog path, defaulting to `data/catalog.jsonl`.
- Accepts an optional injected `llm_client`; the default is `None` and does not import or construct the OpenAI client.
- Creates an in-memory SQLite database and FTS5 `products` table.
- Keeps `parent_asin` as an unindexed identifier field.
- Indexes `title`, `categories`, `features`, `details`, `store`, and `description`.
- Uses `unicode61` tokenization with diacritic removal.
- Reads JSONL products sequentially and inserts them in batches of 1,000.
- Flattens dict values as `key value` text and converts list items to strings.

Not implemented in the index:

- structured price/rating indexes;
- normalized color, material, size, brand, style, or use-case fields;
- catalog schema, row-count, ID-uniqueness, or checksum validation;
- dense vectors or attribute inverted indexes.

### Session handling

File: `starter/agent.py`

- `reset(session_id, user_profile)` records only that the session ID exists.
- `respond` rejects a session that has not been reset.

The supplied aggregate user profile is not stored or used. There is no turn history, slot ledger, asked-attribute registry, candidate cache, override version, or session-local personalization.

### Query processing and retrieval

File: `starter/agent.py`

- Uses only the current `user_message`.
- Extracts ASCII alphanumeric tokens with `[a-z0-9]+`.
- Lowercases tokens, removes a built-in English stopword list, removes one-character terms, deduplicates in first-seen order, and keeps at most 40 terms.
- Quotes each term and joins all terms with FTS `OR`.
- Returns no recommendations for an empty expression.
- Executes one SQLite BM25 route with weights:

```text
parent_asin  0.0
title        6.0
categories   4.0
features     2.5
details      2.5
store        1.5
description  1.0
```

- Applies the requested SQL `LIMIT top_k` and preserves the resulting BM25 order.

There is no query rewrite, structured filtering, hard/soft constraint distinction, Buying/Browsing strategy, dense retrieval, rank fusion, relaxation, diversity policy, or reranker.

### Turn response

File: `starter/agent.py`

- Returns a fixed customer-facing message.
- Always returns `ask_attribute: null`.
- Returns recommendations as objects containing `parent_asin`.
- Reports zero token usage when no client is injected; otherwise consumes usage from the injected client.

The Agent never calls the model's `generate_json` method. The default BM25 path now runs without LLM configuration and reports zero model tokens; model-assisted features must explicitly construct and inject a client.

### OpenAI-compatible LLM client

File: `utils/llm_client.py`

- Loads `.env` from the current working directory.
- Requires `LLM_API_KEY` and `LLM_MODEL`; accepts an optional `LLM_BASE_URL`.
- Creates an OpenAI-compatible SDK client.
- Sends non-streaming chat-completion requests in JSON-object mode.
- Parses the response and requires a JSON object at its root.
- Records last-call, total, and unreported prompt/completion token counts.
- Allows the Agent to consume unreported usage once per response.
- Logs and raises clear errors for empty, invalid JSON, or non-object responses.

It does not provide an application-level timeout policy, fallback, business-output schema validation, or retrieval/ranking integration. SDK retry and timeout behavior remains dependent on the installed OpenAI package version.

### Public evaluator

File: `evaluator/local_evaluator.py`

- Loads the public sessions and full product catalog.
- Builds catalog ID, category, and product lookup structures.
- Derives hidden intent cards deterministically from target metadata when they are absent from the public set.
- Simulates Buying, Browsing, Intent Override, and Boundary sessions for up to 10 turns.
- Prevents an Intent Override session from converting before the replacement intent is sent.
- Filters invalid and duplicate product IDs and scores only the first 10 valid unique values.
- Uses exact `parent_asin` equality for a hit.
- Calculates Hit Rate@10, MRR, MTTC, Efficiency, the recommended TechnicalScore, per-scenario core metrics, and reported token use.
- Treats misses as turn 11 for MTTC.
- Writes detailed session results to `results.json` through the module entry point.

The participant commit changed the evaluator only by introducing an `AgentBase` protocol type; scoring behavior was not changed.

Important boundaries:

- Only `ask_attribute`, not question prose in `message`, drives the simulated customer's next answer.
- The public evaluator accepts some outputs that are looser than the JSON contract, including string recommendation entries. The project must not depend on those leniencies.
- `respond` exceptions are converted to empty turn results, but catalog loading, `Agent` construction, and `reset` exceptions can terminate the whole run.
- The evaluator aggregates token use but does not enforce a real timeout or measure latency and memory.
- Overall score output uses `recommended_technical_score`; the published baseline artifact uses `technical_score`.

### Tests currently present

Files: `tests/test_agent.py`, `tests/test_evaluator.py`, `tests/test_llm_client.py`

Current tests cover:

- LLM configuration loading and required variables;
- provider-neutral key handling;
- JSON-object generation, malformed/empty/non-object errors;
- token usage accumulation and consumption;
- no-credential Agent startup, zero usage, and injected-client usage reporting;
- recommendation validity, deduplication, and order;
- miss-as-turn-11 metric behavior;
- evaluator derivation of hidden fields from product metadata.

There are no focused unit tests yet for actual BM25 ranking, strict API schema, multi-turn state, Override, Boundary, constraints, hybrid retrieval, reranking, deterministic repeated evaluation, timeout, latency, or peak memory. Full-catalog construction and the 200-session evaluator are covered by recorded manual verification rather than unit tests.

### Configuration and supporting artifacts

- `.env.example` documents the LLM variables and currently uses a DeepSeek-compatible base URL example.
- `requirements.txt` declares `openai>=1.0.0` and `python-dotenv>=1.0.0`; versions are not upper-bounded or locked.
- `notebooks/test_llm_client.ipynb` contains a historical provider smoke-test output. It is not evidence of Agent-level evaluation or metric improvement.
- `data/public_set.jsonl` is present and statically confirmed to contain 200 sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary.
- `data/catalog.jsonl` is intentionally ignored and is now present locally. The downloaded official `catalog.jsonl.gz` is 19,235,996 bytes and its independently calculated SHA256 is `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`, matching the Release API.
- The decompressed catalog is 60,546,327 bytes and was validated as 50,000 parseable JSON rows, 50,000 unique `parent_asin` values, and zero missing `parent_asin` values.
- The existing Conda environment `D:\450\conda\envs\tiktok` runs Python 3.11.16. `requirements.txt` installed `openai==3.5.0` and `python-dotenv==1.2.3` plus transitive dependencies; these resolved versions are environment facts, not repository locks.

## Contract and competition alignment

| Requirement | Current status | Notes |
| --- | --- | --- |
| Export `Agent.reset` and `Agent.respond` | Implemented | Correct method shapes |
| Return message/ask/recommendations | Implemented | `ask_attribute` always null |
| Use catalog-valid exact IDs | Implemented and verified across the full 200-session public evaluation | Evaluator still normalizes invalid/duplicate outputs defensively |
| Return no more than scored Top 10 | Implemented under official `top_k=10` | SQL limit uses input `top_k` |
| Preserve frozen catalog | Implemented | Reads catalog; does not mutate it |
| Support up to 10 turns | Interface-compatible but stateless | Turn argument is unused |
| Buying/Browsing adaptation | Not implemented | One BM25 route |
| Intent Override state rewrite | Not implemented | No history or slots |
| Boundary/no-preference handling | Not implemented | No questions or exhausted state |
| Aggregate-profile use | Not implemented | Profile ignored |
| Offline/no-network execution | Default BM25 path implemented and verified without credentials | Optional model features still require their own fallback policy |
| Declared external dependencies | Partially implemented | Requirements exist; no lock/upper bound |
| Token disclosure | Implemented for SDK calls | Agent currently makes no calls |
| Latency/cost/fallback disclosure | Not yet implemented | Required before submission |
| Reproducible one-command run | Verified without LLM environment variables | Run from the repository root with the existing environment |
| UI/multimodal/heavy vector DB avoidance | Aligned | None are present |

## Competition timing and compliance status

The [Devpost overview](https://tiktoktechjam2026.devpost.com/) and [Official Rules](https://tiktoktechjam2026.devpost.com/rules) were checked on 2026-08-27 SGT.

- The Submission Period is 2026-08-29 12:00 through 2026-09-01 12:00 SGT. Existing projects must be significantly updated after that period begins, so the implementation history must make the post-start work clear.
- Technical Workshops are listed for 2026-08-28. The exact Track 4 time of 16:00–16:45 appears in the local challenge brief but was not independently found on the public Devpost schedule.
- The Devpost overview requires a written solution/technology description, a public repository with a comprehensive README, and a public three-minute YouTube end-to-end demo.
- The Official Rules require English materials or English translations, free judge access through the judging period, authorization for third-party integrations, and compliance with open-source licenses.
- The Official Rules prohibit attempts to re-identify people represented in the anonymized data and require competition data used or processed by the participant to be deleted when the competition is complete.

There is a material rubric discrepancy. The controlling Official Rules describe Stage Two using four equally weighted criteria: Technical Execution, Innovation & Problem Insight, Feasibility & Practicality, and Impact & Relevance. The Devpost overview also lists Presentation & Communication but marks it final-only and does not publish weights. The local `problem-statement.md` instead records 35/20/20/15/10. Because the Official Rules state that they prevail over inconsistent TechJam materials, the current plan uses the four equal Stage Two evidence tracks, still prepares presentation evidence for the final, and records this as an organizer-clarification question. TechnicalScore remains only an objective input to Technical Execution.

This section records competition obligations and planning constraints, not implemented Agent functionality.

## Evaluation status

The following values are inherited organizer-published references from `docs/baseline_results.json`:

```text
sample_count       200
Hit Rate@10        0.125
MRR                0.068034
MTTC               9.81
Efficiency         0.119
TechnicalScore     0.10671
```

They were first reproduced exactly for commit `914879c` with process-local placeholder configuration, then reproduced again after the optional-client change with all LLM environment variables absent. The second evaluator completed all 200 sessions and wrote `results.json` in approximately 25.4 seconds with zero token usage. This proves score preservation and no-credential startup, but it is not a controlled latency/RSS benchmark.

Scenario results were Boundary HR `0.0`, Browsing HR `0.025`, Buying HR `0.2375`, and Intent Override HR `0.133333`. These reproduce the weak starter, not an improved IntentGraph result.

## Known gaps and immediate risks

1. Current-message-only retrieval loses category/context after clarification replies and cannot invalidate old override preferences.
2. The fixed null question policy cannot obtain simulator information in Browsing or Boundary sessions.
3. The observer exposes per-turn candidate survival, but the Agent still lacks a strict output guard before evaluator normalization.
4. There is no controlled resource/repeatability/offline benchmark or experiment manifest.
5. Dependencies have lower bounds but no upper bounds or lockfile; the resolved environment is not yet portable evidence.
6. The official `upstream` is configured, but periodic rule/source checks are not automated.
7. Judging materials conflict: Official Rules specify four equally weighted Stage Two criteria, while the local brief records 35/20/20/15/10; the Rules-first interpretation is documented, but the organizer should still clarify the Track/final mapping.

## Change log

### 2026-08-27 — `pre` branch and official baseline alignment

- Created `pre` from the participant `main` while preserving all intended implementation work.
- Configured the official repository as `upstream` and fetched its current `main` at `3407835`.
- Confirmed the histories share official commits `2a6cc8e` and `9a35be5`, then merge-aligned `pre` so `3407835` is an explicit ancestor.
- Kept the ignored catalog, release assets, evaluator output, internal plan, current-architecture snapshot, and caches outside the commit.

### 2026-08-27 — Local Agent Layer Observer

- Added a standard-library local HTTP server and offline web UI under `observer/`; no frontend framework, CDN, external service, or new dependency is required.
- Added a single-session trace runner that reuses official public simulator helpers without modifying the evaluator.
- Exposed Input, Parse, Session, Retrieval, Ranking, Policy, and Score views per turn, including BM25 candidate count, public target retrieval rank, Top-10 target rank, override eligibility, normalized recommendations, and session score contribution.
- Kept public ground truth and derived intent cards outside `Agent.respond`; the UI labels them as diagnostic-only data.
- Added public-session search/scenario/result filters, aggregate metric cards, turn navigation, conversation I/O, target metadata, and ranked product inspection.
- Added trace and HTTP API tests. All 14 repository tests pass.
- Verified the real server with 200 sessions, the `public_0001` trace, JSON endpoints, static HTML, and a headless Chromium render at 1600x1100.

### 2026-08-27 — Optional LLM client and no-credential baseline

- Removed the default Agent's import-time and construction-time dependency on `LLMClient`.
- Added optional client injection so later model-assisted features can still report consumed usage without coupling baseline startup to credentials.
- Added regression tests for no-environment startup/zero usage and injected-client usage; all 11 tests pass.
- Re-ran the complete public evaluator with `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` absent. All baseline and scenario metrics remained exactly unchanged and token usage remained zero.
- Added `docs/development_workflow.md` to explain the task, score, debug funnel, experiment discipline, commands, and improvement order.

### 2026-08-27 — Full baseline reproduction and setup cleanup

- Ran the official 200-session public evaluator with the existing `tiktok` Conda environment and verified exact agreement with the organizer-published weak BM25 metrics.
- Recorded zero prompt/completion tokens and the four scenario metric groups in `results.json`.
- Confirmed the run used process-local placeholder LLM configuration only; no `.env`, credential, or persistent environment variable was created.
- Removed the mistakenly installed `D:\tiktok\miniconda3` tree and its 133,071,768-byte installer after resolving and checking both exact targets. The existing Anaconda installation and `D:\450\conda\envs\tiktok` were preserved.
- No Agent, evaluator, dependency declaration, public labels, or catalog content changed.

### 2026-08-27 — Local Conda, dependency, and catalog bootstrap

Environment and data changes:

- Reused the existing named Conda environment at `D:\450\conda\envs\tiktok` with Python 3.11.16.
- Installed the repository-declared `openai` and `python-dotenv` dependencies into that environment.
- Downloaded `catalog.jsonl.gz` from the official `participant-kit` GitHub Release into ignored `data/releases/` storage.
- Independently matched the downloaded SHA256 to the official Release digest, then decompressed it to ignored `data/catalog.jsonl`.

Verification performed:

- Parsed all 50,000 catalog rows and confirmed 50,000 unique, nonmissing `parent_asin` values.
- Ran all 10 existing unit tests successfully.
- Loaded the full catalog into the current SQLite FTS5 Agent and verified one query returned 10 recommendations with zero token usage.
- Verified PowerShell activation resolves `tiktok` to `D:\450\conda\envs\tiktok`; the current shell requires a process-scoped ExecutionPolicy bypass to load the Conda hook.

Verification not performed:

- The full 200-session evaluator, baseline metric reproduction, latency/RSS benchmark, and no-network run.
- No Agent, evaluator, test, requirement, or catalog source content was modified by this bootstrap.

### 2026-08-27 — Devpost rules and execution-plan validation

Documentation and planning changes:

- Verified the official Submission Period, judging/final milestones, submission artifacts, and the significant-post-start-update requirement.
- Replaced the earlier provisional 35/20/20/15/10 assumption with the Rules-first interpretation: four equally weighted Stage Two criteria, with Presentation & Communication retained as final-stage evidence.
- Added a 72-hour execution map, Workshop questions, internal resource/quality gates, and a submission/compliance checklist to the ignored internal plan.
- Kept dense retrieval, RRF, cross-encoder reranking, and EVSI as measured experiments. Dense is not an unconditional P0 before baseline, state, contract, and sparse/attribute diagnostics are working.
- Recorded English/translation, free judge access, third-party authorization/license, no-re-identification, and post-competition data-deletion obligations.

Verification boundary:

- This pass changed documentation only. It did not implement or modify Agent behavior, retrieval, session state, evaluator logic, dependencies, or data.
- No new metric, latency, memory, offline, or model-performance claim was produced.
- The public Devpost page confirmed only the Workshop date; the local brief's exact Track 4 time remains pending independent organizer confirmation.

### 2026-08-27 — Requirements, provenance, and documentation audit

Implemented documentation changes:

- Confirmed official participant-kit ancestry using identical commit objects.
- Distinguished the official base, participant LLM-client extension, and later official clarification.
- Synchronized the official `3407835` TechnicalScore clarification into `README.md` and `docs/competition_specification.md`.
- Removed the stale README reference to a participant release checklist that is not present in the kit.
- Added ignored internal planning and current-architecture documents while keeping this implementation record tracked.
- Reframed dense retrieval, RRF, reranking, and score-aware clarification as planned experiments rather than current features or mandatory requirements.
- Recorded the current startup regression, local artifact gaps, evaluator semantics, contract alignment, and verification boundary.

Verification performed:

- Audited Git history, remotes, official tag and current official main refs.
- Reviewed the Agent, evaluator, LLM client, tests, notebook, requirements, API contract, evaluation config, baseline artifact, submission rules, public-set shape, README, and challenge brief.
- Confirmed the public scenario counts as 80/80/30/10.
- Confirmed `docs/internal_plan.md` and `docs/current_architecture.md` are ignored while this file is not ignored.

Verification not performed:

- Unit tests and evaluator execution, because no usable Python is installed.
- Catalog checksum, row/ID validation, BM25 index construction, and baseline reproduction, because `data/catalog.jsonl` is absent.
- LLM or network smoke tests; no credentials were used.

### 2026-08-27 — Participant LLM client extension (`914879c`)

Already present before this documentation audit:

- Added the OpenAI-compatible JSON client, `.env.example`, Python dependencies, usage accounting, tests, and notebook.
- Initialized that client from the starter Agent and reported consumed token usage.
- Added a structural evaluator protocol type without changing metric behavior.
- Added the local challenge brief and repository working guidelines.

This extension did not implement LLM-based intent parsing, query rewriting, retrieval, recommendation, or reranking.
