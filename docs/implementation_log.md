# Implementation Log

This tracked document records only code and behavior that exist in the repository and have been verified. Planning, hypotheses, and unvalidated designs belong in ignored `docs/internal_plan.md`; the current working-tree architecture belongs in ignored `docs/current_architecture.md`.

Last updated: 2026-08-27 SGT.

## Current verified integration baseline

- Branch: `pre-v06-integration`
- Integration commit: `5fed7a7` (`feat: integrate stateful sparse shopping agent`)
- Parent checkpoint: `f4e435b` (`feat: add agent layer workbench`)
- Official upstream main checked on 2026-08-27: `34078351e1c3615e5505a2e829600b56a542e462`
- Runtime: Python 3.11.16 in the existing `tiktok` Conda environment
- Catalog: 50,000 parseable rows and 50,000 unique non-empty `parent_asin` values
- Public set: 200 sessions and 200 unique targets, split 80 Buying / 80 Browsing / 30 Intent Override / 10 Boundary
- Default execution: offline, no LLM object, no API key, no network call, zero reported tokens

## Current public evaluation

The integrated Agent was run against the complete released 200-session evaluator after all code changes in this entry.

| Scope | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 0.925000 | 0.586265 | 3.175000 |
| Browsing | 80 | 0.975000 | 0.600724 | 3.012500 |
| Intent Override | 30 | 0.900000 | 0.655265 | 4.700000 |
| Boundary | 10 | 0.900000 | 0.643452 | 4.000000 |
| Overall | 200 | 0.940000 | 0.605258 | 3.380000 |

Overall Efficiency is `0.762000`; recommended TechnicalScore is `0.803977`; prompt and completion token usage are both zero.

The complete parsed JSON, including all 200 ordered session rows, is equal to the independently verified v0.6 handoff result. Raw Windows output may use CRLF while the reference uses LF; JSON semantic equality and normalized content are identical.

The pre-integration Workbench checkpoint reproduced the official weak baseline at HR@10 `0.125`, MRR `0.068034`, MTTC `9.81`, Efficiency `0.119`, and TechnicalScore `0.10671`. The current gain therefore comes from the stateful sparse Agent integration, not from the browser observer.

These are public-development metrics, not a claim about the private 800 sessions.

## Implemented Agent behavior

### Session and lifecycle

`starter/agent.py` now implements a per-session `SessionState` containing:

- aggregate profile copy;
- active category;
- active and excluded retrieval terms;
- known, asked, and exhausted attribute classes;
- per-turn terms and attribute classes;
- version, version anchor, and override count;
- the fast-policy `prefer_other_next` event.

`reset` replaces prior state for that session. `respond` validates turn 1–10 and positive `top_k`, updates state, ranks products, selects at most one allowed clarification attribute, and returns at most 10 catalog-backed recommendation objects. `drop_session` releases development replay/Lab state.

The SQLite connection uses `check_same_thread=False`; state and query operations are protected by an `RLock` for the multi-threaded local Workbench.

### Parsing and state transitions

Implemented deterministic parsing includes:

- explicit shopping-category extraction;
- public-protocol constraint fragments;
- material, color, size, style, use-case, and budget class detection;
- simple `not`, `no`, and `without` negative-term extraction, including `not too X`;
- no-preference/exhausted attribute handling;
- explicit ignore/change-mind/replace detection;
- repeated override version-anchor movement;
- category-goal changes that clear the previous goal's term and question lifecycle.

A plain sentence such as `Actually, cotton sounds fine` no longer triggers an override solely because it contains `actually`.

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

Ties use broad rank and then `parent_asin`. The response returns the first `min(top_k, 10)` fused IDs.

### Clarification policy

Three explicit policies are supported:

- `fast` (default): after a no-preference reply, prefer `other` to obtain the remaining disclosed constraints quickly;
- `boundary`: use that shortcut only for a direct Boundary-style reply;
- `conservative`: continue through the fixed allowed-attribute order.

Known, asked, and exhausted attributes are not re-asked. Turn 10 always returns `ask_attribute=null`.

The selected policy can be passed to `Agent` or set with `TECHJAM_QUESTION_POLICY`. Workbench experiment manifests now record it because it directly changes results.

### Optional LLM boundary

The default Agent does not import, construct, or call an LLM client. An explicitly injected compatible client is used only to consume and report token usage; it does not currently parse intent, retrieve, rerank, or write response prose.

`utils/llm_client.py` remains available for measured future experiments and is covered by configuration, JSON-response, error, and usage tests.

## Implemented Agent Workbench

The local Workbench is a loopback-only development control plane, not part of official scoring.

### Startup and pages

- `Start Observer.vbs`: hidden `pythonw.exe` launch using the existing `tiktok` environment.
- `Start Observer.cmd` and `python -m observer.launcher`: troubleshooting fallbacks.
- Overview: runtime, Git, source fingerprint, data/hash/index health, metrics, and truthful algorithm registry.
- Session Diagnostics: deterministic public replay, actual Agent events, output validation, and post-hoc score diagnosis.
- Catalog & Index: 50k catalog browsing, field-weighted BM25 search, and raw product JSON.
- Runs & Experiments: fixed test/evaluator jobs, progress, logs, cancellation, metrics, and versioned manifests.
- Lab: target-free calls to the real `reset/respond` interface with opaque session IDs.
- Documents: read-only allowlisted project documentation and source.

### Trace integration

The Agent can emit optional versioned, target-blind events for:

```text
session -> parse -> retrieval -> state -> policy -> output
```

Retrieval events expose broad/strict/fused counts, the weighted-RRF formula, and actual Top-10 route evidence. State events expose only information derived from the profile and conversation.

For a public replay, the Observer calls `Agent.respond` with a random UUID session. Only after the response does it compare the target with target-blind `debug_rankings` route IDs to compute broad, strict, fused, and Top-10 target ranks. Target, scenario, intent card, behavior, prior result, and public sample ID are never passed into Agent decision features.

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
- loaded-vs-disk Agent/evaluator plus catalog/public-set fingerprints, with stale-runtime blocking for every replay, evaluation, and Lab call;
- evaluation provenance captured before the background job and rechecked before artifact finalization, so a manifest cannot mix loaded code/data with later disk hashes.

The Workbench must not be publicly deployed or connected to private final labels.

## Strict result verification

`scripts/compare_results.py` supports both metric reporting and strict verification.

```powershell
python scripts/compare_results.py run_a.json run_b.json
python scripts/compare_results.py --assert-equal run_a.json run_b.json
```

Strict mode recursively compares the complete parsed objects, including key presence, list order, scenario metrics, usage, and every session row. Any semantic difference exits with code 1 and prints the first JSON paths that differ. Whitespace, indentation, CRLF, and LF formatting differences do not fail.

This replaces the handoff comparator behavior that printed aggregate deltas but always exited successfully.

## Verification completed

- 32 Python unit/integration tests pass.
- Agent tests cover accumulation, category changes, negative phrases, false override prevention, first/repeated/selective overrides, Boundary exhaustion, question policies, broad/strict/fused routes, output cap/final turn, optional usage, and target-blind trace events.
- Comparator tests cover formatting/line-ending equality, session-level mismatch, missing keys/list order, and invalid JSON.
- Existing evaluator, LLM client, Workbench replay, catalog/Lab/background evaluation, HTTP token/cross-site, and exclusive-listener tests pass.
- `node --check observer/static/app.js` passes.
- The complete 200-session evaluator completed successfully with no LLM environment variables.
- `scripts/compare_results.py --assert-equal` reports a strict match against the independently verified v0.6 result.
- A headless Chrome smoke test rendered the live loopback Workbench against 50,000 indexed products and 200 sessions; Overview, state/fusion pipeline, public Trace, two-turn Lab state, and background Tests were exercised successfully with `restart_required=false`.

## Provenance and audit corrections

The v0.6 release material was independently audited before integration:

- outer ZIP hash and 73/73 declared file hashes matched;
- source.zip, bundle target tree, and the shared project source files matched byte-for-byte;
- bundle history is complete and contains official `3407835` as an ancestor;
- the patch exactly represents `367f1bf -> 89ef66c`, not `3407835 -> 89ef66c`;
- the participant evaluator differs from official `3407835` by an `AgentBase` Protocol/type annotation only; scoring behavior was cross-run with the official blob and produced the same complete result;
- the packaged `original_baseline_reference/starter_agent.py` is a participant optional-client baseline, not the pristine official starter.

Documentation must therefore say “official scoring behavior preserved,” not “unmodified official evaluator.”

## Current limitations

1. The state model is a term/turn ledger, not a normalized slot-level IntentGraph. An override can still remove unrelated preferences introduced in the same anchor turn.
2. Parser patterns are strongly aligned with the released simulator's English templates.
3. The default fast policy benefits from the public simulator's `other` disclosure behavior. This is protocol adaptation, not direct label leakage, but it creates public-strategy overfitting risk.
4. There is no independent product-disjoint or phrase-perturbed holdout result yet.
5. Clarification uses a fixed order, not candidate entropy or expected information gain.
6. No explicit Buying/Browsing router is implemented; hidden scenario labels are never available to the Agent.
7. Profile data is stored but not used for personalization.
8. There is no structured hard filter/relaxation ledger, dense retrieval, learned reranker, or semantic reranker.
9. Full controlled latency, P95, peak RSS, repeated no-network, and dependency-lock evidence are not yet recorded.

The current implementation should be described as a **versioned stateful sparse retrieval and weighted-RRF baseline with heuristic clarification**, not as the complete IntentGraph target architecture.

## Change history

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
