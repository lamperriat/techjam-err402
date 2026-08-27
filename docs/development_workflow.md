# Shopping Copilot Development Workflow

This document explains what the project is optimizing, how to tell whether a change is better, and how to debug failures without guessing.

## 1. What this project actually does

This is not primarily a chatbot-writing task. It is a multi-turn ranked retrieval task.

For each hidden shopping session, the Agent receives an aggregate profile and one user message at a time. On every turn it may ask for one structured attribute, return up to 10 ordered catalog IDs, or do both. The evaluator succeeds when the hidden target `parent_asin` first appears in those scored Top 10 results.

```text
user message + current session state
                 |
                 v
          query / policy decision
                 |
                 v
          candidate retrieval
                 |
                 v
           ordered Top 10
                 |
                 v
       target hit? rank? turn?
```

Natural-language fluency is useful for the demo, but the evaluator reacts to structured `ask_attribute` and recommendation IDs. A pleasant message with the wrong IDs does not score.

## 2. How the score tells us what “better” means

```text
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency = (11 - MTTC) / 10
```

- HitRate@10 asks: did the correct product ever enter Top 10?
- MRR asks: how high was it ranked when first found?
- MTTC asks: how many turns were needed?

The practical order is recall first, ranking second, turns third. A system that asks elegant questions but removes the target from its candidates is worse.

Always compare both overall and per-scenario results. An overall improvement can hide a severe regression in Boundary or Intent Override.

Current reproducible weak baseline:

| Metric | Value |
| --- | ---: |
| HitRate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| TechnicalScore | 0.10671 |

## 3. Debug from the outside inward

Do not start by reading random ranking code. Locate the first layer where the expected target stops surviving.

| Layer | Debug question | Evidence |
| --- | --- | --- |
| Startup/contract | Did Agent construct and return valid schema/IDs? | exception, response validation, invalid/duplicate count |
| Session state | Does the active intent contain the latest constraints only? | per-turn active/superseded slots |
| Query plan | Did the query represent the active intent? | compiled query, route, hard/soft constraints |
| Retrieval | Was the target present in the wider candidate pool? | target rank at Recall@10/50/100 |
| Constraint gate | Did filtering remove a retrievable target? | before/after candidate counts, `FILTER_KILLED_TARGET` |
| Fusion/rerank | Did later ranking push the target out of Top 10? | route ranks and final rank |
| Clarification | Did the question reveal useful information without repeating? | asked/exhausted attributes and candidate reduction |
| Runtime | Did a slow or optional component fail? | elapsed time, RSS, fallback reason |

This produces an actionable miss taxonomy:

```text
STARTUP_FAILURE
INVALID_OUTPUT
RETRIEVAL_MISS
FILTER_KILLED_TARGET
FUSION_MISS
RERANK_MISS
WRONG_QUESTION
STALE_OVERRIDE_STATE
TIMEOUT
```

## 4. The development loop

Every improvement should change one causal layer at a time.

1. State one hypothesis, such as “remembering the category will improve Browsing recall after a short clarification reply.”
2. Add a focused unit test that fails before the change.
3. Implement the smallest change that passes it.
4. Run all unit tests.
5. Run the full public evaluator.
6. Compare overall metrics, four scenarios, runtime, and the relevant intermediate diagnostic.
7. Keep the change only when the evidence matches the hypothesis and no unacceptable regression appears.

An experiment record should contain:

```text
hypothesis
code/config version
data hash
overall metrics
scenario metrics
intermediate metric
runtime/resources
decision: keep / revise / revert
```

Do not combine state, dense retrieval, reranking, and question policy into one change. If the score moves, there would be no reliable explanation.

## 5. Commands on this machine

In PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& 'C:\Users\danie\anaconda3\shell\condabin\conda-hook.ps1'
conda activate tiktok
Set-Location D:\tiktok\techjam-err402
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run the deterministic public evaluator without LLM credentials:

```powershell
python -m evaluator.local_evaluator
```

The evaluator overwrites ignored `results.json`. Copy important experiment outputs into another ignored experiment directory before the next run.

## 6. Agent Workbench

For normal development on this machine, double-click `Start Observer.vbs`. It launches through the existing `tiktok` environment without a terminal and opens `http://127.0.0.1:8765`. Use the in-page **停止** action when finished. `Start Observer.cmd` and `python -m observer.launcher` remain fallback launch paths.

The Workbench now covers the complete local development loop:

```text
environment / data / Git / index health
-> current algorithm registry
-> public single-session replay
-> actual target-blind Agent events
-> post-hoc target survival and score diagnosis
-> full evaluator or unit-test background job
-> progress, logs, metrics, and versioned experiment artifact
```

It also includes catalog/FTS5 search, full product JSON, a target-free manual Agent lab, experiment comparison, and an allowlisted document library. The control plane runs only fixed test/evaluation actions and does not execute arbitrary shell input. A per-instance token and same-origin/Host checks protect the local API.

The Workbench fingerprints the Agent/evaluator code loaded at startup and compares it with disk. After either file changes, evaluation, refreshed replay, and Lab execution are blocked until the Workbench is restarted, preventing an experiment from silently running an old imported class.

Trace events are emitted by the actual Agent through an optional versioned callback. The Session layer explicitly reports `reset-only / stateless baseline`; it does not display simulator disclosure as if it were Agent memory. Public target rank is joined after `Agent.respond` and is labelled post-hoc.

Every public replay gives the Agent an opaque random session ID. The Agent receives only profile, generated user message, turn, and `top_k`; it never receives `sample_id`, target, intent card, scenario, behavior, or prior results. The server is loopback-only and must not be attached to private final labels.

Successful browser-started evaluations refresh `results.json` and write ignored versioned artifacts under `experiments/`. See `docs/agent_workbench.md` for pages, endpoints, safety boundaries, and the trace maintenance contract.

## 7. Current improvement order

1. Baseline reliability: no-credential startup, tests, catalog validation, reproducible score.
2. Contract and diagnostics: strict response guard, per-turn trace, candidate survival and miss taxonomy.
3. Versioned session state: accumulation, negation, replacement, category change, Override and no-preference exhaustion.
4. Candidate-aware clarification and Buying/Browsing strategy.
5. Fielded sparse and attribute retrieval with safe constraint relaxation.
6. Optional dense/fusion and reranking only after measured recall/resource gates pass.

The current repository has completed baseline reliability and the local Workbench/control-plane layer. It has not yet implemented the IntentGraph state, clarification policy, attribute gate, hybrid retrieval, fusion, or semantic reranking.
