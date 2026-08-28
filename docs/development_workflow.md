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

Current integrated stateful-sparse result:

| Metric | Value |
| --- | ---: |
| HitRate@10 | 0.945000 |
| MRR | 0.606175 |
| MTTC | 3.335000 |
| Efficiency | 0.766500 |
| TechnicalScore | 0.807652 |

The weak-starter reference remains HR@10 `0.125`, MRR `0.068034`, MTTC `9.81`, and TechnicalScore `0.10671`. Keep it as the original control, not as the description of the current Agent.

## 3. Debug from the outside inward

Do not start by reading random ranking code. Locate the first current layer where the expected target stops surviving. The Workbench currently exposes the following evidence:

| Current layer | Debug question | Current evidence |
| --- | --- | --- |
| Startup/contract | Did Agent construct and return valid schema/IDs? | exception, response validation, invalid/duplicate count |
| Session state | Was the latest message accumulated, negated, or treated as an override? Was the previous question answered or interrupted? | active/excluded terms, known/asked/exhausted/pending attributes, parsed events, version, turn ledger |
| Query plan | Did the active state compile into the intended sparse query? | compiled query and parser/override events |
| Retrieval | Was the public target present in broad OR or strict AND retrieval? | post-hoc target broad/strict ranks and route counts |
| Fusion control | Did weighted RRF push the target out of Top 10? | fused rank and fusion evidence |
| Coverage cascade | Did visible-term coverage safely move it relative to fused control? | fused/final ranks, per-result matched-term counts, mode and schema provenance |
| Attribute rerank | Did comparable normalized evidence move it safely in an experiment? | fused→reranked→final ranks, component score, matched evidence, mode |
| Output | Did the actual final route or normalization lose it? | final rank and normalized Top 10 |
| Clarification | Which fixed policy asked what, and was it repeated? | policy, `ask_attribute`, asked/exhausted attributes |
| Runtime | Did the Agent fail or slow down? | per-turn elapsed time and exception event |

The currently emitted trace codes are:

```text
AGENT_ERROR
HIT
PRE_OVERRIDE_NOT_SCORABLE
RETRIEVAL_MISS
LOW_FINAL_RANK
OUTPUT_OR_NORMALIZATION_MISS
```

Hard/soft slot objects and candidate-aware information gain now exist as target-blind
shadow diagnostics. Structured filter/relaxation counts and codes such as
`FILTER_KILLED_TARGET` or `WRONG_QUESTION` still belong to the target architecture.
Normalized product evidence and reranker ranks are real trace fields, but active v1/v2
remain rejected. The served output is now the R08 coverage-ordered `final` route; the
weighted-RRF `fused` route remains the explicit control and diagnostic reference.

## 4. The development loop

Every improvement should change one causal layer at a time.

1. State one hypothesis, such as “remembering the category will improve Browsing recall after a short clarification reply.”
2. Add a focused unit test that fails before the change.
3. Implement the smallest change that passes it.
4. Run all unit tests.
5. Select or reject the hypothesis on the frozen public-target-disjoint corpus.
6. Only the single surviving configuration may run the full public evaluator and phrase suites.
7. Run repeated determinism, latency, RSS, no-key, and leakage gates before activation.
8. Compare overall metrics, four scenarios, robustness, runtime, and the relevant intermediate diagnostic.
9. Keep the change only when the evidence matches the hypothesis and no unacceptable regression appears.

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

Run an explicit, provenance-recorded rerank experiment without changing the released
evaluator flow:

```powershell
python scripts/evaluate_agent.py --retrieval-mode coverage --rerank-mode off --output experiments/p4_coverage.json
python scripts/evaluate_agent.py --retrieval-mode control --rerank-mode off --output experiments/p4_control.json
python scripts/evaluate_agent.py --retrieval-mode control --rerank-mode shadow --output experiments/p2_shadow.json
python scripts/evaluate_agent.py --retrieval-mode control --rerank-mode active --output experiments/p2_active.json
```

`coverage + off` is the production/default path. `control + off` preserves the pre-P4
weighted-RRF output for paired comparisons. `control + shadow` computes P2/P3
diagnostics without serving them; `control + active` executes the rejected v2
Top-10-member-safe experiment. Coverage is deliberately restricted to rerank off so an
ungated combination cannot silently change the promoted architecture.

The evaluator overwrites ignored `results.json`. Copy important experiment outputs into another ignored experiment directory before the next run. Compare complete results, including all session rows, with:

```powershell
python scripts/compare_results.py --assert-equal expected.json actual.json
```

Strict mode exits with code 1 on any semantic difference while ignoring JSON formatting and line-ending differences.

Run the fixed P1 robustness gate:

```powershell
python scripts/evaluate_generalization.py --corpus both --suite default
```

Use `--suite all --corpus public` for the frozen dev/challenge/audit phrase families. The runner wraps the Agent and transforms only the visible user message; it never receives target, scenario, intent card, sample ID, or prior result. Its derived corpus deterministically excludes all 200 released-public target IDs and records the seed, sample hash, and overlap count. This derived data is a local stress test, not a substitute for the private 800 sessions.

## 6. Agent Workbench

For normal development on this machine, double-click `Start Observer.vbs`. It launches
through the existing `tiktok` environment without a terminal and opens
`http://127.0.0.1:8765`. The P4-aligned launcher uses the served `coverage + off` path;
the UI separates the weighted-RRF fused control from the coverage-ordered final route and
records retrieval-mode provenance. Use the in-page **停止** action when finished.
`Start Observer.cmd` and `python -m observer.launcher` remain fallback launch paths.

The Workbench now covers the complete local development loop:

```text
environment / data / Git / index health
-> current algorithm registry
-> public single-session replay
-> actual target-blind Agent events
-> post-hoc target survival and score diagnosis
-> full evaluator, generalization gate, or unit-test background job
-> progress, logs, metrics, and versioned experiment artifact
```

It also includes catalog/FTS5 search, full product JSON, a target-free manual Agent lab, experiment comparison, and an allowlisted document library. The control plane runs only fixed test/evaluation actions and does not execute arbitrary shell input. A per-instance token and same-origin/Host checks protect the local API.

The Workbench fingerprints Agent, coverage, attributes, reranker, slot-ledger,
clarification, shadow-analysis, evaluator, and generalization code plus catalog/public-
set inputs loaded at startup and compares them with disk. After any monitored file
changes, evaluation, every replay, and Lab execution are blocked until restart.
Background evaluation checks freshness both before execution and before artifact
finalization, while its manifest uses the captured start-of-run provenance. This
prevents an experiment from silently mixing an old imported class or cached data with
later disk hashes.

Trace events are emitted by the actual Agent through an optional versioned callback. The
state layer reports the Agent's version, category, active/excluded terms,
known/asked/exhausted attributes, override count, and full shadow slot lifecycle.
Retrieval reports broad, strict, fused, reranked, and final routes; coverage mode also
reports its schema, matched visible terms, per-result coverage, and whether Top 10 changed.
Policy shows actual and QuestionValue shadow decisions. Public target ranks and candidate
breakdowns are joined only after `Agent.respond` and are labelled post-hoc.

Every public replay gives the Agent an opaque random session ID. The Agent receives only profile, generated user message, turn, and `top_k`; it never receives `sample_id`, target, intent card, scenario, behavior, or prior results. The server is loopback-only and must not be attached to private final labels.

Successful browser-started evaluations refresh `results.json`; evaluator and generalization jobs write ignored, versioned artifacts under `experiments/`. See `docs/agent_workbench.md` for pages, endpoints, safety boundaries, and the trace maintenance contract.

## 7. Current improvement order

1. Keep the promoted R08 served path reproducible; retain explicit weighted RRF as the
   paired control and do not add sample/ASIN exceptions.
2. Treat the normalized attributes, SlotLedger, QuestionValue, and Workbench analysis as
   experiment infrastructure, not production claims.
3. The first architecture wave is complete: 14 raw candidates were run, 13 were
   semantically valid/effective after the R12 false-activation audit, and R08 alone passed
   selection and promotion gates.
4. P5 PRF, P6 depth, P7 BGE, and P8 explicit-negative partition are frozen rejected
   experiments. P8 showed a quality gain on its local stress selection but failed wall,
   P95, and RSS gates; it must not be tuned or rerun on that corpus.
5. Use only unlabeled microbenchmarks to diagnose P8 allocation/resource overhead. Any
   optimized executor requires a new frozen target-disjoint P9 protocol and still counts
   as execution engineering, not another ranking architecture.
6. Advance only a survivor that passes fresh selection, repeat, confirmation, strict
   quality/resource gates, then one released-public/phrase confirmation. Semantic
   dependencies additionally require recall, licensing, offline, and packaging evidence.

The repository has completed no-credential reliability, Workbench diagnostics,
versioned term state, pending-question lifecycle, broader target-blind phrase parsing,
Override/Boundary handling, broad/strict sparse routes, weighted RRF, promoted visible-
term coverage ordering, heuristic clarification, strict result comparison, P1/P3/P4
robustness and resource gates, normalized product attributes, a shadow SlotLedger,
candidate-aware QuestionValue diagnostics, and cross-session shadow analysis. It has not
made the slot ledger the retrieval source of truth, activated candidate-aware questioning,
implemented served hard filtering/relaxation or numeric budget execution, or added
dense/profile/semantic ranking. P8 explicit-negative partition and deterministic active
rerank v1/v2 remain rejected experiment-only implementations.
