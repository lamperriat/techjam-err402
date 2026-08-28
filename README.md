# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Agent

Python 3.10 or later is recommended. The current stateful sparse Agent uses only the Python standard library and does not require LLM credentials.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to improve the system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The official weak BM25 starter reference scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

The current integrated Agent implements versioned multi-turn term state, a target-blind parsed-turn layer, pending-question lifecycle, explicit Override and Boundary handling, broad/strict FTS5 retrieval, weighted RRF, and heuristic clarification. P2/P3 also include normalized product attributes, an auditable slot ledger, a deterministic constraint scorer, and candidate-aware QuestionValue diagnostics behind explicit `off / shadow / active` rerank modes. These additions remain diagnostic: active rerank v1/v2 failed their quality gates and shadow failed the repeated resource time gate. `off` remains the default. Its verified served result is:

| Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 0.940000 | 0.605258 | 3.375000 | 0.762500 | 0.804077 |

These are public-development metrics and do not predict the private 800-session result. P1 also supplies fixed phrase-perturbation suites and a deterministic 200-product-derived, public-target-disjoint stress corpus. All released-public dev/challenge/audit phrase suites currently retain HR@10 `0.94`; the derived corpus is a local stress tool, not organizer data or a hidden-score estimate. The deterministic parser and default `fast` policy remain documented overfitting risks.

The final P3 shadow mode is strictly output-equal to off on both the public and frozen product-disjoint corpora and exposes five auditable routes:
`broad`, `strict`, `fused`, `reranked`, and `final`. Experimental active v1 scored HR@10
`0.93`, MRR `0.599974`, MTTC `3.43`, and TechnicalScore `0.796392`, so it failed the
activation gate and is deliberately not the default.

Run the fixed robustness gate from the browser Workbench or directly:

```bash
python3 scripts/evaluate_generalization.py --corpus both --suite default
```

Run a mode-controlled evaluator experiment with a separate provenance manifest:

```bash
python3 scripts/evaluate_agent.py --rerank-mode shadow --output experiments/p2_shadow.json
```

## LLM Client Configuration

The OpenAI-compatible client is optional and is not constructed by the default offline Agent. Core evaluation uses only the Python standard library. Install the optional client dependencies only when developing or testing model-assisted features:

```bash
python3 -m pip install -r requirements-llm.txt
```

When explicitly constructing `utils.llm_client.LLMClient`, copy `.env.example` to `.env` and set `LLM_API_KEY`, `LLM_MODEL`, and, for a third-party OpenAI-compatible service, `LLM_BASE_URL`. The client sends
non-streaming chat-completion requests in JSON-object mode and records the
provider-reported prompt and completion token counts.

See `docs/development_workflow.md` for the project mental model, debugging funnel, experiment loop, and current improvement order.

## Agent Workbench

On Windows, double-click `Start Observer.vbs` to start the local Workbench without opening a terminal. It uses the existing `tiktok` Conda environment, starts through `pythonw.exe`, and opens `http://127.0.0.1:8765`. The one-click development launcher uses output-safe `shadow` mode so rerank evidence is visible while recommendations retain the fused order. The normal Agent/evaluator default remains `off`. `Start Observer.cmd` and `python -m observer.launcher` are troubleshooting fallbacks.

The Workbench provides:

- runtime, Git, data, catalog-hash, and FTS5-index health;
- an honest algorithm registry that distinguishes implemented, baseline-only, and planned layers;
- all 200 public sessions with actual Agent events and separately labelled post-hoc target diagnostics;
- frozen-catalog search and raw product inspection;
- browser controls for the complete public evaluator, fixed phrase/product-disjoint generalization gate, unit tests, progress, cancellation, logs, versioned local experiments, and cross-session target-blind shadow-policy analysis;
- a target-free manual Agent playground and read-only project document library;
- a safe in-page shutdown action.

The server refuses non-loopback bind addresses, rejects cross-site API requests, requires an ephemeral local control token, and does not expose an arbitrary shell runner. It fingerprints the loaded Agent/attributes/reranker/slot-ledger/clarification/shadow-analysis/evaluator sources plus catalog/public-set inputs and blocks stale or mixed-version runs until the Workbench is restarted. Every public replay gives the Agent a fresh opaque session ID. The released simulator uses hidden target/scenario state only to generate the permitted user messages; raw labels, intent cards, behavior, and prior results are never passed into Agent decision features. Target-rank and scoring annotations are joined after `Agent.respond`.

The Workbench displays the current versioned state, full slot-ledger lifecycle, all five ranking routes, normalized attribute/rerank evidence, weighted fusion, actual heuristic policy, candidate-aware shadow components, and post-hoc target ranks. It continues to label slot-ledger-driven retrieval, hard filtering/relaxation, numeric budget execution, dense retrieval, active candidate-aware clarification, profile ranking, and semantic reranking as missing rather than presenting roadmap layers as implemented. See `docs/agent_workbench.md` for the full usage, API, isolation, and maintenance contract.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable stateful sparse Agent
starter/attributes.py             target-blind normalized product/constraint views
starter/reranker.py               deterministic gated Top-50 scorer
starter/slot_ledger.py             auditable normalized conversation shadow
starter/clarification.py           candidate-aware QuestionValue shadow
evaluator/local_evaluator.py      public-set simulator and scorer
scripts/compare_results.py        report and strict complete-result comparison
scripts/evaluate_agent.py         mode-controlled evaluator + provenance manifest
scripts/evaluate_generalization.py target-blind phrase/product-disjoint stress gate
scripts/benchmark_resources.py    repeatability, RSS, latency, and route-recall audit
observer/shadow_analysis.py        target-blind cross-session question diagnostics
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only judging runbooks and private-release procedures are not distributed in this participant repository.

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
