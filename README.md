# TechJam Conversational E-Commerce Search Challenge
By team err402

## Get Started

Select an agent explicitly with `--agent`:

```bash
python3 -m evaluator.local_evaluator --agent baseline
python3 -m evaluator.local_evaluator --agent v1 --output results/v1_initial.json
```

`baseline` is the original stateless weighted-BM25 implementation. `v1` adds
stateful intent routing, category and FTS candidate generation, documented
intent-weighted reranking, popularity and Bayesian-rating priors, and
deterministic information-gain clarification questions. No LLM involved in `v1`. 

The evaluator prints the selected agent description and displays session
progress with cumulative prompt, completion, and total token counts. Use
`--quiet` to suppress the description and progress bar; stdout will contain
only the aggregate JSON summary. The full per-session result is still written
to the path specified by `--output`.

Add versioned agents under `agents/` and register them in `agents/registry.py`.

Result:
| Agent | HitRate@10 | MRR | MTTC | Score | Tokens |
|:------|:-----------|:----|:-----|:------|:-------|
| Baseline (BM25) | 0.125 | 0.068034 | 9.81  | 0.10671  | 0 |
| v1              | 0.98  | 0.67896  | 2.465 | 0.864388 | 0 |


## Generate an Expanded Development Set

With `local-data/valid_records.csv` available, generate 1,000 additional
samples using purchase-frequency-weighted unique product sampling:

```bash
python3 generate_evaluation_set.py
```

The generator excludes public target products, uses the official scenario
quotas, and writes `local-data/generated_set_1000.jsonl`. Its profile tags are
sampled from the public profile distribution because metadata for most history
ASINs is not present in the frozen catalog. Rating summaries use the selected
purchase record's rating, matching the relationship observed in the public
set; they are not verified averages of historical ratings. Change `--seed`,
`--samples`, or the input/output paths through the documented CLI options.


## LLM Client Configuration

Install the OpenAI-compatible client dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `LLM_API_KEY`, `LLM_MODEL`, and, for a
third-party OpenAI-compatible service, `LLM_BASE_URL`. The client sends
non-streaming chat-completion requests in JSON-object mode and records the
provider-reported prompt and completion token counts.

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
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
