# 2026 Tiktok TechJam Conversational E-Commerce Search Challenge

By team err402.

This project implements conversational product search over the given frozen Amazon 2023 50K catalog. The deterministic retrieval pipeline combines category, lexical, constraint, rating, popularity, and optional dense signals, and outputs a recommendation score. V3 adds online LLM-based conversational parsing and question wording. 

Note: due to the design of the benchmark, it is pointless to apply LLM to do the parsing because all the conversations have a fixed format and can be parsed by regex perfectly. However, for users in the real world, the input will be more complicated and an LLM will help improve the performance a lot. As a result, we present two versions. V1 is tuned for the benchmark, while V3 is designed for interacting with humans. V2 and V2 embedding are obsolete versions. They will be introduced in detail later.

## Get Started

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
```

For V3, add `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` to the copied `.env` for an OpenAI-compatible service. For V1 and V2, no need to edit the `.env` as LLM is not called. 

### Interactive mode

Run the final V3 agent as an in-memory shopping REPL:

```bash
python3 -m interactive
```

Enter shopping requests directly at the `You>` prompt. The REPL presents recommendations and asks up to two native facets per turn. Use `/quit` to leave. At the end of the conversation it reports provider-reported token cost. As our prompt is simple, usually the cost is about 1K token per round of conversation.

### Benchmark mode

Run any registered agent on the public evaluator. `v3` uses the live configured LLM and therefore incurs API cost; the other documented agents are local.

```bash
python3 -m evaluator.local_evaluator --agent v1 --output results/v1.json
```

To select a different catalog or evaluator input set, pass `--catalog` and
`--dataset`. `--quiet` prints only the aggregate JSON to stdout.

```bash
python3 -m evaluator.local_evaluator \
  --agent v1 \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results/v1_public.json \
  --quiet
```

The primary agent choices are:
- `baseline`: stateless BM25 provided by default.
- `v1`: deterministic category-aware reranking and clarification. Recommended for benchmarking.
- `v2`: V1 retrieval with offline-extracted fine-grained catalog attributes. The benchmark score is slightly lower than `v1` because the evaluator cannot take full use of the attribute generated.
- `v2-embedding`: V2 plus local Qwen3 dense retrieval. This is not recommended as it costs higher but brings little performance boost.
- `v3`: V2 retrieval with LLM intent probability, state parsing, profile updates, and context-aware question wording. Very slow for benchmark. 

V2 and V3 require `results/catalog_attributes_processed.jsonl`. V2-embedding also requires its configured local embedding artifact. As V2-embedding is obsolete, the generated embedding file is not provided. 
The `catalog_attributes_process.jsonl` can be found in the release page. 

## Current Benchmark Results

Current deterministic results. V3 is not included because it takes too long to benchmark. 

| Dataset | Agent | HitRate@10 | MRR | MTTC | Score | Tokens |
|:--|:--|--:|--:|--:|--:|--:|
| Public 200 | Baseline (BM25) | 0.125 | 0.068034 | 9.810 | 0.106710 | 0 |
| Public 200 | V1 | 0.995 | 0.703766 | 2.110 | 0.886430 | 0 |
| Public 200 | V2 | 0.990 | 0.649599 | 2.205 | 0.865780 | 0 |
| Public 200 | V2-embedding | 0.990 | 0.648002 | 2.205 | 0.865301 | 0 |
| Generated 1,000 | V1 | 0.994 | 0.643010 | 2.232 | 0.865263 | 0 |
| Generated 1,000 | V2 | 0.985 | 0.613687 | 2.418 | 0.848246 | 0 |
| Generated 1,000 | V2-embedding | 0.985 | 0.618703 | 2.428 | 0.849551 | 0 |

## Evaluation

Each turn may return a message, one evaluator-facing `ask_attribute`, up to ten
ranked `parent_asin` recommendations, and prompt/completion token counts. See
[the agent contract](docs/agent_api_contract.json) for the complete interface.

The evaluator reports:

- **Hit Rate@10:** fraction of sessions that find the target within ten turns.
- **MRR:** reciprocal rank of the target; a miss contributes zero.
- **MTTC:** first-hit turn; a miss is assigned turn 11.
- **Token usage:** provider-reported prompt and completion tokens.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

The public evaluator and scoring configuration are in
[`evaluator/local_evaluator.py`](evaluator/local_evaluator.py) and
[`docs/evaluation_config.json`](docs/evaluation_config.json). Submission and
release requirements are in [`docs/submission_rules.md`](docs/submission_rules.md).

## Data Attribution

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab,
UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) before using or
redistributing the data. Sessions are sampled deterministically from the
official Clothing 5-core leave-last-out split and joined to the frozen catalog.
