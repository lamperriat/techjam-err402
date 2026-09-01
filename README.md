# 2026 TikTok TechJam Conversational E-Commerce Search Challenge
Project ESearch. 

By team err402.

This project implements conversational search over the provided frozen 50K Amazon Reviews 2023 catalog. The retrieval pipeline combines category, lexical, constraint, rating, and popularity signals. V1 is the deterministic offline submission agent; AgentV212 adds a frozen learned reranking and pagination stack; V3 adds LLM-based parsing and question wording for natural interactive conversations.

## Get Started

The project was tested with Python 3.13.3. Other versions may also work.

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
```

V1 and AgentV212 require neither credentials nor network access. To use V3, set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `.env` for an OpenAI-compatible LLM service.

V1 and AgentV212 only require the provided `data/catalog.jsonl`; AgentV212's frozen P11 sidecar and small-ranker artifact are bundled with the code. V2 and V3 additionally require `results/catalog_attributes_processed.jsonl`, distributed through the [project releases](https://github.com/lamperriat/techjam-err402/releases/tag/data%2Fattributes_processed).

### Interactive mode

Run V3 as an in-memory shopping REPL:

```bash
python3 -m interactive
```

Enter requests at the `You>` prompt and use `/quit` to exit. The REPL prints recommended products and provider-reported input and output token counts.

### Official benchmark

The organizer's unchanged evaluator imports the exported `Agent`, which is V1:

```bash
python3 -m evaluator.local_evaluator --output results/v1.json
```

### Agent selection and custom inputs

The development evaluator adds agent selection, a progress bar, custom input paths, and `--quiet` JSON-only output:

```bash
mkdir results
python3 -m evaluator.agent_evaluator --agent v1 --output results/v1.json
python3 -m evaluator.agent_evaluator --agent v212 --output results/v212.json
python3 -m evaluator.agent_evaluator \
  --agent v1 \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results/v1_public.json \
  --quiet
```

Available development agents are:

- `baseline`: the original stateless BM25 reference.
- `v1`: deterministic category-aware reranking and clarification; the official offline default.
- `v2`: V1 with offline-extracted fine-grained catalog attributes.
- `v212`: frozen offline coverage retrieval, P11 reranking, fold-safe small-ranker, and two-page unseen pagination.
- `v2-embedding`(obsolete): V2 plus local Qwen3 dense embedding retrieval; retained for comparison.
- `v3`: V2 retrieval with LLM intent estimation, state updates, and context-aware question wording.

## Submission Package

`submission/` is the self-contained source package. It deliberately excludes the evaluator, benchmark datasets, API credentials, and the 39 MB processed attribute artifact. The catalog is supplied by the organizer, and the processed attribute artifact for V2/V3 is supplied through the project release.

```text
submission/
  agent.py                 # exports Agent, AgentV1, AgentV2, AgentV212, and AgentV3
  interactive.py
  requirements.txt
  .env.example
  README.md
  DATA_ATTRIBUTION.md
  starter/agent.py         # compatibility with the original evaluator import
  src/err402/              # runtime agents, bundled V212 assets, retrieval, and LLM modules
  data/README.md
  results/README.md
```

`Agent` remains a direct alias of `AgentV1`; AgentV212 is an explicitly selectable alternative. V1 is therefore still used when the organizer replaces the local evaluator with the original version. The submission package does not contain or modify any evaluator code.

## Current Benchmark Results

V3 is omitted because live-LLM latency makes it unsuitable for the benchmark. In addition, the ranking is still determined by the same model in V1, not by the LLM. 

| Dataset | Agent | HitRate@10 | MRR | MTTC | Score | Tokens |
|:--|:--|--:|--:|--:|--:|--:|
| Public 200 | Baseline (BM25) | 0.125 | 0.068034 | 9.810 | 0.106710 | 0 |
| Public 200 | V1 | 0.995 | 0.703766 | 2.110 | 0.886430 | 0 |
| Public 200 | V2 | 0.990 | 0.649599 | 2.205 | 0.865780 | 0 |
| Public 200 | V2-embedding | 0.990 | 0.648002 | 2.205 | 0.865301 | 0 |
| Generated 1,000 | V1 | 0.994 | 0.643010 | 2.232 | 0.865263 | 0 |
| Generated 1,000 | V2 | 0.985 | 0.613687 | 2.418 | 0.848246 | 0 |
| Generated 1,000 | V2-embedding | 0.985 | 0.618703 | 2.428 | 0.849551 | 0 |
| Generated 1,000 | AgentV212 | 0.988 | 0.703367 | 2.917 | 0.866670 | 0 |

AgentV212 narrowly improves the Generated 1,000 TechnicalScore through higher MRR, while V1 retains higher hit rate, lower MTTC, and substantially lower runtime. In a four-process exact-repeat run on our local M2 MacBook Air, the V1 and V2 completed in 44 seconds and AgentV212 completed in 211 seconds.

## Method and Operational Notes

- V1 parses the evaluator's fixed conversation templates, maintains session constraints, retrieves lexical and category candidates, and reranks them using intent-specific weights, Bayesian-adjusted ratings, and popularity. Clarification facets (the next attribute to ask the user about) are selected deterministically from candidate coverage, information gain, catalog-specific priors, and question history.
- V2 uses preprocessed LLM-extracted attributes at runtime but makes no online model calls.
- AgentV212 uses coverage retrieval, a frozen P11 Top-10 scorer, a fold-safe small-ranker, and two-page unseen pagination. Its runtime is deterministic, offline, and fixed to the tested configuration.
- V3 uses an OpenAI-compatible LLM for conversational parsing and question construction while retaining deterministic product retrieval.
- V1, V2, and AgentV212 have zero runtime model cost. V3's latency and cost depend on the configured provider; the REPL reports its measured tokens, typically around 1K tokens per conversation turn.

Limitations:

- V1 is optimized for the evaluator's structured phrasing and is less robust to unrestricted human language.
- AgentV212 bundles a roughly 31 MB feature sidecar and uses considerably more CPU time than V1.
- Most catalog prices are missing, limiting budget-aware ranking. In practice budget is very important in the decision-making. 
- V3 should actually use a smaller model. The REPL and the chatting logic is not polished.
- If time permits, we want to experiment more with the LLM attribute extraction. There are many papers published related to this technique. However, due to the limitation of the current benchmark, we cannot evaluate its effect properly.

## Contributions
Qizhen Sun: development of agent V1, V2, V2-embedding, and V3 interactive; documentations.
Zou Seak Pang: development agent V212 and visualization.
Yiwen Xu: testing of agent workflow.

## Evaluation and Data

`evaluator/local_evaluator.py` is exactly the same as the one provided by the organizer. `evaluator/agent_evaluator.py` is a separate development harness and is not part of the submission package.

The catalog and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) before using or redistributing the data.
