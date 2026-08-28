# Track 4 Research Source Registry

Last checked: 2026-08-28 SGT.

This is the canonical source registry for the P4 architecture search. It separates
organizer requirements, primary research, and repository-local measurements. A paper
can motivate an experiment; only this repository's frozen evaluator artifacts can
establish a result for this project.

## Organizer and data sources

| Claim supported | Primary source | Use in this repository |
| --- | --- | --- |
| Submission window, significant-update rule, submission obligations, and four equally weighted Stage Two criteria | [TikTok TechJam 2026 Official Rules](https://tiktoktechjam2026.devpost.com/rules) | Rules-first compliance boundary |
| Track task, 50,000 released products, 200 public sessions, 800 organizer-private sessions, exact-ID scoring, metric formula, and allowed model choices | [Official Track 4 repository](https://github.com/TechJam2026/techjam-conversational-search) | Executable task definition and participant-facing data inventory |
| Official downloadable catalog package and checksums | [Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit) | Catalog provenance and integrity |
| Exact request/response schema | [Agent API contract](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/agent_api_contract.json) | Strict response validation |
| Deterministic public simulation and scoring behavior | [Official local evaluator](https://github.com/TechJam2026/techjam-conversational-search/blob/main/evaluator/local_evaluator.py) | Frozen local measurement only |
| Source corpus provenance | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html) | Attribution; not an authorization to reconstruct private labels |

## Primary research sources

| Design question | Primary source | Bounded inference for Track 4 |
| --- | --- | --- |
| How should independent lexical rankings be combined without labels? | Cormack, Clarke, and Büttcher, [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Keep RRF as a strong control; compare genuinely different score/rank aggregation mechanisms rather than weight-only copies |
| How can a browsing list balance relevance and novelty? | Goldstein and Carbonell, [Using MMR for Diversity-Based Reranking](https://aclanthology.org/X98-1025/) | Test target-blind attribute-signature MMR only for visibly open browsing |
| How can ambiguous results cover multiple explicit aspects? | Santos et al., [Explicit Search Result Diversification through Sub-Queries](https://doi.org/10.1007/978-3-642-12275-0_11) | Treat category, field, and slot routes as observable aspects; do not use hidden scenario labels |
| Why model conversation history selectively? | Yu et al., [Few-Shot Conversational Dense Retrieval](https://arxiv.org/abs/2105.04166) | Scope history/carry-over to the current goal version so overrides cannot retain unrelated context |
| What is a practical zero-shot dense-retrieval candidate? | Wang et al., [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) | E5 is a future optional local dense route, subject to license, asset, memory, latency, and offline gates |
| What are the robustness trade-offs across lexical, dense, sparse, late-interaction, and reranking systems? | Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663) | Do not assume neural retrieval is automatically more robust; keep BM25/RRF as the measured baseline and gate resources |
| Can learned sparse expansion bridge lexical mismatch? | Formal et al., [SPLADE v2](https://arxiv.org/abs/2109.10086) | Learned sparse retrieval is a future independent route, not a synonym-table parameter tweak |
| Can document expansion reduce query-document vocabulary mismatch? | Nogueira et al., [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375) | Consider offline catalog expansion only after verifying redistribution, size, and no-target-training boundaries |
| Can fine-grained semantic interaction be made cheaper than full cross-encoding? | Khattab and Zaharia, [ColBERT](https://arxiv.org/abs/2004.12832) | Late interaction is a later-stage option if a dense shortlist and resource budget justify it |
| Can a cross-encoder improve shortlist order? | Nogueira and Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) | Apply any cross-encoder only after bounded target-blind retrieval; never expose the full 50,000 catalog per turn |
| Why explicitly evaluate clarification selection? | Aliannejadi et al., [ClariQ](https://arxiv.org/abs/2009.11352) | Clarification is a ranking/policy problem, but Track 4 promotion must also account for MTTC and final-turn limits |

## Evidence boundary

- Official facts above are checked against organizer-controlled sources.
- Research papers motivate hypotheses in `docs/algorithm_architecture_research.md`;
  their reported gains are not transferred to this catalog or evaluator.
- Local data integrity is established by `scripts/verify_official_assets.py` and
  `docs/data_inventory.md`.
- Architecture selection uses the frozen public-target-disjoint local stress corpus.
  It is not organizer private data and is not a hidden-score estimate.
- The released public 200 is reserved for a final gate after selection. The private 800
  remains inaccessible, so no local result can prove private-set superiority.
