# Track 4 Research and Compliance Source Registry

Last checked: 2026-08-28 SGT.

Audience: Track 4 implementation team, reviewers, and submission auditors.

Scope: organizer-controlled requirements and data, P4-P8 architecture decisions, and
submission-readiness claims. This is the repository's canonical research report and
claim/source ledger. It separates organizer requirements, primary research, and
repository-local measurements. A paper can motivate an experiment; only this
repository's frozen evaluator artifacts can establish a result for this project.

## Direct answer and current boundary

The participant-facing dataset is complete. At the verified official snapshot, the
organizer releases exactly one frozen 50,000-product catalog and 200 labeled public
development sessions. The additional 800 final sessions are organizer-private, so
their absence locally is intentional rather than a failed download. No additional
official train split, embeddings, images, review text, purchase history, private labels,
or hidden intent cards were found in the repository or participant release.

On 2026-08-28 SGT, the official remote still resolved to `main` commit
`34078351e1c3615e5505a2e829600b56a542e462` and participant tag
`2a6cc8e776da66ce69b1cbd237838fbc43f32587`. The release still exposed the three authored
assets `catalog.jsonl.gz`, `SHA256SUMS`, and `techjam-participant-kit.zip`. The local
offline verifier passed 14/14 checks. These facts establish participant-kit completeness;
they do not expose or validate the private 800 sessions or the final judging hardware.

The current evidence also does **not** justify saying that every project plan or
submission obligation is complete. P8 remains an experiment in progress, and the
competition-window significant update, public/default-branch release, short report,
team-contribution disclosure, and three-minute demonstration remain separate gates.

## Organizer and data sources

| Claim supported | Primary source | Use in this repository |
| --- | --- | --- |
| Submission window, significant-update rule, submission obligations, and four equally weighted Stage Two criteria | [TikTok TechJam 2026 Official Rules](https://tiktoktechjam2026.devpost.com/rules) | Rules-first compliance boundary |
| Track task, 50,000 released products, 200 public sessions, 800 organizer-private sessions, exact-ID scoring, metric formula, and allowed model choices | [Official Track 4 repository](https://github.com/TechJam2026/techjam-conversational-search) | Executable task definition and participant-facing data inventory |
| Official downloadable catalog package and checksums | [Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit) | Catalog provenance and integrity |
| Exact request/response schema | [Agent API contract](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/agent_api_contract.json) | Strict response validation |
| Deterministic public simulation and scoring behavior | [Official local evaluator](https://github.com/TechJam2026/techjam-conversational-search/blob/main/evaluator/local_evaluator.py) | Frozen local measurement only |
| Source corpus provenance | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html) | Attribution; not an authorization to reconstruct private labels |
| Participant-facing data file boundary | [Official Competition Data README](https://github.com/TechJam2026/techjam-conversational-search/blob/main/data/README.md) | Confirms the released 200 sessions and separately downloaded 50,000-product catalog |
| A large public product-search relevance dataset exists but is not part of this challenge kit | Reddy et al., [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588) | External research evidence only; ESCI labels must not be represented as organizer data or private-evaluation evidence |

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
| Why isolate explicit negative constraints instead of treating negation as ordinary terms? | Weller, Lawrie, and Van Durme, [NevIR: Negation in Neural Information Retrieval](https://aclanthology.org/2024.eacl-long.139/) | Negation remains difficult across retrieval families; P8 should test a transparent compatibility partition before paying for a neural reranker |
| Can fine-grained negative feedback improve product search? | Bi et al., [Conversational Product Search Based on Negative Feedback](https://arxiv.org/abs/1909.02071) | Test only explicit visible `not/without X` evidence; do not infer undisclosed dislikes from a miss or simulator state |
| How should override and no-preference updates be represented? | Kim et al., [SOM-DST: Selectively Overwriting Memory for Dialogue State Tracking](https://aclanthology.org/2020.acl-main.53/) | Motivate auditable carry-over/delete/don't-care/update operations; current deterministic ledger remains the control rather than importing a neural DST claim |
| Why selectively retain conversational history? | Voskarides et al., [Query Resolution for Conversational Search with Limited Supervision](https://arxiv.org/abs/2005.11723) | Motivate target-blind copy/drop decisions scoped to the current goal version, not a repeated per-turn RRF variant |
| Can retrieval-risk estimation decide whether to ask or recommend? | Meng et al., [Query Performance Prediction for Conversational Search](https://arxiv.org/abs/2305.10923) | Test unsupervised score/coverage signals first because this project lacks a large independent policy-training set |
| How can clarification be treated as expected-value planning? | Aliannejadi et al., [Asking Clarifying Questions in Open-Domain Information-Seeking Conversations](https://arxiv.org/abs/1907.06554) | A question must have candidate-set value greater than its MTTC cost; paper gains are not transferred to this evaluator |
| Can a catalog graph constrain conversational question paths? | Lei et al., [Interactive Path Reasoning on Graph for Conversational Recommendation](https://www.kdd.org/kdd2020/accepted-papers/view/interactive-path-reasoning-on-graph-for-conversational-recommendation.html) | Motivate deterministic attribute-graph pruning only; the paper's RL setting and interaction volume are not available here |
| Can character-level evidence address morphology and spelling mismatch in product search? | Lakshman et al., [Deep Semantic Product Search](https://kdd.org/kdd2019/accepted-papers/view/deep-semantic-product-search) | A bounded char n-gram sidecar is a distinct future lexical/OOV arm; behavior-label and GPU results are out of scope |
| Can a small feature model learn shortlist order? | Burges, [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) | A future tree ranker requires product-disjoint training and a leakage audit; 200 public labels are too small for unguarded fitting |

## Claim-gap matrix

| Question | Evidence status | Consequence |
| --- | --- | --- |
| Are the participant data complete and current? | Confirmed by remote refs/release inventory plus local 14/14 integrity checks | Continue from the current 50,000 + 200 snapshot; do not search for a nonexistent downloadable private split |
| Are the private 800 rows or final hardware limits knowable? | Not published | Treat them as known unknowns and preserve deterministic offline/resource fallbacks |
| May the team change the Agent architecture? | Yes; the contract and evaluator are fixed, while retrieval, state, routing, clarification, reranking, and legal model choices are participant work | Keep official artifacts byte-stable while improving `starter/` and experiment-only modules |
| Is R08 proven best on the private set? | No; it is the best eligible design under one frozen local selection protocol and a bounded public confirmation | Keep claims local and scenario-specific |
| Should P5 PRF, P6 depth expansion, or the P7 BGE formulation be rerun on their frozen corpora? | No; each already failed a preregistered value/resource gate | Use a fresh target-disjoint protocol and a mechanism-level change |
| Is a generic dense/model swap the next priority? | No; local sparse union already recovered 198/200 P7 targets and the tested dense route rescued zero while failing wall/RSS gates | Prioritize shortlist discrimination, negative constraints, state, and question policy |
| Are submission obligations complete? | No | Preserve a separate post-start implementation commit, before/after evidence, public repository/default-branch verification, report, licensing, and demo checklist |

## P8 research recommendation

The first isolated P8 arm should implement a stable
`compatible -> unknown -> explicit_violation` partition for active, high-confidence hard
negative slots. Missing product metadata stays `unknown`, ties retain served R08 order,
and a short candidate list falls back deterministically. This is materially different
from P4 R09, which mixed negative protection with positive-constraint relaxation.

Subsequent fresh-corpus arms should remain mechanism-distinct: slot-dominance partial
ordering, catalog graph disambiguation, explicit state operations, selective history,
unsupervised retrieval-risk routing, finite-horizon question planning, attribute-graph
question paths, char n-gram OOV retrieval, document-side expansion, product-disjoint
learning-to-rank, a frozen shortlist cross-encoder, browsing-only xQuAD, and contextual
term-impact sidecars. Weight, cutoff, synonym-list, penalty, or checkpoint-only changes
do not count as new architectures.

The P5 PRF graph, P6 120-to-240 depth graph, and P7 BGE single-vector graph must not be
rerun or renamed as new candidates. SPLADE, COIL, ColBERT, and another generic embedding
checkpoint remain lower-priority feasibility studies because current recall and resource
evidence do not justify immediate activation.

The recommendation is now implemented as a pre-metric isolated P8 protocol. Its frozen
selection and confirmation corpus hashes are respectively
`1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee` and
`3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde`.
Independent review aligned builder/runtime evidence confidence, removed an uncovered slot,
sanitized the worker spec, expanded source/evaluator locks, and added a fresh direct served-
Agent reference bridge. The full project suite passes `387/387`; no P8 selection,
confirmation, or released-public metric has yet been run, so no P8 performance claim exists.

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
- Direct source verification cannot establish unpublished final hardware, timeout,
  network, archive-size, or simulator-equivalence details. These remain explicit
  limitations rather than assumed requirements.
