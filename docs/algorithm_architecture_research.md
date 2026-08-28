# P4 Target-Blind Architecture Search

Last updated: 2026-08-28 SGT.

## Decision summary

The current served Agent remains unchanged while P4 compares isolated architectures.
The existing sparse control already has fused recall `0.995` at 50 on the released
public audit, and 11 of its 12 public misses entered that Top-50 pool. That local
evidence makes shortlist discrimination and safe intent handling the first search
priority; it does not prove that semantic retrieval is unnecessary on the private 800.

The research basis and official sources are frozen in `report-source.md`. RRF, MMR,
conversational context selection, dense retrieval, learned sparse retrieval,
late-interaction, and cross-encoder work are used only as design evidence. Every local
claim still requires a target-blind evaluator artifact.

## Frozen experimental protocol

- Selection corpus: 200 deterministic catalog-derived sessions with 200 unique targets,
  zero overlap with all released-public targets, and scenario mix 80/80/30/10.
- Fixed policy: `question_policy=fast`, `rerank_mode=off`, Top-10, at most 10 turns.
- Agent inputs: profile plus visible messages only; never target, sample ID, scenario,
  intent card, simulator behavior, prior result, or target rank.
- Control: `C00.control_rrf`, required to be response-equal to the default Agent.
- A design counts only when it is non-control, contract-clean, fully evaluated, has at
  least one activation, and changes at least one Top-10 output.
- Promotion requires no HR, MRR, TechnicalScore, MTTC, scenario-HR, or hit-to-miss
  regression versus control. Aggregate score alone cannot override a failed gate.
- The raw-score winner, eligible winner, and deterministic confirmation are reported
  separately. The released public set is not used to choose among variants.

Runner:

```powershell
python scripts/evaluate_architectures.py `
  --derived-count 200 `
  --seed track4-p1-product-disjoint-v1 `
  --variants all `
  --confirm-top 3 `
  --output experiments/p4_architecture_search.json
```

## Frozen architecture registry

| ID | Mechanism | Hypothesis and boundary |
| --- | --- | --- |
| C00 | Control weighted RRF | Exact current served sparse ranking; not counted as a new experiment |
| R01 | Field RRF | Independent title/category, feature/detail, and description/store routes can recover field-specific evidence |
| R02 | Category guard | A strict category-field route can suppress cross-category lexical matches, with deterministic fallback |
| R03 | Turn RRF | Versioned per-turn routes can preserve useful evidence without flattening all terms into one query |
| R04 | Phrase route | Exact multiword slot and visible n-gram evidence can distinguish products sharing individual tokens |
| R05 | Alias expansion | A low-trust catalog-domain alias route can reduce lexical mismatch while preserving the original query |
| R06 | Rare anchor | The lowest-document-frequency visible term can anchor precision when frequent attributes dominate |
| R07 | CombSUM BM25 | Normalized raw route scores can discriminate differently from reciprocal rank |
| R08 | Coverage cascade | Distinct visible-term coverage can prioritize candidates satisfying more of the expressed need |
| R09 | Slot filter/relax | Known negative conflicts are never backfilled; positive hard constraints relax deterministically, missing metadata remains unknown, and fewer than ten results are permitted |
| R10 | Candidate carry-over | A decayed prior shortlist can stabilize refinement, but only inside the current goal version |
| R11 | Browsing MMR | Visibly open browsing can benefit from target-blind catalog-aspect diversity; buying/hard constraints bypass it |
| R12 | Numeric budget | Visible under/over/around constraints can rank known prices while reserving space for unknown prices; the frozen derived corpus has no visible budget turns, so it must be `not_counted` there |
| R13 | Intent router | Visible hard constraints, visible open browsing, and ordinary refinement can use different experts without hidden scenario labels |
| R14 | Borda fusion | Normalized route-relative rank votes provide a genuinely distinct aggregation control from RRF and raw-score CombSUM |

Changing only a depth, weight, synonym list, or MMR penalty does not create another
architecture. Unique mechanism and stage-graph fingerprints are enforced by tests.

## Safety and reproducibility gates

The runner refuses to compare a control with contract errors or an incomplete session
set. It validates response keys, allowed question attributes, catalog membership,
uniqueness, Top-10 size, finite optional scores, and usage shape. Contract-invalid
variants cannot count toward the ten-experiment requirement or enter confirmation.

Each artifact records Git branch/commit/dirty state, Python/platform/SQLite versions,
catalog/public/derived hashes, all direct source hashes, timing, activation/fallback
statistics, complete session results, scenario deltas, hit-to-miss changes, and repeated
functional hashes. The full matrix must be run from a clean committed tree.

## Next architecture wave

If no isolated local design passes the gate, the correct conclusion is that the sparse
control remains the best eligible local architecture. The next justified wave is not
more public-session rules. It is an offline, legally distributable semantic route—such
as small E5 dense retrieval or SPLADE-style learned sparse expansion—followed by bounded
hybrid fusion and, only if resources permit, a shortlist cross-encoder or late-
interaction reranker. Each model path must declare model/version/license, asset hash,
disk/RAM, latency, token/network behavior, and an offline fallback before public gating.

## Result status

The 8-session runner smoke test is an implementation check only and is not used for
selection. The frozen 200-session matrix and eligible winner will be recorded here in a
separate committed result after the implementation tree is clean.
