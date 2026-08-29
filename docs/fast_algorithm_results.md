# Fast Algorithm Results

Last updated: 2026-08-29 SGT. This aggregate-only registry prevents exact-repeat runs; it
contains no blind trace, public-session row, target/ASIN, or confirmation result.

## Old matrix identity

- Frozen checkpoint: `761b9a3`
- Config canonical SHA-256: `492b42c19708b0e528755cb00374b368afaf037ce2c8b1f5d33f52685de3638c`
- Actions: `KEEP_R08`, `KEEP_P11`, `CANDIDATE_RERANK`,
  `FROZEN_SEMANTIC_RERANK`, `RESULT_AWARE_REWRITE_RETRIEVE`, `ASK`
- Parameters: four workers, fixed ten-turn replay, Top-10 response, candidate-only C50
  structured/semantic actions. `ASK` is observed and gate-excluded.

## Successful-run reuse ledger

`m→h / h→m / net` is reported for Candidate; Semantic follows in the next column. The v1
aggregate schema did not emit a separate activation count, so that field is recorded as
`not emitted` rather than inferred.

| Split | Limit | Runner source SHA prefix | Output | Wall | Baseline / C50 / Oracle HR | Candidate m→h / h→m / net | Semantic m→h / h→m / net | Rewrite net | Activation | Decision |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| train/explore | 10 | `eb01b2504f20` | `train_explore-smoke-10-aggregate.json` | 15.867314s | 1.000 / 1.000 / 1.000 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | not emitted | REUSE; protocol-only |
| train/explore | 100 | `910d5229b95b` | `train_explore-smoke-100-aggregate.json` | 76.357433s | 0.940 / 0.970 / 0.940 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | not emitted | STOP; no rescue |
| train/explore | 200 | `327b72b41988` | `train_explore-smoke-200-aggregate.json` | 103.241977s | 0.935 / 0.985 / 0.940 | 1 / 1 / 0 | 0 / 5 / -5 | 0 | not emitted | STOP; no net rescue |
| train/explore | full 2,000 | `327b72b41988` | `train_explore-full-aggregate.json` | 947.950242s | 0.9475 / 0.991 / 0.956 | 14 / 21 / -7 | 5 / 48 / -43 | 0 | not emitted | STOP old matrix |
| calibration | full 2,000 | `327b72b41988` | `calibration-full-aggregate.json` | 1076.397994s | 0.932 / 0.987 / 0.950 | 16 / 13 / +3 | 22 / 36 / -14 | 0 | not emitted | STOP; inconsistent with train |

The 10/100 rows are legacy successful aggregate artifacts with earlier runner source
closures; they remain non-decision evidence and must not be rerun merely to normalize the
runner hash. Before any future run, compare commit/config/action IDs/split/limit/parameters
and output path with this ledger. An exact match must reuse the existing aggregate.

## Interrupted selection

The old-matrix `selection` run was stopped by Ctrl+C after about eight minutes. Parent and
all four workers exited, and no `selection-full` artifact was produced. The opened split is
not valid formal one-shot evidence; any future formal selection must be newly generated,
previously unopened, and target/product-family-disjoint. Confirmation remains sealed.

## Next compact-action batch

| Commit / config | Actions added | Split / limit | Wall | HR delta | m→h | h→m | Net rescue | Activation | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PENDING | `COMPACT_NEGATIVE_C50`, `GUARDED_COMPACT_SLOT10` | train/explore / 100 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Only one shared `limit=100` run is permitted after both actions are implemented and the
single targeted test batch passes. Expand to `limit=200` only if an individual new action
has activation, at least one rescue, positive net rescue, and positive HR delta.
