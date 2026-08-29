# Small-ranker v1.2 metric-safe gate diagnostic

This bounded diagnostic used only frozen `train_explore` numeric caches and the
existing `ndcg_d4_lr003` outer-OOF score matrix. It did not start the Agent,
retrain the ranker, or open calibration, selection, confirmation, or public.

## Question

The v1 admission gate optimized zero session hit-to-miss and maximum rescue.
That policy improved HR@10 but caused a large MRR regression in both OOF and
untouched calibration. This experiment asked whether changing only the nested
inner-OOF threshold rule could retain meaningful rescue while requiring:

1. zero hit-to-miss;
2. nonnegative MRR delta;
3. nonpositive MTTC delta; and
4. maximum TechnicalScore among valid thresholds.

## Result

| Policy | miss->hit | hit->miss | net | Fold net | HR delta | MRR delta | MTTC delta | TechnicalScore delta | Active sessions |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Existing gate reference | 56 | 0 | 56 | 7/15/14/7/13 | +0.0280 | -0.059318 | -0.356 | +0.003325 | 1,461 |
| Metric-safe threshold | 7 | 0 | 7 | 0/0/0/0/7 | +0.0035 | -0.000517 | -0.010 | +0.001795 | 22 |

Four inner selections chose the exact KEEP fallback because no active threshold
could meet all metric constraints. Only the fifth selected an active threshold;
on its held outer fold it rescued seven sessions but still regressed MRR by
`0.002583`. Total wall time was `2.441047` seconds.

## Decision

**NO-GO.** Threshold-only tuning does not solve the rank-quality tradeoff. The
challenger misses the existing `+10` net gate, rescues only 1/5 outer folds, and
does not preserve MRR on outer OOF. No runtime or held-out evaluation is allowed.

The next useful algorithm is not another scalar-threshold sweep. It should add
an explicit target-blind harm head for loss of the earliest reciprocal-rank hit,
or learn direct session utility combining HR, reciprocal rank, and first-hit
turn. Existing P11/R08 behavior and the v1.1 runtime fallback remain unchanged.

The complete ignored artifact is
`experiments/fast_track/small_ranker_metric_gate_v1.json`; the tracked evidence
record is `configs/small_ranker_v1_2.metric_gate.manifest.json`.
