# Small-ranker v1.3 first-hit-harm gate

This bounded experiment reused the frozen `train_explore` feature, label, and
`ndcg_d4_lr003` outer-OOF score caches. It did not retrain the ranker, start the
Agent/runtime, or open calibration, selection, confirmation, or public data.

## Question

The v1 ranker recovered 56 misses but lost `0.059318` MRR. A threshold-only
metric gate then preserved only seven rescues and still lost MRR. This experiment
therefore learned a second target-blind logistic head for a narrower failure:
replacing slot 10 exactly when it contains the session's earliest eligible hit.
The admission utility was `P(rescue) - lambda * P(first-hit harm)`. Both `lambda`
and the threshold were selected inside each outer fold using inner OOF only.

## Result

| Policy | miss->hit | hit->miss | net | Fold net | HR@10 | HR delta | MRR delta | MTTC delta | TechnicalScore delta |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| P11 baseline | 0 | 0 | 0 | 0/0/0/0/0 | 0.9475 | 0 | 0 | 0 | 0 |
| v1.3 nested OOF | 34 | 0 | 34 | 0/12/7/6/9 | 0.9645 | +0.0170 | +0.000490 | -0.0625 | +0.009898 |

The fold MRR deltas were `0`, `+0.002786`, `+0.001750`, `-0.003000`, and
`+0.000917`. The policy activated 508 turns in 188 sessions and activated none
of the nine labeled first-hit-harm rows. Total offline wall time was `3.343320`
seconds.

## Decision

**Promising OOF; not promoted.** This is the first cached challenger in this
line that improves all four aggregate metrics while retaining zero hit-to-miss.
It is still too fragile to serve: one fold regressed MRR, one fold rescued
nothing, and nine harm-positive rows are too few for a stable learned head.

Before any held-out check, the result required an exact deterministic
reproduction from the frozen hashes. Selection, confirmation, public evaluation,
runtime integration, and the served default remained closed/off. P11/R08
fallback behavior was unchanged.

## Exact reproduction

The frozen experiment was repeated on the same branch without changing code,
features, labels, scores, seeds, or protocol. Excluding only `timing_seconds`,
both artifacts have the identical canonical SHA-256
`2a9d46cd756a600707481d9187ac03658232202413c4d904d0c02366a275974a`.
All global and fold metrics reproduced exactly: `34` miss-to-hit, `0`
hit-to-miss, fold net `0/12/7/6/9`, HR delta `+0.0170`, MRR delta
`+0.000490`, MTTC delta `-0.0625`, and TechnicalScore delta `+0.009898`.
The repeat took `3.402735` seconds.

The exact OOF result is reproducible, but the repository's existing calibration
split is **not untouched for this design**: it was opened for v1 and its observed
MRR regression directly motivated the harm head. It must not be rerun or renamed
as independent evidence. The next legitimate validation must be a newly frozen,
product-family-disjoint source split. Selection, confirmation, public, and
runtime integration remain closed.

The complete ignored artifact is
`experiments/fast_track/small_ranker_mrr_harm_gate_v1.json`; the tracked evidence
record is `configs/small_ranker_v1_3.mrr_harm_gate.manifest.json`.
