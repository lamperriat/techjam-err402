# Small-ranker v1.4 reciprocal-rank-regret gate

This bounded experiment used only the frozen `train_explore` numeric caches. It
did not retrain the ranker, start the Agent/runtime, or read calibration,
selection, confirmation, or public data.

## Corrected failure model

The v1.3 label covered removal of an existing earliest rank-10 hit. That was not
the dominant MRR failure. The evaluator stops at the first hit, so inserting the
target at rank 10 on an early turn can preempt a later baseline hit at rank 1–9.
HR and MTTC improve while reciprocal rank falls. This explains how v1.3 could
activate zero labeled harm rows yet still lose MRR in one fold.

v1.4 computed the exact isolated reciprocal-rank regret for every candidate
action before target removal, then learned a target-blind binary regret head.
This yielded 412 regret-positive actions instead of only nine v1.3 harm labels.

## Result

| Policy | m→h | h→m | net | Fold net | HR@10 | MRR delta | MTTC delta | TechnicalScore delta |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| v1.3 first-hit harm | 34 | 0 | 34 | 0/12/7/6/9 | 0.9645 | +0.000490 | -0.0625 | +0.009898 |
| v1.4 RR regret | 48 | 3 | 45 | 7/13/5/7/13 | 0.9700 | +0.001183 | -0.1160 | +0.013925 |

Fold hit-to-miss was `0/0/3/0/0`; fold MRR delta was
`+0.001750/+0.002250/+0.001250/+0.001750/-0.001083`. The policy activated 8,802
turns across 1,450 sessions and the cached experiment took `3.541212` seconds.

## Decision

**NO-GO.** The corrected label materially improves the aggregate frontier and
reaches 97.0% OOF HR@10, but three hit-to-miss regressions violate the hard
safety gate and one fold still loses MRR. It does not replace v1.3.

The next algorithm should use three distinct target-blind heads: rescue,
reciprocal-rank regret, and direct hit-loss. Nested admission must veto predicted
hit-loss before optimizing the rescue/regret utility. Existing P11/R08 fallback
and the served default remain unchanged.

The repository calibration result was observed before v1.3 and influenced this
design, so it cannot be reused as untouched evidence. A future survivor requires
a newly frozen product-family-disjoint validation split.
