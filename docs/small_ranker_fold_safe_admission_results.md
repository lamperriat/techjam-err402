# Small-ranker v1.5 fold-safe admission

This bounded experiment used only frozen `train_explore` caches. It did not
retrain the ranker, start the Agent/runtime, or read calibration, selection,
confirmation, or public data.

## What changed

v1.4 showed that reciprocal-rank modeling can reach 97% HR@10, but an admission
threshold safe on pooled inner OOF still caused three outer hit-to-miss failures.
v1.5 therefore required every selected threshold to satisfy zero hit-to-miss,
nonnegative MRR, and nonpositive MTTC separately in all inner folds as well as
in aggregate.

A third isolated hit-loss head was also tested. It had zero positive labels:
none of the 40 direct-risk actions caused a session miss by itself. The v1.4
losses were a joint-action effect—several individually safe removals could erase
all hits in one session. Consequently the measurable improvement comes from the
fold-robust admission rule, not the degenerate third head.

## Result

| Policy | m→h | h→m | net | Fold net | HR@10 | MRR delta | MTTC delta | TechnicalScore delta |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| v1.3 first-hit harm | 34 | 0 | 34 | 0/12/7/6/9 | 0.9645 | +0.000490 | -0.0625 | +0.009898 |
| v1.4 pooled-safe RR regret | 48 | 3 | 45 | 7/13/5/7/13 | 0.9700 | +0.001183 | -0.1160 | +0.013925 |
| v1.5 fold-safe RR regret | 48 | 0 | 48 | 7/13/8/7/13 | 0.9715 | +0.001933 | -0.1115 | +0.014810 |

Every fold has positive net hits, zero hit-to-miss, positive MRR, and improved
MTTC. Fold MRR deltas are `+0.001750/+0.002250/+0.002000/+0.001750/+0.001917`.
The cached nested experiment took `8.222593` seconds.

## Decision

**Best OOF candidate so far; not promoted.** It improves HR@10 by 2.4 percentage
points while satisfying the hard safety constraints in all outer folds. It is
still aggressive—8,098 activated turns across 1,412 of 2,000 sessions—and all
evidence remains within the development split.

The next checkpoint is an exact frozen reproduction. If reproducible, validation
must use a newly frozen product-family-disjoint split. The already observed
calibration cannot be reused as untouched evidence. P11/R08 fallback and the
served default remain unchanged.
