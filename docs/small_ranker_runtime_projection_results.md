# Small-ranker v1.6 deployable-feature projection

The v1.5 OOF result used the full rich feature cache, including a semantic route
that is absent from the lightweight served runtime. This bounded experiment
recomputed every C100 feature group with the frozen semantic-route-off projection
and used the already frozen projected OOF ranker scores. It did not start the
Agent/runtime or read any held-out split.

## Result

| Feature surface | m→h | h→m | net | Fold net | HR@10 | MRR delta | MTTC delta | TechnicalScore delta |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Full semantic v1.5 | 48 | 0 | 48 | 7/13/8/7/13 | 0.9715 | +0.001933 | -0.1115 | +0.014810 |
| Runtime semantic-off | 48 | 0 | 48 | 7/13/8/7/13 | 0.9715 | +0.001933 | -0.1015 | +0.014610 |

All five folds retain positive net hits, positive MRR, improved MTTC, and zero
hit-to-miss. Runtime projection changes the selected action surface slightly
(414 versus 412 RR-regret labels and 8,226 versus 8,098 activated turns), but it
does not change HR or reciprocal-rank outcomes. Projection plus nested analysis
took `11.825018` seconds.

## Decision

**Deployable feature parity passed; not promoted.** The earlier concern that the
97.15% result depended on the unavailable semantic route is rejected. The
candidate remains OOF-only and is not yet one full-data artifact.

The next checkpoint is an exact repeat of this projected experiment. Only then
may the preregistered full-data ranker and rescue/RR-regret heads be exported,
using the cross-fitted activation quantile and unlabeled full-model quantile
mapping. P11/R08 fallback and the served default remain unchanged.
