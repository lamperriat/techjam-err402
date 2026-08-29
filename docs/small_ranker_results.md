# Small Ranker v1 results

This ledger covers only the frozen `train_explore` research path. Calibration,
selection, confirmation, public, and full-Agent evaluation remain unopened.

| Stage | Commit | Feature schema | Model config | OOF HR delta | miss→hit | hit→miss | Fold coverage | Wall | Decision |
|---|---|---|---|---:|---:|---:|---|---:|---|
| Rich C100 cache | pending checkpoint | `92795134` (133 features) | n/a | n/a | n/a | n/a | 400 sessions × 5; 3,703–3,794 positive query groups/fold | 1,623.098 s feature phase | proceed to one preregistered LambdaMART batch |

Cache facts: 2,000,000 float32 rows, 20,000 query groups, 1,064,000,128
bytes, feature SHA-256
`2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089`,
label SHA-256
`9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb`.
The official baseline join is 1,895 hits and 105 misses. Numeric and ephemeral
visible-context scans found zero identifier-shaped values.

The catalog-metadata family signature produced 747 groups for 747 unique
labels (no cross-parent merges), so this run is exact-parent grouped in effect.
That is an explicitly optimistic evidence boundary, not proof of generalization
to unseen near-duplicate product families.
