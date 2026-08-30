# Small-ranker v1.5 candidate freeze and fresh validation protocol

## Why validation cannot start yet

The reproducible 97.15% result is outer OOF, not a deployable policy artifact.
Each outer fold currently owns a different fitted model and threshold. Opening a
new split before producing one frozen full-data model would leave room to change
calibration after seeing results.

The old proxy calibration is already observed, proxy selection was partially
opened, and the old confirmation remains sealed. None can be relabeled as fresh
selection evidence for v1.5. Amazon's benchmark test split is explicitly
forbidden.

## Candidate freeze

The next implementation must do only the following on `train_explore`:

1. Train the frozen `ndcg_d4_lr003` ranker and the rescue/RR-regret heads once on
   all development rows. Remove the zero-label isolated hit-loss head.
2. Lock the RR multiplier to `1.0`, selected in four of five outer folds.
3. Choose an activation quantile from cross-fitted development utilities only.
   It must be safe in every fold and in aggregate. Raw thresholds cannot be
   copied between separately fitted models because their probability scales can
   differ. Map the locked quantile to the full model's unlabeled development
   utility distribution; no outcome may adjust that mapping.
4. Export a target-blind, causal artifact with complete input/source hashes and
   P11/R08 fallback. Exact repeat and offline parity are required before any new
   data is opened.

No Agent or full evaluator is needed for this freeze.

## Fresh source without leaking sealed confirmation

The pinned Amazon validation CSV generated every existing proxy split. A fresh
source should therefore use the same revision's `train` split, never `test`.
To guarantee disjointness without reading the sealed confirmation, eligible
targets must satisfy both conditions:

- the target occurs in Amazon `train` but nowhere in the complete pinned
  validation CSV; and
- its catalog family signature occurs for no catalog target in that validation
  CSV.

This excludes the entire universe from which train/explore, calibration,
selection, and sealed confirmation were drawn. Fresh selection and confirmation
must then be mutually family-disjoint. Source bytes, SHA-256, header, revision,
and row count must be pinned in a separate source-freeze commit before corpus
generation. The train source is not currently present, so no download or split
build is authorized by this preregistration.

## Compact validation cache

Repeating a 20,000-turn Agent run for every change is unnecessary. A target-blind
worker should build rich C100 features once, apply the frozen ranker and heads,
and persist only baseline Top10, chosen slot-10 candidate, frozen scores,
probabilities, and the activation decision. It must not receive labels. After it
closes, a separate join may compute official metrics.

The compact cache deliberately omits the roughly 1 GiB full C100 feature tensor.
One expensive build then supports sub-second metric audits and exact repeats.

## Frozen decision gates

Both fresh 2,000-row splits require at least `+0.01` HR@10 (`+20` net hits), zero
hit-to-miss, nonnegative MRR, nonpositive MTTC, and positive net hits in at least
three taxonomies. Confirmation may open only after selection passes and repeats
exactly. Public evaluation and default-on serving remain unauthorized.
