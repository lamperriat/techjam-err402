# Implementation Log

This tracked document records only code and behavior that exist in the repository and have been verified. Planning, hypotheses, and unvalidated designs belong in ignored `docs/internal_plan.md`; the current working-tree architecture belongs in ignored `docs/current_architecture.md`.

Last updated: 2026-08-29 SGT.

## Current verified implementation

- Branch: `p11-p12-fast-track`
- P11 preregistration lock: `c6efa5f` (`chore: freeze P11 preregistration lock`)
- P11 frozen source: `639cf78` (`fix: align P11 protocol hash with corpus builder`),
  including the P11 implementation checkpoint at parent `4f27ee8`
- P11 formal decision: `promote_p11_r01`; aggregate result SHA-256
  `fe0f8820b22c07136db44fb3739809d22b8edc5d1125707c5b0523dec312b912`
- P9 preregistration lock: `e36d515` (`chore: freeze p9 preregistration lock`)
- P9 compact-negative protocol: `d03690d` (`feat: add hardened compact-negative p9 protocol`)
- P4 served/reference bridge: `1f8fd3c` (`test: bridge frozen and promoted response traces`)
- P4 served promotion: `97bc89c` (`feat: promote coverage cascade into served agent`)
- P4 frozen matrix: `e5d0d49` (`feat: add target-blind architecture search lab`)
- P4 Workbench alignment: `04b6e21` (`feat: align observer with promoted retrieval`)
- P4 R12 hygiene fix: `eb626bc` (`fix: reject measurement-only budget signals`)
- P3 implementation: `9cc9262` (`feat: add auditable clarification shadows`)
- P3 verification and data inventory: `87447fb` (`docs: record p3 gates and official data inventory`)
- Frozen P1 head: `02f0741` on `p1-generalization`
- P2 core implementation: `586f3dd` (`feat: add target-blind shortlist reranker`)
- P2 Workbench/tooling: `4610480` (`feat: expose rerank experiments in workbench`)
- P2 v1 gate record: `f91b547` (`docs: record p2 rerank gate results`)
- Optional dependency isolation: `71383b5` (`build: isolate optional LLM dependencies`)
- Resource/route benchmark: `38ca016` (`test: add resource and route recall benchmark`)
- P1 implementation commit: `abae926` (`feat: add generalization gate and robust intent state`)
- P1 parent checkpoint: `66cb1cf` (`docs: finalize integration verification`)
- Stateful Agent integration: `5fed7a7` (`feat: integrate stateful sparse shopping agent`)
- Workbench baseline: `f4e435b` (`feat: add agent layer workbench`)
- Official upstream main rechecked on 2026-08-28: `34078351e1c3615e5505a2e829600b56a542e462`
- Runtime: Python 3.11.16 in the existing `tiktok` Conda environment
- Catalog: 50,000 parseable rows and 50,000 unique non-empty `parent_asin` values
- Public set: 200 sessions and 200 unique targets, split 80 Buying / 80 Browsing / 30 Intent Override / 10 Boundary
- Official catalog release SHA-256: `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8`; local compressed asset is identical
- Official public-set Git blob: `121dbec9c1368c81cd887d6959e62507512139c0`; local Git-normalized content is identical
- Default execution: offline, no LLM object, no API key, no network call, zero reported tokens
- Direct `Agent()` defaults are `TECHJAM_RETRIEVAL_MODE=coverage`,
  `TECHJAM_RERANK_MODE=off`, `TECHJAM_QUESTION_POLICY=fast`, and
  `TECHJAM_P11_MODE=active`; clear inherited values or set them explicitly for a
  production run. `TECHJAM_P11_MODE=off` restores the complete R08 order, while
  `retrieval_mode=control` preserves the pre-P4 weighted-RRF output for paired experiments.

## P12 validation-only proxy builder

- Added a fixed-source, standard-library streaming builder for the pinned Amazon Reviews
  2023 Clothing_Shoes_and_Jewelry 5-core validation CSV; filename, byte count, SHA-256,
  five-column schema, official catalog identity, 16 consumed corpora, and the empty manual
  exclusion ledger all fail closed before outputs are published.
- The local-only source contains 2,524,981 rows. After the frozen-catalog inner join and
  2,720-target exclusion, 35,717 rows / 2,986 targets remain. No test split was read.
- Four 2,000-row target-group-disjoint splits use exact 40/40/15/5 scenario quotas,
  catalog-only taxonomy/popularity/difficulty, safe aggregate profiles, and an explicitly
  sealed confirmation filename. Raw user/rating/timestamp/prior-ID sequences never persist.
- One outcome-independent validation-source-frequency outlier is assigned to train/explore before the
  held-out hash split. Purchase-frequency uses only a bucketed pre-validation history
  length; preference tags use joined frozen-catalog metadata. The manifest exposes raw
  source weights, target-uniform and taxonomy stress views, concentration/effective-target
  diagnostics, and the 2.4736% prior-catalog join limitation.
- Production load/build pins the complete tracked config, records canonical/LF hash modes,
  re-verifies inputs after parsing, rejects input/output collisions, and refuses both flat
  CLI configs and forged production dataclasses. Publication uses exclusive hard links,
  verifies identical pre-tracked aggregate evidence, and rolls back ordinary exceptions;
  it is explicitly not crash-atomic as a six-file set.
- Thirteen fixture/security tests pass, including config/source drift, fresh-checkout
  aggregate evidence, production-marker forgery, and injected publish failure. Final
  commit-bound aggregate hashes are generated only after this implementation checkpoint;
  derived rows remain ignored and are never eligible for tracking.

## P11 Top-10-preserving reranker: frozen formal promote decision

The sole preregistered P11 formal run completed with decision `promote_p11_r01` and winner
`P11.R01.top10_linear`. The frozen winner is now connected to the production path through
`starter/p11_bridge.py`: direct served construction defaults to P11 `active`, while the
complete fallback remains R08 `coverage/off/fast`. P9 stayed frozen and was neither
modified nor rerun. Released public was not evaluated during this integration.

- The bridge verifies the frozen catalog, scorer contract, SQLite-internal feature
  metadata, sidecar byte count and SHA-256 before serving any P11 order. The external
  manifest is audit/provenance metadata. Its read-only SQLite fetch is capped at the ten
  existing R08 head members per turn.
- P11 may only permute those exact ten members; every item from rank 11 onward is unchanged.
  Equal final scores preserve R08 order. Any initialization, identity, fetch, subtype,
  scoring, adapter, or boundary failure records a diagnostic and keeps R08 output;
  shutdown failures propagate instead of producing a false completed state.
- The tracked sidecar is 32,501,760 bytes with SHA-256
  `83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473`.
- Explicit historical retrieval/rerank/question settings retain their prior off semantics
  unless P11 mode is also explicitly requested. The no-terminal Workbench launcher pins
  the new served preset instead of inheriting stale shell values.
- Verification on the isolated fast-track worktree passed 68 focused P11/core tests,
  all 580 repository tests, all 14 official-asset checks, and `compileall`. No released-
  public evaluation was run.

- Added an exact future-experiment metric bridge that validates HR, MRR, MTTC,
  Efficiency, TechnicalScore, `best_rank`, and reciprocal-rank consistency in the official
  order: the evaluator first rounds aggregate HR/MRR/MTTC to six decimals, then computes
  Efficiency and TechnicalScore from those rounded values. Focused fixtures cover the P9
  one-millionth boundary without altering the frozen P9 artifact or decision.
- Added a fixed P11 B00/C00/S00/R01 experiment layer. B00 is a fresh direct served-Agent
  reference, C00 must be response/trace-equal to R08, S00 computes diagnostics but must
  return C00 output, and R01 may only permute the exact R08 Top 10. The member set and
  every rank after 10 are invariant; any feature, sidecar, or instrumentation failure
  returns the complete R08 order.
- Added scorer `p11.top10-linear.v3` with Broad/Strict/RRF rank priors, frozen-IDF query
  coverage, three catalog field groups, subtype consistency, positive
  observed/inferred/unknown evidence, P9 explicit-conflict partitioning, and constraint
  source-turn/version weighting. Hard-clause evidence combines exact field-local 2/3-grams
  with an exact 4-12-token full-clause match; the complete phrase contributes 50% of that
  component. Within each conflict bucket, relevance is partitioned into anchored,
  non-chaining near-tie groups at `<=0.002`; subtype-normalized Bayesian rating/popularity
  is used only inside a group and cannot cross a material relevance gap.
- The scorer performs one at-most-ten-row sidecar fetch per turn and avoids runtime JSON,
  candidate-loop regex compilation, model initialization, dense retrieval, and LLM calls.
  Its precise added-work bound is `O(B+S+U+10×F)`, with frozen route caps `B<=120`,
  `S<=80`, and `U<=200`; feature fetch/scoring itself is `O(10×F)` and remains
  independent of the 50,000-row catalog size.
- Added a catalog-only, label-free, target-blind compressed SQLite sidecar builder and a
  read-only immutable loader. The current verified 50,000-row asset is `32,501,760` bytes,
  below the 32 MiB cap, with SHA-256
  `83b6d8c04be6666173806b6e9cb03301eecb8ca58a60272bfa719e6533380473`.
  Its feature registry is
  `c2c6b4309e5bbf8e092f625957ae5f0cdeb193adcc48d552e5291837803749b1`, and its semantics
  hash is `abae7be9ab9073593ca40309177408adf20e460e3153fe95ec942fb53b47a488`.
- Added seven deterministic fresh corpora. Their 920 targets are mutually disjoint and
  disjoint from the 1,800-target released-public/P1/P5/P6/P7/P8/P9 opened registry:

| P11 split | Rows | SHA-256 |
| --- | ---: | --- |
| Primary representative | 200 | `1d578694c3226d1b008d2c9f2f252ed63d114a544c82c218c06116b13c00cf84` |
| Uniform tail | 200 | `87d2334dd28dded92df2d8c8897f7f9552efb655bc74488d49dafe2f6efc1dfd` |
| Confirmation | 200 | `6dfdcdaf8cd6a091a9b82c192b076ad4e48a89b4023d5ef65394a6d6daf737ba` |
| Negative failure slice | 80 | `c0c593dc90af45ec9f3dcdfaaace286f9b0a53c52d0a833c8d292a4488126290` |
| Budget failure slice | 80 | `a522134897f7ab8348c327a9a53d30075033dc379bcec610777201abfbb6ee91` |
| Override failure slice | 80 | `1eeb7e552f2ef0ce8aae413adb7a6393891f1e265c774728ed6ba3b35685df95` |
| Missing-evidence failure slice | 80 | `2aca6b723b592b84caf173fb55c231e8572d0844da9f57cd0c89b9e0489f4ef9` |

  Primary and confirmation each preserve the 80/80/30/10 scenario mix and the same
  catalog-only popularity quota. Uniform-tail is a separate non-regression surface.
  Corpus metadata SHA-256 is
  `40995692dda99dbca7d94382568e656f45aa1575874f37625d07feb1d8866b8e`; it records all 7
  opened-to-new comparisons, all 36 opened-pair comparisons, and all 21 new-pair
  comparisons as zero overlap. The four 80-row failure slices are only prebuilt and
  hash/schema/disjointness-validated: `runs_per_slice=0`, so they are not effect estimates
  or promotion evidence. Released public is used only for identity and target exclusion,
  never weight search.
- The first real-asset lock attempt correctly stopped on a validator canonicalization bug:
  the corpus builder hashes one compact JSON record with its terminating LF, while the lock
  validator had omitted that byte. The validator now reproduces the builder's exact JSONL
  contract, a cross-module test binds both implementations, and a negative test rejects the
  former LF-free hash. Existing corpus metadata, protocol-file identity, corpus-builder
  source identity, and every corpus hash were already correct, so no corpus was rebuilt and
  confirmation semantics were not parsed.
- Added a fail-closed preregistration builder and parent/worker runner. Formal lock
  generation requires a completely clean, pushed HEAD, exact official origin/upstream
  proof, official catalog/public/evaluator/config identities, the complete source closure,
  all corpus identities, and the final sidecar identity. The lock builder hashes
  confirmation bytes but does not JSON-parse confirmation.
- Added fresh staged workers with role-specific read access, source/runtime manifests,
  exact B00/C00 and C00/S00 equality, Top-10/tail guards, absolute role deadlines, a
  frozen 5,400-second `time.monotonic` whole-formal-run deadline, and persistent one-shot
  attempt/confirmation markers. The pre-import audit reports every denial through a
  parent-owned, role/nonce/sequence-bound stdout event stream and immediately terminates
  with code 96; an event-channel identity/write failure terminates with code 98. A clean
  exit is accepted only after the reader reaches EOF and the supervisor counts exactly
  equal the O_EXCL final-record counts. Candidate `atexit` callbacks and agent cleanup
  remain inside this audit boundary. The v3 final record measures peak RSS only after
  those callbacks finish; the parent loads it only after a clean exit, validates its exact
  schema, and replaces the worker's untrusted pre-exit memory field. Windows uses typed
  `GetProcessMemoryInfo/PeakWorkingSetSize`. The staged import root is the fixed bootstrap
  stage, not a path inferred from a candidate worker.
- SQLite access is also role-scoped inside the bootstrap. Every exposed `connect` and
  `Connection` alias is wrapped, every connection receives an authorizer, returned
  connection/cursor objects do not expose the raw handle, and ATTACH/DETACH, authorizer
  mutation, and extension loading fail-stop through the same bound event stream. Tests
  cover the direct `Connection` constructor and statically concatenated SQL/method names;
  B00/C00 still report that they never opened the sidecar, while S00/R01 verify the exact
  immutable sidecar identity.
- The preregistration source scan now combines raw text, decoded bytes, recursive compiled
  constants, and bounded static-expression extraction for concatenation, repetition,
  f-strings, and literal joins. It rejects every complete ASIN-shaped value and the frozen
  target-blind forbidden operations even when assembled from bounded static fragments.
  After each of primary, uniform-tail, and confirmation is semantically loaded, the parent
  independently scans the complete 24-file source closure for that split's exact target
  identifiers before launching any worker for that split. Each proof is bound to the
  recomputed target-registry hash and exposes only aggregate counts and hashes; the final
  artifact scrub covers all loaded target and sample identifiers.
  This remains a trusted-Python boundary, not a hostile-native OS sandbox against arbitrary
  dynamic construction, reflection, re-imported low-level modules, or native code.
- Promotion is fail-closed and requires strict TechnicalScore gains on primary and
  confirmation, primary delta at least `0.005`, paired 10,000-resample 95% bootstrap lower
  bounds above zero, uniform-tail non-regression, no HR/MRR/scenario-HR/MTTC regression,
  zero hit-to-miss, wall/P95/RSS ratios at most `1.15/1.20/1.10` against both B00 served
  and C00 control, exact fresh-process repeats, and clean
  contract/target-blind/network/token/exception audits. The paired bootstrap uses the
  explicitly frozen unrounded per-session TechnicalScore contribution; exact rounded
  official aggregate scores are validated separately.
- Runner hardening had independent GO reviews for the parent audit stream and all three
  target-source scans; a final readiness re-review of the SQLite/RSS hardening also
  returned GO. Source commit `639cf78` was pushed before lock generation. The lock-only
  commit `c6efa5f6250c013e4a6618661ef947623e709cc5` was then pushed with lock SHA-256
  `1fbcd9b52062cd342b2f29b0a1c66b4eb1d892ffdbe2cd9b1fd35894eb412325`.
  Locked `--dry-preflight` passed identity validation and the four-role zero-session smoke;
  confirmation remained byte/hash-only and no one-shot marker existed at that point.
- The one formal run then followed the frozen order: primary initial, primary fresh exact
  repeat, uniform-tail non-regression, confirmation consumption and semantic parse,
  confirmation initial, confirmation fresh exact repeat, final gate. It finished in
  `477.032` seconds under the 5,400-second whole-run deadline. Primary and confirmation
  repeats were exact, all 18 formal subprocesses were fresh, and every contract,
  target-blind, network, token, generic-exception, pre-import audit, Top-10 membership, and
  tail-preservation check passed.

| Split | Role | HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Primary | C00 R08 | 0.930000 | 0.583204 | 3.370000 | 0.763000 | 0.792561 |
| Primary | R01 Top-10 | 0.930000 | 0.625488 | 3.370000 | 0.763000 | 0.805246 |
| Uniform tail | C00 R08 | 0.900000 | 0.607732 | 3.540000 | 0.746000 | 0.781520 |
| Uniform tail | R01 Top-10 | 0.900000 | 0.618764 | 3.540000 | 0.746000 | 0.784829 |
| Confirmation | C00 R08 | 0.955000 | 0.581927 | 3.140000 | 0.786000 | 0.809278 |
| Confirmation | R01 Top-10 | 0.955000 | 0.629524 | 3.140000 | 0.786000 | 0.823557 |

Initial-run absolute resources were:

| Split | Role | Wall seconds | Response P95 ms | Peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Primary | C00 R08 | 25.644050 | 61.1538 | 147,890,176 |
| Primary | R01 Top-10 | 26.960285 | 63.9380 | 151,748,608 |
| Uniform tail | C00 R08 | 27.889994 | 64.2017 | 147,480,576 |
| Uniform tail | R01 Top-10 | 28.020347 | 64.8137 | 150,405,120 |
| Confirmation | C00 R08 | 23.525219 | 70.2335 | 147,615,744 |
| Confirmation | R01 Top-10 | 24.113710 | 72.5760 | 151,543,808 |

- TechnicalScore deltas were `+0.012685` primary, `+0.003309` uniform-tail, and
  `+0.014279` confirmation. The preregistered 10,000-resample paired 95% CIs were
  `[0.004354166667, 0.021275000000]` primary and
  `[0.004538690476, 0.024201190476]` confirmation, both strictly above zero. Their
  unrounded paired observed means were `0.012685119048` and `0.014279166667`.
  Every split had zero HR and MTTC delta, zero hit-to-miss, and zero scenario-HR delta.
  Shared control/candidate scenario HRs were primary `0.800000/0.925000/0.937500/0.966667`,
  uniform-tail `0.900000/0.925000/0.887500/0.866667`, and confirmation
  `0.900000/0.987500/0.937500/0.933333` for
  boundary/browsing/buying/intent-override respectively. The preregistration constrained
  scenario HR, not scenario MRR: uniform-tail intent-override MRR changed
  `0.712037 -> 0.685370`, and confirmation boundary MRR changed
  `0.428571 -> 0.420952`. These do not invalidate the frozen decision but remain explicit
  production-integration risks to monitor.
- Initial-run candidate ratios against the worse of C00 or B00 were, by split, primary
  wall/P95/RSS `1.051327/1.045528/1.027000`, uniform-tail
  `1.004674/1.009532/1.024582`, and confirmation
  `1.027087/1.033353/1.026610`. Including exact repeats, the global worst observed
  wall/P95/RSS ratios were `1.051327/1.052578/1.027000`; all are below the frozen
  `1.15/1.20/1.10` limits.
- The aggregate-only artifacts are fixed by SHA-256: formal-attempt marker
  `3638d4f7f95c3d877bf47b77210b6f7a448330bd6f1d77f885ffd1073d0fd669`,
  confirmation-consumed marker
  `b72590359d10ee1f52ea6e0876be669f9da21256ac8486f449fac05fc1865df3`, and result
  `fe0f8820b22c07136db44fb3739809d22b8edc5d1125707c5b0523dec312b912`.
  Failure slices remained non-gating with `runs_per_slice=0`. Post-result verification
  passes the complete `559/559` suite, official assets `14/14`, and `compileall`.
- At the time of the formal gate, R01 was promoted only for a later reversible production
  integration and direct/default behavior still remained R08. That later integration is
  the served bridge recorded at the start of this section; the released-public checkpoint
  itself remains unchanged because public was not rerun.

## P9 compact-negative experiment: frozen retain decision

- Added a compact catalog-only evidence sidecar, P9-only C00/S00/R01 layers, deterministic
  selection/confirmation builders, staged fresh-process worker, preregistration builder,
  strict runner, and focused tests without importing P9 into `starter/agent.py`.
- The 50,000-row sidecar is 1,486,848 bytes, label-free and target-blind, with SHA-256
  `2bc5846b7f6efb2e8395ea99b6bca5b585fb1507d23d6289dbc00d7600d22128`.
  The two 200-target P9 splits are mutually disjoint and exclude released-public plus
  P1/P5/P6/P7/P8; their SHA-256 values are
  `6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f` and
  `4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9`.
- Source/spec was frozen and pushed in `d03690d`; the separate preregistration lock commit
  is `e36d515`, with lock SHA-256
  `32d113e4927925039786054faf9fe35a1ee86606f971b0b60904b6cad9453ced`.
  The complete suite passed `458/458` before the sole formal run and again after result
  documentation; official assets passed `14/14`.
- Selection C00 to R01 changed hits `42 -> 50`, HR `0.210000 -> 0.250000`, MRR
  `0.065454 -> 0.089877`, MTTC `9.175000 -> 8.785000`, and Score
  `0.161136 -> 0.196263`. The exact B00/C00/R01 repeat passed.
- Confirmation initial C00 to R01 changed hits `37 -> 45`, HR `0.185000 -> 0.225000`,
  MRR `0.056688 -> 0.084885`, MTTC `9.330000 -> 8.960000`, and Score
  `0.142906 -> 0.178765`. Both splits had eight miss-to-hit, zero hit-to-miss, 11 earlier
  hits, 25 rank improvements, and no scenario hit-count regression.
- All R01 bootstrap/wall/P95/RSS ratios passed on selection initial, selection repeat, and
  confirmation initial. Eleven fresh worker runs used distinct PIDs/nonces and recorded
  zero network, denied-read, process-creation, contract, integrity, or generic-exception
  events. The worker controls trusted Python with staging, audit hooks, read boundaries,
  and direct AST scans; it is not an OS sandbox against hostile native code.
- The frozen decision is `retain_p9_c00`. Confirmation B00/C00/S00 failed only the exact-
  metric bridge: the official evaluator first rounds aggregate metrics and reports Score
  `0.142906`; the bridge rounded the exact contribution sum to `0.142907`. Confirmation
  repeat was therefore not attempted. This is an inconclusive repeat boundary, not an
  algorithm/resource failure or a confirmation pass. No P9 rerun, released-public run, or
  production promotion occurred.
- The ignored redacted aggregate artifact SHA-256 is
  `62134b9555cb33df5c1009f341ff15eccd2782d5f33c00cb5d86699b18a4ee66`.

## P8 explicit-negative experiment: completed frozen selection

- Added a deterministic P8 corpus builder with frozen official/prior input hashes and two
  frozen 200-session outputs. Selection SHA-256 is
  `1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee`;
  confirmation SHA-256 is
  `3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde`.
  The 400 targets are mutually disjoint and exclude released-public plus P1/P5/P6/P7.
- Negative values are catalog-derived only: same reliable category bucket, minimum three
  supporting documents, leaf before coarse, no global fallback, and no description, Agent,
  FTS, prior result, or metric input.
- Added pure high-confidence explicit-negative compilation and a stable
  `compatible -> unknown -> explicit_violation` partition over the first 50 R08 candidates.
  Unknown metadata is never treated as a conflict; the untouched tail and deterministic
  violation fallback preserve complete catalog-only output.
- Added experiment-only C00/S00/R01 Agents. C00 is response/route-equal to explicit served
  `coverage/off/fast`; S00 is output-equal shadow diagnostics; R01 is the sole active arm.
  No P8 module is imported by `starter/agent.py`.
- Added a fresh-process offline worker and parent runner. The worker receives no label,
  target, sample ID, scenario, corpus/public/prior path, evaluator, or result. The parent
  owns the official evaluator and emits aggregate metrics, exact totals, gates, resources,
  and hashes without per-session records.
- The focused P8 suite passes `63/63` and the complete project suite passes `387/387`,
  including lifecycle, control/shadow equality, catalog support, source isolation, exact
  metric reconstruction, resource/repeat gates, confirmation non-disclosure, and artifact
  redaction.
- Independent review found and fixed two pre-freeze validity gaps: catalog evidence in the
  builder now uses the same `>=0.90` confidence threshold as runtime, and the untested
  `feature` execution slot was removed. The worker now receives a sanitized mechanism spec,
  the source lock covers canonical transitive paths, and the evaluator has a direct official-
  blob gate. A fresh direct `Agent(coverage/off/fast)` worker is the formal B00 reference;
  C00 must match its complete functional result, ordered response trace, and exact totals on
  every opened split.
- The preregistration source lock records both the raw runtime file SHA-256 and the
  Git-filtered blob SHA-1. This locks the exact imported bytes while correctly proving
  commit membership for legitimate Windows CRLF checkouts; a direct raw-byte comparison
  against Git's normalized LF blob was rejected during dry lock generation before any
  lock or metric existed.
- P8 source/spec was frozen and pushed in `8f5a5e8`; the Git-filter-aware source-lock fix
  was frozen and pushed in `1b1edd5`. The separately committed preregistration lock is
  `2847459`; it binds source commit `1b1edd5`, 15 canonical source files, the official
  evaluator blob, catalog/public/prior identities, both P8 splits, and metadata. Lock
  SHA-256 is `357f5b81897e25b80830ff46d0fb1efcadbf2f25ac31afa05fe837246f9bce7d`.
- The only formal P8 selection run completed from clean pushed commit `2847459`. The fresh
  B00 served reference and C00 control matched exactly. On this local catalog-derived,
  product-disjoint stress split, C00 recorded 46/200 hits, HR@10 `0.230000`, MRR
  `0.078494`, MTTC `8.980000`, and TechnicalScore `0.178948`; R01 recorded 54/200 hits,
  HR@10 `0.270000`, MRR `0.113218`, MTTC `8.615000`, and TechnicalScore `0.216665`.
  R01 produced eight miss-to-hit changes, zero hit-to-miss changes, 11 earlier hits, and
  31 rank improvements; all four scenario hit-rate non-regression gates passed.
- R01 was nevertheless rejected by the pre-registered resource gates: wall ratio
  `1.303476 > 1.30`, response-P95 ratio `1.836056 > 1.30`, and peak-RSS ratio
  `1.261027 > 1.20`. All contract, integrity, network, exception, activation, and quality
  gates passed. Because the active arm was ineligible, repeat was not attempted,
  confirmation remained unopened, and no released-public evaluation was run. The decision
  is `retain_p8_c00`; `starter/agent.py` and the served public score remain unchanged.
- The ignored aggregate artifact is
  `experiments/p8_explicit_negative_evaluation.json`, SHA-256
  `0b29d13c59796582385bdec32c877a8de2518ee7464b0f96709a71ef139d4670`.
  It contains no per-session records or target/sample identifiers. These P8 numbers are a
  synthetic stress-test result, not an official public score or a private-800 estimate.

## Current public evaluation

These metrics are the last verified served-public checkpoint. The released public set was
not rerun for P5-P9, and P9 did not modify the served Agent.

| Scope | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 0.925000 | 0.586057 | 3.162500 |
| Browsing | 80 | 0.987500 | 0.603224 | 2.925000 |
| Intent Override | 30 | 0.900000 | 0.655265 | 4.666667 |
| Boundary | 10 | 0.900000 | 0.643452 | 4.000000 |
| Overall | 200 | 0.945000 | 0.606175 | 3.335000 |

Overall Efficiency is `0.766500`; recommended TechnicalScore is `0.807652`; prompt and
completion token usage are both zero.

Compared with the explicit current-tree weighted-RRF control, promoted coverage changes
HR@10 by `+0.005000`, MRR by `+0.000917`, MTTC by `-0.040000`, and TechnicalScore by
`+0.003575`. The paired result has zero hit-to-miss and one miss-to-hit change.

The pre-integration Workbench checkpoint reproduced the official weak baseline at HR@10 `0.125`, MRR `0.068034`, MTTC `9.81`, Efficiency `0.119`, and TechnicalScore `0.10671`. The current gain therefore comes from the stateful sparse Agent integration, not from the browser observer.

These are public-development metrics, not a claim about the private 800 sessions.

## P4 coverage-cascade promotion

The frozen product-disjoint matrix recorded 14 raw non-control variants. A semantic
activation audit found R12's only apparent activation was caused by parsing a head-
circumference range (`21.25inch-25inch`) as a price. After rejecting that false
activation, 13 genuinely independent effective designs remain, still above the required
minimum of ten. A hygiene-only rerun on the same frozen corpus records zero R12
activations, zero output changes, and exact control metrics; its ignored artifact SHA-256
is `6428a2f4049f0b17dc7d9d6287716803aee596ff2e6d383ed625d86e84a7324f`.
This confirms classification without rerunning winner selection. The raw matrix artifact
is retained unchanged.

R08 was the sole eligible selection winner and is now implemented in
`starter/coverage.py` and served by `starter.agent.Agent` in the default
`coverage + rerank off` configuration. It counts distinct visible query terms matched
across the same catalog fields used by the frozen experiment, sorts by descending
coverage, and preserves weighted-RRF fused rank on ties.

The actual served Agent was independently run on canonical plus all eight registered
phrase suites. Every complete result hash equals the frozen winner; canonical HR@10 is
`0.945000`, MRR `0.606175`, MTTC `3.335000`, and TechnicalScore `0.807652`. The strict
response contract, no-key execution, two-run functional determinism, public/phrase
robustness, and resource measurement checks pass. A reference bridge additionally
proves exact complete response traces and broad/strict/fused/final route equality. The
combined artifact `experiments/p4_promoted_verification.json` has SHA-256
`8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.

The original architecture artifacts remain the selection evidence. Because promotion
changed `architecture_lab.py` to pin the old control and share the coverage helper, its
post-promotion working-tree bytes are no longer identical to the selection commit. The
old unchanged-source gate is therefore legacy/frozen evidence; the served implementation
is validated by `scripts/verify_promoted_agent.py`, not by pretending the promoted file
never changed. None of this is evidence about the organizer-private 800 sessions.

## P2 shortlist-reranker evaluation

P2 adds an explicitly gated `off / shadow / active` rerank mode. The rerank default remains
`off`; neither the released evaluator nor the normal Agent path activates an unproven
scorer.

- `off`: the complete 200-session result strictly matches frozen P1, including every
  session row and list order.
- `shadow`: computes normalized attributes and Top-50 scores but serves the original
  fused order. Its complete evaluator JSON strictly matches `off`.
- `active`: serves the experimental reranked order. The first frozen-weight run was
  rejected by the public gate.

| Mode | HR@10 | MRR | MTTC | TechnicalScore | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| off / frozen P1 | 0.940000 | 0.605258 | 3.375000 | 0.804077 | retained default |
| shadow | 0.940000 | 0.605258 | 3.375000 | 0.804077 | diagnostic only |
| active v1 | 0.930000 | 0.599974 | 3.430000 | 0.796392 | rejected |

Active v1 caused two baseline hit-to-miss regressions and reduced Buying HR from
`0.925` to `0.900`; it produced no compensating overall gain. Post-hoc diagnosis showed
that incomplete attribute coverage can incorrectly promote products with explicit
metadata above otherwise strong sparse matches. For example, one target's cotton/color
evidence existed only in the catalog description, which the conservative v1 extractor
does not treat as normalized attribute evidence. This is recorded as a failed
experiment, not presented as an improvement. Because the public activation gate failed,
active v1 was not advanced to the more expensive generalization and resource gates.

The preliminary P2 observation already suggested a time regression. The later controlled
P3 two-run artifact, recorded below against the final current source, confirms that
shadow exceeds the planned `1.5x` time gate.

## P2 v2 Top-10-member-safe control

The next target-blind control computes the same Top-50 score diagnostics but permits
movement only inside the original Fused Top 10. It preserves the Top-10 member set and
the complete order below rank 10. Adjacent candidates may cross only when both expose
the same requested-slot coverage signature. This removes the v1 hit-to-miss failure mode
by construction, but it does not bound rank displacement or guarantee an MRR gain.

The strategy was selected on the fixed 200-session product-disjoint corpus before any
new public gate:

| Derived canonical | HR@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Frozen P1 | 0.935000 | 0.630183 | 3.185000 | 0.812855 |
| P2 v2 control | 0.935000 | 0.624960 | 3.185000 | 0.811288 |

The control produced 6 best-rank improvements, 7 regressions, and zero hit-to-miss or
miss-to-hit changes. MRR fell by `0.005223` and TechnicalScore by `0.001567`, so v2 was
rejected without using the public set as a tuning loop and without running an active
resource gate. Reranking `off` remains the default.

## P3 slot and clarification shadows

`starter/slot_ledger.py` adds an auditable, target-blind normalized constraint history.
Each immutable record contains slot, normalized value, polarity, hardness, source,
confidence, source turn, state version, and an `active`, `superseded`, or `deleted`
lifecycle. Selective changes retire only the removed constraint; a no-preference event
deletes the named slot in the shadow view; explicit later evidence can reopen it. A
later, locally scoped hard restatement supersedes the earlier soft record without
upgrading unrelated values in a contrast clause. The ledger is diagnostic and does not
yet compile retrieval queries.

`starter/clarification.py` ranks unanswered attributes over the Fused Top-50 normalized
product views using:

```text
normalized information gain * catalog coverage * answerability - turn cost
```

Known, asked, exhausted, pending, category, and active-ledger attributes are omitted.
Each candidate contributes one primary value so multi-value combinations do not create
artificial entropy. Brand and feature are cardinality-penalized rather than rewarded for
raw long-tail entropy. Catalog prices are preserved in a shadow-only metadata side table,
so budget buckets can be diagnosed when price coverage exists. Turn 10 can expose the
evidence but never selects another question. The selected QuestionValue is exposed in
trace and Workbench beside the actual fixed-order question, but it does not change
`ask_attribute` or recommendations.

After the final current-tree source freeze, a fresh `off` run and a fresh `shadow` run both produced
HR@10 `0.940000`, MRR `0.605258`, MTTC `3.375000`, and TechnicalScore `0.804077`.
Their complete evaluator JSONs strictly match each other and frozen P1. The run manifests
include Agent, attributes, reranker, slot-ledger, clarification, evaluator, catalog, and
public-set hashes.

The same final source produced identical canonical session results on the frozen
public-target-disjoint corpus: HR@10 `0.935000`, MRR `0.630183`, MTTC `3.185000`, and
TechnicalScore `0.812855`. Both corpora therefore pass the non-interference gate; this
does not establish that the shadow question policy is better.

The controlled two-run resource audit passed determinism in both modes but failed the
planned `1.5x` activation budget:

| Mode | Mean total | Mean evaluator | Mean respond P95 | Mean peak RSS |
| --- | ---: | ---: | ---: | ---: |
| off | 25.436 s | 22.090 s | 71.156 ms | 369.7 MiB |
| shadow | 51.145 s | 47.742 s | 133.037 ms | 434.6 MiB |

Shadow is `2.01x` the off total wall time, `1.87x` the respond P95, and `1.18x` the
mean peak RSS. It remains a development diagnostic. The in-memory ranking-diagnostic
cache is bounded to 128 sessions to prevent unbounded Top-50 breakdown accumulation.

## Implemented Agent behavior

### Session and lifecycle

`starter/agent.py` now implements a per-session `SessionState` containing:

- aggregate profile copy;
- active category;
- active and excluded retrieval terms;
- known, asked, and exhausted attribute classes;
- pending clarification attribute and originating turn;
- per-turn terms and attribute classes;
- version, version anchor, and override count;
- the fast-policy `prefer_other_next` event.
- an auditable normalized shadow slot ledger.

`reset` replaces prior state for that session. `respond` validates turn 1–10 and positive `top_k`, updates state, ranks products, selects at most one allowed clarification attribute, and returns at most 10 catalog-backed recommendation objects. `drop_session` releases development replay/Lab state.

The SQLite connection uses `check_same_thread=False`; state and query operations are protected by an `RLock` for the multi-threaded local Workbench.

### Parsing and state transitions

Implemented deterministic parsing now produces a target-blind `ParsedTurn` and includes:

- anchored natural shopping openers such as looking/searching/shopping, need/want, show/find, and help-find;
- separation of category text, vague browsing suffixes, and actual constraint fragments;
- material, color, size, style, use-case, and budget class detection;
- conservative `not`, `no`, and `without` negative-term extraction, including `not too X` while excluding false negations such as `not only`, `not quite`, and `not sure`;
- explicit and pending-context no-preference/exhausted attribute handling;
- explicit ignore/disregard/forget, change-mind, no-longer, switch/replace, and context-bound `instead` detection;
- explicit `old -> new` spans are replaced selectively while vague `ignore earlier` events retain the auditable version-anchor behavior;
- loose `I need/want/show me` category openers establish the first goal only; later short color/material replies remain constraints, while product-head spans support explicit category switches;
- retry detection separated from negative constraints;
- repeated override version-anchor movement;
- category-goal changes that clear the previous goal's term and question lifecycle.

A plain sentence such as `Actually, cotton sounds fine` no longer triggers an override solely because it contains `actually`.

A selected clarification remains pending until the next ordinary user response. If an evaluator Override interrupts the expected answer, the pending attribute is released instead of being permanently marked asked; an interrupted `other` fallback also restores its disclosure preference.

### Sparse retrieval and fusion

The Agent builds one in-memory SQLite FTS5 catalog index over title, categories, features, details, store, and description.

Each turn compiles the current active state into two retrieval routes:

- Broad: quoted terms joined by OR, field-weighted BM25, Top 120.
- Strict: up to 16 quoted terms joined by AND, field-weighted BM25, Top 80.

The routes are fused deterministically with:

```text
score(d) = I_b(d) / (60 + broad_rank)
         + 1.8 * I_s(d) / (20 + strict_rank)
```

Ties use broad rank and then `parent_asin`. The promoted `coverage` retrieval mode loads
title, categories, features, details, store, and description for the fused candidate set,
counts distinct active query terms found in those visible catalog fields, sorts by
descending coverage, and preserves fused rank on ties. The response returns the first
`min(top_k, 10)` IDs from the explicit `final` route. With default `coverage + off`,
`fused` remains the control order, `reranked` remains equal to fused, and `final` is the
coverage order. Explicit `control + off` leaves final equal to fused.

### Normalized attributes and shortlist reranking

`starter/attributes.py` builds immutable, target-blind product and visible-conversation
views. The first frozen schema normalizes category, audience, material, color, closure,
style, use case, size, width, brand, price, and atomic feature phrases; records source,
confidence, and raw evidence; filters numeric/generic catalog noise; and uses no public
labels, sample IDs, target IDs, profile priors, network calls, or evaluator imports. Its
registry SHA-256 is
`1d85fc42f49fd9374238d98b8feaeab8d76269b0987740256fe60e666757d2ca`.

`starter/reranker.py` is a deterministic pure scorer over the fused Top 50. It exposes
RRF prior, category consistency, positive slot match, exact feature match, negative
violation, total score, and matched evidence. Missing values remain unknown rather than
violations. Scoring diagnostics cover Top 50; active v2 can move only members of the
original Top 10 within equal requested-slot coverage groups and preserves the full order
from rank 11 onward. The mode is rejected and off remains the default.

The Agent exposes five auditable routes: `broad`, `strict`, `fused`, `reranked`, and
`final`. In explicit control mode, `off` skips attribute scoring, `shadow` computes it
without changing output, and `active` uses it only when explicitly requested. Coverage
is restricted to rerank off to prevent an ungated composition. A bounded 10,000-view LRU
cache avoids re-extracting common shortlist products. Target-blind debug diagnostics
expose component and coverage evidence; the Observer joins public targets only after
`respond` returns.

### Clarification policy

Three explicit policies are supported:

- `fast` (default): after a no-preference reply, prefer `other` to obtain the remaining disclosed constraints quickly;
- `boundary`: use that shortcut only for a direct Boundary-style reply;
- `conservative`: continue through the fixed allowed-attribute order.

Known, asked, and exhausted attributes are not re-asked. Turn 10 always returns `ask_attribute=null`.

The selected policy can be passed to `Agent` or set with `TECHJAM_QUESTION_POLICY`. Workbench experiment manifests now record it because it directly changes results.

### Optional LLM boundary

The default Agent does not import, construct, or call an LLM client. An explicitly injected compatible client is used only to consume and report token usage; it does not currently parse intent, retrieve, rerank, or write response prose.

`utils/llm_client.py` remains available for measured future experiments and is covered by
configuration, JSON-response, error, and usage tests. Core execution uses the stdlib-only
`requirements.txt`; optional OpenAI and dotenv packages are isolated in
`requirements-llm.txt`. Agent/evaluator imports were verified under `python -S` without
site packages.

## Implemented Agent Workbench

The local Workbench is a loopback-only development control plane, not part of official scoring.

### Startup and pages

- `Start Observer.vbs`: hidden `pythonw.exe` launch using the existing `tiktok` environment.
- `Start Observer.cmd` and `python -m observer.launcher`: troubleshooting fallbacks.
- Overview: runtime, Git, source fingerprint, data/hash/index health, metrics, and truthful algorithm registry.
- Session Diagnostics: deterministic public replay, actual Agent events, output validation, and post-hoc score diagnosis.
- Catalog & Index: 50k catalog browsing, field-weighted BM25 search, and raw product JSON.
- Runs & Experiments: fixed test/evaluator/generalization jobs, progress, logs, cancellation, metrics, versioned manifests, and target-blind cross-session shadow-policy summaries.
- Lab: target-free calls to the real `reset/respond` interface with opaque session IDs.
- Documents: read-only allowlisted project documentation and source.

### Trace integration

The Agent can emit optional versioned, target-blind events for:

```text
session -> parse -> retrieval -> state -> policy -> output
```

Retrieval events expose broad/strict/fused/reranked/final counts, the weighted-RRF
formula, retrieval/rerank modes, coverage schema and matched-term evidence, and
raw-fused/reranked/final Top-10 evidence. State events expose only information derived
from the profile and conversation.

For a public replay, the Observer calls `Agent.respond` with a random UUID session. Only
after the response does it compare the target with target-blind route IDs and component
diagnostics. It records broad, strict, fused, reranked, and final ranks; `final` is the
actual output route. Target, scenario, intent card, behavior, prior result, and public
sample ID are never passed into Agent decision features.

Completed replays and evicted Lab sessions release Agent and recorder state.

### Local control-plane safety

- loopback bind only;
- project fingerprint check before reusing port 8765;
- per-process API control token;
- Host, Origin, and browser-site checks;
- JSON-only mutation bodies;
- CSP and frame protections;
- fixed allowlisted test/evaluation/Lab/shutdown controls;
- no arbitrary shell or filesystem browser;
- loaded-vs-disk Agent/coverage/attributes/reranker/slot-ledger/clarification/
  shadow-analysis/evaluator/generalization sources plus catalog/public-set fingerprints,
  with stale-runtime blocking for every replay, evaluation, generalization, and Lab call;
- evaluation provenance captured before the background job and rechecked before artifact finalization, so a manifest cannot mix loaded code/data with later disk hashes.

The Workbench must not be publicly deployed or connected to private final labels.

## P1 generalization and reliability gate

`scripts/evaluate_generalization.py` adds a deterministic, target-blind robustness runner without changing the released evaluator. Its `PerturbedAgent` wrapper transforms only the visible `user_message` before delegating to `Agent.respond`; it is never given sample ID, scenario, target ID, intent card, or prior result.

The frozen phrase registry contains independent development, challenge, and audit wording for shopping openers, requirement disclosures, no-preference responses, overrides, and retry feedback. It records a hash over every regex/replacement and suite composition, applied-rule coverage/counts, examples, per-suite metrics, deltas, and paired session changes using the official per-session score contribution. A released-public suite fails instead of reporting robustness if any selected rule transforms zero messages. The current registry SHA-256 is `7ebc55c38f3389da2d2d01f549763c2c6b39908f89b60095fafb2c1964cc940b`.

Released-public phrase results before and after P1:

| Phrase suite | Before HR@10 | Before Score | After HR@10 | After Score |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 0.940000 | 0.803977 | 0.940000 | 0.804077 |
| Combined development | 0.845000 | 0.700243 | 0.940000 | 0.804077 |
| Combined challenge | 0.835000 | 0.687278 | 0.940000 | 0.804077 |
| Combined audit | not used for the initial baseline | not used for the initial baseline | 0.940000 | 0.804077 |

After P1, every individual suite plus the combined development, challenge, and audit suites has the same released-public HR@10, MRR, MTTC, and TechnicalScore as canonical. The all-suite robust hit count is 188/200 (`0.940000`). This demonstrates resilience to these frozen equivalent phrasings; it does not prove unrestricted natural-language understanding.

The same runner can build 200 deterministic catalog-derived sessions after excluding every released-public target. The fixed seed `track4-p1-product-disjoint-v1` produces SHA-256 `38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae`, 200 unique targets, zero public-target overlap, and an 80 Buying / 80 Browsing / 30 Intent Override / 10 Boundary split.

| Derived suite | Before HR@10 | Before Score | After HR@10 | After Score |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 0.935000 | 0.813096 | 0.935000 | 0.812855 |
| Combined development | 0.855000 | 0.740916 | 0.935000 | 0.812855 |
| Combined challenge | 0.835000 | 0.727386 | 0.935000 | 0.812855 |
| Combined audit | not used for the initial baseline | not used for the initial baseline | 0.935000 | 0.812855 |

The pending-question change slightly lowers derived canonical Score by `0.000241`: derived Buying HR changes from `0.950000` to `0.937500`, while Intent Override HR improves from `0.966667` to `1.000000` and its MTTC from `3.966667` to `3.800000`. This mixed scenario result is recorded rather than hidden. The derived corpus is a local metadata-based stress set, not organizer private data, and is not a prediction of private evaluation performance.

The Workbench exposes this fixed run as **运行泛化压力测试** and `POST /api/jobs/generalization`. Evaluation and generalization jobs are mutually exclusive because both build repeated 50,000-product in-memory indexes. The versioned artifact records Git state, source/input hashes, corpus metadata, phrase transforms, metrics, and robustness summaries.

## Strict result verification

`scripts/compare_results.py` supports both metric reporting and strict verification.

```powershell
python scripts/compare_results.py run_a.json run_b.json
python scripts/compare_results.py --assert-equal run_a.json run_b.json
```

Strict mode recursively compares the complete parsed objects, including key presence, list order, scenario metrics, usage, and every session row. Any semantic difference exits with code 1 and prints the first JSON paths that differ. Whitespace, indentation, CRLF, and LF formatting differences do not fail.

This replaces the handoff comparator behavior that printed aggregate deltas but always exited successfully.

## Verification completed

- The P4 promotion checkpoint passed `153/153` Python unit/integration tests after adding
  the P4 architecture lab, official-asset integrity, contract, lifecycle, budget,
  promotion bridge, R12 hygiene, and Workbench retrieval-mode coverage tests.
- Agent tests cover accumulation, natural openers/requirements/no-preference, pending-question interruption, category changes, negative phrases and false negations, false override prevention, first/repeated/selective overrides, Boundary exhaustion, question policies, five ranking routes, mode safety, catalog-price shadow ingestion, bounded diagnostic memory, output cap/final turn, optional usage, and target-blind trace/component diagnostics.
- Attribute/reranker/ledger/QuestionValue tests cover normalization boundaries, immutable provenance, unknown values, source confidence, noise removal, scorer arithmetic, negative penalties, deterministic ties, immutable fused input, Top-10 member and tail safety, lifecycle retirement/hard restatement, multi-value entropy control, final-turn suppression, and candidate-price coverage.
- Generalization tests cover phrase payload preservation, adapter input isolation, deterministic stratified public-target-disjoint generation, and rerank-mode propagation.
- Comparator tests cover formatting/line-ending equality, session-level mismatch, missing keys/list order, and invalid JSON.
- Existing evaluator, LLM client, Workbench replay, catalog/Lab/background evaluation,
  HTTP token/cross-site, and exclusive-listener tests pass. Observer tests also cover
  retrieval-mode propagation, coverage evidence/provenance, fused-versus-final route
  semantics, source fingerprints/schema metadata, cross-session target-blind shadow
  artifacts, visible ledger/QuestionValue components, and stale-source rejection.
- `node --check observer/static/app.js` passes.
- The complete 200-session evaluator completed successfully with no LLM environment variables.
- The final direct public evaluator result strictly matches the canonical result produced through the target-blind robustness wrapper.
- A headless Chrome smoke test rendered the live loopback Workbench against 50,000 indexed products and 200 sessions; Overview, state/fusion pipeline, public Trace, two-turn Lab state, and background Tests were exercised successfully with `restart_required=false`.

## Provenance and audit corrections

The v0.6 release material was independently audited before integration:

- outer ZIP hash and 73/73 declared file hashes matched;
- source.zip, bundle target tree, and the shared project source files matched byte-for-byte;
- bundle history is complete and contains official `3407835` as an ancestor;
- the patch exactly represents `367f1bf -> 89ef66c`, not `3407835 -> 89ef66c`;
- historical participant commit `914879c` added an `AgentBase` Protocol/type annotation only; the current file has been restored and has no diff from official upstream blob `7c808347b31ef3121a9cbc4810ac3eb325f950ba`;
- the packaged `original_baseline_reference/starter_agent.py` is a participant optional-client baseline, not the pristine official starter.

The current repository may describe the evaluator as restored to the official upstream
file. Historical audit reports must still distinguish the earlier type-only wrapper.

## Current limitations

1. The retrieval source of truth remains the term/turn state. A normalized slot ledger now exists in shadow, but it does not yet compile structured filters or retrieval queries. Explicit old→new spans are selective, while a vague `ignore earlier` override can still remove unrelated preferences introduced in the same anchor turn.
2. The parser now passes three frozen equivalent-phrase families, but it remains deterministic and English-pattern based; this is not unrestricted semantic parsing.
3. The default fast policy benefits from the public simulator's `other` disclosure behavior. This is protocol adaptation, not direct label leakage, but it creates public-strategy overfitting risk.
4. The product-disjoint corpus is derived from the same frozen catalog and official simulator, so it tests target overlap and wording robustness but is not an independent approximation of the private distribution.
5. Served clarification still uses a fixed order. Candidate-aware QuestionValue exists only in shadow; its Top-50 candidates are equally weighted, missing values are not modeled as a bucket, and its constants have not passed an activation gate.
6. No explicit Buying/Browsing router is implemented; hidden scenario labels are never available to the Agent.
7. Profile data is stored but not used for personalization.
8. The served path has no structured hard filter/relaxation execution, dense retrieval,
   learned reranker, or semantic reranker. P8 implements explicit-negative partitioning only
   in an isolated experiment; despite selection quality gains it failed the frozen resource
   gates and is not imported by the served Agent. Active rerank v1/v2 also remain disabled.
9. Budget buckets are visible in QuestionValue shadow, but a user budget such as `under $50` does not yet become a numeric range filter or ranking constraint. Budget questioning cannot be activated until that downstream path exists.
10. The controlled P3 shadow resource audit is deterministic but fails the planned time gate at `2.01x` off total wall time, so shadow remains development-only.

The current served implementation should be described as **versioned stateful sparse
retrieval with weighted-RRF candidate fusion, promoted visible-query-term coverage
ordering, heuristic clarification, and a frozen Top-10-only P11 linear reranker, plus
normalized slot/attribute/rerank/QuestionValue diagnostics**, not as the complete
IntentGraph target architecture.

## Change history

### 2026-08-28 - P7 quadruple-disjoint corpus and semantic feasibility freeze

- Added a fourth deterministic corpus builder that hard-checks the official catalog
  SHA-256, public Git blob, canonical P1/P5/P6 sample hashes, the expected P7 output hash,
  catalog/prior counts, sample-ID families, all six prior pairwise target
  overlaps, four selected-target overlaps, scenario mix, and output-path safety before it
  writes. The ignored result has 200 unique targets, prior union 800, mix 80/80/30/10,
  every overlap zero, and SHA-256
  `bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546`.
- Fixed public identity checking before freeze to use the official LF-normalized Git blob
  `121dbec9c1368c81cd887d6959e62507512139c0`, avoiding false failure from Windows/Linux
  line-ending differences. Derived raw hashes are recorded only as reference metadata;
  canonical sample hashes are the cross-platform hard gate.
- Added an optional, pinned CPU semantic runtime manifest and a machine-readable model
  spec for MIT-licensed `BAAI/bge-small-en-v1.5` revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. It freezes all eleven asset hashes,
  preprocessing, query/document schemas, CLS pooling, float32 normalization/indexing,
  exact tie ordering, and byte-for-byte sparse fallback boundaries. The config now also
  contains the normative eligibility, recall, timing, RSS, byte-scope, repeatability, tie,
  and decision definitions; 256 tokens is explicitly a resource cutoff below the model's
  512-token capability. OpenBLAS/OMP/MKL and ORT execution/optimization are explicitly
  pinned before imports. The upstream MIT notice is bundled under `third_party/`.
- P7 is pre-registered as C00 plus output-identical dense shadow only. It has no active
  recommendation candidate and cannot authorize public evaluation. Dense recall must
  rescue at least five sessions across two scenarios and pass strict local asset,
  initialization, P95, evaluation-time, RSS, determinism, and target-blindness gates
  before any hybrid admission design can begin on a fresh corpus.
- No P7 route metric was read while the corpus, model choice, runtime versions, constants,
  or gates were being frozen. The served `starter/agent.py` remains unchanged and does
  not import the optional semantic runtime.
- No P7 route metric or outcome is claimed in this entry. At the hardened preregistration
  checkpoint, the full suite passes `273/273`, P7 corpus/model-spec tests pass `15/15`,
  official participant assets pass `14/14`, and `pip check` reports no broken dependency.

### 2026-08-28 - P7 offline encoder and catalog-only index builder

- Added a standard-library-at-import semantic core. It validates the full model spec and
  model assets before dynamically importing the exact NumPy/tokenizers/ONNX Runtime stack,
  pins the pre-import environment, and constructs CPU-only sequential BGE inference with
  the frozen tokenizer, 256-token cutoff, CLS pooling, and float32 L2 normalization.
- Added the exact recursive catalog serializer and a validated `.npy` memory-mapped index
  loader. The loader hard-checks canonical model-spec identity, catalog identity, matrix
  and ASIN size/SHA, float32 shape/C ordering, LF-only unique ascending ASIN rows, and
  stable score-descending/ASIN-ascending search. Empty queries return no route without
  invoking the model.
- Added a catalog-only offline builder with no evaluator/session import. It validates the
  official catalog and all model/license assets, sorts 50,000 products by ASIN, encodes in
  frozen batches, verifies finite unit vectors, records preprocessing/runtime/resources,
  and publishes the matrix, ASIN file, and manifest through one same-filesystem rename.
  Existing output is never replaced and failures clean only the builder-owned temp path.
- The real ignored BGE assets passed a 32-document CPU smoke: shape `(32, 384)`, dtype
  float32, CPU provider only, and vector norms within float32 tolerance of one. This is a
  runtime compatibility check, not a P7 recall/resource result.
- Semantic/builder focused tests pass `20/20`; the full suite passes `293/293`, official
  participant assets remain `14/14`, and `pip check` is clean. `starter/agent.py` and the
  served R08 output path remain unchanged. No evaluator or P7 route metric was run/read.

### 2026-08-28 - P7 target-blind capture and isolated gate runner

- Added one `P7CaptureAgent` subclass for both C00 and S00. It freezes the actual served
  `coverage/off/fast` path, calls the parent ranking method once, copies the real
  Broad-120/Strict-80 routes, and computes Dense-120 only in shadow. Dense or capture
  exceptions are counted and cannot alter the response object returned by the sparse
  Agent. Empty queries never invoke the semantic model.
- Added a strict tracked-index-lock validator that runs before optional semantic imports.
  It bridges the build commit and source hashes, raw/canonical model spec, official catalog,
  manifest, float32 matrix, ordered ASINs, canonical-document digest, all model assets,
  bundled license, independently recomputed asset bytes, and build resource observation.
- Added a parent-simulator/child-Agent JSONL protocol with a separate minimal worker entry.
  Fresh workers receive only catalog and semantic bootstrap paths, a corpus ordinal, the
  official profile, and the currently visible message/turn/top-k. The child module and
  namespace contain no selection, target, sample ID, scenario, evaluator, or post-hoc
  surface. Lab and RPC response captures must be exactly equal before labels can be joined.
- Added frozen integrity, Dense@10/40/120 recall, rescue/scenario, cold-init, query P95,
  evaluation-wall, absolute-RSS, no-network, exception, and repeatability gates. Canonical
  response and dense-route hashes exclude PID, UUID, labels, timestamps, and durations.
  A fresh repeat worker is launched only when every initial gate passes.
- Formal execution is restricted to the default P7-only corpus and model directory, exact
  catalog/spec/index lock, a clean branch whose origin equals HEAD, and unchanged hashes for
  the evaluator, Agent dependency closure, model/license/index assets, and inputs before
  and after evaluation. Builder, semantic, and spec bytes must also match their Git blobs
  in the locked build commit. Windows RSS uses OS lifetime `PeakWorkingSetSize`; existing
  output is never overwritten and publication is atomic.
- P7 lab/worker/runner tests pass `29/29`; the complete suite passes `322/322`, source
  compilation passes, and `pip check` is clean. The 50,000-row production index lock and
  the sole formal P7 run remain pending at this checkpoint. No evaluator or P7 route metric
  was run/read, released-public was not evaluated, and `starter/agent.py` remains unchanged.

### 2026-08-28 - P7 production index and pre-metric CLI bootstrap correction

- Built the frozen catalog-only index once from semantic-source commit `09a911a`: 50,000
  finite unit-normalized float32 rows by 384 dimensions. Matrix SHA-256 is
  `84897381c106b909b9e3d44229187d12f23796f108cfec97904db1cbeeb2d407`;
  the 50,000 unique sorted-ASIN file SHA-256 is
  `3af465b23ff2d33614501472edf02d2953ccfc170d2fe3348d55cd51c8ef0d54`;
  the external manifest SHA-256 is
  `cca932a8b4d0a160e0a409ec6ce9cf3b68c99e3b95bddb911b9c7d83b67365ba`.
- The build took `7422.871804s`. Its sampled Windows working-set peak was
  `4,844,965,888` bytes from a `25,243,648`-byte baseline. Required model, index,
  manifest, ordered-ASIN, and license assets total `211,493,793` bytes. Independent
  verification recomputed the catalog documents, matrix properties, ASIN order, eleven
  model assets, license, byte scope, and atomic publication without reading session data.
- Added and pushed tracked lock `configs/p7_semantic_index_lock.json` in commit `b1a802b`.
  Both lock validators passed, including build-commit Git blobs and current bytes; the
  lock SHA-256 is
  `be9304a358aaa29de9337d56cc7d6c86bfdcf9a19fe694bc6291107aa444376b`.
- The first direct CLI invocation exited before importing the evaluator because Python
  placed `scripts/`, not the project root, on `sys.path`. It produced no result file and
  read no P7 metric. Added the project root at bootstrap plus an isolated `python -I`
  subprocess regression that imports the official evaluator from an arbitrary working
  directory. No corpus, model, query, route, response, metric, threshold, or gate changed.
- P7 focused tests now pass `30/30`; the complete suite passes `323/323`. The corrected
  CLI remains metric-free at this checkpoint, released-public remains untouched, and the
  served `coverage/off/fast` Agent remains unchanged.

### 2026-08-28 - P7 frozen BGE feasibility rejection

- Ran the only metric-bearing P7 study from clean/pushed commit `be29edf`. The ignored
  15,736-byte artifact SHA-256 is
  `c487f55e1d3ca3da93553eaf6d2782bac0d07925150b568dc8b73476b60c1b56`.
  Pre/post source snapshots match HEAD and origin exactly across all 34 code, evaluator,
  corpus, catalog, model, license, lock, and index files.
- C00 and S00 response hashes are exactly equal; all 6 alignment checks, all 33 integrity
  checks, worker isolation, output contract, catalog membership, zero exception, and zero
  network gates pass. The two workers each captured 583 reached non-empty turns; labels
  were joined only after they exited, and no per-target/sample/session identifier is in
  the artifact.
- Sparse Broad-120 union Strict-80 recalled 198/200 sessions. Dense recall was 53/200 at
  10, 75/200 at 40, and 115/200 at 120; sparse plus Dense-120 still recalled 198/200.
  Dense rescued zero sessions across zero scenario types, so all three recall gates fail.
- Asset bytes (`211,493,793`), cold initialization (`0.648087s`), query-plus-search P95
  (`30.6435ms`), RSS availability, network, and exception gates pass. S00/C00 evaluation
  wall ratio `1.552145` and absolute peak RSS ratio `2.759477` exceed their `1.50` limits.
- The frozen condition therefore did not launch the repeat worker and correctly returned
  `reject_p7_bge`. P7 will not be rerun or tuned, released-public was not evaluated, and
  `starter/agent.py` remains R08 `coverage/off/fast` with public HR `0.945000`, MRR
  `0.606175`, MTTC `3.335000`, and TechnicalScore `0.807652`.
- Because recall failed independently of resources, no smaller dense fallback was promoted.
  The planned next fresh-corpus preregistration became P8 explicit-negative execution; its
  completed frozen result and resource rejection are recorded at the top of this document.

### 2026-08-28 - P6 frozen adaptive-depth selection rejection

- Ran the P6 selection exactly once from clean, pushed pre-metric commit `873cbd2`.
  Preflight/postflight branch, commit, status, source hashes, and input hashes are exact.
  Artifact SHA-256 is
  `5c2608b874f53b42e687b10ca5be352f8926856a8160712b468f8896df5331bf`.
- C00 and R01 tied exactly at HR `0.905000`, MRR `0.581312`, MTTC `3.470000`,
  TechnicalScore `0.777494`, 181 hits, and official-contribution sum x25200 `3918567`.
  R01 made 193 deep queries and 44 safe rank-10 output changes, but changed zero target
  outcomes, ranks, hit turns, or official per-session contributions.
- Post-hoc pool evidence rejected the cutoff hypothesis independently of Top-10 scoring:
  base@120 and deep@240 both found targets on the same 144 eligible turns and recalled the
  same 126 sessions; deep-only recovery was zero turns and zero sessions.
- R01 evaluation time was `1.720x` the real served Agent and `1.710x` C00; response P95
  was `2.213x` served and `2.282x` C00; absolute peak RSS was `1.266x` served, while RSS
  increment was `1.327x` C00. It failed the frozen quality, recall, and resource gates.
- Decision: `retain_control_active_rejected`. Confirmation was correctly not attempted,
  released-public was not evaluated, `starter/agent.py` remains unchanged, and P6 will not
  be tuned or rerun on this corpus. The formal served public result remains the prior R08
  result documented above.
- A separate read-only result audit recomputed the artifact digest, all 15 source and five
  input hashes, 29 R01 gates, exact totals, resource ratios, worker metadata, and target-
  isolation boundary. It found no decision-level discrepancy and did not rerun evaluation.

### 2026-08-28 - P6 guarded adaptive-depth pre-metric implementation freeze

- Added a P6-only pure guard and experiment Agent. It repeats the exact broad FTS query
  only when the served Top-120 is saturated, validates an exact 120-item prefix at depth
  240, protects Top 9, and may replace only rank 10 with one target-blind newcomer whose
  visible-query coverage is strictly higher and whose fields match no excluded term.
- Added exact C00/S00/R01 control, shadow, and active semantics without importing P6 code
  into `starter/agent.py`. C00 remains response-identical to the served R08 Agent; S00
  computes diagnostics but returns C00 output; every failure path in R01 returns the full
  original order.
- Added a frozen runner that verifies official/P1/P5/P6 identities and target isolation,
  exact integer score contributions, response and route invariants, post-hoc session-level
  candidate-pool recall, override timing, and public-use boundaries. Labels are joined only
  after target-blind turn records have been copied from a closed Agent.
- Resource observations use four fresh workers for initial selection (actual served Agent,
  C00, S00, R01). Only a fully eligible R01 launches three fresh confirmation workers
  (served, C00, R01). Parent-issued nonces prove fresh results while allowing legal PID
  reuse. R01 is gated against both C00 and the real served Agent; a confirmation-worker
  failure is recorded as a deterministic retain-control decision instead of inviting a
  rerun.
- Independent pre-metric review found and fixed excluded-term ordering across Python hash
  seeds, the initial shared-process resource design, the instrumented-control-only resource
  comparison, PID-reuse false rejection, and confirmation-failure artifact loss. No P6
  metric or result artifact existed during these changes.
- The complete project suite passes `258/258`; the focused P6 suite passes `65/65`, source
  compilation and whitespace checks pass, and the served Agent remains unchanged pending
  the single frozen P6 selection run.

### 2026-08-28 - P6 triple-disjoint selection corpus freeze

- Added a deterministic P6 corpus builder that excludes the union of all released-public,
  frozen P1-derived, and frozen P5-derived targets. It hard-checks the P1 canonical hash,
  the P5 file hash, counts, sample-ID families, pairwise input disjointness, catalog
  membership, scenario mix, and output-path safety before writing anything.
- The ignored P6 corpus contains 200 unique targets with zero overlap against each of the
  three prior target sets, scenario mix 80/80/30/10, and canonical SHA-256
  `27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7`.
  This freezes a fresh local selection surface for the next pre-registered experiment;
  it remains catalog-derived stress data and is not evidence about the private 800.
- The complete suite passes `201/201` tests at this corpus-freeze checkpoint.
- Pre-registered the sole next mechanism before reading any P6 metric: a saturated
  broad-120 route may repeat the identical FTS query at depth 240, then a target-blind
  strict-coverage guard may admit at most one new item at rank 10 while preserving Top 9.
  The registry, constants, exact-quality comparisons, pool-recall audit, fallback rules,
  clean repeat, resource confirmation, and public-use boundary are frozen in
  `docs/algorithm_architecture_research.md`.

### 2026-08-28 - P5 independent selection corpus and pre-registered guarded PRF

- Added a deterministic P5 corpus builder that excludes both released-public and frozen
  P1-derived targets, validates the 50,000/200/200 inputs, preserves the official scenario
  mix, and writes cross-platform canonical hashes. The ignored frozen P5 corpus contains
  200 unique targets with both overlaps equal to zero; SHA-256 is
  `0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c`.
- Added isolated target-blind PRF primitives and a P5-only control/shadow/active Agent
  registry without changing the served Agent or the frozen P4 architecture evidence.
  Feedback uses cross-seed catalog-IDF evidence, an original-query-conjoined second FTS
  route, low-weight rank fusion, and a protected Top-9/one-newcomer safety boundary.
- Added a frozen P5 runner that hard-checks corpus identity and both exclusion sets,
  compares C00 with evaluator and ordered-response hashes from a complete run of the
  actual served coverage Agent, proves shadow output equality, enforces metric/scenario/
  hit-to-miss/runtime gates, and repeats only an
  eligible active candidate. Released-public rows are not evaluated by this runner.
- The protocol, constants, formulas, and rejection gates were committed before opening
  P5 metrics. The clean-tree run matched served/C00/shadow outputs exactly. Active R01
  changed 21 turn-level Top-10 tails but changed none of the 200 target outcomes, so its
  metrics tied control and its `30.962s` evaluation time was `1.574x` control. It failed
  both strict score-improvement and `1.30x` time gates, was rejected without a public
  run, and did not change the served Agent. Artifact SHA-256:
  `d0fce8879cd19f0853aeb632b56195a7496b690939581e8f4c731d4a0795d90f`.

### 2026-08-28 — P4 target-blind architecture search and promotion

- Added an experiment-only `ArchitectureAgent` registry with one exact control and 14
  materially different retrieval, fusion, constraint, state, diversification, budget,
  and routing candidates. Selection initially left `starter.agent.Agent` unchanged.
- Added a frozen product-disjoint matrix runner with strict response-contract validation,
  control-integrity failure, activation/output-change accounting, session/scenario gates,
  hit-to-miss rejection, separate raw/eligible winners, and deterministic confirmation.
- Contract-invalid or incomplete variants cannot count toward the ten-experiment
  requirement. R09 never backfills a known negative conflict; R11/R13 scope browsing
  evidence to the current goal version. The raw runner counted R12 after one output
  change, but semantic audit proved that change came from a measurement-to-price regex
  false positive. The corrected semantic-effective count is 13, not 14.
- Added preflight and postflight Git/source/input snapshots. A long run is discarded if
  any direct source, input, Git branch/commit, dirty state, or derived-path state changes.
  The artifact records all parsed invocation values and hashes direct Agent dependencies.
- Added the complete offline official-asset verifier and a rules-first audit. The local
  catalog/public/evaluator assets pass all row, schema, uniqueness, membership, scenario,
  Git-blob, and release-hash checks.
- The frozen clean-tree 200-session matrix recorded 14 mechanically effective,
  contract-clean non-control variants; 13 remain genuinely effective after the R12
  audit. `R08.coverage_cascade` is the sole eligible selection winner:
  HR `0.935→0.945`, MRR `0.630183→0.643516`, MTTC `3.185→3.115`, Score
  `0.812855→0.823255`, zero score regressions, and exact repeated functional output.
  That table is selection-corpus evidence only.
- Promoted R08 into the served Agent and independently verified the actual default Agent
  against the frozen/reference winner on all nine public phrase suites, complete response
  traces, broad/strict/fused/final routes, strict contract, determinism, resources, and
  no-key execution. The combined verification artifact SHA-256 is
  `8a72f81dc9290f40c17384de49167c0bdfe080dbcf80f063ebc3a0d601152ec7`.

### 2026-08-28 — P3 auditable slot and clarification shadows

- Added immutable normalized slot history with active/superseded/deleted lifecycles,
  scoped hard restatements, selective override retirement, and no-preference deletion.
- Added candidate-aware QuestionValue diagnostics, candidate-price ingestion, active-slot
  blocking, final-turn suppression, and bounded ranking-diagnostic memory without changing
  served questions or recommendations.
- Added full Workbench ledger/QuestionValue cards, cross-session target-blind policy
  artifacts, schema/source provenance, and stale-runtime coverage.
- Restored `evaluator/local_evaluator.py` to the official upstream Git blob and verified
  public and product-disjoint off/shadow functional equality.
- Expanded the suite from 94 to 114 tests. The two-run resource audit failed the shadow
  time gate, so P3 remains diagnostic and reranking remains off by default.

### 2026-08-28 — P2 normalized attributes and gated rerank (`586f3dd`, `4610480`)

- Added catalog-only normalized attribute views and visible-dialogue-only constraint
  evidence with frozen registries and provenance.
- Added deterministic Top-50 component scoring with immutable fused order, untouched
  tail, stable ties, and explicit off/shadow/active/final route semantics.
- Added target-blind component diagnostics, bounded attribute caching, experiment runners,
  five-route recall/resource audit, and Workbench visualization/source-stale protection.
- Expanded the suite from 63 to 94 tests and preserved the P1 result exactly in both off
  and shadow modes.
- Rejected active v1 after HR, MRR, MTTC, TechnicalScore, Buying HR, and preliminary
  resource evidence failed the activation gates; reranking remains off by default.

### 2026-08-28 — P1 generalization and intent-state reliability (`abae926`)

- Added frozen target-blind development/challenge/audit phrase suites and deterministic public-target-disjoint derived sessions.
- Recorded pre-change failures before expanding parser recognition, preserving a causal before/after comparison.
- Added `ParsedTurn`, broader but conservative phrase recognition, and category/constraint separation without changing retrieval weights.
- Added a pending-question lifecycle so Override messages do not silently consume unanswered clarification opportunities.
- Added a fixed Workbench robustness action, provenance manifests, progress/logs, experiment comparison, and source-stale protection.
- Expanded the test suite from 32 to 55 tests; public HR/MRR remain unchanged while overall MTTC improves by `0.005`.
- Rejected immediate activation of feature-first/candidate-aware clarification because a read-only experiment improved overall score but reduced Boundary HR from `0.9` to `0.8`; candidate evidence will be developed in shadow mode first.

### 2026-08-27 — v0.6 integration into Workbench (`5fed7a7`)

- Preserved the Workbench checkpoint in commit `f4e435b` on a new integration branch.
- Replaced the stateless current-message-only Agent with the audited versioned state, broad/strict sparse routes, weighted RRF, and question policies.
- Preserved and extended thread-safe target-blind trace events.
- Replaced the Observer's old current-message BM25 diagnosis with post-response broad/strict/fused route diagnosis.
- Added state lifecycle cleanup and policy fingerprints to experiment manifests.
- Updated Workbench pipeline/UI labels to reflect state and fusion without claiming dense/reranking layers.
- Added strict complete-result comparison and expanded the test suite from 16 to 32 tests.
- Reproduced the complete v0.6 public result exactly.

### 2026-08-27 — Agent Workbench checkpoint (`f4e435b`)

- Added the loopback browser control plane, one-click Windows launcher, target-free Lab, public replay diagnostics, catalog/index explorer, fixed background jobs, versioned experiment manifests, document viewer, source-stale guard, API security checks, and tests.
- The checkpoint intentionally retained the stateless weak BM25 Agent and reproduced TechnicalScore `0.10671` before the v0.6 integration.

### Earlier participant history

- `367f1bf`: recorded official upstream alignment on `pre`.
- `1496fec`: merged official `3407835` into the participant history.
- `8f9e64d`: reliable no-credential baseline and first Layer Observer.
- `914879c`: optional OpenAI-compatible client, usage tests, challenge notes, and the type-only evaluator Protocol wrapper.
- `2a6cc8e` and `9a35be5`: official participant-kit history shared byte-for-byte with upstream.

## Competition boundary

The Devpost Rules and event pages were checked on 2026-08-27 SGT. The submission window begins on 2026-08-29 at 12:00 SGT. This integration is a pre-window technical baseline and does not by itself prove the required significant post-start update. After the window opens, substantial code work must have clear commits, tests, results, and documentation.

TechnicalScore remains an objective input to Technical Execution rather than the complete judging result. Public metrics, architecture quality, feasibility, limitations, impact, and communication evidence all remain necessary.
