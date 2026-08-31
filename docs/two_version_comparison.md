# Deadline fusion A/B result

Both requested versions are implemented, independently committed, default-off, and tested on Public200 plus the same cached 2,000-session OOF cohort. Cached 2k is an offline OOF benchmark, not a private-set score.

| Version | 2k HR@10 | m→h | h→m | MRR | MTTC | Score | First-turn HR | Repeat slots | Recovery wall | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v2.12 incumbent | 0.9910 | — | — | 0.695795 | 2.8690 | 0.866858 | — | — | cached | global default |
| A: Fusion Core | 0.9025 | 3* | 180* | 0.525192 | 3.2850 | 0.763108 | 0.358 | 0 | 1778.19 s | No-Go |
| B: A + `other` | 0.9410 | 105† | 28† | 0.614587 | 2.6025 | 0.822826 | 0.358 | 0 | 1699.32 s | A/B score-max; below incumbent |

\* A transitions are versus v2.12. † B transitions are versus A. Direct B versus v2.12 is m→h 5, h→m 105, net −100.

| Public200 | HR@10 | MRR | MTTC | Score | First-turn HR | Exact repeat |
|---|---:|---:|---:|---:|---:|---|
| A | 0.955 | 0.658046 | 2.505 | 0.844814 | 0.475 | yes |
| B | **0.995** | **0.714518** | **1.930** | **0.893255** | 0.475 | yes |

## What the experiment shows

- B is unambiguously better than A: on cached 2k it rescues 105 A misses while harming 28 A hits (net +77), and every fold improves. On Public200 it adds +0.040 HR, +0.056472 MRR, and reduces MTTC by 0.575.
- The B gain comes entirely after turn one; A and B have identical first-turn HR. `other` activates in every 2k session (3,695 asks), so this is an evaluator-aligned interaction gain rather than a better first-page ranker.
- Directly stacking V1 FTS1000/scoring, the parent state parser, hard masks, P11/C100 union, and immediate unseen serving is not safe. A loses 177 net hits versus v2.12; B recovers 77 of them but still loses 100 net hits versus v2.12.
- Therefore B is the winner if the choice is strictly A versus B, but neither should replace the current v2.12 default. B remains a one-flag opt-in: `other_mode=active`.

## Highest-value A* ablation

Do not tune weights. Preserve the v2.12 first two pages byte-for-byte and admit the frozen teammate-V1 fusion only as an unseen exploration tail from page three. If one additional isolated ablation is affordable, disable the supported-slot hard-conflict guard while keeping unknown-neutral scoring; A recorded 853,915 conflict events, making that guard the clearest negative-interaction suspect. Apply B's unchanged `other` wrapper only after the safer A* core is frozen.

Artifacts: [Version A](version_a_fusion_core.json) · [Version B](version_b_fusion_other.json)
