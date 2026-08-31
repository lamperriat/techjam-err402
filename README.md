# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Agent

Python 3.10 or later is recommended. The current stateful sparse Agent uses only the Python standard library and does not require LLM credentials.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to improve the system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The official weak BM25 starter reference scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

The current integrated Agent implements versioned multi-turn term state, a target-blind
parsed-turn layer, pending-question lifecycle, explicit Override and Boundary handling,
broad/strict FTS5 retrieval, weighted RRF, and heuristic clarification. P4 promotes the
target-blind `R08.coverage_cascade`: with reranking off, the served default orders the
weighted-RRF candidate set by the number of distinct visible query terms matched in
catalog fields, preserving fused order on ties. P2/P3 normalized attributes, slot ledger,
constraint scorer, and QuestionValue components remain diagnostics; active rerank v1/v2
failed their gates. The verified served result is:

| Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 0.945000 | 0.606175 | 3.335000 | 0.766500 | 0.807652 |

These are local measurements on the organizer-released 200-session development set and
do not predict the organizer-private 800-session result. For canonical and each of eight
phrase suites, the actual served Agent exactly reproduces the corresponding frozen-winner
result. Against the explicit weighted-RRF control, coverage improves HR@10 by `0.005000`,
MRR by `0.000917`, MTTC by `0.040000` turns, and TechnicalScore by `0.003575`, with zero
hit-to-miss and one miss-to-hit change. The deterministic product-derived,
public-target-disjoint corpus is a
local stress tool, not organizer data or a hidden-score estimate.

The served default now adds the frozen P11 Top-10-only linear reranker after R08.
It may change order inside the existing Top 10, but it cannot add a product to that set
and therefore cannot improve Hit Rate@10. Under the served `coverage/off/fast` preset, set
`TECHJAM_P11_MODE=off` to recover the exact R08 response/ranking without rebuilding any
asset. Diagnostic traces still label P11 as disabled. This integration did not rerun the released
200-session set; the table above remains the last released-public R08 checkpoint.

Historical P3 validation showed shadow strictly output-equal to control/off on both the
public and frozen product-disjoint corpora and exposed five auditable routes:
`broad`, `strict`, `fused`, `reranked`, and `final`. Experimental active v1 scored HR@10
`0.93`, MRR `0.599974`, MTTC `3.43`, and TechnicalScore `0.796392`, so it failed the
activation gate and is deliberately not the default.

Run the fixed robustness gate from the browser Workbench or directly:

```bash
python3 scripts/evaluate_generalization.py --corpus both --suite default
```

Run a mode-controlled evaluator experiment with a separate provenance manifest:

```bash
python3 scripts/evaluate_agent.py --retrieval-mode control --rerank-mode shadow --output experiments/p2_shadow.json
```

Verify that the official catalog, public sessions, and evaluator are complete and
unchanged:

```bash
python3 scripts/verify_official_assets.py
```

Run the isolated P4 target-blind architecture matrix only from a clean committed tree:

```bash
python3 scripts/evaluate_architectures.py --variants all --confirm-top 3
```

This selected on the frozen public-target-disjoint local stress corpus, not on the
released-public labels. The matrix recorded 14 non-control candidates; a semantic audit
found R12's sole activation was a measurement-to-price regex false positive, so 13 are
counted as genuinely effective independent designs. R08 was the sole eligible winner and
has now been promoted into the default Agent after public, phrase, contract,
determinism, repeated resource-measurement, no-key, and reference-bridge checks. See
`docs/algorithm_architecture_research.md` for the protocol and evidence boundary.

The next isolated P5 experiment tested guarded session-local pseudo-relevance feedback
on a fresh 200-target corpus disjoint from both released-public and the P4 selection
targets. Run its frozen reproduction with:

```bash
python3 scripts/evaluate_p5.py
```

R01 made 21 guarded turn-level Top-10 tail changes but produced no target-outcome gain
and took `1.574x` the control evaluation time, so it failed its pre-registered score and
runtime gates. It remains experiment-only; the default Agent is still R08 coverage.

P6 was the next isolated experiment. It tested whether the unchanged broad FTS query is
occasionally truncated at Top 120: only a saturated route may be recomputed at depth 240,
and at most one catalog-text candidate with strictly higher visible-query coverage may
challenge rank 10. It generates no feedback terms and does not modify the served Agent
unless every pre-registered quality, safety, determinism, and resource gate passes. Its
fresh 200-target selection corpus is disjoint from released-public, P1, and P5 targets.
The sole frozen run changed rank 10 on 44 turns but changed no session outcome, and
deep@240 rescued zero targets beyond base@120. It also failed the pre-registered time,
P95, and memory gates, so R01 was rejected without a released-public run. The runner
remains in `scripts/evaluate_p6.py` as an auditable record, but this corpus must not be
rerun for tuning; a future selection requires a fresh disjoint corpus and a mechanism-
level different hypothesis. The default Agent remains R08 coverage.

P7 therefore changes mechanism rather than sparse constants. It freezes an MIT-licensed,
CPU-only BGE-small ONNX route on a fourth corpus disjoint from public/P1/P5/P6. P7 is
shadow recall feasibility only: it cannot change recommendations or trigger a public run.
The model revision, every local asset hash, preprocessing, exact-search order, runtime
versions, recall threshold, and resource limits are frozen in
`configs/p7_bge_small_en_v1_5.json` and `docs/algorithm_architecture_research.md`.
That machine-readable contract also fixes eligible-turn, rescue, byte, timing, RSS, tie,
and repeat-worker definitions; the model's upstream MIT notice is retained under
`third_party/`. The sole frozen P7 run has now completed and rejected BGE; the default
Agent remains R08 coverage.
The optional packages are isolated in `requirements-semantic.txt`; the default Agent
continues to use only the standard library.

The P7 offline semantic core, catalog-only index builder, target-blind C00/S00 capture
layer, and process-isolated gate runner are now implemented, but they remain experiment
infrastructure and are not imported by the served Agent. After installing the optional
pinned environment, build the ignored local index from the frozen catalog and
already-downloaded model assets with:

```powershell
python scripts/build_p7_semantic_index.py
```

The command validates every source/model hash, refuses to replace an existing output
directory, encodes products in ascending `parent_asin` order, and atomically publishes
`experiments/p7_index/embeddings.npy`, `parent_asins.txt`, and
`semantic-index.manifest.json`. It reads no session file or evaluation label. The index
is locked by `configs/p7_semantic_index_lock.json` but has not been admitted into
recommendation output because P7 failed its frozen recall and resource gates.

`starter/p7_lab.py` uses the same capture subclass for control and shadow. The shadow
computes Dense-120 after the served sparse routes but returns the untouched response
object. `scripts/evaluate_p7.py` keeps simulator labels in the parent; a separate minimal
`scripts/p7_worker.py` receives only semantic bootstrap paths, ordinary profile and
visible-turn inputs, and a corpus ordinal. Labels are joined only after both initial
workers have exited. Formal execution is refused until the built index has a tracked hash
lock and the complete preflight is clean and pushed. No P7 route or target metric was run
while implementing this infrastructure. The sole metric-bearing run used clean/pushed
commit `be29edf`; its ignored aggregate artifact SHA-256 is
`c487f55e1d3ca3da93553eaf6d2782bac0d07925150b568dc8b73476b60c1b56`.

P7 passed response alignment, all 33 integrity checks, cold initialization, query P95,
asset size, offline/network, and exception gates. Sparse Broad-120 union Strict-80 recalled
198/200 sessions; Dense-120 alone recalled 115/200 and the sparse-plus-dense union remained
198/200, so dense rescued zero sessions. It also used `1.552x` C00 evaluation time and
`2.759x` C00 absolute peak RSS, above both `1.50x` limits. The repeat worker therefore did
not run, BGE was rejected without a released-public evaluation, and this P7 corpus must not
be rerun or tuned.

The subsequent frozen P8 experiment isolated high-confidence explicit-negative execution.
On its 200-session catalog-derived selection stress split, R01 improved C00 HR from `0.23`
to `0.27`, MRR from `0.078494` to `0.113218`, and MTTC from `8.98` to `8.615`, with eight
miss-to-hit and zero hit-to-miss changes. It nevertheless failed all three pre-registered
resource ratios (wall `1.303476x`, response P95 `1.836056x`, peak RSS `1.261027x`), so the
frozen decision is `retain_p8_c00`. Repeat was not attempted, confirmation was never opened,
and released-public was not rerun. P8 therefore remains experiment-only and the served R08
Agent/public score is unchanged; the same P8 corpus will not be tuned or rerun.

P9 then tested the same frozen semantics with a compact catalog-only evidence sidecar on
two new target-disjoint 200-session splits. On selection, C00 to R01 changed
HR/MRR/MTTC/Score from `0.210000/0.065454/9.175000/0.161136` to
`0.250000/0.089877/8.785000/0.196263`; on confirmation, the initial runs changed them from
`0.185000/0.056688/9.330000/0.142906` to
`0.225000/0.084885/8.960000/0.178765`. Each split had eight miss-to-hit, zero hit-to-miss,
and no scenario hit-count regression. All frozen bootstrap, wall, response-P95, and RSS
ratios passed, and selection exact repeat passed. The formal decision is nevertheless
`retain_p9_c00`: confirmation B00/C00/S00 failed only a metric bridge check because the
official evaluator computes Score from six-decimal rounded aggregate metrics (`0.142906`),
while the bridge rounded the exact per-session contribution sum (`0.142907`). Confirmation
repeat was therefore not attempted and remains inconclusive. P9 was not rerun, released-
public was not evaluated, and no production promotion occurred. The served Agent remains
R08 `coverage/off/fast` with the public checkpoint above. P9's isolated Python audit
boundary is not an OS sandbox against hostile native code.

P11 corrected the future metric bridge with synthetic fixtures, then evaluated one fixed,
catalog-only `p11.top10-linear.v3` scorer on three new product-disjoint splits while
preserving the exact R08 Top-10 member set and complete tail. Source `639cf78` and the
separate lock-only commit `c6efa5f` were pushed before the sole formal run. Primary C00 to
R01 changed MRR/Score from `0.583204/0.792561` to `0.625488/0.805246`; uniform-tail changed
`0.607732/0.781520` to `0.618764/0.784829`; confirmation changed
`0.581927/0.809278` to `0.629524/0.823557`. HR and MTTC were unchanged on every split,
hit-to-miss was zero, primary and confirmation paired 95% CI lower bounds were positive,
fresh exact repeats passed, and all preregistered quality/resource/audit gates passed. The
formal experiment decision is `promote_p11_r01`; aggregate result SHA-256 is
`fe0f8820b22c07136db44fb3739809d22b8edc5d1125707c5b0523dec312b912`.
Scenario HR was preserved, but scenario MRR was not a frozen gate and regressed on the
uniform-tail intent-override and confirmation boundary slices; production integration
must keep that residual risk visible without retuning on the consumed evidence.
Released public was not run. The frozen candidate is now installed as a reversible served
layer: P11 defaults to `active`, may only permute R08's exact Top 10, and returns the complete
R08 order on any pre-response identity, feature, scoring, adapter, or boundary failure.
Shutdown failures are surfaced to the caller instead of being reported as successful.

Direct `Agent()` construction reads three optional experiment variables:
`TECHJAM_RETRIEVAL_MODE` (`coverage` or `control`),
`TECHJAM_RERANK_MODE` (`off`, `shadow`, or experimental `active`) and
`TECHJAM_QUESTION_POLICY` (`fast`, `boundary`, or `conservative`). It also reads
`TECHJAM_P11_MODE` (`off`, `control`, `shadow`, or `active`) and the optional
`TECHJAM_P11_SIDECAR_PATH`. Production and
official evaluation should clear inherited values or explicitly use `coverage`, `off`,
`fast`, and P11 `active`. With no overrides, `Agent()` serves R08 coverage with the older
P2 reranker off and the frozen P11 Top-10 reranker active.

## LLM Client Configuration

The OpenAI-compatible client is optional and is not constructed by the default offline Agent. Core evaluation uses only the Python standard library. Install the optional client dependencies only when developing or testing model-assisted features:

```bash
python3 -m pip install -r requirements-llm.txt
```

When explicitly constructing `utils.llm_client.LLMClient`, copy `.env.example` to `.env` and set `LLM_API_KEY`, `LLM_MODEL`, and, for a third-party OpenAI-compatible service, `LLM_BASE_URL`. The client sends
non-streaming chat-completion requests in JSON-object mode and records the
provider-reported prompt and completion token counts.

See `docs/development_workflow.md` for the project mental model, debugging funnel, experiment loop, and current improvement order.

## Agent Workbench

On Windows, double-click `Start Observer.vbs` to start the local Workbench without opening
a terminal. It uses the existing `tiktok` Conda environment, starts through
`pythonw.exe`, and opens `http://127.0.0.1:8765`. The Workbench launcher uses the served
`coverage + rerank off + P11 active` path and labels the weighted-RRF `fused` control separately
from the coverage-ordered `final` route. `Start Observer.cmd` and
`python -m observer.launcher` are troubleshooting fallbacks.

The Workbench provides:

- a read-only Fusion Studio for switching between teammate T0, strict Fusion A, and
  bounded-`other` Fusion B, including a step-through compute graph and the frozen
  Public200/local-2k-OOF result comparison;
- runtime, Git, data, catalog-hash, and FTS5-index health;
- an honest algorithm registry that distinguishes implemented, baseline-only, and planned layers;
- all 200 public sessions with actual Agent events and separately labelled post-hoc target diagnostics;
- frozen-catalog search and raw product inspection;
- browser controls for the complete public evaluator, fixed phrase/product-disjoint generalization gate, unit tests, progress, cancellation, logs, versioned local experiments, and cross-session target-blind shadow-policy analysis;
- a target-free manual Agent playground and read-only project document library;
- a safe in-page shutdown action.

The server refuses non-loopback bind addresses, rejects cross-site API requests, requires
an ephemeral local control token, and does not expose an arbitrary shell runner. It
fingerprints the loaded Agent/coverage/attributes/reranker/slot-ledger/clarification/
shadow-analysis/evaluator sources plus catalog/public-set inputs and blocks stale or
mixed-version runs until the Workbench is restarted. Every public replay gives the Agent
a fresh opaque session ID. The released simulator uses hidden target/scenario state only
to generate the permitted user messages; raw labels, intent cards, behavior, and prior
results are never passed into Agent decision features. Target-rank and scoring
annotations are joined after `Agent.respond`.

The Workbench displays the current versioned state, full slot-ledger lifecycle, all five
ranking routes, the control-fused versus served-coverage ordering, coverage provenance,
normalized attribute evidence, actual heuristic policy, candidate-aware shadow
components, and post-hoc target ranks. It continues to label slot-ledger-driven retrieval,
hard filtering/relaxation, numeric budget execution, dense retrieval, active candidate-
aware clarification, profile ranking, and semantic reranking as missing rather than
presenting roadmap layers as implemented. See `docs/agent_workbench.md` for the full
usage, API, isolation, and maintenance contract, and
`docs/teammate_ab_website_handoff.md` for the Fusion A/B architecture and evidence map.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable stateful sparse Agent
starter/coverage.py               promoted target-blind R08 ordering helper
starter/attributes.py             target-blind normalized product/constraint views
starter/reranker.py               deterministic gated Top-50 scorer
starter/slot_ledger.py             auditable normalized conversation shadow
starter/clarification.py           candidate-aware QuestionValue shadow
starter/p11_features.py            frozen catalog-only Top-10 feature/scoring contract
starter/p11_lab.py                 isolated B00/C00/S00/R01 P11 role layer
starter/p11_bridge.py              fail-closed served P11/R08 integration boundary
starter/assets/p11_features.sqlite frozen read-only 50k catalog feature sidecar
evaluator/local_evaluator.py      public-set simulator and scorer
scripts/compare_results.py        report and strict complete-result comparison
scripts/evaluate_agent.py         mode-controlled evaluator + provenance manifest
scripts/evaluate_generalization.py target-blind phrase/product-disjoint stress gate
scripts/official_metric_bridge.py exact future official aggregate-score reconstruction
scripts/build_p11_corpora.py      deterministic fresh P11 corpus builder
scripts/build_p11_sidecar.py      catalog-only compressed feature builder
scripts/build_p11_prereg_lock.py  fail-closed source/data/sidecar lock builder
scripts/evaluate_p11.py           one-shot fresh-process P11 formal runner
scripts/benchmark_resources.py    repeatability, RSS, latency, and route-recall audit
scripts/verify_promoted_agent.py  frozen-reference to served-Agent promotion bridge
observer/shadow_analysis.py        target-blind cross-session question diagnostics
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only judging runbooks and private-release procedures are not distributed in this participant repository.

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
