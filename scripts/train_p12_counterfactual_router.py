"""Train a deterministic, identifier-free P12 counterfactual admission router.

This command is deliberately bound to the already-closed train/explore P12-v1
artifact.  Product identifiers are used transiently to reconstruct atomic
rank-10 swaps and labels, but are forbidden from the serialized model/report.
No scenario, taxonomy, difficulty, ordinal, identifier, or hash-derived value is
ever included in the runtime feature matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT = "train_explore"
SESSION_COUNT = 2_000
TURN_COUNT = 10
PROPOSAL_COUNT = 40
TRACE_ROWS = SESSION_COUNT * TURN_COUNT
LAMBDA_HARM = 2.0
SEED = "track4-p12-counterfactual-router-v1"

AGGREGATE = ROOT / "experiments/fast_track/action_oracle_v1/train_explore-full-aggregate.json"
PROXY = ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
MANIFEST = ROOT / "experiments/fast_track/proxy_v1/manifest.json"
AGGREGATE_SHA256 = "11ad3e24aec412f6cb3b146d248aa7e2335a12dafccc20241eeb3301af97ca24"
PROXY_SHA256 = "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e"
MANIFEST_SHA256 = "8058973426bbc76ea856a5c48a61e91ed9e35ae44988a21a6d7b2195e88a7193"
CONFIG_SHA256 = "492b42c19708b0e528755cb00374b368afaf037ce2c8b1f5d33f52685de3638c"
TRACE_REGISTRY_SHA256 = "713a058b39c7fba7dc1775b48f852a0e28744e3702372e361733766c2de4bb8a"
COMBINED_TRACE_SHA256 = "f9a441220926aebf49f4b4d54a0f50df99f72ad4f8c0342e5528517503473e7b"
TRACE_SPECS = (
    ("train_explore-full-blind-shard-01-of-04.jsonl", "fac3bc71e6210d1a449de706d335cc5bb945d4d3daf01e8cbecbe15c0600bf1a"),
    ("train_explore-full-blind-shard-02-of-04.jsonl", "63812776b374fc0041871600a5781fbf1ea6046a3219334e7263338abbab6657"),
    ("train_explore-full-blind-shard-03-of-04.jsonl", "36a8706a2f8c51635e4feb4cde905a9789c7953ffeab25ae036ef824061f36b3"),
    ("train_explore-full-blind-shard-04-of-04.jsonl", "1f9968795ab5490968badcf82c39ec11bedd00f22797569dfec8c2ff3fb7ed99"),
)
OLD_ACTIONS = (
    "KEEP_R08",
    "KEEP_P11",
    "CANDIDATE_RERANK",
    "FROZEN_SEMANTIC_RERANK",
    "RESULT_AWARE_REWRITE_RETRIEVE",
    "ASK",
)
FEATURE_NAMES = (
    "turn_fraction",
    "proposal_rank_fraction",
    "proposal_distance_fraction",
    "structured_support",
    "structured_rank_utility",
    "semantic_support",
    "semantic_rank_utility",
    "proposal_support_fraction",
    "incumbent_structured_support",
    "incumbent_structured_rank_utility",
    "incumbent_semantic_support",
    "incumbent_semantic_rank_utility",
    "support_advantage_fraction",
    "p11_structured_overlap_fraction",
    "p11_semantic_overlap_fraction",
    "structured_semantic_overlap_fraction",
    "previous_pool_presence",
    "previous_rank_utility",
    "previous_top10_presence",
)
FEATURE_FORMULAS = (
    "one-based visible turn / 10",
    "one-based R08 candidate rank / 50",
    "(one-based R08 candidate rank - 10) / 40",
    "candidate is in structured Top10",
    "candidate structured Top10 rank utility; zero when absent",
    "candidate is in semantic Top10",
    "candidate semantic Top10 rank utility; zero when absent",
    "mean of candidate structured and semantic support flags",
    "P11 rank10 incumbent is in structured Top10",
    "incumbent structured Top10 rank utility; zero when absent",
    "P11 rank10 incumbent is in semantic Top10",
    "incumbent semantic Top10 rank utility; zero when absent",
    "candidate support mean minus incumbent support mean",
    "P11 and structured Top10 set intersection size / 10",
    "P11 and semantic Top10 set intersection size / 10",
    "structured and semantic Top10 set intersection size / 10",
    "candidate is in the immediately previous visible turn C50",
    "candidate previous-turn C50 rank utility; zero when absent",
    "candidate is in the immediately previous visible turn P11 Top10",
)
FORBIDDEN_PATH_WORDS = ("calibration", "selection", "confirmation", "sealed")
FORBIDDEN_KEYS = (
    "asin",
    "identifier",
    "ordinal",
    "sample_id",
    "target_id",
    "ground_truth",
    "scenario",
    "taxonomy",
    "difficulty",
)
ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$")


class RouterTrainingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TargetBlindRows:
    x: np.ndarray
    baseline_rankings: tuple[tuple[tuple[str, ...], ...], ...]
    proposals: tuple[tuple[tuple[str, ...], ...], ...]
    incumbents: tuple[tuple[str, ...], ...]
    feature_table_sha256: str


@dataclass(frozen=True, slots=True)
class Dataset:
    x: np.ndarray
    rescue: np.ndarray
    harm: np.ndarray
    targets: tuple[str, ...]
    eligible_from: np.ndarray
    session_weights: np.ndarray
    baseline_rankings: tuple[tuple[tuple[str, ...], ...], ...]
    proposals: tuple[tuple[tuple[str, ...], ...], ...]
    incumbents: tuple[tuple[str, ...], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink():
        raise RouterTrainingError(f"{label} must be a regular non-symlink file")
    attrs = getattr(resolved.stat(), "st_file_attributes", 0)
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise RouterTrainingError(f"{label} must not be a reparse point")
    return resolved


def _reject_forbidden_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(word in lowered for word in FORBIDDEN_PATH_WORDS):
        raise RouterTrainingError("forbidden held-out path requested")


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RouterTrainingError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise RouterTrainingError(f"{label} root must be an object")
    return value


def _validate_aggregate() -> dict[str, Any]:
    path = _regular_file(AGGREGATE, "aggregate")
    if _sha256(path) != AGGREGATE_SHA256:
        raise RouterTrainingError("aggregate SHA-256 mismatch")
    manifest = _regular_file(MANIFEST, "proxy manifest")
    if _sha256(manifest) != MANIFEST_SHA256:
        raise RouterTrainingError("proxy manifest SHA-256 mismatch")
    value = _strict_json(path, "aggregate")
    if value.get("schema_version") != "track4.p12-action-oracle-result.v1":
        raise RouterTrainingError("aggregate schema mismatch")
    protocol = value.get("protocol")
    provenance = value.get("provenance")
    worker = value.get("worker")
    action_oracle = value.get("action_oracle")
    if not all(isinstance(item, dict) for item in (protocol, provenance, worker, action_oracle)):
        raise RouterTrainingError("aggregate sections are missing")
    if protocol.get("split") != SPLIT or protocol.get("sample_count") != SESSION_COUNT:
        raise RouterTrainingError("aggregate split/sample mismatch")
    if not protocol.get("full_split") or protocol.get("fixed_turns_per_session") != TURN_COUNT:
        raise RouterTrainingError("aggregate is not the complete ten-turn grid")
    if protocol.get("confirmation_accessed") is not False:
        raise RouterTrainingError("aggregate indicates held-out access")
    expected_provenance = {
        "manifest_sha256": MANIFEST_SHA256,
        "config_canonical_sha256": CONFIG_SHA256,
        "split_sha256": PROXY_SHA256,
        "blind_trace_registry_sha256": TRACE_REGISTRY_SHA256,
        "combined_blind_trace_sha256": COMBINED_TRACE_SHA256,
    }
    if any(provenance.get(key) != expected for key, expected in expected_provenance.items()):
        raise RouterTrainingError("aggregate provenance mismatch")
    registry = provenance.get("blind_trace_registry")
    expected_registry = [
        {"shard_index": index, "record_count": 5_000, "trace_sha256": digest}
        for index, (_, digest) in enumerate(TRACE_SPECS)
    ]
    if registry != expected_registry:
        raise RouterTrainingError("aggregate trace registry mismatch")
    registry_sha256 = hashlib.sha256(
        (
            json.dumps(
                expected_registry,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if registry_sha256 != TRACE_REGISTRY_SHA256:
        raise RouterTrainingError("canonical trace registry SHA-256 mismatch")
    if action_oracle.get("session_count") != SESSION_COUNT:
        raise RouterTrainingError("aggregate oracle session count mismatch")
    if set(action_oracle.get("actions", {})) != set(OLD_ACTIONS):
        raise RouterTrainingError("aggregate is not the frozen old-action matrix")
    trajectory = worker.get("trajectory", {})
    if trajectory != {
        "completed_sessions": SESSION_COUNT,
        "fixed_turns": TURN_COUNT,
        "respond_count": TRACE_ROWS,
        "top_k": 10,
    }:
        raise RouterTrainingError("aggregate worker grid mismatch")
    if worker.get("parallel_workers") != 4 or worker.get("actions", {}).get("ids") != list(OLD_ACTIONS):
        raise RouterTrainingError("aggregate worker/action identity mismatch")
    for key in (
        "network_attempt_count",
        "full_catalog_search_calls",
        "p11_invariant_failure_count",
        "semantic_failure_count",
        "rewrite_failure_count",
    ):
        if worker.get(key) != 0:
            raise RouterTrainingError(f"aggregate worker integrity failed: {key}")
    shards = worker.get("per_shard")
    if not isinstance(shards, list) or len(shards) != 4:
        raise RouterTrainingError("aggregate shard summaries missing")
    for index, shard in enumerate(shards):
        summary = shard.get("summary", {}) if isinstance(shard, dict) else {}
        if (
            shard.get("shard_index") != index
            or shard.get("sample_count") != 500
            or summary.get("schema_version") != "p12.action-worker.v1"
            or summary.get("trace_written_after_components_closed") is not True
            or summary.get("trajectory", {}).get("respond_count") != 5_000
            or summary.get("actions", {}).get("ids") != list(OLD_ACTIONS)
        ):
            raise RouterTrainingError("closed worker shard marker mismatch")
    if _sha256(path) != AGGREGATE_SHA256:
        raise RouterTrainingError("aggregate changed during validation")
    return value


def _ranking(value: object, limit: int, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 0 < len(value) <= limit
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise RouterTrainingError(f"invalid ranking: {label}")
    return tuple(value)


def _load_traces() -> tuple[tuple[dict[str, Any], ...], frozenset[str]]:
    rows_by_coordinate: dict[tuple[int, int], dict[str, Any]] = {}
    identifiers: set[str] = set()
    combined_digest = hashlib.sha256()
    trace_root = AGGREGATE.parent
    for shard_number, (filename, expected_digest) in enumerate(TRACE_SPECS, start=1):
        path = _regular_file(trace_root / filename, f"trace shard {shard_number}")
        if _sha256(path) != expected_digest:
            raise RouterTrainingError(f"trace shard {shard_number} SHA-256 mismatch")
        count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    raise RouterTrainingError("trace contains a blank row")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RouterTrainingError("trace contains invalid JSON") from exc
                if not isinstance(raw, dict) or set(raw) != {"ordinal", "turn", "actions", "candidate_pools"}:
                    raise RouterTrainingError("trace row schema mismatch")
                local_ordinal, turn = raw["ordinal"], raw["turn"]
                if (
                    not isinstance(local_ordinal, int)
                    or isinstance(local_ordinal, bool)
                    or not 1 <= local_ordinal <= 500
                    or not isinstance(turn, int)
                    or isinstance(turn, bool)
                    or not 1 <= turn <= TURN_COUNT
                ):
                    raise RouterTrainingError("trace coordinate mismatch")
                ordinal = (shard_number - 1) * 500 + local_ordinal
                if (ordinal, turn) in rows_by_coordinate:
                    raise RouterTrainingError("trace coordinate mismatch")
                actions = raw["actions"]
                pools = raw["candidate_pools"]
                if not isinstance(actions, dict) or set(actions) != set(OLD_ACTIONS):
                    raise RouterTrainingError("trace action registry mismatch")
                if not isinstance(pools, dict) or set(pools) != {"c20", "c50", "c100"}:
                    raise RouterTrainingError("trace pool registry mismatch")
                clean_actions = {name: _ranking(actions[name], 10, name) for name in OLD_ACTIONS}
                c20 = _ranking(pools["c20"], 20, "c20")
                c50 = _ranking(pools["c50"], 50, "c50")
                c100 = _ranking(pools["c100"], 100, "c100")
                if (
                    len(c20) != 20
                    or len(c50) != 50
                    or len(c100) != 100
                    or c20 != c50[:20]
                    or c50 != c100[:50]
                ):
                    raise RouterTrainingError("trace pool prefix mismatch")
                r08 = clean_actions["KEEP_R08"]
                p11 = clean_actions["KEEP_P11"]
                if r08 != c20[:10] or set(r08) != set(p11) or clean_actions["ASK"] != p11:
                    raise RouterTrainingError("trace R08/P11 invariant mismatch")
                if any(not set(clean_actions[name]).issubset(set(c50)) for name in (
                    "CANDIDATE_RERANK", "FROZEN_SEMANTIC_RERANK"
                )):
                    raise RouterTrainingError("old action escaped C50")
                clean = {
                    "actions": clean_actions,
                    "c20": c20,
                    "c50": c50,
                    "c100": c100,
                }
                rows_by_coordinate[(ordinal, turn)] = clean
                identifiers.update(c100)
                normalized = dict(raw)
                normalized["ordinal"] = ordinal
                combined_digest.update(
                    (
                        json.dumps(
                            normalized,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                count += 1
        if count != 5_000 or _sha256(path) != expected_digest:
            raise RouterTrainingError("trace shard count or post-parse identity mismatch")
    if len(rows_by_coordinate) != TRACE_ROWS:
        raise RouterTrainingError("trace is not an exact 20,000-row grid")
    if combined_digest.hexdigest() != COMBINED_TRACE_SHA256:
        raise RouterTrainingError("combined canonical trace SHA-256 mismatch")
    sessions = tuple(
        tuple(rows_by_coordinate[(ordinal, turn)] for turn in range(1, 11))
        for ordinal in range(1, SESSION_COUNT + 1)
    )
    return sessions, frozenset(identifiers)


def _eligible_turn(sample: Mapping[str, Any]) -> int:
    scenario = sample.get("scenario_type")
    if scenario != "intent_override":
        return 1
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise RouterTrainingError("proxy sample ID is invalid")
    rng = random.Random(f"{sample_id}\0intent_override")
    return rng.choice([3, 4])


def _load_proxy() -> tuple[tuple[str, ...], tuple[int, ...]]:
    path = _regular_file(PROXY, "train proxy")
    if _sha256(path) != PROXY_SHA256:
        raise RouterTrainingError("train proxy SHA-256 mismatch")
    targets: list[str] = []
    eligible: list[int] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                raise RouterTrainingError("proxy contains a blank row")
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RouterTrainingError("proxy contains invalid JSON") from exc
            if not isinstance(sample, dict):
                raise RouterTrainingError("proxy row is not an object")
            sample_id = sample.get("sample_id")
            target = sample.get("ground_truth", {}).get("parent_asin") if isinstance(sample.get("ground_truth"), dict) else None
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in sample_ids
                or not isinstance(target, str)
                or ASIN_RE.fullmatch(target) is None
            ):
                raise RouterTrainingError("proxy label binding is invalid")
            sample_ids.add(sample_id)
            targets.append(target)
            eligible.append(_eligible_turn(sample))
    if len(targets) != SESSION_COUNT or _sha256(path) != PROXY_SHA256:
        raise RouterTrainingError("proxy row count or post-parse identity mismatch")
    return tuple(targets), tuple(eligible)


def _rank_utility(ranking: Sequence[str], identifier: str) -> float:
    try:
        return (len(ranking) - ranking.index(identifier) - 1) / max(1, len(ranking) - 1)
    except ValueError:
        return 0.0


def _features(
    turn_index: int,
    candidate_rank: int,
    candidate: str,
    incumbent: str,
    p11: Sequence[str],
    structured: Sequence[str],
    semantic: Sequence[str],
    previous: Mapping[str, Any] | None,
) -> tuple[float, ...]:
    s_set, e_set, p_set = set(structured), set(semantic), set(p11)
    s_support = float(candidate in s_set)
    e_support = float(candidate in e_set)
    i_s = float(incumbent in s_set)
    i_e = float(incumbent in e_set)
    previous_pool = previous["c50"] if previous is not None else ()
    previous_p11 = previous["actions"]["KEEP_P11"] if previous is not None else ()
    return (
        (turn_index + 1) / TURN_COUNT,
        candidate_rank / 50.0,
        (candidate_rank - 10) / 40.0,
        s_support,
        _rank_utility(structured, candidate),
        e_support,
        _rank_utility(semantic, candidate),
        (s_support + e_support) / 2.0,
        i_s,
        _rank_utility(structured, incumbent),
        i_e,
        _rank_utility(semantic, incumbent),
        (s_support + e_support - i_s - i_e) / 2.0,
        len(p_set & s_set) / 10.0,
        len(p_set & e_set) / 10.0,
        len(s_set & e_set) / 10.0,
        float(candidate in previous_pool),
        _rank_utility(previous_pool, candidate),
        float(candidate in previous_p11),
    )


def _build_target_blind_rows(
    trace: tuple[tuple[dict[str, Any], ...], ...],
) -> TargetBlindRows:
    if len(FEATURE_NAMES) != len(FEATURE_FORMULAS):
        raise RouterTrainingError("feature name/formula contract mismatch")
    row_count = SESSION_COUNT * TURN_COUNT * PROPOSAL_COUNT
    x = np.empty((row_count, len(FEATURE_NAMES)), dtype=np.float32)
    baseline_rankings: list[tuple[tuple[str, ...], ...]] = []
    proposals: list[tuple[tuple[str, ...], ...]] = []
    incumbents: list[tuple[str, ...]] = []
    cursor = 0
    for turns in trace:
        baseline = tuple(tuple(turn["actions"]["KEEP_P11"]) for turn in turns)
        session_proposals: list[tuple[str, ...]] = []
        for turn_index, turn in enumerate(turns):
            p11 = turn["actions"]["KEEP_P11"]
            c50 = turn["c50"]
            if len(p11) != 10 or len(c50) != 50 or set(p11) != set(c50[:10]):
                raise RouterTrainingError("proposal reconstruction invariant failed")
            incumbent = p11[9]
            candidates = tuple(c50[10:50])
            session_proposals.append(candidates)
            structured = turn["actions"]["CANDIDATE_RERANK"]
            semantic = turn["actions"]["FROZEN_SEMANTIC_RERANK"]
            previous = turns[turn_index - 1] if turn_index else None
            for offset, candidate in enumerate(candidates):
                x[cursor] = _features(
                    turn_index,
                    11 + offset,
                    candidate,
                    incumbent,
                    p11,
                    structured,
                    semantic,
                    previous,
                )
                cursor += 1
        baseline_rankings.append(baseline)
        proposals.append(tuple(session_proposals))
        incumbents.append(tuple(turn["actions"]["KEEP_P11"][9] for turn in turns))
    if cursor != row_count or not np.isfinite(x).all():
        raise RouterTrainingError("feature matrix construction failed")
    feature_digest = hashlib.sha256()
    feature_digest.update(
        (
            json.dumps(
                {
                    "dtype": str(x.dtype),
                    "formulas": FEATURE_FORMULAS,
                    "names": FEATURE_NAMES,
                    "shape": list(x.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    feature_digest.update(memoryview(x).cast("B"))
    return TargetBlindRows(
        x=x,
        baseline_rankings=tuple(baseline_rankings),
        proposals=tuple(proposals),
        incumbents=tuple(incumbents),
        feature_table_sha256=feature_digest.hexdigest(),
    )


def _join_labels(
    rows: TargetBlindRows,
    targets: tuple[str, ...],
    eligible_from: tuple[int, ...],
) -> Dataset:
    rescue = np.zeros(len(rows.x), dtype=np.float64)
    harm = np.zeros(len(rows.x), dtype=np.float64)
    cursor = 0
    for session_index, target in enumerate(targets):
        eligible = eligible_from[session_index]
        baseline = rows.baseline_rankings[session_index]
        baseline_hit_turns = {
            turn
            for turn, ranking in enumerate(baseline, start=1)
            if turn >= eligible and target in ranking
        }
        baseline_hit = bool(baseline_hit_turns)
        for turn_index in range(TURN_COUNT):
            current_turn = turn_index + 1
            other_hit = any(turn != current_turn for turn in baseline_hit_turns)
            p11 = baseline[turn_index]
            for candidate in rows.proposals[session_index][turn_index]:
                atomic_hit = other_hit or (
                    current_turn >= eligible
                    and (target in p11[:9] or target == candidate)
                )
                rescue[cursor] = float(not baseline_hit and atomic_hit)
                harm[cursor] = float(baseline_hit and not atomic_hit)
                cursor += 1
    if cursor != len(rows.x):
        raise RouterTrainingError("label join did not cover the closed feature table")
    cluster_sizes: dict[str, int] = {}
    for target in targets:
        cluster_sizes[target] = cluster_sizes.get(target, 0) + 1
    session_weights = np.asarray(
        [1.0 / cluster_sizes[target] for target in targets], dtype=np.float64
    )
    return Dataset(
        x=rows.x,
        rescue=rescue,
        harm=harm,
        targets=targets,
        eligible_from=np.asarray(eligible_from, dtype=np.int16),
        session_weights=session_weights,
        baseline_rankings=rows.baseline_rankings,
        proposals=rows.proposals,
        incumbents=rows.incumbents,
    )


def _folds(targets: Sequence[str], count: int, salt: str) -> np.ndarray:
    values = []
    for target in targets:
        digest = hashlib.sha256(f"{SEED}\0{salt}\0{target}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") % count)
    result = np.asarray(values, dtype=np.int8)
    if set(result.tolist()) != set(range(count)):
        raise RouterTrainingError("target-cluster fold assignment is incomplete")
    return result


def _row_mask(session_mask: np.ndarray) -> np.ndarray:
    return np.repeat(session_mask, TURN_COUNT * PROPOSAL_COUNT)


def _fit_head(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    train_x = x[mask].astype(np.float64, copy=False)
    train_y = y[mask]
    train_weights = weights[mask]
    mean = np.average(train_x, axis=0, weights=train_weights)
    centered = train_x - mean
    scale = np.sqrt(np.average(centered * centered, axis=0, weights=train_weights))
    train_weights = train_weights / train_weights.mean()
    scale[scale < 1e-12] = 1.0
    z = centered / scale
    design = np.column_stack((np.ones(len(z)), z))
    gram = design.T @ (design * train_weights[:, None])
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    rhs = design.T @ (train_y * train_weights)
    try:
        coefficient = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError as exc:
        raise RouterTrainingError("ridge solve failed") from exc
    if not all(np.isfinite(value).all() for value in (mean, scale, coefficient)):
        raise RouterTrainingError("ridge model is non-finite")
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def _predict_head(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    values = model["coefficient"][0] + z @ model["coefficient"][1:]
    return np.clip(values, 0.0, 1.0)


def _fit_pair(dataset: Dataset, session_mask: np.ndarray, alpha: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mask = _row_mask(session_mask)
    row_weights = np.repeat(dataset.session_weights, TURN_COUNT * PROPOSAL_COUNT)
    return (
        _fit_head(dataset.x, dataset.rescue, mask, row_weights, alpha),
        _fit_head(dataset.x, dataset.harm, mask, row_weights, alpha),
    )


def _predict_pair(models: tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], x: np.ndarray) -> np.ndarray:
    return _predict_head(models[0], x) - LAMBDA_HARM * _predict_head(models[1], x)


def _official_metrics(
    sessions: int,
    hits: int,
    reciprocal_rank_sum: float,
    first_turn_sum: int,
) -> dict[str, Any]:
    hit_rate = round(hits / sessions, 6)
    mrr = round(reciprocal_rank_sum / sessions, 6)
    mttc = round(first_turn_sum / sessions, 6)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": sessions,
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _evaluate_policy(
    dataset: Dataset,
    gains: np.ndarray,
    sessions: np.ndarray,
    threshold: float,
    runner_gap: float,
) -> dict[str, Any]:
    cube = gains.reshape(SESSION_COUNT, TURN_COUNT, PROPOSAL_COUNT)
    miss_to_hit = hit_to_miss = activation_turns = activation_sessions = hits = baseline_hits = 0
    baseline_rr_sum = policy_rr_sum = 0.0
    baseline_turn_sum = policy_turn_sum = 0
    for session_index in np.flatnonzero(sessions):
        target = dataset.targets[session_index]
        eligible = int(dataset.eligible_from[session_index])
        session_active = False
        baseline_first_turn: int | None = None
        baseline_first_rank: int | None = None
        policy_first_turn: int | None = None
        policy_first_rank: int | None = None
        for turn_index in range(TURN_COUNT):
            baseline = dataset.baseline_rankings[session_index][turn_index]
            current_turn = turn_index + 1
            if (
                current_turn >= eligible
                and baseline_first_turn is None
                and target in baseline
            ):
                baseline_first_turn = turn_index + 1
                baseline_first_rank = baseline.index(target) + 1
            if policy_first_turn is None:
                values = cube[session_index, turn_index]
                order = np.argsort(-values, kind="stable")
                best, second = int(order[0]), int(order[1])
                active = bool(
                    values[best] >= threshold
                    and values[best] - values[second] >= runner_gap
                )
                activation_turns += int(active)
                session_active = session_active or active
                if current_turn >= eligible:
                    if active:
                        candidate = dataset.proposals[session_index][turn_index][best]
                        policy_ranking = (*baseline[:9], candidate)
                    else:
                        policy_ranking = baseline
                    if target in policy_ranking:
                        policy_first_turn = current_turn
                        policy_first_rank = policy_ranking.index(target) + 1
        activation_sessions += int(session_active)
        base_hit = baseline_first_turn is not None
        policy_hit = policy_first_turn is not None
        baseline_hits += int(base_hit)
        hits += int(policy_hit)
        miss_to_hit += int(not base_hit and policy_hit)
        hit_to_miss += int(base_hit and not policy_hit)
        baseline_rr_sum += 0.0 if baseline_first_rank is None else 1.0 / baseline_first_rank
        policy_rr_sum += 0.0 if policy_first_rank is None else 1.0 / policy_first_rank
        baseline_turn_sum += 11 if baseline_first_turn is None else baseline_first_turn
        policy_turn_sum += 11 if policy_first_turn is None else policy_first_turn
    count = int(sessions.sum())
    return {
        "sessions": count,
        "baseline_hits": baseline_hits,
        "policy_hits": hits,
        "baseline_hr": round(baseline_hits / count, 6),
        "policy_hr": round(hits / count, 6),
        "miss_to_hit": miss_to_hit,
        "hit_to_miss": hit_to_miss,
        "net": miss_to_hit - hit_to_miss,
        "objective": miss_to_hit - int(LAMBDA_HARM) * hit_to_miss,
        "activation_turns": activation_turns,
        "activation_sessions": activation_sessions,
        "baseline_reciprocal_rank_sum": baseline_rr_sum,
        "policy_reciprocal_rank_sum": policy_rr_sum,
        "baseline_first_turn_sum": baseline_turn_sum,
        "policy_first_turn_sum": policy_turn_sum,
        "baseline_official": _official_metrics(
            count, baseline_hits, baseline_rr_sum, baseline_turn_sum
        ),
        "policy_official": _official_metrics(
            count, hits, policy_rr_sum, policy_turn_sum
        ),
    }


def _select(
    dataset: Dataset,
    session_mask: np.ndarray,
    inner_folds: np.ndarray,
) -> tuple[float, float, float, dict[str, Any]]:
    alphas = (1.0, 10.0, 100.0)
    gaps = (0.0, 0.001, 0.005, 0.01)
    best: tuple[tuple[float, ...], tuple[float, float, float, dict[str, Any]]] | None = None
    for alpha in alphas:
        gains = np.full(len(dataset.x), -math.inf, dtype=np.float64)
        for fold in sorted(set(inner_folds[session_mask].tolist())):
            valid_sessions = session_mask & (inner_folds == fold)
            train_sessions = session_mask & (inner_folds != fold)
            models = _fit_pair(dataset, train_sessions, alpha)
            row_valid = _row_mask(valid_sessions)
            gains[row_valid] = _predict_pair(models, dataset.x[row_valid])
        best_per_turn = gains.reshape(SESSION_COUNT, TURN_COUNT, PROPOSAL_COUNT).max(axis=2)
        finite = best_per_turn[np.repeat(session_mask[:, None], TURN_COUNT, axis=1)]
        thresholds = sorted(set(float(np.quantile(finite, q)) for q in (0.50, 0.75, 0.90, 0.95, 0.975, 0.99)))
        for threshold in thresholds:
            for gap in gaps:
                metrics = _evaluate_policy(dataset, gains, session_mask, threshold, gap)
                key = (
                    float(metrics["objective"]),
                    -float(metrics["hit_to_miss"]),
                    -float(metrics["activation_turns"]),
                    threshold,
                    gap,
                    alpha,
                )
                candidate = (alpha, threshold, gap, metrics)
                if best is None or key > best[0]:
                    best = (key, candidate)
    if best is None:
        raise RouterTrainingError("inner model selection produced no candidate")
    return best[1]


def _serialize_head(model: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "mean": [round(float(value), 15) for value in model["mean"]],
        "scale": [round(float(value), 15) for value in model["scale"]],
        "coefficient": [round(float(value), 15) for value in model["coefficient"]],
    }


def _recursive_audit(value: object, identifiers: frozenset[str], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RouterTrainingError(f"non-string output key at {path}")
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_KEYS):
                raise RouterTrainingError(f"forbidden identifier key at {path}.{key}")
            _recursive_audit(child, identifiers, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _recursive_audit(child, identifiers, f"{path}[{index}]")
    elif isinstance(value, str):
        if value in identifiers or ASIN_RE.fullmatch(value):
            raise RouterTrainingError(f"identifier-like output value at {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RouterTrainingError(f"non-finite output value at {path}")


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _reject_forbidden_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise RouterTrainingError("output parent must be an existing directory")
    attrs = getattr(parent.stat(), "st_file_attributes", 0)
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise RouterTrainingError("output parent must not be a reparse point")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def train(output: Path) -> dict[str, Any]:
    aggregate = _validate_aggregate()
    trace, trace_identifiers = _load_traces()
    target_blind_rows = _build_target_blind_rows(trace)
    targets, eligible = _load_proxy()
    all_identifiers = trace_identifiers | frozenset(targets)
    dataset = _join_labels(target_blind_rows, targets, eligible)

    outer = _folds(targets, 5, "outer")
    oof_gains = np.full(len(dataset.x), -math.inf, dtype=np.float64)
    outer_choices: list[dict[str, Any]] = []
    for fold in range(5):
        held = outer == fold
        train_sessions = ~held
        inner = _folds(targets, 4, f"outer-{fold}-inner")
        alpha, threshold, gap, inner_metrics = _select(dataset, train_sessions, inner)
        models = _fit_pair(dataset, train_sessions, alpha)
        held_rows = _row_mask(held)
        oof_gains[held_rows] = _predict_pair(models, dataset.x[held_rows])
        outer_choices.append({
            "fold": fold,
            "alpha": alpha,
            "threshold": round(threshold, 15),
            "runner_gap": gap,
            "inner": inner_metrics,
        })
    # Evaluate outer predictions with each fold's independently selected gate.
    combined: dict[str, int | float | dict[str, int | float]] = {key: 0 for key in (
        "sessions", "baseline_hits", "policy_hits", "miss_to_hit", "hit_to_miss",
        "activation_turns", "activation_sessions", "baseline_reciprocal_rank_sum",
        "policy_reciprocal_rank_sum", "baseline_first_turn_sum", "policy_first_turn_sum",
    )}
    for choice in outer_choices:
        held = outer == int(choice["fold"])
        metrics = _evaluate_policy(dataset, oof_gains, held, float(choice["threshold"]), float(choice["runner_gap"]))
        for key in tuple(combined):
            if key.endswith("reciprocal_rank_sum"):
                combined[key] = float(combined[key]) + float(metrics[key])
            else:
                combined[key] = int(combined[key]) + int(metrics[key])
    baseline_official = _official_metrics(
        SESSION_COUNT,
        int(combined["baseline_hits"]),
        float(combined["baseline_reciprocal_rank_sum"]),
        int(combined["baseline_first_turn_sum"]),
    )
    policy_official = _official_metrics(
        SESSION_COUNT,
        int(combined["policy_hits"]),
        float(combined["policy_reciprocal_rank_sum"]),
        int(combined["policy_first_turn_sum"]),
    )
    combined.update({
        "baseline_hr": baseline_official["hit_rate_at_10"],
        "policy_hr": policy_official["hit_rate_at_10"],
        "net": int(combined["miss_to_hit"]) - int(combined["hit_to_miss"]),
        "objective": int(combined["miss_to_hit"]) - int(LAMBDA_HARM) * int(combined["hit_to_miss"]),
        "baseline_official": baseline_official,
        "policy_official": policy_official,
    })

    all_sessions = np.ones(SESSION_COUNT, dtype=bool)
    full_inner = _folds(targets, 5, "final-inner")
    alpha, threshold, gap, final_inner = _select(dataset, all_sessions, full_inner)
    final_models = _fit_pair(dataset, all_sessions, alpha)
    fold_summary = [
        {
            "fold": fold,
            "group_count": len({targets[index] for index in np.flatnonzero(outer == fold)}),
            "session_count": int((outer == fold).sum()),
        }
        for fold in range(5)
    ]
    historical_sources = aggregate["provenance"].get("source_sha256_lf", {})
    if not isinstance(historical_sources, dict) or not historical_sources:
        raise RouterTrainingError("historical source closure is missing")
    source_closure_sha256 = hashlib.sha256(
        json.dumps(
            historical_sources,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact = {
        "schema_version": "p12.counterfactual-router.v1",
        "training_split": SPLIT,
        "promotion_status": "OOF_RESEARCH_ONLY_NOT_RUNTIME_DEPLOYABLE",
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "formulas_in_name_order": list(FEATURE_FORMULAS),
            "numeric_dtype": "float32 input; float64 standardization and ridge solve",
            "forbidden_runtime_inputs": [
                "product identity", "sample identity", "group labels", "scenario labels",
                "taxonomy labels", "difficulty labels", "future turns",
            ],
            "proposal": "exact R08 C50 ranks 11-to-50; exact P11 rank-10 membership swap",
            "membership_only": "P11 ranks 1-to-9 remain exact; HR@1/MRR ordering is not optimized",
            "missing_rank": "zero utility plus a separate support flag",
            "rank_utility": "(ranking length - one-based rank) / (ranking length - 1)",
            "history": "only the immediately previous visible turn; no future turn",
        },
        "model": {
            "type": "deterministic_two_head_clipped_ridge_scores",
            "harm_multiplier": LAMBDA_HARM,
            "alpha": alpha,
            "threshold": round(threshold, 15),
            "runner_gap": gap,
            "standardization": "(x - mean) / scale; scale below 1e-12 becomes 1",
            "head_score": "clip(intercept + standardized_x dot coefficient, 0, 1)",
            "routing_score": "rescue_head_score - 2 * harm_head_score",
            "tie_break": "stable original R08 candidate order",
            "activation": "best score >= threshold and best-minus-runner >= runner_gap",
            "output": "when active replace only P11 rank10; otherwise exact KEEP_P11",
            "rescue_head": _serialize_head(final_models[0]),
            "harm_head": _serialize_head(final_models[1]),
        },
        "cross_fit": {
            "outer_folds": 5,
            "group_unit": "same-label-product cluster (offline fold use only)",
            "assignment": "seeded SHA-256 modulo; digest never enters features or model",
            "training_weight": "each product cluster has equal total ridge weight",
            "selection_objective": "miss_to_hit - 2 * hit_to_miss; MRR is reported separately",
            "fold_summary": fold_summary,
            "outer_choices": outer_choices,
            "oof_membership": combined,
            "final_inner": final_inner,
        },
        "label_summary": {
            "atomic_rescue_positive_rows": int(dataset.rescue.sum()),
            "atomic_harm_positive_rows": int(dataset.harm.sum()),
            "labels_joined_only_after_feature_table_hashed": True,
        },
        "source_provenance": {
            "aggregate_sha256": AGGREGATE_SHA256,
            "proxy_sha256": PROXY_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "historical_config_canonical_sha256": CONFIG_SHA256,
            "trace_registry_sha256": TRACE_REGISTRY_SHA256,
            "combined_trace_sha256": COMBINED_TRACE_SHA256,
            "feature_table_sha256": target_blind_rows.feature_table_sha256,
            "historical_source_closure_sha256": source_closure_sha256,
            "historical_source_file_count": len(historical_sources),
            "trainer_sha256": _sha256(Path(__file__)),
            "trace_rows": TRACE_ROWS,
            "sessions": SESSION_COUNT,
            "trust_boundary": "raw aggregate hash plus verified shard and combined-trace hashes",
        },
        "privacy_audit": {
            "catalog_identity_values_serialized": 0,
            "catalog_identity_keys_serialized": 0,
            "runtime_feature_count": len(FEATURE_NAMES),
            "feature_extraction_completed_before_proxy_label_load": True,
        },
    }
    _recursive_audit(artifact, all_identifiers)
    _write_exclusive(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split != SPLIT:
        raise RouterTrainingError("only train_explore is permitted")
    output = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    _reject_forbidden_path(output)
    artifact = train(output)
    summary = artifact["cross_fit"]["oof_membership"]
    print(json.dumps({"output": str(output.resolve()), "oof_membership": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RouterTrainingError, ValueError) as exc:
        print(f"[p12-counterfactual-router] {exc}", file=sys.stderr)
        raise SystemExit(1)
