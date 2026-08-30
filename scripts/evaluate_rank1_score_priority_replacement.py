"""One-shot evaluation of the preregistered v2.13 score-priority rank-1 policy.

Everything before the durable O_EXCL receipt is target-free.  The fixed label
archive is opened exactly once, after the receipt is durable, solely to
reproduce the frozen v1.9 comparator and to compute aggregate/fold outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, BinaryIO, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_rank1_seen_replacement as v12  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


# Preserve the audited v2.12 helper surface for focused synthetic tests.
Rank1ReplayError = v12.Rank1ReplayError
Rank1ReplayConsumedError = v12.Rank1ReplayConsumedError
OutcomeBundle = v12.OutcomeBundle
TargetFreeInputs = v12.TargetFreeInputs
ReplayBundle = v12.ReplayBundle
MetricValues = v12.MetricValues
Transition = v12.Transition
decode_intent_versions = v12.decode_intent_versions
reconstruct_v19_order = v12.reconstruct_v19_order
derive_baseline_session_hit = v12.derive_baseline_session_hit
load_outcomes_from_open_handle = v12.load_outcomes_from_open_handle
state_from_positive_index = v12.state_from_positive_index
dominance_audit = v12.dominance_audit
metric_values = v12.metric_values
transition_metrics = v12.transition_metrics
VERSION_FEATURE_INDEX = v12.VERSION_FEATURE_INDEX
GOAL_AGE_FEATURE_INDEX = v12.GOAL_AGE_FEATURE_INDEX
CURRENT_OVERRIDE_FEATURE_INDEX = v12.CURRENT_OVERRIDE_FEATURE_INDEX
OVERRIDE_COUNT_FEATURE_INDEX = v12.OVERRIDE_COUNT_FEATURE_INDEX
_dual_nonnegative = v12._dual_nonnegative
_dual_nonpositive = v12._dual_nonpositive
_dual_strict_positive = v12._dual_strict_positive


SCHEMA_VERSION = "small-ranker-score-priority-rank1-outcome.v1"
EXPERIMENT_ID = "SR-V2.13-SCORE-PRIORITY-RANK1"
BRANCH = "small-ranker-v2.13-score-priority-rank1"
REMOTE = "origin"
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
REMOTE_REF = "refs/remotes/origin/" + BRANCH
LINEAGE_BASE_COMMIT = "ddc963d569b300c8e590272b9c4e65ad5164b670"
COMPARABILITY_COMMIT = "fed015af46bdf20c4fe240fe7118af5cd55dc23f"
PREREG_COMMIT = "9240ace8a588eb82a3898ab805126bf1e322911e"
PREREG_BLOB_OID = "f49fa7ba9cee5962c6f6203f8d9f7675cc59a7a2"
PREREG_RAW_SHA256 = "99f4aa2bc549a2846364706fe167b02fd50cc54fe1e3cd51a48d2ee9d31ee331"
PREREG_CANONICAL_SHA256 = "2769f345af4fb863faf917d76579a874f1cc58d840a8a28be121c84269954df6"
COMPARABILITY_BLOB_OID = "344cc82b1ce06144a0a29c6754cfdb1c4859cfdb"
COMPARABILITY_RAW_SHA256 = "f2fff7144baf7c45200015ab1184060b8f51abdfe1db5a07bf9c40f665357884"
COMPARABILITY_CANONICAL_SHA256 = "76a8e3047cb2f3737003ec208e06030fa3ad3054b82e3fa53ca1b0c0bc248091"
PREREG_PATH = ROOT / "configs/small_ranker_v2_13.score_priority_rank1_replacement_preregistration.json"
COMPARABILITY_PATH = ROOT / "configs/small_ranker_global_benchmark_comparability_v1.manifest.json"
PREREG_PATH_SET = {
    "configs/small_ranker_v2_13.score_priority_rank1_replacement_preregistration.json"
}
COMPARABILITY_PATH_SET = {
    "configs/small_ranker_global_benchmark_comparability_v1.manifest.json"
}
IMPLEMENTATION_PATHS = {
    "scripts/evaluate_rank1_score_priority_replacement.py",
    "tests/test_rank1_score_priority_replacement.py",
}

SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
FEATURE_COUNT = 133
OUTER_FOLDS = 5
BASE_SEED = 40_220_260_830
EXPECTED_INNER_FITS = 50
EXPECTED_OUTER_FITS = 10
EXPECTED_FIT_INVOCATIONS = 60
EXPECTED_SELECTION_INVOCATIONS = 5
EXPECTED_FOLD_QUANTILES = (25 / 64, 44 / 64, 26 / 64, 55 / 64, 32 / 64)
RESOURCE_BYTES_MAXIMUM = 2_147_483_648
RESOURCE_SECONDS_MAXIMUM = 120.0

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
PROJECTED_FEATURES_PATH = PROJECTION_ROOT / (
    "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
)
PROJECTED_FEATURES_SHA256 = "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
PROJECTED_FEATURES_BYTES = 1_064_000_128
OOF_SCORES_PATH = SOURCE_ROOT / (
    "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
    "oof_scores_runtime_projection_no_semantic.npy"
)
OOF_SCORES_SHA256 = "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e"
OOF_SCORES_BYTES = 8_000_128
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
LABEL_BYTES = 1_702_876
LABEL_MEMBER_SPECS = v12.LABEL_MEMBER_SPECS
OUTPUT_PATH = ROOT / (
    "experiments/fast_track/small_ranker_v2_13/"
    "score_priority_rank1_one_shot_20260831/score_priority_rank1_result.json"
)

EXPECTED_CHOSEN_SHA256 = "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
EXPECTED_ACTIVATION_SHA256 = "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
EXPECTED_PRIORITY_SHA256 = "54bccc6f930a696f80cc754104efbbaecab96e15f3abc97534289c1f191e6505"
EXPECTED_PRIORITY_TIE_ROWS = 18_364
EXPECTED_PRIORITY_DUPLICATE_EXCESS = 630_767
EXPECTED_BASELINE_ASCII_SHA256 = "2d5fa0ea12ab02b74e2b6c3a3f92b3ba83191c35eb176f95aed9361f785496e9"
EXPECTED_GRACE_ASCII_SHA256 = "6f84d2aa03d792027634a358bdcef9adf646717c94cc3ec0916e87e33410448a"
EXPECTED_LOCAL_ARRAY_SHA256 = {
    "baseline_pages": "3b6dec60039b93f3b2704baf82fd16536e587f92a3ef113988ed66f383b5af8d",
    "candidate_pages": "c4041222828b0c9a8dbdf94857b333c93b731c249156cf78a6963c822c54046c",
    "changed": "27006b9ff7ed5ba7293cca9413fc7d0fb82db34056e5a477ac582c68038a5cab",
    "last_reset_turn": "580a474cfb3031c138aa4908bb4195102e2a4d000509c20630cdc007acc5db34",
}
EXPECTED_LOCAL_IDENTITY_SHA256 = "272bc0cbabec7ecc77120cbe7db85d0e3635afb87d85cc66159f72fc917a3745"
EXPECTED_BASELINE_OFFICIAL = {
    "hit_rate_at_10": 0.9715,
    "mrr": 0.676861,
    "mttc": 3.056,
    "technical_score": 0.847688,
}
EXPECTED_LOCAL_OFFICIAL = {
    "hit_rate_at_10": 0.9765,
    "mrr": 0.684856,
    "mttc": 3.015,
    "technical_score": 0.853407,
}
EXPECTED_LOCAL_FOLDS = (
    {"hit_rate_at_10": 0.9675, "mrr": 0.709984, "mttc": 2.9475, "technical_score": 0.857795},
    {"hit_rate_at_10": 0.9675, "mrr": 0.658691, "mttc": 3.09, "technical_score": 0.839557},
    {"hit_rate_at_10": 0.99, "mrr": 0.715605, "mttc": 2.9775, "technical_score": 0.870132},
    {"hit_rate_at_10": 0.9675, "mrr": 0.68047, "mttc": 3.0625, "technical_score": 0.846641},
    {"hit_rate_at_10": 0.99, "mrr": 0.659528, "mttc": 2.9975, "technical_score": 0.852908},
)
EXPECTED_GRACE_OFFICIAL = {
    "hit_rate_at_10": 0.991,
    "mrr": 0.695795,
    "mttc": 2.869,
    "technical_score": 0.866858,
}
EXPECTED_GRACE_FOLDS = (
    {"hit_rate_at_10": 0.995, "mrr": 0.721803, "mttc": 2.7025, "technical_score": 0.879991},
    {"hit_rate_at_10": 0.9875, "mrr": 0.665996, "mttc": 2.9275, "technical_score": 0.854999},
    {"hit_rate_at_10": 0.9925, "mrr": 0.733901, "mttc": 2.9075, "technical_score": 0.87827},
    {"hit_rate_at_10": 0.9875, "mrr": 0.678388, "mttc": 2.925, "technical_score": 0.858766},
    {"hit_rate_at_10": 0.9925, "mrr": 0.678887, "mttc": 2.8825, "technical_score": 0.862266},
)

ASIN_SHAPE_RE = re.compile(rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class TargetFreePreflight:
    environment: Mapping[str, Any]
    preregistration: Mapping[str, Any]
    git: Mapping[str, Any]
    target_free: TargetFreeInputs
    priority_audit: Mapping[str, Any]
    memory_before_receipt: tuple[int, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def score_priority_ordinals(scores: np.ndarray) -> np.ndarray:
    """Return raw-C100 zero-based ordinals ordered by score desc, ordinal asc."""

    values = np.asarray(scores)
    if (
        values.ndim < 1
        or values.shape[-1] != CANDIDATE_COUNT
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise Rank1ReplayError("score-priority surface schema failed")
    ordinal = np.broadcast_to(
        np.arange(CANDIDATE_COUNT, dtype=np.uint8), values.shape
    )
    ordered = np.lexsort((ordinal, -values), axis=-1).astype(np.uint8, copy=False)
    ordered = np.ascontiguousarray(ordered)
    expected = np.broadcast_to(
        np.arange(CANDIDATE_COUNT, dtype=np.uint8), ordered.shape
    )
    if not np.array_equal(np.sort(ordered, axis=-1), expected):
        raise Rank1ReplayError("score-priority row is not a raw-C100 permutation")
    ordered.setflags(write=False)
    return ordered


def _priority_audit(scores: np.ndarray) -> dict[str, Any]:
    priority = score_priority_ordinals(scores)
    ordered_scores = np.take_along_axis(
        np.asarray(scores), priority.astype(np.int64), axis=-1
    )
    adjacent_equal = ordered_scores[..., 1:] == ordered_scores[..., :-1]
    audit = {
        **_array_identity(priority),
        "order": "C",
        "rows_with_at_least_one_exact_score_tie": int(
            np.any(adjacent_equal, axis=-1).sum()
        ),
        "duplicate_score_excess": int(adjacent_equal.sum()),
        "all_scores_finite": True,
        "zero_based_permutation": True,
    }
    if not (
        audit["raw_sha256"] == EXPECTED_PRIORITY_SHA256
        and audit["rows_with_at_least_one_exact_score_tie"]
        == EXPECTED_PRIORITY_TIE_ROWS
        and audit["duplicate_score_excess"]
        == EXPECTED_PRIORITY_DUPLICATE_EXCESS
    ):
        raise Rank1ReplayError("target-free score-priority identity drifted")
    return audit


def _choose_priority_replacement(
    order: Sequence[str],
    raw_c100: Sequence[str],
    priority: Sequence[int],
    served: set[str],
) -> tuple[str, ...]:
    ranked = tuple(str(value) for value in order)
    raw = tuple(str(value) for value in raw_c100)
    if (
        len(ranked) != CANDIDATE_COUNT
        or len(raw) != CANDIDATE_COUNT
        or len(set(ranked)) != CANDIDATE_COUNT
        or len(set(raw)) != CANDIDATE_COUNT
        or set(ranked) != set(raw)
    ):
        raise Rank1ReplayError("reconstructed/raw C100 identity failed")
    baseline = ranked[:10]
    if baseline[0] not in served:
        return baseline
    legal = set(ranked[10:])
    replacement: str | None = None
    for raw_ordinal in priority:
        index = int(raw_ordinal)
        if not 0 <= index < CANDIDATE_COUNT:
            raise Rank1ReplayError("score-priority ordinal escaped C100")
        identifier = raw[index]
        if identifier in legal and identifier not in served:
            replacement = identifier
            break
    if replacement is None:
        return baseline
    page = (replacement, *baseline[1:])
    if len(page) != 10 or len(set(page)) != 10:
        raise Rank1ReplayError("score-priority replacement page is invalid")
    return page


def rank1_score_priority_replacement(
    order: Sequence[str],
    raw_c100: Sequence[str],
    scores: np.ndarray,
    served: set[str],
) -> tuple[str, ...]:
    """Apply the frozen rank-1 envelope with score-prioritized legal tail."""

    values = np.asarray(scores)
    if values.shape != (CANDIDATE_COUNT,) or values.dtype != np.float32:
        raise Rank1ReplayError("one-row score-priority schema failed")
    priority = score_priority_ordinals(values)
    return _choose_priority_replacement(order, raw_c100, priority, served)


def _iter_ascii_pages(value: Any) -> Sequence[Sequence[str]]:
    pages: list[tuple[str, ...]] = []

    def walk(node: Any) -> None:
        if isinstance(node, np.ndarray):
            node = node.tolist()
        if isinstance(node, (list, tuple)):
            if node and all(isinstance(item, str) for item in node):
                page = tuple(node)
                if len(set(page)) != len(page):
                    raise Rank1ReplayError("ASCII digest page contains duplicates")
                pages.append(page)
                return
            for child in node:
                walk(child)
            return
        raise Rank1ReplayError("ASCII digest input is not nested pages")

    walk(value)
    if not pages:
        raise Rank1ReplayError("ASCII digest input contains no pages")
    return pages


def ascii_page_digest(pages: Any) -> str:
    digest = hashlib.sha256()
    for page in _iter_ascii_pages(pages):
        try:
            payload = ("|".join(page) + "\n").encode("ascii")
        except UnicodeEncodeError as error:
            raise Rank1ReplayError("page identifier is not ASCII") from error
        digest.update(payload)
    return digest.hexdigest()


def _digest_page(digest: Any, page: Sequence[str]) -> None:
    try:
        digest.update(("|".join(page) + "\n").encode("ascii"))
    except UnicodeEncodeError as error:
        raise Rank1ReplayError("page identifier is not ASCII") from error


def _finish_replay_bundle(
    baseline_pages: np.ndarray,
    candidate_pages: np.ndarray,
    changed: np.ndarray,
    last_reset_turn: np.ndarray,
    baseline_digest: str,
    candidate_digest: str,
    structural: Mapping[str, Any],
    latency_ns: Sequence[int],
) -> ReplayBundle:
    for value in (baseline_pages, candidate_pages, changed, last_reset_turn):
        value.setflags(write=False)
    identity = {
        "baseline_pages": _array_identity(baseline_pages),
        "candidate_pages": _array_identity(candidate_pages),
        "changed": _array_identity(changed),
        "last_reset_turn": _array_identity(last_reset_turn),
        "baseline_ascii_page_sha256": baseline_digest,
        "candidate_ascii_page_sha256": candidate_digest,
    }
    timing = {
        "sample_count": len(latency_ns),
        "p50_microseconds": (
            round(float(np.percentile(latency_ns, 50)) / 1_000.0, 6)
            if latency_ns
            else None
        ),
        "p95_microseconds": (
            round(float(np.percentile(latency_ns, 95)) / 1_000.0, 6)
            if latency_ns
            else None
        ),
        "maximum_microseconds": (
            round(max(latency_ns) / 1_000.0, 6) if latency_ns else None
        ),
    }
    return ReplayBundle(
        baseline_pages,
        candidate_pages,
        changed,
        last_reset_turn,
        identity,
        dict(structural),
        timing,
    )


def replay_score_priority_pages(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    scores: np.ndarray,
    chosen: np.ndarray,
    activation: np.ndarray,
    versions: np.ndarray,
    measure_timing: bool = False,
) -> ReplayBundle:
    values = np.asarray(scores)
    chosen = np.asarray(chosen)
    activation = np.asarray(activation, dtype=bool)
    versions = np.asarray(versions)
    session_count, turn_count = versions.shape
    if (
        values.shape != (session_count, turn_count, CANDIDATE_COUNT)
        or values.dtype != np.float32
        or chosen.shape != (session_count, turn_count)
        or activation.shape != (session_count, turn_count)
        or len(traces) != session_count
    ):
        raise Rank1ReplayError("score-priority replay surface failed")
    priorities = score_priority_ordinals(values)
    baseline_pages = np.empty((session_count, turn_count, 10), dtype=np.int16)
    candidate_pages = np.empty_like(baseline_pages)
    changed = np.zeros((session_count, turn_count), dtype=bool)
    last_reset_turn = np.ones(session_count, dtype=np.int16)
    baseline_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    latency_ns: list[int] = []
    reset_count = changed_turns = changed_sessions = 0
    baseline_distinct_total = candidate_distinct_total = 0
    for session, turns in enumerate(traces):
        if len(turns) != turn_count:
            raise Rank1ReplayError("blind trace turn count drifted")
        served: set[str] = set()
        last_version: int | None = None
        session_changed = False
        baseline_distinct: set[str] = set()
        candidate_distinct: set[str] = set()
        for turn_index, turn in enumerate(turns):
            version = int(versions[session, turn_index])
            is_reset = last_version is None or version != last_version
            if is_reset:
                served.clear()
                reset_count += 1
                last_reset_turn[session] = turn_index + 1
            order = reconstruct_v19_order(
                turn,
                int(chosen[session, turn_index]),
                bool(activation[session, turn_index]),
            )
            raw = tuple(str(value) for value in turn["c100"])
            baseline = order[:10]
            tick = time.perf_counter_ns() if measure_timing else 0
            candidate = _choose_priority_replacement(
                order, raw, priorities[session, turn_index], served
            )
            if measure_timing:
                latency_ns.append(time.perf_counter_ns() - tick)
            page_changed = candidate != baseline
            if is_reset and page_changed:
                raise Rank1ReplayError("score-priority reset page is not identity")
            if page_changed and not (
                baseline[0] in served
                and candidate[0] not in served
                and candidate[0] in order[10:]
                and candidate[1:] == baseline[1:]
            ):
                raise Rank1ReplayError("score-priority structural invariant failed")
            raw_index = {identifier: index for index, identifier in enumerate(raw)}
            try:
                baseline_pages[session, turn_index] = [
                    raw_index[identifier] for identifier in baseline
                ]
                candidate_pages[session, turn_index] = [
                    raw_index[identifier] for identifier in candidate
                ]
            except KeyError as error:
                raise Rank1ReplayError("score-priority page escaped raw C100") from error
            changed[session, turn_index] = page_changed
            changed_turns += int(page_changed)
            session_changed |= page_changed
            _digest_page(baseline_digest, baseline)
            _digest_page(candidate_digest, candidate)
            served.update(candidate)
            baseline_distinct.update(baseline)
            candidate_distinct.update(candidate)
            last_version = version
        changed_sessions += int(session_changed)
        baseline_distinct_total += len(baseline_distinct)
        candidate_distinct_total += len(candidate_distinct)
    structural = {
        "reset_count": reset_count,
        "changed_turns": changed_turns,
        "changed_sessions": changed_sessions,
        "changed_slots_per_changed_turn": 1 if changed_turns else 0,
        "ranks_2_to_10_byte_identical": True,
        "removed_rank1_already_served": True,
        "inserted_rank1_unseen_legal_tail": True,
        "reset_pages_identity": True,
        "baseline_mean_distinct_products": round(
            baseline_distinct_total / session_count, 6
        ),
        "candidate_mean_distinct_products": round(
            candidate_distinct_total / session_count, 6
        ),
    }
    return _finish_replay_bundle(
        baseline_pages,
        candidate_pages,
        changed,
        last_reset_turn,
        baseline_digest.hexdigest(),
        candidate_digest.hexdigest(),
        structural,
        latency_ns,
    )


def _stable_unseen_first(
    order: Sequence[str], served: set[str]
) -> tuple[str, ...]:
    ranked = tuple(order)
    unseen = [identifier for identifier in ranked if identifier not in served]
    seen = [identifier for identifier in ranked if identifier in served]
    page = tuple((unseen + seen)[:10])
    if len(page) != 10 or len(set(page)) != 10:
        raise Rank1ReplayError("grace unseen-first page is invalid")
    return page


def replay_grace_pages(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    chosen: np.ndarray,
    activation: np.ndarray,
    versions: np.ndarray,
) -> ReplayBundle:
    chosen = np.asarray(chosen)
    activation = np.asarray(activation, dtype=bool)
    versions = np.asarray(versions)
    session_count, turn_count = versions.shape
    if (
        len(traces) != session_count
        or chosen.shape != (session_count, turn_count)
        or activation.shape != (session_count, turn_count)
    ):
        raise Rank1ReplayError("grace replay surface failed")
    if any(len(turns) != turn_count for turns in traces):
        raise Rank1ReplayError("blind trace turn count drifted")
    baseline_pages = np.empty((session_count, turn_count, 10), dtype=np.int16)
    candidate_pages = np.empty_like(baseline_pages)
    changed = np.zeros((session_count, turn_count), dtype=bool)
    last_reset_turn = np.ones(session_count, dtype=np.int16)
    baseline_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    reset_count = changed_turns = changed_sessions = 0
    for session, turns in enumerate(traces):
        served: set[str] = set()
        last_version: int | None = None
        intent_age = 0
        session_changed = False
        for turn_index, turn in enumerate(turns):
            version = int(versions[session, turn_index])
            is_reset = last_version is None or version != last_version
            if is_reset:
                served.clear()
                intent_age = 1
                reset_count += 1
                last_reset_turn[session] = turn_index + 1
            else:
                intent_age += 1
            order = reconstruct_v19_order(
                turn,
                int(chosen[session, turn_index]),
                bool(activation[session, turn_index]),
            )
            raw = tuple(str(value) for value in turn["c100"])
            if len(raw) != CANDIDATE_COUNT or set(raw) != set(order):
                raise Rank1ReplayError("grace raw/reconstructed C100 mismatch")
            baseline = order[:10]
            candidate = baseline if intent_age <= 2 else _stable_unseen_first(order, served)
            page_changed = candidate != baseline
            if is_reset and page_changed:
                raise Rank1ReplayError("grace reset page is not identity")
            raw_index = {identifier: index for index, identifier in enumerate(raw)}
            baseline_pages[session, turn_index] = [raw_index[value] for value in baseline]
            candidate_pages[session, turn_index] = [raw_index[value] for value in candidate]
            changed[session, turn_index] = page_changed
            changed_turns += int(page_changed)
            session_changed |= page_changed
            _digest_page(baseline_digest, baseline)
            _digest_page(candidate_digest, candidate)
            served.update(candidate)
            last_version = version
        changed_sessions += int(session_changed)
    structural = {
        "reset_count": reset_count,
        "changed_turns": changed_turns,
        "changed_sessions": changed_sessions,
        "grace_pages_per_intent_version": 2,
        "reset_pages_identity": True,
        "stable_unseen_first_after_grace": True,
    }
    return _finish_replay_bundle(
        baseline_pages,
        candidate_pages,
        changed,
        last_reset_turn,
        baseline_digest.hexdigest(),
        candidate_digest.hexdigest(),
        structural,
        (),
    )


def audit_nested_partition(
    outer: np.ndarray, inner: np.ndarray, action: np.ndarray
) -> dict[str, Any]:
    outer = np.asarray(outer)
    inner = np.asarray(inner)
    action = np.asarray(action)
    if (
        outer.shape != (SESSION_COUNT,)
        or inner.shape != (SESSION_COUNT,)
        or outer.dtype != np.uint8
        or inner.dtype != np.uint8
        or action.dtype != np.bool_
        or action.shape not in {(SESSION_COUNT,), (SESSION_COUNT, TURN_COUNT)}
        or set(np.unique(outer).tolist()) != set(range(OUTER_FOLDS))
        or set(np.unique(inner).tolist()) != set(range(OUTER_FOLDS))
    ):
        raise Rank1ReplayError("nested partition schema failed")
    outer_counts = [int(np.sum(outer == fold)) for fold in range(OUTER_FOLDS)]
    inner_counts = [int(np.sum(inner == fold)) for fold in range(OUTER_FOLDS)]
    if outer_counts != [400] * OUTER_FOLDS:
        raise Rank1ReplayError("outer folds are not exactly balanced")
    per_session_actions = (
        action.astype(np.int64)
        if action.ndim == 1
        else action.sum(axis=1, dtype=np.int64)
    )
    outer_train_inner_counts: list[list[int]] = []
    model_train_action_counts: list[list[int]] = []
    model_valid_action_counts: list[list[int]] = []
    for held in range(OUTER_FOLDS):
        train_sessions = outer != held
        outer_train_inner_counts.append(
            [
                int(np.sum(train_sessions & (inner == fold)))
                for fold in range(OUTER_FOLDS)
            ]
        )
        train_action_row: list[int] = []
        valid_action_row: list[int] = []
        for fold in range(OUTER_FOLDS):
            model_train = train_sessions & (inner != fold)
            model_valid = train_sessions & (inner == fold)
            train_action_row.append(int(per_session_actions[model_train].sum()))
            valid_action_row.append(int(per_session_actions[model_valid].sum()))
        model_train_action_counts.append(train_action_row)
        model_valid_action_counts.append(valid_action_row)
    if (
        any(value <= 0 for row in outer_train_inner_counts for value in row)
        or any(value <= 0 for row in model_train_action_counts for value in row)
        or any(value <= 0 for row in model_valid_action_counts for value in row)
    ):
        raise Rank1ReplayError("nested comparator action cell is empty")
    return {
        "valid": True,
        "outer_counts": outer_counts,
        "inner_counts": inner_counts,
        "global_inner_equal_counts_required": False,
        "outer_train_inner_counts": outer_train_inner_counts,
        "model_train_action_counts": model_train_action_counts,
        "model_valid_action_counts": model_valid_action_counts,
    }


def _official(metric: MetricValues, name: str) -> float:
    return float(metric.official()[name])


def _dual_ge(candidate: MetricValues, incumbent: MetricValues, name: str) -> bool:
    return getattr(candidate, name) >= getattr(incumbent, name) and _official(
        candidate, name
    ) >= _official(incumbent, name)


def _dual_gt(candidate: MetricValues, incumbent: MetricValues, name: str) -> bool:
    return getattr(candidate, name) > getattr(incumbent, name) and _official(
        candidate, name
    ) > _official(incumbent, name)


def _dual_le(candidate: MetricValues, incumbent: MetricValues, name: str) -> bool:
    return getattr(candidate, name) <= getattr(incumbent, name) and _official(
        candidate, name
    ) <= _official(incumbent, name)


def passes_local_gates(
    candidate: MetricValues,
    incumbent: MetricValues,
    fold_pairs: Sequence[tuple[MetricValues, MetricValues]],
    *,
    transition: Transition | None = None,
    fold_transitions: Sequence[Transition] | None = None,
    require_membership: bool = False,
) -> bool:
    numeric = bool(
        len(fold_pairs) == OUTER_FOLDS
        and _dual_gt(candidate, incumbent, "hit_rate_at_10")
        and _official(candidate, "hit_rate_at_10") > 0.9765
        and _dual_ge(candidate, incumbent, "mrr")
        and _dual_le(candidate, incumbent, "mttc")
        and _dual_gt(candidate, incumbent, "technical_score")
        and all(
            _dual_ge(cand, base_metric, "hit_rate_at_10")
            and _dual_ge(cand, base_metric, "mrr")
            and _dual_le(cand, base_metric, "mttc")
            and _dual_ge(cand, base_metric, "technical_score")
            for cand, base_metric in fold_pairs
        )
    )
    if not numeric:
        return False
    if transition is None or fold_transitions is None:
        return not require_membership
    return bool(
        transition.miss_to_hit >= 1
        and transition.hit_to_miss == 0
        and len(fold_transitions) == OUTER_FOLDS
        and all(row.net_hits >= 0 and row.hit_to_miss == 0 for row in fold_transitions)
    )


def passes_global_gates(
    candidate: MetricValues,
    incumbent: MetricValues,
    fold_pairs: Sequence[tuple[MetricValues, MetricValues]],
    dominance: Mapping[str, int] | None = None,
    exact_repeat: bool = True,
    *,
    transition: Transition | None = None,
    fold_transitions: Sequence[Transition] | None = None,
    require_membership: bool = False,
) -> bool:
    structural = bool(
        dominance is not None
        and set(dominance)
        == {
            "hit_to_miss",
            "later_first_hit",
            "same_turn_worse_rank",
            "earlier_hit_not_rank1",
            "new_hit_not_rank1",
            "reset_eligibility_mismatch",
        }
        and not any(int(value) for value in dominance.values())
    )
    hits = int(candidate.hit_rate_at_10 * candidate.sample_count)
    numeric = bool(
        exact_repeat
        and structural
        and len(fold_pairs) == OUTER_FOLDS
        and hits >= 1_983
        and _dual_gt(candidate, incumbent, "hit_rate_at_10")
        and _official(candidate, "hit_rate_at_10") > 0.991
        and _dual_ge(candidate, incumbent, "mrr")
        and _official(candidate, "mrr") >= 0.695795
        and _dual_le(candidate, incumbent, "mttc")
        and _official(candidate, "mttc") <= 2.869
        and _dual_gt(candidate, incumbent, "technical_score")
        and _official(candidate, "technical_score") > 0.866858
        and all(
            _dual_ge(cand, base_metric, "hit_rate_at_10")
            and _dual_ge(cand, base_metric, "mrr")
            and _dual_le(cand, base_metric, "mttc")
            and _dual_ge(cand, base_metric, "technical_score")
            for cand, base_metric in fold_pairs
        )
    )
    if not numeric:
        return False
    if transition is None or fold_transitions is None:
        return not require_membership
    return bool(
        transition.hit_to_miss == 0
        and len(fold_transitions) == OUTER_FOLDS
        and all(row.net_hits >= 0 and row.hit_to_miss == 0 for row in fold_transitions)
    )


_sha256_path = v12._sha256_path
_sha256_handle = v12._sha256_handle
_validate_environment = v12._validate_environment
_process_memory = v12._process_memory
_trace_baseline_rank_mismatches = v12._trace_baseline_rank_mismatches
_reproduce_nested_activation = v12._reproduce_nested_activation


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Rank1ReplayError("duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_hook)
    if not isinstance(value, dict):
        raise Rank1ReplayError("expected JSON object")
    return value


def _validate_preregistration() -> dict[str, Any]:
    if (
        not PREREG_PATH.is_file()
        or PREREG_PATH.is_symlink()
        or not COMPARABILITY_PATH.is_file()
        or COMPARABILITY_PATH.is_symlink()
    ):
        raise Rank1ReplayError("frozen protocol evidence is unavailable")
    raw = PREREG_PATH.read_bytes()
    prereg = _load_json_no_duplicates(PREREG_PATH)
    comparability_raw = COMPARABILITY_PATH.read_bytes()
    comparability = _load_json_no_duplicates(COMPARABILITY_PATH)
    outcome = prereg.get("outcome_protocol", {})
    score = prereg.get("score_priority_contract", {})
    observed = score.get("target_free_priority_observation", {})
    reproduction = prereg.get("frozen_comparator_reproduction", {})
    labels = prereg.get("label_and_partition_contract", {})
    resources = prereg.get("resource_and_privacy", {})
    choreography = prereg.get("checkpoint_choreography", {})
    checkpoint = prereg.get("benchmark_comparability_checkpoint", {})
    relative_output = OUTPUT_PATH.relative_to(ROOT).as_posix()
    if not (
        hashlib.sha256(raw).hexdigest() == PREREG_RAW_SHA256
        and _canonical_sha256(prereg) == PREREG_CANONICAL_SHA256
        and hashlib.sha256(comparability_raw).hexdigest()
        == COMPARABILITY_RAW_SHA256
        and _canonical_sha256(comparability) == COMPARABILITY_CANONICAL_SHA256
        and prereg.get("schema_version")
        == "small-ranker-score-priority-rank1-preregistration.v1"
        and prereg.get("status")
        == "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_OUTCOME"
        and prereg.get("experiment_id") == EXPERIMENT_ID
        and prereg.get("parent_commit") == COMPARABILITY_COMMIT
        and prereg.get("lineage_base_commit") == LINEAGE_BASE_COMMIT
        and checkpoint.get("commit") == COMPARABILITY_COMMIT
        and checkpoint.get("git_blob_oid_sha1") == COMPARABILITY_BLOB_OID
        and outcome.get("attach_count") == 1
        and outcome.get("completed_attach_count_before_run") == 0
        and outcome.get("fixed_output") == relative_output
        and outcome.get("output_not_cli_overridable") is True
        and outcome.get("result_schema_version") == SCHEMA_VERSION
        and score.get("priority_order")
        == "numpy.lexsort((original_raw_c100_ordinal, -float32_score)) without rounding, tolerance, normalization, softmax, blend, threshold, or refill"
        and observed.get("shape")
        == [SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT]
        and observed.get("dtype") == "uint8"
        and observed.get("order") == "C"
        and observed.get("ndarray_raw_sha256") == EXPECTED_PRIORITY_SHA256
        and observed.get("rows_with_at_least_one_exact_score_tie")
        == EXPECTED_PRIORITY_TIE_ROWS
        and observed.get("duplicate_score_excess")
        == EXPECTED_PRIORITY_DUPLICATE_EXCESS
        and observed.get("all_scores_finite") is True
        and observed.get("every_row_is_exact_permutation_of_uint8_0_through_99")
        is True
        and reproduction.get("fit_invocations_exact") == EXPECTED_FIT_INVOCATIONS
        and reproduction.get("inner_fit_invocations_exact") == EXPECTED_INNER_FITS
        and reproduction.get("outer_fit_invocations_exact") == EXPECTED_OUTER_FITS
        and reproduction.get("quantile_selection_invocations_exact")
        == EXPECTED_SELECTION_INVOCATIONS
        and reproduction.get("base_seed") == BASE_SEED
        and tuple(reproduction.get("expected_fold_quantiles", ()))
        == EXPECTED_FOLD_QUANTILES
        and reproduction.get("expected_activation_ndarray_raw_sha256")
        == EXPECTED_ACTIVATION_SHA256
        and labels.get("archive_open_count") == 1
        and labels.get("member_access_count") == len(LABEL_MEMBER_SPECS)
        and tuple(labels.get("member_access_order", ()))
        == tuple(name for name, _shape, _dtype in LABEL_MEMBER_SPECS)
        and labels.get("inner", {}).get("global_equal_counts_required") is False
        and resources.get("cached_replay_wall_seconds_maximum")
        == int(RESOURCE_SECONDS_MAXIMUM)
        and resources.get("peak_working_set_bytes_maximum")
        == RESOURCE_BYTES_MAXIMUM
        and resources.get("new_v2_13_fit_or_selection_allowed") is False
        and resources.get("frozen_comparator_fits_allowed_exact")
        == EXPECTED_FIT_INVOCATIONS
        and choreography.get("branch") == BRANCH
        and choreography.get("remote") == REMOTE
        and choreography.get("remote_url") == REMOTE_URL
        and choreography.get("remote_tracking_ref") == REMOTE_REF
        and frozen.RR_MULTIPLIER == 1.0
        and tuple(frozen.QUANTILES)
        == tuple(float(value) / 64.0 for value in range(64))
        and frozen.KEEP_QUANTILE == 1.0
    ):
        raise Rank1ReplayError("preregistration binding drifted")
    return {
        "preregistration_commit": PREREG_COMMIT,
        "comparability_commit": COMPARABILITY_COMMIT,
        "git_blob_oid": PREREG_BLOB_OID,
        "raw_sha256": PREREG_RAW_SHA256,
        "canonical_sha256": PREREG_CANONICAL_SHA256,
        "comparability": {
            "git_blob_oid": COMPARABILITY_BLOB_OID,
            "raw_sha256": COMPARABILITY_RAW_SHA256,
            "canonical_sha256": COMPARABILITY_CANONICAL_SHA256,
        },
    }


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Rank1ReplayError("Git checkpoint validation failed")
    return completed.stdout.strip()


def _commit_parent(commit: str) -> str:
    values = _git(("rev-list", "--parents", "-n", "1", commit)).split()
    if len(values) != 2 or values[0] != commit:
        raise Rank1ReplayError("commit must have exactly one parent")
    return values[1]


def _changed_paths(commit: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in _git(
            ("diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        ).splitlines()
        if line.strip()
    }


def _validate_git_checkpoint(implementation_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise Rank1ReplayError("implementation commit must be a full SHA")
    if _git(("status", "--porcelain=v1", "--untracked-files=all")):
        raise Rank1ReplayError("tracked or untracked worktree state is not clean")
    head = _git(("rev-parse", "HEAD"))
    branch = _git(("symbolic-ref", "--short", "HEAD"))
    remote_url = _git(("remote", "get-url", REMOTE))
    remote_head = _git(("rev-parse", REMOTE_REF))
    prereg_relative = PREREG_PATH.relative_to(ROOT).as_posix()
    comparability_relative = COMPARABILITY_PATH.relative_to(ROOT).as_posix()
    if not (
        head == implementation_commit
        and branch == BRANCH
        and remote_url == REMOTE_URL
        and remote_head == head
        and _commit_parent(head) == PREREG_COMMIT
        and _commit_parent(PREREG_COMMIT) == COMPARABILITY_COMMIT
        and _commit_parent(COMPARABILITY_COMMIT) == LINEAGE_BASE_COMMIT
        and _changed_paths(head) == IMPLEMENTATION_PATHS
        and _changed_paths(PREREG_COMMIT) == PREREG_PATH_SET
        and _changed_paths(COMPARABILITY_COMMIT) == COMPARABILITY_PATH_SET
        and _git(("rev-parse", PREREG_COMMIT + ":" + prereg_relative))
        == PREREG_BLOB_OID
        and _git(
            ("rev-parse", COMPARABILITY_COMMIT + ":" + comparability_relative)
        )
        == COMPARABILITY_BLOB_OID
    ):
        raise Rank1ReplayError("commit choreography drifted")

    prereg = _load_json_no_duplicates(PREREG_PATH)
    pinned = prereg.get("pinned_source_blobs", {})
    default_commit = pinned.get("default_commit")
    if default_commit != COMPARABILITY_COMMIT:
        raise Rank1ReplayError("pinned source default commit drifted")
    source_blobs: dict[str, dict[str, str]] = {}
    for registered_name, specification in pinned.items():
        if registered_name == "default_commit":
            continue
        if not isinstance(specification, Mapping):
            raise Rank1ReplayError("pinned source specification drifted")
        commit = str(specification.get("commit", default_commit))
        path = registered_name.split("@", 1)[0]
        expected_blob = str(specification.get("git_blob_oid_sha1", ""))
        actual_blob = _git(("rev-parse", commit + ":" + path))
        if actual_blob != expected_blob:
            raise Rank1ReplayError("pinned source Git blob drifted")
        source_blobs[registered_name] = {
            "commit": commit,
            "path": path,
            "git_blob_oid": actual_blob,
        }
    artifact = prereg.get("deployable_full_artifact", {})
    artifact_path = str(artifact.get("path", ""))
    artifact_commit = str(artifact.get("source_commit", ""))
    artifact_blob = _git(("rev-parse", artifact_commit + ":" + artifact_path))
    if artifact_blob != artifact.get("git_blob_oid_sha1"):
        raise Rank1ReplayError("deployable artifact Git blob drifted")
    implementation_files: dict[str, dict[str, Any]] = {}
    for relative in sorted(IMPLEMENTATION_PATHS):
        digest, size = _sha256_path(ROOT / relative)
        implementation_files[relative] = {
            "git_blob_oid": _git(("rev-parse", head + ":" + relative)),
            "sha256": digest,
            "bytes": size,
        }
    return {
        "implementation_commit": head,
        "preregistration_commit": PREREG_COMMIT,
        "comparability_commit": COMPARABILITY_COMMIT,
        "lineage_base_commit": LINEAGE_BASE_COMMIT,
        "branch": branch,
        "remote": REMOTE,
        "remote_url": remote_url,
        "remote_tracking_ref": REMOTE_REF,
        "remote_equal": True,
        "clean_including_untracked": True,
        "exact_commit_chain": True,
        "exact_changed_path_sets": True,
        "pinned_source_blobs": source_blobs,
        "deployable_artifact_blob": artifact_blob,
        "implementation_files": implementation_files,
    }


def _load_target_free_inputs() -> TargetFreeInputs:
    return v12._load_target_free_inputs()


def _validate_source_snapshots(
    snapshots: Mapping[str, tuple[int, int]],
) -> bool:
    return v12._validate_source_snapshots(snapshots)


def _perform_target_free_preflight(
    implementation_commit: str | None,
) -> TargetFreePreflight:
    environment = _validate_environment()
    preregistration = _validate_preregistration()
    git = (
        _validate_git_checkpoint(implementation_commit)
        if implementation_commit is not None
        else {"formal_git_gate_skipped": True}
    )
    _check_output_components(OUTPUT_PATH, ROOT)
    target_free = _load_target_free_inputs()
    if not (
        target_free.projected_features.shape
        == (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT, FEATURE_COUNT)
        and target_free.projected_features.dtype == np.float32
        and not target_free.projected_features.flags.writeable
        and target_free.oof_scores.shape
        == (SESSION_COUNT, TURN_COUNT, CANDIDATE_COUNT)
        and target_free.oof_scores.dtype == np.float32
        and not target_free.oof_scores.flags.writeable
        and np.isfinite(target_free.oof_scores).all()
        and target_free.chosen.shape == (SESSION_COUNT, TURN_COUNT)
        and hashlib.sha256(
            np.ascontiguousarray(target_free.chosen).tobytes()
        ).hexdigest()
        == EXPECTED_CHOSEN_SHA256
        and len(target_free.traces) == SESSION_COUNT
        and all(len(turns) == TURN_COUNT for turns in target_free.traces)
        and _validate_source_snapshots(target_free.source_snapshots)
    ):
        raise Rank1ReplayError("target-free preflight surface drifted")
    priority_audit = _priority_audit(target_free.oof_scores)
    working_set, peak = _process_memory()
    if not (
        0 < working_set <= RESOURCE_BYTES_MAXIMUM
        and 0 < peak <= RESOURCE_BYTES_MAXIMUM
    ):
        raise Rank1ReplayError("pre-receipt memory gate failed")
    return TargetFreePreflight(
        environment,
        preregistration,
        git,
        target_free,
        priority_audit,
        (working_set, peak),
    )


def preflight_only(
    implementation_commit: str | None = None,
) -> TargetFreePreflight:
    """Run the target-free checks without creating or consuming the receipt."""

    return _perform_target_free_preflight(implementation_commit)


def _check_output_components(path: Path, root: Path) -> None:
    root_resolved = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    parent_resolved = candidate.parent.resolve(strict=False)
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise Rank1ReplayError("one-shot output escapes the worktree")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise Rank1ReplayError("one-shot output escapes the worktree") from error
    current = root
    for part in relative.parts[:-1]:
        if part in {"", ".", ".."}:
            raise Rank1ReplayError("one-shot output contains unsafe traversal")
        current = current / part
        if current.exists() and current.is_symlink():
            raise Rank1ReplayError("one-shot output has a symlink component")
    if candidate.exists() or candidate.is_symlink():
        raise Rank1ReplayError("one-shot output is already consumed")


def _write_receipt_payload(
    handle: BinaryIO, value: Mapping[str, Any]
) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    handle.seek(0)
    handle.truncate(0)
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
    return len(payload), hashlib.sha256(payload).hexdigest()


def _invalid_receipt_payload(
    implementation_commit: str, error: BaseException
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "INVALID_ONE_SHOT_CONSUMED",
        "implementation_commit": implementation_commit,
        "error_class": type(error).__name__,
        "rerun_forbidden": True,
    }


def _write_descriptor_payload(
    descriptor: int, value: Mapping[str, Any]
) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise OSError("short receipt descriptor write")
        view = view[count:]
    os.fsync(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _safe_close(handle: BinaryIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass


def open_one_shot_receipt(
    path: Path, root: Path, implementation_commit: str
) -> BinaryIO:
    _check_output_components(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_output_components(path, root)
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError as error:
        raise Rank1ReplayError("one-shot output is already consumed") from error
    marker = {
        "schema_version": "small-ranker-score-priority-one-shot-marker.v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_commit": implementation_commit,
        "status": "CONSUMED_PENDING_RERUN_FORBIDDEN",
    }
    try:
        handle = os.fdopen(descriptor, "r+b")
    except Exception as error:
        try:
            _write_descriptor_payload(
                descriptor, _invalid_receipt_payload(implementation_commit, error)
            )
        except Exception:
            pass
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise Rank1ReplayConsumedError(
            "v2.13 receipt wrapping failed after consumption; rerun is forbidden"
        ) from error
    try:
        _write_receipt_payload(handle, marker)
    except Exception as error:
        try:
            _write_receipt_payload(
                handle, _invalid_receipt_payload(implementation_commit, error)
            )
        except Exception:
            pass
        finally:
            _safe_close(handle)
        raise Rank1ReplayConsumedError(
            "v2.13 receipt durability failed after consumption; rerun is forbidden"
        ) from error
    return handle


def _label_stat_identity(handle: BinaryIO) -> tuple[int, int, int]:
    stat = os.fstat(handle.fileno())
    return int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0))


def _result_privacy_scan(result: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "session_id",
        "sample_id",
        "target_id",
        "target_asin",
        "ground_truth",
        "per_session",
        "membership_vector",
        "positive_index",
        "eligible_from",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys & {str(key) for key in value}:
                raise Rank1ReplayError("result contains a forbidden outcome key")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            if len(value) >= SESSION_COUNT:
                raise Rank1ReplayError("result contains a per-session vector")
            for child in value:
                walk(child)
        elif isinstance(value, np.ndarray):
            raise Rank1ReplayError("result contains an ndarray")

    walk(result)
    payload = _canonical_bytes(result)
    if ASIN_SHAPE_RE.search(payload):
        raise Rank1ReplayError("result contains a product identifier")


def _bundle_exact_repeat(first: ReplayBundle, repeat: ReplayBundle) -> bool:
    arrays_equal = all(
        np.array_equal(getattr(first, name), getattr(repeat, name))
        for name in (
            "baseline_pages",
            "candidate_pages",
            "changed",
            "last_reset_turn",
        )
    )
    return bool(
        arrays_equal
        and first.identity == repeat.identity
        and first.structural == repeat.structural
        and _canonical_sha256(first.identity)
        == _canonical_sha256(repeat.identity)
    )


def _matches_official(metric: MetricValues, expected: Mapping[str, float]) -> bool:
    actual = metric.official()
    return all(actual.get(name) == value for name, value in expected.items())


def _validate_local_page_identities(
    local: ReplayBundle,
    local_repeat: ReplayBundle,
    score: ReplayBundle,
) -> dict[str, Any]:
    """Validate target-free page identities before any outcome metric is built."""

    local_arrays_exact = all(
        local.identity.get(name, {}).get("raw_sha256") == expected
        for name, expected in EXPECTED_LOCAL_ARRAY_SHA256.items()
    )
    local_identity_sha = _canonical_sha256(local.identity)
    local_repeat_sha = _canonical_sha256(local_repeat.identity)
    score_baseline_ascii = str(
        score.identity.get("baseline_ascii_page_sha256", "")
    )
    if not (
        local_arrays_exact
        and local_identity_sha == EXPECTED_LOCAL_IDENTITY_SHA256
        and local_repeat_sha == EXPECTED_LOCAL_IDENTITY_SHA256
        and _bundle_exact_repeat(local, local_repeat)
        and score_baseline_ascii == EXPECTED_BASELINE_ASCII_SHA256
        and np.array_equal(score.baseline_pages, local.baseline_pages)
    ):
        raise Rank1ReplayError("frozen local page identity drifted")
    return {
        "all_four_local_array_hashes_exact": local_arrays_exact,
        "local_identity_canonical_sha256": local_identity_sha,
        "local_exact_repeat": True,
        "baseline_ascii_page_sha256": score_baseline_ascii,
    }


def _validate_comparator_identities(
    local: ReplayBundle,
    local_repeat: ReplayBundle,
    grace: ReplayBundle,
    grace_repeat: ReplayBundle,
    score: ReplayBundle,
    baseline_metric: MetricValues,
    local_metric: MetricValues,
    local_folds: Sequence[MetricValues],
    grace_metric: MetricValues,
    grace_folds: Sequence[MetricValues],
) -> dict[str, Any]:
    local_page_audit = _validate_local_page_identities(
        local, local_repeat, score
    )
    grace_baseline_ascii = str(
        grace.identity.get("baseline_ascii_page_sha256", "")
    )
    grace_candidate_ascii = str(
        grace.identity.get("candidate_ascii_page_sha256", "")
    )
    valid = bool(
        _bundle_exact_repeat(grace, grace_repeat)
        and grace_baseline_ascii == EXPECTED_BASELINE_ASCII_SHA256
        and grace_candidate_ascii == EXPECTED_GRACE_ASCII_SHA256
        and np.array_equal(score.baseline_pages, grace.baseline_pages)
        and _matches_official(baseline_metric, EXPECTED_BASELINE_OFFICIAL)
        and _matches_official(local_metric, EXPECTED_LOCAL_OFFICIAL)
        and len(local_folds) == OUTER_FOLDS
        and all(
            _matches_official(metric, expected)
            for metric, expected in zip(local_folds, EXPECTED_LOCAL_FOLDS)
        )
        and _matches_official(grace_metric, EXPECTED_GRACE_OFFICIAL)
        and len(grace_folds) == OUTER_FOLDS
        and all(
            _matches_official(metric, expected)
            for metric, expected in zip(grace_folds, EXPECTED_GRACE_FOLDS)
        )
    )
    if not valid:
        raise Rank1ReplayError("frozen comparator identity or metric drifted")
    return {
        **local_page_audit,
        "grace_candidate_ascii_page_sha256": grace_candidate_ascii,
        "grace_exact_repeat": True,
        "official_global_and_fold_metrics_exact": True,
    }


def _metric_list(
    state: Mapping[str, np.ndarray], outer: np.ndarray
) -> list[MetricValues]:
    return [
        metric_values(state, np.asarray(outer) == fold)
        for fold in range(OUTER_FOLDS)
    ]


def _transition_list(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    changed: np.ndarray,
    outer: np.ndarray,
) -> list[Transition]:
    return [
        transition_metrics(before, after, changed, np.asarray(outer) == fold)
        for fold in range(OUTER_FOLDS)
    ]


def run(implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    receipt: BinaryIO | None = None
    label_handle: BinaryIO | None = None
    consumed = False
    final_written = False
    try:
        preflight = _perform_target_free_preflight(implementation_commit)
        target_free = preflight.target_free
        receipt = open_one_shot_receipt(OUTPUT_PATH, ROOT, implementation_commit)
        consumed = True

        # The first label path operation in the formal run is deliberately here,
        # after the durable pending receipt returned above.
        if not LABEL_PATH.is_file() or LABEL_PATH.is_symlink():
            raise Rank1ReplayError("sealed label archive is unavailable")
        label_handle = LABEL_PATH.open("rb")
        label_identity_start = _label_stat_identity(label_handle)
        label_start_sha, label_start_bytes = _sha256_handle(label_handle)
        if label_start_sha != LABEL_SHA256 or label_start_bytes != LABEL_BYTES:
            raise Rank1ReplayError("sealed label archive identity drifted")
        label_handle.seek(0)
        outcomes = load_outcomes_from_open_handle(label_handle)

        baseline_session_hit = derive_baseline_session_hit(
            outcomes.baseline_rank, outcomes.eligible_from
        )
        labels = {
            "baseline_rank": outcomes.baseline_rank,
            "positive_index": outcomes.positive_index,
            "eligible_from": outcomes.eligible_from,
            "outer_fold": outcomes.outer_fold,
            "inner_fold": outcomes.inner_fold,
            "baseline_session_hit": baseline_session_hit,
        }
        surface = frozen._action_surface(
            target_free.projected_features,
            target_free.oof_scores,
            labels,
        )
        if not np.array_equal(surface.chosen, target_free.chosen):
            raise Rank1ReplayError("post-attach chosen surface drifted")
        partition = audit_nested_partition(
            outcomes.outer_fold, outcomes.inner_fold, surface.action
        )
        activation, selections, comparator_reproduction = (
            _reproduce_nested_activation(surface, labels, seed=BASE_SEED)
        )
        activation_sha = hashlib.sha256(
            np.ascontiguousarray(activation).tobytes()
        ).hexdigest()
        if activation_sha != EXPECTED_ACTIVATION_SHA256:
            raise Rank1ReplayError("frozen activation identity drifted")

        score = replay_score_priority_pages(
            target_free.traces,
            target_free.oof_scores,
            surface.chosen,
            activation,
            target_free.versions,
            measure_timing=True,
        )
        score_repeat = replay_score_priority_pages(
            target_free.traces,
            target_free.oof_scores,
            surface.chosen,
            activation,
            target_free.versions,
        )
        local = v12.replay_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
            measure_timing=False,
        )
        local_repeat = v12.replay_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
            measure_timing=False,
        )
        if not (
            score.baseline_pages.dtype == np.int16
            and score.candidate_pages.dtype == np.int16
            and not score.baseline_pages.flags.writeable
            and not score.candidate_pages.flags.writeable
            and np.all((score.baseline_pages >= 0) & (score.baseline_pages < 100))
            and np.all((score.candidate_pages >= 0) & (score.candidate_pages < 100))
        ):
            raise Rank1ReplayError("score replay ordinal surface drifted")

        # This preregistered identity check precedes every target-derived state
        # or metric.  Identity is a valid consumed No-Go, not an implementation
        # failure and not another look at the outcome.
        local_page_audit = _validate_local_page_identities(
            local, local_repeat, score
        )
        score_equals_local = bool(
            np.array_equal(score.candidate_pages, local.candidate_pages)
        )
        if score_equals_local:
            score_exact_repeat = _bundle_exact_repeat(score, score_repeat)
            reset_mismatches = int(
                np.sum(score.last_reset_turn != outcomes.eligible_from)
            )
            trace_rank_mismatches = _trace_baseline_rank_mismatches(
                target_free.traces, outcomes
            )
            baseline_hit_count = int(baseline_session_hit.sum())
            identity_structural_ok = bool(
                score_exact_repeat
                and baseline_hit_count == 1_895
                and trace_rank_mismatches == 0
                and reset_mismatches == 0
                and score.structural.get("ranks_2_to_10_byte_identical") is True
                and score.structural.get("removed_rank1_already_served") is True
                and score.structural.get("inserted_rank1_unseen_legal_tail")
                is True
                and score.structural.get("reset_pages_identity") is True
            )
            if not identity_structural_ok:
                raise Rank1ReplayError(
                    "identity short-circuit structural audit failed"
                )

            label_end_sha, label_end_bytes = _sha256_handle(label_handle)
            label_identity_end = _label_stat_identity(label_handle)
            if not (
                label_end_sha == label_start_sha
                and label_end_bytes == label_start_bytes
                and label_identity_end == label_identity_start
            ):
                raise Rank1ReplayError(
                    "sealed label archive changed during identity attach"
                )
            label_handle.close()
            label_handle = None
            if not _validate_source_snapshots(target_free.source_snapshots):
                raise Rank1ReplayError(
                    "target-free input changed during identity run"
                )
            working_set_after, peak_after = _process_memory()
            wall_seconds = time.perf_counter() - started
            if not (
                0 < working_set_after <= RESOURCE_BYTES_MAXIMUM
                and 0 < peak_after <= RESOURCE_BYTES_MAXIMUM
                and wall_seconds <= RESOURCE_SECONDS_MAXIMUM
            ):
                raise Rank1ReplayError(
                    "identity cached replay resource gate failed"
                )

            status = "NO_GO_SCORE_PRIORITY_IDENTICAL_TO_LOCAL_INCUMBENT"
            decision_payload = {
                "status": status,
                "structural_integrity": True,
                "local_gate_pass": False,
                "global_gate_pass": False,
                "score_equals_local_identity_no_go": True,
                "promote_to_default_off_runtime_patch": False,
            }
            score_identity_sha = _canonical_sha256(score.identity)
            result = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "status": status,
                "environment": dict(preflight.environment),
                "git": dict(preflight.git),
                "preregistration": dict(preflight.preregistration),
                "sources": {
                    "projected_features": {
                        "path": str(PROJECTED_FEATURES_PATH),
                        "bytes": PROJECTED_FEATURES_BYTES,
                        "sha256": PROJECTED_FEATURES_SHA256,
                    },
                    "oof_scores": {
                        "path": str(OOF_SCORES_PATH),
                        "bytes": OOF_SCORES_BYTES,
                        "sha256": OOF_SCORES_SHA256,
                    },
                    "chosen_raw_sha256": EXPECTED_CHOSEN_SHA256,
                    "activation_raw_sha256": activation_sha,
                    "priority": dict(preflight.priority_audit),
                },
                "candidate_recall_frozen_context": {
                    "C10": 0.9475,
                    "C20": 0.9715,
                    "C50": 0.991,
                    "C100": 0.993,
                    "not_recomputed_in_this_outcome": True,
                },
                "target_free_reset": dict(target_free.reset_audit),
                "replay": {
                    "policy": "SCORE_PRIORITY_RANK1_REPLACEMENT",
                    "identity": dict(score.identity),
                    "identity_sha256": score_identity_sha,
                    "structural": dict(score.structural),
                },
                "exact_repeat": {
                    "equal": True,
                    "score_equal": True,
                    "score_first_identity_sha256": score_identity_sha,
                    "score_repeat_identity_sha256": _canonical_sha256(
                        score_repeat.identity
                    ),
                    "local_equal": True,
                },
                "comparator_reconstruction": {
                    **local_page_audit,
                    "frozen_v1_9": comparator_reproduction,
                    "outer_selections": selections,
                    "grace_reconstruction_skipped_for_identity_no_go": True,
                },
                "partition": partition,
                "label_attach": {
                    "archive_open_count": 1,
                    "member_access_count": len(LABEL_MEMBER_SPECS),
                    "member_access_order": [
                        name for name, _shape, _dtype in LABEL_MEMBER_SPECS
                    ],
                    "archive_sha256": label_start_sha,
                    "archive_bytes": label_start_bytes,
                    "same_handle_start_end": True,
                    "baseline_p11_hit_count": baseline_hit_count,
                    "trace_baseline_rank_mismatches": trace_rank_mismatches,
                    "last_reset_mismatches": reset_mismatches,
                },
                "dominance_audit_vs_v1_9": {
                    "hit_to_miss": 0,
                    "later_first_hit": 0,
                    "same_turn_worse_rank": 0,
                    "earlier_hit_not_rank1": 0,
                    "new_hit_not_rank1": 0,
                    "reset_eligibility_mismatch": 0,
                    "inherited_from_exact_local_page_identity": True,
                },
                "comparison": {
                    "identity_short_circuit": {
                        "candidate_equals_local_pages": True,
                        "target_metrics_computed": False,
                        "miss_to_hit": 0,
                        "hit_to_miss": 0,
                        "net_hits": 0,
                        "fold_net_hits": [0, 0, 0, 0, 0],
                        "local_incumbent_official_6dp": dict(
                            EXPECTED_LOCAL_OFFICIAL
                        ),
                        "candidate_official_6dp": dict(
                            EXPECTED_LOCAL_OFFICIAL
                        ),
                        "candidate_fold_official_6dp": [
                            dict(row) for row in EXPECTED_LOCAL_FOLDS
                        ],
                        "global_incumbent_official_6dp": dict(
                            EXPECTED_GRACE_OFFICIAL
                        ),
                    }
                },
                "privacy": {
                    "split": "train_explore_shared_cohort_oof",
                    "receipt_preceded_label_existence_stat_open_hash_and_access": True,
                    "target_runtime_features": 0,
                    "target_metrics_computed": False,
                    "proxy_opened": False,
                    "new_fit_or_selection": False,
                    "per_session_outcomes_serialized": False,
                    "product_identifiers_serialized": False,
                    "agent_or_full_evaluator_started": False,
                    "forbidden_split_opened": False,
                },
                "resource": {
                    "wall_seconds": round(wall_seconds, 6),
                    "working_set_before_receipt_bytes": (
                        preflight.memory_before_receipt[0]
                    ),
                    "peak_before_receipt_bytes": (
                        preflight.memory_before_receipt[1]
                    ),
                    "working_set_final_bytes": working_set_after,
                    "peak_final_bytes": peak_after,
                    "policy_transform_latency": dict(score.timing),
                    "workers": 1,
                    "gpu_used": False,
                    "gpu_peak_bytes": 0,
                },
                "decision": {
                    **decision_payload,
                    "strict_decision_sha256": _canonical_sha256(
                        decision_payload
                    ),
                    "served_default": "off",
                    "fallback_order": [
                        "SR-V2.12-FIXED-TWO-PAGE-GRACE",
                        "v1.9",
                        "P11",
                        "R08",
                    ],
                    "shared_cohort_not_independent_confirmation": True,
                },
                "receipt": {
                    "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                    "durable": True,
                    "self_hash_omitted": True,
                    "rerun_forbidden": True,
                },
            }
            _result_privacy_scan(result)
            _write_receipt_payload(receipt, result)
            final_written = True
            _safe_close(receipt)
            receipt = None
            return result

        grace = replay_grace_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
        )
        grace_repeat = replay_grace_pages(
            target_free.traces,
            surface.chosen,
            activation,
            target_free.versions,
        )

        baseline_state = state_from_positive_index(
            score.baseline_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        local_state = state_from_positive_index(
            local.candidate_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        grace_state = state_from_positive_index(
            grace.candidate_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        score_state = state_from_positive_index(
            score.candidate_pages,
            outcomes.positive_index,
            outcomes.eligible_from,
        )
        all_sessions = np.ones(SESSION_COUNT, dtype=bool)
        baseline_metric = metric_values(baseline_state, all_sessions)
        local_metric = metric_values(local_state, all_sessions)
        grace_metric = metric_values(grace_state, all_sessions)
        score_metric = metric_values(score_state, all_sessions)
        local_fold_metrics = _metric_list(local_state, outcomes.outer_fold)
        grace_fold_metrics = _metric_list(grace_state, outcomes.outer_fold)
        score_fold_metrics = _metric_list(score_state, outcomes.outer_fold)

        # Comparator reconstruction is a hard precondition.  No candidate-vs-
        # comparator transition or promotion gate is evaluated before this call.
        comparator_audit = _validate_comparator_identities(
            local,
            local_repeat,
            grace,
            grace_repeat,
            score,
            baseline_metric,
            local_metric,
            local_fold_metrics,
            grace_metric,
            grace_fold_metrics,
        )

        score_exact_repeat = _bundle_exact_repeat(score, score_repeat)
        all_exact_repeat = bool(
            score_exact_repeat
            and comparator_audit["local_exact_repeat"]
            and comparator_audit["grace_exact_repeat"]
        )
        local_changed = np.any(
            local.candidate_pages != score.candidate_pages, axis=2
        )
        global_changed = np.any(
            grace.candidate_pages != score.candidate_pages, axis=2
        )
        v19_transition = transition_metrics(
            baseline_state, score_state, score.changed, all_sessions
        )
        local_transition = transition_metrics(
            local_state, score_state, local_changed, all_sessions
        )
        global_transition = transition_metrics(
            grace_state, score_state, global_changed, all_sessions
        )
        local_fold_transitions = _transition_list(
            local_state,
            score_state,
            local_changed,
            outcomes.outer_fold,
        )
        global_fold_transitions = _transition_list(
            grace_state,
            score_state,
            global_changed,
            outcomes.outer_fold,
        )
        dominance = dominance_audit(baseline_state, score_state)
        reset_mismatches = int(
            np.sum(score.last_reset_turn != outcomes.eligible_from)
        )
        dominance["reset_eligibility_mismatch"] = reset_mismatches
        trace_rank_mismatches = _trace_baseline_rank_mismatches(
            target_free.traces, outcomes
        )
        baseline_hit_count = int(baseline_session_hit.sum())
        structural_ok = bool(
            all_exact_repeat
            and baseline_hit_count == 1_895
            and trace_rank_mismatches == 0
            and not any(dominance.values())
            and score.structural.get("ranks_2_to_10_byte_identical") is True
            and score.structural.get("removed_rank1_already_served") is True
            and score.structural.get("inserted_rank1_unseen_legal_tail") is True
            and score.structural.get("reset_pages_identity") is True
        )
        if not structural_ok:
            raise Rank1ReplayError(
                "score-priority structural audit failed"
            )
        local_pass = bool(
            passes_local_gates(
                score_metric,
                local_metric,
                list(zip(score_fold_metrics, local_fold_metrics)),
                transition=local_transition,
                fold_transitions=local_fold_transitions,
                require_membership=True,
            )
        )
        global_pass = bool(
            local_pass
            and passes_global_gates(
                score_metric,
                grace_metric,
                list(zip(score_fold_metrics, grace_fold_metrics)),
                dominance,
                all_exact_repeat,
                transition=global_transition,
                fold_transitions=global_fold_transitions,
                require_membership=True,
            )
        )
        if global_pass:
            status = "GLOBAL_GO_DEFAULT_OFF_RUNTIME_INTEGRATION"
        elif local_pass:
            status = "LOCAL_GO_GLOBAL_NO_GO"
        else:
            status = "NO_GO_CLOSE_SCORE_PRIORITY_RANK1"

        label_end_sha, label_end_bytes = _sha256_handle(label_handle)
        label_identity_end = _label_stat_identity(label_handle)
        if not (
            label_end_sha == label_start_sha
            and label_end_bytes == label_start_bytes
            and label_identity_end == label_identity_start
        ):
            raise Rank1ReplayError("sealed label archive changed during attach")
        label_handle.close()
        label_handle = None
        if not _validate_source_snapshots(target_free.source_snapshots):
            raise Rank1ReplayError("target-free input changed during run")

        working_set_after, peak_after = _process_memory()
        wall_seconds = time.perf_counter() - started
        if not (
            0 < working_set_after <= RESOURCE_BYTES_MAXIMUM
            and 0 < peak_after <= RESOURCE_BYTES_MAXIMUM
            and wall_seconds <= RESOURCE_SECONDS_MAXIMUM
        ):
            raise Rank1ReplayError("formal cached replay resource gate failed")

        local_fold_reports = [
            {"fold": fold, **transition.report()}
            for fold, transition in enumerate(local_fold_transitions)
        ]
        global_fold_reports = [
            {"fold": fold, **transition.report()}
            for fold, transition in enumerate(global_fold_transitions)
        ]
        decision_payload = {
            "status": status,
            "structural_integrity": structural_ok,
            "local_gate_pass": local_pass,
            "global_gate_pass": global_pass,
            "score_equals_local_identity_no_go": score_equals_local,
            "promote_to_default_off_runtime_patch": global_pass,
        }
        score_identity_sha = _canonical_sha256(score.identity)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "environment": dict(preflight.environment),
            "git": dict(preflight.git),
            "preregistration": dict(preflight.preregistration),
            "sources": {
                "projected_features": {
                    "path": str(PROJECTED_FEATURES_PATH),
                    "bytes": PROJECTED_FEATURES_BYTES,
                    "sha256": PROJECTED_FEATURES_SHA256,
                },
                "oof_scores": {
                    "path": str(OOF_SCORES_PATH),
                    "bytes": OOF_SCORES_BYTES,
                    "sha256": OOF_SCORES_SHA256,
                },
                "chosen_raw_sha256": EXPECTED_CHOSEN_SHA256,
                "activation_raw_sha256": activation_sha,
                "priority": dict(preflight.priority_audit),
            },
            "candidate_recall_frozen_context": {
                "C10": 0.9475,
                "C20": 0.9715,
                "C50": 0.991,
                "C100": 0.993,
                "not_recomputed_in_this_outcome": True,
            },
            "target_free_reset": dict(target_free.reset_audit),
            "replay": {
                "policy": "SCORE_PRIORITY_RANK1_REPLACEMENT",
                "identity": dict(score.identity),
                "identity_sha256": score_identity_sha,
                "structural": dict(score.structural),
                "raw_c100_ordinals_int16_read_only": True,
            },
            "exact_repeat": {
                "equal": all_exact_repeat,
                "score_equal": score_exact_repeat,
                "score_first_identity_sha256": score_identity_sha,
                "score_repeat_identity_sha256": _canonical_sha256(
                    score_repeat.identity
                ),
                "local_equal": comparator_audit["local_exact_repeat"],
                "grace_equal": comparator_audit["grace_exact_repeat"],
            },
            "comparator_reconstruction": {
                **comparator_audit,
                "frozen_v1_9": comparator_reproduction,
                "outer_selections": selections,
            },
            "partition": partition,
            "label_attach": {
                "archive_open_count": 1,
                "member_access_count": len(LABEL_MEMBER_SPECS),
                "member_access_order": [
                    name for name, _shape, _dtype in LABEL_MEMBER_SPECS
                ],
                "archive_sha256": label_start_sha,
                "archive_bytes": label_start_bytes,
                "same_handle_start_end": True,
                "baseline_p11_hit_count": baseline_hit_count,
                "trace_baseline_rank_mismatches": trace_rank_mismatches,
                "last_reset_mismatches": reset_mismatches,
            },
            "dominance_audit_vs_v1_9": dominance,
            "comparison": {
                "v1_9_structural": v19_transition.report(),
                "local_v2_12": {
                    "global": local_transition.report(),
                    "folds": local_fold_reports,
                    "gate_pass": local_pass,
                },
                "global_fixed_two_page_grace": {
                    "global": global_transition.report(),
                    "folds": global_fold_reports,
                    "gate_pass": global_pass,
                },
            },
            "privacy": {
                "split": "train_explore_shared_cohort_oof",
                "receipt_preceded_label_existence_stat_open_hash_and_access": True,
                "target_runtime_features": 0,
                "proxy_opened": False,
                "new_fit_or_selection": False,
                "per_session_outcomes_serialized": False,
                "product_identifiers_serialized": False,
                "agent_or_full_evaluator_started": False,
                "forbidden_split_opened": False,
            },
            "resource": {
                "wall_seconds": round(wall_seconds, 6),
                "working_set_before_receipt_bytes": preflight.memory_before_receipt[0],
                "peak_before_receipt_bytes": preflight.memory_before_receipt[1],
                "working_set_final_bytes": working_set_after,
                "peak_final_bytes": peak_after,
                "policy_transform_latency": dict(score.timing),
                "workers": 1,
                "gpu_used": False,
                "gpu_peak_bytes": 0,
            },
            "decision": {
                **decision_payload,
                "strict_decision_sha256": _canonical_sha256(decision_payload),
                "served_default": "off",
                "fallback_order": [
                    "SR-V2.12-FIXED-TWO-PAGE-GRACE",
                    "v1.9",
                    "P11",
                    "R08",
                ],
                "shared_cohort_not_independent_confirmation": True,
            },
            "receipt": {
                "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "durable": True,
                "self_hash_omitted": True,
                "rerun_forbidden": True,
            },
        }
        _result_privacy_scan(result)
        _write_receipt_payload(receipt, result)
        final_written = True
        _safe_close(receipt)
        receipt = None
        return result
    except Exception as error:
        _safe_close(label_handle)
        label_handle = None
        if consumed and receipt is not None and not final_written:
            invalid = _invalid_receipt_payload(implementation_commit, error)
            try:
                _write_receipt_payload(receipt, invalid)
            except Exception:
                pass
            finally:
                _safe_close(receipt)
                receipt = None
            raise Rank1ReplayConsumedError(
                "v2.13 one-shot was consumed; inspect the durable invalid receipt"
            ) from error
        raise
    finally:
        _safe_close(label_handle)
        _safe_close(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="Exact clean pushed implementation commit bound before label access.",
    )
    args = parser.parse_args()
    result = run(str(args.implementation_commit))
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparison": result["comparison"],
                "dominance_audit_vs_v1_9": result[
                    "dominance_audit_vs_v1_9"
                ],
                "resource": result["resource"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
