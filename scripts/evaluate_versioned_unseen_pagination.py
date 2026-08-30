"""Replay version-scoped unseen-first serving over the frozen v1.9 OOF policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_p12_counterfactual_router as trace_source  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-versioned-unseen-pagination.v1"
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
EXPECTED_BASELINE = {
    "hit_rate_at_10": 0.9715,
    "mrr": 0.676861,
    "mttc": 3.056,
    "technical_score": 0.847688,
}


class PaginationReplayError(RuntimeError):
    pass


def stable_unseen_first(
    order: Sequence[str], served: set[str], top_k: int = 10
) -> tuple[str, ...]:
    """Return a stable unseen-first page, falling back to seen items if required."""

    if top_k <= 0 or len(order) < top_k or len(order) != len(set(order)):
        raise PaginationReplayError("invalid ranked order")
    unseen = [identifier for identifier in order if identifier not in served]
    seen = [identifier for identifier in order if identifier in served]
    page = tuple((unseen + seen)[:top_k])
    if len(page) != top_k or len(page) != len(set(page)):
        raise PaginationReplayError("invalid unseen-first page")
    return page


def reconstruct_current_order(
    turn: Mapping[str, Any], chosen_index: int, activated: bool
) -> tuple[str, ...]:
    """Reconstruct P11 full order and the frozen v1.9 slot-10 swap."""

    c100 = tuple(str(value) for value in turn["c100"])
    p11 = tuple(str(value) for value in turn["actions"]["KEEP_P11"])
    if len(c100) != 100 or len(p11) != 10 or set(p11) != set(c100[:10]):
        raise PaginationReplayError("P11/C100 membership invariant failed")
    order = list(p11 + c100[10:])
    if len(order) != 100 or len(order) != len(set(order)):
        raise PaginationReplayError("P11 full-order reconstruction failed")
    if activated:
        if not 0 <= chosen_index < len(c100):
            raise PaginationReplayError("chosen C100 index is invalid")
        challenger = c100[chosen_index]
        challenger_rank = order.index(challenger)
        order[9], order[challenger_rank] = order[challenger_rank], order[9]
    return tuple(order)


def _policy_state(
    pages: Sequence[Sequence[Sequence[str]]],
    targets: Sequence[str],
    eligible_from: np.ndarray,
) -> dict[str, np.ndarray]:
    hit = np.zeros(base.SESSION_COUNT, dtype=bool)
    first_rank = np.zeros(base.SESSION_COUNT, dtype=np.int16)
    first_turn = np.full(base.SESSION_COUNT, 11, dtype=np.int16)
    for session, turns in enumerate(pages):
        for turn, page in enumerate(turns, start=1):
            if turn < int(eligible_from[session]):
                continue
            try:
                rank = page.index(targets[session]) + 1
            except ValueError:
                continue
            hit[session] = True
            first_rank[session] = rank
            first_turn[session] = turn
            break
    return {"hit": hit, "first_rank": first_rank, "first_turn": first_turn}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _simulate(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    proxy_rows: Sequence[Mapping[str, Any]],
    chosen: np.ndarray,
    activation: np.ndarray,
    labels: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    baseline_pages: list[list[tuple[str, ...]]] = []
    candidate_pages: list[list[tuple[str, ...]]] = []
    targets = [str(row["ground_truth"]["parent_asin"]) for row in proxy_rows]
    eligible_from = np.asarray(labels["eligible_from"])
    output_changed = np.zeros((base.SESSION_COUNT, base.TURN_COUNT), dtype=bool)
    baseline_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    baseline_unique_total = 0
    candidate_unique_total = 0
    same_version_repeat_slots = 0
    reset_count = 0

    for session in range(base.SESSION_COUNT):
        reset_turn = trace_source._eligible_turn(proxy_rows[session])
        if int(eligible_from[session]) != reset_turn:
            raise PaginationReplayError("proxy/label eligible-turn mismatch")
        served: set[str] = set()
        all_baseline: set[str] = set()
        all_candidate: set[str] = set()
        session_baseline: list[tuple[str, ...]] = []
        session_candidate: list[tuple[str, ...]] = []
        for turn_index, turn in enumerate(traces[session]):
            turn_number = turn_index + 1
            if reset_turn > 1 and turn_number == reset_turn:
                served.clear()
                reset_count += 1
            order = reconstruct_current_order(
                turn,
                int(chosen[session, turn_index]),
                bool(activation[session, turn_index]),
            )
            baseline = order[:10]
            candidate = stable_unseen_first(order, served)
            if turn_number == 1 or (reset_turn > 1 and turn_number == reset_turn):
                if candidate != baseline:
                    raise PaginationReplayError("new intent version is not identity")
            same_version_repeat_slots += sum(
                identifier in served for identifier in candidate
            )
            served.update(candidate)
            all_baseline.update(baseline)
            all_candidate.update(candidate)
            session_baseline.append(baseline)
            session_candidate.append(candidate)
            output_changed[session, turn_index] = candidate != baseline
            baseline_digest.update(("|".join(baseline) + "\n").encode("ascii"))
            candidate_digest.update(("|".join(candidate) + "\n").encode("ascii"))
        baseline_unique_total += len(all_baseline)
        candidate_unique_total += len(all_candidate)
        baseline_pages.append(session_baseline)
        candidate_pages.append(session_candidate)

    baseline_state = _policy_state(
        baseline_pages, targets, eligible_from
    )
    candidate_state = _policy_state(
        candidate_pages, targets, eligible_from
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    aggregate = metric.transition_metrics(
        baseline_state, candidate_state, output_changed, all_sessions
    )
    outer = np.asarray(labels["outer_fold"])
    folds = [
        {
            "fold": fold,
            **metric.transition_metrics(
                baseline_state,
                candidate_state,
                output_changed,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    structural = {
        "baseline_output_sha256": baseline_digest.hexdigest(),
        "candidate_output_sha256": candidate_digest.hexdigest(),
        "changed_turns": int(output_changed.sum()),
        "changed_sessions": int(np.any(output_changed, axis=1).sum()),
        "reset_count": reset_count,
        "same_version_repeat_slots": same_version_repeat_slots,
        "baseline_mean_distinct_products": round(
            baseline_unique_total / base.SESSION_COUNT, 6
        ),
        "candidate_mean_distinct_products": round(
            candidate_unique_total / base.SESSION_COUNT, 6
        ),
        "candidate_exposure_duplicate_fraction": round(
            1.0
            - candidate_unique_total
            / float(base.SESSION_COUNT * base.TURN_COUNT * 10),
            6,
        ),
    }
    return {"global": aggregate, "folds": folds, "structural": structural}


def run(source_root: Path, projection_root: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = output.resolve()
    if output.exists() or ROOT not in output.parents:
        raise PaginationReplayError("output must be a new path below this worktree")
    inputs = frozen._load_inputs(source_root, projection_root)
    surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    activation, selections = attribution._reproduce_nested_activation(
        surface, inputs.labels, seed=40220260830
    )
    activation_sha = hashlib.sha256(activation.tobytes()).hexdigest()
    chosen_sha = hashlib.sha256(surface.chosen.tobytes()).hexdigest()
    if (
        activation_sha != EXPECTED_ACTIVATION_SHA256
        or chosen_sha != EXPECTED_CHOSEN_SHA256
    ):
        raise PaginationReplayError("frozen v1.9 policy identity mismatch")
    traces = attribution._load_traces(source_root.resolve())
    proxy_rows = attribution._load_proxy_rows(source_root.resolve())
    first = _simulate(
        traces, proxy_rows, surface.chosen, activation, inputs.labels
    )
    repeat = _simulate(
        traces, proxy_rows, surface.chosen, activation, inputs.labels
    )
    first_sha = _canonical_sha256(first)
    repeat_sha = _canonical_sha256(repeat)
    if first_sha != repeat_sha:
        raise PaginationReplayError("cached replay did not repeat exactly")
    baseline = first["global"]["baseline"]
    if any(baseline[name] != expected for name, expected in EXPECTED_BASELINE.items()):
        raise PaginationReplayError("reconstructed v1.9 metrics changed")
    global_metrics = first["global"]
    folds = first["folds"]
    promote = bool(
        global_metrics["policy"]["hit_rate_at_10"] > 0.9715
        and global_metrics["hit_to_miss"] == 0
        and global_metrics["mrr_delta"] >= 0.0
        and global_metrics["mttc_delta"] <= 0.0
        and global_metrics["technical_score_delta"] > 0.0
        and all(
            row["net_hits"] >= 0
            and row["hit_to_miss"] == 0
            and row["mrr_delta"] >= 0.0
            and row["mttc_delta"] <= 0.0
            for row in folds
        )
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.10-VERSIONED-UNSEEN-PAGINATION",
        "status": "GO_RUNTIME_INTEGRATION" if promote else "NO_GO_CACHED_REPLAY",
        "source": {
            "activation_sha256": activation_sha,
            "chosen_sha256": chosen_sha,
            "proxy_sha256": trace_source.PROXY_SHA256,
            "blind_trace_aggregate_sha256": trace_source.AGGREGATE_SHA256,
            "combined_blind_trace_sha256": trace_source.COMBINED_TRACE_SHA256,
        },
        "policy": {
            "target_blind": True,
            "intent_version_scoped": True,
            "stable_unseen_first": True,
            "fallback_to_seen_only_when_required": True,
            "ranker_or_admission_changed": False,
        },
        "comparison": first,
        "outer_selections": selections,
        "exact_repeat": {
            "equal": True,
            "first_sha256": first_sha,
            "repeat_sha256": repeat_sha,
        },
        "privacy": {
            "split": "train_explore",
            "target_used_only_for_aggregate_metrics": True,
            "target_or_identifier_serialized": False,
            "held_out_split_opened": False,
            "agent_or_full_evaluator_started": False,
        },
        "resource": {
            "wall_seconds": round(time.perf_counter() - started, 6)
        },
        "decision": {
            "promote_to_default_off_runtime_patch": promote,
            "served_default": "off",
        },
    }
    if result["resource"]["wall_seconds"] > 120:
        raise PaginationReplayError("cached replay exceeded resource gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        (
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\tiktok\techjam-err402-fast-track"),
    )
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=Path(r"D:\tiktok\techjam-v1-2-metric-gate"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root, args.projection_root, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "global": result["comparison"]["global"],
                "folds": result["comparison"]["folds"],
                "structural": result["comparison"]["structural"],
                "exact_repeat": result["exact_repeat"],
                "resource": result["resource"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
