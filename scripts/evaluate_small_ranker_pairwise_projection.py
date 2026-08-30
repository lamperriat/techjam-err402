"""Evaluate pairwise semantic-off OOF scores with the frozen admission protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-pairwise-projection-evaluation.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_1.pairwise_projection_preregistration.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_CURRENT_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
CURRENT_HR = 0.9715


class PairwiseEvaluationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _policy_metrics(
    labels: Mapping[str, np.ndarray],
    surface: frozen.ActionSurface,
    activation: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    state = metric.policy_session_state(labels, surface.chosen, activation)
    zero = np.zeros_like(activation, dtype=bool)
    baseline = metric.policy_session_state(labels, surface.chosen, zero)
    outer = np.asarray(labels["outer_fold"])
    all_sessions = np.ones(len(outer), dtype=bool)
    global_metrics = metric.transition_metrics(
        baseline, state, activation, all_sessions
    )
    folds = [
        {
            "fold": fold,
            **metric.transition_metrics(
                baseline, state, activation, outer == fold
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    return global_metrics, folds, state


def _promotion_gate(
    current_state: Mapping[str, np.ndarray],
    challenger_state: Mapping[str, np.ndarray],
    challenger_activation: np.ndarray,
    labels: Mapping[str, np.ndarray],
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    outer = np.asarray(labels["outer_fold"])
    all_sessions = np.ones(len(outer), dtype=bool)
    global_delta = metric.transition_metrics(
        current_state,
        challenger_state,
        challenger_activation,
        all_sessions,
    )
    folds = [
        {
            "fold": fold,
            **metric.transition_metrics(
                current_state,
                challenger_state,
                challenger_activation,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    passed = bool(
        float(global_delta["policy"]["hit_rate_at_10"]) > CURRENT_HR
        and int(global_delta["hit_to_miss"]) == 0
        and float(global_delta["mrr_delta"]) >= 0.0
        and float(global_delta["mttc_delta"]) <= 0.0
        and float(global_delta["technical_score_delta"]) > 0.0
        and all(
            int(row["net_hits"]) >= 0
            and int(row["hit_to_miss"]) == 0
            and float(row["mrr_delta"]) >= 0.0
            and float(row["mttc_delta"]) <= 0.0
            for row in folds
        )
    )
    return passed, global_delta, folds


def run(
    source_root: Path,
    projection_root: Path,
    projection_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    if output_path.exists() or ROOT not in output_path.parents:
        raise PairwiseEvaluationError("evaluation output must be new and local")
    projection_result_path = projection_result_path.resolve()
    projection_result = json.loads(
        projection_result_path.read_text(encoding="utf-8")
    )
    if (
        projection_result.get("schema_version")
        != "small-ranker-pairwise-semantic-off-projection.v1"
        or projection_result.get("scores", {}).get("byte_identical") is not True
        or projection_result.get("scope", {}).get("target_label_read") is not False
    ):
        raise PairwiseEvaluationError("pairwise projection protocol mismatch")
    score_path = ROOT / str(projection_result["scores"]["path"])
    repeat_path = ROOT / str(projection_result["scores"]["repeat_path"])
    expected_score_hash = str(projection_result["scores"]["sha256"])
    if (
        not score_path.is_file()
        or not repeat_path.is_file()
        or _sha256(score_path) != expected_score_hash
        or _sha256(repeat_path) != expected_score_hash
    ):
        raise PairwiseEvaluationError("pairwise projected score identity mismatch")

    inputs = frozen._load_inputs(source_root, projection_root)
    challenger_scores = np.load(score_path, mmap_mode="r")
    if challenger_scores.shape != inputs.oof_scores.shape:
        raise PairwiseEvaluationError("pairwise projected score schema mismatch")
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    challenger_surface = frozen._action_surface(
        inputs.projected_features, challenger_scores, inputs.labels
    )
    current_activation, current_selections = (
        attribution._reproduce_nested_activation(
            current_surface, inputs.labels, seed=40220260830
        )
    )
    if (
        hashlib.sha256(current_activation.tobytes()).hexdigest()
        != EXPECTED_CURRENT_ACTIVATION_SHA256
    ):
        raise PairwiseEvaluationError("current frozen policy did not reproduce")
    first_activation, first_selections = (
        attribution._reproduce_nested_activation(
            challenger_surface, inputs.labels, seed=40220260830
        )
    )
    repeat_activation, repeat_selections = (
        attribution._reproduce_nested_activation(
            challenger_surface, inputs.labels, seed=40220260830
        )
    )
    if not np.array_equal(first_activation, repeat_activation):
        raise PairwiseEvaluationError("pairwise admission activation repeat differs")
    if _canonical_sha256(first_selections) != _canonical_sha256(repeat_selections):
        raise PairwiseEvaluationError("pairwise admission selection repeat differs")

    current_global, current_folds, current_state = _policy_metrics(
        inputs.labels, current_surface, current_activation
    )
    challenger_global, challenger_folds, challenger_state = _policy_metrics(
        inputs.labels, challenger_surface, first_activation
    )
    passed, relative_global, relative_folds = _promotion_gate(
        current_state,
        challenger_state,
        first_activation,
        inputs.labels,
    )
    positive = np.asarray(inputs.labels["positive_index"])
    eligible_from = np.asarray(inputs.labels["eligible_from"])
    current_miss = ~np.asarray(current_state["hit"], dtype=bool)
    correct_proposal_current_misses = 0
    correct_proposal_turns = 0
    for session in np.flatnonzero(current_miss):
        eligible_index = int(eligible_from[session]) - 1
        correct = (
            (positive[session, eligible_index:] >= 0)
            & (
                challenger_surface.chosen[session, eligible_index:]
                == positive[session, eligible_index:]
            )
        )
        correct_proposal_current_misses += int(np.any(correct))
        correct_proposal_turns += int(correct.sum())
    valid = positive >= 0
    current_correct = current_surface.chosen == positive
    challenger_correct = challenger_surface.chosen == positive
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.1-PAIRWISE-PROJECTION",
        "scope": {
            "split": "train_explore",
            "ranker_retrained": False,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "full_model_or_artifact_trained": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "projection_result_sha256": _sha256(projection_result_path),
            "projected_score_sha256": expected_score_hash,
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "repeat": {
            "score_files_byte_identical": True,
            "activation_exact": True,
            "activation_sha256": hashlib.sha256(
                first_activation.tobytes()
            ).hexdigest(),
            "selection_canonical_sha256": _canonical_sha256(first_selections),
        },
        "ranker_proposals": {
            "positive_query_groups": int(valid.sum()),
            "current_correct_slot10_proposal_fraction": round(
                float(current_correct[valid].mean()), 6
            ),
            "pairwise_correct_slot10_proposal_fraction": round(
                float(challenger_correct[valid].mean()), 6
            ),
            "current_remaining_miss_sessions": int(current_miss.sum()),
            "pairwise_correct_proposal_remaining_miss_sessions": int(
                correct_proposal_current_misses
            ),
            "pairwise_correct_proposal_remaining_miss_turns": int(
                correct_proposal_turns
            ),
        },
        "current": {
            "global": current_global,
            "folds": current_folds,
            "selections": current_selections,
        },
        "challenger": {
            "global": challenger_global,
            "folds": challenger_folds,
            "selections": first_selections,
        },
        "relative_to_current": {
            "global": relative_global,
            "folds": relative_folds,
        },
        "decision": {
            "promotion_gate_passed": passed,
            "status": "PROMOTE" if passed else "NO_GO",
            "next": (
                "freeze full pairwise artifact"
                if passed
                else "close pairwise objective and preregister a materially different ranker mechanism"
            ),
        },
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 6)
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--projection-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.source_root,
        args.projection_root,
        args.projection_result,
        args.output,
    )
    print(
        json.dumps(
            {
                "ranker_proposals": result["ranker_proposals"],
                "challenger": result["challenger"]["global"],
                "relative_to_current": result["relative_to_current"],
                "decision": result["decision"],
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
