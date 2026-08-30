"""Evaluate fixed semantic-off RRF-3 as a current-policy supplement."""

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
from scripts import evaluate_small_ranker_supplemental_pairwise as supplemental  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-rrf3-evaluation.v1"
PREREGISTRATION = ROOT / "configs/small_ranker_v2_4.rrf3_preregistration.json"
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
CURRENT_SCORE_SHA256 = (
    "5000deb9b77b3e7b326ccab6455222b291d2ec859ddab2043fe67d23a3217c5e"
)
RRF_K = 60.0
RRF_FEATURE_NAMES = (
    "current_policy_active",
    "current_choice_rank_fraction_under_rrf3",
    "rrf3_choice_rank_fraction_under_current_ranker",
    "rrf3_choice_coverage_rank_fraction",
    "rrf3_minus_current_top10_route_agreement",
    "rrf3_minus_current_active_token_recall",
    "rrf3_minus_current_hard_clause_coverage",
    "rrf3_minus_current_constraint_conflict_sum",
)


class RRF3EvaluationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_ranks(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 3 or scores.shape[2] != base.CANDIDATE_COUNT:
        raise RRF3EvaluationError("RRF member score shape mismatch")
    order = np.argsort(-np.asarray(scores), axis=2, kind="stable")
    ranks = np.empty(order.shape, dtype=np.uint8)
    values = np.broadcast_to(
        np.arange(1, base.CANDIDATE_COUNT + 1, dtype=np.uint8), order.shape
    )
    np.put_along_axis(ranks, order, values, axis=2)
    return ranks


def rrf_scores(members: Sequence[np.ndarray]) -> np.ndarray:
    if len(members) != 3:
        raise RRF3EvaluationError("RRF-3 requires exactly three members")
    result = np.zeros(members[0].shape, dtype=np.float32)
    for member in members:
        result += 1.0 / (np.float32(RRF_K) + stable_ranks(member))
    if not np.isfinite(result).all():
        raise RRF3EvaluationError("RRF scores are non-finite")
    return result


def _load_member_scores(
    projection_result_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    result = json.loads(projection_result_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version")
        != "small-ranker-rrf3-semantic-off-projection.v1"
        or result.get("scope", {}).get("target_label_read") is not False
        or len(result.get("members", [])) != 2
    ):
        raise RRF3EvaluationError("RRF projection protocol mismatch")
    first: list[np.ndarray] = []
    repeat: list[np.ndarray] = []
    for row in result["members"]:
        first_path = ROOT / str(row["score_path"])
        repeat_path = ROOT / str(row["repeat_path"])
        expected = str(row["score_sha256"])
        if (
            row.get("byte_identical") is not True
            or not first_path.is_file()
            or not repeat_path.is_file()
            or _sha256(first_path) != expected
            or _sha256(repeat_path) != expected
        ):
            raise RRF3EvaluationError("RRF member score identity mismatch")
        first.append(np.load(first_path, mmap_mode="r"))
        repeat.append(np.load(repeat_path, mmap_mode="r"))
    return first, repeat, result


def _oracle(
    surface: supplemental.SupplementalSurface,
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    eligible = (
        np.arange(1, base.TURN_COUNT + 1)[None, :]
        >= eligible_from[:, None]
    )
    current_miss = ~np.asarray(current_state["hit"], dtype=bool)
    correct = (
        surface.action
        & eligible
        & (positive >= 0)
        & (surface.pairwise_chosen == positive)
    )
    reachable = current_miss & np.any(correct, axis=1)
    outer = np.asarray(labels["outer_fold"])
    return {
        "current_miss_sessions": int(current_miss.sum()),
        "correct_action_turns_on_current_misses": int(
            correct[current_miss].sum()
        ),
        "reachable_current_miss_sessions": int(reachable.sum()),
        "maximum_zero_harm_hr_at_10": round(
            float((np.asarray(current_state["hit"]) | reachable).mean()), 6
        ),
        "reachable_by_fold": [
            int(np.sum(reachable & (outer == fold)))
            for fold in range(base.OUTER_FOLDS)
        ],
        "posthoc_target_informed_not_runtime": True,
    }


def run(
    source_root: Path,
    projection_root: Path,
    rrf_projection_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    if output_path.exists() or ROOT not in output_path.parents:
        raise RRF3EvaluationError("output must be new and local")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version") != "small-ranker-rrf3-preregistration.v1"
        or tuple(prereg["admission"]["feature_names"]) != RRF_FEATURE_NAMES
        or float(prereg["fusion"]["k"]) != RRF_K
    ):
        raise RRF3EvaluationError("RRF preregistration mismatch")
    inputs = frozen._load_inputs(source_root, projection_root)
    if _sha256(inputs.score_path) != CURRENT_SCORE_SHA256:
        raise RRF3EvaluationError("current semantic-off score mismatch")
    projected_first, projected_repeat, projection_result = _load_member_scores(
        rrf_projection_result_path.resolve()
    )
    first_rrf = rrf_scores([inputs.oof_scores, *projected_first])
    repeat_rrf = rrf_scores([inputs.oof_scores, *projected_repeat])
    if not np.array_equal(first_rrf, repeat_rrf):
        raise RRF3EvaluationError("RRF score repeat differs")

    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = (
        attribution._reproduce_nested_activation(
            current_surface, inputs.labels, seed=40220260830
        )
    )
    if (
        hashlib.sha256(current_activation.tobytes()).hexdigest()
        != supplemental.EXPECTED_CURRENT_ACTIVATION_SHA256
        or hashlib.sha256(current_surface.chosen.tobytes()).hexdigest()
        != supplemental.EXPECTED_CURRENT_CHOSEN_SHA256
    ):
        raise RRF3EvaluationError("current policy did not reproduce")
    surface = supplemental._surface(
        inputs.projected_features,
        inputs.oof_scores,
        first_rrf,
        inputs.labels,
        current_activation,
    )
    current_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, surface.current_activation
    )
    zero = np.zeros_like(surface.current_activation, dtype=bool)
    p11_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, zero
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    current_vs_p11 = metric.transition_metrics(
        p11_state, current_state, surface.current_activation, all_sessions
    )
    if not (
        float(current_vs_p11["policy"]["hit_rate_at_10"])
        == supplemental.CURRENT_HR
        and int(current_vs_p11["miss_to_hit"]) == 48
        and int(current_vs_p11["hit_to_miss"]) == 0
    ):
        raise RRF3EvaluationError("current comparator drifted")

    first_started = time.perf_counter()
    first = supplemental._nested_oof(surface, inputs.labels, seed=40220260830)
    first_seconds = time.perf_counter() - first_started
    repeat_started = time.perf_counter()
    repeat = supplemental._nested_oof(surface, inputs.labels, seed=40220260830)
    repeat_seconds = time.perf_counter() - repeat_started
    exact = bool(
        np.array_equal(first.supplement, repeat.supplement)
        and np.array_equal(first.final_chosen, repeat.final_chosen)
        and np.array_equal(first.final_activation, repeat.final_activation)
        and np.array_equal(
            first.rescue_probability, repeat.rescue_probability
        )
        and np.array_equal(
            first.regret_probability, repeat.regret_probability
        )
        and _canonical_sha256(first.selections)
        == _canonical_sha256(repeat.selections)
    )
    if not exact:
        raise RRF3EvaluationError("RRF nested admission repeat differs")
    relative, folds, final_state = supplemental._metrics(
        inputs.labels, current_state, first
    )
    passed = supplemental._promotion_gate(relative, folds)
    final_vs_p11 = metric.transition_metrics(
        p11_state, final_state, first.final_activation, all_sessions
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.4-RRF3",
        "scope": {
            "split": "train_explore",
            "cached_inputs_only": True,
            "ranker_retrained": False,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "runtime_features_target_blind": True,
            "target_is_training_or_posthoc_label_only": True,
            "full_model_or_artifact_trained": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "projection_result_sha256": _sha256(
                rrf_projection_result_path.resolve()
            ),
            "current_oof_score_sha256": CURRENT_SCORE_SHA256,
            "projected_member_score_sha256": [
                str(row["score_sha256"])
                for row in projection_result["members"]
            ],
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "projected_features_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "fusion": {
            "members": [
                "ndcg_d4_lr003",
                "ndcg_d6_lr006",
                "ndcg_d4_regularized",
            ],
            "k": RRF_K,
            "score_sha256": hashlib.sha256(first_rrf.tobytes()).hexdigest(),
            "repeat_exact": True,
        },
        "feature_contract": {
            "names": list(RRF_FEATURE_NAMES),
            "count": len(RRF_FEATURE_NAMES),
            "schema_sha256": _canonical_sha256(list(RRF_FEATURE_NAMES)),
        },
        "surface": {
            "action_rows": int(surface.action.sum()),
            "action_sessions": int(np.any(surface.action, axis=1).sum()),
            "rescue_label_rows": int(surface.rescue.sum()),
            "regret_label_rows": int(surface.regret.sum()),
        },
        "current": {
            "activation_sha256": hashlib.sha256(
                surface.current_activation.tobytes()
            ).hexdigest(),
            "chosen_sha256": hashlib.sha256(
                surface.current_chosen.tobytes()
            ).hexdigest(),
            "selections_sha256": _canonical_sha256(current_selections),
            "versus_p11": current_vs_p11,
        },
        "challenger": {
            "relative_to_current": relative,
            "folds_relative_to_current": folds,
            "versus_p11": final_vs_p11,
            "outer_selections": first.selections,
            "supplemental_activation_sha256": hashlib.sha256(
                first.supplement.tobytes()
            ).hexdigest(),
            "final_chosen_sha256": hashlib.sha256(
                first.final_chosen.tobytes()
            ).hexdigest(),
            "final_activation_sha256": hashlib.sha256(
                first.final_activation.tobytes()
            ).hexdigest(),
        },
        "repeat": {
            "exact": exact,
            "selection_canonical_sha256": _canonical_sha256(
                first.selections
            ),
            "rescue_probability_sha256": hashlib.sha256(
                first.rescue_probability.tobytes()
            ).hexdigest(),
            "regret_probability_sha256": hashlib.sha256(
                first.regret_probability.tobytes()
            ).hexdigest(),
        },
        "rrf3_supplemental_oracle": _oracle(
            surface, inputs.labels, current_state
        ),
        "decision": {
            "promotion_gate_passed": passed,
            "status": "PROMOTE" if passed else "NO_GO",
            "full_artifact_authorized": passed,
            "next": (
                "preregister export/distillation and target-free resource work"
                if passed
                else "close fixed RRF-3 and proceed to focused nonlinear LambdaMART"
            ),
        },
        "timing_seconds": {
            "first_nested_oof": round(first_seconds, 6),
            "repeat_nested_oof": round(repeat_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
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
    parser.add_argument("--rrf-projection-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.source_root,
        args.projection_root,
        args.rrf_projection_result,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
