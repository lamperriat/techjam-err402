"""Measure posthoc overlap among three already-frozen proposal surfaces."""

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
from scripts import evaluate_small_ranker_rrf3 as rrf3  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-proposal-overlap-diagnostic.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_6.proposal_overlap_preregistration.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
SURFACE_ORDER = ("pairwise", "rrf3", "focused_lambdamart")


class ProposalOverlapError(RuntimeError):
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


def _local_score(path_value: object, expected_sha256: str) -> np.ndarray:
    relative = Path(str(path_value))
    if relative.is_absolute():
        raise ProposalOverlapError("proposal score path must be repo-relative")
    unresolved = ROOT / relative
    path = unresolved.resolve()
    if (
        ROOT not in path.parents
        or not path.is_file()
        or unresolved.is_symlink()
        or _sha256(path) != expected_sha256
    ):
        raise ProposalOverlapError("proposal score identity mismatch")
    score = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        score.shape
        != (base.SESSION_COUNT, base.TURN_COUNT, base.CANDIDATE_COUNT)
        or score.dtype != np.float32
        or not np.isfinite(np.asarray(score)).all()
    ):
        raise ProposalOverlapError("proposal score schema mismatch")
    return score


def _reachable_surface(
    scores: np.ndarray,
    incumbent: np.ndarray,
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    labels: Mapping[str, np.ndarray],
    current_hit: np.ndarray,
) -> dict[str, np.ndarray]:
    proposal = base.choose_slot10(scores, incumbent)[0]
    current_choice = np.where(
        current_activation, current_chosen, incumbent
    ).astype(np.uint8)
    action = (proposal != current_choice) & (proposal != incumbent)
    positive = np.asarray(labels["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(labels["eligible_from"], dtype=np.int16)
    eligible = (
        np.arange(base.TURN_COUNT, dtype=np.int16)[None, :]
        >= eligible_from[:, None] - 1
    )
    correct_turn = (
        action
        & eligible
        & (positive >= 0)
        & (proposal == positive)
        & (~current_hit[:, None])
    )
    return {
        "proposal": proposal,
        "action": action,
        "correct_turn": correct_turn,
        "reachable": np.any(correct_turn, axis=1),
    }


def _membership_summary(
    reachable: Mapping[str, np.ndarray], outer: np.ndarray, current_hits: int
) -> dict[str, Any]:
    masks = [np.asarray(reachable[name], dtype=bool) for name in SURFACE_ORDER]
    if any(mask.shape != (base.SESSION_COUNT,) for mask in masks):
        raise ProposalOverlapError("proposal reachability shape mismatch")
    code = np.zeros(base.SESSION_COUNT, dtype=np.uint8)
    for bit, mask in enumerate(masks):
        code |= mask.astype(np.uint8) << bit
    pattern_counts = {}
    for value in range(1, 8):
        names = [
            name for bit, name in enumerate(SURFACE_ORDER) if value & (1 << bit)
        ]
        pattern_counts["+".join(names)] = int(np.sum(code == value))
    union = code > 0
    union_by_fold = [
        int(np.sum(union & (outer == fold))) for fold in range(base.OUTER_FOLDS)
    ]
    surfaces = {}
    for index, name in enumerate(SURFACE_ORDER):
        mask = masks[index]
        other = np.zeros_like(mask)
        for other_index, other_mask in enumerate(masks):
            if other_index != index:
                other |= other_mask
        surfaces[name] = {
            "sessions": int(mask.sum()),
            "by_outer_fold": [
                int(np.sum(mask & (outer == fold)))
                for fold in range(base.OUTER_FOLDS)
            ],
            "unique_over_other_two": int(np.sum(mask & ~other)),
        }
    intersections = {
        "pairwise_and_rrf3": int(np.sum(masks[0] & masks[1])),
        "pairwise_and_focused": int(np.sum(masks[0] & masks[2])),
        "rrf3_and_focused": int(np.sum(masks[1] & masks[2])),
        "all_three": int(np.sum(masks[0] & masks[1] & masks[2])),
    }
    union_sessions = int(union.sum())
    union_folds = int(sum(value > 0 for value in union_by_fold))
    portfolio_worth_testing = bool(
        union_sessions >= 14 and union_folds >= 3 and union_sessions > 13
    )
    return {
        "surfaces": surfaces,
        "membership_patterns": pattern_counts,
        "intersections": intersections,
        "union": {
            "sessions": union_sessions,
            "by_outer_fold": union_by_fold,
            "outer_folds": union_folds,
            "maximum_zero_harm_hits": int(current_hits) + union_sessions,
            "maximum_zero_harm_hr_at_10": round(
                (int(current_hits) + union_sessions) / base.SESSION_COUNT, 6
            ),
        },
        "direction_gate": {
            "portfolio_worth_testing": portfolio_worth_testing,
            "minimum_union_sessions": 14,
            "minimum_union_folds": 3,
            "must_exceed_best_single_surface": 13,
        },
        "membership_code_sha256": hashlib.sha256(code.tobytes()).hexdigest(),
    }


def run(source_root: Path, projection_root: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_path.exists()
        or output_path.is_symlink()
        or experiments_root not in output_path.parents
    ):
        raise ProposalOverlapError("output must be new and below experiments")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("schema_version") != "small-ranker-proposal-overlap-preregistration.v1":
        raise ProposalOverlapError("proposal-overlap preregistration mismatch")
    fixed = prereg["fixed_surfaces"]
    inputs = frozen._load_inputs(source_root, projection_root)
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = attribution._reproduce_nested_activation(
        current_surface, inputs.labels, seed=40220260830
    )
    if (
        hashlib.sha256(current_activation.tobytes()).hexdigest()
        != EXPECTED_ACTIVATION_SHA256
        or hashlib.sha256(current_surface.chosen.tobytes()).hexdigest()
        != EXPECTED_CHOSEN_SHA256
    ):
        raise ProposalOverlapError("frozen current policy did not reproduce")
    current_state = metric.policy_session_state(
        inputs.labels, current_surface.chosen, current_activation
    )
    current_hit = np.asarray(current_state["hit"], dtype=bool)
    if int(current_hit.sum()) != 1943:
        raise ProposalOverlapError("current hit count drifted")

    pairwise = _local_score(
        fixed["pairwise"]["path"], fixed["pairwise"]["sha256"]
    )
    focused = _local_score(
        fixed["focused_lambdamart"]["path"],
        fixed["focused_lambdamart"]["sha256"],
    )
    member_paths = fixed["rrf3"]["members"][1:]
    member_hashes = fixed["rrf3"]["member_sha256"][1:]
    members = [
        _local_score(path, expected)
        for path, expected in zip(member_paths, member_hashes, strict=True)
    ]
    rrf_scores = rrf3.rrf_scores([inputs.oof_scores, *members])
    if (
        hashlib.sha256(rrf_scores.tobytes()).hexdigest()
        != fixed["rrf3"]["combined_score_sha256"]
    ):
        raise ProposalOverlapError("RRF-3 combined score identity mismatch")

    score_surfaces = {
        "pairwise": pairwise,
        "rrf3": rrf_scores,
        "focused_lambdamart": focused,
    }
    reachability = {}
    correct_turn_counts = {}
    action_counts = {}
    decision_hashes = {}
    for name in SURFACE_ORDER:
        surface = _reachable_surface(
            score_surfaces[name],
            current_surface.incumbent,
            current_surface.chosen,
            current_activation,
            inputs.labels,
            current_hit,
        )
        reachable = np.asarray(surface["reachable"], dtype=bool)
        expected = int(fixed[name]["expected_reachable_current_misses"])
        if int(reachable.sum()) != expected:
            raise ProposalOverlapError(f"{name} oracle count drifted")
        reachability[name] = reachable
        correct_turn_counts[name] = int(surface["correct_turn"].sum())
        action_counts[name] = {
            "turns": int(surface["action"].sum()),
            "sessions": int(np.any(surface["action"], axis=1).sum()),
        }
        decision_hashes[name] = hashlib.sha256(
            surface["proposal"].tobytes()
        ).hexdigest()

    first = _membership_summary(
        reachability,
        np.asarray(inputs.labels["outer_fold"], dtype=np.uint8),
        int(current_hit.sum()),
    )
    repeat = _membership_summary(
        reachability,
        np.asarray(inputs.labels["outer_fold"], dtype=np.uint8),
        int(current_hit.sum()),
    )
    if _canonical_sha256(first) != _canonical_sha256(repeat):
        raise ProposalOverlapError("proposal overlap exact repeat failed")
    portfolio = bool(first["direction_gate"]["portfolio_worth_testing"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.6-PROPOSAL-OVERLAP-DIAGNOSTIC",
        "scope": {
            "split": "train_explore",
            "cached_frozen_surfaces_only": True,
            "target_posthoc_membership_only": True,
            "identifiers_serialized": False,
            "ranker_or_gate_trained": False,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "external_data_downloaded": False,
            "full_model_or_runtime_artifact_trained": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "pairwise_score_sha256": fixed["pairwise"]["sha256"],
            "rrf3_member_score_sha256": fixed["rrf3"]["member_sha256"],
            "rrf3_combined_score_sha256": fixed["rrf3"][
                "combined_score_sha256"
            ],
            "focused_score_sha256": fixed["focused_lambdamart"]["sha256"],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
        },
        "current": {
            "hits": int(current_hit.sum()),
            "misses": int((~current_hit).sum()),
            "activation_sha256": hashlib.sha256(
                current_activation.tobytes()
            ).hexdigest(),
            "chosen_sha256": hashlib.sha256(
                current_surface.chosen.tobytes()
            ).hexdigest(),
            "selections_sha256": _canonical_sha256(current_selections),
        },
        "proposal_surfaces": {
            "correct_action_turns": correct_turn_counts,
            "action_counts": action_counts,
            "decision_sha256": decision_hashes,
        },
        "overlap": first,
        "repeat": {
            "exact": True,
            "canonical_sha256": _canonical_sha256(first),
        },
        "decision": {
            "status": "PORTFOLIO_SIGNAL" if portfolio else "PORTFOLIO_CLOSED",
            "train_selector_next": portfolio,
            "next": (
                "preregister one target-blind selector over the three frozen proposals"
                if portfolio
                else "preregister a materially new proposal signal; do not train a selector or retune these surfaces"
            ),
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.source_root, args.projection_root, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
