"""Evaluate the fixed v2.5 focused LambdaMART proposal surface.

This is evaluator-side, cached ``train_explore`` posthoc analysis.  Targets are
used only for the preregistered proposal oracle and metric ledger; they are not
available to the ranker scores or any runtime feature.
"""

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
from scripts import evaluate_small_ranker_supplemental_pairwise as supplemental  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-focused-stage-a-evaluation.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_implementation_amendment.json"
)
FOCUSED_CACHE_MANIFEST = ROOT / (
    "configs/small_ranker_v2_5.focused_cache.manifest.json"
)
TRAINING_MANIFEST = ROOT / (
    "configs/small_ranker_v2_5.focused_outer_oof.manifest.json"
)
ATTRIBUTION_MANIFEST = ROOT / (
    "configs/small_ranker_v2_0.remaining_miss_attribution.manifest.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
CURRENT_HR = 0.9715
MIN_REACHABLE_SESSIONS = 14
MIN_REACHABLE_FOLDS = 3


class FocusedStageAError(RuntimeError):
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


def _local_regular_file(value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise FocusedStageAError("training artifact path must be repo-relative")
    unresolved = ROOT / relative
    path = unresolved.resolve()
    if ROOT not in path.parents or not path.is_file() or unresolved.is_symlink():
        raise FocusedStageAError("training artifact is not a local regular file")
    return path


def stage_a_gate(reachable_sessions: int, reachable_folds: int) -> bool:
    return (
        int(reachable_sessions) >= MIN_REACHABLE_SESSIONS
        and int(reachable_folds) >= MIN_REACHABLE_FOLDS
    )


def _load_training_result(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    unresolved = path
    path = unresolved.resolve()
    if ROOT not in path.parents or not path.is_file() or unresolved.is_symlink():
        raise FocusedStageAError("training result must be a local regular file")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not FOCUSED_CACHE_MANIFEST.is_file() or not TRAINING_MANIFEST.is_file():
        raise FocusedStageAError("tracked focused cache/training manifest is unavailable")
    cache_manifest = json.loads(
        FOCUSED_CACHE_MANIFEST.read_text(encoding="utf-8")
    )
    training_manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    amendment = json.loads(IMPLEMENTATION_AMENDMENT.read_text(encoding="utf-8"))
    relative_result = path.relative_to(ROOT).as_posix()
    if (
        result.get("schema_version")
        != "small-ranker-focused-outer-oof-training.v1"
        or result.get("scope", {}).get("split") != "train_explore"
        or result.get("scope", {}).get("score_projection_target_free") is not True
        or result.get("scope", {}).get("agent_or_evaluator_started") is not False
        or result.get("sources", {}).get("preregistration_sha256")
        != _sha256(PREREGISTRATION)
        or result.get("sources", {}).get("implementation_amendment_sha256")
        != _sha256(IMPLEMENTATION_AMENDMENT)
        or result.get("sources", {}).get("trainer_sha256")
        != _sha256(ROOT / "scripts/train_small_ranker_focused_outer_oof.py")
        or result.get("sources", {}).get("projected_features_sha256")
        != frozen.EXPECTED_HASHES["projected_features"]
        or result.get("sources", {}).get("label_cache_sha256")
        != frozen.EXPECTED_HASHES["labels"]
        or result.get("sources", {}).get("cache_manifest_sha256")
        != _sha256(FOCUSED_CACHE_MANIFEST)
        or training_manifest.get("schema_version")
        != "small-ranker-focused-outer-oof-manifest.v1"
        or training_manifest.get("experiment_id")
        != "SR-V2.5-FOCUSED-LAMBDAMART-STAGE-A-TRAIN"
        or training_manifest.get("result", {}).get("path") != relative_result
        or training_manifest.get("result", {}).get("sha256") != _sha256(path)
        or training_manifest.get("sources", {}).get("cache_manifest_sha256")
        != _sha256(FOCUSED_CACHE_MANIFEST)
        or training_manifest.get("sources", {}).get("trainer_sha256")
        != _sha256(ROOT / "scripts/train_small_ranker_focused_outer_oof.py")
        or cache_manifest.get("result", {}).get("sha256")
        != result.get("sources", {}).get("cache_result_sha256")
        or result.get("model", {}).get("parameters")
        != amendment.get("training_contract", {}).get("parameters")
        or int(result.get("model", {}).get("rounds", -1)) != 300
        or int(result.get("model", {}).get("seed", -1)) != 40220260830
        or result.get("model", {}).get("xgboost_version") != "1.7.6"
        or result.get("exact_repeat", {}).get("score_bytes_identical") is not True
        or result.get("exact_repeat", {}).get("proposal_decisions_identical")
        is not True
    ):
        raise FocusedStageAError("focused training protocol mismatch")
    first_path = _local_regular_file(result.get("first", {}).get("score_path"))
    repeat_path = _local_regular_file(result.get("repeat", {}).get("score_path"))
    first_hash = _sha256(first_path)
    repeat_hash = _sha256(repeat_path)
    if (
        first_hash != result.get("first", {}).get("score_sha256")
        or repeat_hash != result.get("repeat", {}).get("score_sha256")
        or first_hash != repeat_hash
        or first_path.stat().st_size != int(result.get("first", {}).get("score_bytes", -1))
        or repeat_path.stat().st_size != int(result.get("repeat", {}).get("score_bytes", -1))
    ):
        raise FocusedStageAError("focused OOF score identity mismatch")
    score_manifest_records = {}
    model_manifest_records = []
    for pass_name in ("first", "repeat"):
        pass_record = result.get(pass_name, {})
        folds = pass_record.get("folds", [])
        parity = pass_record.get("serialized_model_parity", [])
        if (
            [row.get("fold") for row in folds] != list(range(base.OUTER_FOLDS))
            or [row.get("fold") for row in parity] != list(range(base.OUTER_FOLDS))
            or any(
                int(row.get("held_sessions", -1)) != 400
                or int(row.get("held_fold_overlap", -1)) != 0
                or int(row.get("seed", -1)) != 40220260830 + int(row.get("fold", -1))
                for row in folds
            )
            or any(
                float(row.get("maximum_absolute_error", -1.0)) != 0.0
                or row.get("c100_order_exact") is not True
                for row in parity
            )
        ):
            raise FocusedStageAError("focused fold/model parity audit mismatch")
        score_manifest_records[pass_name] = {
            "path": pass_record["score_path"],
            "sha256": pass_record["score_sha256"],
            "bytes": int(pass_record["score_bytes"]),
        }
        for row in folds:
            model_path = _local_regular_file(row.get("model_path"))
            if _sha256(model_path) != row.get("model_sha256"):
                raise FocusedStageAError("focused fold model identity mismatch")
            model_manifest_records.append(
                {
                    "pass": pass_name,
                    "fold": int(row["fold"]),
                    "path": row["model_path"],
                    "sha256": row["model_sha256"],
                }
            )
    if training_manifest.get("files") != {
        "scores": score_manifest_records,
        "models": model_manifest_records,
    }:
        raise FocusedStageAError("tracked training file records drifted")
    first = np.load(first_path, mmap_mode="r", allow_pickle=False)
    repeat = np.load(repeat_path, mmap_mode="r", allow_pickle=False)
    expected = (base.SESSION_COUNT, base.TURN_COUNT, base.CANDIDATE_COUNT)
    if (
        first.shape != expected
        or repeat.shape != expected
        or first.dtype != np.float32
        or repeat.dtype != np.float32
        or not np.isfinite(np.asarray(first)).all()
        or not np.isfinite(np.asarray(repeat)).all()
        or not np.array_equal(first, repeat)
    ):
        raise FocusedStageAError("focused OOF score schema/repeat mismatch")
    return result, first, repeat


def _proposal_oracle(
    surface: supplemental.SupplementalSurface,
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    positive = np.asarray(labels["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(labels["eligible_from"], dtype=np.int16)
    outer = np.asarray(labels["outer_fold"], dtype=np.uint8)
    current_hit = np.asarray(current_state["hit"], dtype=bool)
    eligible = (
        np.arange(base.TURN_COUNT, dtype=np.int16)[None, :]
        >= eligible_from[:, None] - 1
    )
    correct_turn = (
        surface.action
        & eligible
        & (positive >= 0)
        & (surface.pairwise_chosen == positive)
        & (~current_hit[:, None])
    )
    reachable = np.any(correct_turn, axis=1)
    by_fold = [
        int(np.sum(reachable & (outer == fold)))
        for fold in range(base.OUTER_FOLDS)
    ]
    sessions = int(reachable.sum())
    fold_count = int(sum(value > 0 for value in by_fold))
    return {
        "target_informed_posthoc_only": True,
        "current_miss_sessions": int((~current_hit).sum()),
        "reachable_current_miss_sessions": sessions,
        "correct_action_turns": int(correct_turn.sum()),
        "reachable_by_outer_fold": by_fold,
        "reachable_outer_folds": fold_count,
        "maximum_zero_harm_hits": int(current_hit.sum()) + sessions,
        "maximum_zero_harm_hr_at_10": round(
            (int(current_hit.sum()) + sessions) / base.SESSION_COUNT, 6
        ),
        "stage_b_gate": {
            "minimum_sessions": MIN_REACHABLE_SESSIONS,
            "minimum_outer_folds": MIN_REACHABLE_FOLDS,
            "passed": stage_a_gate(sessions, fold_count),
        },
    }


def _metrics(
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    final_chosen: np.ndarray,
    final_activation: np.ndarray,
    supplement: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    final_state = metric.policy_session_state(labels, final_chosen, final_activation)
    outer = np.asarray(labels["outer_fold"], dtype=np.uint8)
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    global_metrics = metric.transition_metrics(
        current_state, final_state, supplement, all_sessions
    )
    folds = [
        {
            "fold": fold,
            **metric.transition_metrics(
                current_state,
                final_state,
                supplement,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    return global_metrics, folds, final_state


def run(
    training_result_path: Path,
    source_root: Path,
    projection_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_path.exists()
        or output_path.is_symlink()
        or experiments_root not in output_path.parents
    ):
        raise FocusedStageAError("output must be new and below experiments")
    training, focused_scores, focused_repeat = _load_training_result(
        training_result_path
    )
    inputs = frozen._load_inputs(source_root, projection_root)
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = attribution._reproduce_nested_activation(
        current_surface, inputs.labels, seed=40220260830
    )
    activation_sha = hashlib.sha256(current_activation.tobytes()).hexdigest()
    chosen_sha = hashlib.sha256(current_surface.chosen.tobytes()).hexdigest()
    if (
        activation_sha != EXPECTED_ACTIVATION_SHA256
        or chosen_sha != EXPECTED_CHOSEN_SHA256
    ):
        raise FocusedStageAError("frozen current policy did not reproduce")

    incumbent = current_surface.incumbent
    first_choice = base.choose_slot10(focused_scores, incumbent)[0]
    repeat_choice = base.choose_slot10(focused_repeat, incumbent)[0]
    decision_sha = hashlib.sha256(first_choice.tobytes()).hexdigest()
    if (
        not np.array_equal(first_choice, repeat_choice)
        or decision_sha
        != training.get("exact_repeat", {}).get("proposal_decision_sha256")
    ):
        raise FocusedStageAError("focused proposal decision repeat mismatch")

    surface = supplemental._surface(
        inputs.projected_features,
        inputs.oof_scores,
        focused_scores,
        inputs.labels,
        current_activation,
    )
    if not np.array_equal(surface.pairwise_chosen, first_choice):
        raise FocusedStageAError("focused supplemental surface choice drifted")
    current_state = metric.policy_session_state(
        inputs.labels, surface.current_chosen, surface.current_activation
    )
    zero = np.zeros_like(surface.current_activation, dtype=bool)
    p11_state = metric.policy_session_state(inputs.labels, surface.current_chosen, zero)
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    current_vs_p11 = metric.transition_metrics(
        p11_state, current_state, surface.current_activation, all_sessions
    )
    if (
        float(current_vs_p11["policy"]["hit_rate_at_10"]) != CURRENT_HR
        or int(current_vs_p11["miss_to_hit"]) != 48
        or int(current_vs_p11["hit_to_miss"]) != 0
    ):
        raise FocusedStageAError("current comparator metric drifted")

    oracle = _proposal_oracle(surface, inputs.labels, current_state)
    final_chosen, final_activation = supplemental._compose_policy(
        surface.current_chosen,
        surface.current_activation,
        surface.pairwise_chosen,
        surface.action,
    )
    ungated, fold_ungated, final_state = _metrics(
        inputs.labels,
        current_state,
        final_chosen,
        final_activation,
        surface.action,
    )
    final_vs_p11 = metric.transition_metrics(
        p11_state, final_state, final_activation, all_sessions
    )
    passed = bool(oracle["stage_b_gate"]["passed"])
    attribution_manifest = json.loads(
        ATTRIBUTION_MANIFEST.read_text(encoding="utf-8")
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.5-FOCUSED-LAMBDAMART-STAGE-A",
        "scope": {
            "split": "train_explore",
            "cached_inputs_only": True,
            "target_used_as_training_or_posthoc_label_only": True,
            "ranker_and_runtime_features_target_blind": True,
            "strict_outer_product_family_oof_for_focused_ranker": True,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "calibration_selection_confirmation_or_public_opened": False,
            "external_data_downloaded": False,
            "full_model_or_runtime_artifact_trained": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "training_result_path": training_result_path.resolve()
            .relative_to(ROOT)
            .as_posix(),
            "training_result_sha256": _sha256(training_result_path.resolve()),
            "trainer_sha256": training["sources"]["trainer_sha256"],
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "projected_features_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "attribution_manifest_sha256": _sha256(ATTRIBUTION_MANIFEST),
        },
        "current": {
            "activation_sha256": activation_sha,
            "chosen_sha256": chosen_sha,
            "selections_sha256": _canonical_sha256(current_selections),
            "versus_p11": current_vs_p11,
        },
        "proposal": {
            "action_turns": int(surface.action.sum()),
            "action_sessions": int(np.any(surface.action, axis=1).sum()),
            "choice_agreement_turns": int(
                np.sum(surface.pairwise_chosen == surface.current_choice)
            ),
            "incumbent_choice_turns": int(
                np.sum(surface.pairwise_chosen == current_surface.incumbent)
            ),
            "decision_sha256": decision_sha,
        },
        "stage_a_oracle": oracle,
        "ungated_diagnostic": {
            "relative_to_current": ungated,
            "folds_relative_to_current": fold_ungated,
            "versus_p11": final_vs_p11,
            "final_chosen_sha256": hashlib.sha256(
                final_chosen.tobytes()
            ).hexdigest(),
            "final_activation_sha256": hashlib.sha256(
                final_activation.tobytes()
            ).hexdigest(),
            "not_a_promotable_policy_without_stage_b": True,
        },
        "candidate_recall_unchanged": attribution_manifest["candidate_recall"],
        "repeat": {
            "score_bytes_identical": True,
            "proposal_decisions_identical": True,
            "score_sha256": training["first"]["score_sha256"],
            "repeat_score_sha256": training["repeat"]["score_sha256"],
        },
        "resource": {
            "first_training_and_projection_seconds": training["first"][
                "timing_seconds"
            ]["total"],
            "repeat_training_and_projection_seconds": training["repeat"][
                "timing_seconds"
            ]["total"],
            "prediction_ms_per_session": training["first"][
                "prediction_ms_per_session"
            ],
            "peak_working_set_bytes": max(
                int(training["first"]["peak_working_set_bytes"]),
                int(training["repeat"]["peak_working_set_bytes"]),
            ),
        },
        "decision": {
            "stage_b_gate_passed": passed,
            "status": "STAGE_B_AUTHORIZED" if passed else "NO_GO",
            "stage_b_started": False,
            "full_artifact_authorized": False,
            "fixed_model_closed_without_tuning": not passed,
            "next": (
                "execute the preregistered 30-model nested admission"
                if passed
                else "close v2.5 without cohort, weight, tree, or oracle-threshold tuning and preregister one different mechanism"
            ),
        },
        "timing_seconds": {"evaluation": round(time.perf_counter() - started, 6)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.training_result,
        args.source_root,
        args.projection_root,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
