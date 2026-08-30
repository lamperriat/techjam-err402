"""Nested-OOF evaluation of one hard-case pairwise residual ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import evaluate_small_ranker_pairwise_projection as comparison  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-hardcase-residual-evaluation.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_2.hardcase_residual_preregistration.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_CURRENT_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
HARD_CASE_WEIGHT = 10.0
CONTROL_WEIGHT = 1.0
EXTRA_HARD_NEGATIVES = 4
MODEL_C = 0.2
ASIN_SHAPE_RE = re.compile(
    rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE
)


class ResidualEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairCache:
    x: np.ndarray
    y: np.ndarray
    session: np.ndarray
    weight: np.ndarray
    hard_session: np.ndarray


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


def _allowed_indices(incumbent: int) -> np.ndarray:
    return np.asarray(
        [incumbent, *range(10, base.CANDIDATE_COUNT)], dtype=np.int64
    )


def _build_pair_cache(
    features: np.ndarray,
    current_scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    current_surface: frozen.ActionSurface,
    current_policy_hit: np.ndarray,
) -> PairCache:
    positive = np.asarray(labels["positive_index"])
    eligible_from = np.asarray(labels["eligible_from"])
    hard_session = np.zeros(base.SESSION_COUNT, dtype=bool)
    for session in np.flatnonzero(~current_policy_hit):
        eligible_index = int(eligible_from[session]) - 1
        present = positive[session, eligible_index:] >= 0
        correct = (
            current_surface.chosen[session, eligible_index:]
            == positive[session, eligible_index:]
        ) & present
        hard_session[session] = bool(np.any(present) and not np.any(correct))

    pair_rows: list[np.ndarray] = []
    pair_labels: list[int] = []
    pair_sessions: list[int] = []
    pair_weights: list[float] = []
    for session in range(base.SESSION_COUNT):
        session_pairs: list[np.ndarray] = []
        eligible_index = int(eligible_from[session]) - 1
        for turn in range(eligible_index, base.TURN_COUNT):
            target = int(positive[session, turn])
            incumbent = int(current_surface.incumbent[session, turn])
            if target < 0 or (target < 10 and target != incumbent):
                continue
            allowed = _allowed_indices(incumbent)
            order = allowed[
                np.argsort(
                    -np.asarray(current_scores[session, turn, allowed]),
                    kind="stable",
                )
            ]
            current = int(current_surface.chosen[session, turn])
            negatives: list[int] = []
            if current != target:
                negatives.append(current)
            else:
                negatives.append(
                    next(int(value) for value in order if int(value) != target)
                )
            if hard_session[session]:
                for candidate in order:
                    candidate = int(candidate)
                    if candidate != target and candidate not in negatives:
                        negatives.append(candidate)
                    if len(negatives) >= 1 + EXTRA_HARD_NEGATIVES:
                        break
            target_row = np.asarray(
                features[session, turn, target], dtype=np.float32
            )
            target_augmented = np.concatenate(
                (
                    target_row,
                    np.asarray(
                        [current_scores[session, turn, target]],
                        dtype=np.float32,
                    ),
                )
            )
            for negative in negatives:
                negative_augmented = np.concatenate(
                    (
                        np.asarray(
                            features[session, turn, negative], dtype=np.float32
                        ),
                        np.asarray(
                            [current_scores[session, turn, negative]],
                            dtype=np.float32,
                        ),
                    )
                )
                session_pairs.append(target_augmented - negative_augmented)
        if not session_pairs:
            continue
        total_weight = (
            HARD_CASE_WEIGHT if hard_session[session] else CONTROL_WEIGHT
        )
        row_weight = total_weight / (2.0 * len(session_pairs))
        for difference in session_pairs:
            pair_rows.extend((difference, -difference))
            pair_labels.extend((1, 0))
            pair_sessions.extend((session, session))
            pair_weights.extend((row_weight, row_weight))
    x = np.asarray(pair_rows, dtype=np.float32)
    y = np.asarray(pair_labels, dtype=np.uint8)
    sessions = np.asarray(pair_sessions, dtype=np.int32)
    weights = np.asarray(pair_weights, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != base.FEATURE_COUNT + 1
        or len(x) != len(y)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(weights).all()
    ):
        raise ResidualEvaluationError("residual pair cache is invalid")
    if int(hard_session.sum()) != 35:
        raise ResidualEvaluationError("hard-case cache does not match attribution")
    return PairCache(x, y, sessions, weights, hard_session)


def _fit_rank_weights(
    cache: PairCache, session_mask: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    rows = session_mask[cache.session]
    x = np.asarray(cache.x[rows], dtype=np.float64)
    y = cache.y[rows]
    weights = cache.weight[rows]
    if not len(x) or len(np.unique(y)) != 2:
        raise ResidualEvaluationError("residual training partition is empty")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    model = LogisticRegression(
        C=MODEL_C,
        solver="liblinear",
        max_iter=300,
        random_state=base._library_seed(seed),
    )
    model.fit((x - mean) / scale, y, sample_weight=weights)
    rank_weights = np.asarray(model.coef_[0] / scale, dtype=np.float64)
    if not np.isfinite(rank_weights).all() or not np.any(rank_weights):
        raise ResidualEvaluationError("residual rank weights are invalid")
    return rank_weights, {
        "pair_rows": int(rows.sum()),
        "sessions": int(np.unique(cache.session[rows]).size),
        "hard_sessions": int(
            np.sum(session_mask & cache.hard_session)
        ),
        "coefficient_l2": float(np.linalg.norm(rank_weights)),
        "iterations": int(model.n_iter_[0]),
    }


def _score_sessions(
    features: np.ndarray,
    current_scores: np.ndarray,
    sessions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    block = np.asarray(features[sessions], dtype=np.float32)
    scores = np.tensordot(
        block,
        np.asarray(weights[:-1], dtype=np.float32),
        axes=([-1], [0]),
    )
    scores += np.asarray(current_scores[sessions], dtype=np.float32) * np.float32(
        weights[-1]
    )
    result = np.asarray(scores, dtype=np.float32)
    if result.shape != (
        len(sessions),
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
    ) or not np.isfinite(result).all():
        raise ResidualEvaluationError("residual score block is invalid")
    return result


def _outer_admission(
    inner_surface: frozen.ActionSurface,
    held_surface: frozen.ActionSurface,
    labels: Mapping[str, np.ndarray],
    outer_fold: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    train_sessions = outer != outer_fold
    held_sessions = outer == outer_fold
    flat_session = np.repeat(np.arange(len(outer)), base.TURN_COUNT)
    flat_x = inner_surface.gate_features.reshape(
        -1, len(base.GATE_FEATURE_NAMES)
    )
    flat_action = inner_surface.action.reshape(-1)
    targets = (
        inner_surface.rescue.reshape(-1),
        inner_surface.regret.reshape(-1),
    )
    weights = (
        inner_surface.rescue_weights.reshape(-1),
        inner_surface.regret_weights.reshape(-1),
    )
    inner_probability = [
        np.zeros_like(inner_surface.action, dtype=np.float32) for _ in range(2)
    ]
    for inner_index in range(base.OUTER_FOLDS):
        model_train = train_sessions & (inner != inner_index)
        model_valid = train_sessions & (inner == inner_index)
        train_rows = flat_action & model_train[flat_session]
        valid_rows = flat_action & model_valid[flat_session]
        for head in range(2):
            inner_probability[head].reshape(-1)[valid_rows] = (
                frozen._fit_predict(
                    flat_x,
                    train_rows,
                    valid_rows,
                    targets[head],
                    weights[head],
                    seed
                    + head * 10_000
                    + outer_fold * 31
                    + inner_index,
                )
            )
    selected = frozen._select_inner_quantile(
        inner_probability[0] - inner_probability[1],
        inner_surface,
        labels,
        train_sessions,
        inner,
    )

    train_rows = flat_action & train_sessions[flat_session]
    held_flat_x = held_surface.gate_features.reshape(
        -1, len(base.GATE_FEATURE_NAMES)
    )
    held_flat_action = held_surface.action.reshape(-1)
    held_rows = held_flat_action & held_sessions[flat_session]
    train_probability: list[np.ndarray] = []
    held_probability: list[np.ndarray] = []
    for head in range(2):
        model, mean, scale = base._fit_gate_model(
            flat_x[train_rows],
            targets[head][train_rows],
            weights[head][train_rows],
            seed + head * 10_000 + outer_fold * 101,
        )
        train_probability.append(
            base._predict_gate(
                model, mean, scale, flat_x[train_rows]
            ).astype(np.float32)
        )
        held_probability.append(
            base._predict_gate(
                model, mean, scale, held_flat_x[held_rows]
            ).astype(np.float32)
        )
    threshold = frozen._threshold_at_quantile(
        train_probability[0] - train_probability[1],
        float(selected["quantile"]),
    )
    held_activation = np.zeros_like(held_surface.action, dtype=bool)
    held_activation.reshape(-1)[held_rows] = (
        held_probability[0] - held_probability[1] >= threshold
    )
    return held_activation, {
        "fold": outer_fold,
        "quantile": float(selected["quantile"]),
        "threshold": threshold,
        "inner_selection": selected,
    }


def _cross_fit_residual_policy(
    features: np.ndarray,
    current_scores: np.ndarray,
    labels: Mapping[str, np.ndarray],
    cache: PairCache,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    outer = np.asarray(labels["outer_fold"])
    inner = np.asarray(labels["inner_fold"])
    chosen = np.zeros((base.SESSION_COUNT, base.TURN_COUNT), dtype=np.uint8)
    activation = np.zeros_like(chosen, dtype=bool)
    records: list[dict[str, Any]] = []
    for outer_fold in range(base.OUTER_FOLDS):
        train_sessions = outer != outer_fold
        held_sessions = outer == outer_fold
        inner_scores = np.zeros_like(current_scores, dtype=np.float32)
        inner_models: list[dict[str, Any]] = []
        for inner_index in range(base.OUTER_FOLDS):
            model_train = train_sessions & (inner != inner_index)
            model_valid = train_sessions & (inner == inner_index)
            rank_weights, audit = _fit_rank_weights(
                cache,
                model_train,
                seed + outer_fold * 101 + inner_index,
            )
            sessions = np.flatnonzero(model_valid)
            inner_scores[sessions] = _score_sessions(
                features, current_scores, sessions, rank_weights
            )
            inner_models.append({"inner_fold": inner_index, **audit})
        inner_surface = frozen._action_surface(
            features, inner_scores, labels
        )

        outer_weights, outer_audit = _fit_rank_weights(
            cache, train_sessions, seed + 50_000 + outer_fold
        )
        held_scores = np.zeros_like(current_scores, dtype=np.float32)
        held_indices = np.flatnonzero(held_sessions)
        held_scores[held_indices] = _score_sessions(
            features, current_scores, held_indices, outer_weights
        )
        held_surface = frozen._action_surface(
            features, held_scores, labels
        )
        held_activation, admission = _outer_admission(
            inner_surface,
            held_surface,
            labels,
            outer_fold,
            seed + 100_000,
        )
        chosen[held_sessions] = held_surface.chosen[held_sessions]
        activation[held_sessions] = held_activation[held_sessions]
        records.append(
            {
                "fold": outer_fold,
                "inner_residual_models": inner_models,
                "outer_residual_model": outer_audit,
                "admission": admission,
            }
        )
    return chosen, activation, records


def run(
    source_root: Path, projection_root: Path, output_path: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = output_path.resolve()
    if output_path.exists() or ROOT not in output_path.parents:
        raise ResidualEvaluationError("output must be a new local path")
    inputs = frozen._load_inputs(source_root, projection_root)
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, _current_selections = (
        attribution._reproduce_nested_activation(
            current_surface, inputs.labels, seed=40220260830
        )
    )
    if (
        hashlib.sha256(current_activation.tobytes()).hexdigest()
        != EXPECTED_CURRENT_ACTIVATION_SHA256
    ):
        raise ResidualEvaluationError("current policy identity mismatch")
    _current_global, _current_folds, current_state = comparison._policy_metrics(
        inputs.labels, current_surface, current_activation
    )
    pair_started = time.perf_counter()
    cache = _build_pair_cache(
        inputs.projected_features,
        inputs.oof_scores,
        inputs.labels,
        current_surface,
        np.asarray(current_state["hit"], dtype=bool),
    )
    pair_seconds = time.perf_counter() - pair_started

    first_started = time.perf_counter()
    first_chosen, first_activation, first_records = (
        _cross_fit_residual_policy(
            inputs.projected_features,
            inputs.oof_scores,
            inputs.labels,
            cache,
            seed=40222060830,
        )
    )
    first_seconds = time.perf_counter() - first_started
    repeat_started = time.perf_counter()
    repeat_chosen, repeat_activation, repeat_records = (
        _cross_fit_residual_policy(
            inputs.projected_features,
            inputs.oof_scores,
            inputs.labels,
            cache,
            seed=40222060830,
        )
    )
    repeat_seconds = time.perf_counter() - repeat_started
    if (
        not np.array_equal(first_chosen, repeat_chosen)
        or not np.array_equal(first_activation, repeat_activation)
        or _canonical_sha256(first_records) != _canonical_sha256(repeat_records)
    ):
        raise ResidualEvaluationError("residual nested OOF repeat differs")

    challenger_state = comparison.metric.policy_session_state(
        inputs.labels, first_chosen, first_activation
    )
    outer = np.asarray(inputs.labels["outer_fold"])
    zero = np.zeros_like(first_activation, dtype=bool)
    p11_state = comparison.metric.policy_session_state(
        inputs.labels, first_chosen, zero
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    challenger_global = comparison.metric.transition_metrics(
        p11_state, challenger_state, first_activation, all_sessions
    )
    challenger_folds = [
        {
            "fold": fold,
            **comparison.metric.transition_metrics(
                p11_state,
                challenger_state,
                first_activation,
                outer == fold,
            ),
        }
        for fold in range(base.OUTER_FOLDS)
    ]
    passed, relative_global, relative_folds = comparison._promotion_gate(
        current_state,
        challenger_state,
        first_activation,
        inputs.labels,
    )

    positive = np.asarray(inputs.labels["positive_index"])
    eligible_from = np.asarray(inputs.labels["eligible_from"])
    current_miss = ~np.asarray(current_state["hit"], dtype=bool)
    correct_sessions = 0
    correct_turns = 0
    for session in np.flatnonzero(current_miss):
        eligible_index = int(eligible_from[session]) - 1
        correct = (
            (positive[session, eligible_index:] >= 0)
            & (
                first_chosen[session, eligible_index:]
                == positive[session, eligible_index:]
            )
        )
        correct_sessions += int(np.any(correct))
        correct_turns += int(correct.sum())
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.2-HARDCASE-RESIDUAL",
        "scope": {
            "split": "train_explore",
            "target_is_training_label_only": True,
            "runtime_features_target_blind": True,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "full_model_or_artifact_trained": False,
            "hyperparameter_sweep": False,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "pair_cache": {
            "rows": len(cache.x),
            "features": cache.x.shape[1],
            "sessions": int(np.unique(cache.session).size),
            "hard_sessions": int(cache.hard_session.sum()),
            "positive_rows": int(cache.y.sum()),
            "bytes": int(
                cache.x.nbytes
                + cache.y.nbytes
                + cache.session.nbytes
                + cache.weight.nbytes
                + cache.hard_session.nbytes
            ),
            "hard_case_weight": HARD_CASE_WEIGHT,
            "control_weight": CONTROL_WEIGHT,
            "extra_hard_negatives": EXTRA_HARD_NEGATIVES,
        },
        "repeat": {
            "chosen_exact": True,
            "activation_exact": True,
            "chosen_sha256": hashlib.sha256(
                first_chosen.tobytes()
            ).hexdigest(),
            "activation_sha256": hashlib.sha256(
                first_activation.tobytes()
            ).hexdigest(),
            "records_canonical_sha256": _canonical_sha256(first_records),
        },
        "ranker_proposals": {
            "current_remaining_miss_sessions": int(current_miss.sum()),
            "correct_proposal_remaining_miss_sessions": correct_sessions,
            "correct_proposal_remaining_miss_turns": correct_turns,
        },
        "challenger": {
            "global": challenger_global,
            "folds": challenger_folds,
            "cross_fit": first_records,
        },
        "relative_to_current": {
            "global": relative_global,
            "folds": relative_folds,
        },
        "decision": {
            "promotion_gate_passed": passed,
            "status": "PROMOTE" if passed else "NO_GO",
            "next": (
                "train and freeze one full residual model"
                if passed
                else "close this residual formulation and choose a materially different mechanism"
            ),
        },
        "timing_seconds": {
            "pair_cache": round(pair_seconds, 6),
            "first_nested_oof": round(first_seconds, 6),
            "repeat_nested_oof": round(repeat_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if ASIN_SHAPE_RE.search(raw):
        raise ResidualEvaluationError("result contains an identity-shaped token")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(raw)
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
    print(
        json.dumps(
            {
                "pair_cache": result["pair_cache"],
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
