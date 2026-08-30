from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

from scripts import probe_small_ranker_strict_outer_restack as subject


def test_choose_slot10_any_matches_frozen_full_shape() -> None:
    rng = np.random.default_rng(402)
    scores = rng.standard_normal(
        (
            subject.base.SESSION_COUNT,
            subject.base.TURN_COUNT,
            subject.base.CANDIDATE_COUNT,
        )
    ).astype(np.float32)
    incumbent = rng.integers(
        0,
        10,
        size=(subject.base.SESSION_COUNT, subject.base.TURN_COUNT),
        dtype=np.uint8,
    )

    expected = subject.base.choose_slot10(scores, incumbent)
    actual = subject.choose_slot10_any(scores, incumbent)

    assert len(actual) == len(expected)
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_gate_feature_matrix_any_matches_small_reference() -> None:
    rng = np.random.default_rng(403)
    features = rng.standard_normal(
        (
            3,
            subject.base.TURN_COUNT,
            subject.base.CANDIDATE_COUNT,
            subject.base.FEATURE_COUNT,
        )
    ).astype(np.float32)
    sessions = np.asarray([2, 0], dtype=np.int16)
    scores = rng.standard_normal(
        (
            len(sessions),
            subject.base.TURN_COUNT,
            subject.base.CANDIDATE_COUNT,
        )
    ).astype(np.float32)
    incumbent = np.broadcast_to(
        np.arange(subject.base.TURN_COUNT, dtype=np.uint8)[None, :] % 10,
        (len(sessions), subject.base.TURN_COUNT),
    ).copy()
    chosen, margin, top_gap = subject.choose_slot10_any(scores, incumbent)

    actual = subject.gate_feature_matrix_any(
        features,
        sessions,
        scores,
        chosen,
        incumbent,
        margin,
        top_gap,
        batch_size=1,
    )

    feature_rows = features[sessions].reshape(
        -1, subject.base.CANDIDATE_COUNT, subject.base.FEATURE_COUNT
    )
    score_rows = scores.reshape(-1, subject.base.CANDIDATE_COUNT)
    chosen_rows = chosen.reshape(-1).astype(np.int64)
    incumbent_rows = incumbent.reshape(-1).astype(np.int64)
    rows = np.arange(len(chosen_rows))
    chosen_features = feature_rows[rows, chosen_rows]
    incumbent_features = feature_rows[rows, incumbent_rows]
    static_columns = [
        subject.base.FEATURE_INDEX[name]
        for name in subject.base.GATE_STATIC_FEATURES
    ]
    conflict_columns = [
        subject.base.FEATURE_INDEX[f"{slot}_conflict"]
        for slot in subject.base.CONSTRAINT_SLOTS
    ]
    active_recall = subject.base.FEATURE_INDEX["active_token_recall"]
    hard_coverage = subject.base.FEATURE_INDEX["hard_clause_coverage"]
    expected = np.column_stack(
        (
            margin.reshape(-1),
            top_gap.reshape(-1),
            score_rows[rows, chosen_rows],
            score_rows[rows, incumbent_rows],
            chosen_features[:, static_columns],
            chosen_features[:, active_recall]
            - incumbent_features[:, active_recall],
            chosen_features[:, hard_coverage]
            - incumbent_features[:, hard_coverage],
            chosen_features[:, conflict_columns].sum(axis=1)
            - incumbent_features[:, conflict_columns].sum(axis=1),
        )
    ).astype(np.float32)
    expected = expected.reshape(
        len(sessions), subject.base.TURN_COUNT, len(subject.base.GATE_FEATURE_NAMES)
    )

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_focused_a00_cohort_excludes_non_a00_sessions() -> None:
    t0_sessions = np.asarray([10, 20, 30, 40], dtype=np.int16)
    positive = np.full((4, subject.base.TURN_COUNT), -1, dtype=np.int16)
    positive[0, :] = 15  # V00: would qualify as hard without the domain mask.
    positive[1, 0] = 12  # A00 hard session.
    positive[2, :2] = 13  # A00 rank-10 control session.
    positive[3, 0] = 14  # A00 protected hit, therefore neither cohort.
    labels_t0 = {
        "positive_index": positive,
        "eligible_from": np.ones(4, dtype=np.uint8),
        "inner_fold": np.asarray([0, 1, 2, 3], dtype=np.uint8),
    }
    inputs = subject.ProbeInputs(
        raw_features=np.empty(0, dtype=np.float32),
        projected_features=np.empty(0, dtype=np.float32),
        outer_fold=np.zeros(50, dtype=np.uint8),
        inner_fold=np.zeros(50, dtype=np.uint8),
        family_index=np.arange(50, dtype=np.int32),
        t0_sessions=t0_sessions,
        labels_t0=labels_t0,
        paths={},
    )
    state = {
        "hit": np.asarray([False, False, True, True]),
        "first_rank": np.asarray([0, 0, 10, 1], dtype=np.int16),
        "first_turn": np.asarray([11, 11, 2, 1], dtype=np.int16),
    }
    incumbent = np.zeros((4, subject.base.TURN_COUNT), dtype=np.uint8)

    groups = subject._select_focused_a00_groups(inputs, state, incumbent)

    assert groups.local_session.tolist() == [1, 2, 2]
    assert groups.global_session.tolist() == [20, 30, 30]
    assert groups.turn.tolist() == [0, 0, 1]
    assert groups.hard.tolist() == [1, 0, 0]
    assert not groups.hard_session[0]
    assert not groups.control_session[0]
    assert np.all(labels_t0["inner_fold"][groups.local_session] != 0)


def test_crossfit_current_uses_other_inner_folds_once_and_keeps_safely(
    monkeypatch,
) -> None:
    session_count = subject.base.OUTER_FOLDS
    shape = (session_count, subject.base.TURN_COUNT)
    gate = np.zeros((*shape, len(subject.base.GATE_FEATURE_NAMES)), dtype=np.float32)
    gate[..., 0] = np.arange(session_count, dtype=np.float32)[:, None]
    action = np.ones(shape, dtype=bool)
    zeros = np.zeros(shape, dtype=np.uint8)
    ones = np.ones(shape, dtype=np.float64)
    surface = subject.frozen.ActionSurface(
        incumbent=zeros,
        chosen=np.full(shape, 10, dtype=np.uint8),
        action=action,
        gate_features=gate,
        rescue=zeros,
        rescue_weights=ones,
        regret=zeros,
        regret_weights=ones,
    )
    labels = {"inner_fold": np.arange(session_count, dtype=np.uint8)}
    fit_calls: list[tuple[int, tuple[int, ...]]] = []
    selection_calls = []

    class FakeModel:
        def __init__(self, seed: int) -> None:
            self.probability = (seed % 97) / 100.0

    def fake_fit(x, _y, _weights, seed):
        fit_calls.append(
            (
                seed,
                tuple(sorted(set(x[:, 0].astype(int).tolist()))),
            )
        )
        return FakeModel(seed), np.zeros(x.shape[1]), np.ones(x.shape[1])

    def fake_predict(model, _mean, _scale, x):
        return np.full(len(x), model.probability, dtype=np.float64)

    def fake_select(utility, received_surface, _labels, session_mask, inner):
        selection_calls.append(utility.copy())
        assert received_surface is surface
        assert np.all(session_mask)
        np.testing.assert_array_equal(inner, labels["inner_fold"])
        return {"quantile": 1.0, "inner_threshold": math.inf}

    def fake_state(_labels, _chosen, activation):
        assert not np.any(activation)
        return {
            "hit": np.zeros(session_count, dtype=bool),
            "first_rank": np.zeros(session_count, dtype=np.int16),
            "first_turn": np.full(session_count, 11, dtype=np.int16),
        }

    monkeypatch.setattr(subject.base, "_fit_gate_model", fake_fit)
    monkeypatch.setattr(subject.base, "_predict_gate", fake_predict)
    monkeypatch.setattr(subject.frozen, "_select_inner_quantile", fake_select)
    monkeypatch.setattr(subject.metric, "policy_session_state", fake_state)

    _state, activation, record = subject._crossfit_current_t0(surface, labels)

    assert len(selection_calls) == 1
    assert not np.any(activation)
    assert record["selected_quantile"] == 1.0
    assert record["inner_threshold"] == "inf"
    assert len(fit_calls) == 10
    for inner_fold in range(subject.base.OUTER_FOLDS):
        expected_sessions = tuple(
            value
            for value in range(subject.base.OUTER_FOLDS)
            if value != inner_fold
        )
        assert fit_calls[inner_fold * 2][1] == expected_sessions
        assert fit_calls[inner_fold * 2 + 1][1] == expected_sessions
        assert fit_calls[inner_fold * 2][0] == subject.BASE_SEED + inner_fold
        assert (
            fit_calls[inner_fold * 2 + 1][0]
            == subject.BASE_SEED + 10_000 + inner_fold
        )
        assert record["folds"][inner_fold]["train_action_rows"] == 40
        assert record["folds"][inner_fold]["valid_action_rows"] == 10
    json.dumps(record, allow_nan=False)


def _pass_record() -> dict[str, object]:
    return {
        "name": "first",
        "models": [
            {
                "model_id": "pairwise_d4_control",
                "domain": "A_00",
                "seed": 123,
                "rounds": 397,
                "train_sessions": 1296,
                "train_session_mask_sha256": "train-mask",
                "train_rows": 483680,
                "train_query_groups": 12092,
                "score_session_sha256": "score-sessions",
                "training_seconds": 1.0,
                "prediction": {"total_seconds": 2.0},
                "model": {"sha256": "model-hash"},
                "score": {"array_sha256": "score-hash"},
                "choice": {"array_sha256": "choice-hash"},
                "serialized_model_parity": {
                    "sample_session_sha256": "sample-hash",
                    "rows": 2000,
                    "maximum_absolute_error": 0.0,
                    "c100_order_exact": True,
                },
            }
        ],
        "current": {
            "files": {
                "scores": {"array_sha256": "current-score-hash"},
                "activation": {"array_sha256": "current-activation-hash"},
            },
            "gate": {
                "head_0_probability_sha256": "head-0-hash",
                "head_1_probability_sha256": "head-1-hash",
            },
        },
        "focused_cache": {
            "query_groups": 3,
            "hard_sessions": 1,
            "control_sessions": 1,
            "hard_groups": 1,
            "control_groups": 2,
            "group_weight_sha256": "weight-hash",
            "files": {
                "features": {"array_sha256": "focused-feature-hash"},
                "relevance": {"array_sha256": "focused-label-hash"},
            },
            "timing_seconds": {"total": 3.0},
        },
        "rrf3": {
            "score": {"array_sha256": "rrf-score-hash"},
            "choice": {"array_sha256": "rrf-choice-hash"},
        },
        "timing_seconds": {"total": 4.0},
        "peak_working_set_bytes": 100,
    }


def test_repeat_identity_ignores_timing_but_detects_hash_change() -> None:
    first = _pass_record()
    changed_timing = deepcopy(first)
    changed_timing["timing_seconds"]["total"] = 400.0
    changed_timing["peak_working_set_bytes"] = 999
    changed_timing["models"][0]["training_seconds"] = 100.0
    changed_timing["models"][0]["prediction"]["total_seconds"] = 200.0
    changed_timing["focused_cache"]["timing_seconds"]["total"] = 300.0

    expected = subject._repeat_identity(first)
    assert subject._repeat_identity(changed_timing) == expected

    changed_hash = deepcopy(changed_timing)
    changed_hash["models"][0]["score"]["array_sha256"] = "different"
    assert subject._repeat_identity(changed_hash) != expected
