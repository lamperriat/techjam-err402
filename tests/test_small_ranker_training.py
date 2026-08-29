from __future__ import annotations

import numpy as np
import pytest

from scripts import train_small_ranker as trainer


def _empty_labels() -> dict[str, np.ndarray]:
    return {
        "baseline_rank": np.zeros((trainer.SESSION_COUNT, trainer.TURN_COUNT), dtype=np.uint8),
        "positive_index": np.full((trainer.SESSION_COUNT, trainer.TURN_COUNT), -1, dtype=np.int16),
        "eligible_from": np.ones(trainer.SESSION_COUNT, dtype=np.uint8),
        "baseline_session_hit": np.zeros(trainer.SESSION_COUNT, dtype=np.uint8),
    }


def test_qid_validation_requires_contiguous_one_positive_groups() -> None:
    qid = np.asarray([0, 0, 0, 4, 4], dtype=np.int32)
    y = np.asarray([0, 1, 0, 0, 1], dtype=np.float32)
    assert trainer.validate_grouped_qid(qid, y) == 2
    with pytest.raises(trainer.SmallRankerTrainingError):
        trainer.validate_grouped_qid(np.asarray([1, 0]), np.asarray([1.0, 1.0]))
    with pytest.raises(trainer.SmallRankerTrainingError):
        trainer.validate_grouped_qid(qid, np.asarray([0, 0, 0, 0, 1], dtype=np.float32))


def test_slot10_choice_never_selects_protected_top9() -> None:
    scores = np.zeros(
        (trainer.SESSION_COUNT, trainer.TURN_COUNT, trainer.CANDIDATE_COUNT),
        dtype=np.float32,
    )
    scores[..., 0] = 100.0
    scores[..., 9] = 1.0
    scores[..., 20] = 2.0
    incumbent = np.full((trainer.SESSION_COUNT, trainer.TURN_COUNT), 9, dtype=np.uint8)
    chosen, margin, top_gap = trainer.choose_slot10(scores, incumbent)
    assert np.all(chosen == 20)
    assert np.allclose(margin, 1.0)
    assert np.all(top_gap >= 1.0)


def test_policy_preserves_ranks_one_to_nine_and_can_swap_rank10() -> None:
    labels = _empty_labels()
    labels["baseline_rank"][0, 0] = 3
    labels["baseline_rank"][1, 0] = 10
    labels["baseline_session_hit"][:2] = 1
    labels["positive_index"][0, 0] = 2
    labels["positive_index"][1, 0] = 9
    chosen = np.zeros((trainer.SESSION_COUNT, trainer.TURN_COUNT), dtype=np.uint8)
    activate = np.zeros_like(chosen, dtype=bool)
    chosen[:2, 0] = 20
    activate[:2, 0] = True
    policy = trainer.policy_session_hits(
        labels["baseline_rank"],
        labels["positive_index"],
        labels["eligible_from"],
        chosen,
        activate,
    )
    assert policy[0] == 1
    assert policy[1] == 0


def test_vectorized_threshold_prefers_rescue_above_harm() -> None:
    labels = _empty_labels()
    labels["positive_index"][0, 0] = 20
    labels["baseline_session_hit"][1] = 1
    labels["baseline_rank"][1, 0] = 10
    labels["positive_index"][1, 0] = 9
    probabilities = np.zeros((trainer.SESSION_COUNT, trainer.TURN_COUNT), dtype=np.float32)
    probabilities[0, 0] = 0.9
    probabilities[1, 0] = 0.8
    action = np.zeros_like(probabilities, dtype=bool)
    action[:2, 0] = True
    chosen = np.zeros_like(labels["positive_index"], dtype=np.uint8)
    chosen[0, 0] = 20
    chosen[1, 0] = 21
    session_mask = np.zeros(trainer.SESSION_COUNT, dtype=bool)
    session_mask[:2] = True
    result = trainer.select_zero_harm_threshold(
        probabilities, action, chosen, labels, session_mask, maximum_thresholds=17
    )
    assert result["miss_to_hit"] == 1
    assert result["hit_to_miss"] == 0
    assert result["activation_turns"] == 1
    assert 0.8 < result["threshold"] <= 0.9


def test_offline_xgboost_compatibility_is_explicit() -> None:
    spec = {
        "objective": "rank:ndcg",
        "max_rounds": 10,
        "max_depth": 4,
        "eta": 0.03,
        "min_child_weight": 8.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "lambdarank_pair_method": "topk",
    }
    old = trainer._model_params(spec, 1, "1.7.6")
    modern = trainer._model_params(spec, 1, "3.2.0")
    assert "lambdarank_pair_method" not in old
    assert modern["lambdarank_pair_method"] == "topk"
    assert modern["lambdarank_num_pair_per_sample"] == 10


def test_gate_schema_contains_no_identity_feature() -> None:
    assert len(trainer.GATE_FEATURE_NAMES) == len(set(trainer.GATE_FEATURE_NAMES))
    assert not any(
        forbidden in name.casefold()
        for name in trainer.GATE_FEATURE_NAMES
        for forbidden in ("asin", "target", "sample_id", "user_id")
    )
