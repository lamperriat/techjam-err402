from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from scripts import evaluate_small_ranker_portfolio_selector as subject


def test_dedup_is_scoped_to_session_turn_and_ors_family_support() -> None:
    family_choices = np.asarray(
        [[[12, 12, 12], [12, 13, 13], [9, 8, 14]]], dtype=np.int16
    )
    current_choice = np.asarray([[9, 9, 9]], dtype=np.int16)
    incumbent = np.asarray([[9, 8, 8]], dtype=np.int16)
    candidates, support, available = subject._deduplicate_actions(
        family_choices, current_choice, incumbent
    )

    observed = {}
    for turn in range(3):
        for slot in range(3):
            if available[0, turn, slot]:
                observed[(0, turn, int(candidates[0, turn, slot]))] = int(
                    support[0, turn, slot]
                )
    assert observed == {
        (0, 0, 12): 0b111,
        (0, 1, 12): 0b001,
        (0, 1, 13): 0b110,
        (0, 2, 14): 0b100,
    }


def test_causal_latch_uses_first_passing_turn_and_is_future_invariant() -> None:
    candidate = np.asarray([[11, 12, 13]], dtype=np.int16)
    utility = np.asarray([[0.4, 0.7, 0.99]], dtype=np.float32)
    available = np.ones_like(candidate, dtype=bool)
    first, first_choice = subject._causal_latch(
        candidate, utility, available, 0.5
    )
    changed = utility.copy()
    changed[0, 2] = 10_000.0
    second, second_choice = subject._causal_latch(
        candidate, changed, available, 0.5
    )

    assert first.tolist() == [[False, True, False]]
    assert first_choice.tolist() == [[-1, 12, -1]]
    assert np.array_equal(first, second)
    assert np.array_equal(first_choice, second_choice)
    assert np.all(first.sum(axis=1) <= 1)


def test_within_turn_tie_break_is_support_then_lower_ordinal() -> None:
    utility = np.asarray([[[0.8, 0.8]]], dtype=np.float32)
    available = np.ones((1, 1, 2), dtype=bool)

    candidates = np.asarray([[[25, 40]]], dtype=np.int16)
    support = np.asarray([[[0b100, 0b011]]], dtype=np.uint8)
    _, chosen, _, _ = subject._within_turn_winner(
        candidates, support, available, utility
    )
    _, reversed_chosen, _, _ = subject._within_turn_winner(
        candidates[..., ::-1],
        support[..., ::-1],
        available[..., ::-1],
        utility[..., ::-1],
    )
    assert chosen.tolist() == [[40]]
    assert np.array_equal(chosen, reversed_chosen)

    candidates = np.asarray([[[25, 17]]], dtype=np.int16)
    support = np.asarray([[[0b001, 0b100]]], dtype=np.uint8)
    _, chosen, _, _ = subject._within_turn_winner(
        candidates, support, available, utility
    )
    _, reversed_chosen, _, _ = subject._within_turn_winner(
        candidates[..., ::-1],
        support[..., ::-1],
        available[..., ::-1],
        utility[..., ::-1],
    )
    assert chosen.tolist() == [[17]]
    assert np.array_equal(chosen, reversed_chosen)


def test_runtime_policy_accepts_early_action_and_api_has_no_labels() -> None:
    candidate = np.asarray([[15, 16]], dtype=np.int16)
    utility = np.asarray([[0.9, 1.0]], dtype=np.float32)
    available = np.ones_like(candidate, dtype=bool)
    supplement, choice = subject._causal_latch(
        candidate, utility, available, 0.5
    )
    assert supplement.tolist() == [[True, False]]
    assert choice.tolist() == [[15, -1]]
    equal_candidate = np.asarray([[21], [22]], dtype=np.int16)
    equal_utility = np.asarray([[0.5], [100.0]], dtype=np.float32)
    equal_available = np.ones_like(equal_candidate, dtype=bool)
    equal_supplement, _ = subject._causal_latch(
        equal_candidate,
        equal_utility,
        equal_available,
        0.5,
        np.asarray([True, False]),
    )
    assert equal_supplement.tolist() == [[True], [False]]

    forbidden = {
        "labels",
        "target",
        "positive_index",
        "current_hit",
        "eligible_from",
        "outer_fold",
        "inner_fold",
    }
    helpers = (
        subject._deduplicate_actions,
        subject._within_turn_winner,
        subject._per_turn_winner_utilities,
        subject._causal_latch,
        subject._map_outer_quantile,
        subject._build_runtime_surface,
        subject._causal_policy,
        subject._compose_policy,
    )
    for helper in helpers:
        assert forbidden.isdisjoint(inspect.signature(helper).parameters)
    assert "rescue" not in subject.RuntimePortfolioSurface.__dataclass_fields__
    assert "regret" not in subject.RuntimePortfolioSurface.__dataclass_fields__


def test_outer_quantile_maps_over_train_per_turn_winners_only() -> None:
    candidates = np.asarray(
        [
            [[10, 11], [-1, -1], [12, 13], [14, 15]],
            [[20, 21], [22, 23], [24, 25], [26, 27]],
        ],
        dtype=np.int16,
    )
    support = np.ones_like(candidates, dtype=np.uint8)
    available = np.ones_like(candidates, dtype=bool)
    available[0, 0, 1] = False
    support[0, 0, 1] = 0
    available[0, 1] = False
    support[0, 1] = 0
    utility = np.asarray(
        [
            [
                [0.2, 1_000_000.0],
                [1_000_000.0, 1_000_000.0],
                [0.5, -200.0],
                [0.9, -300.0],
            ],
            [[99.0, 98.0], [97.0, 96.0], [95.0, 94.0], [93.0, 92.0]],
        ],
        dtype=np.float32,
    )
    _, _, winners, winner_available = subject._per_turn_winner_utilities(
        candidates, support, available, utility
    )
    assert winner_available[0].tolist() == [True, False, True, True]
    assert winners[0, 1] == -np.inf
    train = np.asarray([True, False])
    threshold = subject._map_outer_quantile(
        winners, winner_available, train, 0.5
    )
    assert np.isclose(threshold, 0.5)

    changed = utility.copy()
    changed[0, [0, 2, 3], 1] = -999_999.0
    changed[1] = 1_000_000.0
    _, _, changed_winners, changed_available = (
        subject._per_turn_winner_utilities(
            candidates, support, available, changed
        )
    )
    assert np.isclose(
        subject._map_outer_quantile(
            changed_winners, changed_available, train, 0.5
        ),
        0.5,
    )
    assert math.isinf(
        subject._map_outer_quantile(
            winners,
            winner_available,
            train,
            subject.frozen.KEEP_QUANTILE,
        )
    )
    assert subject.frozen.QUANTILES == tuple(value / 64 for value in range(64))


def test_portfolio_weights_and_policy_membership_fail_closed() -> None:
    available = np.asarray(
        [[[True, True], [False, False]], [[False, False], [False, False]]]
    )
    raw = np.asarray(
        [[[1.0, 3.0], [9.0, 9.0]], [[7.0, 7.0], [7.0, 7.0]]]
    )
    normalized = subject._session_normalize_weights(raw, available)
    assert np.allclose(normalized[0], [[0.25, 0.75], [0.0, 0.0]])
    assert np.all(normalized[1] == 0.0)
    with pytest.raises(subject.PortfolioSelectorError):
        bad = raw.copy()
        bad[0, 0, 0] = np.nan
        subject._session_normalize_weights(bad, available)

    current = np.asarray([[9, 9]], dtype=np.uint8)
    activation = np.asarray([[False, True]])
    candidates = np.asarray([[[12, -1], [13, -1]]], dtype=np.int16)
    action_available = candidates >= 0
    supplement = np.asarray([[True, False]])
    choice = np.asarray([[12, -1]], dtype=np.int16)
    chosen, final_activation = subject._compose_policy(
        current,
        activation,
        candidates,
        action_available,
        supplement,
        choice,
    )
    assert chosen.tolist() == [[12, 9]]
    assert final_activation.tolist() == [[True, True]]
    with pytest.raises(subject.PortfolioSelectorError):
        subject._compose_policy(
            current,
            activation,
            candidates,
            action_available,
            supplement,
            np.asarray([[14, -1]], dtype=np.int16),
        )


def test_fold_integrity_rejects_family_crossing(monkeypatch) -> None:
    monkeypatch.setattr(subject.base, "SESSION_COUNT", 4)
    monkeypatch.setattr(subject.base, "OUTER_FOLDS", 2)
    labels = {
        "family_index": np.asarray([0, 1, 2, 3]),
        "outer_fold": np.asarray([0, 0, 1, 1]),
        "inner_fold": np.asarray([0, 1, 0, 1]),
    }
    summary = subject._validate_fold_integrity(labels)
    assert summary["every_family_outer_and_inner_unique"] is True
    assert summary["outer_by_inner_session_counts"] == [[1, 1], [1, 1]]
    with pytest.raises(subject.PortfolioSelectorError):
        subject._validate_fold_integrity(
            {**labels, "family_index": np.asarray([0, 0, 2, 3])}
        )
