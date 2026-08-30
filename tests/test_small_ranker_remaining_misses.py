from __future__ import annotations

from scripts import analyze_small_ranker_remaining_misses as analyzer


def test_remaining_miss_classes_are_exclusive_and_ordered() -> None:
    assert (
        analyzer._exclusive_class(
            candidate_present=False,
            correct_proposal=True,
            ambiguous=True,
        )
        == "A_candidate_absent"
    )
    assert (
        analyzer._exclusive_class(
            candidate_present=True,
            correct_proposal=True,
            ambiguous=True,
        )
        == "C_admission_failure"
    )
    assert (
        analyzer._exclusive_class(
            candidate_present=True,
            correct_proposal=False,
            ambiguous=True,
        )
        == "D_irrecoverable_ambiguous"
    )
    assert (
        analyzer._exclusive_class(
            candidate_present=True,
            correct_proposal=False,
            ambiguous=False,
        )
        == "B_ranker_failure"
    )


def test_candidate_recall_honors_eligible_turn_and_pool_prefix() -> None:
    turns = [
        {"c100": tuple(["target", *[f"a-{index}" for index in range(99)]])},
        {"c100": tuple([*[f"b-{index}" for index in range(75)], "target", *[f"c-{index}" for index in range(24)]])},
    ]
    flags = analyzer._candidate_recall_flags(turns, "target", eligible_turn=2)
    assert flags == {10: False, 20: False, 50: False, 100: True}


def test_nonpositional_ambiguity_projection_has_no_rank_features() -> None:
    assert analyzer.NONPOSITIONAL_EVIDENCE_NAMES
    assert not any(
        "rank_fraction" in name
        or "reciprocal_rank" in name
        or "presence" == name
        for name in analyzer.NONPOSITIONAL_EVIDENCE_NAMES
    )
