from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.p12_actions import (
    ACTION_IDS,
    ASK,
    BudgetConstraint,
    CANDIDATE_RERANK,
    COMPACT_NEGATIVE_C50,
    FROZEN_SEMANTIC_RERANK,
    GUARDED_COMPACT_SLOT10,
    GUARDED_COMPACT_SLOT10_STRICT,
    HARD_CLAUSE_NOVEL_SLOT10,
    KEEP_P11,
    KEEP_R08,
    P11_EVIDENCE_NOVEL_SLOT10,
    RESULT_AWARE_REWRITE_RETRIEVE,
    TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
    latest_budget_constraint,
    numeric_budget_match,
    rank_compact_negative_c50,
    rank_frozen_semantic_c50,
    rank_guarded_compact_slot10,
    rank_guarded_compact_slot10_strict,
    rank_hard_clause_novel_slot10,
    rank_p11_evidence_novel_slot10,
    rank_structured_c50,
    rank_two_signal_consensus_novel_slot10,
)
from starter.attributes import (
    ProductAttributeView,
    build_conversation_constraint_view,
    build_product_attribute_view,
)
from starter.p8_negative import ExecutableNegative
from starter.p11_features import CandidateScore
from starter.p9_evidence import SLOT_ORDER, VALUE_BITS


def _view(
    identifier: str,
    *,
    price: float | None = None,
    color: str | None = None,
) -> ProductAttributeView:
    product: dict[str, object] = {
        "parent_asin": identifier,
        "title": "plain item",
        "price": price,
    }
    if color is not None:
        product["features"] = [f"color: {color}"]
    return build_product_attribute_view(product)


def _negative(slot: str = "color", value: str = "red") -> ExecutableNegative:
    return ExecutableNegative(slot, value, 1, 1, 1)


def _masks(**values: str) -> tuple[int, ...]:
    return tuple(
        VALUE_BITS[slot].get(values.get(slot, ""), 0)
        for slot in SLOT_ORDER
    )


def _p11_score(
    *,
    relevance: float = 0.20,
    conflict_state: str = "not_applicable",
    idf: float = 0.10,
    title: float = 0.10,
    features: float = 0.10,
    description: float = 0.10,
    hard: float = 0.10,
    subtype: float = 0.10,
    positive: float = 0.10,
) -> CandidateScore:
    return CandidateScore(
        total=relevance,
        relevance=relevance,
        tie_bonus=0.0,
        conflict_state=conflict_state,
        broad_rank_prior=0.0,
        strict_rank_prior=0.0,
        rrf_rank_prior=0.0,
        idf_any_field_coverage=idf,
        title_category_coverage=title,
        features_details_coverage=features,
        description_store_coverage=description,
        latest_hard_clause_coverage=hard,
        subtype_consistency=subtype,
        positive_constraint_evidence=positive,
    )


def _novel_inputs() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, CandidateScore],
]:
    pool = tuple(f"item-{index:02d}" for index in range(15))
    structured = (*pool[:9], pool[10], pool[9], *pool[11:])
    semantic = (*pool[:9], pool[11], pool[9], pool[10], *pool[12:])
    scores = {identifier: _p11_score() for identifier in pool}
    return pool, structured, semantic, scores


class BudgetParsingTests(unittest.TestCase):
    def test_latest_around_max_and_min_budget_wins(self) -> None:
        history = [
            "My budget is around $1,200.50.",
            "Actually, I can spend at most $900.",
            "On second thought, at least USD $750 is fine.",
        ]

        self.assertEqual(
            latest_budget_constraint(history),
            BudgetConstraint("min", 750.0),
        )

    def test_no_preference_for_budget_clears_earlier_value(self) -> None:
        self.assertIsNone(
            latest_budget_constraint(
                [
                    "I need a budget under $80.",
                    "I don't have an additional preference for budget.",
                ]
            )
        )

    def test_budget_after_no_preference_becomes_active(self) -> None:
        self.assertEqual(
            latest_budget_constraint(
                ["Any budget works.", "Actually, budget around $45 matters."]
            ),
            BudgetConstraint("around", 45.0),
        )

    def test_non_message_history_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            latest_budget_constraint("budget around $50")

    def test_v1_numeric_budget_match_formula_is_frozen(self) -> None:
        around = BudgetConstraint("around", 100.0)
        self.assertEqual(numeric_budget_match(100.0, around), 1.0)
        self.assertEqual(numeric_budget_match(50.0, around), 0.5)
        self.assertEqual(numeric_budget_match(250.0, around), 0.0)
        self.assertEqual(
            numeric_budget_match(80.0, BudgetConstraint("max", 80.0)),
            1.0,
        )
        self.assertEqual(
            numeric_budget_match(80.01, BudgetConstraint("max", 80.0)),
            0.0,
        )
        self.assertEqual(
            numeric_budget_match(80.0, BudgetConstraint("min", 80.0)),
            1.0,
        )
        self.assertEqual(
            numeric_budget_match(79.99, BudgetConstraint("min", 80.0)),
            0.0,
        )


class StructuredC50Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.empty_intent = build_conversation_constraint_view("", (), ())

    def test_negative_constraint_can_change_complete_pool_order(self) -> None:
        pool = ("red-item", "blue-item")
        views = {
            "red-item": _view("red-item", color="red"),
            "blue-item": _view("blue-item", color="blue"),
        }
        intent = build_conversation_constraint_view("", (), ("red",))

        ranked = rank_structured_c50(
            pool,
            intent,
            views,
            {identifier: 0.5 for identifier in pool},
        )

        self.assertEqual(ranked, ("blue-item", "red-item"))
        self.assertEqual(set(ranked), set(pool))

    def test_around_budget_prefers_closest_numeric_price(self) -> None:
        pool = ("low", "high", "near")
        views = {
            "low": _view("low", price=40.0),
            "high": _view("high", price=160.0),
            "near": _view("near", price=100.0),
        }

        ranked = rank_structured_c50(
            pool,
            self.empty_intent,
            views,
            {identifier: 0.0 for identifier in pool},
            ["The budget is around $100."],
        )

        self.assertEqual(ranked[0], "near")
        self.assertEqual(set(ranked), set(pool))

    def test_max_and_min_budget_apply_directional_match(self) -> None:
        pool = ("expensive", "cheap")
        views = {
            "expensive": _view("expensive", price=150.0),
            "cheap": _view("cheap", price=50.0),
        }
        priors = {identifier: 0.0 for identifier in pool}

        maximum = rank_structured_c50(
            pool,
            self.empty_intent,
            views,
            priors,
            ["Please keep it below $80."],
        )
        minimum = rank_structured_c50(
            pool,
            self.empty_intent,
            views,
            priors,
            ["I want something at least $100."],
        )

        self.assertEqual(maximum, ("cheap", "expensive"))
        self.assertEqual(minimum, ("expensive", "cheap"))

    def test_later_no_preference_removes_budget_reranking(self) -> None:
        pool = ("far", "near")
        views = {
            "far": _view("far", price=180.0),
            "near": _view("near", price=100.0),
        }

        ranked = rank_structured_c50(
            pool,
            self.empty_intent,
            views,
            {identifier: 0.0 for identifier in pool},
            ["Budget around $100.", "Budget doesn't matter."],
        )

        self.assertEqual(ranked, pool)

    def test_equal_scores_preserve_original_order_and_exact_membership(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(50))
        views = {identifier: _view(identifier) for identifier in pool}

        ranked = rank_structured_c50(
            pool,
            self.empty_intent,
            views,
            {identifier: 0.25 for identifier in pool},
        )

        self.assertEqual(ranked, pool)
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_invalid_unique_finite_and_coverage_inputs_fail_closed(self) -> None:
        duplicate = ("same", "same")
        self.assertEqual(
            rank_structured_c50(
                duplicate,
                self.empty_intent,
                {"same": _view("same")},
                {"same": 0.5},
            ),
            duplicate,
        )

        pool = ("first", "second")
        views = {identifier: _view(identifier) for identifier in pool}
        self.assertEqual(
            rank_structured_c50(
                pool,
                self.empty_intent,
                views,
                {"first": math.nan, "second": 0.5},
            ),
            pool,
        )
        self.assertEqual(
            rank_structured_c50(
                pool,
                self.empty_intent,
                {"first": views["first"]},
                {identifier: 0.5 for identifier in pool},
            ),
            pool,
        )

    def test_budget_with_missing_price_fails_closed(self) -> None:
        pool = ("known", "unknown")
        views = {
            "known": _view("known", price=50.0),
            "unknown": _view("unknown"),
        }

        self.assertEqual(
            rank_structured_c50(
                pool,
                self.empty_intent,
                views,
                {identifier: 0.0 for identifier in pool},
                ["Budget under $80."],
            ),
            pool,
        )

    def test_non_finite_scorer_output_fails_closed(self) -> None:
        pool = ("first", "second")
        views = {identifier: _view(identifier) for identifier in pool}
        with patch(
            "scripts.p12_actions.score_candidate",
            return_value=SimpleNamespace(total=math.nan),
        ):
            ranked = rank_structured_c50(
                pool,
                self.empty_intent,
                views,
                {identifier: 0.5 for identifier in pool},
            )
        self.assertEqual(ranked, pool)

    def test_more_than_c50_fails_closed(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(51))
        self.assertEqual(
            rank_structured_c50(pool, self.empty_intent, {}, {}),
            pool,
        )


class FrozenSemanticC50Tests(unittest.TestCase):
    def test_fixed_hybrid_is_deterministic_and_preserves_membership(self) -> None:
        pool = ("a", "b", "c", "d")
        cosine = {"a": 0.0, "b": -0.5, "c": 1.0, "d": -0.5}

        first = rank_frozen_semantic_c50(pool, cosine)
        second = rank_frozen_semantic_c50(pool, cosine)

        self.assertEqual(first, second)
        self.assertEqual(first, ("a", "c", "b", "d"))
        self.assertEqual(set(first), set(pool))

    def test_all_equal_cosine_fails_closed_to_original_order(self) -> None:
        pool = ("a", "b", "c")
        self.assertEqual(
            rank_frozen_semantic_c50(
                pool,
                {identifier: 0.25 for identifier in pool},
            ),
            pool,
        )

    def test_missing_or_non_finite_cosine_fails_closed(self) -> None:
        pool = ("a", "b", "c")
        self.assertEqual(
            rank_frozen_semantic_c50(pool, {"a": 0.0, "b": 1.0}),
            pool,
        )
        self.assertEqual(
            rank_frozen_semantic_c50(
                pool,
                {"a": 0.0, "b": math.inf, "c": 1.0},
            ),
            pool,
        )

    def test_duplicate_pool_fails_closed(self) -> None:
        pool = ("a", "a", "b")
        self.assertEqual(
            rank_frozen_semantic_c50(pool, {"a": 0.0, "b": 1.0}),
            pool,
        )


class CompactNegativeActionTests(unittest.TestCase):
    def test_no_executable_negative_is_exact_p11_noop(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        evidence = {identifier: _masks(color="blue") for identifier in pool}

        self.assertEqual(rank_compact_negative_c50(pool, evidence, ()), pool)
        self.assertEqual(rank_guarded_compact_slot10(pool, evidence, ()), pool)
        self.assertEqual(
            rank_guarded_compact_slot10_strict(pool, evidence, ()), pool
        )

    def test_full_partition_is_stable_and_uses_unknown_before_violation(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        evidence = {identifier: _masks() for identifier in pool}
        evidence[pool[0]] = _masks(color="red")
        evidence[pool[10]] = _masks(color="blue")
        evidence[pool[11]] = _masks(color="red")

        ranked = rank_compact_negative_c50(pool, evidence, (_negative(),))

        self.assertEqual(ranked, (pool[10], *pool[1:10], pool[0], pool[11]))
        self.assertEqual(set(ranked), set(pool))

    def test_guarded_action_uses_first_strictly_better_class(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        evidence = {identifier: _masks(color="blue") for identifier in pool}
        evidence[pool[9]] = _masks(color="red")
        evidence[pool[10]] = _masks()

        ranked = rank_guarded_compact_slot10(pool, evidence, (_negative(),))

        expected = list(pool)
        expected[9], expected[10] = expected[10], expected[9]
        self.assertEqual(ranked, tuple(expected))
        self.assertEqual(ranked[:9], pool[:9])
        self.assertEqual(set(ranked[:10]) ^ set(pool[:10]), {pool[9], pool[10]})

    def test_strict_action_only_repairs_compatible_adjacent_rank11(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        evidence = {identifier: _masks(color="blue") for identifier in pool}
        evidence[pool[9]] = _masks(color="red")

        ranked = rank_guarded_compact_slot10_strict(
            pool, evidence, (_negative(),)
        )

        expected = list(pool)
        expected[9], expected[10] = expected[10], expected[9]
        self.assertEqual(ranked, tuple(expected))
        self.assertEqual(ranked[:9], pool[:9])
        self.assertEqual(ranked[11:], pool[11:])

        evidence[pool[10]] = _masks()
        self.assertEqual(
            rank_guarded_compact_slot10_strict(
                pool, evidence, (_negative(),)
            ),
            pool,
        )

    def test_guarded_action_rejects_no_better_class_or_earlier_violation(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        unknown_tail = {identifier: _masks(color="blue") for identifier in pool}
        unknown_tail[pool[9]] = _masks()
        unknown_tail[pool[10]] = _masks()
        unknown_tail[pool[11]] = _masks(color="red")
        self.assertEqual(
            rank_guarded_compact_slot10(pool, unknown_tail, (_negative(),)),
            pool,
        )

        earlier_violation = {identifier: _masks(color="blue") for identifier in pool}
        earlier_violation[pool[0]] = _masks(color="red")
        earlier_violation[pool[9]] = _masks(color="red")
        self.assertEqual(
            rank_guarded_compact_slot10(pool, earlier_violation, (_negative(),)),
            pool,
        )
        self.assertEqual(
            rank_guarded_compact_slot10_strict(
                pool, earlier_violation, (_negative(),)
            ),
            pool,
        )

    def test_invalid_compact_inputs_fail_closed(self) -> None:
        pool = tuple(f"item-{index:02d}" for index in range(12))
        incomplete = {identifier: _masks(color="blue") for identifier in pool[:-1]}
        self.assertEqual(
            rank_compact_negative_c50(pool, incomplete, (_negative(),)),
            pool,
        )
        self.assertEqual(
            rank_compact_negative_c50(
                pool,
                {identifier: _masks(color="blue") for identifier in pool},
                (_negative(value="outside-registry"),),
            ),
            pool,
        )

        oversized = tuple(f"item-{index:02d}" for index in range(51))
        self.assertEqual(
            rank_compact_negative_c50(
                oversized,
                {identifier: _masks(color="blue") for identifier in oversized},
                (_negative(),),
            ),
            oversized,
        )


class NovelSlot10ActionTests(unittest.TestCase):
    def test_evidence_action_uses_only_novel_tail_and_swaps_rank10_once(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        scores[pool[10]] = _p11_score(
            relevance=1.0,
            idf=1.0,
            title=1.0,
            features=1.0,
            description=1.0,
            hard=1.0,
            subtype=1.0,
            positive=1.0,
        )
        scores[pool[13]] = _p11_score(
            relevance=0.30,
            idf=0.40,
            title=0.40,
            hard=0.70,
        )

        ranked = rank_p11_evidence_novel_slot10(
            pool,
            structured,
            semantic,
            scores,
            has_non_category_signal=True,
        )

        expected = list(pool)
        expected[9], expected[13] = expected[13], expected[9]
        self.assertEqual(ranked, tuple(expected))
        self.assertEqual(ranked[:9], pool[:9])
        self.assertEqual(set(ranked[:10]) ^ set(pool[:10]), {pool[9], pool[13]})
        self.assertEqual(ranked.index(pool[10]), 10)

    def test_evidence_action_fails_closed_without_signal_or_unique_margin(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        challenger = _p11_score(
            relevance=0.30,
            idf=0.40,
            title=0.40,
            hard=0.70,
        )
        scores[pool[12]] = challenger
        scores[pool[13]] = challenger

        self.assertEqual(
            rank_p11_evidence_novel_slot10(
                pool,
                structured,
                semantic,
                scores,
                has_non_category_signal=False,
            ),
            pool,
        )
        self.assertEqual(
            rank_p11_evidence_novel_slot10(
                pool,
                structured,
                semantic,
                scores,
                has_non_category_signal=True,
            ),
            pool,
        )

    def test_hard_clause_action_requires_long_unique_field_local_match(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        scores[pool[9]] = _p11_score(
            relevance=0.40,
            idf=0.20,
            hard=0.30,
            subtype=0.20,
            positive=0.20,
        )
        scores[pool[12]] = _p11_score(
            relevance=0.20,
            idf=0.20,
            hard=0.60,
            subtype=0.20,
            positive=0.20,
        )
        scores[pool[13]] = _p11_score(
            relevance=0.39,
            idf=0.20,
            hard=0.90,
            subtype=0.20,
            positive=0.20,
        )

        ranked = rank_hard_clause_novel_slot10(
            pool,
            structured,
            semantic,
            scores,
            hard_clause_term_count=4,
        )
        expected = list(pool)
        expected[9], expected[13] = expected[13], expected[9]
        self.assertEqual(ranked, tuple(expected))

        self.assertEqual(
            rank_hard_clause_novel_slot10(
                pool,
                structured,
                semantic,
                scores,
                hard_clause_term_count=3,
            ),
            pool,
        )
        close_runner = dict(scores)
        close_runner[pool[12]] = _p11_score(
            relevance=0.20,
            idf=0.20,
            hard=0.66,
            subtype=0.20,
            positive=0.20,
        )
        self.assertEqual(
            rank_hard_clause_novel_slot10(
                pool,
                structured,
                semantic,
                close_runner,
                hard_clause_term_count=4,
            ),
            pool,
        )

    def test_two_signal_consensus_activates_at_inclusive_boundaries(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        scores[pool[10]] = _p11_score(
            relevance=1.0,
            idf=1.0,
            title=1.0,
            features=1.0,
            hard=1.0,
            subtype=1.0,
            positive=1.0,
        )
        scores[pool[12]] = _p11_score(
            relevance=0.20,
            idf=0.20,
            title=0.20,
            features=0.20,
            hard=0.20,
            subtype=0.20,
            positive=0.20,
        )
        scores[pool[13]] = _p11_score(
            relevance=0.22,
            idf=0.25,
            title=0.25,
            features=0.25,
            hard=0.30,
            subtype=0.30,
            positive=0.30,
        )

        ranked = rank_two_signal_consensus_novel_slot10(
            pool,
            structured,
            semantic,
            scores,
            has_non_category_signal=True,
        )
        expected = list(pool)
        expected[9], expected[13] = expected[13], expected[9]
        self.assertEqual(ranked, tuple(expected))
        self.assertEqual(ranked[:9], pool[:9])
        self.assertEqual(set(ranked[:10]) ^ set(pool[:10]), {pool[9], pool[13]})

        below_relevance = dict(scores)
        below_relevance[pool[13]] = _p11_score(
            relevance=0.219999999999,
            idf=0.25,
            title=0.25,
            features=0.25,
            hard=0.30,
            subtype=0.30,
            positive=0.30,
        )
        self.assertEqual(
            rank_two_signal_consensus_novel_slot10(
                pool,
                structured,
                semantic,
                below_relevance,
                has_non_category_signal=True,
            ),
            pool,
        )

    def test_two_signal_consensus_requires_same_argmax_and_visible_signal(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        scores[pool[12]] = _p11_score(
            relevance=0.40,
            idf=0.25,
            title=0.25,
            features=0.25,
            hard=0.60,
            subtype=0.60,
            positive=0.60,
        )
        scores[pool[13]] = _p11_score(
            relevance=0.40,
            idf=0.50,
            title=0.50,
            features=0.50,
            hard=0.25,
            subtype=0.25,
            positive=0.25,
        )

        self.assertEqual(
            rank_two_signal_consensus_novel_slot10(
                pool,
                structured,
                semantic,
                scores,
                has_non_category_signal=True,
            ),
            pool,
        )
        self.assertEqual(
            rank_two_signal_consensus_novel_slot10(
                pool,
                structured,
                semantic,
                scores,
                has_non_category_signal=False,
            ),
            pool,
        )

    def test_invalid_family2_inputs_are_exact_noops(self) -> None:
        pool, structured, semantic, scores = _novel_inputs()
        scores[pool[13]] = _p11_score(
            relevance=0.30,
            idf=0.40,
            title=0.40,
            hard=0.70,
        )
        incomplete_scores = dict(scores)
        incomplete_scores.pop(pool[-1])
        self.assertEqual(
            rank_p11_evidence_novel_slot10(
                pool,
                structured,
                semantic,
                incomplete_scores,
                has_non_category_signal=True,
            ),
            pool,
        )

        non_finite_scores = dict(scores)
        non_finite_scores[pool[-1]] = _p11_score(idf=math.nan)
        self.assertEqual(
            rank_p11_evidence_novel_slot10(
                pool,
                structured,
                semantic,
                non_finite_scores,
                has_non_category_signal=True,
            ),
            pool,
        )
        self.assertEqual(
            rank_p11_evidence_novel_slot10(
                pool,
                structured,
                semantic[:-1],
                scores,
                has_non_category_signal=True,
            ),
            pool,
        )


class ActionIdTests(unittest.TestCase):
    def test_goal_action_ids_are_frozen_and_unique(self) -> None:
        self.assertEqual(
            ACTION_IDS,
            (
                KEEP_R08,
                KEEP_P11,
                CANDIDATE_RERANK,
                FROZEN_SEMANTIC_RERANK,
                RESULT_AWARE_REWRITE_RETRIEVE,
                COMPACT_NEGATIVE_C50,
                GUARDED_COMPACT_SLOT10,
                GUARDED_COMPACT_SLOT10_STRICT,
                P11_EVIDENCE_NOVEL_SLOT10,
                HARD_CLAUSE_NOVEL_SLOT10,
                TWO_SIGNAL_CONSENSUS_NOVEL_SLOT10,
                ASK,
            ),
        )
        self.assertEqual(len(ACTION_IDS), len(set(ACTION_IDS)))


if __name__ == "__main__":
    unittest.main()
