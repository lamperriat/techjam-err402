from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from starter.attributes import (
    AttributeValue,
    ConstraintValue,
    ConversationConstraintView,
    ProductAttributeView,
)
from starter.reranker import (
    PRESERVED_TOP_K,
    RERANK_TOP_N,
    WEIGHTS,
    has_usable_evidence,
    rerank_top_n,
    score_candidate,
)


def attribute(value: str, *, confidence: float = 1.0) -> AttributeValue:
    return AttributeValue(value, "test", confidence, value)


class CandidateScoringTests(unittest.TestCase):
    def test_score_breakdown_matches_documented_weighted_formula(self) -> None:
        intent = ConversationConstraintView(
            category_terms=("dress",),
            positive=(ConstraintValue("material", "cotton", 1, 1.0, "test"),),
            negative=(ConstraintValue("material", "polyester", -1, 1.0, "test"),),
            exact_terms=("pocket",),
        )
        product = ProductAttributeView(
            parent_asin="MATCH",
            category=(attribute("dress"),),
            material=(
                attribute("cotton", confidence=0.9),
                attribute("polyester", confidence=0.9),
            ),
            feature_phrases=(attribute("pocket"),),
        )

        score = score_candidate(intent, product, normalized_rrf=0.5)

        self.assertEqual(score.rrf_prior, 0.5)
        self.assertEqual(score.category_consistency, 1.0)
        self.assertEqual(score.positive_slot_match, 0.9)
        self.assertEqual(score.exact_feature_match, 1.0)
        self.assertEqual(score.negative_violation, 0.9)
        expected = 0.45 * 0.5 + 0.15 + 0.25 * 0.9 + 0.15 - 0.10 * 0.9
        self.assertAlmostEqual(score.total, expected)
        self.assertEqual(
            score.matched_evidence,
            ("material=cotton", "excluded:material=polyester", "exact=pocket"),
        )

    def test_unknown_product_attributes_do_not_create_negative_violations(self) -> None:
        intent = ConversationConstraintView(
            positive=(ConstraintValue("color", "blue", 1, 1.0, "test"),),
            negative=(ConstraintValue("material", "polyester", -1, 1.0, "test"),),
        )

        score = score_candidate(
            intent,
            ProductAttributeView(parent_asin="UNKNOWN"),
            normalized_rrf=0.4,
        )

        self.assertEqual(score.positive_slot_match, 0.0)
        self.assertEqual(score.negative_violation, 0.0)
        self.assertAlmostEqual(score.total, WEIGHTS["rrf_prior"] * 0.4)

    def test_exact_negative_term_is_a_full_violation(self) -> None:
        intent = ConversationConstraintView(excluded_exact_terms=("sequins",))
        product = ProductAttributeView(
            parent_asin="SEQUINS",
            feature_phrases=(attribute("sequins"),),
        )

        score = score_candidate(intent, product, normalized_rrf=0.0)

        self.assertEqual(score.negative_violation, 1.0)
        self.assertEqual(score.total, -0.1)
        self.assertIn("excluded:exact=sequins", score.matched_evidence)

    def test_inputs_are_clamped_and_breakdown_is_immutable(self) -> None:
        intent = ConversationConstraintView(category_terms=("dress",))
        product = ProductAttributeView(
            parent_asin="DRESS",
            category=(attribute("dress"),),
        )
        high = score_candidate(intent, product, normalized_rrf=9.0)
        low = score_candidate(intent, product, normalized_rrf=-4.0)

        self.assertEqual(high.rrf_prior, 1.0)
        self.assertEqual(low.rrf_prior, 0.0)
        self.assertFalse(hasattr(high, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            high.total = 0.0  # type: ignore[misc]
        self.assertEqual(high.as_dict()["total"], high.total)


class RerankTopNTests(unittest.TestCase):
    def test_attribute_match_can_promote_within_pool_and_tail_is_untouched(self) -> None:
        fused = ["RAW-FIRST", "MATCH", "TAIL"]
        intent = ConversationConstraintView(
            negative=(ConstraintValue("material", "polyester", -1, 1.0, "test"),),
            exact_terms=("pocket",),
        )
        products = {
            "RAW-FIRST": ProductAttributeView(
                parent_asin="RAW-FIRST",
                material=(attribute("polyester"),),
            ),
            "MATCH": ProductAttributeView(
                parent_asin="MATCH",
                material=(attribute("cotton"),),
                feature_phrases=(attribute("pocket"),),
            ),
        }

        reranked, breakdowns = rerank_top_n(
            fused,
            {"RAW-FIRST": 1.0, "MATCH": 0.9},
            products,
            intent,
            top_n=2,
        )

        self.assertEqual(reranked, ["MATCH", "RAW-FIRST", "TAIL"])
        self.assertEqual(fused, ["RAW-FIRST", "MATCH", "TAIL"])
        self.assertEqual(set(breakdowns), {"RAW-FIRST", "MATCH"})
        self.assertGreater(breakdowns["MATCH"].total, breakdowns["RAW-FIRST"].total)

    def test_incomparable_attribute_coverage_cannot_cross(self) -> None:
        fused = ["UNKNOWN", "MATCH"]
        intent = ConversationConstraintView(
            positive=(ConstraintValue("material", "cotton", 1, 1.0, "test"),),
        )
        products = {
            "UNKNOWN": ProductAttributeView(parent_asin="UNKNOWN"),
            "MATCH": ProductAttributeView(
                parent_asin="MATCH",
                material=(attribute("cotton"),),
            ),
        }

        reranked, breakdowns = rerank_top_n(
            fused,
            {"UNKNOWN": 1.0, "MATCH": 0.9},
            products,
            intent,
        )

        self.assertEqual(reranked, fused)
        self.assertEqual(breakdowns["UNKNOWN"].coverage_signature, ())
        self.assertEqual(breakdowns["MATCH"].coverage_signature, ("material",))

    def test_raw_top_ten_membership_and_tail_order_are_preserved(self) -> None:
        fused = [f"RAW-{index:02d}" for index in range(1, 12)]
        intent = ConversationConstraintView(exact_terms=("pocket",))
        products = {
            asin: ProductAttributeView(parent_asin=asin)
            for asin in fused
        }
        products["RAW-11"] = ProductAttributeView(
            parent_asin="RAW-11",
            feature_phrases=(attribute("pocket"),),
        )

        reranked, _ = rerank_top_n(
            fused,
            {asin: 1.0 - index / 100.0 for index, asin in enumerate(fused)},
            products,
            intent,
        )

        self.assertEqual(set(reranked[:PRESERVED_TOP_K]), set(fused[:PRESERVED_TOP_K]))
        self.assertEqual(reranked[PRESERVED_TOP_K:], fused[PRESERVED_TOP_K:])

    def test_ties_preserve_original_fused_order(self) -> None:
        fused = ["Z", "A"]
        intent = ConversationConstraintView(category_terms=("dress",))
        products = {
            asin: ProductAttributeView(
                parent_asin=asin,
                category=(attribute("dress"),),
            )
            for asin in fused
        }

        reranked, _ = rerank_top_n(
            fused,
            {"Z": 1.0, "A": 1.0},
            products,
            intent,
        )

        self.assertEqual(reranked, fused)

    def test_missing_view_is_unknown_but_deterministic(self) -> None:
        intent = ConversationConstraintView(exact_terms=("pocket",))
        arguments = (
            ["MISSING", "KNOWN"],
            {"MISSING": 1.0, "KNOWN": 0.9},
            {
                "KNOWN": ProductAttributeView(
                    parent_asin="KNOWN",
                    feature_phrases=(attribute("pocket"),),
                )
            },
            intent,
        )

        first = rerank_top_n(*arguments)
        second = rerank_top_n(*arguments)

        self.assertEqual(first, second)
        self.assertEqual(first[0][0], "KNOWN")
        self.assertIn("MISSING", first[1])

    def test_no_evidence_nonpositive_prior_and_empty_pool_are_safe_noops(self) -> None:
        fused = ["A", "B"]
        empty_intent = ConversationConstraintView()
        useful_intent = ConversationConstraintView(exact_terms=("pocket",))

        self.assertEqual(rerank_top_n(fused, {"A": 1.0}, {}, empty_intent), (fused, {}))
        self.assertEqual(
            rerank_top_n(fused, {"A": 0.0, "B": -1.0}, {}, useful_intent),
            (fused, {}),
        )
        self.assertEqual(rerank_top_n([], {}, {}, useful_intent), ([], {}))
        self.assertEqual(rerank_top_n(fused, {"A": 1.0}, {}, useful_intent, top_n=0), (fused, {}))

    def test_default_pool_limit_is_fixed_and_auditable(self) -> None:
        self.assertEqual(RERANK_TOP_N, 50)
        self.assertEqual(PRESERVED_TOP_K, 10)
        self.assertEqual(
            WEIGHTS,
            {
                "rrf_prior": 0.45,
                "category_consistency": 0.15,
                "positive_slot_match": 0.25,
                "exact_feature_match": 0.15,
                "negative_violation": -0.10,
            },
        )


if __name__ == "__main__":
    unittest.main()
