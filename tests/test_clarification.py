from __future__ import annotations

import unittest

from starter.attributes import AttributeValue, ProductAttributeView
from starter.clarification import rank_question_values


def attribute(value: str) -> AttributeValue:
    return AttributeValue(value, "test", 1.0, value)


class CandidateAwareClarificationTests(unittest.TestCase):
    def test_balanced_high_coverage_slot_ranks_first(self) -> None:
        products = {
            "A": ProductAttributeView(
                parent_asin="A",
                color=(attribute("blue"),),
                material=(attribute("cotton"),),
            ),
            "B": ProductAttributeView(
                parent_asin="B",
                color=(attribute("red"),),
            ),
            "C": ProductAttributeView(
                parent_asin="C",
                color=(attribute("blue"),),
                material=(attribute("linen"),),
            ),
            "D": ProductAttributeView(
                parent_asin="D",
                color=(attribute("red"),),
            ),
        }

        result = rank_question_values(
            products,
            products,
            blocked_attributes=(),
            turn=1,
        )

        self.assertEqual(result["selected_attribute"], "color")
        color, material = result["candidates"][:2]
        self.assertEqual(color["coverage"], 1.0)
        self.assertEqual(material["coverage"], 0.5)
        self.assertGreater(color["score"], material["score"])

    def test_known_exhausted_and_pending_attributes_are_blocked(self) -> None:
        products = {
            "A": ProductAttributeView(parent_asin="A", color=(attribute("blue"),)),
            "B": ProductAttributeView(parent_asin="B", color=(attribute("red"),)),
        }

        result = rank_question_values(
            products,
            products,
            blocked_attributes={"color", "budget"},
            turn=2,
        )

        self.assertIsNone(result["selected_attribute"])
        self.assertEqual(result["blocked_attributes"], ["color", "price"])

    def test_long_tail_brand_is_penalized_and_output_is_deterministic(self) -> None:
        products = {
            str(index): ProductAttributeView(
                parent_asin=str(index),
                brand=(attribute(f"brand {index}"),),
                color=(attribute("blue" if index < 5 else "red"),),
            )
            for index in range(10)
        }
        arguments = (products, products)

        first = rank_question_values(*arguments, blocked_attributes=(), turn=3)
        second = rank_question_values(*arguments, blocked_attributes=(), turn=3)

        self.assertEqual(first, second)
        self.assertEqual(first["selected_attribute"], "color")
        brand = next(item for item in first["candidates"] if item["attribute"] == "brand")
        self.assertLess(brand["answerability"], 0.55)

    def test_empty_candidates_have_no_question(self) -> None:
        result = rank_question_values({}, [], blocked_attributes=(), turn=10)

        self.assertIsNone(result["selected_attribute"])
        self.assertEqual(result["candidate_count"], 0)

    def test_final_turn_reports_evidence_but_never_selects_a_question(self) -> None:
        products = {
            "A": ProductAttributeView(parent_asin="A", color=(attribute("blue"),)),
            "B": ProductAttributeView(parent_asin="B", color=(attribute("red"),)),
        }

        result = rank_question_values(
            products,
            products,
            blocked_attributes=(),
            turn=10,
        )

        self.assertTrue(result["candidates"])
        self.assertIsNone(result["selected_attribute"])
        self.assertEqual(result["selection_reason"], "turn_limit")

    def test_multi_value_combinations_do_not_create_artificial_entropy(self) -> None:
        products = {
            "A": ProductAttributeView(
                parent_asin="A",
                material=(attribute("cotton"), attribute("polyester")),
            ),
            "B": ProductAttributeView(
                parent_asin="B",
                material=(attribute("cotton"), attribute("wool")),
            ),
        }

        result = rank_question_values(
            products,
            products,
            blocked_attributes=set(),
            turn=1,
        )

        self.assertNotIn(
            "material",
            {candidate["attribute"] for candidate in result["candidates"]},
        )


if __name__ == "__main__":
    unittest.main()
