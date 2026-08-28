from __future__ import annotations

import unittest

from starter.attributes import AttributeValue, ProductAttributeView, build_product_attribute_view
from starter.agent import _text
from starter.p8_negative import (
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    UNKNOWN,
    ExecutableNegative,
    classify_candidate,
    stable_negative_partition,
)
from starter.p9_evidence import (
    SLOT_ORDER,
    classify_masks,
    compile_mask_constraints,
    masks_from_view,
    runtime_attribute_view,
    stable_compact_partition,
)


def constraint(slot: str, value: str, record_id: int = 1) -> ExecutableNegative:
    return ExecutableNegative(slot, value, record_id, 1, 1)


def attribute(
    value: str,
    source: str = "details.value",
    confidence: float = 0.98,
) -> AttributeValue:
    return AttributeValue(value, source, confidence, value)


class P9EvidenceTests(unittest.TestCase):
    def test_all_six_slots_have_exact_known_and_violation_states(self) -> None:
        values = {
            "audience": ("men", "women"),
            "material": ("cotton", "polyester"),
            "color": ("red", "blue"),
            "closure": ("zipper", "button"),
            "style": ("casual", "formal"),
            "use_case": ("beach", "work"),
        }
        for slot, (excluded, alternative) in values.items():
            negative = (constraint(slot, excluded),)
            compiled = compile_mask_constraints(negative)
            violating = ProductAttributeView(
                "violating", **{slot: (attribute(excluded),)}
            )
            compatible = ProductAttributeView(
                "compatible", **{slot: (attribute(alternative),)}
            )
            with self.subTest(slot=slot):
                for view in (violating, compatible):
                    self.assertEqual(
                        classify_masks(masks_from_view(view), compiled),
                        classify_candidate(view, negative).state,
                    )

    def test_single_and_double_negative_states_are_differentially_exact(self) -> None:
        views = (
            ProductAttributeView(
                "compatible",
                color=(attribute("blue"),),
                material=(attribute("cotton"),),
            ),
            ProductAttributeView("unknown", color=(attribute("blue"),)),
            ProductAttributeView("violation", color=(attribute("red"),)),
            ProductAttributeView("empty"),
        )
        constraint_sets = (
            (constraint("color", "red"),),
            (
                constraint("color", "red"),
                constraint("material", "polyester", 2),
            ),
        )
        for constraints in constraint_sets:
            compiled = compile_mask_constraints(constraints)
            for view in views:
                with self.subTest(constraints=constraints, identifier=view.parent_asin):
                    self.assertEqual(
                        classify_masks(masks_from_view(view), compiled),
                        classify_candidate(view, constraints).state,
                    )

    def test_source_and_confidence_boundaries_match_p8(self) -> None:
        cases = (
            ProductAttributeView("description", color=(attribute("red", "description", 1.0),)),
            ProductAttributeView("low_title", color=(attribute("red", "title", 0.82),)),
            ProductAttributeView("feature_boundary", color=(attribute("red", "features", 0.90),)),
            ProductAttributeView("detail", color=(attribute("red", "details.Color", 0.98),)),
            ProductAttributeView("trusted_title", color=(attribute("red", "title", 1.0),)),
        )
        negative = (constraint("color", "red"),)
        compiled = compile_mask_constraints(negative)
        for view in cases:
            with self.subTest(identifier=view.parent_asin):
                self.assertEqual(
                    classify_masks(masks_from_view(view), compiled),
                    classify_candidate(view, negative).state,
                )
        self.assertEqual(classify_masks(masks_from_view(cases[0]), compiled), UNKNOWN)
        self.assertEqual(classify_masks(masks_from_view(cases[1]), compiled), UNKNOWN)
        self.assertEqual(
            classify_masks(masks_from_view(cases[2]), compiled), EXPLICIT_VIOLATION
        )

    def test_unknown_value_has_the_same_known_slot_semantics(self) -> None:
        negative = (constraint("color", "chartreuse"),)
        compiled = compile_mask_constraints(negative)
        known = ProductAttributeView("known", color=(attribute("blue"),))
        absent = ProductAttributeView("absent")
        self.assertEqual(classify_masks(masks_from_view(known), compiled), COMPATIBLE)
        self.assertEqual(classify_masks(masks_from_view(absent), compiled), UNKNOWN)
        self.assertEqual(
            classify_masks(masks_from_view(known), compiled),
            classify_candidate(known, negative).state,
        )

    def test_stable_partition_is_differentially_exact_including_tail(self) -> None:
        identifiers = [f"P-{index:02d}" for index in range(52)]
        views = {
            identifier: ProductAttributeView(
                identifier,
                color=(attribute("red" if index % 3 == 0 else "blue"),),
            )
            for index, identifier in enumerate(identifiers)
            if index % 3 != 2
        }
        constraints = (constraint("color", "red"),)
        p8 = stable_negative_partition(identifiers, views, constraints)
        p9 = stable_compact_partition(
            identifiers,
            {identifier: masks_from_view(view) for identifier, view in views.items()},
            constraints,
        )
        self.assertEqual(p9.identifiers, p8.identifiers)
        self.assertEqual(dict(p9.counts), dict(p8.counts))
        self.assertEqual(p9.violation_fallback_count, p8.violation_fallback_count)
        self.assertEqual(p9.identifiers[-2:], ("P-50", "P-51"))

    def test_catalog_projection_matches_agent_fts_reconstruction(self) -> None:
        product = {
            "parent_asin": "A",
            "title": "Red linen dress",
            "categories": ["Women", "Dresses"],
            "features": ["Blue cotton casual"],
            "details": {"Unexpected": "polyester green", "Color": "red"},
            "store": "Example",
            "description": "black leather description only",
        }
        expected = build_product_attribute_view({
            "parent_asin": "A",
            "title": _text(product["title"]),
            "categories": _text(product["categories"]),
            "features": _text(product["features"]),
            "details": _text(product["details"]),
            "store": _text(product["store"]),
        })
        observed = runtime_attribute_view(product)
        self.assertEqual(observed, expected)
        self.assertEqual(len(masks_from_view(observed)), len(SLOT_ORDER))

    def test_input_validation_is_bounded_and_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "six masks"):
            classify_masks((0,), ())
        with self.assertRaisesRegex(ValueError, "top_k"):
            stable_compact_partition(["A"], {}, (), top_k=0)
        with self.assertRaisesRegex(ValueError, "candidate_pool"):
            stable_compact_partition(["A"], {}, (), top_k=10, candidate_pool=9)


if __name__ == "__main__":
    unittest.main()
