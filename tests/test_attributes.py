from __future__ import annotations

import re
import unittest
from dataclasses import FrozenInstanceError

from starter.attributes import (
    AttributeValue,
    ProductAttributeView,
    attribute_registry_sha256,
    build_conversation_constraint_view,
    build_product_attribute_view,
    normalize_value,
    product_slot,
    product_view_as_dict,
)


def values(items: tuple[AttributeValue, ...]) -> set[str]:
    return {item.value for item in items}


class ProductAttributeViewTests(unittest.TestCase):
    def test_normalization_canonicalizes_unicode_case_hyphens_and_aliases(self) -> None:
        self.assertEqual(
            normalize_value("ＧＲＥＹ MEN’S PULL‐ON"),
            "gray men pull on",
        )

    def test_phrase_matching_keeps_token_boundaries_and_ignores_generic_root(self) -> None:
        view = build_product_attribute_view({
            "parent_asin": "BOUNDARY",
            "title": "Cottonwood Pocketsize Womenswear Pullonboard",
            "categories": ["Clothing, Shoes & Jewelry"],
        })

        self.assertEqual(view.category, ())
        self.assertEqual(view.audience, ())
        self.assertEqual(view.material, ())
        self.assertEqual(view.closure, ())
        self.assertEqual(view.feature_phrases, ())

    def test_product_view_combines_evidence_and_drops_catalog_noise(self) -> None:
        product = {
            "parent_asin": "A-1",
            "title": "Acme Men's Grey Cotton Pull-On Running Shoes with Pockets",
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Men",
                "Shoes",
                "Loafers & Slip-Ons",
            ],
            "features": [
                "100% Cotton",
                "90% Polyester, 10% Spandex",
                "Machine Wash",
                "All Other Heathers",
            ],
            "description": ["Purple silk description-only text"],
            "details": {
                "Department": "mens",
                "Color": "Grey",
                "Closure Type": "Pull-on",
                "Shoe Width": "Wide",
                "Material": "Cotton",
                "Manufacturer": "Acme Manufacturing",
            },
            "store": "Acme",
            "price": "$29.99",
        }

        view = build_product_attribute_view(product)

        self.assertEqual(view.parent_asin, "A-1")
        self.assertIn("shoe", values(view.category))
        self.assertNotIn("clothing shoes and jewelry", values(view.category))
        self.assertEqual(values(view.audience), {"men"})
        self.assertEqual(values(view.material), {"cotton", "polyester", "spandex"})
        self.assertEqual(values(view.color), {"gray"})
        self.assertTrue({"pull on", "slip on"} <= values(view.closure))
        self.assertEqual(values(view.use_case), {"running"})
        self.assertEqual(values(view.width), {"wide"})
        self.assertIn("acme", values(view.brand))
        self.assertEqual(
            next(item for item in view.brand if item.value == "acme").source,
            "store",
        )
        self.assertTrue({"machine wash", "pocket"} <= values(view.feature_phrases))
        self.assertNotIn("pockets", values(view.feature_phrases))
        self.assertEqual(view.price, 29.99)

        all_values = {
            item.value
            for slot in (
                view.category,
                view.audience,
                view.material,
                view.color,
                view.closure,
                view.style,
                view.use_case,
                view.size,
                view.width,
                view.brand,
                view.feature_phrases,
            )
            for item in slot
        }
        self.assertTrue({"100", "90", "10", "all", "other", "heathers"}.isdisjoint(all_values))
        self.assertNotIn("purple", values(view.color))
        self.assertNotIn("silk", values(view.material))

    def test_strongest_source_wins_and_keeps_raw_provenance(self) -> None:
        view = build_product_attribute_view({
            "parent_asin": "A-2",
            "title": "Gray cotton dress",
            "features": ["Cotton fabric"],
            "details": {"Material Type": "Cotton", "Color": "Grey"},
            "categories": ["Women", "Dresses"],
        })

        cotton = next(item for item in view.material if item.value == "cotton")
        gray = next(item for item in view.color if item.value == "gray")
        self.assertEqual(cotton.source, "details.material type")
        self.assertEqual(cotton.confidence, 0.98)
        self.assertEqual(cotton.raw, "Material Type: Cotton")
        self.assertEqual(gray.source, "details.color")

    def test_flattened_fts_fields_are_supported(self) -> None:
        view = build_product_attribute_view({
            "parent_asin": "A-3",
            "title": ["Women's", "Blue Dress"],
            "categories": "Clothing Shoes and Jewelry Women Clothing Dresses",
            "features": "Machine-wash pockets",
            "details": "Department women Material cotton Closure pull-on",
            "store": {"name": "Example Store"},
        })

        self.assertIn("dress", values(view.category))
        self.assertIn("women", values(view.audience))
        self.assertIn("cotton", values(view.material))
        self.assertIn("blue", values(view.color))
        self.assertIn("pull on", values(view.closure))
        self.assertEqual(values(view.brand), {"name example store"})
        self.assertTrue({"machine wash", "pocket"} <= values(view.feature_phrases))

    def test_missing_values_remain_unknown_and_description_is_not_evidence(self) -> None:
        view = build_product_attribute_view({
            "parent_asin": "UNKNOWN",
            "description": ["red leather running jacket"],
        })

        self.assertEqual(view, ProductAttributeView(parent_asin="UNKNOWN"))
        self.assertIsNone(view.price)
        serialized = product_view_as_dict(view)
        self.assertFalse(
            any(
                item["value"] == "unknown"
                for slot, slot_values in serialized.items()
                if slot not in {"parent_asin", "price"}
                for item in slot_values
            )
        )

    def test_views_are_compact_immutable_and_product_slot_is_stable(self) -> None:
        view = build_product_attribute_view({
            "parent_asin": "A-4",
            "features": ["Pockets"],
        })

        with self.assertRaises(FrozenInstanceError):
            view.parent_asin = "changed"  # type: ignore[misc]
        self.assertFalse(hasattr(view, "__dict__"))
        self.assertEqual(product_slot(view, "feature"), view.feature_phrases)
        self.assertEqual(product_slot(view, "missing"), ())

    def test_registry_hash_and_extraction_are_deterministic(self) -> None:
        first = build_product_attribute_view({
            "parent_asin": "A-5",
            "details": {"Color": "Grey", "Material": "Cotton"},
        })
        second = build_product_attribute_view({
            "details": {"Material": "Cotton", "Color": "Grey"},
            "parent_asin": "A-5",
        })
        digest = attribute_registry_sha256()

        self.assertEqual(first, second)
        self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(digest, attribute_registry_sha256())


class ConversationConstraintViewTests(unittest.TestCase):
    def test_visible_state_is_normalized_without_profile_or_numeric_noise(self) -> None:
        view = build_conversation_constraint_view(
            "Women's Casual Dresses",
            ["cotton", "grey", "pull", "on", "pockets", "100", "90", "10", "all", "soft"],
            ["polyester", "other", "heathers"],
            {"use_case": ["Wedding"], "brand": "Acme"},
        )

        positive = {(item.slot, item.value) for item in view.positive}
        negative = {(item.slot, item.value) for item in view.negative}
        self.assertEqual(view.category_terms, ("dress",))
        self.assertTrue({
            ("audience", "women"),
            ("style", "casual"),
            ("material", "cotton"),
            ("color", "gray"),
            ("closure", "pull on"),
            ("feature", "pocket"),
            ("use_case", "wedding"),
            ("brand", "acme"),
        } <= positive)
        self.assertIn(("material", "polyester"), negative)
        self.assertEqual(view.exact_terms, ("soft",))
        self.assertEqual(view.excluded_exact_terms, ())
        self.assertFalse(hasattr(view, "profile"))

    def test_existing_slot_classifications_can_be_declared_without_values(self) -> None:
        view = build_conversation_constraint_view(
            "",
            [],
            [],
            {"material", "budget", "feature_phrases"},
        )

        self.assertEqual(view.positive, ())
        self.assertEqual(view.classified_slots, ("feature", "material", "price"))

    def test_unrecognized_visible_terms_remain_exact_and_missing_is_empty(self) -> None:
        view = build_conversation_constraint_view(
            "evening gowns",
            ["embroidered"],
            ["sequins"],
        )
        empty = build_conversation_constraint_view("", [], [])

        self.assertEqual(view.category_terms, ("evening", "gowns"))
        self.assertEqual(view.exact_terms, ("embroidered",))
        self.assertEqual(view.excluded_exact_terms, ("sequins",))
        self.assertEqual(empty.category_terms, ())
        self.assertEqual(empty.positive, ())
        self.assertEqual(empty.negative, ())


if __name__ == "__main__":
    unittest.main()
