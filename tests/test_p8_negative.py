from __future__ import annotations

import unittest
from types import SimpleNamespace

from starter.attributes import AttributeValue, ProductAttributeView
from starter.p8_negative import (
    CANDIDATE_POOL,
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    MIN_EVIDENCE_CONFIDENCE,
    UNKNOWN,
    ExecutableNegative,
    classify_candidate,
    compile_negative_constraints,
    stable_negative_partition,
)


def record(**changes: object) -> SimpleNamespace:
    values = {
        "record_id": 1,
        "slot": "color",
        "value": "red",
        "polarity": -1,
        "hardness": "hard",
        "source": "excluded_terms",
        "confidence": 1.0,
        "source_turn": 1,
        "version": 2,
        "status": "active",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def constraint(slot: str, value: str, record_id: int = 1) -> ExecutableNegative:
    return ExecutableNegative(slot, value, record_id, 1, 2)


def attribute(value: str, source: str = "details.color", confidence: float = 0.98) -> AttributeValue:
    return AttributeValue(value, source, confidence, value)


class P8NegativePrimitiveTests(unittest.TestCase):
    def test_compiler_accepts_only_the_frozen_current_version_contract(self) -> None:
        records = [
            record(),
            record(record_id=2, source="classification"),
            record(record_id=3, confidence=0.999),
            record(record_id=4, version=1),
            record(record_id=5, status="superseded"),
            record(record_id=6, polarity=1),
            record(record_id=7, hardness="soft"),
            record(record_id=8, slot="price"),
            record(record_id=9, value="water resistant", slot="feature"),
            record(record_id=10, value="réd"),
            record(record_id=11, slot="size", value="large"),
            record(record_id=12, slot="width", value="wide"),
        ]

        compiled = compile_negative_constraints(records, current_version=2)

        self.assertEqual(
            compiled.constraints,
            (ExecutableNegative("color", "red", 1, 1, 2),),
        )
        self.assertEqual(dict(compiled.rejection_counts), {
            "not_active": 1,
            "not_full_confidence": 1,
            "not_hard": 1,
            "not_negative": 1,
            "slot_not_allowed": 4,
            "stale_goal_version": 1,
            "untrusted_source": 1,
            "value_not_single_token": 1,
        })

    def test_single_negative_has_three_states_and_ignores_description(self) -> None:
        negative = (constraint("color", "red"),)
        red = ProductAttributeView("red", color=(attribute("red"),))
        blue = ProductAttributeView("blue", color=(attribute("blue"),))
        absent = ProductAttributeView("absent")
        description_only = ProductAttributeView(
            "description", color=(attribute("red", "description", 1.0),)
        )
        low_confidence_title = ProductAttributeView(
            "title", color=(attribute("red", "title", 0.82),)
        )

        self.assertEqual(classify_candidate(red, negative).state, EXPLICIT_VIOLATION)
        self.assertEqual(classify_candidate(blue, negative).state, COMPATIBLE)
        self.assertEqual(classify_candidate(absent, negative).state, UNKNOWN)
        self.assertEqual(classify_candidate(description_only, negative).state, UNKNOWN)
        self.assertEqual(classify_candidate(low_confidence_title, negative).state, UNKNOWN)
        self.assertEqual(MIN_EVIDENCE_CONFIDENCE, 0.90)

    def test_double_negative_uses_violation_first_then_unknown_then_compatible(self) -> None:
        negatives = (
            constraint("color", "red", 1),
            constraint("material", "polyester", 2),
        )
        compatible = ProductAttributeView(
            "compatible",
            color=(attribute("blue"),),
            material=(attribute("cotton", "details.material"),),
        )
        unknown = ProductAttributeView(
            "unknown",
            color=(attribute("blue"),),
        )
        violation = ProductAttributeView(
            "violation",
            color=(attribute("red"),),
        )

        self.assertEqual(classify_candidate(compatible, negatives).state, COMPATIBLE)
        self.assertEqual(classify_candidate(unknown, negatives).state, UNKNOWN)
        result = classify_candidate(violation, negatives)
        self.assertEqual(result.state, EXPLICIT_VIOLATION)
        self.assertEqual(result.violations, ("color=red",))
        self.assertEqual(result.unknown, ("material=polyester",))

    def test_partition_is_stable_and_unknown_is_not_a_conflict(self) -> None:
        identifiers = ["V1", "U1", "C1", "V2", "C2", "U2"]
        views = {
            "V1": ProductAttributeView("V1", color=(attribute("red"),)),
            "V2": ProductAttributeView("V2", color=(attribute("red"),)),
            "C1": ProductAttributeView("C1", color=(attribute("blue"),)),
            "C2": ProductAttributeView("C2", color=(attribute("green"),)),
        }

        partition = stable_negative_partition(
            identifiers, views, (constraint("color", "red"),), top_k=5
        )

        self.assertEqual(
            partition.identifiers,
            ("C1", "C2", "U1", "U2", "V1", "V2"),
        )
        self.assertEqual(dict(partition.counts), {
            COMPATIBLE: 2,
            UNKNOWN: 2,
            EXPLICIT_VIOLATION: 2,
        })
        self.assertEqual(partition.violation_fallback_count, 1)

    def test_only_first_fifty_are_partitioned_and_base_tail_is_untouched(self) -> None:
        identifiers = [f"V{index:02d}" for index in range(52)]
        views = {
            identifier: ProductAttributeView(identifier, color=(attribute("red"),))
            for identifier in identifiers
        }
        views["V49"] = ProductAttributeView("V49", color=(attribute("blue"),))

        partition = stable_negative_partition(
            identifiers, views, (constraint("color", "red"),)
        )

        self.assertEqual(CANDIDATE_POOL, 50)
        self.assertEqual(partition.identifiers[0], "V49")
        self.assertEqual(partition.identifiers[-2:], ("V50", "V51"))
        self.assertEqual(sum(dict(partition.counts).values()), 50)
        self.assertEqual(partition.as_dict()["base_tail_count"], 2)

    def test_no_constraint_is_exact_and_input_validation_is_strict(self) -> None:
        identifiers = ["B", "A", "B"]
        partition = stable_negative_partition(identifiers, {}, ())
        self.assertEqual(partition.identifiers, ("B", "A"))
        with self.assertRaisesRegex(ValueError, "top_k"):
            stable_negative_partition(identifiers, {}, (), top_k=0)
        with self.assertRaisesRegex(ValueError, "candidate_pool"):
            stable_negative_partition(identifiers, {}, (), top_k=10, candidate_pool=9)


if __name__ == "__main__":
    unittest.main()
