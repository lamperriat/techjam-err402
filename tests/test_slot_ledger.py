from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from starter.attributes import ConstraintValue, ConversationConstraintView
from starter.slot_ledger import DELETED, SUPERSEDED, SlotLedger


def constraint(slot: str, value: str, polarity: int = 1) -> ConstraintValue:
    return ConstraintValue(slot, value, polarity, 1.0, "test")


class SlotLedgerTests(unittest.TestCase):
    def test_selective_override_keeps_an_independent_slot(self) -> None:
        ledger = SlotLedger()
        ledger.reconcile(
            ConversationConstraintView(
                positive=(constraint("material", "cotton"), constraint("color", "blue")),
            ),
            turn=1,
            version=1,
            message="blue cotton",
        )
        ledger.reconcile(
            ConversationConstraintView(
                positive=(constraint("material", "linen"), constraint("color", "blue")),
            ),
            turn=2,
            version=2,
            message="replace cotton with linen",
            retired_status=SUPERSEDED,
        )

        active = {(record.slot, record.value) for record in ledger.active_records()}
        cotton = next(record for record in ledger.records if record.value == "cotton")
        blue = next(record for record in ledger.records if record.value == "blue")
        self.assertEqual(active, {("material", "linen"), ("color", "blue")})
        self.assertEqual(cotton.status, SUPERSEDED)
        self.assertEqual(cotton.ended_turn, 2)
        self.assertEqual(blue.source_turn, 1)

    def test_no_preference_deletes_only_the_named_slot(self) -> None:
        ledger = SlotLedger()
        intent = ConversationConstraintView(
            positive=(constraint("material", "cotton"), constraint("color", "blue")),
        )
        ledger.reconcile(intent, turn=1, version=1, message="blue cotton")
        ledger.reconcile(
            intent,
            turn=2,
            version=1,
            message="no preference for material",
            suppressed_slots={"material"},
        )

        active = {(record.slot, record.value) for record in ledger.active_records()}
        cotton = next(record for record in ledger.records if record.value == "cotton")
        self.assertEqual(active, {("color", "blue")})
        self.assertEqual(cotton.status, DELETED)

    def test_negative_evidence_supersedes_the_same_positive_value(self) -> None:
        ledger = SlotLedger()
        ledger.reconcile(
            ConversationConstraintView(positive=(constraint("color", "red"),)),
            turn=1,
            version=1,
            message="red",
        )
        ledger.reconcile(
            ConversationConstraintView(
                positive=(constraint("color", "red"),),
                negative=(constraint("color", "red", -1),),
            ),
            turn=2,
            version=1,
            message="not red",
        )

        positive, negative = ledger.records
        self.assertEqual(positive.status, SUPERSEDED)
        self.assertEqual(negative.status, "active")
        self.assertEqual(negative.polarity, -1)
        self.assertEqual(negative.hardness, "hard")

    def test_records_are_immutable_versioned_and_serializable(self) -> None:
        ledger = SlotLedger()
        ledger.reconcile(
            ConversationConstraintView(category_terms=("dress",)),
            turn=3,
            version=2,
            message="I need a dress",
        )
        record = ledger.active_records()[0]

        self.assertEqual((record.slot, record.value), ("category", "dress"))
        self.assertEqual((record.source_turn, record.version), (3, 2))
        self.assertEqual(record.hardness, "hard")
        self.assertEqual(ledger.as_dict()["active_count"], 1)
        with self.assertRaises(FrozenInstanceError):
            record.status = DELETED  # type: ignore[misc]

    def test_explicit_hard_restatement_versions_only_the_mentioned_value(self) -> None:
        ledger = SlotLedger()
        intent = ConversationConstraintView(
            positive=(constraint("material", "cotton"), constraint("color", "blue")),
        )
        ledger.reconcile(intent, turn=1, version=1, message="blue cotton")
        ledger.reconcile(intent, turn=2, version=2, message="I must have cotton")

        cotton_records = [record for record in ledger.records if record.value == "cotton"]
        blue = next(record for record in ledger.records if record.value == "blue")
        self.assertEqual(len(cotton_records), 2)
        self.assertEqual(cotton_records[0].status, SUPERSEDED)
        self.assertEqual(cotton_records[0].ended_turn, 2)
        self.assertEqual(cotton_records[1].hardness, "hard")
        self.assertEqual((cotton_records[1].source_turn, cotton_records[1].version), (2, 2))
        self.assertEqual(blue.hardness, "soft")
        self.assertEqual(blue.source_turn, 1)

    def test_hard_marker_does_not_leak_across_contrast_clause(self) -> None:
        ledger = SlotLedger()
        intent = ConversationConstraintView(
            positive=(constraint("material", "cotton"), constraint("feature", "pockets")),
        )
        ledger.reconcile(
            intent,
            turn=1,
            version=1,
            message="cotton is fine, but pockets are a must",
        )

        hardness = {record.value: record.hardness for record in ledger.active_records()}
        self.assertEqual(hardness, {"cotton": "soft", "pockets": "hard"})


if __name__ == "__main__":
    unittest.main()
