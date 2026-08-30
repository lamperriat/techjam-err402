from __future__ import annotations

import unittest

from starter.progressive_other import (
    MAX_FAMILY_COUNT,
    MAX_FAMILY_RECORDS,
    FamilyKey,
    ProgressiveOtherContractError,
    family_key,
    progressive_other_order,
    update_family_memory,
)


class ProgressiveOtherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.order = [f"p{index:03d}" for index in range(40)]
        self.served = self.order[:20]
        self.red = family_key("Color", " Red ")
        self.blue = family_key("color", "blue")

    def test_normalizes_and_bounds_memory(self) -> None:
        memory = update_family_memory({}, rejected=[("Color", " Red "), self.red])
        self.assertEqual(memory, {FamilyKey("color", "red"): 1})
        for _ in range(10):
            memory = update_family_memory(memory, rejected=[self.red])
        self.assertEqual(memory[self.red], MAX_FAMILY_COUNT)

        many = [("feature", f"value {index}") for index in range(30)]
        bounded = update_family_memory({}, rejected=many)
        self.assertEqual(len(bounded), MAX_FAMILY_RECORDS)

    def test_affirmation_clears_matching_family_only(self) -> None:
        memory = {self.red: 2, self.blue: 1}
        updated = update_family_memory(
            memory,
            rejected=[self.red],
            affirmed=[self.red],
        )
        self.assertEqual(updated, {self.blue: 1})
        self.assertEqual(memory, {self.red: 2, self.blue: 1})

    def test_reset_discards_old_intent_memory(self) -> None:
        updated = update_family_memory(
            {self.red: 3},
            rejected=[self.blue],
            reset=True,
        )
        self.assertEqual(updated, {self.blue: 1})

    def test_first_two_pages_are_frozen_identity(self) -> None:
        families = {"p000": [self.red], "p020": [self.red]}
        for intent_age in (1, 2):
            self.assertEqual(
                progressive_other_order(
                    self.order,
                    self.served,
                    intent_age=intent_age,
                    family_memory={self.red: 2},
                    candidate_families=families,
                ),
                self.order,
            )

    def test_third_page_stably_demotes_rejected_unseen_family(self) -> None:
        families = {
            "p020": [self.red],
            "p021": [self.blue],
            "p022": [self.red],
        }
        ranked = progressive_other_order(
            self.order,
            self.served,
            intent_age=3,
            family_memory={self.red: 1},
            candidate_families=families,
        )
        self.assertEqual(ranked[:3], ["p021", "p023", "p024"])
        self.assertLess(ranked.index("p039"), ranked.index("p020"))
        self.assertLess(ranked.index("p020"), ranked.index("p022"))
        self.assertEqual(ranked[-20:], self.served)
        self.assertEqual(set(ranked), set(self.order))

    def test_no_eligible_memory_is_v2_12_identity(self) -> None:
        expected = self.order[20:] + self.order[:20]
        ranked = progressive_other_order(
            self.order,
            self.served,
            intent_age=3,
            family_memory={},
            candidate_families={"p020": [self.red]},
        )
        self.assertEqual(ranked, expected)

    def test_unknown_candidate_attributes_are_neutral(self) -> None:
        ranked = progressive_other_order(
            self.order,
            self.served,
            intent_age=3,
            family_memory={self.red: 1},
            candidate_families={"p020": [self.red]},
        )
        self.assertEqual(ranked[0], "p021")
        self.assertEqual(ranked[18], "p039")
        self.assertEqual(ranked[19], "p020")
        self.assertEqual(ranked[20], "p000")

    def test_malformed_evidence_fails_closed_for_runtime_to_catch(self) -> None:
        bad_inputs = [
            {("color", "red"): 0},
            {("color", "red"): MAX_FAMILY_COUNT + 1},
            {("color", "red!"): 1},
        ]
        for memory in bad_inputs:
            with self.subTest(memory=memory):
                with self.assertRaises(ProgressiveOtherContractError):
                    progressive_other_order(
                        self.order,
                        self.served,
                        intent_age=3,
                        family_memory=memory,
                        candidate_families={},
                    )

        with self.assertRaises(ProgressiveOtherContractError):
            progressive_other_order(
                self.order + [self.order[-1]],
                self.served,
                intent_age=3,
                family_memory={self.red: 1},
                candidate_families={},
            )


if __name__ == "__main__":
    unittest.main()
