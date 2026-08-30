from __future__ import annotations

import unittest

from scripts.evaluate_versioned_unseen_pagination import (
    PaginationReplayError,
    fixed_two_page_grace,
    reconstruct_current_order,
    seen_hole_replacement,
    stable_unseen_first,
)


class VersionedUnseenPaginationTests(unittest.TestCase):
    def test_unseen_items_are_stable_and_seen_items_are_fallback(self) -> None:
        order = tuple(f"item-{index}" for index in range(15))
        page = stable_unseen_first(order, {"item-0", "item-2", "item-4"})
        self.assertEqual(
            page,
            (
                "item-1",
                "item-3",
                "item-5",
                "item-6",
                "item-7",
                "item-8",
                "item-9",
                "item-10",
                "item-11",
                "item-12",
            ),
        )

    def test_seen_fallback_keeps_page_full(self) -> None:
        order = tuple(f"item-{index}" for index in range(12))
        page = stable_unseen_first(order, set(order[:10]))
        self.assertEqual(page, ("item-10", "item-11", *order[:8]))

    def test_slot10_swap_preserves_full_membership(self) -> None:
        c100 = tuple(f"item-{index}" for index in range(100))
        p11 = tuple(reversed(c100[:10]))
        turn = {"c100": c100, "actions": {"KEEP_P11": p11}}
        order = reconstruct_current_order(turn, 20, True)
        self.assertEqual(order[:9], p11[:9])
        self.assertEqual(order[9], "item-20")
        self.assertEqual(order[20], p11[9])
        self.assertEqual(set(order), set(c100))

    def test_invalid_duplicate_order_fails_closed(self) -> None:
        with self.assertRaises(PaginationReplayError):
            stable_unseen_first(("same",) * 10, set())

    def test_seen_holes_receive_tail_without_moving_unseen_top10(self) -> None:
        order = tuple(f"item-{index}" for index in range(15))
        served = {"item-0", "item-2", "item-4"}
        page = seen_hole_replacement(order, served)
        self.assertEqual(
            page,
            (
                "item-10",
                "item-1",
                "item-11",
                "item-3",
                "item-12",
                "item-5",
                "item-6",
                "item-7",
                "item-8",
                "item-9",
            ),
        )
        self.assertEqual(set(page), set(stable_unseen_first(order, served)))

    def test_fixed_grace_exploits_two_pages_then_uses_unseen_order(self) -> None:
        order = tuple(f"item-{index}" for index in range(15))
        served = {"item-0", "item-2", "item-4"}
        self.assertEqual(
            fixed_two_page_grace(order, served, intent_age=1),
            order[:10],
        )
        self.assertEqual(
            fixed_two_page_grace(order, served, intent_age=2),
            order[:10],
        )
        self.assertEqual(
            fixed_two_page_grace(order, served, intent_age=3),
            stable_unseen_first(order, served),
        )

    def test_fixed_grace_rejects_invalid_intent_age(self) -> None:
        order = tuple(f"item-{index}" for index in range(10))
        with self.assertRaises(PaginationReplayError):
            fixed_two_page_grace(order, set(), intent_age=0)


if __name__ == "__main__":
    unittest.main()
