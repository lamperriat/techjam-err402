from __future__ import annotations

import unittest

from starter.coverage import order_by_query_coverage


class CoverageCascadeTest(unittest.TestCase):
    def test_orders_by_coverage_and_preserves_fused_ties(self) -> None:
        ordered, diagnostics = order_by_query_coverage(
            ["blue", "cotton", "dress"],
            ["ONE", "TWO", "THREE"],
            {
                "ONE": ("blue dress",),
                "TWO": ("blue cotton dress",),
                "THREE": ("cotton dress",),
            },
            lambda value: value.split(),
        )

        self.assertEqual(ordered, ["TWO", "ONE", "THREE"])
        self.assertTrue(diagnostics["changed_top_10"])
        self.assertEqual(diagnostics["maximum_coverage"], 3)
        self.assertEqual(diagnostics["coverage_histogram"], {"2": 2, "3": 1})

    def test_missing_fields_have_zero_coverage_without_disappearing(self) -> None:
        ordered, diagnostics = order_by_query_coverage(
            ["blue"],
            ["UNKNOWN", "MATCH"],
            {"MATCH": ("blue",)},
            lambda value: value.split(),
        )

        self.assertEqual(ordered, ["MATCH", "UNKNOWN"])
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["covered_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
