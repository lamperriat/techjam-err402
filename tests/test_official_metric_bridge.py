from __future__ import annotations

import math
import unittest

from scripts.official_metric_bridge import (
    METRIC_FIELDS,
    OfficialMetricBridgeError,
    rebuild_official_metrics,
    validate_official_metrics,
)


def _hit(rank: int, turn: int = 1) -> dict[str, object]:
    return {
        "hit": True,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 1.0 / rank,
    }


def _miss() -> dict[str, object]:
    return {
        "hit": False,
        "first_hit_turn": None,
        "best_rank": None,
        "reciprocal_rank": 0.0,
    }


class OfficialMetricBridgeTests(unittest.TestCase):
    def test_reproduces_p9_rounding_boundary(self) -> None:
        ranks = [1] * 8 + [5] + [7] * 6 + [8] + [9] * 5 + [10] * 16
        sessions = [
            _hit(rank, 1 if index == 0 else 2)
            for index, rank in enumerate(ranks)
        ] + [_miss() for _ in range(163)]

        metrics = rebuild_official_metrics(sessions)

        self.assertEqual(metrics["sample_count"], 200)
        self.assertEqual(metrics["hit_rate_at_10"], 0.185)
        self.assertEqual(metrics["mrr"], 0.056688)
        self.assertEqual(metrics["mttc"], 9.33)
        self.assertEqual(metrics["efficiency"], 0.167)
        self.assertEqual(metrics["recommended_technical_score"], 0.142906)
        old_shortcut = round(720_249 / (25_200 * 200), 6)
        self.assertEqual(old_shortcut, 0.142907)

    def test_score_uses_unrounded_efficiency(self) -> None:
        sessions = [_hit(7), _hit(10), _miss()]

        metrics = rebuild_official_metrics(sessions)

        self.assertEqual(metrics["efficiency"], 0.666667)
        self.assertEqual(metrics["recommended_technical_score"], 0.490952)
        rounded_efficiency_score = round(
            0.5 * metrics["hit_rate_at_10"]
            + 0.3 * metrics["mrr"]
            + 0.2 * metrics["efficiency"],
            6,
        )
        self.assertEqual(rounded_efficiency_score, 0.490953)

    def test_each_one_micro_unit_metric_difference_fails(self) -> None:
        sessions = [_hit(1), _miss()]
        expected = rebuild_official_metrics(sessions)
        for field in METRIC_FIELDS:
            with self.subTest(field=field):
                observed = dict(expected)
                observed[field] = float(observed[field]) + 0.000001
                validation = validate_official_metrics(sessions, observed)
                self.assertFalse(validation["passed"])
                self.assertEqual(
                    validation["failure_reasons"],
                    [f"official_rounding_mismatch:{field}"],
                )

    def test_missing_invalid_and_sample_count_fail_closed(self) -> None:
        sessions = [_hit(1), _miss()]
        expected = rebuild_official_metrics(sessions)

        missing = dict(expected)
        del missing["efficiency"]
        self.assertEqual(
            validate_official_metrics(sessions, missing)["failure_reasons"],
            ["missing_metric:efficiency"],
        )

        for invalid in (math.nan, math.inf, -math.inf, True, "0.5"):
            with self.subTest(invalid=invalid):
                observed = dict(expected)
                observed["mrr"] = invalid
                self.assertEqual(
                    validate_official_metrics(sessions, observed)["failure_reasons"],
                    ["invalid_metric:mrr"],
                )

        wrong_count = dict(expected)
        wrong_count["sample_count"] = 3
        self.assertEqual(
            validate_official_metrics(sessions, wrong_count)["failure_reasons"],
            ["official_rounding_mismatch:sample_count"],
        )

    def test_empty_or_inconsistent_ledger_is_invalid(self) -> None:
        with self.assertRaises(OfficialMetricBridgeError):
            rebuild_official_metrics([])

        invalid_ledgers = [
            [{"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.1}],
            [{"hit": True, "first_hit_turn": None, "reciprocal_rank": 1.0}],
            [{"hit": True, "first_hit_turn": 1, "reciprocal_rank": math.nan}],
            [
                {
                    "hit": True,
                    "first_hit_turn": 1,
                    "best_rank": 11,
                    "reciprocal_rank": 1.0 / 11,
                }
            ],
            [
                {
                    "hit": True,
                    "first_hit_turn": 1,
                    "best_rank": 2,
                    "reciprocal_rank": 0.4,
                }
            ],
            [
                {
                    "hit": False,
                    "first_hit_turn": None,
                    "best_rank": 1,
                    "reciprocal_rank": 0.0,
                }
            ],
        ]
        for sessions in invalid_ledgers:
            with self.subTest(sessions=sessions):
                validation = validate_official_metrics(sessions, {})
                self.assertFalse(validation["passed"])
                self.assertEqual(validation["failure_reasons"], ["invalid_session_ledger"])


if __name__ == "__main__":
    unittest.main()
