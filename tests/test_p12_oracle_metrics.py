from __future__ import annotations

import hashlib
import unittest

from scripts.official_metric_bridge import rebuild_official_metrics
from scripts.p12_oracle_metrics import (
    P12OracleMetricsError,
    aggregate_action_oracle,
)


ACTIONS = ("KEEP_P11", "RERANK", "SEMANTIC", "ASK")
ELIGIBLE = ("KEEP_P11", "RERANK", "SEMANTIC")


def _miss() -> dict[str, int | None]:
    return {"first_hit_turn": None, "first_rank": None}


def _hit(turn: int, rank: int) -> dict[str, int]:
    return {"first_hit_turn": turn, "first_rank": rank}


def _record(
    ordinal: int,
    target: str,
    taxonomy: str,
    weight: float,
    *,
    keep: dict,
    rerank: dict,
    semantic: dict,
    ask: dict | None = None,
) -> dict:
    return {
        "ordinal": ordinal,
        "target_id": target,
        "scenario": "synthetic",
        "taxonomy": taxonomy,
        "difficulty": "synthetic",
        "popularity": "synthetic",
        "source_weight": weight,
        "actions": {
            "KEEP_P11": keep,
            "RERANK": rerank,
            "SEMANTIC": semantic,
            "ASK": ask if ask is not None else keep,
        },
    }


class P12OracleMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record(
                0,
                "target-a",
                "tax-a",
                9.0,
                keep=_hit(2, 2),
                rerank=_hit(1, 1),
                semantic=_miss(),
            ),
            _record(
                1,
                "target-a",
                "tax-a",
                1.0,
                keep=_miss(),
                rerank=_hit(4, 4),
                semantic=_hit(2, 2),
            ),
            _record(
                2,
                "target-b",
                "tax-b",
                1.0,
                keep=_hit(10, 10),
                rerank=_miss(),
                semantic=_hit(3, 1),
                ask=_hit(1, 1),
            ),
        ]

    def _aggregate(self, **overrides: object) -> dict:
        arguments = {
            "action_ids": ACTIONS,
            "oracle_eligible_actions": ELIGIBLE,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 77,
        }
        arguments.update(overrides)
        return aggregate_action_oracle(self.records, **arguments)

    def test_row_uniform_exactly_uses_official_rounding(self) -> None:
        result = self._aggregate()
        expected = rebuild_official_metrics(
            [
                {
                    "hit": True,
                    "first_hit_turn": 2,
                    "best_rank": 2,
                    "reciprocal_rank": 0.5,
                },
                {
                    "hit": False,
                    "first_hit_turn": None,
                    "best_rank": None,
                    "reciprocal_rank": 0.0,
                },
                {
                    "hit": True,
                    "first_hit_turn": 10,
                    "best_rank": 10,
                    "reciprocal_rank": 0.1,
                },
            ]
        )
        self.assertEqual(
            result["actions"]["KEEP_P11"]["metrics"]["row_uniform_official"],
            expected,
        )

    def test_weighted_views_have_distinct_and_expected_weighting(self) -> None:
        metrics = self._aggregate()["actions"]["KEEP_P11"]["metrics"]

        self.assertEqual(metrics["source_weighted"]["hit_rate_at_10"], 0.909091)
        # target-a contributes its two-row mean and target-b contributes equally.
        self.assertEqual(metrics["target_uniform"]["hit_rate_at_10"], 0.75)
        self.assertEqual(metrics["taxonomy_balanced"]["hit_rate_at_10"], 0.75)
        self.assertEqual(metrics["source_weighted"]["mttc"], 3.545455)

    def test_relative_rescues_and_oracle_upper_bound(self) -> None:
        result = self._aggregate()
        relative = result["actions"]["RERANK"]["relative_to_baseline"]

        self.assertEqual(relative["miss_to_hit"], 1)
        self.assertEqual(relative["hit_to_miss"], 1)
        self.assertEqual(relative["hit_count_delta"], 0)
        self.assertEqual(relative["net_rescues"], 0)
        self.assertEqual(relative["hit_rate_delta"], 0.0)
        self.assertEqual(relative["rescue_scenario_span"], 1)
        self.assertEqual(relative["rescue_taxonomy_span"], 1)
        self.assertEqual(relative["positive_net_scenario_span"], 0)
        self.assertEqual(relative["positive_net_taxonomy_span"], 1)
        self.assertEqual(result["oracle"]["selection_counts"], {
            "KEEP_P11": 0,
            "RERANK": 1,
            "SEMANTIC": 2,
        })
        self.assertEqual(
            result["oracle"]["metrics"]["row_uniform_official"]["hit_rate_at_10"],
            1.0,
        )
        self.assertEqual(result["oracle"]["relative_to_baseline"]["miss_to_hit"], 1)
        self.assertEqual(
            result["oracle"]["relative_to_baseline"]["positive_net_scenario_span"],
            1,
        )
        self.assertEqual(
            result["oracle"]["relative_to_baseline"]["positive_net_taxonomy_span"],
            1,
        )
        # ASK is deliberately stronger on row 3 but is not oracle eligible.
        self.assertEqual(
            result["oracle"]["metrics"]["row_uniform_official"]["mrr"],
            round((1.0 + 0.5 + 1.0) / 3, 6),
        )

    def test_bootstrap_is_target_clustered_and_deterministic(self) -> None:
        first = self._aggregate()["oracle"]["paired_utility_bootstrap_ci"]
        second = self._aggregate()["oracle"]["paired_utility_bootstrap_ci"]

        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["resamples"], 1000)
        self.assertGreaterEqual(first["upper"], first["observed_mean"])
        self.assertLessEqual(first["lower"], first["observed_mean"])

    def test_string_bootstrap_seed_is_stably_normalized(self) -> None:
        seed = "p12-selection-v1"
        expected = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")

        first = self._aggregate(bootstrap_seed=seed)["oracle"][
            "paired_utility_bootstrap_ci"
        ]
        second = self._aggregate(bootstrap_seed=seed)["oracle"][
            "paired_utility_bootstrap_ci"
        ]

        self.assertEqual(first, second)
        self.assertEqual(first["seed"], expected)

    def test_output_contains_no_join_identifiers_or_stratum_values(self) -> None:
        result_text = repr(self._aggregate())
        for forbidden in ("target-a", "target-b", "tax-a", "tax-b", "ordinal"):
            self.assertNotIn(forbidden, result_text)

    def test_rejects_bad_ledgers_and_action_contracts(self) -> None:
        with self.assertRaises(P12OracleMetricsError):
            aggregate_action_oracle(
                [], action_ids=ACTIONS, oracle_eligible_actions=ELIGIBLE
            )
        with self.assertRaises(P12OracleMetricsError):
            self._aggregate(oracle_eligible_actions=("UNKNOWN",))

        duplicate = [dict(self.records[0]), dict(self.records[1])]
        duplicate[1]["ordinal"] = duplicate[0]["ordinal"]
        with self.assertRaises(P12OracleMetricsError):
            aggregate_action_oracle(
                duplicate, action_ids=ACTIONS, oracle_eligible_actions=ELIGIBLE
            )

        incomplete_hit = [dict(self.records[0])]
        incomplete_hit[0]["actions"] = dict(incomplete_hit[0]["actions"])
        incomplete_hit[0]["actions"]["RERANK"] = {
            "first_hit_turn": 1,
            "first_rank": None,
        }
        with self.assertRaises(P12OracleMetricsError):
            aggregate_action_oracle(
                incomplete_hit, action_ids=ACTIONS, oracle_eligible_actions=ELIGIBLE
            )

        negative_weight = [dict(self.records[0])]
        negative_weight[0]["source_weight"] = -1.0
        with self.assertRaises(P12OracleMetricsError):
            aggregate_action_oracle(
                negative_weight, action_ids=ACTIONS, oracle_eligible_actions=ELIGIBLE
            )

    def test_zero_weight_sum_fails_closed(self) -> None:
        rows = [dict(self.records[0]), dict(self.records[2])]
        for row in rows:
            row["source_weight"] = 0.0
        with self.assertRaises(P12OracleMetricsError):
            aggregate_action_oracle(
                rows, action_ids=ACTIONS, oracle_eligible_actions=ELIGIBLE
            )


if __name__ == "__main__":
    unittest.main()
