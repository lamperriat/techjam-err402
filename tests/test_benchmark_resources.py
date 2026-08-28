from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_resources import (
    ArchitectureRankCaptureAgent,
    ROUTES,
    SessionCapture,
    TurnCapture,
    _nearest_rank,
    _parser,
    build_benchmark,
    build_route_audit,
    latency_summary,
)
from starter.frozen_winner import FROZEN_WINNER_ID


PRODUCTS = [
    {
        "parent_asin": "TARGET-BLUE",
        "title": "Women's blue cotton casual dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["cotton", "blue", "casual"],
        "details": {"material": "cotton", "color": "blue"},
        "description": ["summer dress with pockets"],
        "store": "Example",
        "price": 49.0,
    },
    {
        "parent_asin": "OTHER-RED",
        "title": "Women's red polyester formal dress",
        "categories": ["Clothing", "Women", "Dresses"],
        "features": ["polyester", "red", "formal"],
        "details": {"material": "polyester", "color": "red"},
        "description": ["formal dress"],
        "store": "Example",
        "price": 59.0,
    },
    {
        "parent_asin": "OTHER-SHOE",
        "title": "Men's black mesh running shoe",
        "categories": ["Clothing", "Men", "Shoes"],
        "features": ["black", "mesh", "running"],
        "details": {"material": "mesh", "color": "black"},
        "description": ["wide training sneaker"],
        "store": "Example",
        "price": 69.0,
    },
]


class BenchmarkResourcesTest(unittest.TestCase):
    def _files(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory)
        catalog = root / "catalog.jsonl"
        dataset = root / "public.jsonl"
        catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        samples = [
            {
                "sample_id": "public_tiny_1",
                "scenario_type": "buying",
                "user_profile": {"summary": "neutral"},
                "ground_truth": {"parent_asin": "TARGET-BLUE"},
            },
            {
                "sample_id": "public_tiny_2",
                "scenario_type": "browsing",
                "user_profile": {"summary": "neutral"},
                "ground_truth": {"parent_asin": "OTHER-SHOE"},
            },
        ]
        dataset.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        return catalog, dataset

    def test_nearest_rank_latency_summary_uses_observed_values(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        self.assertEqual(_nearest_rank(values, 0.50), 3.0)
        self.assertEqual(_nearest_rank(values, 0.95), 100.0)
        summary = latency_summary([1_000_000, 2_000_000, 3_000_000])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertEqual(summary["p99_ms"], 3.0)
        self.assertEqual(summary["max_ms"], 3.0)

    def test_posthoc_audit_excludes_pre_override_rank_and_lists_miss(self) -> None:
        sample = {
            "sample_id": "public_override",
            "scenario_type": "intent_override",
            "user_profile": {},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "target_category": "dress",
                "hard_constraints": ["blue"],
                "soft_preferences": ["casual"],
            },
            "behavior": {
                "scenario_type": "intent_override",
                "override": {"turn": 3, "new_value": "blue", "message": "blue"},
            },
        }
        captures = [SessionCapture(turns=[
            TurnCapture(1, {
                "broad": ("TARGET",),
                "strict": ("TARGET",),
                "fused": ("TARGET",),
                "reranked": ("TARGET",),
                "final": ("TARGET",),
            }),
            TurnCapture(3, {
                "broad": tuple([f"B{i}" for i in range(14)] + ["TARGET"]),
                "strict": (),
                "fused": tuple([f"F{i}" for i in range(14)] + ["TARGET"]),
                "reranked": tuple([f"R{i}" for i in range(9)] + ["TARGET"]),
                "final": tuple([f"F{i}" for i in range(14)] + ["TARGET"]),
            }),
        ])]
        evaluator_result = {
            "sessions": [{"sample_id": "public_override", "hit": False}]
        }

        audit = build_route_audit([sample], evaluator_result, captures, {})

        self.assertEqual(audit["routes"]["fused"]["recall_at_k"]["10"], 0.0)
        self.assertEqual(audit["routes"]["fused"]["recall_at_k"]["20"], 1.0)
        self.assertEqual(
            audit["public_misses"][0]["best_route_ranks"],
            {
                "broad": 15,
                "strict": None,
                "fused": 15,
                "reranked": 10,
                "final": 15,
            },
        )
        self.assertEqual(
            audit["public_misses"][0]["best_route_turns"],
            {
                "broad": 3,
                "strict": None,
                "fused": 3,
                "reranked": 3,
                "final": 3,
            },
        )
        self.assertEqual(audit["public_misses"][0]["best_fused_rank"], 15)
        self.assertEqual(audit["public_misses"][0]["best_fused_turn"], 3)
        self.assertNotIn(1, audit["public_misses"][0]["observed_eligible_turns"])

    def test_two_run_smoke_is_deterministic_target_blind_and_no_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, dataset = self._files(directory)
            with patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "must-not-appear-in-artifact"},
                clear=False,
            ):
                artifact = build_benchmark(
                    catalog,
                    dataset,
                    runs=2,
                    rerank_mode="shadow",
                    sample_limit=2,
                    rss_sample_ms=1.0,
                )

        serialized = json.dumps(artifact)
        self.assertEqual(artifact["determinism"]["status"], "passed")
        self.assertTrue(artifact["all_runs_no_key_default_verified"])
        self.assertNotIn("must-not-appear-in-artifact", serialized)
        self.assertNotIn("OPENAI_API_KEY", serialized)
        self.assertEqual(artifact["configuration"]["rerank_mode"], "shadow")
        self.assertEqual(artifact["configuration"]["retrieval_mode"], "control")
        self.assertEqual(artifact["configuration"]["captured_routes"], list(ROUTES))
        self.assertEqual(len(artifact["configuration"]["attribute_source_sha256"]), 64)
        self.assertEqual(len(artifact["configuration"]["reranker_source_sha256"]), 64)
        self.assertEqual(len(artifact["configuration"]["slot_ledger_source_sha256"]), 64)
        self.assertEqual(len(artifact["configuration"]["clarification_source_sha256"]), 64)
        for run in artifact["runs"]:
            self.assertGreater(run["respond_call_count"], 0)
            self.assertEqual(
                run["respond_call_count"], run["respond_latency"]["count"]
            )
            self.assertIn("p95_ms", run["respond_latency"])
            self.assertIn("index_build", run["timing_seconds"])
            self.assertIn("evaluator_wall", run["timing_seconds"])
            self.assertIn("run_peak_rss_bytes", run["memory"])
            self.assertEqual(
                run["route_audit"]["cutoffs"], [10, 20, 50, 80, 120]
            )
            self.assertEqual(set(run["route_audit"]["routes"]), set(ROUTES))
            self.assertTrue(
                run["no_key_default"]["agent_closed_before_posthoc_label_join"]
            )

    def test_runtime_selection_rejects_empty_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, dataset = self._files(directory)
            with self.assertRaisesRegex(ValueError, "sample selection is empty"):
                build_benchmark(
                    catalog,
                    dataset,
                    runs=1,
                    scenarios=("boundary",),
                    rss_sample_ms=1.0,
                )

    def test_cli_defaults_to_off_reranking(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.rerank_mode, "off")
        self.assertIsNone(args.architecture_variant)
        self.assertIsNone(args.retrieval_mode)

    def test_frozen_winner_capture_matches_served_final_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, _ = self._files(directory)
            agent = ArchitectureRankCaptureAgent(
                catalog,
                FROZEN_WINNER_ID,
                question_policy="fast",
            )
            try:
                agent.reset("opaque", {})
                response = agent.respond(
                    "opaque",
                    "I need a women's blue cotton casual dress.",
                    10,
                    10,
                )
                rankings = agent.take_last_rankings()
            finally:
                agent.connection.close()

        self.assertEqual(
            [value.__name__ for value in ArchitectureRankCaptureAgent.mro()[:4]],
            [
                "ArchitectureRankCaptureAgent",
                "RankCaptureAgent",
                "ArchitectureAgent",
                "Agent",
            ],
        )
        self.assertIsNotNone(rankings)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            rankings["final"][:10],
        )
        self.assertEqual(agent.architecture_spec.variant_id, FROZEN_WINNER_ID)
        self.assertEqual(agent.rerank_mode, "off")

    def test_frozen_winner_two_run_gate_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, dataset = self._files(directory)
            artifact = build_benchmark(
                catalog,
                dataset,
                runs=2,
                architecture_variant=FROZEN_WINNER_ID,
                rss_sample_ms=1.0,
            )

        self.assertTrue(artifact["frozen_winner_gate"]["passed"])
        self.assertEqual(
            artifact["configuration"]["retrieval_mode"],
            f"architecture:{FROZEN_WINNER_ID}",
        )
        self.assertEqual(artifact["determinism"]["status"], "passed")
        for run in artifact["runs"]:
            self.assertEqual(len(run["target_blind_trace_sha256"]), 64)
            self.assertEqual(len(run["architecture_stats_sha256"]), 64)
            self.assertIsNotNone(run["architecture_stats"])

    def test_frozen_winner_rejects_unfrozen_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, dataset = self._files(directory)
            for kwargs in (
                {"architecture_variant": "R07.combsum_bm25"},
                {
                    "architecture_variant": FROZEN_WINNER_ID,
                    "question_policy": "boundary",
                },
                {
                    "architecture_variant": FROZEN_WINNER_ID,
                    "rerank_mode": "shadow",
                },
                {
                    "architecture_variant": FROZEN_WINNER_ID,
                    "runs": 1,
                },
                {
                    "architecture_variant": FROZEN_WINNER_ID,
                    "sample_limit": 1,
                },
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    build_benchmark(
                        catalog,
                        dataset,
                        rss_sample_ms=1.0,
                        **kwargs,
                    )

    def test_invalid_programmatic_rerank_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, dataset = self._files(directory)
            with self.assertRaisesRegex(ValueError, "rerank mode"):
                build_benchmark(
                    catalog,
                    dataset,
                    runs=1,
                    rerank_mode="invalid",
                    rss_sample_ms=1.0,
                )


if __name__ == "__main__":
    unittest.main()
