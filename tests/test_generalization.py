from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_generalization import (
    AUDIT_RULES,
    CHALLENGE_RULES,
    DEV_RULES,
    PerturbedAgent,
    _parser,
    _session_changes,
    _suite_names,
    _suite_registry_sha256,
    build_product_disjoint_samples,
    evaluate_suites,
    perturb_message,
)
from starter.frozen_winner import FROZEN_WINNER_ID


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.profile = (session_id, user_profile)

    def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
        self.calls.append((session_id, message, turn, top_k))
        return {"message": "ok", "ask_attribute": None, "recommendations": []}


class GeneralizationTest(unittest.TestCase):
    def test_cli_defaults_to_off_reranking(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.rerank_mode, "off")
        self.assertIsNone(args.architecture_variant)

    def test_dev_and_challenge_paraphrases_preserve_visible_constraint(self) -> None:
        original = (
            "Actually, ignore my earlier preference. "
            "What I need is: blue cotton casual."
        )
        dev, dev_applied = perturb_message(original, DEV_RULES)
        challenge, challenge_applied = perturb_message(original, CHALLENGE_RULES)
        audit, audit_applied = perturb_message(original, AUDIT_RULES)

        self.assertIn("blue cotton casual", dev)
        self.assertIn("blue cotton casual", challenge)
        self.assertIn("blue cotton casual", audit)
        self.assertIn("override_dev", dev_applied)
        self.assertIn("override_challenge", challenge_applied)
        self.assertIn("override_audit", audit_applied)
        self.assertNotEqual(dev, challenge)

    def test_adapter_changes_only_visible_message(self) -> None:
        delegate = FakeAgent()
        adapter = PerturbedAgent(delegate, DEV_RULES)
        profile = {"summary": "neutral"}
        adapter.reset("opaque", profile)
        response = adapter.respond(
            "opaque", "I'm looking for shoes, but I'm still exploring.", 1, 10
        )

        self.assertEqual(delegate.profile, ("opaque", profile))
        self.assertEqual(delegate.calls[0][0::2], ("opaque", 1))
        self.assertIn("I need shoes", delegate.calls[0][1])
        self.assertEqual(response["message"], "ok")
        self.assertEqual(adapter.stats()["transformed_messages"], 1)

    def test_product_disjoint_samples_are_deterministic_and_stratified(self) -> None:
        public = [
            {
                "sample_id": "public_1",
                "ground_truth": {"parent_asin": "PUBLIC"},
            }
        ]
        products = {
            "PUBLIC": {"title": "Public", "categories": ["Clothing"]},
            **{
                f"P{index:03d}": {
                    "title": f"Product {index}",
                    "categories": ["Clothing", "Test"],
                }
                for index in range(250)
            },
        }
        first, first_metadata = build_product_disjoint_samples(
            public, products, 200, "fixed-seed"
        )
        second, second_metadata = build_product_disjoint_samples(
            public, products, 200, "fixed-seed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_metadata["samples_sha256"], second_metadata["samples_sha256"])
        self.assertEqual(first_metadata["public_target_overlap"], 0)
        self.assertEqual(first_metadata["unique_target_count"], 200)
        self.assertEqual(
            first_metadata["scenario_counts"],
            {"boundary": 10, "browsing": 80, "buying": 80, "intent_override": 30},
        )
        self.assertNotIn(
            "PUBLIC", {sample["ground_truth"]["parent_asin"] for sample in first}
        )

    def test_suite_selection_composes_default_and_audit(self) -> None:
        self.assertEqual(
            _suite_names(["default", "combined_audit"]),
            ["combined_dev", "combined_challenge", "combined_audit"],
        )
        self.assertIn("override_dev", _suite_names(["all"]))
        self.assertEqual(len(_suite_registry_sha256()), 64)

    def test_session_change_uses_official_weighted_contribution(self) -> None:
        baseline = {
            "sessions": [{
                "sample_id": "paired",
                "hit": True,
                "first_hit_turn": 1,
                "best_rank": 10,
                "reciprocal_rank": 0.1,
            }]
        }
        current = {
            "sessions": [{
                "sample_id": "paired",
                "hit": True,
                "first_hit_turn": 2,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            }]
        }

        changes = _session_changes(current, baseline)

        self.assertEqual(changes["official_score_improvement_count"], 1)
        self.assertEqual(changes["later_hit_count"], 1)
        self.assertEqual(changes["rank_improvement_count"], 1)

    def test_suite_runner_propagates_rerank_mode_to_agent(self) -> None:
        calls: list[dict] = []

        class Connection:
            def close(self) -> None:
                pass

        class CapturingAgent:
            def __init__(self, catalog_path: Path, **kwargs: object) -> None:
                calls.append({"catalog_path": catalog_path, **kwargs})
                self.connection = Connection()

        result = {
            "sample_count": 1,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": 11.0,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
            "reported_token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "scenario_metrics": {},
            "sessions": [],
        }
        with (
            patch("scripts.evaluate_generalization.Agent", CapturingAgent),
            patch("scripts.evaluate_generalization.evaluate", return_value=result),
        ):
            artifact = evaluate_suites(
                Path("catalog.jsonl"),
                [{"sample_id": "sample"}],
                set(),
                {},
                {},
                [],
                "fast",
                "unit",
                "shadow",
            )

        self.assertEqual(calls[0]["question_policy"], "fast")
        self.assertEqual(calls[0]["rerank_mode"], "shadow")
        self.assertEqual(artifact["sample_count"], 1)

    def test_suite_runner_uses_only_the_frozen_winner(self) -> None:
        calls: list[dict] = []

        class Connection:
            def close(self) -> None:
                pass

        class CapturingArchitectureAgent:
            def __init__(
                self,
                catalog_path: Path,
                variant_id: str,
                **kwargs: object,
            ) -> None:
                calls.append({
                    "catalog_path": catalog_path,
                    "variant_id": variant_id,
                    **kwargs,
                })
                self.connection = Connection()

            def experiment_stats(self) -> dict:
                return {"activations": 1, "output_changes": 1}

        result = {
            "sample_count": 1,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": 11.0,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
            "reported_token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "scenario_metrics": {},
            "sessions": [],
        }
        with (
            patch(
                "scripts.evaluate_generalization.ArchitectureAgent",
                CapturingArchitectureAgent,
            ),
            patch("scripts.evaluate_generalization.evaluate", return_value=result),
        ):
            artifact = evaluate_suites(
                Path("catalog.jsonl"),
                [{"sample_id": "sample"}],
                set(),
                {},
                {},
                [],
                "fast",
                "unit",
                "off",
                FROZEN_WINNER_ID,
            )

        self.assertEqual(calls[0]["variant_id"], FROZEN_WINNER_ID)
        self.assertEqual(calls[0]["question_policy"], "fast")
        self.assertEqual(
            artifact["suites"]["canonical"]["architecture_stats"],
            {"activations": 1, "output_changes": 1},
        )

    def test_suite_runner_rejects_unfrozen_winner_configuration(self) -> None:
        for variant, question_policy, rerank_mode in (
            ("R07.combsum_bm25", "fast", "off"),
            (FROZEN_WINNER_ID, "boundary", "off"),
            (FROZEN_WINNER_ID, "fast", "shadow"),
        ):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                evaluate_suites(
                    Path("catalog.jsonl"),
                    [],
                    set(),
                    {},
                    {},
                    [],
                    question_policy,
                    "unit",
                    rerank_mode,
                    variant,
                )


if __name__ == "__main__":
    unittest.main()
