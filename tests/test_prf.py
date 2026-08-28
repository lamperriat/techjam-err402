from __future__ import annotations

import dataclasses
import json
import math
import re
import unittest

from starter.prf import (
    MIN_FEEDBACK_TERMS_FOR_FUSION,
    SCHEMA_VERSION,
    PrfConfig,
    build_prf_expression,
    extract_feedback_terms,
    guarded_prf_fusion,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _fields(
    title: str = "",
    categories: str = "",
    features: str = "",
    details: str = "",
    store: str = "",
    description: str = "",
) -> tuple[str, str, str, str, str, str]:
    return title, categories, features, details, store, description


class PrfTest(unittest.TestCase):
    def test_config_is_frozen_and_validated(self) -> None:
        config = PrfConfig()
        self.assertEqual(
            [field.name for field in dataclasses.fields(config)],
            [
                "seed_depth",
                "min_query_terms",
                "min_seed_count",
                "min_seed_coverage",
                "min_support_count",
                "min_support_ratio",
                "max_feedback_terms",
                "max_df_ratio",
                "min_novel_documents",
                "route_depth",
                "rrf_k",
                "prf_weight",
                "max_top10_newcomers",
            ],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.prf_weight = 0.5  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "min_seed_count"):
            PrfConfig(seed_depth=2, min_seed_count=3)
        with self.assertRaisesRegex(ValueError, "max_top10_newcomers"):
            PrfConfig(max_top10_newcomers=2)
        self.assertEqual(config.prf_weight, 0.15)
        self.assertEqual(config.max_feedback_terms, 4)
        self.assertEqual(config.min_seed_coverage, 2)

    def test_extracts_only_supported_novel_multifield_ascii_terms(self) -> None:
        config = PrfConfig(max_feedback_terms=4)
        seed_ids = ["s1", "s2", "s3", "s4", "s5"]
        seed_fields = {
            "s1": _fields("lightweight blue runner", "shoe", "breathable", "mesh", "Nike"),
            "s2": _fields("lightweight common", "breathable shoe", "mesh", "rare", "Nike"),
            "s3": _fields(
                "lightweight singlefield",
                "shoe",
                "breathable lightweight",
                "rare",
                "Adidas",
            ),
            "s4": _fields("common singlefield", "shoe", "breathable", "", "Adidas"),
            "s5": _fields("common singlefield café cotton2", "shoe", "", "", "Puma", "descriptiononly"),
        }
        document_frequencies = {
            "lightweight": 10,
            "breathable": 12,
            "common": 100,
            "singlefield": 9,
            "rare": 4,
            "shoe": 20,
            "blue": 10,
            "nike": 8,
            "mesh": 8,
        }

        terms, diagnostics = extract_feedback_terms(
            ["blue", "shoe"],
            ["mesh"],
            seed_ids,
            seed_fields,
            1_000,
            document_frequencies,
            config,
        )
        by_term = {item["term"]: item for item in diagnostics["term_diagnostics"]}

        self.assertEqual(set(terms), {"breathable", "lightweight"})
        self.assertEqual(diagnostics["schema_version"], SCHEMA_VERSION)
        self.assertFalse(diagnostics["fallback"])
        self.assertIn("original_query_term", by_term["shoe"]["rejection_reasons"])
        self.assertIn("excluded_term", by_term["mesh"]["rejection_reasons"])
        self.assertIn(
            "document_frequency_above_maximum",
            by_term["common"]["rejection_reasons"],
        )
        self.assertIn(
            "field_group_support_below_minimum",
            by_term["singlefield"]["rejection_reasons"],
        )
        self.assertIn(
            "novel_document_support_below_minimum",
            by_term["rare"]["rejection_reasons"],
        )
        self.assertNotIn("descriptiononly", by_term)
        self.assertNotIn("caf", by_term)
        self.assertNotIn("cotton", by_term)
        expected_idf = math.log(1 + (1_000 - 10 + 0.5) / (10 + 0.5))
        self.assertAlmostEqual(by_term["lightweight"]["bm25_idf"], expected_idf, 11)
        expected_discount = sum(
            1 / math.log2(rank + 1) for rank in (1, 2, 3)
        ) / sum(1 / math.log2(rank + 1) for rank in range(1, 6))
        self.assertAlmostEqual(
            by_term["lightweight"]["normalized_rank_discount"],
            expected_discount,
            11,
        )
        self.assertAlmostEqual(
            by_term["lightweight"]["score"],
            expected_idf * expected_discount,
            11,
        )

    def test_dynamic_store_brand_is_excluded_even_when_seen_in_content(self) -> None:
        seeds = ["s1", "s2", "s3", "s4", "s5"]
        fields = {
            identifier: _fields("nike durable", "shoe", "nike comfort", "", "Nike Store")
            for identifier in seeds
        }
        terms, diagnostics = extract_feedback_terms(
            ["blue", "shoe"], [], seeds, fields, 1_000, {"nike": 10, "durable": 10, "comfort": 10}, PrfConfig()
        )
        by_term = {item["term"]: item for item in diagnostics["term_diagnostics"]}

        self.assertNotIn("nike", terms)
        self.assertIn("dynamic_store_brand", by_term["nike"]["rejection_reasons"])
        self.assertIn("nike", diagnostics["dynamic_store_brand_terms"])

    def test_feedback_terms_shorter_than_three_characters_are_noise(self) -> None:
        seeds = ["s1", "s2", "s3", "s4", "s5"]
        fields = {
            identifier: _fields(
                "uv durable", "shoe", "uv comfort", "durable comfort", "Store"
            )
            for identifier in seeds
        }

        terms, diagnostics = extract_feedback_terms(
            ["blue", "shoe"],
            [],
            seeds,
            fields,
            1_000,
            {"uv": 10, "durable": 10, "comfort": 10},
            PrfConfig(),
        )
        observed = {item["term"] for item in diagnostics["term_diagnostics"]}

        self.assertEqual(set(terms), {"comfort", "durable"})
        self.assertNotIn("uv", observed)

    def test_feedback_falls_back_for_short_query_or_fewer_than_two_terms(self) -> None:
        seeds = ["s1", "s2", "s3", "s4", "s5"]
        fields = {
            identifier: _fields("durable", "shoe", "durable", "", "Store")
            for identifier in seeds
        }
        short, short_diagnostics = extract_feedback_terms(
            ["shoe"], [], seeds, fields, 1_000, {"shoe": 10, "durable": 10}, PrfConfig()
        )
        sparse, sparse_diagnostics = extract_feedback_terms(
            ["blue", "shoe"], [], seeds, fields, 1_000, {"shoe": 10, "durable": 10}, PrfConfig()
        )

        self.assertEqual(short, [])
        self.assertIn(
            "query_term_count_below_minimum", short_diagnostics["fallback_reasons"]
        )
        self.assertEqual(sparse, [])
        self.assertIn(
            "fewer_than_two_feedback_terms", sparse_diagnostics["fallback_reasons"]
        )

    def test_seed_count_and_coverage_are_explicit_fallbacks(self) -> None:
        fields = {"s1": _fields("durable", "shoe", "durable", "", "Store")}
        terms, diagnostics = extract_feedback_terms(
            ["blue", "shoe"], [], ["s1", "missing", "missing2"], fields, 1_000, {"durable": 10, "shoe": 10}, PrfConfig()
        )

        self.assertEqual(terms, [])
        self.assertIn("seed_count_below_minimum", diagnostics["fallback_reasons"])
        self.assertNotIn("seed_coverage_below_minimum", diagnostics["fallback_reasons"])

    def test_three_available_seeds_do_not_fail_an_availability_ratio_gate(self) -> None:
        seed_ids = ["s1", "s2", "s3", "missing1", "missing2"]
        fields = {
            identifier: _fields(
                "breathable durable", "shoe breathable", "durable", "", "Store"
            )
            for identifier in seed_ids[:3]
        }
        terms, diagnostics = extract_feedback_terms(
            ["blue", "shoe"],
            [],
            seed_ids,
            fields,
            1_000,
            {"breathable": 8, "durable": 8, "shoe": 10},
            PrfConfig(),
        )

        self.assertEqual(set(terms), {"breathable", "durable"})
        self.assertFalse(diagnostics["fallback"])
        self.assertEqual(diagnostics["available_seed_ratio"], 0.6)

    def test_build_expression_is_unique_ascii_and_query_first(self) -> None:
        expression = build_prf_expression(
            ["Blue", "shoe", "blue", "cotton2"],
            ["Breathable", "shoe", "café", "Lightweight"],
        )
        self.assertEqual(
            expression,
            '("blue" OR "shoe") AND ("breathable" OR "lightweight")',
        )
        self.assertEqual(build_prf_expression(["blue"], []), "")

    @staticmethod
    def _base_rankings() -> dict[str, list[str]]:
        base = [f"b{index}" for index in range(1, 13)]
        return {
            "broad": list(base),
            "strict": ["b1", "b2", "b11"],
            "fused": list(base),
            "final": list(base),
        }

    @staticmethod
    def _base_fields() -> dict[str, tuple[str, str, str, str, str, str]]:
        fields = {f"b{index}": _fields("blue", "", "", "", "", "") for index in range(1, 13)}
        fields["b11"] = _fields(
            "blue breathable", "", "cotton lightweight", "", "", ""
        )
        return fields

    def test_same_coverage_base_pool_newcomer_requires_route_evidence(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        final, diagnostics = guarded_prf_fusion(
            ["blue", "cotton"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            PrfConfig(),
            _tokenize,
        )

        self.assertEqual(final[:9], rankings["final"][:9])
        self.assertEqual(final[9], "b11")
        self.assertEqual(final[10], "b10")
        self.assertEqual(diagnostics["guard"]["newcomer_count"], 1)
        self.assertTrue(diagnostics["guard"]["top9_unchanged"])

        rankings["broad"].remove("b11")
        rankings["strict"].remove("b11")
        guarded, rejected = guarded_prf_fusion(
            ["blue"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            PrfConfig(min_query_terms=1),
            _tokenize,
        )
        self.assertEqual(guarded[:10], rankings["final"][:10])
        decision = next(
            item
            for item in rejected["guard"]["candidate_decisions"]
            if item["identifier"] == "b11"
        )
        self.assertIn(
            "same_coverage_requires_broad_and_strict_route_evidence",
            decision["rejection_reasons"],
        )

    def test_prf_only_requires_strict_original_query_coverage_advantage(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        fields["x"] = _fields(
            "blue breathable", "", "lightweight", "", "", ""
        )
        unchanged, rejected = guarded_prf_fusion(
            ["blue"], [], ["breathable", "lightweight"], rankings, ["x"], fields, PrfConfig(min_query_terms=1), _tokenize
        )
        self.assertEqual(unchanged[:10], rankings["final"][:10])
        candidate_decision = next(
            item
            for item in rejected["guard"]["candidate_decisions"]
            if item["identifier"] == "x"
        )
        self.assertIn(
            "prf_only_requires_strict_coverage_advantage",
            candidate_decision["rejection_reasons"],
        )

        fields["x"] = _fields(
            "blue cotton breathable", "", "lightweight", "", "", ""
        )
        promoted, accepted = guarded_prf_fusion(
            ["blue", "cotton"], [], ["breathable", "lightweight"], rankings, ["x"], fields, PrfConfig(), _tokenize
        )
        self.assertEqual(promoted[:9], rankings["final"][:9])
        self.assertEqual(promoted[9], "x")
        self.assertEqual(accepted["guard"]["newcomers"], ["x"])

    def test_feedback_and_excluded_guards_block_newcomer(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        fields["x"] = _fields(
            "blue cotton leather onlyone one", "", "two", "", "", ""
        )
        for feedback, excluded, expected_reason in (
            (["onlyone"], [], "fewer_than_two_feedback_terms"),
            (["one", "two"], ["leather"], "excluded_term_match"),
        ):
            final, diagnostics = guarded_prf_fusion(
                ["blue", "cotton"], excluded, feedback, rankings, ["x"], fields, PrfConfig(), _tokenize
            )
            self.assertEqual(final[:10], rankings["final"][:10])
            candidate_decision = next(
                item
                for item in diagnostics["guard"]["candidate_decisions"]
                if item["identifier"] == "x"
            )
            reasons = candidate_decision["rejection_reasons"]
            self.assertIn(expected_reason, reasons)

    def test_candidate_must_match_at_least_two_feedback_terms(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        fields["x"] = _fields(
            "blue cotton breathable", "", "", "", "", ""
        )
        final, diagnostics = guarded_prf_fusion(
            ["blue", "cotton"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["x"],
            fields,
            PrfConfig(),
            _tokenize,
        )
        candidate = next(
            item
            for item in diagnostics["guard"]["candidate_decisions"]
            if item["identifier"] == "x"
        )

        self.assertEqual(final, rankings["final"])
        self.assertEqual(candidate["feedback_match_count"], 1)
        self.assertIn(
            "feedback_match_count_below_minimum",
            candidate["rejection_reasons"],
        )

    def test_same_coverage_route_evidence_when_strict_route_is_empty(self) -> None:
        rankings = self._base_rankings()
        rankings["strict"] = []
        fields = self._base_fields()
        accepted, diagnostics = guarded_prf_fusion(
            ["blue"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            PrfConfig(min_query_terms=1),
            _tokenize,
        )
        self.assertEqual(accepted[9], "b11")
        self.assertEqual(diagnostics["guard"]["newcomers"], ["b11"])

        rankings["broad"] = [
            *[f"z{index}" for index in range(1, 31)],
            "b11",
        ]
        rejected, rejected_diagnostics = guarded_prf_fusion(
            ["blue"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            PrfConfig(min_query_terms=1),
            _tokenize,
        )
        candidate = next(
            item
            for item in rejected_diagnostics["guard"]["candidate_decisions"]
            if item["identifier"] == "b11"
        )
        self.assertEqual(rejected, rankings["final"])
        self.assertIn(
            "same_coverage_requires_broad_top30_when_strict_empty",
            candidate["rejection_reasons"],
        )

    def test_proposal_b_score_uses_fused_rank_and_must_beat_incumbent(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        config = PrfConfig(min_query_terms=1, prf_weight=0.0)

        rejected, rejected_diagnostics = guarded_prf_fusion(
            ["blue"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            config,
            _tokenize,
        )
        rejected_candidate = next(
            item
            for item in rejected_diagnostics["guard"]["candidate_decisions"]
            if item["identifier"] == "b11"
        )
        self.assertEqual(rejected, rankings["final"])
        self.assertIn(
            "same_coverage_proposal_score_not_strictly_higher",
            rejected_candidate["rejection_reasons"],
        )
        self.assertIn(
            "proposal_not_ranked_ahead_of_incumbent",
            rejected_candidate["rejection_reasons"],
        )

        rankings["fused"] = [
            "b11",
            *[identifier for identifier in rankings["fused"] if identifier != "b11"],
        ]
        accepted, accepted_diagnostics = guarded_prf_fusion(
            ["blue"],
            [],
            ["breathable", "lightweight"],
            rankings,
            ["b11"],
            fields,
            config,
            _tokenize,
        )
        proposal = {
            item["identifier"]: item for item in accepted_diagnostics["proposal"]
        }
        self.assertEqual(accepted[9], "b11")
        self.assertEqual(proposal["b11"]["base_rank"], 11)
        self.assertEqual(proposal["b11"]["fused_rank"], 1)
        self.assertAlmostEqual(proposal["b11"]["base_rrf_score"], 1 / 61, 11)
        self.assertLess(
            proposal["b11"]["proposal_rank"], proposal["b10"]["proposal_rank"]
        )

    def test_fallback_preserves_full_baseline_beyond_route_depth(self) -> None:
        base = [f"b{index}" for index in range(1, 151)]
        rankings = {
            "broad": list(base),
            "strict": list(base),
            "fused": list(base),
            "final": list(base),
        }
        fields = {
            identifier: _fields("blue", "", "", "", "", "")
            for identifier in base
        }
        final, diagnostics = guarded_prf_fusion(
            ["blue", "shoe"],
            [],
            [],
            rankings,
            ["b125"],
            fields,
            PrfConfig(route_depth=120),
            _tokenize,
        )

        self.assertEqual(final, base)
        self.assertEqual(len(final), 150)
        proposal_by_id = {
            item["identifier"]: item for item in diagnostics["proposal"]
        }
        self.assertEqual(proposal_by_id["b125"]["broad_rank"], 125)

    def test_proposal_ties_are_deterministic_and_guard_limits_newcomers(self) -> None:
        rankings = self._base_rankings()
        fields = self._base_fields()
        fields.update({
            "x": _fields("blue cotton one", "", "two", "", "", ""),
            "y": _fields("blue cotton one", "", "two", "", "", ""),
        })
        config = PrfConfig(prf_weight=0.0)
        first, first_diagnostics = guarded_prf_fusion(
            ["blue", "cotton"], [], ["one", "two"], rankings, ["y", "x"], fields, config, _tokenize
        )
        second, second_diagnostics = guarded_prf_fusion(
            ["blue", "cotton"], [], ["one", "two"], rankings, ["y", "x"], fields, config, _tokenize
        )

        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(first[9], "y")
        self.assertLessEqual(first_diagnostics["guard"]["newcomer_count"], 1)
        serialized = json.dumps(first_diagnostics, sort_keys=True)
        for forbidden in ("target_id", "sample_id", "scenario_type"):
            self.assertNotIn(forbidden, serialized)

    def test_fusion_constant_requires_two_feedback_terms(self) -> None:
        self.assertEqual(MIN_FEEDBACK_TERMS_FOR_FUSION, 2)


if __name__ == "__main__":
    unittest.main()
