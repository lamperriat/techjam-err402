from __future__ import annotations

import dataclasses
import inspect
import json
import re
import unittest

import starter.adaptive_depth as adaptive_depth
from starter.adaptive_depth import (
    BASE_BROAD,
    DEEP,
    PROTECTED_PREFIX,
    SCHEMA_VERSION,
    TOP_K,
    DepthConfig,
    DepthProposal,
    depth_precheck,
    guarded_depth_admission,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [value.lower() for value in TOKEN_RE.findall(text)]


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(1, count + 1)]


def _fields(text: str) -> tuple[str, str, str, str, str, str]:
    return text, "", "", "", "", ""


def _fixture() -> tuple[
    dict[str, list[str]],
    list[str],
    dict[str, int],
    dict[str, tuple[str, str, str, str, str, str]],
]:
    broad = _ids("b", BASE_BROAD)
    final = broad[:TOP_K]
    tail = ["z-winner", "a-lower", *_ids("d", DEEP - BASE_BROAD - 2)]
    rankings = {
        "broad": broad,
        "strict": broad[:20],
        "fused": broad[:40],
        "final": final,
    }
    coverage = {identifier: 1 for identifier in final}
    fields = {
        "z-winner": _fields("blue cotton shirt"),
        "a-lower": _fields("blue shirt"),
    }
    return rankings, [*broad, *tail], coverage, fields


def _run(
    *,
    query: list[str] | None = None,
    excluded: list[str] | None = None,
    rankings: dict[str, list[str]] | None = None,
    deep: list[str] | None = None,
    coverage: dict[str, int] | None = None,
    fields: dict[str, tuple[str, str, str, str, str, str]] | None = None,
    config: DepthConfig | None = None,
) -> tuple[list[str], dict[str, object]]:
    base_rankings, base_deep, base_coverage, base_fields = _fixture()
    return guarded_depth_admission(
        query if query is not None else ["blue", "cotton", "shirt"],
        excluded if excluded is not None else [],
        rankings if rankings is not None else base_rankings,
        deep if deep is not None else base_deep,
        coverage if coverage is not None else base_coverage,
        fields if fields is not None else base_fields,
        config or DepthConfig(),
        _tokenize,
    )


class AdaptiveDepthTest(unittest.TestCase):
    def test_config_is_frozen_has_preregistered_defaults_and_is_validated(self) -> None:
        config = DepthConfig()
        self.assertEqual(
            [field.name for field in dataclasses.fields(config)],
            [
                "base_broad_depth",
                "deep_depth",
                "top_k",
                "protected_prefix",
                "max_top10_newcomers",
                "min_query_terms",
                "strict_coverage_margin",
            ],
        )
        self.assertEqual(
            dataclasses.asdict(config),
            {
                "base_broad_depth": 120,
                "deep_depth": 240,
                "top_k": 10,
                "protected_prefix": 9,
                "max_top10_newcomers": 1,
                "min_query_terms": 2,
                "strict_coverage_margin": 1,
            },
        )
        self.assertEqual((BASE_BROAD, DEEP, TOP_K, PROTECTED_PREFIX), (120, 240, 10, 9))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.deep_depth = 360  # type: ignore[misc]
        for invalid in (
            {"base_broad_depth": 0},
            {"deep_depth": 120},
            {"top_k": 121},
            {"protected_prefix": 8},
            {"max_top10_newcomers": 2},
            {"min_query_terms": True},
            {"strict_coverage_margin": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DepthConfig(**invalid)

    def test_depth_proposal_is_frozen(self) -> None:
        proposal = DepthProposal("x", 2, 121, ("blue", "shirt"), (), True, ())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            proposal.coverage = 3  # type: ignore[misc]

    def test_trigger_truth_table(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        incumbent = rankings["final"][9]
        cases = (
            (
                "query_term_count_below_minimum",
                {"query": ["blue"]},
                "query_terms_ready",
            ),
            (
                "broad_route_not_saturated",
                {
                    "rankings": {**rankings, "broad": rankings["broad"][:-1]},
                    "deep": deep[:-1],
                },
                "broad_saturated",
            ),
            (
                "final_top10_incomplete",
                {"rankings": {**rankings, "final": rankings["final"][:-1]}},
                "final_top10_ready",
            ),
            (
                "incumbent_coverage_missing",
                {"coverage": {}},
                "incumbent_coverage_known",
            ),
            (
                "incumbent_coverage_already_full",
                {"coverage": {**coverage, incumbent: 3}},
                "incumbent_below_full_coverage",
            ),
        )
        for reason, arguments, condition in cases:
            with self.subTest(reason=reason):
                result, diagnostics = _run(**arguments)
                self.assertEqual(result, arguments.get("rankings", rankings)["final"])
                trigger = diagnostics["trigger"]
                self.assertFalse(trigger["enabled"])
                self.assertFalse(trigger["conditions"][condition])
                self.assertIn(reason, trigger["rejection_reasons"])

        result, diagnostics = _run()
        self.assertTrue(diagnostics["trigger"]["enabled"])
        self.assertTrue(all(diagnostics["trigger"]["conditions"].values()))
        self.assertEqual(result[9], "z-winner")

    def test_precheck_exposes_the_same_truth_table_without_a_deep_route(self) -> None:
        rankings, _, coverage, _ = _fixture()
        accepted = depth_precheck(
            ["blue", "cotton", "shirt"], rankings, coverage, DepthConfig()
        )
        self.assertTrue(accepted["enabled"])
        self.assertEqual(accepted["incumbent"], rankings["final"][9])
        self.assertEqual(accepted["incumbent_coverage"], 1)
        self.assertNotIn("prefix_matches", accepted["conditions"])

        rejected = depth_precheck(
            ["blue"], rankings, coverage, DepthConfig()
        )
        self.assertFalse(rejected["enabled"])
        self.assertEqual(
            rejected["rejection_reasons"],
            ["query_term_count_below_minimum"],
        )

    def test_prefix_mismatch_is_an_exact_no_op(self) -> None:
        rankings, deep, _, _ = _fixture()
        deep[0], deep[1] = deep[1], deep[0]
        result, diagnostics = _run(rankings=rankings, deep=deep)
        self.assertEqual(result, rankings["final"])
        self.assertFalse(diagnostics["prefix"]["matches"])
        self.assertTrue(diagnostics["prefix"]["checked"])
        self.assertEqual(diagnostics["guard"]["reason"], "prefix_mismatch")
        self.assertEqual(diagnostics["tail"]["proposals"], [])

    def test_failed_precheck_does_not_claim_a_prefix_mismatch(self) -> None:
        rankings, _, _, _ = _fixture()
        rankings["broad"] = rankings["broad"][:-1]
        result, diagnostics = _run(
            rankings=rankings,
            deep=list(rankings["broad"]),
        )
        self.assertEqual(result, rankings["final"])
        self.assertFalse(diagnostics["triggered"])
        self.assertFalse(diagnostics["prefix"]["checked"])
        self.assertIsNone(diagnostics["prefix"]["matches"])
        self.assertEqual(diagnostics["prefix"]["checked_count"], 0)
        self.assertEqual(
            diagnostics["guard"]["reason"], "broad_route_not_saturated"
        )
        self.assertNotIn(
            "prefix_mismatch", diagnostics["trigger"]["rejection_reasons"]
        )

    def test_candidates_overlapping_any_base_route_are_removed(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        overlaps = [
            rankings["broad"][50],
            "strict-only",
            "fused-only",
            rankings["final"][0],
        ]
        rankings["strict"].append("strict-only")
        rankings["fused"].append("fused-only")
        deep[120:124] = overlaps
        deep[124] = "fresh"
        fields.update(
            {identifier: _fields("blue cotton shirt") for identifier in overlaps}
        )
        fields["fresh"] = _fields("blue cotton shirt")

        result, diagnostics = _run(
            rankings=rankings, deep=deep, coverage=coverage, fields=fields
        )
        self.assertEqual(result[9], "fresh")
        self.assertEqual(diagnostics["tail"]["base_pool_overlap_count"], 4)
        proposal_ids = [
            item["identifier"] for item in diagnostics["tail"]["proposals"]
        ]
        self.assertFalse(set(overlaps) & set(proposal_ids))

    def test_winner_orders_by_coverage_then_deep_rank_then_identifier(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        deep[120:124] = ["z-two", "a-two", "a-one", "a-three"]
        fields.update(
            {
                "z-two": _fields("blue cotton"),
                "a-two": _fields("blue cotton"),
                "a-one": _fields("blue"),
                "a-three": _fields("blue cotton shirt"),
            }
        )
        result, diagnostics = _run(
            rankings=rankings, deep=deep, coverage=coverage, fields=fields
        )
        self.assertEqual(result[9], "a-three")
        proposals = diagnostics["tail"]["proposals"]
        self.assertEqual(
            [item["identifier"] for item in proposals[:3]],
            ["a-three", "z-two", "a-two"],
        )
        self.assertEqual(
            diagnostics["selection_order"],
            ["coverage_desc", "deep_rank_asc", "identifier_asc"],
        )

    def test_excluded_match_blocks_an_otherwise_best_candidate(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        fields["z-winner"] = _fields("blue cotton shirt leather")
        result, diagnostics = _run(
            excluded=["leather"],
            rankings=rankings,
            deep=deep,
            coverage=coverage,
            fields=fields,
        )
        self.assertEqual(result[9], "a-lower")
        winner = next(
            item
            for item in diagnostics["tail"]["proposals"]
            if item["identifier"] == "z-winner"
        )
        self.assertFalse(winner["eligible"])
        self.assertEqual(winner["matched_excluded_terms"], ("leather",))
        self.assertIn("excluded_term_match", winner["rejection_reasons"])

    def test_excluded_term_order_is_canonical_in_complete_diagnostics(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        fields["z-winner"] = _fields("blue cotton shirt red formal")
        forward_result, forward = _run(
            excluded=["red", "formal"],
            rankings=rankings,
            deep=deep,
            coverage=coverage,
            fields=fields,
        )
        reverse_result, reverse = _run(
            excluded=["formal", "red"],
            rankings=rankings,
            deep=deep,
            coverage=coverage,
            fields=fields,
        )

        self.assertEqual(forward_result, reverse_result)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["excluded_terms"], ["formal", "red"])
        self.assertEqual(
            forward["coverage"]["matched_excluded_terms_by_parent_asin"][
                "z-winner"
            ],
            ("formal", "red"),
        )

    def test_protects_top9_and_inserts_only_one_newcomer(self) -> None:
        rankings, _, _, _ = _fixture()
        result, diagnostics = _run()
        self.assertEqual(result[:9], rankings["final"][:9])
        self.assertEqual(result[10:], rankings["final"][9:])
        self.assertEqual(len(result), len(rankings["final"]) + 1)
        self.assertTrue(diagnostics["guard"]["top9_unchanged"])
        self.assertEqual(diagnostics["guard"]["newcomer_count"], 1)
        self.assertEqual(diagnostics["guard"]["newcomers"], ["z-winner"])

    def test_no_eligible_candidate_preserves_every_final_item_and_order(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        fields["z-winner"] = _fields("blue")
        fields["a-lower"] = _fields("blue")
        result, diagnostics = _run(
            rankings=rankings, deep=deep, coverage=coverage, fields=fields
        )
        self.assertEqual(result, rankings["final"])
        self.assertEqual(diagnostics["guard"]["reason"], "no_eligible_tail_candidate")
        self.assertEqual(diagnostics["guard"]["newcomer_count"], 0)

    def test_strict_margin_is_relative_to_rank10_coverage(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        fields["z-winner"] = _fields("blue")
        fields["a-lower"] = _fields("blue cotton")
        result, diagnostics = _run(
            rankings=rankings, deep=deep, coverage=coverage, fields=fields
        )
        self.assertEqual(result[9], "a-lower")
        rejected = next(
            item
            for item in diagnostics["tail"]["proposals"]
            if item["identifier"] == "z-winner"
        )
        self.assertIn("strict_coverage_margin_not_met", rejected["rejection_reasons"])
        self.assertEqual(diagnostics["guard"]["replacement_coverage"], 2)

    def test_default_deep_cutoff_ignores_items_after_rank240(self) -> None:
        rankings, deep, coverage, fields = _fixture()
        for identifier in deep[120:]:
            fields[identifier] = _fields("blue")
        beyond = "rank-241"
        deep.append(beyond)
        fields[beyond] = _fields("blue cotton shirt")
        result, diagnostics = _run(
            rankings=rankings, deep=deep, coverage=coverage, fields=fields
        )
        self.assertEqual(result, rankings["final"])
        self.assertEqual(diagnostics["tail"]["route_count"], 120)
        self.assertNotIn(
            beyond,
            [item["identifier"] for item in diagnostics["tail"]["proposals"]],
        )

    def test_diagnostics_are_json_serializable_and_label_free(self) -> None:
        _, diagnostics = _run()
        encoded = json.dumps(diagnostics, sort_keys=True)
        self.assertIn(SCHEMA_VERSION, encoded)
        self.assertTrue(diagnostics["label_free"])
        self.assertTrue(diagnostics["target_blind"])
        self.assertTrue(diagnostics["triggered"])
        self.assertTrue(diagnostics["active"])
        self.assertEqual(
            diagnostics["coverage"]["coverage_by_parent_asin"]["z-winner"],
            3,
        )
        for forbidden in (
            "ground_truth",
            "sample_id",
            "scenario",
            "intent_card",
            "evaluator",
            '"result"',
            "elapsed",
        ):
            self.assertNotIn(forbidden, encoded.lower())

    def test_visible_and_fts_query_counts_follow_the_base_expression(self) -> None:
        query = ["a", "blue", "blue", *[f"t{index:02d}" for index in range(60)]]
        rankings, deep, coverage, fields = _fixture()
        coverage[rankings["final"][9]] = 50
        fields["z-winner"] = _fields(" ".join(dict.fromkeys(query)))
        final, diagnostics = _run(
            query=query,
            rankings=rankings,
            deep=deep,
            coverage=coverage,
            fields=fields,
        )
        self.assertEqual(diagnostics["visible_query_term_count"], 62)
        self.assertEqual(diagnostics["fts_query_term_count"], 50)
        self.assertEqual(len(diagnostics["query_terms"]), 62)
        self.assertEqual(diagnostics["trigger"]["query_term_count"], 62)
        self.assertEqual(final[9], "z-winner")
        self.assertEqual(diagnostics["guard"]["replacement_coverage"], 62)

        _, short_diagnostics = _run(query=["a", "blue"])
        self.assertTrue(short_diagnostics["triggered"])
        self.assertEqual(short_diagnostics["visible_query_term_count"], 2)
        self.assertEqual(short_diagnostics["fts_query_term_count"], 1)

    def test_source_has_no_hidden_answer_or_evaluation_vocabulary(self) -> None:
        source = inspect.getsource(adaptive_depth).lower()
        self.assertIn('"target_blind": true', source)
        for forbidden in (
            "ground_truth",
            "sample_id",
            "scenario",
            "intent_card",
            "evaluator",
            "result",
            "elapsed",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
