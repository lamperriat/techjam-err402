from __future__ import annotations

import inspect
import unittest

from starter.attributes import AttributeValue, ProductAttributeView
from starter.p8_negative import ExecutableNegative
from starter.p9_evidence import masks_from_view
from starter.p11_features import (
    NEAR_TIE_MAX_DELTA,
    TIE_WEIGHTS,
    WEIGHTS,
    CandidateEvidence,
    CandidateScore,
    FeatureBatch,
    PositiveConstraint,
    _hard_clause_coverage,
    _near_tie_groups,
    _order_with_near_tie_quality,
    rerank_top10_preserving_membership,
)


def attribute(value: str) -> AttributeValue:
    return AttributeValue(value, "details.value", 0.98, value)


def evidence(
    identifier: str,
    *,
    title_category: tuple[str, ...] = (),
    features_details: tuple[str, ...] = (),
    description_store: tuple[str, ...] = (),
    sequences: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] = ((), (), ()),
    observed: tuple[str, ...] = (),
    inferred: tuple[str, ...] = (),
    observed_subtypes: tuple[str, ...] = (),
    inferred_subtypes: tuple[str, ...] = (),
    view: ProductAttributeView | None = None,
    bayesian: float = 0.5,
    popularity: float = 0.5,
) -> CandidateEvidence:
    return CandidateEvidence(
        parent_asin=identifier,
        field_tokens=(
            frozenset(title_category),
            frozenset(features_details),
            frozenset(description_store),
        ),
        field_sequences=sequences,
        observed_values=frozenset(observed),
        inferred_values=frozenset(inferred),
        observed_subtypes=frozenset(observed_subtypes),
        inferred_subtypes=frozenset(inferred_subtypes),
        negative_masks=masks_from_view(view or ProductAttributeView(identifier)),
        bayesian_rating_percentile=bayesian,
        popularity_percentile=popularity,
    )


def equal_ranks(identifiers: list[str]) -> dict[str, int]:
    return {identifier: 1 for identifier in identifiers}


def score(
    relevance: float,
    tie_bonus: float = 0.0,
    conflict_state: str = "not_applicable",
) -> CandidateScore:
    return CandidateScore(
        total=round(relevance + tie_bonus, 12),
        relevance=round(relevance, 12),
        tie_bonus=round(tie_bonus, 12),
        conflict_state=conflict_state,
        **{name: 0.0 for name in WEIGHTS},
    )


class P11Top10ScorerTests(unittest.TestCase):
    def test_fixed_weights_are_bounded_and_frozen(self) -> None:
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(TIE_WEIGHTS.values()), 0.002)
        self.assertEqual(NEAR_TIE_MAX_DELTA, 0.002)
        self.assertEqual(
            WEIGHTS,
            {
                "broad_rank_prior": 0.03,
                "strict_rank_prior": 0.06,
                "rrf_rank_prior": 0.16,
                "idf_any_field_coverage": 0.24,
                "title_category_coverage": 0.11,
                "features_details_coverage": 0.08,
                "description_store_coverage": 0.03,
                "latest_hard_clause_coverage": 0.10,
                "subtype_consistency": 0.10,
                "positive_constraint_evidence": 0.09,
            },
        )

    def test_near_tie_groups_include_exact_boundary_without_chaining(self) -> None:
        identifiers = ("ANCHOR", "BOUNDARY", "OUTSIDE", "SECOND_BOUNDARY")
        breakdowns = {
            "ANCHOR": score(0.500000000000),
            "BOUNDARY": score(0.498000000000),
            "OUTSIDE": score(0.497999999999),
            "SECOND_BOUNDARY": score(0.495999999999),
        }

        self.assertEqual(
            _near_tie_groups(
                identifiers,
                breakdowns,
                {identifier: rank for rank, identifier in enumerate(identifiers, 1)},
            ),
            (("ANCHOR", "BOUNDARY"), ("OUTSIDE", "SECOND_BOUNDARY")),
        )

    def test_quality_bonus_only_reorders_inside_near_tie_group(self) -> None:
        identifiers = ("HIGH", "NEAR", "FAR")
        breakdowns = {
            "HIGH": score(0.500, 0.0),
            "NEAR": score(0.499, 0.002),
            "FAR": score(0.496, 0.002),
        }

        self.assertEqual(
            _order_with_near_tie_quality(
                identifiers,
                breakdowns,
                {identifier: rank for rank, identifier in enumerate(identifiers, 1)},
            ),
            ("NEAR", "HIGH", "FAR"),
        )

    def test_exact_final_score_ties_preserve_r08_order(self) -> None:
        identifiers = [
            "P-07", "P-02", "P-10", "P-01", "P-09",
            "P-03", "P-08", "P-04", "P-06", "P-05", "TAIL",
        ]
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(
                {identifier: evidence(identifier) for identifier in identifiers[:10]},
                {},
            ),
            query_terms=(),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
        )

        self.assertFalse(result.fallback)
        self.assertEqual(result.reason, "scored")
        self.assertEqual(len({item.total for item in result.breakdowns.values()}), 1)
        self.assertFalse(result.changed_top10_order)
        self.assertEqual(result.identifiers, tuple(identifiers))

    def test_idf_rerank_preserves_exact_top10_membership_and_tail(self) -> None:
        identifiers = [f"P-{index:02d}" for index in range(1, 12)]
        features = {identifier: evidence(identifier) for identifier in identifiers[:10]}
        features["P-05"] = evidence("P-05", title_category=("rare",))
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {"rare": 4.0}),
            query_terms=("rare",),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
        )

        self.assertFalse(result.fallback)
        self.assertEqual(result.identifiers[0], "P-05")
        self.assertEqual(set(result.identifiers[:10]), set(identifiers[:10]))
        self.assertEqual(result.identifiers[10:], tuple(identifiers[10:]))

    def test_p9_conflict_partition_is_monotonic_and_unknown_is_not_violation(self) -> None:
        identifiers = ["VIOLATION", "UNKNOWN", "COMPATIBLE"]
        features = {
            "VIOLATION": evidence(
                "VIOLATION",
                view=ProductAttributeView("VIOLATION", color=(attribute("red"),)),
            ),
            "UNKNOWN": evidence("UNKNOWN"),
            "COMPATIBLE": evidence(
                "COMPATIBLE",
                view=ProductAttributeView("COMPATIBLE", color=(attribute("blue"),)),
            ),
        }
        negative = ExecutableNegative("color", "red", 1, 1, 1)
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {}),
            query_terms=(),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
            negative_constraints=(negative,),
        )

        self.assertEqual(
            result.identifiers,
            ("COMPATIBLE", "UNKNOWN", "VIOLATION"),
        )
        self.assertEqual(result.breakdowns["UNKNOWN"].conflict_state, "unknown")
        self.assertEqual(
            result.breakdowns["VIOLATION"].conflict_state,
            "explicit_violation",
        )

    def test_positive_observed_inferred_unknown_and_recency_are_explainable(self) -> None:
        identifiers = ["UNKNOWN", "INFERRED", "OBSERVED"]
        features = {
            "UNKNOWN": evidence("UNKNOWN"),
            "INFERRED": evidence("INFERRED", inferred=("material=cotton",)),
            "OBSERVED": evidence("OBSERVED", observed=("material=cotton",)),
        }
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {}),
            query_terms=(),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
            positive_constraints=(
                PositiveConstraint("material", "cotton", "hard", 3, 2),
            ),
            current_turn=3,
            current_version=2,
        )

        self.assertEqual(result.identifiers, ("OBSERVED", "INFERRED", "UNKNOWN"))
        self.assertEqual(
            result.breakdowns["OBSERVED"].positive_constraint_evidence,
            1.0,
        )
        self.assertEqual(
            result.breakdowns["INFERRED"].positive_constraint_evidence,
            0.45,
        )
        self.assertEqual(
            result.breakdowns["UNKNOWN"].positive_constraint_evidence,
            0.0,
        )

    def test_hard_ngram_and_subtype_evidence_are_prepared_once(self) -> None:
        identifiers = ["BASE", "MATCH"]
        features = {
            "BASE": evidence("BASE"),
            "MATCH": evidence(
                "MATCH",
                sequences=(("lightweight waterproof running shoe",), (), ()),
                observed_subtypes=("running shoe",),
            ),
        }
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {}),
            query_terms=(),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
            hard_clause_terms=("lightweight", "waterproof", "running", "shoe"),
            query_subtypes=("running shoe",),
        )

        self.assertEqual(result.identifiers[0], "MATCH")
        self.assertEqual(
            result.breakdowns["MATCH"].latest_hard_clause_coverage,
            1.0,
        )
        self.assertEqual(result.breakdowns["MATCH"].subtype_consistency, 1.0)
        source = inspect.getsource(_hard_clause_coverage)
        self.assertNotIn("_terms(", source)
        self.assertNotIn("json.", source)
        self.assertNotIn("re.", source)

    def test_exact_full_clause_outranks_fragmented_local_ngrams(self) -> None:
        identifiers = ["FRAGMENTED", "EXACT"]
        features = {
            "FRAGMENTED": evidence(
                "FRAGMENTED",
                sequences=(
                    (
                        "lightweight waterproof running",
                        "waterproof running shoe",
                    ),
                    (),
                    (),
                ),
            ),
            "EXACT": evidence(
                "EXACT",
                sequences=(("lightweight waterproof running shoe",), (), ()),
            ),
        }
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {}),
            query_terms=(),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
            hard_clause_terms=("lightweight", "waterproof", "running", "shoe"),
        )

        self.assertEqual(result.identifiers[0], "EXACT")
        self.assertEqual(
            result.breakdowns["EXACT"].latest_hard_clause_coverage,
            1.0,
        )
        self.assertEqual(
            result.breakdowns["FRAGMENTED"].latest_hard_clause_coverage,
            0.5,
        )

    def test_maximum_tie_bonus_cannot_flip_a_material_relevance_gap(self) -> None:
        identifiers = ["QUALITY", "RELEVANT"]
        features = {
            "QUALITY": evidence("QUALITY", bayesian=1.0, popularity=1.0),
            "RELEVANT": evidence(
                "RELEVANT",
                description_store=("rare",),
                bayesian=0.0,
                popularity=0.0,
            ),
        }
        result = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch(features, {"rare": 3.0}),
            query_terms=("rare",),
            broad_ranks=equal_ranks(identifiers),
            strict_ranks=equal_ranks(identifiers),
            fused_ranks=equal_ranks(identifiers),
        )

        self.assertEqual(result.identifiers[0], "RELEVANT")
        self.assertAlmostEqual(result.breakdowns["QUALITY"].tie_bonus, 0.002)
        self.assertGreater(
            result.breakdowns["RELEVANT"].relevance
            - result.breakdowns["QUALITY"].relevance,
            0.002,
        )

    def test_missing_or_misbound_evidence_falls_back_exactly(self) -> None:
        identifiers = ["A", "B", "TAIL"]
        missing = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch({"A": evidence("A")}, {}),
            query_terms=(),
            broad_ranks={},
            strict_ranks={},
            fused_ranks={},
        )
        misbound = rerank_top10_preserving_membership(
            identifiers,
            FeatureBatch({
                "A": evidence("WRONG"),
                "B": evidence("B"),
                "TAIL": evidence("TAIL"),
            }, {}),
            query_terms=(),
            broad_ranks={},
            strict_ranks={},
            fused_ranks={},
        )

        for result in (missing, misbound):
            self.assertTrue(result.fallback)
            self.assertEqual(result.identifiers, tuple(identifiers))
            self.assertEqual(result.breakdowns, {})


if __name__ == "__main__":
    unittest.main()
