from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from scripts import build_small_ranker_cache as cache


def _evidence() -> cache.StaticEvidence:
    tokens = frozenset({"women", "blue", "cotton", "dress", "summer"})
    return cache.StaticEvidence(
        field_tokens=(tokens, frozenset({"cotton", "lightweight"}), frozenset({"summer"})),
        combined_tokens=tokens | {"lightweight"},
        observed_values=frozenset({"color=blue", "material=cotton", "category=dress"}),
        inferred_values=frozenset({"use_case=summer"}),
        observed_by_slot={
            "color": frozenset({"blue"}),
            "material": frozenset({"cotton"}),
            "category": frozenset({"dress"}),
        },
        inferred_by_slot={"use_case": frozenset({"summer"})},
        char3_bits=cache._hash_char_ngrams("women blue cotton dress summer", 3),
        char4_bits=cache._hash_char_ngrams("women blue cotton dress summer", 4),
        bigram_bits=cache._hash_ngrams(("women", "blue", "cotton", "dress", "summer"), 2),
        trigram_bits=cache._hash_ngrams(("women", "blue", "cotton", "dress", "summer"), 3),
        bayesian=0.7,
        popularity=0.4,
        price=39.0,
    )


def test_feature_schema_is_rich_unique_and_identity_free() -> None:
    assert len(cache.FEATURE_NAMES) >= 100
    assert len(cache.FEATURE_NAMES) == len(set(cache.FEATURE_NAMES))
    assert not any(
        forbidden in name.casefold()
        for name in cache.FEATURE_NAMES
        for forbidden in ("asin", "target", "sample_id", "user_id")
    )
    assert {"broad_presence", "strict_presence", "semantic_presence", "p11_presence"} <= set(cache.FEATURE_NAMES)
    assert {"turn_char3_overlap", "goal_bigram_coverage", "active_rare_term_coverage"} <= set(cache.FEATURE_NAMES)
    assert {"price_unknown", "material_conflict", "explicit_negative_violation"} <= set(cache.FEATURE_NAMES)


def test_feature_phase_api_has_no_proxy_or_label_input() -> None:
    parameters = inspect.signature(cache.build_target_blind_features).parameters
    assert "proxy_path" not in parameters
    assert "label_path" not in parameters
    assert set(parameters) == {
        "context_path",
        "catalog_path",
        "sidecar_path",
        "output_path",
        "phase_manifest_path",
    }


def test_rich_feature_row_is_finite_and_exact_width() -> None:
    identifier = "candidate"
    incumbent = "incumbent"
    route_maps = {
        route: {identifier: 2, incumbent: 10}
        for route in cache.RANK_ROUTES
    }
    route_top10 = {route: (identifier, incumbent) for route in cache.RANK_ROUTES}
    idf = {token: 2.0 for token in ("blue", "cotton", "dress", "summer")}
    views = tuple(cache._query_view("blue cotton summer dress", idf) for _ in cache.QUERY_VIEWS)
    context = {
        "query_terms": ["blue", "cotton", "summer", "dress"],
        "version": 1,
        "version_anchor_turn": 1,
        "override_count": 0,
        "current_turn_override": False,
        "active_records": [
            {"slot": "color", "value": "blue", "polarity": 1, "hardness": "soft"},
            {"slot": "material", "value": "cotton", "polarity": 1, "hardness": "hard"},
        ],
        "retired_records": [],
        "hard_clause_terms": ["cotton"],
        "budget_upper": 50.0,
    }
    row = cache.build_feature_row(
        identifier=identifier,
        turn=2,
        route_maps=route_maps,
        route_top10=route_top10,
        incumbent=incumbent,
        previous_ranks=({identifier: 3},),
        query_views=views,
        context=context,
        evidence=_evidence(),
        idf=idf,
        group_top10_jaccards=[1.0] * 5,
        vote_entropy=0.5,
    )
    assert row.dtype == np.float32
    assert row.shape == (len(cache.FEATURE_NAMES),)
    assert np.isfinite(row).all()
    assert row[cache.FEATURE_INDEX["material_observed"]] == 1.0
    assert row[cache.FEATURE_INDEX["price_compatible"]] == 1.0


def test_family_grouping_keeps_variants_in_one_fold() -> None:
    base = {
        "categories": ["Women", "Dresses"],
        "store": "Example Brand",
    }
    first = cache._family_signature({**base, "title": "Classic Blue Dress Size Small 2 Pack"})
    second = cache._family_signature({**base, "title": "Classic Red Dress Size Large 4 Pack"})
    other = cache._family_signature({**base, "title": "Linen Beach Tunic"})
    assert first == second
    family, outer, inner, count = cache._family_folds(
        [first, second, other], np.asarray([0, 1, 0], dtype=np.uint8), 5, 402
    )
    assert count == 2
    assert family[0] == family[1]
    assert outer[0] == outer[1]
    assert inner[0] == inner[1]


def test_hard_negative_selection_includes_positive_and_top10() -> None:
    candidates = tuple(f"c{index}" for index in range(100))
    features = np.zeros((100, len(cache.FEATURE_NAMES)), dtype=np.float32)
    for route in ("broad", "strict"):
        features[:, cache.FEATURE_INDEX[f"{route}_presence"]] = 1.0
        features[:, cache.FEATURE_INDEX[f"{route}_rank_fraction"]] = np.arange(1, 101) / 100.0
    turn = {
        "c100": candidates,
        "actions": {
            "KEEP_P11": candidates[:10],
            "CANDIDATE_RERANK": candidates[20:30],
            "FROZEN_SEMANTIC_RERANK": candidates[30:40],
        },
    }
    selected, length = cache._deterministic_training_indices(7, 73, turn, features)
    assert length == cache.TRAINING_ROWS
    assert 73 in selected
    assert set(range(10)) <= set(selected.tolist())
    assert selected.dtype == np.int16


def test_binary_identity_scan_detects_asin_shape(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"safe-prefix-B012345678-safe-suffix")
    assert cache._identity_shape_scan(path) == 1
