from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections.abc import Callable

import numpy as np
import pytest

from scripts import analyze_grace_remaining_misses as attribution
from scripts import train_small_ranker as base


DEPTHS = (10, 20, 50, 100)


def _positive(*, turn: int | None = None, ordinal: int = -1) -> np.ndarray:
    values = np.full(10, -1, dtype=np.int16)
    if turn is not None:
        values[turn] = ordinal
    return values


def _feature_cube() -> np.ndarray:
    return np.zeros((10, 100, len(base.FEATURE_NAMES)), dtype=np.float32)


def _set_query_evidence(
    features: np.ndarray,
    *,
    start: int,
    query_specificity: float,
    active_constraints: float,
) -> None:
    features[
        start:, :, base.FEATURE_INDEX["query_specificity_fraction"]
    ] = query_specificity
    features[
        start:, :, base.FEATURE_INDEX["active_constraint_count_fraction"]
    ] = active_constraints


def _complete_product() -> dict[str, object]:
    return {
        "title": "Synthetic walking shoe",
        "categories": ["Shoes"],
        "features": ["Synthetic feature"],
        "details": {"material": "synthetic"},
        "description": "Synthetic description",
        "store": "Synthetic store",
        "price": 12.5,
    }


def _assert_privacy_rejected(payload: object) -> None:
    with pytest.raises(RuntimeError):
        attribution.privacy_scan(payload)


@pytest.mark.parametrize(
    ("ordinal", "expected"),
    [
        (-1, {10: False, 20: False, 50: False, 100: False}),
        (0, {10: True, 20: True, 50: True, 100: True}),
        (9, {10: True, 20: True, 50: True, 100: True}),
        (10, {10: False, 20: True, 50: True, 100: True}),
        (19, {10: False, 20: True, 50: True, 100: True}),
        (20, {10: False, 20: False, 50: True, 100: True}),
        (49, {10: False, 20: False, 50: True, 100: True}),
        (50, {10: False, 20: False, 50: False, 100: True}),
        (99, {10: False, 20: False, 50: False, 100: True}),
    ],
)
def test_candidate_recall_flags_use_zero_based_prefix_boundaries(
    ordinal: int, expected: dict[int, bool]
) -> None:
    assert attribution.candidate_recall_flags(
        _positive(turn=4, ordinal=ordinal), eligible_from=1
    ) == expected


def test_candidate_recall_ignores_pre_eligibility_occurrences() -> None:
    positive = _positive(turn=0, ordinal=0)
    assert attribution.candidate_recall_flags(positive, eligible_from=2) == {
        depth: False for depth in DEPTHS
    }


def test_primary_funnel_is_exclusive_and_uses_frozen_precedence() -> None:
    assert (
        attribution.classify_primary(
            c100_reachable=False,
            admission_rejected=True,
            state_rejected=True,
        )
        == "candidate_absent_at_c100"
    )
    assert (
        attribution.classify_primary(
            c100_reachable=True,
            admission_rejected=True,
            state_rejected=False,
        )
        == "admission_grace_state_rejection"
    )
    assert (
        attribution.classify_primary(
            c100_reachable=True,
            admission_rejected=False,
            state_rejected=True,
        )
        == "admission_grace_state_rejection"
    )
    assert (
        attribution.classify_primary(
            c100_reachable=True,
            admission_rejected=False,
            state_rejected=False,
        )
        == "candidate_present_but_ranker_failure"
    )


def test_admission_rejection_requires_an_available_action() -> None:
    positive = _positive(turn=3, ordinal=17)
    chosen = np.zeros(10, dtype=np.int16)
    chosen[3] = 17
    action = np.zeros(10, dtype=bool)
    activation = np.zeros(10, dtype=bool)

    assert (
        attribution.admission_rejected_flag(
            positive, 1, chosen, action, activation
        )
        is False
    )
    action[3] = True
    assert (
        attribution.admission_rejected_flag(
            positive, 1, chosen, action, activation
        )
        is True
    )
    activation[3] = True
    assert (
        attribution.admission_rejected_flag(
            positive, 1, chosen, action, activation
        )
        is False
    )


def test_largest_bottleneck_uses_the_separate_preregistered_tie_order() -> None:
    tied = {name: 4 for name in attribution.PRIMARY_ORDER}
    assert attribution.largest_primary_bottleneck(tied) == "candidate_absent_at_c100"
    tied["candidate_absent_at_c100"] = 0
    assert (
        attribution.largest_primary_bottleneck(tied)
        == "candidate_present_but_ranker_failure"
    )


def test_lifecycle_requires_pre_eligible_evidence_and_no_eligible_recall() -> None:
    pages = np.tile(np.arange(10, dtype=np.int16), (10, 1))
    positive = _positive(turn=0, ordinal=73)

    assert attribution.lifecycle_flag(positive, pages, eligible_from=3) is True

    no_old_evidence = _positive()
    assert (
        attribution.lifecycle_flag(no_old_evidence, pages, eligible_from=3) is False
    )

    positive[4] = 88
    assert attribution.lifecycle_flag(positive, pages, eligible_from=3) is False


def test_lifecycle_accepts_actual_pre_eligible_grace_page_evidence() -> None:
    pages = np.tile(np.arange(10, dtype=np.int16), (10, 1))
    positive = _positive(turn=1, ordinal=73)
    pages[1, 4] = 73
    assert attribution.lifecycle_flag(positive, pages, eligible_from=3) is True


def test_state_counterfactual_clears_served_without_regranting_grace() -> None:
    identifiers = [f"item-{index:03d}" for index in range(100)]
    turn = {
        "c100": tuple(identifiers),
        "actions": {"KEEP_P11": tuple(identifiers[:10])},
    }
    pages = attribution._eligibility_reset_grace_pages(
        ((turn, turn, turn, turn, turn),),
        np.zeros((1, 5), dtype=np.int16),
        np.zeros((1, 5), dtype=bool),
        np.ones((1, 5), dtype=np.int16),
        np.asarray([3], dtype=np.int16),
    )
    assert pages[0, 2].tolist() == list(range(10))
    assert pages[0, 3].tolist() == list(range(10, 20))


def test_information_query_zero_is_per_turn_or_and_not_vacuous_attribute() -> None:
    features = _feature_cube()
    eligible_index = 2
    _set_query_evidence(
        features,
        start=eligible_index,
        query_specificity=1.0,
        active_constraints=1.0,
    )
    query_index = base.FEATURE_INDEX["query_specificity_fraction"]
    active_index = base.FEATURE_INDEX["active_constraint_count_fraction"]
    for turn in range(eligible_index, 10):
        if turn % 2:
            features[turn, :, query_index] = 0.0
        else:
            features[turn, :, active_index] = 0.0

    flags = attribution.information_insufficient(
        features, _positive(), eligible_from=3
    )
    assert flags == {
        "query_zero": True,
        "attribute_missing": False,
        "combined": True,
    }

    features[5, :, query_index] = 1.0
    features[5, :, active_index] = 1.0
    flags = attribution.information_insufficient(
        features, _positive(), eligible_from=3
    )
    assert flags["query_zero"] is False
    assert flags["attribute_missing"] is False
    assert flags["combined"] is False


def test_information_attribute_missing_requires_every_reachable_target_row() -> None:
    features = _feature_cube()
    _set_query_evidence(
        features,
        start=2,
        query_specificity=1.0,
        active_constraints=1.0,
    )
    positive = _positive()
    positive[2] = 7
    positive[5] = 11
    missing_index = base.FEATURE_INDEX["missing_positive_evidence_fraction"]
    unknown_index = base.FEATURE_INDEX["category_unknown"]
    features[2, 7, missing_index] = 1.0
    features[5, 11, unknown_index] = 1.0

    flags = attribution.information_insufficient(features, positive, eligible_from=3)
    assert flags == {
        "query_zero": False,
        "attribute_missing": True,
        "combined": True,
    }

    features[5, 11, base.FEATURE_INDEX["category_compatible"]] = 1.0
    flags = attribution.information_insufficient(features, positive, eligible_from=3)
    assert flags["attribute_missing"] is False
    assert flags["combined"] is False


def test_information_attribute_missing_requires_positive_missing_or_unknown() -> None:
    features = _feature_cube()
    _set_query_evidence(
        features,
        start=0,
        query_specificity=1.0,
        active_constraints=1.0,
    )
    positive = _positive(turn=0, ordinal=4)
    flags = attribution.information_insufficient(features, positive, eligible_from=1)
    assert flags == {
        "query_zero": False,
        "attribute_missing": False,
        "combined": False,
    }


def test_catalog_missing_fields_for_complete_and_absent_rows() -> None:
    complete = attribution.catalog_missing_fields(_complete_product())
    assert complete == {
        "row_absent": False,
        "title_empty": False,
        "categories_empty": False,
        "descriptive_evidence_empty": False,
        "price_null": False,
        "any_missing": False,
    }

    absent = attribution.catalog_missing_fields(None)
    assert absent["row_absent"] is True
    assert absent["any_missing"] is True
    assert set(absent) == set(complete)


@pytest.mark.parametrize(
    ("mutate", "flag"),
    [
        (lambda row: row.update(title=""), "title_empty"),
        (lambda row: row.update(categories=[]), "categories_empty"),
        (lambda row: row.update(price=None), "price_null"),
    ],
)
def test_catalog_missing_fields_report_individual_conditions(
    mutate: Callable[[dict[str, object]], None], flag: str
) -> None:
    product = _complete_product()
    mutate(product)
    result = attribution.catalog_missing_fields(product)
    assert result[flag] is True
    assert result["any_missing"] is True


def test_catalog_missing_fields_require_all_descriptive_fields_empty() -> None:
    product = _complete_product()
    product.update(features=[], details={}, description="", store=None)
    result = attribution.catalog_missing_fields(product)
    assert result["descriptive_evidence_empty"] is True
    assert result["any_missing"] is True

    product["features"] = ["evidence"]
    assert (
        attribution.catalog_missing_fields(product)["descriptive_evidence_empty"]
        is False
    )


def test_privacy_scan_accepts_only_small_anonymous_aggregates() -> None:
    result = {
        "primary": {
            "candidate_absent_at_c100": {"count": 7, "fraction": 0.388889},
            "by_outer_fold": [1, 2, 1, 1, 2],
        },
        "candidate_frontier": attribution.candidate_frontier_pending(),
        "source_hashes": {"trace": "a" * 64},
    }
    assert attribution.privacy_scan(result) is None


def test_privacy_scan_does_not_confuse_reproduction_with_product_token() -> None:
    assert (
        attribution.privacy_scan(
            {"comparator_reproduction": {"fit_invocations": 60}}
        )
        is None
    )


def test_privacy_scan_accepts_hashed_frozen_access_schema() -> None:
    payload = {
        "access_audit": {
            "label_member_order_sha256": "a" * 64,
            "label_member_order_exact": True,
        },
        "grace_replay_identity": {
            "first": {"candidate_pages": {"raw_sha256": "a" * 64}}
        },
    }
    assert attribution.privacy_scan(payload) is None


@pytest.mark.parametrize("schema_name", ["parent_asin", "positive_index", "eligible_from"])
def test_privacy_scan_rejects_forbidden_schema_names_as_values(
    schema_name: str,
) -> None:
    _assert_privacy_rejected({"safe": schema_name})


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "session_id",
        "sample_id",
        "product_id",
        "target",
        "positive_index",
        "eligible_from",
        "per_session",
    ],
)
def test_privacy_scan_rejects_forbidden_keys_recursively(
    forbidden_key: str,
) -> None:
    _assert_privacy_rejected({"safe": {forbidden_key: 1}})


def test_privacy_scan_rejects_identifiers_arrays_and_session_vectors() -> None:
    _assert_privacy_rejected({"safe": "B0ABCDEFGH"})
    _assert_privacy_rejected({"safe": np.zeros(2, dtype=np.int8)})
    _assert_privacy_rejected({"safe": [0] * 2000})


def test_canonical_hash_is_exact_repeat_stable_and_content_sensitive() -> None:
    first = {"b": 2, "a": [1, 2]}
    reordered = {"a": [1, 2], "b": 2}
    expected = hashlib.sha256(b'{"a":[1,2],"b":2}').hexdigest()

    assert attribution._canonical_sha256(first) == expected
    assert attribution._canonical_sha256(reordered) == expected
    assert attribution._canonical_sha256({"a": [2, 1], "b": 2}) != expected
    assert re.fullmatch(r"[0-9a-f]{64}", expected)


def test_canonical_hash_rejects_non_json_nan() -> None:
    with pytest.raises(ValueError):
        attribution._canonical_sha256({"not_finite": float("nan")})


def test_candidate_frontier_remains_explicitly_pending_without_inference() -> None:
    frontier = attribution.candidate_frontier_pending()
    assert set(frontier) == {
        "present_by_c200",
        "present_by_c400",
        "absent_at_c400",
        "rule",
    }
    for name in ("present_by_c200", "present_by_c400", "absent_at_c400"):
        entry = frontier[name]
        assert entry["count"] is None
        assert entry["status"] == "pending_not_observed"
        assert "C100" in entry["reason"]
    assert "do not infer" in frontier["rule"].lower()


def test_receipt_does_not_create_missing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "missing" / "result.json"
    monkeypatch.setattr(attribution, "OUTPUT_PATH", output)
    monkeypatch.setattr(attribution, "ROOT", tmp_path)
    with pytest.raises(attribution.GraceAttributionError):
        attribution._open_receipt("f" * 40)
    assert not output.parent.exists()


def test_receipt_interrupt_is_durably_consumed_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ready" / "result.json"
    output.parent.mkdir()
    monkeypatch.setattr(attribution, "OUTPUT_PATH", output)
    monkeypatch.setattr(attribution, "ROOT", tmp_path)
    original = attribution._write_descriptor
    calls = 0

    def interrupt_once(descriptor: int, value: dict[str, object]) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original(descriptor, value)

    monkeypatch.setattr(attribution, "_write_descriptor", interrupt_once)
    with pytest.raises(attribution.GraceAttributionConsumedError):
        attribution._open_receipt("f" * 40)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "INVALID_ATTRIBUTION_CONSUMED"
    assert payload["error_class"] == "KeyboardInterrupt"
    assert payload["rerun_forbidden"] is True


def test_receipt_rejects_existing_path_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ready" / "result.json"
    output.parent.mkdir()
    output.write_bytes(b"already-consumed")
    monkeypatch.setattr(attribution, "OUTPUT_PATH", output)
    monkeypatch.setattr(attribution, "ROOT", tmp_path)
    with pytest.raises(attribution.GraceAttributionError):
        attribution._open_receipt("f" * 40)
    assert output.read_bytes() == b"already-consumed"


def test_receipt_rejects_reparse_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ready" / "result.json"
    output.parent.mkdir()
    monkeypatch.setattr(attribution, "OUTPUT_PATH", output)
    monkeypatch.setattr(attribution, "ROOT", tmp_path)
    monkeypatch.setattr(
        attribution,
        "_is_link_or_reparse",
        lambda path: path == output.parent,
    )
    with pytest.raises(attribution.GraceAttributionError):
        attribution._open_receipt("f" * 40)
    assert not output.exists()
