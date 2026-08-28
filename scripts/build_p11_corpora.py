from __future__ import annotations

"""Build fresh, frozen P11 corpora without opening any evaluation result.

The builder consumes only the catalog and nine conversation JSONL files whose targets
have already been opened by P1-P9.  It validates their canonical identities before
selecting any new target.  Raw outputs stay under the ignored ``experiments`` folder;
the tracked protocol contains the seeds, strata, identities, and expected output hashes.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.build_p8_selection_corpus import (  # noqa: E402
    NEGATIVE_VOCABULARIES,
    _eligible_plans,
    _file_sha256,
    _load_catalog,
    _load_jsonl,
    _materialize_sample,
    _samples_sha256,
    _stable_digest,
    _target_ids,
)


SCHEMA_VERSION = "p11.corpora.v1"
PROTOCOL_SCHEMA_VERSION = "p11.corpus-protocol.v1"
DEFAULT_PROTOCOL = Path("configs/p11_corpus_protocol.json")
DEFAULT_OUTPUT_DIR = Path("experiments")
DEFAULT_METADATA_FILENAME = "p11_corpora.metadata.json"

_SPACE_RE = re.compile(r"\s+")
_NEGATED_PREFIX_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\bfree\s+of\b)[^.;,:]{0,18}$",
    re.IGNORECASE,
)
_DESCRIPTION_PATTERNS = {
    (slot, value): re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
    for slot, values in NEGATIVE_VOCABULARIES.items()
    for value in values
}


class CorpusBuildError(ValueError):
    """Raised when a frozen input or deterministic output invariant fails."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _target(sample: Mapping[str, Any]) -> str:
    return str(sample.get("ground_truth", {}).get("parent_asin", "")).strip()


def _rating_number(product: Mapping[str, Any]) -> int:
    value = product.get("rating_number")
    if isinstance(value, bool):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric) or numeric < 0:
        return 0
    return int(numeric)


def popularity_percentiles(
    products: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Return catalog-only mid-rank percentiles with ties kept together."""

    if not products:
        raise CorpusBuildError("catalog is empty")
    frequencies = Counter(_rating_number(product) for product in products.values())
    midrank: dict[int, float] = {}
    lower = 0
    total = len(products)
    for value in sorted(frequencies):
        count = frequencies[value]
        midrank[value] = (lower + 0.5 * count) / total
        lower += count
    return {
        parent_asin: midrank[_rating_number(product)]
        for parent_asin, product in products.items()
    }


def _stable_take(
    identifiers: Iterable[str], count: int, seed: str, purpose: str
) -> list[str]:
    ordered = sorted(
        set(identifiers),
        key=lambda value: (_stable_digest(seed, purpose, value), value),
    )
    if len(ordered) < count:
        raise CorpusBuildError(
            f"{purpose} needs {count} targets but only {len(ordered)} are eligible"
        )
    return ordered[:count]


def _scenario_sequence(counts: Mapping[str, int], seed: str) -> list[str]:
    labelled = [
        (scenario, ordinal)
        for scenario, count in sorted(counts.items())
        for ordinal in range(int(count))
    ]
    labelled.sort(
        key=lambda item: _stable_digest(
            seed, "scenario", item[0], str(item[1])
        )
    )
    return [scenario for scenario, _ in labelled]


def _clean_title(product: Mapping[str, Any], limit: int = 180) -> str:
    return _SPACE_RE.sub(" ", str(product.get("title") or "product")).strip()[:limit]


def _profile(summary: str) -> dict[str, Any]:
    return {
        "purchase_frequency": "not provided",
        "average_prior_rating": None,
        "rating_style": "not provided",
        "preference_tags": [],
        "summary": summary,
    }


def _base_rows(
    identifiers: Sequence[str],
    scenario_counts: Mapping[str, int],
    prefix: str,
    category_bucket: str,
    seed: str,
) -> list[dict[str, Any]]:
    if sum(int(value) for value in scenario_counts.values()) != len(identifiers):
        raise CorpusBuildError(f"{prefix} scenario counts do not equal target count")
    ordered = sorted(
        identifiers,
        key=lambda value: (_stable_digest(seed, "row-order", value), value),
    )
    scenarios = _scenario_sequence(scenario_counts, seed)
    return [
        {
            "category_bucket": category_bucket,
            "difficulty_bucket": "unlabeled",
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"{prefix}{index:04d}",
            "scenario_type": scenario,
            "user_profile": _profile(
                "Neutral profile for a fresh catalog-derived P11 session."
            ),
        }
        for index, (parent_asin, scenario) in enumerate(
            zip(ordered, scenarios, strict=True), start=1
        )
    ]


def _bin_index(value: float, bins: Sequence[Mapping[str, Any]]) -> int | None:
    for index, spec in enumerate(bins):
        low = float(spec["low"])
        high = float(spec["high"])
        if low <= value < high or (index == len(bins) - 1 and value == high):
            return index
    return None


def _select_representative(
    available: set[str],
    percentiles: Mapping[str, float],
    bins: Sequence[Mapping[str, Any]],
    seed: str,
    purpose: str,
) -> tuple[list[str], list[int]]:
    selected: list[str] = []
    observed: list[int] = []
    for index, spec in enumerate(bins):
        candidates = [
            parent_asin
            for parent_asin in available
            if _bin_index(percentiles[parent_asin], bins) == index
        ]
        quota = int(spec["count"])
        chosen = _stable_take(candidates, quota, seed, f"{purpose}-bin-{index}")
        selected.extend(chosen)
        observed.append(len(chosen))
    if len(set(selected)) != len(selected):
        raise RuntimeError(f"{purpose} selected a duplicate target")
    return selected, observed


def _price(product: Mapping[str, Any]) -> float | None:
    value = product.get("price")
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _leaf_category(product: Mapping[str, Any]) -> str:
    categories = product.get("categories") or []
    return _SPACE_RE.sub(" ", str(categories[-1])).strip().casefold() if categories else ""


def _budget_plans(
    products: Mapping[str, Mapping[str, Any]],
    thresholds: Sequence[float],
    minimum_peer_count: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for parent_asin, product in products.items():
        price = _price(product)
        leaf = _leaf_category(product)
        if price is not None and leaf:
            grouped[leaf].append((parent_asin, price))

    plans: dict[str, dict[str, Any]] = {}
    for rows in grouped.values():
        prices = [price for _, price in rows]
        for parent_asin, target_price in rows:
            for threshold in thresholds:
                threshold = float(threshold)
                if target_price > threshold:
                    continue
                lower_peers = sum(price <= threshold for price in prices) - 1
                upper_peers = sum(price > threshold for price in prices)
                if lower_peers >= minimum_peer_count and upper_peers >= minimum_peer_count:
                    plans[parent_asin] = {
                        "target_price": target_price,
                        "threshold": threshold,
                        "lower_peer_count": lower_peers,
                        "upper_peer_count": upper_peers,
                    }
                    break
    return plans


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            for text in (str(key), *_flatten_text(item))
        ]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _flatten_text(item)]
    return [] if value in (None, "") else [str(value)]


def _positive_description_match(text: str, pattern: re.Pattern[str]) -> bool:
    return any(
        not _NEGATED_PREFIX_RE.search(text[max(0, match.start() - 28) : match.start()])
        for match in pattern.finditer(text)
    )


def _missing_evidence_plans(
    products: Mapping[str, Mapping[str, Any]], seed: str
) -> dict[str, dict[str, str]]:
    plans: dict[str, dict[str, str]] = {}
    for parent_asin, product in products.items():
        description = " ".join(_flatten_text(product.get("description"))).casefold()
        if not description:
            continue
        structured = " ".join(
            _flatten_text(
                {
                    field: product.get(field)
                    for field in ("title", "categories", "features", "details", "store")
                }
            )
        ).casefold()
        candidates: list[tuple[str, str]] = []
        for slot, values in NEGATIVE_VOCABULARIES.items():
            for value in values:
                pattern = _DESCRIPTION_PATTERNS[(slot, value)]
                if pattern.search(structured):
                    continue
                if _positive_description_match(description, pattern):
                    candidates.append((slot, value))
        if candidates:
            slot, value = min(
                candidates,
                key=lambda item: _stable_digest(
                    seed, parent_asin, "description-only", item[0], item[1]
                ),
            )
            plans[parent_asin] = {"slot": slot, "value": value}
    return plans


def _positive_failure_row(
    parent_asin: str,
    product: Mapping[str, Any],
    phrase: str,
    scenario: str,
    sample_id: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "category_bucket": "derived_p11_failure",
        "difficulty_bucket": "pre_materialized",
        "ground_truth": {"parent_asin": parent_asin},
        "sample_id": sample_id,
        "scenario_type": scenario,
        "user_profile": _profile("Neutral profile for a P11 failure-slice session."),
        "intent_card": {
            "target_category": _clean_title(product),
            "hard_constraints": [phrase],
            "soft_preferences": [],
        },
        "behavior": {"scenario_type": scenario, "p11_failure_audit": dict(audit)},
    }


def _materialize_positive_slice(
    identifiers: Sequence[str],
    products: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    scenario_counts: Mapping[str, int],
    seed: str,
    prefix: str,
    kind: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        identifiers,
        key=lambda value: (_stable_digest(seed, "row-order", value), value),
    )
    scenarios = _scenario_sequence(scenario_counts, seed)
    rows: list[dict[str, Any]] = []
    for index, (parent_asin, scenario) in enumerate(
        zip(ordered, scenarios, strict=True), start=1
    ):
        plan = plans[parent_asin]
        if kind == "budget":
            threshold = float(plan["threshold"])
            rendered = int(threshold) if threshold.is_integer() else threshold
            phrase = f"budget under ${rendered}"
            audit = {"kind": kind, **plan}
        elif kind == "missing_evidence":
            phrase = f"{plan['slot']}: {plan['value']}"
            audit = {
                "kind": kind,
                "slot": plan["slot"],
                "value": plan["value"],
                "evidence_source": "description_only",
                "structured_evidence_present": False,
            }
        else:
            raise ValueError(f"unsupported positive failure kind {kind}")
        rows.append(
            _positive_failure_row(
                parent_asin,
                products[parent_asin],
                phrase,
                scenario,
                f"{prefix}{index:04d}",
                audit,
            )
        )
    return rows


def _materialize_override_slice(
    identifiers: Sequence[str],
    products: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    seed: str,
    prefix: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        identifiers,
        key=lambda value: (_stable_digest(seed, "row-order", value), value),
    )
    rows: list[dict[str, Any]] = []
    for index, parent_asin in enumerate(ordered, start=1):
        plan = plans[parent_asin]
        old_value = f"{plan['slot']}: {plan['negative_value']}"
        new_value = f"{plan['slot']}: {plan['positive_value']}"
        turn = 3 + int(_stable_digest(seed, parent_asin, "override-turn"), 16) % 2
        rows.append(
            {
                "category_bucket": "derived_p11_failure",
                "difficulty_bucket": "pre_materialized",
                "ground_truth": {"parent_asin": parent_asin},
                "sample_id": f"{prefix}{index:04d}",
                "scenario_type": "intent_override",
                "user_profile": _profile(
                    "Neutral profile for a P11 override failure session."
                ),
                "intent_card": {
                    "target_category": _clean_title(products[parent_asin]),
                    "hard_constraints": [new_value],
                    "soft_preferences": [],
                },
                "behavior": {
                    "scenario_type": "intent_override",
                    "override": {
                        "turn": turn,
                        "old_value": old_value,
                        "new_value": new_value,
                        "message": (
                            "Actually, ignore my earlier preference. What I need is: "
                            f"{new_value}."
                        ),
                    },
                    "p11_failure_audit": {
                        "kind": "override",
                        "slot": plan["slot"],
                        "old_value": plan["negative_value"],
                        "new_value": plan["positive_value"],
                        "new_evidence_source": plan["positive_evidence_source"],
                    },
                },
            }
        )
    return rows


def _inspect_opened(
    opened_rows: Mapping[str, list[dict[str, Any]]],
    products: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
    expected_union_count: int,
) -> tuple[set[str], dict[str, Any]]:
    if set(opened_rows) != set(specs):
        raise CorpusBuildError("opened corpus names do not match protocol registry")
    target_sets: dict[str, set[str]] = {}
    observations: dict[str, Any] = {}
    for name, spec in specs.items():
        rows = opened_rows[name]
        canonical_hash = _samples_sha256(rows)
        expected_hash = str(spec["canonical_samples_sha256"]).lower()
        if canonical_hash != expected_hash:
            raise CorpusBuildError(f"{name} canonical sample SHA-256 mismatch")
        expected_rows = int(spec["rows"])
        prefix = str(spec["sample_id_prefix"])
        sample_ids = [str(row.get("sample_id", "")) for row in rows]
        targets = _target_ids(rows)
        if len(rows) != expected_rows or len(targets) != expected_rows:
            raise CorpusBuildError(f"{name} row/unique-target count mismatch")
        if len(set(sample_ids)) != expected_rows or any(
            not sample_id.startswith(prefix) for sample_id in sample_ids
        ):
            raise CorpusBuildError(f"{name} sample IDs are invalid or duplicated")
        missing = targets - set(products)
        if missing:
            raise CorpusBuildError(f"{name} has {len(missing)} targets outside catalog")
        target_sets[name] = targets
        observations[name] = {
            "rows": len(rows),
            "unique_targets": len(targets),
            "canonical_samples_sha256": canonical_hash,
        }

    overlaps = {
        f"{left}__{right}": len(target_sets[left] & target_sets[right])
        for left, right in combinations(sorted(target_sets), 2)
    }
    nonzero = {name: value for name, value in overlaps.items() if value}
    if nonzero:
        raise CorpusBuildError(f"opened target registry overlaps: {nonzero}")
    union = set().union(*target_sets.values())
    if len(union) != expected_union_count:
        raise CorpusBuildError(
            f"opened target union {len(union)} != expected {expected_union_count}"
        )
    return union, {
        "corpora": observations,
        "pairwise_target_overlaps": overlaps,
        "target_union_count": len(union),
    }


def _split_summary(
    rows: Sequence[Mapping[str, Any]],
    percentiles: Mapping[str, float],
    bins: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    targets = [_target(row) for row in rows]
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "unique_target_count": len(set(targets)),
        "scenario_counts": dict(sorted(Counter(str(row["scenario_type"]) for row in rows).items())),
        "samples_sha256": _sha256_bytes(_canonical_jsonl_bytes(rows)),
    }
    values = [percentiles[target] for target in targets]
    if values:
        ordered = sorted(values)
        summary["popularity_percentile"] = {
            "minimum": round(ordered[0], 9),
            "median": round((ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2, 9),
            "mean": round(sum(ordered) / len(ordered), 9),
            "maximum": round(ordered[-1], 9),
        }
    if bins is not None:
        counts = [0] * len(bins)
        for value in values:
            index = _bin_index(value, bins)
            if index is None:
                raise RuntimeError("selected representative target is outside frozen bins")
            counts[index] += 1
        summary["popularity_bin_counts"] = counts
    return summary


def build_p11_corpora(
    opened_rows: Mapping[str, list[dict[str, Any]]],
    products: Mapping[str, dict[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build all seven corpora in memory and return aggregate-only metadata."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise CorpusBuildError("unsupported P11 corpus protocol schema")
    expected_catalog_count = int(protocol["catalog"]["count"])
    if len(products) != expected_catalog_count:
        raise CorpusBuildError(
            f"catalog product count {len(products)} != expected {expected_catalog_count}"
        )
    opened_targets, opened_audit = _inspect_opened(
        opened_rows,
        products,
        protocol["opened_corpora"],
        int(protocol["opened_target_union_count"]),
    )
    eligible = {
        parent_asin
        for parent_asin, product in products.items()
        if parent_asin not in opened_targets
        and str(product.get("title") or "").strip()
        and product.get("categories")
    }
    percentiles = popularity_percentiles(products)
    split_specs = protocol["splits"]
    representative_bins = protocol["representative_popularity_bins"]

    primary_ids, primary_bins = _select_representative(
        eligible,
        percentiles,
        representative_bins,
        str(split_specs["primary"]["seed"]),
        "primary",
    )
    eligible -= set(primary_ids)
    confirmation_ids, confirmation_bins = _select_representative(
        eligible,
        percentiles,
        representative_bins,
        str(split_specs["confirmation"]["seed"]),
        "confirmation",
    )
    eligible -= set(confirmation_ids)

    uniform_spec = split_specs["uniform_tail"]
    tail_cutoff = float(uniform_spec["maximum_popularity_percentile_exclusive"])
    uniform_ids = _stable_take(
        (value for value in eligible if percentiles[value] < tail_cutoff),
        int(uniform_spec["count"]),
        str(uniform_spec["seed"]),
        "uniform-tail",
    )
    eligible -= set(uniform_ids)

    primary = _base_rows(
        primary_ids,
        split_specs["primary"]["scenario_counts"],
        str(split_specs["primary"]["sample_id_prefix"]),
        "derived_representative",
        str(split_specs["primary"]["seed"]),
    )
    confirmation = _base_rows(
        confirmation_ids,
        split_specs["confirmation"]["scenario_counts"],
        str(split_specs["confirmation"]["sample_id_prefix"]),
        "derived_representative_confirmation",
        str(split_specs["confirmation"]["seed"]),
    )
    uniform_tail = _base_rows(
        uniform_ids,
        uniform_spec["scenario_counts"],
        str(uniform_spec["sample_id_prefix"]),
        "derived_uniform_tail",
        str(uniform_spec["seed"]),
    )

    negative_plans = _eligible_plans(products, str(protocol["failure_plan_seed"]))
    negative_spec = split_specs["failure_negative"]
    negative_ids = _stable_take(
        set(negative_plans) & eligible,
        int(negative_spec["count"]),
        str(negative_spec["seed"]),
        "failure-negative",
    )
    eligible -= set(negative_ids)
    negative_ordered = sorted(
        negative_ids,
        key=lambda value: (
            _stable_digest(str(negative_spec["seed"]), "row-order", value),
            value,
        ),
    )
    negative_scenarios = _scenario_sequence(
        negative_spec["scenario_counts"], str(negative_spec["seed"])
    )
    failure_negative = [
        _materialize_sample(
            parent_asin,
            products[parent_asin],
            negative_plans[parent_asin],
            scenario,
            f"{negative_spec['sample_id_prefix']}{index:04d}",
            str(negative_spec["seed"]),
        )
        for index, (parent_asin, scenario) in enumerate(
            zip(negative_ordered, negative_scenarios, strict=True), start=1
        )
    ]

    budget_spec = split_specs["failure_budget"]
    budget_plans = _budget_plans(
        products,
        protocol["budget_thresholds"],
        int(protocol["budget_minimum_peer_count_per_side"]),
    )
    budget_ids = _stable_take(
        set(budget_plans) & eligible,
        int(budget_spec["count"]),
        str(budget_spec["seed"]),
        "failure-budget",
    )
    eligible -= set(budget_ids)
    failure_budget = _materialize_positive_slice(
        budget_ids,
        products,
        budget_plans,
        budget_spec["scenario_counts"],
        str(budget_spec["seed"]),
        str(budget_spec["sample_id_prefix"]),
        "budget",
    )

    override_spec = split_specs["failure_override"]
    override_ids = _stable_take(
        set(negative_plans) & eligible,
        int(override_spec["count"]),
        str(override_spec["seed"]),
        "failure-override",
    )
    eligible -= set(override_ids)
    failure_override = _materialize_override_slice(
        override_ids,
        products,
        negative_plans,
        str(override_spec["seed"]),
        str(override_spec["sample_id_prefix"]),
    )

    missing_spec = split_specs["failure_missing_evidence"]
    missing_plans = _missing_evidence_plans(
        products, str(missing_spec["seed"])
    )
    missing_ids = _stable_take(
        set(missing_plans) & eligible,
        int(missing_spec["count"]),
        str(missing_spec["seed"]),
        "failure-missing-evidence",
    )
    failure_missing = _materialize_positive_slice(
        missing_ids,
        products,
        missing_plans,
        missing_spec["scenario_counts"],
        str(missing_spec["seed"]),
        str(missing_spec["sample_id_prefix"]),
        "missing_evidence",
    )

    corpora = {
        "primary": primary,
        "uniform_tail": uniform_tail,
        "confirmation": confirmation,
        "failure_negative": failure_negative,
        "failure_budget": failure_budget,
        "failure_override": failure_override,
        "failure_missing_evidence": failure_missing,
    }
    target_sets = {name: _target_ids(rows) for name, rows in corpora.items()}
    for name, rows in corpora.items():
        expected_count = int(split_specs[name]["count"])
        if len(rows) != expected_count or len(target_sets[name]) != expected_count:
            raise RuntimeError(f"{name} output row/unique-target invariant failed")
        if target_sets[name] & opened_targets:
            raise RuntimeError(f"{name} overlaps an opened P1-P9 target")
    new_overlaps = {
        f"{left}__{right}": len(target_sets[left] & target_sets[right])
        for left, right in combinations(sorted(target_sets), 2)
    }
    opened_vs_new_overlaps = {
        name: len(targets & opened_targets)
        for name, targets in sorted(target_sets.items())
    }
    if any(new_overlaps.values()):
        raise RuntimeError("P11 output corpora are not mutually target-disjoint")
    if any(opened_vs_new_overlaps.values()):
        raise RuntimeError("P11 output corpora overlap the opened target registry")

    summaries = {
        name: _split_summary(
            rows,
            percentiles,
            representative_bins if name in {"primary", "confirmation"} else None,
        )
        for name, rows in corpora.items()
    }
    if summaries["primary"]["popularity_bin_counts"] != primary_bins:
        raise RuntimeError("primary popularity-bin audit mismatch")
    if summaries["confirmation"]["popularity_bin_counts"] != confirmation_bins:
        raise RuntimeError("confirmation popularity-bin audit mismatch")
    for name, summary in summaries.items():
        expected_hash = split_specs[name].get("expected_samples_sha256")
        if expected_hash and summary["samples_sha256"] != str(expected_hash).lower():
            raise CorpusBuildError(f"{name} frozen output SHA-256 mismatch")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_sha256": _sha256_bytes(_canonical_json_bytes(protocol)),
        "catalog": {
            "product_count": len(products),
            "sha256": str(protocol["catalog"]["sha256"]).lower(),
        },
        "opened_registry": opened_audit,
        "opened_vs_new_target_overlaps": opened_vs_new_overlaps,
        "outputs": summaries,
        "new_pairwise_target_overlaps": new_overlaps,
        "new_target_union_count": len(set().union(*target_sets.values())),
        "representative_popularity_bins": representative_bins,
        "selection_boundaries": {
            "primary_role": "selection and weight choice",
            "uniform_tail_role": "long-tail non-regression only",
            "confirmation_role": "unopened until candidate and weights are frozen",
            "failure_slice_role": "separate mechanism diagnostics; never a primary proxy",
            "released_public_used_for_weight_search": False,
            "evaluation_result_json_read": False,
            "agent_used": False,
            "fts_used": False,
        },
        "failure_semantics": {
            "negative": "catalog-supported explicit conflict from the frozen P8 planner",
            "budget": "target price is under a fixed threshold with peers on both sides",
            "override": "old catalog-supported conflict is replaced by target-observed evidence",
            "missing_evidence": "positive value occurs in description but not structured fields",
        },
    }
    return corpora, metadata


def _exclusive_publish(payloads: Mapping[Path, bytes]) -> None:
    """Publish several files without ever replacing an existing destination."""

    destinations = [path.resolve() for path in payloads]
    if len(destinations) != len(set(destinations)):
        raise ValueError("P11 output paths must be unique")
    existing = [path for path in payloads if path.exists()]
    if existing:
        raise FileExistsError(f"P11 output already exists: {existing[0]}")

    temporary: dict[Path, Path] = {}
    created: list[Path] = []
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temp = Path(name)
            temporary[destination] = temp
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temp in temporary.items():
            os.link(temp, destination)
            created.append(destination)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CorpusBuildError("P11 protocol must be a JSON object")
    return value


def build_and_write_p11_corpora(
    project_root: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    split_specs = protocol["splits"]
    output_paths = {
        name: output_dir / str(spec["filename"])
        for name, spec in split_specs.items()
    }
    metadata_path = output_dir / str(
        protocol.get("metadata_filename", DEFAULT_METADATA_FILENAME)
    )
    all_outputs = [*output_paths.values(), metadata_path]
    if len({path.resolve() for path in all_outputs}) != len(all_outputs):
        raise ValueError("P11 corpus and metadata output paths must differ")
    existing = [path for path in all_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"P11 output already exists: {existing[0]}")

    catalog_path = project_root / str(protocol["catalog"]["path"])
    catalog_hash = _file_sha256(catalog_path)
    if catalog_hash != str(protocol["catalog"]["sha256"]).lower():
        raise CorpusBuildError("catalog frozen SHA-256 mismatch")
    opened_paths = {
        name: project_root / str(spec["path"])
        for name, spec in protocol["opened_corpora"].items()
    }
    input_paths = {catalog_path.resolve(), *(path.resolve() for path in opened_paths.values())}
    if input_paths & {path.resolve() for path in all_outputs}:
        raise ValueError("P11 outputs must not overwrite an input")

    products = _load_catalog(catalog_path)
    opened_rows = {name: _load_jsonl(path) for name, path in opened_paths.items()}
    corpora, metadata = build_p11_corpora(opened_rows, products, protocol)
    metadata["catalog"]["path"] = str(catalog_path)
    metadata["protocol_path"] = str(protocol_path)
    metadata["protocol_file_sha256"] = _file_sha256(protocol_path)
    metadata["builder_source"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _file_sha256(Path(__file__).resolve()),
    }
    metadata["input_files"] = {
        name: {
            "path": str(opened_paths[name]),
            "file_sha256": _file_sha256(opened_paths[name]),
            **metadata["opened_registry"]["corpora"][name],
        }
        for name in sorted(opened_paths)
    }
    metadata["output_files"] = {
        name: {
            "path": str(output_paths[name]),
            "sha256": metadata["outputs"][name]["samples_sha256"],
        }
        for name in output_paths
    }
    payloads = {
        output_paths[name]: _canonical_jsonl_bytes(rows)
        for name, rows in corpora.items()
    }
    payloads[metadata_path] = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _exclusive_publish(payloads)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = project_root / protocol_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    metadata = build_and_write_p11_corpora(project_root, protocol_path, output_dir)
    print(
        "[p11-corpora] "
        + " ".join(
            f"{name}={summary['sample_count']}:{summary['samples_sha256']}"
            for name, summary in metadata["outputs"].items()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
