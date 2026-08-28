from __future__ import annotations

"""Build two frozen, target-disjoint P8 explicit-negative corpora."""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_official_assets import git_blob_sha1  # noqa: E402
from starter.attributes import (  # noqa: E402
    build_product_attribute_view,
    normalize_value,
)
from starter.p8_negative import (  # noqa: E402
    ALLOWED_NEGATIVE_SLOTS,
    MIN_EVIDENCE_CONFIDENCE,
)


SCHEMA_VERSION = "p8.explicit-negative-corpora.v1"
DEFAULT_COUNT = 200
DEFAULT_SELECTION_SEED = "track4-p8-explicit-negative-selection-v1"
DEFAULT_CONFIRMATION_SEED = "track4-p8-explicit-negative-confirmation-v1"
DEFAULT_SELECTION_OUTPUT = Path(
    "experiments/p8_selection_product_disjoint.jsonl"
)
DEFAULT_CONFIRMATION_OUTPUT = Path(
    "experiments/p8_confirmation_product_disjoint.jsonl"
)
DEFAULT_METADATA_OUTPUT = Path(
    "experiments/p8_explicit_negative_corpora.metadata.json"
)
DEFAULT_P1_PATH = Path("experiments/p1_derived_product_disjoint.jsonl")
DEFAULT_P5_PATH = Path("experiments/p5_selection_product_disjoint.jsonl")
DEFAULT_P6_PATH = Path("experiments/p6_selection_product_disjoint.jsonl")
DEFAULT_P7_PATH = Path("experiments/p7_selection_product_disjoint.jsonl")

OFFICIAL_CATALOG_COUNT = 50_000
OFFICIAL_PUBLIC_COUNT = 200
PRIOR_DERIVED_COUNT = 200
CATALOG_FROZEN_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
PUBLIC_FROZEN_GIT_BLOB_SHA1 = "121dbec9c1368c81cd887d6959e62507512139c0"
PUBLIC_FROZEN_SAMPLES_SHA256 = (
    "6c726257fec25575716ee65b095f94c48402b6e14e83341518610f45fbfbec6d"
)
P1_FROZEN_SAMPLES_SHA256 = (
    "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
)
P5_FROZEN_SAMPLES_SHA256 = (
    "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c"
)
P6_FROZEN_SAMPLES_SHA256 = (
    "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7"
)
P7_FROZEN_SAMPLES_SHA256 = (
    "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546"
)
P8_SELECTION_FROZEN_SAMPLES_SHA256 = (
    "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee"
)
P8_CONFIRMATION_FROZEN_SAMPLES_SHA256 = (
    "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde"
)

SELECTION_SAMPLE_ID_PREFIX = "derived_p8_selection_"
CONFIRMATION_SAMPLE_ID_PREFIX = "derived_p8_confirmation_"
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}
SCENARIO_WEIGHTS = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)
NEGATIVE_TEMPLATES = (
    "not {value}",
    "without {value}",
    "do not want {value}",
)
MIN_NEGATIVE_SUPPORT = 3
BUCKET_FALLBACK_ORDER = ("leaf", "coarse")

# Every value is a single-token canonical value recognized by the current
# deterministic constraint parser. Product evidence is taken only from fields
# consumed by starter.attributes; that module deliberately ignores description.
NEGATIVE_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "material": (
        "canvas", "cotton", "denim", "fleece", "leather", "linen", "mesh",
        "nylon", "polyester", "rayon", "rubber", "silk", "spandex", "suede",
        "wool",
    ),
    "color": (
        "beige", "black", "blue", "brown", "gold", "gray", "green", "khaki",
        "navy", "orange", "pink", "purple", "red", "silver", "tan", "white",
        "yellow",
    ),
    "style": (
        "athletic", "casual", "classic", "elegant", "formal", "modern",
        "oversized", "relaxed", "slim", "sporty", "vintage",
    ),
    "use_case": (
        "beach", "cycling", "gym", "hiking", "office", "outdoor", "rain",
        "running", "school", "snow", "travel", "walking", "wedding", "winter",
        "work", "workout",
    ),
    "audience": ("baby", "boys", "girls", "kids", "men", "unisex", "women"),
    "closure": ("buckle", "button", "clasp", "drawstring", "snap", "zipper"),
}
EVIDENCE_SOURCE_RULE = (
    "runtime-aligned confidence >= 0.90 evidence from categories, title, features, "
    "store, or structured details; never description"
)
_ALLOWED_DIRECT_SOURCES = {"categories", "title", "features", "store"}
_SPACE_RE = re.compile(r"\s+")

_INPUT_SPECS = (
    ("released_public", "released public", "public_"),
    ("prior_p1_derived", "prior P1-derived", "derived_p1_"),
    ("prior_p5_derived", "prior P5-derived", "derived_p5_"),
    ("prior_p6_derived", "prior P6-derived", "derived_p6_"),
    ("prior_p7_derived", "prior P7-derived", "derived_p7_"),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} row {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        parent_asin = str(row.get("parent_asin") or "").strip()
        if not parent_asin:
            raise ValueError("catalog contains an empty parent_asin")
        if parent_asin in products:
            raise ValueError(f"catalog contains duplicate parent_asin {parent_asin}")
        products[parent_asin] = row
    return products


def _target_ids(samples: list[dict[str, Any]]) -> set[str]:
    return {
        target
        for sample in samples
        if (
            target := str(
                sample.get("ground_truth", {}).get("parent_asin", "")
            ).strip()
        )
    }


def _canonical_jsonl_bytes(samples: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            sample,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for sample in samples
    ).encode("utf-8")


def _samples_sha256(samples: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_jsonl_bytes(samples)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode("utf-8")).hexdigest()


def _stable_choice(values: list[Any] | tuple[Any, ...], seed: str, *parts: str) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    index = int(_stable_digest(seed, *parts), 16) % len(values)
    return values[index]


def _input_rows(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "released_public": public_samples,
        "prior_p1_derived": p1_samples,
        "prior_p5_derived": p5_samples,
        "prior_p6_derived": p6_samples,
        "prior_p7_derived": p7_samples,
    }


def _input_target_sets(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
) -> dict[str, set[str]]:
    return {
        name: _target_ids(samples)
        for name, samples in _input_rows(
            public_samples, p1_samples, p5_samples, p6_samples, p7_samples
        ).items()
    }


def _pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _input_invariant_failures(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
) -> list[str]:
    rows = _input_rows(
        public_samples, p1_samples, p5_samples, p6_samples, p7_samples
    )
    targets = {
        name: _target_ids(samples) for name, samples in rows.items()
    }
    failures: list[str] = []
    for key, label, prefix in _INPUT_SPECS:
        samples = rows[key]
        sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
        if len(targets[key]) != len(samples):
            failures.append(f"{label} targets are missing or duplicated")
        if len(set(sample_ids)) != len(sample_ids) or any(
            not sample_id.startswith(prefix) for sample_id in sample_ids
        ):
            failures.append(f"{label} corpus has invalid or duplicate sample IDs")

    for (left_key, left_label, _), (right_key, right_label, _) in combinations(
        _INPUT_SPECS, 2
    ):
        if targets[left_key] & targets[right_key]:
            failures.append(f"{left_label}/{right_label} input targets overlap")
    return failures


def _scenario_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("P8 corpus count must be positive")
    raw = [(name, total * weight) for name, weight in SCENARIO_WEIGHTS]
    counts = {name: int(value) for name, value in raw}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda item: (-(item[1] - int(item[1])), item[0]))
    for name, _ in order[:remainder]:
        counts[name] += 1
    return counts


def _scenario_sequence(count: int, seed: str) -> list[str]:
    counts = _scenario_counts(count)
    labelled = [
        (scenario, ordinal)
        for scenario, _ in SCENARIO_WEIGHTS
        for ordinal in range(counts[scenario])
    ]
    labelled.sort(
        key=lambda item: _stable_digest(
            seed, "scenario", item[0], str(item[1])
        )
    )
    return [scenario for scenario, _ in labelled]


def _source_allowed(source: str) -> bool:
    return source in _ALLOWED_DIRECT_SOURCES or source.startswith("details.")


def _reliable_category_buckets(
    product: Mapping[str, Any], view: Any
) -> tuple[tuple[str, str], ...]:
    raw_positions = {
        normalize_value(raw): index
        for index, raw in enumerate(product.get("categories") or [])
    }
    category_items = sorted(
        (item for item in view.category if item.source == "categories"),
        key=lambda item: (
            raw_positions.get(normalize_value(item.raw), len(raw_positions)),
            item.value,
        ),
    )
    ordered_values = list(dict.fromkeys(item.value for item in category_items))
    if not ordered_values:
        return ()
    return (
        ("leaf", ordered_values[-1]),
        ("coarse", ordered_values[0]),
    )


def _product_evidence(product: Mapping[str, Any]) -> dict[str, Any] | None:
    if not str(product.get("title") or "").strip() or not product.get("categories"):
        return None
    view = build_product_attribute_view(product)
    buckets = _reliable_category_buckets(product, view)
    if not buckets:
        return None
    slots: dict[str, dict[str, str]] = {}
    for slot, vocabulary in NEGATIVE_VOCABULARIES.items():
        values = {
            item.value: item.source
            for item in getattr(view, slot)
            if item.value in vocabulary
            and " " not in item.value
            and _source_allowed(item.source)
            and item.confidence >= MIN_EVIDENCE_CONFIDENCE
        }
        if values:
            slots[slot] = values
    if not slots:
        return None
    return {"buckets": buckets, "slots": slots}


def _bucket_document_frequencies(
    evidence_by_id: Mapping[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Counter[str]]]:
    frequencies: dict[tuple[str, str], dict[str, Counter[str]]] = {}
    for evidence in evidence_by_id.values():
        for bucket in evidence["buckets"]:
            slot_frequencies = frequencies.setdefault(bucket, {})
            for slot, values in evidence["slots"].items():
                slot_frequencies.setdefault(slot, Counter()).update(values.keys())
    return frequencies


def _constraint_plan(
    evidence: Mapping[str, Any],
    bucket_frequencies: Mapping[tuple[str, str], Mapping[str, Counter[str]]],
    seed: str,
    parent_asin: str,
    minimum_support: int = MIN_NEGATIVE_SUPPORT,
) -> dict[str, Any] | None:
    if minimum_support <= 0:
        raise ValueError("negative minimum support must be positive")
    chosen: tuple[str, str, str, str, int, str] | None = None
    chosen_level = ""
    for level, bucket_value in evidence["buckets"]:
        candidates: list[tuple[str, str, str, str, int, str]] = []
        slot_frequencies = bucket_frequencies.get((level, bucket_value), {})
        for slot, positive_values in evidence["slots"].items():
            target_values = set(positive_values)
            document_frequencies = slot_frequencies.get(slot, Counter())
            for negative_value in NEGATIVE_VOCABULARIES[slot]:
                support = int(document_frequencies.get(negative_value, 0))
                if negative_value in target_values or support < minimum_support:
                    continue
                for positive_value, source in sorted(positive_values.items()):
                    candidates.append(
                        (
                            slot,
                            positive_value,
                            source,
                            negative_value,
                            support,
                            bucket_value,
                        )
                    )
        if candidates:
            chosen = min(
                candidates,
                key=lambda item: (
                    -item[4],
                    _stable_digest(
                        seed,
                        parent_asin,
                        "negative-tie",
                        item[0],
                        item[1],
                        item[3],
                    ),
                ),
            )
            chosen_level = level
            break
    if chosen is None:
        return None

    (
        slot,
        positive_value,
        evidence_source,
        negative_value,
        negative_support,
        bucket_value,
    ) = chosen
    template = _stable_choice(
        NEGATIVE_TEMPLATES, seed, parent_asin, slot, negative_value, "template"
    )
    negative_phrase = template.format(value=negative_value)
    return {
        "slot": slot,
        "positive_value": positive_value,
        "positive_phrase": f"{slot}: {positive_value}",
        "positive_evidence_source": evidence_source,
        "negative_value": negative_value,
        "negative_phrase": negative_phrase,
        "negative_template": template,
        "negative_support": negative_support,
        "negative_bucket_level": chosen_level,
        "negative_bucket_value": bucket_value,
    }


def _eligible_plans(
    products: dict[str, dict[str, Any]],
    seed: str,
    minimum_support: int = MIN_NEGATIVE_SUPPORT,
) -> dict[str, dict[str, Any]]:
    evidence_by_id = {
        parent_asin: evidence
        for parent_asin, product in products.items()
        if (evidence := _product_evidence(product)) is not None
    }
    frequencies = _bucket_document_frequencies(evidence_by_id)
    plans: dict[str, dict[str, Any]] = {}
    for parent_asin in products:
        evidence = evidence_by_id.get(parent_asin)
        if evidence is None:
            continue
        plan = _constraint_plan(
            evidence, frequencies, seed, parent_asin, minimum_support
        )
        if plan is not None:
            plans[parent_asin] = plan
    return plans


def _clean_title(product: Mapping[str, Any], limit: int = 180) -> str:
    title = _SPACE_RE.sub(" ", str(product.get("title") or "product")).strip()
    return title[:limit].rstrip()


def _profile() -> dict[str, Any]:
    return {
        "purchase_frequency": "not provided",
        "average_prior_rating": None,
        "rating_style": "not provided",
        "preference_tags": [],
        "summary": "Neutral profile for a derived explicit-negative stress session.",
    }


def _materialize_sample(
    parent_asin: str,
    product: Mapping[str, Any],
    plan: dict[str, Any],
    scenario: str,
    sample_id: str,
    seed: str,
) -> dict[str, Any]:
    negative_audit = {
        "slot": plan["slot"],
        "excluded_value": plan["negative_value"],
        "phrase": plan["negative_phrase"],
        "template": plan["negative_template"],
        "positive_anchor": plan["positive_phrase"],
        "positive_value": plan["positive_value"],
        "positive_evidence_source": plan["positive_evidence_source"],
        "catalog_document_support": plan["negative_support"],
        "category_bucket_level": plan["negative_bucket_level"],
        "description_used_as_evidence": False,
    }
    behavior: dict[str, Any] = {
        "scenario_type": scenario,
        "explicit_negative": negative_audit,
    }
    if scenario == "intent_override":
        combined = f"{plan['negative_phrase']}; {plan['positive_phrase']}"
        turn = 3 + int(_stable_digest(seed, parent_asin, "override-turn"), 16) % 2
        behavior["override"] = {
            "turn": turn,
            "old_value": f"{plan['slot']}: {plan['negative_value']}",
            "new_value": combined,
            "message": (
                "Actually, ignore my earlier preference. What I need is: "
                f"{combined}."
            ),
        }

    return {
        "category_bucket": "derived_explicit_negative",
        "difficulty_bucket": "pre_materialized",
        "ground_truth": {"parent_asin": parent_asin},
        "sample_id": sample_id,
        "scenario_type": scenario,
        "user_profile": _profile(),
        "intent_card": {
            "target_category": _clean_title(product),
            "hard_constraints": [plan["negative_phrase"]],
            "soft_preferences": [plan["positive_phrase"]],
        },
        "behavior": behavior,
    }


def _build_one_corpus(
    name: str,
    prefix: str,
    selected_ids: list[str],
    products: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenarios = _scenario_sequence(len(selected_ids), seed)
    samples = [
        _materialize_sample(
            parent_asin,
            products[parent_asin],
            plans[parent_asin],
            scenario,
            f"{prefix}{index:04d}",
            seed,
        )
        for index, (parent_asin, scenario) in enumerate(
            zip(selected_ids, scenarios, strict=True), start=1
        )
    ]
    scenario_counts = dict(sorted(Counter(scenarios).items()))
    if len(samples) == DEFAULT_COUNT and scenario_counts != EXPECTED_SCENARIO_COUNTS:
        raise RuntimeError(f"P8 {name} default scenario-mix invariant failed")
    slot_counts = Counter(
        sample["behavior"]["explicit_negative"]["slot"] for sample in samples
    )
    template_counts = Counter(
        sample["behavior"]["explicit_negative"]["template"] for sample in samples
    )
    bucket_level_counts = Counter(
        sample["behavior"]["explicit_negative"]["category_bucket_level"]
        for sample in samples
    )
    supports = [
        int(sample["behavior"]["explicit_negative"]["catalog_document_support"])
        for sample in samples
    ]
    return samples, {
        "sample_count": len(samples),
        "unique_target_count": len(_target_ids(samples)),
        "sample_id_prefix": prefix,
        "seed": seed,
        "samples_sha256": _samples_sha256(samples),
        "scenario_counts": scenario_counts,
        "negative_slot_counts": dict(sorted(slot_counts.items())),
        "negative_template_counts": dict(sorted(template_counts.items())),
        "negative_bucket_level_counts": dict(sorted(bucket_level_counts.items())),
        "selected_negative_support": {
            "min": min(supports),
            "median": statistics.median(supports),
            "max": max(supports),
            "min_support": MIN_NEGATIVE_SUPPORT,
        },
        "description_evidence_count": 0,
    }


def build_p8_selection_corpora(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int = DEFAULT_COUNT,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    confirmation_seed: str = DEFAULT_CONFIRMATION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic selection and confirmation samples plus aggregates."""

    if count <= 0:
        raise ValueError("P8 corpus count must be positive")
    if set(NEGATIVE_VOCABULARIES) != set(ALLOWED_NEGATIVE_SLOTS):
        raise RuntimeError("P8 builder and runtime negative-slot registries differ")
    failures = _input_invariant_failures(
        public_samples, p1_samples, p5_samples, p6_samples, p7_samples
    )
    if failures:
        raise ValueError("invalid P8 exclusion inputs: " + "; ".join(failures))

    target_sets = _input_target_sets(
        public_samples, p1_samples, p5_samples, p6_samples, p7_samples
    )
    excluded_targets = set().union(*target_sets.values())
    missing = excluded_targets - set(products)
    if missing:
        raise ValueError(f"{len(missing)} excluded targets are missing from catalog")

    plan_seed = f"{selection_seed}\0{confirmation_seed}\0constraint-plans"
    plans = _eligible_plans(products, plan_seed)
    eligible_ids = set(plans) - excluded_targets
    if len(eligible_ids) < count * 2:
        raise ValueError(
            f"requested two P8 corpora of {count} samples but only "
            f"{len(eligible_ids)} eligible disjoint targets remain"
        )

    selection_ids = sorted(
        eligible_ids,
        key=lambda parent_asin: _stable_digest(selection_seed, parent_asin),
    )[:count]
    confirmation_ids = sorted(
        eligible_ids - set(selection_ids),
        key=lambda parent_asin: _stable_digest(confirmation_seed, parent_asin),
    )[:count]
    selection, selection_summary = _build_one_corpus(
        "selection",
        SELECTION_SAMPLE_ID_PREFIX,
        selection_ids,
        products,
        plans,
        selection_seed,
    )
    confirmation, confirmation_summary = _build_one_corpus(
        "confirmation",
        CONFIRMATION_SAMPLE_ID_PREFIX,
        confirmation_ids,
        products,
        plans,
        confirmation_seed,
    )

    selection_targets = _target_ids(selection)
    confirmation_targets = _target_ids(confirmation)
    selected_overlaps = {
        name: len((selection_targets | confirmation_targets) & targets)
        for name, targets in target_sets.items()
    }
    cross_overlap = len(selection_targets & confirmation_targets)
    if (
        len(selection_targets) != count
        or len(confirmation_targets) != count
        or cross_overlap
        or any(selected_overlaps.values())
    ):
        raise RuntimeError("P8 target uniqueness or exclusion invariant failed")

    pairwise_input_overlaps = {
        _pair_key(left, right): len(target_sets[left] & target_sets[right])
        for left, right in combinations(target_sets, 2)
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "p8_explicit_negative_product_disjoint",
        "canonical_serialization": (
            "UTF-8 JSON Lines; object keys sorted; compact separators; LF after every row"
        ),
        "corpora": {
            "selection": selection_summary,
            "confirmation": confirmation_summary,
        },
        "exclusions": {
            "input_target_counts": {
                name: len(targets) for name, targets in target_sets.items()
            },
            "combined_unique_input_target_count": len(excluded_targets),
            "pairwise_input_target_overlaps": pairwise_input_overlaps,
            "selected_target_overlaps": selected_overlaps,
            "selection_confirmation_target_overlap": cross_overlap,
        },
        "catalog_source": {
            "description": "official frozen 50,000-product participant catalog",
            "expected_product_count": OFFICIAL_CATALOG_COUNT,
            "loaded_product_count": len(products),
        },
        "generator": {
            "target_selection": (
                "SHA-256(seed + NUL + parent_asin) over catalog-eligible products, "
                "excluding released-public and frozen P1/P5/P6/P7 targets; confirmation "
                "also excludes the complete selection corpus"
            ),
            "agent_used": False,
            "fts_used": False,
            "prior_results_used": False,
            "intent_fields_pre_materialized": True,
            "evidence_source_rule": EVIDENCE_SOURCE_RULE,
            "description_used_as_evidence": False,
            "negative_templates": list(NEGATIVE_TEMPLATES),
            "allowed_negative_slots": list(NEGATIVE_VOCABULARIES),
            "negative_values_single_token": True,
            "negative_support_policy": {
                "min_support": MIN_NEGATIVE_SUPPORT,
                "frequency_unit": "catalog documents",
                "scope": "shared reliable category bucket",
                "selection": "highest document frequency, then seed+ASIN hash tie-break",
            },
            "category_bucket_fallback_order": list(BUCKET_FALLBACK_ORDER),
            "global_category_fallback_used": False,
        },
        "boundary": (
            "These are deterministic catalog-derived local stress corpora. They are "
            "not organizer private data, not a private-distribution proxy, and not a "
            "hidden-leaderboard estimate. Metadata contains only aggregate counts and "
            "hashes, never selected target IDs."
        ),
    }
    return selection, confirmation, metadata


def _validate_frozen_inputs(
    rows: dict[str, list[dict[str, Any]]],
    products: dict[str, dict[str, Any]],
    expected_counts: dict[str, int],
) -> None:
    failures = _input_invariant_failures(
        rows["released_public"],
        rows["prior_p1_derived"],
        rows["prior_p5_derived"],
        rows["prior_p6_derived"],
        rows["prior_p7_derived"],
    )
    if len(products) != expected_counts["catalog"]:
        failures.append(
            f"catalog products {len(products)} != expected {expected_counts['catalog']}"
        )
    for name, samples in rows.items():
        if len(samples) != expected_counts[name]:
            failures.append(
                f"{name} samples {len(samples)} != expected {expected_counts[name]}"
            )

    excluded_targets = set().union(*(_target_ids(samples) for samples in rows.values()))
    missing = excluded_targets - set(products)
    if missing:
        failures.append(f"{len(missing)} excluded targets are missing from catalog")
    if failures:
        raise ValueError("invalid frozen P8 inputs: " + "; ".join(failures))


def _atomic_write_many(payloads: Mapping[Path, bytes]) -> None:
    temporary_paths: dict[Path, Path] = {}
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            temporary_paths[destination] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temporary_path in temporary_paths.items():
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


def build_and_write_p8_selection_corpora(
    catalog_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
    p6_path: Path,
    p7_path: Path,
    selection_output_path: Path,
    confirmation_output_path: Path,
    metadata_path: Path,
    *,
    count: int = DEFAULT_COUNT,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    confirmation_seed: str = DEFAULT_CONFIRMATION_SEED,
    expected_catalog_count: int = OFFICIAL_CATALOG_COUNT,
    expected_public_count: int = OFFICIAL_PUBLIC_COUNT,
    expected_p1_count: int = PRIOR_DERIVED_COUNT,
    expected_p5_count: int = PRIOR_DERIVED_COUNT,
    expected_p6_count: int = PRIOR_DERIVED_COUNT,
    expected_p7_count: int = PRIOR_DERIVED_COUNT,
    expected_catalog_sha256: str = CATALOG_FROZEN_SHA256,
    expected_public_git_blob_sha1: str = PUBLIC_FROZEN_GIT_BLOB_SHA1,
    expected_public_samples_sha256: str = PUBLIC_FROZEN_SAMPLES_SHA256,
    expected_p1_samples_sha256: str = P1_FROZEN_SAMPLES_SHA256,
    expected_p5_samples_sha256: str = P5_FROZEN_SAMPLES_SHA256,
    expected_p6_samples_sha256: str = P6_FROZEN_SAMPLES_SHA256,
    expected_p7_samples_sha256: str = P7_FROZEN_SAMPLES_SHA256,
    expected_selection_output_sha256: str | None = P8_SELECTION_FROZEN_SAMPLES_SHA256,
    expected_confirmation_output_sha256: str | None = P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
) -> dict[str, Any]:
    """Validate all frozen inputs, then safely write both canonical corpora."""

    input_paths = {
        "released_public": public_path,
        "prior_p1_derived": p1_path,
        "prior_p5_derived": p5_path,
        "prior_p6_derived": p6_path,
        "prior_p7_derived": p7_path,
    }
    output_paths = {
        "selection": selection_output_path,
        "confirmation": confirmation_output_path,
        "metadata": metadata_path,
    }
    resolved_inputs = {catalog_path.resolve()} | {
        path.resolve() for path in input_paths.values()
    }
    resolved_outputs = {path.resolve() for path in output_paths.values()}
    if len(resolved_outputs) != len(output_paths):
        raise ValueError("P8 selection, confirmation, and metadata outputs must differ")
    if resolved_inputs & resolved_outputs:
        raise ValueError("P8 outputs must not overwrite a frozen input file")

    catalog_sha256 = _file_sha256(catalog_path)
    public_blob = git_blob_sha1(public_path)
    hash_failures: list[str] = []
    if catalog_sha256 != expected_catalog_sha256.lower():
        hash_failures.append(
            "catalog frozen SHA-256 mismatch: "
            f"{catalog_sha256} != expected {expected_catalog_sha256.lower()}"
        )
    if public_blob != expected_public_git_blob_sha1.lower():
        hash_failures.append(
            "released_public normalized Git blob mismatch: "
            f"{public_blob} != expected {expected_public_git_blob_sha1.lower()}"
        )
    if hash_failures:
        raise ValueError("; ".join(hash_failures))

    rows = {
        name: _load_jsonl(path) for name, path in input_paths.items()
    }
    expected_sample_hashes = {
        "released_public": expected_public_samples_sha256.lower(),
        "prior_p1_derived": expected_p1_samples_sha256.lower(),
        "prior_p5_derived": expected_p5_samples_sha256.lower(),
        "prior_p6_derived": expected_p6_samples_sha256.lower(),
        "prior_p7_derived": expected_p7_samples_sha256.lower(),
    }
    actual_sample_hashes = {
        name: _samples_sha256(samples) for name, samples in rows.items()
    }
    sample_hash_failures = [
        f"{name} frozen canonical sample SHA-256 mismatch: "
        f"{actual_sample_hashes[name]} != expected {expected}"
        for name, expected in expected_sample_hashes.items()
        if actual_sample_hashes[name] != expected
    ]
    if sample_hash_failures:
        raise ValueError("; ".join(sample_hash_failures))

    products = _load_catalog(catalog_path)
    _validate_frozen_inputs(
        rows,
        products,
        {
            "catalog": expected_catalog_count,
            "released_public": expected_public_count,
            "prior_p1_derived": expected_p1_count,
            "prior_p5_derived": expected_p5_count,
            "prior_p6_derived": expected_p6_count,
            "prior_p7_derived": expected_p7_count,
        },
    )
    selection, confirmation, metadata = build_p8_selection_corpora(
        rows["released_public"],
        rows["prior_p1_derived"],
        rows["prior_p5_derived"],
        rows["prior_p6_derived"],
        rows["prior_p7_derived"],
        products,
        count,
        selection_seed,
        confirmation_seed,
    )

    output_expectations = {
        "selection": expected_selection_output_sha256,
        "confirmation": expected_confirmation_output_sha256,
    }
    for name, expected in output_expectations.items():
        if expected is not None and metadata["corpora"][name]["samples_sha256"] != expected.lower():
            raise ValueError(
                f"P8 {name} frozen output SHA-256 mismatch: "
                f"{metadata['corpora'][name]['samples_sha256']} != expected {expected.lower()}"
            )

    metadata["catalog_source"] = {
        **metadata["catalog_source"],
        "path": str(catalog_path),
        "sha256": catalog_sha256,
        "expected_frozen_sha256": expected_catalog_sha256.lower(),
        "frozen_sha256_verified": True,
        "expected_count_verified": len(products) == expected_catalog_count,
    }
    metadata["input_sources"] = {
        name: {
            "path": str(input_paths[name]),
            "sha256": _file_sha256(input_paths[name]),
            "sample_count": len(rows[name]),
            "unique_target_count": len(_target_ids(rows[name])),
            "canonical_samples_sha256": actual_sample_hashes[name],
            "expected_frozen_samples_sha256": expected_sample_hashes[name],
            "frozen_samples_sha256_verified": True,
        }
        for name in input_paths
    }
    metadata["input_sources"]["released_public"].update(
        {
            "git_blob_sha1_lf": public_blob,
            "expected_frozen_git_blob_sha1_lf": expected_public_git_blob_sha1.lower(),
            "frozen_git_blob_verified": True,
        }
    )
    metadata["outputs"] = {
        name: {
            "path": str(output_paths[name]),
            "samples_file_sha256": metadata["corpora"][name]["samples_sha256"],
            "expected_frozen_samples_sha256": (
                output_expectations[name].lower()
                if output_expectations[name] is not None
                else None
            ),
            "frozen_samples_sha256_verified": output_expectations[name] is not None,
        }
        for name in ("selection", "confirmation")
    }
    metadata["outputs"]["metadata"] = {"path": str(metadata_path)}

    selection_payload = _canonical_jsonl_bytes(selection)
    confirmation_payload = _canonical_jsonl_bytes(confirmation)
    metadata_payload = (
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_many(
        {
            selection_output_path: selection_payload,
            confirmation_output_path: confirmation_payload,
            metadata_path: metadata_payload,
        }
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build P8 selection and confirmation explicit-negative corpora while "
            "excluding released-public and frozen P1/P5/P6/P7 targets."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--prior-p1", type=Path, default=DEFAULT_P1_PATH)
    parser.add_argument("--prior-p5", type=Path, default=DEFAULT_P5_PATH)
    parser.add_argument("--prior-p6", type=Path, default=DEFAULT_P6_PATH)
    parser.add_argument("--prior-p7", type=Path, default=DEFAULT_P7_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--confirmation-seed", default=DEFAULT_CONFIRMATION_SEED)
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT
    )
    parser.add_argument(
        "--confirmation-output", type=Path, default=DEFAULT_CONFIRMATION_OUTPUT
    )
    parser.add_argument(
        "--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = build_and_write_p8_selection_corpora(
        args.catalog,
        args.public,
        args.prior_p1,
        args.prior_p5,
        args.prior_p6,
        args.prior_p7,
        args.selection_output,
        args.confirmation_output,
        args.metadata_output,
        count=args.count,
        selection_seed=args.selection_seed,
        confirmation_seed=args.confirmation_seed,
    )
    selection = metadata["corpora"]["selection"]
    confirmation = metadata["corpora"]["confirmation"]
    print(
        "[p8-corpora] "
        f"selection={selection['sample_count']} "
        f"selection_sha256={selection['samples_sha256']} "
        f"confirmation={confirmation['sample_count']} "
        f"confirmation_sha256={confirmation['samples_sha256']}",
        flush=True,
    )
    print(
        "[p8-corpora] wrote "
        f"{args.selection_output}, {args.confirmation_output}, and "
        f"{args.metadata_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
