from __future__ import annotations

"""Build two frozen P9 corpora outside every public and P1-P8 target set."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_p8_selection_corpus import (  # noqa: E402
    BUCKET_FALLBACK_ORDER,
    CATALOG_FROZEN_SHA256,
    EVIDENCE_SOURCE_RULE,
    EXPECTED_SCENARIO_COUNTS,
    MIN_NEGATIVE_SUPPORT,
    NEGATIVE_TEMPLATES,
    NEGATIVE_VOCABULARIES,
    P1_FROZEN_SAMPLES_SHA256,
    P5_FROZEN_SAMPLES_SHA256,
    P6_FROZEN_SAMPLES_SHA256,
    P7_FROZEN_SAMPLES_SHA256,
    P8_CONFIRMATION_FROZEN_SAMPLES_SHA256,
    P8_SELECTION_FROZEN_SAMPLES_SHA256,
    PUBLIC_FROZEN_GIT_BLOB_SHA1,
    PUBLIC_FROZEN_SAMPLES_SHA256,
    _atomic_write_many,
    _build_one_corpus,
    _canonical_jsonl_bytes,
    _eligible_plans,
    _file_sha256,
    _load_catalog,
    _load_jsonl,
    _samples_sha256,
    _stable_digest,
    _target_ids,
)
from scripts.verify_official_assets import git_blob_sha1  # noqa: E402
from starter.p8_negative import ALLOWED_NEGATIVE_SLOTS  # noqa: E402


SCHEMA_VERSION = "p9.explicit-negative-corpora.v1"
DEFAULT_COUNT = 200
DEFAULT_SELECTION_SEED = "track4-p9-explicit-negative-selection-v1"
DEFAULT_CONFIRMATION_SEED = "track4-p9-explicit-negative-confirmation-v1"
DEFAULT_SELECTION_OUTPUT = Path(
    "experiments/p9_selection_product_disjoint.jsonl"
)
DEFAULT_CONFIRMATION_OUTPUT = Path(
    "experiments/p9_confirmation_product_disjoint.jsonl"
)
DEFAULT_METADATA_OUTPUT = Path(
    "experiments/p9_explicit_negative_corpora.metadata.json"
)
DEFAULT_P1_PATH = Path("experiments/p1_derived_product_disjoint.jsonl")
DEFAULT_P5_PATH = Path("experiments/p5_selection_product_disjoint.jsonl")
DEFAULT_P6_PATH = Path("experiments/p6_selection_product_disjoint.jsonl")
DEFAULT_P7_PATH = Path("experiments/p7_selection_product_disjoint.jsonl")
DEFAULT_P8_SELECTION_PATH = Path(
    "experiments/p8_selection_product_disjoint.jsonl"
)
DEFAULT_P8_CONFIRMATION_PATH = Path(
    "experiments/p8_confirmation_product_disjoint.jsonl"
)

OFFICIAL_CATALOG_COUNT = 50_000
OFFICIAL_PUBLIC_COUNT = 200
PRIOR_DERIVED_COUNT = 200
P9_SELECTION_FROZEN_SAMPLES_SHA256 = (
    "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f"
)
P9_CONFIRMATION_FROZEN_SAMPLES_SHA256 = (
    "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9"
)

SELECTION_SAMPLE_ID_PREFIX = "derived_p9_selection_"
CONFIRMATION_SAMPLE_ID_PREFIX = "derived_p9_confirmation_"

_INPUT_SPECS = (
    ("released_public", "released public", "public_"),
    ("prior_p1_derived", "prior P1-derived", "derived_p1_"),
    ("prior_p5_derived", "prior P5-derived", "derived_p5_"),
    ("prior_p6_derived", "prior P6-derived", "derived_p6_"),
    ("prior_p7_derived", "prior P7-derived", "derived_p7_"),
    ("prior_p8_selection", "prior P8 selection", "derived_p8_selection_"),
    (
        "prior_p8_confirmation",
        "prior P8 confirmation",
        "derived_p8_confirmation_",
    ),
)


def _input_rows(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
    p8_selection_samples: list[dict[str, Any]],
    p8_confirmation_samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "released_public": public_samples,
        "prior_p1_derived": p1_samples,
        "prior_p5_derived": p5_samples,
        "prior_p6_derived": p6_samples,
        "prior_p7_derived": p7_samples,
        "prior_p8_selection": p8_selection_samples,
        "prior_p8_confirmation": p8_confirmation_samples,
    }


def _input_target_sets(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
    p8_selection_samples: list[dict[str, Any]],
    p8_confirmation_samples: list[dict[str, Any]],
) -> dict[str, set[str]]:
    rows = _input_rows(
        public_samples,
        p1_samples,
        p5_samples,
        p6_samples,
        p7_samples,
        p8_selection_samples,
        p8_confirmation_samples,
    )
    return {name: _target_ids(samples) for name, samples in rows.items()}


def _pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _input_invariant_failures(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
    p8_selection_samples: list[dict[str, Any]],
    p8_confirmation_samples: list[dict[str, Any]],
) -> list[str]:
    rows = _input_rows(
        public_samples,
        p1_samples,
        p5_samples,
        p6_samples,
        p7_samples,
        p8_selection_samples,
        p8_confirmation_samples,
    )
    targets = {name: _target_ids(samples) for name, samples in rows.items()}
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


def build_p9_selection_corpora(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    p7_samples: list[dict[str, Any]],
    p8_selection_samples: list[dict[str, Any]],
    p8_confirmation_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int = DEFAULT_COUNT,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    confirmation_seed: str = DEFAULT_CONFIRMATION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return P9 selection/confirmation corpora outside all seven inputs."""

    if count <= 0:
        raise ValueError("P9 corpus count must be positive")
    if set(NEGATIVE_VOCABULARIES) != set(ALLOWED_NEGATIVE_SLOTS):
        raise RuntimeError("P9 builder and runtime negative-slot registries differ")
    failures = _input_invariant_failures(
        public_samples,
        p1_samples,
        p5_samples,
        p6_samples,
        p7_samples,
        p8_selection_samples,
        p8_confirmation_samples,
    )
    if failures:
        raise ValueError("invalid P9 exclusion inputs: " + "; ".join(failures))

    target_sets = _input_target_sets(
        public_samples,
        p1_samples,
        p5_samples,
        p6_samples,
        p7_samples,
        p8_selection_samples,
        p8_confirmation_samples,
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
            f"requested two P9 corpora of {count} samples but only "
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
        "P9 selection",
        SELECTION_SAMPLE_ID_PREFIX,
        selection_ids,
        products,
        plans,
        selection_seed,
    )
    confirmation, confirmation_summary = _build_one_corpus(
        "P9 confirmation",
        CONFIRMATION_SAMPLE_ID_PREFIX,
        confirmation_ids,
        products,
        plans,
        confirmation_seed,
    )

    selection_targets = _target_ids(selection)
    confirmation_targets = _target_ids(confirmation)
    output_targets = selection_targets | confirmation_targets
    selected_overlaps = {
        name: len(output_targets & targets) for name, targets in target_sets.items()
    }
    cross_overlap = len(selection_targets & confirmation_targets)
    if (
        len(selection_targets) != count
        or len(confirmation_targets) != count
        or cross_overlap
        or any(selected_overlaps.values())
    ):
        raise RuntimeError("P9 target uniqueness or exclusion invariant failed")
    if count == DEFAULT_COUNT:
        for name, summary in (
            ("selection", selection_summary),
            ("confirmation", confirmation_summary),
        ):
            if summary["scenario_counts"] != EXPECTED_SCENARIO_COUNTS:
                raise RuntimeError(f"P9 {name} default scenario-mix invariant failed")
            if summary["selected_negative_support"]["min"] < MIN_NEGATIVE_SUPPORT:
                raise RuntimeError(f"P9 {name} negative-support invariant failed")

    pairwise_input_overlaps = {
        _pair_key(left, right): len(target_sets[left] & target_sets[right])
        for left, right in combinations(target_sets, 2)
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "p9_explicit_negative_product_disjoint",
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
            "base_protocol": "scripts.build_p8_selection_corpus catalog-only helpers",
            "target_selection": (
                "SHA-256(seed + NUL + parent_asin) over catalog-eligible products, "
                "excluding released-public, frozen P1/P5/P6/P7, and both frozen P8 "
                "target sets; confirmation also excludes the complete P9 selection"
            ),
            "agent_used": False,
            "fts_used": False,
            "prior_results_used": False,
            "prior_metrics_used": False,
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
            "hidden-leaderboard estimate. Metadata contains aggregate counts and "
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
        rows["prior_p8_selection"],
        rows["prior_p8_confirmation"],
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
        raise ValueError("invalid frozen P9 inputs: " + "; ".join(failures))


def build_and_write_p9_selection_corpora(
    catalog_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
    p6_path: Path,
    p7_path: Path,
    p8_selection_path: Path,
    p8_confirmation_path: Path,
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
    expected_p8_selection_count: int = PRIOR_DERIVED_COUNT,
    expected_p8_confirmation_count: int = PRIOR_DERIVED_COUNT,
    expected_catalog_sha256: str = CATALOG_FROZEN_SHA256,
    expected_public_git_blob_sha1: str = PUBLIC_FROZEN_GIT_BLOB_SHA1,
    expected_public_samples_sha256: str = PUBLIC_FROZEN_SAMPLES_SHA256,
    expected_p1_samples_sha256: str = P1_FROZEN_SAMPLES_SHA256,
    expected_p5_samples_sha256: str = P5_FROZEN_SAMPLES_SHA256,
    expected_p6_samples_sha256: str = P6_FROZEN_SAMPLES_SHA256,
    expected_p7_samples_sha256: str = P7_FROZEN_SAMPLES_SHA256,
    expected_p8_selection_samples_sha256: str = P8_SELECTION_FROZEN_SAMPLES_SHA256,
    expected_p8_confirmation_samples_sha256: str = (
        P8_CONFIRMATION_FROZEN_SAMPLES_SHA256
    ),
    expected_selection_output_sha256: str | None = P9_SELECTION_FROZEN_SAMPLES_SHA256,
    expected_confirmation_output_sha256: str | None = (
        P9_CONFIRMATION_FROZEN_SAMPLES_SHA256
    ),
) -> dict[str, Any]:
    """Validate seven frozen inputs and atomically write both P9 corpora."""

    input_paths = {
        "released_public": public_path,
        "prior_p1_derived": p1_path,
        "prior_p5_derived": p5_path,
        "prior_p6_derived": p6_path,
        "prior_p7_derived": p7_path,
        "prior_p8_selection": p8_selection_path,
        "prior_p8_confirmation": p8_confirmation_path,
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
        raise ValueError("P9 selection, confirmation, and metadata outputs must differ")
    if resolved_inputs & resolved_outputs:
        raise ValueError("P9 outputs must not overwrite a frozen input file")

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

    rows = {name: _load_jsonl(path) for name, path in input_paths.items()}
    expected_sample_hashes = {
        "released_public": expected_public_samples_sha256.lower(),
        "prior_p1_derived": expected_p1_samples_sha256.lower(),
        "prior_p5_derived": expected_p5_samples_sha256.lower(),
        "prior_p6_derived": expected_p6_samples_sha256.lower(),
        "prior_p7_derived": expected_p7_samples_sha256.lower(),
        "prior_p8_selection": expected_p8_selection_samples_sha256.lower(),
        "prior_p8_confirmation": expected_p8_confirmation_samples_sha256.lower(),
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
            "prior_p8_selection": expected_p8_selection_count,
            "prior_p8_confirmation": expected_p8_confirmation_count,
        },
    )
    selection, confirmation, metadata = build_p9_selection_corpora(
        rows["released_public"],
        rows["prior_p1_derived"],
        rows["prior_p5_derived"],
        rows["prior_p6_derived"],
        rows["prior_p7_derived"],
        rows["prior_p8_selection"],
        rows["prior_p8_confirmation"],
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
        if (
            expected is not None
            and metadata["corpora"][name]["samples_sha256"] != expected.lower()
        ):
            raise ValueError(
                f"P9 {name} frozen output SHA-256 mismatch: "
                f"{metadata['corpora'][name]['samples_sha256']} != expected "
                f"{expected.lower()}"
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

    _atomic_write_many(
        {
            selection_output_path: _canonical_jsonl_bytes(selection),
            confirmation_output_path: _canonical_jsonl_bytes(confirmation),
            metadata_path: (
                json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8"),
        }
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build P9 selection and confirmation corpora while excluding public, "
            "P1/P5/P6/P7, and both frozen P8 target sets."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--prior-p1", type=Path, default=DEFAULT_P1_PATH)
    parser.add_argument("--prior-p5", type=Path, default=DEFAULT_P5_PATH)
    parser.add_argument("--prior-p6", type=Path, default=DEFAULT_P6_PATH)
    parser.add_argument("--prior-p7", type=Path, default=DEFAULT_P7_PATH)
    parser.add_argument(
        "--prior-p8-selection", type=Path, default=DEFAULT_P8_SELECTION_PATH
    )
    parser.add_argument(
        "--prior-p8-confirmation", type=Path, default=DEFAULT_P8_CONFIRMATION_PATH
    )
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
    metadata = build_and_write_p9_selection_corpora(
        args.catalog,
        args.public,
        args.prior_p1,
        args.prior_p5,
        args.prior_p6,
        args.prior_p7,
        args.prior_p8_selection,
        args.prior_p8_confirmation,
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
        "[p9-corpora] "
        f"selection={selection['sample_count']} "
        f"selection_sha256={selection['samples_sha256']} "
        f"confirmation={confirmation['sample_count']} "
        f"confirmation_sha256={confirmation['samples_sha256']}",
        flush=True,
    )
    print(
        "[p9-corpora] wrote "
        f"{args.selection_output}, {args.confirmation_output}, and "
        f"{args.metadata_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
