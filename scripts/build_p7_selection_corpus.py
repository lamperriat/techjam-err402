from __future__ import annotations

"""Build the frozen P7 selection corpus from a fourth disjoint product set."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from scripts.evaluate_generalization import (  # noqa: E402
    _samples_sha256,
    build_product_disjoint_samples,
)
from scripts.verify_official_assets import git_blob_sha1  # noqa: E402


SCHEMA_VERSION = "p7.selection-corpus.v1"
DEFAULT_COUNT = 200
DEFAULT_SEED = "track4-p7-product-disjoint-v1"
DEFAULT_OUTPUT = Path("experiments/p7_selection_product_disjoint.jsonl")
DEFAULT_METADATA_OUTPUT = Path(
    "experiments/p7_selection_product_disjoint.metadata.json"
)
DEFAULT_P1_PATH = Path("experiments/p1_derived_product_disjoint.jsonl")
DEFAULT_P5_PATH = Path("experiments/p5_selection_product_disjoint.jsonl")
DEFAULT_P6_PATH = Path("experiments/p6_selection_product_disjoint.jsonl")
OFFICIAL_CATALOG_COUNT = 50_000
OFFICIAL_PUBLIC_COUNT = 200
PRIOR_DERIVED_COUNT = 200
CATALOG_FROZEN_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
PUBLIC_FROZEN_GIT_BLOB_SHA1 = (
    "121dbec9c1368c81cd887d6959e62507512139c0"
)
P1_FROZEN_SHA256 = (
    "265a6dae0d9029d54333fbce980b23981b5332d967fc2b450924b05443cadc46"
)
P1_FROZEN_SAMPLES_SHA256 = (
    "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
)
P5_FROZEN_SHA256 = (
    "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c"
)
P5_FROZEN_SAMPLES_SHA256 = P5_FROZEN_SHA256
P6_FROZEN_SHA256 = (
    "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7"
)
P6_FROZEN_SAMPLES_SHA256 = P6_FROZEN_SHA256
P7_FROZEN_SHA256 = (
    "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546"
)
SAMPLE_ID_PREFIX = "derived_p7_"
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}

_INPUT_SPECS = (
    ("released_public", "released public", "public_"),
    ("prior_p1_derived", "prior P1-derived", "derived_p1_"),
    ("prior_p5_derived", "prior P5-derived", "derived_p5_"),
    ("prior_p6_derived", "prior P6-derived", "derived_p6_"),
)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_rows(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "released_public": public_samples,
        "prior_p1_derived": p1_samples,
        "prior_p5_derived": p5_samples,
        "prior_p6_derived": p6_samples,
    }


def _input_target_sets(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
) -> dict[str, set[str]]:
    return {
        name: _target_ids(samples)
        for name, samples in _input_rows(
            public_samples, p1_samples, p5_samples, p6_samples
        ).items()
    }


def _pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _input_invariant_failures(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
) -> list[str]:
    rows = _input_rows(public_samples, p1_samples, p5_samples, p6_samples)
    targets = _input_target_sets(public_samples, p1_samples, p5_samples, p6_samples)
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


def build_p7_selection_corpus(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic samples outside all four frozen prior target sets."""

    failures = _input_invariant_failures(
        public_samples, p1_samples, p5_samples, p6_samples
    )
    if failures:
        raise ValueError("invalid P7 exclusion inputs: " + "; ".join(failures))

    target_sets = _input_target_sets(
        public_samples, p1_samples, p5_samples, p6_samples
    )
    generated, base_metadata = build_product_disjoint_samples(
        [*public_samples, *p1_samples, *p5_samples, *p6_samples],
        products,
        count,
        seed,
    )
    samples = [
        {**sample, "sample_id": f"{SAMPLE_ID_PREFIX}{index:04d}"}
        for index, sample in enumerate(generated, start=1)
    ]
    selected_targets = _target_ids(samples)
    overlaps = {
        name: selected_targets & targets for name, targets in target_sets.items()
    }
    if len(selected_targets) != len(samples) or any(overlaps.values()):
        raise RuntimeError("P7 target uniqueness or exclusion invariant failed")

    scenario_counts = dict(
        sorted(Counter(str(sample["scenario_type"]) for sample in samples).items())
    )
    if count == DEFAULT_COUNT and scenario_counts != EXPECTED_SCENARIO_COUNTS:
        raise RuntimeError("P7 default scenario-mix invariant failed")

    pairwise_overlaps = {
        _pair_key(left, right): len(target_sets[left] & target_sets[right])
        for left, right in combinations(target_sets, 2)
    }
    payload = _canonical_jsonl_bytes(samples)
    union_targets = set().union(*target_sets.values())
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "p7_selection_product_disjoint",
        "seed": seed,
        "sample_count": len(samples),
        "sample_id_prefix": SAMPLE_ID_PREFIX,
        "samples_sha256": _sha256_bytes(payload),
        "canonical_serialization": (
            "UTF-8 JSON Lines; object keys sorted; compact separators; LF after every row"
        ),
        "unique_target_count": len(selected_targets),
        "scenario_counts": scenario_counts,
        "target_overlaps": {name: len(value) for name, value in overlaps.items()},
        "exclusions": {
            "input_target_counts": {
                name: len(targets) for name, targets in target_sets.items()
            },
            "combined_unique_target_count": len(union_targets),
            "pairwise_input_target_overlaps": pairwise_overlaps,
            "selected_target_overlaps": {
                name: len(value) for name, value in overlaps.items()
            },
        },
        "catalog_source": {
            "description": "official frozen 50,000-product participant catalog",
            "expected_product_count": OFFICIAL_CATALOG_COUNT,
            "loaded_product_count": len(products),
        },
        "generator": {
            "base": "scripts.evaluate_generalization.build_product_disjoint_samples",
            "base_selection": base_metadata.get("selection"),
            "prior_generators_modified": False,
            "selection": (
                "SHA-256(seed + NUL + parent_asin), excluding the union of "
                "released-public, prior P1-derived, prior P5-derived, and prior "
                "P6-derived targets; requires non-empty title and categories"
            ),
        },
        "boundary": (
            "This is a deterministic catalog-derived local selection stress set. It is "
            "not organizer private data, not a private-distribution proxy, and not a "
            "hidden-leaderboard estimate."
        ),
    }
    return samples, metadata


def _validate_frozen_inputs(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    p6_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    *,
    expected_catalog_count: int,
    expected_public_count: int,
    expected_p1_count: int,
    expected_p5_count: int,
    expected_p6_count: int,
) -> None:
    failures = _input_invariant_failures(
        public_samples, p1_samples, p5_samples, p6_samples
    )
    expected_counts = (
        ("catalog products", len(products), expected_catalog_count),
        ("public samples", len(public_samples), expected_public_count),
        ("P1 samples", len(p1_samples), expected_p1_count),
        ("P5 samples", len(p5_samples), expected_p5_count),
        ("P6 samples", len(p6_samples), expected_p6_count),
    )
    for label, actual, expected in expected_counts:
        if actual != expected:
            failures.append(f"{label} {actual} != expected {expected}")

    excluded_targets = set().union(
        *_input_target_sets(
            public_samples, p1_samples, p5_samples, p6_samples
        ).values()
    )
    missing = excluded_targets - set(products)
    if missing:
        failures.append(f"{len(missing)} excluded targets are missing from catalog")
    if failures:
        raise ValueError("invalid frozen P7 inputs: " + "; ".join(failures))


def build_and_write_p7_selection_corpus(
    catalog_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
    p6_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
    expected_catalog_count: int = OFFICIAL_CATALOG_COUNT,
    expected_public_count: int = OFFICIAL_PUBLIC_COUNT,
    expected_p1_count: int = PRIOR_DERIVED_COUNT,
    expected_p5_count: int = PRIOR_DERIVED_COUNT,
    expected_p6_count: int = PRIOR_DERIVED_COUNT,
    expected_catalog_sha256: str = CATALOG_FROZEN_SHA256,
    expected_public_git_blob_sha1: str = PUBLIC_FROZEN_GIT_BLOB_SHA1,
    expected_p1_samples_sha256: str = P1_FROZEN_SAMPLES_SHA256,
    expected_p5_samples_sha256: str = P5_FROZEN_SAMPLES_SHA256,
    expected_p6_samples_sha256: str = P6_FROZEN_SAMPLES_SHA256,
    expected_output_sha256: str | None = P7_FROZEN_SHA256,
) -> dict[str, Any]:
    """Validate four frozen inputs and write canonical P7 JSONL plus metadata."""

    input_paths = {
        "released_public": public_path,
        "prior_p1_derived": p1_path,
        "prior_p5_derived": p5_path,
        "prior_p6_derived": p6_path,
    }
    resolved_inputs = {catalog_path.resolve()} | {
        path.resolve() for path in input_paths.values()
    }
    resolved_outputs = {output_path.resolve(), metadata_path.resolve()}
    if len(resolved_outputs) != 2:
        raise ValueError("sample and metadata outputs must be different files")
    if resolved_inputs & resolved_outputs:
        raise ValueError("P7 outputs must not overwrite a frozen input file")

    actual_catalog_sha256 = _file_sha256(catalog_path)
    actual_hashes = {name: _file_sha256(path) for name, path in input_paths.items()}
    actual_public_blob = git_blob_sha1(public_path)
    hash_failures = []
    if actual_catalog_sha256 != expected_catalog_sha256.lower():
        hash_failures.append(
            "catalog frozen SHA-256 mismatch: "
            f"{actual_catalog_sha256} != expected {expected_catalog_sha256.lower()}"
        )
    if actual_public_blob != expected_public_git_blob_sha1.lower():
        hash_failures.append(
            "released_public normalized Git blob mismatch: "
            f"{actual_public_blob} != expected {expected_public_git_blob_sha1.lower()}"
        )
    if hash_failures:
        raise ValueError("; ".join(hash_failures))

    public_samples = load_jsonl(public_path)
    p1_samples = load_jsonl(p1_path)
    p5_samples = load_jsonl(p5_path)
    p6_samples = load_jsonl(p6_path)
    rows_by_name = {
        "prior_p1_derived": p1_samples,
        "prior_p5_derived": p5_samples,
        "prior_p6_derived": p6_samples,
    }
    expected_sample_hashes = {
        "prior_p1_derived": expected_p1_samples_sha256.lower(),
        "prior_p5_derived": expected_p5_samples_sha256.lower(),
        "prior_p6_derived": expected_p6_samples_sha256.lower(),
    }
    actual_sample_hashes = {
        name: _samples_sha256(rows) for name, rows in rows_by_name.items()
    }
    sample_hash_failures = [
        f"{name} frozen canonical sample SHA-256 mismatch: "
        f"{actual_sample_hashes[name]} != expected {expected}"
        for name, expected in expected_sample_hashes.items()
        if actual_sample_hashes[name] != expected
    ]
    if sample_hash_failures:
        raise ValueError("; ".join(sample_hash_failures))
    _, _, products = catalog_index(catalog_path)
    _validate_frozen_inputs(
        public_samples,
        p1_samples,
        p5_samples,
        p6_samples,
        products,
        expected_catalog_count=expected_catalog_count,
        expected_public_count=expected_public_count,
        expected_p1_count=expected_p1_count,
        expected_p5_count=expected_p5_count,
        expected_p6_count=expected_p6_count,
    )
    samples, metadata = build_p7_selection_corpus(
        public_samples, p1_samples, p5_samples, p6_samples, products, count, seed
    )
    if (
        expected_output_sha256 is not None
        and metadata["samples_sha256"] != expected_output_sha256.lower()
    ):
        raise ValueError(
            "P7 frozen output SHA-256 mismatch: "
            f"{metadata['samples_sha256']} != expected "
            f"{expected_output_sha256.lower()}"
        )

    metadata["catalog_source"] = {
        **metadata["catalog_source"],
        "path": str(catalog_path),
        "sha256": actual_catalog_sha256,
        "expected_frozen_sha256": expected_catalog_sha256.lower(),
        "frozen_sha256_verified": True,
        "expected_count_verified": len(products) == expected_catalog_count,
    }
    rows = _input_rows(public_samples, p1_samples, p5_samples, p6_samples)
    metadata["input_sources"] = {
        name: {
            "path": str(input_paths[name]),
            "sha256": actual_hashes[name],
            "sample_count": len(rows[name]),
            "unique_target_count": len(_target_ids(rows[name])),
        }
        for name in input_paths
    }
    metadata["input_sources"]["released_public"].update(
        {
            "git_blob_sha1_lf": actual_public_blob,
            "expected_frozen_git_blob_sha1_lf": (
                expected_public_git_blob_sha1.lower()
            ),
            "frozen_git_blob_verified": True,
        }
    )
    reference_raw_hashes = {
        "prior_p1_derived": P1_FROZEN_SHA256,
        "prior_p5_derived": P5_FROZEN_SHA256,
        "prior_p6_derived": P6_FROZEN_SHA256,
    }
    for name in expected_sample_hashes:
        metadata["input_sources"][name].update(
            {
                "reference_raw_file_sha256": reference_raw_hashes[name],
                "raw_file_matches_reference": (
                    actual_hashes[name] == reference_raw_hashes[name]
                ),
                "canonical_samples_sha256": actual_sample_hashes[name],
                "expected_frozen_samples_sha256": expected_sample_hashes[name],
                "frozen_samples_sha256_verified": True,
            }
        )
    metadata["outputs"] = {
        "samples_path": str(output_path),
        "metadata_path": str(metadata_path),
        "expected_frozen_samples_sha256": (
            expected_output_sha256.lower()
            if expected_output_sha256 is not None
            else None
        ),
        "frozen_samples_sha256_verified": expected_output_sha256 is not None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_jsonl_bytes(samples))
    written_hash = _file_sha256(output_path)
    if written_hash != metadata["samples_sha256"]:
        raise RuntimeError("written P7 JSONL hash does not match canonical sample hash")
    metadata["outputs"]["samples_file_sha256"] = written_hash
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen P7 catalog-derived selection corpus while excluding "
            "released-public, P1-derived, P5-derived, and P6-derived targets."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--prior-p1", type=Path, default=DEFAULT_P1_PATH)
    parser.add_argument("--prior-p5", type=Path, default=DEFAULT_P5_PATH)
    parser.add_argument("--prior-p6", type=Path, default=DEFAULT_P6_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = build_and_write_p7_selection_corpus(
        args.catalog,
        args.public,
        args.prior_p1,
        args.prior_p5,
        args.prior_p6,
        args.output,
        args.metadata_output,
        count=args.count,
        seed=args.seed,
    )
    overlaps = metadata["target_overlaps"]
    print(
        "[p7-selection] "
        f"samples={metadata['sample_count']} "
        f"public_overlap={overlaps['released_public']} "
        f"p1_overlap={overlaps['prior_p1_derived']} "
        f"p5_overlap={overlaps['prior_p5_derived']} "
        f"p6_overlap={overlaps['prior_p6_derived']} "
        f"sha256={metadata['samples_sha256']}",
        flush=True,
    )
    print(
        f"[p7-selection] wrote {args.output} and {args.metadata_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
