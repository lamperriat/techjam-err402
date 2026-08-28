from __future__ import annotations

"""Build the frozen P6 selection corpus from a third disjoint product set."""

import argparse
import hashlib
import json
import sys
from collections import Counter
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


SCHEMA_VERSION = "p6.selection-corpus.v1"
DEFAULT_COUNT = 200
DEFAULT_SEED = "track4-p6-product-disjoint-v1"
DEFAULT_OUTPUT = Path("experiments/p6_selection_product_disjoint.jsonl")
DEFAULT_METADATA_OUTPUT = Path(
    "experiments/p6_selection_product_disjoint.metadata.json"
)
DEFAULT_P1_PATH = Path("experiments/p1_derived_product_disjoint.jsonl")
DEFAULT_P5_PATH = Path("experiments/p5_selection_product_disjoint.jsonl")
OFFICIAL_CATALOG_COUNT = 50_000
OFFICIAL_PUBLIC_COUNT = 200
PRIOR_DERIVED_COUNT = 200
P5_FROZEN_SHA256 = (
    "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c"
)
P1_FROZEN_SAMPLES_SHA256 = (
    "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
)
SAMPLE_ID_PREFIX = "derived_p6_"
EXPECTED_SCENARIO_COUNTS = {
    "boundary": 10,
    "browsing": 80,
    "buying": 80,
    "intent_override": 30,
}


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


def _input_target_sets(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
) -> dict[str, set[str]]:
    return {
        "released_public": _target_ids(public_samples),
        "prior_p1_derived": _target_ids(p1_samples),
        "prior_p5_derived": _target_ids(p5_samples),
    }


def _input_invariant_failures(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
) -> list[str]:
    rows = {
        "released public": public_samples,
        "prior P1-derived": p1_samples,
        "prior P5-derived": p5_samples,
    }
    targets = _input_target_sets(public_samples, p1_samples, p5_samples)
    failures = [
        f"{name} targets are missing or duplicated"
        for name, samples in rows.items()
        if len(_target_ids(samples)) != len(samples)
    ]
    for name, samples, prefix in (
        ("P1", p1_samples, "derived_p1_"),
        ("P5", p5_samples, "derived_p5_"),
    ):
        sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
        if len(set(sample_ids)) != len(sample_ids) or any(
            not sample_id.startswith(prefix) for sample_id in sample_ids
        ):
            failures.append(f"prior {name} corpus has invalid or duplicate sample IDs")

    pairs = (
        ("public/P1", "released_public", "prior_p1_derived"),
        ("public/P5", "released_public", "prior_p5_derived"),
        ("P1/P5", "prior_p1_derived", "prior_p5_derived"),
    )
    for label, left, right in pairs:
        if targets[left] & targets[right]:
            failures.append(f"{label} input targets overlap")
    return failures


def build_p6_selection_corpus(
    public_samples: list[dict[str, Any]],
    p1_samples: list[dict[str, Any]],
    p5_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic samples outside public, P1, and P5 target sets."""

    failures = _input_invariant_failures(public_samples, p1_samples, p5_samples)
    if failures:
        raise ValueError("invalid P6 exclusion inputs: " + "; ".join(failures))

    target_sets = _input_target_sets(public_samples, p1_samples, p5_samples)
    exclusion_samples = [*public_samples, *p1_samples, *p5_samples]
    generated, base_metadata = build_product_disjoint_samples(
        exclusion_samples,
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
        raise RuntimeError("P6 target uniqueness or exclusion invariant failed")

    scenario_counts = dict(
        sorted(Counter(str(sample["scenario_type"]) for sample in samples).items())
    )
    if count == DEFAULT_COUNT and scenario_counts != EXPECTED_SCENARIO_COUNTS:
        raise RuntimeError("P6 default scenario-mix invariant failed")

    payload = _canonical_jsonl_bytes(samples)
    union_targets = set().union(*target_sets.values())
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "p6_selection_product_disjoint",
        "seed": seed,
        "sample_count": len(samples),
        "sample_id_prefix": SAMPLE_ID_PREFIX,
        "samples_sha256": _sha256_bytes(payload),
        "canonical_serialization": (
            "UTF-8 JSON Lines; object keys sorted; compact separators; LF after every row"
        ),
        "unique_target_count": len(selected_targets),
        "scenario_counts": scenario_counts,
        "public_target_overlap": len(overlaps["released_public"]),
        "prior_p1_target_overlap": len(overlaps["prior_p1_derived"]),
        "prior_p5_target_overlap": len(overlaps["prior_p5_derived"]),
        "exclusions": {
            "released_public_target_count": len(target_sets["released_public"]),
            "prior_p1_derived_target_count": len(target_sets["prior_p1_derived"]),
            "prior_p5_derived_target_count": len(target_sets["prior_p5_derived"]),
            "combined_unique_target_count": len(union_targets),
            "public_p1_input_overlap": len(
                target_sets["released_public"] & target_sets["prior_p1_derived"]
            ),
            "public_p5_input_overlap": len(
                target_sets["released_public"] & target_sets["prior_p5_derived"]
            ),
            "p1_p5_input_overlap": len(
                target_sets["prior_p1_derived"] & target_sets["prior_p5_derived"]
            ),
            "selected_public_target_overlap": len(overlaps["released_public"]),
            "selected_p1_target_overlap": len(overlaps["prior_p1_derived"]),
            "selected_p5_target_overlap": len(overlaps["prior_p5_derived"]),
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
                "SHA-256(seed + NUL + parent_asin), excluding the union of released-"
                "public, prior P1-derived, and prior P5-derived targets; requires "
                "non-empty title and categories"
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
    products: dict[str, dict[str, Any]],
    *,
    expected_catalog_count: int,
    expected_public_count: int,
    expected_p1_count: int,
    expected_p5_count: int,
) -> None:
    failures = _input_invariant_failures(public_samples, p1_samples, p5_samples)
    expected_counts = (
        ("catalog products", len(products), expected_catalog_count),
        ("public samples", len(public_samples), expected_public_count),
        ("P1 samples", len(p1_samples), expected_p1_count),
        ("P5 samples", len(p5_samples), expected_p5_count),
    )
    for label, actual, expected in expected_counts:
        if actual != expected:
            failures.append(f"{label} {actual} != expected {expected}")

    excluded_targets = set().union(
        *_input_target_sets(public_samples, p1_samples, p5_samples).values()
    )
    missing = excluded_targets - set(products)
    if missing:
        failures.append(f"{len(missing)} excluded targets are missing from catalog")
    if failures:
        raise ValueError("invalid frozen P6 inputs: " + "; ".join(failures))


def build_and_write_p6_selection_corpus(
    catalog_path: Path,
    public_path: Path,
    p1_path: Path,
    p5_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
    expected_catalog_count: int = OFFICIAL_CATALOG_COUNT,
    expected_public_count: int = OFFICIAL_PUBLIC_COUNT,
    expected_p1_count: int = PRIOR_DERIVED_COUNT,
    expected_p5_count: int = PRIOR_DERIVED_COUNT,
    expected_p1_samples_sha256: str = P1_FROZEN_SAMPLES_SHA256,
    expected_p5_sha256: str = P5_FROZEN_SHA256,
) -> dict[str, Any]:
    """Validate frozen inputs and write canonical P6 JSONL plus audit metadata."""

    resolved_inputs = {
        path.resolve()
        for path in (catalog_path, public_path, p1_path, p5_path)
    }
    resolved_outputs = {output_path.resolve(), metadata_path.resolve()}
    if len(resolved_outputs) != 2:
        raise ValueError("sample and metadata outputs must be different files")
    if resolved_inputs & resolved_outputs:
        raise ValueError("P6 outputs must not overwrite a frozen input file")
    actual_p5_sha256 = _file_sha256(p5_path)
    if actual_p5_sha256 != expected_p5_sha256.lower():
        raise ValueError(
            "P5 frozen SHA-256 mismatch: "
            f"{actual_p5_sha256} != expected {expected_p5_sha256.lower()}"
        )

    public_samples = load_jsonl(public_path)
    p1_samples = load_jsonl(p1_path)
    p5_samples = load_jsonl(p5_path)
    actual_p1_samples_sha256 = _samples_sha256(p1_samples)
    if actual_p1_samples_sha256 != expected_p1_samples_sha256.lower():
        raise ValueError(
            "P1 frozen sample SHA-256 mismatch: "
            f"{actual_p1_samples_sha256} != expected "
            f"{expected_p1_samples_sha256.lower()}"
        )
    _, _, products = catalog_index(catalog_path)
    _validate_frozen_inputs(
        public_samples,
        p1_samples,
        p5_samples,
        products,
        expected_catalog_count=expected_catalog_count,
        expected_public_count=expected_public_count,
        expected_p1_count=expected_p1_count,
        expected_p5_count=expected_p5_count,
    )
    samples, metadata = build_p6_selection_corpus(
        public_samples,
        p1_samples,
        p5_samples,
        products,
        count,
        seed,
    )

    metadata["catalog_source"] = {
        **metadata["catalog_source"],
        "path": str(catalog_path),
        "sha256": _file_sha256(catalog_path),
        "expected_count_verified": len(products) == expected_catalog_count,
    }
    input_specs = (
        ("released_public", public_path, public_samples),
        ("prior_p1_derived", p1_path, p1_samples),
        ("prior_p5_derived", p5_path, p5_samples),
    )
    metadata["input_sources"] = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
            "sample_count": len(rows),
            "unique_target_count": len(_target_ids(rows)),
        }
        for name, path, rows in input_specs
    }
    metadata["input_sources"]["prior_p5_derived"].update(
        {
            "expected_frozen_sha256": expected_p5_sha256.lower(),
            "frozen_sha256_verified": True,
        }
    )
    metadata["input_sources"]["prior_p1_derived"].update(
        {
            "canonical_samples_sha256": actual_p1_samples_sha256,
            "expected_frozen_samples_sha256": expected_p1_samples_sha256.lower(),
            "frozen_samples_sha256_verified": True,
        }
    )
    metadata["outputs"] = {
        "samples_path": str(output_path),
        "metadata_path": str(metadata_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_jsonl_bytes(samples))
    written_hash = _file_sha256(output_path)
    if written_hash != metadata["samples_sha256"]:
        raise RuntimeError("written P6 JSONL hash does not match canonical sample hash")
    metadata["outputs"]["samples_file_sha256"] = written_hash
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen P6 catalog-derived selection corpus while excluding "
            "released-public, P1-derived, and frozen P5-derived targets."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--prior-p1", type=Path, default=DEFAULT_P1_PATH)
    parser.add_argument("--prior-p5", type=Path, default=DEFAULT_P5_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_OUTPUT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = build_and_write_p6_selection_corpus(
        args.catalog,
        args.public,
        args.prior_p1,
        args.prior_p5,
        args.output,
        args.metadata_output,
        count=args.count,
        seed=args.seed,
    )
    print(
        "[p6-selection] "
        f"samples={metadata['sample_count']} "
        f"public_overlap={metadata['public_target_overlap']} "
        f"p1_overlap={metadata['prior_p1_target_overlap']} "
        f"p5_overlap={metadata['prior_p5_target_overlap']} "
        f"sha256={metadata['samples_sha256']}",
        flush=True,
    )
    print(
        f"[p6-selection] wrote {args.output} and {args.metadata_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
