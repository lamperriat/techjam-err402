from __future__ import annotations

"""Build the frozen P5 selection corpus without changing the P1 generator.

P5 reuses the deterministic catalog-derived generator, but excludes both released-
public targets and every target already used by the frozen P1 derived corpus.
"""

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
    build_product_disjoint_samples,
)


SCHEMA_VERSION = "p5.selection-corpus.v1"
DEFAULT_COUNT = 200
DEFAULT_SEED = "track4-p5-product-disjoint-v1"
DEFAULT_OUTPUT = Path("experiments/p5_selection_product_disjoint.jsonl")
DEFAULT_METADATA_OUTPUT = Path(
    "experiments/p5_selection_product_disjoint.metadata.json"
)
OFFICIAL_CATALOG_COUNT = 50_000
OFFICIAL_PUBLIC_COUNT = 200
PRIOR_DERIVED_COUNT = 200
SAMPLE_ID_PREFIX = "derived_p5_"


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


def _samples_payload(samples: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            sample,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for sample in samples
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_p5_selection_corpus(
    public_samples: list[dict[str, Any]],
    prior_derived_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic P5 samples and target-overlap audit metadata."""

    public_targets = _target_ids(public_samples)
    prior_targets = _target_ids(prior_derived_samples)
    exclusion_samples = [*public_samples, *prior_derived_samples]
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
    public_overlap = selected_targets & public_targets
    prior_overlap = selected_targets & prior_targets
    if public_overlap or prior_overlap:
        raise RuntimeError("P5 target exclusion invariant failed")

    payload = _samples_payload(samples).encode("utf-8")
    scenario_counts = dict(
        sorted(Counter(str(sample["scenario_type"]) for sample in samples).items())
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "p5_selection_product_disjoint",
        "seed": seed,
        "sample_count": len(samples),
        "sample_id_prefix": SAMPLE_ID_PREFIX,
        "samples_sha256": _sha256_bytes(payload),
        "unique_target_count": len(selected_targets),
        "scenario_counts": scenario_counts,
        "public_target_overlap": len(public_overlap),
        "prior_derived_target_overlap": len(prior_overlap),
        "exclusions": {
            "released_public_target_count": len(public_targets),
            "prior_p1_derived_target_count": len(prior_targets),
            "combined_unique_target_count": len(public_targets | prior_targets),
            "public_prior_input_overlap": len(public_targets & prior_targets),
            "selected_public_target_overlap": len(public_overlap),
            "selected_prior_derived_target_overlap": len(prior_overlap),
        },
        "catalog_source": {
            "description": "official frozen 50,000-product participant catalog",
            "expected_product_count": OFFICIAL_CATALOG_COUNT,
            "loaded_product_count": len(products),
        },
        "generator": {
            "base": (
                "scripts.evaluate_generalization.build_product_disjoint_samples"
            ),
            "base_selection": base_metadata.get("selection"),
            "p1_generator_modified": False,
            "selection": (
                "SHA-256(seed + NUL + parent_asin), excluding the union of released-"
                "public and prior P1-derived targets; requires non-empty title and "
                "categories"
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
    prior_derived_samples: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    *,
    expected_catalog_count: int,
    expected_public_count: int,
    expected_prior_count: int,
) -> None:
    public_targets = _target_ids(public_samples)
    prior_targets = _target_ids(prior_derived_samples)
    failures: list[str] = []
    if len(products) != expected_catalog_count:
        failures.append(
            f"catalog products {len(products)} != expected {expected_catalog_count}"
        )
    if len(public_samples) != expected_public_count:
        failures.append(
            f"public samples {len(public_samples)} != expected {expected_public_count}"
        )
    if len(public_targets) != expected_public_count:
        failures.append(
            f"public unique targets {len(public_targets)} != expected {expected_public_count}"
        )
    if len(prior_derived_samples) != expected_prior_count:
        failures.append(
            "prior derived samples "
            f"{len(prior_derived_samples)} != expected {expected_prior_count}"
        )
    if len(prior_targets) != expected_prior_count:
        failures.append(
            "prior derived unique targets "
            f"{len(prior_targets)} != expected {expected_prior_count}"
        )
    if public_targets & prior_targets:
        failures.append("prior P1-derived targets overlap released-public targets")
    catalog_targets = set(products)
    missing = (public_targets | prior_targets) - catalog_targets
    if missing:
        failures.append(f"{len(missing)} excluded targets are missing from catalog")
    invalid_prior_ids = [
        str(sample.get("sample_id", ""))
        for sample in prior_derived_samples
        if not str(sample.get("sample_id", "")).startswith("derived_p1_")
    ]
    if invalid_prior_ids:
        failures.append("prior corpus contains non-derived_p1 sample IDs")
    if failures:
        raise ValueError("invalid frozen P5 inputs: " + "; ".join(failures))


def build_and_write_p5_selection_corpus(
    catalog_path: Path,
    public_path: Path,
    prior_derived_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    count: int = DEFAULT_COUNT,
    seed: str = DEFAULT_SEED,
    expected_catalog_count: int = OFFICIAL_CATALOG_COUNT,
    expected_public_count: int = OFFICIAL_PUBLIC_COUNT,
    expected_prior_count: int = PRIOR_DERIVED_COUNT,
) -> dict[str, Any]:
    """Load frozen inputs, validate them, and write deterministic P5 artifacts."""

    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("sample and metadata outputs must be different files")
    public_samples = load_jsonl(public_path)
    prior_samples = load_jsonl(prior_derived_path)
    _, _, products = catalog_index(catalog_path)
    _validate_frozen_inputs(
        public_samples,
        prior_samples,
        products,
        expected_catalog_count=expected_catalog_count,
        expected_public_count=expected_public_count,
        expected_prior_count=expected_prior_count,
    )
    samples, metadata = build_p5_selection_corpus(
        public_samples,
        prior_samples,
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
    metadata["input_sources"] = {
        "released_public": {
            "path": str(public_path),
            "sha256": _file_sha256(public_path),
            "sample_count": len(public_samples),
            "unique_target_count": len(_target_ids(public_samples)),
        },
        "prior_p1_derived": {
            "path": str(prior_derived_path),
            "sha256": _file_sha256(prior_derived_path),
            "sample_count": len(prior_samples),
            "unique_target_count": len(_target_ids(prior_samples)),
        },
    }
    metadata["outputs"] = {
        "samples_path": str(output_path),
        "metadata_path": str(metadata_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_samples_payload(samples).encode("utf-8"))
    written_hash = _file_sha256(output_path)
    if written_hash != metadata["samples_sha256"]:
        raise RuntimeError("written P5 JSONL hash does not match canonical sample hash")
    metadata["outputs"]["samples_file_sha256"] = written_hash
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen P5 catalog-derived selection corpus while excluding "
            "released-public and prior P1-derived targets."
        )
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--prior-derived",
        type=Path,
        default=Path("experiments/p1_derived_product_disjoint.jsonl"),
    )
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
    metadata = build_and_write_p5_selection_corpus(
        args.catalog,
        args.public,
        args.prior_derived,
        args.output,
        args.metadata_output,
        count=args.count,
        seed=args.seed,
    )
    print(
        "[p5-selection] "
        f"samples={metadata['sample_count']} "
        f"public_overlap={metadata['public_target_overlap']} "
        f"prior_overlap={metadata['prior_derived_target_overlap']} "
        f"sha256={metadata['samples_sha256']}",
        flush=True,
    )
    print(
        f"[p5-selection] wrote {args.output} and {args.metadata_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
