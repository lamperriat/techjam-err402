from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generate_evaluation_set import (
    SCENARIO_DIFFICULTY,
    load_catalog_categories,
    load_public_data,
    load_record_groups,
    scenario_labels,
    user_profile,
    weighted_unique_sample,
)


SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    priority = sorted(
        ratios,
        key=lambda name: (raw_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def stratified_product_partition(
    parent_asins: list[str],
    categories: dict[str, str],
    seed: int,
) -> dict[str, list[str]]:
    """Assign every product to one category-stratified split."""
    if len(parent_asins) != len(set(parent_asins)):
        raise ValueError("product pool must contain unique parent ASINs")
    strata: dict[str, list[str]] = defaultdict(list)
    for parent_asin in parent_asins:
        strata[categories[parent_asin]].append(parent_asin)

    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_RATIOS}
    for category in sorted(strata):
        products = strata[category]
        rng.shuffle(products)
        counts = _allocate_counts(len(products), SPLIT_RATIOS)
        start = 0
        for name in SPLIT_RATIOS:
            end = start + counts[name]
            splits[name].extend(products[start:end])
            start = end
    for products in splits.values():
        rng.shuffle(products)
    return splits


def balanced_session_targets(
    parent_asins: list[str],
    session_count: int,
    rng: random.Random,
) -> list[str]:
    """Repeat products uniformly; per-product counts differ by at most one."""
    if not parent_asins:
        raise ValueError("each split must contain at least one product")
    if session_count < len(parent_asins):
        raise ValueError("session count cannot be smaller than split product count")
    repetitions, remainder = divmod(session_count, len(parent_asins))
    targets = parent_asins * repetitions
    targets.extend(rng.sample(parent_asins, remainder))
    rng.shuffle(targets)
    return targets


def generate_product_disjoint_splits(
    records_path: Path,
    catalog_path: Path,
    public_set_path: Path,
    sample_count: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict[str, list[dict]], dict]:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    catalog_categories = load_catalog_categories(catalog_path)
    public_targets, tag_templates = load_public_data(public_set_path)
    groups = load_record_groups(
        records_path,
        set(catalog_categories),
        public_targets,
        show_progress=show_progress,
    )
    if not groups:
        raise ValueError("no eligible products found")

    rng = random.Random(seed)
    product_count = min(sample_count, len(groups))
    if product_count == len(groups):
        selected_products = sorted(groups)
    else:
        selected_products = weighted_unique_sample(groups, product_count, rng)
    product_splits = stratified_product_partition(
        selected_products,
        catalog_categories,
        seed + 1,
    )
    session_counts = (
        {name: len(products) for name, products in product_splits.items()}
        if sample_count == len(selected_products)
        else _allocate_counts(sample_count, SPLIT_RATIOS)
    )

    splits: dict[str, list[dict]] = {}
    for split_index, (name, products) in enumerate(product_splits.items()):
        split_rng = random.Random(seed + 100 + split_index)
        targets = balanced_session_targets(products, session_counts[name], split_rng)
        scenarios = scenario_labels(session_counts[name], split_rng)
        samples: list[dict] = []
        for index, (parent_asin, scenario) in enumerate(
            tqdm(
                zip(targets, scenarios, strict=True),
                total=session_counts[name],
                desc=f"Generating {name}",
                unit="session",
                disable=not show_progress,
            ),
            start=1,
        ):
            record = split_rng.choice(groups[parent_asin])
            samples.append(
                {
                    "category_bucket": catalog_categories[parent_asin],
                    "difficulty_bucket": SCENARIO_DIFFICULTY[scenario],
                    "ground_truth": {"parent_asin": parent_asin},
                    "sample_id": f"v1_{name}_{index:05d}",
                    "scenario_type": scenario,
                    "user_profile": user_profile(record, tag_templates, split_rng),
                }
            )
        splits[name] = samples

    metadata = {
        "eligible_product_count": len(groups),
        "selected_product_count": len(selected_products),
        "split_product_counts": {
            name: len(products) for name, products in product_splits.items()
        },
        "balanced_repeated_targets": sample_count > len(selected_products),
    }
    return splits, metadata


def _distribution(samples: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(sample[field]) for sample in samples).items()))


def _split_summary(samples: list[dict]) -> dict:
    target_counts = Counter(
        str(sample["ground_truth"]["parent_asin"]) for sample in samples
    )
    return {
        "sample_count": len(samples),
        "unique_product_count": len(target_counts),
        "sessions_per_product_min": min(target_counts.values()),
        "sessions_per_product_max": max(target_counts.values()),
        "scenario_distribution": _distribution(samples, "scenario_type"),
        "category_distribution": _distribution(samples, "category_bucket"),
    }


def _write_split(path: Path, samples: list[dict], show_progress: bool) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for sample in tqdm(
            samples,
            desc=f"Writing {path.stem}",
            unit="session",
            disable=not show_progress,
        ):
            handle.write(json.dumps(sample, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate product-disjoint benchmark tuning splits."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=PROJECT_ROOT / "local-data/valid_records.csv",
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--public-set",
        type=Path,
        default=PROJECT_ROOT / "data/public_set.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "local-data/v1_tuning",
    )
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate at most 100 sessions in memory and do not write files.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_count = min(args.samples, 100) if args.dry_run else args.samples
    splits, metadata = generate_product_disjoint_splits(
        records_path=args.records,
        catalog_path=args.catalog,
        public_set_path=args.public_set,
        sample_count=sample_count,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    manifest = {
        "schema_version": 2,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "requested_sample_count": args.samples,
        "generated_sample_count": sum(map(len, splits.values())),
        "split_ratios": SPLIT_RATIOS,
        "product_disjoint": True,
        "public_targets_excluded": True,
        **metadata,
        "splits": {name: _split_summary(split) for name, split in splits.items()},
    }
    if not args.dry_run:
        outputs = [
            *(args.output_dir / f"{name}.jsonl" for name in splits),
            args.output_dir / "manifest.json",
        ]
        occupied = [path for path in outputs if path.exists()]
        if occupied:
            raise FileExistsError(f"refusing to overwrite existing output: {occupied[0]}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            _write_split(args.output_dir / f"{name}.jsonl", split, not args.quiet)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
