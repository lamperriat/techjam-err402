"""Generate an expanded development set from locally filtered purchase records.

Targets follow the inferred organizer sampling process. User profiles are
approximations because metadata and ratings for most historical ASINs are not
available in the frozen catalog.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm


SCENARIO_PROPORTIONS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
SCENARIO_DIFFICULTY = {
    "buying": "easy",
    "browsing": "medium",
    "intent_override": "hard",
    "boundary": "medium",
}


def category_bucket(categories: list[str]) -> str:
    """Map a catalog path to the three development-set buckets."""
    nodes = [str(value).strip().lower() for value in categories[1:]]
    if "shoes" in nodes or "boot shop" in nodes:
        return "shoes"
    if "jewelry" in nodes or any("jewelry" in node for node in nodes):
        return "jewelry"
    return "clothing"


def load_catalog_categories(path: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            categories[str(product["parent_asin"])] = category_bucket(
                product.get("categories") or []
            )
    return categories


def load_public_data(path: Path) -> tuple[set[str], list[list[str]]]:
    excluded_asins: set[str] = set()
    tag_templates: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            excluded_asins.add(str(sample["ground_truth"]["parent_asin"]))
            tag_templates.append(list(sample["user_profile"]["preference_tags"]))
    if not tag_templates:
        raise ValueError("public set must contain at least one profile")
    return excluded_asins, tag_templates


def load_record_groups(
    path: Path,
    catalog_asins: set[str],
    excluded_asins: set[str],
    show_progress: bool = False,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"parent_asin", "rating", "history"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"records CSV must contain columns: {sorted(required)}")
        for row in tqdm(
            reader,
            desc="Reading purchase records",
            unit="record",
            disable=not show_progress,
        ):
            asin = row["parent_asin"]
            if asin in catalog_asins and asin not in excluded_asins:
                groups[asin].append(row)
    return dict(groups)


def weighted_unique_sample(
    groups: dict[str, list[dict[str, str]]],
    sample_count: int,
    rng: random.Random,
) -> list[str]:
    """Sample unique ASINs with probability proportional to record count."""
    if sample_count > len(groups):
        raise ValueError(
            f"requested {sample_count} unique products, but only {len(groups)} are eligible"
        )
    # Exponential-race keys implement sequential weighted sampling without
    # replacement while assigning exactly one key to every eligible product.
    ranked = (
        (rng.expovariate(len(records)), asin)
        for asin, records in groups.items()
    )
    return [asin for _, asin in heapq.nsmallest(sample_count, ranked)]


def scenario_labels(sample_count: int, rng: random.Random) -> list[str]:
    raw_counts = {
        scenario: sample_count * proportion
        for scenario, proportion in SCENARIO_PROPORTIONS.items()
    }
    counts = {scenario: math.floor(value) for scenario, value in raw_counts.items()}
    remaining = sample_count - sum(counts.values())
    priority = sorted(
        SCENARIO_PROPORTIONS,
        key=lambda scenario: (
            raw_counts[scenario] - counts[scenario],
            SCENARIO_PROPORTIONS[scenario],
        ),
        reverse=True,
    )
    for scenario in priority[:remaining]:
        counts[scenario] += 1

    labels = [
        scenario
        for scenario in SCENARIO_PROPORTIONS
        for _ in range(counts[scenario])
    ]
    rng.shuffle(labels)
    return labels


def purchase_frequency(history: str) -> str:
    # Match the privacy-capped history range exposed by the public profiles.
    count = min(len(history.split()), 4)
    if count == 0:
        return "no prior purchases"
    if count == 1:
        return "1 prior purchase"
    if count == 2:
        return "2 prior purchases"
    return "3-4 prior purchases"


def rating_style(rating: float) -> str:
    if rating >= 5:
        return "usually positive"
    if rating >= 4:
        return "mixed"
    return "critical"


def user_profile(
    record: dict[str, str],
    tag_templates: list[list[str]],
    rng: random.Random,
) -> dict:
    """Build an approximate safe profile without using target-product metadata."""
    rating = float(record["rating"])
    style = rating_style(rating)
    tags = list(rng.choice(tag_templates))
    tag_summary = ", ".join(tags) if tags else "general shopping"
    return {
        "average_prior_rating": rating,
        "preference_tags": tags,
        "purchase_frequency": purchase_frequency(record["history"]),
        "rating_style": style,
        "summary": f"Prior purchases emphasize {tag_summary}; ratings are {style}.",
    }


def generate_samples(
    records_path: Path,
    catalog_path: Path,
    public_set_path: Path,
    sample_count: int,
    seed: int,
    show_progress: bool = False,
) -> list[dict]:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")

    catalog_categories = load_catalog_categories(catalog_path)
    excluded_asins, tag_templates = load_public_data(public_set_path)
    groups = load_record_groups(
        records_path,
        set(catalog_categories),
        excluded_asins,
        show_progress=show_progress,
    )
    rng = random.Random(seed)
    selected_asins = weighted_unique_sample(groups, sample_count, rng)
    scenarios = scenario_labels(sample_count, rng)

    samples: list[dict] = []
    selected = enumerate(zip(selected_asins, scenarios, strict=True), start=1)
    for index, (asin, scenario) in tqdm(
        selected,
        total=sample_count,
        desc="Generating sessions",
        unit="session",
        disable=not show_progress,
    ):
        record = rng.choice(groups[asin])
        samples.append(
            {
                "category_bucket": catalog_categories[asin],
                "difficulty_bucket": SCENARIO_DIFFICULTY[scenario],
                "ground_truth": {"parent_asin": asin},
                "sample_id": f"generated_{index:04d}",
                "scenario_type": scenario,
                "user_profile": user_profile(record, tag_templates, rng),
            }
        )
    return samples


def summary(samples: list[dict], output_path: Path, seed: int) -> dict:
    return {
        "output": str(output_path),
        "seed": seed,
        "sample_count": len(samples),
        "category_distribution": dict(
            sorted(Counter(sample["category_bucket"] for sample in samples).items())
        ),
        "scenario_distribution": dict(
            sorted(Counter(sample["scenario_type"] for sample in samples).items())
        ),
        "rating_style_distribution": dict(
            sorted(
                Counter(
                    sample["user_profile"]["rating_style"] for sample in samples
                ).items()
            )
        ),
        "preference_tag_distribution": dict(
            sorted(
                Counter(
                    tag
                    for sample in samples
                    for tag in sample["user_profile"]["preference_tags"]
                ).items()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluator-compatible samples by purchase-weighted unique "
            "sampling from valid records."
        )
    )
    parser.add_argument("--records", type=Path, default=Path("local-data/valid_records.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local-data/generated_set_1000.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    samples = generate_samples(
        records_path=args.records,
        catalog_path=args.catalog,
        public_set_path=args.public_set,
        sample_count=args.samples,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    print(json.dumps(summary(samples, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
