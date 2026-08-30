"""Profile the category hierarchy stored in the catalog JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# These are the conventional shopper departments present in this catalog.
# Other level-two labels are reported separately rather than classified implicitly.
DEPARTMENT_LABELS = {"Baby", "Boys", "Girls", "Men", "Women"}


def _ranked(counter: Counter[str], limit: int | None = None) -> list[dict]:
    items = counter.most_common(limit)
    return [{"label": label, "products": count} for label, count in items]


def profile_catalog_categories(catalog_path: Path, top: int = 10) -> dict:
    """Return label and taxonomy-node counts at every category depth."""
    if top <= 0:
        raise ValueError("top must be positive")

    product_count = 0
    missing_category_count = 0
    depth_counts: Counter[int] = Counter()
    labels_by_level: defaultdict[int, Counter[str]] = defaultdict(Counter)
    nodes_by_level: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)
    leaf_labels: Counter[str] = Counter()
    leaf_nodes: set[tuple[str, ...]] = set()
    penultimate_labels: Counter[str] = Counter()
    penultimate_nodes: set[tuple[str, ...]] = set()

    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product_count += 1
            raw_categories = json.loads(line).get("categories")
            if not isinstance(raw_categories, list) or not raw_categories:
                missing_category_count += 1
                continue

            categories = tuple(str(label).strip() for label in raw_categories)
            depth_counts[len(categories)] += 1
            for index, label in enumerate(categories, start=1):
                labels_by_level[index][label] += 1
                nodes_by_level[index].add(categories[:index])

            leaf_labels[categories[-1]] += 1
            leaf_nodes.add(categories)
            if len(categories) >= 2:
                penultimate_labels[categories[-2]] += 1
                penultimate_nodes.add(categories[:-1])

    second_level = labels_by_level[2]
    department_product_count = sum(
        second_level[label] for label in DEPARTMENT_LABELS
    )
    products_with_second_level = second_level.total()
    non_department = Counter(
        {
            label: count
            for label, count in second_level.items()
            if label not in DEPARTMENT_LABELS
        }
    )

    return {
        "catalog": str(catalog_path),
        "products": product_count,
        "products_without_categories": missing_category_count,
        "path_depth_distribution": dict(sorted(depth_counts.items())),
        "unique_taxonomy_nodes": sum(len(nodes) for nodes in nodes_by_level.values()),
        "levels": [
            {
                "depth": depth,
                "unique_labels": len(labels_by_level[depth]),
                "unique_nodes": len(nodes_by_level[depth]),
                "top_labels": _ranked(labels_by_level[depth], top),
            }
            for depth in sorted(labels_by_level)
        ],
        "finest_level": {
            "unique_labels": len(leaf_labels),
            "unique_nodes": len(leaf_nodes),
            "top_labels": _ranked(leaf_labels, top),
        },
        "second_last_level": {
            "unique_labels": len(penultimate_labels),
            "unique_nodes": len(penultimate_nodes),
            "top_labels": _ranked(penultimate_labels, top),
        },
        "second_level_department_audit": {
            "department_definition": sorted(DEPARTMENT_LABELS),
            "products_with_second_level": products_with_second_level,
            "department_products": department_product_count,
            "department_fraction": (
                department_product_count / products_with_second_level
                if products_with_second_level
                else 0.0
            ),
            "is_always_department": not non_department,
            "non_department_unique_labels": len(non_department),
            "top_non_department_labels": _ranked(non_department, top),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog.jsonl"),
        help="catalog JSONL path (default: data/catalog.jsonl)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of most common labels to show per section (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            profile_catalog_categories(arguments.catalog, arguments.top),
            indent=2,
        )
    )
