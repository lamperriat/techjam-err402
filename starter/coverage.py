"""Deterministic query-term coverage ordering promoted from Architecture Lab R08."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "p4.coverage-cascade.v1"


def order_by_query_coverage(
    query_terms: Iterable[str],
    identifiers: Sequence[str],
    searchable_fields: Mapping[str, Sequence[str]],
    tokenize: Callable[[str], Iterable[str]],
) -> tuple[list[str], dict[str, Any]]:
    """Order by matched visible query-term count, preserving fused rank on ties."""

    terms = set(query_terms)
    original = list(identifiers)
    original_rank = {
        identifier: rank for rank, identifier in enumerate(original, start=1)
    }
    coverage = {
        identifier: len(
            terms
            & set(tokenize(" ".join(searchable_fields.get(identifier, ()))))
        )
        for identifier in original
    }
    ordered = sorted(
        original,
        key=lambda identifier: (
            -coverage.get(identifier, 0),
            original_rank[identifier],
        ),
    )
    histogram: dict[str, int] = {}
    for value in coverage.values():
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return ordered, {
        "schema_version": SCHEMA_VERSION,
        "query_term_count": len(terms),
        "candidate_count": len(original),
        "covered_candidate_count": sum(value > 0 for value in coverage.values()),
        "maximum_coverage": max(coverage.values(), default=0),
        "coverage_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
        "changed_top_10": ordered[:10] != original[:10],
    }
