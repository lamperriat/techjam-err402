from __future__ import annotations

from collections.abc import Sequence


SCHEMA_VERSION = "small-ranker-versioned-pagination-runtime.v1"
GRACE_PAGES = 2
PAGE_SIZE = 10


class VersionedPaginationError(RuntimeError):
    pass


def fixed_two_page_grace_order(
    order: Sequence[str],
    served: set[str],
    intent_age: int,
    top_k: int = PAGE_SIZE,
) -> tuple[str, ...]:
    """Return the full order for the frozen two-page-grace policy."""

    ranked = tuple(order)
    if intent_age <= 0:
        raise VersionedPaginationError("intent age must be positive")
    if top_k <= 0 or len(ranked) < top_k or len(ranked) != len(set(ranked)):
        raise VersionedPaginationError("invalid ranked order")
    if intent_age <= GRACE_PAGES:
        return ranked
    unseen = tuple(identifier for identifier in ranked if identifier not in served)
    seen = tuple(identifier for identifier in ranked if identifier in served)
    proposed = unseen + seen
    if len(proposed) != len(ranked) or set(proposed) != set(ranked):
        raise VersionedPaginationError("pagination changed ranked membership")
    if len(proposed[:top_k]) != len(set(proposed[:top_k])):
        raise VersionedPaginationError("pagination produced a duplicate page")
    return proposed
