"""Deterministic guarded admission from a deeper copy of the broad route."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "p6.adaptive-depth.v1"
BASE_BROAD = 120
DEEP = 240
TOP_K = 10
PROTECTED_PREFIX = 9
SEARCHABLE_FIELD_COUNT = 6


@dataclass(frozen=True, slots=True)
class DepthConfig:
    base_broad_depth: int = BASE_BROAD
    deep_depth: int = DEEP
    top_k: int = TOP_K
    protected_prefix: int = PROTECTED_PREFIX
    max_top10_newcomers: int = 1
    min_query_terms: int = 2
    strict_coverage_margin: int = 1

    def __post_init__(self) -> None:
        positive = {
            "base_broad_depth": self.base_broad_depth,
            "deep_depth": self.deep_depth,
            "top_k": self.top_k,
            "min_query_terms": self.min_query_terms,
            "strict_coverage_margin": self.strict_coverage_margin,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.deep_depth <= self.base_broad_depth:
            raise ValueError("deep_depth must exceed base_broad_depth")
        if self.base_broad_depth < self.top_k:
            raise ValueError("base_broad_depth must be at least top_k")
        if self.protected_prefix != self.top_k - 1:
            raise ValueError("protected_prefix must equal top_k - 1")
        if (
            isinstance(self.max_top10_newcomers, bool)
            or not isinstance(self.max_top10_newcomers, int)
            or not 0 <= self.max_top10_newcomers <= 1
        ):
            raise ValueError("max_top10_newcomers must be 0 or 1")


@dataclass(frozen=True, slots=True)
class DepthProposal:
    identifier: str
    coverage: int
    deep_rank: int
    matched_query_terms: tuple[str, ...]
    matched_excluded_terms: tuple[str, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]


def _normalized_terms(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        term = str(value).strip().lower()
        if term and term.isascii() and term.isalnum() and term not in normalized:
            normalized.append(term)
    return normalized


def _route(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def _field_text(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        fields = value[:SEARCHABLE_FIELD_COUNT]
    else:
        fields = (value,)
    return " ".join(str(field or "") for field in fields)


def _coverage_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def depth_precheck(
    query_terms: Iterable[str],
    rankings: Mapping[str, Sequence[str]],
    baseline_coverage: Mapping[str, int],
    config: DepthConfig,
) -> dict[str, Any]:
    """Check whether a deeper broad query is allowed before issuing it."""

    visible_query = _normalized_terms(query_terms)
    fts_query = [term for term in visible_query if len(term) > 1][:50]
    broad = _route(rankings.get("broad", ()))
    final = _route(rankings.get("final", ()))
    incumbent = (
        final[config.protected_prefix]
        if len(final) >= config.top_k
        else None
    )
    incumbent_coverage = (
        _coverage_value(baseline_coverage.get(incumbent))
        if incumbent is not None
        else None
    )
    conditions = {
        "query_terms_ready": len(visible_query) >= config.min_query_terms,
        "broad_saturated": len(broad) == config.base_broad_depth,
        "final_top10_ready": len(final) >= config.top_k,
        "incumbent_coverage_known": incumbent_coverage is not None,
        "incumbent_below_full_coverage": (
            incumbent_coverage is not None
            and incumbent_coverage < len(visible_query)
        ),
        "newcomers_enabled": config.max_top10_newcomers == 1,
    }
    reasons: list[str] = []
    if not conditions["query_terms_ready"]:
        reasons.append("query_term_count_below_minimum")
    if not conditions["broad_saturated"]:
        reasons.append("broad_route_not_saturated")
    if not conditions["final_top10_ready"]:
        reasons.append("final_top10_incomplete")
    elif not conditions["incumbent_coverage_known"]:
        reasons.append("incumbent_coverage_missing")
    elif (
        conditions["query_terms_ready"]
        and not conditions["incumbent_below_full_coverage"]
    ):
        reasons.append("incumbent_coverage_already_full")
    if not conditions["newcomers_enabled"]:
        reasons.append("top10_newcomers_disabled")
    return {
        "enabled": not reasons,
        "conditions": conditions,
        "rejection_reasons": reasons,
        "query_terms": visible_query,
        "fts_query_terms": fts_query,
        "visible_query_term_count": len(visible_query),
        "fts_query_term_count": len(fts_query),
        "broad_count": len(broad),
        "final_count": len(final),
        "incumbent": incumbent,
        "incumbent_coverage": incumbent_coverage,
    }


def guarded_depth_admission(
    query_terms: Iterable[str],
    excluded_terms: Iterable[str],
    rankings: Mapping[str, Sequence[str]],
    deep_ids: Sequence[str],
    baseline_coverage: Mapping[str, int],
    searchable_fields: Mapping[str, Sequence[object]],
    config: DepthConfig,
    tokenize: Callable[[str], Iterable[str]],
) -> tuple[list[str], dict[str, Any]]:
    """Admit at most one unseen deep-route item behind a protected prefix."""

    precheck = depth_precheck(
        query_terms,
        rankings,
        baseline_coverage,
        config,
    )
    query = list(precheck["query_terms"])
    # ``SessionState.excluded_terms`` is a set. Canonicalize its order so the
    # full audit remains byte-for-byte stable across independent hash seeds.
    excluded = sorted(_normalized_terms(excluded_terms))
    broad = _route(rankings.get("broad", ()))
    strict = _route(rankings.get("strict", ()))
    fused = _route(rankings.get("fused", ()))
    final = _route(rankings.get("final", ()))
    deep = _route(deep_ids[: config.deep_depth])

    prefix_checked = bool(precheck["enabled"])
    prefix_matches = (
        deep[: config.base_broad_depth] == broad if prefix_checked else None
    )
    incumbent = precheck["incumbent"]
    incumbent_coverage = precheck["incumbent_coverage"]
    conditions = {**precheck["conditions"], "prefix_matches": prefix_matches}
    trigger_reasons = list(precheck["rejection_reasons"])
    if prefix_checked and not prefix_matches:
        trigger_reasons.append("prefix_mismatch")
    trigger_enabled = not trigger_reasons and prefix_matches is True

    pool = set(broad) | set(strict) | set(fused) | set(final)
    tail = deep[config.base_broad_depth : config.deep_depth]
    unique_tail: list[tuple[str, int]] = []
    seen: set[str] = set()
    pool_overlap_count = 0
    for deep_rank, identifier in enumerate(
        tail, start=config.base_broad_depth + 1
    ):
        if identifier in seen:
            continue
        seen.add(identifier)
        if identifier in pool:
            pool_overlap_count += 1
            continue
        unique_tail.append((identifier, deep_rank))

    proposals: list[DepthProposal] = []
    missing_fields_count = 0
    if trigger_enabled and incumbent_coverage is not None:
        query_set = set(query)
        excluded_set = set(excluded)
        required_coverage = incumbent_coverage + config.strict_coverage_margin
        for identifier, deep_rank in unique_tail:
            fields = searchable_fields.get(identifier)
            if fields is None:
                missing_fields_count += 1
            terms = set(_normalized_terms(tokenize(_field_text(fields or ()))))
            matched_query = tuple(term for term in query if term in terms)
            matched_excluded = tuple(term for term in excluded if term in terms)
            coverage = len(query_set & terms)
            reasons: list[str] = []
            if coverage < required_coverage:
                reasons.append("strict_coverage_margin_not_met")
            if excluded_set & terms:
                reasons.append("excluded_term_match")
            proposals.append(
                DepthProposal(
                    identifier=identifier,
                    coverage=coverage,
                    deep_rank=deep_rank,
                    matched_query_terms=matched_query,
                    matched_excluded_terms=matched_excluded,
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                )
            )

    proposals.sort(
        key=lambda item: (-item.coverage, item.deep_rank, item.identifier)
    )
    accepted = [item for item in proposals if item.eligible]
    winner = accepted[0] if accepted else None
    if winner is None:
        final_order = list(final)
    else:
        final_order = [
            *final[: config.protected_prefix],
            winner.identifier,
            *final[config.protected_prefix :],
        ]

    base_top10 = final[: config.top_k]
    final_top10 = final_order[: config.top_k]
    protected = base_top10[: config.protected_prefix]
    newcomers = [value for value in final_top10 if value not in set(base_top10)]
    bounded_coverage = {
        identifier: value
        for identifier in base_top10
        if (value := _coverage_value(baseline_coverage.get(identifier))) is not None
    }
    bounded_coverage.update(
        {item.identifier: item.coverage for item in proposals}
    )
    if winner is not None:
        reason = "accepted"
    elif trigger_reasons:
        reason = trigger_reasons[0]
    else:
        reason = "no_eligible_tail_candidate"

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "query_terms": query,
        "excluded_terms": excluded,
        "visible_query_term_count": precheck["visible_query_term_count"],
        "fts_query_term_count": precheck["fts_query_term_count"],
        "triggered": trigger_enabled,
        "active": winner is not None,
        "trigger": {
            "enabled": trigger_enabled,
            "conditions": conditions,
            "rejection_reasons": trigger_reasons,
            "query_term_count": precheck["visible_query_term_count"],
            "fts_query_term_count": precheck["fts_query_term_count"],
            "broad_count": precheck["broad_count"],
            "final_count": precheck["final_count"],
            "incumbent": incumbent,
            "incumbent_coverage": incumbent_coverage,
        },
        "prefix": {
            "checked": prefix_checked,
            "matches": prefix_matches,
            "checked_count": config.base_broad_depth if prefix_checked else 0,
        },
        "selection_order": [
            "coverage_desc",
            "deep_rank_asc",
            "identifier_asc",
        ],
        "tail": {
            "route_count": len(tail),
            "unique_count": len(seen),
            "base_pool_overlap_count": pool_overlap_count,
            "candidate_count": len(unique_tail),
            "missing_fields_count": missing_fields_count,
            "eligible_count": len(accepted),
            "coverage_rejection_count": sum(
                "strict_coverage_margin_not_met" in item.rejection_reasons
                for item in proposals
            ),
            "excluded_rejection_count": sum(
                "excluded_term_match" in item.rejection_reasons
                for item in proposals
            ),
            "proposals": [asdict(item) for item in proposals],
        },
        "coverage": {
            "coverage_by_parent_asin": bounded_coverage,
            "matched_excluded_terms_by_parent_asin": {
                item.identifier: item.matched_excluded_terms
                for item in proposals
                if item.matched_excluded_terms
            },
        },
        "guard": {
            "protected_top9": protected,
            "incumbent": incumbent,
            "incumbent_coverage": incumbent_coverage,
            "replacement": winner.identifier if winner is not None else None,
            "replacement_coverage": winner.coverage if winner is not None else None,
            "applied": winner is not None,
            "reason": reason,
            "top9_unchanged": final_top10[: len(protected)] == protected,
            "newcomers": newcomers,
            "newcomer_count": len(newcomers),
        },
        "final_top10": final_top10,
        "target_blind": True,
        "label_free": True,
    }
    return final_order, diagnostics
