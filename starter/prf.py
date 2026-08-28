"""Target-blind deterministic pseudo-relevance feedback primitives for P5."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "p5.prf.v1"
FIELD_NAMES = ("title", "categories", "features", "details")
STORE_INDEX = 4
SEARCHABLE_FIELD_COUNT = 6
MIN_FEEDBACK_TERMS_FOR_FUSION = 2
MIN_FIELD_GROUPS = 2
MIN_FEEDBACK_TERM_LENGTH = 3
TOP_K = 10
PROTECTED_PREFIX = 9
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class PrfConfig:
    seed_depth: int = 5
    min_query_terms: int = 2
    min_seed_count: int = 3
    min_seed_coverage: int = 2
    min_support_count: int = 3
    min_support_ratio: float = 0.60
    max_feedback_terms: int = 4
    max_df_ratio: float = 0.02
    min_novel_documents: int = 3
    route_depth: int = 120
    rrf_k: float = 60.0
    prf_weight: float = 0.15
    max_top10_newcomers: int = 1

    def __post_init__(self) -> None:
        integer_positive = {
            "seed_depth": self.seed_depth,
            "min_query_terms": self.min_query_terms,
            "min_seed_count": self.min_seed_count,
            "min_seed_coverage": self.min_seed_coverage,
            "min_support_count": self.min_support_count,
            "max_feedback_terms": self.max_feedback_terms,
            "route_depth": self.route_depth,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_seed_count > self.seed_depth:
            raise ValueError("min_seed_count cannot exceed seed_depth")
        if self.min_support_count > self.seed_depth:
            raise ValueError("min_support_count cannot exceed seed_depth")
        for name, value in (
            ("min_support_ratio", self.min_support_ratio),
            ("max_df_ratio", self.max_df_ratio),
        ):
            if isinstance(value, bool) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if (
            isinstance(self.min_novel_documents, bool)
            or not isinstance(self.min_novel_documents, int)
            or self.min_novel_documents < 0
        ):
            raise ValueError("min_novel_documents must be a non-negative integer")
        if self.route_depth < TOP_K:
            raise ValueError(f"route_depth must be at least {TOP_K}")
        if isinstance(self.rrf_k, bool) or float(self.rrf_k) <= 0.0:
            raise ValueError("rrf_k must be positive")
        if isinstance(self.prf_weight, bool) or float(self.prf_weight) < 0.0:
            raise ValueError("prf_weight must be non-negative")
        if (
            isinstance(self.max_top10_newcomers, bool)
            or not isinstance(self.max_top10_newcomers, int)
            or not 0 <= self.max_top10_newcomers <= 1
        ):
            raise ValueError("max_top10_newcomers must be 0 or 1")


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _flatten_text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{_flatten_text(key)} {_flatten_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _ascii_alpha_tokens(value: object) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(_flatten_text(value))
        if (
            len(token) >= MIN_FEEDBACK_TERM_LENGTH
            and token.isascii()
            and token.isalpha()
        )
    }


def _normalized_terms(values: Iterable[str], *, alpha_only: bool = False) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if not text or not text.isascii():
            continue
        if alpha_only and not text.isalpha():
            continue
        if not alpha_only and not text.isalnum():
            continue
        if text not in result:
            result.append(text)
    return result


def _field_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _bm25_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        1.0
        + (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def extract_feedback_terms(
    query_terms: Iterable[str],
    excluded_terms: Iterable[str],
    seed_ids: Sequence[str],
    seed_fields: Mapping[str, Sequence[object]],
    document_count: int,
    document_frequencies: Mapping[str, int],
    config: PrfConfig,
) -> tuple[list[str], dict[str, Any]]:
    """Select conservative catalog terms from ranked target-blind seed documents."""

    if isinstance(document_count, bool) or document_count <= 0:
        raise ValueError("document_count must be positive")
    query = _normalized_terms(query_terms, alpha_only=True)
    excluded = _normalized_terms(excluded_terms, alpha_only=True)
    ranked_seed_ids = _dedupe(seed_ids)[: config.seed_depth]
    available = [
        identifier
        for identifier in ranked_seed_ids
        if len(_field_tuple(seed_fields.get(identifier, ()))) >= SEARCHABLE_FIELD_COUNT
    ]
    available_seed_ratio = len(available) / config.seed_depth
    store_terms: set[str] = set()
    document_terms: dict[str, set[str]] = {}
    field_groups: dict[str, set[str]] = defaultdict(set)
    supporting_ids: dict[str, list[str]] = defaultdict(list)
    rank_discounts: dict[str, float] = defaultdict(float)
    all_seed_rank_discount = sum(
        1.0 / math.log2(rank + 1.0)
        for rank in range(1, len(ranked_seed_ids) + 1)
    )

    available_set = set(available)
    for rank, identifier in enumerate(ranked_seed_ids, start=1):
        fields = _field_tuple(seed_fields.get(identifier, ()))
        if identifier not in available_set:
            continue
        store_terms.update(_ascii_alpha_tokens(fields[STORE_INDEX]))
        seen_in_document: set[str] = set()
        for index, field_name in enumerate(FIELD_NAMES):
            terms = _ascii_alpha_tokens(fields[index])
            seen_in_document.update(terms)
            for term in terms:
                field_groups[term].add(field_name)
        document_terms[identifier] = seen_in_document
        for term in seen_in_document:
            supporting_ids[term].append(identifier)
            rank_discounts[term] += 1.0 / math.log2(rank + 1.0)

    normalized_df: dict[str, int] = {}
    for raw_term, raw_frequency in document_frequencies.items():
        terms = _ascii_alpha_tokens(raw_term)
        if len(terms) != 1:
            continue
        term = next(iter(terms))
        if isinstance(raw_frequency, bool) or not isinstance(raw_frequency, int):
            continue
        normalized_df[term] = raw_frequency

    maximum_df = float(config.max_df_ratio) * document_count
    query_set = set(query)
    excluded_set = set(excluded)
    vocabulary = sorted({term for terms in document_terms.values() for term in terms})
    diagnostics_by_term: dict[str, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    denominator = len(available)
    for term in vocabulary:
        support = len(supporting_ids[term])
        support_ratio = support / denominator if denominator else 0.0
        groups = sorted(field_groups[term])
        df = normalized_df.get(term)
        reasons: list[str] = []
        if term in query_set:
            reasons.append("original_query_term")
        if term in excluded_set:
            reasons.append("excluded_term")
        if term in store_terms:
            reasons.append("dynamic_store_brand")
        if support < config.min_support_count:
            reasons.append("seed_support_below_minimum")
        if support_ratio + 1e-12 < config.min_support_ratio:
            reasons.append("seed_support_ratio_below_minimum")
        if len(groups) < MIN_FIELD_GROUPS:
            reasons.append("field_group_support_below_minimum")
        if df is None or df < support or df > document_count:
            reasons.append("invalid_or_missing_document_frequency")
            idf = None
            novel_documents = None
            score = 0.0
        else:
            novel_documents = df - support
            if df > maximum_df + 1e-12:
                reasons.append("document_frequency_above_maximum")
            if novel_documents < config.min_novel_documents:
                reasons.append("novel_document_support_below_minimum")
            idf = _bm25_idf(document_count, df)
            normalized_rank_discount = (
                rank_discounts[term] / all_seed_rank_discount
                if all_seed_rank_discount
                else 0.0
            )
            score = idf * normalized_rank_discount
        item = {
            "term": term,
            "eligible": not reasons,
            "selected": False,
            "rejection_reasons": reasons,
            "seed_support_count": support,
            "seed_support_ratio": round(support_ratio, 9),
            "supporting_seed_ids": list(supporting_ids[term]),
            "field_groups": groups,
            "field_group_count": len(groups),
            "document_frequency": df,
            "maximum_document_frequency": maximum_df,
            "novel_document_count": novel_documents,
            "bm25_idf": round(idf, 12) if idf is not None else None,
            "rank_discount_sum": round(rank_discounts[term], 12),
            "all_seed_rank_discount_sum": round(all_seed_rank_discount, 12),
            "normalized_rank_discount": round(
                rank_discounts[term] / all_seed_rank_discount
                if all_seed_rank_discount
                else 0.0,
                12,
            ),
            "score": round(score, 12),
        }
        diagnostics_by_term[term] = item
        if not reasons:
            eligible.append(item)

    eligible.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["seed_support_count"]),
            int(item["document_frequency"]),
            str(item["term"]),
        )
    )
    fallback_reasons: list[str] = []
    if len(query) < config.min_query_terms:
        fallback_reasons.append("query_term_count_below_minimum")
    if len(available) < config.min_seed_count:
        fallback_reasons.append("seed_count_below_minimum")
    provisional = [str(item["term"]) for item in eligible[: config.max_feedback_terms]]
    if len(provisional) < MIN_FEEDBACK_TERMS_FOR_FUSION:
        fallback_reasons.append("fewer_than_two_feedback_terms")
    selected = [] if fallback_reasons else provisional
    selected_set = set(selected)
    for item in diagnostics_by_term.values():
        item["selected"] = item["term"] in selected_set

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "query_terms": query,
        "excluded_terms": excluded,
        "ranked_seed_ids": ranked_seed_ids,
        "available_seed_ids": available,
        "seed_count": len(available),
        "available_seed_ratio": round(available_seed_ratio, 9),
        "document_count": document_count,
        "dynamic_store_brand_terms": sorted(store_terms),
        "selected_terms": selected,
        "fallback": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "term_diagnostics": [diagnostics_by_term[term] for term in vocabulary],
        "score_formula": (
            "BM25_IDF=ln(1+(N-df+0.5)/(df+0.5)); "
            "score=BM25_IDF*sum_support(1/log2(seed_rank+1))"
            "/sum_all_seeds(1/log2(seed_rank+1))"
        ),
        "target_blind": True,
    }
    return selected, diagnostics


def build_prf_expression(
    query_terms: Iterable[str], feedback_terms: Iterable[str]
) -> str:
    """Build a deterministic FTS OR expression from original and feedback terms."""

    query = _normalized_terms(query_terms, alpha_only=True)
    feedback = _normalized_terms(feedback_terms, alpha_only=True)
    feedback = [term for term in feedback if term not in set(query)]
    if not query or not feedback:
        return ""
    query_group = " OR ".join(f'"{term}"' for term in query)
    feedback_group = " OR ".join(f'"{term}"' for term in feedback)
    return f"({query_group}) AND ({feedback_group})"


def _search_terms(
    value: object,
    tokenize: Callable[[str], Iterable[str]],
) -> set[str]:
    fields = _field_tuple(value)
    text = " ".join(_flatten_text(field) for field in fields[:SEARCHABLE_FIELD_COUNT])
    return {
        normalized
        for token in tokenize(text)
        if (normalized := str(token).strip().lower())
        and normalized.isascii()
        and normalized.isalnum()
    }


def guarded_prf_fusion(
    query_terms: Iterable[str],
    excluded_terms: Iterable[str],
    feedback_terms: Iterable[str],
    rankings: Mapping[str, Sequence[str]],
    prf_ids: Sequence[str],
    searchable_fields: Mapping[str, Sequence[object]],
    config: PrfConfig,
    tokenize: Callable[[str], Iterable[str]],
) -> tuple[list[str], dict[str, Any]]:
    """Fuse one PRF route, then admit at most one guarded Top-10 newcomer."""

    query = _normalized_terms(query_terms)
    excluded = _normalized_terms(excluded_terms)
    feedback = _normalized_terms(feedback_terms, alpha_only=True)
    base = _dedupe(rankings.get("final", ()))
    prf = _dedupe(prf_ids)[: config.route_depth]
    broad_ids = _dedupe(rankings.get("broad", ()))
    strict_ids = _dedupe(rankings.get("strict", ()))
    fused_ids = _dedupe(rankings.get("fused", ()))
    broad = set(broad_ids)
    strict = set(strict_ids)
    fused = set(fused_ids)
    final_pool = set(base)
    base_pool = broad | strict | fused | final_pool
    base_rank = {identifier: rank for rank, identifier in enumerate(base, start=1)}
    prf_rank = {identifier: rank for rank, identifier in enumerate(prf, start=1)}
    broad_rank = {
        identifier: rank for rank, identifier in enumerate(broad_ids, start=1)
    }
    strict_rank = {
        identifier: rank for rank, identifier in enumerate(strict_ids, start=1)
    }
    fused_rank = {
        identifier: rank for rank, identifier in enumerate(fused_ids, start=1)
    }
    candidate_ids = _dedupe([*base, *prf])
    query_set = set(query)
    excluded_set = set(excluded)
    feedback_set = set(feedback)
    evidence: dict[str, dict[str, Any]] = {}
    proposal_scores: dict[str, float] = {}
    for identifier in candidate_ids:
        terms = _search_terms(searchable_fields.get(identifier, ()), tokenize)
        coverage_terms = sorted(query_set & terms)
        excluded_matches = sorted(excluded_set & terms)
        feedback_matches = sorted(feedback_set & terms)
        b_rank = base_rank.get(identifier)
        p_rank = prf_rank.get(identifier)
        f_rank = fused_rank.get(identifier)
        base_score = (
            1.0 / (float(config.rrf_k) + f_rank) if f_rank is not None else 0.0
        )
        prf_score = (
            float(config.prf_weight) / (float(config.rrf_k) + p_rank)
            if p_rank is not None
            else 0.0
        )
        routes = [
            route
            for route, members in (("broad", broad), ("strict", strict))
            if identifier in members
        ]
        if p_rank is not None:
            routes.append("prf")
        proposal_scores[identifier] = base_score + prf_score
        evidence[identifier] = {
            "identifier": identifier,
            "original_query_coverage": len(coverage_terms),
            "matched_query_terms": coverage_terms,
            "excluded_match_count": len(excluded_matches),
            "matched_excluded_terms": excluded_matches,
            "feedback_match_count": len(feedback_matches),
            "matched_feedback_terms": feedback_matches,
            "base_rank": b_rank,
            "prf_rank": p_rank,
            "broad_rank": broad_rank.get(identifier),
            "strict_rank": strict_rank.get(identifier),
            "fused_rank": f_rank,
            "base_rrf_score": round(base_score, 12),
            "prf_rrf_score": round(prf_score, 12),
            "route_score": round(proposal_scores[identifier], 12),
            "base_pool": identifier in base_pool,
            "prf_only": identifier not in base_pool,
            "route_evidence": routes,
        }

    proposal = sorted(
        candidate_ids,
        key=lambda identifier: (
            -int(evidence[identifier]["original_query_coverage"]),
            -proposal_scores[identifier],
            fused_rank.get(identifier, 10**9),
            prf_rank.get(identifier, 10**9),
            identifier,
        ),
    )
    for rank, identifier in enumerate(proposal, start=1):
        evidence[identifier]["proposal_rank"] = rank
    base_top10 = base[:TOP_K]
    protected = base_top10[:PROTECTED_PREFIX]
    incumbent = base_top10[PROTECTED_PREFIX] if len(base_top10) > PROTECTED_PREFIX else None
    incumbent_coverage = (
        int(evidence[incumbent]["original_query_coverage"])
        if incumbent is not None
        else -1
    )
    incumbent_proposal_rank = (
        int(evidence[incumbent]["proposal_rank"])
        if incumbent is not None
        else len(proposal) + 1
    )
    incumbent_proposal_score = (
        proposal_scores[incumbent] if incumbent is not None else -1.0
    )
    global_reasons: list[str] = []
    if len(query) < config.min_query_terms:
        global_reasons.append("query_term_count_below_minimum")
    if len(feedback) < MIN_FEEDBACK_TERMS_FOR_FUSION:
        global_reasons.append("fewer_than_two_feedback_terms")
    if not prf:
        global_reasons.append("empty_prf_route")
    if config.max_top10_newcomers == 0:
        global_reasons.append("top10_newcomers_disabled")

    decisions: list[dict[str, Any]] = []
    replacement: str | None = None
    original_top10 = set(base_top10)
    for identifier in proposal:
        if identifier in original_top10:
            continue
        item = evidence[identifier]
        reasons = list(global_reasons)
        coverage = int(item["original_query_coverage"])
        if item["prf_rank"] is None:
            reasons.append("not_in_prf_route")
        if int(item["excluded_match_count"]) != 0:
            reasons.append("excluded_term_match")
        if int(item["feedback_match_count"]) < MIN_FEEDBACK_TERMS_FOR_FUSION:
            reasons.append("feedback_match_count_below_minimum")
        if int(item["proposal_rank"]) >= incumbent_proposal_rank:
            reasons.append("proposal_not_ranked_ahead_of_incumbent")
        if bool(item["prf_only"]):
            if coverage <= incumbent_coverage:
                reasons.append("prf_only_requires_strict_coverage_advantage")
        else:
            if coverage < incumbent_coverage:
                reasons.append("coverage_below_incumbent")
            elif coverage == incumbent_coverage:
                if proposal_scores[identifier] <= incumbent_proposal_score:
                    reasons.append(
                        "same_coverage_proposal_score_not_strictly_higher"
                    )
                if strict_ids:
                    if item["broad_rank"] is None or item["strict_rank"] is None:
                        reasons.append(
                            "same_coverage_requires_broad_and_strict_route_evidence"
                        )
                elif item["broad_rank"] is None or int(item["broad_rank"]) > 30:
                    reasons.append(
                        "same_coverage_requires_broad_top30_when_strict_empty"
                    )
        decision = {
            "identifier": identifier,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            **item,
        }
        decisions.append(decision)
        if not reasons:
            replacement = identifier
            break

    if replacement is None:
        final = list(base)
    else:
        final = [
            *protected,
            replacement,
            *(identifier for identifier in base if identifier not in protected and identifier != replacement),
        ]
        if len(base) >= TOP_K:
            final.extend(identifier for identifier in prf if identifier not in final)
        final = _dedupe(final)

    final_top10 = final[:TOP_K]
    newcomers = [identifier for identifier in final_top10 if identifier not in original_top10]
    if len(newcomers) > config.max_top10_newcomers:
        raise RuntimeError("PRF Top-10 newcomer invariant failed")
    if final_top10[: len(protected)] != protected:
        raise RuntimeError("PRF protected-prefix invariant failed")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "query_terms": query,
        "excluded_terms": excluded,
        "feedback_terms": feedback,
        "proposal_formula": (
            "sort by original-query coverage, then B+0.15P (configurable prf_weight) "
            "where B=1/(rrf_k+fused_rank) and P=1/(rrf_k+prf_rank), "
            "then fused rank, PRF rank, and ID"
        ),
        "proposal": [evidence[identifier] for identifier in proposal],
        "guard": {
            "base_top10": base_top10,
            "protected_top9": protected,
            "incumbent": incumbent,
            "replacement": replacement,
            "applied": replacement is not None,
            "global_rejection_reasons": global_reasons,
            "candidate_decisions": decisions,
            "top9_unchanged": final_top10[: len(protected)] == protected,
            "newcomers": newcomers,
            "newcomer_count": len(newcomers),
        },
        "final_top10": final_top10,
        "target_blind": True,
    }
    return final, diagnostics
