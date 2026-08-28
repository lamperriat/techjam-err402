"""Target-blind architecture ablations isolated from the submitted Agent path.

The default ``starter.agent.Agent`` is deliberately unchanged.  Every architecture in
this module consumes only the same profile, visible conversation state, and frozen
catalog fields available to that Agent.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from starter.agent import Agent, SessionState, _terms
from starter.attributes import (
    ProductAttributeView,
    build_product_attribute_view,
    normalize_value,
    product_slot,
)
from starter.coverage import order_by_query_coverage


SCHEMA_VERSION = "p4.architecture-lab.v1"
CONTROL_ID = "C00.control_rrf"


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    variant_id: str
    family: str
    mechanism: str
    stage_graph: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SPECS = (
    ArchitectureSpec(
        CONTROL_ID,
        "control",
        "control_rrf",
        ("visible_state", "broad_bm25+strict_bm25", "weighted_rrf", "top10"),
        "Current served sparse control without output changes.",
    ),
    ArchitectureSpec(
        "R01.field_rrf",
        "retrieval",
        "field_rrf",
        ("visible_state", "three_field_routes", "multi_route_rrf", "top10"),
        "Independent title/category, feature/detail, and description/store routes.",
    ),
    ArchitectureSpec(
        "R02.category_guard",
        "retrieval",
        "category_guard",
        ("category_route", "requirement_rrf", "guard_then_fallback", "top10"),
        "Category-field guard with deterministic fused fallback.",
    ),
    ArchitectureSpec(
        "R03.turn_rrf",
        "state_retrieval",
        "turn_rrf",
        ("versioned_turn_routes", "recency_rrf", "top10"),
        "Retrieves each retained turn separately instead of flattening all terms.",
    ),
    ArchitectureSpec(
        "R04.phrase_route",
        "retrieval",
        "phrase_route",
        ("slot_and_turn_phrases", "exact_phrase_fts", "rrf", "top10"),
        "Adds exact bigram/trigram and multiword-slot phrase evidence.",
    ),
    ArchitectureSpec(
        "R05.alias_expansion",
        "query_representation",
        "alias_expansion",
        ("original_query", "low_trust_alias_route", "rrf", "top10"),
        "Keeps the original query and fuses a target-blind domain-alias route.",
    ),
    ArchitectureSpec(
        "R06.rare_anchor",
        "retrieval",
        "rare_anchor",
        ("fts5vocab", "rarest_visible_anchor", "guarded_route", "fallback", "top10"),
        "Uses the lowest-document-frequency visible term as a lexical anchor.",
    ),
    ArchitectureSpec(
        "R07.combsum_bm25",
        "fusion",
        "combsum_bm25",
        ("broad_bm25+strict_bm25", "route_normalization", "combsum", "top10"),
        "Fuses normalized raw BM25 scores rather than rank-only RRF.",
    ),
    ArchitectureSpec(
        "R08.coverage_cascade",
        "reranking",
        "coverage_cascade",
        ("fused_pool", "visible_term_coverage", "lexicographic_cascade", "top10"),
        "Ranks by distinct visible-term coverage before fused rank.",
    ),
    ArchitectureSpec(
        "R09.slot_filter_relax",
        "constraint_retrieval",
        "slot_filter_relax",
        ("slot_ledger", "negative_guard", "positive_relaxation", "bounded_top10"),
        "Never backfills known negative conflicts; relaxes only positive constraints and may return fewer than ten.",
        (("candidate_pool", 200), ("missing_metadata", "unknown_not_conflict")),
    ),
    ArchitectureSpec(
        "R10.candidate_carryover",
        "belief_state",
        "candidate_carryover",
        ("current_posterior", "same_goal_prior", "temporal_rrf", "top10"),
        "Carries a decayed shortlist only inside the current goal version.",
        (("previous_route_weight", 0.55), ("rrf_k", 60.0), ("carryover_depth", 50)),
    ),
    ArchitectureSpec(
        "R11.browse_mmr",
        "diversification",
        "browse_mmr",
        ("visible_browse_router", "attribute_mmr", "top10"),
        "Diversifies ambiguous visible browsing turns by catalog attribute aspects.",
        (("candidate_pool", 30), ("redundancy_penalty", 0.20)),
    ),
    ArchitectureSpec(
        "R12.numeric_budget",
        "constraint_reranking",
        "numeric_budget",
        ("visible_budget_parser", "price_distance_or_bound", "unknown_neutral", "top10"),
        "Executes visible numeric budget constraints against catalog prices.",
        (
            ("around_fraction", 0.20),
            ("around_absolute_floor", 10.0),
            ("unknown_top10_reserve", 2),
        ),
    ),
    ArchitectureSpec(
        "R13.intent_router",
        "joint_router",
        "intent_router",
        ("visible_intent_router", "browse_or_constraint_expert", "top10"),
        "Routes visible browsing to field/MMR and hard constraints to guarded relaxation.",
        (("browse_redundancy_penalty", 0.20), ("negative_backfill", False)),
    ),
    ArchitectureSpec(
        "R14.borda_fusion",
        "fusion",
        "borda_fusion",
        ("broad+strict+field_routes", "normalized_borda_votes", "top10"),
        "Aggregates route-relative rank votes instead of reciprocal-rank or raw-score fusion.",
        (("route_depth", 120),),
    ),
)

SPEC_BY_ID = {spec.variant_id: spec for spec in SPECS}


def validate_registry(specs: Iterable[ArchitectureSpec] = SPECS) -> None:
    materialized = tuple(specs)
    ids = [spec.variant_id for spec in materialized]
    mechanisms = [spec.mechanism for spec in materialized]
    stage_graphs = [spec.stage_graph for spec in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("architecture variant IDs must be unique")
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("architecture mechanisms must be unique")
    if len(stage_graphs) != len(set(stage_graphs)):
        raise ValueError("architecture stage graphs must be unique")
    controls = [spec for spec in materialized if spec.family == "control"]
    if len(controls) != 1 or controls[0].variant_id != CONTROL_ID:
        raise ValueError("registry must contain exactly one canonical control")
    if len(materialized) - 1 < 10:
        raise ValueError("architecture search requires at least 10 non-control designs")


validate_registry()


@dataclass(slots=True)
class VariantStats:
    turns: int = 0
    activations: int = 0
    output_changes: int = 0
    fallbacks: int = 0
    route_counts: Counter[str] = field(default_factory=Counter)
    relaxation_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "activations": self.activations,
            "output_changes": self.output_changes,
            "fallbacks": self.fallbacks,
            "route_counts": dict(sorted(self.route_counts.items())),
            "relaxation_counts": dict(sorted(self.relaxation_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class StrategyResult:
    identifiers: tuple[str, ...]
    activated: bool
    fallback: bool = False
    backfill_base: bool = True
    routes: tuple[str, ...] = ()
    relaxation: str | None = None


_ALIAS_EXPANSIONS = {
    "women": ("female", "womens"),
    "womens": ("women", "female"),
    "men": ("male", "mens"),
    "mens": ("men", "male"),
    "shoe": ("shoes", "footwear"),
    "shoes": ("shoe", "footwear"),
    "sneaker": ("trainers", "athletic shoe"),
    "sneakers": ("trainers", "athletic shoes"),
    "shirt": ("top", "tee"),
    "pants": ("trousers",),
    "purse": ("handbag",),
    "handbag": ("purse",),
    "jewelry": ("jewellery",),
    "formal": ("dressy",),
    "casual": ("everyday",),
    "running": ("jogging", "athletic"),
    "hiking": ("trekking", "outdoor"),
    "comfortable": ("comfort", "comfy"),
    "waterproof": ("water resistant", "rain"),
}

_UNDER_RE = re.compile(
    r"\b(?:under|below|less than|at most|max(?:imum)?|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OVER_RE = re.compile(
    r"\b(?:over|above|more than|at least|min(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_AROUND_RE = re.compile(
    r"\b(?:budget\s+)?(?:around|about|near)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BARE_PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def _dedupe(*routes: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(identifier for route in routes for identifier in route))


def _quote(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _rrf(routes: Iterable[tuple[Iterable[str], float]]) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for identifiers, weight in routes:
        for rank, identifier in enumerate(identifiers, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (60.0 + rank)
            best_rank[identifier] = min(best_rank.get(identifier, 10**9), rank)
    return sorted(scores, key=lambda value: (-scores[value], best_rank[value], value))


class ArchitectureAgent(Agent):
    """Experiment-only Agent that applies one explicit target-blind architecture."""

    def __init__(
        self,
        catalog_path: str | Path,
        variant_id: str = CONTROL_ID,
        *,
        question_policy: str = "fast",
    ) -> None:
        if variant_id not in SPEC_BY_ID:
            raise ValueError(f"unknown architecture variant: {variant_id}")
        self.architecture_spec = SPEC_BY_ID[variant_id]
        self.variant_stats = VariantStats()
        self._rowids: dict[str, int] = {}
        self._prices: dict[str, float] = {}
        self._carryover: dict[int, tuple[int, list[str]]] = {}
        super().__init__(
            catalog_path,
            question_policy=question_policy,
            rerank_mode="off",
            retrieval_mode="control",
        )

    def _build_index(self) -> None:
        super()._build_index()
        self._rowids = {
            str(parent_asin): int(rowid)
            for rowid, parent_asin in self.connection.execute(
                "SELECT rowid, parent_asin FROM products"
            )
        }
        mechanism = self.architecture_spec.mechanism
        if mechanism in {"rare_anchor"}:
            self.connection.execute(
                "CREATE VIRTUAL TABLE products_vocab USING fts5vocab(products, 'row')"
            )
        if mechanism in {"numeric_budget", "intent_router"}:
            with self.catalog_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    product = json.loads(line)
                    price = self._parse_price(product.get("price"))
                    if price is not None:
                        self._prices[str(product["parent_asin"])] = price

    @staticmethod
    def _parse_price(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if float(value) >= 0 else None
        match = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
        if not match:
            return None
        parsed = float(match.group(0))
        return parsed if parsed >= 0 else None

    def reset(self, session_id: str, user_profile: dict) -> None:
        previous = self._sessions.get(session_id)
        if previous is not None:
            self._carryover.pop(id(previous), None)
        super().reset(session_id, user_profile)

    def drop_session(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is not None:
            self._carryover.pop(id(state), None)
        super().drop_session(session_id)

    def experiment_stats(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "spec": self.architecture_spec.as_dict(),
            **self.variant_stats.as_dict(),
        }

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        self.variant_stats.turns += 1
        mechanism = self.architecture_spec.mechanism
        if mechanism == "control_rrf" or not rankings["fused"]:
            return rankings
        strategy = getattr(self, f"_strategy_{mechanism}")
        result: StrategyResult = strategy(state, rankings)
        final = (
            _dedupe(result.identifiers, rankings["fused"])
            if result.backfill_base
            else _dedupe(result.identifiers)
        )
        self.variant_stats.activations += int(result.activated)
        self.variant_stats.fallbacks += int(result.fallback)
        self.variant_stats.output_changes += int(final[:10] != rankings["fused"][:10])
        self.variant_stats.route_counts.update(result.routes)
        if result.relaxation:
            self.variant_stats.relaxation_counts[result.relaxation] += 1
        return {
            **rankings,
            "reranked": final,
            "final": final,
        }

    def _route(
        self,
        expression: str,
        *,
        limit: int = 120,
        with_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        if not expression:
            return []
        select = "parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
        rows = self.connection.execute(
            f"SELECT {select} FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0), "
            f"parent_asin ASC LIMIT ?",
            (expression, limit),
        ).fetchall()
        if with_scores:
            return [(str(row[0]), float(row[1])) for row in rows]
        return [str(row[0]) for row in rows]

    @staticmethod
    def _column_expression(columns: tuple[str, ...], terms: list[str], strict: bool = False) -> str:
        unique = [term for term in dict.fromkeys(terms) if len(term) > 1][:24]
        if not unique:
            return ""
        operator = " AND " if strict else " OR "
        return "{" + " ".join(columns) + "} : (" + operator.join(_quote(term) for term in unique) + ")"

    def _rows(self, identifiers: list[str]) -> dict[str, tuple[str, str, str, str, str, str]]:
        rowids = [self._rowids[value] for value in identifiers if value in self._rowids]
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description "
            f"FROM products WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        return {
            str(row[0]): tuple(str(value or "") for value in row[1:])
            for row in rows
        }

    def _views(self, identifiers: list[str]) -> dict[str, ProductAttributeView]:
        rows = self._rows(identifiers)
        result: dict[str, ProductAttributeView] = {}
        for identifier, values in rows.items():
            title, categories, features, details, store, description = values
            result[identifier] = build_product_attribute_view({
                "parent_asin": identifier,
                "title": title,
                "categories": categories,
                "features": features,
                "details": details,
                "store": store,
                "description": description,
                "price": self._prices.get(identifier),
            })
        return result

    def _strategy_field_rrf(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        terms = self._query_terms(state)
        routes = [
            self._route(self._column_expression(columns, terms))
            for columns in (
                ("title", "categories"),
                ("features", "details"),
                ("description", "store"),
            )
        ]
        final = _rrf([
            (rankings["broad"], 1.0),
            (rankings["strict"], 1.0),
            *((route, 1.0) for route in routes),
        ])
        return StrategyResult(tuple(final), bool(any(routes)), routes=("field_rrf",))

    def _strategy_category_guard(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        category_terms = _terms(state.category_text)
        route = self._route(
            self._column_expression(("title", "categories"), category_terms, strict=True)
        )
        allowed = set(route)
        guarded = [value for value in rankings["fused"] if value in allowed]
        fallback = len(guarded) < 10
        final = _dedupe(guarded, route, rankings["fused"])
        return StrategyResult(
            tuple(final),
            bool(category_terms and route),
            fallback=fallback,
            routes=("category_guard",),
        )

    def _strategy_turn_rrf(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        routes: list[tuple[list[str], float]] = []
        category_terms = _terms(state.category_text)
        if category_terms:
            routes.append((self._route(self._fts_expression(category_terms)), 1.0))
        active = set(state.active_terms)
        turn_items = [
            (turn, [term for term in terms if term in active])
            for turn, terms in sorted(state.turn_terms.items())
        ]
        maximum_turn = max((turn for turn, _ in turn_items), default=1)
        for turn, terms in turn_items:
            if terms:
                routes.append((
                    self._route(self._fts_expression(terms)),
                    0.75 + 0.25 * turn / maximum_turn,
                ))
        final = _rrf([*routes, (rankings["fused"], 1.0)])
        return StrategyResult(tuple(final), bool(routes), routes=("turn_rrf",))

    def _strategy_phrase_route(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        phrases: list[str] = [
            record.value
            for record in state.slot_ledger.active_records()
            if len(record.value.split()) >= 2
        ]
        for terms in state.turn_terms.values():
            for width in (3, 2):
                phrases.extend(
                    " ".join(terms[index:index + width])
                    for index in range(0, max(0, len(terms) - width + 1))
                )
        phrases = list(dict.fromkeys(value for value in phrases if len(value) >= 5))[:16]
        expression = " OR ".join(_quote(value) for value in phrases)
        route = self._route(expression)
        final = _rrf(((rankings["fused"], 1.0), (route, 1.0)))
        return StrategyResult(tuple(final), bool(route), routes=("phrase_route",))

    def _strategy_alias_expansion(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        original = self._query_terms(state)
        expanded = [
            value
            for term in original
            for value in _ALIAS_EXPANSIONS.get(term, ())
        ]
        route = self._route(self._fts_expression([*original, *expanded])) if expanded else []
        final = _rrf(((rankings["fused"], 1.0), (route, 0.6)))
        return StrategyResult(tuple(final), bool(expanded and route), routes=("alias_route",))

    def _strategy_rare_anchor(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        terms = self._query_terms(state)
        frequencies: list[tuple[int, str]] = []
        for term in terms:
            row = self.connection.execute(
                "SELECT doc FROM products_vocab WHERE term = ?", (term,)
            ).fetchone()
            if row is not None:
                frequencies.append((int(row[0]), term))
        if len(frequencies) < 2:
            return StrategyResult(tuple(rankings["fused"]), False, fallback=True)
        _, anchor = min(frequencies)
        others = [term for term in terms if term != anchor]
        expression = f"{_quote(anchor)} AND (" + " OR ".join(_quote(term) for term in others) + ")"
        route = self._route(expression)
        fallback = len(route) < 10
        final = _dedupe(route, rankings["fused"])
        return StrategyResult(
            tuple(final), True, fallback=fallback, routes=("rare_anchor",)
        )

    def _strategy_combsum_bm25(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        terms = self._query_terms(state)
        broad = self._route(self._fts_expression(terms), with_scores=True)
        strict = (
            self._route(self._strict_fts_expression(terms), limit=80, with_scores=True)
            if len(terms) >= 2
            else []
        )

        combined: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for rows in (broad, strict):
            if not rows:
                continue
            values = [score for _, score in rows]
            low, high = min(values), max(values)
            scale = high - low
            for rank, (identifier, score) in enumerate(rows, start=1):
                normalized = 1.0 if scale <= 1e-12 else (high - score) / scale
                combined[identifier] = combined.get(identifier, 0.0) + normalized
                best_rank[identifier] = min(best_rank.get(identifier, 10**9), rank)
        final = sorted(
            combined,
            key=lambda value: (-combined[value], best_rank[value], value),
        )
        return StrategyResult(tuple(final), bool(final), routes=("combsum",))

    def _strategy_coverage_cascade(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        terms = self._query_terms(state)
        rows = self._rows(rankings["fused"][:200])
        final, _ = order_by_query_coverage(
            terms,
            rankings["fused"],
            rows,
            _terms,
        )
        return StrategyResult(tuple(final), bool(terms), routes=("coverage",))

    @staticmethod
    def _record_matches(record: object, view: ProductAttributeView) -> bool | None:
        slot = str(getattr(record, "slot"))
        if slot == "price":
            return None
        values = product_slot(view, slot)
        if not values:
            return None
        wanted = set(str(getattr(record, "value")).split())
        return any(
            str(getattr(record, "value")) == value.value
            or wanted <= set(value.value.split())
            for value in values
        )

    def _slot_relaxation(
        self, state: SessionState, base_ids: list[str]
    ) -> tuple[list[str], bool, str]:
        pool = base_ids[:200]
        views = self._views(pool)
        records = [
            record
            for record in state.slot_ledger.active_records()
            if record.slot != "price"
        ]
        hard = [record for record in records if record.hardness == "hard" or record.polarity < 0]
        soft = [record for record in records if record not in hard]

        def conflicts(identifier: str, constraints: list[object]) -> int:
            view = views.get(identifier, ProductAttributeView(parent_asin=identifier))
            total = 0
            for record in constraints:
                matched = self._record_matches(record, view)
                if matched is None:
                    continue
                if (record.polarity > 0 and not matched) or (record.polarity < 0 and matched):
                    total += 1
            return total

        active_hard = list(hard)
        safe = [identifier for identifier in pool if conflicts(identifier, active_hard) == 0]
        relaxed = 0
        relaxable = sorted(
            (record for record in active_hard if record.polarity > 0),
            key=lambda record: (record.confidence, record.source_turn, record.record_id),
        )
        while len(safe) < 10 and relaxable:
            removed = relaxable.pop(0)
            active_hard.remove(removed)
            relaxed += 1
            safe = [identifier for identifier in pool if conflicts(identifier, active_hard) == 0]
        rank = {value: index for index, value in enumerate(pool, start=1)}

        def soft_matches(identifier: str) -> int:
            view = views.get(identifier, ProductAttributeView(parent_asin=identifier))
            return sum(
                1
                for record in soft
                if self._record_matches(record, view) == (record.polarity > 0)
            )

        safe.sort(key=lambda value: (-soft_matches(value), rank[value]))
        final = _dedupe(safe)
        shortfall = len(final) < 10 and any(
            record.polarity < 0 for record in active_hard
        )
        labels = []
        if relaxed:
            labels.append(f"relaxed_{relaxed}")
        if shortfall:
            labels.append("negative_guard_shortfall")
        label = "+".join(labels) if labels else "none"
        return final, bool(records), label

    def _strategy_slot_filter_relax(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        final, activated, relaxation = self._slot_relaxation(state, rankings["fused"])
        return StrategyResult(
            tuple(final),
            activated,
            fallback=relaxation != "none",
            backfill_base=False,
            routes=("slot_filter",),
            relaxation=relaxation,
        )

    def _strategy_candidate_carryover(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        key = id(state)
        previous = self._carryover.get(key)
        current = rankings["fused"]
        activated = previous is not None and previous[0] == state.version
        if activated:
            final = _rrf(((current, 1.0), (previous[1], 0.55)))
        else:
            final = list(current)
        self._carryover[key] = (state.version, list(current[:50]))
        return StrategyResult(tuple(final), activated, routes=("carryover",))

    @staticmethod
    def _view_signature(view: ProductAttributeView) -> set[str]:
        fields = (view.category, view.brand, view.material, view.style, view.use_case)
        return {
            f"{name}:{value.value}"
            for name, values in zip(
                ("category", "brand", "material", "style", "use_case"), fields
            )
            for value in values[:2]
        }

    def _browse_mmr(self, state: SessionState, base_ids: list[str]) -> tuple[list[str], bool]:
        current_goal_messages = state.messages[max(0, state.version_anchor_turn - 1):]
        visible_browse = any(
            "still exploring" in message.casefold()
            for message in current_goal_messages
        )
        hard = any(record.hardness == "hard" for record in state.slot_ledger.active_records())
        if not visible_browse or hard:
            return list(base_ids), False
        pool = base_ids[:30]
        views = self._views(pool)
        signatures = {
            identifier: self._view_signature(
                views.get(identifier, ProductAttributeView(parent_asin=identifier))
            )
            for identifier in pool
        }
        selected: list[str] = []
        remaining = list(pool)
        while remaining and len(selected) < 10:
            def score(identifier: str) -> tuple[float, int]:
                relevance = 1.0 / (1 + pool.index(identifier))
                maximum_overlap = 0.0
                for chosen in selected:
                    union = signatures[identifier] | signatures[chosen]
                    overlap = (
                        len(signatures[identifier] & signatures[chosen]) / len(union)
                        if union
                        else 0.0
                    )
                    maximum_overlap = max(maximum_overlap, overlap)
                return relevance - 0.20 * maximum_overlap, -pool.index(identifier)

            chosen = max(remaining, key=score)
            selected.append(chosen)
            remaining.remove(chosen)
        return _dedupe(selected, base_ids), True

    def _strategy_browse_mmr(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        final, activated = self._browse_mmr(state, rankings["fused"])
        return StrategyResult(tuple(final), activated, routes=("browse_mmr",))

    @staticmethod
    def _budget_constraint(messages: list[str]) -> tuple[str, float] | None:
        text = " ".join(messages)
        semantic: list[tuple[int, str, float]] = []
        for kind, pattern in (("under", _UNDER_RE), ("over", _OVER_RE), ("around", _AROUND_RE)):
            semantic.extend(
                (match.start(), kind, float(match.group(1)))
                for match in pattern.finditer(text)
            )
        if semantic:
            _, kind, value = max(semantic)
            return kind, value
        matches = list(_BARE_PRICE_RE.finditer(text))
        return ("around", float(matches[-1].group(1))) if matches else None

    def _budget_rank(self, state: SessionState, base_ids: list[str]) -> tuple[list[str], bool, bool]:
        if {"budget", "price"} & state.exhausted_attributes:
            return list(base_ids), False, False
        current_goal_messages = state.messages[max(0, state.version_anchor_turn - 1):]
        constraint = self._budget_constraint(current_goal_messages)
        if constraint is None:
            return list(base_ids), False, False
        kind, value = constraint
        known = [identifier for identifier in base_ids if identifier in self._prices]
        unknown = [identifier for identifier in base_ids if identifier not in self._prices]
        rank = {identifier: index for index, identifier in enumerate(base_ids)}
        unknown.sort(key=lambda identifier: rank[identifier])
        if kind == "under":
            inside = [identifier for identifier in known if self._prices[identifier] <= value]
            outside = [identifier for identifier in known if self._prices[identifier] > value]
            inside.sort(key=lambda identifier: rank[identifier])
            outside.sort(key=lambda identifier: (self._prices[identifier] - value, rank[identifier]))
        elif kind == "over":
            inside = [identifier for identifier in known if self._prices[identifier] >= value]
            outside = [identifier for identifier in known if self._prices[identifier] < value]
            inside.sort(key=lambda identifier: rank[identifier])
            outside.sort(key=lambda identifier: (value - self._prices[identifier], rank[identifier]))
        else:
            tolerance = max(10.0, value * 0.20)
            near = [
                identifier
                for identifier in known
                if abs(self._prices[identifier] - value) <= tolerance
            ]
            near_set = set(near)
            outside = sorted(
                (identifier for identifier in known if identifier not in near_set),
                key=lambda identifier: (abs(self._prices[identifier] - value), rank[identifier]),
            )
            inside = sorted(
                near,
                key=lambda identifier: (abs(self._prices[identifier] - value), rank[identifier]),
            )
        unknown_reserve = min(2, len(unknown))
        known_head = max(0, 10 - unknown_reserve)
        neutral_top = [*inside[:known_head], *unknown[:unknown_reserve]]
        fallback = len(inside) + len(unknown) < 10
        return _dedupe(
            neutral_top,
            inside[known_head:],
            unknown[unknown_reserve:],
            outside,
            base_ids,
        ), True, fallback

    def _strategy_numeric_budget(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        final, activated, fallback = self._budget_rank(state, rankings["fused"])
        return StrategyResult(
            tuple(final), activated, fallback=fallback, routes=("numeric_budget",)
        )

    def _strategy_intent_router(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        hard = any(record.hardness == "hard" for record in state.slot_ledger.active_records())
        current_goal_messages = state.messages[max(0, state.version_anchor_turn - 1):]
        browse = any(
            "still exploring" in message.casefold()
            for message in current_goal_messages
        )
        if hard:
            guarded = self._strategy_category_guard(state, rankings)
            routed = {**rankings, "fused": list(guarded.identifiers)}
            final, activated, relaxation = self._slot_relaxation(state, routed["fused"])
            return StrategyResult(
                tuple(final),
                activated or guarded.activated,
                fallback=guarded.fallback or relaxation != "none",
                backfill_base=False,
                routes=("router_constraint",),
                relaxation=relaxation,
            )
        if browse:
            fielded = self._strategy_field_rrf(state, rankings)
            final, activated = self._browse_mmr(state, list(fielded.identifiers))
            return StrategyResult(
                tuple(final),
                activated or fielded.activated,
                routes=("router_browse",),
            )
        turn = self._strategy_turn_rrf(state, rankings)
        return StrategyResult(
            turn.identifiers,
            turn.activated,
            fallback=turn.fallback,
            routes=("router_turn",),
        )

    def _strategy_borda_fusion(
        self, state: SessionState, rankings: dict[str, list[str]]
    ) -> StrategyResult:
        terms = self._query_terms(state)
        field_routes = [
            self._route(self._column_expression(columns, terms))
            for columns in (
                ("title", "categories"),
                ("features", "details"),
                ("description", "store"),
            )
        ]
        routes = [rankings["broad"], rankings["strict"], *field_routes]
        votes: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for route in routes:
            depth = len(route)
            if not depth:
                continue
            for rank, identifier in enumerate(route, start=1):
                votes[identifier] = votes.get(identifier, 0.0) + (
                    (depth - rank + 1) / depth
                )
                best_rank[identifier] = min(best_rank.get(identifier, 10**9), rank)
        final = sorted(
            votes,
            key=lambda identifier: (
                -votes[identifier],
                best_rank[identifier],
                identifier,
            ),
        )
        return StrategyResult(tuple(final), bool(votes), routes=("borda_fusion",))
