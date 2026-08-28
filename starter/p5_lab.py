"""P5-only guarded pseudo-relevance-feedback experiment.

This module is deliberately isolated from the submitted Agent and the frozen P4
architecture matrix.  It consumes only visible conversation state and catalog text.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from starter.agent import Agent, SessionState, _terms
from starter.prf import (
    SCHEMA_VERSION as PRF_SCHEMA_VERSION,
    PrfConfig,
    build_prf_expression,
    extract_feedback_terms,
    guarded_prf_fusion,
)


SCHEMA_VERSION = "p5.prf-lab.v1"
C00 = "P5.C00.r08_coverage"
S00 = "P5.S00.prf_shadow"
R01 = "P5.R01.guarded_session_prf"
CONTROL_ID = C00
SHADOW_ID = S00
ACTIVE_ID = R01
_FIELD_QUERY_CHUNK = 400


@dataclass(frozen=True, slots=True)
class P5Spec:
    variant_id: str
    family: str
    mechanism: str
    stage_graph: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_PRF_PARAMETERS = (
    ("seed_depth", 5),
    ("min_query_terms", 2),
    ("min_seed_count", 3),
    ("min_seed_coverage", 2),
    ("min_support_count", 3),
    ("min_support_ratio", 0.60),
    ("max_feedback_terms", 4),
    ("max_df_ratio", 0.02),
    ("min_novel_documents", 3),
    ("route_depth", 120),
    ("rrf_k", 60.0),
    ("prf_weight", 0.15),
    ("max_top10_newcomers", 1),
)

SPECS = (
    P5Spec(
        C00,
        "control",
        "r08_coverage",
        ("visible_state", "broad+strict_rrf", "coverage_cascade", "top10"),
        "Exact served R08 coverage baseline.",
    ),
    P5Spec(
        S00,
        "diagnostic",
        "prf_shadow",
        (
            "r08_coverage",
            "top5_catalog_feedback",
            "guarded_second_fts",
            "shadow_only",
        ),
        "Computes guarded session-local feedback without changing recommendations.",
        _PRF_PARAMETERS,
    ),
    P5Spec(
        R01,
        "retrieval",
        "guarded_session_prf",
        (
            "r08_coverage",
            "top5_catalog_feedback",
            "low_weight_second_fts",
            "guarded_tail_promotion",
            "top10",
        ),
        "Allows at most one guarded feedback candidate into the top-10 tail.",
        _PRF_PARAMETERS,
    ),
)
SPEC_BY_ID = {spec.variant_id: spec for spec in SPECS}


def validate_registry(specs: Iterable[P5Spec] = SPECS) -> None:
    materialized = tuple(specs)
    ids = [spec.variant_id for spec in materialized]
    mechanisms = [spec.mechanism for spec in materialized]
    stage_graphs = [spec.stage_graph for spec in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("P5 variant IDs must be unique")
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("P5 mechanisms must be unique")
    if len(stage_graphs) != len(set(stage_graphs)):
        raise ValueError("P5 stage graphs must be unique")
    if set(ids) != {C00, S00, R01}:
        raise ValueError("P5 registry must contain the frozen control, shadow, and active variants")


validate_registry()


@dataclass(slots=True)
class P5Stats:
    turns: int = 0
    activations: int = 0
    output_changes: int = 0
    shadow_changes: int = 0
    fallbacks: int = 0
    prf_candidate_total: int = 0
    top10_newcomer_total: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    route_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "activations": self.activations,
            "output_changes": self.output_changes,
            "shadow_changes": self.shadow_changes,
            "fallbacks": self.fallbacks,
            "prf_candidate_total": self.prf_candidate_total,
            "top10_newcomer_total": self.top10_newcomer_total,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "route_counts": dict(sorted(self.route_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class PrfComputation:
    identifiers: tuple[str, ...]
    route: tuple[str, ...]
    diagnostics: dict[str, Any]


class P5Agent(Agent):
    """Experiment-only R08 wrapper with optional guarded session-local PRF."""

    def __init__(
        self,
        catalog_path: str | Path,
        variant_id: str = C00,
        *,
        question_policy: str = "fast",
    ) -> None:
        if variant_id not in SPEC_BY_ID:
            raise ValueError(f"unknown P5 variant: {variant_id}")
        self.p5_spec = SPEC_BY_ID[variant_id]
        self.p5_stats = P5Stats()
        self.prf_config = PrfConfig(
            seed_depth=5,
            min_query_terms=2,
            min_seed_count=3,
            min_seed_coverage=2,
            min_support_count=3,
            min_support_ratio=0.60,
            max_feedback_terms=4,
            max_df_ratio=0.02,
            min_novel_documents=3,
            route_depth=120,
            rrf_k=60.0,
            prf_weight=0.15,
            max_top10_newcomers=1,
        )
        self._p5_rowids: dict[str, int] = {}
        self._p5_document_count = 0
        super().__init__(
            catalog_path,
            question_policy=question_policy,
            rerank_mode="off",
            retrieval_mode="coverage",
        )

    def _build_index(self) -> None:
        super()._build_index()
        if self.p5_spec.variant_id == C00:
            return
        self._p5_rowids = {
            str(parent_asin): int(rowid)
            for rowid, parent_asin in self.connection.execute(
                "SELECT rowid, parent_asin FROM products"
            )
        }
        self._p5_document_count = len(self._p5_rowids)
        self.connection.execute(
            "CREATE VIRTUAL TABLE products_prf_vocab USING fts5vocab(products, 'row')"
        )
        self.connection.commit()

    def experiment_stats(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "prf_schema_version": PRF_SCHEMA_VERSION,
            "spec": self.p5_spec.as_dict(),
            **self.p5_stats.as_dict(),
        }

    def _empty_prf_diagnostics(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": PRF_SCHEMA_VERSION,
            "lab_schema_version": SCHEMA_VERSION,
            "variant_id": self.p5_spec.variant_id,
            "mode": self.p5_spec.mechanism,
            "active": False,
            "affects_output": self.p5_spec.variant_id == R01,
            "reason": reason,
            "query_term_count": 0,
            "seed_depth": self.prf_config.seed_depth,
            "seed_ids": [],
            "seed_coverages": {},
            "dual_route_seed_count": 0,
            "feedback_terms": [],
            "feedback_term_details": [],
            "fts_expression": "",
            "route_candidate_count": 0,
            "new_candidate_count": 0,
            "would_change_top_10": False,
            "changed_top_10": False,
            "top10_added": [],
            "top10_removed": [],
            "fusion": {},
        }

    def _attach_prf_diagnostics(
        self, state: SessionState, diagnostics: dict[str, Any]
    ) -> None:
        existing = self._ranking_diagnostics.get(
            id(state), self._empty_rerank_diagnostics()
        )
        self._store_ranking_diagnostics(state, {**existing, "prf": diagnostics})

    @staticmethod
    def _dedupe(identifiers: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(identifiers))

    def _load_p5_fields(
        self, identifiers: Iterable[str]
    ) -> dict[str, tuple[str, str, str, str, str, str]]:
        ordered = self._dedupe(identifiers)
        rowids = [self._p5_rowids[value] for value in ordered if value in self._p5_rowids]
        result: dict[str, tuple[str, str, str, str, str, str]] = {}
        for start in range(0, len(rowids), _FIELD_QUERY_CHUNK):
            chunk = rowids[start:start + _FIELD_QUERY_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store, description "
                f"FROM products WHERE rowid IN ({placeholders})",
                chunk,
            ).fetchall()
            result.update({
                str(row[0]): tuple(str(value or "") for value in row[1:])
                for row in rows
            })
        return result

    def _document_frequencies(self, terms: Iterable[str]) -> dict[str, int]:
        ordered = sorted(set(terms))
        result: dict[str, int] = {}
        for start in range(0, len(ordered), _FIELD_QUERY_CHUNK):
            chunk = ordered[start:start + _FIELD_QUERY_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT term, doc FROM products_prf_vocab "
                f"WHERE term IN ({placeholders})",
                chunk,
            ).fetchall()
            result.update((str(term), int(doc)) for term, doc in rows)
        return result

    def _select_seeds(
        self,
        state: SessionState,
        query_terms: list[str],
        rankings: dict[str, list[str]],
    ) -> tuple[list[str], dict[str, Any]]:
        diagnostics = self._ranking_diagnostics.get(
            id(state), self._empty_rerank_diagnostics()
        )
        coverage = diagnostics.get("coverage", {}).get(
            "coverage_by_parent_asin", {}
        )
        top = list(rankings["final"][: self.prf_config.seed_depth])
        maximum = max((int(coverage.get(value, 0)) for value in top), default=0)
        floor = max(self.prf_config.min_seed_coverage, maximum - 1)
        seeds = [value for value in top if int(coverage.get(value, 0)) >= floor]
        detail = {
            "query_term_count": len(set(query_terms)),
            "seed_depth": self.prf_config.seed_depth,
            "seed_ids": seeds,
            "seed_coverages": {
                value: int(coverage.get(value, 0)) for value in seeds
            },
            "maximum_seed_coverage": maximum,
            "seed_coverage_floor": floor,
            "dual_route_seed_count": 0,
        }
        if len(set(query_terms)) < self.prf_config.min_query_terms:
            return [], {**detail, "reason": "insufficient_query_terms"}
        if len(seeds) < self.prf_config.min_seed_count:
            return [], {**detail, "reason": "insufficient_consistent_seeds"}

        broad = set(rankings["broad"][:40])
        strict = set(rankings["strict"][:40])
        dual_count = sum(value in broad and value in strict for value in seeds)
        detail["dual_route_seed_count"] = dual_count
        if strict:
            consistent = dual_count >= 2
        else:
            consistent = all(value in set(rankings["broad"][:30]) for value in seeds)
        if not consistent:
            return [], {**detail, "reason": "route_disagreement"}
        return seeds, {**detail, "reason": "seed_consensus"}

    def _prf_route(self, expression: str) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0), "
            "parent_asin ASC LIMIT ?",
            (expression, self.prf_config.route_depth),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _compute_prf(
        self,
        state: SessionState,
        query_terms: list[str],
        rankings: dict[str, list[str]],
    ) -> PrfComputation:
        baseline = list(rankings["final"])
        seeds, seed_diagnostics = self._select_seeds(state, query_terms, rankings)
        if not seeds:
            diagnostics = {
                **self._empty_prf_diagnostics(seed_diagnostics["reason"]),
                **seed_diagnostics,
            }
            return PrfComputation(tuple(baseline), (), diagnostics)

        seed_fields = self._load_p5_fields(seeds)
        possible_terms = {
            term
            for values in seed_fields.values()
            for term in _terms(" ".join(values[:4]))
        }
        document_frequencies = self._document_frequencies(possible_terms)
        feedback_terms, extraction = extract_feedback_terms(
            query_terms,
            state.excluded_terms,
            seeds,
            seed_fields,
            self._p5_document_count,
            document_frequencies,
            self.prf_config,
        )
        if len(feedback_terms) < 2:
            diagnostics = {
                **self._empty_prf_diagnostics("no_safe_feedback_terms"),
                **seed_diagnostics,
                "reason": "no_safe_feedback_terms",
                "feedback_terms": list(feedback_terms),
                "feedback_term_details": extraction.get("term_diagnostics", []),
                "extraction": extraction,
            }
            return PrfComputation(tuple(baseline), (), diagnostics)

        expression = build_prf_expression(query_terms, feedback_terms)
        route = self._prf_route(expression)
        if not route:
            diagnostics = {
                **self._empty_prf_diagnostics("empty_prf_route"),
                **seed_diagnostics,
                "reason": "empty_prf_route",
                "feedback_terms": list(feedback_terms),
                "feedback_term_details": extraction.get("term_diagnostics", []),
                "extraction": extraction,
                "fts_expression": expression,
            }
            return PrfComputation(tuple(baseline), (), diagnostics)

        union = self._dedupe([*rankings["fused"], *route])
        searchable_fields = self._load_p5_fields(union)
        proposed, fusion = guarded_prf_fusion(
            query_terms,
            state.excluded_terms,
            feedback_terms,
            rankings,
            route,
            searchable_fields,
            self.prf_config,
            _terms,
        )
        baseline_top = baseline[:10]
        proposed_top = proposed[:10]
        added = [value for value in proposed_top if value not in baseline_top]
        removed = [value for value in baseline_top if value not in proposed_top]
        changed = proposed_top != baseline_top
        diagnostics = {
            **self._empty_prf_diagnostics(
                "safe_promotion" if changed else "no_safe_promotion"
            ),
            **seed_diagnostics,
            "reason": "safe_promotion" if changed else "no_safe_promotion",
            "active": True,
            "feedback_terms": list(feedback_terms),
            "feedback_term_details": extraction.get("term_diagnostics", []),
            "extraction": extraction,
            "fts_expression": expression,
            "route_candidate_count": len(route),
            "new_candidate_count": len(set(route) - set(rankings["fused"])),
            "would_change_top_10": changed,
            "top10_added": added,
            "top10_removed": removed,
            "fusion": fusion,
        }
        return PrfComputation(tuple(proposed), tuple(route), diagnostics)

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        baseline = list(rankings["final"])
        self.p5_stats.turns += 1
        if self.p5_spec.variant_id == C00:
            diagnostics = self._empty_prf_diagnostics("control")
            self._attach_prf_diagnostics(state, diagnostics)
            return {**rankings, "coverage": baseline, "prf": []}

        query_terms = self._query_terms(state)
        computation = self._compute_prf(state, query_terms, rankings)
        proposed = list(computation.identifiers)
        served = proposed if self.p5_spec.variant_id == R01 else baseline
        proposed_changed = proposed[:10] != baseline[:10]
        output_changed = served[:10] != baseline[:10]
        diagnostics = {
            **computation.diagnostics,
            "affects_output": self.p5_spec.variant_id == R01,
            "changed_top_10": output_changed,
        }
        self._attach_prf_diagnostics(state, diagnostics)

        reason = str(diagnostics["reason"])
        self.p5_stats.activations += int(bool(diagnostics["active"]))
        self.p5_stats.output_changes += int(output_changed)
        self.p5_stats.shadow_changes += int(proposed_changed)
        self.p5_stats.fallbacks += int(not proposed_changed)
        self.p5_stats.prf_candidate_total += len(computation.route)
        self.p5_stats.top10_newcomer_total += len(diagnostics["top10_added"])
        self.p5_stats.reason_counts[reason] += 1
        if computation.route:
            self.p5_stats.route_counts["prf"] += 1
        return {
            **rankings,
            "coverage": baseline,
            "prf": list(computation.route),
            "final": served,
        }

    def debug_prf_diagnostics(self, session_id: str) -> dict[str, Any]:
        """Return a JSON-safe copy of current label-blind feedback diagnostics."""

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            state = self._sessions[session_id]
            diagnostics = self._ranking_diagnostics.get(id(state))
            if diagnostics is None or "prf" not in diagnostics:
                self._rank_candidates(state)
                diagnostics = self._ranking_diagnostics[id(state)]
            return json.loads(json.dumps(diagnostics["prf"]))


__all__ = [
    "ACTIVE_ID",
    "C00",
    "CONTROL_ID",
    "R01",
    "S00",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "SPECS",
    "SPEC_BY_ID",
    "P5Agent",
    "P5Spec",
    "PrfComputation",
    "validate_registry",
]
