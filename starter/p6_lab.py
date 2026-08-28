"""P6-only guarded broad-depth experiment.

The submitted :class:`starter.agent.Agent` remains unchanged.  This wrapper
reuses its exact served R08 pipeline, optionally computes the same broad FTS5
route at depth 240, and delegates every admission decision to
``starter.adaptive_depth``.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from starter.adaptive_depth import (
    SCHEMA_VERSION as ADAPTIVE_DEPTH_SCHEMA_VERSION,
    DepthConfig,
    depth_precheck,
    guarded_depth_admission,
)
from starter.agent import Agent, SessionState, _terms


SCHEMA_VERSION = "p6.adaptive-depth-lab.v1"
C00 = "P6.C00.r08_coverage"
S00 = "P6.S00.adaptive_depth_shadow"
R01 = "P6.R01.guarded_broad_depth_doubling"
CONTROL_ID = C00
SHADOW_ID = S00
ACTIVE_ID = R01
_AUDIT_RECORD_LIMIT = 2_000


@dataclass(frozen=True, slots=True)
class P6Spec:
    variant_id: str
    family: str
    mechanism: str
    stage_graph: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_DEPTH_PARAMETERS = (
    ("base_broad_depth", 120),
    ("deep_depth", 240),
    ("top_k", 10),
    ("protected_prefix", 9),
    ("max_top10_newcomers", 1),
    ("min_query_terms", 2),
    ("strict_coverage_margin", 1),
)

SPECS = (
    P6Spec(
        C00,
        "control",
        "r08_coverage",
        ("visible_state", "broad120+strict80_rrf", "coverage_cascade", "top10"),
        "Exact served R08 coverage baseline.",
    ),
    P6Spec(
        S00,
        "diagnostic",
        "adaptive_depth_shadow",
        (
            "r08_coverage",
            "same_broad_fts_depth240",
            "prefix_validation",
            "guarded_tail_proposal",
            "shadow_only",
        ),
        "Computes a guarded depth proposal without changing recommendations.",
        _DEPTH_PARAMETERS,
    ),
    P6Spec(
        R01,
        "retrieval",
        "guarded_broad_depth_doubling",
        (
            "r08_coverage",
            "same_broad_fts_depth240",
            "prefix_validation",
            "guarded_tail_admission",
            "top10",
        ),
        "Allows at most one higher-coverage deep-route item at rank ten.",
        _DEPTH_PARAMETERS,
    ),
)
SPEC_BY_ID = {spec.variant_id: spec for spec in SPECS}


def validate_registry(specs: Iterable[P6Spec] = SPECS) -> None:
    materialized = tuple(specs)
    ids = [spec.variant_id for spec in materialized]
    mechanisms = [spec.mechanism for spec in materialized]
    stage_graphs = [spec.stage_graph for spec in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("P6 variant IDs must be unique")
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("P6 mechanisms must be unique")
    if len(stage_graphs) != len(set(stage_graphs)):
        raise ValueError("P6 stage graphs must be unique")
    if set(ids) != {C00, S00, R01}:
        raise ValueError("P6 registry must contain the frozen control, shadow, and active variants")


validate_registry()


@dataclass(slots=True)
class P6Stats:
    turns: int = 0
    triggers: int = 0
    deep_queries: int = 0
    guard_admissions: int = 0
    activations: int = 0
    output_changes: int = 0
    shadow_changes: int = 0
    fallbacks: int = 0
    prefix_mismatches: int = 0
    exceptions: int = 0
    deep_candidate_total: int = 0
    newcomer_total: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "triggers": self.triggers,
            "deep_queries": self.deep_queries,
            "guard_admissions": self.guard_admissions,
            "activations": self.activations,
            "output_changes": self.output_changes,
            "shadow_changes": self.shadow_changes,
            "fallbacks": self.fallbacks,
            "prefix_mismatches": self.prefix_mismatches,
            "exceptions": self.exceptions,
            "deep_candidate_total": self.deep_candidate_total,
            "newcomer_total": self.newcomer_total,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class DepthComputation:
    identifiers: tuple[str, ...]
    route: tuple[str, ...]
    diagnostics: dict[str, Any]


class P6Agent(Agent):
    """Experiment-only R08 wrapper with optional guarded broad-depth admission."""

    def __init__(
        self,
        catalog_path: str | Path,
        variant_id: str = C00,
        *,
        question_policy: str = "fast",
    ) -> None:
        if variant_id not in SPEC_BY_ID:
            raise ValueError(f"unknown P6 variant: {variant_id}")
        self.p6_spec = SPEC_BY_ID[variant_id]
        self.p6_stats = P6Stats()
        self.depth_config = DepthConfig(
            base_broad_depth=120,
            deep_depth=240,
            top_k=10,
            protected_prefix=9,
            max_top10_newcomers=1,
            min_query_terms=2,
            strict_coverage_margin=1,
        )
        self._state_session_indexes: dict[int, int] = {}
        self._next_session_index = 0
        self._p6_audit: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        super().__init__(
            catalog_path,
            question_policy=question_policy,
            rerank_mode="off",
            retrieval_mode="coverage",
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._lock:
            previous = self._sessions.get(session_id)
            if previous is not None:
                self._state_session_indexes.pop(id(previous), None)
            session_index = self._next_session_index
            self._next_session_index += 1
            super().reset(session_id, user_profile)
            self._state_session_indexes[id(self._sessions[session_id])] = session_index

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            super().drop_session(session_id)
            if state is not None:
                self._state_session_indexes.pop(id(state), None)

    def experiment_stats(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "adaptive_depth_schema_version": ADAPTIVE_DEPTH_SCHEMA_VERSION,
            "spec": self.p6_spec.as_dict(),
            **self.p6_stats.as_dict(),
        }

    def experiment_audit(self) -> list[dict[str, Any]]:
        """Return ordered, JSON-safe records derived only from visible state."""

        with self._lock:
            return json.loads(json.dumps(list(self._p6_audit.values())))

    def _empty_depth_diagnostics(self, reason: str) -> dict[str, Any]:
        config = asdict(self.depth_config)
        return {
            "schema_version": ADAPTIVE_DEPTH_SCHEMA_VERSION,
            "lab_schema_version": SCHEMA_VERSION,
            "variant_id": self.p6_spec.variant_id,
            "mode": self.p6_spec.mechanism,
            "active": False,
            "triggered": False,
            "affects_output": self.p6_spec.variant_id == R01,
            "reason": reason,
            "deep_query_executed": False,
            "config": config,
            "query_terms": [],
            "excluded_terms": [],
            "trigger": {
                "enabled": False,
                "rejection_reasons": [reason],
                "broad_count": 0,
                "final_count": 0,
            },
            "prefix": {"matches": None, "checked_count": 0},
            "tail": {"candidate_count": 0, "new_candidate_count": 0, "proposals": []},
            "guard": {
                "applied": False,
                "reason": reason,
                "protected_top9": [],
                "incumbent": None,
                "replacement": None,
                "top9_unchanged": True,
                "newcomers": [],
            },
            "final_top10": [],
            "would_change_top_10": False,
            "changed_top_10": False,
            "target_blind": True,
            "label_free": True,
            "route_audit": {
                "base_broad_ids": [],
                "deep_broad_ids": [],
                "baseline_top10": [],
                "proposal_top10": [],
                "output_top10": [],
            },
        }

    def _attach_depth_diagnostics(
        self, state: SessionState, diagnostics: dict[str, Any]
    ) -> None:
        existing = self._ranking_diagnostics.get(
            id(state), self._empty_rerank_diagnostics()
        )
        self._store_ranking_diagnostics(
            state, {**existing, "adaptive_depth": diagnostics}
        )

    def _deep_broad_route(
        self, query_terms: list[str]
    ) -> tuple[list[str], dict[str, int]]:
        expression = self._fts_expression(query_terms)
        if not expression:
            return [], {}
        rows = self.connection.execute(
            "SELECT rowid, parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 240",
            (expression,),
        ).fetchall()
        return (
            [str(row[1]) for row in rows],
            {str(row[1]): int(row[0]) for row in rows},
        )

    @staticmethod
    def _reason(diagnostics: dict[str, Any]) -> str:
        reason = diagnostics.get("reason")
        if reason:
            return str(reason)
        guard = diagnostics.get("guard", {})
        if guard.get("reason"):
            return str(guard["reason"])
        rejected = diagnostics.get("trigger", {}).get("rejection_reasons", [])
        return str(rejected[0]) if rejected else "unknown"

    def _compute_depth(
        self,
        state: SessionState,
        query_terms: list[str],
        rankings: dict[str, list[str]],
    ) -> DepthComputation:
        baseline = list(rankings["final"])
        existing = self._ranking_diagnostics.get(
            id(state), self._empty_rerank_diagnostics()
        )
        baseline_coverage = existing.get("coverage", {}).get(
            "coverage_by_parent_asin", {}
        )
        precheck = depth_precheck(
            query_terms,
            rankings,
            baseline_coverage,
            self.depth_config,
        )
        if not precheck["enabled"]:
            proposed, diagnostics = guarded_depth_admission(
                query_terms,
                state.excluded_terms,
                rankings,
                list(rankings.get("broad", [])),
                baseline_coverage,
                {},
                self.depth_config,
                _terms,
            )
            diagnostics = {**diagnostics, "deep_query_executed": False}
            return DepthComputation(tuple(proposed), (), diagnostics)

        deep_ids: list[str] = []
        try:
            deep_ids, rowids = self._deep_broad_route(query_terms)
            existing_pool = {
                identifier
                for route in ("broad", "strict", "fused", "final")
                for identifier in rankings.get(route, [])
            }
            field_ids = [
                identifier
                for identifier in deep_ids[
                    self.depth_config.base_broad_depth : self.depth_config.deep_depth
                ]
                if identifier not in existing_pool
            ]
            searchable_fields = self._load_coverage_fields(field_ids, rowids)
            proposed, diagnostics = guarded_depth_admission(
                query_terms,
                state.excluded_terms,
                rankings,
                deep_ids,
                baseline_coverage,
                searchable_fields,
                self.depth_config,
                _terms,
            )
            diagnostics = {**diagnostics, "deep_query_executed": True}
            return DepthComputation(tuple(proposed), tuple(deep_ids), diagnostics)
        except Exception as error:  # preserve the exact served order on lab failures
            diagnostics = self._empty_depth_diagnostics("exception_fallback")
            diagnostics.update({
                "query_terms": list(precheck["query_terms"]),
                "excluded_terms": sorted(state.excluded_terms),
                "deep_query_executed": True,
                "exception_class": type(error).__name__,
                "trigger": {
                    "enabled": False,
                    "conditions": {
                        **precheck["conditions"],
                        "prefix_matches": None,
                    },
                    "rejection_reasons": ["exception_fallback"],
                    "broad_count": precheck["broad_count"],
                    "final_count": precheck["final_count"],
                    "incumbent": precheck["incumbent"],
                    "incumbent_coverage": precheck["incumbent_coverage"],
                },
                "coverage": {
                    "coverage_by_parent_asin": {
                        identifier: baseline_coverage[identifier]
                        for identifier in baseline[: self.depth_config.top_k]
                        if identifier in baseline_coverage
                    },
                    "matched_excluded_terms_by_parent_asin": {},
                },
                "guard": {
                    **diagnostics["guard"],
                    "protected_top9": baseline[: self.depth_config.protected_prefix],
                    "incumbent": precheck["incumbent"],
                    "incumbent_coverage": precheck["incumbent_coverage"],
                },
                "final_top10": baseline[: self.depth_config.top_k],
            })
            return DepthComputation(tuple(baseline), tuple(deep_ids), diagnostics)

    def _record_audit(
        self,
        state: SessionState,
        rankings: dict[str, list[str]],
        route: list[str],
        baseline: list[str],
        proposed: list[str],
        served: list[str],
        diagnostics: dict[str, Any],
    ) -> None:
        session_index = self._state_session_indexes.get(id(state), -1)
        turn = len(state.messages)
        coverage = diagnostics.get("coverage", {})
        broad_pool = list(rankings.get("broad", []))[
            : self.depth_config.base_broad_depth
        ]
        strict_pool = list(rankings.get("strict", []))
        deep_pool = list(route)[: self.depth_config.deep_depth]
        base_union_pool = list(dict.fromkeys([*broad_pool, *strict_pool]))
        deep_query_executed = bool(diagnostics.get("deep_query_executed"))
        valid_deep_evidence = (
            deep_query_executed
            and diagnostics.get("prefix", {}).get("matches") is True
        )
        deep_union_pool = (
            list(dict.fromkeys([*deep_pool, *strict_pool]))
            if valid_deep_evidence
            else list(base_union_pool)
        )
        record = {
            "session_index": session_index,
            "turn": turn,
            "query_terms": self._query_terms(state),
            "excluded_terms": sorted(state.excluded_terms),
            "base_pool": broad_pool,
            "deep_pool": deep_pool,
            "strict_pool": strict_pool,
            "base_union_pool": base_union_pool,
            "deep_union_pool": deep_union_pool,
            "baseline_top10": baseline[: self.depth_config.top_k],
            "proposal_top10": proposed[: self.depth_config.top_k],
            "served_top10": served[: self.depth_config.top_k],
            "active": bool(diagnostics.get("active")),
            "deep_query_executed": deep_query_executed,
            "prefix": diagnostics.get("prefix", {}),
            "trigger": diagnostics.get("trigger", {}),
            "guard": diagnostics.get("guard", {}),
            "coverage_by_parent_asin": coverage.get(
                "coverage_by_parent_asin", {}
            ),
            "matched_excluded_terms_by_parent_asin": coverage.get(
                "matched_excluded_terms_by_parent_asin", {}
            ),
            "reason": self._reason(diagnostics),
        }
        key = (session_index, turn)
        self._p6_audit[key] = record
        self._p6_audit.move_to_end(key)
        while len(self._p6_audit) > _AUDIT_RECORD_LIMIT:
            self._p6_audit.popitem(last=False)

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        rankings = super()._rank_candidates(state)
        baseline = list(rankings["final"])
        self.p6_stats.turns += 1

        if self.p6_spec.variant_id == C00:
            computation = DepthComputation(
                tuple(baseline),
                (),
                self._empty_depth_diagnostics("control"),
            )
        else:
            query_terms = self._query_terms(state)
            try:
                computation = self._compute_depth(state, query_terms, rankings)
            except Exception as error:  # exact control fallback is an experiment invariant
                diagnostics = self._empty_depth_diagnostics("exception_fallback")
                diagnostics["exception_class"] = type(error).__name__
                computation = DepthComputation(tuple(baseline), (), diagnostics)

        proposed = list(computation.identifiers)
        route = list(computation.route)
        served = proposed if self.p6_spec.variant_id == R01 else baseline
        proposed_changed = proposed[:10] != baseline[:10]
        output_changed = served[:10] != baseline[:10]
        diagnostics = {
            **computation.diagnostics,
            "lab_schema_version": SCHEMA_VERSION,
            "variant_id": self.p6_spec.variant_id,
            "mode": self.p6_spec.mechanism,
            "triggered": bool(
                computation.diagnostics.get(
                    "triggered",
                    computation.diagnostics.get("trigger", {}).get("enabled", False),
                )
            ),
            "active": bool(
                computation.diagnostics.get(
                    "active",
                    computation.diagnostics.get("guard", {}).get("applied", False),
                )
            ),
            "deep_query_executed": bool(
                computation.diagnostics.get("deep_query_executed", False)
            ),
            "affects_output": self.p6_spec.variant_id == R01,
            "would_change_top_10": proposed_changed,
            "changed_top_10": output_changed,
            "route_audit": {
                "base_broad_ids": list(rankings.get("broad", []))[:120],
                "deep_broad_ids": route[:240],
                "baseline_top10": baseline[:10],
                "proposal_top10": proposed[:10],
                "output_top10": served[:10],
            },
        }
        self._attach_depth_diagnostics(state, diagnostics)
        self._record_audit(
            state, rankings, route, baseline, proposed, served, diagnostics
        )

        reason = self._reason(diagnostics)
        prefix_matches = diagnostics.get("prefix", {}).get("matches")
        guard = diagnostics.get("guard", {})
        newcomers = guard.get("newcomers", [])
        triggered = bool(diagnostics.get("triggered"))
        deep_queried = bool(diagnostics.get("deep_query_executed"))
        admitted = bool(guard.get("applied"))
        self.p6_stats.triggers += int(triggered)
        self.p6_stats.deep_queries += int(deep_queried)
        self.p6_stats.guard_admissions += int(admitted)
        self.p6_stats.activations += int(admitted)
        self.p6_stats.output_changes += int(output_changed)
        self.p6_stats.shadow_changes += int(proposed_changed)
        self.p6_stats.fallbacks += int(not proposed_changed)
        self.p6_stats.prefix_mismatches += int(prefix_matches is False)
        self.p6_stats.exceptions += int(reason == "exception_fallback")
        self.p6_stats.deep_candidate_total += len(route)
        self.p6_stats.newcomer_total += len(newcomers)
        self.p6_stats.reason_counts[reason] += 1
        return {
            **rankings,
            "adaptive_depth": route,
            "final": served,
        }

    def debug_adaptive_depth_diagnostics(self, session_id: str) -> dict[str, Any]:
        """Return a JSON-safe copy of current visible-state depth diagnostics."""

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            state = self._sessions[session_id]
            diagnostics = self._ranking_diagnostics.get(id(state))
            if diagnostics is None or "adaptive_depth" not in diagnostics:
                self._rank_candidates(state)
                diagnostics = self._ranking_diagnostics[id(state)]
            return json.loads(json.dumps(diagnostics["adaptive_depth"]))


__all__ = [
    "ACTIVE_ID",
    "C00",
    "CONTROL_ID",
    "DepthComputation",
    "P6Agent",
    "P6Spec",
    "R01",
    "S00",
    "SCHEMA_VERSION",
    "SHADOW_ID",
    "SPECS",
    "SPEC_BY_ID",
    "validate_registry",
]
