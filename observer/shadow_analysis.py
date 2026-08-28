"""Target-blind aggregate diagnostics for candidate-aware clarification shadow runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Any, Iterable

from starter.slot_ledger import normalize_slot


SCHEMA_VERSION = "p3.shadow-policy-analysis.v1"
_COMPONENTS = (
    "score",
    "information_gain",
    "coverage",
    "answerability",
    "turn_cost",
)


class ShadowPolicyRecorder:
    """Collect policy diagnostics without reading targets, intent cards, or behavior."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        sample_id: object,
        scenario_type: object,
        turn: int,
        actual_attribute: object,
        question_shadow: dict[str, Any] | None,
    ) -> None:
        shadow = question_shadow or {}
        candidates = shadow.get("candidates")
        top = candidates[0] if isinstance(candidates, list) and candidates else {}
        selected = shadow.get("selected_attribute")
        actual = actual_attribute if isinstance(actual_attribute, str) else None
        selected = selected if isinstance(selected, str) else None
        blocked = {
            normalize_slot(value)
            for value in shadow.get("blocked_attributes", [])
            if isinstance(value, str)
        }
        event = {
            "sample_id": str(sample_id),
            "scenario_type": str(scenario_type),
            "turn": int(turn),
            "actual_attribute": actual,
            "shadow_attribute": selected,
            "disagrees": actual != selected,
            "candidate_count": int(shadow.get("candidate_count") or 0),
            "blocked_selection_violation": bool(
                selected and normalize_slot(selected) in blocked
            ),
            "reason": shadow.get("reason"),
        }
        for component in _COMPONENTS:
            value = top.get(component) if isinstance(top, dict) else None
            event[component] = float(value) if isinstance(value, (int, float)) else None
        self.events.append(event)

    @staticmethod
    def _summary(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = list(events)
        actual_counts = Counter(row["actual_attribute"] or "none" for row in rows)
        shadow_counts = Counter(row["shadow_attribute"] or "none" for row in rows)
        disagreements = sum(bool(row["disagrees"]) for row in rows)
        component_means = {
            component: round(fmean(values), 6) if values else None
            for component in _COMPONENTS
            if (values := [row[component] for row in rows if row[component] is not None])
        }
        return {
            "turn_count": len(rows),
            "candidate_evidence_turns": sum(row["candidate_count"] > 0 for row in rows),
            "actual_question_turns": sum(row["actual_attribute"] is not None for row in rows),
            "shadow_question_turns": sum(row["shadow_attribute"] is not None for row in rows),
            "disagreement_count": disagreements,
            "disagreement_rate": round(disagreements / len(rows), 6) if rows else 0.0,
            "blocked_selection_violations": sum(
                bool(row["blocked_selection_violation"]) for row in rows
            ),
            "actual_attribute_counts": dict(sorted(actual_counts.items())),
            "shadow_attribute_counts": dict(sorted(shadow_counts.items())),
            "selected_component_means": component_means,
        }

    def artifact(self) -> dict[str, Any]:
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            by_scenario[event["scenario_type"]].append(event)
        return {
            "schema_version": SCHEMA_VERSION,
            "target_blind": True,
            "excluded_inputs": ["ground_truth", "intent_card", "behavior", "target rank"],
            "summary": self._summary(self.events),
            "scenario_summaries": {
                scenario: self._summary(events)
                for scenario, events in sorted(by_scenario.items())
            },
            "events": list(self.events),
        }
