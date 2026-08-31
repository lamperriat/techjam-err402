"""Default-off adapter around the vendored teammate V1 retrieval core.

The adapter consumes only current visible conversation state.  It preserves V1's
CatalogIndex candidate generation and ProductScorer ordering, then applies three
bounded safety transforms: explicit hard-conflict demotion, immediate no-repeat,
and atomic selective state updates.  Any contract failure returns the caller's
fallback order without committing adapter memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vendor.teammate_v1.err402.agents.v1 import (
    AgentV1,
    Constraint as V1Constraint,
    QUESTION_TEMPLATES,
    SessionState as V1SessionState,
)
from vendor.teammate_v1.err402.retrieval.catalog import (
    CandidatePool,
    CatalogIndex,
    extract_department,
    normalized_text,
)
from vendor.teammate_v1.err402.retrieval.scoring import (
    ProductScorer,
    QueryContext,
    ScoredProduct,
    ScoringConfig,
)


SCHEMA_VERSION = "fusion-core.version-a.v1"
MODE_OFF = "off"
MODE_ACTIVE = "active"
_ALLOWED_MODES = frozenset({MODE_OFF, MODE_ACTIVE})
_CLOSED_VALUE_FIELDS = {
    "audience": "department",
    "brand": "brand",
    "color": "color",
    "material": "material",
    "size": "size",
    "style": "style",
    "use_case": "use_case",
}
_SLOT_ALIASES = {
    "budget": "price",
    "department": "audience",
    "feature_phrases": "feature",
    "size_fit": "size",
}
_BUDGET_PATTERNS = (
    re.compile(r"(?:under|below|less than|up to|maximum|max)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"budget(?:\s+(?:around|of))?\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
)


class FusionCoreContractError(ValueError):
    """Raised internally when visible state or V1 output violates Version A."""


@dataclass(frozen=True, slots=True)
class VisibleConstraint:
    slot: str
    value: str
    polarity: int
    hardness: str
    source_turn: int
    version: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.slot, self.value, self.polarity)


@dataclass(frozen=True, slots=True)
class VisibleProjection:
    category: str
    intent: str
    version: int
    constraints: tuple[VisibleConstraint, ...]
    asked_attributes: frozenset[str]
    no_preference_attributes: frozenset[str]
    profile: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FusionOutcome:
    identifiers: tuple[str, ...]
    ask_attribute: str | None
    diagnostics: Mapping[str, Any]
    scores: tuple[float, ...] = ()


@dataclass(slots=True)
class _SessionMemory:
    category: str = ""
    version: int = 0
    last_turn: int = 0
    served: set[str] = field(default_factory=set)
    constraint_keys: frozenset[tuple[str, str, int]] = frozenset()

    def clone(self) -> _SessionMemory:
        return _SessionMemory(
            category=self.category,
            version=self.version,
            last_turn=self.last_turn,
            served=set(self.served),
            constraint_keys=self.constraint_keys,
        )


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalized_slot(value: object) -> str:
    slot = normalized_text(value).replace(" ", "_")
    return _SLOT_ALIASES.get(slot, slot)


def _identifiers(values: Sequence[object], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise FusionCoreContractError(f"{label}_shape")
    result = tuple(str(item) for item in values)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise FusionCoreContractError(f"{label}_identity")
    return result


def _active_records(state: object) -> tuple[object, ...]:
    ledger = _value(state, "slot_ledger")
    if ledger is not None and callable(getattr(ledger, "active_records", None)):
        return tuple(ledger.active_records())
    records = _value(state, "active_records")
    if records is not None:
        return tuple(records)
    if isinstance(ledger, Mapping):
        return tuple(ledger.get("active", ()))
    return ()


def _string_set(source: object, name: str) -> frozenset[str]:
    raw = _value(source, name, ()) or ()
    if isinstance(raw, (str, bytes, bytearray)):
        raise FusionCoreContractError(f"{name}_shape")
    return frozenset(
        item for item in (normalized_text(value) for value in raw) if item
    )


def project_visible_state(state: object) -> VisibleProjection:
    """Build the immutable Version A view from active ledger state only."""

    category = str(_value(state, "category_text", _value(state, "category", ""))).strip()
    version = _value(state, "version", 1)
    if not category or type(version) is not int or version < 1:
        raise FusionCoreContractError("visible_state_header")
    intent = normalized_text(_value(state, "intent", "buying"))
    if intent not in {"buying", "browsing"}:
        intent = "buying"

    constraints: list[VisibleConstraint] = []
    seen: set[tuple[str, str, int]] = set()
    for record in _active_records(state):
        if _value(record, "status", "active") != "active":
            continue
        slot = _normalized_slot(_value(record, "slot", ""))
        value = normalized_text(_value(record, "value", ""))
        polarity = _value(record, "polarity", 0)
        hardness = normalized_text(_value(record, "hardness", "soft"))
        source_turn = _value(record, "source_turn", 0)
        record_version = _value(record, "version", version)
        if (
            not slot
            or not value
            or polarity not in {-1, 1}
            or hardness not in {"hard", "soft"}
            or type(source_turn) is not int
            or source_turn < 1
            or type(record_version) is not int
            or not 1 <= record_version <= version
        ):
            raise FusionCoreContractError("visible_constraint")
        key = (slot, value, polarity)
        if key in seen:
            continue
        if (slot, value, -polarity) in seen:
            raise FusionCoreContractError("opposed_active_constraints")
        seen.add(key)
        constraints.append(
            VisibleConstraint(slot, value, polarity, hardness, source_turn, record_version)
        )

    profile = _value(state, "profile", _value(state, "user_profile", {})) or {}
    if not isinstance(profile, Mapping):
        raise FusionCoreContractError("profile_shape")
    asked = _string_set(state, "asked_attributes")
    exhausted = _string_set(state, "exhausted_attributes")
    return VisibleProjection(
        category=category,
        intent=intent,
        version=version,
        constraints=tuple(constraints),
        asked_attributes=asked - {"other"},
        no_preference_attributes=exhausted - {"other"},
        profile=dict(profile),
    )


def _constraint_text(constraint: VisibleConstraint) -> str:
    return (
        f"not {constraint.value}"
        if constraint.polarity < 0
        else constraint.value
    )


def _budget(constraints: Iterable[VisibleConstraint]) -> tuple[float, bool] | None:
    result: tuple[float, bool] | None = None
    for constraint in constraints:
        if constraint.polarity < 0:
            continue
        for pattern in _BUDGET_PATTERNS:
            match = pattern.search(constraint.value)
            if match:
                number = float(match.group(1))
                if math.isfinite(number) and number >= 0.0:
                    result = (number, constraint.hardness == "hard")
                break
    return result


def _known_candidate_value(product: object, slot: str) -> str | None:
    field = _CLOSED_VALUE_FIELDS.get(slot)
    if field is None:
        return None
    raw = getattr(product, field, None)
    return normalized_text(raw) if raw not in (None, "") else None


def _explicit_hard_conflict(
    product: object,
    constraints: Iterable[VisibleConstraint],
) -> bool:
    """Return true only for known contradictory evidence; missing is neutral."""

    budget = _budget(constraints)
    if budget is not None and budget[1]:
        price = getattr(product, "price", None)
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            if math.isfinite(float(price)) and float(price) > budget[0]:
                return True
    for constraint in constraints:
        if constraint.hardness != "hard":
            continue
        known = _known_candidate_value(product, constraint.slot)
        if known is None:
            continue
        if constraint.polarity < 0 and known == constraint.value:
            return True
        if constraint.polarity > 0 and known != constraint.value:
            return True
    return False


class FusionCore:
    """Version A runtime. Disabled mode is an exact no-op fallback."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        mode: str = MODE_OFF,
        scoring_config: ScoringConfig | None = None,
        catalog: Any | None = None,
        scorer: Any | None = None,
    ) -> None:
        if mode not in _ALLOWED_MODES:
            raise ValueError("mode must be off or active")
        self.mode = mode
        self.catalog = catalog
        self.scorer = scorer
        if mode == MODE_ACTIVE:
            self.catalog = catalog if catalog is not None else CatalogIndex(catalog_path)
            self.scorer = scorer if scorer is not None else ProductScorer(
                self.catalog, scoring_config
            )
        self._sessions: dict[str, _SessionMemory] = {}

    def reset(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        self._sessions[session_id] = _SessionMemory()

    def close(self) -> None:
        close = getattr(self.catalog, "close", None)
        if callable(close):
            close()

    def debug_memory(self, session_id: str) -> Mapping[str, Any]:
        memory = self._sessions[session_id]
        return {
            "category": memory.category,
            "version": memory.version,
            "last_turn": memory.last_turn,
            "served": tuple(sorted(memory.served)),
            "constraint_keys": tuple(sorted(memory.constraint_keys)),
        }

    def _fallback(
        self,
        fallback: tuple[str, ...],
        reason: str,
    ) -> FusionOutcome:
        return FusionOutcome(
            fallback,
            None,
            {
                "schema_version": SCHEMA_VERSION,
                "configured_mode": self.mode,
                "effective_mode": "off" if self.mode == MODE_OFF else "fallback",
                "fallback": self.mode != MODE_OFF,
                "reason_code": reason,
                "output_changed": False,
                "state_committed": False,
            },
            (),
        )

    def apply(
        self,
        session_id: str,
        visible_state: object,
        *,
        turn: int,
        top_k: int,
        fallback_order: Sequence[object],
        supplemental_order: Sequence[object] = (),
    ) -> FusionOutcome:
        fallback = _identifiers(fallback_order, "fallback_order")
        if self.mode == MODE_OFF:
            return self._fallback(fallback, "disabled")
        if session_id not in self._sessions:
            return self._fallback(fallback, "session_not_reset")
        if type(turn) is not int or turn < 1 or type(top_k) is not int or top_k < 1:
            return self._fallback(fallback, "call_contract")

        original = self._sessions[session_id]
        working = original.clone()
        try:
            if turn <= working.last_turn:
                raise FusionCoreContractError("turn_not_monotonic")
            projection = project_visible_state(visible_state)
            if working.version and projection.version < working.version:
                raise FusionCoreContractError("version_regressed")
            category_changed = bool(
                working.category
                and normalized_text(working.category) != normalized_text(projection.category)
            )
            intent_version_changed = bool(
                working.version and projection.version != working.version
            )
            if category_changed or intent_version_changed:
                working.served.clear()
            if category_changed:
                working.constraint_keys = frozenset()

            v1_constraints = [
                V1Constraint(
                    _constraint_text(item),
                    item.hardness == "hard",
                    "fusion_visible_state",
                    f"{item.slot}:{item.source_turn}:{item.version}",
                )
                for item in projection.constraints
            ]
            constraint_texts = tuple(item.text for item in v1_constraints)
            query_text = " ".join((projection.category, *constraint_texts))
            pool = self.catalog.candidates(projection.category, query_text)
            supplemental = _identifiers(supplemental_order, "supplemental_order")
            v1_candidate_count = len(pool.parent_asins)
            combined = tuple(dict.fromkeys((*pool.parent_asins, *supplemental)))
            pool = CandidatePool(combined, dict(pool.lexical_ranks))
            budget = AgentV1._budget_constraint(v1_constraints)
            context = QueryContext(
                intent=projection.intent,
                category=projection.category,
                constraints=constraint_texts,
                department=extract_department(query_text),
                budget=budget,
            )
            scored = self.scorer.score(pool, context)
            ranked = AgentV1._prioritize_exact_category(scored, projection.category)
            identifiers = tuple(item.product.parent_asin for item in ranked)
            if len(identifiers) != len(set(identifiers)) or len(identifiers) < top_k:
                raise FusionCoreContractError("v1_ranked_order")

            safe: list[ScoredProduct] = []
            conflicts: list[ScoredProduct] = []
            for item in ranked:
                (conflicts if _explicit_hard_conflict(
                    item.product, projection.constraints
                ) else safe).append(item)
            conflict_safe = [*safe, *conflicts]

            unseen = [
                item for item in conflict_safe
                if item.product.parent_asin not in working.served
            ]
            seen = [
                item for item in conflict_safe
                if item.product.parent_asin in working.served
            ]
            proposed_items = [*unseen, *seen]
            proposed = tuple(item.product.parent_asin for item in proposed_items)
            if len(proposed) != len(identifiers) or set(proposed) != set(identifiers):
                raise FusionCoreContractError("membership_changed")

            question_state = V1SessionState(
                user_profile=dict(projection.profile),
                intent=projection.intent,
                category=projection.category,
                constraints=v1_constraints,
                asked_attributes=set(projection.asked_attributes),
                no_preference_attributes=set(projection.no_preference_attributes),
            )
            question_host = AgentV1.__new__(AgentV1)
            question_host.catalog = self.catalog
            ask_attribute = AgentV1._select_question(
                question_host, question_state, proposed_items[:100]
            )
            if ask_attribute == "other":
                raise FusionCoreContractError("other_question_forbidden")

            new_keys = frozenset(item.key for item in projection.constraints)
            removed = working.constraint_keys - new_keys
            added = new_keys - working.constraint_keys
            working.category = projection.category
            working.version = projection.version
            working.last_turn = turn
            working.constraint_keys = new_keys
            working.served.update(proposed[:top_k])
            self._sessions[session_id] = working
            return FusionOutcome(
                proposed,
                ask_attribute,
                {
                    "schema_version": SCHEMA_VERSION,
                    "configured_mode": self.mode,
                    "effective_mode": MODE_ACTIVE,
                    "fallback": False,
                    "reason_code": "ranked",
                    "output_changed": proposed[:top_k] != fallback[:top_k],
                    "state_committed": True,
                    "candidate_count": len(proposed),
                    "v1_candidate_count": v1_candidate_count,
                    "parent_route_count": len(supplemental),
                    "parent_route_added": len(combined) - v1_candidate_count,
                    "hard_conflict_count": len(conflicts),
                    "unknown_is_neutral": True,
                    "immediate_no_repeat": True,
                    "category_reset": category_changed,
                    "intent_version_reset": intent_version_changed,
                    "selective_removed_constraints": len(removed),
                    "selective_added_constraints": len(added),
                    "served_before_count": len(original.served),
                    "served_after_count": len(working.served),
                    "ask_other_forbidden": True,
                },
                tuple(float(item.score) for item in proposed_items),
            )
        except Exception:
            return self._fallback(fallback, "adapter_failure")


class FusionCoreAgent:
    """Evaluator-compatible wrapper which advances the frozen parent first.

    Off mode returns the parent response object directly.  Active mode uses the
    already-advanced parent state as the sole visible-state source and as the
    exact functional fallback.  It changes only recommendations, message, and
    ask_attribute; usage and every other parent response field are retained.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        mode: str | None = None,
        parent_agent: Any | None = None,
        fusion_core: FusionCore | None = None,
        force_v212_parent: bool = False,
        parent_options: Mapping[str, Any] | None = None,
    ) -> None:
        mode = str(mode or os.getenv("TECHJAM_FUSION_CORE_MODE", MODE_OFF)).strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ValueError("mode must be off or active")
        if parent_agent is None:
            from starter.agent import Agent as ParentAgent

            options = dict(parent_options or {})
            if force_v212_parent:
                options.update({
                    "p11_mode": "active",
                    "small_ranker_mode": "active",
                    "pagination_mode": "active",
                })
                options.setdefault(
                    "small_ranker_artifact_path",
                    Path(__file__).resolve().parent
                    / "assets"
                    / "small_ranker_fold_safe_v1.json",
                )
            parent_agent = ParentAgent(catalog_path, **options)
        self.mode = mode
        self.parent = parent_agent
        self.fusion = fusion_core or FusionCore(catalog_path, mode=mode)
        self._profiles: dict[str, dict[str, Any]] = {}
        self._intents: dict[str, str] = {}
        self._served_versions: dict[str, tuple[int, set[str]]] = {}
        self._all_served: dict[str, set[str]] = {}
        self._diagnostic_totals = {
            "turns": 0,
            "fusion_active_turns": 0,
            "fallback_turns": 0,
            "same_version_repeat_slots": 0,
            "hard_conflicts": 0,
            "parent_route_added": 0,
            "candidate_count": 0,
        }

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.parent.reset(session_id, user_profile)
        self.fusion.reset(session_id)
        self._profiles[session_id] = dict(user_profile)
        self._intents[session_id] = "buying"
        self._served_versions[session_id] = (1, set())
        self._all_served[session_id] = set()

    def close(self) -> None:
        try:
            self.fusion.close()
        finally:
            close = getattr(self.parent, "close", None)
            if callable(close):
                close()

    def _active_response(
        self,
        parent_response: Mapping[str, Any],
        outcome: FusionOutcome,
        *,
        turn: int,
        top_k: int,
    ) -> dict[str, Any]:
        response = dict(parent_response)
        if outcome.diagnostics.get("effective_mode") == MODE_ACTIVE:
            recommendations = []
            for index, identifier in enumerate(outcome.identifiers[:top_k]):
                score = outcome.scores[index] if index < len(outcome.scores) else 0.0
                recommendations.append({
                    "parent_asin": identifier,
                    "score": round(float(score), 6),
                })
            response["recommendations"] = recommendations
            ask_attribute = None if turn >= 10 else outcome.ask_attribute
            if ask_attribute == "other":
                ask_attribute = None
            response["ask_attribute"] = ask_attribute
            response["message"] = (
                QUESTION_TEMPLATES[ask_attribute]
                if ask_attribute is not None
                else "Here are the closest matches based on what you have told me."
            )
        elif response.get("ask_attribute") == "other":
            # Active Version A forbids the parent's generic other question even
            # when V1 itself falls back.  The recommendation fallback stays exact.
            response["ask_attribute"] = None
            response["message"] = (
                "Here are the closest matches based on what you have told me."
            )
        return response

    def _record_diagnostics(
        self,
        session_id: str,
        version: int,
        response: Mapping[str, Any],
        outcome: FusionOutcome,
    ) -> None:
        totals = self._diagnostic_totals
        totals["turns"] += 1
        effective = outcome.diagnostics.get("effective_mode")
        totals["fusion_active_turns"] += int(effective == MODE_ACTIVE)
        totals["fallback_turns"] += int(effective != MODE_ACTIVE)
        totals["hard_conflicts"] += int(outcome.diagnostics.get("hard_conflict_count", 0))
        totals["parent_route_added"] += int(outcome.diagnostics.get("parent_route_added", 0))
        totals["candidate_count"] += int(outcome.diagnostics.get("candidate_count", 0))
        identifiers = {
            str(item.get("parent_asin"))
            for item in response.get("recommendations", ())
            if isinstance(item, Mapping) and item.get("parent_asin")
        }
        previous_version, served = self._served_versions.get(
            session_id, (version, set())
        )
        if previous_version != version:
            served = set()
        totals["same_version_repeat_slots"] += len(identifiers & served)
        served.update(identifiers)
        self._served_versions[session_id] = (version, served)
        self._all_served.setdefault(session_id, set()).update(identifiers)

    def evaluation_diagnostics(self) -> dict[str, object]:
        turns = self._diagnostic_totals["turns"]
        sessions = len(self._all_served)
        return {
            "schema_version": "fusion-core-evaluation-diagnostics.v1",
            **self._diagnostic_totals,
            "sessions": sessions,
            "mean_distinct_served": round(
                sum(map(len, self._all_served.values())) / sessions, 6
            ) if sessions else 0.0,
            "mean_candidate_count": round(
                self._diagnostic_totals["candidate_count"] / turns, 6
            ) if turns else 0.0,
        }

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if turn == 1:
            self._intents[session_id] = (
                "browsing"
                if "but I'm still exploring." in user_message
                else "buying"
            )
        elif user_message.startswith("Actually, ignore my earlier preference."):
            self._intents[session_id] = "buying"
        parent_response = self.parent.respond(
            session_id, user_message, turn, top_k
        )
        if self.mode == MODE_OFF:
            return parent_response
        raw_snapshot = self.parent.debug_snapshot(session_id)
        snapshot = (
            dict(raw_snapshot)
            if isinstance(raw_snapshot, Mapping)
            else dict(vars(raw_snapshot))
        )
        snapshot["profile"] = dict(self._profiles.get(session_id, {}))
        snapshot["intent"] = self._intents.get(session_id, "buying")
        supplemental: Sequence[object] = ()
        debug_rankings = getattr(self.parent, "debug_rankings", None)
        if callable(debug_rankings):
            try:
                rankings = debug_rankings(session_id)
                if isinstance(rankings, Mapping):
                    supplemental = rankings.get("final", ())
            except Exception:
                supplemental = ()
        fallback = tuple(
            item["parent_asin"] for item in parent_response.get("recommendations", ())
        )
        outcome = self.fusion.apply(
            session_id,
            snapshot,
            turn=turn,
            top_k=top_k,
            fallback_order=fallback,
            supplemental_order=supplemental,
        )
        response = self._active_response(
            parent_response, outcome, turn=turn, top_k=top_k
        )
        self._record_diagnostics(
            session_id,
            int(snapshot.get("version", 1)),
            response,
            outcome,
        )
        return response


__all__ = [
    "FusionCore",
    "FusionCoreAgent",
    "FusionCoreContractError",
    "FusionOutcome",
    "MODE_ACTIVE",
    "MODE_OFF",
    "SCHEMA_VERSION",
    "VisibleConstraint",
    "VisibleProjection",
    "project_visible_state",
]
