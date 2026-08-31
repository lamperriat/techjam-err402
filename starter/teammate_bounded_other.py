"""Bounded evaluator-aligned ``other`` questions over an unchanged base agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import Counter
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from vendor.teammate_v1.err402.agents.v1 import AgentV1


SCHEMA_VERSION = "teammate-bounded-other.v1"
DIAGNOSTIC_SCHEMA_VERSION = "teammate-bounded-other-evaluation-diagnostics.v1"
OTHER_MESSAGE = "What other requirement matters most?"
INFORMATIVE_RE = re.compile(r"^\s*For that, what matters is:\s*(.*?)\s*\.?\s*$", re.I | re.S)
EXHAUSTED_RE = re.compile(
    r"^\s*I don'?t have (?:an|any) additional preference(?:s| for other)?\.?\s*$", re.I
)
BOUNDARY_RE = re.compile(
    r"^\s*I don'?t have (?:a|any) preference for other;\s*please use your judgment\.?\s*$",
    re.I,
)
OVERRIDE_PATTERNS = (
    re.compile(r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:earlier|previous|prior|before)\b", re.I),
    re.compile(r"\bchange(?:d)?\s+my\s+mind\b", re.I),
    re.compile(r"\bno\s+longer\s+(?:want|need|prefer)\b", re.I),
    re.compile(r"\b(?:switch|change|replace)\b.{0,40}\bfrom\b.{0,40}\bto\b", re.I),
    re.compile(r"\breplace\b.{0,40}\bwith\b", re.I),
    re.compile(r"^\s*instead\b", re.I),
    re.compile(
        r"\b(?:want|need|prefer|choose|change)\b.{0,40}\binstead\b|"
        r"\binstead\b.{0,40}\b(?:want|need|prefer|choose|change)\b",
        re.I,
    ),
)


@dataclass(frozen=True, slots=True)
class OtherLifecycle:
    version: int = 1
    asks: int = 0
    informative_replies: int = 0
    pending: bool = False
    exhausted: bool = False
    boundary_seen: bool = False
    disclosed_constraints: int = 0


def is_intent_override(message: str) -> bool:
    """Recognize the official override and equivalent visible phrasings."""

    return any(pattern.search(message) for pattern in OVERRIDE_PATTERNS)


class TeammateBoundedOtherAgent:
    """Ask at most two ``other`` questions per visible intent version."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        base_agent: Any | None = None,
        base_factory: Callable[..., Any] = AgentV1,
        base_options: Mapping[str, Any] | None = None,
        enabled: bool = True,
        replace_specific: bool = False,
    ) -> None:
        self.base = (
            base_agent
            if base_agent is not None
            else base_factory(catalog_path, **dict(base_options or {}))
        )
        self.enabled = bool(enabled)
        self.replace_specific = bool(replace_specific)
        self._states: dict[str, OtherLifecycle] = {}
        self._activation_sessions: set[str] = set()
        self._totals = {
            "other_activation_turns": 0,
            "other_informative_replies": 0,
            "other_boundary_sentinel_replies": 0,
            "other_exhausted_replies": 0,
            "other_disclosed_constraints": 0,
            "other_override_resets": 0,
            "other_replaced_specific_questions": 0,
        }

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.base.reset(session_id, user_profile)
        self._states[session_id] = OtherLifecycle()

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()

    def _cancel_base_question(self, session_id: str, attribute: object) -> None:
        if not isinstance(attribute, str) or not attribute:
            return
        cancel = getattr(self.base, "cancel_last_question", None)
        if callable(cancel) and cancel(session_id, attribute):
            return
        sessions = getattr(self.base, "_sessions", None)
        state = sessions.get(session_id) if isinstance(sessions, dict) else None
        if state is None or getattr(state, "last_asked_attribute", None) != attribute:
            return
        asked = getattr(state, "asked_attributes", None)
        if isinstance(asked, set):
            asked.discard(attribute)
        follow_up = getattr(state, "follow_up_attributes", None)
        if isinstance(follow_up, set):
            follow_up.discard(attribute)
        counts = getattr(state, "question_counts", None)
        if isinstance(counts, Counter):
            counts[attribute] -= 1
            if counts[attribute] <= 0:
                del counts[attribute]
        state.last_asked_attribute = None

    @staticmethod
    def _consume(state: OtherLifecycle, message: str) -> tuple[OtherLifecycle, str | None, int]:
        if not state.pending:
            return state, None, 0
        if BOUNDARY_RE.match(message):
            return replace(state, pending=False, boundary_seen=True), "boundary", 0
        if EXHAUSTED_RE.match(message):
            return replace(state, pending=False, exhausted=True), "exhausted", 0
        match = INFORMATIVE_RE.match(message)
        if match:
            count = len([value for value in match.group(1).split(";") if value.strip(" .")])
            return replace(
                state,
                pending=False,
                informative_replies=state.informative_replies + 1,
                disclosed_constraints=state.disclosed_constraints + count,
            ), "informative", count
        return replace(state, pending=False), None, 0

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.base.respond(session_id, user_message, turn, top_k)
        if not self.enabled:
            return response
        original = self._states.get(session_id, OtherLifecycle())
        try:
            override = is_intent_override(user_message)
            start = OtherLifecycle(version=original.version + 1) if override else original
            working, reply_kind, disclosed = self._consume(start, user_message)
            base_attribute = response.get("ask_attribute")
            should_ask = (
                turn < 10
                and (base_attribute is None or self.replace_specific)
                and not working.exhausted
                and working.asks < (
                    3 if self.replace_specific and working.boundary_seen else 2
                )
            )
            candidate = response
            if should_ask:
                if base_attribute is not None:
                    self._cancel_base_question(session_id, base_attribute)
                    self._totals["other_replaced_specific_questions"] += 1
                candidate = dict(response)
                candidate["ask_attribute"] = "other"
                candidate["message"] = OTHER_MESSAGE
                working = replace(working, asks=working.asks + 1, pending=True)

            if override:
                self._totals["other_override_resets"] += 1
            if reply_kind:
                self._totals[f"other_{reply_kind}_replies" if reply_kind != "boundary" else "other_boundary_sentinel_replies"] += 1
            self._totals["other_disclosed_constraints"] += disclosed
            if should_ask:
                self._totals["other_activation_turns"] += 1
                self._activation_sessions.add(session_id)
            self._states[session_id] = working
            return candidate
        except Exception:
            return response

    def debug_other(self, session_id: str) -> dict[str, object]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self._states[session_id])}

    def evaluation_diagnostics(self) -> dict[str, object]:
        base_diagnostics: dict[str, object] = {}
        diagnostics = getattr(self.base, "evaluation_diagnostics", None)
        if callable(diagnostics):
            try:
                value = diagnostics()
                if isinstance(value, Mapping):
                    base_diagnostics = dict(value)
            except Exception:
                base_diagnostics = {"available": False}
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "fusion_core": base_diagnostics,
            "other_activation_sessions": len(self._activation_sessions),
            **self._totals,
        }


def create_teammate_bounded_other_agent(
    catalog_path: str | Path = "data/catalog.jsonl", **options: Any
) -> TeammateBoundedOtherAgent:
    """Importable evaluator factory using the untouched vendored AgentV1."""

    return TeammateBoundedOtherAgent(catalog_path, **options)


def create_teammate_forced_other_agent(
    catalog_path: str | Path = "data/catalog.jsonl", **options: Any
) -> TeammateBoundedOtherAgent:
    """Replace V1's hidden specific question with bounded evaluator ``other``."""

    return TeammateBoundedOtherAgent(
        catalog_path, replace_specific=True, **options
    )


__all__ = [
    "OTHER_MESSAGE", "OtherLifecycle", "SCHEMA_VERSION",
    "TeammateBoundedOtherAgent", "create_teammate_bounded_other_agent",
    "create_teammate_forced_other_agent",
    "is_intent_override",
]
