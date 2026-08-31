"""Version B: evaluator-aligned ``other`` elicitation over frozen Fusion Core A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

from starter.fusion_core import FusionCoreAgent


SCHEMA_VERSION = "fusion-other.version-b.v1"
MODE_OFF = "off"
MODE_ACTIVE = "active"
OTHER_MESSAGE = "What other requirement matters most?"
INFORMATIVE_PREFIX = "For that, what matters is:"
EXHAUSTED_REPLY = "I don't have an additional preference for other."
BOUNDARY_REPLY = "I don't have a preference for other; please use your judgment."


@dataclass(frozen=True, slots=True)
class OtherLifecycle:
    version: int = 1
    other_asks: int = 0
    other_informative_replies: int = 0
    other_boundary_sentinel_seen: bool = False
    other_exhausted: bool = False
    pending_other: bool = False
    disclosed_constraints: int = 0


class FusionOtherAgent:
    """Change only A's question policy; ranking and exposure stay in ``base``."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        other_mode: str | None = None,
        base_agent: Any | None = None,
        mode: str | None = None,
        force_v212_parent: bool = False,
        parent_options: Mapping[str, Any] | None = None,
    ) -> None:
        resolved = str(
            other_mode or os.getenv("TECHJAM_FUSION_OTHER_MODE", MODE_OFF)
        ).strip().lower()
        if resolved not in {MODE_OFF, MODE_ACTIVE}:
            raise ValueError("other_mode must be off or active")
        self.other_mode = resolved
        self.base = base_agent or FusionCoreAgent(
            catalog_path,
            mode=mode,
            force_v212_parent=force_v212_parent,
            parent_options=parent_options,
        )
        self._lifecycles: dict[str, OtherLifecycle] = {}
        self._other_diagnostic_totals = {
            "other_activation_turns": 0,
            "other_informative_replies": 0,
            "other_boundary_sentinel_replies": 0,
            "other_exhausted_replies": 0,
            "other_disclosed_constraints": 0,
        }
        self._other_activation_sessions: set[str] = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.base.reset(session_id, user_profile)
        self._lifecycles[session_id] = OtherLifecycle()

    def close(self) -> None:
        self.base.close()

    def _commit_lifecycle(
        self,
        session_id: str,
        before: OtherLifecycle,
        after: OtherLifecycle,
    ) -> None:
        """Commit current-version state and append-only evaluation counters."""

        ask_delta = max(0, after.other_asks - before.other_asks)
        informative_delta = max(
            0,
            after.other_informative_replies - before.other_informative_replies,
        )
        disclosed_delta = max(
            0, after.disclosed_constraints - before.disclosed_constraints
        )
        boundary_delta = int(
            after.other_boundary_sentinel_seen
            and not before.other_boundary_sentinel_seen
        )
        exhausted_delta = int(
            after.other_exhausted and not before.other_exhausted
        )
        totals = self._other_diagnostic_totals
        totals["other_activation_turns"] += ask_delta
        totals["other_informative_replies"] += informative_delta
        totals["other_boundary_sentinel_replies"] += boundary_delta
        totals["other_exhausted_replies"] += exhausted_delta
        totals["other_disclosed_constraints"] += disclosed_delta
        if ask_delta:
            self._other_activation_sessions.add(session_id)
        self._lifecycles[session_id] = after

    def _intent_version(self, session_id: str, fallback: int) -> int:
        parent = getattr(self.base, "parent", None)
        snapshot = getattr(parent, "debug_snapshot", None)
        if not callable(snapshot):
            raise RuntimeError("A parent has no visible version snapshot")
        value = snapshot(session_id)
        version = value.get("version") if isinstance(value, Mapping) else None
        if type(version) is not int or version < 1:
            raise RuntimeError("A parent returned an invalid intent version")
        return version

    @staticmethod
    def _consume_reply(state: OtherLifecycle, message: str) -> OtherLifecycle:
        if not state.pending_other:
            return state
        cleaned = message.strip()
        if cleaned == BOUNDARY_REPLY:
            return replace(
                state,
                other_boundary_sentinel_seen=True,
                pending_other=False,
            )
        if cleaned == EXHAUSTED_REPLY:
            return replace(state, other_exhausted=True, pending_other=False)
        if cleaned.startswith(INFORMATIVE_PREFIX):
            values = cleaned[len(INFORMATIVE_PREFIX):].strip(" .")
            disclosed = len([value for value in values.split(";") if value.strip()])
            return replace(
                state,
                other_informative_replies=min(
                    2, state.other_informative_replies + 1
                ),
                disclosed_constraints=state.disclosed_constraints + disclosed,
                pending_other=False,
            )
        return replace(state, pending_other=False)

    @staticmethod
    def _can_ask_other(state: OtherLifecycle, turn: int) -> bool:
        maximum_asks = 3 if state.other_boundary_sentinel_seen else 2
        return bool(
            turn < 10
            and not state.other_exhausted
            and state.other_asks < maximum_asks
            and state.other_informative_replies < 2
        )

    def _cancel_unserved_a_question(self, session_id: str) -> None:
        """Prevent A's hidden proposal from entering visible question history."""

        parent = getattr(self.base, "parent", None)
        sessions = getattr(parent, "_sessions", None)
        if not isinstance(sessions, dict):
            return
        state = sessions.get(session_id)
        if state is None:
            return
        if hasattr(state, "pending_attribute"):
            state.pending_attribute = None
        if hasattr(state, "pending_turn"):
            state.pending_turn = None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = self.base.respond(session_id, user_message, turn, top_k)
        if self.other_mode == MODE_OFF:
            return response
        original = self._lifecycles.get(session_id, OtherLifecycle())
        try:
            version = self._intent_version(session_id, original.version)
            working = OtherLifecycle(version=version) if version != original.version else original
            version_start = working
            working = self._consume_reply(working, user_message)
            if turn >= 10:
                self._commit_lifecycle(session_id, version_start, working)
                if response.get("ask_attribute") is None:
                    return response
                candidate = dict(response)
                candidate["ask_attribute"] = None
                candidate["message"] = (
                    "Here are the closest matches based on what you have told me."
                )
                return candidate
            if not self._can_ask_other(working, turn):
                self._commit_lifecycle(session_id, version_start, working)
                return response
            working = replace(
                working,
                other_asks=working.other_asks + 1,
                pending_other=True,
            )
            candidate = dict(response)
            candidate["ask_attribute"] = "other"
            candidate["message"] = OTHER_MESSAGE
            self._cancel_unserved_a_question(session_id)
            self._commit_lifecycle(session_id, version_start, working)
            return candidate
        except Exception:
            return response

    def debug_other(self, session_id: str) -> dict[str, object]:
        state = self._lifecycles[session_id]
        return {"schema_version": SCHEMA_VERSION, **asdict(state)}

    def evaluation_diagnostics(self) -> dict[str, object]:
        """Return deterministic observation-only A and ``other`` aggregates."""

        base_diagnostics: dict[str, object] = {}
        diagnostics_fn = getattr(self.base, "evaluation_diagnostics", None)
        if callable(diagnostics_fn):
            try:
                value = diagnostics_fn()
                if isinstance(value, Mapping):
                    base_diagnostics = dict(value)
            except Exception:
                # Diagnostics must never change or invalidate a served response.
                base_diagnostics = {"available": False}
        return {
            "schema_version": "fusion-other-evaluation-diagnostics.v1",
            "fusion_core": base_diagnostics,
            "other_activation_sessions": len(self._other_activation_sessions),
            **self._other_diagnostic_totals,
        }


__all__ = [
    "BOUNDARY_REPLY",
    "EXHAUSTED_REPLY",
    "FusionOtherAgent",
    "INFORMATIVE_PREFIX",
    "MODE_ACTIVE",
    "MODE_OFF",
    "OTHER_MESSAGE",
    "OtherLifecycle",
    "SCHEMA_VERSION",
]
