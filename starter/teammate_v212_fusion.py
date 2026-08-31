"""Strict deadline A/B fusion of teammate V1 and the frozen v2.12 rank stack."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from starter.agent import Agent as V212Agent
from starter.teammate_bounded_other import is_intent_override
from vendor.teammate_v1.err402.agents.v1 import AgentV1


SCHEMA_VERSION = "teammate-v212-tail-fusion-a.v1"
FOLD_SAFE_ARTIFACT = (
    Path(__file__).resolve().parent / "assets" / "small_ranker_fold_safe_v1.json"
)


class TeammateV212FusionA:
    """T0 exploitation for two pages, then v2.12 unseen exploration.

    Questions always come from teammate V1, whose supported attribute set does
    not include ``other``.  The v2.12 expert runs as a shadow rank provider: P11
    and the frozen fold-safe small ranker remain active, while this wrapper owns
    the actual two-page grace and served ledger.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        teammate: Any | None = None,
        rank_expert: Any | None = None,
    ) -> None:
        self.teammate = teammate if teammate is not None else AgentV1(catalog_path)
        self.rank_expert = rank_expert if rank_expert is not None else V212Agent(
            catalog_path,
            p11_mode="active",
            small_ranker_mode="active",
            small_ranker_artifact_path=FOLD_SAFE_ARTIFACT,
            pagination_mode="off",
        )
        self._pages: dict[str, int] = {}
        self._served: dict[str, set[str]] = {}
        self._sessions: set[str] = set()
        self._stats = Counter(
            turns=0,
            grace_turns=0,
            tail_turns=0,
            tail_changes=0,
            rank_fallbacks=0,
            intent_resets=0,
            forbidden_other=0,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.teammate.reset(session_id, user_profile)
        self.rank_expert.reset(session_id, user_profile)
        self._pages[session_id] = 0
        self._served[session_id] = set()
        self._sessions.add(session_id)

    @staticmethod
    def _ids(response: Mapping[str, Any]) -> tuple[str, ...]:
        rows = response.get("recommendations")
        if not isinstance(rows, list):
            raise TypeError("recommendations must be a list")
        identifiers = tuple(str(row["parent_asin"]) for row in rows)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recommendations must be unique")
        return identifiers

    def _cancel_shadow_question(self, session_id: str) -> None:
        sessions = getattr(self.rank_expert, "_sessions", None)
        state = sessions.get(session_id) if isinstance(sessions, dict) else None
        if state is not None:
            if hasattr(state, "pending_attribute"):
                state.pending_attribute = None
            if hasattr(state, "pending_turn"):
                state.pending_turn = None

    def cancel_last_question(self, session_id: str, attribute: str) -> bool:
        """Cancel A's unserved T0 question when Version B substitutes ``other``."""

        sessions = getattr(self.teammate, "_sessions", None)
        state = sessions.get(session_id) if isinstance(sessions, dict) else None
        if state is None or getattr(state, "last_asked_attribute", None) != attribute:
            return False
        state.asked_attributes.discard(attribute)
        state.follow_up_attributes.discard(attribute)
        state.question_counts[attribute] -= 1
        if state.question_counts[attribute] <= 0:
            del state.question_counts[attribute]
        state.last_asked_attribute = None
        return True

    def _rank_order(self, session_id: str) -> tuple[str, ...]:
        debug = getattr(self.rank_expert, "debug_rankings", None)
        if not callable(debug):
            raise RuntimeError("rank expert has no debug_rankings")
        rankings = debug(session_id)
        if not isinstance(rankings, Mapping):
            raise TypeError("rank expert returned malformed rankings")
        values = rankings.get("final")
        if not isinstance(values, (list, tuple)):
            raise TypeError("rank expert final order is malformed")
        order = tuple(map(str, values))
        if not order or len(order) != len(set(order)):
            raise ValueError("rank expert final order is invalid")
        return order

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        if turn > 1 and is_intent_override(user_message):
            self._pages[session_id] = 0
            self._served[session_id].clear()
            self._stats["intent_resets"] += 1

        teammate_response = self.teammate.respond(
            session_id, user_message, turn, top_k
        )
        self.rank_expert.respond(session_id, user_message, turn, top_k)
        self._cancel_shadow_question(session_id)
        self._stats["turns"] += 1
        if teammate_response.get("ask_attribute") == "other":
            self._stats["forbidden_other"] += 1
            raise RuntimeError("Version A forbids ask_attribute=other")

        self._pages[session_id] += 1
        page = self._pages[session_id]
        baseline_ids = self._ids(teammate_response)
        if page <= 2:
            self._served[session_id].update(baseline_ids)
            self._stats["grace_turns"] += 1
            return teammate_response

        try:
            served = self._served[session_id]
            selected = [value for value in self._rank_order(session_id) if value not in served]
            selected = selected[:top_k]
            if len(selected) < top_k:
                selected.extend(
                    value for value in baseline_ids
                    if value not in served and value not in selected
                )
                selected = selected[:top_k]
            if not selected:
                raise RuntimeError("empty unseen exploration page")
            candidate = dict(teammate_response)
            candidate["recommendations"] = [
                {"parent_asin": identifier} for identifier in selected
            ]
            served.update(selected)
            self._stats["tail_turns"] += 1
            self._stats["tail_changes"] += int(tuple(selected) != baseline_ids)
            return candidate
        except Exception:
            self._stats["rank_fallbacks"] += 1
            self._served[session_id].update(baseline_ids)
            return teammate_response

    def evaluation_diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            **dict(self._stats),
            "sessions": len(self._sessions),
        }

    def close(self) -> None:
        try:
            self.teammate.close()
        finally:
            self.rank_expert.close()


def create_fusion_a(
    catalog_path: str | Path = "data/catalog.jsonl",
) -> TeammateV212FusionA:
    return TeammateV212FusionA(catalog_path)


__all__ = [
    "FOLD_SAFE_ARTIFACT",
    "SCHEMA_VERSION",
    "TeammateV212FusionA",
    "create_fusion_a",
]
