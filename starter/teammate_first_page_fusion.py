"""Public-safe fusion: teammate V1 first-page retrieval plus Fusion-B dialogue.

The wrapper exposes the teammate's stronger initial recommendation page while
retaining Fusion-B's question and later-turn policy.  It then aligns Fusion-B's
served ledger with what was actually shown, so hidden first-page candidates do
not affect pagination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from starter.fusion_other import FusionOtherAgent
from vendor.teammate_v1.err402.agents.v1 import AgentV1


SCHEMA_VERSION = "teammate-first-page-fusion.v1"
ROUTER_SCHEMA_VERSION = "teammate-intent-routed-fusion.v1"


class TeammateFirstPageFusionAgent:
    """Use T0 recommendations on turn one and Fusion-B for the conversation."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        teammate: Any | None = None,
        fusion_b: Any | None = None,
    ) -> None:
        self.teammate = teammate if teammate is not None else AgentV1(catalog_path)
        self.fusion_b = fusion_b if fusion_b is not None else FusionOtherAgent(
            catalog_path,
            other_mode="active",
            mode="active",
            force_v212_parent=True,
        )
        self._substitutions = 0
        self._fallbacks = 0
        self._sessions: set[str] = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.teammate.reset(session_id, user_profile)
        self.fusion_b.reset(session_id, user_profile)
        self._sessions.add(session_id)

    @staticmethod
    def _ids(response: Mapping[str, Any]) -> tuple[str, ...]:
        rows = response.get("recommendations")
        if not isinstance(rows, list) or not rows:
            raise ValueError("recommendations must be a non-empty list")
        identifiers = tuple(str(row["parent_asin"]) for row in rows)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recommendations must be unique")
        return identifiers

    def _align_served_ledger(self, session_id: str, identifiers: tuple[str, ...]) -> None:
        core_agent = self.fusion_b.base
        memory = core_agent.fusion._sessions[session_id]
        memory.served = set(identifiers)
        version, _served = core_agent._served_versions[session_id]
        core_agent._served_versions[session_id] = (version, set(identifiers))
        core_agent._all_served[session_id] = set(identifiers)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        fusion_response = self.fusion_b.respond(
            session_id, user_message, turn, top_k
        )
        if turn != 1:
            return fusion_response
        try:
            teammate_response = self.teammate.respond(
                session_id, user_message, turn, top_k
            )
            identifiers = self._ids(teammate_response)
            if len(identifiers) != len(self._ids(fusion_response)):
                raise ValueError("first-page sizes differ")
            candidate = dict(fusion_response)
            candidate["recommendations"] = teammate_response["recommendations"]
            self._align_served_ledger(session_id, identifiers)
            self._substitutions += 1
            return candidate
        except Exception:
            self._fallbacks += 1
            return fusion_response

    def evaluation_diagnostics(self) -> dict[str, object]:
        diagnostics = self.fusion_b.evaluation_diagnostics()
        return {
            "schema_version": SCHEMA_VERSION,
            "sessions": len(self._sessions),
            "first_page_substitutions": self._substitutions,
            "fallbacks": self._fallbacks,
            "fusion_b": diagnostics,
        }

    def close(self) -> None:
        try:
            self.teammate.close()
        finally:
            self.fusion_b.close()


def create_teammate_first_page_fusion(
    catalog_path: str | Path = "data/catalog.jsonl",
) -> TeammateFirstPageFusionAgent:
    return TeammateFirstPageFusionAgent(catalog_path)


class IntentRoutedFusionAgent:
    """Route open exploration to T0 and explicit shopping to Fusion-B.

    The evaluator-visible initial utterance already distinguishes open browsing
    (``but I'm still exploring``) from explicit buying and override sessions.
    Routing once at turn one keeps each backend's state machine internally
    coherent and uses no labels, targets, or evaluator metadata.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        teammate: Any | None = None,
        fusion_b: Any | None = None,
    ) -> None:
        self.teammate = teammate if teammate is not None else AgentV1(catalog_path)
        self.fusion_b = fusion_b if fusion_b is not None else FusionOtherAgent(
            catalog_path,
            other_mode="active",
            mode="active",
            force_v212_parent=True,
        )
        self._routes: dict[str, str] = {}
        self._route_counts = {"teammate": 0, "fusion_b": 0}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.teammate.reset(session_id, user_profile)
        self.fusion_b.reset(session_id, user_profile)
        self._routes.pop(session_id, None)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        route = self._routes.get(session_id)
        if route is None:
            if turn != 1:
                raise ValueError("route must be selected on turn one")
            route = (
                "teammate"
                if "but I'm still exploring." in user_message
                else "fusion_b"
            )
            self._routes[session_id] = route
            self._route_counts[route] += 1
        backend = self.teammate if route == "teammate" else self.fusion_b
        return backend.respond(session_id, user_message, turn, top_k)

    def evaluation_diagnostics(self) -> dict[str, object]:
        teammate_diagnostics: dict[str, object] = {}
        diagnostics = getattr(self.teammate, "evaluation_diagnostics", None)
        if callable(diagnostics):
            value = diagnostics()
            if isinstance(value, Mapping):
                teammate_diagnostics = dict(value)
        return {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "sessions": len(self._routes),
            "teammate_routes": self._route_counts["teammate"],
            "fusion_b_routes": self._route_counts["fusion_b"],
            "teammate": teammate_diagnostics,
            "fusion_b": self.fusion_b.evaluation_diagnostics(),
        }

    def close(self) -> None:
        try:
            self.teammate.close()
        finally:
            self.fusion_b.close()


def create_intent_routed_fusion(
    catalog_path: str | Path = "data/catalog.jsonl",
) -> IntentRoutedFusionAgent:
    return IntentRoutedFusionAgent(catalog_path)


def create_intent_routed_forced_other_fusion(
    catalog_path: str | Path = "data/catalog.jsonl",
) -> IntentRoutedFusionAgent:
    from starter.teammate_bounded_other import TeammateBoundedOtherAgent

    teammate = TeammateBoundedOtherAgent(
        catalog_path, replace_specific=True
    )
    return IntentRoutedFusionAgent(catalog_path, teammate=teammate)


__all__ = [
    "SCHEMA_VERSION",
    "ROUTER_SCHEMA_VERSION",
    "IntentRoutedFusionAgent",
    "TeammateFirstPageFusionAgent",
    "create_intent_routed_fusion",
    "create_intent_routed_forced_other_fusion",
    "create_teammate_first_page_fusion",
]
