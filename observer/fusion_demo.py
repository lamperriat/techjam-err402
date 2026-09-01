"""Target-blind, local T0/A/B demonstration runtime for the Workbench."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping


VARIANTS = ("t0", "a", "b")
DEFAULT_PROFILE = {
    "summary": "Local demonstration profile",
    "preference_tags": ["comfort", "practical"],
}


def _default_factories() -> dict[str, Callable[[Path], Any]]:
    from starter.teammate_v212_fusion import create_fusion_a, create_fusion_b
    from vendor.teammate_v1.err402.agents.v1 import AgentV1

    return {"t0": AgentV1, "a": create_fusion_a, "b": create_fusion_b}


def _close(agent: Any | None) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()


class FusionDemo:
    """Run one real teammate/fusion variant at a time for a manual demo.

    The manager never receives evaluator labels, target identifiers, sample IDs, or
    scenario state. Switching variants closes the previous catalog index before the
    next one is created, which keeps the local demonstration resource-bounded.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        products: Mapping[str, Mapping[str, Any]],
        *,
        factories: Mapping[str, Callable[[Path], Any]] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.products = products
        self.factories = dict(factories or _default_factories())
        self._agent: Any | None = None
        self._variant: str | None = None
        self._session_id: str | None = None
        self._turn = 0
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.catalog_path.is_file()

    def unavailable(self) -> dict[str, Any]:
        return {
            "available": False,
            "reason": "The frozen catalog is required for a live T0/A/B run.",
            "catalog_path": str(self.catalog_path),
            "setup": "Download the official catalog.jsonl.gz release and extract it to data/catalog.jsonl.",
        }

    def reset(
        self, variant: str, profile: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown fusion demo variant: {variant}")
        if not self.available:
            return self.unavailable()
        with self._lock:
            reused_index = self._agent is not None and self._variant == variant
            if not reused_index:
                previous = self._agent
                self._agent = None
                self._variant = None
                self._session_id = None
                _close(previous)
                candidate = self.factories[variant](self.catalog_path)
                self._agent = candidate
                self._variant = variant
            self._session_id = f"fusion_demo_{uuid.uuid4().hex}"
            self._turn = 0
            self._agent.reset(
                self._session_id, dict(profile or DEFAULT_PROFILE)
            )
            return {
                "available": True,
                "variant": variant,
                "session_id": self._session_id,
                "turn": 0,
                "reused_index": reused_index,
                "catalog_products": len(self.products),
            }

    def respond(self, session_id: str, message: str) -> dict[str, Any]:
        text = str(message).strip()
        if not text:
            raise ValueError("message is required")
        with self._lock:
            if not self.available:
                return self.unavailable()
            if self._agent is None or session_id != self._session_id:
                raise KeyError("unknown or expired fusion demo session")
            if self._turn >= 10:
                raise ValueError("the live demo is limited to 10 turns; start a new session")
            self._turn += 1
            _, fusion, _ = self._components()
            diagnostics_before = self._diagnostics(fusion)
            response = self._agent.respond(session_id, text, self._turn, 10)
            diagnostics_after = self._diagnostics(fusion)
            return self._payload(text, response, diagnostics_before, diagnostics_after)

    def _components(self) -> tuple[Any, Any | None, Any | None]:
        if self._variant == "b":
            fusion = self._agent.base
            return fusion.teammate, fusion, self._agent
        if self._variant == "a":
            return self._agent.teammate, self._agent, None
        return self._agent, None, None

    @staticmethod
    def _diagnostics(fusion: Any | None) -> dict[str, Any]:
        getter = getattr(fusion, "evaluation_diagnostics", None)
        if not callable(getter):
            return {}
        try:
            value = getter()
        except Exception:
            return {}
        return dict(value) if isinstance(value, Mapping) else {}

    def _payload(
        self,
        message: str,
        response: Mapping[str, Any],
        diagnostics_before: Mapping[str, Any],
        diagnostics_after: Mapping[str, Any],
    ) -> dict[str, Any]:
        teammate, fusion, other_adapter = self._components()
        teammate_state = teammate._sessions[self._session_id]
        constraint_texts = tuple(
            constraint.text for constraint in teammate_state.constraints
        )
        query_text = " ".join((teammate_state.category, *constraint_texts))
        pool = teammate.catalog.candidates(teammate_state.category, query_text)
        lexical_count = len(pool.lexical_ranks)
        category_only = max(0, len(pool.parent_asins) - lexical_count)
        page = fusion._pages.get(self._session_id, self._turn) if fusion else self._turn
        served = (
            len(fusion._served.get(self._session_id, set()))
            if fusion
            else len(teammate_state.shown_product_ids)
        )
        expert_top: list[str] = []
        if fusion is not None:
            try:
                expert_top = list(fusion._rank_order(self._session_id)[:10])
            except Exception:
                expert_top = []
        other_state: dict[str, Any] | None = None
        if other_adapter is not None:
            try:
                other_state = other_adapter.debug_other(self._session_id)
            except Exception:
                other_state = None

        rows = response.get("recommendations")
        if not isinstance(rows, list):
            raise TypeError("demo Agent recommendations must be a list")
        identifiers: list[str] = []
        seen: set[str] = set()
        for item in rows:
            if not isinstance(item, Mapping) or not item.get("parent_asin"):
                continue
            identifier = str(item["parent_asin"])
            if identifier in seen or identifier not in self.products:
                continue
            seen.add(identifier)
            identifiers.append(identifier)
            if len(identifiers) == 10:
                break
        contract_drops = len(rows) - len(identifiers)
        recommendations = []
        for rank, identifier in enumerate(identifiers, start=1):
            product = self.products.get(identifier, {})
            recommendations.append({
                "rank": rank,
                "parent_asin": identifier,
                "title": str(product.get("title") or "Untitled product"),
                "price": product.get("price"),
                "categories": product.get("categories") or [],
            })

        if fusion is None:
            route = "T0 ProductScorer unseen page"
        elif page <= 2:
            route = f"exact T0 grace page {page}/2"
        elif int(diagnostics_after.get("rank_fallbacks", 0)) > int(
            diagnostics_before.get("rank_fallbacks", 0)
        ):
            route = "T0 fallback after rank-expert exception"
        else:
            route = "v2.12 unseen expert tail with T0 fill"
        question = response.get("ask_attribute")
        events = [
            {"layer": "Visible input", "kind": "actual", "status": "done", "value": f"turn {self._turn}", "detail": message},
            {"layer": "Intent + state", "kind": "observer-derived", "status": "done", "value": teammate_state.intent, "detail": f"category={teammate_state.category or 'unknown'} · constraints={len(constraint_texts)}"},
            {"layer": "FTS1000 + category", "kind": "observer-derived", "status": "done", "value": f"{len(pool.parent_asins)} candidates", "detail": f"deterministic replay: lexical={lexical_count} · category-only={category_only}"},
            {"layer": "ProductScorer", "kind": "observer-derived", "status": "done", "value": "multi-signal rank", "detail": "lexical · category · constraint · department · budget · rating · popularity"},
            {"layer": "v2.12 rank expert", "kind": "observer-derived", "status": "done" if fusion else "bypassed", "value": f"{len(expert_top)} inspected" if fusion else "T0 only", "detail": "R08/P11 + fold-safe small-ranker shadow" if fusion else "No shadow expert in teammate T0"},
            {"layer": "Page router", "kind": "observer-derived", "status": "done", "value": route, "detail": f"intent page={page} · served ledger={served}"},
            {"layer": "Question policy", "kind": "actual", "status": "done", "value": question or "none", "detail": "bounded other lifecycle" if other_adapter else "specific attribute policy; other disabled"},
            {"layer": "Catalog-valid Top10", "kind": "actual", "status": "done", "value": f"{len(recommendations)} products", "detail": f"official-style unique valid IDs · dropped={contract_drops}"},
        ]
        constraints = [
            {"text": item.text, "hard": item.hard, "source": item.source}
            for item in teammate_state.constraints
        ]
        return {
            "available": True,
            "variant": self._variant,
            "session_id": self._session_id,
            "turn": self._turn,
            "response": {
                "message": str(response.get("message") or ""),
                "ask_attribute": question,
            },
            "recommendations": recommendations,
            "events": events,
            "state": {
                "intent": teammate_state.intent,
                "category": teammate_state.category,
                "constraints": constraints,
                "asked_attributes": sorted(teammate_state.asked_attributes),
                "shown_by_t0": len(teammate_state.shown_product_ids),
                "fusion_page": page if fusion else None,
                "fusion_served": served if fusion else None,
                "other": other_state,
            },
            "expert_top": expert_top,
            "contract": {
                "raw_recommendations": len(rows),
                "valid_unique_top10": len(recommendations),
                "dropped_or_beyond_top10": contract_drops,
            },
        }

    def close(self) -> None:
        with self._lock:
            _close(self._agent)
            self._agent = None
            self._variant = None
            self._session_id = None
            self._turn = 0


__all__ = ["DEFAULT_PROFILE", "FusionDemo", "VARIANTS"]
