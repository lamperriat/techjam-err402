from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from starter.agent import _parse_turn
from vendor.teammate_v1.err402.agents.v1 import AgentV1, OVERRIDE_RE
from vendor.teammate_v1.err402.retrieval.catalog import (
    CandidatePool,
    extract_department,
)
from vendor.teammate_v1.err402.retrieval.scoring import QueryContext
SCHEMA_VERSION = "teammate-safe-deltas.v1"
VARIANTS = ("t0", "s1", "s2", "s1+s2")
TailProvider = Callable[
    [str, Mapping[str, Any], Sequence[tuple[int, str]]], Sequence[str]
]
class _LazyP11C100:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.agent: Any | None = None
        self.synced: dict[str, int] = {}

    def _ensure_agent(self) -> Any:
        if self.agent is None:
            from starter.agent import Agent

            self.agent = Agent(
                self.catalog_path,
                question_policy="fast",
                rerank_mode="off",
                retrieval_mode="coverage",
                p11_mode="active",
                small_ranker_mode="off",
                pagination_mode="off",
            )
        return self.agent

    def reset(self, session_id: str, profile: Mapping[str, Any]) -> None:
        self.synced.pop(session_id, None)
        if self.agent is not None:
            self.agent.reset(session_id, dict(profile))
            self.synced[session_id] = 0

    def __call__(
        self,
        session_id: str,
        profile: Mapping[str, Any],
        history: Sequence[tuple[int, str]],
    ) -> Sequence[str]:
        agent = self._ensure_agent()
        if session_id not in self.synced:
            agent.reset(session_id, dict(profile))
            self.synced[session_id] = 0
        start = self.synced[session_id]
        if start > len(history):
            raise RuntimeError("P11 replay state exceeds visible history")
        for turn, message in history[start:]:
            agent.respond(session_id, message, turn, 10)
            self.synced[session_id] += 1
        rankings = agent.debug_rankings(session_id)
        final = rankings.get("final")
        if not isinstance(final, list):
            raise RuntimeError("P11 final ranking is unavailable")
        return tuple(str(identifier) for identifier in final[:100])

    def close(self) -> None:
        if self.agent is not None:
            self.agent.close()
            self.agent = None
class P11TailUnionAgent:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        base_agent: Any | None = None,
        tail_provider: TailProvider | None = None,
        state_guard: bool = False,
        tail_enabled: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.base = base_agent if base_agent is not None else AgentV1(self.catalog_path)
        self._tail_provider: TailProvider | None = tail_provider
        self._state_guard, self._tail_enabled = state_guard, tail_enabled
        self._profiles: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[tuple[int, str]]] = {}
        self._versions: dict[str, tuple[int, str]] = {}
        self._sessions_seen: set[str] = set()
        self._stats = {
            "turns": 0,
            "tail_attempts": 0,
            "activations": 0,
            "tail_added": 0,
            "fallbacks": 0,
            "s1_activations": 0,
        }

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.base.reset(session_id, user_profile)
        self._profiles[session_id] = dict(user_profile)
        self._history[session_id] = []
        self._versions[session_id] = (1, "")
        self._sessions_seen.add(session_id)
        reset = getattr(self._tail_provider, "reset", None)
        if callable(reset):
            reset(session_id, user_profile)
    def _provider(self) -> TailProvider:
        if self._tail_provider is None:
            self._tail_provider = _LazyP11C100(self.catalog_path)
        return self._tail_provider

    def _rank_newcomers(
        self,
        session_id: str,
        p11_c100: Sequence[str],
        limit: int,
    ) -> list[Any]:
        state = self.base._sessions[session_id]
        texts = tuple(constraint.text for constraint in state.constraints)
        query_text = " ".join((state.category, *texts))
        teammate_pool = self.base.catalog.candidates(state.category, query_text)
        teammate_ids = set(teammate_pool.parent_asins)
        products = self.base.catalog.products
        newcomers = tuple(
            identifier
            for identifier in dict.fromkeys(map(str, p11_c100))
            if identifier not in teammate_ids
            and identifier not in state.shown_product_ids
            and identifier in products
        )
        if not newcomers:
            return []
        context = QueryContext(
            intent=state.intent,
            category=state.category,
            constraints=texts,
            department=extract_department(query_text),
            budget=AgentV1._budget_constraint(state.constraints),
        )
        ranked = self.base.scorer.score(CandidatePool(newcomers, {}), context)
        return AgentV1._prioritize_exact_category(ranked, state.category)[:limit]
    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if self._state_guard and turn > 1:
            version, category = self._versions[session_id]
            state = self.base._sessions[session_id]
            category = category or state.category
            parsed = _parse_turn(user_message, current_category=category)
            changed = bool(parsed.category_text and parsed.category_text.lower() != category.lower())
            noncanonical = parsed.is_override and OVERRIDE_RE.match(user_message) is None
            if changed or noncanonical:
                state.shown_product_ids.clear()
                self._stats["s1_activations"] += 1
            if changed or parsed.is_override:
                version += 1
            self._versions[session_id] = (version, parsed.category_text or category)
        baseline = self.base.respond(session_id, user_message, turn, top_k)
        self._stats["turns"] += 1
        self._history.setdefault(session_id, []).append((turn, user_message))
        version, category = self._versions[session_id]
        self._versions[session_id] = (version, category or self.base._sessions[session_id].category)
        if not self._tail_enabled:
            return baseline
        recommendations = baseline.get("recommendations")
        if turn == 1 or not isinstance(recommendations, list) or len(recommendations) >= top_k:
            return baseline
        self._stats["tail_attempts"] += 1
        try:
            p11_c100 = self._provider()(
                session_id,
                self._profiles[session_id],
                tuple(self._history[session_id]),
            )
            additions = self._rank_newcomers(
                session_id, p11_c100, top_k - len(recommendations)
            )
            if not additions:
                return baseline
            extended = dict(baseline)
            extended["recommendations"] = [
                *recommendations,
                *(
                    {
                        "parent_asin": item.product.parent_asin,
                        "score": round(float(item.score), 6),
                    }
                    for item in additions
                ),
            ]
            self.base._sessions[session_id].shown_product_ids.update(
                item.product.parent_asin for item in additions
            )
            self._stats["activations"] += 1
            self._stats["tail_added"] += len(additions)
            return extended
        except Exception:
            self._stats["fallbacks"] += 1
            return baseline
    def evaluation_diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            **self._stats,
            "sessions": len(self._sessions_seen),
        }

    def close(self) -> None:
        try:
            close = getattr(self._tail_provider, "close", None)
            if callable(close):
                close()
        finally:
            self.base.close()
def create_teammate_safe_agent(
    catalog_path: str | Path = "data/catalog.jsonl",
    *,
    variant: str = "t0",
    tail_provider: TailProvider | None = None,
) -> Any:
    normalized = str(variant).strip().lower()
    if normalized not in VARIANTS:
        raise ValueError("variant must be t0, s1, s2, or s1+s2")
    if normalized == "t0":
        return AgentV1(catalog_path)
    return P11TailUnionAgent(
        catalog_path,
        tail_provider=tail_provider,
        state_guard=normalized in {"s1", "s1+s2"},
        tail_enabled=normalized in {"s2", "s1+s2"},
    )


__all__ = [
    "P11TailUnionAgent",
    "SCHEMA_VERSION",
    "VARIANTS",
    "create_teammate_safe_agent",
]
