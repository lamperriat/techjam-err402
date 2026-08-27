from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent, _terms


def _product_view(product: dict[str, Any], target: str) -> dict[str, Any]:
    parent_asin = str(product.get("parent_asin", ""))
    return {
        "parent_asin": parent_asin,
        "title": str(product.get("title") or "Untitled product"),
        "categories": [str(value) for value in product.get("categories") or []],
        "price": product.get("price"),
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
        "store": str(product.get("store") or ""),
        "is_target": parent_asin == target,
    }


def _retrieval_snapshot(agent: Any, user_message: str, target: str) -> dict[str, Any]:
    terms = list(dict.fromkeys(_terms(user_message)))[:40]
    expression = " OR ".join(f'"{term}"' for term in terms)
    connection = getattr(agent, "connection", None)
    if not expression or connection is None:
        return {
            "terms": terms,
            "expression": expression,
            "candidate_count": 0 if not expression else None,
            "target_retrieval_rank": None,
        }
    rows = connection.execute(
        "SELECT parent_asin FROM products WHERE products MATCH ? "
        "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)",
        (expression,),
    ).fetchall()
    ranked_ids = [str(row[0]) for row in rows]
    target_rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
    return {
        "terms": terms,
        "expression": expression,
        "candidate_count": len(ranked_ids),
        "target_retrieval_rank": target_rank,
    }


class TraceRunner:
    def __init__(
        self,
        agent: Any,
        samples: list[dict[str, Any]],
        catalog_ids: set[str],
        categories: dict[str, list[str]],
        products: dict[str, dict[str, Any]],
        evaluation_result: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.samples = {str(sample["sample_id"]): sample for sample in samples}
        self.catalog_ids = catalog_ids
        self.categories = categories
        self.products = products
        result_sessions = (evaluation_result or {}).get("sessions") or []
        self.result_by_sample = {str(item["sample_id"]): item for item in result_sessions}
        self.metrics = {
            key: value
            for key, value in (evaluation_result or {}).items()
            if key not in {"sessions"}
        }
        self._cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_paths(
        cls,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        results_path: str | Path = "results.json",
    ) -> TraceRunner:
        samples = load_jsonl(dataset_path)
        catalog_ids, categories, products = catalog_index(catalog_path)
        result_file = Path(results_path)
        evaluation_result = (
            json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
        )
        return cls(
            Agent(catalog_path),
            samples,
            catalog_ids,
            categories,
            products,
            evaluation_result,
        )

    def list_sessions(self) -> dict[str, Any]:
        sessions: list[dict[str, Any]] = []
        for sample_id, sample in self.samples.items():
            target = str(sample["ground_truth"]["parent_asin"])
            product = self.products[target]
            prior_result = self.result_by_sample.get(sample_id, {})
            sessions.append({
                "sample_id": sample_id,
                "scenario_type": str(sample["scenario_type"]),
                "difficulty_bucket": sample.get("difficulty_bucket"),
                "target_title": str(product.get("title") or "Untitled product"),
                "hit": prior_result.get("hit"),
                "first_hit_turn": prior_result.get("first_hit_turn"),
                "best_rank": prior_result.get("best_rank"),
            })
        return {"metrics": self.metrics, "sessions": sessions}

    def trace(self, sample_id: str, refresh: bool = False) -> dict[str, Any]:
        if sample_id not in self.samples:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        if not refresh and sample_id in self._cache:
            return self._cache[sample_id]

        sample = self.samples[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        session_id = f"observer_{sample_id}"
        self.agent.reset(session_id, sample["user_profile"])

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(self.categories.get(target, [])),
            disclosed,
        )
        turns: list[dict[str, Any]] = []
        hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            disclosed_before = sorted(disclosed)
            retrieval = _retrieval_snapshot(self.agent, user_message, target)
            error: str | None = None
            try:
                response = self.agent.respond(session_id, user_message, turn, TOP_K)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                error = error or "Invalid response object or message"
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
            target_top10_rank = ranked.index(target) + 1 if target in ranked else None
            eligible_for_hit = override_applied
            hit = eligible_for_hit and target_top10_rank is not None
            if hit:
                hit_turn = turn
                best_rank = target_top10_rank

            next_user_message: str | None = None
            event = "hit" if hit else "continue"
            if not hit and turn < MAX_TURNS:
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    next_user_message = str(
                        override.get("message", "Actually, please ignore my earlier preference.")
                    )
                    event = "override_next"
                else:
                    next_user_message, boundary_used = customer_reply(
                        effective_sample,
                        response.get("ask_attribute"),
                        disclosed,
                        boundary_used,
                    )

            turns.append({
                "turn": turn,
                "user_message": user_message,
                "agent_message": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "usage": response.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0},
                "error": error,
                "retrieval": retrieval,
                "recommendations": [
                    _product_view(self.products[parent_asin], target) for parent_asin in ranked
                ],
                "raw_recommendation_count": (
                    len(response.get("recommendations"))
                    if isinstance(response.get("recommendations"), list)
                    else 0
                ),
                "valid_recommendation_count": len(ranked),
                "target_top10_rank": target_top10_rank,
                "eligible_for_hit": eligible_for_hit,
                "hit": hit,
                "event": event,
                "simulator_disclosed_before": disclosed_before,
                "simulator_disclosed_after": sorted(disclosed),
                "next_user_message": next_user_message,
            })
            if hit or turn == MAX_TURNS:
                break
            user_message = str(next_user_message)

        contribution = (
            0.0
            if hit_turn is None or best_rank is None
            else 0.5 + 0.3 / best_rank + 0.02 * (11 - hit_turn)
        )
        trace = {
            "sample_id": sample_id,
            "scenario_type": str(sample["scenario_type"]),
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "profile": sample["user_profile"],
            "target": _product_view(self.products[target], target),
            "intent_card": card,
            "behavior": behavior,
            "result": {
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else round(1.0 / best_rank, 6),
                "technical_contribution": round(contribution, 6),
            },
            "turns": turns,
            "observer_note": (
                "Ground truth and intent card are visible only in this public-set debug observer; "
                "they are never passed into Agent.respond."
            ),
        }
        self._cache[sample_id] = trace
        return trace
