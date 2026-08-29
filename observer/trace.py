from __future__ import annotations

import inspect
import json
import threading
import time
import uuid
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
from observer.events import TraceRecorder
from starter.agent import P11_MODES, RETRIEVAL_MODES, Agent, _terms


RERANK_MODES = {"off", "shadow", "active"}


def _normalize_rerank_mode(value: object) -> str:
    mode = str(value or "off").strip().lower()
    if mode not in RERANK_MODES:
        raise ValueError("rerank_mode must be one of: off, shadow, active")
    return mode


def _agent_rerank_mode(agent: Any) -> str:
    value = getattr(agent, "rerank_mode", "off")
    try:
        return _normalize_rerank_mode(value)
    except ValueError:
        return "off"


def _normalize_retrieval_mode(value: object) -> str:
    mode = str(value or "control").strip().lower()
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            "retrieval_mode must be one of: " + ", ".join(RETRIEVAL_MODES)
        )
    return mode


def _agent_retrieval_mode(agent: Any) -> str:
    value = getattr(agent, "retrieval_mode", "control")
    try:
        return _normalize_retrieval_mode(value)
    except ValueError:
        return "control"


def _agent_p11_mode(agent: Any) -> str:
    mode = str(getattr(agent, "p11_mode", "off")).strip().lower()
    return mode if mode in P11_MODES else "off"


def _agent_p11_status(agent: Any) -> dict[str, Any]:
    status = getattr(agent, "_p11_status", None)
    if callable(status):
        try:
            value = status()
            if isinstance(value, dict):
                return dict(value)
        except Exception:
            pass
    mode = _agent_p11_mode(agent)
    return {
        "configured_mode": mode,
        "effective_mode": mode,
        "fallback": False,
        "identity_verified": False,
        "reason_code": "status_unavailable",
    }


def _create_agent(
    catalog_path: str | Path,
    *,
    trace_sink: Any | None = None,
    rerank_mode: str | None = None,
    retrieval_mode: str | None = None,
    p11_mode: str | None = None,
) -> Agent:
    """Construct current or pre-reranker Agents without misreporting their mode."""
    parameters = inspect.signature(Agent).parameters
    kwargs: dict[str, Any] = {}
    if "trace_sink" in parameters:
        kwargs["trace_sink"] = trace_sink
    if "rerank_mode" in parameters:
        kwargs["rerank_mode"] = rerank_mode
    if "retrieval_mode" in parameters:
        kwargs["retrieval_mode"] = retrieval_mode
    if "p11_mode" in parameters and p11_mode is not None:
        kwargs["p11_mode"] = p11_mode
    return Agent(catalog_path, **kwargs)


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


def _retrieval_snapshot(user_message: str) -> dict[str, Any]:
    terms = list(dict.fromkeys(_terms(user_message)))[:40]
    expression = " OR ".join(f'"{term}"' for term in terms)
    return {
        "terms": terms,
        "expression": expression,
        "candidate_count": 0 if not expression else None,
        "diagnostic_adapter": "Agent events and target-blind debug routes",
    }


def _rank_of(target: str, identifiers: object) -> int | None:
    if not isinstance(identifiers, list):
        return None
    try:
        return identifiers.index(target) + 1
    except ValueError:
        return None


def _agent_diagnostics(agent: Any, session_id: str, target: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    rankings: dict[str, list[str]] = {}
    rerank_diagnostics: dict[str, Any] = {}
    error: str | None = None
    try:
        debug_snapshot = getattr(agent, "debug_snapshot", None)
        if callable(debug_snapshot):
            value = debug_snapshot(session_id)
            if isinstance(value, dict):
                snapshot = value
        debug_rankings = getattr(agent, "debug_rankings", None)
        if callable(debug_rankings):
            value = debug_rankings(session_id)
            if isinstance(value, dict):
                rankings = {
                    route: [str(identifier) for identifier in identifiers]
                    for route, identifiers in value.items()
                    if isinstance(route, str) and isinstance(identifiers, list)
                }
        debug_rerank = getattr(agent, "debug_rerank_diagnostics", None)
        if callable(debug_rerank):
            value = debug_rerank(session_id)
            if isinstance(value, dict):
                rerank_diagnostics = value
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    broad = rankings.get("broad", [])
    strict = rankings.get("strict", [])
    fused = rankings.get("fused", [])
    reranked = rankings["reranked"] if "reranked" in rankings else fused
    final = rankings["final"] if "final" in rankings else fused
    return {
        "state": snapshot,
        "rerank_mode": _agent_rerank_mode(agent),
        "retrieval_mode": _agent_retrieval_mode(agent),
        "p11": rerank_diagnostics.get("p11", _agent_p11_status(agent)),
        "route_counts": {
            "broad": len(broad),
            "strict": len(strict),
            "fused": len(fused),
            "reranked": len(reranked),
            "final": len(final),
        },
        "target_broad_rank": _rank_of(target, broad),
        "target_strict_rank": _rank_of(target, strict),
        "target_fused_rank": _rank_of(target, fused),
        "target_reranked_rank": _rank_of(target, reranked),
        "target_final_rank": _rank_of(target, final),
        "target_rerank_breakdown": (
            (rerank_diagnostics.get("breakdowns") or {}).get(target)
            if isinstance(rerank_diagnostics.get("breakdowns"), dict)
            else None
        ),
        "rerank_diagnostics": {
            key: value
            for key, value in rerank_diagnostics.items()
            if key != "breakdowns"
        },
        "coverage_diagnostics": (
            rerank_diagnostics.get("coverage")
            if isinstance(rerank_diagnostics.get("coverage"), dict)
            else {"active": False}
        ),
        "actual_route": "final" if "final" in rankings else "fused",
        "error": error,
    }


def _output_diagnostics(payload: object, catalog_ids: set[str]) -> dict[str, Any]:
    if not isinstance(payload, list):
        return {
            "payload_is_list": False,
            "raw_count": 0,
            "malformed_count": 1,
            "invalid_catalog_count": 0,
            "duplicate_count": 0,
            "valid_unique_count": 0,
        }
    seen: set[str] = set()
    malformed = 0
    invalid = 0
    duplicates = 0
    valid = 0
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("parent_asin"), str):
            malformed += 1
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin not in catalog_ids:
            invalid += 1
            continue
        if parent_asin in seen:
            duplicates += 1
            continue
        seen.add(parent_asin)
        valid += 1
    return {
        "payload_is_list": True,
        "raw_count": len(payload),
        "malformed_count": malformed,
        "invalid_catalog_count": invalid,
        "duplicate_count": duplicates,
        "valid_unique_count": valid,
        "scored_count": min(valid, TOP_K),
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
        recorder: TraceRecorder | None = None,
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
        self.recorder = recorder
        self._cache: dict[str, dict[str, Any]] = {}
        self._trace_lock = threading.Lock()

    @classmethod
    def from_paths(
        cls,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        results_path: str | Path = "results.json",
        rerank_mode: str | None = None,
        retrieval_mode: str | None = None,
        p11_mode: str | None = None,
    ) -> TraceRunner:
        samples = load_jsonl(dataset_path)
        catalog_ids, categories, products = catalog_index(catalog_path)
        result_file = Path(results_path)
        evaluation_result = (
            json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
        )
        recorder = TraceRecorder()
        return cls(
            _create_agent(
                catalog_path,
                trace_sink=recorder.emit,
                rerank_mode=rerank_mode,
                retrieval_mode=retrieval_mode,
                p11_mode=p11_mode,
            ),
            samples,
            catalog_ids,
            categories,
            products,
            evaluation_result,
            recorder,
        )

    def replace_evaluation_result(self, evaluation_result: dict[str, Any]) -> None:
        result_sessions = evaluation_result.get("sessions") or []
        self.result_by_sample = {str(item["sample_id"]): item for item in result_sessions}
        self.metrics = {
            key: value for key, value in evaluation_result.items() if key != "sessions"
        }

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

        with self._trace_lock:
            return self._run_trace(sample_id)

    def _run_trace(self, sample_id: str) -> dict[str, Any]:
        session_id = f"observer_{uuid.uuid4().hex}"
        if self.recorder is not None:
            self.recorder.clear(session_id)
        try:
            return self._run_trace_session(sample_id, session_id)
        finally:
            drop_session = getattr(self.agent, "drop_session", None)
            if callable(drop_session):
                drop_session(session_id)
            if self.recorder is not None:
                self.recorder.clear(session_id)

    def _run_trace_session(self, sample_id: str, session_id: str) -> dict[str, Any]:
        trace_started = time.perf_counter()
        sample = self.samples[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
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
            turn_started = time.perf_counter()
            disclosed_before = sorted(disclosed)
            retrieval = _retrieval_snapshot(user_message)
            error: str | None = None
            try:
                response = self.agent.respond(session_id, user_message, turn, TOP_K)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                error = error or "Invalid response object or message"
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            agent_events = (
                self.recorder.events(session_id, turn) if self.recorder is not None else []
            )
            state_snapshot: dict[str, Any] = {}
            for agent_event in agent_events:
                event_data = agent_event.get("data") or {}
                if agent_event.get("layer") == "parse":
                    retrieval["terms"] = event_data.get("terms", retrieval["terms"])
                    retrieval["expression"] = event_data.get(
                        "fts_expression", retrieval["expression"]
                    )
                    retrieval["strict_expression"] = event_data.get("strict_fts_expression")
                elif agent_event.get("layer") == "retrieval":
                    retrieval["candidate_count"] = event_data.get("candidate_count")
                    retrieval["route_counts"] = event_data.get("route_counts") or {}
                    retrieval["engine"] = event_data.get("engine")
                    retrieval["retrieval_mode"] = event_data.get("retrieval_mode")
                    retrieval["coverage_diagnostics"] = event_data.get("coverage")
                elif agent_event.get("layer") == "state":
                    state_snapshot = dict(event_data)

            diagnostics = _agent_diagnostics(self.agent, session_id, target)
            state_snapshot = diagnostics["state"] or state_snapshot
            retrieval["route_counts"] = diagnostics["route_counts"]
            retrieval["target_broad_rank"] = diagnostics["target_broad_rank"]
            retrieval["target_strict_rank"] = diagnostics["target_strict_rank"]
            retrieval["target_fused_rank"] = diagnostics["target_fused_rank"]
            retrieval["target_reranked_rank"] = diagnostics["target_reranked_rank"]
            retrieval["target_final_rank"] = diagnostics["target_final_rank"]
            retrieval["target_rerank_breakdown"] = diagnostics[
                "target_rerank_breakdown"
            ]
            retrieval["rerank_diagnostics"] = diagnostics["rerank_diagnostics"]
            retrieval["coverage_diagnostics"] = diagnostics[
                "coverage_diagnostics"
            ]
            retrieval["posthoc_target_rank"] = diagnostics["target_final_rank"]
            retrieval["actual_route"] = diagnostics["actual_route"]
            retrieval["rerank_mode"] = diagnostics["rerank_mode"]
            retrieval["retrieval_mode"] = diagnostics["retrieval_mode"]
            retrieval["diagnostic_error"] = diagnostics["error"]
            retrieval["posthoc_note"] = (
                "Public ground truth was joined after Agent.respond and ranked locally against "
                "target-blind broad, strict, fused, reranked, and final route IDs. The final "
                "route is the actual output order; in coverage mode fused is the control and "
                "final is the promoted coverage order. Legacy Agents fall back to fused."
            )
            ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
            validation = _output_diagnostics(response.get("recommendations"), self.catalog_ids)
            target_top10_rank = ranked.index(target) + 1 if target in ranked else None
            eligible_for_hit = override_applied
            hit = eligible_for_hit and target_top10_rank is not None
            if hit:
                hit_turn = turn
                best_rank = target_top10_rank

            if error:
                failure_code = "AGENT_ERROR"
            elif hit:
                failure_code = "HIT"
            elif not eligible_for_hit and target_top10_rank is not None:
                failure_code = "PRE_OVERRIDE_NOT_SCORABLE"
            elif retrieval["posthoc_target_rank"] is None:
                failure_code = "RETRIEVAL_MISS"
            elif int(retrieval["posthoc_target_rank"]) > TOP_K:
                failure_code = "LOW_FINAL_RANK"
            else:
                failure_code = "OUTPUT_OR_NORMALIZATION_MISS"

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
                "state_snapshot": state_snapshot,
                "agent_events": agent_events,
                "validation": validation,
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
                "failure_code": failure_code,
                "simulator_disclosed_before": disclosed_before,
                "simulator_disclosed_after": sorted(disclosed),
                "next_user_message": next_user_message,
                "elapsed_ms": round((time.perf_counter() - turn_started) * 1000.0, 3),
            })
            if hit or turn == MAX_TURNS:
                break
            user_message = str(next_user_message)

        contribution = (
            0.0
            if hit_turn is None or best_rank is None
            else 0.5 + 0.3 / best_rank + 0.02 * (11 - hit_turn)
        )
        if hit_turn is not None:
            diagnosis = "SUCCESS"
        elif any(turn["failure_code"] == "AGENT_ERROR" for turn in turns):
            diagnosis = "AGENT_ERROR"
        elif all(turn["retrieval"]["posthoc_target_rank"] is None for turn in turns):
            diagnosis = "RETRIEVAL_MISS"
        elif min(
            turn["retrieval"]["posthoc_target_rank"]
            for turn in turns
            if turn["retrieval"]["posthoc_target_rank"] is not None
        ) > TOP_K:
            diagnosis = "LOW_FINAL_RANK"
        elif any(turn["failure_code"] == "PRE_OVERRIDE_NOT_SCORABLE" for turn in turns):
            diagnosis = "PRE_OVERRIDE_NOT_SCORABLE"
        else:
            diagnosis = "OUTPUT_OR_NORMALIZATION_MISS"
        trace = {
            "sample_id": sample_id,
            "scenario_type": str(sample["scenario_type"]),
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "profile": sample["user_profile"],
            "target": _product_view(self.products[target], target),
            "intent_card": card,
            "behavior": behavior,
            "rerank_mode": _agent_rerank_mode(self.agent),
            "retrieval_mode": _agent_retrieval_mode(self.agent),
            "result": {
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else round(1.0 / best_rank, 6),
                "technical_contribution": round(contribution, 6),
                "diagnosis": diagnosis,
            },
            "turns": turns,
            "elapsed_ms": round((time.perf_counter() - trace_started) * 1000.0, 3),
            "observer_note": (
                "Ground truth and intent card are visible only in this public-set debug observer; "
                "the Agent receives only an opaque session ID, profile, user message, turn, and top_k."
            ),
        }
        self._cache[sample_id] = trace
        return trace
