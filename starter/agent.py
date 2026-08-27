from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CATEGORY_RE = re.compile(
    r"(?:i(?:'m| am)\s+looking\s+for|looking\s+for)\s+(.+?)(?:\.|,|;|\bbut\b|$)",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"(?:don'?t|do not)\s+have\s+(?:(?:an?|any)\s+)?(?:additional\s+)?"
    r"preference\s+for\s+([a-z_]+)",
    re.IGNORECASE,
)
NEGATIVE_TERM_RE = re.compile(
    r"\b(?:not|no|without)\s+(?:too\s+)?([a-z0-9-]+)", re.IGNORECASE
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking", "need",
    "what", "those", "options", "quite", "right", "yet", "ask", "about", "one",
    "specific", "attribute", "matters", "key", "requirement", "currently", "still",
    "exploring", "additional", "preference", "use", "your", "judgment", "actually",
    "ignore", "earlier", "prioritize", "target", "requirements", "here", "closest",
    "found", "not", "too",
}

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
}
QUESTION_ORDER = (
    "material", "color", "feature", "style", "use_case", "size", "budget", "brand", "other"
)

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
    "linen", "denim", "fleece", "mesh", "suede", "canvas", "rubber",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple",
    "yellow", "orange", "beige", "navy", "gold", "silver", "tan", "khaki",
}
SIZE_TERMS = {
    "size", "sizing", "width", "wide", "narrow", "xs", "small", "medium", "large", "xl",
    "xxl", "petite", "tall",
}
STYLE_TERMS = {
    "department", "style", "fit", "sleeve", "neck", "casual", "formal", "vintage",
    "classic", "modern", "sporty", "athletic", "slim", "relaxed", "oversized", "elegant",
}
USE_CASE_TERMS = {
    "hiking", "running", "gym", "winter", "outdoor", "work", "walking", "workout", "office",
    "wedding", "travel", "school", "beach", "rain", "snow", "cycling",
}
BUDGET_MARKERS = {"budget", "under", "below", "price", "cheaper", "less"}

OVERRIDE_PATTERNS = (
    re.compile(r"\bignore\s+my\s+earlier\s+preference\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bchange(?:d)?\s+my\s+mind\b", re.IGNORECASE),
)

CONTENT_PATTERNS = (
    re.compile(r"key requirement is:\s*(.+)", re.IGNORECASE),
    re.compile(r"for that, what matters is:\s*(.+)", re.IGNORECASE),
    re.compile(r"what matters is:\s*(.+)", re.IGNORECASE),
    re.compile(r"what i need is:\s*(.+)", re.IGNORECASE),
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _content_fragment(message: str) -> str:
    for pattern in CONTENT_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip(" .")
    return message


def _classify_message(message: str) -> set[str]:
    """Return the attribute classes explicitly represented in a message."""
    lowered = message.lower()
    tokens = set(_terms(lowered))
    attributes: set[str] = set()
    if tokens & MATERIALS:
        attributes.add("material")
    if tokens & COLORS or "color" in tokens:
        attributes.add("color")
    if tokens & SIZE_TERMS:
        attributes.add("size")
    if tokens & STYLE_TERMS:
        attributes.add("style")
    if tokens & USE_CASE_TERMS:
        attributes.add("use_case")
    if tokens & BUDGET_MARKERS or "$" in lowered:
        attributes.add("budget")
    if any(pattern.search(message) for pattern in CONTENT_PATTERNS) and not attributes:
        attributes.add("feature")
    return attributes


@dataclass
class SessionState:
    profile: dict[str, Any]
    category_text: str = ""
    active_terms: list[str] = field(default_factory=list)
    excluded_terms: set[str] = field(default_factory=set)
    known_attributes: set[str] = field(default_factory=set)
    asked_attributes: list[str] = field(default_factory=list)
    exhausted_attributes: set[str] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    turn_terms: dict[int, list[str]] = field(default_factory=dict)
    turn_excluded_terms: dict[int, set[str]] = field(default_factory=dict)
    turn_attributes: dict[int, set[str]] = field(default_factory=dict)
    version: int = 1
    override_count: int = 0
    version_anchor_turn: int = 1
    prefer_other_next: bool = False


class Agent:
    """Offline stateful sparse agent with auditable rank fusion and question policy."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        llm_client: Any | None = None,
        question_policy: str | None = None,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.llm_client = llm_client
        self.trace_sink = trace_sink
        self.question_policy = (
            question_policy or os.getenv("TECHJAM_QUESTION_POLICY", "fast")
        ).strip().lower()
        if self.question_policy not in {"conservative", "boundary", "fast"}:
            raise ValueError(
                "question_policy must be one of: conservative, boundary, fast"
            )
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._build_index()

    def _trace(
        self,
        session_id: str,
        turn: int | None,
        layer: str,
        data: dict[str, Any],
    ) -> None:
        if self.trace_sink is not None:
            self.trace_sink({
                "session_id": session_id,
                "turn": turn,
                "layer": layer,
                "data": data,
            })

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._lock:
            self._sessions[session_id] = SessionState(profile=dict(user_profile))
            self._trace(session_id, None, "session", {
                "memory_mode": "versioned multi-turn state",
                "question_policy": self.question_policy,
                "profile_keys_received": sorted(str(key) for key in user_profile),
            })

    def drop_session(self, session_id: str) -> None:
        """Release optional development-session state after a replay or Lab eviction."""
        with self._lock:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _is_override(message: str) -> bool:
        return any(pattern.search(message) for pattern in OVERRIDE_PATTERNS)

    @staticmethod
    def _extract_category(message: str) -> str:
        match = CATEGORY_RE.search(message)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")[:160]

    @staticmethod
    def _negative_terms(message: str) -> set[str]:
        return {match.group(1).lower() for match in NEGATIVE_TERM_RE.finditer(message)}

    def _update_state(self, state: SessionState, user_message: str, turn: int) -> None:
        state.messages.append(user_message)

        no_preference_match = NO_PREFERENCE_RE.search(user_message)
        if no_preference_match:
            attribute = no_preference_match.group(1).lower()
            if attribute in ALLOWED_ATTRIBUTES:
                state.exhausted_attributes.add(attribute)
                is_direct_boundary_reply = "additional preference" not in user_message.lower()
                if self.question_policy == "fast":
                    state.prefer_other_next = True
                elif self.question_policy == "boundary" and is_direct_boundary_reply:
                    state.prefer_other_next = True

        category = self._extract_category(user_message)
        if category:
            if state.category_text and category.lower() != state.category_text.lower():
                state.active_terms.clear()
                state.excluded_terms.clear()
                state.known_attributes.clear()
                state.asked_attributes.clear()
                state.exhausted_attributes.clear()
                state.turn_terms.clear()
                state.turn_excluded_terms.clear()
                state.turn_attributes.clear()
                state.prefer_other_next = False
                state.version += 1
                state.version_anchor_turn = turn
            state.category_text = category

        if self._is_override(user_message):
            stale_turn = state.version_anchor_turn
            stale_terms = set(state.turn_terms.pop(stale_turn, []))
            state.turn_excluded_terms.pop(stale_turn, None)
            state.turn_attributes.pop(stale_turn, None)
            state.active_terms = [
                term for term in state.active_terms if term not in stale_terms
            ]
            state.excluded_terms = (
                set().union(*state.turn_excluded_terms.values())
                if state.turn_excluded_terms
                else set()
            )
            state.known_attributes = (
                set().union(*state.turn_attributes.values())
                if state.turn_attributes
                else set()
            )
            state.version += 1
            state.override_count += 1
            state.version_anchor_turn = turn

        negative_terms = self._negative_terms(user_message)
        if negative_terms:
            state.turn_excluded_terms[turn] = negative_terms
            state.excluded_terms.update(negative_terms)

        lowered = user_message.lower()
        if no_preference_match or "not quite right yet" in lowered:
            return

        fragment = _content_fragment(user_message)
        added_terms: list[str] = []
        for term in _terms(fragment):
            if term not in state.excluded_terms and term not in state.active_terms:
                state.active_terms.append(term)
                added_terms.append(term)
        if added_terms:
            state.turn_terms[turn] = added_terms
        attributes = _classify_message(fragment)
        if attributes:
            state.turn_attributes[turn] = attributes
            state.known_attributes.update(attributes)

    @staticmethod
    def _fts_expression(terms: list[str]) -> str:
        unique = [term for term in dict.fromkeys(terms) if len(term) > 1][:50]
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique)

    def _query_terms(self, state: SessionState) -> list[str]:
        category_terms = _terms(state.category_text)
        return [
            term
            for term in dict.fromkeys([*category_terms, *state.active_terms])
            if term not in state.excluded_terms
        ]

    @staticmethod
    def _strict_fts_expression(terms: list[str]) -> str:
        unique = [term for term in dict.fromkeys(terms) if len(term) > 1][:16]
        return " AND ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique
        )

    @staticmethod
    def _fusion_score(
        parent_asin: str,
        broad_rank: dict[str, int],
        strict_rank: dict[str, int],
    ) -> float:
        score = 0.0
        if parent_asin in broad_rank:
            score += 1.0 / (60.0 + broad_rank[parent_asin])
        if parent_asin in strict_rank:
            score += 1.8 / (20.0 + strict_rank[parent_asin])
        return score

    def _rank_candidates(self, state: SessionState) -> dict[str, list[str]]:
        query_terms = self._query_terms(state)
        broad_expression = self._fts_expression(query_terms)
        if not broad_expression:
            return {"broad": [], "strict": [], "fused": []}

        broad_rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 120",
            (broad_expression,),
        ).fetchall()
        broad_ids = [str(row[0]) for row in broad_rows]

        strict_ids: list[str] = []
        if len(query_terms) >= 2:
            strict_expression = self._strict_fts_expression(query_terms)
            if strict_expression:
                strict_rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 80",
                    (strict_expression,),
                ).fetchall()
                strict_ids = [str(row[0]) for row in strict_rows]

        broad_rank = {asin: rank for rank, asin in enumerate(broad_ids, start=1)}
        strict_rank = {asin: rank for rank, asin in enumerate(strict_ids, start=1)}
        candidates = dict.fromkeys([*broad_ids, *strict_ids])
        fused_ids = sorted(
            candidates,
            key=lambda asin: (
                -self._fusion_score(asin, broad_rank, strict_rank),
                broad_rank.get(asin, 10**9),
                asin,
            ),
        )
        return {"broad": broad_ids, "strict": strict_ids, "fused": fused_ids}

    def debug_rankings(self, session_id: str) -> dict[str, list[str]]:
        """Return the exact route rankings used by the current Agent."""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            return self._rank_candidates(self._sessions[session_id])

    @staticmethod
    def _select_question(state: SessionState, turn: int) -> str | None:
        if turn >= 10:
            return None
        blocked = (
            state.known_attributes
            | state.exhausted_attributes
            | set(state.asked_attributes)
        )
        if state.prefer_other_next and "other" not in blocked:
            state.prefer_other_next = False
            state.asked_attributes.append("other")
            return "other"
        for attribute in QUESTION_ORDER:
            if attribute not in blocked:
                state.asked_attributes.append(attribute)
                return attribute
        return None

    @staticmethod
    def _message(state: SessionState, ask_attribute: str | None) -> str:
        if ask_attribute is None:
            return "Here are the strongest matches for your current requirements."
        if ask_attribute == "other":
            return (
                "Is there anything else that matters, such as comfort, closure, "
                "or a specific feature?"
            )
        return f"Do you have a preference for {ask_attribute.replace('_', ' ')}?"

    def _snapshot(self, state: SessionState) -> dict[str, Any]:
        return {
            "version": state.version,
            "override_count": state.override_count,
            "version_anchor_turn": state.version_anchor_turn,
            "prefer_other_next": state.prefer_other_next,
            "question_policy": self.question_policy,
            "category_text": state.category_text,
            "active_terms": list(state.active_terms),
            "excluded_terms": sorted(state.excluded_terms),
            "known_attributes": sorted(state.known_attributes),
            "asked_attributes": list(state.asked_attributes),
            "turn_terms": {str(key): list(value) for key, value in state.turn_terms.items()},
            "turn_excluded_terms": {
                str(key): sorted(value) for key, value in state.turn_excluded_terms.items()
            },
            "exhausted_attributes": sorted(state.exhausted_attributes),
            "query_terms": self._query_terms(state),
        }

    def debug_snapshot(self, session_id: str) -> dict[str, Any]:
        """Return state derived only from profile and conversation messages."""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            return self._snapshot(self._sessions[session_id])

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        started = time.perf_counter()
        with self._lock:
            if session_id not in self._sessions:
                raise RuntimeError("reset must be called before respond")
            if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 10:
                raise ValueError("turn must be an integer from 1 to 10")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")

            state = self._sessions[session_id]
            self._update_state(state, user_message, turn)
            query_terms = self._query_terms(state)
            broad_expression = self._fts_expression(query_terms)
            strict_expression = (
                self._strict_fts_expression(query_terms) if len(query_terms) >= 2 else ""
            )
            self._trace(session_id, turn, "parse", {
                "input": user_message,
                "terms": query_terms,
                "fts_expression": broad_expression,
                "strict_fts_expression": strict_expression,
            })

            rankings = self._rank_candidates(state)
            recommendations = [
                {"parent_asin": asin}
                for asin in rankings["fused"][: min(max(top_k, 1), 10)]
            ]
            if self.trace_sink is not None:
                broad_rank = {
                    asin: rank for rank, asin in enumerate(rankings["broad"], start=1)
                }
                strict_rank = {
                    asin: rank for rank, asin in enumerate(rankings["strict"], start=1)
                }
                top_results = [
                    {
                        "parent_asin": asin,
                        "broad_rank": broad_rank.get(asin),
                        "strict_rank": strict_rank.get(asin),
                        "fusion_score": round(
                            self._fusion_score(asin, broad_rank, strict_rank), 8
                        ),
                    }
                    for asin in rankings["fused"][:10]
                ]
                self._trace(session_id, turn, "retrieval", {
                    "engine": "SQLite FTS5 BM25 + weighted RRF",
                    "candidate_count": len(rankings["fused"]),
                    "route_counts": {
                        "broad": len(rankings["broad"]),
                        "strict": len(rankings["strict"]),
                        "fused": len(rankings["fused"]),
                    },
                    "fusion_formula": "1/(60+broad_rank) + 1.8/(20+strict_rank)",
                    "top_results": top_results,
                })

            ask_attribute = self._select_question(state, turn)
            snapshot = self._snapshot(state)
            self._trace(session_id, turn, "state", {
                "memory_mode": "versioned multi-turn state",
                "active_slots": {
                    "category": snapshot["category_text"],
                    "terms": snapshot["active_terms"],
                    "known_attributes": snapshot["known_attributes"],
                },
                **snapshot,
            })
            if ask_attribute is None:
                policy_reason = "No answerable question remains or this is the final turn."
            elif ask_attribute == "other":
                policy_reason = "Fast disclosure fallback after an exhausted attribute."
            else:
                policy_reason = "First unanswered attribute in the deterministic policy order."
            self._trace(session_id, turn, "policy", {
                "ask_attribute": ask_attribute,
                "question_policy": self.question_policy,
                "reason": policy_reason,
            })

            usage = (
                self.llm_client.consume_usage().as_dict()
                if self.llm_client is not None
                else {"prompt_tokens": 0, "completion_tokens": 0}
            )
            response = {
                "message": self._message(state, ask_attribute),
                "ask_attribute": ask_attribute,
                "recommendations": recommendations,
                "usage": usage,
            }
            self._trace(session_id, turn, "output", {
                "recommendation_count": len(recommendations),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "usage": usage,
            })
            return response
