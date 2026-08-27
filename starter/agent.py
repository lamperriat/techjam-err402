from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


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


class Agent:
    """Editable weak baseline with BM25 retrieval and optional LLM usage reporting."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        llm_client: Any | None = None,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.llm_client = llm_client
        self.trace_sink = trace_sink
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        self._sessions: set[str] = set()
        self._build_index()

    def _trace(self, session_id: str, turn: int | None, layer: str, data: dict[str, Any]) -> None:
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
        # The profile is anonymized and may be used for personalization.
        with self._lock:
            self._sessions.add(session_id)
            self._trace(session_id, None, "session", {
                "memory_mode": "reset-only / stateless baseline",
                "active_slots": {},
                "profile_keys_received": sorted(str(key) for key in user_profile),
            })

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
            unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
            expression = " OR ".join(f'"{term}"' for term in unique_terms)
            self._trace(session_id, turn, "parse", {
                "input": user_message,
                "terms": unique_terms,
                "fts_expression": expression,
            })
            rows: list[tuple[str, float]] = []
            candidate_count = 0
            if expression:
                rows = self.connection.execute(
                    "SELECT parent_asin, "
                    "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS rank_score "
                    "FROM products WHERE products MATCH ? ORDER BY rank_score LIMIT ?",
                    (expression, top_k),
                ).fetchall()
                if self.trace_sink is not None:
                    candidate_count = int(self.connection.execute(
                        "SELECT count(*) FROM products WHERE products MATCH ?",
                        (expression,),
                    ).fetchone()[0])
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
            self._trace(session_id, turn, "retrieval", {
                "engine": "SQLite FTS5 / BM25",
                "candidate_count": candidate_count,
                "field_weights": [0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0],
                "top_results": [
                    {"parent_asin": str(parent_asin), "bm25_score": round(float(score), 8)}
                    for parent_asin, score in rows
                ],
            })
            self._trace(session_id, turn, "policy", {
                "ask_attribute": None,
                "reason": "The current weak baseline has no clarification policy.",
            })
            usage = (
                self.llm_client.consume_usage().as_dict()
                if self.llm_client is not None
                else {"prompt_tokens": 0, "completion_tokens": 0}
            )
            response = {
                "message": "Here are the closest matches I found.",
                "ask_attribute": None,
                "recommendations": recommendations,
                "usage": usage,
            }
            self._trace(session_id, turn, "output", {
                "recommendation_count": len(recommendations),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "usage": usage,
            })
            return response
