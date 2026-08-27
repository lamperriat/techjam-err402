from __future__ import annotations

import threading
from collections import defaultdict
from copy import deepcopy
from typing import Any


TRACE_SCHEMA_VERSION = "2.0"


class TraceRecorder:
    """Thread-safe collector for optional Agent development events."""

    def __init__(self, max_sessions: int = 64) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_sessions = max(1, max_sessions)

    def emit(self, event: dict[str, Any]) -> None:
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return
        normalized = {**deepcopy(event), "schema_version": TRACE_SCHEMA_VERSION}
        with self._lock:
            if session_id not in self._events and len(self._events) >= self._max_sessions:
                self._events.pop(next(iter(self._events)))
            self._events[session_id].append(normalized)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._events.pop(session_id, None)

    def events(self, session_id: str, turn: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            values = deepcopy(self._events.get(session_id, []))
        if turn is None:
            return values
        return [event for event in values if event.get("turn") in {None, turn}]
