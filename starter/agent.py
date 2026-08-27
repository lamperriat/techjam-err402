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
CATEGORY_PATTERNS = (
    re.compile(
        r"^\s*(?:i(?:'m| am)\s+)?(?:looking|searching|shopping)\s+for\s+"
        r"(.+?)(?:[.,;]|\s+\bbut\b|\s+\bwith\b|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:i\s+(?:need|want|am\s+after)|i(?:'d|\s+would)\s+like|"
        r"show\s+me|find\s+me|help\s+me\s+find)\s+"
        r"(.+?)(?:[.,;]|\s+\bbut\b|\s+\bwith\b|$)",
        re.IGNORECASE,
    ),
)
NO_PREFERENCE_PATTERNS = (
    re.compile(
        r"(?:don'?t|do not)\s+have\s+(?:(?:an?|any)\s+)?(?:additional\s+)?"
        r"preference\s+(?:for|on|about)\s+([a-z_]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:strong\s+)?preference\s+(?:for|on|about)\s+([a-z_]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:any|either)\s+([a-z_]+)\s+(?:works|is\s+(?:fine|okay|ok))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([a-z_]+)\s+(?:is\s+flexible\s+for\s+me|doesn'?t\s+matter|"
        r"is\s+not\s+important\s+to\s+me)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi\s+don'?t\s+care\s+about\s+([a-z_]+)\b", re.IGNORECASE),
)
GENERIC_NO_PREFERENCE_RE = re.compile(
    r"\b(?:no\s+(?:strong\s+)?preference|you\s+(?:can\s+)?decide|"
    r"use\s+your\s+judgment|anything\s+(?:is|would\s+be)\s+fine)\b",
    re.IGNORECASE,
)
RETRY_RE = re.compile(
    r"\b(?:not\s+quite\s+right\s+yet|do\s+not\s+fit\s+yet|"
    r"not\s+a\s+match\s+yet|none\s+of\s+(?:these|those).{0,30}(?:fit|match))\b",
    re.IGNORECASE,
)
VAGUE_RE = re.compile(
    r"^(?:(?:but\s+)?i(?:'m| am)\s+)?(?:still\s+exploring|just\s+browsing|"
    r"not\s+sure\s+yet)[.!\s]*$",
    re.IGNORECASE,
)
NEGATIVE_TERM_RE = re.compile(
    r"\b(?:(?:not(?!\s+(?:only|quite|sure)\b)|"
    r"no(?!\s+(?:preference|additional|longer)\b)|without)\s+(?:too\s+)?|"
    r"(?:don'?t|do\s+not)\s+(?:want|need|prefer)\s+)"
    r"([a-z0-9-]+)",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking", "need",
    "what", "those", "options", "quite", "right", "yet", "ask", "about", "one",
    "specific", "attribute", "matters", "key", "requirement", "currently", "still",
    "exploring", "additional", "preference", "use", "your", "judgment", "actually",
    "ignore", "earlier", "prioritize", "target", "requirements", "here", "closest",
    "found", "not", "too", "only", "shopping", "searching", "main", "thing",
    "important", "detail", "deciding", "factor", "strong", "flexible", "choose",
    "fits", "best", "works", "decide", "previous", "choice", "before", "rather",
    "disregard", "said", "forget", "instead", "now", "longer", "switch", "change",
    "replace", "don", "dont", "fine", "okay", "ok", "sounds",
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
CATEGORY_HEAD_TERMS = {
    "accessories", "apparel", "bag", "bags", "belt", "belts", "blazer", "blazers",
    "boot", "boots", "bracelet", "bracelets", "bra", "bras", "coat", "coats",
    "dress", "dresses", "earring", "earrings", "glove", "gloves", "handbag",
    "handbags", "hat", "hats", "hoodie", "hoodies", "jacket", "jackets", "jeans",
    "jewelry", "jumpsuit", "jumpsuits", "leggings", "necklace", "necklaces", "pants",
    "ring", "rings", "sandal", "sandals", "shirt", "shirts", "shoe", "shoes",
    "shorts", "skirt", "skirts", "sneaker", "sneakers", "sock", "socks", "suit",
    "suits", "sweater", "sweaters", "swimwear", "tie", "ties", "top", "tops",
    "underwear", "wallet", "wallets", "watch", "watches",
}

OVERRIDE_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:earlier|previous|prior|before)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bchange(?:d)?\s+my\s+mind\b", re.IGNORECASE),
    re.compile(r"\bno\s+longer\s+(?:want|need|prefer)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:switch|change|replace)\b.{0,40}\bfrom\b.{0,40}\bto\b",
        re.IGNORECASE,
    ),
    re.compile(r"\breplace\b.{0,40}\bwith\b", re.IGNORECASE),
    re.compile(r"^\s*instead\b", re.IGNORECASE),
    re.compile(
        r"\b(?:want|need|prefer|choose|change)\b.{0,40}\binstead\b|"
        r"\binstead\b.{0,40}\b(?:want|need|prefer|choose|change)\b",
        re.IGNORECASE,
    ),
)

CONTENT_PATTERNS = (
    re.compile(r"\b(?:key|main|primary)\s+(?:requirement|priority)\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\b(?:for\s+that,?\s*)?what\s+matters\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+i\s+need\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\bthe\s+main\s+thing\s+that\s+matters\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\bthe\s+most\s+important\s+detail\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\bthe\s+deciding\s+factor\s+for\s+me\s+(?:is\s*:|is|:)\s*(.+)", re.IGNORECASE),
    re.compile(r"\b(?:i\s+would\s+rather\s+have|please\s+prioritize)\s*:\s*(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+(?:would\s+)?prefer\s*:?[ ]+(.+)", re.IGNORECASE),
    re.compile(r"\bmust\s+have\s*:?[ ]+(.+)", re.IGNORECASE),
)

SWITCH_SPAN_RE = re.compile(
    r"\b(?:switch|change|replace)\b.{0,40}?\bfrom\b\s*(?P<old>.+?)\s+"
    r"\bto\b\s*(?P<new>.+)$",
    re.IGNORECASE,
)
REPLACE_WITH_SPAN_RE = re.compile(
    r"\breplace\s+(?P<old>.+?)\s+\bwith\b\s+(?P<new>.+)$",
    re.IGNORECASE,
)
INSTEAD_OF_SPAN_RE = re.compile(
    r"^\s*instead\s+of\s+(?P<old>.+?)[,;]\s*(?P<new>.+)$",
    re.IGNORECASE,
)
NO_LONGER_REPLACEMENT_RE = re.compile(
    r"\bno\s+longer\s+(?:want|need|prefer)\b\s*(?P<old>[^;,.]+?)\s*"
    r"[;,.]\s*(?P<new>.+)$",
    re.IGNORECASE,
)
NO_LONGER_ONLY_RE = re.compile(
    r"\bno\s+longer\s+(?:want|need|prefer)\b\s*(?P<old>.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
INSTEAD_PREFIX_RE = re.compile(r"^\s*instead\b[,;:]?\s*(?P<new>.+)$", re.IGNORECASE)
INSTEAD_SUFFIX_RE = re.compile(
    r"\b(?:want|need|prefer|choose)\s+(?P<new>.+?)\s+instead\b",
    re.IGNORECASE,
)
CHANGED_MIND_CONTENT_RE = re.compile(
    r"\bchange(?:d)?\s+my\s+mind\b\s*[:.;,-]\s*(?P<new>.+)$",
    re.IGNORECASE,
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


def _category_and_remainder(
    message: str, *, allow_loose_openers: bool = True
) -> tuple[str, str]:
    patterns = CATEGORY_PATTERNS if allow_loose_openers else CATEGORY_PATTERNS[:1]
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            category = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")[:160]
            remainder = message[match.end():].strip(" .,:;-")
            remainder = re.sub(r"^but\s+", "", remainder, flags=re.IGNORECASE)
            return category, remainder
    return "", ""


def _content_fragment(
    message: str,
    category_text: str = "",
    category_remainder: str = "",
) -> tuple[str, bool]:
    for pattern in CONTENT_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip(" ."), True
    if category_text:
        if not category_remainder or VAGUE_RE.fullmatch(category_remainder):
            return "", False
        return category_remainder, False
    return message, False


def _clean_override_replacement(fragment: str) -> str:
    fragment = fragment.strip(" .")
    fragment = re.sub(r"^(?:but\s+)?", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(
        r"^(?:i\s+)?(?:now\s+)?(?:want|need|prefer|choose)\s+",
        "",
        fragment,
        flags=re.IGNORECASE,
    )
    fragment = re.sub(
        r"\s+(?:instead|now|please)\s*$", "", fragment, flags=re.IGNORECASE
    )
    return fragment.strip(" .")


def _override_content_fragment(message: str) -> tuple[str, bool, str]:
    for pattern in (
        SWITCH_SPAN_RE,
        REPLACE_WITH_SPAN_RE,
        INSTEAD_OF_SPAN_RE,
        NO_LONGER_REPLACEMENT_RE,
    ):
        match = pattern.search(message)
        if match:
            return (
                _clean_override_replacement(match.group("new")),
                True,
                match.group("old").strip(" ."),
            )
    for pattern in (INSTEAD_PREFIX_RE, INSTEAD_SUFFIX_RE):
        match = pattern.search(message)
        if match:
            return _clean_override_replacement(match.group("new")), True, ""

    fragment, wrapped = _content_fragment(message)
    if wrapped:
        return fragment, True, ""

    changed_mind = CHANGED_MIND_CONTENT_RE.search(message)
    if changed_mind:
        return _clean_override_replacement(changed_mind.group("new")), True, ""

    no_longer = NO_LONGER_ONLY_RE.search(message)
    if no_longer:
        return "", True, no_longer.group("old").strip(" .")
    if any(pattern.search(message) for pattern in OVERRIDE_PATTERNS):
        return "", True, ""
    return message, False, ""


def _no_preference(message: str, pending_attribute: str | None) -> tuple[bool, str | None]:
    for pattern in NO_PREFERENCE_PATTERNS:
        match = pattern.search(message)
        if match:
            attribute = match.group(1).lower()
            return True, attribute if attribute in ALLOWED_ATTRIBUTES else None
    if GENERIC_NO_PREFERENCE_RE.search(message):
        attribute = pending_attribute if pending_attribute in ALLOWED_ATTRIBUTES else None
        return True, attribute
    return False, None


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
    return attributes


@dataclass(frozen=True)
class ParsedTurn:
    category_text: str
    content_fragment: str
    content_wrapped: bool
    is_override: bool
    override_old_fragment: str
    is_retry: bool
    is_no_preference: bool
    no_preference_attribute: str | None
    no_preference_is_additional: bool
    negative_terms: frozenset[str]
    attributes: frozenset[str]


def _parse_turn(
    message: str,
    pending_attribute: str | None = None,
    current_category: str = "",
) -> ParsedTurn:
    is_override = any(pattern.search(message) for pattern in OVERRIDE_PATTERNS)
    override_old_fragment = ""
    if is_override:
        fragment, content_wrapped, override_old_fragment = _override_content_fragment(
            message
        )
        category_text, category_remainder = "", ""
        old_category_terms = set(_terms(override_old_fragment))
        new_category_terms = set(_terms(fragment))
        current_category_terms = set(_terms(current_category))
        old_heads = old_category_terms & CATEGORY_HEAD_TERMS
        new_heads = new_category_terms & CATEGORY_HEAD_TERMS
        if (
            fragment
            and old_category_terms
            and old_category_terms <= current_category_terms
            and (
                (old_heads and new_heads and old_heads != new_heads)
                or old_category_terms == current_category_terms
            )
        ):
            category_text = fragment
            fragment = ""
    else:
        category_text, category_remainder = _category_and_remainder(
            message, allow_loose_openers=not bool(current_category)
        )
        fragment, content_wrapped = "", False
        if (
            current_category
            and category_text
            and not (set(_terms(category_text)) & CATEGORY_HEAD_TERMS)
        ):
            fragment = " ".join(
                part for part in (category_text, category_remainder) if part
            )
            category_text = ""
            category_remainder = ""
    if not is_override and not fragment:
        fragment, content_wrapped = _content_fragment(
            message, category_text, category_remainder
        )
    is_no_preference, no_preference_attribute = _no_preference(
        message, pending_attribute
    )
    is_retry = bool(RETRY_RE.search(message))
    suppress_fragment_semantics = is_retry or (
        is_no_preference and not content_wrapped
    )
    negative_terms = (
        frozenset()
        if suppress_fragment_semantics
        else frozenset(
            match.group(1).lower() for match in NEGATIVE_TERM_RE.finditer(fragment)
        )
    )
    attributes = (
        frozenset()
        if suppress_fragment_semantics or not fragment
        else frozenset(_classify_message(fragment))
    )
    lowered = message.lower()
    return ParsedTurn(
        category_text=category_text,
        content_fragment=fragment,
        content_wrapped=content_wrapped,
        is_override=is_override,
        override_old_fragment=override_old_fragment,
        is_retry=is_retry,
        is_no_preference=is_no_preference,
        no_preference_attribute=no_preference_attribute,
        no_preference_is_additional=(
            "additional" in lowered
            or "flexible" in lowered
            or "not important" in lowered
        ),
        negative_terms=negative_terms,
        attributes=attributes,
    )


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
    pending_attribute: str | None = None
    pending_turn: int | None = None


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
        return _category_and_remainder(message)[0]

    @staticmethod
    def _negative_terms(message: str) -> set[str]:
        return {match.group(1).lower() for match in NEGATIVE_TERM_RE.finditer(message)}

    @staticmethod
    def _finish_pending(state: SessionState) -> None:
        if (
            state.pending_attribute is not None
            and state.pending_attribute not in state.asked_attributes
        ):
            state.asked_attributes.append(state.pending_attribute)
        state.pending_attribute = None
        state.pending_turn = None

    @staticmethod
    def _cancel_pending(state: SessionState) -> None:
        if state.pending_attribute == "other":
            state.prefer_other_next = True
        state.pending_attribute = None
        state.pending_turn = None

    def _update_state(
        self, state: SessionState, user_message: str, turn: int
    ) -> ParsedTurn:
        state.messages.append(user_message)
        parsed = _parse_turn(
            user_message, state.pending_attribute, state.category_text
        )

        category = parsed.category_text
        category_changed = bool(
            category
            and state.category_text
            and category.lower() != state.category_text.lower()
        )
        if category:
            if category_changed:
                state.active_terms.clear()
                state.excluded_terms.clear()
                state.known_attributes.clear()
                state.asked_attributes.clear()
                state.exhausted_attributes.clear()
                state.turn_terms.clear()
                state.turn_excluded_terms.clear()
                state.turn_attributes.clear()
                state.prefer_other_next = False
                state.pending_attribute = None
                state.pending_turn = None
                state.version += 1
                state.version_anchor_turn = turn
            state.category_text = category

        if not category_changed:
            if parsed.is_override:
                self._cancel_pending(state)
            else:
                self._finish_pending(state)

        if parsed.is_override and not category_changed:
            old_terms = set(_terms(parsed.override_old_fragment))
            if not old_terms:
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

            if old_terms:
                for old_term in old_terms:
                    state.category_text = re.sub(
                        rf"\b{re.escape(old_term)}\b",
                        " ",
                        state.category_text,
                        flags=re.IGNORECASE,
                    )
                state.category_text = re.sub(
                    r"\s+", " ", state.category_text
                ).strip(" .,:;-")
                state.active_terms = [
                    term for term in state.active_terms if term not in old_terms
                ]
                for source_turn, source_terms in list(state.turn_terms.items()):
                    kept_terms = [term for term in source_terms if term not in old_terms]
                    if kept_terms:
                        state.turn_terms[source_turn] = kept_terms
                        remaining_attributes = _classify_message(" ".join(kept_terms))
                        if remaining_attributes:
                            state.turn_attributes[source_turn] = remaining_attributes
                        else:
                            state.turn_attributes.pop(source_turn, None)
                    else:
                        state.turn_terms.pop(source_turn, None)
                        state.turn_attributes.pop(source_turn, None)
                state.known_attributes = (
                    set().union(*state.turn_attributes.values())
                    if state.turn_attributes
                    else set()
                )
                old_attributes = _classify_message(parsed.override_old_fragment)
                replacement_attributes = set(parsed.attributes)
                reopened_attributes = old_attributes - replacement_attributes
                if reopened_attributes:
                    state.asked_attributes = [
                        attribute
                        for attribute in state.asked_attributes
                        if attribute not in reopened_attributes
                    ]
                    state.exhausted_attributes.difference_update(reopened_attributes)

        if parsed.is_no_preference and parsed.no_preference_attribute:
            attribute = parsed.no_preference_attribute
            state.exhausted_attributes.add(attribute)
            if self.question_policy == "fast":
                state.prefer_other_next = True
            elif self.question_policy == "boundary" and not parsed.no_preference_is_additional:
                state.prefer_other_next = True

        negative_terms = set(parsed.negative_terms)
        if negative_terms:
            state.turn_excluded_terms[turn] = negative_terms
            state.excluded_terms.update(negative_terms)

        if parsed.is_retry or (
            parsed.is_no_preference and not parsed.content_wrapped
        ):
            return parsed

        fragment = parsed.content_fragment
        added_terms: list[str] = []
        for term in _terms(fragment):
            if term not in state.excluded_terms and term not in state.active_terms:
                state.active_terms.append(term)
                added_terms.append(term)
        if added_terms:
            state.turn_terms[turn] = added_terms
        attributes = set(parsed.attributes)
        if attributes:
            state.turn_attributes[turn] = attributes
            state.known_attributes.update(attributes)
        return parsed

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
        if state.pending_attribute is not None:
            blocked.add(state.pending_attribute)
        if state.prefer_other_next and "other" not in blocked:
            state.prefer_other_next = False
            state.pending_attribute = "other"
            state.pending_turn = turn
            return "other"
        for attribute in QUESTION_ORDER:
            if attribute not in blocked:
                state.pending_attribute = attribute
                state.pending_turn = turn
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
            "pending_attribute": state.pending_attribute,
            "pending_turn": state.pending_turn,
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
            parsed = self._update_state(state, user_message, turn)
            query_terms = self._query_terms(state)
            broad_expression = self._fts_expression(query_terms)
            strict_expression = (
                self._strict_fts_expression(query_terms) if len(query_terms) >= 2 else ""
            )
            self._trace(session_id, turn, "parse", {
                "input": user_message,
                "events": {
                    "category": parsed.category_text or None,
                    "content_fragment": parsed.content_fragment or None,
                    "content_wrapped": parsed.content_wrapped,
                    "override": parsed.is_override,
                    "override_old_fragment": parsed.override_old_fragment or None,
                    "retry": parsed.is_retry,
                    "no_preference": parsed.is_no_preference,
                    "no_preference_attribute": parsed.no_preference_attribute,
                },
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
                "pending_turn": state.pending_turn,
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
