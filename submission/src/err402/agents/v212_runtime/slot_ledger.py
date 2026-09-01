"""Target-blind audit ledger for normalized conversation constraints."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable

from .attributes import ConversationConstraintView, normalize_value


SCHEMA_VERSION = "p3.slot-ledger.v1"
ACTIVE = "active"
SUPERSEDED = "superseded"
DELETED = "deleted"
_VALID_RETIRED_STATUSES = {SUPERSEDED, DELETED}
_HARD_REQUIREMENT_RE = re.compile(
    r"\b(?:must|need|required|requirement|cannot|can'?t|without)\b",
    re.IGNORECASE,
)
_SLOT_ALIASES = {"budget": "price", "feature_phrases": "feature"}


def normalize_slot(value: object) -> str:
    slot = normalize_value(value).replace(" ", "_")
    return _SLOT_ALIASES.get(slot, slot)


def _message_mentions(normalized_message: str, value: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized_message))


def _hard_requirement_mentions(message: str, value: str) -> bool:
    normalized_message = normalize_value(message)
    clauses = re.split(
        r"(?:[,.;!?]+|\b(?:but|however|although)\b)", normalized_message
    )
    return any(
        _HARD_REQUIREMENT_RE.search(clause) and _message_mentions(clause, value)
        for clause in clauses
    )


@dataclass(frozen=True, slots=True)
class SlotRecord:
    record_id: int
    slot: str
    value: str
    polarity: int
    hardness: str
    source: str
    confidence: float
    source_turn: int
    version: int
    status: str = ACTIVE
    ended_turn: int | None = None
    ended_version: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _desired_constraints(
    intent: ConversationConstraintView,
    suppressed_slots: set[str],
) -> dict[tuple[str, str, int], tuple[str, float]]:
    desired: dict[tuple[str, str, int], tuple[str, float]] = {}

    def add(slot: str, value: str, polarity: int, source: str, confidence: float) -> None:
        normalized_slot = normalize_slot(slot)
        normalized_value = normalize_value(value)
        if not normalized_slot or not normalized_value or normalized_slot in suppressed_slots:
            return
        opposite = (normalized_slot, normalized_value, -polarity)
        if polarity < 0:
            desired.pop(opposite, None)
        elif opposite in desired:
            return
        desired[(normalized_slot, normalized_value, polarity)] = (source, confidence)

    for value in intent.category_terms:
        add("category", value, 1, "category_text", 1.0)
    for constraint in intent.positive:
        add(
            constraint.slot,
            constraint.value,
            1,
            constraint.source,
            constraint.confidence,
        )
    for constraint in intent.negative:
        add(
            constraint.slot,
            constraint.value,
            -1,
            constraint.source,
            constraint.confidence,
        )
    for value in intent.exact_terms:
        add("feature", value, 1, "active_terms", 1.0)
    for value in intent.excluded_exact_terms:
        add("feature", value, -1, "excluded_terms", 1.0)
    return desired


@dataclass(slots=True)
class SlotLedger:
    records: list[SlotRecord] = field(default_factory=list)
    _next_record_id: int = 1

    def reconcile(
        self,
        intent: ConversationConstraintView,
        *,
        turn: int,
        version: int,
        message: str,
        suppressed_slots: Iterable[str] = (),
        retired_status: str = DELETED,
    ) -> None:
        if retired_status not in _VALID_RETIRED_STATUSES:
            raise ValueError("retired_status must be superseded or deleted")
        suppressed = {normalize_slot(slot) for slot in suppressed_slots}
        desired = _desired_constraints(intent, suppressed)
        active = {
            (record.slot, record.value, record.polarity): index
            for index, record in enumerate(self.records)
            if record.status == ACTIVE
        }

        refreshed: set[tuple[str, str, int]] = set()
        for key, index in active.items():
            if key in desired:
                record = self.records[index]
                if (
                    record.hardness != "hard"
                    and _hard_requirement_mentions(message, record.value)
                ):
                    self.records[index] = replace(
                        record,
                        status=SUPERSEDED,
                        ended_turn=turn,
                        ended_version=version,
                    )
                    refreshed.add(key)
                continue
            record = self.records[index]
            opposite_exists = (record.slot, record.value, -record.polarity) in desired
            status = (
                DELETED
                if record.slot in suppressed
                else SUPERSEDED if opposite_exists else retired_status
            )
            self.records[index] = replace(
                record,
                status=status,
                ended_turn=turn,
                ended_version=version,
            )

        for key in sorted(desired):
            if key in active and key not in refreshed:
                continue
            slot, value, polarity = key
            source, confidence = desired[key]
            self.records.append(SlotRecord(
                record_id=self._next_record_id,
                slot=slot,
                value=value,
                polarity=polarity,
                hardness=(
                    "hard"
                    if polarity < 0 or _hard_requirement_mentions(message, value)
                    else "soft"
                ),
                source=source,
                confidence=round(float(confidence), 3),
                source_turn=turn,
                version=version,
            ))
            self._next_record_id += 1

    def active_records(self) -> tuple[SlotRecord, ...]:
        return tuple(record for record in self.records if record.status == ACTIVE)

    def as_dict(self) -> dict[str, object]:
        active = self.active_records()
        return {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(self.records),
            "active_count": len(active),
            "active": [record.as_dict() for record in active],
            "records": [record.as_dict() for record in self.records],
        }
