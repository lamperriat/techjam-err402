"""Compact catalog evidence for the isolated P9 negative experiment.

The offline builder projects catalog rows exactly as ``Agent._build_index`` does,
then stores six reliable attribute masks.  Runtime code reads only the masks for
the current bounded candidate pool; it never materializes the catalog or product
attribute views.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from starter.attributes import (
    ProductAttributeView,
    SLOT_VOCABULARIES,
    build_product_attribute_view,
    product_slot,
)
from starter.p8_negative import (
    CANDIDATE_POOL,
    COMPATIBLE,
    EXPLICIT_VIOLATION,
    MIN_EVIDENCE_CONFIDENCE,
    UNKNOWN,
    ExecutableNegative,
)


SCHEMA_VERSION = "p9.compact-negative-evidence.v1"
OFFICIAL_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
OFFICIAL_CATALOG_ROWS = 50_000
SLOT_ORDER = (
    "audience",
    "material",
    "color",
    "closure",
    "style",
    "use_case",
)
RELIABLE_SOURCES = frozenset({"categories", "title", "features", "details", "store"})
SQL_MASK_COLUMNS = tuple(f"{slot}_mask" for slot in SLOT_ORDER)

VALUE_ORDER = {
    slot: tuple(sorted(set(SLOT_VOCABULARIES[slot].values())))
    for slot in SLOT_ORDER
}
VALUE_BITS = {
    slot: {value: 1 << index for index, value in enumerate(values)}
    for slot, values in VALUE_ORDER.items()
}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


REGISTRY_SHA256 = _canonical_sha256({
    "schema_version": SCHEMA_VERSION,
    "slot_order": SLOT_ORDER,
    "value_order": VALUE_ORDER,
})
SEMANTICS_SHA256 = _canonical_sha256({
    "catalog_projection": "starter.agent._text.v1",
    "minimum_confidence": MIN_EVIDENCE_CONFIDENCE,
    "reliable_sources": sorted(RELIABLE_SOURCES),
    "description_is_evidence": False,
    "partition_order": [COMPATIBLE, UNKNOWN, EXPLICIT_VIOLATION],
    "candidate_pool": CANDIDATE_POOL,
})


def _agent_text(value: object) -> str:
    """Mirror the catalog projection used by ``starter.agent.Agent`` exactly."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def runtime_attribute_view(product: Mapping[str, object]) -> ProductAttributeView:
    """Build the view P8 would reconstruct from Agent's in-memory FTS row."""

    return build_product_attribute_view({
        "parent_asin": str(product.get("parent_asin") or ""),
        "title": _agent_text(product.get("title")),
        "categories": _agent_text(product.get("categories")),
        "features": _agent_text(product.get("features")),
        "details": _agent_text(product.get("details")),
        "store": _agent_text(product.get("store")),
    })


def _is_reliable_source(source: str) -> bool:
    return source in RELIABLE_SOURCES or source.startswith("details.")


def masks_from_view(view: ProductAttributeView) -> tuple[int, ...]:
    """Encode precisely the catalog evidence visible to P8 classification."""

    masks: list[int] = []
    for slot in SLOT_ORDER:
        mask = 0
        bits = VALUE_BITS[slot]
        for item in product_slot(view, slot):
            if (
                _is_reliable_source(item.source)
                and item.confidence >= MIN_EVIDENCE_CONFIDENCE
            ):
                mask |= bits.get(item.value, 0)
        masks.append(mask)
    return tuple(masks)


def masks_from_catalog_product(product: Mapping[str, object]) -> tuple[int, ...]:
    return masks_from_view(runtime_attribute_view(product))


def compile_mask_constraints(
    constraints: Iterable[ExecutableNegative],
) -> tuple[tuple[int, int], ...]:
    """Resolve constraints to compact ``(slot index, value bit)`` pairs."""

    compiled: list[tuple[int, int]] = []
    slot_indexes = {slot: index for index, slot in enumerate(SLOT_ORDER)}
    for constraint in constraints:
        slot_index = slot_indexes.get(constraint.slot)
        if slot_index is None:
            raise ValueError(f"unsupported compact evidence slot: {constraint.slot}")
        compiled.append((slot_index, VALUE_BITS[constraint.slot].get(constraint.value, 0)))
    return tuple(compiled)


def classify_masks(
    masks: Sequence[int],
    compiled_constraints: Iterable[tuple[int, int]],
) -> str:
    """Return the P8-compatible C/U/V state without allocating attribute views."""

    if len(masks) != len(SLOT_ORDER):
        raise ValueError("compact evidence must contain exactly six masks")
    saw_unknown = False
    for slot_index, value_bit in compiled_constraints:
        slot_mask = masks[slot_index]
        if value_bit and slot_mask & value_bit:
            return EXPLICIT_VIOLATION
        if not slot_mask:
            saw_unknown = True
    return UNKNOWN if saw_unknown else COMPATIBLE


@dataclass(frozen=True, slots=True)
class CompactPartition:
    identifiers: tuple[str, ...]
    compatible_count: int
    unknown_count: int
    explicit_violation_count: int
    violation_fallback_count: int

    @property
    def counts(self) -> tuple[tuple[str, int], ...]:
        return (
            (COMPATIBLE, self.compatible_count),
            (UNKNOWN, self.unknown_count),
            (EXPLICIT_VIOLATION, self.explicit_violation_count),
        )


def stable_compact_partition(
    identifiers: Iterable[str],
    evidence: Mapping[str, Sequence[int]],
    constraints: Iterable[ExecutableNegative],
    *,
    top_k: int = 10,
    candidate_pool: int = CANDIDATE_POOL,
) -> CompactPartition:
    """Stable C -> U -> V partition with the same bounded fallback as P8."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (
        not isinstance(candidate_pool, int)
        or isinstance(candidate_pool, bool)
        or candidate_pool < top_k
    ):
        raise ValueError("candidate_pool must be an integer at least as large as top_k")
    ordered = tuple(dict.fromkeys(str(identifier) for identifier in identifiers))
    materialized = tuple(constraints)
    if not materialized:
        return CompactPartition(ordered, 0, len(ordered), 0, 0)

    compiled = compile_mask_constraints(materialized)
    compatible: list[str] = []
    unknown: list[str] = []
    violation: list[str] = []
    groups = {
        COMPATIBLE: compatible,
        UNKNOWN: unknown,
        EXPLICIT_VIOLATION: violation,
    }
    pool = ordered[:candidate_pool]
    empty_masks = (0,) * len(SLOT_ORDER)
    for identifier in pool:
        groups[classify_masks(evidence.get(identifier, empty_masks), compiled)].append(
            identifier
        )
    final = (*compatible, *unknown, *violation, *ordered[candidate_pool:])
    safe_count = len(compatible) + len(unknown)
    violation_fallback_count = max(0, min(top_k, len(final)) - min(top_k, safe_count))
    return CompactPartition(
        identifiers=tuple(final),
        compatible_count=len(compatible),
        unknown_count=len(unknown),
        explicit_violation_count=len(violation),
        violation_fallback_count=violation_fallback_count,
    )


class CompactEvidenceStore:
    """Read-only, bounded-query access to a frozen P9 evidence sidecar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"P9 evidence sidecar does not exist: {self.path}")
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.connection.execute("PRAGMA query_only=ON")
        try:
            self.metadata = self._validate()
        except Exception:
            self.connection.close()
            raise

    def _validate(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM metadata").fetchall()
        metadata = {str(key): str(value) for key, value in rows}
        required = {
            "schema_version": SCHEMA_VERSION,
            "registry_sha256": REGISTRY_SHA256,
            "semantics_sha256": SEMANTICS_SHA256,
            "slot_order": ",".join(SLOT_ORDER),
            "catalog_sha256": OFFICIAL_CATALOG_SHA256,
            "catalog_rows": str(OFFICIAL_CATALOG_ROWS),
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"P9 evidence {key} mismatch")

        columns = self.connection.execute("PRAGMA table_info(evidence)").fetchall()
        expected_columns = ("catalog_rowid", "parent_asin", *SQL_MASK_COLUMNS)
        if tuple(str(column[1]) for column in columns) != expected_columns:
            raise ValueError("P9 evidence table schema mismatch")
        if str(columns[0][2]).upper() != "INTEGER" or int(columns[0][5]) != 1:
            raise ValueError("P9 evidence catalog_rowid must be the integer primary key")

        row_count, minimum_rowid, maximum_rowid, distinct_asins = (
            self.connection.execute(
                "SELECT COUNT(*), MIN(catalog_rowid), MAX(catalog_rowid), "
                "COUNT(DISTINCT parent_asin) FROM evidence"
            ).fetchone()
        )
        if (
            row_count != OFFICIAL_CATALOG_ROWS
            or minimum_rowid != 1
            or maximum_rowid != OFFICIAL_CATALOG_ROWS
        ):
            raise ValueError("P9 evidence rowids must cover the official catalog continuously")
        if distinct_asins != OFFICIAL_CATALOG_ROWS:
            raise ValueError("P9 evidence parent_asin values must be unique")

        invalid_masks = " OR ".join(
            (
                f"typeof({column}) <> 'integer' OR {column} < 0 OR "
                f"({column} & {~((1 << len(VALUE_ORDER[slot])) - 1)}) <> 0"
            )
            for slot, column in zip(SLOT_ORDER, SQL_MASK_COLUMNS)
        )
        if self.connection.execute(
            f"SELECT 1 FROM evidence WHERE {invalid_masks} LIMIT 1"
        ).fetchone() is not None:
            raise ValueError("P9 evidence contains mask bits outside the frozen registry")
        return metadata

    def fetch(
        self,
        requested: Sequence[tuple[int, str]],
    ) -> dict[str, tuple[int, ...]]:
        """Fetch masks by rowid and verify every row remains bound to its ASIN."""

        if not requested:
            return {}
        rowids = [rowid for rowid, _identifier in requested]
        if len(rowids) != len(set(rowids)):
            raise ValueError("candidate rowids must be unique")
        placeholders = ",".join("?" for _ in rowids)
        columns = ", ".join(SQL_MASK_COLUMNS)
        rows = self.connection.execute(
            f"SELECT catalog_rowid, parent_asin, {columns} "
            f"FROM evidence WHERE catalog_rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        by_rowid = {int(row[0]): row for row in rows}
        if len(by_rowid) != len(requested):
            raise ValueError("P9 evidence is missing candidate rows")
        result: dict[str, tuple[int, ...]] = {}
        for rowid, expected_asin in requested:
            row = by_rowid[rowid]
            if str(row[1]) != expected_asin:
                raise ValueError("P9 evidence rowid-to-ASIN binding mismatch")
            masks = tuple(int(value) for value in row[2:])
            if len(masks) != len(SLOT_ORDER) or any(value < 0 for value in masks):
                raise ValueError("P9 evidence mask is invalid")
            result[expected_asin] = masks
        return result

    def close(self) -> None:
        self.connection.close()


__all__ = [
    "CompactEvidenceStore",
    "CompactPartition",
    "OFFICIAL_CATALOG_ROWS",
    "OFFICIAL_CATALOG_SHA256",
    "REGISTRY_SHA256",
    "RELIABLE_SOURCES",
    "SCHEMA_VERSION",
    "SEMANTICS_SHA256",
    "SLOT_ORDER",
    "SQL_MASK_COLUMNS",
    "VALUE_BITS",
    "VALUE_ORDER",
    "classify_masks",
    "compile_mask_constraints",
    "masks_from_catalog_product",
    "masks_from_view",
    "runtime_attribute_view",
    "stable_compact_partition",
]
