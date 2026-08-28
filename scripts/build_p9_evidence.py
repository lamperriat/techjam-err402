"""Build the frozen, catalog-only P9 compact evidence sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from starter.p9_evidence import (
    OFFICIAL_CATALOG_ROWS,
    OFFICIAL_CATALOG_SHA256,
    REGISTRY_SHA256,
    SCHEMA_VERSION,
    SEMANTICS_SHA256,
    SLOT_ORDER,
    SQL_MASK_COLUMNS,
    CompactEvidenceStore,
    masks_from_catalog_product,
)


EXPECTED_CATALOG_SHA256 = OFFICIAL_CATALOG_SHA256
EXPECTED_CATALOG_ROWS = OFFICIAL_CATALOG_ROWS
DEFAULT_CATALOG = Path("data/catalog.jsonl")
DEFAULT_SIDECAR = Path("experiments/p9_negative_evidence.sqlite")
DEFAULT_METADATA = Path("experiments/p9_negative_evidence.metadata.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)


def _create_database(
    path: Path,
    catalog_path: Path,
    *,
    catalog_sha256: str,
) -> tuple[int, dict[str, int]]:
    connection = sqlite3.connect(path)
    nonzero_counts: Counter[str] = Counter()
    seen_asins: set[str] = set()
    row_count = 0
    try:
        connection.execute("PRAGMA page_size=4096")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("PRAGMA auto_vacuum=NONE")
        mask_columns = ", ".join(
            f"{column} INTEGER NOT NULL CHECK({column} >= 0)"
            for column in SQL_MASK_COLUMNS
        )
        connection.execute(
            "CREATE TABLE evidence("
            "catalog_rowid INTEGER PRIMARY KEY, "
            "parent_asin TEXT NOT NULL, "
            f"{mask_columns})"
        )
        connection.execute(
            "CREATE TABLE metadata("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        insert_columns = ", ".join(("catalog_rowid", "parent_asin", *SQL_MASK_COLUMNS))
        placeholders = ",".join("?" for _ in range(2 + len(SQL_MASK_COLUMNS)))
        insert_sql = f"INSERT INTO evidence({insert_columns}) VALUES ({placeholders})"
        batch: list[tuple[object, ...]] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid catalog JSON at line {line_number}") from error
                if not isinstance(product, dict):
                    raise ValueError(f"catalog line {line_number} must be a JSON object")
                asin = str(product.get("parent_asin") or "")
                if not asin:
                    raise ValueError(f"catalog line {line_number} has no parent_asin")
                if asin in seen_asins:
                    raise ValueError(f"duplicate catalog parent_asin: {asin}")
                seen_asins.add(asin)
                row_count += 1
                masks = masks_from_catalog_product(product)
                for slot, mask in zip(SLOT_ORDER, masks):
                    nonzero_counts[slot] += int(bool(mask))
                batch.append((row_count, asin, *masks))
                if len(batch) >= 1_000:
                    connection.executemany(insert_sql, batch)
                    batch.clear()
        if batch:
            connection.executemany(insert_sql, batch)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "registry_sha256": REGISTRY_SHA256,
            "semantics_sha256": SEMANTICS_SHA256,
            "slot_order": ",".join(SLOT_ORDER),
            "catalog_sha256": catalog_sha256,
            "catalog_rows": str(row_count),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return row_count, dict(nonzero_counts)


def build_sidecar(
    catalog_path: str | Path,
    sidecar_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_catalog_sha256: str | None = EXPECTED_CATALOG_SHA256,
    expected_catalog_rows: int | None = EXPECTED_CATALOG_ROWS,
) -> dict[str, Any]:
    """Build atomically and return aggregate, catalog-derived metadata."""

    catalog = Path(catalog_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"catalog does not exist: {catalog}")
    if sidecar == metadata_file:
        raise ValueError("sidecar and metadata paths must be different")
    catalog_sha256 = _sha256_file(catalog)
    if (
        expected_catalog_sha256 is not None
        and catalog_sha256 != expected_catalog_sha256.lower()
    ):
        raise ValueError("catalog SHA-256 does not match the frozen identity")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".p9-evidence-", dir=sidecar.parent)
    )
    temporary_sidecar = temporary_directory / sidecar.name
    temporary_metadata = temporary_directory / metadata_file.name
    try:
        row_count, nonzero_counts = _create_database(
            temporary_sidecar,
            catalog,
            catalog_sha256=catalog_sha256,
        )
        if expected_catalog_rows is not None and row_count != expected_catalog_rows:
            raise ValueError(
                f"catalog row count {row_count} does not match {expected_catalog_rows}"
            )
        if (
            isinstance(expected_catalog_sha256, str)
            and expected_catalog_sha256.lower() == OFFICIAL_CATALOG_SHA256
            and expected_catalog_rows == OFFICIAL_CATALOG_ROWS
        ):
            store = CompactEvidenceStore(temporary_sidecar)
            store.close()
        sidecar_sha256 = _sha256_file(temporary_sidecar)
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "catalog": {
                "bytes": catalog.stat().st_size,
                "rows": row_count,
                "sha256": catalog_sha256,
            },
            "evidence": {
                "bytes": temporary_sidecar.stat().st_size,
                "sha256": sidecar_sha256,
                "registry_sha256": REGISTRY_SHA256,
                "semantics_sha256": SEMANTICS_SHA256,
                "slot_order": list(SLOT_ORDER),
                "nonzero_rows_by_slot": {
                    slot: nonzero_counts.get(slot, 0) for slot in SLOT_ORDER
                },
            },
            "target_blind": True,
            "label_free": True,
        }
        _write_json(temporary_metadata, metadata)
        os.replace(temporary_sidecar, sidecar)
        os.replace(temporary_metadata, metadata_file)
        return metadata
    finally:
        for child in temporary_directory.iterdir():
            child.unlink()
        temporary_directory.rmdir()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata = build_sidecar(args.catalog, args.output, args.metadata)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
