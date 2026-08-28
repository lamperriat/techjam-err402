"""Build the catalog-only SQLite sidecar used by the P11 Top-10 scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from starter.agent import _terms
from starter.attributes import (
    AUDIENCE,
    CATEGORIES,
    GENERIC_CATEGORIES,
    SLOT_VOCABULARIES,
    build_product_attribute_view,
    normalize_value,
    product_slot,
)
from starter.p9_evidence import masks_from_catalog_product
from starter.p11_features import (
    FIELD_GROUPS,
    FEATURE_ENCODING,
    REGISTRY_SHA256,
    SCHEMA_VERSION,
    SEMANTICS_SHA256,
    SQL_NEGATIVE_MASK_COLUMNS,
    P11FeatureStore,
    encode_feature_blob,
)


OFFICIAL_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
OFFICIAL_CATALOG_ROWS = 50_000
DEFAULT_CATALOG = Path("data/catalog.jsonl")
DEFAULT_SIDECAR = Path("experiments/p11_features.sqlite")
DEFAULT_METADATA = Path("experiments/p11_features.metadata.json")
MAX_SIDECAR_BYTES = 33_554_432
POSITIVE_SLOTS = ("category", *SLOT_VOCABULARIES, "brand")
OBSERVED_SOURCES = frozenset({"categories", "features", "store", "details"})
_CANONICAL_VALUES = {
    "category": tuple(sorted(set(CATEGORIES.values()))),
    **{
        slot: tuple(sorted(set(vocabulary.values())))
        for slot, vocabulary in SLOT_VOCABULARIES.items()
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)


def _flatten(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key in sorted(value, key=lambda item: normalize_value(item)):
            for item in _flatten(value[key]):
                flattened.append(f"{key} {item}")
        return flattened
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _flatten(child)]
    if isinstance(value, (set, frozenset)):
        return sorted((str(item) for item in value), key=normalize_value)
    return [str(value)]


def _field_values(product: Mapping[str, object]) -> tuple[list[str], list[str], list[str]]:
    return (
        [*_flatten(product.get("title")), *_flatten(product.get("categories"))],
        [*_flatten(product.get("features")), *_flatten(product.get("details"))],
        [*_flatten(product.get("description")), *_flatten(product.get("store"))],
    )


def _field_payload(values: Iterable[str]) -> tuple[tuple[str, ...], frozenset[str]]:
    sequences = tuple(
        dict.fromkeys(
            sequence
            for raw in values
            if (sequence := " ".join(_terms(str(raw))))
        )
    )
    tokens = frozenset(token for sequence in sequences for token in sequence.split())
    return sequences, tokens


def _catalog_subtypes(product: Mapping[str, object]) -> tuple[str, ...]:
    generic = set(GENERIC_CATEGORIES) | set(AUDIENCE) | set(AUDIENCE.values())
    values: list[str] = []
    for raw in _flatten(product.get("categories")):
        normalized = normalize_value(raw)
        if not normalized or normalized in generic or len(normalized.split()) > 4:
            continue
        canonical = CATEGORIES.get(normalized, normalized)
        if canonical not in values:
            values.append(canonical)
    return tuple(values)


def _description_inferred_values(product: Mapping[str, object]) -> set[str]:
    normalized = normalize_value(" ".join(_flatten(product.get("description"))))
    if not normalized:
        return set()
    padded = f" {normalized} "
    return {
        f"{slot}={value}"
        for slot, values in _CANONICAL_VALUES.items()
        for value in values
        if f" {value} " in padded
    }


def _attribute_evidence(
    product: Mapping[str, object],
    observed_subtypes: tuple[str, ...],
) -> tuple[set[str], set[str], set[str]]:
    view = build_product_attribute_view(product)
    observed: set[str] = {f"category={value}" for value in observed_subtypes}
    inferred: set[str] = set()
    inferred_subtypes: set[str] = set()
    for slot in POSITIVE_SLOTS:
        for item in product_slot(view, slot):
            key = f"{slot}={item.value}"
            reliable = (
                item.confidence >= 0.90
                and (
                    item.source in OBSERVED_SOURCES
                    or item.source.startswith("details.")
                )
            )
            if reliable:
                observed.add(key)
            elif item.source == "title":
                inferred.add(key)
                if slot == "category":
                    inferred_subtypes.add(item.value)
    inferred.update(_description_inferred_values(product))
    inferred.difference_update(observed)
    inferred_subtypes.difference_update(observed_subtypes)
    return observed, inferred, inferred_subtypes


def _rating(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 5.0 else None


def _rating_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _percentiles(values: list[tuple[int, float]]) -> dict[int, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {values[0][0]: 0.5}
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    result: dict[int, float] = {}
    index = 0
    denominator = len(ordered) - 1
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        percentile = ((index + end - 1) / 2.0) / denominator
        for rowid, _ in ordered[index:end]:
            result[rowid] = percentile
        index = end
    return result


def _quality_priors(
    groups: Mapping[str, list[tuple[int, float | None, int]]],
) -> tuple[dict[int, float], dict[int, float]]:
    bayesian_percentiles: dict[int, float] = {}
    popularity_percentiles: dict[int, float] = {}
    all_ratings = [
        rating
        for records in groups.values()
        for _rowid, rating, _count in records
        if rating is not None
    ]
    global_mean = statistics.fmean(all_ratings) if all_ratings else 3.0
    for records in groups.values():
        ratings = [rating for _rowid, rating, _count in records if rating is not None]
        counts = [count for _rowid, _rating_value, count in records if count > 0]
        prior_mean = statistics.fmean(ratings) if ratings else global_mean
        prior_count = max(10.0, float(statistics.median(counts)) if counts else 10.0)
        bayesian: list[tuple[int, float]] = []
        popularity: list[tuple[int, float]] = []
        for rowid, rating, count in records:
            observed_rating = prior_mean if rating is None else rating
            bayes = (
                count * observed_rating + prior_count * prior_mean
            ) / (count + prior_count)
            bayesian.append((rowid, bayes))
            popularity.append((rowid, math.log1p(count)))
        bayesian_percentiles.update(_percentiles(bayesian))
        popularity_percentiles.update(_percentiles(popularity))
    return bayesian_percentiles, popularity_percentiles


def _create_database(
    path: Path,
    catalog_path: Path,
    *,
    catalog_sha256: str,
) -> tuple[int, dict[str, int]]:
    connection = sqlite3.connect(path)
    term_df: Counter[str] = Counter()
    subtype_df: Counter[str] = Counter()
    rating_groups: dict[str, list[tuple[int, float | None, int]]] = defaultdict(list)
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
            for column in SQL_NEGATIVE_MASK_COLUMNS
        )
        connection.execute(
            "CREATE TABLE evidence("
            "catalog_rowid INTEGER PRIMARY KEY, parent_asin TEXT NOT NULL UNIQUE, "
            "feature_blob BLOB NOT NULL, "
            f"{mask_columns}, "
            "bayesian_rating_percentile REAL NOT NULL CHECK("
            "bayesian_rating_percentile BETWEEN 0.0 AND 1.0), "
            "popularity_percentile REAL NOT NULL CHECK("
            "popularity_percentile BETWEEN 0.0 AND 1.0))"
        )
        connection.execute(
            "CREATE TABLE term_stats("
            "term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL "
            "CHECK(document_frequency > 0)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE subtype_stats("
            "subtype TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE metadata("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )

        column_count = 3 + len(SQL_NEGATIVE_MASK_COLUMNS) + 2
        insert_sql = "INSERT INTO evidence VALUES (" + ",".join(
            "?" for _ in range(column_count)
        ) + ")"
        batch: list[tuple[object, ...]] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid catalog JSON at line {line_number}") from error
                if not isinstance(product, Mapping):
                    raise ValueError(f"catalog line {line_number} must be an object")
                asin = str(product.get("parent_asin") or "")
                if not asin:
                    raise ValueError(f"catalog line {line_number} has no parent_asin")
                if asin in seen_asins:
                    raise ValueError(f"duplicate catalog parent_asin: {asin}")
                seen_asins.add(asin)
                row_count += 1

                fields = tuple(_field_payload(values) for values in _field_values(product))
                document_terms = set().union(*(field[1] for field in fields))
                term_df.update(document_terms)
                observed_subtypes = _catalog_subtypes(product)
                subtype_df.update(set(observed_subtypes))
                observed, inferred, inferred_subtypes = _attribute_evidence(
                    product,
                    observed_subtypes,
                )
                negative_masks = masks_from_catalog_product(product)
                group = observed_subtypes[-1] if observed_subtypes else "__global__"
                rating_groups[group].append((
                    row_count,
                    _rating(product.get("average_rating")),
                    _rating_count(product.get("rating_number")),
                ))
                batch.append((
                    row_count,
                    asin,
                    sqlite3.Binary(encode_feature_blob(
                        (fields[0][0], fields[1][0], fields[2][0]),
                        observed,
                        inferred,
                        observed_subtypes,
                        inferred_subtypes,
                    )),
                    *negative_masks,
                    0.5,
                    0.5,
                ))
                if len(batch) >= 500:
                    connection.executemany(insert_sql, batch)
                    batch.clear()
        if batch:
            connection.executemany(insert_sql, batch)

        term_rows = (
            (term, df)
            for term, df in sorted(term_df.items())
        )
        connection.executemany(
            "INSERT INTO term_stats(term, document_frequency) VALUES (?, ?)",
            term_rows,
        )
        connection.executemany(
            "INSERT INTO subtype_stats(subtype, document_frequency) VALUES (?, ?)",
            sorted(subtype_df.items()),
        )
        bayesian, popularity = _quality_priors(rating_groups)
        connection.executemany(
            "UPDATE evidence SET bayesian_rating_percentile=?, "
            "popularity_percentile=? WHERE catalog_rowid=?",
            (
                (bayesian.get(rowid, 0.5), popularity.get(rowid, 0.5), rowid)
                for rowid in range(1, row_count + 1)
            ),
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "registry_sha256": REGISTRY_SHA256,
            "semantics_sha256": SEMANTICS_SHA256,
            "negative_slot_order": ",".join(
                column.removeprefix("negative_").removesuffix("_mask")
                for column in SQL_NEGATIVE_MASK_COLUMNS
            ),
            "catalog_sha256": catalog_sha256,
            "catalog_rows": str(row_count),
            "feature_encoding": FEATURE_ENCODING,
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return row_count, {
        "term_count": len(term_df),
        "subtype_count": len(subtype_df),
    }


def build_sidecar(
    catalog_path: str | Path,
    sidecar_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_catalog_sha256: str | None = OFFICIAL_CATALOG_SHA256,
    expected_catalog_rows: int | None = OFFICIAL_CATALOG_ROWS,
) -> dict[str, Any]:
    """Build the deterministic P11 sidecar atomically from the catalog only."""

    catalog = Path(catalog_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"catalog does not exist: {catalog}")
    if sidecar == metadata_file:
        raise ValueError("sidecar and metadata paths must differ")
    if sidecar.exists() or metadata_file.exists():
        existing = sidecar if sidecar.exists() else metadata_file
        raise FileExistsError(f"P11 output already exists: {existing}")
    catalog_sha256 = _sha256_file(catalog)
    if (
        expected_catalog_sha256 is not None
        and catalog_sha256 != expected_catalog_sha256.lower()
    ):
        raise ValueError("catalog SHA-256 does not match the frozen identity")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".p11-sidecar-", dir=sidecar.parent)
    )
    temporary_sidecar = temporary_directory / sidecar.name
    temporary_metadata = temporary_directory / metadata_file.name
    try:
        row_count, counts = _create_database(
            temporary_sidecar,
            catalog,
            catalog_sha256=catalog_sha256,
        )
        if expected_catalog_rows is not None and row_count != expected_catalog_rows:
            raise ValueError(
                f"catalog row count {row_count} does not match {expected_catalog_rows}"
            )
        sidecar_bytes = temporary_sidecar.stat().st_size
        if sidecar_bytes > MAX_SIDECAR_BYTES:
            raise ValueError(
                f"P11 sidecar exceeds {MAX_SIDECAR_BYTES} bytes: {sidecar_bytes}"
            )
        store = P11FeatureStore(
            temporary_sidecar,
            expected_catalog_sha256=catalog_sha256,
            expected_catalog_rows=row_count,
        )
        store.close()
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "catalog": {
                "bytes": catalog.stat().st_size,
                "rows": row_count,
                "sha256": catalog_sha256,
            },
            "sidecar": {
                "bytes": sidecar_bytes,
                "sha256": _sha256_file(temporary_sidecar),
                "registry_sha256": REGISTRY_SHA256,
                "semantics_sha256": SEMANTICS_SHA256,
                **counts,
            },
            "field_groups": list(FIELD_GROUPS),
            "target_blind": True,
            "label_free": True,
        }
        _write_json(temporary_metadata, metadata)
        # Publish with exclusive hard links. A concurrently created output
        # fails closed; existing evidence or metadata is never replaced.
        os.link(temporary_sidecar, sidecar)
        try:
            os.link(temporary_metadata, metadata_file)
        except BaseException:
            sidecar.unlink()
            raise
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
