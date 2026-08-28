from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_p9_evidence import build_sidecar
from starter.p9_evidence import (
    REGISTRY_SHA256,
    SCHEMA_VERSION,
    SEMANTICS_SHA256,
    CompactEvidenceStore,
    SLOT_ORDER,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(identifier: str, *, color: str | None = None) -> dict[str, object]:
    details = {"Material": "cotton", "Department": "Women"}
    if color is not None:
        details["Color"] = color
    return {
        "parent_asin": identifier,
        "title": "Everyday dress",
        "categories": ["Women", "Dresses"],
        "features": ["casual cotton"],
        "details": details,
        "store": "Example",
        "description": "red" if color is None else "blue",
    }


class P9EvidenceBuilderTests(unittest.TestCase):
    def _catalog(self, root: Path, products: list[dict[str, object]]) -> Path:
        path = root / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        return path

    def _build(self, root: Path, catalog: Path, stem: str = "evidence") -> tuple[Path, Path, dict]:
        sidecar = root / f"{stem}.sqlite"
        metadata = root / f"{stem}.metadata.json"
        payload = build_sidecar(
            catalog,
            sidecar,
            metadata,
            expected_catalog_sha256=None,
            expected_catalog_rows=None,
        )
        return sidecar, metadata, payload

    def _open_synthetic_store(
        self,
        sidecar: Path,
        catalog: Path,
        rows: int,
    ) -> CompactEvidenceStore:
        with (
            patch("starter.p9_evidence.OFFICIAL_CATALOG_SHA256", _sha256(catalog)),
            patch("starter.p9_evidence.OFFICIAL_CATALOG_ROWS", rows),
        ):
            return CompactEvidenceStore(sidecar)

    def test_sidecar_binds_one_based_catalog_rowids_to_asins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = [_product("A", color="red"), _product("B"), _product("C", color="blue")]
            catalog = self._catalog(root, products)
            sidecar, metadata_path, metadata = self._build(root, catalog)
            connection = sqlite3.connect(sidecar)
            try:
                rows = connection.execute(
                    "SELECT catalog_rowid, parent_asin FROM evidence ORDER BY catalog_rowid"
                ).fetchall()
            finally:
                connection.close()
            store = self._open_synthetic_store(sidecar, catalog, 3)
            try:
                evidence = store.fetch([(1, "A"), (2, "B"), (3, "C")])
            finally:
                store.close()

            self.assertEqual(rows, [(1, "A"), (2, "B"), (3, "C")])
            self.assertEqual(set(evidence), {"A", "B", "C"})
            self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
            self.assertEqual(metadata["catalog"]["rows"], 3)
            self.assertEqual(metadata["evidence"]["registry_sha256"], REGISTRY_SHA256)
            self.assertEqual(metadata["evidence"]["semantics_sha256"], SEMANTICS_SHA256)
            self.assertEqual(metadata["evidence"]["slot_order"], list(SLOT_ORDER))
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8")), metadata)

    def test_build_is_byte_deterministic_and_metadata_is_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root, [_product("PRIVATE-A", color="red"), _product("PRIVATE-B")])
            first, first_metadata, first_payload = self._build(root, catalog, "first")
            second, second_metadata, second_payload = self._build(root, catalog, "second")

            self.assertEqual(_sha256(first), _sha256(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_payload, second_payload)
            metadata_text = first_metadata.read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE-A", metadata_text)
            self.assertNotIn("PRIVATE-B", metadata_text)
            self.assertEqual(first_metadata.read_bytes(), second_metadata.read_bytes())

    def test_binding_mismatch_and_missing_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root, [_product("A"), _product("B")])
            sidecar, _metadata, _payload = self._build(root, catalog)
            store = self._open_synthetic_store(sidecar, catalog, 2)
            try:
                with self.assertRaisesRegex(ValueError, "binding"):
                    store.fetch([(1, "B")])
                with self.assertRaisesRegex(ValueError, "missing"):
                    store.fetch([(3, "C")])
            finally:
                store.close()

    def test_runtime_store_rejects_identity_structure_and_mask_tampering(self) -> None:
        mutations = {
            "catalog identity": (
                "UPDATE metadata SET value = ? WHERE key = 'catalog_sha256'",
                ("0" * 64,),
                "catalog_sha256",
            ),
            "non-contiguous rowids": (
                "UPDATE evidence SET catalog_rowid = 4 WHERE catalog_rowid = 2",
                (),
                "continuously",
            ),
            "duplicate ASIN": (
                "UPDATE evidence SET parent_asin = 'A' WHERE catalog_rowid = 2",
                (),
                "unique",
            ),
            "unknown mask bit": (
                "UPDATE evidence SET audience_mask = ? WHERE catalog_rowid = 1",
                (1 << 30,),
                "outside the frozen registry",
            ),
        }
        for index, (name, (statement, parameters, message)) in enumerate(mutations.items()):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                catalog = self._catalog(
                    root,
                    [_product("A"), _product("B"), _product("C")],
                )
                sidecar, _metadata, _payload = self._build(
                    root, catalog, f"tamper-{index}"
                )
                connection = sqlite3.connect(sidecar)
                try:
                    connection.execute(statement, parameters)
                    connection.commit()
                finally:
                    connection.close()
                with self.assertRaisesRegex(ValueError, message):
                    self._open_synthetic_store(sidecar, catalog, 3)

    def test_runtime_store_requires_catalog_rowid_as_the_primary_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root, [_product("A"), _product("B")])
            sidecar, _metadata, _payload = self._build(root, catalog)
            connection = sqlite3.connect(sidecar)
            try:
                connection.execute("ALTER TABLE evidence RENAME TO original_evidence")
                mask_columns = ", ".join(
                    f"{slot}_mask INTEGER NOT NULL" for slot in SLOT_ORDER
                )
                connection.execute(
                    "CREATE TABLE evidence("
                    "catalog_rowid INTEGER NOT NULL, parent_asin TEXT NOT NULL, "
                    f"{mask_columns})"
                )
                columns = ", ".join(
                    ("catalog_rowid", "parent_asin", *(f"{slot}_mask" for slot in SLOT_ORDER))
                )
                connection.execute(
                    f"INSERT INTO evidence({columns}) SELECT {columns} FROM original_evidence"
                )
                connection.execute("DROP TABLE original_evidence")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "primary key"):
                self._open_synthetic_store(sidecar, catalog, 2)

    def test_frozen_identity_and_row_count_are_checked_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root, [_product("A")])
            sidecar = root / "evidence.sqlite"
            metadata = root / "evidence.metadata.json"
            sidecar.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_sidecar(
                    catalog,
                    sidecar,
                    metadata,
                    expected_catalog_sha256="0" * 64,
                    expected_catalog_rows=1,
                )
            self.assertEqual(sidecar.read_bytes(), b"preserve")
            with self.assertRaisesRegex(ValueError, "row count"):
                build_sidecar(
                    catalog,
                    sidecar,
                    metadata,
                    expected_catalog_sha256=_sha256(catalog),
                    expected_catalog_rows=2,
                )
            self.assertEqual(sidecar.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
