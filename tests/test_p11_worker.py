from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import p11_worker


class P11WorkerAssetTests(unittest.TestCase):
    def _catalog(self, directory: Path) -> tuple[Path, bytes]:
        payload = b'{"parent_asin":"A"}\n{"parent_asin":"B"}\n'
        path = directory / "catalog.jsonl"
        path.write_bytes(payload)
        return path, payload

    @staticmethod
    def _args(
        role: str,
        catalog: Path,
        sidecar: object,
        *,
        sidecar_bytes: int = 0,
        sidecar_sha256: str = "",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            role=role,
            nonce="0" * 32,
            catalog=catalog,
            sidecar=sidecar,
            sidecar_bytes=sidecar_bytes,
            sidecar_sha256=sidecar_sha256,
        )

    def _official_patches(self, payload: bytes):
        return (
            patch.object(p11_worker, "OFFICIAL_CATALOG_ROWS", 2),
            patch.object(
                p11_worker,
                "OFFICIAL_CATALOG_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ),
        )

    def test_baseline_and_control_validate_catalog_without_touching_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            catalog, payload = self._catalog(Path(raw_directory))
            rows_patch, hash_patch = self._official_patches(payload)
            with rows_patch, hash_patch:
                for role in (p11_worker.BASELINE_ID, p11_worker.CONTROL_ID):
                    report = p11_worker._validate_assets(
                        self._args(role, catalog, object())
                    )
                    self.assertTrue(report["catalog"]["verified_official"])
                    self.assertEqual(report["catalog"]["rows"], 2)
                    self.assertEqual(
                        report["sidecar"],
                        {
                            "required": False,
                            "opened_for_identity": False,
                            "verified": True,
                            "bytes": None,
                            "sha256": None,
                        },
                    )

    def test_active_and_shadow_require_exact_bounded_sidecar_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog, payload = self._catalog(directory)
            sidecar = directory / "features.sqlite"
            sidecar.write_bytes(b"catalog-only-sidecar")
            sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            rows_patch, hash_patch = self._official_patches(payload)
            with rows_patch, hash_patch:
                for role in (p11_worker.SHADOW_ID, p11_worker.ACTIVE_ID):
                    report = p11_worker._validate_assets(
                        self._args(
                            role,
                            catalog,
                            sidecar,
                            sidecar_bytes=sidecar.stat().st_size,
                            sidecar_sha256=sidecar_hash,
                        )
                    )
                    self.assertTrue(report["sidecar"]["opened_for_identity"])
                    self.assertTrue(report["sidecar"]["verified"])

                with self.assertRaisesRegex(
                    p11_worker.WorkerProtocolError,
                    "sidecar identity",
                ):
                    p11_worker._validate_assets(
                        self._args(
                            p11_worker.ACTIVE_ID,
                            catalog,
                            sidecar,
                            sidecar_bytes=sidecar.stat().st_size,
                            sidecar_sha256="0" * 64,
                        )
                    )
                with self.assertRaisesRegex(
                    p11_worker.WorkerProtocolError,
                    "sidecar identity",
                ):
                    p11_worker._validate_assets(
                        self._args(
                            p11_worker.ACTIVE_ID,
                            catalog,
                            sidecar,
                            sidecar_bytes=p11_worker.MAX_SIDECAR_BYTES + 1,
                            sidecar_sha256=sidecar_hash,
                        )
                    )

    def test_catalog_sha_and_nonblank_row_count_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            catalog, payload = self._catalog(directory)
            rows_patch, hash_patch = self._official_patches(payload)
            with rows_patch, hash_patch:
                catalog.write_bytes(payload + b"\n")
                with self.assertRaisesRegex(
                    p11_worker.WorkerProtocolError,
                    "catalog identity",
                ):
                    p11_worker._validate_assets(
                        self._args(p11_worker.BASELINE_ID, catalog, object())
                    )

    def test_control_agent_constructor_receives_no_sidecar_arguments(self) -> None:
        args = self._args(p11_worker.CONTROL_ID, Path("catalog.jsonl"), object())
        sentinel = object()
        with patch.object(
            p11_worker,
            "create_p11_agent",
            return_value=sentinel,
        ) as create:
            self.assertIs(p11_worker._build_agent(args), sentinel)
        create.assert_called_once_with(args.catalog, p11_worker.CONTROL_ID)


if __name__ == "__main__":
    unittest.main()
