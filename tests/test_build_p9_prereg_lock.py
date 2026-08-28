from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_p9_prereg_lock as lock_builder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(lock_builder._canonical_bytes(row) + b"\n" for row in rows))


def _row(identifier: str, target: str, scenario: str = "buying") -> dict:
    return {
        "sample_id": identifier,
        "scenario_type": scenario,
        "user_profile": {},
        "ground_truth": {"parent_asin": target},
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        for relative in lock_builder.SOURCE_PATHS.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen source\n", encoding="utf-8")
        runner = root / lock_builder.SOURCE_PATHS["evaluate_p9"]
        runner.write_text(
            f"REQUIRED_SOURCE_PATHS = {lock_builder.SOURCE_PATHS!r}\n",
            encoding="utf-8",
        )

        self.spec = root / "configs" / "p9_compact_negative_matrix.json"
        self.spec.parent.mkdir(parents=True, exist_ok=True)
        self.spec.write_text(
            json.dumps({
                "schema_version": lock_builder.SPEC_SCHEMA_VERSION,
                "mechanism": {
                    "evidence_asset": {
                        "schema_version": lock_builder.EVIDENCE_SCHEMA_VERSION,
                        "registry_sha256": "a" * 64,
                        "semantics_sha256": "b" * 64,
                        "catalog_only": True,
                        "label_free": True,
                        "maximum_bytes": lock_builder.EVIDENCE_MAX_BYTES,
                    },
                },
                "resource_limits": {
                    "evidence_asset_max_bytes": lock_builder.EVIDENCE_MAX_BYTES,
                    "bootstrap_ratio": 1.2,
                    "wall_ratio": 1.3,
                    "response_p95_ratio": 1.3,
                    "peak_rss_ratio": 1.2,
                    "rss_sample_ms": 10.0,
                    "bootstrap_timeout_seconds": 120.0,
                    "request_timeout_seconds": 30.0,
                    "finalize_timeout_seconds": 30.0,
                    "exit_timeout_seconds": 10.0,
                    "cumulative_worker_io_timeout_seconds": 180.0,
                },
            }) + "\n",
            encoding="utf-8",
        )
        self.catalog = root / "data" / "catalog.jsonl"
        _write_jsonl(self.catalog, [{"parent_asin": "CATALOG001"}])
        self.public = root / "data" / "public_set.jsonl"
        _write_jsonl(self.public, [_row("public", "TARGET001")])
        prior_names = ("p1", "p5", "p6", "p7", "p8_selection", "p8_confirmation")
        self.priors = {name: root / "experiments" / f"{name}.jsonl" for name in prior_names}
        for index, (name, path) in enumerate(self.priors.items(), start=2):
            _write_jsonl(path, [_row(name, f"TARGET{index:03d}")])

        scenarios = sorted(lock_builder.EXPECTED_SCENARIO_COUNTS)
        self.corpora = {
            "selection": root / "experiments" / "selection.jsonl",
            "confirmation": root / "experiments" / "confirmation.jsonl",
        }
        _write_jsonl(self.corpora["selection"], [
            _row(f"selection_{index}", f"TARGET{index + 20:03d}", scenario)
            for index, scenario in enumerate(scenarios)
        ])
        _write_jsonl(self.corpora["confirmation"], [
            _row(f"confirmation_{index}", f"TARGET{index + 30:03d}", scenario)
            for index, scenario in enumerate(scenarios)
        ])
        observations = {
            "released_public": lock_builder._inspect_conversation_rows(self.public),
            **{name: lock_builder._inspect_conversation_rows(path) for name, path in self.priors.items()},
        }
        split_observations = {
            name: lock_builder._inspect_conversation_rows(path) for name, path in self.corpora.items()
        }
        public_blob = lock_builder._git_blob_sha1(self.public)

        self.evidence = root / "experiments" / "evidence.sqlite"
        self.evidence.write_bytes(b"fixture-evidence")
        self.evidence_metadata = root / "experiments" / "evidence.metadata.json"
        evidence_metadata = {
            "schema_version": lock_builder.EVIDENCE_SCHEMA_VERSION,
            "catalog": {
                "bytes": self.catalog.stat().st_size,
                "rows": 1,
                "sha256": lock_builder._sha256_file(self.catalog),
            },
            "evidence": {
                "bytes": self.evidence.stat().st_size,
                "sha256": lock_builder._sha256_file(self.evidence),
                "registry_sha256": "a" * 64,
                "semantics_sha256": "b" * 64,
            },
            "target_blind": True,
            "label_free": True,
        }
        self.evidence_metadata.write_text(json.dumps(evidence_metadata) + "\n", encoding="utf-8")

        metadata_names = {
            "released_public": "released_public",
            "prior_p1_derived": "p1",
            "prior_p5_derived": "p5",
            "prior_p6_derived": "p6",
            "prior_p7_derived": "p7",
            "prior_p8_selection": "p8_selection",
            "prior_p8_confirmation": "p8_confirmation",
        }
        metadata = {
            "schema_version": lock_builder.METADATA_SCHEMA_VERSION,
            "catalog_source": {
                "sha256": lock_builder._sha256_file(self.catalog),
                "loaded_product_count": 1,
                "frozen_sha256_verified": True,
                "expected_count_verified": True,
            },
            "input_sources": {},
            "corpora": {},
            "outputs": {"metadata": {"path": "experiments/metadata.json"}},
            "exclusions": {
                "pairwise_input_target_overlaps": {"all": 0},
                "selected_target_overlaps": {"all": 0},
                "selection_confirmation_target_overlap": 0,
            },
        }
        for metadata_name, source_name in metadata_names.items():
            observed = observations[source_name]
            metadata["input_sources"][metadata_name] = {
                "sample_count": observed["row_count"],
                "unique_target_count": observed["row_count"],
                "canonical_samples_sha256": observed["canonical_samples_sha256"],
                "frozen_samples_sha256_verified": True,
            }
        metadata["input_sources"]["released_public"].update({
            "git_blob_sha1_lf": public_blob,
            "frozen_git_blob_verified": True,
        })
        for split, observed in split_observations.items():
            metadata["corpora"][split] = {
                "sample_count": observed["row_count"],
                "unique_target_count": observed["row_count"],
                "samples_sha256": observed["canonical_samples_sha256"],
                "scenario_counts": observed["scenario_counts"],
            }
            metadata["outputs"][split] = {
                "expected_frozen_samples_sha256": observed["canonical_samples_sha256"],
                "samples_file_sha256": observed["canonical_samples_sha256"],
                "frozen_samples_sha256_verified": True,
            }
        self.metadata = root / "experiments" / "metadata.json"
        self.metadata.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        self.output = root / "configs" / "p9_prereg_lock.json"
        self.paths = lock_builder.LockPaths(
            spec=self.spec,
            catalog=self.catalog,
            released_public=self.public,
            priors=self.priors,
            evidence=self.evidence,
            evidence_metadata=self.evidence_metadata,
            corpus_metadata=self.metadata,
            corpora=self.corpora,
            output=self.output,
        )
        canonical = {name: observed["canonical_samples_sha256"] for name, observed in observations.items()}
        canonical.update({name: observed["canonical_samples_sha256"] for name, observed in split_observations.items()})
        self.expectations = lock_builder.FrozenExpectations(
            catalog_sha256=lock_builder._sha256_file(self.catalog),
            catalog_rows=1,
            public_git_blob_sha1=public_blob,
            public_rows=1,
            prior_rows=1,
            split_rows=4,
            evidence_bytes=self.evidence.stat().st_size,
            evidence_sha256=lock_builder._sha256_file(self.evidence),
            evidence_metadata_sha256=lock_builder._sha256_file(self.evidence_metadata),
            registry_sha256="a" * 64,
            semantics_sha256="b" * 64,
            canonical_sha256=canonical,
            scenario_counts={name: 1 for name in scenarios},
        )


class BuildP9PreregLockTests(unittest.TestCase):
    def test_builds_strict_target_free_lock_and_binds_p8_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            lock = lock_builder.build_prereg_lock(
                project_root=fixture.root,
                paths=fixture.paths,
                expectations=fixture.expectations,
                enforce_git=False,
                require_defaults=False,
            )
        self.assertEqual(set(lock), {
            "schema_version", "source", "spec", "catalog", "released_public", "priors",
            "evidence", "evidence_metadata", "corpus_metadata", "corpora",
            "source_target_scan",
        })
        self.assertEqual(
            set(lock["priors"]),
            {"p1", "p5", "p6", "p7", "p8_selection", "p8_confirmation"},
        )
        self.assertEqual(lock["evidence"]["bytes"], fixture.expectations.evidence_bytes)
        self.assertEqual(lock["source_target_scan"]["match_count"], 0)
        self.assertTrue(lock["source_target_scan"]["passed"])
        encoded = json.dumps(lock)
        for identifier in ("TARGET001", "TARGET002", "TARGET007", "TARGET020", "TARGET030"):
            self.assertNotIn(identifier, encoded)

    def test_live_source_registry_matches_runner_and_includes_builder(self) -> None:
        required = lock_builder._required_paths_from_runner(
            lock_builder.PROJECT_ROOT / "scripts" / "evaluate_p9.py"
        )
        self.assertEqual(required, lock_builder.SOURCE_PATHS)
        self.assertEqual(required["lock_builder"], "scripts/build_p9_prereg_lock.py")

    def test_clean_pushed_revision_is_strict(self) -> None:
        head = "a" * 40

        def successful(_root: Path, *arguments: str, binary: bool = False):
            return {
                ("branch", "--show-current"): "pre",
                ("rev-parse", "HEAD"): head,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("remote", "get-url", "origin"): lock_builder.EXPECTED_ORIGIN_URL,
                ("ls-remote", "--heads", "origin", "refs/heads/pre"): (
                    f"{head}\trefs/heads/pre"
                ),
            }[arguments]

        with patch.object(lock_builder, "_git", side_effect=successful):
            self.assertEqual(
                lock_builder.capture_pushed_clean_revision(Path(".")),
                {
                    "git_commit": head,
                    "git_branch": "pre",
                    "remote_proof": {
                        "remote": "origin",
                        "head_ref": "refs/heads/pre",
                        "advertised_head": head,
                        "url_sha256": lock_builder.EXPECTED_ORIGIN_URL_SHA256,
                        "verified": True,
                    },
                },
            )
        for expected, override in {
            "named": {("branch", "--show-current"): ""},
            "clean": {("status", "--porcelain=v1", "--untracked-files=all"): " M file"},
            "origin": {
                ("ls-remote", "--heads", "origin", "refs/heads/pre"): (
                    f"{'b' * 40}\trefs/heads/pre"
                )
            },
            "official HTTPS": {
                ("remote", "get-url", "origin"): "file:///tmp/fake.git"
            },
        }.items():
            def failing(_root: Path, *arguments: str, binary: bool = False, values=override):
                defaults = {
                    ("branch", "--show-current"): "pre",
                    ("rev-parse", "HEAD"): head,
                    ("status", "--porcelain=v1", "--untracked-files=all"): "",
                    ("remote", "get-url", "origin"): lock_builder.EXPECTED_ORIGIN_URL,
                    ("ls-remote", "--heads", "origin", "refs/heads/pre"): (
                        f"{head}\trefs/heads/pre"
                    ),
                }
                return values.get(arguments, defaults[arguments])

            with self.subTest(expected=expected), patch.object(lock_builder, "_git", side_effect=failing):
                with self.assertRaisesRegex(lock_builder.PreregLockError, expected):
                    lock_builder.capture_pushed_clean_revision(Path("."))

    def test_source_scan_rejects_hardcoded_locked_identifier_without_echoing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            source = fixture.root / lock_builder.SOURCE_PATHS["p9_worker"]
            source.write_text("FROZEN_VALUE = 'TARGET001'\n", encoding="utf-8")
            with self.assertRaisesRegex(
                lock_builder.PreregLockError, "hardcodes a locked product identifier"
            ) as caught:
                lock_builder.build_prereg_lock(
                    project_root=fixture.root,
                    paths=fixture.paths,
                    expectations=fixture.expectations,
                    enforce_git=False,
                    require_defaults=False,
                )
            self.assertNotIn("TARGET001", str(caught.exception))

    def test_tracked_source_rejects_working_vs_head_blob_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"working\n")

            def fake_git(_root: Path, *arguments: str, binary: bool = False):
                if arguments[0] == "hash-object":
                    return "a" * 40
                if arguments[0] == "rev-parse":
                    return "b" * 40
                return ""

            with patch.object(lock_builder, "_git", side_effect=fake_git):
                with self.assertRaisesRegex(lock_builder.PreregLockError, "HEAD Git blob"):
                    lock_builder._tracked_head_identity(source, root)

    def test_asset_size_gate_is_hard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            maximum = fixture.evidence.stat().st_size - 1
            spec = lock_builder._load_json_object(fixture.spec)
            spec["mechanism"]["evidence_asset"]["maximum_bytes"] = maximum
            spec["resource_limits"]["evidence_asset_max_bytes"] = maximum
            fixture.spec.write_text(json.dumps(spec) + "\n", encoding="utf-8")
            with patch.object(lock_builder, "EVIDENCE_MAX_BYTES", maximum):
                with self.assertRaisesRegex(lock_builder.PreregLockError, "16 MiB"):
                    lock_builder.build_prereg_lock(
                        project_root=fixture.root,
                        paths=fixture.paths,
                        expectations=fixture.expectations,
                        enforce_git=False,
                        require_defaults=False,
                    )

    def test_p9_selection_must_be_disjoint_from_p8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            rows = lock_builder._read_jsonl(fixture.corpora["selection"])
            rows[0]["ground_truth"]["parent_asin"] = "TARGET006"
            _write_jsonl(fixture.corpora["selection"], rows)
            observed = lock_builder._inspect_conversation_rows(fixture.corpora["selection"])
            metadata = lock_builder._load_json_object(fixture.metadata)
            metadata["corpora"]["selection"].update({
                "samples_sha256": observed["canonical_samples_sha256"],
                "scenario_counts": observed["scenario_counts"],
            })
            metadata["outputs"]["selection"].update({
                "expected_frozen_samples_sha256": observed["canonical_samples_sha256"],
                "samples_file_sha256": observed["canonical_samples_sha256"],
            })
            fixture.metadata.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            canonical = fixture.expectations.canonical()
            canonical["selection"] = observed["canonical_samples_sha256"]
            expectations = lock_builder.FrozenExpectations(
                catalog_sha256=fixture.expectations.catalog_sha256,
                catalog_rows=fixture.expectations.catalog_rows,
                public_git_blob_sha1=fixture.expectations.public_git_blob_sha1,
                public_rows=fixture.expectations.public_rows,
                prior_rows=fixture.expectations.prior_rows,
                split_rows=fixture.expectations.split_rows,
                evidence_bytes=fixture.expectations.evidence_bytes,
                evidence_sha256=fixture.expectations.evidence_sha256,
                evidence_metadata_sha256=fixture.expectations.evidence_metadata_sha256,
                registry_sha256=fixture.expectations.registry_sha256,
                semantics_sha256=fixture.expectations.semantics_sha256,
                canonical_sha256=canonical,
                scenario_counts=fixture.expectations.scenarios(),
            )
            with self.assertRaisesRegex(lock_builder.PreregLockError, "including P8"):
                lock_builder.build_prereg_lock(
                    project_root=fixture.root,
                    paths=fixture.paths,
                    expectations=expectations,
                    enforce_git=False,
                    require_defaults=False,
                )

    def test_live_frozen_corpora_and_evidence_identities_are_current(self) -> None:
        paths = lock_builder.default_paths()
        self.assertEqual(
            (paths.evidence.stat().st_size, lock_builder._sha256_file(paths.evidence)),
            (lock_builder.EXPECTED_EVIDENCE_BYTES, lock_builder.EXPECTED_EVIDENCE_SHA256),
        )
        self.assertEqual(
            lock_builder._sha256_file(paths.evidence_metadata),
            lock_builder.EXPECTED_EVIDENCE_METADATA_SHA256,
        )
        self.assertEqual(
            lock_builder._sha256_file(paths.corpora["selection"]),
            lock_builder.EXPECTED_CANONICAL_SHA256["selection"],
        )
        self.assertEqual(
            lock_builder._sha256_file(paths.corpora["confirmation"]),
            lock_builder.EXPECTED_CANONICAL_SHA256["confirmation"],
        )

    def test_atomic_create_refuses_overwrite_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "lock.json"
            lock_builder.atomic_create(output, {"schema_version": "one"})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                lock_builder.atomic_create(output, {"schema_version": "two"})
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(root.glob(".lock.json.*.tmp")), [])

    def test_build_refuses_existing_output_before_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.output.write_text("keep", encoding="utf-8")
            with patch.object(lock_builder, "_git") as git:
                with self.assertRaises(FileExistsError):
                    lock_builder.build_prereg_lock(
                        project_root=fixture.root,
                        paths=fixture.paths,
                        expectations=fixture.expectations,
                        require_defaults=False,
                    )
            git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
