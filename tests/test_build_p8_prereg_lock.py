from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_p8_prereg_lock as lock_builder


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
        runner = root / lock_builder.SOURCE_PATHS["evaluate_p8"]
        runner.write_text(
            "REQUIRED_SOURCE_NAMES = {'builder', 'p8_negative', 'p8_lab', "
            "'p8_worker', 'evaluate_p8', 'agent', 'coverage', 'attributes', "
            "'slot_ledger', 'response_contract', 'evaluator'}\n",
            encoding="utf-8",
        )

        self.spec = root / "configs" / "p8_explicit_negative_matrix.json"
        self.spec.parent.mkdir(parents=True, exist_ok=True)
        self.spec.write_text(
            json.dumps({"schema_version": lock_builder.SPEC_SCHEMA_VERSION}) + "\n",
            encoding="utf-8",
        )
        self.catalog = root / "data" / "catalog.jsonl"
        _write_jsonl(self.catalog, [{"parent_asin": "CATALOG001"}])
        self.public = root / "data" / "public_set.jsonl"
        _write_jsonl(self.public, [_row("public", "TARGET001")])
        self.priors = {
            name: root / "experiments" / f"{name}.jsonl"
            for name in ("p1", "p5", "p6", "p7")
        }
        for index, (name, path) in enumerate(self.priors.items(), start=2):
            _write_jsonl(path, [_row(name, f"TARGET{index:03d}")])
        scenarios = sorted(lock_builder.EXPECTED_SCENARIO_COUNTS)
        self.corpora = {
            "selection": root / "experiments" / "selection.jsonl",
            "confirmation": root / "experiments" / "confirmation.jsonl",
        }
        selection = [
            _row(f"selection_{index}", f"TARGET{index + 10:03d}", scenario)
            for index, scenario in enumerate(scenarios)
        ]
        confirmation = [
            _row(f"confirmation_{index}", f"TARGET{index + 20:03d}", scenario)
            for index, scenario in enumerate(scenarios)
        ]
        _write_jsonl(self.corpora["selection"], selection)
        _write_jsonl(self.corpora["confirmation"], confirmation)

        observations = {
            "released_public": lock_builder._inspect_conversation_rows(self.public),
            **{
                name: lock_builder._inspect_conversation_rows(path)
                for name, path in self.priors.items()
            },
        }
        split_observations = {
            name: lock_builder._inspect_conversation_rows(path)
            for name, path in self.corpora.items()
        }
        public_blob = lock_builder._git_blob_sha1(self.public)
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
        metadata_names = {
            "released_public": "released_public",
            "prior_p1_derived": "p1",
            "prior_p5_derived": "p5",
            "prior_p6_derived": "p6",
            "prior_p7_derived": "p7",
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
        self.output = root / "configs" / "p8_prereg_lock.json"
        self.paths = lock_builder.LockPaths(
            spec=self.spec,
            catalog=self.catalog,
            released_public=self.public,
            priors=self.priors,
            corpus_metadata=self.metadata,
            corpora=self.corpora,
            output=self.output,
        )
        canonical = {
            name: observed["canonical_samples_sha256"]
            for name, observed in observations.items()
        }
        canonical.update({
            name: observed["canonical_samples_sha256"]
            for name, observed in split_observations.items()
        })
        self.expectations = lock_builder.FrozenExpectations(
            catalog_sha256=lock_builder._sha256_file(self.catalog),
            catalog_rows=1,
            public_git_blob_sha1=public_blob,
            public_rows=1,
            prior_rows=1,
            split_rows=4,
            canonical_sha256=canonical,
            scenario_counts={name: 1 for name in scenarios},
        )


class BuildP8PreregLockTests(unittest.TestCase):
    def test_builds_strict_target_free_lock_from_frozen_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            lock = lock_builder.build_prereg_lock(
                project_root=fixture.root,
                paths=fixture.paths,
                expectations=fixture.expectations,
                enforce_git=False,
                require_defaults=False,
            )
            required_paths = lock_builder._required_paths_from_runner(
                fixture.root / lock_builder.SOURCE_PATHS["evaluate_p8"]
            )
        self.assertEqual(
            set(lock),
            {
                "schema_version", "source", "spec", "catalog", "released_public",
                "priors", "corpus_metadata", "corpora",
            },
        )
        self.assertEqual(lock["schema_version"], lock_builder.SCHEMA_VERSION)
        self.assertEqual(
            set(lock["source"]["files"]), set(required_paths) | {"lock_builder"}
        )
        self.assertEqual(
            lock["corpora"]["selection"]["canonical_samples_sha256"],
            fixture.expectations.canonical()["selection"],
        )
        encoded = json.dumps(lock)
        for identifier in (
            "TARGET001", "TARGET002", "TARGET003", "TARGET004", "TARGET005",
            "TARGET010", "TARGET020",
        ):
            self.assertNotIn(identifier, encoded)

    def test_live_source_registry_matches_runner_and_includes_builder_itself(self) -> None:
        required = lock_builder._required_paths_from_runner(
            lock_builder.PROJECT_ROOT / "scripts" / "evaluate_p8.py"
        )
        self.assertTrue(required)
        for name, path in required.items():
            self.assertEqual(path, lock_builder.SOURCE_PATHS[name])
        self.assertEqual(
            lock_builder.SOURCE_PATHS["lock_builder"],
            "scripts/build_p8_prereg_lock.py",
        )

    def test_clean_pushed_revision_is_strict(self) -> None:
        head = "a" * 40

        def successful(_root: Path, *arguments: str, binary: bool = False):
            mapping = {
                ("branch", "--show-current"): "pre",
                ("rev-parse", "HEAD"): head,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("rev-parse", "origin/pre"): head,
            }
            return mapping[arguments]

        with patch.object(lock_builder, "_git", side_effect=successful):
            self.assertEqual(
                lock_builder.capture_pushed_clean_revision(Path(".")),
                {"git_commit": head, "git_branch": "pre"},
            )

        cases = {
            "named": {("branch", "--show-current"): ""},
            "clean": {("status", "--porcelain=v1", "--untracked-files=all"): " M file"},
            "origin": {("rev-parse", "origin/pre"): "b" * 40},
        }
        for expected, override in cases.items():
            def failing(_root: Path, *arguments: str, binary: bool = False, values=override):
                defaults = {
                    ("branch", "--show-current"): "pre",
                    ("rev-parse", "HEAD"): head,
                    ("status", "--porcelain=v1", "--untracked-files=all"): "",
                    ("rev-parse", "origin/pre"): head,
                }
                return values.get(arguments, defaults[arguments])

            with self.subTest(expected=expected), patch.object(
                lock_builder, "_git", side_effect=failing
            ):
                with self.assertRaisesRegex(lock_builder.PreregLockError, expected):
                    lock_builder.capture_pushed_clean_revision(Path("."))

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

    def test_build_refuses_existing_output_before_any_git_call(self) -> None:
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

    def test_current_frozen_split_and_metadata_hashes_are_latest(self) -> None:
        defaults = lock_builder.default_paths()
        self.assertEqual(
            lock_builder._sha256_file(defaults.corpora["selection"]),
            "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
        )
        self.assertEqual(
            lock_builder._sha256_file(defaults.corpora["confirmation"]),
            "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
        )
        self.assertEqual(
            lock_builder._sha256_file(defaults.corpus_metadata),
            "57a1085db52623fac974705f4ed5394c0f8388f982c6a39bae2be9aab4a363ea",
        )


if __name__ == "__main__":
    unittest.main()
