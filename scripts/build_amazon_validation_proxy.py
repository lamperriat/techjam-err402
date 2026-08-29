from __future__ import annotations

"""Build a privacy-reduced, validation-only Amazon 5-core proxy benchmark.

The builder is deliberately fail closed.  Its CLI accepts only the pinned
validation CSV, verifies every frozen input before and after parsing, excludes
every previously consumed target, and publishes an exclusive target-disjoint
output set with rollback on ordinary Python exceptions.  Raw Amazon identifiers
other than the evaluation target, ratings, timestamps, and prior item sequences
are never written to an output file.  Multi-file publication is not crash-atomic.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "track4.amazon-validation-proxy.v1"
DEFAULT_CONFIG = Path("configs/amazon_validation_proxy_v1.json")
EXPECTED_HEADER = ("user_id", "parent_asin", "rating", "timestamp", "history")
OFFICIAL_VALIDATION_REVISION = "2aa726ef444e72c6a1364c4baa0bcdfb1de55db6"
OFFICIAL_VALIDATION_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
    f"{OFFICIAL_VALIDATION_REVISION}/benchmark/5core/last_out_w_his/"
    "Clothing_Shoes_and_Jewelry.valid.csv"
)
OFFICIAL_VALIDATION_BYTES = 345_027_412
OFFICIAL_VALIDATION_SHA256 = (
    "94b00815eb883ee41ceea08229139c5a0711ee2b6b212a64ecff1450044a21ba"
)
OFFICIAL_CATALOG_ROWS = 50_000
OFFICIAL_CATALOG_SHA256 = (
    "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
)
PRODUCTION_VALIDATION_PATH = (
    "data/external/amazon_reviews_2023/Clothing_Shoes_and_Jewelry.valid.csv"
)
PRODUCTION_CATALOG_PATH = "data/catalog.jsonl"
PRODUCTION_OUTPUT_ROOT = "experiments/fast_track/proxy_v1"
PRODUCTION_MANUAL_LEDGER = "configs/p12_manual_target_exclusions.json"
PRODUCTION_MANUAL_SHA256 = (
    "355f4a25d65f24ed5d39f929d8309aa5ee527ae74dab157913f712b186de2235"
)
PRODUCTION_SEED = "track4-amazon-validation-proxy-v1"
PRODUCTION_SPLIT_ROWS = 2_000
PRODUCTION_TOTAL_ROWS = 8_000
PRODUCTION_SCENARIO_COUNTS = {
    "buying": 800,
    "browsing": 800,
    "intent_override": 300,
    "boundary": 100,
}
PINNED_CONSUMED_CORPORA = {
    "released_public": (
        "data/public_set.jsonl",
        200,
        "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    ),
    "p1_derived": (
        "experiments/p1_derived_product_disjoint.jsonl",
        200,
        "69c413e1cdc032d437af2dcda4a33dc4104860454402ced72001952582cfe687",
    ),
    "p5_selection": (
        "experiments/p5_selection_product_disjoint.jsonl",
        200,
        "0d58a32f65b67c9408558a59df461c340691928a791117099a56049e177efa0c",
    ),
    "p6_selection": (
        "experiments/p6_selection_product_disjoint.jsonl",
        200,
        "27544cdb6ed9495808c35bbab09b4dbadcb88a1d75d162f17bb4fba6ee8841c7",
    ),
    "p7_selection": (
        "experiments/p7_selection_product_disjoint.jsonl",
        200,
        "bad13262ca5cccd3585a80c255918a91c894c8d44d538435006064c3596f9546",
    ),
    "p8_selection": (
        "experiments/p8_selection_product_disjoint.jsonl",
        200,
        "1c11d73d7c8ced617ce874e15a563f240731ca9654ed42bcc4f773b7b4da81ee",
    ),
    "p8_confirmation": (
        "experiments/p8_confirmation_product_disjoint.jsonl",
        200,
        "3ae6f8ff7ab0362399b348c3443daa5b7138aab9cf72e944b7e11dd71d7d3dde",
    ),
    "p9_selection": (
        "experiments/p9_selection_product_disjoint.jsonl",
        200,
        "6298cbd6d7507f4b163ab4979a86ff109e0dffa90557e3b28e5d20d129e5be9f",
    ),
    "p9_confirmation": (
        "experiments/p9_confirmation_product_disjoint.jsonl",
        200,
        "4bbd9d53f32e3773de18bab881ba6e5ef0887ca86701897798ee086430ed08d9",
    ),
    "p11_primary": (
        "experiments/p11_primary_representative.jsonl",
        200,
        "1d578694c3226d1b008d2c9f2f252ed63d114a544c82c218c06116b13c00cf84",
    ),
    "p11_uniform_tail": (
        "experiments/p11_uniform_tail.jsonl",
        200,
        "87d2334dd28dded92df2d8c8897f7f9552efb655bc74488d49dafe2f6efc1dfd",
    ),
    "p11_confirmation": (
        "experiments/p11_confirmation.jsonl",
        200,
        "6dfdcdaf8cd6a091a9b82c192b076ad4e48a89b4023d5ef65394a6d6daf737ba",
    ),
    "p11_failure_negative": (
        "experiments/p11_failure_negative.jsonl",
        80,
        "c0c593dc90af45ec9f3dcdfaaace286f9b0a53c52d0a833c8d292a4488126290",
    ),
    "p11_failure_budget": (
        "experiments/p11_failure_budget.jsonl",
        80,
        "a522134897f7ab8348c327a9a53d30075033dc379bcec610777201abfbb6ee91",
    ),
    "p11_failure_override": (
        "experiments/p11_failure_override.jsonl",
        80,
        "1eeb7e552f2ef0ce8aae413adb7a6393891f1e265c774728ed6ba3b35685df95",
    ),
    "p11_failure_missing_evidence": (
        "experiments/p11_failure_missing_evidence.jsonl",
        80,
        "2aca6b723b592b84caf173fb55c231e8572d0844da9f57cd0c89b9e0489f4ef9",
    ),
}
NORMALIZED_TEXT_HASH_MODE = (
    "SHA-256 over UTF-8 bytes after CRLF/CR to LF normalization"
)
SOURCE_LICENSE_NOTE = (
    "No clear redistribution license for the review data; keep raw and derived rows local."
)
SPLIT_FILENAMES = {
    "train_explore": "proxy_train_explore.jsonl",
    "calibration": "proxy_calibration.jsonl",
    "selection": "proxy_selection.jsonl",
    "confirmation": "proxy_confirmation.sealed.jsonl",
}
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
FORBIDDEN_SESSION_KEYS = {"user_id", "rating", "timestamp", "history"}
_SPACE_RE = re.compile(r"\s+")
_SAFE_TAGS = (
    "clothing",
    "shoes",
    "jewelry",
    "accessories",
    "sportswear",
    "costumes",
    "bags",
)


class ProxyBuildError(ValueError):
    """Raised when an input identity or proxy invariant is not satisfied."""


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    path: Path | str
    expected_sha256: str | None = None
    expected_rows: int | None = None


@dataclass(frozen=True)
class ProxyBuildConfig:
    validation_csv: Path | str
    catalog_path: Path | str
    public_path: Path | str
    manual_exclusions_path: Path | str | None
    output_dir: Path | str
    seed: str
    split_counts: Mapping[str, int]
    expected_validation_sha256: str
    expected_catalog_sha256: str
    expected_validation_bytes: int | None = None
    expected_catalog_rows: int | None = None
    expected_header: Sequence[str] = EXPECTED_HEADER
    scenario_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    source_frequency_outlier_train_threshold: float = 1.0
    consumed_corpora: Sequence[CorpusSpec | Mapping[str, Any] | Path | str] = field(default_factory=tuple)
    expected_public_sha256: str | None = None
    expected_consumed_sha256: Sequence[str] = field(default_factory=tuple)
    expected_consumed_union_count: int | None = None
    expected_manual_exclusions_sha256: str | None = None
    expected_manual_count: int | None = None
    source_url: str = ""
    source_revision: str = ""
    source_license_note: str = SOURCE_LICENSE_NOTE
    config_path: Path | str | None = None
    loaded_config_canonical_sha256: str | None = None
    production_pinned: bool = False


def _canonical_json_bytes(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return (rendered + "\n").encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_file_identity_lf(path: Path) -> tuple[str, int]:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(payload), len(payload)


def _text_file_sha256_lf(path: Path) -> str:
    return _text_file_identity_lf(path)[0]


def _canonical_json_file_sha256(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProxyBuildError(f"invalid JSON identity input: {path}") from error
    if not isinstance(value, Mapping):
        raise ProxyBuildError(f"JSON identity input must be an object: {path}")
    return _sha256_bytes(_canonical_json_bytes(value))


def _git_code_provenance(
    root: Path, *, builder_sha256_lf: str, config_canonical_sha256: str
) -> dict[str, Any]:
    """Bind production evidence to the commit that last changed its code/config."""

    relative_paths = (
        "scripts/build_amazon_validation_proxy.py",
        "configs/amazon_validation_proxy_v1.json",
    )

    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ProxyBuildError(f"cannot establish proxy code provenance: {detail}")
        return completed.stdout

    commit = run("rev-list", "-1", "HEAD", "--", *relative_paths).decode(
        "ascii", errors="strict"
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProxyBuildError("proxy builder/config do not have a committed source identity")
    builder_blob = run("show", f"{commit}:{relative_paths[0]}")
    config_blob = run("show", f"{commit}:{relative_paths[1]}")
    try:
        config_value = json.loads(config_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProxyBuildError("committed proxy config is not canonicalizable JSON") from error
    if not isinstance(config_value, Mapping):
        raise ProxyBuildError("committed proxy config must be a JSON object")
    builder_blob_hash = _sha256_bytes(
        builder_blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    )
    config_blob_hash = _sha256_bytes(_canonical_json_bytes(config_value))
    if builder_blob_hash != builder_sha256_lf or config_blob_hash != config_canonical_sha256:
        raise ProxyBuildError(
            "working proxy builder/config do not match their committed source identity"
        )
    return {
        "commit": commit,
        "config_blob_canonical_sha256": config_blob_hash,
        "paths": list(relative_paths),
        "builder_blob_sha256_lf": builder_blob_hash,
        "working_sources_match_commit": True,
    }


def _stable_digest(seed: str, *parts: str) -> str:
    value = "\0".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _target_set_sha256(targets: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(targets))).encode("utf-8")
    return _sha256_bytes(payload)


def _resolve(path: Path | str | None, root: Path) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _reject_test_source(path: Path, source_url: str) -> None:
    names = [path.name.casefold()]
    if source_url:
        names.append(Path(urlparse(source_url).path).name.casefold())
    if any("test" in name for name in names):
        raise ProxyBuildError("test data is forbidden; only the pinned validation CSV is accepted")
    if "valid" not in path.name.casefold():
        raise ProxyBuildError("validation filename must contain 'valid'")


def _validate_pinned_production_source(config: ProxyBuildConfig) -> None:
    if not config.production_pinned:
        return
    checks = {
        "validation filename": Path(config.validation_csv).name
        == "Clothing_Shoes_and_Jewelry.valid.csv",
        "validation URL": config.source_url == OFFICIAL_VALIDATION_URL,
        "validation revision": config.source_revision == OFFICIAL_VALIDATION_REVISION,
        "validation SHA-256": config.expected_validation_sha256
        == OFFICIAL_VALIDATION_SHA256,
        "validation bytes": config.expected_validation_bytes
        == OFFICIAL_VALIDATION_BYTES,
        "validation header": tuple(config.expected_header) == EXPECTED_HEADER,
        "catalog SHA-256": config.expected_catalog_sha256 == OFFICIAL_CATALOG_SHA256,
        "catalog rows": config.expected_catalog_rows == OFFICIAL_CATALOG_ROWS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ProxyBuildError(
            "production config is not the pinned validation-only source: "
            + ", ".join(failed)
        )


def _require_config(condition: bool, message: str) -> None:
    if not condition:
        raise ProxyBuildError(f"invalid pinned production config: {message}")


def _validate_nested_production_config(value: Mapping[str, Any]) -> None:
    """Validate every production declaration that the builder relies on or emits."""

    try:
        source = value["source"]
        catalog = value["catalog"]
        exclusions = value["exclusions"]
        selection = value["selection"]
        outputs = value["outputs"]
        annotations = value["annotations"]
        privacy = value["privacy"]
        distribution = value["distribution"]
        corpora = list(exclusions["consumed_corpora"])
        manual = exclusions["manual_target_ledger"]
        split_specs = outputs["splits"]
    except (KeyError, TypeError) as error:
        raise ProxyBuildError("pinned production config is missing a required section") from error

    _require_config(value.get("schema_version") == SCHEMA_VERSION, "schema_version")
    source_fields = tuple(str(item) for item in source.get("fields", ()))
    _require_config(source.get("dataset") == "McAuley-Lab/Amazon-Reviews-2023", "source.dataset")
    _require_config(source.get("domain") == "Clothing_Shoes_and_Jewelry", "source.domain")
    _require_config(source.get("split") == "validation", "source.split must be validation")
    _require_config(source.get("revision") == OFFICIAL_VALIDATION_REVISION, "source.revision")
    _require_config(source.get("url") == OFFICIAL_VALIDATION_URL, "source.url")
    _require_config(source.get("path") == PRODUCTION_VALIDATION_PATH, "source.path")
    _require_config(source.get("bytes") == OFFICIAL_VALIDATION_BYTES, "source.bytes")
    _require_config(source.get("sha256") == OFFICIAL_VALIDATION_SHA256, "source.sha256")
    _require_config(source_fields == EXPECTED_HEADER, "source.fields")
    _require_config(source.get("header") == ",".join(source_fields), "source.header")
    _require_config(source.get("history_delimiter") == " ", "source.history_delimiter")

    _require_config(catalog.get("path") == PRODUCTION_CATALOG_PATH, "catalog.path")
    _require_config(catalog.get("rows") == OFFICIAL_CATALOG_ROWS, "catalog.rows")
    _require_config(catalog.get("sha256") == OFFICIAL_CATALOG_SHA256, "catalog.sha256")

    _require_config(exclusions.get("group_key") == "ground_truth.parent_asin", "exclusions.group_key")
    _require_config(exclusions.get("file_hash_mode") == NORMALIZED_TEXT_HASH_MODE, "exclusions.file_hash_mode")
    _require_config(exclusions.get("consumed_target_union_count") == 2_720, "exclusions.consumed_target_union_count")
    _require_config(exclusions.get("require_consumed_corpora_pairwise_target_disjoint") is True, "exclusions pairwise requirement")
    corpus_by_id: dict[str, Mapping[str, Any]] = {}
    for item in corpora:
        _require_config(isinstance(item, Mapping), "consumed corpus entry type")
        corpus_id = str(item.get("id", ""))
        _require_config(bool(corpus_id) and corpus_id not in corpus_by_id, "consumed corpus ids")
        corpus_by_id[corpus_id] = item
    _require_config(set(corpus_by_id) == set(PINNED_CONSUMED_CORPORA), "consumed corpus registry")
    for corpus_id, expected in PINNED_CONSUMED_CORPORA.items():
        item = corpus_by_id[corpus_id]
        observed = (item.get("path"), item.get("rows"), item.get("file_sha256"))
        _require_config(observed == expected, f"consumed corpus {corpus_id}")
    _require_config(manual.get("path") == PRODUCTION_MANUAL_LEDGER, "manual ledger path")
    _require_config(manual.get("schema_version") == "track4.p12-manual-exclusions.v1", "manual ledger schema")
    _require_config(manual.get("expected_count") == 0, "manual ledger count")
    _require_config(manual.get("file_sha256") == PRODUCTION_MANUAL_SHA256, "manual ledger hash")

    _require_config(selection.get("seed") == PRODUCTION_SEED, "selection.seed")
    _require_config(selection.get("source_frequency_outlier_train_threshold") == 0.1, "selection outlier threshold")
    _require_config(
        selection.get("source_frequency_outlier_policy")
        == "assign outcome-independent validation-source-frequency outliers to train_explore before held-out hash grouping",
        "selection outlier policy",
    )
    _require_config(selection.get("target_join") == "inner_join_frozen_catalog", "selection.target_join")
    group_split = selection.get("group_split", {})
    _require_config(group_split.get("key") == "parent_asin", "selection.group_split.key")
    _require_config(
        group_split.get("algorithm")
        == "outcome-independent validation-source-frequency outliers to train_explore, then taxonomy/popularity-stratified SHA-256 order with deterministic per-stratum rotation and round-robin target-group assignment",
        "selection group-split algorithm",
    )
    _require_config(group_split.get("pairwise_target_overlap_required") == 0, "selection group overlap")
    row_sampling = selection.get("row_sampling", {})
    _require_config(
        row_sampling.get("algorithm")
        == "breadth-first deterministic source-occurrence sampling inside each fixed target group, using safe-profile histograms only",
        "selection row-sampling algorithm",
    )
    _require_config(
        row_sampling.get("weighting")
        == "raw per-target source frequency divided across emitted rows as evaluator-only source_weight",
        "selection weighting",
    )
    _require_config(row_sampling.get("raw_user_id_persisted") is False, "selection raw user id policy")
    scenario_assignment = selection.get("scenario_assignment", {})
    _require_config(
        scenario_assignment.get("algorithm")
        == "stable hash shuffle of a fixed scenario multiset",
        "scenario assignment algorithm",
    )
    scenario_counts = scenario_assignment.get("counts_per_split")
    _require_config(scenario_counts == PRODUCTION_SCENARIO_COUNTS, "scenario counts")

    _require_config(outputs.get("root") == PRODUCTION_OUTPUT_ROOT, "outputs.root")
    _require_config(outputs.get("manifest") == f"{PRODUCTION_OUTPUT_ROOT}/manifest.json", "outputs.manifest")
    _require_config(outputs.get("audit") == f"{PRODUCTION_OUTPUT_ROOT}/audit.json", "outputs.audit")
    _require_config(isinstance(split_specs, Mapping), "outputs.splits")
    _require_config(set(split_specs) == set(SPLIT_FILENAMES), "output split names")
    declared_paths: set[str] = set()
    for split, filename in SPLIT_FILENAMES.items():
        spec = split_specs[split]
        expected_path = f"{PRODUCTION_OUTPUT_ROOT}/{filename}"
        _require_config(spec.get("path") == expected_path, f"output path for {split}")
        _require_config(spec.get("rows") == PRODUCTION_SPLIT_ROWS, f"output rows for {split}")
        _require_config(spec.get("sealed") is (split == "confirmation"), f"sealed flag for {split}")
        if split == "confirmation":
            _require_config(spec.get("requires_explicit_unseal") is True, "confirmation unseal guard")
        declared_paths.add(str(spec.get("path")))
    _require_config(len(declared_paths) == len(SPLIT_FILENAMES), "output paths must be unique")
    _require_config(outputs.get("total_rows") == PRODUCTION_TOTAL_ROWS, "outputs.total_rows")
    _require_config(sum(int(spec["rows"]) for spec in split_specs.values()) == PRODUCTION_TOTAL_ROWS, "split row sum")

    taxonomy = annotations.get("taxonomy", {})
    popularity = annotations.get("popularity", {})
    views = annotations.get("evaluation_views", {})
    _require_config(taxonomy.get("version") == "catalog-path-coarse-v1", "taxonomy version")
    _require_config(taxonomy.get("values") == ["clothing", "shoes", "jewelry", "accessories-other"], "taxonomy values")
    _require_config(taxonomy.get("leaf_path_included") is True, "taxonomy leaf path")
    _require_config(popularity.get("version") == "catalog-rating-number-midrank-v1", "popularity version")
    _require_config(popularity.get("values") == ["head", "mid", "tail"], "popularity values")
    _require_config(set(views) == {"source_weighted", "target_uniform", "taxonomy_balanced"}, "evaluation views")
    _require_config(
        views["source_weighted"].get("aggregation")
        == "micro_average_over_sampled_source_rows",
        "source-weighted aggregation",
    )
    _require_config(
        views["target_uniform"].get("aggregation")
        == "one equal contribution per unique target after aggregating its repeated sessions",
        "target-uniform aggregation",
    )
    _require_config(
        views["taxonomy_balanced"].get("aggregation")
        == "unweighted_macro_average_over_nonempty_taxonomy_groups",
        "taxonomy-balanced aggregation",
    )
    _require_config(
        views["taxonomy_balanced"].get("reuses_source_rows") is True,
        "taxonomy-balanced source-row policy",
    )

    _require_config(
        privacy.get("agent_profile_source")
        == "pre-validation history length and frozen-catalog coarse metadata aggregates only",
        "privacy profile source",
    )
    _require_config(set(privacy.get("raw_fields_never_emitted", ())) == FORBIDDEN_SESSION_KEYS, "privacy forbidden fields")
    _require_config(privacy.get("target_rating_used_as_prior_profile") is False, "privacy target rating")
    _require_config(privacy.get("raw_history_asins_persisted") is False, "privacy raw history")
    _require_config(distribution.get("nonredistributable") is True, "distribution nonredistributable")
    _require_config(distribution.get("local_research_use_only") is True, "distribution local-only")
    _require_config(distribution.get("commit_raw_or_derived_rows") is False, "distribution commit guard")
    _require_config(distribution.get("include_raw_or_derived_rows_in_release_or_zip") is False, "distribution archive guard")
    _require_config(
        distribution.get("boundary")
        == "validation-derived proxy; not organizer-private data and not a byte-identical organizer generator",
        "distribution boundary",
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None,
    expected_bytes: int | None = None,
    normalize_text_eol: bool = False,
) -> tuple[str, int]:
    if not path.is_file():
        raise ProxyBuildError(f"{label} is missing: {path}")
    raw_size = path.stat().st_size
    if expected_bytes is not None and raw_size != int(expected_bytes):
        raise ProxyBuildError(
            f"{label} byte count mismatch: expected {expected_bytes}, observed {raw_size}"
        )
    if normalize_text_eol:
        digest, size = _text_file_identity_lf(path)
    else:
        digest, size = _file_sha256(path), raw_size
    if expected_sha256 and digest != expected_sha256.casefold():
        raise ProxyBuildError(f"{label} SHA-256 mismatch")
    return digest, size


def _load_jsonl_targets(path: Path) -> tuple[set[str], int]:
    targets: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                value = json.loads(line)
                target = str(value["ground_truth"]["parent_asin"]).strip()
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ProxyBuildError(
                    f"invalid exclusion row in {path.name}:{line_number}"
                ) from error
            if not target:
                raise ProxyBuildError(f"empty exclusion target in {path.name}:{line_number}")
            targets.add(target)
    return targets, rows


def _load_manual_targets(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        try:
            value = json.loads(text)
            if value.get("schema_version") != "track4.p12-manual-exclusions.v1":
                raise ProxyBuildError("manual exclusion ledger schema mismatch")
            raw_targets = value["targets"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ProxyBuildError("manual exclusion ledger is invalid") from error
    else:
        raw_targets = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not isinstance(raw_targets, list) or any(not isinstance(item, str) for item in raw_targets):
        raise ProxyBuildError("manual exclusion ledger targets must be a string list")
    cleaned = {item.strip() for item in raw_targets if item.strip()}
    if len(cleaned) != len(raw_targets):
        raise ProxyBuildError("manual exclusion ledger contains blank or duplicate targets")
    return cleaned


def _coerce_corpus_specs(
    values: Sequence[CorpusSpec | Mapping[str, Any] | Path | str],
    expected_hashes: Sequence[str] = (),
) -> tuple[CorpusSpec, ...]:
    result: list[CorpusSpec] = []
    if expected_hashes and len(expected_hashes) != len(values):
        raise ProxyBuildError("consumed corpus hash count does not match path count")
    for index, value in enumerate(values):
        if isinstance(value, CorpusSpec):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(
                CorpusSpec(
                    name=str(value["name"]),
                    path=str(value["path"]),
                    expected_sha256=(
                        str(value["expected_sha256"])
                        if value.get("expected_sha256")
                        else None
                    ),
                    expected_rows=(
                        int(value["expected_rows"])
                        if value.get("expected_rows") is not None
                        else None
                    ),
                )
            )
        elif isinstance(value, (str, Path)):
            result.append(
                CorpusSpec(
                    name=f"consumed_{index + 1}",
                    path=value,
                    expected_sha256=(
                        str(expected_hashes[index]) if expected_hashes else None
                    ),
                )
            )
        else:
            raise ProxyBuildError("consumed_corpora entries must be objects")
    if len({spec.name for spec in result}) != len(result):
        raise ProxyBuildError("consumed corpus names must be unique")
    return tuple(result)


def _load_exclusions(
    config: ProxyBuildConfig, root: Path, public_path: Path
) -> tuple[set[str], dict[str, Any]]:
    specs = list(
        _coerce_corpus_specs(config.consumed_corpora, config.expected_consumed_sha256)
    )
    if any(spec.name == "released_public" for spec in specs):
        raise ProxyBuildError("released_public must use the dedicated public_path field")
    specs.insert(
        0,
        CorpusSpec(
            "released_public",
            public_path,
            expected_sha256=config.expected_public_sha256,
        ),
    )

    union: set[str] = set()
    observations: dict[str, Any] = {}
    target_sets: dict[str, set[str]] = {}
    for spec in specs:
        path = _resolve(spec.path, root)
        assert path is not None
        digest, size = _verify_file(
            path,
            label=f"consumed corpus {spec.name}",
            expected_sha256=spec.expected_sha256,
            normalize_text_eol=True,
        )
        targets, rows = _load_jsonl_targets(path)
        if spec.expected_rows is not None and rows != int(spec.expected_rows):
            raise ProxyBuildError(f"consumed corpus {spec.name} row count mismatch")
        if len(targets) != rows:
            raise ProxyBuildError(f"consumed corpus {spec.name} has duplicate targets")
        target_sets[spec.name] = targets
        union.update(targets)
        observations[spec.name] = {
            "bytes_lf": size,
            "rows": rows,
            "sha256_lf": digest,
            "unique_targets": len(targets),
        }

    overlaps = {
        f"{left}__{right}": len(target_sets[left] & target_sets[right])
        for left, right in combinations(sorted(target_sets), 2)
    }
    nonzero = {name: count for name, count in overlaps.items() if count}
    if nonzero:
        raise ProxyBuildError(f"consumed target registries overlap: {nonzero}")
    if (
        config.expected_consumed_union_count is not None
        and len(union) != int(config.expected_consumed_union_count)
    ):
        raise ProxyBuildError(
            "consumed exclusion union mismatch: "
            f"expected {config.expected_consumed_union_count}, observed {len(union)}"
        )

    consumed_union = set(union)
    manual_path = _resolve(config.manual_exclusions_path, root)
    manual_digest: str | None = None
    manual_bytes = 0
    if manual_path is not None:
        manual_digest, manual_bytes = _verify_file(
            manual_path,
            label="manual exclusion ledger",
            expected_sha256=config.expected_manual_exclusions_sha256,
            normalize_text_eol=True,
        )
    manual = _load_manual_targets(manual_path)
    if (
        config.expected_manual_count is not None
        and len(manual) != int(config.expected_manual_count)
    ):
        raise ProxyBuildError("manual exclusion target count mismatch")
    overlap_with_consumed = len(manual & union)
    union.update(manual)
    public_count = len(target_sets["released_public"])
    consumed_only = set().union(
        *(targets for name, targets in target_sets.items() if name != "released_public")
    ) if len(target_sets) > 1 else set()
    return union, {
        "consumed": observations,
        "file_hash_mode": NORMALIZED_TEXT_HASH_MODE,
        "consumed_pairwise_overlap_nonzero": nonzero,
        "consumed_target_union_count": len(consumed_union),
        "consumed_target_union_sha256": _target_set_sha256(consumed_union),
        "manual": {
            "bytes_lf": manual_bytes,
            "sha256_lf": manual_digest,
            "target_count": len(manual),
            "overlap_with_consumed": overlap_with_consumed,
        },
        "public_target_count": public_count,
        "manual_target_count": len(manual),
        "consumed_target_count": len(consumed_only),
        "union_target_count": len(union),
        "total_excluded_target_count": len(union),
        "total_excluded_target_sha256": _target_set_sha256(union),
    }


def _clean_categories(product: Mapping[str, Any]) -> list[str]:
    values = product.get("categories") or []
    return [_SPACE_RE.sub(" ", str(value)).strip() for value in values if str(value).strip()]


def _taxonomy(product: Mapping[str, Any]) -> str:
    categories = _clean_categories(product)[1:]
    lowered = " | ".join(value.casefold() for value in categories)
    if any(token in lowered for token in ("jewelry", "earring", "necklace", "bracelet", "ring")):
        return "jewelry"
    if any(
        token in lowered
        for token in ("shoe", "footwear", "boot", "sneaker", "sandal", "slipper", "loafer")
    ):
        return "shoes"
    if any(
        token in lowered
        for token in (
            "clothing",
            "apparel",
            "shirt",
            "dress",
            "pant",
            "jean",
            "skirt",
            "coat",
            "jacket",
            "underwear",
            "sock",
            "swimwear",
            "costume",
        )
    ):
        return "clothing"
    return "accessories-other"


def _taxonomy_leaf(product: Mapping[str, Any]) -> str:
    categories = _clean_categories(product)
    values = categories[-2:] if categories else ["unknown"]
    return " > ".join(values)


def _safe_product_tag(product: Mapping[str, Any]) -> str:
    taxonomy = _taxonomy(product)
    if taxonomy != "accessories-other":
        return taxonomy
    lowered = " ".join(value.casefold() for value in _clean_categories(product)[1:])
    for tag in _SAFE_TAGS[3:]:
        if tag in lowered:
            return tag
    return "accessories"


def _load_catalog(path: Path, expected_rows: int | None) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
                target = str(product["parent_asin"]).strip()
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ProxyBuildError(f"invalid catalog row {line_number}") from error
            if not target or target in products:
                raise ProxyBuildError(f"blank or duplicate catalog target at row {line_number}")
            products[target] = product
    if expected_rows is not None and len(products) != int(expected_rows):
        raise ProxyBuildError(
            f"catalog row count mismatch: expected {expected_rows}, observed {len(products)}"
        )
    return products


def _rating_number(product: Mapping[str, Any]) -> int:
    value = product.get("rating_number")
    if isinstance(value, bool):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return int(numeric) if math.isfinite(numeric) and numeric >= 0 else 0


def _popularity_bins(products: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    frequencies = Counter(_rating_number(product) for product in products.values())
    total = len(products)
    lower = 0
    by_value: dict[int, str] = {}
    for value in sorted(frequencies):
        count = frequencies[value]
        percentile = (lower + count / 2) / total
        by_value[value] = "tail" if percentile < 1 / 3 else "mid" if percentile < 2 / 3 else "head"
        lower += count
    return {
        target: by_value[_rating_number(product)] for target, product in products.items()
    }


def _purchase_frequency(count: int) -> str:
    if count <= 0:
        return "not provided"
    if count <= 2:
        return "1-2 prior purchases"
    if count <= 4:
        return "3-4 prior purchases"
    if count <= 9:
        return "5-9 prior purchases"
    return "10+ prior purchases"


def _safe_profile(
    prior_ids: Sequence[str], products: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], int]:
    joined = [products[value] for value in prior_ids if value in products]
    counts = Counter(_safe_product_tag(product) for product in joined)
    tags = [
        tag
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:3]
    if tags:
        summary = "Prior catalog activity most often falls in: " + ", ".join(tags) + "."
    else:
        summary = "No stable coarse category preference is available from prior catalog activity."
    return (
        {
            "average_prior_rating": None,
            "preference_tags": tags,
            "purchase_frequency": _purchase_frequency(len(prior_ids)),
            "rating_style": "unknown",
            "summary": summary,
        },
        len(joined),
    )


def _metadata_difficulty(product: Mapping[str, Any]) -> str:
    score = sum(
        (
            bool(str(product.get("title") or "").strip()),
            len(_clean_categories(product)) >= 3,
            bool(product.get("features")),
            bool(product.get("description")),
            bool(product.get("details")),
            product.get("price") not in (None, ""),
            bool(str(product.get("store") or "").strip()),
        )
    )
    return "easy" if score >= 6 else "medium" if score >= 4 else "hard"


def _read_validation(
    path: Path,
    products: Mapping[str, Mapping[str, Any]],
    excluded: set[str],
    expected_header: Sequence[str],
    seed: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    representatives: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ProxyBuildError("validation CSV is empty") from error
        if header != tuple(expected_header):
            raise ProxyBuildError(
                "validation CSV schema/header column mismatch: "
                f"expected {tuple(expected_header)}, observed {header}"
            )
        for row_number, row in enumerate(reader, start=2):
            stats["source_rows"] += 1
            if len(row) != len(expected_header):
                raise ProxyBuildError(f"validation row {row_number} does not have five columns")
            target = row[1].strip()
            if target not in products:
                stats["target_not_in_catalog"] += 1
                continue
            if target in excluded:
                stats["excluded_target_rows"] += 1
                continue
            source_counts[target] += 1
            prior_ids = tuple(value for value in row[4].split() if value)
            profile, joined_count = _safe_profile(prior_ids, products)
            stats["prior_items_total"] += len(prior_ids)
            stats["prior_items_joined"] += joined_count
            profile_hash = _sha256_bytes(_canonical_json_bytes(profile))
            candidate_key = (-joined_count, _stable_digest(seed, "representative", target, profile_hash))
            current = representatives.get(target)
            if current is None:
                current = {
                    "profile": profile,
                    "joined_count": joined_count,
                    "selection_key": candidate_key,
                    "profiles": {},
                }
                representatives[target] = current
            elif candidate_key < current["selection_key"]:
                current["profile"] = profile
                current["joined_count"] = joined_count
                current["selection_key"] = candidate_key
            profile_record = current["profiles"].setdefault(
                profile_hash,
                {"count": 0, "joined_count": joined_count, "profile": profile},
            )
            profile_record["count"] += 1
    stats["eligible_source_rows"] = sum(source_counts.values())
    stats["eligible_unique_targets"] = len(representatives)
    for target, record in representatives.items():
        record["source_weight"] = source_counts[target]
    return representatives, dict(stats)


def _scenario_quotas(
    config: ProxyBuildConfig, split: str, count: int
) -> dict[str, int]:
    if split in config.scenario_counts:
        quotas = {name: int(value) for name, value in config.scenario_counts[split].items()}
    elif count % 20 == 0:
        unit = count // 20
        quotas = {"buying": 8 * unit, "browsing": 8 * unit, "intent_override": 3 * unit, "boundary": unit}
    else:
        raise ProxyBuildError(f"{split} requires explicit scenario counts")
    if set(quotas) != set(SCENARIOS) or sum(quotas.values()) != count or min(quotas.values()) < 0:
        raise ProxyBuildError(f"{split} scenario quotas are invalid")
    return quotas


def _assign_scenarios(
    tokens: Sequence[tuple[str, str, int]],
    quotas: Mapping[str, int],
    seed: str,
    split: str,
) -> dict[tuple[str, str, int], str]:
    ordered = sorted(
        tokens,
        key=lambda value: (
            _stable_digest(
                seed, split, "scenario", value[0], value[1], str(value[2])
            ),
            value,
        ),
    )
    labels = [name for name in SCENARIOS for _ in range(int(quotas[name]))]
    return dict(zip(ordered, labels, strict=True))


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        bad = FORBIDDEN_SESSION_KEYS & {str(key).casefold() for key in value}
        if bad:
            raise RuntimeError(f"proxy session contains forbidden raw keys: {sorted(bad)}")
        for item in value.values():
            _validate_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _validate_no_forbidden_keys(item)


def _allocate_target_groups(
    representatives: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    popularity: Mapping[str, str],
    seed: str,
    outlier_train_threshold: float,
) -> dict[str, list[str]]:
    """Hash-split targets evenly within catalog-only taxonomy/popularity strata."""

    if not 0 < outlier_train_threshold <= 1:
        raise ProxyBuildError("source-frequency outlier threshold must be in (0, 1]")
    split_names = tuple(SPLIT_FILENAMES)
    groups: dict[str, list[str]] = {name: [] for name in split_names}
    total_source_weight = sum(
        int(record["source_weight"]) for record in representatives.values()
    )
    outliers = {
        target
        for target, record in representatives.items()
        if total_source_weight
        and int(record["source_weight"]) / total_source_weight
        >= outlier_train_threshold
    }
    groups["train_explore"].extend(
        sorted(
            outliers,
            key=lambda target: (
                _stable_digest(seed, "source-frequency-outlier", target),
                target,
            ),
        )
    )
    strata: dict[tuple[str, str], list[str]] = {}
    for target in representatives:
        if target in outliers:
            continue
        key = (_taxonomy(products[target]), popularity[target])
        strata.setdefault(key, []).append(target)

    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda target: (
                _stable_digest(seed, "target-group", *stratum, target),
                target,
            ),
        )
        rotation = int(_stable_digest(seed, "stratum-rotation", *stratum), 16) % len(
            split_names
        )
        for index, target in enumerate(ordered):
            split = split_names[(rotation + index) % len(split_names)]
            groups[split].append(target)
    return groups


def _build_rows(
    representatives: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    popularity: Mapping[str, str],
    config: ProxyBuildConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    counts = {name: int(config.split_counts.get(name, 0)) for name in SPLIT_FILENAMES}
    if set(config.split_counts) != set(SPLIT_FILENAMES) or min(counts.values()) <= 0:
        raise ProxyBuildError("split_counts must define four positive proxy splits")
    required = sum(counts.values())
    source_capacity = sum(int(record["source_weight"]) for record in representatives.values())
    if source_capacity < required:
        raise ProxyBuildError(
            f"proxy needs {required} source sessions but only {source_capacity} are eligible "
            f"across {len(representatives)} unique targets"
        )

    target_groups = _allocate_target_groups(
        representatives,
        products,
        popularity,
        config.seed,
        config.source_frequency_outlier_train_threshold,
    )

    target_sets: dict[str, set[str]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_FILENAMES:
        group = target_groups[split]
        occurrences: dict[str, list[tuple[str, int]]] = {}
        for target in group:
            tokens = [
                (profile_hash, ordinal)
                for profile_hash, profile_record in representatives[target]["profiles"].items()
                for ordinal in range(int(profile_record["count"]))
            ]
            tokens.sort(
                key=lambda value: (
                    _stable_digest(
                        config.seed,
                        split,
                        "within-target-occurrence",
                        target,
                        value[0],
                        str(value[1]),
                    ),
                    value,
                )
            )
            occurrences[target] = tokens
        capacity = sum(len(tokens) for tokens in occurrences.values())
        if capacity < counts[split]:
            raise ProxyBuildError(
                f"{split} needs {counts[split]} sessions but its target-disjoint "
                f"group has source capacity {capacity} across {len(group)} unique targets"
            )
        selected_tokens: list[tuple[str, str, int]] = []
        depth = 0
        while len(selected_tokens) < counts[split]:
            available = [target for target in group if len(occurrences[target]) > depth]
            if not available:
                raise RuntimeError("source occurrence capacity audit drifted during sampling")
            available.sort(
                key=lambda target: (
                    _stable_digest(
                        config.seed, split, "breadth-first", str(depth), target
                    ),
                    target,
                )
            )
            for target in available:
                profile_hash, ordinal = occurrences[target][depth]
                selected_tokens.append((target, profile_hash, ordinal))
                if len(selected_tokens) == counts[split]:
                    break
            depth += 1
        target_sets[split] = {target for target, _, _ in selected_tokens}
        quotas = _scenario_quotas(config, split, counts[split])
        scenarios = _assign_scenarios(selected_tokens, quotas, config.seed, split)
        row_order = sorted(
            selected_tokens,
            key=lambda value: (
                _stable_digest(
                    config.seed,
                    split,
                    "row",
                    value[0],
                    value[1],
                    str(value[2]),
                ),
                value,
            ),
        )
        emissions = Counter(target for target, _, _ in selected_tokens)
        rows: list[dict[str, Any]] = []
        for index, token in enumerate(row_order, start=1):
            target, profile_hash, _source_ordinal = token
            product = products[target]
            record = representatives[target]
            taxonomy = _taxonomy(product)
            source_weight = int(record["source_weight"]) / emissions[target]
            row = {
                "category_bucket": taxonomy,
                "difficulty_bucket": _metadata_difficulty(product),
                "evaluation_strata": {
                    "popularity": popularity[target],
                    "source_weight": round(source_weight, 12),
                    "taxonomy": taxonomy,
                    "taxonomy_leaf": _taxonomy_leaf(product),
                },
                "ground_truth": {"parent_asin": target},
                "sample_id": f"amazon_proxy_{split}_{index:04d}",
                "scenario_type": scenarios[token],
                "taxonomy": {
                    "group": taxonomy,
                    "leaf_path": _clean_categories(product),
                },
                "user_profile": dict(record["profiles"][profile_hash]["profile"]),
            }
            _validate_no_forbidden_keys(row)
            rows.append(row)
        rows_by_split[split] = rows
    return rows_by_split, target_sets


def _exclusive_publish(
    payloads: Mapping[Path, bytes],
    *,
    allow_identical_existing: Iterable[Path] = (),
) -> None:
    destinations = [path.resolve() for path in payloads]
    if len(destinations) != len(set(destinations)):
        raise ProxyBuildError("proxy output paths must be unique")
    allowed = {path.resolve() for path in allow_identical_existing}
    pending: dict[Path, bytes] = {}
    for path, payload in payloads.items():
        if not path.exists():
            pending[path] = payload
            continue
        if (
            path.resolve() in allowed
            and path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == payload
        ):
            continue
        raise FileExistsError(f"proxy output already exists or differs: {path}")
    temporary: dict[Path, Path] = {}
    created: list[tuple[Path, int, int]] = []
    try:
        for destination, payload in pending.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temp = Path(name)
            temporary[destination] = temp
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temp in temporary.items():
            os.link(temp, destination)
            identity = destination.stat(follow_symlinks=False)
            created.append((destination, identity.st_dev, identity.st_ino))
    except BaseException:
        for destination, device, inode in created:
            try:
                identity = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if identity.st_dev == device and identity.st_ino == inode:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def _split_summary(
    rows: Sequence[Mapping[str, Any]], targets: set[str], payload: bytes, *, sealed: bool
) -> dict[str, Any]:
    digest = _sha256_bytes(payload)
    weight_by_target: Counter[str] = Counter()
    weight_by_scenario: Counter[str] = Counter()
    weight_by_taxonomy: Counter[str] = Counter()
    weight_by_popularity: Counter[str] = Counter()
    for row in rows:
        weight = float(row["evaluation_strata"]["source_weight"])
        target = str(row["ground_truth"]["parent_asin"])
        weight_by_target[target] += weight
        weight_by_scenario[str(row["scenario_type"])] += weight
        weight_by_taxonomy[str(row["evaluation_strata"]["taxonomy"])] += weight
        weight_by_popularity[str(row["evaluation_strata"]["popularity"])] += weight
    total_weight = sum(weight_by_target.values())
    squared_weight = sum(value * value for value in weight_by_target.values())
    def weighted_shares(values: Mapping[str, float]) -> dict[str, float]:
        return {
            key: round(value / total_weight if total_weight else 0.0, 6)
            for key, value in sorted(values.items())
        }
    return {
        "bytes": len(payload),
        "difficulty_counts": dict(sorted(Counter(str(row["difficulty_bucket"]) for row in rows).items())),
        "file_sha256": digest,
        "sha256": digest,
        "popularity_counts": dict(sorted(Counter(str(row["evaluation_strata"]["popularity"]) for row in rows).items())),
        "row_count": len(rows),
        "rows": len(rows),
        "scenario_counts": dict(sorted(Counter(str(row["scenario_type"]) for row in rows).items())),
        "sealed": sealed,
        "source_weight_sum": round(total_weight, 6),
        "source_weight_effective_target_count": round(
            total_weight * total_weight / squared_weight if squared_weight else 0.0,
            6,
        ),
        "source_weight_max_target_share": round(
            max(weight_by_target.values(), default=0.0) / total_weight
            if total_weight
            else 0.0,
            6,
        ),
        "source_weighted_popularity_shares": weighted_shares(weight_by_popularity),
        "source_weighted_scenario_shares": weighted_shares(weight_by_scenario),
        "source_weighted_taxonomy_shares": weighted_shares(weight_by_taxonomy),
        "target_set_sha256": _target_set_sha256(targets),
        "taxonomy_counts": dict(sorted(Counter(str(row["evaluation_strata"]["taxonomy"]) for row in rows).items())),
        "unique_target_count": len(targets),
        "unique_targets": len(targets),
    }


def _build_distribution_audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_stats = manifest["source"]["stats"]
    split_summaries = manifest["splits"]
    prior_total = int(source_stats.get("prior_items_total", 0))
    prior_joined = int(source_stats.get("prior_items_joined", 0))
    taxonomy_values = ("clothing", "shoes", "jewelry", "accessories-other")
    taxonomy_share_ranges: dict[str, dict[str, float]] = {}
    for taxonomy in taxonomy_values:
        shares = {
            split: summary["taxonomy_counts"].get(taxonomy, 0) / summary["rows"]
            for split, summary in split_summaries.items()
        }
        taxonomy_share_ranges[taxonomy] = {
            "max": round(max(shares.values()), 6),
            "min": round(min(shares.values()), 6),
            "range": round(max(shares.values()) - min(shares.values()), 6),
        }
    checks = {
        "catalog_inner_join_only": True,
        "confirmation_sealed": bool(split_summaries["confirmation"]["sealed"]),
        "exact_split_rows": all(
            int(summary["rows"])
            == int(manifest["build"]["expected_split_counts"][split])
            for split, summary in split_summaries.items()
        ),
        "excluded_output_overlap_zero": manifest["exclusions"]["output_overlap_count"] == 0,
        "pairwise_target_overlap_zero": all(
            value == 0 for value in manifest["pairwise_split_target_overlaps"].values()
        ),
        "scenario_ratio_exact": all(
            int(summary["scenario_counts"].get("buying", 0)) * 20
            == int(summary["rows"]) * 8
            and int(summary["scenario_counts"].get("browsing", 0)) * 20
            == int(summary["rows"]) * 8
            and int(summary["scenario_counts"].get("intent_override", 0)) * 20
            == int(summary["rows"]) * 3
            and int(summary["scenario_counts"].get("boundary", 0)) * 20
            == int(summary["rows"])
            for summary in split_summaries.values()
        ),
        "test_rows_read_zero": manifest["source"]["test_rows_read"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"proxy distribution audit failed: {checks}")
    return {
        "schema_version": "track4.amazon-validation-proxy-audit.v1",
        "checks": checks,
        "coverage": {
            "eligible_source_rows": int(source_stats["eligible_source_rows"]),
            "eligible_unique_targets": int(source_stats["eligible_unique_targets"]),
            "prior_catalog_join_rate": (
                round(prior_joined / prior_total, 6) if prior_total else 0.0
            ),
            "proxy_rows": sum(int(summary["rows"]) for summary in split_summaries.values()),
            "proxy_unique_target_union": sum(
                int(summary["unique_targets"]) for summary in split_summaries.values()
            ),
        },
        "split_distributions": {
            split: {
                "difficulty_counts": summary["difficulty_counts"],
                "popularity_counts": summary["popularity_counts"],
                "repeated_target_rows": int(summary["rows"])
                - int(summary["unique_targets"]),
                "scenario_counts": summary["scenario_counts"],
                "source_weight_effective_target_count": summary[
                    "source_weight_effective_target_count"
                ],
                "source_weight_max_target_share": summary[
                    "source_weight_max_target_share"
                ],
                "source_weight_sum": summary["source_weight_sum"],
                "source_weighted_popularity_shares": summary[
                    "source_weighted_popularity_shares"
                ],
                "source_weighted_scenario_shares": summary[
                    "source_weighted_scenario_shares"
                ],
                "source_weighted_taxonomy_shares": summary[
                    "source_weighted_taxonomy_shares"
                ],
                "taxonomy_counts": summary["taxonomy_counts"],
                "unique_targets": int(summary["unique_targets"]),
            }
            for split, summary in split_summaries.items()
        },
        "taxonomy_share_ranges": taxonomy_share_ranges,
        "warnings": [
            "Most raw validation targets are outside the frozen 50k catalog; proxy coverage is an inner-join subset.",
            "Only a small fraction of prior item identifiers join the frozen catalog, so many preference-tag aggregates are neutral; purchase-frequency bins still use the raw pre-validation history length.",
            "Target repetition is allowed only within a split because fewer than 8,000 eligible unique targets exist.",
            "Use source-row micro metrics as primary and taxonomy macro metrics as a required stress view.",
            "Target-group disjointness can place a very high-frequency product in only one split; inspect source_weight_max_target_share and reject gains isolated to that concentration.",
            "Repeated rows are target-clustered, not independent observations; confidence intervals and promotion gates must resample/aggregate by target and agree with target-uniform metrics.",
        ],
    }


def build_proxy(config: ProxyBuildConfig) -> dict[str, Any]:
    root = PROJECT_ROOT
    config_path = _resolve(config.config_path, root)
    if config.production_pinned:
        expected_config_path = (PROJECT_ROOT / DEFAULT_CONFIG).resolve()
        if config_path != expected_config_path:
            raise ProxyBuildError(
                "production builds require the tracked pinned proxy config path"
            )
        if config.loaded_config_canonical_sha256 is None:
            raise ProxyBuildError("production config is missing its loaded canonical identity")
        if config != load_config(expected_config_path):
            raise ProxyBuildError(
                "production config fields differ from the tracked pinned proxy config"
            )
    if config_path is not None:
        root = config_path.parent.parent.resolve()
    validation_path = _resolve(config.validation_csv, root)
    catalog_path = _resolve(config.catalog_path, root)
    public_path = _resolve(config.public_path, root)
    output_dir = _resolve(config.output_dir, root)
    assert validation_path is not None and catalog_path is not None
    assert public_path is not None and output_dir is not None

    # This check intentionally precedes every stat/open/hash operation.
    _reject_test_source(validation_path, config.source_url)
    _validate_pinned_production_source(config)
    if config.production_pinned and not _is_within(
        output_dir, (root / "experiments" / "fast_track").resolve()
    ):
        raise ProxyBuildError(
            "production proxy output must stay under experiments/fast_track"
        )
    config_hash = None
    if config_path is not None:
        config_hash = _canonical_json_file_sha256(config_path)
        if (
            config.loaded_config_canonical_sha256 is not None
            and config_hash != config.loaded_config_canonical_sha256
        ):
            raise ProxyBuildError("proxy config changed after it was loaded")
    builder_hash = _text_file_sha256_lf(Path(__file__))
    code_provenance = (
        _git_code_provenance(
            root,
            builder_sha256_lf=builder_hash,
            config_canonical_sha256=str(config_hash),
        )
        if config.production_pinned
        else None
    )

    output_paths = {
        split: output_dir / filename for split, filename in SPLIT_FILENAMES.items()
    }
    manifest_path = output_dir / "manifest.json"
    audit_path = output_dir / "audit.json"
    manual_path = _resolve(config.manual_exclusions_path, root)
    corpus_paths = {
        _resolve(spec.path, root)
        for spec in _coerce_corpus_specs(
            config.consumed_corpora, config.expected_consumed_sha256
        )
    }
    input_paths = {
        validation_path,
        catalog_path,
        public_path,
        *(path for path in (manual_path, config_path, *corpus_paths) if path is not None),
    }
    declared_outputs = {*output_paths.values(), manifest_path, audit_path}
    if {path.resolve() for path in input_paths} & {
        path.resolve() for path in declared_outputs
    }:
        raise ProxyBuildError("proxy output path collides with a frozen input")
    existing_split = next((path for path in output_paths.values() if path.exists()), None)
    if existing_split is not None:
        raise FileExistsError(f"proxy split output already exists: {existing_split}")

    validation_hash, validation_bytes = _verify_file(
        validation_path,
        label="Amazon validation CSV",
        expected_sha256=config.expected_validation_sha256,
        expected_bytes=config.expected_validation_bytes,
    )
    catalog_hash, catalog_bytes = _verify_file(
        catalog_path,
        label="official catalog",
        expected_sha256=config.expected_catalog_sha256,
    )
    excluded, exclusion_audit = _load_exclusions(config, root, public_path)
    products = _load_catalog(catalog_path, config.expected_catalog_rows)
    representatives, source_stats = _read_validation(
        validation_path,
        products,
        excluded,
        config.expected_header,
        config.seed,
    )
    popularity = _popularity_bins(products)
    rows_by_split, target_sets = _build_rows(representatives, products, popularity, config)

    final_validation_identity = _verify_file(
        validation_path,
        label="Amazon validation CSV post-parse",
        expected_sha256=config.expected_validation_sha256,
        expected_bytes=config.expected_validation_bytes,
    )
    final_catalog_identity = _verify_file(
        catalog_path,
        label="official catalog post-parse",
        expected_sha256=config.expected_catalog_sha256,
    )
    final_excluded, final_exclusion_audit = _load_exclusions(config, root, public_path)
    if (
        final_validation_identity != (validation_hash, validation_bytes)
        or final_catalog_identity != (catalog_hash, catalog_bytes)
        or final_excluded != excluded
        or final_exclusion_audit != exclusion_audit
    ):
        raise ProxyBuildError("a frozen input changed while the proxy was being built")
    if config_path is not None and _canonical_json_file_sha256(config_path) != config_hash:
        raise ProxyBuildError("proxy config changed while the proxy was being built")
    if _text_file_sha256_lf(Path(__file__)) != builder_hash:
        raise ProxyBuildError("proxy builder source changed while it was running")
    if config.production_pinned and _git_code_provenance(
        root,
        builder_sha256_lf=builder_hash,
        config_canonical_sha256=str(config_hash),
    ) != code_provenance:
        raise ProxyBuildError("proxy committed code provenance changed during the build")

    overlaps = {
        f"{left}__{right}": len(target_sets[left] & target_sets[right])
        for left, right in combinations(SPLIT_FILENAMES, 2)
    }
    if any(overlaps.values()):
        raise RuntimeError("proxy target splits are not disjoint")
    if any(targets & excluded for targets in target_sets.values()):
        raise RuntimeError("excluded target escaped into proxy output")
    exclusion_audit["output_overlap_count"] = sum(
        len(targets & excluded) for targets in target_sets.values()
    )

    payloads = {
        output_paths[split]: _canonical_jsonl_bytes(rows)
        for split, rows in rows_by_split.items()
    }
    split_summaries = {
        split: {
            "filename": SPLIT_FILENAMES[split],
            **_split_summary(
                rows_by_split[split],
                target_sets[split],
                payloads[output_paths[split]],
                sealed=split == "confirmation",
            ),
        }
        for split in SPLIT_FILENAMES
    }
    total_source_weight = sum(
        int(record["source_weight"]) for record in representatives.values()
    )
    outlier_count = sum(
        1
        for record in representatives.values()
        if total_source_weight
        and int(record["source_weight"]) / total_source_weight
        >= config.source_frequency_outlier_train_threshold
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build": {
            "allocation_algorithm": (
                "taxonomy/popularity-stratified SHA-256 target groups; breadth-first source-occurrence quotas"
            ),
            "builder_source_sha256": builder_hash,
            "builder_source_hash_mode": "SHA-256 over source bytes after CRLF/CR to LF normalization",
            "code_provenance": code_provenance,
            "config_canonical_sha256": config_hash,
            "config_hash_mode": "SHA-256 over canonical UTF-8 JSON with sorted keys, compact separators, and an LF terminator",
            "expected_split_counts": {
                name: int(config.split_counts[name]) for name in SPLIT_FILENAMES
            },
            "inputs_reverified_after_parse": True,
            "source_frequency_outlier_policy": {
                "destination": "train_explore",
                "observed_target_count": outlier_count,
                "threshold_share": config.source_frequency_outlier_train_threshold,
            },
            "representative_algorithm": (
                "safe-profile histogram per target; deterministic occurrence ordering without raw source fields"
            ),
            "seed": config.seed,
        },
        "catalog": {
            "bytes": catalog_bytes,
            "product_count": len(products),
            "sha256": catalog_hash,
        },
        "evaluation_views": {
            "primary": "source-weighted micro metrics using source_weight",
            "secondary": "target-uniform metrics to expose source-frequency concentration",
            "stress": "equal-weight macro mean across the four taxonomy buckets on the same rows",
        },
        "exclusions": exclusion_audit,
        "pairwise_split_target_overlaps": overlaps,
        "privacy": {
            "average_prior_rating": None,
            "raw_amazon_user_ids_retained": False,
            "raw_prior_item_sequences_retained": False,
            "target_validation_rating_used": False,
            "target_validation_timestamp_used": False,
        },
        "profile_derivation": {
            "fields": ["purchase_frequency", "average_prior_rating", "rating_style", "preference_tags", "summary"],
            "preference_tags": "up to three catalog-only coarse tags supported by at least two prior items",
            "purchase_frequency": "binned from pre-validation history length without persisting prior identifiers",
            "ratings": "not used",
        },
        "source": {
            "bytes": validation_bytes,
            "category": "Clothing_Shoes_and_Jewelry",
            "dataset": "Amazon Reviews 2023 5-core leave-last-out",
            "header": list(config.expected_header),
            "license_note": config.source_license_note,
            "redistributable": False,
            "revision": config.source_revision,
            "role": "validation",
            "sha256": validation_hash,
            "split": "validation_only",
            "stats": source_stats,
            "test_rows_read": 0,
            "url": config.source_url,
        },
        "splits": split_summaries,
        "limitations": [
            "Validation-derived proxy only; it is not a byte-identical or distribution-identical reconstruction of organizer-private evaluation.",
            "Raw and derived Amazon rows remain local because redistribution rights are unclear.",
            "The frozen-catalog intersection has fewer than 8,000 unique eligible targets, so a target may repeat only within its assigned split; target groups remain pairwise disjoint.",
            "Targets representing at least the configured share of eligible source rows are assigned to train/explore before hash splitting; held-out source-weighted metrics therefore describe the non-extreme population.",
            "The confirmation split is materialized but sealed from generic runners unless explicitly authorized.",
            "Six-file publication uses exclusive creation and exception rollback but is not crash-atomic; an interrupted partial build requires integrity review before rebuilding.",
        ],
    }
    audit = _build_distribution_audit(manifest)
    audit_payload = _canonical_json_bytes(audit, pretty=True)
    manifest["audit"] = {
        "filename": audit_path.name,
        "sha256": _sha256_bytes(audit_payload),
    }
    manifest_payload = _canonical_json_bytes(manifest, pretty=True)
    all_payloads = {
        **payloads,
        audit_path: audit_payload,
        manifest_path: manifest_payload,
    }
    _exclusive_publish(
        all_payloads,
        allow_identical_existing=(manifest_path, audit_path),
    )
    return manifest


def load_config(path: Path | str) -> ProxyBuildConfig:
    config_path = Path(path).resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProxyBuildError(f"cannot load proxy config: {config_path}") from error
    if not isinstance(value, dict):
        raise ProxyBuildError("proxy config must be a JSON object")
    loaded_config_hash = _sha256_bytes(_canonical_json_bytes(value))

    # The tracked production config is deliberately descriptive.  A compact
    # flat form is accepted only by the programmatic API for unit-test fixtures.
    if "source" in value:
        _validate_nested_production_config(value)
        source = value["source"]
        catalog = value["catalog"]
        exclusions = value["exclusions"]
        selection = value["selection"]
        outputs = value["outputs"]
        corpus_values = list(exclusions["consumed_corpora"])
        public_values = [item for item in corpus_values if item.get("id") == "released_public"]
        if len(public_values) != 1:
            raise ProxyBuildError("production config must identify exactly one released_public corpus")
        public_value = public_values[0]
        consumed = [
            {
                "name": str(item["id"]),
                "path": str(item["path"]),
                "expected_sha256": str(item["file_sha256"]),
                "expected_rows": int(item["rows"]),
            }
            for item in corpus_values
            if item.get("id") != "released_public"
        ]
        manual = exclusions["manual_target_ledger"]
        split_counts = {
            name: int(spec["rows"]) for name, spec in outputs["splits"].items()
        }
        common_scenarios = selection["scenario_assignment"]["counts_per_split"]
        header = source.get("fields") or str(source["header"]).split(",")
        return ProxyBuildConfig(
            validation_csv=str(source["path"]),
            catalog_path=str(catalog["path"]),
            public_path=str(public_value["path"]),
            manual_exclusions_path=str(manual["path"]),
            output_dir=str(outputs["root"]),
            seed=str(selection["seed"]),
            split_counts=split_counts,
            expected_validation_sha256=str(source["sha256"]),
            expected_catalog_sha256=str(catalog["sha256"]),
            expected_validation_bytes=int(source["bytes"]),
            expected_catalog_rows=int(catalog["rows"]),
            expected_header=tuple(str(item) for item in header),
            scenario_counts={name: dict(common_scenarios) for name in split_counts},
            source_frequency_outlier_train_threshold=float(
                selection.get("source_frequency_outlier_train_threshold", 1.0)
            ),
            consumed_corpora=tuple(consumed),
            expected_public_sha256=str(public_value["file_sha256"]),
            expected_consumed_union_count=int(exclusions["consumed_target_union_count"]),
            expected_manual_exclusions_sha256=manual.get("file_sha256"),
            expected_manual_count=int(manual["expected_count"]),
            source_url=str(source["url"]),
            source_revision=str(source["revision"]),
            source_license_note=SOURCE_LICENSE_NOTE,
            config_path=config_path,
            loaded_config_canonical_sha256=loaded_config_hash,
            production_pinned=True,
        )

    return ProxyBuildConfig(
        validation_csv=value["validation_csv"],
        catalog_path=value["catalog_path"],
        public_path=value["public_path"],
        manual_exclusions_path=value.get("manual_exclusions_path"),
        output_dir=value["output_dir"],
        seed=str(value["seed"]),
        split_counts=value["split_counts"],
        expected_validation_sha256=str(value["expected_validation_sha256"]),
        expected_catalog_sha256=str(value["expected_catalog_sha256"]),
        expected_validation_bytes=(
            int(value["expected_validation_bytes"])
            if value.get("expected_validation_bytes") is not None
            else None
        ),
        expected_catalog_rows=int(value.get("expected_catalog_rows", 50_000)),
        expected_header=tuple(value.get("expected_header", EXPECTED_HEADER)),
        scenario_counts=value.get("scenario_counts", {}),
        source_frequency_outlier_train_threshold=float(
            value.get("source_frequency_outlier_train_threshold", 1.0)
        ),
        consumed_corpora=tuple(value.get("consumed_corpora", ())),
        expected_public_sha256=value.get("expected_public_sha256"),
        expected_consumed_sha256=tuple(value.get("expected_consumed_sha256", ())),
        expected_consumed_union_count=(
            int(value["expected_consumed_union_count"])
            if value.get("expected_consumed_union_count") is not None
            else None
        ),
        expected_manual_exclusions_sha256=value.get("expected_manual_exclusions_sha256"),
        expected_manual_count=(
            int(value["expected_manual_count"])
            if value.get("expected_manual_count") is not None
            else None
        ),
        source_url=str(value.get("source_url", "")),
        source_revision=str(value.get("source_revision", "")),
        source_license_note=str(value.get("source_license_note", SOURCE_LICENSE_NOTE)),
        config_path=config_path,
        loaded_config_canonical_sha256=loaded_config_hash,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    if not config.production_pinned:
        raise ProxyBuildError("CLI accepts only the pinned production proxy config")
    manifest = build_proxy(config)
    print(
        "[amazon-validation-proxy] "
        + " ".join(
            f"{name}={summary['row_count']}:{summary['file_sha256']}"
            for name, summary in manifest["splits"].items()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
