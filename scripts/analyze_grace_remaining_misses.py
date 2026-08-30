"""One-shot anonymous attribution of misses left by fixed two-page grace.

All catalog, trace, feature, and score work precedes the durable receipt and is
target-free.  The numeric label archive and train_explore proxy are opened only
after that receipt, solely for evaluator-side aggregate attribution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat as stat_module
import subprocess
import sys
import time
from typing import Any, BinaryIO, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_rank1_score_priority_replacement as replay  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402

feature_schema = replay.base


SCHEMA_VERSION = "small-ranker-grace-miss-attribution.v1"
EXPERIMENT_ID = "SR-V2.14-GRACE-MISS-ATTRIBUTION"
BRANCH = "small-ranker-v2.14-grace-miss-attribution"
REMOTE_REF = "refs/remotes/origin/" + BRANCH
REMOTE_URL = "https://github.com/lamperriat/techjam-err402.git"
AMENDMENT_COMMIT = "4f015f6cbb867a60e6d56cd17aa127931885a4b1"
AMENDMENT_RAW_SHA256 = "28921c699ab057468bf06e5b60bf2d55e44bd43d27f7b21d2239cfd71e5aafce"
AMENDMENT_V2_COMMIT = "cd8af48a4ce8fab63dac08b80d1354bdc0057de1"
AMENDMENT_V2_BLOB = "329a40f23c193458a7bd44634606fee908fcb9bc"
AMENDMENT_V2_RAW_SHA256 = "1be83ff651cdc38387da44d5fb58ae216c6c55800d2ca3338061c759f521a7e5"
PREREG_PATH = ROOT / "configs/small_ranker_v2_14.grace_remaining_miss_attribution_preregistration.json"
CONTRACT_PATH = ROOT / "configs/small_ranker_v2_14.grace_remaining_miss_attribution_implementation_contract.json"
AMENDMENT_PATH = ROOT / "configs/small_ranker_v2_14.grace_remaining_miss_attribution_contract_amendment.json"
AMENDMENT_V2_PATH = ROOT / "configs/small_ranker_v2_14.grace_remaining_miss_attribution_contract_amendment_v2.json"
PREREG_RAW_SHA256 = "986c4cf07fb589e76c2e352b3f559d80c8920e0b8fca2943eeb6e26bceab1be8"
CONTRACT_RAW_SHA256 = "93fa610eccd620fde8a0df162d2c6aee26e6ad6d4b96739d5c6208f4f8feca34"
IMPLEMENTATION_PATHS = {
    "scripts/analyze_grace_remaining_misses.py",
    "tests/test_grace_remaining_miss_attribution.py",
}
IMPLEMENTATION_BASE_COMMIT = "7fa028876ddff4ab285c599c4c71896157cdddc9"
CORRECTION_PATHS = {"scripts/analyze_grace_remaining_misses.py"}

SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
LABEL_PATH = SOURCE_ROOT / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
PROXY_PATH = SOURCE_ROOT / "experiments/fast_track/proxy_v1/proxy_train_explore.jsonl"
CATALOG_PATH = SOURCE_ROOT / "data/catalog.jsonl"
OUTPUT_PATH = ROOT / "experiments/fast_track/small_ranker_v2_14/grace_remaining_miss_attribution_20260831/result.json"
LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
LABEL_BYTES = 1_702_876
PROXY_SHA256 = "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e"
PROXY_BYTES = 1_315_338
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
CATALOG_BYTES = 60_546_327
EXPECTED_CHOSEN_SHA256 = "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
EXPECTED_ACTIVATION_SHA256 = "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
EXPECTED_BASELINE_ASCII_SHA256 = "2d5fa0ea12ab02b74e2b6c3a3f92b3ba83191c35eb176f95aed9361f785496e9"
EXPECTED_GRACE_ASCII_SHA256 = "6f84d2aa03d792027634a358bdcef9adf646717c94cc3ec0916e87e33410448a"
EXPECTED_FOLD_HITS = (398, 395, 397, 395, 397)
EXPECTED_FOLD_MISSES = (2, 5, 3, 5, 3)
SESSION_COUNT = 2_000
TURN_COUNT = 10
CANDIDATE_COUNT = 100
RESOURCE_BYTES_MAXIMUM = 2_147_483_648
RESOURCE_SECONDS_MAXIMUM = 120.0
DEPTHS = (10, 20, 50, 100)
SLOTS = ("category", "material", "color", "size", "style", "brand", "price", "feature", "use_case")
PRIMARY_ORDER = (
    "candidate_absent_at_c100",
    "admission_grace_state_rejection",
    "candidate_present_but_ranker_failure",
)
DECISION_TIE_ORDER = (
    "candidate_absent_at_c100",
    "candidate_present_but_ranker_failure",
    "admission_grace_state_rejection",
)
ASIN_SHAPE_RE = re.compile(rb"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)


class GraceAttributionError(RuntimeError):
    pass


class GraceAttributionConsumedError(GraceAttributionError):
    pass


@dataclass(frozen=True)
class Preflight:
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    protocol: Mapping[str, str]
    inputs: Any
    catalog_flags: Mapping[str, int]
    catalog_snapshot: tuple[int, int, int]
    memory: tuple[int, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sha256_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _handle_identity(handle: BinaryIO) -> tuple[int, int, int]:
    stat = os.fstat(handle.fileno())
    return int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0))


def candidate_recall_flags(positive: np.ndarray, eligible_from: int) -> dict[int, bool]:
    values = np.asarray(positive)
    if values.shape != (TURN_COUNT,) or not 1 <= int(eligible_from) <= TURN_COUNT:
        raise GraceAttributionError("candidate recall input schema failed")
    eligible_values = values[int(eligible_from) - 1 :]
    if np.any((eligible_values < -1) | (eligible_values >= CANDIDATE_COUNT)):
        raise GraceAttributionError("candidate ordinal escaped C100")
    return {
        depth: bool(np.any((eligible_values >= 0) & (eligible_values < depth)))
        for depth in DEPTHS
    }


def classify_primary(
    *, c100_reachable: bool, admission_rejected: bool, state_rejected: bool
) -> str:
    if not c100_reachable:
        return PRIMARY_ORDER[0]
    if admission_rejected or state_rejected:
        return PRIMARY_ORDER[1]
    return PRIMARY_ORDER[2]


def largest_primary_bottleneck(counts: Mapping[str, int]) -> str:
    if set(counts) != set(PRIMARY_ORDER) or any(int(counts[name]) < 0 for name in PRIMARY_ORDER):
        raise GraceAttributionError("primary count schema failed")
    return max(
        DECISION_TIE_ORDER,
        key=lambda name: (int(counts[name]), -DECISION_TIE_ORDER.index(name)),
    )


def admission_rejected_flag(
    positive: np.ndarray,
    eligible_from: int,
    chosen: np.ndarray,
    action: np.ndarray,
    activation: np.ndarray,
) -> bool:
    positive = np.asarray(positive)
    chosen = np.asarray(chosen)
    action = np.asarray(action, dtype=bool)
    activation = np.asarray(activation, dtype=bool)
    if (
        positive.shape != (TURN_COUNT,)
        or chosen.shape != (TURN_COUNT,)
        or action.shape != (TURN_COUNT,)
        or activation.shape != (TURN_COUNT,)
        or not 1 <= int(eligible_from) <= TURN_COUNT
    ):
        raise GraceAttributionError("admission input schema failed")
    eligible = np.arange(TURN_COUNT) >= int(eligible_from) - 1
    return bool(
        np.any(
            eligible
            & action
            & (positive >= 0)
            & (chosen == positive)
            & (~activation)
        )
    )


def lifecycle_flag(
    positive: np.ndarray, grace_pages: np.ndarray, eligible_from: int
) -> bool:
    values = np.asarray(positive)
    pages = np.asarray(grace_pages)
    if (
        values.shape != (TURN_COUNT,)
        or pages.shape != (TURN_COUNT, 10)
        or not 1 <= int(eligible_from) <= TURN_COUNT
    ):
        raise GraceAttributionError("lifecycle input schema failed")
    boundary = int(eligible_from) - 1
    pre_raw = bool(np.any(values[:boundary] >= 0))
    pre_page = any(
        int(values[turn]) >= 0 and int(values[turn]) in pages[turn]
        for turn in range(boundary)
    )
    eligible_absent = bool(np.all(values[boundary:] < 0))
    return bool((pre_raw or pre_page) and eligible_absent)


def information_insufficient(
    features: np.ndarray, positive: np.ndarray, eligible_from: int
) -> dict[str, bool]:
    values = np.asarray(features)
    positive = np.asarray(positive)
    if (
        values.shape != (TURN_COUNT, CANDIDATE_COUNT, len(feature_schema.FEATURE_NAMES))
        or positive.shape != (TURN_COUNT,)
        or not 1 <= int(eligible_from) <= TURN_COUNT
    ):
        raise GraceAttributionError("information input schema failed")
    start = int(eligible_from) - 1
    query = values[start:, 0, feature_schema.FEATURE_INDEX["query_specificity_fraction"]]
    active = values[start:, 0, feature_schema.FEATURE_INDEX["active_constraint_count_fraction"]]
    query_zero = bool(np.all((query == 0.0) | (active == 0.0)))
    compatible_columns = [feature_schema.FEATURE_INDEX[f"{slot}_compatible"] for slot in SLOTS]
    unknown_columns = [feature_schema.FEATURE_INDEX[f"{slot}_unknown"] for slot in SLOTS]
    missing_column = feature_schema.FEATURE_INDEX["missing_positive_evidence_fraction"]
    reachable = [turn for turn in range(start, TURN_COUNT) if int(positive[turn]) >= 0]
    attribute_missing = bool(
        reachable
        and all(
            float(values[turn, int(positive[turn]), compatible_columns].sum()) == 0.0
            and (
                float(values[turn, int(positive[turn]), missing_column]) > 0.0
                or float(values[turn, int(positive[turn]), unknown_columns].sum()) > 0.0
            )
            for turn in reachable
        )
    )
    return {
        "query_zero": query_zero,
        "attribute_missing": attribute_missing,
        "combined": bool(query_zero or attribute_missing),
    }


def _flatten_metadata(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            flattened.extend(_flatten_metadata(key))
            flattened.extend(_flatten_metadata(value[key]))
        return tuple(flattened)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(part for item in value for part in _flatten_metadata(item))
    text = str(value).strip()
    return (text,) if text else ()


def catalog_missing_fields(product: Mapping[str, Any] | None) -> dict[str, bool]:
    if product is None:
        return {
            "row_absent": True,
            "title_empty": False,
            "categories_empty": False,
            "descriptive_evidence_empty": False,
            "price_null": False,
            "any_missing": True,
        }
    title_empty = not _flatten_metadata(product.get("title"))
    categories_empty = not _flatten_metadata(product.get("categories"))
    descriptive_empty = not any(
        _flatten_metadata(product.get(name))
        for name in ("features", "description", "details", "store")
    )
    price_null = product.get("price") is None
    flags = {
        "row_absent": False,
        "title_empty": bool(title_empty),
        "categories_empty": bool(categories_empty),
        "descriptive_evidence_empty": bool(descriptive_empty),
        "price_null": bool(price_null),
    }
    return {**flags, "any_missing": any(flags.values())}


def candidate_frontier_pending() -> dict[str, Any]:
    reason = "frozen blind trace ends at exact C100"
    return {
        "present_by_c200": {"count": None, "status": "pending_not_observed", "reason": reason},
        "present_by_c400": {"count": None, "status": "pending_not_observed", "reason": reason},
        "absent_at_c400": {"count": None, "status": "pending_not_observed", "reason": reason},
        "rule": "do not infer C200/C400 from C100 or unrelated adaptive-depth artifacts",
    }


def privacy_scan(result: object) -> None:
    forbidden_tokens = {
        "session",
        "sample",
        "product",
        "target",
        "message",
    }
    forbidden_exact = {
        "parent_asin",
        "positive_index",
        "eligible_from",
        "per_session",
        "outcome_by_session",
    }
    forbidden_value_exact = {"parent_asin", "positive_index", "eligible_from"}

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                folded = str(key).casefold()
                tokens = {token for token in re.split(r"[^a-z0-9]+", folded) if token}
                if folded in forbidden_exact or tokens & forbidden_tokens:
                    raise GraceAttributionError("result contains a forbidden key")
                walk(child)
        elif isinstance(value, np.ndarray):
            raise GraceAttributionError("result contains an ndarray")
        elif isinstance(value, str) and value.casefold() in forbidden_value_exact:
            raise GraceAttributionError("result contains a forbidden schema name")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) >= SESSION_COUNT:
                raise GraceAttributionError("result contains a cohort-length vector")
            for child in value:
                walk(child)

    walk(result)
    if ASIN_SHAPE_RE.search(_canonical_bytes(result)):
        raise GraceAttributionError("result contains an identifier-shaped token")


def _catalog_mask(flags: Mapping[str, bool]) -> int:
    names = ("row_absent", "title_empty", "categories_empty", "descriptive_evidence_empty", "price_null")
    return sum((1 << index) for index, name in enumerate(names) if flags[name])


def _mask_flags(mask: int) -> dict[str, bool]:
    names = ("row_absent", "title_empty", "categories_empty", "descriptive_evidence_empty", "price_null")
    flags = {name: bool(mask & (1 << index)) for index, name in enumerate(names)}
    return {**flags, "any_missing": any(flags.values())}


def _load_catalog_flags() -> tuple[dict[str, int], tuple[int, int, int]]:
    if CATALOG_PATH.is_symlink():
        raise GraceAttributionError("catalog path is a symlink")
    flags: dict[str, int] = {}
    digest = hashlib.sha256()
    with CATALOG_PATH.open("rb") as handle:
        identity = _handle_identity(handle)
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            if not isinstance(row, dict):
                raise GraceAttributionError("catalog row is invalid")
            identifier = row.get("parent_asin")
            if not isinstance(identifier, str) or not identifier or identifier in flags:
                raise GraceAttributionError("catalog identity is invalid")
            flags[identifier] = _catalog_mask(catalog_missing_fields(row))
        end_identity = _handle_identity(handle)
    if (
        identity != end_identity
        or identity[0] != CATALOG_BYTES
        or digest.hexdigest() != CATALOG_SHA256
        or len(flags) != 50_000
    ):
        raise GraceAttributionError("catalog binding drifted")
    return flags, identity


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GraceAttributionError("duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise GraceAttributionError("protocol JSON is invalid")
    return value


def _validate_protocol() -> dict[str, str]:
    expected = (
        (PREREG_PATH, PREREG_RAW_SHA256, "small-ranker-grace-miss-attribution-preregistration.v1"),
        (CONTRACT_PATH, CONTRACT_RAW_SHA256, "small-ranker-grace-miss-attribution-implementation-contract.v1"),
        (AMENDMENT_PATH, AMENDMENT_RAW_SHA256, "small-ranker-grace-miss-attribution-contract-amendment.v1"),
        (AMENDMENT_V2_PATH, AMENDMENT_V2_RAW_SHA256, "small-ranker-grace-miss-attribution-contract-amendment.v2"),
    )
    result: dict[str, str] = {}
    for path, expected_hash, schema in expected:
        raw = path.read_bytes()
        value = _load_json_no_duplicates(path)
        if hashlib.sha256(raw).hexdigest() != expected_hash or value.get("schema_version") != schema:
            raise GraceAttributionError("protocol identity drifted")
        result[path.name] = expected_hash
    if (
        _load_json_no_duplicates(AMENDMENT_PATH).get("direct_parent")
        != "a07b22b3e343c9ea690c9fa9c942c2abc14628b9"
        or _load_json_no_duplicates(AMENDMENT_V2_PATH).get("direct_parent")
        != AMENDMENT_COMMIT
    ):
        raise GraceAttributionError("protocol lineage drifted")
    return result


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise GraceAttributionError("Git checkpoint validation failed")
    return completed.stdout.strip()


def _validate_git(implementation_commit: str) -> dict[str, Any]:
    head = _git(("rev-parse", "HEAD"))
    parent = _git(("rev-parse", "HEAD^"))
    branch = _git(("branch", "--show-current"))
    remote = _git(("remote", "get-url", "origin"))
    remote_head = _git(("rev-parse", REMOTE_REF))
    status = _git(("status", "--porcelain=v1", "--untracked-files=all"))
    paths = set(_git(("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")).splitlines())
    base_parent = _git(("rev-parse", IMPLEMENTATION_BASE_COMMIT + "^"))
    base_paths = set(
        _git(
            (
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                IMPLEMENTATION_BASE_COMMIT,
            )
        ).splitlines()
    )
    amendment_blob = _git(("rev-parse", AMENDMENT_V2_COMMIT + ":" + AMENDMENT_V2_PATH.relative_to(ROOT).as_posix()))
    if not (
        implementation_commit == head
        and parent == IMPLEMENTATION_BASE_COMMIT
        and base_parent == AMENDMENT_V2_COMMIT
        and branch == BRANCH
        and remote.rstrip("/").removesuffix(".git") == REMOTE_URL.removesuffix(".git")
        and remote_head == head
        and not status
        and paths == CORRECTION_PATHS
        and base_paths == IMPLEMENTATION_PATHS
        and amendment_blob == AMENDMENT_V2_BLOB
    ):
        raise GraceAttributionError("implementation Git checkpoint drifted")
    return {
        "commit": head,
        "parent": parent,
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "branch": branch,
        "remote_equal": True,
        "clean": True,
        "paths_exact": True,
    }


def _check_output() -> None:
    root = ROOT.resolve(strict=True)
    parent = OUTPUT_PATH.parent.resolve(strict=False)
    if parent != root and root not in parent.parents:
        raise GraceAttributionError("receipt escapes worktree")
    current = ROOT
    for part in OUTPUT_PATH.relative_to(ROOT).parts[:-1]:
        current /= part
        if current.exists() and _is_link_or_reparse(current):
            raise GraceAttributionError("receipt parent is a link or reparse point")
    if not OUTPUT_PATH.parent.is_dir() or _is_link_or_reparse(OUTPUT_PATH.parent):
        raise GraceAttributionError("receipt parent must be a prepared real directory")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise GraceAttributionError("attribution receipt is already consumed")


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    marker = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _query_columns_identical(features: np.ndarray) -> bool:
    for name in ("query_specificity_fraction", "active_constraint_count_fraction"):
        column = feature_schema.FEATURE_INDEX[name]
        values = np.asarray(features[..., column])
        if not np.array_equal(values, np.broadcast_to(values[..., :1], values.shape)):
            return False
    return True


def preflight_only(implementation_commit: str) -> Preflight:
    environment = replay._validate_environment()
    protocol = _validate_protocol()
    git = _validate_git(implementation_commit)
    _check_output()
    inputs = replay._load_target_free_inputs()
    if (
        hashlib.sha256(np.ascontiguousarray(inputs.chosen).tobytes()).hexdigest() != EXPECTED_CHOSEN_SHA256
        or not _query_columns_identical(inputs.projected_features)
        or not replay._validate_source_snapshots(inputs.source_snapshots)
    ):
        raise GraceAttributionError("target-free surface drifted")
    catalog_flags, catalog_snapshot = _load_catalog_flags()
    working, peak = replay._process_memory()
    if not (0 < working <= RESOURCE_BYTES_MAXIMUM and 0 < peak <= RESOURCE_BYTES_MAXIMUM):
        raise GraceAttributionError("pre-receipt memory gate failed")
    return Preflight(
        environment,
        git,
        protocol,
        inputs,
        catalog_flags,
        catalog_snapshot,
        (working, peak),
    )


def _write_descriptor(descriptor: int, value: Mapping[str, Any]) -> tuple[int, str]:
    payload = _canonical_bytes(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short receipt write")
        view = view[written:]
    os.fsync(descriptor)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _open_receipt(implementation_commit: str) -> int:
    _check_output()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    try:
        descriptor = os.open(str(OUTPUT_PATH), flags, 0o600)
        _write_descriptor(
            descriptor,
            {
                "schema_version": "small-ranker-grace-miss-attribution-marker.v1",
                "experiment_id": EXPERIMENT_ID,
                "implementation_commit": implementation_commit,
                "status": "CONSUMED_PENDING_RERUN_FORBIDDEN",
            },
        )
    except BaseException as error:
        if descriptor is not None:
            _invalid_receipt(descriptor, implementation_commit, error)
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise GraceAttributionConsumedError(
                "receipt creation failed after consumption; rerun is forbidden"
            ) from error
        if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
            raise GraceAttributionConsumedError(
                "receipt creation may have consumed the path; rerun is forbidden"
            ) from error
        raise
    if descriptor is None:
        raise GraceAttributionError("receipt descriptor was not created")
    return descriptor


def _invalid_receipt(descriptor: int, implementation_commit: str, error: BaseException) -> None:
    try:
        _write_descriptor(
            descriptor,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "implementation_commit": implementation_commit,
                "status": "INVALID_ATTRIBUTION_CONSUMED",
                "error_class": type(error).__name__,
                "rerun_forbidden": True,
            },
        )
    except BaseException:
        pass


def _load_proxy(handle: BinaryIO) -> tuple[list[str], tuple[int, int, int], str]:
    identity = _handle_identity(handle)
    digest = hashlib.sha256()
    values: list[str] = []
    handle.seek(0)
    for line in handle:
        digest.update(line)
        row = json.loads(line)
        truth = row.get("ground_truth") if isinstance(row, dict) else None
        identifier = truth.get("parent_asin") if isinstance(truth, dict) else None
        if not isinstance(identifier, str) or not identifier:
            raise GraceAttributionError("proxy binding row is invalid")
        values.append(identifier)
    if (
        identity[0] != PROXY_BYTES
        or digest.hexdigest() != PROXY_SHA256
        or len(values) != SESSION_COUNT
        or _handle_identity(handle) != identity
    ):
        raise GraceAttributionError("proxy binding drifted")
    return values, identity, digest.hexdigest()


def _reproduce_activation(surface: Any, labels: Mapping[str, np.ndarray]) -> tuple[np.ndarray, Mapping[str, Any]]:
    original = replay.base._fit_gate_model
    constants = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal constants
        result = original(*args, **kwargs)
        constants += int(isinstance(result[0], replay.base._ConstantGate))
        return result

    replay.base._fit_gate_model = counted
    try:
        activation, _selections, audit = replay._reproduce_nested_activation(surface, labels, replay.BASE_SEED)
    finally:
        replay.base._fit_gate_model = original
    digest = hashlib.sha256(np.ascontiguousarray(activation).tobytes()).hexdigest()
    if digest != EXPECTED_ACTIVATION_SHA256 or constants != 0:
        raise GraceAttributionError("comparator activation drifted")
    if not (
        audit.get("fit_invocations") == 60
        and audit.get("inner_fit_invocations") == 50
        and audit.get("outer_fit_invocations") == 10
        and audit.get("inner_quantile_selection_invocations") == 5
        and tuple(audit.get("ordered_fold_quantiles", ())) == replay.EXPECTED_FOLD_QUANTILES
    ):
        raise GraceAttributionError("comparator reproduction audit drifted")
    return activation, {**audit, "constant_fits": constants, "activation_raw_sha256": digest}


def _eligibility_reset_grace_pages(
    traces: Sequence[Sequence[Mapping[str, Any]]],
    chosen: np.ndarray,
    activation: np.ndarray,
    versions: np.ndarray,
    eligible_values: np.ndarray,
) -> np.ndarray:
    versions = np.asarray(versions)
    chosen = np.asarray(chosen)
    activation = np.asarray(activation, dtype=bool)
    eligible_values = np.asarray(eligible_values)
    if (
        versions.ndim != 2
        or chosen.shape != versions.shape
        or activation.shape != versions.shape
        or eligible_values.shape != (versions.shape[0],)
        or len(traces) != versions.shape[0]
        or any(len(turns) != versions.shape[1] for turns in traces)
        or np.any((eligible_values < 1) | (eligible_values > versions.shape[1]))
    ):
        raise GraceAttributionError("state counterfactual schema failed")
    cohort_count, turn_count = versions.shape
    pages = np.empty((cohort_count, turn_count, 10), dtype=np.int16)
    for cohort_index, turns in enumerate(traces):
        served: set[str] = set()
        last_version: int | None = None
        intent_age = 0
        boundary = int(eligible_values[cohort_index])
        for turn_index, turn in enumerate(turns):
            version = int(versions[cohort_index, turn_index])
            if last_version is None or version != last_version:
                served.clear()
                intent_age = 1
            else:
                intent_age += 1
            if turn_index + 1 == boundary:
                served.clear()
            order = replay.reconstruct_v19_order(
                turn,
                int(chosen[cohort_index, turn_index]),
                bool(activation[cohort_index, turn_index]),
            )
            raw = tuple(str(item) for item in turn["c100"])
            page = order[:10] if intent_age <= 2 else replay._stable_unseen_first(order, served)
            raw_index = {identifier: index for index, identifier in enumerate(raw)}
            pages[cohort_index, turn_index] = [raw_index[identifier] for identifier in page]
            served.update(page)
            last_version = version
    pages.setflags(write=False)
    return pages


def _proxy_matches_numeric(
    targets: Sequence[str], traces: Sequence[Sequence[Mapping[str, Any]]], positive: np.ndarray
) -> bool:
    for cohort_index, turns in enumerate(traces):
        identifier = targets[cohort_index]
        for turn_index, turn in enumerate(turns):
            try:
                observed = tuple(turn["c100"]).index(identifier)
            except ValueError:
                observed = -1
            if observed != int(positive[cohort_index, turn_index]):
                return False
    return True


def _recall_summary(positive: np.ndarray, eligible: np.ndarray, indices: Sequence[int]) -> dict[str, Any]:
    counts = Counter({depth: 0 for depth in DEPTHS})
    for cohort_index in indices:
        for depth, value in candidate_recall_flags(positive[cohort_index], int(eligible[cohort_index])).items():
            counts[depth] += int(value)
    total = len(indices)
    return {
        f"c{depth}": {"count": int(counts[depth]), "fraction": round(counts[depth] / total, 6) if total else None}
        for depth in DEPTHS
    }


def _aggregate(
    primary: Sequence[str],
    folds: Sequence[int],
    info: Sequence[Mapping[str, bool]],
    lifecycle: Sequence[bool],
    catalog: Sequence[Mapping[str, bool]],
) -> dict[str, Any]:
    primary_counts = Counter(primary)
    fold_rows = []
    for fold in range(5):
        fold_counts = Counter(value for value, assigned in zip(primary, folds) if assigned == fold)
        fold_rows.append({"fold": fold, **{name: int(fold_counts[name]) for name in PRIMARY_ORDER}})
    info_counts = {
        name: sum(int(row[name]) for row in info)
        for name in ("query_zero", "attribute_missing", "combined")
    }
    catalog_counts = {
        name: sum(int(row[name]) for row in catalog)
        for name in ("row_absent", "title_empty", "categories_empty", "descriptive_evidence_empty", "price_null", "any_missing")
    }
    total = len(primary)
    return {
        "primary": {
            name: {"count": int(primary_counts[name]), "fraction": round(primary_counts[name] / total, 6)}
            for name in PRIMARY_ORDER
        },
        "primary_by_outer_fold": fold_rows,
        "diagnostics": {
            "old_intent_or_question_lifecycle": {"count": sum(map(int, lifecycle)), "nonexclusive": True},
            "query_or_attribute_information_insufficient": {**info_counts, "nonexclusive": True},
            "catalog_metadata_missing": {**catalog_counts, "nonexclusive": True},
        },
    }


def _rehash_target_free(preflight: Preflight) -> bool:
    if not replay._validate_source_snapshots(preflight.inputs.source_snapshots):
        return False
    checks = (
        (replay.PROJECTED_FEATURES_PATH, replay.PROJECTED_FEATURES_SHA256),
        (replay.OOF_SCORES_PATH, replay.OOF_SCORES_SHA256),
        (replay.v12.TRACE_AGGREGATE_PATH, replay.v12.TRACE_AGGREGATE_SHA256),
        (CATALOG_PATH, CATALOG_SHA256),
    )
    if any(_sha256_path(path)[0] != expected for path, expected in checks):
        return False
    for filename, expected in replay.v12.TRACE_SPECS:
        if _sha256_path(replay.v12.TRACE_AGGREGATE_PATH.parent / filename)[0] != expected:
            return False
    stat = CATALOG_PATH.stat()
    return preflight.catalog_snapshot == (int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0)))


def run(implementation_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    descriptor: int | None = None
    label_handle: BinaryIO | None = None
    proxy_handle: BinaryIO | None = None
    try:
        preflight = preflight_only(implementation_commit)
        descriptor = _open_receipt(implementation_commit)

        if LABEL_PATH.is_symlink():
            raise GraceAttributionError("label archive is a symlink")
        label_handle = LABEL_PATH.open("rb")
        label_identity = _handle_identity(label_handle)
        label_sha, label_size = _sha256_handle(label_handle)
        if label_identity[0] != LABEL_BYTES or label_size != LABEL_BYTES or label_sha != LABEL_SHA256:
            raise GraceAttributionError("label archive binding drifted")
        label_handle.seek(0)
        outcomes = replay.load_outcomes_from_open_handle(label_handle)
        labels = {
            "baseline_rank": outcomes.baseline_rank,
            "positive_index": outcomes.positive_index,
            "eligible_from": outcomes.eligible_from,
            "outer_fold": outcomes.outer_fold,
            "inner_fold": outcomes.inner_fold,
            "baseline_session_hit": replay.derive_baseline_session_hit(outcomes.baseline_rank, outcomes.eligible_from),
        }
        if [int(np.sum(outcomes.outer_fold == fold)) for fold in range(5)] != [400] * 5:
            raise GraceAttributionError("outer partition drifted")
        if set(np.unique(outcomes.inner_fold).tolist()) != set(range(5)):
            raise GraceAttributionError("inner partition drifted")
        surface = frozen._action_surface(preflight.inputs.projected_features, preflight.inputs.oof_scores, labels)
        if not np.array_equal(surface.chosen, preflight.inputs.chosen):
            raise GraceAttributionError("chosen surface drifted after attach")
        activation, reproduction = _reproduce_activation(surface, labels)
        grace = replay.replay_grace_pages(preflight.inputs.traces, surface.chosen, activation, preflight.inputs.versions)
        grace_repeat = replay.replay_grace_pages(preflight.inputs.traces, surface.chosen, activation, preflight.inputs.versions)
        if not replay._bundle_exact_repeat(grace, grace_repeat):
            raise GraceAttributionError("grace replay is not exact")
        if (
            grace.identity["baseline_ascii_page_sha256"] != EXPECTED_BASELINE_ASCII_SHA256
            or grace.identity["candidate_ascii_page_sha256"] != EXPECTED_GRACE_ASCII_SHA256
        ):
            raise GraceAttributionError("grace page identity drifted")
        grace_state = replay.state_from_positive_index(grace.candidate_pages, outcomes.positive_index, outcomes.eligible_from)
        fold_hits = tuple(int(np.sum(grace_state["hit"] & (outcomes.outer_fold == fold))) for fold in range(5))
        if int(grace_state["hit"].sum()) != 1_982 or fold_hits != EXPECTED_FOLD_HITS:
            raise GraceAttributionError("grace reference outcome drifted")
        remaining = np.flatnonzero(~grace_state["hit"])
        if len(remaining) != 18 or tuple(400 - value for value in fold_hits) != EXPECTED_FOLD_MISSES:
            raise GraceAttributionError("grace miss cohort drifted")

        # Proxy access is deliberately after every comparator identity gate.
        if PROXY_PATH.is_symlink():
            raise GraceAttributionError("proxy path is a symlink")
        proxy_handle = PROXY_PATH.open("rb")
        targets, proxy_identity, proxy_sha = _load_proxy(proxy_handle)
        if not _proxy_matches_numeric(targets, preflight.inputs.traces, outcomes.positive_index):
            raise GraceAttributionError("proxy/numeric ordinal binding drifted")

        reset_pages = _eligibility_reset_grace_pages(
            preflight.inputs.traces,
            surface.chosen,
            activation,
            preflight.inputs.versions,
            outcomes.eligible_from,
        )
        reset_repeat = _eligibility_reset_grace_pages(
            preflight.inputs.traces,
            surface.chosen,
            activation,
            preflight.inputs.versions,
            outcomes.eligible_from,
        )
        if not np.array_equal(reset_pages, reset_repeat):
            raise GraceAttributionError("state counterfactual is not exact")

        primary: list[str] = []
        folds: list[int] = []
        lifecycle: list[bool] = []
        information: list[dict[str, bool]] = []
        metadata: list[dict[str, bool]] = []
        latency_ns: list[int] = []
        for raw_index in remaining:
            cohort_index = int(raw_index)
            tick = time.perf_counter_ns()
            positive = outcomes.positive_index[cohort_index]
            boundary = int(outcomes.eligible_from[cohort_index]) - 1
            reachable = candidate_recall_flags(positive, boundary + 1)[100]
            admission = admission_rejected_flag(
                positive,
                boundary + 1,
                surface.chosen[cohort_index],
                surface.action[cohort_index],
                activation[cohort_index],
            )
            reset_state = replay.state_from_positive_index(
                reset_pages[cohort_index : cohort_index + 1],
                outcomes.positive_index[cohort_index : cohort_index + 1],
                outcomes.eligible_from[cohort_index : cohort_index + 1],
            )
            state_rejected = bool(reset_state["hit"][0])
            primary.append(
                classify_primary(
                    c100_reachable=reachable,
                    admission_rejected=admission,
                    state_rejected=state_rejected,
                )
            )
            folds.append(int(outcomes.outer_fold[cohort_index]))
            lifecycle.append(lifecycle_flag(positive, grace.candidate_pages[cohort_index], boundary + 1))
            information.append(
                information_insufficient(
                    preflight.inputs.projected_features[cohort_index], positive, boundary + 1
                )
            )
            mask = preflight.catalog_flags.get(targets[cohort_index])
            metadata.append(catalog_missing_fields(None) if mask is None else _mask_flags(mask))
            latency_ns.append(time.perf_counter_ns() - tick)

        aggregate = _aggregate(primary, folds, information, lifecycle, metadata)
        aggregate_repeat = _aggregate(primary, folds, information, lifecycle, metadata)
        if _canonical_sha256(aggregate) != _canonical_sha256(aggregate_repeat):
            raise GraceAttributionError("aggregate attribution is not exact")
        primary_total = sum(int(aggregate["primary"][name]["count"]) for name in PRIMARY_ORDER)
        fold_total = sum(
            int(row[name]) for row in aggregate["primary_by_outer_fold"] for name in PRIMARY_ORDER
        )
        if primary_total != 18 or fold_total != 18:
            raise GraceAttributionError("primary attribution is incomplete")

        label_end_sha, label_end_size = _sha256_handle(label_handle)
        proxy_end_sha, proxy_end_size = _sha256_handle(proxy_handle)
        if not (
            label_end_sha == LABEL_SHA256
            and label_end_size == LABEL_BYTES
            and _handle_identity(label_handle) == label_identity
            and proxy_end_sha == proxy_sha == PROXY_SHA256
            and proxy_end_size == PROXY_BYTES
            and _handle_identity(proxy_handle) == proxy_identity
            and _rehash_target_free(preflight)
        ):
            raise GraceAttributionError("source changed during attribution")
        _validate_protocol()
        _validate_git(implementation_commit)
        label_handle.close()
        label_handle = None
        proxy_handle.close()
        proxy_handle = None

        working, peak = replay._process_memory()
        wall = time.perf_counter() - started
        if not (
            0 < working <= RESOURCE_BYTES_MAXIMUM
            and 0 < peak <= RESOURCE_BYTES_MAXIMUM
            and wall <= RESOURCE_SECONDS_MAXIMUM
        ):
            raise GraceAttributionError("attribution resource gate failed")
        largest = largest_primary_bottleneck(
            {name: int(aggregate["primary"][name]["count"]) for name in PRIMARY_ORDER}
        )
        all_indices = tuple(range(SESSION_COUNT))
        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "COMPLETE_ANONYMOUS_ATTRIBUTION",
            "interpretation": "shared-cohort OOF evaluator-side attribution; not private or independent validation",
            "environment": dict(preflight.environment),
            "git": dict(preflight.git),
            "protocol_hashes": dict(preflight.protocol),
            "source_hashes": {
                "projected_features": replay.PROJECTED_FEATURES_SHA256,
                "oof_scores": replay.OOF_SCORES_SHA256,
                "labels": LABEL_SHA256,
                "proxy": PROXY_SHA256,
                "catalog": CATALOG_SHA256,
                "combined_blind_trace": replay.v12.COMBINED_TRACE_SHA256,
            },
            "source_physical": {
                "projected_features": {"bytes": replay.PROJECTED_FEATURES_BYTES, "sha256": replay.PROJECTED_FEATURES_SHA256},
                "oof_scores": {"bytes": replay.OOF_SCORES_BYTES, "sha256": replay.OOF_SCORES_SHA256},
                "labels": {"bytes": LABEL_BYTES, "sha256": LABEL_SHA256},
                "proxy": {"bytes": PROXY_BYTES, "sha256": PROXY_SHA256},
                "catalog": {"bytes": CATALOG_BYTES, "sha256": CATALOG_SHA256},
                "blind_aggregate": {"bytes": 23_371, "sha256": replay.v12.TRACE_AGGREGATE_SHA256},
                "blind_shards": [
                    {"name": filename, "bytes": 15_939_420, "rows": 5_000, "sha256": digest}
                    for filename, digest in replay.v12.TRACE_SPECS
                ],
            },
            "reference": {
                "cohort_size": SESSION_COUNT,
                "hits": 1_982,
                "misses": 18,
                "hr_at_10": 0.991,
                "mrr": 0.695795,
                "mttc": 2.869,
                "technical_score": 0.866858,
                "fold_hits": list(fold_hits),
                "fold_misses": list(EXPECTED_FOLD_MISSES),
                "baseline_ascii_page_sha256": EXPECTED_BASELINE_ASCII_SHA256,
                "grace_ascii_page_sha256": EXPECTED_GRACE_ASCII_SHA256,
                "exact_repeat": True,
            },
            "candidate_recall": {
                "all_2000": _recall_summary(outcomes.positive_index, outcomes.eligible_from, all_indices),
                "remaining_18": _recall_summary(outcomes.positive_index, outcomes.eligible_from, tuple(int(value) for value in remaining)),
            },
            **aggregate,
            "candidate_frontier": candidate_frontier_pending(),
            "identity": {
                "chosen_raw_sha256": EXPECTED_CHOSEN_SHA256,
                "activation_raw_sha256": EXPECTED_ACTIVATION_SHA256,
                "state_counterfactual_raw_sha256": hashlib.sha256(np.ascontiguousarray(reset_pages).tobytes()).hexdigest(),
                "aggregate_canonical_sha256": _canonical_sha256(aggregate),
                "all_sources_unchanged": True,
                "proxy_numeric_binding_exact": True,
            },
            "grace_replay_identity": {
                "first": dict(grace.identity),
                "repeat_canonical_sha256": _canonical_sha256(grace_repeat.identity),
                "first_repeat_all_arrays_equal": True,
            },
            "access_audit": {
                "receipt_durable_before_label": True,
                "label_archive_open_count": 1,
                "label_member_access_count": 5,
                "label_member_order_sha256": _canonical_sha256(
                    [
                        "baseline_rank",
                        "positive_index",
                        "eligible_from",
                        "outer_fold",
                        "inner_fold",
                    ]
                ),
                "label_member_order_exact": True,
                "proxy_open_count": 1,
                "evaluator_or_agent_started": False,
            },
            "comparator_reproduction": dict(reproduction),
            "resources": {
                "pre_receipt_working_set_bytes": preflight.memory[0],
                "pre_receipt_peak_working_set_bytes": preflight.memory[1],
                "wall_seconds": round(wall, 6),
                "working_set_bytes": working,
                "peak_working_set_bytes": peak,
                "attribution_latency_p95_microseconds": round(float(np.percentile(latency_ns, 95)) / 1_000.0, 6),
                "gpu_used": False,
                "gpu_peak_bytes": 0,
            },
            "decision": {
                "largest_primary_bottleneck": largest,
                "next_stage": "single fixed-two-page-grace plus frozen score-priority cached smoke",
                "runtime_changed": False,
                "runtime_outcome_features": 0,
                "default": "off",
            },
            "receipt": {"path": OUTPUT_PATH.relative_to(ROOT).as_posix(), "durable": True, "self_hash_omitted": True, "rerun_forbidden": True},
        }
        privacy_scan(result)
        _write_descriptor(descriptor, result)
        os.close(descriptor)
        descriptor = None
        return result
    except BaseException as error:
        if label_handle is not None:
            try:
                label_handle.close()
            except BaseException:
                pass
        if proxy_handle is not None:
            try:
                proxy_handle.close()
            except BaseException:
                pass
        was_consumed = descriptor is not None
        if descriptor is not None:
            _invalid_receipt(descriptor, implementation_commit, error)
            try:
                os.close(descriptor)
            except OSError:
                pass
        if was_consumed:
            raise GraceAttributionConsumedError(
                "attribution failed after receipt consumption; rerun is forbidden"
            ) from error
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        preflight = preflight_only(args.implementation_commit)
        print(json.dumps({"status": "TARGET_FREE_PREFLIGHT_PASS", "git": preflight.git, "memory": preflight.memory}, sort_keys=True))
        return 0
    result = run(args.implementation_commit)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary": result["primary"],
                "candidate_frontier": result["candidate_frontier"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
