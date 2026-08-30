"""Fit the frozen selector on T-only labels and freeze target-free H policies.

This Stage-1b orchestrator consumes the already-frozen v2.9 Stage-1a runtime
surfaces.  It may copy only the current outer fold's T_o label rows.  H_o labels
are never copied, retained, supplied to a selector or metric, or serialized.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


PREREGISTRATION = ROOT / "configs/small_ranker_v2_9.top5_proposal_depth_preregistration.json"
STAGE1A_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_"
    "stage1a_implementation_amendment.json"
)
STAGE1A_MANIFEST = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_stage1a.manifest.json"
)
STAGE1A_RESULT = ROOT / (
    "experiments/fast_track/small_ranker_v2_9/"
    "stage1a_20260830T225937/stage1a_result.json"
)
STAGE1B_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_9.top5_proposal_depth_"
    "stage1b_implementation_amendment.json"
)
STAGE1A_BUILDER = ROOT / "scripts/build_small_ranker_top5_proposal_depth_stage1a.py"
V28_BUILDER_SOURCE = ROOT / "scripts/build_small_ranker_strict_outer_restack.py"
V28_FREEZER_SOURCE = ROOT / "scripts/freeze_small_ranker_strict_outer_restack.py"
MECHANICS_SOURCE = ROOT / "scripts/probe_small_ranker_top5_proposal_depth.py"
SELECTOR_SOURCE = ROOT / "scripts/small_ranker_portfolio_selector_py39.py"
METRIC_SOURCE = ROOT / "scripts/analyze_small_ranker_metric_gate.py"
FROZEN_SOURCE = ROOT / "scripts/export_small_ranker_fold_safe_artifact.py"
BASE_SOURCE = ROOT / "scripts/train_small_ranker.py"
RR_SOURCE = ROOT / "scripts/analyze_small_ranker_rr_regret_gate.py"
CACHE_MANIFEST = ROOT / "configs/small_ranker_v1.cache.manifest.json"
LABEL_ARCHIVE_RELATIVE = Path("experiments/fast_track/small_ranker_v1/labels_v2.npz")
DEFAULT_LABEL_ARCHIVE = Path(r"D:\tiktok\techjam-err402-fast-track") / LABEL_ARCHIVE_RELATIVE

EXPECTED_HASHES = {
    "preregistration": "51c0a9d909e7e8d21604ff29981c8a35ca217b94e0ec9d6f8c98ca12d700cebb",
    "stage1a_amendment": "b0c7e1cfe6ef9a56657f9898dd2f7358628471fe9380de5bae4ac564cf3324d3",
    "stage1a_manifest": "d85f75a2e09ee9ad7a39e12e5b5cf858acd5c134ce122978bcab40c5d4081704",
    "stage1a_result": "54577998a25dbf3054f2f393e837b0d49ea1fb5a87d95103d860595c97b067b1",
    "stage1b_amendment": "5c1e6a6e78b56c3c22a8329daac941be0ab46bc5c84c989141e30a402fdf7d7c",
    "stage1a_builder": "07c864b6e490ddf3a614bfc6a45e1756c645a71f007f1c46899f97cbd9ce9a3d",
    "v28_builder": "461b0caccfaa8a13ce3373d6b3860eb72188e8ce4fd0e281f2841028d164519a",
    "v28_freezer": "3c164aceedbe684bdcb86c4cab2ffa34280b24ed4043f6e3072b13f381ce2797",
    "mechanics": "6ca765d0df519da789ebbeecf82b9629ac32298b41180e1ca951d769f1a94e64",
    "selector": "35b7b68af7c52b7ecf0fe37ee686ed2e737ff2f6643622abd26dbc97e192cba8",
    "metric": "8c0cbffa6cd3dc62ddee3bb386c16bd60592a6324ecf6fcf4bcd4cf37951ca83",
    "frozen": "5115026c53b21d4d5930cb9af7783c0988b049a0e259f5a0a588901ad44f5e8b",
    "base": "db7f4a3e19da118abb7d37fc1530babd6928894e51e85010b11d9dcdc1d7e583",
    "rr": "793e3615df38cd995f55e57decaeea35b549e40ad50ee3bf8a6dbf1055ca7e80",
    "cache_manifest": "a930d184672bc29d9dd4bc1c2e908da035712ab061f2127a9771b2f3ed6a5c1a",
}
SOURCE_PATHS = {
    "preregistration": PREREGISTRATION,
    "stage1a_amendment": STAGE1A_AMENDMENT,
    "stage1a_manifest": STAGE1A_MANIFEST,
    "stage1a_result": STAGE1A_RESULT,
    "stage1b_amendment": STAGE1B_AMENDMENT,
    "stage1a_builder": STAGE1A_BUILDER,
    "v28_builder": V28_BUILDER_SOURCE,
    "v28_freezer": V28_FREEZER_SOURCE,
    "mechanics": MECHANICS_SOURCE,
    "selector": SELECTOR_SOURCE,
    "metric": METRIC_SOURCE,
    "frozen": FROZEN_SOURCE,
    "base": BASE_SOURCE,
    "rr": RR_SOURCE,
    "cache_manifest": CACHE_MANIFEST,
}

for _name in (
    "stage1a_builder",
    "v28_builder",
    "v28_freezer",
    "mechanics",
    "selector",
    "metric",
    "frozen",
    "base",
    "rr",
    "cache_manifest",
):
    _path = SOURCE_PATHS[_name]
    if not _path.is_file() or _path.is_symlink() or _sha256(_path) != EXPECTED_HASHES[_name]:
        raise RuntimeError("hash-pinned Stage-1b dependency is unavailable: %s" % _name)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import build_small_ranker_top5_proposal_depth_stage1a as stage1a  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import probe_small_ranker_top5_proposal_depth as mechanics  # noqa: E402
from scripts import small_ranker_portfolio_selector_py39 as selector  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


for _module, _source, _name in (
    (stage1a, STAGE1A_BUILDER, "stage1a_builder"),
    (mechanics, MECHANICS_SOURCE, "mechanics"),
    (selector, SELECTOR_SOURCE, "selector"),
    (metric, METRIC_SOURCE, "metric"),
    (frozen, FROZEN_SOURCE, "frozen"),
    (frozen.rr, RR_SOURCE, "rr"),
    (base, BASE_SOURCE, "base"),
):
    if Path(_module.__file__).resolve() != _source.resolve() or _sha256(_source) != EXPECTED_HASHES[_name]:
        raise RuntimeError("imported Stage-1b dependency drifted: %s" % _name)


SCHEMA_VERSION = "small-ranker-top5-proposal-depth-stage1b.v1"
PARENT_STAGE1A_COMMIT = "bf29d629b4dc81877648480ea33046f3ac663db6"
STAGE1A_RESULT_BYTES = 219_510
STAGE1A_IDENTITY = "b0abefdd23b4364c336e10035baa8a62b5dd1b4dd17f435b53244aec72b9c1da"
STAGE1A_ARRAY_FILES = 270
STAGE1A_ARRAY_BYTES = 438_874_560
PRE_STAGE1B_CACHE_BYTES = 526_907_778
EXPECTED_LABEL_SHA256 = "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
EXPECTED_LABEL_BYTES = 1_702_876
PASSES = ("first", "repeat")
OUTER_FOLDS = tuple(range(5))
SELECTOR_ARRAY_FIELDS = (
    "inner_rescue_probability",
    "inner_regret_probability",
    "inner_coverage",
    "reference_rescue_probability",
    "reference_regret_probability",
    "reference_utility",
    "held_rescue_probability",
    "held_regret_probability",
    "held_utility",
    "supplement",
    "supplemental_choice",
    "final_chosen",
    "final_activation",
)
FROZEN_ARRAY_FIELDS = (
    "domain_local_current_chosen",
    "domain_local_current_activation",
    "final_chosen",
    "final_activation",
)
DERIVED_LABEL_FIELDS = (
    "rescue",
    "rescue_weights",
    "regret",
    "regret_weights",
    "rr_loss",
    "mttc_loss",
)
LABEL_MEMBER_SPECS = {
    "baseline_rank": ((2000, 10), "uint8"),
    "positive_index": ((2000, 10), "int16"),
    "eligible_from": ((2000,), "uint8"),
    "inner_fold": ((2000,), "uint8"),
    "family_index": ((2000,), "int32"),
}
EXPECTED_OUTPUT_ARRAYS = 2 * 5 * len(SELECTOR_ARRAY_FIELDS) + len(FROZEN_ARRAY_FIELDS)
TRACKED_CHECKPOINT_PATHS = (
    ".gitattributes",
    "configs/small_ranker_v2_9.top5_proposal_depth_stage1b_implementation_amendment.json",
    "scripts/build_small_ranker_top5_proposal_depth_stage1b.py",
    "tests/test_small_ranker_top5_proposal_depth_stage1b.py",
)


class Stage1BError(RuntimeError):
    pass


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1BError("expected a JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_environment() -> Dict[str, Any]:
    import sklearn
    import xgboost

    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "workers": 1,
    }
    expected = {
        "python": "3.9.19",
        "numpy": "1.26.4",
        "xgboost": "1.7.6",
        "sklearn": "1.1.3",
        "workers": 1,
    }
    if actual != expected:
        raise Stage1BError("Stage-1b dependency identity mismatch")
    return actual


def _path_below(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved != parent and parent in resolved.parents


def _source_snapshot() -> Dict[str, str]:
    snapshot = {"orchestrator": _sha256(Path(__file__).resolve())}
    for name, path in SOURCE_PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise Stage1BError("Stage-1b source is unavailable: %s" % name)
        snapshot[name] = _sha256(path)
    return snapshot


def _iter_stage1a_records(result: Mapping[str, Any]):
    for pass_name in PASSES:
        outers = result.get(pass_name, ())
        if len(outers) != len(OUTER_FOLDS):
            raise Stage1BError("Stage-1a outer registry is incomplete")
        for outer_fold, outer in enumerate(outers):
            if int(outer.get("outer_fold", -1)) != outer_fold:
                raise Stage1BError("Stage-1a outer registry order drifted")
            for phase in mechanics.PHASES:
                files = outer.get("phases", {}).get(phase, {}).get("files", {})
                if set(files) != set(mechanics.SURFACE_FIELDS):
                    raise Stage1BError("Stage-1a surface registry drifted")
                for field in mechanics.SURFACE_FIELDS:
                    yield pass_name, outer_fold, phase, field, files[field]


def _audit_stage1a_physical(result: Mapping[str, Any]) -> Dict[str, Any]:
    stage_root = STAGE1A_RESULT.parent.resolve()
    seen = set()
    total_bytes = 0
    for _pass_name, _outer_fold, _phase, _field, record in _iter_stage1a_records(result):
        raw = Path(str(record.get("path", "")))
        unresolved = ROOT / raw if not raw.is_absolute() else raw
        if unresolved.is_symlink():
            raise Stage1BError("Stage-1a surface symlink is forbidden")
        path = unresolved.resolve()
        key = str(path).lower()
        if (
            path.suffix.lower() != ".npy"
            or not _path_below(path, stage_root)
            or not path.is_file()
            or key in seen
            or path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(path) != str(record.get("sha256"))
            or int(record.get("asin_shape_matches", -1)) != 0
            or mechanics._identity_shape_scan(path) != 0
        ):
            raise Stage1BError("Stage-1a physical surface audit failed")
        seen.add(key)
        total_bytes += path.stat().st_size
    if len(seen) != STAGE1A_ARRAY_FILES or total_bytes != STAGE1A_ARRAY_BYTES:
        raise Stage1BError("Stage-1a physical surface totals drifted")
    return {
        "registered_files": len(seen),
        "registered_bytes": total_bytes,
        "all_file_hashes_and_identity_scans_verified": True,
    }


def _validate_protocol_without_labels() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for name, path in SOURCE_PATHS.items():
        expected = EXPECTED_HASHES[name]
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256(path) != expected
        ):
            raise Stage1BError("Stage-1b source mismatch: %s" % name)
    prereg = _load_json(PREREGISTRATION)
    manifest = _load_json(STAGE1A_MANIFEST)
    result = _load_json(STAGE1A_RESULT)
    amendment = _load_json(STAGE1B_AMENDMENT)
    if not (
        prereg.get("schema_version")
        == "small-ranker-top5-proposal-depth-preregistration.v1"
        and manifest.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage1a-manifest.v1"
        and manifest.get("status") == "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
        and manifest.get("git", {}).get("stage1a_implementation_commit")
        == "17c8e95c5b5ef2d04645e7a79d97eb4c6315ee69"
        and manifest.get("result", {}).get("path") == STAGE1A_RESULT.relative_to(ROOT).as_posix()
        and manifest.get("result", {}).get("sha256") == EXPECTED_HASHES["stage1a_result"]
        and manifest.get("result", {}).get("bytes") == STAGE1A_RESULT_BYTES
        and manifest.get("result", {}).get("aggregate_identity_sha256") == STAGE1A_IDENTITY
        and manifest.get("decision", {}).get("stage1b_implementation_preparation_authorized")
        is True
        and manifest.get("decision", {}).get("stage1b_t_only_label_attach_authorized_now")
        is False
        and result.get("status") == "TARGET_FREE_ALL_OUTER_SURFACES_FROZEN"
        and result.get("exact_repeat", {}).get("identity_sha256") == STAGE1A_IDENTITY
        and result.get("privacy", {}).get("label_archive_opened") is False
        and result.get("decision", {}).get("stage1b_t_only_label_attach_authorized")
        is False
        and amendment.get("schema_version")
        == "small-ranker-top5-proposal-depth-stage1b-implementation-amendment.v1"
        and amendment.get("parent_stage1a_evidence_commit") == PARENT_STAGE1A_COMMIT
        and amendment.get("source_binding", {}).get("stage1a_manifest_sha256")
        == EXPECTED_HASHES["stage1a_manifest"]
        and amendment.get("source_binding", {}).get("stage1a_result_sha256")
        == EXPECTED_HASHES["stage1a_result"]
        and amendment.get("source_binding", {}).get("stage1a_aggregate_identity_sha256")
        == STAGE1A_IDENTITY
        and amendment.get("source_binding", {}).get("sealed_label_archive_expected_sha256")
        == EXPECTED_LABEL_SHA256
        and amendment.get("source_binding", {}).get(
            "v2_7_python39_selector_reference_sha256"
        )
        == EXPECTED_HASHES["selector"]
        and amendment.get("source_binding", {}).get(
            "v2_8_selector_builder_reference_sha256"
        )
        == EXPECTED_HASHES["v28_builder"]
        and amendment.get("source_binding", {}).get(
            "v2_8_freezer_reference_sha256"
        )
        == EXPECTED_HASHES["v28_freezer"]
        and amendment.get("commit_and_open_gate", {}).get(
            "parent_stage1a_evidence_commit_must_be_HEAD_ancestor"
        )
        is True
        and amendment.get("commit_and_open_gate", {}).get(
            "stage1a_manifest_last_change_must_equal_parent_evidence_commit"
        )
        is True
        and amendment.get("environment_contract", {}).get("python") == "3.9.19"
        and amendment.get("environment_contract", {}).get("numpy") == "1.26.4"
        and amendment.get("environment_contract", {}).get("xgboost") == "1.7.6"
        and amendment.get("environment_contract", {}).get("sklearn") == "1.1.3"
        and amendment.get("environment_contract", {}).get("workers") == 1
        and amendment.get("label_loader_contract", {}).get("archive_open_count") == 5
        and amendment.get("label_loader_contract", {}).get("member_access_count") == 25
        and amendment.get("label_loader_contract", {}).get(
            "allowed_member_names_in_fixed_order"
        )
        == list(LABEL_MEMBER_SPECS)
        and amendment.get("output_schema", {}).get("supervised_label_arrays_serialized")
        is False
        and amendment.get("output_schema", {}).get("total_npy_files")
        == EXPECTED_OUTPUT_ARRAYS
        and amendment.get("output_schema", {}).get("total_json_files") == 1
    ):
        raise Stage1BError("Stage-1b protocol semantics drifted")
    physical = _audit_stage1a_physical(result)
    if physical != {
        "registered_files": STAGE1A_ARRAY_FILES,
        "registered_bytes": STAGE1A_ARRAY_BYTES,
        "all_file_hashes_and_identity_scans_verified": True,
    }:
        raise Stage1BError("Stage-1a physical prerequisite drifted")
    if selector.MAX_ACTIONS != 3 or mechanics.MAX_ACTIONS != 15:
        raise Stage1BError("selector width adapter prerequisite drifted")
    if tuple(selector.FEATURE_NAMES) != tuple(mechanics.FEATURE_NAMES):
        raise Stage1BError("selector feature schema drifted")
    return prereg, manifest, result, amendment


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=%s" % ROOT.as_posix(), *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise Stage1BError("Stage-1b Git checkpoint validation failed")
    return completed.stdout.strip()


def _validate_git_checkpoint() -> str:
    if _git(("status", "--porcelain")):
        raise Stage1BError("Stage-1b requires a clean committed worktree")
    head = _git(("rev-parse", "HEAD"))
    if head == PARENT_STAGE1A_COMMIT:
        raise Stage1BError("Stage-1b implementation checkpoint is absent")
    _git(("merge-base", "--is-ancestor", PARENT_STAGE1A_COMMIT, head))
    stage1a_manifest_path = STAGE1A_MANIFEST.relative_to(ROOT).as_posix()
    if _git(("log", "-1", "--format=%H", "--", stage1a_manifest_path)) != PARENT_STAGE1A_COMMIT:
        raise Stage1BError("Stage-1a evidence provenance drifted")
    for relative in TRACKED_CHECKPOINT_PATHS:
        _git(("cat-file", "-e", "HEAD:%s" % relative))
        if _git(("log", "-1", "--format=%H", "--", relative)) != head:
            raise Stage1BError("Stage-1b checkpoint is not an exact implementation snapshot")
    return head


def _validate_training_order(t_sessions: np.ndarray) -> np.ndarray:
    training = np.asarray(t_sessions, dtype=np.int16)
    if (
        training.shape != (1600,)
        or len(np.unique(training)) != 1600
        or np.any(training < 0)
        or np.any(training >= 2000)
        or not np.array_equal(training, np.sort(training))
    ):
        raise Stage1BError("T_o session order is invalid")
    return training


def _validate_t_label_values(name: str, value: np.ndarray) -> None:
    if name == "baseline_rank" and np.any(value > 10):
        raise Stage1BError("T_o baseline rank is invalid")
    if name == "positive_index" and np.any((value < -1) | (value >= 100)):
        raise Stage1BError("T_o positive index is invalid")
    if name == "eligible_from" and np.any((value < 1) | (value > 10)):
        raise Stage1BError("T_o eligibility is invalid")
    if name == "inner_fold" and set(np.unique(value).tolist()) != set(OUTER_FOLDS):
        raise Stage1BError("T_o inner-fold coverage is invalid")
    if name == "family_index" and np.any(value < 0):
        raise Stage1BError("T_o family index is invalid")


def _audit_label_archive(
    path: Path,
    expected_sha256: str = EXPECTED_LABEL_SHA256,
    expected_bytes: int = EXPECTED_LABEL_BYTES,
) -> Dict[str, Any]:
    label_path = Path(path)
    if (
        label_path.suffix.lower() != ".npz"
        or label_path.is_symlink()
        or not label_path.is_file()
        or label_path.stat().st_size != int(expected_bytes)
        or _sha256(label_path) != expected_sha256
    ):
        raise Stage1BError("sealed label archive preflight failed")
    return {
        "path": str(label_path.resolve()),
        "sha256": expected_sha256,
        "bytes": int(expected_bytes),
        "regular_file": True,
        "symlink": False,
    }


def _load_t_only_labels(
    path: Path,
    t_sessions: np.ndarray,
    expected_sha256: str = EXPECTED_LABEL_SHA256,
    expected_bytes: int = EXPECTED_LABEL_BYTES,
    np_load=np.load,
) -> Dict[str, np.ndarray]:
    label_path = Path(path)
    training = _validate_training_order(t_sessions)
    _audit_label_archive(label_path, expected_sha256, expected_bytes)
    labels_t: Dict[str, np.ndarray] = {}
    try:
        with np_load(label_path, allow_pickle=False) as archive:
            for name, (shape, dtype) in LABEL_MEMBER_SPECS.items():
                complete_member = archive[name]
                if (
                    not isinstance(complete_member, np.ndarray)
                    or complete_member.shape != tuple(shape)
                    or str(complete_member.dtype) != dtype
                ):
                    raise Stage1BError("sealed label member schema failed: %s" % name)
                selected = np.asarray(complete_member[training]).copy()
                del complete_member
                if selected.shape[0] != 1600 or str(selected.dtype) != dtype:
                    raise Stage1BError("T_o label slice schema failed: %s" % name)
                _validate_t_label_values(name, selected)
                selected.setflags(write=False)
                labels_t[name] = selected
    except Stage1BError:
        raise
    except Exception as exc:
        raise Stage1BError("sealed T_o-only label load failed") from exc
    if tuple(labels_t) != tuple(LABEL_MEMBER_SPECS):
        raise Stage1BError("T_o label member registry drifted")
    inner = labels_t["inner_fold"]
    family = labels_t["family_index"]
    for family_id in np.unique(family):
        if len(np.unique(inner[family == family_id])) != 1:
            raise Stage1BError("product family crosses an inner-fold boundary")
    return labels_t


def _record_path(record: Mapping[str, Any], source_root: Path) -> Path:
    raw = Path(str(record.get("path", "")))
    unresolved = ROOT / raw if not raw.is_absolute() else raw
    if unresolved.is_symlink():
        raise Stage1BError("surface symlink is forbidden")
    path = unresolved.resolve()
    if (
        path.suffix.lower() != ".npy"
        or not _path_below(path, source_root)
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or _sha256(path) != str(record.get("sha256"))
        or int(record.get("asin_shape_matches", -1)) != 0
        or mechanics._identity_shape_scan(path) != 0
    ):
        raise Stage1BError("frozen Stage-1a surface file failed validation")
    return path


def _load_surface(
    stage_root: Path, outer: Mapping[str, Any], phase: str
) -> selector.RuntimePortfolioSurface:
    files = outer.get("phases", {}).get(phase, {}).get("files", {})
    if set(files) != set(mechanics.SURFACE_FIELDS):
        raise Stage1BError("surface registry is incomplete")
    values: Dict[str, np.ndarray] = {}
    for name in mechanics.SURFACE_FIELDS:
        record = files[name]
        path = _record_path(record, stage_root)
        value = mechanics._load_npy_mmap(path)
        if (
            value.shape != tuple(int(item) for item in record.get("shape", []))
            or str(value.dtype) != str(record.get("dtype"))
            or mechanics._array_sha256(value) != str(record.get("array_sha256"))
            or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all())
        ):
            raise Stage1BError("surface logical array validation failed")
        values[name] = value
    sessions = 400 if phase == "held_H" else 1600
    expected = {
        "current_chosen": ((sessions, 10), "uint8"),
        "current_activation": ((sessions, 10), "bool"),
        "current_choice": ((sessions, 10), "uint8"),
        "incumbent": ((sessions, 10), "uint8"),
        "family_choices": ((sessions, 10, 3, 5), "uint8"),
        "candidates": ((sessions, 10, 15), "int16"),
        "source_mask": ((sessions, 10, 15), "uint8"),
        "available": ((sessions, 10, 15), "bool"),
        "features": ((sessions, 10, 15, 19), "float32"),
    }
    if any(values[name].shape != shape or str(values[name].dtype) != dtype for name, (shape, dtype) in expected.items()):
        raise Stage1BError("surface schema is not the frozen top-5 schema")
    if (
        np.any(values["available"].sum(axis=2) > 15)
        or np.any(values["source_mask"][~values["available"]])
        or np.any(values["features"][~values["available"]])
    ):
        raise Stage1BError("surface padding or width is invalid")
    return selector.RuntimePortfolioSurface(**values)


def _attach_t_only_labels(
    runtime: selector.RuntimePortfolioSurface, labels_t: Mapping[str, np.ndarray]
) -> selector.PortfolioSurface:
    available = np.asarray(runtime.available)
    candidates = np.asarray(runtime.candidates)
    if (
        available.ndim != 3
        or candidates.shape != available.shape
        or runtime.features.shape[:3] != available.shape
        or len(labels_t["eligible_from"]) != available.shape[0]
    ):
        raise Stage1BError("dynamic selector surface shape failed")
    rescue = np.zeros(available.shape, dtype=np.uint8)
    regret = np.zeros(available.shape, dtype=np.uint8)
    rr_loss = np.zeros(available.shape, dtype=np.float32)
    mttc_loss = np.zeros(available.shape, dtype=np.float32)
    for slot in range(available.shape[2]):
        safe_candidate = np.where(
            available[..., slot], candidates[..., slot], runtime.incumbent
        ).astype(np.uint8)
        slot_labels = selector._isolated_action_labels(
            labels_t,
            runtime.current_chosen,
            runtime.current_activation,
            safe_candidate,
            available[..., slot],
        )
        rescue[..., slot], regret[..., slot], rr_loss[..., slot], mttc_loss[..., slot] = slot_labels
    rescue_weights = selector._session_normalize_weights(
        np.where(rescue > 0, 1.0, np.where(regret > 0, 5.0, 0.05)), available
    )
    regret_weights = selector._session_normalize_weights(
        np.where(
            regret > 0,
            5.0 + 20.0 * rr_loss + 0.2 * mttc_loss,
            np.where(rescue > 0, 0.2, 0.05),
        ),
        available,
    )
    return selector.PortfolioSurface(
        current_chosen=runtime.current_chosen,
        current_activation=runtime.current_activation,
        current_choice=runtime.current_choice,
        incumbent=runtime.incumbent,
        family_choices=runtime.family_choices,
        candidates=runtime.candidates,
        source_mask=runtime.source_mask,
        available=runtime.available,
        features=runtime.features,
        rescue=rescue,
        rescue_weights=rescue_weights,
        regret=regret,
        regret_weights=regret_weights,
        rr_loss=rr_loss,
        mttc_loss=mttc_loss,
    )


def _gate_model_sha256(model: Any, mean: np.ndarray, scale: np.ndarray) -> str:
    payload: Dict[str, Any] = {
        "type": type(model).__name__,
        "mean": mechanics._array_sha256(np.asarray(mean)),
        "scale": mechanics._array_sha256(np.asarray(scale)),
    }
    if hasattr(model, "coef_"):
        payload.update(
            {
                "coef": mechanics._array_sha256(np.asarray(model.coef_)),
                "intercept": mechanics._array_sha256(np.asarray(model.intercept_)),
                "classes": mechanics._array_sha256(np.asarray(model.classes_)),
            }
        )
    else:
        payload["probability"] = float(model.probability)
    return _canonical_sha256(payload)


def _write_array(path: Path, value: np.ndarray) -> Dict[str, Any]:
    return mechanics._write_npy_exclusive(path, np.asarray(value))


def _label_identity(surface: selector.PortfolioSurface) -> Dict[str, Any]:
    return {
        "arrays": {
            name: {
                "array_sha256": mechanics._array_sha256(
                    np.asarray(getattr(surface, name))
                ),
                "shape": [int(item) for item in getattr(surface, name).shape],
                "dtype": str(getattr(surface, name).dtype),
            }
            for name in DERIVED_LABEL_FIELDS
        },
        "rescue_rows": int(surface.rescue.sum()),
        "regret_rows": int(surface.regret.sum()),
        "serialized": False,
    }


def _apply_held_policy(
    held: selector.RuntimePortfolioSurface,
    rescue_probability: np.ndarray,
    regret_probability: np.ndarray,
    mapped_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rescue_probability = np.asarray(rescue_probability)
    regret_probability = np.asarray(regret_probability)
    if (
        rescue_probability.shape != held.available.shape
        or regret_probability.shape != held.available.shape
        or not np.isfinite(rescue_probability).all()
        or not np.isfinite(regret_probability).all()
        or np.any(rescue_probability[~held.available])
        or np.any(regret_probability[~held.available])
        or math.isnan(float(mapped_threshold))
        or (not np.isfinite(mapped_threshold) and mapped_threshold != math.inf)
    ):
        raise Stage1BError("held target-free probability contract failed")
    held_utility = (
        rescue_probability
        - selector.REGRET_MULTIPLIER * regret_probability
    )
    supplement, supplemental_choice = selector._causal_policy(
        held.candidates,
        held.source_mask,
        held.available,
        held_utility,
        mapped_threshold,
        np.ones(len(held.current_chosen), dtype=bool),
    )
    final_chosen, final_activation = selector._compose_policy(
        held.current_chosen,
        held.current_activation,
        held.candidates,
        held.available,
        supplement,
        supplemental_choice,
    )
    if (
        supplement.shape != held.current_chosen.shape
        or supplemental_choice.shape != held.current_chosen.shape
        or final_chosen.shape != held.current_chosen.shape
        or final_activation.shape != held.current_activation.shape
        or np.any(supplement.sum(axis=1) > 1)
    ):
        raise Stage1BError("held chronological latch contract failed")
    if mapped_threshold == math.inf and (
        np.any(supplement)
        or not np.array_equal(final_chosen, held.current_chosen)
        or not np.array_equal(final_activation, held.current_activation)
    ):
        raise Stage1BError("KEEP did not preserve the domain-local current policy")
    return (
        supplement,
        supplemental_choice,
        final_chosen,
        final_activation,
        held_utility,
    )


def _run_selector(
    training: selector.PortfolioSurface,
    labels_t: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    reference: selector.RuntimePortfolioSurface,
    held: selector.RuntimePortfolioSurface,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    inner = np.asarray(labels_t["inner_fold"], dtype=np.uint8)
    family = np.asarray(labels_t["family_index"], dtype=np.int32)
    width = int(training.available.shape[2])
    if (
        width <= 0
        or reference.available.shape[2] != width
        or held.available.shape[2] != width
        or training.features.shape[3] != len(selector.FEATURE_NAMES)
    ):
        raise Stage1BError("selector dynamic width is inconsistent")
    flat_x = training.features.reshape(-1, len(selector.FEATURE_NAMES))
    flat_available = training.available.reshape(-1)
    flat_session = np.repeat(np.arange(len(inner)), base.TURN_COUNT * width)
    flat_family = np.repeat(family, base.TURN_COUNT * width)
    targets = (training.rescue.reshape(-1), training.regret.reshape(-1))
    weights = (
        training.rescue_weights.reshape(-1),
        training.regret_weights.reshape(-1),
    )
    inner_probability = [
        np.zeros(training.available.shape, dtype=np.float32) for _ in range(2)
    ]
    coverage = np.zeros_like(flat_available, dtype=np.uint8)
    readiness_records = []
    blocked = False
    selector_fits = 0
    for inner_fold in OUTER_FOLDS:
        train_sessions = inner != inner_fold
        valid_sessions = inner == inner_fold
        train_rows = flat_available & train_sessions[flat_session]
        valid_rows = flat_available & valid_sessions[flat_session]
        readiness = selector._fit_readiness(
            train_rows,
            targets[0],
            targets[1],
            flat_session,
            flat_family,
        )
        readiness["inner_fold"] = inner_fold
        readiness["valid_action_rows"] = int(valid_rows.sum())
        head_hashes = []
        if not readiness["ready"] or not np.any(valid_rows):
            blocked = True
        else:
            coverage[valid_rows] += 1
            for head in range(2):
                model, mean, scale = selector._fit_gate_model(
                    flat_x[train_rows],
                    targets[head][train_rows],
                    weights[head][train_rows],
                    selector.MODEL_SEED,
                )
                selector._validate_fitted_model(model)
                inner_probability[head].reshape(-1)[valid_rows] = (
                    base._predict_gate(
                        model, mean, scale, flat_x[valid_rows]
                    ).astype(np.float32)
                )
                head_hashes.append(_gate_model_sha256(model, mean, scale))
                selector_fits += 1
        readiness["head_model_sha256"] = head_hashes
        readiness_records.append(readiness)
    if not blocked and (
        not np.all(coverage[flat_available] == 1)
        or np.any(coverage[~flat_available])
    ):
        raise Stage1BError("selector inner coverage is invalid")

    selected: Dict[str, Any]
    outer_readiness: Optional[Dict[str, Any]] = None
    outer_heads = []
    mapped_threshold = math.inf
    reference_probability = [
        np.zeros(reference.available.shape, dtype=np.float32) for _ in range(2)
    ]
    held_probability = [
        np.zeros(held.available.shape, dtype=np.float32) for _ in range(2)
    ]
    inner_utility = (
        inner_probability[0]
        - selector.REGRET_MULTIPLIER * inner_probability[1]
    )
    if blocked:
        selected = {
            "quantile": frozen.KEEP_QUANTILE,
            "status": "KEEP_INSUFFICIENT_INNER_FIT",
        }
    else:
        selected = selector._select_inner_quantile(
            training,
            inner_utility,
            labels_t,
            current_state,
            np.ones(len(inner), dtype=bool),
            inner,
        )
        if float(selected["quantile"]) < frozen.KEEP_QUANTILE:
            outer_readiness = selector._fit_readiness(
                flat_available,
                targets[0],
                targets[1],
                flat_session,
                flat_family,
            )
            if outer_readiness["ready"]:
                reference_flat = reference.features.reshape(
                    -1, len(selector.FEATURE_NAMES)
                )
                held_flat = held.features.reshape(-1, len(selector.FEATURE_NAMES))
                reference_available = reference.available.reshape(-1)
                held_available = held.available.reshape(-1)
                if not np.any(reference_available):
                    raise Stage1BError("finite quantile has no reference actions")
                for head in range(2):
                    model, mean, scale = selector._fit_gate_model(
                        flat_x[flat_available],
                        targets[head][flat_available],
                        weights[head][flat_available],
                        selector.MODEL_SEED,
                    )
                    selector._validate_fitted_model(model)
                    reference_probability[head].reshape(-1)[reference_available] = (
                        base._predict_gate(
                            model,
                            mean,
                            scale,
                            reference_flat[reference_available],
                        ).astype(np.float32)
                    )
                    if np.any(held_available):
                        held_probability[head].reshape(-1)[held_available] = (
                            base._predict_gate(
                                model, mean, scale, held_flat[held_available]
                            ).astype(np.float32)
                        )
                    outer_heads.append(
                        {
                            "head": head,
                            "seed": selector.MODEL_SEED,
                            "model_sha256": _gate_model_sha256(model, mean, scale),
                        }
                    )
                    selector_fits += 1
                reference_utility = (
                    reference_probability[0]
                    - selector.REGRET_MULTIPLIER * reference_probability[1]
                )
                (
                    _slot,
                    _candidate,
                    reference_winner,
                    reference_winner_available,
                ) = selector._per_turn_winner_utilities(
                    reference.candidates,
                    reference.source_mask,
                    reference.available,
                    reference_utility,
                )
                mapped_threshold = selector._map_outer_quantile(
                    reference_winner,
                    reference_winner_available,
                    np.ones(len(reference.current_chosen), dtype=bool),
                    float(selected["quantile"]),
                )
                selected = {**selected, "status": "FINITE_SELECTED"}
            else:
                selected = {
                    **selected,
                    "proposed_quantile": float(selected["quantile"]),
                    "quantile": frozen.KEEP_QUANTILE,
                    "status": "KEEP_INSUFFICIENT_OUTER_FIT",
                }
        else:
            selected = {**selected, "status": "KEEP_SELECTED"}

    reference_utility = (
        reference_probability[0]
        - selector.REGRET_MULTIPLIER * reference_probability[1]
    )
    (
        supplement,
        supplemental_choice,
        final_chosen,
        final_activation,
        held_utility,
    ) = _apply_held_policy(
        held,
        held_probability[0],
        held_probability[1],
        mapped_threshold,
    )
    files = {
        "inner_rescue_probability": _write_array(
            output_dir / "inner_rescue_probability.npy", inner_probability[0]
        ),
        "inner_regret_probability": _write_array(
            output_dir / "inner_regret_probability.npy", inner_probability[1]
        ),
        "inner_coverage": _write_array(output_dir / "inner_coverage.npy", coverage),
        "reference_rescue_probability": _write_array(
            output_dir / "reference_rescue_probability.npy", reference_probability[0]
        ),
        "reference_regret_probability": _write_array(
            output_dir / "reference_regret_probability.npy", reference_probability[1]
        ),
        "reference_utility": _write_array(
            output_dir / "reference_utility.npy", reference_utility
        ),
        "held_rescue_probability": _write_array(
            output_dir / "held_rescue_probability.npy", held_probability[0]
        ),
        "held_regret_probability": _write_array(
            output_dir / "held_regret_probability.npy", held_probability[1]
        ),
        "held_utility": _write_array(output_dir / "held_utility.npy", held_utility),
        "supplement": _write_array(output_dir / "supplement.npy", supplement),
        "supplemental_choice": _write_array(
            output_dir / "supplemental_choice.npy", supplemental_choice
        ),
        "final_chosen": _write_array(output_dir / "final_chosen.npy", final_chosen),
        "final_activation": _write_array(
            output_dir / "final_activation.npy", final_activation
        ),
    }
    if set(files) != set(SELECTOR_ARRAY_FIELDS):
        raise Stage1BError("selector output schema drifted")
    return final_chosen, final_activation, {
        "selected_quantile": float(selected["quantile"]),
        "mapped_reference_threshold": (
            float(mapped_threshold) if np.isfinite(mapped_threshold) else "KEEP"
        ),
        "inner_selection": selected,
        "inner_fit_readiness": readiness_records,
        "inner_oof_coverage_sha256": mechanics._array_sha256(coverage),
        "inner_utility_sha256": mechanics._array_sha256(inner_utility),
        "outer_fit_readiness": outer_readiness,
        "outer_head_models": outer_heads,
        "selector_fits": selector_fits,
        "files": files,
        "supplement_turns": int(supplement.sum()),
        "supplement_sessions": int(np.any(supplement, axis=1).sum()),
        "action_axis_width": width,
        "keep_quantile": float(frozen.KEEP_QUANTILE),
        "regret_multiplier": float(selector.REGRET_MULTIPLIER),
        "maximum_supplements_per_session": int(
            supplement.sum(axis=1).max(initial=0)
        ),
        "within_turn_tie_break": "higher utility, greater support-bit count, lower candidate ordinal",
        "chronological_first_passing_latch_verified": True,
        "keep_preserves_domain_local_current_verified": bool(
            mapped_threshold != math.inf
            or (
                not np.any(supplement)
                and np.array_equal(final_chosen, held.current_chosen)
                and np.array_equal(final_activation, held.current_activation)
            )
        ),
        "held_builder_accepts_labels_state_metric_or_oracle": False,
    }


def _file_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: record[name]
        for name in (
            "sha256",
            "array_sha256",
            "bytes",
            "shape",
            "dtype",
            "asin_shape_matches",
        )
    }


def _selector_identity(
    outer_fold: int,
    stage1a_outer: Mapping[str, Any],
    training_order: np.ndarray,
    held_order: np.ndarray,
    labels_t: Mapping[str, np.ndarray],
    label_record: Mapping[str, Any],
    current_state: Mapping[str, np.ndarray],
    selector_record: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "outer_fold": outer_fold,
        "stage1a_outer_identity": stage1a_outer["identity"],
        "training_order_sha256": mechanics._array_sha256(training_order),
        "held_order_sha256": mechanics._array_sha256(held_order),
        "label_archive_sha256": EXPECTED_LABEL_SHA256,
        "t_only_member_sha256": {
            name: mechanics._array_sha256(value) for name, value in labels_t.items()
        },
        "derived_label_evidence": dict(label_record),
        "current_training_state_sha256": {
            name: mechanics._array_sha256(np.asarray(current_state[name]))
            for name in ("hit", "first_rank", "first_turn")
        },
        "selector": {
            name: selector_record[name]
            for name in (
                "selected_quantile",
                "mapped_reference_threshold",
                "inner_selection",
                "inner_fit_readiness",
                "inner_oof_coverage_sha256",
                "inner_utility_sha256",
                "outer_fit_readiness",
                "outer_head_models",
                "selector_fits",
                "supplement_turns",
                "supplement_sessions",
                "action_axis_width",
                "keep_quantile",
                "regret_multiplier",
                "maximum_supplements_per_session",
                "within_turn_tie_break",
                "chronological_first_passing_latch_verified",
                "keep_preserves_domain_local_current_verified",
                "held_builder_accepts_labels_state_metric_or_oracle",
            )
        },
        "output_arrays": {
            name: _file_identity(selector_record["files"][name])
            for name in SELECTOR_ARRAY_FIELDS
        },
        "privacy": {
            "retained_label_scope": "T_%d only" % outer_fold,
            "held_outcome_rows_copied_retained_serialized_or_supplied": 0,
            "held_state_or_metric_computed": False,
            "runtime_features_mutated_after_label_open": False,
        },
    }


def _validate_repeat(
    first: Mapping[str, Any], repeat: Mapping[str, Any]
) -> Dict[str, Any]:
    if not (
        first.get("outer_fold") == repeat.get("outer_fold")
        and first.get("identity") == repeat.get("identity")
        and first.get("identity_sha256") == repeat.get("identity_sha256")
        == _canonical_sha256(first.get("identity"))
    ):
        raise Stage1BError("selector repeat identity drifted")
    first_files = first.get("files", {})
    repeat_files = repeat.get("files", {})
    if set(first_files) != set(repeat_files):
        raise Stage1BError("selector repeat file registry drifted")
    for name in first_files:
        if _file_identity(first_files[name]) != _file_identity(repeat_files[name]):
            raise Stage1BError("selector repeat physical array drifted")
    return {
        "outer_fold": int(first["outer_fold"]),
        "equal": True,
        "identity_sha256": first["identity_sha256"],
        "physical_arrays_per_pass": len(first_files),
    }


def _validate_held_coverage(held_orders: Sequence[np.ndarray]) -> Dict[str, Any]:
    if len(held_orders) != len(OUTER_FOLDS):
        raise Stage1BError("held fold count is invalid")
    owner = np.full(2000, 255, dtype=np.uint8)
    coverage = np.zeros(2000, dtype=np.uint8)
    order_hashes = []
    for outer_fold, raw in enumerate(held_orders):
        held = np.asarray(raw, dtype=np.int16)
        if (
            held.shape != (400,)
            or len(np.unique(held)) != 400
            or np.any(held < 0)
            or np.any(held >= 2000)
            or np.any(owner[held] != 255)
        ):
            raise Stage1BError("held shard overlap or schema failure")
        owner[held] = outer_fold
        coverage[held] += 1
        order_hashes.append(mechanics._array_sha256(held))
    if np.any(owner == 255) or not np.all(coverage == 1):
        raise Stage1BError("held shards do not cover every session exactly once")
    return {
        "sessions": 2000,
        "unique_sessions": 2000,
        "missing_sessions": 0,
        "overlap_sessions": 0,
        "per_outer_counts": [400] * 5,
        "per_outer_held_order_sha256": order_hashes,
        "coverage_count_array_sha256": mechanics._array_sha256(coverage),
        "outer_fold_by_session_sha256": mechanics._array_sha256(owner),
    }


def _stage2_decision(
    current_chosen: np.ndarray,
    current_activation: np.ndarray,
    final_chosen: np.ndarray,
    final_activation: np.ndarray,
) -> Dict[str, Any]:
    if not (
        current_chosen.shape
        == current_activation.shape
        == final_chosen.shape
        == final_activation.shape
        == (2000, 10)
    ):
        raise Stage1BError("stitched policy schema failed")
    chosen_equal = bool(np.array_equal(current_chosen, final_chosen))
    activation_equal = bool(np.array_equal(current_activation, final_activation))
    identity = chosen_equal and activation_equal
    return {
        "identity_short_circuit": identity,
        "final_chosen_equals_domain_local_current": chosen_equal,
        "final_activation_equals_domain_local_current": activation_equal,
        "stage2_eligible_after_tracked_manifest": not identity,
        "stage2_preparation_authorized_now": False,
        "stage2_outcome_protocol_authorized": False,
        "held_outcome_attach_authorized_now": False,
        "held_outcome_attach_runs": 0,
        "status": (
            "NO_GO_TARGET_FREE_POLICY_IDENTITY"
            if identity
            else "TARGET_FREE_NON_IDENTITY_POLICY_FROZEN"
        ),
    }


def _validate_output_root(output_root: Path) -> Path:
    root = (
        ROOT / "experiments/fast_track/small_ranker_v2_9"
    ).resolve()
    path = Path(output_root).resolve()
    if (
        path.parent != root
        or not path.name.startswith("stage1b_")
        or path.exists()
        or path.is_symlink()
    ):
        raise Stage1BError("output root must be a new direct Stage-1b child")
    return path


def _build_worker(
    output_dir: Path,
    pass_name: str,
    outer_fold: int,
    stage1a_outer: Mapping[str, Any],
    training_order: np.ndarray,
    held_order: np.ndarray,
    labels_t: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    started = time.perf_counter()
    stage_root = STAGE1A_RESULT.parent.resolve()
    runtime_oof = _load_surface(stage_root, stage1a_outer, "oof_T")
    runtime_reference = _load_surface(stage_root, stage1a_outer, "reference_T")
    runtime_held = _load_surface(stage_root, stage1a_outer, "held_H")
    if not (
        mechanics._array_sha256(training_order)
        == stage1a_outer["phases"]["oof_T"]["session_order_sha256"]
        == stage1a_outer["phases"]["reference_T"]["session_order_sha256"]
        and mechanics._array_sha256(held_order)
        == stage1a_outer["phases"]["held_H"]["session_order_sha256"]
    ):
        raise Stage1BError("worker session lineage drifted")
    training = _attach_t_only_labels(runtime_oof, labels_t)
    label_record = _label_identity(training)
    current_state = metric.policy_session_state(
        labels_t, training.current_chosen, training.current_activation
    )
    final_chosen, final_activation, selector_record = _run_selector(
        training,
        labels_t,
        current_state,
        runtime_reference,
        runtime_held,
        output_dir,
    )
    identity = _selector_identity(
        outer_fold,
        stage1a_outer,
        training_order,
        held_order,
        labels_t,
        label_record,
        current_state,
        selector_record,
    )
    record = {
        "pass_name": pass_name,
        "outer_fold": outer_fold,
        "status": "T_ONLY_SELECTOR_COMPLETE",
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "selected_quantile": selector_record["selected_quantile"],
        "mapped_reference_threshold": selector_record["mapped_reference_threshold"],
        "selector_fits": selector_record["selector_fits"],
        "supplement_turns": selector_record["supplement_turns"],
        "supplement_sessions": selector_record["supplement_sessions"],
        "files": selector_record["files"],
        "privacy": {
            "retained_label_scope": "T_%d only" % outer_fold,
            "held_outcome_rows_copied_retained_serialized_or_supplied": 0,
            "held_state_or_metric_computed": False,
            "held_builder_accepts_labels_state_metric_or_oracle": False,
        },
        "resource": {"wall_seconds": round(time.perf_counter() - started, 6)},
    }
    stitched = (
        np.asarray(runtime_held.current_chosen).copy(),
        np.asarray(runtime_held.current_activation).copy(),
        np.asarray(final_chosen).copy(),
        np.asarray(final_activation).copy(),
    )
    return record, stitched


def _audit_output_records(
    output_root: Path,
    worker_records: Mapping[str, Sequence[Mapping[str, Any]]],
    frozen_files: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    records = []
    for pass_name in PASSES:
        workers = worker_records.get(pass_name, ())
        if len(workers) != 5:
            raise Stage1BError("output worker registry is incomplete")
        for outer_fold, worker in enumerate(workers):
            if int(worker.get("outer_fold", -1)) != outer_fold:
                raise Stage1BError("output worker registry order drifted")
            files = worker.get("files", {})
            if set(files) != set(SELECTOR_ARRAY_FIELDS):
                raise Stage1BError("output selector file registry drifted")
            records.extend(files[name] for name in SELECTOR_ARRAY_FIELDS)
    if set(frozen_files) != set(FROZEN_ARRAY_FIELDS):
        raise Stage1BError("frozen stitched file registry drifted")
    records.extend(frozen_files[name] for name in FROZEN_ARRAY_FIELDS)
    seen = set()
    total_bytes = 0
    for record in records:
        raw = Path(str(record.get("path", "")))
        unresolved = ROOT / raw if not raw.is_absolute() else raw
        if unresolved.is_symlink():
            raise Stage1BError("output symlink is forbidden")
        path = unresolved.resolve()
        key = str(path).lower()
        if (
            path.suffix.lower() != ".npy"
            or not _path_below(path, output_root)
            or not path.is_file()
            or key in seen
            or path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(path) != str(record.get("sha256"))
            or mechanics._identity_shape_scan(path) != 0
            or int(record.get("asin_shape_matches", -1)) != 0
        ):
            raise Stage1BError("output physical file audit failed")
        value = mechanics._load_npy_mmap(path)
        if (
            value.shape != tuple(int(item) for item in record.get("shape", []))
            or str(value.dtype) != str(record.get("dtype"))
            or mechanics._array_sha256(value) != str(record.get("array_sha256"))
            or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all())
        ):
            raise Stage1BError("output logical array audit failed")
        seen.add(key)
        total_bytes += path.stat().st_size
    if len(seen) != EXPECTED_OUTPUT_ARRAYS:
        raise Stage1BError("output array count failed")
    actual_files = [path for path in output_root.rglob("*") if path.is_file()]
    if len(actual_files) != len(seen) or sum(path.stat().st_size for path in actual_files) != total_bytes:
        raise Stage1BError("unregistered Stage-1b output file detected")
    return {
        "registered_array_files": len(seen),
        "registered_array_bytes": total_bytes,
        "all_paths_unique_npy_below_output_root": True,
        "all_file_and_array_hashes_verified": True,
        "all_shapes_dtypes_and_finite_values_verified": True,
        "symlink_count": 0,
        "identity_shape_matches": 0,
    }


def _aggregate_identity(
    pairs: Sequence[Mapping[str, Any]],
    held_coverage: Mapping[str, Any],
    frozen_files: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, str],
    git_provenance: Mapping[str, Any],
    environment: Mapping[str, Any],
    label_archive_audit: Mapping[str, Any],
    stage1a_input_audit: Mapping[str, Any],
    physical_output_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage1a_aggregate_identity_sha256": STAGE1A_IDENTITY,
        "per_outer_identity_sha256": [row["identity_sha256"] for row in pairs],
        "held_partition": dict(held_coverage),
        "stitched_policy_arrays": {
            name: _file_identity(frozen_files[name]) for name in FROZEN_ARRAY_FIELDS
        },
        "git_provenance": dict(git_provenance),
        "environment": dict(environment),
        "label_archive_physical_audit": dict(label_archive_audit),
        "stage1a_input_physical_audit": dict(stage1a_input_audit),
        "stage1b_output_physical_audit": dict(physical_output_audit),
        "sources": dict(sources),
    }


def run(output_root: Path, label_archive: Path = DEFAULT_LABEL_ARCHIVE) -> Dict[str, Any]:
    started = time.perf_counter()
    output_root = _validate_output_root(output_root)
    prereg, _stage1a_manifest, stage1a_result, amendment = (
        _validate_protocol_without_labels()
    )
    implementation_commit = _validate_git_checkpoint()
    environment = _validate_environment()
    git_provenance = {
        "parent_stage1a_evidence_commit": PARENT_STAGE1A_COMMIT,
        "stage1b_implementation_commit": implementation_commit,
        "parent_is_ancestor": True,
        "stage1a_manifest_last_change_equals_parent": True,
        "checkpoint_paths_last_change_equals_implementation": True,
    }
    source_start = _source_snapshot()
    for name, expected in EXPECTED_HASHES.items():
        if name in source_start and source_start[name] != expected:
            raise Stage1BError("source drifted before Stage-1b: %s" % name)
    if Path(label_archive).resolve() != DEFAULT_LABEL_ARCHIVE.resolve():
        raise Stage1BError("production Stage-1b label path is fixed")

    _stage0_prereg, frozen_v28_result = stage1a._validate_protocol()
    first_training, first_held = stage1a._load_partition_orders(
        frozen_v28_result, "first"
    )
    repeat_training, repeat_held = stage1a._load_partition_orders(
        frozen_v28_result, "repeat"
    )
    if any(
        not np.array_equal(first_training[fold], repeat_training[fold])
        or not np.array_equal(first_held[fold], repeat_held[fold])
        for fold in OUTER_FOLDS
    ):
        raise Stage1BError("Stage-1a first/repeat partition order drifted")
    held_coverage = _validate_held_coverage(first_held)
    validated_training = tuple(
        _validate_training_order(first_training[fold]) for fold in OUTER_FOLDS
    )
    for outer_fold in OUTER_FOLDS:
        held = np.asarray(first_held[outer_fold], dtype=np.int16)
        complement = np.ones(2000, dtype=bool)
        complement[held] = False
        if not np.array_equal(
            validated_training[outer_fold], np.flatnonzero(complement).astype(np.int16)
        ):
            raise Stage1BError("T_o is not the exact ordered H_o complement")
    first_training = validated_training
    expected_held = _stage1a_manifest.get("held_partition", {})
    if not (
        held_coverage["coverage_count_array_sha256"]
        == expected_held.get("coverage_count_array_sha256")
        and held_coverage["outer_fold_by_session_sha256"]
        == expected_held.get("outer_fold_by_session_sha256")
        and held_coverage["per_outer_held_order_sha256"]
        == expected_held.get("per_outer_held_order_sha256")
        and [mechanics._array_sha256(value) for value in first_training]
        == expected_held.get("per_outer_training_order_sha256")
    ):
        raise Stage1BError("held partition differs from tracked Stage-1a evidence")
    label_archive_start = _audit_label_archive(Path(label_archive))

    output_root.mkdir(parents=True)
    worker_records: Dict[str, list] = {name: [] for name in PASSES}
    pairs = []
    stitched = {
        pass_name: {
            "current_chosen": np.empty((2000, 10), dtype=np.uint8),
            "current_activation": np.empty((2000, 10), dtype=bool),
            "final_chosen": np.empty((2000, 10), dtype=np.uint8),
            "final_activation": np.empty((2000, 10), dtype=bool),
        }
        for pass_name in PASSES
    }
    label_archive_open_count = 0
    label_member_access_count = 0
    for outer_fold in OUTER_FOLDS:
        training_order = _validate_training_order(first_training[outer_fold])
        held_order = np.asarray(first_held[outer_fold], dtype=np.int16)
        labels_t = _load_t_only_labels(Path(label_archive), training_order)
        label_archive_open_count += 1
        label_member_access_count += len(LABEL_MEMBER_SPECS)
        per_pass_stitched = {}
        for pass_name in PASSES:
            stage1a_outer = stage1a_result[pass_name][outer_fold]
            record, shard = _build_worker(
                output_root / pass_name / ("outer_%d" % outer_fold),
                pass_name,
                outer_fold,
                stage1a_outer,
                training_order,
                held_order,
                labels_t,
            )
            worker_records[pass_name].append(record)
            per_pass_stitched[pass_name] = shard
        pair = _validate_repeat(
            worker_records["first"][outer_fold],
            worker_records["repeat"][outer_fold],
        )
        pairs.append(pair)
        first_shard = per_pass_stitched["first"]
        repeat_shard = per_pass_stitched["repeat"]
        if any(
            not np.array_equal(first_shard[index], repeat_shard[index])
            for index in range(4)
        ):
            raise Stage1BError("stitched first/repeat shard drifted")
        for pass_name, shard in per_pass_stitched.items():
            stitched[pass_name]["current_chosen"][held_order] = shard[0]
            stitched[pass_name]["current_activation"][held_order] = shard[1]
            stitched[pass_name]["final_chosen"][held_order] = shard[2]
            stitched[pass_name]["final_activation"][held_order] = shard[3]
        if _audit_label_archive(Path(label_archive)) != label_archive_start:
            raise Stage1BError("sealed label archive changed during outer processing")
        del labels_t, per_pass_stitched, first_shard, repeat_shard
        gc.collect()

    if label_archive_open_count != 5 or label_member_access_count != 25:
        raise Stage1BError("T_o-only archive access count drifted")
    label_archive_end = _audit_label_archive(Path(label_archive))
    if label_archive_end != label_archive_start:
        raise Stage1BError("sealed label archive final physical audit drifted")
    label_archive_audit = {
        "before_first_open": label_archive_start,
        "after_all_training_side_use": label_archive_end,
        "exact": True,
    }
    if any(
        not np.array_equal(stitched["first"][name], stitched["repeat"][name])
        for name in stitched["first"]
    ):
        raise Stage1BError("stitched first/repeat policy drifted")
    first_policy = stitched["first"]
    frozen_files = {
        "domain_local_current_chosen": _write_array(
            output_root / "frozen/domain_local_current_chosen.npy",
            first_policy["current_chosen"],
        ),
        "domain_local_current_activation": _write_array(
            output_root / "frozen/domain_local_current_activation.npy",
            first_policy["current_activation"],
        ),
        "final_chosen": _write_array(
            output_root / "frozen/final_chosen.npy", first_policy["final_chosen"]
        ),
        "final_activation": _write_array(
            output_root / "frozen/final_activation.npy",
            first_policy["final_activation"],
        ),
    }
    decision = _stage2_decision(
        first_policy["current_chosen"],
        first_policy["current_activation"],
        first_policy["final_chosen"],
        first_policy["final_activation"],
    )
    output_audit = _audit_output_records(output_root, worker_records, frozen_files)
    stage1a_input_audit = {
        "validated_before_label_open": True,
        "after_all_stage1b_use": _audit_stage1a_physical(stage1a_result),
        "exact_pre_and_post_physical_registry": True,
    }
    source_end = _source_snapshot()
    if source_end != source_start:
        raise Stage1BError("Stage-1b source changed during execution")
    if _validate_git_checkpoint() != implementation_commit:
        raise Stage1BError("Stage-1b Git checkpoint changed during execution")
    git_provenance["validated_after_all_stage1b_use"] = True
    aggregate_identity = _aggregate_identity(
        pairs,
        held_coverage,
        frozen_files,
        source_end,
        git_provenance,
        environment,
        label_archive_audit,
        stage1a_input_audit,
        output_audit,
    )
    aggregate_identity_sha256 = _canonical_sha256(aggregate_identity)
    rss, peak = mechanics._process_memory()
    total_wall = time.perf_counter() - started
    build_wall = sum(
        float(record["resource"]["wall_seconds"])
        for pass_name in PASSES
        for record in worker_records[pass_name]
    )
    selector_fits = sum(
        int(record["selector_fits"])
        for pass_name in PASSES
        for record in worker_records[pass_name]
    )
    output_bytes = int(output_audit["registered_array_bytes"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.9-STRICT-TOP5-PROPOSAL-DEPTH-STAGE1B",
        "status": "PENDING_RESOURCE_GATE",
        "evidence_boundary": "T_o-only supervised selector freeze and target-free H_o policy identity; no H_o outcome",
        "git": git_provenance,
        "first": worker_records["first"],
        "repeat": worker_records["repeat"],
        "outer_pairs": pairs,
        "held_partition": held_coverage,
        "exact_repeat": {
            "equal": True,
            "outer_shards": 5,
            "aggregate_identity": aggregate_identity,
            "aggregate_identity_sha256": aggregate_identity_sha256,
        },
        "frozen_policy": {"files": frozen_files, **decision},
        "dependencies": environment,
        "stage1a_input_physical_audit": stage1a_input_audit,
        "physical_output_audit": output_audit,
        "sources": source_end,
        "label_scope": {
            "archive_path": str(Path(label_archive).resolve()),
            "archive_sha256": EXPECTED_LABEL_SHA256,
            "archive_bytes": EXPECTED_LABEL_BYTES,
            "archive_open_count": label_archive_open_count,
            "allowed_member_names": list(LABEL_MEMBER_SPECS),
            "member_access_count": label_member_access_count,
            "physical_audit": label_archive_audit,
            "retained_scope": "one current-outer T_o-only bundle at a time",
            "raw_member_arrays_serialized": False,
            "derived_supervised_arrays_serialized": False,
            "compressed_members_may_physically_decompress_H_o": True,
        },
        "privacy": {
            "label_archive_opened": True,
            "t_only_label_rows_per_outer": 1600,
            "simultaneous_outer_label_bundles": 1,
            "held_outcome_rows_copied_retained_serialized_or_supplied": 0,
            "held_state_or_metric_computed": False,
            "offline_held_outcome_attach_runs": 0,
            "agent_or_full_evaluator_started": False,
            "forbidden_split_or_external_data_opened": False,
        },
        "resource": {
            "wall_seconds": round(total_wall, 6),
            "build_worker_wall_seconds": round(build_wall, 6),
            "rss_bytes": int(rss),
            "peak_working_set_bytes": int(peak),
            "output_array_files": EXPECTED_OUTPUT_ARRAYS,
            "output_array_bytes": output_bytes,
            "stage1b_result_bytes": 0,
            "incremental_cache_bytes": output_bytes,
            "stage0_plus_stage1a_plus_stage1b_cache_bytes": PRE_STAGE1B_CACHE_BYTES
            + output_bytes,
            "selector_fits": selector_fits,
            "xgboost_fits": 0,
            "retrieval_queries": 0,
            "new_133_feature_query_candidate_cache_builds": 0,
            "full_agent_or_official_evaluator_runs": 0,
            "offline_held_outcome_attach_runs": 0,
            "workers": 1,
            "budget": {},
        },
        "decision": {
            **decision,
            "stage2_preparation_authorized": False,
            "tracked_stage1b_manifest_required_before_stage2_preparation": not decision[
                "identity_short_circuit"
            ],
            "stage2_outcome_protocol_commit_required_after_manifest": not decision[
                "identity_short_circuit"
            ],
            "stage2_outcome_implementation_commit_required_after_protocol": not decision[
                "identity_short_circuit"
            ],
            "held_outcome_attach_authorized_now": False,
            "runtime_artifact_authorized": False,
            "hr_mrr_mttc_or_technical_score_improvement_claimed": False,
        },
    }
    budget = amendment["resource_gate"]
    previous_state = None
    for _iteration in range(10):
        result_bytes = len(mechanics._serialized_json(result))
        incremental = output_bytes + result_bytes
        cumulative = PRE_STAGE1B_CACHE_BYTES + incremental
        cumulative_wall = 87.409108 + total_wall
        cumulative_peak = max(773_652_480, int(peak))
        checks = {
            "stage1b_wall_seconds": {
                "actual": round(total_wall, 6),
                "maximum": int(
                    budget["first_plus_repeat_stage1b_wall_seconds_maximum"]
                ),
                "pass": total_wall
                <= int(budget["first_plus_repeat_stage1b_wall_seconds_maximum"]),
            },
            "cumulative_wall_seconds": {
                "actual": round(cumulative_wall, 6),
                "maximum": int(
                    budget[
                        "stage0_plus_stage1a_plus_stage1b_wall_seconds_maximum"
                    ]
                ),
                "pass": cumulative_wall
                <= int(
                    budget[
                        "stage0_plus_stage1a_plus_stage1b_wall_seconds_maximum"
                    ]
                ),
            },
            "peak_working_set_bytes": {
                "actual": cumulative_peak,
                "observed_stage1b": int(peak),
                "historical_stage0_plus_stage1a": 773_652_480,
                "maximum": int(budget["peak_working_set_bytes_maximum"]),
                "pass": 0 < int(peak)
                and cumulative_peak
                <= int(budget["peak_working_set_bytes_maximum"]),
            },
            "incremental_cache_bytes": {
                "actual": incremental,
                "maximum": int(budget["stage1b_incremental_cache_bytes_maximum"]),
                "pass": incremental
                <= int(budget["stage1b_incremental_cache_bytes_maximum"]),
            },
            "cumulative_cache_bytes": {
                "actual": cumulative,
                "maximum": int(
                    budget[
                        "stage0_plus_stage1a_plus_stage1b_cache_bytes_maximum"
                    ]
                ),
                "pass": cumulative
                <= int(
                    budget[
                        "stage0_plus_stage1a_plus_stage1b_cache_bytes_maximum"
                    ]
                ),
            },
        }
        budget_pass = all(item["pass"] for item in checks.values())
        result["resource"]["stage1b_result_bytes"] = result_bytes
        result["resource"]["incremental_cache_bytes"] = incremental
        result["resource"][
            "stage0_plus_stage1a_plus_stage1b_cache_bytes"
        ] = cumulative
        result["resource"]["budget"] = checks
        result["status"] = (
            decision["status"]
            if budget_pass
            else "IMPLEMENTATION_FAIL_STAGE1B_RESOURCE_BUDGET"
        )
        if not budget_pass:
            for decision_record in (result["frozen_policy"], result["decision"]):
                decision_record["status"] = "IMPLEMENTATION_FAIL_STAGE1B_RESOURCE_BUDGET"
                decision_record["stage2_eligible_after_tracked_manifest"] = False
                decision_record["stage2_preparation_authorized_now"] = False
                decision_record["stage2_outcome_protocol_authorized"] = False
                decision_record["held_outcome_attach_authorized_now"] = False
            result["decision"]["stage2_preparation_authorized"] = False
            result["decision"][
                "tracked_stage1b_manifest_required_before_stage2_preparation"
            ] = False
            result["decision"][
                "stage2_outcome_protocol_commit_required_after_manifest"
            ] = False
            result["decision"][
                "stage2_outcome_implementation_commit_required_after_protocol"
            ] = False
        state = (result_bytes, incremental, cumulative, budget_pass, result["status"])
        if state == previous_state:
            break
        previous_state = state
    else:
        raise Stage1BError("Stage-1b result-size accounting did not converge")
    if len(mechanics._serialized_json(result)) != result["resource"][
        "stage1b_result_bytes"
    ]:
        raise Stage1BError("Stage-1b result-size accounting drifted")
    mechanics._assert_no_identity_matches(result)
    mechanics._write_json_exclusive(output_root / "stage1b_cache_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args.output_root)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "IMPLEMENTATION_FAIL_STAGE1B_MECHANICS",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "identity_sha256": result["exact_repeat"][
                    "aggregate_identity_sha256"
                ],
                "wall_seconds": result["resource"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {
        "NO_GO_TARGET_FREE_POLICY_IDENTITY",
        "TARGET_FREE_NON_IDENTITY_POLICY_FROZEN",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
