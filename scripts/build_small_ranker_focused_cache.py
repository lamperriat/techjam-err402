"""Build the one-shot v2.5 focused LambdaMART query-group cache.

Only frozen ``train_explore`` numeric artifacts are opened.  Product targets
are used to select supervised groups and create binary relevance labels; they
are never copied into the feature tensor or a deployable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_remaining_misses as attribution  # noqa: E402
from scripts import export_small_ranker_fold_safe_artifact as frozen  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-focused-cache-build.v1"
PREREGISTRATION = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_preregistration.json"
)
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_5.focused_lambdamart_implementation_amendment.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
EXPECTED_ACTIVATION_SHA256 = (
    "48ad9137cb3b99985d3d7e4035575bf06225d8c6b4f9f3c134a468f404d1c410"
)
EXPECTED_CHOSEN_SHA256 = (
    "229952c9ced7f6eec1ff1938480adc85ba5093ad865336465749029576e47051"
)
EXPECTED_HARD_SESSIONS = 43
ROWS_PER_GROUP = base.CANDIDATE_COUNT - 9


class FocusedCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class FocusedGroups:
    session: np.ndarray
    turn: np.ndarray
    hard: np.ndarray
    outer_fold: np.ndarray
    inner_fold: np.ndarray
    hard_session: np.ndarray
    control_session: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if ROOT not in resolved.parents:
        raise FocusedCacheError("cache artifact escaped the worktree")
    return resolved.relative_to(ROOT).as_posix()


def _exclusive_npy_memmap(
    path: Path, *, dtype: np.dtype[Any] | type, shape: Sequence[int]
) -> np.memmap:
    """Create a standard C-order ``.npy`` memmap without overwrite semantics."""

    resolved_dtype = np.dtype(dtype)
    resolved_shape = tuple(int(value) for value in shape)
    with path.open("xb") as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {
                "descr": np.lib.format.dtype_to_descr(resolved_dtype),
                "fortran_order": False,
                "shape": resolved_shape,
            },
        )
        offset = handle.tell()
        payload_bytes = int(np.prod(resolved_shape, dtype=np.int64)) * resolved_dtype.itemsize
        if payload_bytes <= 0:
            raise FocusedCacheError("exclusive memmap shape is empty")
        handle.seek(offset + payload_bytes - 1)
        handle.write(b"\0")
    return np.memmap(
        path,
        dtype=resolved_dtype,
        mode="r+",
        offset=offset,
        shape=resolved_shape,
        order="C",
    )


def _allowed_indices(incumbent: int) -> np.ndarray:
    if not 0 <= int(incumbent) < 10:
        raise FocusedCacheError("slot-10 incumbent must be in protected C10")
    allowed = np.asarray(
        [int(incumbent), *range(10, base.CANDIDATE_COUNT)], dtype=np.int16
    )
    if len(allowed) != ROWS_PER_GROUP or len(np.unique(allowed)) != len(allowed):
        raise FocusedCacheError("allowed-91 candidate order is invalid")
    return allowed


def _target_allowed(target: int, incumbent: int) -> bool:
    return int(target) == int(incumbent) or 10 <= int(target) < base.CANDIDATE_COUNT


def _select_groups(
    labels: Mapping[str, np.ndarray],
    current_state: Mapping[str, np.ndarray],
    incumbent: np.ndarray,
) -> FocusedGroups:
    positive = np.asarray(labels["positive_index"], dtype=np.int16)
    eligible_from = np.asarray(labels["eligible_from"], dtype=np.int16)
    current_hit = np.asarray(current_state["hit"], dtype=bool)
    first_rank = np.asarray(current_state["first_rank"], dtype=np.int16)
    first_turn = np.asarray(current_state["first_turn"], dtype=np.int16)
    if positive.shape != incumbent.shape or positive.shape != (
        base.SESSION_COUNT,
        base.TURN_COUNT,
    ):
        raise FocusedCacheError("focused cohort input shape mismatch")

    hard_session = np.zeros(base.SESSION_COUNT, dtype=bool)
    control_session = np.zeros(base.SESSION_COUNT, dtype=bool)
    allowed_turn = np.zeros_like(positive, dtype=bool)
    for session in range(base.SESSION_COUNT):
        eligible_index = int(eligible_from[session]) - 1
        if not 0 <= eligible_index < base.TURN_COUNT:
            raise FocusedCacheError("eligible turn is outside the official horizon")
        for turn in range(eligible_index, base.TURN_COUNT):
            allowed_turn[session, turn] = _target_allowed(
                int(positive[session, turn]), int(incumbent[session, turn])
            ) and int(positive[session, turn]) >= 0
        if not current_hit[session]:
            hard_session[session] = bool(np.any(allowed_turn[session]))
        elif int(first_rank[session]) == 10:
            stop = min(int(first_turn[session]), base.TURN_COUNT)
            control_session[session] = bool(
                np.any(allowed_turn[session, eligible_index:stop])
            )

    if int(hard_session.sum()) != EXPECTED_HARD_SESSIONS:
        raise FocusedCacheError("reachable current-miss cohort drifted")
    if np.any(hard_session & control_session) or not np.any(control_session):
        raise FocusedCacheError("hard/control focused cohorts are invalid")

    sessions: list[int] = []
    turns: list[int] = []
    hard_flags: list[bool] = []
    for session in range(base.SESSION_COUNT):
        if hard_session[session]:
            stop = base.TURN_COUNT
            is_hard = True
        elif control_session[session]:
            stop = min(int(first_turn[session]), base.TURN_COUNT)
            is_hard = False
        else:
            continue
        eligible_index = int(eligible_from[session]) - 1
        for turn in range(eligible_index, stop):
            if allowed_turn[session, turn]:
                sessions.append(session)
                turns.append(turn)
                hard_flags.append(is_hard)

    session_array = np.asarray(sessions, dtype=np.int16)
    turn_array = np.asarray(turns, dtype=np.uint8)
    hard_array = np.asarray(hard_flags, dtype=np.uint8)
    if (
        not len(session_array)
        or len(session_array) != len(turn_array)
        or len(session_array) != len(hard_array)
        or not np.all(np.diff(session_array.astype(np.int32)) >= 0)
    ):
        raise FocusedCacheError("focused query groups are invalid")
    outer = np.asarray(labels["outer_fold"], dtype=np.uint8)[session_array]
    inner = np.asarray(labels["inner_fold"], dtype=np.uint8)[session_array]
    return FocusedGroups(
        session=session_array,
        turn=turn_array,
        hard=hard_array,
        outer_fold=outer,
        inner_fold=inner,
        hard_session=hard_session,
        control_session=control_session,
    )


def _fold_counts(values: np.ndarray, mask: np.ndarray) -> list[int]:
    return [int(np.sum(mask & (values == fold))) for fold in range(base.OUTER_FOLDS)]


def _file_record(path: Path, shape: Sequence[int], dtype: str) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": [int(value) for value in shape],
        "dtype": dtype,
        "asin_shape_matches": base._identity_shape_scan(path),
    }


def run(
    source_root: Path,
    projection_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = output_dir.resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or experiments_root not in output_dir.parents
    ):
        raise FocusedCacheError("output directory must be new and below experiments")
    if not PREREGISTRATION.is_file() or not IMPLEMENTATION_AMENDMENT.is_file():
        raise FocusedCacheError("focused protocol files are unavailable")

    inputs = frozen._load_inputs(source_root, projection_root)
    current_surface = frozen._action_surface(
        inputs.projected_features, inputs.oof_scores, inputs.labels
    )
    current_activation, current_selections = attribution._reproduce_nested_activation(
        current_surface, inputs.labels, seed=40220260830
    )
    activation_sha = hashlib.sha256(current_activation.tobytes()).hexdigest()
    chosen_sha = hashlib.sha256(current_surface.chosen.tobytes()).hexdigest()
    if (
        activation_sha != EXPECTED_ACTIVATION_SHA256
        or chosen_sha != EXPECTED_CHOSEN_SHA256
    ):
        raise FocusedCacheError("frozen current policy did not reproduce")
    current_state = metric.policy_session_state(
        inputs.labels, current_surface.chosen, current_activation
    )
    zero = np.zeros_like(current_activation, dtype=bool)
    p11_state = metric.policy_session_state(
        inputs.labels, current_surface.chosen, zero
    )
    all_sessions = np.ones(base.SESSION_COUNT, dtype=bool)
    current_metrics = metric.transition_metrics(
        p11_state, current_state, current_activation, all_sessions
    )
    if (
        int(np.asarray(current_state["hit"]).sum()) != 1943
        or float(current_metrics["policy"]["hit_rate_at_10"]) != 0.9715
        or int(current_metrics["miss_to_hit"]) != 48
        or int(current_metrics["hit_to_miss"]) != 0
    ):
        raise FocusedCacheError("frozen current comparator metric drifted")

    groups = _select_groups(inputs.labels, current_state, current_surface.incumbent)
    group_count = len(groups.session)
    row_count = group_count * ROWS_PER_GROUP
    output_dir.mkdir(parents=True)
    feature_path = output_dir / "focused_features.npy"
    label_path = output_dir / "focused_relevance.npy"
    metadata_path = output_dir / "focused_groups.npz"
    result_path = output_dir / "cache_result.json"
    features = _exclusive_npy_memmap(
        feature_path,
        dtype=np.float32,
        shape=(group_count, ROWS_PER_GROUP, base.FEATURE_COUNT),
    )
    relevance = _exclusive_npy_memmap(
        label_path,
        dtype=np.uint8,
        shape=(group_count, ROWS_PER_GROUP),
    )
    positive = np.asarray(inputs.labels["positive_index"], dtype=np.int16)
    for group in range(group_count):
        session = int(groups.session[group])
        turn = int(groups.turn[group])
        allowed = _allowed_indices(int(current_surface.incumbent[session, turn]))
        target = int(positive[session, turn])
        features[group] = np.asarray(
            inputs.projected_features[session, turn, allowed], dtype=np.float32
        )
        relevance[group] = (allowed == target).astype(np.uint8)
        if int(relevance[group].sum()) != 1:
            raise FocusedCacheError("focused query group lacks one positive")
    features.flush()
    relevance.flush()
    if not np.isfinite(np.asarray(features)).all():
        raise FocusedCacheError("focused feature cache is non-finite")
    with metadata_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            session_ordinal=groups.session,
            turn_index=groups.turn,
            hard_cohort=groups.hard,
            outer_fold=groups.outer_fold,
            inner_fold=groups.inner_fold,
        )
    feature_record = _file_record(feature_path, features.shape, str(features.dtype))
    label_record = _file_record(label_path, relevance.shape, str(relevance.dtype))
    metadata_record = _file_record(metadata_path, (group_count,), "numeric npz")
    if any(
        int(record["asin_shape_matches"]) != 0
        for record in (feature_record, label_record, metadata_record)
    ):
        raise FocusedCacheError("focused cache identity-shape scan failed")

    outer_sessions = np.asarray(inputs.labels["outer_fold"], dtype=np.uint8)
    inner_sessions = np.asarray(inputs.labels["inner_fold"], dtype=np.uint8)
    family = np.asarray(inputs.labels["family_index"], dtype=np.int32)
    hard_group_mask = groups.hard.astype(bool)
    control_group_mask = ~hard_group_mask
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.5-FOCUSED-LAMBDAMART-CACHE",
        "scope": {
            "split": "train_explore",
            "cached_inputs_only": True,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
            "external_data_downloaded": False,
            "target_used_as_supervised_label_only": True,
            "runtime_features_target_blind": True,
        },
        "sources": {
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "feature_cache_sha256": frozen.EXPECTED_HASHES["features"],
            "projected_features_sha256": frozen.EXPECTED_HASHES[
                "projected_features"
            ],
            "label_cache_sha256": frozen.EXPECTED_HASHES["labels"],
            "current_oof_score_sha256": frozen.EXPECTED_HASHES[
                "projected_oof_scores"
            ],
        },
        "current": {
            "activation_sha256": activation_sha,
            "chosen_sha256": chosen_sha,
            "selection_sha256": _canonical_sha256(current_selections),
            "versus_p11": current_metrics,
        },
        "cache": {
            "features": feature_record,
            "relevance": label_record,
            "groups": metadata_record,
            "feature_count": base.FEATURE_COUNT,
            "candidates_per_group": ROWS_PER_GROUP,
            "query_groups": group_count,
            "rows": row_count,
            "one_positive_per_group": bool(
                np.all(np.asarray(relevance).sum(axis=1) == 1)
            ),
            "feature_names_sha256": _canonical_sha256(list(base.FEATURE_NAMES)),
            "candidate_order": "incumbent,then_indices_10_through_99",
        },
        "cohorts": {
            "hard": {
                "sessions": int(groups.hard_session.sum()),
                "query_groups": int(hard_group_mask.sum()),
                "sessions_by_outer_fold": _fold_counts(
                    outer_sessions, groups.hard_session
                ),
                "groups_by_outer_fold": _fold_counts(
                    groups.outer_fold, hard_group_mask
                ),
                "sessions_by_inner_fold": _fold_counts(
                    inner_sessions, groups.hard_session
                ),
                "groups_by_inner_fold": _fold_counts(
                    groups.inner_fold, hard_group_mask
                ),
                "unique_product_families": int(
                    len(np.unique(family[groups.hard_session]))
                ),
            },
            "control": {
                "sessions": int(groups.control_session.sum()),
                "query_groups": int(control_group_mask.sum()),
                "sessions_by_outer_fold": _fold_counts(
                    outer_sessions, groups.control_session
                ),
                "groups_by_outer_fold": _fold_counts(
                    groups.outer_fold, control_group_mask
                ),
                "sessions_by_inner_fold": _fold_counts(
                    inner_sessions, groups.control_session
                ),
                "groups_by_inner_fold": _fold_counts(
                    groups.inner_fold, control_group_mask
                ),
                "unique_product_families": int(
                    len(np.unique(family[groups.control_session]))
                ),
                "matching_semantics": "behavioral rank-10 controls, not same-target-family matching",
            },
        },
        "privacy": {
            "numeric_arrays_only": True,
            "string_or_object_arrays": 0,
            "identity_shape_matches": 0,
            "session_ordinal_is_grouping_metadata_not_feature": True,
            "target_or_current_score_or_rank_feature_columns": 0,
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.source_root, args.projection_root, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
