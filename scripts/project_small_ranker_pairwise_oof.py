"""Score frozen pairwise outer-fold models on semantic-off cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-pairwise-semantic-off-projection.v1"
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
PROJECTED_FEATURE_SHA256 = (
    "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
)
LABEL_SHA256 = (
    "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
)
RAW_PAIRWISE_SCORE_SHA256 = (
    "cfaccf1a26c6987a142493142cf0ac32e84b2bcbd41187da111f8c4547777245"
)
MODEL_HASHES = (
    "bd28443e989f781fafe15853774cc59479eedf678833582061a3100efe49d26e",
    "0ee45185c3bb98411cb212af2ba5a538481a6fdf20427e9913a9e8d001f480a6",
    "2ace1279a863ece15f1021198fb3892fe68788df0e51ecf694a15ed4f0c27f71",
    "d73c4339edac4cbac9ef25cc15d97553068767e09ec1ae01dc3aec3d059cbaf0",
    "6b634f5aed8e21a39820b964bb1bfdb25787cf5e50b14971927599b9338422a5",
)


class PairwiseProjectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_paths(source_root: Path) -> tuple[Path, ...]:
    root = source_root / (
        "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
        "models/pairwise_d4_control"
    )
    paths = tuple(root / f"fold_{fold}.json" for fold in range(base.OUTER_FOLDS))
    for fold, path in enumerate(paths):
        if not path.is_file() or _sha256(path) != MODEL_HASHES[fold]:
            raise PairwiseProjectionError(f"pairwise fold model mismatch: {fold}")
    return paths


def _score_once(
    projected: np.ndarray,
    outer_fold: np.ndarray,
    model_paths: Sequence[Path],
    output_path: Path,
) -> float:
    import xgboost as xgb

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(base.SESSION_COUNT, base.TURN_COUNT, base.CANDIDATE_COUNT),
    )
    started = time.perf_counter()
    for fold, model_path in enumerate(model_paths):
        booster = xgb.Booster()
        booster.load_model(model_path)
        sessions = np.flatnonzero(outer_fold == fold)
        if len(sessions) != 400 or int(booster.num_features()) != base.FEATURE_COUNT:
            raise PairwiseProjectionError("pairwise model/fold shape mismatch")
        for offset in range(0, len(sessions), 25):
            selected = sessions[offset : offset + 25]
            block = np.asarray(projected[selected], dtype=np.float32)
            prediction = booster.predict(
                xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT)),
                output_margin=True,
            )
            scores[selected] = np.asarray(prediction, dtype=np.float32).reshape(
                len(selected), base.TURN_COUNT, base.CANDIDATE_COUNT
            )
    scores.flush()
    if not np.isfinite(np.asarray(scores)).all():
        raise PairwiseProjectionError("pairwise projection produced non-finite scores")
    return time.perf_counter() - started


def _raw_reference_audit(
    source_root: Path,
    outer_fold: np.ndarray,
    model_paths: Sequence[Path],
) -> dict[str, Any]:
    import xgboost as xgb

    feature_path = source_root / "experiments/fast_track/small_ranker_v1/features.npy"
    score_path = source_root / (
        "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
        "oof_scores_pairwise_d4_control.npy"
    )
    if (
        not feature_path.is_file()
        or not score_path.is_file()
        or _sha256(score_path) != RAW_PAIRWISE_SCORE_SHA256
    ):
        raise PairwiseProjectionError("raw pairwise reference is unavailable")
    features = np.load(feature_path, mmap_mode="r")
    reference = np.load(score_path, mmap_mode="r")
    groups = np.asarray([0, 1999, 4001, 7999, 8000, 11999, 12000, 15999, 16000, 19999])
    rows: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for group in groups:
        session = int(group // base.TURN_COUNT)
        turn = int(group % base.TURN_COUNT)
        fold = int(outer_fold[session])
        booster = xgb.Booster()
        booster.load_model(model_paths[fold])
        matrix = np.asarray(features[session, turn], dtype=np.float32)
        rows.append(matrix)
        expected.append(np.asarray(reference[session, turn], dtype=np.float32))
        actual.append(
            np.asarray(
                booster.predict(xgb.DMatrix(matrix), output_margin=True),
                dtype=np.float32,
            )
        )
    expected_matrix = np.stack(expected)
    actual_matrix = np.stack(actual)
    expected_order = np.argsort(-expected_matrix, axis=1, kind="stable")
    actual_order = np.argsort(-actual_matrix, axis=1, kind="stable")
    return {
        "groups": len(groups),
        "rows": sum(len(row) for row in rows),
        "maximum_absolute_error": float(
            np.max(np.abs(expected_matrix - actual_matrix))
        ),
        "c100_order_exact": bool(np.array_equal(expected_order, actual_order)),
    }


def run(source_root: Path, projection_root: Path, output_dir: Path) -> dict[str, Any]:
    import xgboost as xgb

    started = time.perf_counter()
    source_root = source_root.resolve()
    projection_root = projection_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or ROOT not in output_dir.parents:
        raise PairwiseProjectionError("output directory must be new and local")
    projected_path = projection_root / (
        "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
    )
    label_path = source_root / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
    if (
        not projected_path.is_file()
        or _sha256(projected_path) != PROJECTED_FEATURE_SHA256
        or not label_path.is_file()
        or _sha256(label_path) != LABEL_SHA256
    ):
        raise PairwiseProjectionError("projection input identity mismatch")
    projected = np.load(projected_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        outer_fold = np.asarray(archive["outer_fold"])
    if projected.shape != (
        base.SESSION_COUNT,
        base.TURN_COUNT,
        base.CANDIDATE_COUNT,
        base.FEATURE_COUNT,
    ) or projected.dtype != np.float32:
        raise PairwiseProjectionError("projected feature schema mismatch")
    model_paths = _model_paths(source_root)
    raw_audit = _raw_reference_audit(source_root, outer_fold, model_paths)
    if raw_audit["maximum_absolute_error"] != 0.0 or not raw_audit["c100_order_exact"]:
        raise PairwiseProjectionError("pairwise source-model scoring parity failed")

    output_dir.mkdir(parents=True)
    first_path = output_dir / "pairwise_semantic_off_oof.npy"
    repeat_path = output_dir / "pairwise_semantic_off_oof.repeat.npy"
    first_seconds = _score_once(projected, outer_fold, model_paths, first_path)
    repeat_seconds = _score_once(projected, outer_fold, model_paths, repeat_path)
    first_hash = _sha256(first_path)
    repeat_hash = _sha256(repeat_path)
    byte_identical = bool(
        first_hash == repeat_hash
        and first_path.stat().st_size == repeat_path.stat().st_size
    )
    if not byte_identical:
        raise PairwiseProjectionError("pairwise projected OOF repeat differs")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.1-PAIRWISE-PROJECTION",
        "scope": {
            "ranker_retrained": False,
            "labels_read": ["outer_fold"],
            "target_label_read": False,
            "agent_or_evaluator_started": False,
            "held_out_split_opened": False,
        },
        "inputs": {
            "projected_feature_sha256": PROJECTED_FEATURE_SHA256,
            "label_cache_sha256": LABEL_SHA256,
            "model_sha256": list(MODEL_HASHES),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "raw_model_reference_parity": raw_audit,
        "scores": {
            "path": first_path.relative_to(ROOT).as_posix(),
            "bytes": first_path.stat().st_size,
            "sha256": first_hash,
            "repeat_path": repeat_path.relative_to(ROOT).as_posix(),
            "repeat_sha256": repeat_hash,
            "byte_identical": byte_identical,
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "timing_seconds": {
            "first_scoring": round(first_seconds, 6),
            "repeat_scoring": round(repeat_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    result_path = output_dir / "projection_result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"))
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
