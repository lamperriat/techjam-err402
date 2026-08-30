"""Project the two missing frozen RRF-3 members onto semantic-off features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-rrf3-semantic-off-projection.v1"
PREREGISTRATION = ROOT / "configs/small_ranker_v2_4.rrf3_preregistration.json"
IMPLEMENTATION_AMENDMENT = ROOT / (
    "configs/small_ranker_v2_4.rrf3_implementation_amendment.json"
)
DEFAULT_SOURCE_ROOT = Path(r"D:\tiktok\techjam-err402-fast-track")
DEFAULT_PROJECTION_ROOT = Path(r"D:\tiktok\techjam-v1-2-metric-gate")
PROJECTED_FEATURE_SHA256 = (
    "cd9b075b923c31afe10b9a9c6d720de22e221c93de961595ff484ea4f532b90a"
)
FEATURE_SHA256 = (
    "2b19835a1bced7f21322610296c712e3d06d915274719e11c268d31f7f596089"
)
LABEL_SHA256 = (
    "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb"
)
MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "ndcg_d6_lr006": {
        "raw_score_sha256": "70a7aaf47640b7d3169fe1c731f9452fb3f8719b6e207500508ef324dbfbaa59",
        "best_iterations": (297, 170, 251, 226, 59),
        "model_sha256": (
            "dd408ab245486383cc2a31972a77dabf471539bee8cec97cc49cc72061fb47a5",
            "fc9f0756459b75358143ed657e15c566014bcf2d426551e8f53e24158356bdf2",
            "82a9d002c42c34849b3f0ede731bd19ba261094810a9ad32ff279b400166df5f",
            "9dc01d4f9abbcae37e10d6504553da25b47fd16d69cb58db7e3643aaba512474",
            "22c1cea8aff72208400b86d5a8642c8aa275fa87c5cf665f1f8921cbee672746",
        ),
    },
    "ndcg_d4_regularized": {
        "raw_score_sha256": "e3d90b5b82778e2a0f50d8a921dcd88ed3eb1c962c64c815e778e90f005977c3",
        "best_iterations": (3, 336, 372, 5, 4),
        "model_sha256": (
            "c929c178d0359efb7e2929d793627fae8f61d584a8376f39c3326a92af00e659",
            "cf7d982d53d90b5d629fe4177e5896e891722a4f6f7e52582edce3d375cc3112",
            "e172bf7f38156a8c6224636fb2d453b4cffdb71435c19c323d9b9968df06b08f",
            "8e7a6268bd463fc4ad8c52eb4e72e53cc7e0b60aeb59c4fe5203c98573b97fda",
            "fa4a989115c927e0b150a28624bbf3bf08b07dca11e63e7490239f561af2c54d",
        ),
    },
}


class RRFProjectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(source_root: Path, model_id: str) -> tuple[Path, tuple[Path, ...]]:
    batch = source_root / "experiments/fast_track/small_ranker_v1/oof_batch_v1"
    raw = batch / f"oof_scores_{model_id}.npy"
    models = tuple(
        batch / "models" / model_id / f"fold_{fold}.json"
        for fold in range(base.OUTER_FOLDS)
    )
    spec = MODEL_SPECS[model_id]
    if not raw.is_file() or _sha256(raw) != spec["raw_score_sha256"]:
        raise RRFProjectionError(f"raw OOF score mismatch: {model_id}")
    for fold, path in enumerate(models):
        if (
            not path.is_file()
            or _sha256(path) != spec["model_sha256"][fold]
        ):
            raise RRFProjectionError(f"fold model mismatch: {model_id}/{fold}")
    return raw, models


def _raw_reference_audit(
    features: np.ndarray,
    reference: np.ndarray,
    outer_fold: np.ndarray,
    model_paths: Sequence[Path],
    best_iterations: Sequence[int],
) -> dict[str, Any]:
    import xgboost as xgb

    groups: list[int] = []
    sampled_folds: list[int] = []
    for fold in range(base.OUTER_FOLDS):
        sessions = np.flatnonzero(outer_fold == fold)
        if len(sessions) != 400:
            raise RRFProjectionError("raw audit fold size mismatch")
        groups.extend((int(sessions[0]) * base.TURN_COUNT, int(sessions[-1]) * base.TURN_COUNT + 9))
        sampled_folds.extend((fold, fold))
    expected: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for group in groups:
        session = int(group // base.TURN_COUNT)
        turn = int(group % base.TURN_COUNT)
        fold = int(outer_fold[session])
        booster = xgb.Booster()
        booster.load_model(model_paths[fold])
        matrix = np.asarray(features[session, turn], dtype=np.float32)
        expected.append(np.asarray(reference[session, turn], dtype=np.float32))
        actual.append(
            np.asarray(
                booster.predict(
                    xgb.DMatrix(matrix),
                    output_margin=True,
                    iteration_range=(0, int(best_iterations[fold]) + 1),
                ),
                dtype=np.float32,
            )
        )
    expected_matrix = np.stack(expected)
    actual_matrix = np.stack(actual)
    return {
        "groups": len(groups),
        "sampled_folds": sampled_folds,
        "rows": int(expected_matrix.size),
        "maximum_absolute_error": float(
            np.max(np.abs(expected_matrix - actual_matrix))
        ),
        "c100_order_exact": bool(
            np.array_equal(
                np.argsort(-expected_matrix, axis=1, kind="stable"),
                np.argsort(-actual_matrix, axis=1, kind="stable"),
            )
        ),
    }


def _score_once(
    projected: np.ndarray,
    outer_fold: np.ndarray,
    model_paths: Sequence[Path],
    best_iterations: Sequence[int],
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
            raise RRFProjectionError("model/fold schema mismatch")
        for offset in range(0, len(sessions), 25):
            selected = sessions[offset : offset + 25]
            block = np.asarray(projected[selected], dtype=np.float32)
            prediction = booster.predict(
                xgb.DMatrix(block.reshape(-1, base.FEATURE_COUNT)),
                output_margin=True,
                iteration_range=(0, int(best_iterations[fold]) + 1),
            )
            scores[selected] = np.asarray(prediction, dtype=np.float32).reshape(
                len(selected), base.TURN_COUNT, base.CANDIDATE_COUNT
            )
    scores.flush()
    if not np.isfinite(np.asarray(scores)).all():
        raise RRFProjectionError("semantic-off projection is non-finite")
    return time.perf_counter() - started


def run(source_root: Path, projection_root: Path, output_dir: Path) -> dict[str, Any]:
    import xgboost as xgb

    started = time.perf_counter()
    source_root = source_root.resolve()
    projection_root = projection_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or ROOT not in output_dir.parents:
        raise RRFProjectionError("output directory must be new and local")
    projected_path = projection_root / (
        "experiments/fast_track/small_ranker_fold_safe_projected_features.npy"
    )
    feature_path = source_root / "experiments/fast_track/small_ranker_v1/features.npy"
    label_path = source_root / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
    if (
        not projected_path.is_file()
        or _sha256(projected_path) != PROJECTED_FEATURE_SHA256
        or not feature_path.is_file()
        or _sha256(feature_path) != FEATURE_SHA256
        or not label_path.is_file()
        or _sha256(label_path) != LABEL_SHA256
    ):
        raise RRFProjectionError("projection input identity mismatch")
    projected = np.load(projected_path, mmap_mode="r")
    features = np.load(feature_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        outer_fold = np.asarray(archive["outer_fold"])
    if projected.shape != features.shape or projected.dtype != np.float32:
        raise RRFProjectionError("projected feature schema mismatch")

    output_dir.mkdir(parents=True)
    members: list[dict[str, Any]] = []
    for model_id, spec in MODEL_SPECS.items():
        raw_path, model_paths = _paths(source_root, model_id)
        raw_reference = np.load(raw_path, mmap_mode="r")
        audit = _raw_reference_audit(
            features,
            raw_reference,
            outer_fold,
            model_paths,
            spec["best_iterations"],
        )
        if (
            audit["maximum_absolute_error"] != 0.0
            or not audit["c100_order_exact"]
            or sorted(set(audit["sampled_folds"])) != list(range(base.OUTER_FOLDS))
        ):
            raise RRFProjectionError(f"raw reference parity failed: {model_id}")
        first_path = output_dir / f"{model_id}_semantic_off_oof.npy"
        repeat_path = output_dir / f"{model_id}_semantic_off_oof.repeat.npy"
        first_seconds = _score_once(
            projected,
            outer_fold,
            model_paths,
            spec["best_iterations"],
            first_path,
        )
        repeat_seconds = _score_once(
            projected,
            outer_fold,
            model_paths,
            spec["best_iterations"],
            repeat_path,
        )
        first_hash = _sha256(first_path)
        repeat_hash = _sha256(repeat_path)
        if first_hash != repeat_hash:
            raise RRFProjectionError(f"projection repeat differs: {model_id}")
        members.append(
            {
                "id": model_id,
                "raw_score_sha256": str(spec["raw_score_sha256"]),
                "raw_reference_parity": audit,
                "model_sha256": list(spec["model_sha256"]),
                "best_iterations_zero_based": list(spec["best_iterations"]),
                "score_path": first_path.relative_to(ROOT).as_posix(),
                "repeat_path": repeat_path.relative_to(ROOT).as_posix(),
                "score_sha256": first_hash,
                "repeat_sha256": repeat_hash,
                "byte_identical": True,
                "bytes": first_path.stat().st_size,
                "timing_seconds": {
                    "first": round(first_seconds, 6),
                    "repeat": round(repeat_seconds, 6),
                },
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "SR-V2.4-RRF3-PROJECTION",
        "scope": {
            "labels_read": ["outer_fold"],
            "target_label_read": False,
            "ranker_retrained": False,
            "agent_or_evaluator_started": False,
            "held_out_splits_opened": False,
        },
        "inputs": {
            "projected_feature_sha256": PROJECTED_FEATURE_SHA256,
            "feature_cache_sha256": FEATURE_SHA256,
            "label_cache_sha256": LABEL_SHA256,
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "implementation_amendment_sha256": _sha256(
                IMPLEMENTATION_AMENDMENT
            ),
            "projector_sha256": _sha256(Path(__file__).resolve()),
        },
        "members": members,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    result_path = output_dir / "projection_result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
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
