"""Revalidate v1.5 with the deployable semantic-route-off feature projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_small_ranker_metric_gate as metric  # noqa: E402
from scripts import analyze_small_ranker_three_head_gate as candidate  # noqa: E402
from scripts import train_small_ranker as base  # noqa: E402


SCHEMA_VERSION = "small-ranker-fold-safe-runtime-projection.v1"


def run(source_root: Path, scratch: Path, output: Path) -> dict:
    started = time.perf_counter()
    source_root = source_root.resolve()
    feature_path = source_root / "experiments/fast_track/small_ranker_v1/features.npy"
    label_path = source_root / "experiments/fast_track/small_ranker_v1/labels_v2.npz"
    score_path = source_root / (
        "experiments/fast_track/small_ranker_v1/oof_batch_v1/"
        "oof_scores_runtime_projection_no_semantic.npy"
    )
    for path in (feature_path, label_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if scratch.exists() or output.exists():
        raise FileExistsError(scratch if scratch.exists() else output)

    features = np.load(feature_path, mmap_mode="r")
    projected = np.lib.format.open_memmap(
        scratch,
        mode="w+",
        dtype=np.float32,
        shape=features.shape,
    )
    projection_started = time.perf_counter()
    for offset in range(0, base.SESSION_COUNT, 25):
        projected[offset : offset + 25] = base.project_semantic_route_off(
            np.asarray(features[offset : offset + 25], dtype=np.float32)
        )
    projected.flush()
    projection_seconds = time.perf_counter() - projection_started

    scores = np.load(score_path, mmap_mode="r")
    with np.load(label_path, allow_pickle=False) as archive:
        labels = {name: archive[name] for name in archive.files}
    analysis_started = time.perf_counter()
    comparison = candidate.compare_three_heads(
        projected, scores, labels, seed=40220260830
    )
    analysis_seconds = time.perf_counter() - analysis_started
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAIN_EXPLORE_NESTED_OOF_RUNTIME_PROJECTION_ONLY",
        "source": {
            "feature_cache_sha256": metric._sha256(feature_path),
            "label_cache_sha256": metric._sha256(label_path),
            "projected_score_sha256": metric._sha256(score_path),
            "projected_feature_sha256": metric._sha256(scratch),
            "analyzer_sha256": metric._sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "semantic_route": "missing",
            "projection": "deterministic train_small_ranker.project_semantic_route_off",
            "ranker_retrained": False,
            "heads_retrained_in_nested_oof": True,
            "held_out_splits_opened": False,
            "runtime_or_agent_started": False,
        },
        "comparison": comparison,
        "timing_seconds": {
            "projection": round(projection_seconds, 6),
            "analysis": round(analysis_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "decision": {"promote": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\tiktok\techjam-err402-fast-track"),
    )
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.scratch.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "global": result["comparison"]["global"],
                "folds": result["comparison"]["folds"],
                "label_counts": {
                    "rescue": result["comparison"]["rescue_label_rows"],
                    "rr_regret": result["comparison"]["rr_regret_label_rows"],
                    "hit_loss": result["comparison"]["hit_loss_label_rows"],
                },
                "timing_seconds": result["timing_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
