from __future__ import annotations

import ast
import json
from collections import Counter, OrderedDict
from pathlib import Path

import pytest

from scripts.evaluate_small_ranker_smoke import (
    _nearest_rank_percentile,
    _peak_process_rss_bytes,
)
from starter import small_ranker as runtime_module
from starter.agent import Agent, DEFAULT_SMALL_RANKER_MODE, SMALL_RANKER_MODES
from starter.small_ranker import (
    ARTIFACT_SCHEMA_VERSION,
    FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    SmallRankerRuntime,
    SmallRankerRuntimeError,
    admission_utility,
    gate_probability,
    score_tree_model,
    swap_slot10,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ARTIFACT = (
    ROOT
    / "experiments"
    / "fast_track"
    / "small_ranker_v1"
    / "oof_batch_v1"
    / "research_runtime_v1.json"
)
P11_SIDECAR = ROOT / "starter" / "assets" / "p11_features.sqlite"
FOLD_SAFE_ARTIFACT = (
    ROOT / "starter" / "assets" / "small_ranker_fold_safe_v1.json"
)
from scripts.benchmark_small_ranker_runtime import _percentile


def _fixture_catalog(path: Path, count: int = 24) -> Path:
    rows = [
        {
            "parent_asin": f"FIXTURE-{index:03d}",
            "title": f"comfortable black wedding dress {index}",
            "categories": ["Clothing", "Women", "Dresses"],
            "features": ["black", "comfortable", "wedding"],
            "details": {"Color": "Black", "Style": "Wedding"},
            "store": "Fixture",
            "description": "A comfortable black dress.",
            "price": 30 + index,
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _ids(response: dict) -> list[str]:
    return [str(row["parent_asin"]) for row in response["recommendations"]]


def test_runtime_registry_and_dependency_boundary_are_frozen() -> None:
    assert DEFAULT_SMALL_RANKER_MODE == "off"
    assert SMALL_RANKER_MODES == ("off", "shadow", "active")
    assert len(FEATURE_NAMES) == 133 == len(set(FEATURE_NAMES))
    assert len(GATE_FEATURE_NAMES) == 25 == len(set(GATE_FEATURE_NAMES))
    tree = ast.parse((ROOT / "starter" / "small_ranker.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"numpy", "xgboost", "sklearn", "pandas", "torch", "requests"}


def test_pure_tree_score_and_gate_probability() -> None:
    model = {
        "base_score": 0.5,
        "trees": [
            {
                "l": [1, -1, -1],
                "r": [2, -1, -1],
                "f": [0, 0, 0],
                "v": [0.25, 1.0, -1.0],
                "d": [1, 0, 0],
            }
        ],
    }
    assert score_tree_model(model, [0.0]) == pytest.approx(1.5)
    assert score_tree_model(model, [1.0]) == pytest.approx(-0.5)
    gate = {
        "mean": [0.0, 0.0],
        "scale": [1.0, 2.0],
        "coef": [1.0, 2.0],
        "intercept": 0.0,
    }
    assert gate_probability(gate, [0.0, 0.0]) == pytest.approx(0.5)
    assert gate_probability(gate, [1.0, 2.0]) > 0.95
    admission = {
        "rescue_head": gate,
        "rr_regret_head": {**gate, "intercept": -1.0},
        "rr_multiplier": 1.0,
    }
    rescue, regret, utility = admission_utility(admission, [0.0, 0.0])
    assert rescue == pytest.approx(0.5)
    assert regret < rescue
    assert utility == pytest.approx(rescue - regret)


def test_frozen_fold_safe_artifact_loads_fail_closed_by_default() -> None:
    runtime = SmallRankerRuntime("shadow", FOLD_SAFE_ARTIFACT, P11_SIDECAR)
    try:
        status = runtime.status()
        assert status["artifact_schema"] == ARTIFACT_SCHEMA_VERSION
        assert status["configured_mode"] == "shadow"
        assert status["fallback"] is False
        assert runtime.admission is not None
        assert runtime.gate is None
    finally:
        runtime.close()


def test_float32_reductions_match_the_frozen_numpy_projection() -> None:
    pairwise_jaccards = [
        1.0,
        0.1111111119389534,
        0.10000000149011612,
        0.1111111119389534,
        0.3333333432674408,
        0.0,
        0.1111111119389534,
        0.10000000149011612,
        0.1111111119389534,
        0.3333333432674408,
        0.0,
        0.10000000149011612,
        1.0,
        0.4285714328289032,
        0.0,
        0.10000000149011612,
        0.10000000149011612,
        0.0,
        0.4285714328289032,
        0.0,
        0.0,
    ]
    assert runtime_module._mean_f32(pairwise_jaccards) == 0.21277397871017456
    assert runtime_module._f32(runtime_module._entropy([6] * 10 + [0] * 90)) == 1.0


def test_slot10_swap_preserves_protected_ranks_and_full_membership() -> None:
    baseline = tuple(f"ITEM-{index}" for index in range(20))
    changed = swap_slot10(baseline, "ITEM-15", "ITEM-9")
    assert changed[:9] == baseline[:9]
    assert changed[9] == "ITEM-15"
    assert changed[15] == "ITEM-9"
    assert set(changed) == set(baseline)
    with pytest.raises(SmallRankerRuntimeError):
        swap_slot10(baseline, "ITEM-4", "ITEM-9")


def test_evidence_cache_protects_current_candidates_during_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask_count = len(runtime_module.NEGATIVE_SLOT_ORDER)
    rows = {
        rowid: tuple(
            [rowid, identifier, b"fixture"]
            + [0] * mask_count
            + [0.1, 0.2]
        )
        for rowid, identifier in enumerate(("A", "B", "C", "D", "E"), 1)
    }

    class FixtureCursor:
        def __init__(self, values: list[tuple[object, ...]]) -> None:
            self._values = values

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._values)

    class FixtureConnection:
        def execute(self, _query: str, rowids: list[int]) -> FixtureCursor:
            return FixtureCursor([rows[int(rowid)] for rowid in rowids])

    monkeypatch.setattr(runtime_module, "EVIDENCE_CACHE_LIMIT", 3)
    monkeypatch.setattr(
        runtime_module.P11FeatureStore,
        "_decode_feature_blob",
        staticmethod(
            lambda _blob: (
                (("title token",), ("category token",), ("detail token",)),
                frozenset(),
                frozenset(),
                (),
                (),
            )
        ),
    )
    runtime = object.__new__(SmallRankerRuntime)
    runtime.connection = FixtureConnection()
    runtime._evidence_cache = OrderedDict()
    runtime._stats = Counter()
    rowids = {identifier: rowid for rowid, identifier in enumerate(("A", "B", "C", "D", "E"), 1)}

    runtime._fetch_evidence(("A", "B", "C"), rowids)
    result = runtime._fetch_evidence(("A", "D", "E"), rowids)

    assert list(result) == ["A", "D", "E"]
    assert [result[identifier].catalog_rowid for identifier in result] == [1, 4, 5]
    assert list(runtime._evidence_cache) == ["A", "D", "E"]
    assert runtime._stats["evidence_rows_read"] == 5


def test_smoke_resource_measurements_are_stdlib_and_deterministic() -> None:
    values = [float(value) for value in range(1, 101)]
    assert _nearest_rank_percentile(values, 0.95) == 95.0
    assert _nearest_rank_percentile([], 0.95) is None
    with pytest.raises(ValueError):
        _nearest_rank_percentile(values, 0.0)
    assert _peak_process_rss_bytes() > 0
    assert _percentile([3.0, 1.0, 2.0], 0.5) == 2.0


def test_agent_default_off_and_incompatible_active_mode_fail_closed(tmp_path: Path) -> None:
    catalog = _fixture_catalog(tmp_path / "catalog.jsonl")
    control = Agent(catalog, p11_mode="off")
    guarded = Agent(
        catalog,
        p11_mode="off",
        small_ranker_mode="active",
        small_ranker_artifact_path=tmp_path / "missing.json",
    )
    try:
        control.reset("control", {})
        guarded.reset("guarded", {})
        message = "I am looking for a comfortable black dress for a wedding"
        control_response = control.respond("control", message, 1, 10)
        guarded_response = guarded.respond("guarded", message, 1, 10)
        assert _ids(guarded_response) == _ids(control_response)
        assert control.small_ranker_mode == "off"
        diagnostics = guarded.debug_rerank_diagnostics("guarded")["small_ranker"]
        assert diagnostics["fallback"] is True
        assert diagnostics["reason_code"] == "incompatible_agent_configuration"
        assert diagnostics["output_changed"] is False
    finally:
        control.close()
        guarded.close()


@pytest.mark.skipif(
    not RESEARCH_ARTIFACT.is_file() or not P11_SIDECAR.is_file(),
    reason="ignored research artifact is not present",
)
def test_local_research_artifact_loads_without_training_libraries() -> None:
    runtime = SmallRankerRuntime("shadow", RESEARCH_ARTIFACT, P11_SIDECAR)
    try:
        status = runtime.status()
        assert status["effective_mode"] == "shadow"
        assert status["semantic_route"] == "missing"
        assert status["fallback"] is False
    finally:
        runtime.close()
