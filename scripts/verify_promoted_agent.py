from __future__ import annotations

"""Prove the served coverage Agent is functionally bridged to frozen R08."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.frozen_winner import FROZEN_WINNER_ID


SCHEMA_VERSION = "p4.promoted-agent-verification.v1"
ROUTES = ("broad", "strict", "fused", "reranked", "final")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(
    frozen_public: dict[str, Any],
    promoted_public: dict[str, Any],
    frozen_resource: dict[str, Any],
    reference_bridge: dict[str, Any],
    promoted_resource: dict[str, Any],
) -> dict[str, Any]:
    frozen_suites = frozen_public["corpora"]["released_public"]["suites"]
    promoted_suites = promoted_public["corpora"]["released_public"]["suites"]
    suite_hashes = {
        name: {
            "frozen_result_sha256": _stable_sha256(value["result"]),
            "promoted_result_sha256": _stable_sha256(
                promoted_suites[name]["result"]
            ),
        }
        for name, value in frozen_suites.items()
    }
    for value in suite_hashes.values():
        value["exact"] = (
            value["frozen_result_sha256"] == value["promoted_result_sha256"]
        )

    reference_run = reference_bridge["runs"][0]
    promoted_run = promoted_resource["runs"][0]
    reference_routes = reference_run["target_blind_route_sha256"]
    promoted_routes = promoted_run["target_blind_route_sha256"]
    route_equivalence = {
        route: reference_routes[route] == promoted_routes[route]
        for route in ROUTES
    }
    checks = {
        "frozen_generalization_gate_passed": bool(
            frozen_public.get("frozen_winner_gate", {}).get("passed")
        ),
        "frozen_resource_gate_passed": bool(
            frozen_resource.get("frozen_winner_gate", {}).get("passed")
        ),
        "reference_bridge_gate_passed": bool(
            reference_bridge.get("bridge_gate", {}).get("passed")
        ),
        "promoted_public_uses_served_coverage_mode": (
            promoted_public.get("configuration", {}).get("architecture_variant") is None
            and promoted_public.get("configuration", {}).get("retrieval_mode") == "coverage"
            and promoted_public.get("configuration", {}).get("rerank_mode") == "off"
        ),
        "promoted_resource_uses_served_coverage_mode": (
            promoted_resource.get("configuration", {}).get("architecture_variant") is None
            and promoted_resource.get("configuration", {}).get("retrieval_mode") == "coverage"
            and promoted_resource.get("configuration", {}).get("rerank_mode") == "off"
        ),
        "all_nine_suite_results_exact": (
            len(suite_hashes) == 9 and all(value["exact"] for value in suite_hashes.values())
        ),
        "promoted_complete_trace_deterministic": (
            promoted_resource.get("determinism", {}).get("status") == "passed"
        ),
        "promoted_contract_clean": all(
            not run.get("contract_errors") for run in promoted_resource.get("runs", [])
        ),
        "promoted_no_key_default": bool(
            promoted_resource.get("all_runs_no_key_default_verified")
        ),
        "response_trace_exact": (
            reference_run["target_blind_response_sha256"]
            == promoted_run["target_blind_response_sha256"]
        ),
        "broad_strict_fused_final_routes_exact": all(
            route_equivalence[route] for route in ("broad", "strict", "fused", "final")
        ),
        "promoted_reranked_route_preserves_control_fused": (
            promoted_routes["reranked"] == promoted_routes["fused"]
        ),
        "reference_wrapper_reranked_route_is_coverage_final": (
            reference_routes["reranked"] == reference_routes["final"]
        ),
        "promoted_public_snapshot_stable": bool(
            promoted_public.get("provenance", {}).get("snapshot_stable")
        ),
        "promoted_resource_snapshot_stable": bool(
            promoted_resource.get("provenance", {}).get("snapshot_stable")
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "winner": FROZEN_WINNER_ID,
        "decision": "verified" if all(checks.values()) else "rejected",
        "passed": all(checks.values()),
        "checks": checks,
        "suite_result_hashes": suite_hashes,
        "route_equivalence": route_equivalence,
        "response_trace": {
            "reference_sha256": reference_run["target_blind_response_sha256"],
            "promoted_sha256": promoted_run["target_blind_response_sha256"],
        },
        "boundary": (
            "Frozen artifacts anchor selection before promotion. The bridge rerun proves the "
            "shared helper retains the reference output, while the promoted runs independently "
            "exercise the actual served Agent. No private-session claim is made."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the served P4 coverage promotion.")
    parser.add_argument(
        "--frozen-public", type=Path, default=Path("experiments/p4_r08_public_all.json")
    )
    parser.add_argument(
        "--promoted-public",
        type=Path,
        default=Path("experiments/p4_promoted_public_all.json"),
    )
    parser.add_argument(
        "--frozen-resource", type=Path, default=Path("experiments/p4_r08_resources.json")
    )
    parser.add_argument(
        "--reference-bridge",
        type=Path,
        default=Path("experiments/p4_reference_bridge_resources.json"),
    )
    parser.add_argument(
        "--promoted-resource",
        type=Path,
        default=Path("experiments/p4_promoted_resources.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/p4_promoted_verification.json")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = (
        args.frozen_public,
        args.promoted_public,
        args.frozen_resource,
        args.reference_bridge,
        args.promoted_resource,
    )
    artifact = verify(*(_load(path) for path in paths))
    artifact["inputs"] = {
        str(path): _sha256(path) for path in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[promoted-verification] {artifact['decision']}; wrote {args.output}")
    return 0 if artifact["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
