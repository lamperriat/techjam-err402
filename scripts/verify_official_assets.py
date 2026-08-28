from __future__ import annotations

"""Verify frozen organizer artifacts without modifying them."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EVALUATOR_BLOB = "7c808347b31ef3121a9cbc4810ac3eb325f950ba"
EXPECTED_PUBLIC_BLOB = "121dbec9c1368c81cd887d6959e62507512139c0"
EXPECTED_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
EXPECTED_CATALOG_GZIP_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
EXPECTED_FIELDS = {
    "parent_asin",
    "title",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "average_rating",
    "rating_number",
    "store",
}
EXPECTED_SCENARIOS = {
    "buying": 80,
    "browsing": 80,
    "intent_override": 30,
    "boundary": 10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path, *, normalize_lf: bool = True) -> str:
    payload = path.read_bytes()
    if normalize_lf:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def catalog_summary(path: Path) -> tuple[dict[str, Any], set[str]]:
    identifiers: set[str] = set()
    row_count = 0
    duplicate_count = 0
    empty_id_count = 0
    schema_mismatch_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            product = json.loads(line)
            if set(product) != EXPECTED_FIELDS:
                schema_mismatch_count += 1
            identifier = str(product.get("parent_asin") or "").strip()
            if not identifier:
                empty_id_count += 1
            elif identifier in identifiers:
                duplicate_count += 1
            else:
                identifiers.add(identifier)
    return ({
        "row_count": row_count,
        "unique_id_count": len(identifiers),
        "duplicate_id_count": duplicate_count,
        "empty_id_count": empty_id_count,
        "schema_mismatch_count": schema_mismatch_count,
        "sha256": sha256(path),
    }, identifiers)


def public_summary(path: Path, catalog_ids: set[str]) -> dict[str, Any]:
    samples: set[str] = set()
    targets: set[str] = set()
    scenarios: Counter[str] = Counter()
    row_count = 0
    duplicate_sample_count = 0
    duplicate_target_count = 0
    missing_target_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            sample = json.loads(line)
            sample_id = str(sample.get("sample_id") or "")
            target = str(sample.get("ground_truth", {}).get("parent_asin") or "")
            scenario = str(sample.get("scenario_type") or "")
            duplicate_sample_count += int(sample_id in samples)
            duplicate_target_count += int(target in targets)
            samples.add(sample_id)
            targets.add(target)
            scenarios[scenario] += 1
            missing_target_count += int(target not in catalog_ids)
    return {
        "row_count": row_count,
        "unique_sample_count": len(samples),
        "unique_target_count": len(targets),
        "duplicate_sample_count": duplicate_sample_count,
        "duplicate_target_count": duplicate_target_count,
        "missing_target_count": missing_target_count,
        "scenario_counts": dict(sorted(scenarios.items())),
        "git_blob_sha1_lf": git_blob_sha1(path),
    }


def audit_repository(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    catalog_path = root / "data" / "catalog.jsonl"
    public_path = root / "data" / "public_set.jsonl"
    evaluator_path = root / "evaluator" / "local_evaluator.py"
    gzip_path = root / "data" / "releases" / "catalog.jsonl.gz"
    catalog, catalog_ids = catalog_summary(catalog_path)
    public = public_summary(public_path, catalog_ids)
    evaluator_blob = git_blob_sha1(evaluator_path)
    gzip_hash = sha256(gzip_path) if gzip_path.exists() else None
    checks = {
        "catalog_sha256": catalog["sha256"] == EXPECTED_CATALOG_SHA256,
        "catalog_rows": catalog["row_count"] == 50_000,
        "catalog_unique_ids": catalog["unique_id_count"] == 50_000,
        "catalog_no_duplicates": catalog["duplicate_id_count"] == 0,
        "catalog_no_empty_ids": catalog["empty_id_count"] == 0,
        "catalog_exact_schema": catalog["schema_mismatch_count"] == 0,
        "public_blob": public["git_blob_sha1_lf"] == EXPECTED_PUBLIC_BLOB,
        "public_rows": public["row_count"] == 200,
        "public_unique_samples": public["unique_sample_count"] == 200,
        "public_unique_targets": public["unique_target_count"] == 200,
        "public_targets_in_catalog": public["missing_target_count"] == 0,
        "public_scenario_mix": public["scenario_counts"] == EXPECTED_SCENARIOS,
        "evaluator_blob": evaluator_blob == EXPECTED_EVALUATOR_BLOB,
        "optional_gzip_hash": gzip_hash in {None, EXPECTED_CATALOG_GZIP_SHA256},
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "catalog": catalog,
        "public_set": public,
        "evaluator_git_blob_sha1_lf": evaluator_blob,
        "catalog_gzip_sha256": gzip_hash,
        "catalog_gzip_present": gzip_path.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    report = audit_repository(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
