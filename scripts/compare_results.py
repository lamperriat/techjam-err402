from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


METRICS = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not contain a JSON object")
    return payload


def metric_value(payload: dict[str, Any], metric: str) -> float | None:
    raw = payload.get(metric)
    if raw is None and metric == "recommended_technical_score":
        raw = payload.get("technical_score")
    return (
        float(raw)
        if isinstance(raw, (int, float, Decimal)) and not isinstance(raw, bool)
        else None
    )


def _fmt(number: float | None) -> str:
    return "—" if number is None else f"{number:.6f}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_report(records: list[tuple[str, dict[str, Any]]]) -> None:
    baseline = records[0][1]
    rows: list[list[str]] = []
    for name, payload in records:
        row = [name]
        for metric in METRICS:
            current = metric_value(payload, metric)
            base = metric_value(baseline, metric)
            delta = None if current is None or base is None else current - base
            row.extend([_fmt(current), "—" if delta is None else f"{delta:+.6f}"])
        rows.append(row)
    headers = ["run"]
    for metric in METRICS:
        headers.extend([metric, "delta"])
    print("\nOverall metrics (deltas are relative to the first file)\n")
    _print_table(headers, rows)

    scenario_rows: list[list[str]] = []
    for name, payload in records:
        grouped_value = payload.get("scenario_metrics")
        grouped = grouped_value if isinstance(grouped_value, dict) else {}
        for scenario in SCENARIOS:
            metrics_value = grouped.get(scenario)
            metrics = metrics_value if isinstance(metrics_value, dict) else {}
            scenario_rows.append([
                name,
                scenario,
                _fmt(metric_value(metrics, "hit_rate_at_10")),
                _fmt(metric_value(metrics, "mrr")),
                _fmt(metric_value(metrics, "mttc")),
            ])
    print("\nScenario metrics\n")
    _print_table(["run", "scenario", "HR@10", "MRR", "MTTC"], scenario_rows)


def _short(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return rendered if len(rendered) <= 180 else rendered[:177] + "..."


def _key_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def differences(expected: object, actual: object, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        result: list[str] = []
        for key in sorted(expected.keys() - actual.keys()):
            result.append(f"{_key_path(path, key)}: missing from actual")
        for key in sorted(actual.keys() - expected.keys()):
            result.append(f"{_key_path(path, key)}: unexpected key in actual")
        for key in sorted(expected.keys() & actual.keys()):
            result.extend(differences(expected[key], actual[key], _key_path(path, key)))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        result = []
        if len(expected) != len(actual):
            result.append(f"{path}: list length {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(differences(left, right, f"{path}[{index}]"))
        return result
    numbers = (int, float, Decimal)
    if (
        isinstance(expected, numbers)
        and not isinstance(expected, bool)
        and isinstance(actual, numbers)
        and not isinstance(actual, bool)
    ):
        return [] if expected == actual else [f"{path}: {_short(expected)} != {_short(actual)}"]
    if type(expected) is not type(actual) or expected != actual:
        return [f"{path}: {_short(expected)} != {_short(actual)}"]
    return []


def assert_equal(records: list[tuple[str, dict[str, Any]]]) -> None:
    expected_name, expected = records[0]
    failed = False
    for actual_name, actual in records[1:]:
        found = differences(expected, actual)
        if not found:
            print(f"STRICT MATCH: {actual_name} == {expected_name}")
            continue
        failed = True
        print(
            f"STRICT MISMATCH: {actual_name} != {expected_name} "
            f"({len(found)} differences)",
            file=sys.stderr,
        )
        for item in found[:20]:
            print(f"  {item}", file=sys.stderr)
        if len(found) > 20:
            print(f"  ... {len(found) - 20} more", file=sys.stderr)
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report or strictly compare TechJam evaluator result JSON files."
    )
    parser.add_argument(
        "--assert-equal",
        action="store_true",
        help="Compare complete parsed JSON objects and exit non-zero on any semantic difference.",
    )
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    if args.assert_equal and len(args.results) < 2:
        parser.error("--assert-equal requires at least two result files")
    records = [(path.stem, load_result(path)) for path in args.results]
    if args.assert_equal:
        assert_equal(records)
    print_report(records)


if __name__ == "__main__":
    main()
