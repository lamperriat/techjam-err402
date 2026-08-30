"""Extract raw attributes for the full catalog with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tqdm.auto import tqdm

from extract_attribute_pilot import (
    MAX_OUTPUT_TOKENS,
    SCHEMA_VERSION,
    extraction_messages,
    load_catalog,
    product_input,
    validate_extraction,
)
from utils.llm_client import (
    InvalidJSONError,
    LLMClient,
    LLMConfig,
    TokenUsage,
    parse_json_object,
)


LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = 1
DEFAULT_OUTPUT = Path("results/catalog_attributes_raw.jsonl")


class JSONLLM(Protocol):
    config: LLMConfig

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def consume_usage(self) -> TokenUsage:
        ...


@dataclass(frozen=True)
class CatalogRunConfig:
    workers: int = 10
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

    @property
    def max_in_flight(self) -> int:
        return self.workers * 2


@dataclass(frozen=True)
class Checkpoint:
    completed: frozenset[str]
    usage: TokenUsage
    error_records: int


def experiment_metadata(catalog_path: Path, model: str) -> dict[str, Any]:
    """Return settings that must remain stable across resumed runs."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "catalog": str(catalog_path.resolve()),
        "model": model,
        "temperature": 0,
        "thinking": "disabled",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def _usage_from_record(record: Mapping[str, Any]) -> TokenUsage:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )


def load_checkpoint(
    output_path: Path,
    expected_experiment: Mapping[str, Any],
) -> Checkpoint:
    """Read completed ASINs and repair only an incomplete final JSONL line."""
    if not output_path.exists():
        return Checkpoint(frozenset(), TokenUsage(), 0)

    completed: set[str] = set()
    usage = TokenUsage()
    latest_status: dict[str, str] = {}
    metadata_seen = False
    with output_path.open("rb+") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        handle.seek(0)
        line_number = 0
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            at_end = handle.tell() == file_size
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if not at_end:
                    raise ValueError(
                        f"Invalid JSON before the end of {output_path} at line {line_number}"
                    ) from error
                LOGGER.warning(
                    "Discarding incomplete final output line %d from %s",
                    line_number,
                    output_path,
                )
                handle.truncate(line_start)
                break

            if not metadata_seen:
                if record.get("record_type") != "metadata":
                    raise ValueError(f"Missing metadata header in {output_path}")
                if record.get("experiment") != expected_experiment:
                    raise ValueError(
                        f"Existing output uses incompatible extraction settings: {output_path}"
                    )
                metadata_seen = True
            else:
                parent_asin = str(record.get("parent_asin") or "")
                if not parent_asin:
                    raise ValueError(
                        f"Output line {line_number} has no parent_asin in {output_path}"
                    )
                usage += _usage_from_record(record)
                status = str(record.get("status"))
                latest_status[parent_asin] = status
                if status == "success":
                    completed.add(parent_asin)

            if at_end and not raw_line.endswith(b"\n"):
                handle.seek(0, 2)
                handle.write(b"\n")

    if not metadata_seen and output_path.stat().st_size:
        raise ValueError(f"Missing readable metadata header in {output_path}")
    error_records = sum(
        status == "error" and parent_asin not in completed
        for parent_asin, status in latest_status.items()
    )
    return Checkpoint(frozenset(completed), usage, error_records)


def initialize_output(
    output_path: Path,
    experiment: Mapping[str, Any],
) -> None:
    """Create a new JSONL output with one metadata header."""
    if output_path.exists() and output_path.stat().st_size:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"record_type": "metadata", "experiment": experiment},
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def recover_captured_responses(
    output_path: Path,
    products: Sequence[Mapping[str, Any]],
    model: str,
) -> int:
    """Append successes recovered from auditable invalid-JSON responses."""
    latest: dict[str, dict[str, Any]] = {}
    with output_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if line.strip():
                record = json.loads(line)
                latest[str(record["parent_asin"])] = record

    products_by_id = {str(product["parent_asin"]): product for product in products}
    recovered: list[dict[str, Any]] = []
    for parent_asin, record in latest.items():
        raw_response = record.get("raw_response")
        if record.get("status") != "error" or not isinstance(raw_response, str):
            continue
        try:
            raw = parse_json_object(raw_response, model)
        except ValueError:
            continue
        source = product_input(products_by_id[parent_asin])
        validation = validate_extraction(raw, source)
        recovered.append(
            {
                "parent_asin": parent_asin,
                "status": "success",
                "attributes": validation.attributes,
                "rejected_attributes": validation.rejected_attributes,
                "schema_errors": validation.schema_errors,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "latency_seconds": 0,
                "recovered_from_error_record": True,
            }
        )

    if recovered:
        with output_path.open("a", encoding="utf-8") as output:
            for record in recovered:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    return len(recovered)


def extract_product(
    product: Mapping[str, Any],
    llm: JSONLLM,
) -> dict[str, Any]:
    """Extract and validate one product, always returning an auditable record."""
    parent_asin = str(product["parent_asin"])
    started = time.monotonic()
    record: dict[str, Any] = {"parent_asin": parent_asin}
    try:
        source = product_input(product)
        raw = llm.generate_json(
            extraction_messages(source),
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        validation = validate_extraction(raw, source)
        record.update(
            {
                "status": "success",
                "attributes": validation.attributes,
                "rejected_attributes": validation.rejected_attributes,
                "schema_errors": validation.schema_errors,
            }
        )
        if validation.schema_errors:
            record["raw_extraction"] = raw
    except Exception as error:
        LOGGER.error("Attribute extraction failed for %s: %s", parent_asin, error)
        record.update(
            {
                "status": "error",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        if isinstance(error, InvalidJSONError):
            record["raw_response"] = error.content
    finally:
        usage = llm.consume_usage()
        record["usage"] = {
            **usage.as_dict(),
            "total_tokens": usage.total_tokens,
        }
        record["latency_seconds"] = round(time.monotonic() - started, 3)
    return record


def _thread_worker(
    product: Mapping[str, Any],
    thread_state: threading.local,
    client_factory: Callable[[], JSONLLM],
) -> dict[str, Any]:
    client = getattr(thread_state, "client", None)
    if client is None:
        client = client_factory()
        thread_state.client = client
    return extract_product(product, client)


def _write_record(
    output: Any,
    record: Mapping[str, Any],
    progress: tqdm,
    counters: dict[str, Any],
) -> None:
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()
    counters["usage"] += _usage_from_record(record)
    status = str(record.get("status"))
    counters[status] = counters.get(status, 0) + 1
    progress.update(1)
    progress.set_postfix(
        ok=counters.get("success", 0),
        errors=counters.get("error", 0),
        prompt=counters["usage"].prompt_tokens,
        completion=counters["usage"].completion_tokens,
        total_tokens=counters["usage"].total_tokens,
        refresh=False,
    )


def run_catalog_extraction(
    catalog_path: str | Path,
    output_path: str | Path,
    config: CatalogRunConfig,
    llm_config: LLMConfig,
    *,
    client_factory: Callable[[], JSONLLM] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Extract missing catalog products concurrently and append each result."""
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    products = load_catalog(catalog_path)
    catalog_ids = {str(product["parent_asin"]) for product in products}
    experiment = experiment_metadata(catalog_path, llm_config.model)
    checkpoint = load_checkpoint(output_path, experiment)
    unknown_completed = set(checkpoint.completed) - catalog_ids
    if unknown_completed:
        raise ValueError("Existing output contains products outside the catalog")
    initialize_output(output_path, experiment)
    if recover_captured_responses(output_path, products, llm_config.model):
        checkpoint = load_checkpoint(output_path, experiment)

    missing = [
        product
        for product in products
        if str(product["parent_asin"]) not in checkpoint.completed
    ]
    to_process = missing[: config.limit] if config.limit is not None else missing
    factory = client_factory or (lambda: LLMClient(llm_config))
    thread_state = threading.local()
    attempted_this_run = 0
    current_successes = 0
    current_errors = 0
    counters: dict[str, Any] = {
        "usage": checkpoint.usage,
        "success": len(checkpoint.completed),
        "error": 0,
    }
    interrupted = False
    started = time.monotonic()

    if config.limit is None:
        progress_total = len(products)
        progress_initial = len(checkpoint.completed)
    else:
        progress_total = len(to_process)
        progress_initial = 0
        counters["success"] = 0
        counters["error"] = 0

    progress = tqdm(
        total=progress_total,
        initial=progress_initial,
        desc="Extracting catalog attributes",
        unit="product",
        disable=not show_progress,
    )
    progress.set_postfix(
        ok=counters["success"],
        errors=counters["error"],
        prompt=counters["usage"].prompt_tokens,
        completion=counters["usage"].completion_tokens,
        total_tokens=counters["usage"].total_tokens,
        refresh=False,
    )

    executor = ThreadPoolExecutor(max_workers=config.workers)
    pending: dict[Future[dict[str, Any]], str] = {}
    product_iterator = iter(to_process)

    def submit_until_full() -> None:
        while len(pending) < config.max_in_flight:
            try:
                product = next(product_iterator)
            except StopIteration:
                break
            future = executor.submit(_thread_worker, product, thread_state, factory)
            pending[future] = str(product["parent_asin"])

    def persist_future(future: Future[dict[str, Any]], output: Any) -> None:
        nonlocal attempted_this_run, current_successes, current_errors
        record = future.result()
        _write_record(output, record, progress, counters)
        attempted_this_run += 1
        if record.get("status") == "success":
            current_successes += 1
        else:
            current_errors += 1

    try:
        with output_path.open("a", encoding="utf-8") as output:
            submit_until_full()
            try:
                while pending:
                    completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed_futures:
                        pending.pop(future)
                        persist_future(future, output)
                    submit_until_full()
            except KeyboardInterrupt:
                interrupted = True
                LOGGER.warning(
                    "Interrupt received; stopping new work and saving in-flight results"
                )
                for future in pending:
                    future.cancel()
                running = [future for future in pending if not future.cancelled()]
                for future in running:
                    persist_future(future, output)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        progress.close()

    elapsed = time.monotonic() - started
    successful_total = len(checkpoint.completed) + current_successes
    return {
        "catalog_products": len(products),
        "successful_products_total": successful_total,
        "remaining_products": len(products) - successful_total,
        "attempted_this_run": attempted_this_run,
        "successful_this_run": current_successes,
        "errors_this_run": current_errors,
        "workers": config.workers,
        "max_in_flight": config.max_in_flight,
        "elapsed_seconds": round(elapsed, 3),
        "products_per_second": (
            round(attempted_this_run / elapsed, 3) if elapsed else None
        ),
        "usage": {
            **counters["usage"].as_dict(),
            "total_tokens": counters["usage"].total_tokens,
        },
        "interrupted": interrupted,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many currently missing products during this invocation.",
    )
    parser.add_argument(
        "--confirm-api-cost",
        action="store_true",
        help="Acknowledge that the command makes paid LLM API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_api_cost:
        raise SystemExit(
            "Refusing to make API calls without --confirm-api-cost. "
            "Review the concurrency and output path before running."
        )
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    llm_config = LLMConfig.from_env()
    summary = run_catalog_extraction(
        args.catalog,
        args.output,
        CatalogRunConfig(workers=args.workers, limit=args.limit),
        llm_config,
    )
    print(json.dumps(summary, indent=2))
    if summary["interrupted"]:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
