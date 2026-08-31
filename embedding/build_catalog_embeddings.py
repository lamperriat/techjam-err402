from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOCUMENT_SCHEMA_VERSION = 1
CORE_ATTRIBUTES = ("material", "color", "size_fit", "style", "use_case")


@dataclass(frozen=True)
class ProductDocument:
    parent_asin: str
    text: str


def _first_attribute_value(attributes: dict, name: str) -> str | None:
    entries = attributes.get(name)
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    if not isinstance(first, dict):
        return None
    value = first.get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def build_product_text(product: dict, attribute_record: dict) -> str:
    """Build a concise, evidence-derived passage for dense retrieval."""
    attributes = attribute_record.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError(f"Attributes missing for {product.get('parent_asin', '<unknown>')}")

    fields: list[tuple[str, str]] = []
    title = str(product.get("title") or "").strip()
    if title:
        fields.append(("title", title))

    categories = product.get("categories")
    if isinstance(categories, list):
        category = " > ".join(str(value).strip() for value in categories if str(value).strip())
        if category:
            fields.append(("category", category))

    brand = str(product.get("store") or "").strip()
    if brand:
        fields.append(("brand", brand))

    details = product.get("details")
    if isinstance(details, dict):
        department = str(details.get("Department") or "").strip()
        if department:
            fields.append(("department", department))

    for name in CORE_ATTRIBUTES:
        value = _first_attribute_value(attributes, name)
        if value:
            fields.append((name, value))

    specific = attributes.get("specific_attributes")
    specific_values: list[str] = []
    if isinstance(specific, list):
        for item in specific:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            value = str(item.get("value") or "").strip()
            if name and value:
                specific_values.append(f"{name}: {value}")
    if specific_values:
        fields.append(("specific attributes", "; ".join(specific_values)))

    if not fields:
        raise ValueError(f"No embeddable text for {product.get('parent_asin', '<unknown>')}")
    return "\n".join(f"{name}: {value}" for name, value in fields)


def load_documents(
    catalog_path: Path,
    attributes_path: Path,
    limit: int | None = None,
) -> list[ProductDocument]:
    attribute_records: dict[str, dict] = {}
    with attributes_path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        if metadata.get("record_type") != "metadata":
            raise ValueError(f"Missing metadata header in {attributes_path}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            record = json.loads(line)
            parent_asin = str(record.get("parent_asin") or "")
            if not parent_asin:
                raise ValueError(f"Attribute row {line_number} has no parent_asin")
            if parent_asin in attribute_records:
                raise ValueError(f"Duplicate attributes for {parent_asin}")
            attribute_records[parent_asin] = record

    documents: list[ProductDocument] = []
    catalog_ids: set[str] = set()
    with catalog_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin") or "")
            if not parent_asin:
                raise ValueError(f"Catalog row {line_number} has no parent_asin")
            if parent_asin in catalog_ids:
                raise ValueError(f"Duplicate catalog product {parent_asin}")
            catalog_ids.add(parent_asin)
            attribute_record = attribute_records.get(parent_asin)
            if attribute_record is None:
                raise ValueError(f"Processed attributes missing for {parent_asin}")
            documents.append(
                ProductDocument(parent_asin, build_product_text(product, attribute_record))
            )

    extra_attribute_ids = set(attribute_records) - catalog_ids
    if extra_attribute_ids:
        example = min(extra_attribute_ids)
        raise ValueError(f"Processed attributes contain non-catalog product {example}")
    if limit is not None:
        documents = documents[:limit]
    return documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_statistics(documents: Sequence[ProductDocument]) -> dict:
    characters = [len(document.text) for document in documents]
    words = [len(document.text.split()) for document in documents]
    return {
        "products": len(documents),
        "characters": {
            "mean": round(statistics.fmean(characters), 2),
            "maximum": max(characters),
            "total": sum(characters),
        },
        "whitespace_words": {
            "mean": round(statistics.fmean(words), 2),
            "maximum": max(words),
            "total": sum(words),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_device(torch_module: object, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_embeddings(args: argparse.Namespace, documents: list[ProductDocument]) -> dict:
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError(
            "Real embedding requires the packages in embedding/requirements.txt"
        ) from error

    device = _resolve_device(torch, args.device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device in {"cuda", "mps"} else "float32"
    torch_dtype = getattr(torch, dtype_name)
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=args.dimensions,
    )
    model.max_seq_length = args.max_length
    dimensions = model.get_sentence_embedding_dimension()
    if dimensions != args.dimensions:
        raise RuntimeError(f"Expected {args.dimensions} dimensions, model returned {dimensions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "catalog_embeddings.npy"
    ids_path = args.output_dir / "product_ids.json"
    progress_path = args.output_dir / "progress.json"
    manifest_path = args.output_dir / "manifest.json"
    configuration = {
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "model": args.model,
        "dimensions": dimensions,
        "max_length": args.max_length,
        "storage_dtype": args.storage_dtype,
        "product_count": len(documents),
        "catalog_sha256": _sha256(args.catalog),
        "attributes_sha256": _sha256(args.attributes),
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration:
            raise RuntimeError(f"Completed output has a different configuration: {args.output_dir}")
        return manifest

    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration") != configuration:
            raise RuntimeError(f"Partial output has a different configuration: {args.output_dir}")
        completed = int(progress.get("completed_products", 0))
        embeddings = np.load(embeddings_path, mmap_mode="r+")
    else:
        occupied = [path for path in (embeddings_path, ids_path) if path.exists()]
        if occupied:
            raise RuntimeError(f"Output directory contains incomplete files: {occupied[0]}")
        _write_json(ids_path, [document.parent_asin for document in documents])
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=args.storage_dtype,
            shape=(len(documents), dimensions),
        )
        _write_json(
            progress_path,
            {"configuration": configuration, "completed_products": completed},
        )

    if embeddings.shape != (len(documents), dimensions):
        raise RuntimeError(f"Unexpected embedding array shape: {embeddings.shape}")

    progress_bar = tqdm(
        total=len(documents),
        initial=completed,
        desc="Embedding catalog",
        unit="product",
    )
    try:
        for start in range(completed, len(documents), args.batch_size):
            end = min(start + args.batch_size, len(documents))
            batch = [document.text for document in documents[start:end]]
            encoded = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings[start:end] = encoded.astype(args.storage_dtype, copy=False)
            embeddings.flush()
            completed = end
            _write_json(
                progress_path,
                {"configuration": configuration, "completed_products": completed},
            )
            progress_bar.update(end - start)
            progress_bar.set_postfix(device=device, dimensions=dimensions, refresh=False)
    finally:
        progress_bar.close()

    manifest = {
        "configuration": configuration,
        "runtime": {
            "device": device,
            "compute_dtype": dtype_name,
            "batch_size": args.batch_size,
            "normalized_embeddings": True,
        },
        "documents": _document_statistics(documents),
        "files": {
            "embeddings": embeddings_path.name,
            "product_ids": ids_path.name,
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build concise catalog passages and normalized dense embeddings."
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--attributes",
        type=Path,
        default=PROJECT_ROOT / "results/catalog_attributes_processed.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/embeddings/qwen3_embedding_0_6b",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview inputs without importing or downloading the model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.dimensions <= 0 or args.max_length <= 0:
        raise ValueError("--dimensions and --max-length must be positive")

    documents = load_documents(args.catalog, args.attributes, args.limit)
    if not documents:
        raise ValueError("No catalog products were loaded")

    if args.dry_run:
        bytes_per_value = 2 if args.storage_dtype == "float16" else 4
        summary = {
            "dry_run": True,
            "model_was_loaded": False,
            "configuration": {
                "model": args.model,
                "dimensions": args.dimensions,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "requested_device": args.device,
                "storage_dtype": args.storage_dtype,
            },
            "documents": _document_statistics(documents),
            "estimated_embedding_file_mib": round(
                len(documents) * args.dimensions * bytes_per_value / 1024**2,
                2,
            ),
            "examples": [
                {"parent_asin": document.parent_asin, "text": document.text}
                for document in documents[:3]
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0

    manifest = _run_embeddings(args, documents)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
