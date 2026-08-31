from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


CORE_ATTRIBUTE_FIELDS = ("material", "color", "size_fit", "style", "use_case")


class ExtractedAttributeIndex:
    """Read-only primary values from the processed offline LLM extraction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.product_ids: set[str] = set()
        self._core: dict[str, dict[str, str]] = {}
        self._specific: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            metadata = json.loads(next(handle))
            if metadata.get("record_type") != "metadata":
                raise ValueError(f"Missing metadata header in {self.path}")
            for line_number, line in enumerate(handle, start=2):
                if not line.strip():
                    continue
                record = json.loads(line)
                parent_asin = str(record.get("parent_asin") or "")
                if not parent_asin:
                    raise ValueError(
                        f"Processed attribute row {line_number} has no parent_asin"
                    )
                if parent_asin in self.product_ids:
                    raise ValueError(f"Duplicate processed attributes for {parent_asin}")
                self.product_ids.add(parent_asin)
                attributes = record.get("attributes")
                if not isinstance(attributes, dict):
                    raise ValueError(f"Processed attributes missing for {parent_asin}")

                core: dict[str, str] = {}
                for field in CORE_ATTRIBUTE_FIELDS:
                    entries = attributes.get(field)
                    if isinstance(entries, list) and entries:
                        value = entries[0].get("value")
                        if isinstance(value, str) and value:
                            core[field] = value
                self._core[parent_asin] = core

                specific_values: defaultdict[str, list[str]] = defaultdict(list)
                entries = attributes.get("specific_attributes")
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        name = entry.get("name")
                        value = entry.get("value")
                        if isinstance(name, str) and isinstance(value, str) and value:
                            specific_values[name].append(value)
                self._specific[parent_asin] = {
                    name: values[0] for name, values in specific_values.items()
                }

    def core_value(self, parent_asin: str, field: str) -> str | None:
        return self._core[parent_asin].get(field)

    def specific_value(self, parent_asin: str, name: str) -> str | None:
        return self._specific[parent_asin].get(name)
