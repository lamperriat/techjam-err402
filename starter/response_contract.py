"""Strict, target-blind validation for the released Agent response boundary."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from starter.agent import ALLOWED_ATTRIBUTES


SCHEMA_VERSION = "p4.strict-response-contract.v1"


def validate_response(response: object, catalog_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["response is not an object"]
    allowed_keys = {"message", "ask_attribute", "recommendations", "usage"}
    required_keys = {"message", "ask_attribute", "recommendations"}
    if not required_keys <= set(response):
        errors.append("missing required response keys")
    if set(response) - allowed_keys:
        errors.append("response contains undeclared keys")
    if not isinstance(response.get("message"), str):
        errors.append("message is not a string")
    ask = response.get("ask_attribute")
    if ask is not None and (
        not isinstance(ask, str) or ask not in ALLOWED_ATTRIBUTES
    ):
        errors.append("ask_attribute is outside the official enum")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        errors.append("recommendations is not an array")
        return errors
    if len(recommendations) > 10:
        errors.append("recommendations exceeds the repository Top-10 invariant")
    identifiers: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict) or set(item) - {"parent_asin", "score"}:
            errors.append("recommendation is not a contract object")
            continue
        identifier = item.get("parent_asin")
        if not isinstance(identifier, str) or identifier not in catalog_ids:
            errors.append("recommendation is outside the frozen catalog")
            continue
        identifiers.append(identifier)
        score = item.get("score")
        if score is not None and (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            errors.append("recommendation score is not a finite number")
    if len(identifiers) != len(set(identifiers)):
        errors.append("recommendations contain duplicate IDs")
    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or set(usage) != {
            "prompt_tokens",
            "completion_tokens",
        }:
            errors.append("usage does not match the contract")
        elif any(
            not isinstance(usage[key], int)
            or isinstance(usage[key], bool)
            or usage[key] < 0
            for key in usage
        ):
            errors.append("usage token counts are invalid")
    return errors


@dataclass(slots=True)
class ContractRecorder:
    delegate: Any
    catalog_ids: set[str]
    errors: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        started = time.perf_counter()
        try:
            response = self.delegate.respond(session_id, user_message, turn, top_k)
            violations = validate_response(response, self.catalog_ids)
            if violations:
                self.errors.extend(f"turn {turn}: {value}" for value in violations)
                raise ValueError("; ".join(violations))
            return response
        except Exception as exc:
            marker = f"turn {turn}: {type(exc).__name__}: {exc}"
            if marker not in self.errors:
                self.errors.append(marker)
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
