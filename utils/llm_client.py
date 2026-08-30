from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage


LOGGER = logging.getLogger(__name__)
INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')
STRAY_QUOTE_BEFORE_JSON_KEY = re.compile(
    r'"\s+"(?=[A-Za-z_][A-Za-z0-9_]*"\s*:)',
)


class InvalidJSONError(ValueError):
    """A JSON parsing failure that preserves the provider response for audit."""

    def __init__(self, content: str) -> None:
        super().__init__("LLM returned invalid JSON")
        self.content = content


def parse_json_object(content: str, model: str) -> dict[str, Any]:
    """Parse one JSON object with narrow repairs for observed provider defects."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        if error.msg == "Extra data":
            parsed, end = json.JSONDecoder().raw_decode(content)
            if content[end:].strip():
                LOGGER.warning(
                    "Model %s appended non-JSON text after a complete JSON value; "
                    "discarded the trailing text",
                    model,
                )
                return _require_json_object(parsed, model)

        repaired_escapes = INVALID_JSON_ESCAPE.sub(r"\\\\", content)
        if repaired_escapes != content:
            try:
                parsed = json.loads(repaired_escapes)
            except json.JSONDecodeError:
                pass
            else:
                LOGGER.warning(
                    "Model %s returned JSON with invalid escape characters; "
                    "parsed after escaping literal backslashes",
                    model,
                )
                return _require_json_object(parsed, model)

        repaired_quote = STRAY_QUOTE_BEFORE_JSON_KEY.sub('"', repaired_escapes)
        if repaired_quote != repaired_escapes:
            try:
                parsed = json.loads(repaired_quote)
            except json.JSONDecodeError:
                pass
            else:
                LOGGER.warning(
                    "Model %s returned JSON with a stray quote before an object key; "
                    "parsed after removing that quote",
                    model,
                )
                return _require_json_object(parsed, model)

        LOGGER.error("Model %s returned invalid JSON: %s", model, error)
        raise InvalidJSONError(content) from error
    return _require_json_object(parsed, model)


def _require_json_object(parsed: Any, model: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        LOGGER.error(
            "Model %s returned JSON with a non-object root: %s",
            model,
            type(parsed).__name__,
        )
        raise ValueError("LLM JSON response must be an object")
    return parsed


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load OpenAI-compatible LLM configuration from the local environment."""
        load_dotenv(Path.cwd() / ".env")

        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip() or None

        missing = [
            name
            for name, value in (("LLM_API_KEY", api_key), ("LLM_MODEL", model))
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMClient:
    """Non-streaming client for JSON responses from OpenAI-compatible LLMs."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        client_options: dict[str, str] = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_options["base_url"] = self.config.base_url
        self._client = OpenAI(**client_options)
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()
        self._unreported_usage = TokenUsage()

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object from a chat-completion request."""
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if extra_body is not None:
            request["extra_body"] = dict(extra_body)
        response = self._client.chat.completions.create(**request)
        self._record_usage(response.usage)

        content = response.choices[0].message.content if response.choices else None
        if not content:
            LOGGER.error("Model %s returned an empty JSON response", self.config.model)
            raise ValueError("LLM returned an empty JSON response")

        return parse_json_object(content, self.config.model)

    def consume_usage(self) -> TokenUsage:
        """Return and clear token usage not yet reported by the agent."""
        usage = self._unreported_usage
        self._unreported_usage = TokenUsage()
        return usage

    def _record_usage(self, usage: CompletionUsage | None) -> None:
        if usage is None:
            LOGGER.warning(
                "Model %s response did not include token usage",
                self.config.model,
            )
            call_usage = TokenUsage()
        else:
            call_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        self.last_usage = call_usage
        self.total_usage += call_usage
        self._unreported_usage += call_usage
