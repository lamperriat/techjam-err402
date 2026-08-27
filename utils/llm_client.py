from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from utils.types import TokenUsage


LOGGER = logging.getLogger(__name__)
LLM_INSTALL_HINT = (
    "Install optional LLM dependencies with "
    "`python -m pip install -r requirements-llm.txt`."
)


def _load_dotenv(path: Path) -> None:
    try:
        dotenv = import_module("dotenv")
    except ModuleNotFoundError as error:
        if error.name != "dotenv":
            raise
        if path.is_file():
            raise RuntimeError(
                f"Loading LLM configuration from {path} requires python-dotenv. "
                f"{LLM_INSTALL_HINT}"
            ) from error
        return
    dotenv.load_dotenv(path)


def _create_openai_client(client_options: dict[str, str]) -> Any:
    try:
        openai = import_module("openai")
    except ModuleNotFoundError as error:
        if error.name != "openai":
            raise
        raise RuntimeError(
            f"LLMClient requires the optional openai package. {LLM_INSTALL_HINT}"
        ) from error
    return openai.OpenAI(**client_options)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load OpenAI-compatible LLM configuration from the local environment."""
        _load_dotenv(Path.cwd() / ".env")

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


class LLMClient:
    """Non-streaming client for JSON responses from OpenAI-compatible LLMs."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        client_options: dict[str, str] = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_options["base_url"] = self.config.base_url
        self._client = _create_openai_client(client_options)
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()
        self._unreported_usage = TokenUsage()

    def generate_json(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        """Generate and parse a JSON object from a chat-completion request."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[dict(message) for message in messages],
            response_format={"type": "json_object"},
            stream=False,
        )
        self._record_usage(getattr(response, "usage", None))

        content = response.choices[0].message.content if response.choices else None
        if not content:
            LOGGER.error("Model %s returned an empty JSON response", self.config.model)
            raise ValueError("LLM returned an empty JSON response")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            LOGGER.error(
                "Model %s returned invalid JSON: %s",
                self.config.model,
                error,
            )
            raise ValueError("LLM returned invalid JSON") from error

        if not isinstance(parsed, dict):
            LOGGER.error(
                "Model %s returned JSON with a non-object root: %s",
                self.config.model,
                type(parsed).__name__,
            )
            raise ValueError("LLM JSON response must be an object")

        return parsed

    def consume_usage(self) -> TokenUsage:
        """Return and clear token usage not yet reported by the agent."""
        usage = self._unreported_usage
        self._unreported_usage = TokenUsage()
        return usage

    def _record_usage(self, usage: object) -> None:
        call_usage = TokenUsage(
            prompt_tokens=max(0, int(getattr(usage, "prompt_tokens", 0) or 0)),
            completion_tokens=max(0, int(getattr(usage, "completion_tokens", 0) or 0)),
        )
        self.last_usage = call_usage
        self.total_usage += call_usage
        self._unreported_usage += call_usage
