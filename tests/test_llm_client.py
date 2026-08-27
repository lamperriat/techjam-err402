from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.llm_client import LLMClient, LLMConfig, TokenUsage


class LLMConfigTest(unittest.TestCase):
    @patch("utils.llm_client._load_dotenv")
    def test_loads_provider_neutral_environment_variables(self, load_dotenv: Mock) -> None:
        environment = {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.com/v1",
            "LLM_MODEL": "test-model",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = LLMConfig.from_env()

        self.assertEqual(config, LLMConfig("test-key", "test-model", "https://example.com/v1"))
        load_dotenv.assert_called_once()

    @patch("utils.llm_client._load_dotenv")
    def test_does_not_treat_provider_specific_key_as_fallback(self, load_dotenv: Mock) -> None:
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key", "LLM_MODEL": "test-model"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                LLMConfig.from_env()

    @patch("utils.llm_client.import_module")
    def test_uses_process_environment_without_python_dotenv(
        self, import_module: Mock
    ) -> None:
        import_module.side_effect = ModuleNotFoundError(
            "No module named 'dotenv'", name="dotenv"
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.llm_client.Path.cwd", return_value=Path(directory)):
                with patch.dict(
                    os.environ,
                    {"LLM_API_KEY": "test-key", "LLM_MODEL": "test-model"},
                    clear=True,
                ):
                    config = LLMConfig.from_env()

        self.assertEqual(config, LLMConfig("test-key", "test-model"))

    @patch("utils.llm_client.import_module")
    def test_missing_python_dotenv_has_install_hint_when_dotenv_file_exists(
        self, import_module: Mock
    ) -> None:
        import_module.side_effect = ModuleNotFoundError(
            "No module named 'dotenv'", name="dotenv"
        )
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.touch()
            with patch("utils.llm_client.Path.cwd", return_value=Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "requirements-llm.txt"):
                    LLMConfig.from_env()


class LLMClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk_client = Mock()
        self.config = LLMConfig("test-key", "test-model", "https://example.com/v1")

    def _response(
        self,
        content: str | None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    @patch("utils.llm_client._create_openai_client")
    def test_generates_and_parses_non_streaming_json(self, create_client: Mock) -> None:
        create_client.return_value = self.sdk_client
        self.sdk_client.chat.completions.create.return_value = self._response(
            '{"intent": "buying"}',
            prompt_tokens=12,
            completion_tokens=3,
        )
        client = LLMClient(self.config)

        result = client.generate_json([{"role": "user", "content": "Return JSON"}])

        self.assertEqual(result, {"intent": "buying"})
        self.assertEqual(client.last_usage, TokenUsage(12, 3))
        self.assertEqual(client.total_usage, TokenUsage(12, 3))
        self.sdk_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
            stream=False,
        )

    @patch("utils.llm_client._create_openai_client")
    def test_accumulates_and_consumes_unreported_usage(self, create_client: Mock) -> None:
        create_client.return_value = self.sdk_client
        self.sdk_client.chat.completions.create.side_effect = [
            self._response("{}", 10, 2),
            self._response("{}", 20, 4),
        ]
        client = LLMClient(self.config)

        client.generate_json([{"role": "user", "content": "First"}])
        client.generate_json([{"role": "user", "content": "Second"}])

        self.assertEqual(client.last_usage, TokenUsage(20, 4))
        self.assertEqual(client.total_usage, TokenUsage(30, 6))
        self.assertEqual(client.consume_usage(), TokenUsage(30, 6))
        self.assertEqual(client.consume_usage(), TokenUsage())
        self.assertEqual(client.total_usage, TokenUsage(30, 6))

    @patch("utils.llm_client._create_openai_client")
    def test_logs_invalid_json_and_still_records_its_usage(
        self, create_client: Mock
    ) -> None:
        create_client.return_value = self.sdk_client
        self.sdk_client.chat.completions.create.return_value = self._response(
            "not JSON",
            prompt_tokens=8,
            completion_tokens=2,
        )
        client = LLMClient(self.config)

        with self.assertLogs("utils.llm_client", level="ERROR") as logs:
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                client.generate_json([{"role": "user", "content": "Return JSON"}])

        self.assertIn("returned invalid JSON", logs.output[0])
        self.assertEqual(client.last_usage, TokenUsage(8, 2))
        self.assertEqual(client.total_usage, TokenUsage(8, 2))

    @patch("utils.llm_client._create_openai_client")
    def test_rejects_json_with_non_object_root(self, create_client: Mock) -> None:
        create_client.return_value = self.sdk_client
        self.sdk_client.chat.completions.create.return_value = self._response("[]")
        client = LLMClient(self.config)

        with self.assertLogs("utils.llm_client", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "must be an object"):
                client.generate_json([{"role": "user", "content": "Return JSON"}])

    @patch("utils.llm_client.import_module")
    def test_missing_openai_has_install_hint(self, import_module: Mock) -> None:
        import_module.side_effect = ModuleNotFoundError(
            "No module named 'openai'", name="openai"
        )

        with self.assertRaisesRegex(RuntimeError, "requirements-llm.txt"):
            LLMClient(self.config)


class OptionalDependencyIsolationTest(unittest.TestCase):
    def test_core_agent_and_evaluator_tests_run_without_optional_packages(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = """
import builtins
import unittest

real_import = builtins.__import__

def import_without_optional_packages(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition('.')[0]
    if root in {'openai', 'dotenv'}:
        raise ModuleNotFoundError(f\"No module named '{root}'\", name=root)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_optional_packages

from utils.llm_client import TokenUsage as CompatibleTokenUsage
from utils.types import TokenUsage

assert CompatibleTokenUsage is TokenUsage
suite = unittest.defaultTestLoader.loadTestsFromNames(
    ['tests.test_agent', 'tests.test_evaluator']
)
result = unittest.TextTestRunner(verbosity=0).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
