from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = PROJECT_ROOT / "submission"


class SubmissionPackageTest(unittest.TestCase):
    def test_entry_points_resolve_only_to_submission_sources(self) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        script = """
import json
import tempfile
from pathlib import Path
import agent
from starter.agent import Agent as CompatibleAgent

root = Path.cwd().resolve()
assert agent.Agent is agent.AgentV1
assert CompatibleAgent is agent.AgentV1
for implementation in (agent.AgentV1, agent.AgentV2, agent.AgentV3):
    module = __import__(implementation.__module__, fromlist=["__file__"])
    Path(module.__file__).resolve().relative_to(root)

with tempfile.TemporaryDirectory() as directory:
    catalog_path = Path(directory) / "catalog.jsonl"
    catalog_path.write_text(json.dumps({
        "parent_asin": "A",
        "title": "Blue cotton running shoe",
        "features": ["lightweight"],
        "description": ["walking shoe"],
        "price": 20.0,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
        "details": {"Department": "womens"},
        "average_rating": 4.2,
        "rating_number": 10,
        "store": "Example",
    }) + "\\n", encoding="utf-8")
    implementation = agent.Agent(catalog_path)
    implementation.reset("session", {})
    response = implementation.respond(
        "session",
        "I'm looking for Women Shoes, but I'm still exploring.",
        turn=1,
        top_k=10,
    )
    assert response["recommendations"][0]["parent_asin"] == "A"
    implementation.close()
"""

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=SUBMISSION_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_readme_is_copied_and_large_artifacts_are_excluded(self) -> None:
        self.assertEqual(
            (SUBMISSION_ROOT / "README.md").read_bytes(),
            (PROJECT_ROOT / "README.md").read_bytes(),
        )
        self.assertFalse((SUBMISSION_ROOT / "evaluator").exists())
        self.assertFalse(
            (SUBMISSION_ROOT / "results/catalog_attributes_processed.jsonl").exists()
        )


if __name__ == "__main__":
    unittest.main()
