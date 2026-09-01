from __future__ import annotations

import unittest

from interactive import _LineBuffer, run_repl
from utils.llm_client import TokenUsage


class FakeInteractiveAgent:
    def __init__(self) -> None:
        self.token_usage = TokenUsage(12, 3)
        self.closed = False

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you prefer a wide fit?",
            "recommendations": [
                {
                    "title": "Example shoe",
                    "price": 25.0,
                    "average_rating": 4.5,
                    "rating_number": 25,
                    "score": 0.9,
                    "description": "A " + "very " * 50 + "comfortable shoe.",
                }
            ],
            "end_conversation": True,
        }

    def close(self) -> None:
        self.closed = True


class InteractiveTest(unittest.TestCase):
    def test_line_buffer_inserts_at_the_arrow_adjusted_cursor(self) -> None:
        buffer = _LineBuffer()
        for character in "shoes":
            buffer.insert(character)

        self.assertTrue(buffer.move_left())
        self.assertTrue(buffer.move_left())
        buffer.insert(" ")
        self.assertTrue(buffer.move_right())

        self.assertEqual(buffer.text, "sho es")
        self.assertEqual(buffer.cursor, 5)

    def test_repl_prints_recommendation_details_and_conversation_tokens(self) -> None:
        agent = FakeInteractiveAgent()
        output: list[str] = []

        run_repl(agent, input_fn=lambda _: "running shoes", output_fn=output.append)

        rendered = "\n".join(output)
        self.assertIn("Example shoe", rendered)
        self.assertIn("$25.00", rendered)
        self.assertIn("4.5/5", rendered)
        self.assertIn("score 0.9000", rendered)
        self.assertIn("…", rendered)
        self.assertIn("input=12, output=3, total=15", rendered)
        self.assertTrue(agent.closed)

    def test_quit_ends_without_calling_the_agent(self) -> None:
        agent = FakeInteractiveAgent()
        output: list[str] = []

        run_repl(agent, input_fn=lambda _: "/quit", output_fn=output.append)

        self.assertIn("Goodbye!", output)
        self.assertTrue(agent.closed)


if __name__ == "__main__":
    unittest.main()
