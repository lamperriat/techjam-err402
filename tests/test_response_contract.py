from __future__ import annotations

import unittest

from starter.response_contract import validate_response


class ResponseContractTest(unittest.TestCase):
    def test_valid_minimal_response(self) -> None:
        response = {
            "message": "ok",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A", "score": 1.0}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.assertEqual(validate_response(response, {"A"}), [])

    def test_rejects_duplicate_unknown_and_nonfinite_recommendations(self) -> None:
        response = {
            "message": "bad",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": "A", "score": float("nan")},
                {"parent_asin": "A", "score": 0.0},
                {"parent_asin": "OUTSIDE", "score": 0.0},
            ],
        }
        errors = validate_response(response, {"A"})
        self.assertIn("recommendation score is not a finite number", errors)
        self.assertIn("recommendation is outside the frozen catalog", errors)
        self.assertIn("recommendations contain duplicate IDs", errors)


if __name__ == "__main__":
    unittest.main()
