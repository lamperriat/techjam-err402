from __future__ import annotations

import unittest

from scripts.verify_promoted_agent import _stable_sha256


class VerifyPromotedAgentTest(unittest.TestCase):
    def test_stable_hash_ignores_mapping_order_but_not_list_order(self) -> None:
        self.assertEqual(
            _stable_sha256({"a": 1, "b": 2}),
            _stable_sha256({"b": 2, "a": 1}),
        )
        self.assertNotEqual(_stable_sha256([1, 2]), _stable_sha256([2, 1]))


if __name__ == "__main__":
    unittest.main()
