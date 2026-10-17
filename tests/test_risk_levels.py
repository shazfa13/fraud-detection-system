
import unittest

from src.api.app import assign_risk_level


class TestAssignRiskLevel(unittest.TestCase):
    def test_risk_level_boundaries(self):
        test_cases = [
            (0, "LOW"),
            (29.9999, "LOW"),
            (30, "MEDIUM"),
            (69.9999, "MEDIUM"),
            (70, "HIGH"),
            (99.9999, "HIGH"),
            (100, "HIGH"),
        ]

        for score, expected in test_cases:
            with self.subTest(score=score):
                self.assertEqual(assign_risk_level(score), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
