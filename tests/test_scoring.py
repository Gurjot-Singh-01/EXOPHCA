"""Tests for base score calculation (ESI formula, not ML model)."""

import unittest
import numpy as np
from math import sqrt
from pathlib import Path

from exophca.science.esi import (
    calculate_esi,
    classify_flags,
    validate_nonnegative_finite,
    validate_positive_finite,
)


ROOT = Path(__file__).resolve().parents[1]


class ScoringFormulaTests(unittest.TestCase):
    """Test the base ESI formula without ML model references."""

    def test_earth_has_base_esi_one(self):
        self.assertEqual(1.0, calculate_esi(1.0, 1.0))

    def test_base_formula_uses_the_existing_weights(self):
        radius = 1.25
        insolation = 2.0
        expected = sqrt(
            (1 - abs((radius - 1) / (radius + 1))) ** 0.57
            * (1 - abs((insolation - 1) / (insolation + 1))) ** 0.7
        )
        self.assertAlmostEqual(expected, calculate_esi(radius, insolation))

    def test_zero_insolation_is_a_valid_formula_boundary(self):
        self.assertEqual(0.0, calculate_esi(1.0, 0.0))

    def test_classify_flags_and_independent_separation(self):
        flags = classify_flags(1.2, "K")
        self.assertTrue(flags["host_is_k_class"])
        self.assertTrue(flags["radius_in_1_to_1p5"])
        # Verify base score does not include bonuses
        base = calculate_esi(1.2, 1.0)
        self.assertLessEqual(base, 1.0)

    def test_classify_flags_radius_boundaries(self):
        self.assertEqual(
            {"host_is_k_class": False, "radius_in_1_to_1p5": False},
            classify_flags(1.0, "G (Sun-like)")
        )
        self.assertEqual(
            {"host_is_k_class": True, "radius_in_1_to_1p5": False},
            classify_flags(1.0, "K")
        )
        self.assertEqual(
            {"host_is_k_class": False, "radius_in_1_to_1p5": True},
            classify_flags(1.000001, "G (Sun-like)")
        )
        self.assertEqual(
            {"host_is_k_class": False, "radius_in_1_to_1p5": True},
            classify_flags(1.5, "G (Sun-like)")
        )
        self.assertEqual(
            {"host_is_k_class": False, "radius_in_1_to_1p5": False},
            classify_flags(1.500001, "G (Sun-like)")
        )
        self.assertEqual(
            {"host_is_k_class": True, "radius_in_1_to_1p5": True},
            classify_flags(1.5, "K")
        )

    def test_formula_rejects_missing_nonfinite_zero_and_negative_radius(self):
        for invalid in (None, np.nan, np.inf, -np.inf, 0.0, -1.0):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    calculate_esi(invalid, 1.0)

    def test_formula_rejects_missing_nonfinite_and_negative_insolation(self):
        for invalid in (None, np.nan, np.inf, -np.inf, -0.1):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    calculate_esi(1.0, invalid)

    def test_positive_validator_rejects_nonnumeric_values(self):
        with self.assertRaises(ValueError):
            validate_positive_finite("Orbital period", "not a number")
        with self.assertRaises(ValueError):
            classify_flags(1.0, "unknown")
        with self.assertRaises(ValueError):
            classify_flags(1.0, None)

    def test_hand_computed_reference_cases(self):
        """Case 1: Earth (R=1, S=1) -> Expected 1.0"""
        self.assertAlmostEqual(1.0, calculate_esi(1.0, 1.0), places=9)

    def test_no_score_exceeds_one(self):
        for r in (0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 25.0):
            for s in (0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0):
                score = calculate_esi(r, s)
                self.assertLessEqual(score, 1.0)
                self.assertGreaterEqual(score, 0.0)

    def test_golden_baseline_direct_formula(self):
        import pandas as pd
        golden_path = ROOT / "data" / "reference" / "golden_base_score.csv"
        self.assertTrue(golden_path.exists())
        df = pd.read_csv(golden_path)
        self.assertEqual(len(df), 5613)
        for row in df.itertuples(index=False):
            score = calculate_esi(row.pl_rade, row.pl_insol)
            self.assertLessEqual(score, 1.0)
            self.assertAlmostEqual(score, row.base_score, places=9)


if __name__ == "__main__":
    unittest.main()
