"""Unit tests for Kopparapu et al. habitable zone module."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from exophca.science.habitable_zone import hz_limits, classify_hz


class HabitableZoneTests(unittest.TestCase):

    def test_hz_limits_against_oracle(self):
        """Compare hz_limits against oracle CSV at all 26 grid temperatures with tolerance 1e-5."""
        project_root = Path(__file__).resolve().parent.parent
        oracle_path = project_root / 'data' / 'reference' / 'hz_fluxes_oracle.csv'
        if not oracle_path.is_file():
            self.skipTest("Oracle CSV not found.")

        df_oracle = pd.read_csv(oracle_path)
        for _, row in df_oracle.iterrows():
            teff = row['teff']
            limits = hz_limits(teff)
            self.assertTrue(limits['in_valid_range'])
            # Oracle columns: recent_venus, runaway_greenhouse, maximum_greenhouse, early_mars
            self.assertAlmostEqual(limits['optimistic_inner'], row['recent_venus'], places=5)
            self.assertAlmostEqual(limits['conservative_inner'], row['runaway_greenhouse'], places=5)
            self.assertAlmostEqual(limits['conservative_outer'], row['maximum_greenhouse'], places=5)
            self.assertAlmostEqual(limits['optimistic_outer'], row['early_mars'], places=5)

    def test_hz_limits_at_sun_teff(self):
        """At Teff = 5780 exactly, limits equal S_sun."""
        limits = hz_limits(5780.0)
        self.assertTrue(limits['in_valid_range'])
        # S_sun = [1.776 (RV), 1.107 (RG), 0.356 (MG), 0.320 (EM)]
        self.assertAlmostEqual(limits['optimistic_inner'], 1.776, places=5)
        self.assertAlmostEqual(limits['conservative_inner'], 1.107, places=5)
        self.assertAlmostEqual(limits['conservative_outer'], 0.356, places=5)
        self.assertAlmostEqual(limits['optimistic_outer'], 0.320, places=5)

    def test_out_of_range_inputs(self):
        """Out-of-range (2599, 7201, NaN, inf, negative) returns undetermined/out of range, no exception."""
        # For Teff out of range, classify_hz returns undetermined because limits are NaN and in_valid_range is False
        for val in [2599.0, 7201.0, np.nan, np.inf, -100.0, None]:
            lims = hz_limits(val)
            self.assertFalse(lims['in_valid_range'])
            self.assertTrue(np.isnan(lims['conservative_inner']))

            cls = classify_hz(1.0, val)
            self.assertEqual(cls, "undetermined")

        # Also test invalid flux with valid Teff
        for f_val in [np.nan, np.inf, None]:
            cls = classify_hz(f_val, 5780.0)
            self.assertEqual(cls, "undetermined")

        # Negative flux with valid Teff is physically outside HZ (too cold / negative flux)
        cls_neg = classify_hz(-1.0, 5780.0)
        self.assertEqual(cls_neg, "too_cold")

    def test_classify_hz_boundaries(self):
        """Boundary tests for classify_hz at exact conservative and optimistic limits."""
        teff = 5780.0
        limits = hz_limits(teff)
        ci = limits['conservative_inner']
        co = limits['conservative_outer']
        oi = limits['optimistic_inner']
        oo = limits['optimistic_outer']

        # Exact conservative boundaries
        self.assertEqual(classify_hz(ci, teff), "conservative")
        self.assertEqual(classify_hz(co, teff), "conservative")

        # Exact optimistic boundaries
        self.assertEqual(classify_hz(oi, teff), "optimistic_only")
        self.assertEqual(classify_hz(oo, teff), "optimistic_only")

        # Just outside optimistic inner -> too_hot
        self.assertEqual(classify_hz(oi + 0.001, teff), "too_hot")
        # Just outside optimistic outer -> too_cold
        self.assertEqual(classify_hz(oo - 0.001, teff), "too_cold")

    def test_monotonicity_conservative_inner_greater_than_outer(self):
        """Monotonicity check: for Teff between 2600 and 7200, conservative_inner > conservative_outer."""
        teffs = np.arange(2600, 7201, 100)
        limits = hz_limits(teffs)
        ci = limits['conservative_inner']
        co = limits['conservative_outer']
        self.assertTrue(np.all(ci > co))


if __name__ == '__main__':
    unittest.main()
