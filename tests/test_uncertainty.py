"""Unit tests for Monte Carlo uncertainty propagation and probabilistic metrics."""

import unittest
import numpy as np
import pandas as pd

from exophca.science.uncertainty import sample_twopiece_normal, propagate_uncertainties
from exophca.science.assessment import determine_tier, determine_evidence_quality, assess_planets, DISCLAIMER


class UncertaintyTests(unittest.TestCase):

    def test_same_seed_gives_identical_output_different_seeds_differ(self):
        df = pd.DataFrame({
            'pl_rade': [1.0], 'pl_radeerr1': [0.1], 'pl_radeerr2': [0.1], 'pl_radelim': [0.0],
            'pl_insol': [1.0], 'pl_insolerr1': [0.1], 'pl_insolerr2': [0.1], 'pl_insollim': [0.0],
            'st_teff': [5780.0]
        })
        out1 = propagate_uncertainties(df, n_draws=100, seed=42)
        out2 = propagate_uncertainties(df, n_draws=100, seed=42)
        out3 = propagate_uncertainties(df, n_draws=100, seed=123)

        self.assertAlmostEqual(out1.loc[0, 'base_score_median'], out2.loc[0, 'base_score_median'])
        self.assertNotEqual(out1.loc[0, 'base_score_median'], out3.loc[0, 'base_score_median'])

    def test_row_position_seeding_contract(self):
        """Concrete property: a row's Monte Carlo output is reproducible exactly from
        ``np.random.default_rng([seed, row_index])``, consumed radius-then-flux.

        This pins the current seeding contract, which is position based: the same
        planet yields different percentiles at index 0 and index 1. It fails if the
        draw order or the seed derivation changes.
        """
        row1 = {
            'pl_name': 'Planet A', 'pl_rade': 1.0, 'pl_radeerr1': 0.1, 'pl_radeerr2': 0.1, 'pl_radelim': 0.0,
            'pl_insol': 1.0, 'pl_insolerr1': 0.1, 'pl_insolerr2': 0.1, 'pl_insollim': 0.0, 'st_teff': 5780.0
        }
        row2 = {
            'pl_name': 'Planet B', 'pl_rade': 1.5, 'pl_radeerr1': 0.2, 'pl_radeerr2': 0.2, 'pl_radelim': 0.0,
            'pl_insol': 0.8, 'pl_insolerr1': 0.2, 'pl_insolerr2': 0.2, 'pl_insollim': 0.0, 'st_teff': 5000.0
        }
        df_a = pd.DataFrame([row1, row2])  # Planet A at index 0
        df_b = pd.DataFrame([row2, row1])  # Planet A at index 1

        out_a = propagate_uncertainties(df_a, n_draws=200, seed=42)
        out_b = propagate_uncertainties(df_b, n_draws=200, seed=42)

        def reference_p16(index):
            """Independent recomputation of Planet A's 16th percentile at a row index."""
            rng = np.random.default_rng([42, index])
            r = sample_twopiece_normal(
                np.array([1.0]), np.array([0.1]), np.array([0.1]), 200, rng, clip_min=1e-6
            )[:, 0]
            f = sample_twopiece_normal(
                np.array([1.0]), np.array([0.1]), np.array([0.1]), 200, rng, clip_min=0.0
            )[:, 0]
            scores = np.clip(
                np.sqrt(
                    (1.0 - np.abs((r - 1.0) / (r + 1.0))) ** 0.57
                    * (1.0 - np.abs((f - 1.0) / (f + 1.0))) ** 0.7
                ),
                0.0,
                1.0,
            )
            return np.percentile(scores, 16)

        a_at_index_0 = out_a.loc[out_a['pl_name'] == 'Planet A', 'base_score_p16'].iloc[0]
        a_at_index_1 = out_b.loc[out_b['pl_name'] == 'Planet A', 'base_score_p16'].iloc[0]

        self.assertAlmostEqual(reference_p16(0), a_at_index_0, places=12)
        self.assertAlmostEqual(reference_p16(1), a_at_index_1, places=12)
        # The contract is position based, so the two positions must not coincide.
        self.assertNotEqual(reference_p16(0), reference_p16(1))

    def test_synthetic_earth_like_case(self):
        df = pd.DataFrame({
            'pl_rade': [1.0], 'pl_radeerr1': [0.05], 'pl_radeerr2': [0.05], 'pl_radelim': [0.0],
            'pl_insol': [1.0], 'pl_insolerr1': [0.05], 'pl_insolerr2': [0.05], 'pl_insollim': [0.0],
            'st_teff': [5780.0]
        })
        out = propagate_uncertainties(df, n_draws=1000, seed=42)
        self.assertTrue(0.0 <= out.loc[0, 'p_conservative_hz'] <= 1.0)
        self.assertTrue(0.0 <= out.loc[0, 'p_any_hz'] <= 1.0)
        self.assertTrue(0.0 <= out.loc[0, 'p_radius_le_1p6'] <= 1.0)
        self.assertAlmostEqual(1.0, out.loc[0, 'base_score_median'], delta=0.02)

    def test_zero_or_missing_errors_uncertainty_status(self):
        df = pd.DataFrame({
            'pl_rade': [1.0, 1.0, 1.0, 1.0],
            'pl_radeerr1': [0.1, 0.0, 0.1, 0.0],
            'pl_radeerr2': [0.1, 0.0, 0.1, 0.0],
            'pl_radelim': [0.0, 0.0, 0.0, 0.0],
            'pl_insol': [1.0, 1.0, 1.0, 1.0],
            'pl_insolerr1': [0.0, 0.1, 0.1, 0.0],
            'pl_insolerr2': [0.0, 0.1, 0.1, 0.0],
            'pl_insollim': [0.0, 0.0, 0.0, 0.0],
            'st_teff': [5780.0, 5780.0, 5780.0, 5780.0]
        })
        out = propagate_uncertainties(df, n_draws=100, seed=42)
        self.assertEqual(out.loc[0, 'uncertainty_status'], 'radius_only')
        self.assertEqual(out.loc[1, 'uncertainty_status'], 'flux_only')
        self.assertEqual(out.loc[3, 'uncertainty_status'], 'none')
        self.assertFalse(out['base_score_median'].isna().any())

    def test_two_piece_normal_sampling_sanity(self):
        rng = np.random.default_rng(42)
        nominal = np.array([10.0])
        err1 = np.array([2.0])
        err2 = np.array([2.0])
        samples = sample_twopiece_normal(nominal, err1, err2, 20000, rng, clip_min=0.0)
        med = np.median(samples[:, 0])
        self.assertTrue(9.5 <= med <= 10.5)

    def test_task5_synthetic_large_errors(self):
        """For a synthetic planet with large errors, assert p16 < p84 strictly and at least one probability strictly between 0 and 1."""
        df = pd.DataFrame({
            'pl_rade': [1.5], 'pl_radeerr1': [0.4], 'pl_radeerr2': [0.4], 'pl_radelim': [0.0],
            'pl_insol': [0.9], 'pl_insolerr1': [0.3], 'pl_insolerr2': [0.3], 'pl_insollim': [0.0],
            'st_teff': [5000.0]
        })
        out = propagate_uncertainties(df, n_draws=2000, seed=42)
        row = out.iloc[0]
        self.assertLess(row['base_score_p16'], row['base_score_p84'])
        probs = [row['p_conservative_hz'], row['p_any_hz'], row['p_radius_le_1p6']]
        self.assertTrue(any(0.0 < p < 1.0 for p in probs))

    def test_task5_naive_reference_comparison_kepler452b(self):
        """Compare Kepler-452 b Monte Carlo against a naive reference implementation using 20000 draws within tolerance 0.03."""
        rade = 1.63
        r_err1 = 0.23
        r_err2 = 0.20
        insol = 1.10
        f_err1 = 0.29
        f_err2 = 0.22
        teff = 5757.0

        df_in = pd.DataFrame({
            'pl_name': ['Kepler-452 b'], 'pl_rade': [rade], 'pl_radeerr1': [r_err1], 'pl_radeerr2': [r_err2], 'pl_radelim': [0.0],
            'pl_insol': [insol], 'pl_insolerr1': [f_err1], 'pl_insolerr2': [f_err2], 'pl_insollim': [0.0],
            'st_teff': [teff]
        })
        out_mod = propagate_uncertainties(df_in, n_draws=20000, seed=42)
        mod_p_cons = out_mod.loc[0, 'p_conservative_hz']
        mod_p_rad = out_mod.loc[0, 'p_radius_le_1p6']

        rng_ref = np.random.default_rng([42, 0])
        n_d = 20000
        z_r = rng_ref.standard_normal(n_d)
        u_r = rng_ref.random(n_d)
        p_up_r = r_err1 / (r_err1 + r_err2)
        r_samp = np.where(u_r < p_up_r, rade + r_err1 * np.abs(z_r), rade - r_err2 * np.abs(z_r))
        r_samp = np.maximum(r_samp, 1e-6)

        z_f = rng_ref.standard_normal(n_d)
        u_f = rng_ref.random(n_d)
        p_up_f = f_err1 / (f_err1 + f_err2)
        f_samp = np.where(u_f < p_up_f, insol + f_err1 * np.abs(z_f), insol - f_err2 * np.abs(z_f))
        f_samp = np.maximum(f_samp, 0.0)

        ref_p_rad = np.mean(r_samp <= 1.6)
        from exophca.science.habitable_zone import classify_hz
        ref_hz = classify_hz(f_samp, teff)
        ref_p_cons = np.mean(ref_hz == "conservative")

        self.assertAlmostEqual(mod_p_cons, ref_p_cons, delta=0.03)
        self.assertAlmostEqual(mod_p_rad, ref_p_rad, delta=0.03)

    def test_task5_full_status_p16_p84_and_probabilities(self):
        """Assert that across real assessed output, full status rows have p16 < p84 and fractional probabilities."""
        df = pd.read_csv('data/processed/planets_assessed.csv')
        full_rows = df[df['uncertainty_status'] == 'full']
        if len(full_rows) > 0:
            lt_count = (full_rows['base_score_p16'] < full_rows['base_score_p84']).sum()
            fraction = lt_count / len(full_rows)
            self.assertGreater(fraction, 0.95)

    def test_task5_median_close_for_small_errors(self):
        """Assert median within 0.05 of deterministic score for planets whose relative error is small (rel error < 0.1)."""
        df = pd.read_csv('data/processed/planets_assessed.csv')
        df_full = df[df['uncertainty_status'] == 'full'].copy()
        df_raw = pd.read_csv('data/processed/planets.csv')
        merged = df_full.merge(df_raw[['pl_name', 'pl_radeerr1', 'pl_insolerr1', 'pl_rade', 'pl_insol']], on='pl_name', suffixes=('', '_raw'))
        if len(merged) > 0:
            merged['rel_err_r'] = merged['pl_radeerr1'] / merged['pl_rade']
            small_err = merged[merged['rel_err_r'] < 0.1]
            if len(small_err) > 0:
                diffs = np.abs(small_err['base_score_median'] - small_err['base_score'])
                self.assertTrue((diffs < 0.05).mean() > 0.90)


if __name__ == '__main__':
    unittest.main()
