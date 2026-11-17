"""Unit tests for the radius model: filters, splits, baselines, intervals, guard."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from exophca.ml import radius_model as rm

PLANETS = Path("data/processed/planets.csv")
ARTIFACT = Path("models/radius_model.joblib")

# Values verified against the stage-1 report; they are regression expectations.
EXPECTED_FILTER_COUNTS = [
    ("raw rows", 6319),
    ('radius_provenance == "measured"', 4742),
    ("pl_radeerr1 present and != 0", 4359),
    ('mass_provenance == "Mass"', 1695),
    ("all five features present and > 0", 1658),
]
EXPECTED_SPLITS = {"train": (1069, 826, 16), "calibration": (274, 207, 3), "test": (315, 258, 10)}
EXPECTED_TEST_HOST_HASH = "71117774fc26099a96bb97ce26fcc612a69adeb9d58837fdbb5573cf0d80fd8e"


class FilterAndSplitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = rm.load_planets(PLANETS)
        cls.selected, cls.steps = rm.filter_training_rows(cls.raw)
        cls.split_df, cls.summary = rm.split_by_host(cls.selected)

    def test_filter_counts_are_stable(self):
        self.assertEqual(
            [(step.label, step.rows) for step in self.steps],
            EXPECTED_FILTER_COUNTS,
        )
        self.assertEqual(len(self.selected), EXPECTED_FILTER_COUNTS[-1][1])

    def test_split_sizes_and_rv_counts_are_stable(self):
        table = rm.split_table(self.split_df).set_index("split")
        for name, (rows, hosts, rv) in EXPECTED_SPLITS.items():
            self.assertEqual(int(table.loc[name, "rows"]), rows, name)
            self.assertEqual(int(table.loc[name, "hosts"]), hosts, name)
            self.assertEqual(int(table.loc[name, "rv_rows"]), rv, name)

    def test_test_host_hash_is_stable(self):
        self.assertEqual(self.summary.test_host_hash, EXPECTED_TEST_HOST_HASH)

    def test_split_hosts_never_overlap(self):
        train = set(self.summary.train_hosts)
        calibration = set(self.summary.calibration_hosts)
        test = set(self.summary.test_hosts)
        self.assertFalse(train & calibration)
        self.assertFalse(train & test)
        self.assertFalse(calibration & test)

    def test_features_never_include_the_target(self):
        self.assertNotIn(rm.TARGET, rm.FEATURES)
        self.assertNotIn("pl_insol", rm.FEATURES)
        with self.assertRaises(ValueError):
            rm.assert_no_forbidden_features(["pl_bmasse", "pl_insol"])
        with self.assertRaises(ValueError):
            rm.assert_no_forbidden_features(["p_any_hz"])
        with self.assertRaises(ValueError):
            rm.assert_no_forbidden_features(["base_score"])


class MassStratumTests(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(rm.mass_stratum(9.9), rm.MASS_STRATA[0])
        self.assertEqual(rm.mass_stratum(10.0), rm.MASS_STRATA[1])
        self.assertEqual(rm.mass_stratum(100.0), rm.MASS_STRATA[1])
        self.assertEqual(rm.mass_stratum(100.1), rm.MASS_STRATA[2])
        self.assertIsNone(rm.mass_stratum(float("nan")))

    def test_piecewise_segments_are_half_open(self):
        self.assertEqual(rm.piecewise_segment(9.999), rm.PIECEWISE_SEGMENTS[0])
        self.assertEqual(rm.piecewise_segment(10.0), rm.PIECEWISE_SEGMENTS[1])
        self.assertEqual(rm.piecewise_segment(99.999), rm.PIECEWISE_SEGMENTS[1])
        self.assertEqual(rm.piecewise_segment(100.0), rm.PIECEWISE_SEGMENTS[2])


class PiecewiseBaselineTests(unittest.TestCase):

    def _features(self, masses):
        log_mass = np.log10(masses)
        return np.column_stack(
            [log_mass, np.zeros_like(log_mass), np.zeros_like(log_mass),
             np.zeros_like(log_mass), np.zeros_like(log_mass)]
        )

    def test_recovers_each_segment_exactly(self):
        masses = np.array([1.0, 3.0, 9.0, 10.0, 40.0, 99.0, 100.0, 700.0, 4000.0])
        log_mass = np.log10(masses)
        intercepts = {rm.PIECEWISE_SEGMENTS[0]: 0.1, rm.PIECEWISE_SEGMENTS[1]: -0.2,
                      rm.PIECEWISE_SEGMENTS[2]: 1.1}
        slopes = {rm.PIECEWISE_SEGMENTS[0]: 0.5, rm.PIECEWISE_SEGMENTS[1]: 0.3,
                  rm.PIECEWISE_SEGMENTS[2]: -0.01}
        y = np.array([
            intercepts[rm.piecewise_segment(m)] + slopes[rm.piecewise_segment(m)] * np.log10(m)
            for m in masses
        ])
        model = rm.PiecewisePowerLawBaseline().fit(self._features(masses), y)
        self.assertTrue(np.allclose(model.predict(self._features(masses)), y))
        self.assertAlmostEqual(model.segments_[rm.PIECEWISE_SEGMENTS[2]][1], -0.01)
        self.assertEqual(model.fallbacks_, {})

    def test_single_row_segment_falls_back_to_a_constant(self):
        masses = np.array([2.0, 3.0, 4.0, 50.0, 60.0, 1000.0])
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
        model = rm.PiecewisePowerLawBaseline().fit(self._features(masses), y)
        self.assertEqual(model.fallbacks_[rm.PIECEWISE_SEGMENTS[2]], 1)
        prediction = model.predict(self._features(np.array([1000.0, 5000.0])))[0]
        self.assertAlmostEqual(prediction, 1.0)
        self.assertTrue(np.all(np.isfinite(model.predict(self._features(masses)))))


class ConformalTests(unittest.TestCase):

    def test_order_statistic_is_the_conformal_quantile(self):
        scores = np.arange(1, 101, dtype=float)
        value, achieved, k = rm.conformal_order_statistic(scores, 0.9)
        self.assertEqual(k, int(np.ceil((len(scores) + 1) * 0.9)))
        self.assertEqual(value, float(scores[k - 1]))
        self.assertAlmostEqual(achieved, k / (len(scores) + 1))

    def test_split_conformal_reaches_nominal_coverage(self):
        rng = np.random.default_rng(7)
        calibration = np.abs(rng.normal(0.0, 0.2, 2000))
        q, _, _ = rm.conformal_order_statistic(calibration, 0.9)
        fresh = np.abs(rng.normal(0.0, 0.2, 4000))
        self.assertAlmostEqual(float((fresh <= q).mean()), 0.9, delta=0.03)

    def test_interval_table_reports_every_stratum(self):
        y = np.array([0.0, 0.2, -0.1, 0.3, 0.1, 0.0])
        lower = np.full(6, -0.5)
        upper = np.full(6, 0.5)
        masses = np.array([1.0, 5.0, 20.0, 60.0, 300.0, 900.0])
        table = rm.interval_table(y, lower, upper, masses)
        self.assertEqual(list(table["stratum"]), ["all rows", *rm.MASS_STRATA])
        self.assertTrue((table["coverage"] == 1.0).all())
        self.assertTrue((table["mean_width_dex"] > 0).all())

    def test_interval_check_flags_a_stratum_that_under_covers(self):
        table = pd.DataFrame(
            {
                "stratum": ["all rows", *rm.MASS_STRATA],
                "n": [400, 100, 100, 200],
                "coverage": [0.90, 0.75, 0.93, 0.94],
                "mean_width_dex": [0.5] * 4,
                "mean_width_rearth": [1.0] * 4,
            }
        )
        check = rm.interval_check(table, 0.90, 0.02)
        self.assertFalse(check["passes"])
        self.assertEqual(check["worst_stratum"], rm.MASS_STRATA[0])
        self.assertAlmostEqual(check["shortfall"], 0.15)

    def test_selection_rule_prefers_the_narrower_calibrated_method(self):
        def table(width, coverages):
            return pd.DataFrame(
                {
                    "stratum": ["all rows", *rm.MASS_STRATA],
                    "n": [400, 100, 100, 200],
                    "coverage": coverages,
                    "mean_width_dex": [width] * 4,
                    "mean_width_rearth": [1.0] * 4,
                }
            )

        wide = table(0.60, [0.91, 0.90, 0.91, 0.92])
        narrow = table(0.50, [0.91, 0.90, 0.91, 0.92])
        winner, _ = rm.recommend_interval_method({"split_conformal": wide, "cqr": narrow})
        self.assertEqual(winner, "cqr")

        under = table(0.40, [0.91, 0.70, 0.92, 0.93])
        winner, reason = rm.recommend_interval_method({"split_conformal": wide, "cqr": under})
        self.assertEqual(winner, "split_conformal")
        self.assertIn("coverage", reason)


class TestGuardTests(unittest.TestCase):

    def test_fresh_meta_allows_evaluation(self):
        allowed, message = rm.test_evaluation_allowed(None)
        self.assertTrue(allowed)
        self.assertEqual(message, "")

    def test_second_evaluation_is_refused_without_force(self):
        meta = {"test_evaluated": True, "test_evaluated_on": "2026-10-20",
                "test_host_hash": "abc"}
        allowed, message = rm.test_evaluation_allowed(meta, force=False)
        self.assertFalse(allowed)
        self.assertIn("already evaluated", message)
        self.assertIn("--force", message)

    def test_force_overrides_and_warns(self):
        meta = {"test_evaluated": True, "test_evaluated_on": "2026-10-20"}
        allowed, message = rm.test_evaluation_allowed(meta, force=True)
        self.assertTrue(allowed)
        self.assertIn("--force", message)


@unittest.skipUnless(ARTIFACT.exists(), "run scripts/train_radius_model.py --stage 2 first")
class ArtifactTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import joblib

        cls.artifact = joblib.load(ARTIFACT)

    def test_artifact_has_the_documented_fields(self):
        for key in ("point_model", "features", "target", "feature_bounds",
                    "split_conformal_half_width_dex", "recommended_interval_method",
                    "test_host_hash"):
            self.assertIn(key, self.artifact)
        self.assertEqual(self.artifact["features"], list(rm.FEATURES))
        self.assertEqual(self.artifact["test_host_hash"], EXPECTED_TEST_HOST_HASH)

    def test_artifact_predicts_and_intervals_are_ordered(self):
        X = np.array([[0.0, 1.5, 5800.0, 1.0, 1.0], [2.0, 2.0, 4500.0, 0.8, 0.7]])
        prediction = self.artifact["point_model"].predict(X)
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        q = self.artifact["split_conformal_half_width_dex"]
        self.assertGreater(q, 0.0)
        if self.artifact["recommended_interval_method"] == "cqr":
            self.assertGreater(self.artifact["cqr_quantile_dex"], 0.0)


if __name__ == "__main__":
    unittest.main()
