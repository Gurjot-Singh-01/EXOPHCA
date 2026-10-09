import unittest
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from app.scoring import (
    MODEL_FEATURES,
    TRAINING_RANGES,
    calculate_custom_adjustment,
    calculate_esi,
    predict_compatibility,
    training_range_warnings,
    validate_nonnegative_finite,
    validate_positive_finite,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rf_compat_model.pkl"
TOI_PATH = ROOT / "data" / "toi_for_app.csv"


def small_model(feature_names=MODEL_FEATURES):
    values = pd.DataFrame(
        [[1.0, 5778.0, 10.0], [10.0, 3500.0, 100.0]],
        columns=feature_names,
    )
    return RandomForestRegressor(n_estimators=2, random_state=42).fit(
        values, [0.5, 0.8]
    )


class ScoringFormulaTests(unittest.TestCase):
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
        self.assertEqual(0.0, validate_nonnegative_finite("Insolation flux", 0.0))

    def test_base_esi_is_separate_from_custom_adjustments(self):
        base = calculate_esi(1.2, 1.0)
        adjustment = calculate_custom_adjustment(1.2, "K")
        self.assertEqual(0.1, adjustment)
        adjusted = base + adjustment
        self.assertGreater(adjusted, base)
        self.assertLess(adjusted, 1.1)

    def test_custom_adjustment_radius_boundaries(self):
        self.assertEqual(0.0, calculate_custom_adjustment(1.0, "G (Sun-like)"))
        self.assertEqual(0.05, calculate_custom_adjustment(1.0, "K"))
        self.assertEqual(0.05, calculate_custom_adjustment(1.000001, "G (Sun-like)"))
        self.assertEqual(0.05, calculate_custom_adjustment(1.5, "G (Sun-like)"))
        self.assertEqual(0.0, calculate_custom_adjustment(1.500001, "G (Sun-like)"))
        self.assertEqual(0.1, calculate_custom_adjustment(1.5, "K"))

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
            calculate_custom_adjustment(1.0, "unknown")
        with self.assertRaises(ValueError):
            calculate_custom_adjustment(1.0, None)


class InferenceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)
        cls.toi = pd.read_csv(TOI_PATH)

    def test_saved_artifact_has_expected_raw_feature_contract(self):
        self.assertEqual(tuple(self.model.feature_names_in_), MODEL_FEATURES)
        self.assertEqual(200, self.model.n_estimators)
        self.assertEqual(42, self.model.random_state)
        scores = predict_compatibility(
            self.model, self.toi.loc[:, list(MODEL_FEATURES)].head(1)
        )
        self.assertEqual((1,), scores.shape)
        self.assertTrue(np.isfinite(scores).all())

    def test_mapping_uses_expected_feature_order(self):
        scores = predict_compatibility(
            self.model,
            {"sy_dist": 227.337, "st_teff": 3736.0, "pl_orbper": 60.8344877},
        )
        self.assertEqual(1, scores.size)
        self.assertAlmostEqual(0.8817420379420032, scores[0], places=12)

    def test_frame_with_wrong_order_or_features_is_rejected(self):
        valid = self.toi.loc[:, list(MODEL_FEATURES)].head(1)
        with self.assertRaisesRegex(ValueError, "columns and order"):
            predict_compatibility(self.model, valid.loc[:, list(reversed(MODEL_FEATURES))])
        with self.assertRaisesRegex(ValueError, "exactly these features"):
            predict_compatibility(self.model, {"pl_orbper": 2.0, "st_teff": 5000.0})

    def test_model_with_wrong_feature_order_is_rejected(self):
        wrong_order = small_model(tuple(reversed(MODEL_FEATURES)))
        with self.assertRaisesRegex(ValueError, "Model feature names/order"):
            predict_compatibility(
                wrong_order,
                {"pl_orbper": 2.0, "st_teff": 5000.0, "sy_dist": 10.0},
            )

    def test_invalid_model_features_are_rejected(self):
        for invalid in (None, np.nan, np.inf, -np.inf, 0.0, -1.0):
            for feature in MODEL_FEATURES:
                values = {"pl_orbper": 2.0, "st_teff": 5000.0, "sy_dist": 10.0}
                values[feature] = invalid
                with self.subTest(feature=feature, value=invalid):
                    with self.assertRaises(ValueError):
                        predict_compatibility(self.model, values)

    def test_training_range_endpoints_are_in_range_and_outside_values_warn(self):
        in_range = {
            name: limits[0] if limits[0] > 0 else 0.000001
            for name, limits in TRAINING_RANGES.items()
        }
        self.assertEqual([], training_range_warnings(in_range))

        in_range.update({name: limits[1] for name, limits in TRAINING_RANGES.items()})
        self.assertEqual([], training_range_warnings(in_range))

        outside = {
            name: limits[1] + 1.0
            for name, limits in TRAINING_RANGES.items()
        }
        self.assertEqual(len(TRAINING_RANGES), len(training_range_warnings(outside)))

    def test_candidate_comparison_scores_are_recomputed_by_saved_artifact(self):
        recalculated = predict_compatibility(
            self.model,
            self.toi.loc[:, list(MODEL_FEATURES)],
        )
        np.testing.assert_allclose(
            recalculated,
            self.toi["predicted_score"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.argsort(recalculated),
            np.argsort(self.toi["predicted_score"].to_numpy()),
        )

    def test_nonfinite_or_out_of_range_predictions_are_rejected(self):
        model = small_model()
        for invalid_score in (np.nan, np.inf, -0.01, 1.1):
            model.predict = lambda frame, score=invalid_score: np.full(len(frame), score)
            with self.subTest(score=invalid_score):
                with self.assertRaises(ValueError):
                    predict_compatibility(
                        model,
                        {"pl_orbper": 2.0, "st_teff": 5000.0, "sy_dist": 10.0},
                    )

    def test_multioutput_prediction_is_rejected(self):
        model = small_model()
        model.predict = lambda frame: np.zeros((len(frame), 2))
        with self.assertRaisesRegex(ValueError, "one score per input row"):
            predict_compatibility(
                model,
                {"pl_orbper": 2.0, "st_teff": 5000.0, "sy_dist": 10.0},
            )

    def test_preprocessing_pipeline_is_not_accepted(self):
        pipeline = Pipeline([("forest", small_model())])
        with self.assertRaisesRegex(ValueError, "unwrapped RandomForestRegressor"):
            predict_compatibility(
                pipeline,
                {"pl_orbper": 2.0, "st_teff": 5000.0, "sy_dist": 10.0},
            )


if __name__ == "__main__":
    unittest.main()
