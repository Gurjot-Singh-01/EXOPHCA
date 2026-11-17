"""Contract tests for the radius prediction module and its written artifact.

``exophca.ml.predict`` is the only reader of ``models/radius_model.joblib``. These
tests pin the contract the app depends on: one row per input, finite estimates
inside a correctly sized interval, a minimum-mass flag, the out-of-distribution
flag, the 1.6 R_earth straddle flag, and NaN plus a reason - never a guess - when
a feature is unusable.

They also check that the artifact written for the app is consistent with the
processed tables and that nothing under ``app/`` imports a model library, so the
Streamlit app can never load the pickled model itself.
"""

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exophca.ml.predict import (  # noqa: E402
    OUTPUT_COLUMNS,
    load_radius_artifact,
    predict_radius,
    version_warnings,
)

APP_DIR = ROOT / "app"
ARTIFACT_PATH = ROOT / "models" / "radius_model.joblib"
META_PATH = ROOT / "models" / "radius_model_meta.json"
PLANETS_PATH = ROOT / "data" / "processed" / "planets.csv"
ASSESSED_PATH = ROOT / "data" / "processed" / "planets_assessed.csv"
ESTIMATES_PATH = ROOT / "data" / "processed" / "radius_estimates.csv"

ASSESSED_SHA256_PREFIX = "6044cf78ae5d39df"

# The committed model artifact is a pickle: loading one executes code from it,
# so a replaced artifact is a supply-chain surface. Its digest is pinned here,
# next to the assessed table's, and checked in EstimatorArtifactTests.
ARTIFACT_SHA256 = "6b3d62e901a0598b03bb637b86b53ce003d7637857b975a0ade9e767849cc2de"

# A Planet with a measured mass, inside every training feature bound.
EARTH_LIKE = {
    "pl_name": "in-bounds",
    "pl_bmasse": 5.0,
    "pl_orbper": 10.0,
    "st_teff": 5000.0,
    "st_mass": 1.0,
    "st_rad": 1.0,
    "mass_provenance": "Mass",
}


def input_frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


class ArtifactLoadingTests(unittest.TestCase):
    """Loading reports version differences and never crashes."""

    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.warnings = load_radius_artifact(ARTIFACT_PATH)

    def test_the_artifact_loads_with_no_version_warning(self):
        self.assertIsNotNone(self.artifact)
        self.assertEqual([], self.warnings)

    def test_the_artifact_records_what_the_predictions_need(self):
        for key in (
            "point_model",
            "features",
            "recommended_interval_method",
            "split_conformal_half_width_dex",
            "cqr_quantile_dex",
            "feature_bounds",
            "library_versions",
        ):
            with self.subTest(key=key):
                self.assertIn(key, self.artifact)

    def test_a_recorded_version_that_differs_is_reported_not_raised(self):
        stored = dict(self.artifact["library_versions"])
        stored["lightgbm"] = "0.0.1"
        stored["scikit-learn"] = "0.0.1"

        warnings = version_warnings(stored)

        self.assertEqual(2, len(warnings), warnings)
        for library in ("lightgbm", "scikit-learn"):
            with self.subTest(library=library):
                self.assertTrue(
                    any(library in text and "0.0.1" in text for text in warnings),
                    warnings,
                )

    def test_a_missing_version_is_reported(self):
        warnings = version_warnings({"numpy": self.artifact["library_versions"]["numpy"]})

        self.assertTrue(any("no scikit-learn version" in text for text in warnings))

    def test_a_missing_artifact_returns_none_and_a_warning(self):
        artifact, warnings = load_radius_artifact(ROOT / "models" / "no-such-file.joblib")

        self.assertIsNone(artifact)
        self.assertTrue(warnings)

    def test_a_mismatched_artifact_still_loads(self):
        fake = {
            "point_model": None,
            "features": list(self.artifact["features"]),
            "library_versions": {"scikit-learn": "0.0.1", "lightgbm": "0.0.1"},
        }
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "fake.joblib"
            joblib.dump(fake, path)

            artifact, warnings = load_radius_artifact(path)

        self.assertEqual(fake, artifact)
        self.assertTrue(any("differs" in text for text in warnings))


class PredictContractTests(unittest.TestCase):
    """One row per input, with intervals and flags that match the artifact."""

    @classmethod
    def setUpClass(cls):
        cls.artifact, _ = load_radius_artifact(ARTIFACT_PATH)
        cls.planets = pd.read_csv(PLANETS_PATH)

    def predict_rows(self, *rows, artifact=None):
        return predict_radius(input_frame(*rows), artifact or self.artifact)

    def planet_row(self, name: str) -> dict:
        row = self.planets[self.planets["pl_name"] == name].iloc[0]
        return {
            "pl_name": name,
            "pl_bmasse": row["pl_bmasse"],
            "pl_orbper": row["pl_orbper"],
            "st_teff": row["st_teff"],
            "st_mass": row["st_mass"],
            "st_rad": row["st_rad"],
            "mass_provenance": row["mass_provenance"],
        }

    def test_the_output_has_exactly_the_documented_columns_in_order(self):
        result = self.predict_rows(EARTH_LIKE)

        self.assertEqual(OUTPUT_COLUMNS, list(result.columns))
        self.assertEqual(1, len(result))

    def test_one_row_is_returned_per_input_in_input_order(self):
        names = ["a", "b", "c"]
        rows = [dict(EARTH_LIKE, pl_name=name, pl_bmasse=mass) for name, mass in zip(names, (1.0, 5.0, 50.0))]

        result = self.predict_rows(*rows)

        self.assertEqual(names, list(result["pl_name"]))

    def test_every_estimate_is_finite_and_ordered(self):
        rows = [self.planet_row(name) for name in ("Proxima Cen b", "GJ 1214 b", "51 Peg b")]

        result = self.predict_rows(*rows)

        self.assertTrue(np.isfinite(result[["radius_estimate", "radius_lower", "radius_upper"]].to_numpy()).all())
        self.assertTrue((result["radius_lower"] < result["radius_estimate"]).all())
        self.assertTrue((result["radius_estimate"] < result["radius_upper"]).all())
        self.assertEqual(["", "", ""], result["reason"].tolist())

    def test_the_split_conformal_half_width_is_the_artifact_value(self):
        half_width = self.artifact["split_conformal_half_width_dex"]
        self.assertEqual("split_conformal", self.artifact["recommended_interval_method"])

        result = self.predict_rows(self.planet_row("Proxima Cen b"))

        log_estimate = np.log10(result.loc[0, "radius_estimate"])
        self.assertAlmostEqual(half_width, log_estimate - np.log10(result.loc[0, "radius_lower"]), places=9)
        self.assertAlmostEqual(half_width, np.log10(result.loc[0, "radius_upper"]) - log_estimate, places=9)

    def test_the_cqr_branch_uses_its_own_quantile(self):
        artifact = dict(self.artifact, recommended_interval_method="cqr")
        quantile = artifact["cqr_quantile_dex"]
        row = self.planet_row("Proxima Cen b")

        result = self.predict_rows(row, artifact=artifact)

        features = np.array(
            [[
                np.log10(row["pl_bmasse"]),
                np.log10(row["pl_orbper"]),
                row["st_teff"],
                row["st_mass"],
                row["st_rad"],
            ]]
        )
        expected_lower = 10 ** (artifact["cqr_lower_model"].predict(features)[0] - quantile)
        expected_upper = 10 ** (artifact["cqr_upper_model"].predict(features)[0] + quantile)
        self.assertAlmostEqual(expected_lower, result.loc[0, "radius_lower"], places=9)
        self.assertAlmostEqual(expected_upper, result.loc[0, "radius_upper"], places=9)

    def test_a_missing_feature_gives_nan_and_a_reason(self):
        row = dict(self.planet_row("Proxima Cen b"), st_rad=np.nan)

        result = self.predict_rows(row)

        self.assertTrue(np.isnan(result.loc[0, "radius_estimate"]))
        self.assertTrue(np.isnan(result.loc[0, "radius_lower"]))
        self.assertTrue(np.isnan(result.loc[0, "radius_upper"]))
        self.assertIn("st_rad is missing", result.loc[0, "reason"])
        self.assertFalse(bool(result.loc[0, "ood_flag"]))
        self.assertFalse(bool(result.loc[0, "interval_straddles_1p6"]))

    def test_a_non_positive_feature_gives_nan_and_a_reason(self):
        row = dict(self.planet_row("Proxima Cen b"), pl_bmasse=0.0)

        result = self.predict_rows(row)

        self.assertTrue(np.isnan(result.loc[0, "radius_estimate"]))
        self.assertIn("pl_bmasse is not positive", result.loc[0, "reason"])

    def test_a_table_missing_a_feature_column_is_refused(self):
        frame = input_frame(dict(EARTH_LIKE)).drop(columns=["st_mass"])

        with self.assertRaises(ValueError):
            predict_radius(frame, self.artifact)

    def test_mass_is_minimum_is_true_for_both_minimum_mass_labels(self):
        rows = [
            dict(EARTH_LIKE, pl_name="msini", mass_provenance="Msini"),
            dict(EARTH_LIKE, pl_name="sin-i", mass_provenance="Msin(i)/sin(i)"),
            dict(EARTH_LIKE, pl_name="measured", mass_provenance="Mass"),
            dict(EARTH_LIKE, pl_name="unknown", mass_provenance="Unknown"),
        ]

        result = self.predict_rows(*rows)

        self.assertEqual(
            [True, True, False, False], result["mass_is_minimum"].tolist()
        )

    def test_the_ood_flag_follows_the_stored_feature_bounds(self):
        bounds = self.artifact["feature_bounds"]
        below = 0.5 * bounds["st_teff"][0]
        rows = [
            dict(EARTH_LIKE, pl_name="in-bounds"),
            dict(EARTH_LIKE, pl_name="cold-host", st_teff=below),
        ]

        result = self.predict_rows(*rows)

        self.assertFalse(bool(result.loc[0, "ood_flag"]))
        self.assertTrue(bool(result.loc[1, "ood_flag"]))

    def test_the_straddle_flag_matches_the_interval(self):
        result = self.predict_rows(
            self.planet_row("Proxima Cen b"), self.planet_row("51 Peg b")
        )

        for index in range(len(result)):
            lower = result.loc[index, "radius_lower"]
            upper = result.loc[index, "radius_upper"]
            with self.subTest(planet=result.loc[index, "pl_name"]):
                self.assertEqual(
                    bool(lower < 1.6 < upper),
                    bool(result.loc[index, "interval_straddles_1p6"]),
                )
        self.assertTrue(bool(result.loc[0, "interval_straddles_1p6"]))


class EstimatorArtifactTests(unittest.TestCase):
    """The artifact the app reads is consistent with the processed tables."""

    @classmethod
    def setUpClass(cls):
        cls.estimates = pd.read_csv(ESTIMATES_PATH)
        cls.planets = pd.read_csv(PLANETS_PATH)

    def test_the_written_table_has_the_documented_columns(self):
        self.assertEqual(OUTPUT_COLUMNS, list(self.estimates.columns))

    def test_every_selected_planet_has_exactly_one_row(self):
        selected = self.planets[
            self.planets["radius_provenance"].isin(["modelled", "derived_unclear"])
        ]
        self.assertEqual(sorted(selected["pl_name"]), sorted(self.estimates["pl_name"]))
        self.assertFalse(self.estimates["pl_name"].duplicated().any())

    def test_measured_radius_planets_are_not_in_the_artifact(self):
        measured = set(
            self.planets.loc[self.planets["radius_provenance"] == "measured", "pl_name"]
        )
        self.assertFalse(measured & set(self.estimates["pl_name"]))

    def test_predictions_are_finite_inside_their_interval(self):
        predicted = self.estimates[self.estimates["radius_estimate"].notna()]

        self.assertGreater(len(predicted), 1000)
        self.assertTrue((predicted["radius_lower"] < predicted["radius_estimate"]).all())
        self.assertTrue((predicted["radius_estimate"] < predicted["radius_upper"]).all())
        self.assertTrue((predicted["radius_estimate"] > 0).all())

    def test_skipped_rows_carry_a_reason_and_no_estimate(self):
        skipped = self.estimates[self.estimates["radius_estimate"].isna()]

        self.assertGreater(len(skipped), 0)
        self.assertTrue(skipped["reason"].str.startswith("no estimate:").all())

    def test_recomputing_a_planet_reproduces_its_written_row(self):
        artifact, _ = load_radius_artifact(ARTIFACT_PATH)
        row = self.planets[self.planets["pl_name"] == "Proxima Cen b"]
        written = self.estimates[self.estimates["pl_name"] == "Proxima Cen b"].iloc[0]

        result = predict_radius(row, artifact)

        self.assertAlmostEqual(written["radius_estimate"], result.loc[0, "radius_estimate"], places=5)
        self.assertAlmostEqual(written["radius_lower"], result.loc[0, "radius_lower"], places=5)
        self.assertAlmostEqual(written["radius_upper"], result.loc[0, "radius_upper"], places=5)

    def test_the_assessed_table_was_not_touched(self):
        digest = hashlib.sha256(ASSESSED_PATH.read_bytes()).hexdigest()

        self.assertTrue(digest.startswith(ASSESSED_SHA256_PREFIX), digest)

    def test_the_committed_artifact_still_matches_its_sha256(self):
        digest = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()

        self.assertEqual(ARTIFACT_SHA256, digest)

    def test_the_meta_report_still_declares_the_same_interval(self):
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        artifact, _ = load_radius_artifact(ARTIFACT_PATH)

        self.assertEqual(meta["intervals"]["final_recommended"], artifact["recommended_interval_method"])
        self.assertAlmostEqual(
            meta["intervals"]["split_conformal_half_width_dex"],
            artifact["split_conformal_half_width_dex"],
            places=12,
        )


class AppIsolationTests(unittest.TestCase):
    """The app must never load the pickled model."""

    FORBIDDEN_IMPORT = re.compile(
        r"^\s*(?:import|from)\s+(joblib|sklearn|lightgbm)\b", re.MULTILINE
    )

    def app_sources(self):
        return sorted(path for path in APP_DIR.rglob("*.py") if "__pycache__" not in path.parts)

    def test_no_app_file_imports_a_model_library(self):
        sources = self.app_sources()
        self.assertGreater(len(sources), 5)

        for path in sources:
            with self.subTest(file=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(self.FORBIDDEN_IMPORT.search(text), path)
                self.assertNotIn("joblib.load", text, path)


if __name__ == "__main__":
    unittest.main()
