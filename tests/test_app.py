"""Tests for the Streamlit app entry point and the Calculator page.

The app is now multipage (``st.navigation`` in app/EXOPHCA.py). These tests
exercise the same behaviour as before: the shell renders the default page and
the Calculator page computes the exact score and habitable-zone class with the
same messages for empty, in-range and out-of-range stellar temperatures.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app" / "EXOPHCA.py"
CALCULATOR_FILE = ROOT / "app" / "views" / "calculator.py"
TESS_FILE = ROOT / "app" / "views" / "tess_candidates.py"

APP_DIR = ROOT / "app"
for _path in (str(ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import labelling  # noqa: E402
import presets  # noqa: E402
import query  # noqa: E402
from exophca.science.assessment import DISCLAIMER  # noqa: E402
from exophca.science.esi import calculate_esi  # noqa: E402
from exophca.science.habitable_zone import classify_hz, hz_limits  # noqa: E402


def run_app():
    return AppTest.from_file(str(APP_FILE), default_timeout=180).run()


def run_calculator_page():
    return AppTest.from_file(str(CALCULATOR_FILE), default_timeout=180).run()


def choose_group(app, group):
    """Switch the Calculator's preset group and rerun the page."""
    app.radio[0].set_value(group)
    app.run()
    return app


def choose_preset(app, name, group=None):
    """Pick a preset (optionally after switching group) and rerun the page."""
    if group is not None:
        choose_group(app, group)
    app.selectbox[0].select(name)
    app.run()
    return app


def run_tess_page():
    return AppTest.from_file(str(TESS_FILE), default_timeout=180).run()


def run_calculator(teff=None):
    """Run the Calculator page, optionally set the temperature, and press Calculate."""
    app = run_calculator_page()
    if teff is not None:
        app.number_input[2].set_value(teff)
    app.button[0].click()
    app.run()
    return app


def info_text(app):
    return " | ".join(item.value for item in app.info)


def metric_values(app):
    return {metric.label: metric.value for metric in app.metric}


def rendered_text(app):
    """Every markdown, caption and info string the page rendered, joined."""
    parts = []
    for collection in ("markdown", "caption", "info", "warning", "text"):
        for item in getattr(app, collection, []):
            parts.append(str(getattr(item, "value", "")))
    return " | ".join(parts)


class AppShellTests(unittest.TestCase):
    """The entry point renders the default page through st.navigation."""

    def test_entry_page_renders_overview_through_navigation(self):
        app = run_app()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual("Overview", app.title[0].value)
        self.assertTrue(
            any("Datasets" in str(header.value) for header in app.subheader),
            "Overview should render its Datasets section",
        )


class CalculatorTests(unittest.TestCase):
    """The exact calculator, unchanged, on its own page."""

    def test_earth_matches_its_baseline(self):
        app = run_calculator_page()
        app.selectbox[0].select("Earth")
        app.button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        esi = next(
            metric
            for metric in app.metric
            if metric.label == "ESI-style radius-flux similarity"
        )
        self.assertEqual("1.000", esi.value)

    def test_zero_insolation_is_scored_without_a_custom_cap(self):
        app = run_calculator_page()
        app.number_input[1].set_value(0.0)
        app.button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        esi = next(
            metric
            for metric in app.metric
            if metric.label == "ESI-style radius-flux similarity"
        )
        self.assertEqual("0.000", esi.value)

    def test_k_class_preset_flags_the_host(self):
        app = run_calculator_page()
        choose_preset(app, "TRAPPIST-1 e", group="Archive planets")
        app.button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        text = " ".join(item.value for item in app.text)
        self.assertIn("Host is K-class: No", text)


class CalculatorPresetTests(unittest.TestCase):
    """Both preset groups, with every value computed from the reference tables."""

    NO_PRESET = "-- Select a preset --"

    def setUp(self):
        self.assessed = pd.read_csv(ROOT / "data" / "processed" / "planets_assessed.csv")
        self.solar = presets.solar_system_presets()
        self.archive = presets.archive_presets(self.assessed)

    def test_the_solar_system_group_offers_all_eight_planets(self):
        app = run_calculator_page()

        self.assertEqual("Solar System", app.radio[0].value)
        self.assertEqual(
            [self.NO_PRESET]
            + ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
            list(app.selectbox[0].options),
        )

    def test_the_archive_group_offers_every_scorable_planet(self):
        app = choose_group(run_calculator_page(), "Archive planets")
        options = list(app.selectbox[0].options)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(self.NO_PRESET, options[0])
        self.assertEqual(len(self.assessed) + 1, len(options))
        self.assertEqual(
            ["Proxima Cen b", "TRAPPIST-1 e", "Kepler-452 b", "TOI-700 d"],
            options[1:5],
        )
        self.assertEqual(set(self.assessed["pl_name"]), set(options[1:]))
        # Earth is a Solar System preset, not an archive row.
        self.assertNotIn("Earth", options)
        self.assertEqual(len(self.archive), len(self.assessed))

    def test_solar_system_values_are_computed_from_the_reference_file(self):
        table = pd.read_csv(ROOT / "data" / "reference" / "solar_system.csv")
        earth = table[table["name"] == "Earth"].iloc[0]

        self.assertEqual(8, len(self.solar))
        for row in table.itertuples(index=False):
            with self.subTest(planet=row.name):
                preset = self.solar[row.name]
                self.assertAlmostEqual(
                    row.mean_radius_km / earth["mean_radius_km"], preset["pl_rade"], places=12
                )
                self.assertAlmostEqual(
                    (earth["semimajor_axis_km"] / row.semimajor_axis_km) ** 2,
                    preset["pl_insol"],
                    places=12,
                )

        self.assertEqual(1.0, self.solar["Earth"]["pl_rade"])
        self.assertEqual(1.0, self.solar["Earth"]["pl_insol"])
        self.assertEqual(10.97, round(self.solar["Jupiter"]["pl_rade"], 2))
        self.assertEqual(0.0369, round(self.solar["Jupiter"]["pl_insol"], 4))
        self.assertEqual(6.67, round(self.solar["Mercury"]["pl_insol"], 2))

    def test_solar_system_presets_use_the_sun_and_a_g_host(self):
        for name, preset in self.solar.items():
            with self.subTest(planet=name):
                self.assertEqual(presets.SOLAR_TEFF_K, preset["st_teff"])
                self.assertEqual(presets.SOLAR_STAR_CLASS, preset["star_class"])
                self.assertIn(presets.SOLAR_SYSTEM_LABEL, preset["source"])

    def test_archive_preset_values_equal_the_processed_row(self):
        for name in presets.ARCHIVE_QUICK_PICKS:
            row = self.assessed[self.assessed["pl_name"] == name].iloc[0]
            with self.subTest(preset=name):
                self.assertEqual(float(row["pl_rade"]), self.archive[name]["pl_rade"])
                self.assertEqual(float(row["pl_insol"]), self.archive[name]["pl_insol"])
                self.assertEqual(float(row["st_teff"]), self.archive[name]["st_teff"])
                self.assertEqual(row["star_class"], self.archive[name]["star_class"])
                self.assertEqual("data/processed/planets_assessed.csv", self.archive[name]["source"])

    def test_every_preset_scores_without_exception(self):
        allowed = {"conservative", "optimistic_only", "too_hot", "too_cold", "undetermined"}
        merged = presets.all_presets(self.assessed)

        self.assertEqual(len(self.solar) + len(self.assessed), len(merged))
        for name, preset in merged.items():
            with self.subTest(preset=name):
                score = calculate_esi(preset["pl_rade"], preset["pl_insol"])
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                teff = preset["st_teff"]
                if teff is None:
                    hz_class = "undetermined"
                elif hz_limits(teff)["in_valid_range"]:
                    hz_class = classify_hz(preset["pl_insol"], teff)
                else:
                    hz_class = "undetermined"
                self.assertIn(hz_class, allowed)

    def test_the_page_scores_every_solar_system_preset(self):
        app = run_calculator_page()
        for name in presets.solar_system_presets():
            with self.subTest(planet=name):
                choose_preset(app, name)
                self.assertEqual([], [exception.message for exception in app.exception])
                self.assertIn("ESI-style radius-flux similarity", metric_values(app))
                # The habitable-zone class is a sentence, so it is wrapping text
                # rather than a metric value that Streamlit would clip.
                self.assertIn("Habitable-zone class", rendered_text(app))

    def test_the_page_scores_the_archive_quick_picks(self):
        app = run_calculator_page()
        choose_group(app, "Archive planets")
        for name in presets.ARCHIVE_QUICK_PICKS:
            with self.subTest(planet=name):
                choose_preset(app, name)
                self.assertEqual([], [exception.message for exception in app.exception])
                self.assertIn("ESI-style radius-flux similarity", metric_values(app))

    def test_selecting_a_preset_fills_the_inputs_and_scores_them(self):
        app = choose_preset(
            run_calculator_page(), "TOI-700 d", group="Archive planets"
        )
        values = {item.label: item.value for item in app.number_input}

        self.assertAlmostEqual(1.073, values["Planet Radius (Earth radii)"])
        self.assertAlmostEqual(0.85, values["Insolation Flux (Earth flux = 1.0)"])
        self.assertAlmostEqual(3459.0, values["Stellar Effective Temperature (K, optional)"])
        # Choosing a preset already scores it; no button press needed.
        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertIn("**conservative habitable zone**", rendered_text(app))
        self.assertNotIn("Habitable-zone class", metric_values(app))

    def test_the_page_states_that_radii_are_mean_radii(self):
        text = " ".join(item.value for item in run_calculator_page().caption)

        self.assertIn("mean (volumetric) radii", text)
        self.assertIn("Jupiter is 10.97 R\u2295", text)
        self.assertIn("equatorial 11.2", text)


class CalculatorMetricValueLengthTests(unittest.TestCase):
    """No Calculator metric value is long enough to be clipped or wrapped.

    A metric value is drawn in the metric font and hidden behind an ellipsis
    when it does not fit its column, so a sentence belongs in a wrapping
    component instead. The archive planets are chosen from the data - two for
    every habitable-zone class - rather than named by hand.
    """

    MAX_VALUE_LENGTH = 18
    ASSESSED = ROOT / "data" / "processed" / "planets_assessed.csv"
    HZ_LABELS = {
        "conservative": "conservative habitable zone",
        "optimistic_only": "optimistic habitable zone only",
        "too_hot": "too hot (inside the recent-Venus limit)",
        "too_cold": "too cold (outside the early-Mars limit)",
        "undetermined": "undetermined",
    }

    def setUp(self):
        self.frame = pd.read_csv(self.ASSESSED)

    def planets_covering_every_hz_class(self):
        """``[(name, hz_class)]``: two archive planets per class, from the data."""
        usable = self.frame.dropna(subset=["pl_rade", "pl_insol", "st_teff"]).copy()
        usable["recomputed"] = [
            classify_hz(flux, teff)
            for flux, teff in zip(usable["pl_insol"], usable["st_teff"])
        ]
        usable = usable[usable["recomputed"] == usable["hz_class"]].sort_values("pl_name")

        chosen = []
        for hz_class in self.HZ_LABELS:
            rows = usable[usable["hz_class"] == hz_class]
            self.assertGreaterEqual(len(rows), 2, f"no archive rows for {hz_class}")
            chosen.extend((row["pl_name"], hz_class) for _, row in rows.head(2).iterrows())
        return chosen

    def assert_every_metric_value_is_short(self, app, context):
        self.assertGreater(len(app.metric), 0, f"{context}: no metric was rendered")
        for metric in app.metric:
            with self.subTest(context=context, metric=metric.label):
                value = str(metric.value)
                self.assertLessEqual(
                    len(value),
                    self.MAX_VALUE_LENGTH,
                    f"{context}: {metric.label!r} value {value!r} is {len(value)} "
                    f"characters, longer than {self.MAX_VALUE_LENGTH}",
                )

    def test_solar_system_presets_keep_every_metric_value_short(self):
        app = run_calculator_page()
        for name in presets.solar_system_presets():
            with self.subTest(planet=name):
                choose_preset(app, name)
                self.assertEqual([], [exception.message for exception in app.exception])
                self.assert_every_metric_value_is_short(app, name)

    def test_the_chosen_archive_planets_cover_every_habitable_zone_class(self):
        chosen = self.planets_covering_every_hz_class()

        self.assertEqual(10, len(chosen))
        self.assertEqual(sorted(self.HZ_LABELS), sorted({hz for _, hz in chosen}))

    def test_archive_presets_keep_every_metric_value_short(self):
        app = run_calculator_page()
        choose_group(app, "Archive planets")
        for name, hz_class in self.planets_covering_every_hz_class():
            with self.subTest(planet=name, hz_class=hz_class):
                choose_preset(app, name)
                self.assertEqual([], [exception.message for exception in app.exception])
                self.assert_every_metric_value_is_short(app, name)
                text = rendered_text(app)
                if hz_class == "undetermined":
                    self.assertIn("Habitable zone undetermined", text)
                else:
                    self.assertIn(f"**{self.HZ_LABELS[hz_class]}**", text)
                self.assertNotIn("Habitable-zone class", metric_values(app))

    def test_the_long_habitable_zone_class_is_wrapping_text_not_a_metric(self):
        app = run_calculator_page()
        choose_preset(app, "Jupiter")

        self.assertIn("**too cold (outside the early-Mars limit)**", rendered_text(app))
        self.assertNotIn("Habitable-zone class", metric_values(app))
        self.assertEqual(["0.238"], [m.value for m in app.metric])


class TessCandidateTests(unittest.TestCase):
    """The TESS candidates view: filters, labelled probabilities, export."""

    def setUp(self):
        self.app = run_tess_page()
        self.frame = pd.read_csv(ROOT / "data" / "processed" / "toi_assessed.csv")
        self.scorable = self.frame[self.frame["pl_rade"].notna() & (self.frame["pl_rade"] > 0)]
        self.table = self.app.dataframe[0].value

    def test_tess_candidates_page_exists_and_renders(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])
        self.assertEqual(1, len(self.app.dataframe))

    def test_every_candidate_is_shown_by_default(self):
        self.assertEqual(len(self.scorable), len(self.table))

    def test_every_row_carries_the_unconfirmed_status(self):
        self.assertEqual({"unconfirmed candidate"}, set(self.table["status"]))

    def test_every_probability_is_labelled_with_its_status(self):
        for name in labelling.PROBABILITY_DEPENDENCIES:
            with self.subTest(probability=name):
                self.assertIn(name, self.table.columns)
                self.assertTrue(self.table[name].astype(str).str.contains("uncertainty_status").all())

    def test_flux_probabilities_are_conditional_for_every_tess_candidate(self):
        self.assertTrue(
            set(self.table["uncertainty_status"]) <= {"radius_only", "none"}
        )
        for name in ("p_conservative_hz", "p_any_hz"):
            with self.subTest(probability=name):
                self.assertTrue(
                    self.table[name].astype(str).str.contains(
                        labelling.CONDITIONAL_TEXT, regex=False
                    ).all()
                )

    def test_page_states_that_tess_has_no_flux_uncertainty(self):
        text = " ".join(
            [item.value for item in self.app.info]
            + [item.value for item in self.app.caption]
            + [item.value for item in self.app.markdown]
        )

        self.assertIn("No TESS candidate has a published flux uncertainty", text)
        self.assertIn(labelling.CONDITIONAL_TEXT, text)
        self.assertIn(DISCLAIMER, text)

    def test_tier_filter_narrows_the_table(self):
        app = run_tess_page()
        app.multiselect[0].select("A")
        app.run()
        table = app.dataframe[0].value

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual({"A"}, set(table["tier"]))
        self.assertEqual(len(self.scorable[self.scorable["tier"] == "A"]), len(table))

    def test_hz_and_evidence_filters_narrow_the_table(self):
        app = run_tess_page()
        app.multiselect[1].select("conservative")
        app.multiselect[2].select("medium")
        app.run()
        table = app.dataframe[0].value

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual({"conservative"}, set(table["hz_class"]))
        self.assertEqual({"medium"}, set(table["evidence_quality"]))
        expected = self.scorable[
            (self.scorable["hz_class"] == "conservative")
            & (self.scorable["evidence_quality"] == "medium")
        ]
        self.assertEqual(len(expected), len(table))

    def test_radius_and_teff_range_filters_bound_the_rows(self):
        app = run_tess_page()
        app.slider[0].set_value((0.6, 1.5))
        app.slider[1].set_value((3000.0, 6000.0))
        app.run()
        table = app.dataframe[0].value

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual((0.6, 1.5), app.slider[0].value)
        self.assertEqual((3000.0, 6000.0), app.slider[1].value)
        self.assertTrue(table["pl_rade"].between(0.6, 1.5).all())
        self.assertTrue(table["st_teff"].between(3000.0, 6000.0).all())
        expected = self.scorable[
            self.scorable["pl_rade"].between(0.6, 1.5)
            & self.scorable["st_teff"].between(3000.0, 6000.0)
        ]
        self.assertEqual(len(expected), len(table))
        self.assertLess(len(table), len(self.scorable))

    def test_boundary_filter_has_no_rows_to_show(self):
        app = run_tess_page()
        app.selectbox[0].select("Flagged only")
        app.run()
        table = app.dataframe[0].value

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(0, len(table))
        self.assertEqual(0, int(self.scorable["boundary_sensitive"].fillna(False).sum()))

    def test_export_carries_the_status_the_disclaimer_and_the_date(self):
        export = query.build_toi_export_frame(self.scorable, "2026-07-11")
        name = query.export_filename("2026-07-11", prefix="exophca_tess_candidates")

        self.assertEqual(len(self.scorable), len(export))
        self.assertIn("disclaimer", export.columns)
        self.assertEqual("disclaimer", export.columns[-1])
        self.assertTrue((export["disclaimer"] == DISCLAIMER).all())
        self.assertEqual({"unconfirmed candidate"}, set(export["status"]))
        self.assertEqual({"2026-07-11"}, set(export["data_retrieval_date"]))
        self.assertIn("2026-07-11", name)
        self.assertEqual(1, len(self.app.get("download_button")))

    def test_tess_candidates_view_shows_radius_column(self):
        toi_path = ROOT / "data" / "processed" / "toi_assessed.csv"
        df = pd.read_csv(toi_path)
        self.assertIn("pl_rade", df.columns)
        valid = df[df["pl_rade"].notna() & (df["pl_rade"] > 0)]
        self.assertGreater(len(valid), 0)

    def test_tess_candidates_view_note_present(self):
        app = run_tess_page()

        all_text = ""
        for item in app.markdown:
            all_text += str(item.value).lower() + "\n"
        for item in app.info:
            all_text += str(item.value).lower() + "\n"
        for item in app.caption:
            all_text += str(item.value).lower() + "\n"
        self.assertIn("unconfirmed candidates", all_text)
        self.assertIn("flux uncertainty", all_text)

    def test_tess_candidates_file_reads_successfully(self):
        toi_path = ROOT / "data" / "processed" / "toi_assessed.csv"
        df = pd.read_csv(toi_path)
        self.assertGreater(len(df), 0)


class StellarTemperatureTests(unittest.TestCase):
    """Habitable-zone classification driven by the stellar temperature input."""

    def test_empty_stellar_temperature_reports_undetermined_zone(self):
        app = run_calculator()

        self.assertEqual([], [exception.message for exception in app.exception])
        text = info_text(app)
        self.assertIn("Habitable zone undetermined", text)
        self.assertIn(
            "no stellar effective temperature was provided, so the Kopparapu et al. "
            "flux limits cannot be evaluated.",
            text,
        )
        self.assertNotIn("Habitable-zone class", rendered_text(app))

    def test_sunlike_temperature_gives_conservative_zone(self):
        app = run_calculator_page()
        app.number_input[0].set_value(1.0)
        app.number_input[1].set_value(1.0)
        app.number_input[2].set_value(5780.0)
        app.button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertIn("**conservative habitable zone**", rendered_text(app))

    def test_out_of_range_temperature_is_not_extrapolated(self):
        app = run_calculator(8000.0)

        self.assertEqual([], [exception.message for exception in app.exception])
        text = info_text(app)
        self.assertIn("Habitable zone undetermined", text)
        self.assertIn(
            "the Kopparapu et al. (2013, 2014) limits are fitted over 2600-7200 K",
            text,
        )
        self.assertIn("not extrapolated", text)
        self.assertNotIn("Habitable-zone class", rendered_text(app))


if __name__ == "__main__":
    unittest.main()
