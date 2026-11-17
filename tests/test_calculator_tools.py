"""Tests for the Calculator's drawings, its presets and the batch CSV scorer.

The drawings and the batch rules are pure functions, so the scale ratios, the
shrink note and the row-level errors are checked directly on the rendered HTML
and on the result object the page uses.
"""

import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import batch  # noqa: E402
import calculator_widgets as widgets  # noqa: E402
import presets  # noqa: E402
from exophca.science.assessment import DISCLAIMER  # noqa: E402
from exophca.science.esi import calculate_esi  # noqa: E402

EXAMPLE_CSV = ROOT / "tests" / "data" / "batch_example.csv"

DIAMETER_PATTERN = r'class="{}"[^>]*width:\s*([\d.]+)px'


def drawn_diameter(html: str, css_class: str) -> float:
    match = re.search(DIAMETER_PATTERN.format(css_class), html)
    if match is None:
        raise AssertionError(f"no {css_class} circle in the rendered HTML")
    return float(match.group(1))


def small_csv(rows: int, *, radius: float = 1.0, flux: float = 1.0) -> str:
    body = "\n".join(
        f"Planet {index},{radius},{flux}" for index in range(1, rows + 1)
    )
    return "name,pl_rade,pl_insol\n" + body + "\n"


class SizeComparisonScaleTests(unittest.TestCase):
    """The Size Comparison is drawn to scale: diameter is linear in radius."""

    def test_diameter_ratio_equals_the_radius_ratio(self):
        for radius in (1.0, 2.0, 11.21):
            with self.subTest(radius=radius):
                result = widgets.size_comparison(radius, 1.0)
                earth = drawn_diameter(result["html"], "earth")
                planet = drawn_diameter(result["html"], "planet")

                # Painted sizes, rounded to a thousandth of a pixel.
                self.assertAlmostEqual(radius, planet / earth, places=3)
                # ...and the same check in pixels: within one pixel of rounding.
                self.assertLess(abs(planet - earth * radius), 1.0)

    def test_a_planet_that_fits_keeps_the_full_earth_size(self):
        for radius in (0.5, 1.0, 2.0):
            with self.subTest(radius=radius):
                result = widgets.size_comparison(radius, 1.0)
                self.assertEqual("", result["note"])
                self.assertEqual(widgets.EARTH_DIAMETER_PX, result["earth_px"])

    def test_jupiter_would_not_fit_and_is_shrunk_with_a_note(self):
        """11.21 R-Earth at a 90 px Earth is 1,009 px: the panel shrinks both."""
        result = widgets.size_comparison(11.21, 1.0)

        self.assertEqual(widgets.MAX_PANEL_DIAMETER_PX, result["planet_px"])
        self.assertIn("Earth shown at 8.9% of the planet's scale to fit", result["note"])
        self.assertAlmostEqual(11.21, result["ratio"], places=3)

    def test_a_planet_that_does_not_fit_is_shrunk_with_a_note(self):
        radius = 20.0
        result = widgets.size_comparison(radius, 1.0)
        earth = drawn_diameter(result["html"], "earth")
        planet = drawn_diameter(result["html"], "planet")

        self.assertEqual(widgets.MAX_PANEL_DIAMETER_PX, planet)
        self.assertLess(earth, widgets.EARTH_DIAMETER_PX)
        self.assertAlmostEqual(radius, planet / earth, places=3)
        self.assertIn("Earth shown at", result["note"])
        self.assertIn("% of the planet's scale to fit", result["note"])
        self.assertAlmostEqual(100.0 * earth / planet, 100.0 / radius, places=3)
        # Both circles were scaled by the same factor.
        self.assertAlmostEqual(
            widgets.EARTH_DIAMETER_PX * result["factor"], earth, places=3
        )
        self.assertAlmostEqual(
            widgets.EARTH_DIAMETER_PX * radius * result["factor"], planet, places=3
        )

    def test_earth_alone_is_drawn_at_the_full_scale(self):
        result = widgets.size_comparison(1.0, 1.0)

        self.assertEqual(widgets.EARTH_DIAMETER_PX, result["earth_px"])
        self.assertEqual(widgets.EARTH_DIAMETER_PX, result["planet_px"])
        self.assertEqual(1.0, result["factor"])

    def test_a_tiny_planet_is_not_clamped(self):
        """No minimum size: a clamp would break the linear relation."""
        result = widgets.size_comparison(0.187, 0.00064)

        self.assertAlmostEqual(widgets.EARTH_DIAMETER_PX * 0.187, result["planet_px"])
        self.assertAlmostEqual(0.187, result["ratio"], places=6)

    def test_the_drawing_refuses_a_non_positive_radius(self):
        with self.assertRaises(ValueError):
            widgets.planet_diameter_px(0.0)


class SystemPreviewTests(unittest.TestCase):
    """The System Preview is labelled an illustration, not a scale drawing."""

    def test_the_planet_circle_is_linear_in_radius_until_the_cap(self):
        half = widgets.system_illustration(0.5, 1.0, "G (Sun-like)")
        one = widgets.system_illustration(1.0, 1.0, "G (Sun-like)")

        self.assertAlmostEqual(2.0 * half["planet_px"], one["planet_px"], places=3)
        self.assertIn("Host Star (G (Sun-like))", one["html"])

    def test_large_planets_are_capped(self):
        big = widgets.system_illustration(30.0, 1.0, "M (red dwarf)")

        self.assertEqual(widgets.SYSTEM_MAX_DIAMETER_PX, big["planet_px"])


class PresetTests(unittest.TestCase):
    """Presets are computed from the reference files, never typed in."""

    def setUp(self):
        self.assessed = pd.read_csv(ROOT / "data" / "processed" / "planets_assessed.csv")
        self.solar = presets.solar_system_presets()
        self.archive = presets.archive_presets(self.assessed)

    def test_every_archive_preset_matches_its_processed_row(self):
        for name in presets.ARCHIVE_QUICK_PICKS:
            row = self.assessed[self.assessed["pl_name"] == name].iloc[0]
            with self.subTest(preset=name):
                self.assertEqual(float(row["pl_rade"]), self.archive[name]["pl_rade"])
                self.assertEqual(float(row["pl_insol"]), self.archive[name]["pl_insol"])
                self.assertEqual(float(row["st_teff"]), self.archive[name]["st_teff"])

    def test_the_archive_group_holds_every_scorable_planet_once(self):
        self.assertEqual(len(self.assessed), len(self.archive))
        self.assertEqual(
            set(self.assessed["pl_name"]), set(self.archive)
        )
        self.assertEqual(
            list(presets.ARCHIVE_QUICK_PICKS), list(self.archive)[:4]
        )

    def test_the_solar_group_scales_on_earths_own_row(self):
        table = pd.read_csv(ROOT / "data" / "reference" / "solar_system.csv")
        earth = table[table["name"] == "Earth"].iloc[0]
        jupiter = table[table["name"] == "Jupiter"].iloc[0]

        self.assertEqual(8, len(self.solar))
        self.assertEqual(1.0, self.solar["Earth"]["pl_rade"])
        self.assertEqual(1.0, self.solar["Earth"]["pl_insol"])
        self.assertAlmostEqual(
            jupiter["mean_radius_km"] / earth["mean_radius_km"],
            self.solar["Jupiter"]["pl_rade"],
            places=12,
        )
        self.assertAlmostEqual(
            (earth["semimajor_axis_km"] / jupiter["semimajor_axis_km"]) ** 2,
            self.solar["Jupiter"]["pl_insol"],
            places=12,
        )

    def test_a_missing_quick_pick_is_skipped_rather_than_guessed(self):
        built = presets.archive_presets(self.assessed, quick_picks=("Not A Planet",))

        self.assertNotIn("Not A Planet", built)
        self.assertEqual(len(self.assessed), len(built))

    def test_a_solar_file_without_earth_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "solar.csv"
            path.write_text(
                "name,mean_radius_km,semimajor_axis_km\nMars,3389.5,227920000\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                presets.solar_system_presets(path)

        self.assertIn("no Earth row", str(caught.exception))

    def test_a_solar_file_missing_a_column_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "solar.csv"
            path.write_text("name,mean_radius_km\nEarth,6371.0\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                presets.solar_system_presets(path)

        self.assertIn("semimajor_axis_km", str(caught.exception))


class BatchValidatorTests(unittest.TestCase):
    """The batch scorer accepts good rows and reports the rest line by line."""

    def test_a_valid_file_is_scored_with_the_calculator_formula(self):
        text = "name,pl_rade,pl_insol,st_teff\nA,1.0,1.0,5780\nB,2.0,0.05,3000\n"
        result = batch.score_batch(text)

        self.assertTrue(result.ok)
        self.assertEqual(2, result.accepted)
        self.assertEqual(2, result.total_rows)
        self.assertEqual([], result.errors)
        self.assertAlmostEqual(calculate_esi(1.0, 1.0), result.frame.loc[0, "base_score"])
        self.assertAlmostEqual(calculate_esi(2.0, 0.05), result.frame.loc[1, "base_score"])
        self.assertEqual(["conservative", "too_cold"], list(result.frame["hz_class"]))

    def test_a_missing_required_column_is_reported(self):
        result = batch.score_batch("name,pl_rade\nA,1.0\n")

        self.assertEqual(0, result.accepted)
        self.assertEqual("header", result.errors[0].column)
        self.assertIn("pl_insol", result.errors[0].message)

    def test_an_empty_file_is_reported(self):
        for text in ("", "   \n", b""):
            with self.subTest(text=text):
                result = batch.score_batch(text)
                self.assertEqual(0, result.accepted)
                self.assertEqual(1, len(result.errors))
                self.assertEqual("file", result.errors[0].column)
                self.assertIn("empty", result.errors[0].message)

    def test_a_header_only_file_is_reported(self):
        result = batch.score_batch("name,pl_rade,pl_insol\n")

        self.assertEqual(0, result.accepted)
        self.assertEqual(1, len(result.errors))
        self.assertEqual("header", result.errors[0].column)
        self.assertIn("no data rows", result.errors[0].message)

    def test_bad_values_are_rejected_row_by_row(self):
        text = (
            "name,pl_rade,pl_insol,st_teff\n"
            "good,1.0,1.0,5780\n"
            "text radius,abc,1.0,5780\n"
            "zero radius,0,1.0,5780\n"
            "negative flux,1.0,-0.2,5780\n"
            "no radius,,1.0,5780\n"
            "text teff,1.0,1.0,warm\n"
        )
        result = batch.score_batch(text)

        self.assertEqual(1, result.accepted)
        self.assertEqual(5, result.rejected)
        self.assertEqual(5, len(result.errors))
        self.assertEqual([3, 4, 5, 6, 7], [error.line for error in result.errors])
        self.assertEqual(
            ["pl_rade", "pl_rade", "pl_insol", "pl_rade", "st_teff"],
            [error.column for error in result.errors],
        )
        self.assertIn("is not a number", str(result.errors[0]))
        self.assertIn("must be positive", str(result.errors[1]))
        self.assertIn("must not be negative", str(result.errors[2]))
        self.assertIn("is missing", str(result.errors[3]))

    def test_a_missing_temperature_is_not_an_error_but_leaves_the_zone_undetermined(self):
        result = batch.score_batch("name,pl_rade,pl_insol,st_teff\nA,1.0,1.0,\n")

        self.assertEqual([], result.errors)
        self.assertEqual(1, result.accepted)
        self.assertEqual("undetermined", result.frame.loc[0, "hz_class"])

    def test_a_temperature_outside_the_fitted_range_is_not_extrapolated(self):
        result = batch.score_batch("name,pl_rade,pl_insol,st_teff\nA,1.0,1.0,9000\n")

        self.assertEqual([], result.errors)
        self.assertEqual("undetermined", result.frame.loc[0, "hz_class"])

    def test_more_than_five_thousand_rows_is_refused(self):
        result = batch.score_batch(small_csv(batch.MAX_ROWS + 1))

        self.assertEqual(0, result.accepted)
        self.assertEqual(1, len(result.errors))
        self.assertIn("at most 5,000 rows", result.errors[0].message)

    def test_a_payload_over_the_byte_ceiling_is_refused_before_it_is_parsed(self):
        oversized = small_csv(200_000)
        self.assertGreater(len(oversized), batch.MAX_UPLOAD_BYTES)

        with mock.patch.object(batch.pd, "read_csv") as read_csv:
            result = batch.score_batch(oversized)

        read_csv.assert_not_called()
        self.assertEqual(0, result.accepted)
        self.assertEqual(1, len(result.errors))
        self.assertEqual("file", result.errors[0].column)
        self.assertIn("bytes are accepted", result.errors[0].message)

    def test_the_row_cap_is_applied_by_the_reader_not_after_it(self):
        text = small_csv(batch.MAX_ROWS + 50)
        self.assertLess(len(text), batch.MAX_UPLOAD_BYTES)

        with mock.patch.object(batch.pd, "read_csv", wraps=batch.pd.read_csv) as read_csv:
            result = batch.score_batch(text)

        self.assertEqual(0, result.accepted)
        self.assertIn("at most 5,000 rows", result.errors[0].message)
        self.assertEqual(batch.MAX_ROWS + 1, read_csv.call_args.kwargs["nrows"])

    def test_exactly_five_thousand_rows_is_accepted(self):
        result = batch.score_batch(small_csv(batch.MAX_ROWS))

        self.assertEqual(batch.MAX_ROWS, result.accepted)
        self.assertEqual([], result.errors)

    def test_only_the_first_twenty_errors_are_listed(self):
        body = "".join(f"row {index},0,1.0\n" for index in range(25))
        result = batch.score_batch("name,pl_rade,pl_insol\n" + body)

        self.assertEqual(25, len(result.errors))
        self.assertEqual(20, len(result.error_lines()))
        self.assertEqual(20, len(result.reported_errors))

    def test_the_export_keeps_the_disclaimer_column(self):
        result = batch.score_batch((EXAMPLE_CSV).read_text(encoding="utf-8"))
        exported = batch.export_frame(result)

        self.assertIn("disclaimer", exported.columns)
        self.assertTrue((exported["disclaimer"] == DISCLAIMER).all())
        self.assertEqual(len(result.frame), len(exported))

    def test_the_example_file_scores_four_of_seven_rows(self):
        result = batch.score_batch(EXAMPLE_CSV.read_bytes())

        self.assertEqual(7, result.total_rows)
        self.assertEqual(4, result.accepted)
        self.assertEqual(3, result.rejected)
        self.assertEqual(
            ["Kepler-452 b", "Proxima Cen b", "TOI-700 d", "Hot star"],
            list(result.frame["name"]),
        )
        self.assertEqual(
            ["conservative", "conservative", "undetermined", "undetermined"],
            list(result.frame["hz_class"]),
        )


class BatchNoDiskWriteTests(unittest.TestCase):
    """An upload is processed from memory: nothing is written anywhere."""

    def test_scoring_creates_no_file(self):
        payload = EXAMPLE_CSV.read_bytes()
        before_app = set(path.name for path in APP_DIR.iterdir())
        before_root = set(path.name for path in ROOT.iterdir())

        with tempfile.TemporaryDirectory() as scratch:
            before_tmp = set(Path(scratch).iterdir())
            result = batch.score_batch(payload)
            self.assertEqual(4, result.accepted)
            self.assertEqual(before_tmp, set(Path(scratch).iterdir()))

        self.assertEqual(before_app, set(path.name for path in APP_DIR.iterdir()))
        self.assertEqual(before_root, set(path.name for path in ROOT.iterdir()))

    def test_a_stream_is_consumed_in_memory(self):
        stream = io.BytesIO(EXAMPLE_CSV.read_bytes())

        result = batch.score_batch(stream.getvalue())

        self.assertEqual(7, result.total_rows)
        self.assertFalse(stream.closed)

    def test_the_batch_module_has_no_filesystem_writes(self):
        source = (APP_DIR / "batch.py").read_text(encoding="utf-8")

        for forbidden in (
            "open(",
            "to_csv(",
            "write_bytes",
            "write_text",
            "NamedTemporaryFile",
            "mkstemp",
            "tempfile",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_the_page_reads_the_upload_from_memory(self):
        source = (APP_DIR / "views" / "calculator.py").read_text(encoding="utf-8")

        self.assertIn("uploaded.getvalue()", source)
        self.assertNotIn("uploaded.name", source)
        self.assertNotIn("save_uploaded_file", source)


if __name__ == "__main__":
    unittest.main()
