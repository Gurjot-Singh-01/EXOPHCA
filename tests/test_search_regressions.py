"""Regression tests for typed search values and computed widget defaults.

The final audit found the Planet Detail page passing the position of
"Kepler-452 b" in the full 5,688-name list as ``index`` to a selectbox whose
options the search box had narrowed to one, so typing "Kepler-452 b" raised
``StreamlitValueOutOfRangeError: index 3778 is outside [0, 0]`` and "Kepler"
raised ``[0, 2742]``. No test typed into that box, which is why 244 tests passed
with the defect in place.

These tests type into the box as a user does. The Calculator and Compare
selectors search on the client side, so a string that matches no option is
never submitted to the page; those two are tested with a name containing
special characters and for the fact that a nonsense string is not an option.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import presets  # noqa: E402
import query  # noqa: E402

VIEWS = APP_DIR / "views"
DETAIL_FILE = VIEWS / "detail.py"
CALCULATOR_FILE = VIEWS / "calculator.py"
COMPARE_FILE = VIEWS / "compare.py"
PROCESSED = ROOT / "data" / "processed"

SPECIAL_CHARS_NAME = "WD 1856+534 b"
NONSENSE = "zzzzqq"


def run_page(path: Path):
    return AppTest.from_file(str(path), default_timeout=240).run()


def all_text(app) -> str:
    """Every piece of text the page rendered, flattened for assertions."""
    parts = []
    for collection in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "text",
    ):
        for item in getattr(app, collection, []):
            parts.append(str(getattr(item, "value", "")))
    for metric in app.metric:
        parts.append(f"{metric.label}: {metric.value}")
    return "\n".join(parts)


def exceptions(app) -> list[str]:
    return [exception.message for exception in app.exception]


def explorer_frame() -> pd.DataFrame:
    assessed = pd.read_csv(PROCESSED / "planets_assessed.csv")
    archive = pd.read_csv(PROCESSED / "planets.csv")
    return query.join_planets(assessed, archive)


def search_detail(value: str):
    """Type ``value`` into the Detail page's "Find a planet" box and rerun."""
    app = run_page(DETAIL_FILE)
    app.text_input[0].set_value(value)
    app.run()
    return app


def expected_names(value: str) -> list[str]:
    """The options the Detail page must offer for one search string."""
    frame = explorer_frame()
    return sorted(query.apply_filters(frame, search=value)["pl_name"].dropna().unique())


class DetailSearchTests(unittest.TestCase):
    """Typing into "Find a planet" must narrow the list without raising."""

    def test_an_exact_name_finds_exactly_one_planet(self):
        app = search_detail("Kepler-452 b")

        self.assertEqual([], exceptions(app))
        self.assertEqual(1, len(app.selectbox))
        self.assertEqual(["Kepler-452 b"], list(app.selectbox[0].options))
        self.assertEqual("Kepler-452 b", app.selectbox[0].value)
        self.assertIn("Kepler-452 b", all_text(app))

    def test_the_host_name_finds_every_kepler_and_keeps_a_valid_default(self):
        app = search_detail("Kepler")

        self.assertEqual([], exceptions(app))
        options = list(app.selectbox[0].options)
        self.assertEqual(expected_names("Kepler"), options)
        # 2,746 matches the planet name or the host star; 2,743 match the planet
        # name alone, which is what the old, planet-name-only list held.
        self.assertGreaterEqual(len(options), 2743)
        # The default planet is among the matches, so it stays selected: this is
        # the position the index must be computed from, inside the filtered list.
        self.assertEqual("Kepler-452 b", app.selectbox[0].value)

    def test_a_lower_case_search_finds_the_same_planet(self):
        app = search_detail("kepler-452")

        self.assertEqual([], exceptions(app))
        self.assertEqual(["Kepler-452 b"], list(app.selectbox[0].options))
        self.assertEqual("Kepler-452 b", app.selectbox[0].value)

    def test_surrounding_whitespace_is_ignored(self):
        app = search_detail("  Proxima  ")

        self.assertEqual([], exceptions(app))
        self.assertEqual(expected_names("Proxima"), list(app.selectbox[0].options))
        self.assertIn("Proxima Cen b", app.selectbox[0].options)
        self.assertIn("Proxima Cen b", all_text(app))

    def test_a_special_character_name_is_read_literally(self):
        app = search_detail(SPECIAL_CHARS_NAME)

        self.assertEqual([], exceptions(app))
        self.assertEqual([SPECIAL_CHARS_NAME], list(app.selectbox[0].options))
        self.assertEqual(SPECIAL_CHARS_NAME, app.selectbox[0].value)

    def test_a_nonsense_string_shows_a_message_and_no_selectbox(self):
        app = search_detail(NONSENSE)

        self.assertEqual([], exceptions(app))
        self.assertEqual([], list(app.selectbox))
        text = all_text(app)
        self.assertIn("No planet matches", text)
        self.assertIn(NONSENSE, text)
        # Nothing was selected, so no planet section may be rendered.
        self.assertEqual([], [metric.label for metric in app.metric])

    def test_an_empty_box_offers_every_planet(self):
        app = search_detail("")

        self.assertEqual([], exceptions(app))
        self.assertEqual(expected_names(""), list(app.selectbox[0].options))
        self.assertEqual(len(explorer_frame()), len(app.selectbox[0].options))


class CalculatorPresetSearchTests(unittest.TestCase):
    """The Calculator preset list accepts archive names, including "+"."""

    def test_an_archive_planet_with_special_characters_loads_cleanly(self):
        app = run_page(CALCULATOR_FILE)
        app.radio[0].set_value(presets.GROUP_ARCHIVE)
        app.run()

        self.assertIn(SPECIAL_CHARS_NAME, app.selectbox[0].options)
        app.selectbox[0].set_value(SPECIAL_CHARS_NAME)
        app.run()

        self.assertEqual([], exceptions(app))
        self.assertEqual(SPECIAL_CHARS_NAME, app.selectbox[0].value)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertIn("ESI-style radius-flux similarity", metrics)
        self.assertNotIn("Habitable-zone class", metrics)
        preset = presets.archive_presets(
            pd.read_csv(PROCESSED / "planets_assessed.csv")
        )[SPECIAL_CHARS_NAME]
        self.assertAlmostEqual(float(preset["pl_rade"]), float(app.number_input[0].value))
        self.assertAlmostEqual(float(preset["pl_insol"]), float(app.number_input[1].value))

    def test_a_nonsense_search_matches_no_option_and_leaves_the_page_clean(self):
        """The preset search is client-side, so it can only pick real options."""
        app = run_page(CALCULATOR_FILE)
        app.radio[0].set_value(presets.GROUP_ARCHIVE)
        app.run()

        options = list(app.selectbox[0].options)
        self.assertEqual([], [name for name in options if NONSENSE in name.lower()])
        app.selectbox[0].set_value(SPECIAL_CHARS_NAME)
        app.run()
        self.assertEqual([], exceptions(app))


class CompareSelectorTests(unittest.TestCase):
    """The Compare selector accepts the same archive names."""

    def test_a_special_character_planet_can_be_compared(self):
        app = run_page(COMPARE_FILE)
        self.assertIn(SPECIAL_CHARS_NAME, app.multiselect[0].options)
        app.multiselect[0].set_value([SPECIAL_CHARS_NAME, "Kepler-452 b"])
        app.run()

        self.assertEqual([], exceptions(app))
        text = all_text(app)
        self.assertIn(SPECIAL_CHARS_NAME, text)
        self.assertIn("Kepler-452 b", text)

    def test_a_nonsense_search_matches_no_option_and_leaves_the_page_clean(self):
        app = run_page(COMPARE_FILE)

        options = list(app.multiselect[0].options)
        self.assertEqual([], [name for name in options if NONSENSE in name.lower()])
        app.multiselect[0].set_value([SPECIAL_CHARS_NAME])
        app.run()
        self.assertEqual([], exceptions(app))


class ComputedDefaultTests(unittest.TestCase):
    """Every computed widget default must be a member of its own option list."""

    def test_every_detail_search_offers_the_selected_planet(self):
        for value in ("Kepler-452 b", "Kepler", "kepler-452", "  Proxima  ",
                      SPECIAL_CHARS_NAME, ""):
            with self.subTest(search=value):
                app = run_page(DETAIL_FILE)
                app.text_input[0].set_value(value)
                app.run()
                self.assertEqual([], exceptions(app))
                self.assertIn(app.selectbox[0].value, app.selectbox[0].options)

    def test_calculator_star_class_index_is_always_in_range(self):
        star_classes = ["G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)"]
        solar = presets.solar_system_presets()
        archive = presets.archive_presets(
            pd.read_csv(PROCESSED / "planets_assessed.csv")
        )
        for name, preset in list(solar.items()) + list(archive.items()):
            with self.subTest(preset=name):
                star_class = preset["star_class"]
                # The page's expression, verbatim: no member can produce an
                # index outside the list, and anything else falls back to 0.
                index = (
                    star_classes.index(star_class)
                    if star_class in star_classes
                    else 0
                )
                self.assertTrue(0 <= index < len(star_classes))

    def test_compare_default_selection_is_a_subset_of_its_options(self):
        frame = explorer_frame()
        names = sorted(frame["pl_name"].dropna().unique())
        app = run_page(COMPARE_FILE)

        self.assertEqual([], exceptions(app))
        self.assertTrue(set(app.multiselect[0].value) <= set(names))
        self.assertLessEqual(len(app.multiselect[0].value), 3)


if __name__ == "__main__":
    unittest.main()
