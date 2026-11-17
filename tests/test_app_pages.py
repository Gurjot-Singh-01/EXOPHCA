"""Tests for the multipage shell, the Overview, Explorer, Detail and Methods
pages, and the labelling rule for probabilities and radius provenance."""

import json
import re
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import labelling  # noqa: E402
import plot_data  # noqa: E402
import query  # noqa: E402
import reference_earth  # noqa: E402
import ui  # noqa: E402
from exophca.science.assessment import (  # noqa: E402
    DISCLAIMER,
    determine_size_regime,
    determine_tier,
)
from exophca.science.habitable_zone import classify_hz, hz_limits  # noqa: E402

ENTRY_FILE = APP_DIR / "EXOPHCA.py"
VIEWS_DIR = APP_DIR / "views"
OVERVIEW_FILE = VIEWS_DIR / "overview.py"
EXPLORER_FILE = VIEWS_DIR / "explorer.py"
DETAIL_FILE = VIEWS_DIR / "detail.py"
COMPARE_FILE = VIEWS_DIR / "compare.py"
DIAGRAMS_FILE = VIEWS_DIR / "diagrams.py"
CALCULATOR_FILE = VIEWS_DIR / "calculator.py"
TESS_FILE = VIEWS_DIR / "tess_candidates.py"
METHODS_FILE = VIEWS_DIR / "methods.py"
PAGE_FILES = [
    OVERVIEW_FILE,
    EXPLORER_FILE,
    DETAIL_FILE,
    COMPARE_FILE,
    DIAGRAMS_FILE,
    CALCULATOR_FILE,
    TESS_FILE,
    METHODS_FILE,
]

PROCESSED = ROOT / "data" / "processed"


def run_page(path: Path):
    return AppTest.from_file(str(path), default_timeout=180).run()


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


def provenance_date(key: str) -> str:
    raw = json.loads((PROCESSED / f"{key}.provenance.json").read_text(encoding="utf-8"))
    return date.fromisoformat(
        pd.to_datetime(raw["retrieval_timestamp"], format="%a %b %d %H:%M:%S %Y").date().isoformat()
    ).strftime("%d %B %Y")


def explorer_frame() -> pd.DataFrame:
    assessed = pd.read_csv(PROCESSED / "planets_assessed.csv")
    archive = pd.read_csv(PROCESSED / "planets.csv")
    return query.join_planets(assessed, archive)


def apply_filters(app):
    """Press "Apply filters" on a page whose filter panel is an st.form.

    Streamlit commits form widgets only when the form is submitted, so a test
    must stage the value *and* click the button before the next run.
    """
    buttons = {button.label: button for button in app.button}
    if "Apply filters" not in buttons:
        raise AssertionError(f"no Apply filters button; buttons are {list(buttons)}")
    buttons["Apply filters"].click()
    return app


def explorer_count(app) -> int:
    """The row count the page reports for the applied filters."""
    match = re.search(r"\*\*(\d[\d,]*)", all_text(app))
    if match is None:
        raise AssertionError("no match-count line on the Explorer page")
    return int(match.group(1).replace(",", ""))


# Filter key -> the AppTest accessor for that widget.
FILTER_WIDGET_ACCESSORS = {
    "exo_search": "text_input",
    "exo_methods": "multiselect",
    "exo_star_classes": "multiselect",
    "exo_provenance": "multiselect",
    "exo_evidence": "multiselect",
    "exo_tiers": "multiselect",
    "exo_hz": "multiselect",
    "exo_status": "multiselect",
    "exo_score": "slider",
    "exo_sort": "selectbox",
    "exo_order": "radio",
    "exo_probabilities": "toggle",
}


def stage_filter(app, key, value):
    """Set one filter widget without running: the value is staged for a submit."""
    return getattr(app, FILTER_WIDGET_ACCESSORS[key])(key=key).set_value(value)


def run_explorer(**filters):
    """Open the Explorer, stage the named filters, apply them and return the app."""
    app = run_page(EXPLORER_FILE)
    for key, value in filters.items():
        stage_filter(app, key, value)
    apply_filters(app)
    app.run()
    return app


def run_detail(planet: str | None = None):
    """Open the Detail page, optionally selecting a planet, and return the app."""
    app = run_page(DETAIL_FILE)
    if planet is not None:
        app.selectbox[0].select(planet)
        app.run()
    return app


class PageSmokeTests(unittest.TestCase):
    """Every page, and the entry point, must render without an exception."""

    def test_entry_and_every_page_loads_without_exception(self):
        for path in [ENTRY_FILE] + PAGE_FILES:
            with self.subTest(page=path.name):
                app = run_page(path)
                self.assertEqual(
                    [], [exception.message for exception in app.exception]
                )

    def test_every_navigable_page_has_its_own_file_and_title(self):
        expected_titles = {
            OVERVIEW_FILE: "Overview",
            EXPLORER_FILE: "Planet Explorer",
            DETAIL_FILE: "Planet Detail",
            COMPARE_FILE: "Compare",
            DIAGRAMS_FILE: "Diagrams",
            CALCULATOR_FILE: "Calculator",
            TESS_FILE: "TESS Candidates (Unconfirmed)",
            METHODS_FILE: "Methods and limitations",
        }
        for path, title in expected_titles.items():
            with self.subTest(page=path.name):
                self.assertTrue(path.exists())
                app = run_page(path)
                self.assertEqual([], [exception.message for exception in app.exception])
                self.assertEqual(title, app.title[0].value)


class OverviewTests(unittest.TestCase):
    """The Overview must name the data, date it, and state what scores mean."""

    def setUp(self):
        self.app = run_page(OVERVIEW_FILE)
        self.text = all_text(self.app)

    def test_overview_loads(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])

    def test_overview_shows_both_dataset_names_and_retrieval_dates(self):
        self.assertIn("Planetary Systems Composite Parameters", self.text)
        self.assertIn("TESS Objects of Interest", self.text)
        self.assertIn(provenance_date("PSCD"), self.text)
        self.assertIn(provenance_date("TOI_2026"), self.text)

    def test_overview_shows_raw_and_assessed_row_counts(self):
        assessed = pd.read_csv(PROCESSED / "planets_assessed.csv")
        toi_assessed = pd.read_csv(PROCESSED / "toi_assessed.csv")
        pscd = json.loads((PROCESSED / "PSCD.provenance.json").read_text(encoding="utf-8"))
        toi = json.loads((PROCESSED / "TOI_2026.provenance.json").read_text(encoding="utf-8"))

        self.assertIn(f"{pscd['row_count']:,}", self.text)
        self.assertIn(f"{toi['row_count']:,}", self.text)
        self.assertIn(f"{len(assessed):,}", self.text)
        self.assertIn(f"{len(toi_assessed):,}", self.text)

    def test_overview_states_the_disclaimer_and_plain_language_meaning(self):
        self.assertIn(DISCLAIMER, self.text)
        self.assertIn("not a probability of life", self.text)
        self.assertIn("Conservative", self.text)


class ExplorerTests(unittest.TestCase):
    """The Explorer table, its filters and its export."""

    def setUp(self):
        self.app = run_page(EXPLORER_FILE)
        self.text = all_text(self.app)
        self.frame = explorer_frame()

    def test_explorer_loads_and_shows_every_planet(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])
        self.assertEqual(1, len(self.app.dataframe))
        self.assertIn(f"{len(self.frame):,} of {len(self.frame):,} planets", self.text)

    def test_explorer_has_a_download_button_below_the_disclaimer(self):
        self.assertEqual(1, len(self.app.get("download_button")))
        self.assertIn(DISCLAIMER, self.text)

    def test_explorer_export_contains_provenance_and_uncertainty_columns(self):
        export = query.build_export_frame(self.frame, retrieval_date="2026-07-11")

        for column in query.REQUIRED_EXPORT_COLUMNS:
            self.assertIn(column, export.columns)
        self.assertEqual(len(self.frame), len(export))
        self.assertTrue((export["data_retrieval_date"] == "2026-07-11").all())

    def test_explorer_export_caveat_follows_uncertainty_status(self):
        export = query.build_export_frame(self.frame, retrieval_date="2026-07-11")

        incomplete = export[export["uncertainty_status"] != "full"]
        complete = export[export["uncertainty_status"] == "full"]
        self.assertGreater(len(incomplete), 0)
        self.assertGreater(len(complete), 0)
        self.assertTrue(
            incomplete["probability_caveat"].str.contains(
                labelling.CONDITIONAL_TEXT, regex=False
            ).all()
        )
        self.assertFalse(
            complete["probability_caveat"].str.contains(
                labelling.CONDITIONAL_TEXT, regex=False
            ).any()
        )

    def test_explorer_export_marks_the_modelled_radius(self):
        export = query.build_export_frame(self.frame, retrieval_date="2026-07-11")
        proxima = export[export["pl_name"] == "Proxima Cen b"].iloc[0]

        self.assertEqual("modelled", proxima["radius_provenance"])
        self.assertIn(labelling.MODELLED_RADIUS_TEXT, proxima["radius_note"])

    def test_explorer_probability_toggle_labels_every_value(self):
        app = run_page(EXPLORER_FILE)
        app.toggle[0].set_value(True)
        apply_filters(app)
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        table = app.dataframe[0].value
        self.assertIn("uncertainty_status", table.columns)
        for name in labelling.PROBABILITY_DEPENDENCIES:
            self.assertIn(name, table.columns)
            self.assertTrue(table[name].str.contains("uncertainty_status").all())

    def test_explorer_filters_by_tier(self):
        filtered = query.apply_filters(self.frame, tiers=["A"])

        self.assertGreater(len(filtered), 0)
        self.assertEqual({"A"}, set(filtered["tier"]))
        self.assertLess(len(filtered), len(self.frame))

    def test_explorer_search_matches_planet_names(self):
        filtered = query.apply_filters(self.frame, search="TRAPPIST")

        self.assertGreater(len(filtered), 0)
        self.assertTrue(filtered["pl_name"].str.contains("TRAPPIST", case=False).all())

    def test_the_filter_panel_is_a_form_with_apply_and_reset(self):
        app = self.app
        labels = [button.label for button in app.button]

        self.assertEqual(
            [], [exception.message for exception in app.exception]
        )
        self.assertEqual(1, len(app.get("form")))
        self.assertIn("Apply filters", labels)
        self.assertIn("Reset filters", labels)
        self.assertIn("Apply filters", self.text)
        self.assertIn("typing alone does not change the table", self.text)

    def test_typing_alone_leaves_the_table_untouched_until_it_is_applied(self):
        """The bug: a bare text box commits only on Enter or blur.

        In AppTest a staged form value is dropped by an intervening run, which
        is the same rule the browser follows: the table cannot move until the
        form is submitted.
        """
        app = run_page(EXPLORER_FILE)
        every = len(self.frame)
        stage_filter(app, "exo_search", "Kepler-452")
        app.run()

        self.assertEqual(
            every, len(app.dataframe[0].value), "the table changed before Apply"
        )
        self.assertEqual(every, explorer_count(app), "the count moved before Apply")

        stage_filter(app, "exo_search", "Kepler-452")
        apply_filters(app).run()
        self.assertEqual(1, len(app.dataframe[0].value))
        self.assertEqual("Kepler-452 b", app.dataframe[0].value["pl_name"].iloc[0])

    def test_reset_clears_every_filter(self):
        app = run_explorer(exo_search="TRAPPIST")
        self.assertEqual(7, len(app.dataframe[0].value))

        buttons = {button.label: button for button in app.button}
        buttons["Reset filters"].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(len(self.frame), len(app.dataframe[0].value))
        self.assertEqual("", app.text_input(key="exo_search").value)

    def test_explorer_score_range_filter_bounds_scores(self):
        filtered = query.apply_filters(self.frame, score_range=(0.8, 1.0))

        self.assertGreater(len(filtered), 0)
        self.assertTrue((filtered["base_score"] >= 0.8).all())
        self.assertTrue((filtered["base_score"] <= 1.0).all())

    def test_export_filename_carries_the_retrieval_date(self):
        name = query.export_filename("2026-07-11")

        self.assertTrue(name.startswith("exophca_planets_filtered_"))
        self.assertTrue(name.endswith("2026-07-11.csv"))


class LabellingRuleTests(unittest.TestCase):
    """Probabilities never appear bare, and modelled radii are always named."""

    def test_full_uncertainty_shows_status_and_no_conditional_label(self):
        text = labelling.format_probability("p_conservative_hz", 0.42, "full")

        self.assertIn("0.420", text)
        self.assertIn("uncertainty_status: full", text)
        self.assertNotIn(labelling.CONDITIONAL_TEXT, text)

    def test_no_uncertainty_is_labelled_conditional(self):
        text = labelling.format_probability("p_conservative_hz", 0.42, "none")

        self.assertIn("0.420", text)
        self.assertIn(labelling.CONDITIONAL_TEXT, text)
        self.assertIn("uncertainty_status: none", text)

    def test_radius_only_marks_flux_probabilities_but_not_the_radius_one(self):
        flux_probability = labelling.format_probability("p_any_hz", 0.30, "radius_only")
        radius_probability = labelling.format_probability("p_radius_le_1p6", 0.30, "radius_only")

        self.assertIn(labelling.CONDITIONAL_TEXT, flux_probability)
        self.assertIn("flux held fixed", flux_probability)
        self.assertNotIn(labelling.CONDITIONAL_TEXT, radius_probability)
        self.assertIn("uncertainty_status: radius_only", radius_probability)

    def test_flux_only_marks_radius_probabilities(self):
        radius_probability = labelling.format_probability("p_radius_le_1p6", 0.55, "flux_only")
        joint_probability = labelling.format_probability(
            "p_rocky_and_conservative", 0.55, "flux_only"
        )

        self.assertIn(labelling.CONDITIONAL_TEXT, radius_probability)
        self.assertIn("radius held fixed", radius_probability)
        self.assertIn(labelling.CONDITIONAL_TEXT, joint_probability)

    def test_unknown_status_can_never_look_unconditional(self):
        text = labelling.format_probability("p_radius_le_1p6", 0.55, "unexpected")

        self.assertIn(labelling.CONDITIONAL_TEXT, text)

    def test_modelled_radius_always_names_its_provenance(self):
        modelled = labelling.radius_text(1.02, "modelled")
        measured = labelling.radius_text(1.63, "measured")

        self.assertIn(labelling.MODELLED_RADIUS_TEXT, modelled)
        self.assertNotIn(labelling.MODELLED_RADIUS_TEXT, measured)
        self.assertIn("1.020", modelled)

    def test_labelled_columns_replace_every_bare_probability(self):
        frame = pd.DataFrame(
            {
                "uncertainty_status": ["none", "full"],
                **{name: [0.1, 0.9] for name in labelling.PROBABILITY_DEPENDENCIES},
            }
        )
        labelled = labelling.labelled_probability_columns(frame)

        for name in labelling.PROBABILITY_DEPENDENCIES:
            self.assertTrue(
                labelled[name].map(lambda value: isinstance(value, str)).all()
            )
            self.assertIn(labelling.CONDITIONAL_TEXT, labelled[name].iloc[0])
            self.assertNotIn(labelling.CONDITIONAL_TEXT, labelled[name].iloc[1])
            self.assertIn("uncertainty_status", labelled[name].iloc[1])


class ExplorerSearchTests(unittest.TestCase):
    """The name search: case-insensitive, literal, trimmed, planet or host star.

    The filter helper is checked directly, and the page is then checked through
    the form, because the panel's behaviour (and not only its logic) is what
    made the search look broken.
    """

    def setUp(self):
        self.frame = explorer_frame()

    def names(self, search):
        return list(query.apply_filters(self.frame, search=search)["pl_name"])

    def test_kepler_452_finds_kepler_452_b(self):
        self.assertEqual(["Kepler-452 b"], self.names("Kepler-452"))

    def test_lower_case_finds_the_same_planet(self):
        self.assertEqual(self.names("Kepler-452"), self.names("kepler-452"))
        self.assertEqual(["Kepler-452 b"], self.names("kepler-452"))

    def test_trappist_finds_at_least_seven_planets(self):
        found = self.names("TRAPPIST")

        self.assertGreaterEqual(len(found), 7)
        self.assertTrue(all("TRAPPIST" in name for name in found))

    def test_a_special_character_name_finds_exactly_one_planet(self):
        self.assertEqual(["WD 1856+534 b"], self.names("WD 1856+534 b"))
        self.assertEqual(["WD 1856+534 b"], self.names("wd 1856+534 B"))

    def test_a_regular_expression_is_read_literally(self):
        """``Kepler-4.`` would match 157 names as a regex; literally it is none."""
        as_regex = self.frame[
            self.frame["pl_name"].str.contains("Kepler-4.", case=False, regex=True, na=False)
        ]

        self.assertEqual([], self.names("Kepler-4."))
        self.assertEqual(157, len(as_regex))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(self.names("Kepler-452"), self.names("   Kepler-452   "))

    def test_the_host_star_name_is_searched_too(self):
        host = "Barnard's star"
        in_names = self.frame["pl_name"].str.contains(host, case=False, regex=False, na=False)

        self.assertFalse(bool(in_names.any()), "this host is not in any pl_name")
        found = self.names(host)
        self.assertEqual(4, len(found))
        self.assertTrue(self.frame[self.frame["pl_name"].isin(found)]["hostname"].eq(host).all())

    def test_a_nonsense_string_finds_nothing(self):
        self.assertEqual([], self.names("zzzznotaplanet"))

    def test_an_empty_box_returns_every_row(self):
        self.assertEqual(len(self.frame), len(query.apply_filters(self.frame, search="")))
        self.assertEqual(len(self.frame), len(query.apply_filters(self.frame, search="   ")))

    # ---- the page, through the form -------------------------------------

    def test_a_nonsense_string_shows_a_clear_message_on_the_page(self):
        app = run_explorer(exo_search="zzzznotaplanet")
        text = all_text(app)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(0, len(app.dataframe[0].value))
        self.assertIn(f"0 of {len(self.frame):,} planets", text)
        self.assertIn("No planet matches", text)
        self.assertIn("zzzznotaplanet", text)
        self.assertIn("Reset filters", text)

    def test_the_page_finds_a_planet_by_name_and_by_its_host(self):
        by_name = run_explorer(exo_search="Kepler-452")
        by_host = run_explorer(exo_search="Barnard's star")

        self.assertEqual(["Kepler-452 b"], list(by_name.dataframe[0].value["pl_name"]))
        self.assertEqual(4, len(by_host.dataframe[0].value))

    def test_the_page_ignores_case_and_surrounding_spaces(self):
        for pattern in ("kepler-452", "  KEPLER-452  "):
            with self.subTest(search=pattern):
                app = run_explorer(exo_search=pattern)
                self.assertEqual(["Kepler-452 b"], list(app.dataframe[0].value["pl_name"]))

    def test_an_empty_search_box_returns_every_row_on_the_page(self):
        app = run_explorer(exo_search="")

        self.assertEqual(len(self.frame), len(app.dataframe[0].value))
        self.assertEqual(len(self.frame), explorer_count(app))


class ExplorerExportColumnsTests(unittest.TestCase):
    """The export carries the disclaimer and keeps every earlier column."""

    def setUp(self):
        self.frame = explorer_frame()
        self.export = query.build_export_frame(self.frame, retrieval_date="2026-07-11")

    def test_export_has_a_disclaimer_column_with_the_project_constant(self):
        self.assertIn("disclaimer", self.export.columns)
        self.assertEqual(len(self.frame), len(self.export))
        self.assertTrue((self.export["disclaimer"] == DISCLAIMER).all())

    def test_export_keeps_every_other_column(self):
        for column in query.REQUIRED_EXPORT_COLUMNS:
            self.assertIn(column, self.export.columns)
        for name in labelling.PROBABILITY_DEPENDENCIES:
            self.assertIn(name, self.export.columns)
        # The disclaimer is the last column; nothing before it was replaced.
        self.assertEqual("disclaimer", self.export.columns[-1])
        self.assertEqual(
            ["pl_name", "hostname", "discoverymethod"],
            list(self.export.columns[:3]),
        )


class ExplorerScatterTests(unittest.TestCase):
    """The radius-insolation plane draws every filtered planet, no sampling."""

    def setUp(self):
        self.frame = explorer_frame()
        self.points, self.dropped = query.plot_points(self.frame)
        self.figure = ui.hz_scatter(self.points)

    def test_every_filtered_planet_with_a_positive_radius_and_flux_is_drawn(self):
        usable = self.frame[(self.frame["pl_rade"] > 0) & (self.frame["pl_insol"] > 0)]
        drawn = sum(len(trace.x) for trace in self.figure.data)

        self.assertEqual(len(usable), len(self.points))
        self.assertEqual(len(usable), drawn)
        self.assertEqual(len(self.frame) - len(usable), self.dropped)
        self.assertEqual(
            len(set(self.points["pl_name"])),
            len(set(self.frame["pl_name"])) - self.dropped,
        )

    def test_no_conservative_or_optimistic_planet_is_thinned(self):
        for hz_class in ("conservative", "optimistic_only"):
            with self.subTest(hz_class=hz_class):
                expected = len(self.frame[self.frame["hz_class"] == hz_class])
                self.assertGreater(expected, 0)
                self.assertEqual(
                    expected, len(self.points[self.points["hz_class"] == hz_class])
                )

    def test_a_filtered_subset_plots_all_of_its_rows(self):
        subset = query.apply_filters(self.frame, tiers=["A"])
        points, dropped = query.plot_points(subset)

        self.assertGreater(len(subset), 0)
        self.assertEqual(len(subset), len(points) + dropped)
        self.assertEqual(len(points), sum(len(t.x) for t in ui.hz_scatter(points).data))

    def test_radius_provenance_is_encoded_as_marker_fill(self):
        symbols = {trace.marker.symbol for trace in self.figure.data}
        self.assertEqual(
            {"circle", "circle-open", "diamond-open"}, symbols
        )
        self.assertNotEqual(
            ui.PROVENANCE_MARKERS["measured"][0],
            ui.PROVENANCE_MARKERS["modelled"][0],
        )
        self.assertIn("filled circle is a measured", ui.PROVENANCE_LEGEND)

    def test_every_habitable_zone_class_gets_its_own_colour(self):
        colors = [ui.HZ_COLORS[name] for name in ui.HZ_LABELS]

        self.assertEqual(len(colors), len(set(colors)))
        classes = set(self.points["hz_class"])
        trace_colors = {trace.marker.color for trace in self.figure.data}
        for hz_class in classes:
            self.assertIn(ui.HZ_COLORS[hz_class], trace_colors)

    def test_the_scatter_has_no_sampling_cap_in_its_caption(self):
        app = run_page(EXPLORER_FILE)
        text = all_text(app)

        self.assertIn(f"{len(self.points):,} points drawn", text)
        self.assertIn(f"{len(self.frame):,} filtered planets", text)
        self.assertIn("No sampling cap is applied", text)
        self.assertIn(ui.PROVENANCE_LEGEND, text)
        self.assertEqual(1, len(app.get("plotly_chart")))


class LogAxisTickTests(unittest.TestCase):
    """Log axes use plain powers of ten, never SI prefixes."""

    def test_labels_are_powers_of_ten(self):
        values, labels = ui.power_of_ten_ticks(1e-4, 1e6)

        self.assertEqual(len(values), len(labels))
        self.assertIn("1", labels)
        self.assertIn("10^-4", labels)
        self.assertIn("10^4", labels)
        self.assertIn("10^6", labels)
        for label in labels:
            self.assertNotRegex(label, r"[µumkMG]")

    def test_labels_stay_tied_to_decades(self):
        values, labels = ui.power_of_ten_ticks(0.0027, 21.0)

        for value, label in zip(values, labels):
            exponent = 0 if label == "1" else int(label.split("^")[1])
            self.assertAlmostEqual(value, 10.0**exponent, places=10)
        self.assertLessEqual(len(labels), 9)

    def test_non_positive_bounds_return_no_ticks(self):
        self.assertEqual(([], []), ui.power_of_ten_ticks(0.0, 1.0))

    def test_explorer_figure_uses_the_power_labels(self):
        frame = explorer_frame()
        points, _ = query.plot_points(frame)
        figure = ui.hz_scatter(points)

        for axis in (figure.layout.xaxis, figure.layout.yaxis):
            self.assertEqual("log", axis.type)
            self.assertGreater(len(axis.ticktext), 0)
            for label in axis.ticktext:
                self.assertTrue(label == "1" or label.startswith("10^"))


class DetailPageTests(unittest.TestCase):
    """One planet, every number, and the labelling rule applied to each."""

    def setUp(self):
        self.assessed = pd.read_csv(PROCESSED / "planets_assessed.csv").set_index("pl_name")

    def test_detail_lists_every_scorable_planet(self):
        app = run_page(DETAIL_FILE)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(len(self.assessed), len(app.selectbox[0].options))

    def test_measured_radius_planet_shows_its_score_range_and_criteria(self):
        app = run_detail("Kepler-452 b")
        text = all_text(app)
        metrics = {metric.label: metric.value for metric in app.metric}
        row = self.assessed.loc["Kepler-452 b"]

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(f"{row['base_score']:.3f}", metrics["Score"])
        self.assertEqual(
            f"{row['base_score_p16']:.3f}\u2013{row['base_score_p84']:.3f}",
            metrics["16th-84th percentile"],
        )
        self.assertNotIn("Habitable-zone class", metrics)
        self.assertIn(ui.HZ_LABELS[row["hz_class"]], text)
        self.assertEqual(row["tier"], metrics["Compatibility tier"])
        self.assertIn(row["criteria"], text)
        self.assertIn(labelling.MEASURED_RADIUS_TEXT, text)
        self.assertIn(row["evidence_quality"], text)
        self.assertNotIn("ML radius estimate (indicative)", text)

    def test_measured_radius_planet_shows_no_ml_estimate(self):
        app = run_detail("Kepler-452 b")
        text = all_text(app)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertNotIn("ML radius estimate (indicative)", text)
        self.assertNotIn("minimum mass", " ".join(w.value for w in app.warning))
        self.assertNotIn("No ML radius estimate", " ".join(i.value for i in app.info))

    def test_boundary_sensitive_planet_is_warned_about(self):
        app = run_detail("Kepler-452 b")
        warnings = " | ".join(warning.value for warning in app.warning)

        self.assertTrue(bool(self.assessed.loc["Kepler-452 b", "boundary_sensitive"]))
        self.assertIn("Boundary sensitive", warnings)
        self.assertIn("uncertainty_status: full", warnings)

    def test_modelled_radius_planet_shows_the_estimate_and_the_labelling_rule(self):
        app = run_detail("Proxima Cen b")
        text = all_text(app)
        row = self.assessed.loc["Proxima Cen b"]
        estimate = (
            pd.read_csv(PROCESSED / "radius_estimates.csv")
            .set_index("pl_name")
            .loc["Proxima Cen b"]
        )
        warnings = " ".join(warning.value for warning in app.warning)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertIn("ML radius estimate (indicative)", text)
        self.assertIn(f"{estimate['radius_estimate']:.2f}", text)
        self.assertIn(f"{estimate['radius_lower']:.2f}", text)
        self.assertIn(f"{estimate['radius_upper']:.2f}", text)
        self.assertIn(f"{row['pl_rade']:.2f}", text)
        self.assertIn("Minimum-mass input", warnings)
        self.assertIn("sin(i) = 1", warnings)
        self.assertIn("does not change the score, tier, class or probabilities", text)
        self.assertTrue(bool(estimate["interval_straddles_1p6"]))
        self.assertIn("spans the 1.6 R\u2295 rocky threshold", text)
        self.assertIn(labelling.MODELLED_RADIUS_TEXT, text)
        self.assertIn(labelling.CONDITIONAL_TEXT, text)
        self.assertIn("radius held fixed", text)
        self.assertIn(f"uncertainty_status: {row['uncertainty_status']}", text)
        self.assertEqual("flux_only", row["uncertainty_status"])

    def test_every_probability_on_the_detail_page_carries_its_status(self):
        for planet in ("Kepler-452 b", "Proxima Cen b"):
            with self.subTest(planet=planet):
                app = run_detail(planet)
                text = all_text(app)
                row = self.assessed.loc[planet]
                for name in labelling.PROBABILITY_DEPENDENCIES:
                    with self.subTest(probability=name):
                        self.assertIn(
                            labelling.format_probability(
                                name, row[name], row["uncertainty_status"]
                            ),
                            text,
                        )

    def test_detail_shows_the_retrieval_date_and_the_references(self):
        app = run_detail("Kepler-452 b")
        text = all_text(app)

        self.assertIn(provenance_date("PSCD"), text)
        self.assertIn("Schulze-Makuch", text)
        self.assertIn("Kopparapu et al. (2013)", text)
        self.assertIn("Kopparapu et al. (2014)", text)
        self.assertIn("retrieved", text)

    def test_detail_has_the_methods_expander_and_the_disclaimer(self):
        app = run_detail("Kepler-452 b")
        text = all_text(app)

        self.assertIn(DISCLAIMER, text)
        self.assertIn("two-piece normal", text)
        self.assertIn("not a census", text)


class MethodsPageTests(unittest.TestCase):
    """The methods page states the formula, the limits and the caveats."""

    @classmethod
    def setUpClass(cls):
        cls.app = run_page(METHODS_FILE)
        cls.text = all_text(cls.app)

    def test_methods_page_loads(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])
        self.assertEqual("Methods and limitations", self.app.title[0].value)

    def test_methods_page_states_the_formula_and_the_borrowed_weight(self):
        self.assertIn("0.57", self.text)
        self.assertIn("0.70", self.text)
        self.assertIn("escape-velocity", self.text)
        self.assertIn("not** a published flux weight", self.text)

    def test_stated_temperature_range_matches_the_code(self):
        self.assertIn("2600-7200 K", self.text)
        self.assertIn("not** extrapolated", self.text)
        self.assertTrue(hz_limits(2600.0)["in_valid_range"])
        self.assertTrue(hz_limits(7200.0)["in_valid_range"])
        for outside in (2599.0, 7201.0):
            with self.subTest(teff=outside):
                self.assertFalse(hz_limits(outside)["in_valid_range"])

    def test_methods_page_states_the_uncertainty_method_and_statuses(self):
        self.assertIn("two-piece normal", self.text)
        self.assertIn("5,000 draws", self.text)
        for status in ("full", "radius_only", "flux_only", "none", "limit_value"):
            with self.subTest(status=status):
                self.assertIn(status, self.text)
        self.assertIn(labelling.CONDITIONAL_TEXT, self.text)

    def test_methods_page_states_the_evidence_and_tier_rules(self):
        for phrase in ("high", "medium", "low", "none"):
            with self.subTest(evidence=phrase):
                self.assertIn(phrase, self.text)
        self.assertIn("modelled radius can never be high or medium", self.text)
        for tier in ("**A**", "**B**", "**C**", "**D**", "**U**"):
            with self.subTest(tier=tier):
                self.assertIn(tier, self.text)

    def test_methods_page_lists_the_main_limits(self):
        self.assertIn("Similarity is not habitability", self.text)
        self.assertIn("modelled, not measured", self.text)
        self.assertIn("no flux uncertainty", self.text)
        self.assertIn("not a census", self.text)

    def test_methods_page_radius_model_numbers_come_from_the_training_report(self):
        meta = json.loads(
            (ROOT / "models" / "radius_model_meta.json").read_text(encoding="utf-8")
        )
        decision = meta["decision"]
        intervals = meta["intervals"]
        method = intervals["final_recommended"]

        self.assertIn("Radius model (mass-radius estimate)", self.text)
        self.assertIn(decision["chosen_model"], self.text)
        self.assertIn(decision["best_baseline"], self.text)
        self.assertIn(f"{decision['chosen_cv_mae']:.3f}", self.text)
        self.assertIn(f"{decision['best_baseline_cv_mae']:.3f}", self.text)
        self.assertIn(f"{meta['test_metrics']['chosen_model']['MAE']:.3f}", self.text)
        self.assertIn(f"{intervals['split_conformal_half_width_dex']:.3f}", self.text)
        self.assertIn(method, self.text)
        self.assertIn(intervals["recommended_on_cv"], self.text)
        for entry in meta["test_interval_coverage"][method]:
            with self.subTest(stratum=entry["stratum"]):
                self.assertIn(entry["stratum"], self.text)
                self.assertIn(f"{entry['coverage']:.3f}", self.text)
        self.assertIn("not** fully independent evidence", self.text)

    def test_methods_page_states_the_minimum_mass_caveat(self):
        self.assertIn("Msini", self.text)
        self.assertIn("sin(i) = 1", self.text)

    def test_methods_page_ends_with_the_disclaimer(self):
        self.assertIn(DISCLAIMER, self.text)
        self.assertIn("Kopparapu", self.text)
        self.assertIn("Schulze-Makuch", self.text)


class ComparePageTests(unittest.TestCase):
    """Compare: Earth is always the reference row and caveats follow the data."""

    def setUp(self):
        self.app = run_page(COMPARE_FILE)
        self.text = all_text(self.app)
        self.table = self.app.dataframe[0].value

    def warnings(self, app=None) -> str:
        app = app or self.app
        return " | ".join(warning.value for warning in app.warning)

    def test_compare_loads_with_earth_first(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])
        planets = list(self.table["Planet"])
        self.assertEqual("Earth (reference)", planets[0])
        self.assertGreaterEqual(len(planets), 2)
        self.assertLessEqual(len(planets), 4)
        self.assertEqual(1, len(self.app.get("plotly_chart")))

    def test_earth_is_the_reference_with_unit_values(self):
        earth = self.table.iloc[0]
        self.assertEqual("1.000", earth["Score"])
        self.assertIn("1.000 R⊕", earth["Radius"])
        self.assertEqual("Conservative habitable zone", earth["HZ class"])
        self.assertEqual("A", earth["Tier"])

    def test_earth_stays_when_nothing_is_selected(self):
        app = run_page(COMPARE_FILE)
        app.multiselect[0].set_value([])
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        table = app.dataframe[0].value
        self.assertEqual(["Earth (reference)"], list(table["Planet"]))

    def test_modelled_radius_warning_names_proxima_cen_b(self):
        app = run_page(COMPARE_FILE)
        app.multiselect[0].set_value(["Proxima Cen b"])
        app.run()
        text = self.warnings(app)

        self.assertIn("Modelled radius", text)
        self.assertIn("Proxima Cen b", text)
        self.assertIn(labelling.MODELLED_RADIUS_TEXT, text)

    def test_modelled_warning_is_singular_for_one_planet(self):
        app = run_page(COMPARE_FILE)
        app.multiselect[0].set_value(["Proxima Cen b"])
        app.run()
        text = self.warnings(app)

        self.assertIn(
            "Proxima Cen b has no measured radius: it is estimated from the "
            "planet's mass",
            text,
        )
        self.assertIn("Its size, score", text)
        self.assertNotIn("have no measured radius", text)

    def test_modelled_warning_is_plural_for_two_planets(self):
        app = run_page(COMPARE_FILE)
        app.multiselect[0].set_value(["Proxima Cen b", "11 Com b"])
        app.run()
        text = self.warnings(app)

        self.assertIn("Proxima Cen b and 11 Com b have no measured radius", text)
        self.assertIn("they are estimated from each planet's mass", text)
        self.assertIn("Their size, score", text)
        self.assertNotIn("has no measured radius", text)

    def test_no_modelled_warning_for_a_measured_radius_only(self):
        app = run_page(COMPARE_FILE)
        app.multiselect[0].set_value(["Kepler-452 b"])
        app.run()

        self.assertNotIn("Modelled radius", self.warnings(app))
        self.assertEqual(
            ["Earth (reference)", "Kepler-452 b"], list(app.dataframe[0].value["Planet"])
        )

    def test_every_probability_in_the_table_carries_its_status(self):
        for name in labelling.PROBABILITY_DEPENDENCIES:
            with self.subTest(probability=name):
                cells = self.table[name].astype(str)
                self.assertTrue(cells.str.contains("uncertainty_status").all())

    def test_radius_and_provenance_sentences_are_shown(self):
        self.assertIn("radius measured", " ".join(self.table["Radius"]))
        self.assertIn("radius modelled, not measured", " ".join(self.table["Radius"]))
        self.assertTrue(
            self.table["Radius provenance"].str.contains("Measured radius").any()
        )

    def test_provenance_is_a_short_tag_with_the_full_sentence_below(self):
        """Table cells stay readable; the whole sentence has its own line."""
        self.assertEqual("Modelled radius, not measured", self.table["Radius provenance"].iloc[-1])
        self.assertLessEqual(self.table["Radius provenance"].str.len().max(), 40)
        for planet in self.table["Planet"]:
            with self.subTest(planet=planet):
                self.assertIn(f"**{planet}**", self.text)
        self.assertIn("Radius provenance, in full", self.text)
        self.assertIn("mass-radius relation", self.text)


class CompareEarthReferenceTests(unittest.TestCase):
    """Earth's class and tier are computed, and the chart marks the reference."""

    def setUp(self):
        self.frame = explorer_frame()
        self.earth = reference_earth.earth_reference()
        self.rows = plot_data.compare_rows(
            self.frame, ["Kepler-452 b", "Proxima Cen b"], earth=self.earth
        )
        self.figure = plot_data.score_range_figure(self.rows)

    def test_earth_class_and_tier_come_from_the_science_functions(self):
        hz_class = classify_hz(1.0, 5772.0)
        tier = determine_tier(hz_class, determine_size_regime(1.0))

        self.assertEqual("conservative", hz_class)
        self.assertEqual("A", tier)
        self.assertEqual(hz_class, self.earth["hz_class"])
        self.assertEqual(tier, self.earth["tier"])
        self.assertEqual(ui.HZ_LABELS[hz_class], self.rows.iloc[0]["HZ class"])
        self.assertEqual(tier, self.rows.iloc[0]["Tier"])

    def test_page_says_earth_class_and_tier_are_computed_not_typed(self):
        text = all_text(run_page(COMPARE_FILE))

        self.assertIn("classify_hz", text)
        self.assertIn("the tier rules", text)
        self.assertIn(f"{self.earth['tier']!r}", text)

    def test_the_chart_marks_earth_and_the_reference_line(self):
        markers = [trace for trace in self.figure.data if trace.name.startswith("Earth reference")]
        lines = [shape for shape in self.figure.layout.shapes if shape.x0 == 1.0 and shape.x1 == 1.0]

        self.assertEqual(1, len(markers))
        self.assertEqual("diamond", markers[0].marker.symbol)
        self.assertEqual([1.0], list(markers[0].x))
        self.assertEqual(["Earth (reference)"], list(markers[0].y))
        self.assertEqual(1, len(lines))
        self.assertEqual("dash", lines[0].line.dash)
        self.assertIn("Earth = 1.000", [note.text for note in self.figure.layout.annotations])
        # The axis runs past 1.0 so the marker and the line are not clipped.
        self.assertGreater(self.figure.layout.xaxis.range[1], 1.0)


class DiagramsPageTests(unittest.TestCase):
    """Diagrams: both figures, every point, and bands taken from hz_limits."""

    @classmethod
    def setUpClass(cls):
        cls.frame = explorer_frame()
        cls.app = run_page(DIAGRAMS_FILE)
        cls.text = all_text(cls.app)

    def test_diagrams_page_loads_with_two_figures(self):
        self.assertEqual([], [exception.message for exception in self.app.exception])
        self.assertEqual(2, len(self.app.get("plotly_chart")))

    def test_band_edges_match_hz_limits_at_three_temperatures(self):
        temperatures = [2600.0, 5000.0, 7200.0]
        bands = plot_data.hz_band_frame(temperatures)
        limits = hz_limits(temperatures)

        self.assertEqual(len(temperatures), len(bands))
        for position, teff in enumerate(temperatures):
            with self.subTest(teff=teff):
                self.assertAlmostEqual(teff, float(bands.loc[position, "st_teff"]))
                for column, key in (
                    ("optimistic_inner", "optimistic_inner"),
                    ("conservative_inner", "conservative_inner"),
                    ("conservative_outer", "conservative_outer"),
                    ("optimistic_outer", "optimistic_outer"),
                ):
                    self.assertAlmostEqual(
                        float(limits[key][position]),
                        float(bands.loc[position, column]),
                        places=12,
                    )

    def test_bands_refuse_to_extrapolate_outside_the_fitted_range(self):
        for teff in (2599.0, 7201.0):
            with self.subTest(teff=teff):
                with self.assertRaises(ValueError):
                    plot_data.hz_band_frame([teff])

    def test_the_bands_keep_the_expected_order_at_every_temperature(self):
        bands = plot_data.hz_band_frame()

        self.assertTrue(
            (bands["optimistic_inner"] > bands["conservative_inner"]).all()
        )
        self.assertTrue(
            (bands["conservative_inner"] > bands["conservative_outer"]).all()
        )
        self.assertTrue(
            (bands["conservative_outer"] > bands["optimistic_outer"]).all()
        )

    def test_hz_figure_draws_every_plottable_planet(self):
        points, counts = plot_data.hz_points(self.frame)
        figure = plot_data.hz_band_figure(points)
        class_traces = [trace for trace in figure.data if trace.type == "scattergl"]

        self.assertEqual(counts["available"], counts["drawn"] + counts["left_out"])
        self.assertEqual(
            counts["teff_missing"]
            + counts["teff_out_of_range"]
            + counts["flux_invalid"],
            counts["left_out"] + counts["multiple"],
        )
        self.assertEqual(counts["drawn"], sum(len(trace.x) for trace in class_traces))
        self.assertEqual(counts["drawn"], len(points))
        self.assertTrue(
            points["st_teff"].between(plot_data.HZ_TEFF_MIN, plot_data.HZ_TEFF_MAX).all()
        )

    def test_out_of_range_hosts_are_left_out_rather_than_extrapolated(self):
        drawn, counts = plot_data.hz_points(self.frame)
        teff = self.frame["st_teff"]
        outside = self.frame[
            teff.notna() & ((teff < plot_data.HZ_TEFF_MIN) | (teff > plot_data.HZ_TEFF_MAX))
        ]

        self.assertEqual(counts["teff_out_of_range"], len(outside))
        self.assertGreater(len(outside), 0)
        self.assertFalse(drawn["pl_name"].isin(outside["pl_name"]).any())

    def test_group_counts_reproduce_from_the_processed_csv(self):
        """Every caption number is re-derived here from planets_assessed.csv."""
        raw = pd.read_csv(PROCESSED / "planets_assessed.csv")
        _, counts = plot_data.hz_points(raw)

        teff = raw["st_teff"]
        flux = raw["pl_insol"]
        teff_missing = teff.isna()
        out_of_range = teff.notna() & ((teff < 2600) | (teff > 7200))
        flux_invalid = flux.isna() | (flux <= 0)
        groups = (
            teff_missing.astype(int) + out_of_range.astype(int) + flux_invalid.astype(int)
        )

        self.assertEqual(len(raw), counts["available"])
        self.assertEqual(int(teff_missing.sum()), counts["teff_missing"])
        self.assertEqual(int(out_of_range.sum()), counts["teff_out_of_range"])
        self.assertEqual(int(flux_invalid.sum()), counts["flux_invalid"])
        self.assertEqual(int((groups > 1).sum()), counts["multiple"])
        self.assertEqual(int((groups > 0).sum()), counts["left_out"])
        self.assertEqual(len(raw) - int((groups > 0).sum()), counts["drawn"])

        # ...and those are the numbers the caption actually prints.
        for fragment in (
            f"{counts['drawn']:,} points drawn of {counts['available']:,} assessed planets",
            f"{counts['left_out']:,} are left out",
            f"{counts['teff_missing']:,} have no host effective temperature",
            f"{counts['teff_out_of_range']:,} have a host temperature outside",
            f"{counts['flux_invalid']:,} have a missing or non-positive flux",
            f"{counts['multiple']:,} planets fall into more than one",
            f"{counts['left_out']:,} planets actually left out",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.text)


    def test_mass_radius_figure_draws_every_plottable_planet(self):
        points, counts = plot_data.mass_radius_points(self.frame)
        figure = plot_data.mass_radius_figure(points)

        self.assertEqual(counts["available"], counts["drawn"] + counts["missing"] + counts["non_positive"])
        self.assertEqual(counts["drawn"], sum(len(trace.x) for trace in figure.data))
        self.assertEqual(counts["drawn"], len(points))
        self.assertTrue((points["pl_bmasse"] > 0).all())
        self.assertTrue((points["pl_rade"] > 0).all())

    def test_captions_report_drawn_against_available(self):
        _, hz_counts = plot_data.hz_points(self.frame)
        _, mr_counts = plot_data.mass_radius_points(self.frame)

        self.assertIn(
            f"{hz_counts['drawn']:,} points drawn of {hz_counts['available']:,}",
            self.text,
        )
        self.assertIn(
            f"{mr_counts['drawn']:,} points drawn of {mr_counts['available']:,}",
            self.text,
        )
        self.assertIn("not extrapolated", self.text)
        self.assertIn("by construction", self.text)


if __name__ == "__main__":
    unittest.main()
