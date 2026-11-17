"""Tests for the Architecture page and the diagram it embeds.

The diagram is one self-contained HTML file, so these tests check three things a
screenshot cannot: that the page runs, that the file asks the network for
nothing, and that every repository path it names really exists at that path.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
ENTRY = APP_DIR / "EXOPHCA.py"
ARCHITECTURE_PAGE = APP_DIR / "views" / "architecture.py"
DIAGRAM = APP_DIR / "assets" / "architecture.html"

# A path in this diagram always starts at one of these top-level directories.
PATH_PATTERN = re.compile(r"\b(?:app|exophca|scripts|tests|data|models)/[A-Za-z0-9_./*+%-]+")

CARD_TITLES = (
    "Where the app’s numbers come from",
    "Every probability says how it was made",
    "What this cannot tell you",
)

# The first rule used to claim the app reads CSV files only. The Calculator and
# Diagrams call the science functions in-process, so the rule now says both.
FIRST_RULE = (
    "The data pages read processed CSV files. The Calculator and Diagrams call "
    "the science functions directly. The app never loads the trained model file."
)

# app/views/* imports these modules and calls their functions; each arrow is
# backed by a call in the code (see docs/progress/architecture_v2.md).
SCIENCE_TO_VIEWS = ("esi", "hz", "assess")


def read_diagram() -> str:
    if not DIAGRAM.exists():
        raise AssertionError(f"the diagram file is missing: {DIAGRAM}")
    return DIAGRAM.read_text(encoding="utf-8")


def strip_inert_url_holders(html: str) -> str:
    """Remove the places where a URL is data or attribution, not a request.

    Each removed construct is legitimate: the viewer runtime's JSON data blocks,
    anchor targets, CSS comments (the embedded font's licence) and XML namespace
    declarations. None of them loads anything.
    """
    text = re.sub(r"<script\b.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<a\b[^>]*>", "<a>", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"""xmlns(:\w+)?=("[^"]*"|'[^']*')""", "", text)


class ArchitecturePageTests(unittest.TestCase):
    def test_the_page_runs_without_an_exception(self):
        app = AppTest.from_file(str(ARCHITECTURE_PAGE), default_timeout=180).run()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual("Architecture", app.title[0].value)
        self.assertEqual(1, len(app.get("iframe")))

    def test_the_page_is_listed_under_reference(self):
        source = ENTRY.read_text(encoding="utf-8")
        reference = re.search(r'"Reference":\s*\[(.*?)\n\s*\],', source, re.S)

        self.assertIsNotNone(reference, "the Reference navigation group is missing")
        group = reference.group(1)
        self.assertIn("views/methods.py", group)
        self.assertIn("views/architecture.py", group)
        # Next to "Methods and limitations", not in an earlier group.
        self.assertGreater(source.index('"Reference": ['), source.index('"Tools": ['))

    def test_the_science_modules_are_drawn_into_the_views(self):
        html = read_diagram()

        for source in SCIENCE_TO_VIEWS:
            with self.subTest(source=source):
                self.assertRegex(
                    html, rf'data-edge-from="{source}" data-edge-to="views"'
                )

    def test_ui_is_drawn_into_the_views(self):
        html = read_diagram()

        self.assertRegex(html, r'data-edge-from="ui" data-edge-to="views"')

    def test_the_tests_node_carries_no_test_count(self):
        html = read_diagram()
        group = re.search(r'<g id="node-tests".*?</g>', html, re.S)

        self.assertIsNotNone(group, "the tests node is missing")
        self.assertEqual(
            [],
            re.findall(r"\b\d[\d,]*\s*tests?\b", group.group(0)),
            "the tests node still states a test count",
        )

    def test_the_page_names_the_file_it_embeds(self):
        source = ARCHITECTURE_PAGE.read_text(encoding="utf-8")

        self.assertIn('"assets" / "architecture.html"', source)
        self.assertIn("st.iframe(", source)


class DiagramSelfContainedTests(unittest.TestCase):
    def test_no_external_resource_is_requested(self):
        html = read_diagram()

        self.assertEqual([], re.findall(r"<script[^>]*\bsrc\s*=", html))
        self.assertEqual([], re.findall(r"<link\b", html))
        self.assertNotIn("@import", html)
        self.assertEqual([], re.findall(r"url\(\s*['\"]?https?:", html))
        self.assertEqual(
            [],
            re.findall(
                r"<(?:img|iframe|source|video|audio)\b[^>]*\bsrc\s*=\s*['\"]https?:",
                html,
            ),
        )
        self.assertNotIn("srcset", html)

    def test_every_remaining_url_is_inert(self):
        """No URL survives outside a data block, an anchor, a comment or xmlns."""
        leftovers = re.findall(r"https?://\S+", strip_inert_url_holders(read_diagram()))

        self.assertEqual([], leftovers)

    def test_the_only_link_targets_are_the_credit_and_the_repository(self):
        hrefs = re.findall(r"""<a\b[^>]*\bhref=["']([^"']+)["']""", read_diagram())
        allowed = (
            "https://github.com/tt-a1i/archify",
            "https://github.com/Gurjot-Singh-01/EXOPHCA",
        )

        for href in hrefs:
            self.assertTrue(
                href.startswith(allowed), f"unexpected external link: {href}"
            )


class DiagramContentTests(unittest.TestCase):
    def test_every_path_it_names_exists(self):
        html = read_diagram()
        paths = {match.rstrip(".,)`\"'") for match in PATH_PATTERN.findall(html)}
        named = sorted(path for path in paths if "*" not in path and path.strip())

        self.assertGreaterEqual(len(named), 30, named)
        missing = [path for path in named if not (ROOT / path).exists()]
        self.assertEqual([], missing)

    def test_the_three_design_rules_are_present(self):
        html = read_diagram()

        for title in CARD_TITLES:
            self.assertIn(title, html)

    def test_the_caveat_is_the_project_disclaimer_word_for_word(self):
        from exophca.science.assessment import DISCLAIMER

        self.assertEqual(
            "Similarity and habitable-zone position are not evidence of "
            "habitability or life.",
            DISCLAIMER,
        )
        self.assertIn(DISCLAIMER, read_diagram())

    def test_the_first_rule_states_what_the_app_really_reads_and_calls(self):
        html = read_diagram()

        self.assertIn(FIRST_RULE, html)
        self.assertIn("exophca/ml/predict.py", html)
        self.assertNotIn("Reads CSV only", html)


if __name__ == "__main__":
    unittest.main()
