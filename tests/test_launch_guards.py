"""Launch guards: the version check, the launcher, and the tolerant embed.

The Architecture page once raised ``TypeError: IframeMixin.iframe() got an
unexpected keyword argument 'alt'`` when the app was started with the
machine-wide Python, whose Streamlit is older than the project's. These tests
pin both halves of the fix: the start-up guard reads its minimum version from
``pyproject.toml`` and stops with a sentence, and the page asks
``inspect.signature`` what the installed ``st.iframe`` accepts instead of
assuming the newest API.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

import streamlit
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
ENTRY = APP_DIR / "EXOPHCA.py"
ARCHITECTURE_PAGE = APP_DIR / "views" / "architecture.py"
DIAGRAM = APP_DIR / "assets" / "architecture.html"
LAUNCHER = ROOT / "scripts" / "run_app.ps1"
PYPROJECT = ROOT / "pyproject.toml"

# Read from the page rather than repeated here: the embed tests check that the
# height the page passes is the height the page declares. A literal 900 lived
# here once and went stale the moment the diagram outgrew its frame.
DIAGRAM_HEIGHT = int(
    re.search(
        r"^DIAGRAM_HEIGHT = (\d+)$",
        ARCHITECTURE_PAGE.read_text(encoding="utf-8"),
        re.MULTILINE,
    ).group(1)
)


def run_page(path: Path) -> AppTest:
    return AppTest.from_file(str(path), default_timeout=180).run()


def messages(app: AppTest, name: str) -> list[str]:
    return [element.value for element in getattr(app, name)]


class LaunchGuardTests(unittest.TestCase):
    def test_the_installed_streamlit_is_not_blocked(self):
        app = run_page(ENTRY)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual([], messages(app, "error"))
        self.assertEqual("Overview", app.title[0].value)

    def test_an_older_streamlit_stops_with_a_clear_message(self):
        with mock.patch.object(streamlit, "__version__", "1.59.2"):
            app = run_page(ENTRY)

        self.assertEqual([], [exception.message for exception in app.exception])
        errors = " ".join(messages(app, "error"))
        self.assertIn("This app needs Streamlit", errors)
        self.assertIn("1.59.2", errors)
        self.assertIn(
            r".\.venv\Scripts\python.exe -m streamlit run app/EXOPHCA.py", errors
        )
        # Nothing past the guard ran: no page title, no navigation.
        self.assertEqual([], [title.value for title in app.title])

    def test_the_minimum_version_is_read_from_pyproject_not_copied(self):
        source = ENTRY.read_text(encoding="utf-8")
        declared = PYPROJECT.read_text(encoding="utf-8")

        self.assertIn("streamlit>=1.66.0", declared)
        self.assertIn("tomllib", source)
        self.assertNotIn("1.66.0", source)


class LauncherTests(unittest.TestCase):
    def test_the_launcher_uses_the_project_virtualenv(self):
        self.assertTrue(LAUNCHER.exists(), f"no launcher at {LAUNCHER}")
        source = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(r".venv\Scripts\python.exe", source)
        self.assertIn("$PSScriptRoot", source)
        self.assertIn("Split-Path -Parent $PSScriptRoot", source)
        self.assertIn("streamlit.__version__", source)
        self.assertIn("-m streamlit run", source)


class IframeToleranceTests(unittest.TestCase):
    """The page must render on a Streamlit whose st.iframe is older or gone."""

    def test_a_streamlit_without_the_alt_argument_still_renders(self):
        calls = []

        def iframe_without_alt(src, *, width="stretch", height="content", tab_index=None):
            calls.append({"src": str(src), "height": height, "tab_index": tab_index})

        with mock.patch.object(streamlit, "iframe", iframe_without_alt):
            app = run_page(ARCHITECTURE_PAGE)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual([], messages(app, "error"))
        self.assertEqual(1, len(calls))
        self.assertEqual(str(DIAGRAM), calls[0]["src"])
        self.assertEqual(DIAGRAM_HEIGHT, calls[0]["height"])
        self.assertEqual("Architecture", app.title[0].value)
        self.assertIn("#### How it was made", messages(app, "markdown"))

    def test_without_st_iframe_the_components_fallback_renders(self):
        calls = []

        def fake_components(body, *, height=None, scrolling=True):
            calls.append({"characters": len(body), "height": height, "scrolling": scrolling})

        with mock.patch.object(streamlit, "iframe", None), mock.patch(
            "streamlit.components.v1.html", fake_components
        ):
            app = run_page(ARCHITECTURE_PAGE)

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual([], messages(app, "error"))
        self.assertEqual(1, len(calls))
        self.assertEqual(len(DIAGRAM.read_text(encoding="utf-8")), calls[0]["characters"])
        self.assertEqual(DIAGRAM_HEIGHT, calls[0]["height"])
        self.assertTrue(calls[0]["scrolling"])
        self.assertEqual("Architecture", app.title[0].value)

    def test_the_licence_note_says_apache_2(self):
        source = ARCHITECTURE_PAGE.read_text(encoding="utf-8")

        self.assertIn("**Apache-2.0** licence", source)
        self.assertIn("MIT licence file shipped alongside", source)


if __name__ == "__main__":
    unittest.main()
