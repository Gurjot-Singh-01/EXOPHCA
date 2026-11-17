"""Cold start: every page runs in a fresh interpreter.

The in-process ``AppTest`` suite shares ``sys.modules`` with the test run, so a
page that calls a function which the saved module does not define can still pass
there when an earlier version of that module was imported first. That is exactly
how the Calculator shipped with ``presets.preset_groups`` while the running
server held a ``presets`` module without it.

Every case here starts a **new Python process** (the project virtualenv's
interpreter, never the test process) and runs the page through ``AppTest`` as the
Streamlit server does, so a renamed or missing function in a view fails the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
ENTRY = APP_DIR / "EXOPHCA.py"
VIEWS = sorted((APP_DIR / "views").glob("*.py"))

ARCHITECTURE = APP_DIR / "views" / "architecture.py"
CALCULATOR = APP_DIR / "views" / "calculator.py"
DETAIL = APP_DIR / "views" / "detail.py"
# The project virtualenv. On a developer machine this is the interpreter the suite
# must run under, and the child processes must use it too. On a CI runner there is
# no .venv at all, so the setup-python interpreter is used for both.
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

# Runs inside the child interpreter. argv: page path, JSON list of steps.
RUNNER = r"""
import json
import sys

from streamlit.testing.v1 import AppTest

target, raw_steps = sys.argv[1], sys.argv[2]
steps = json.loads(raw_steps)

app = AppTest.from_file(target, default_timeout=180).run()
for kind, value in steps:
    if kind == "radio":
        app.radio[0].set_value(value)
    elif kind == "select":
        app.selectbox[0].select(value)
    elif kind == "text":
        app.text_input[0].set_value(value)
    else:
        raise SystemExit("unknown step: %s" % kind)
    app.run()

print(json.dumps({
    "exceptions": [exception.message for exception in app.exception],
    "title": app.title[0].value if len(app.title) else None,
    "metrics": {metric.label: metric.value for metric in app.metric},
    "markdown": [item.value for item in app.markdown],
    "captions": [caption.value for caption in app.caption],
    "selectbox_options": [len(widget.options) for widget in app.selectbox],
    "selectbox_values": [widget.value for widget in app.selectbox],
    "warnings": [warning.value for warning in app.warning],
}))
"""


def run_page_in_fresh_process(path: Path, steps=()):
    """Run one page in a brand-new interpreter; return (completed, payload)."""
    completed = subprocess.run(
        [str(PYTHON), "-c", RUNNER, str(path), json.dumps(list(steps))],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=420,
    )
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            payload = json.loads(line)
            break
    return completed, payload


class ColdStartInterpreterTests(unittest.TestCase):
    """The child processes must use the same interpreter as the test process.

    On a developer machine that interpreter is the project virtualenv, and the
    suite must be run with it. On a CI runner there is no project virtualenv, so
    both the suite and the children use the setup-python interpreter.
    """

    def test_the_project_virtualenv_interpreter_exists(self):
        if not VENV_PYTHON.exists():
            self.skipTest("no project virtualenv: running on a CI runner")

        self.assertTrue(
            VENV_PYTHON.exists(), f"no virtualenv interpreter at {VENV_PYTHON}"
        )

    def test_the_child_interpreter_is_the_virtualenv_one(self):
        completed = subprocess.run(
            [str(PYTHON), "-c", "import sys; print(sys.executable)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(str(PYTHON), completed.stdout.strip())
        self.assertEqual(PYTHON.resolve(), Path(sys.executable).resolve())


class ColdStartTests(unittest.TestCase):
    """Every page, and both Calculator preset groups, from a fresh process."""

    def assert_clean(self, path: Path, steps=()):
        completed, payload = run_page_in_fresh_process(path, steps)
        if completed.returncode != 0 or payload is None:
            self.fail(
                f"{path.name} crashed in a fresh interpreter "
                f"(exit {completed.returncode})\n"
                f"stdout:\n{completed.stdout[-2000:]}\n"
                f"stderr:\n{completed.stderr[-2000:]}"
            )
        if payload["exceptions"]:
            self.fail(
                f"{path.name} raised in a fresh interpreter: "
                f"{payload['exceptions']}"
            )
        return payload

    def test_every_view_starts_without_an_exception(self):
        self.assertEqual(9, len(VIEWS), [view.name for view in VIEWS])
        for view in VIEWS:
            with self.subTest(view=view.name):
                self.assert_clean(view)

    def test_the_entry_point_starts_on_the_overview(self):
        payload = self.assert_clean(ENTRY)

        self.assertEqual("Overview", payload["title"])

    def test_detail_search_for_kepler_cold(self):
        """Typing "Kepler" into the Detail search must not raise in a fresh process.

        The audit found the selectbox index taken from the full 5,688-name list,
        so this search raised StreamlitValueOutOfRangeError ("index 3778 is
        outside [0, 2742]") in a cold interpreter while every AppTest still
        passed, because no test typed into the box.
        """
        payload = self.assert_clean(DETAIL, [("text", "Kepler")])

        self.assertEqual("Planet Detail", payload["title"])
        self.assertEqual(1, len(payload["selectbox_options"]))
        self.assertGreaterEqual(payload["selectbox_options"][0], 2743)
        self.assertEqual("Kepler-452 b", payload["selectbox_values"][0])
        self.assertIn("**Boundary sensitive.**", " ".join(payload["warnings"]))

    def test_architecture_page_starts_cold(self):
        """The diagram page reads its HTML file and embeds it, from a new process."""
        payload = self.assert_clean(ARCHITECTURE)

        self.assertEqual("Architecture", payload["title"])
        self.assertTrue(
            any("How it was made" in text for text in payload["markdown"]),
            payload["markdown"],
        )

    def test_calculator_scores_a_solar_system_preset_cold(self):
        payload = self.assert_clean(CALCULATOR, [("select", "Jupiter")])

        self.assertEqual("Calculator", payload["title"])
        self.assertIn("ESI-style radius-flux similarity", payload["metrics"])
        # The class is a sentence, so it is wrapping markdown rather than a metric
        # value that the metric font would clip with an ellipsis.
        self.assertNotIn("Habitable-zone class", payload["metrics"])
        self.assertTrue(
            any("Habitable-zone class" in text for text in payload["markdown"]),
            payload["markdown"],
        )
        self.assertIn("**too cold (outside the early-Mars limit)**", payload["markdown"])

    def test_calculator_scores_an_archive_preset_cold(self):
        payload = self.assert_clean(
            CALCULATOR, [("radio", "Archive planets"), ("select", "Proxima Cen b")]
        )

        self.assertIn("ESI-style radius-flux similarity", payload["metrics"])

    def test_a_missing_function_fails_this_test(self):
        """The runner reports the child's exception instead of hiding it.

        The probe page lives in a temporary directory, never in ``app/views``,
        and calls a function that ``presets`` does not define.
        """
        with tempfile.TemporaryDirectory() as scratch:
            probe = Path(scratch) / "probe_page.py"
            probe.write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(APP_DIR)!r})\n"
                "import presets\n\n"
                "presets.this_function_does_not_exist()\n",
                encoding="utf-8",
            )
            completed, payload = run_page_in_fresh_process(probe)

        self.assertEqual(0, completed.returncode, completed.stderr[-2000:])
        self.assertIsNotNone(payload)
        self.assertTrue(payload["exceptions"], "a missing function must surface")
        self.assertIn("this_function_does_not_exist", payload["exceptions"][0])


if __name__ == "__main__":
    unittest.main()
