import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "app" / "EXOPHCA.py"


def run_app():
    return AppTest.from_file(str(APP_FILE), default_timeout=120).run()


class AppSmokeTests(unittest.TestCase):
    def test_app_starts_with_all_tabs(self):
        app = run_app()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(4, len(app.tabs))

    def test_earth_matches_its_baseline(self):
        app = run_app()
        app.tabs[0].selectbox[0].select("Earth")
        app.tabs[0].button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        score = next(metric for metric in app.metric if metric.label == "Formula-based compatibility score")
        self.assertEqual("1.000", score.value)
        self.assertIn(score.delta, (None, ""))

    def test_tess_candidate_prediction_renders(self):
        app = run_app()
        app.tabs[1].selectbox[0].select("TOI-7347.01 (top candidate)")
        app.tabs[1].button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        score = next(
            metric
            for metric in app.metric
            if metric.label == "Model-estimated compatibility score"
        )
        self.assertEqual("0.882", score.value)


if __name__ == "__main__":
    unittest.main()
