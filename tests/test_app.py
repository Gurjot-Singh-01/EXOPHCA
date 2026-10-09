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
        esi = next(metric for metric in app.metric if metric.label == "Base ESI (radius + insolation)")
        adjusted = next(
            metric
            for metric in app.metric
            if metric.label == "Custom-adjusted score (base ESI + project bonus)"
        )
        adjustment = next(
            metric
            for metric in app.metric
            if metric.label == "Custom research-inspired adjustment"
        )
        self.assertEqual("1.000", esi.value)
        self.assertEqual("+0.000", adjustment.value)
        self.assertEqual("1.000", adjusted.value)
        self.assertIn(adjusted.delta, (None, ""))

    def test_tess_candidate_prediction_renders(self):
        app = run_app()
        app.tabs[1].selectbox[0].select(
            "TOI-7347.01 (formula-ranked lead; unconfirmed TESS candidate "
            "that fails the strict habitability gate)"
        )
        app.tabs[1].button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        score = next(
            metric
            for metric in app.metric
            if metric.label == "Model estimate of its training-target score"
        )
        self.assertEqual("0.882", score.value)

    def test_zero_model_input_is_rejected_in_the_ui(self):
        app = run_app()
        app.tabs[1].number_input[0].set_value(0.0)
        app.tabs[1].button[0].click()
        app.run()

        self.assertIn("Orbital period must be greater than zero.", [error.value for error in app.error])
        self.assertFalse(
            any(metric.label == "Model estimate of its training-target score" for metric in app.metric)
        )

    def test_zero_insolation_is_scored_without_a_custom_cap(self):
        app = run_app()
        app.tabs[0].number_input[1].set_value(0.0)
        app.tabs[0].button[0].click()
        app.run()

        self.assertEqual([], [exception.message for exception in app.exception])
        esi = next(metric for metric in app.metric if metric.label == "Base ESI (radius + insolation)")
        self.assertEqual("0.000", esi.value)


if __name__ == "__main__":
    unittest.main()
