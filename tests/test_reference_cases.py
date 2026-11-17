"""Unit tests for Phase 5 PHL reference validation and forbidden string checks."""

import unittest
from pathlib import Path
import pandas as pd


class PHLValidationTests(unittest.TestCase):

    def test_phl_comparison_csv_exists_and_populated(self):
        csv_path = Path('data/processed/phl_comparison.csv')
        self.assertTrue(csv_path.exists(), "phl_comparison.csv must exist in data/processed/")
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 1000, "phl_comparison.csv should contain thousands of matched planets")
        expected_cols = ['pl_name', 'hwc_name', 'Radius', 'Flux', 'ESI', 'pl_rade', 'pl_insol', 'base_score', 'hz_class']
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column {col} missing from phl_comparison.csv")

    def test_phase5_validation_report_exists(self):
        doc_path = Path('docs/progress/phase5_validation.md')
        if not doc_path.exists():
            self.skipTest(
                "docs/progress/ is not published: the working records are kept "
                "outside the repository, so this report is absent from a clone"
            )
        content = doc_path.read_text(encoding='utf-8')
        self.assertIn("Phase 5 Validation Report", content)
        self.assertIn("Task 1", content)
        self.assertIn("Task 5", content)

    def test_forbidden_strings_not_in_science_modules(self):
        """Ensure no PHL or reference catalog data structures/files feed core science modules (cross-contamination guardrail)."""
        science_dir = Path('exophca/science')
        self.assertTrue(science_dir.exists())

        forbidden = ['hwc.csv', 'hwc_potentially_habitable.csv', 'phl_comparison.csv', 'Habitable Worlds Catalog']

        for py_file in science_dir.glob('**/*.py'):
            text = py_file.read_text(encoding='utf-8')
            for fstr in forbidden:
                self.assertNotIn(
                    fstr,
                    text,
                    f"Forbidden reference string '{fstr}' found in core science module {py_file}!"
                )


if __name__ == '__main__':
    unittest.main()
