"""Tests that README.md only makes claims this repository supports.

Every number is read from the committed file it describes, so the README cannot
drift away from the data, the paths it names or the screenshot placeholders it
still needs. Run with the rest of the suite:

    python -m unittest tests.test_readme_claims -v
"""

import re
import unittest
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PROCESSED = ROOT / "data" / "processed"

# Wording that was wrong in an earlier draft and must not come back.
FORBIDDEN_PHRASES = (
    "published two-term",
    "+0.05",
    "formula-derived",
    "three baselines",
    "to be chosen by the author",
    "two adjustments",
)

# Inline code is treated as a repository path when it starts with one of these.
PATH_PREFIXES = (
    "app/",
    "exophca/",
    "scripts/",
    "tests/",
    "data/processed/",
    "data/reference/",
    "models/",
    "docs/",
)

# A path may also be embedded in a command, e.g. `python scripts/fetch_data.py`.
EMBEDDED_PATH = re.compile(
    r"(?:app|exophca|scripts|tests|models|docs)/[\w./*\-]+|data/(?:processed|reference)/[\w./*\-]+"
)

SCREENSHOTS = (
    "Overview",
    "Planet Explorer",
    "Planet Detail",
    "Compare",
    "Diagrams",
    "Calculator",
    "Architecture",
)


def _thousands(number: int) -> str:
    return f"{number:,}"


def _paths_named_in(text: str) -> list[str]:
    """Return the repository-relative paths the README writes in backticks."""
    found = []
    for token in re.findall(r"`([^`\n]+)`", text):
        # "`a.py`, `b.py`" is written as two tokens; an anchor may name a member.
        for piece in token.split(", "):
            piece = piece.split("`")[0].strip()
            if piece.startswith(PATH_PREFIXES):
                path = piece.split(":")[0].rstrip(".,;")
            else:
                embedded = EMBEDDED_PATH.search(piece)
                path = embedded.group(0).rstrip(".,;") if embedded else ""
            if path:
                found.append(path)
    return sorted(set(found))


class ReadmeClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not README.is_file():
            raise unittest.SkipTest(f"README.md not present: {README}")
        cls.text = README.read_text(encoding="utf-8")

    # (a) the wrong wording ---------------------------------------------------
    def test_wrong_wording_is_gone(self):
        for phrase in FORBIDDEN_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.text)

    def test_the_base_score_is_described_as_esi_style_and_bonus_free(self):
        self.assertIn("not the published\nEarth Similarity Index", self.text)
        self.assertIn("There are no bonuses", self.text)
        self.assertIn("borrowed", self.text)

    # (b) the counts ----------------------------------------------------------
    def test_planet_counts_match_planets_csv(self):
        planets = pd.read_csv(PROCESSED / "planets.csv")
        self.assertIn(_thousands(len(planets)), self.text)
        self.assertIn(_thousands(int(planets["scorable"].sum())), self.text)

    def test_habitable_zone_counts_match_planets_hz_csv(self):
        hz = pd.read_csv(PROCESSED / "planets_hz.csv")
        counts = hz["hz_class"].value_counts()
        self.assertEqual(len(counts), 5, f"expected five classes, got {list(counts.index)}")
        for hz_class, count in counts.items():
            with self.subTest(hz_class=hz_class):
                self.assertIn(f"| `{hz_class}` | {_thousands(count)} |", self.text)

    def test_tier_counts_match_planets_assessed_csv(self):
        assessed = pd.read_csv(PROCESSED / "planets_assessed.csv")
        counts = assessed["tier"].value_counts()
        self.assertEqual(set(counts.index), {"A", "B", "C", "D", "U"})
        for tier, count in counts.items():
            with self.subTest(tier=tier):
                self.assertIn(f"| {tier} | {_thousands(count)} |", self.text)

    def test_phl_spearman_matches_phl_comparison_csv(self):
        comparison = pd.read_csv(PROCESSED / "phl_comparison.csv")
        rho = spearmanr(comparison["base_score"], comparison["ESI"]).statistic
        self.assertIn(f"{rho:.4f}", self.text)
        self.assertIn(_thousands(len(comparison)), self.text)

    def test_radius_estimate_counts_match_radius_estimates_csv(self):
        estimates = pd.read_csv(PROCESSED / "radius_estimates.csv")
        numeric = int(estimates["radius_estimate"].notna().sum())
        self.assertIn(_thousands(len(estimates)), self.text)
        self.assertIn(_thousands(numeric), self.text)
        # The two numbers must be reported as a pair, not as one wrong figure.
        self.assertIn(f"{_thousands(len(estimates))} (1,168 with a numeric estimate)", self.text)

    def test_evidence_quality_counts_match_planets_assessed_csv(self):
        assessed = pd.read_csv(PROCESSED / "planets_assessed.csv")
        counts = assessed["evidence_quality"].value_counts()
        for quality in ("high", "medium", "low"):
            with self.subTest(quality=quality):
                self.assertIn(_thousands(int(counts[quality])), self.text)

    # (c) the paths -----------------------------------------------------------
    def test_every_repository_path_in_backticks_exists(self):
        missing = []
        for path in _paths_named_in(self.text):
            target = ROOT / path
            if path.endswith("/"):
                exists = target.is_dir()
            elif "*" in path:
                exists = bool(list(ROOT.glob(path)))
            else:
                exists = target.exists()
            if not exists:
                missing.append(path)
        self.assertEqual([], missing, f"README names paths that do not exist: {missing}")

    def test_the_path_check_can_see_the_important_files(self):
        paths = _paths_named_in(self.text)
        for expected in (
            "app/views/overview.py",
            "exophca/science/esi.py",
            "scripts/fetch_data.py",
            "tests/test_data_pipeline.py",
            "data/processed/planets.csv",
            "data/processed/toi_assessed.csv",
            "data/reference/hwc.csv",
            "models/radius_model_meta.json",
        ):
            with self.subTest(path=expected):
                self.assertIn(expected, paths)

    def test_the_readme_names_no_unpublished_docs_path(self):
        """The docs/ tree is not published, so the README must not point at it.

        A path named here that does not exist in a fresh clone fails
        ``test_every_repository_path_in_backticks_exists`` on CI, where the
        excluded files are absent. This test fails early and says why.
        """
        unpublished = [
            path for path in _paths_named_in(self.text) if path.startswith("docs/")
        ]
        self.assertEqual(
            [],
            unpublished,
            "README names docs/ paths that are not published: "
            f"{unpublished}",
        )

    # (d) the screenshot placeholders ----------------------------------------
    def test_every_screenshot_placeholder_is_still_present(self):
        placeholders = re.findall(r"\[SCREENSHOT: [^\]]+\]", self.text)
        self.assertEqual(len(placeholders), len(SCREENSHOTS))
        joined = "\n".join(placeholders)
        for label in SCREENSHOTS:
            with self.subTest(label=label):
                self.assertIn(label, joined)


if __name__ == "__main__":
    unittest.main()
