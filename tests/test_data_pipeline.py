"""Unit tests for the EXOPHCA data pipeline, validation, cleaning, and golden baseline regression."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from exophca.data.validate import validate_dataframe
from exophca.data.clean import clean_planets_dataset, clean_toi_dataset


def archive_sha256(path: Path) -> str:
    """Return the sha256 of a raw archive without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def golden_source_sha256(source_path: Path) -> str | None:
    """Return the pscomppars sha256 pinned in data/reference/golden_source.json.

    The file holds one entry per July 2026 export. Anything unreadable, absent or
    without a hash returns None, so the caller can leave the regression running.
    """
    if not source_path.is_file():
        return None
    try:
        document = json.loads(source_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    pscd = document.get('pscd')
    if not isinstance(pscd, dict):
        return None
    recorded = pscd.get('sha256')
    return recorded or None


def archive_hash_skip_reason(pscd_path: Path, source_path: Path) -> str | None:
    """Return a skip message when the raw archive is not the export the golden file was pinned to.

    The golden baseline in data/reference/golden_base_score.csv was computed from
    the July 2026 pscomppars export, whose sha256 is pinned in
    data/reference/golden_source.json. That file is a fixture: no script writes
    it. data/processed/PSCD.provenance.json is NOT usable here, because
    scripts/build_dataset.py rewrites it on every rebuild, which would make the
    guard compare a new archive against itself. A different download is a new
    dataset, not a regression, so the comparison cannot be a pass/fail on the
    numbers. When the hashes match the regression runs unchanged; when the
    fixture or its recorded hash is missing, the check stays out of the way.

    A mismatched archive is a *loud* failure by design (see
    ``test_golden_baseline_regression``): the golden baseline is pinned to the
    July 2026 export, so any other archive makes the comparison meaningless and
    silently skipping it would hide a real regression. Set
    ``EXOPHCA_EXPECT_SNAPSHOT=1`` when you deliberately rebuilt from a different
    export and want the skip instead of the failure.
    """
    recorded = golden_source_sha256(source_path)
    if not recorded:
        return None

    actual = archive_sha256(pscd_path)
    if actual == recorded:
        return None
    return (
        f"different archive: data/PSCD.csv has sha256 {actual[:16]}... but the golden "
        f"baseline was pinned to {recorded[:16]}... (the July 2026 export recorded in "
        "data/reference/golden_source.json); the golden regression only applies to "
        "that export"
    )


class DataPipelineTests(unittest.TestCase):

    def test_validate_catches_errors(self):
        """Test validate_dataframe catches missing columns, negative radius, and duplicate names."""
        df = pd.DataFrame({
            'pl_name': ['Planet A', 'Planet A', 'Planet B'],
            'pl_rade': [-1.0, 1.2, 1.0],
            'pl_insol': [1.0, -0.5, 1.0],
            'st_teff': [5000.0, 5000.0, -100.0]
        })
        errors = validate_dataframe(df, ['pl_name', 'pl_rade', 'pl_insol', 'st_teff', 'non_existent'], 'test_frame')
        self.assertTrue(any("Missing required columns" in e for e in errors))
        self.assertTrue(any("planet radius <= 0" in e for e in errors))
        self.assertTrue(any("insolation flux < 0" in e for e in errors))
        self.assertTrue(any("stellar effective temperature <= 0" in e for e in errors))
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_clean_keeps_every_row_and_separates_recovery(self):
        """Test clean_planets_dataset keeps all input rows, never puts recovered flux into pl_insol."""
        raw_df = pd.DataFrame({
            'pl_name': ['P1', 'P2', 'P3'],
            'hostname': ['H1', 'H2', 'H3'],
            'discoverymethod': ['Transit', 'Radial Velocity', 'Radial Velocity'],
            'disc_year': [2020, 2021, 2022],
            'pl_rade': [1.0, np.nan, 2.0],
            'pl_insol': [1.0, np.nan, 1.0],
            'pl_eqt': [255.0, 300.0, 300.0],
            'pl_orbper': [10.0, 20.0, 20.0],
            'pl_orbsmax': [0.1, np.nan, 0.2],
            'pl_bmasse': [1.0, 2.0, 2.0],
            'st_teff': [5778.0, 4000.0, 4000.0],
            'st_rad': [1.0, 0.8, 0.8],
            'st_mass': [1.0, 0.7, 0.7],
            'sy_dist': [10.0, 20.0, 20.0],
            'pl_rade_reflink': ['Link', 'Calculated Value', 'Calculated Value'],
            'pl_bmassprov': ['Mass', 'M-R relationship', 'M-R relationship']
        })
        cleaned = clean_planets_dataset(raw_df)
        self.assertEqual(len(cleaned), len(raw_df))
        # Ensure pl_insol is NOT populated by recovered values where missing
        self.assertTrue(cleaned.loc[1, 'pl_insol'] is np.nan or pd.isna(cleaned.loc[1, 'pl_insol']))
        # Ensure recovered approx is calculated separately
        self.assertFalse(pd.isna(cleaned.loc[1, 'insol_recovered_approx']))
        # Ensure radius provenance categories are correct
        prov_counts = cleaned['radius_provenance'].value_counts()
        self.assertIn('measured', prov_counts)
        self.assertIn('modelled', prov_counts)
        self.assertIn('missing', prov_counts)
        # Verify synthetic row with no radius gets "missing" and scorable is False
        self.assertEqual(cleaned.loc[1, 'radius_provenance'], 'missing')
        self.assertFalse(cleaned.loc[1, 'scorable'])

    def test_golden_baseline_regression(self):
        """Test base_score equals golden_base_score within 1e-9 for scorable non-recovered planets."""
        project_root = Path(__file__).resolve().parent.parent
        golden_path = project_root / 'data' / 'reference' / 'golden_base_score.csv'
        pscd_path = project_root / 'data' / 'PSCD.csv'
        source_path = project_root / 'data' / 'reference' / 'golden_source.json'

        if not pscd_path.is_file():
            self.skipTest(
                "raw archive not present: run scripts/fetch_data.py "
                "(or download the pscomppars table from the NASA Exoplanet "
                "Archive TAP service) to create data/PSCD.csv"
            )
        if not golden_path.is_file():
            self.skipTest(f"reference file not present: {golden_path}")

        mismatch = archive_hash_skip_reason(pscd_path, source_path)
        if mismatch:
            # An archive that is present but is not the pinned July 2026 export
            # is a failure, not a skip: the golden file cannot be compared with
            # it, and a silent skip would hide a regression behind a re-download.
            # EXOPHCA_EXPECT_SNAPSHOT=1 acknowledges a deliberate rebuild from a
            # different export, where skipping is the honest outcome.
            if os.environ.get("EXOPHCA_EXPECT_SNAPSHOT") == "1":
                self.skipTest(f"EXOPHCA_EXPECT_SNAPSHOT=1: {mismatch}")
            self.fail(
                "archive present but not the pinned snapshot; the golden baseline "
                "cannot be checked against it. Restore the July 2026 export, or "
                "set EXOPHCA_EXPECT_SNAPSHOT=1 if you rebuilt from a different "
                f"one on purpose. Details: {mismatch}"
            )

        df_golden = pd.read_csv(golden_path)
        df_raw = pd.read_csv(pscd_path, comment='#', low_memory=False)
        df_cleaned = clean_planets_dataset(df_raw)

        # Merge golden with cleaned on pl_name (golden has base_score named 'base_score')
        merged = pd.merge(df_golden[['pl_name', 'base_score']], df_cleaned[['pl_name', 'base_score', 'insol_recovered_flag']], on='pl_name', suffixes=('_golden', '_clean'))

        compared = 0
        excluded = 0
        for _, row in merged.iterrows():
            # Exclude rows where flux was recovered or missing
            if row.get('insol_recovered_flag', False) or pd.isna(row['base_score_clean']):
                excluded += 1
                continue

            golden_score = row['base_score_golden']
            clean_score = row['base_score_clean']
            self.assertAlmostEqual(golden_score, clean_score, places=9)
            compared += 1

        print(f"\nGolden regression compared: {compared}, excluded (recovered/missing): {excluded}")
        self.assertGreater(compared, 5000)

    def test_toi_processed_only_pc_apc(self):
        """Test processed TOI dataset contains only PC or APC rows."""
        project_root = Path(__file__).resolve().parent.parent
        toi_path = project_root / 'data' / 'TOI_2026.csv'
        if not toi_path.is_file():
            self.skipTest(
                "raw archive not present: run scripts/fetch_data.py "
                "(or download the TOI table from the NASA Exoplanet "
                "Archive TAP service) to create data/TOI_2026.csv"
            )

        df_raw = pd.read_csv(toi_path, comment='#', low_memory=False)
        df_toi = clean_toi_dataset(df_raw)
        self.assertTrue(df_toi['tfopwg_disp'].isin(['PC', 'APC']).all())
        self.assertGreater(len(df_toi), 5000)


class ArchiveHashGuardTests(unittest.TestCase):
    """The golden regression's archive guard, exercised on temporary files only."""

    def _make(self, directory: Path, raw_bytes: bytes, source: dict | None):
        raw = directory / 'PSCD.csv'
        raw.write_bytes(raw_bytes)
        fixture = directory / 'golden_source.json'
        if source is not None:
            fixture.write_text(json.dumps(source), encoding='utf-8')
        return raw, fixture

    def test_matching_hash_runs_the_regression(self):
        """A raw archive whose sha256 matches golden_source.json is not skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = directory / 'PSCD.csv'
            raw.write_bytes(b'# header\npl_name\nPlanet A\n')
            fixture = directory / 'golden_source.json'
            fixture.write_text(
                json.dumps({'pscd': {'sha256': archive_sha256(raw)}}), encoding='utf-8'
            )

            self.assertIsNone(archive_hash_skip_reason(raw, fixture))

    def test_different_hash_skips_with_a_clear_message(self):
        """A newer download skips, naming both hashes and the file to consult."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            july = hashlib.sha256(b'july export').hexdigest()
            raw, fixture = self._make(directory, b'# header\npl_name\nPlanet A\n',
                                      {'pscd': {'sha256': july}})

            reason = archive_hash_skip_reason(raw, fixture)

            self.assertIsNotNone(reason)
            self.assertIn('different archive', reason)
            self.assertIn(archive_sha256(raw)[:16], reason)
            self.assertIn(july[:16], reason)
            self.assertIn('golden_source.json', reason)
            self.assertNotIn('PSCD.provenance.json', reason)

    def test_a_rewritten_provenance_file_cannot_defeat_the_guard(self):
        """The guard reads the fixture, so rewriting the processed provenance changes nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw, fixture = self._make(
                directory, b'# header\npl_name\nPlanet A\n',
                {'pscd': {'sha256': hashlib.sha256(b'july export').hexdigest()}},
            )
            provenance = directory / 'PSCD.provenance.json'
            provenance.write_text(json.dumps({'sha256': archive_sha256(raw)}), encoding='utf-8')

            reason = archive_hash_skip_reason(raw, fixture)

            self.assertIsNotNone(reason)
            self.assertIn('different archive', reason)

    def test_missing_source_does_not_skip(self):
        """Without the fixture there is nothing to compare against, so the test runs."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw, fixture = self._make(directory, b'# header\npl_name\nPlanet A\n', None)

            self.assertIsNone(archive_hash_skip_reason(raw, fixture))

    def test_source_without_a_hash_does_not_skip(self):
        """A fixture that records no pscd sha256 leaves the regression running."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw, fixture = self._make(directory, b'# header\n', {'pscd': {'row_count': 1}})

            self.assertIsNone(archive_hash_skip_reason(raw, fixture))


class GoldenSourceFixtureTests(unittest.TestCase):
    """data/reference/golden_source.json is the guard's only reference, and nothing writes it."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.source_path = self.project_root / 'data' / 'reference' / 'golden_source.json'

    def _document(self) -> dict:
        self.assertTrue(self.source_path.is_file(), f"reference file not present: {self.source_path}")
        return json.loads(self.source_path.read_text(encoding='utf-8'))

    def test_golden_source_pins_the_july_exports(self):
        """The fixture records the sha256, byte length, row count and timestamp of both exports."""
        document = self._document()

        self.assertEqual(
            document['pscd']['sha256'],
            '01aba7f0cf186fed49a85fe6114aa096650729c8428e338ac7f47df900497474',
        )
        self.assertEqual(
            document['toi']['sha256'],
            '743a9acc54c504a723c484a617cb9f9539bb180bd74dd46f386b1fe0868e2bcc',
        )
        self.assertEqual(document['pscd']['byte_length'], 48107049)
        self.assertEqual(document['toi']['byte_length'], 4587939)
        self.assertEqual(document['pscd']['row_count'], 6319)
        self.assertEqual(document['toi']['row_count'], 8064)
        self.assertEqual(document['pscd']['retrieval_timestamp'], 'Sat Jul 11 08:35:38 2026')
        self.assertEqual(document['toi']['retrieval_timestamp'], 'Sun Jul 19 06:26:22 2026')

    def test_golden_source_matches_the_archives_on_disk(self):
        """When the untracked exports are present, their hashes and sizes are the pinned ones."""
        document = self._document()
        checked = 0
        for key, name in (('pscd', 'PSCD.csv'), ('toi', 'TOI_2026.csv')):
            raw = self.project_root / 'data' / name
            if not raw.is_file():
                continue
            self.assertEqual(archive_sha256(raw), document[key]['sha256'], name)
            self.assertEqual(raw.stat().st_size, document[key]['byte_length'], name)
            checked += 1
        if checked == 0:
            self.skipTest("raw archive not present: neither data/PSCD.csv nor data/TOI_2026.csv")

    def test_no_code_under_scripts_or_exophca_writes_the_fixture(self):
        """Grep-equivalent: no source file in scripts/ or exophca/ even names the fixture."""
        mentions = []
        for folder in ('scripts', 'exophca'):
            for path in sorted((self.project_root / folder).rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                try:
                    text = path.read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                if 'golden_source' in text:
                    mentions.append(str(path.relative_to(self.project_root)))

        self.assertEqual(mentions, [], f"scripts/ or exophca/ references golden_source: {mentions}")



if __name__ == '__main__':
    unittest.main()
