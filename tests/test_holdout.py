"""Guards on the shipped shadow-holdout record.

``models/radius_model_holdout.json`` is the only record of the one evaluation
this release made on hosts no model was fitted on, and the README publishes
eight of its numbers. Nothing recomputed either side until this file. Every
expected value is read from the committed JSON, metadata or data file it
describes - never typed from memory on both sides - so a hand-edit to the JSON
or to the README fails here.

    python -m unittest tests.test_holdout -v
"""

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "models" / "radius_model_holdout.json"
META = ROOT / "models" / "radius_model_meta.json"
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"
CONSTRAINTS = ROOT / "constraints.txt"
PHASE9_REPORT = ROOT / "docs" / "progress" / "phase9_holdout.md"

TABLE_HEADER = "| Quantity | Released test split | Shadow holdout |"

# The JSON's stratum names, as the README writes them.
STRATUM_DISPLAY = {
    "mass < 10": "`mass < 10` M\u2295",
    "10 <= mass <= 100": "`10\u2013100` M\u2295",
    "mass > 100": "`mass > 100` M\u2295",
}


def _rows_in_table(text: str, header: str) -> list[list[str]]:
    """Return the data cells of the markdown table that starts with ``header``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        return []
    rows = []
    # start + 1 is the alignment row.
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _plain(cell: str) -> str:
    """Drop the emphasis a table cell may carry, keeping its text."""
    return cell.replace("**", "").strip()


def _pins() -> dict[str, str]:
    """The exact versions this repository pins, from both pin files."""
    pins: dict[str, str] = {}
    for path in (CONSTRAINTS, REQUIREMENTS):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s#]*)$", line.strip())
            if match:
                pins.setdefault(match.group(1).lower(), match.group(2))
    return pins


def _all_rows(table: list[dict]) -> dict:
    return next(row for row in table if row["stratum"] == "all rows")


def _worst(table: list[dict]) -> dict:
    return min(
        (row for row in table if row["stratum"] != "all rows"),
        key=lambda row: row["coverage"],
    )


class HoldoutRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (HOLDOUT, META, README):
            if not path.is_file():
                raise unittest.SkipTest(f"not present: {path}")
        cls.holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        cls.meta = json.loads(META.read_text(encoding="utf-8"))
        cls.readme = README.read_text(encoding="utf-8")

    # (1) the README publishes numbers this file recomputes -------------------
    def test_the_readme_holdout_table_matches_the_holdout_json(self):
        rows = _rows_in_table(self.readme, TABLE_HEADER)
        self.assertTrue(rows, f"README is missing the table headed {TABLE_HEADER!r}")

        holdout = self.holdout
        split_rows = self.meta["split_sizes"]["rows"]
        split_hosts = self.meta["split_sizes"]["hosts"]
        released_point = self.meta["test_metrics"]["chosen_model"]
        released_baseline = self.meta["test_metrics"]["best_baseline_piecewise_power_law"]
        holdout_point = holdout["holdout_point_metrics"]
        holdout_baseline = holdout["holdout_baseline_metrics"]
        released_sc = _all_rows(self.meta["test_interval_coverage"]["split_conformal"])
        holdout_sc = _all_rows(holdout["holdout_interval_tables"]["split_conformal"])
        released_worst = _worst(self.meta["test_interval_coverage"]["split_conformal"])
        holdout_worst = _worst(holdout["holdout_interval_tables"]["split_conformal"])

        expected = [
            (
                f"{split_rows['test']} / {split_hosts['test']}",
                f"{holdout['holdout']['rows']} / {holdout['holdout']['hosts']}",
            ),
            (f"{released_point['MAE']:.4f}", f"{holdout_point['MAE']:.4f}"),
            (f"{released_point['RMSE']:.4f}", f"{holdout_point['RMSE']:.4f}"),
            (f"{released_point['R2']:.4f}", f"{holdout_point['R2']:.4f}"),
            (
                f"{released_point['MedAPE']:.1f}%",
                f"{holdout_point['MedAPE']:.1f}%",
            ),
            (
                f"{released_baseline['MAE']:.4f}",
                f"{holdout_baseline['MAE']:.4f}",
            ),
            (
                f"{self.meta['intervals']['split_conformal_half_width_dex']:.4f}",
                f"{holdout['calibration']['split_conformal_half_width_dex']:.4f}",
            ),
            (f"{released_sc['coverage']:.4f}", f"{holdout_sc['coverage']:.4f}"),
            (
                f"{released_worst['coverage']:.4f} "
                + f"({STRATUM_DISPLAY[released_worst['stratum']]})",
                f"{holdout_worst['coverage']:.4f} "
                + f"({STRATUM_DISPLAY[holdout_worst['stratum']]})",
            ),
            (
                f"{self.meta['test_ood_rows']} of {split_rows['test']}",
                f"{holdout['holdout']['out_of_distribution_rows']} of "
                + f"{holdout['holdout']['rows']}",
            ),
        ]

        self.assertEqual(
            len(rows),
            len(expected),
            f"the README table has {len(rows)} rows, the holdout record implies "
            f"{len(expected)}",
        )
        for cells, (released, shadow) in zip(rows, expected):
            label = cells[0]
            with self.subTest(row=label):
                self.assertEqual(
                    _plain(cells[1]),
                    released,
                    f"README row {label!r} disagrees with radius_model_meta.json",
                )
                self.assertEqual(
                    _plain(cells[2]),
                    shadow,
                    f"README row {label!r} disagrees with radius_model_holdout.json",
                )

        # The prose delta must be the difference of the two published MAEs. The
        # README publishes four decimals, so the delta a reader can reproduce is
        # the difference of those, not of the unrounded values.
        delta = round(holdout_point["MAE"], 4) - round(released_point["MAE"], 4)
        self.assertIn(f"MAE is {delta:.4f} dex higher", self.readme)

        # The CQR figures in the same paragraph come from the same JSON.
        cqr_overall = _all_rows(holdout["holdout_interval_tables"]["cqr"])["coverage"]
        cqr_worst = _worst(holdout["holdout_interval_tables"]["cqr"])["coverage"]
        self.assertIn(f"covers {cqr_overall * 100:.1f}%", self.readme)
        self.assertIn(f"worst stratum {cqr_worst * 100:.1f}%", self.readme)

    # (2) the numbers came from the published recipe source -------------------
    def test_the_recipe_hash_recomputes_from_the_recorded_files(self):
        names = self.holdout["recipe_files"]
        self.assertTrue(names, "the holdout record names no recipe files")
        digest = hashlib.sha256()
        for name in names:
            path = ROOT / name
            self.assertTrue(path.is_file(), f"recipe file is missing: {name}")
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        self.assertEqual(
            digest.hexdigest(),
            self.holdout["recipe_hash"],
            "the recipe has changed since the holdout numbers were produced; "
            "re-run scripts/holdout_evaluation.py --force if the change is "
            "intended, or revert it if it is not",
        )

    # (3) the holdout really is the released calibration split ----------------
    def test_the_holdout_is_the_released_calibration_split(self):
        from exophca.ml import radius_model as rm

        recorded = self.meta["split_sizes"]
        holdout = self.holdout["holdout"]
        pool = self.holdout["pool"]

        self.assertEqual(holdout["rows"], recorded["rows"]["calibration"])
        self.assertEqual(holdout["hosts"], recorded["hosts"]["calibration"])
        self.assertEqual(pool["rows"], recorded["rows"]["train"])
        self.assertEqual(pool["hosts"], recorded["hosts"]["train"])
        self.assertEqual(pool["fit_rows"] + pool["calibration_rows"], pool["rows"])

        data_file = str(self.meta["data_file"]).replace("\\", "/")
        raw = rm.load_planets(ROOT / data_file)
        selected, _steps = rm.filter_training_rows(raw)
        split, summary = rm.split_by_host(selected, seed=self.meta["random_seed"])

        calibration_hosts = set(split.loc[split["split"] == "calibration", rm.GROUP_COLUMN])
        pool_hosts = set(split.loc[split["split"] == "train", rm.GROUP_COLUMN])
        self.assertEqual(len(calibration_hosts), holdout["hosts"])
        self.assertEqual(
            rm.host_list_hash(calibration_hosts),
            holdout["host_hash"],
            "holdout.host_hash is not the hash of the calibration host list the "
            "released split produces from the committed data",
        )
        self.assertEqual(summary.test_host_hash, self.meta["test_host_hash"])
        self.assertEqual(
            rm.host_list_hash(pool_hosts),
            pool["pool_host_hash"],
            "pool.pool_host_hash is not the hash of the pool's own host list",
        )
        self.assertNotEqual(
            pool["pool_host_hash"],
            hashlib.sha256(b"").hexdigest(),
            "pool.pool_host_hash is the hash of the empty string; the pool split "
            "has no test hosts, so this field must hash the pool's own hosts",
        )

    # (4) who produced it and from which versions -----------------------------
    def test_the_record_carries_its_own_provenance(self):
        holdout = self.holdout
        self.assertIs(holdout["holdout_evaluated"], True)
        self.assertIsInstance(holdout["shadow_seed"], int)
        self.assertIsInstance(holdout["shadow_calibration_fraction"], float)
        self.assertRegex(holdout["holdout_evaluated_on"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertTrue(holdout["caveat"].strip())

        self.assertEqual(
            holdout["released_test_host_hash"],
            holdout["reproduced_test_host_hash"],
            "the holdout did not reproduce the released test-host hash",
        )
        self.assertEqual(holdout["released_test_host_hash"], self.meta["test_host_hash"])

        pins = _pins()
        self.assertTrue(pins, "no version pins were found")
        for name, version in holdout["library_versions"].items():
            if name == "python":
                continue
            with self.subTest(package=name):
                self.assertIn(name, pins, f"{name} is not pinned by this repository")
                self.assertEqual(
                    version,
                    pins[name],
                    f"the holdout ran {name} {version}, the repository pins {pins[name]}",
                )
        self.assertEqual(holdout["library_versions"]["python"], holdout["python"])
        self.assertTrue(
            sys.version.startswith(holdout["python"].split(".")[0] + "."),
            f"the suite runs Python {sys.version.split()[0]}, the record "
            f"{holdout['python']}",
        )

        if PHASE9_REPORT.is_file():
            text = PHASE9_REPORT.read_text(encoding="utf-8")
            self.assertIn(f"shadow seed: {holdout['shadow_seed']}", text)
            self.assertIn(holdout["recipe_hash"], text)


if __name__ == "__main__":
    unittest.main()
