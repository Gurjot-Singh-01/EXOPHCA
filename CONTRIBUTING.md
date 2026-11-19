# Contributing to EXOPHCA

Thanks for looking. This repository is an educational scoring and comparison
project, so the most useful contribution is usually a sharper caveat or a bug in
the pipeline, not a new feature.

## Setting up

Python 3.14 or newer is required — `pyproject.toml` declares
`requires-python = ">=3.14"` and an older interpreter fails the install.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -c constraints.txt
```

`constraints.txt` pins the versions the committed model artifact and the
processed tables were produced with. If you change `pyproject.toml`, refresh the
constraints file in the same commit.

## Running the tests

One command runs everything — `pip check`, `compileall` and the full suite — and
exits non-zero if any step fails:

```bash
python scripts/check.py
```

For a quicker loop:

```bash
python -m unittest tests.test_scoring tests.test_habitable_zone -v
```

Two tests need the untracked raw exports and skip when those files are absent;
that is expected on a fresh clone. The golden regression **fails loudly** when
an archive is present but is not the July 2026 export whose sha256 is pinned in
`data/reference/golden_source.json`, because its golden file is pinned to those
rows and a silent skip would hide a regression behind a re-download. Set
`EXOPHCA_EXPECT_SNAPSHOT=1` when you rebuilt from another export deliberately and
want the skip with its explanation instead.

Coverage, measured once with `coverage run -m unittest discover -s tests` and
`coverage report` on 2026-11-19: **85% of the non-test code** (2,541 statements,
372 missed) and **92% including the test modules** (5,241 statements, 418
missed). The lowest file is `exophca/ml/radius_model.py` at 54%, which is the
training module: only `scripts/train_radius_model.py` exercises it end to end, and
that script is deliberately not run here. The figure is recorded, not gated.

**Never lower the test count to make a change pass.** If a test file loses test
methods, it was damaged; restore it from git before going further.

## Regenerating the model

`scripts/train_radius_model.py` writes `models/radius_model.joblib` and
`models/radius_model_meta.json`. The default run reuses the recorded test-split
evaluation:

```bash
python scripts/train_radius_model.py --stage 2
```

To re-evaluate the test split as well, add `--force`:

```bash
python scripts/train_radius_model.py --stage 2 --force
```

That rewrites the artifact, so the committed sha256 changes with it:
`exophca/ml/predict.py` holds `EXPECTED_ARTIFACT_SHA256`, the loader refuses to
`joblib.load` a file that does not match it, and `tests/test_ml_predict.py`
imports the same constant. Update that one value in the same commit, and expect
the README's model numbers to move with it.

**The artifact is a pickle.** Loading one executes code from the file, which is
why the loader checks the digest *before* it unpickles anything and returns a
warning instead of raising. That guard protects this repository's own artifact:
a fork that copies `models/radius_model.joblib` gets the byte check but no reason
to trust it, so re-train instead of trusting a copied file.

**The shadow holdout is not the trainer.** `scripts/holdout_evaluation.py` writes
`models/radius_model_holdout.json` and a working record, and touches nothing the
release depends on. It re-runs the recipe inside the released training rows and
evaluates once on the calibration hosts, which no model was fitted on. It refuses
a second run without `--force` and records a hash of the recipe source, so a
recipe changed after the number was seen is visible. Do not run
`scripts/estimate_radii.py` to "refresh" anything: the shipped table is pinned.

## Refreshing the archive snapshot

The reported results belong to two NASA Exoplanet Archive exports retrieved in
July 2026. To build a new snapshot:

```bash
python scripts/fetch_data.py        # downloads data/PSCD.csv and data/TOI_2026.csv
python scripts/build_dataset.py     # -> data/processed/planets.csv, toi.csv, provenance
python scripts/generate_oracle.py
python scripts/hz_report.py
python scripts/assess.py
python scripts/estimate_radii.py
python scripts/validate_phl.py
python scripts/make_golden_baseline.py
```

A new snapshot changes every count in the README and the numbers in the
Methods page, so it is not a small change: update
`data/reference/golden_source.json` only if you intend the golden regression to
follow the new export, and re-run the suite afterwards — the README guard tests
read the committed CSVs and will tell you which claims went stale.

## Pre-commit

```bash
python -m pip install pre-commit
pre-commit install
```

The hooks run ruff's bug-finding rules plus unused imports and unused locals
(`E9,F63,F7,F82,F401,F841`) and the whitespace and end-of-file checks.
`pre-commit run --all-files` passes on this tree, with `data/reference/` excluded
because `hzcalc.py` is a vendored Python 2 file. CI runs the same rule set and
the same exclusion, so a finding cannot reach `main` through either path.

`.gitattributes` normalises every text file to LF in the working tree. Do not
remove it: `tests/test_ml_predict.py` pins the digest of a processed CSV, and a
line-ending difference between a Windows clone and the Linux runner is exactly
what broke the first CI run.

## Working notes stay local

`docs/progress/` holds the session log and the working records. It is
**gitignored by design** and must never be committed or published: it is process
history, not documentation of the shipped system. The same applies to the report
and the presentation while they describe a superseded methodology.

Do not add a tracked path under `docs/` and do not reference one from the README
— a clone does not contain it, and the README guard test fails on exactly that.

## Licence

MIT. By contributing you agree that your contribution is licensed under the
same terms.
