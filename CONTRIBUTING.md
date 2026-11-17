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

Two tests need the untracked raw exports and skip when those files are absent,
and the golden regression skips when `data/PSCD.csv` is not the July 2026 export
whose sha256 is pinned in `data/reference/golden_source.json`. Those skips are
expected on a fresh clone.

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
`tests/test_ml_predict.py` pins the artifact's hash and will fail until
`ARTIFACT_SHA256` is updated to the new value. Update the constant in the same
commit, and expect the README's model numbers to move with it.

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

The hooks run ruff's bug-finding rules plus trailing-whitespace and
end-of-file checks. A `pre-commit run --all-files` currently fails the two
whitespace hooks on the existing tree; that backlog is listed in the hook
configuration and is meant to be cleared in its own commit, not mixed into a
functional change.

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
