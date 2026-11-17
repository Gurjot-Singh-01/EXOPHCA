# EXOPHCA — Exoplanet Habitability Compatibility Analysis

[![CI](https://github.com/Gurjot-Singh-01/EXOPHCA/actions/workflows/ci.yml/badge.svg)](https://github.com/Gurjot-Singh-01/EXOPHCA/actions/workflows/ci.yml)
[![Python 3.14 | 3.15](https://img.shields.io/badge/python-3.14%20%7C%203.15-blue)](https://github.com/Gurjot-Singh-01/EXOPHCA)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Data snapshot: July 2026.** Every committed table was built from two NASA
Exoplanet Archive exports retrieved that month; the app reads only those
committed files.

EXOPHCA is an educational data-science project that scores how closely known
exoplanets and TESS Objects of Interest resemble Earth on two measured
properties: planet radius and stellar insolation. It ships a Streamlit app
whose every number is read from committed CSV files, and the whole pipeline that
produced those files.

## What this is, and what it is not

**It is** a transparent scoring and comparison exercise: an ESI-style
radius-and-flux similarity formula applied to archive measurements, a documented
habitable-zone classification, an offline radius estimate for planets whose
radius the archive had to model, and an honest look at how much uncertainty each
number carries.

**It is not** evidence of habitability, and it is not evidence of life.
A high score means "similar to Earth in radius and the flux it receives". It says
nothing about atmosphere, surface pressure, water, geology, magnetic field,
stellar activity, or whether anything lives there. Similarity is not
habitability; habitable-zone position is not habitability. Every page of the app
states this, and `exophca/science/assessment.py` carries the same wording as the
project's single `DISCLAIMER` constant.

The scores are also **not probabilities**. Only the explicitly labelled
probability columns in `data/processed/planets_assessed.csv` and
`data/processed/toi_assessed.csv` are probabilities, and each one is printed with
the `uncertainty_status` that qualifies it.

## Quick start

The code was written, run and tested on **Python 3.14.0**, and `pyproject.toml`
declares `>=3.14` to match. 3.14 is the minimum supported interpreter: it is
the version the model artifact, the processed tables and this suite were
produced with. Python 3.15 is exercised in CI — the blocking matrix in
`.github/workflows/ci.yml` installs the pins in `constraints.txt` and runs the
full suite on both 3.14 and 3.15, and both jobs must pass. Python 3.12 and older
are not supported and are not tested.

```powershell
git clone https://github.com/Gurjot-Singh-01/EXOPHCA.git
cd EXOPHCA

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e .
```

`pip install -e .` alone is enough to run the app and the tests: it makes
`exophca` importable and installs the runtime dependencies declared in
`pyproject.toml`. `requirements.txt` only adds notebook tooling (Jupyter,
notebook) on top of that same runtime set, so install it
(`python -m pip install -r requirements.txt`) only if you want the notebooks.

### Reproducible install

`pyproject.toml` declares lower bounds, so a plain `pip install -e .` takes
whatever version is newest on the day it runs. `constraints.txt` pins the exact
versions the committed `models/radius_model.joblib` and the processed tables
were produced with, so the environment can be rebuilt as it was:

```powershell
python -m pip install -e . -c constraints.txt
```

`-c` constrains what the resolver is allowed to choose; `-e .` still installs
the project itself, so the two flags are used together rather than instead of
each other. Confirm the result before running anything:

```powershell
python -m pip check
```

which must print `No broken requirements found.`. The pins cover the runtime
stack only: `requirements.txt` adds notebook tooling on top and is not
constrained, so install it separately if you want the notebooks.

Start the app with the project launcher:

```powershell
.\scripts\run_app.ps1
```

The launcher resolves `.venv\Scripts\python.exe` next to the repository, prints
the interpreter, the Streamlit version and the entry point it is about to use,
and stops with a clear message if the virtualenv is missing. Arguments are passed
through, for example `.\scripts\run_app.ps1 --server.port 8611`.

Or run Streamlit directly:

```powershell
python -m streamlit run app/EXOPHCA.py
```

Streamlit prints the local URL. Nothing needs training, no NASA credentials are
needed, and the app does not load the model artifact at runtime: every page reads
`data/processed/*.csv` and `data/reference/solar_system.csv`.

### Running the app on a host

The app is deployable on a Streamlit Community Cloud-style host with no extra
pieces. What the host needs:

- **Entry point:** `app/EXOPHCA.py`.
- **Interpreter:** Python 3.14. The host must offer it and be set to it: the
  package declares `requires-python = ">=3.14"`, so an older default fails the
  *install*, not just the app. The pin lives in `.python-version` and
  `pyproject.toml`; select 3.14 in the host's own settings as well.
- **Install:** `python -m pip install -e . -c constraints.txt` from the
  repository root, the same command CI runs.
- **What it reads:** only the committed CSVs — `data/processed/*.csv`,
  `data/reference/solar_system.csv` and the other files under
  `data/reference/`. The app needs **neither** the raw archive exports **nor**
  any working record, so no secret, no download step and none of the unpublished
  documentation has to exist for it to run.

The Calculator's batch upload is capped at 2 MB before it is parsed, and at
5,000 rows after it, so a hosted instance cannot be made to allocate an
unbounded frame (`server.maxUploadSize` in `.streamlit/config.toml` enforces the
same ceiling at the server).

### The pages

| Group | Page | File |
| --- | --- | --- |
| Explore | Overview | `app/views/overview.py` |
| Explore | Planet Explorer | `app/views/explorer.py` |
| Explore | Planet Detail | `app/views/detail.py` |
| Research | Compare | `app/views/compare.py` |
| Research | Diagrams | `app/views/diagrams.py` |
| Tools | Calculator | `app/views/calculator.py` |
| Tools | TESS Candidates | `app/views/tess_candidates.py` |
| Reference | Methods and limitations | `app/views/methods.py` |
| Reference | Architecture | `app/views/architecture.py` |

## Reproducing the data

Three routes. Pick the one that matches what you want.

### A. Just run the app — nothing to download

Every page reads the committed CSVs under `data/processed/` and
`data/reference/solar_system.csv`. `pip install -e .` and the launcher are the
whole setup: no archive access, no credentials, no training, no raw exports.

### B. Exact reproduction of the reported results

This needs the two raw NASA Exoplanet Archive exports the results were built
from, both taken July 2026. They are deliberately not tracked by git. **The
release asset is not published yet.** When it is, it will be attached to the
tagged release; build the same archive locally with:

```powershell
Compress-Archive -Path data\PSCD.csv, data\TOI_2026.csv `
    -DestinationPath EXOPHCA-raw-exports-july-2026.zip
```

| File | sha256 | Bytes | Rows | Columns | Retrieved |
| --- | --- | ---: | ---: | ---: | --- |
| `data/PSCD.csv` | `01aba7f0cf186fed49a85fe6114aa096650729c8428e338ac7f47df900497474` | 48,107,049 | 6,319 | 325 | Sat Jul 11 08:35:38 2026 |
| `data/TOI_2026.csv` | `743a9acc54c504a723c484a617cb9f9539bb180bd74dd46f386b1fe0868e2bcc` | 4,587,939 | 8,064 | 87 | Sun Jul 19 06:26:22 2026 |

Both hashes and timestamps are recorded in `data/processed/PSCD.provenance.json`
and `data/processed/TOI_2026.provenance.json`, and the same July 2026 hashes,
byte lengths, row counts and retrieval timestamps are pinned in
`data/reference/golden_source.json` — the fixture the golden regression compares
against, because every rebuild rewrites the two processed provenance files.
Verify the files before running anything:

```bash
sha256sum data/PSCD.csv data/TOI_2026.csv
```

```powershell
Get-FileHash data\PSCD.csv, data\TOI_2026.csv -Algorithm SHA256
```

Then run the pipeline from the repository root, in this order:

```bash
python scripts/build_dataset.py         # raw exports -> data/processed/planets.csv, toi.csv, *.provenance.json
python scripts/generate_oracle.py       # -> data/reference/hz_fluxes_oracle.csv
python scripts/hz_report.py             # -> data/processed/planets_hz.csv
python scripts/assess.py                # -> data/processed/planets_assessed.csv, toi_assessed.csv
python scripts/estimate_radii.py        # -> data/processed/radius_estimates.csv
python scripts/validate_phl.py          # -> data/processed/phl_comparison.csv
python scripts/make_golden_baseline.py  # -> data/reference/golden_base_score.csv
```

- `scripts/build_dataset.py` reads both exports with `comment='#'`, validates
  them, writes `planets.csv` (one row per planet, `scorable` flag) and `toi.csv`,
  and writes `PSCD.provenance.json` / `TOI_2026.provenance.json` with the hash,
  byte length and retrieval date.
- `scripts/generate_oracle.py` re-runs the Kopparapu reference implementation in
  `data/reference/hzcalc.py` to produce `hz_fluxes_oracle.csv`; `hz_report.py`
  checks the project's habitable-zone limits against it.
- `scripts/make_golden_baseline.py` reads the legacy `data/planets_for_app.csv`,
  not `data/processed/planets.csv`. That file came from the archived notebook and
  is not read by the app, so the golden baseline cannot be regenerated from the
  current pipeline inputs alone.

**`scripts/train_radius_model.py` is deliberately not in that order.** Retraining
overwrites the committed `models/radius_model.joblib` and
`models/radius_model_meta.json`. Because `test_evaluated` is `true` in the meta
file, a second stage-2 run is refused unless `--force` is passed, and `--force`
re-evaluates the held-out test split — so every test number in this README and
on the Methods page would change. The committed artifact is the one the reported
results came from; leave it alone unless you intend to replace it.

### C. Fresh data from the archive

`python scripts/fetch_data.py` downloads today's `pscomppars` and `toi` tables
from `https://exoplanetarchive.ipac.caltech.edu/TAP/sync` (`select * from
pscomppars` and `select * from toi`) into `data/PSCD.csv` and
`data/TOI_2026.csv`, records the query, URL, UTC timestamp, sha256 and row counts
in `data/fetch_provenance.json`, and validates the required columns before
renaming a download into place. It **never overwrites an existing file unless
`--force` is passed**. Then run the route-B scripts above.

**Fresh data will not reproduce the reported numbers.** The archive keeps
changing: a November 2026 download has 6,464 `pscomppars` rows (703 columns) and
8,151 TOI rows (91 columns) against the July export's 6,319 × 325 and 8,064 × 87.
Row counts, values and every derived figure move with it, and
`data/reference/golden_base_score.csv` is pinned to the July rows, so
`tests/test_data_pipeline.py:test_golden_baseline_regression` skips when the
archive hash does not match the recorded one rather than reporting a false
regression.

## Data sources and credits

- **NASA Exoplanet Archive** — the `pscomppars` table (confirmed planets, read as
  `data/PSCD.csv`) and the `toi` table (TESS Objects of Interest, read as
  `data/TOI_2026.csv`), via the TAP service. This archive is the only source of
  the measurements the scores are computed from. Cite the archive and its
  underlying references when reusing the data:
  <https://exoplanetarchive.ipac.caltech.edu/>
- **PHL Habitable Worlds Catalog** (`data/reference/hwc.csv`,
  `hwc_potentially_habitable.csv`) — used **only** for external validation
  (`scripts/validate_phl.py`) and written to
  `data/processed/phl_comparison.csv`. No PHL value is used in scoring, and
  `tests/test_reference_cases.py` asserts the science modules contain no PHL
  reference strings. Credit: Planetary Habitability Laboratory, University of
  Puerto Rico at Arecibo.
- **Schulze-Makuch et al. (2011)**, "A Two-Tiered Approach to Assessing the
  Habitability of Exoplanets", *Astrobiology* 11(10) — the published Earth
  Similarity Index the base score is modelled on. EXOPHCA uses its radius
  weight and its escape-velocity weight (for flux); it is not the published
  four-parameter index.
- **Kopparapu et al. (2013)**, *ApJ* 765, 131, <https://arxiv.org/abs/1301.6674>
  and **Kopparapu et al. (2014)**, *ApJL* 787, L29,
  <https://arxiv.org/abs/1404.5292> — the habitable-zone flux limits implemented
  in `exophca/science/habitable_zone.py`. The papers are not redistributed here;
  the links above are the reference.

## Third-party material in this repository

The files under `data/reference/` come from other people. Each is listed with its
source and credit; none of them is used in scoring.

| File | Source | Credit and terms |
| --- | --- | --- |
| `data/reference/hwc.csv` | PHL Habitable Worlds Catalog, <https://phl.upr.edu/hwc> | Planetary Habitability Laboratory, University of Puerto Rico at Arecibo. Used **only** for external validation (`scripts/validate_phl.py`). PHL asks to be credited when the catalogue is used. |
| `data/reference/hwc_potentially_habitable.csv` | The same catalogue's potentially-habitable subset | Same credit as above; used only to check the habitable-zone subset comparison. |
| `data/reference/hzcalc.py` | R. Kopparapu (2014), translated to Python by J. Armstrong; original at <https://personal.ems.psu.edu/~jfk4/ruk15/planets/hzcalc.py> | Reference implementation of the Kopparapu et al. flux limits. Used to generate the oracle below and to check `exophca/science/habitable_zone.py`; never called by the app. |
| `data/reference/hz_fluxes_oracle.csv` | Generated from `hzcalc.py` by `scripts/generate_oracle.py` | Derived from the file above, so it carries the same credit. |
| `data/reference/solar_system.csv` | NASA NSSDCA planetary fact sheets, <https://nssdc.gsfc.nasa.gov/planetary/factsheet/> | Public NASA data; each row's `source` column records the page it came from. |
| `data/reference/golden_base_score.csv` | Generated by `scripts/make_golden_baseline.py` from `data/planets_for_app.csv`, itself derived from the NASA Exoplanet Archive export | Same archive credit as Data sources. |
| `data/reference/golden_source.json` | Written once, by hand, from `data/processed/PSCD.provenance.json` and `data/processed/TOI_2026.provenance.json` | Not third-party material. It pins the sha256, byte length, row count and retrieval timestamp of the two July 2026 exports, so the golden regression has a reference that a rebuild cannot overwrite; no script writes it. |

The two Kopparapu papers behind the habitable-zone limits are not redistributed
here. They are <https://arxiv.org/abs/1301.6674> (2013, ApJ 765, 131) and
<https://arxiv.org/abs/1404.5292> (2014, ApJL 787, L29); the authors and
publishers keep their own terms.

Redistribution terms of third-party files have not been independently verified by
the author.

## Results, as recorded in the committed files

Everything in this section is read from the files named beside it. There are no
numbers typed from memory.

### Data pipeline

| Quantity | Value | File |
| --- | ---: | --- |
| Confirmed planets in the archive export | 6,319 | `data/processed/planets.csv` |
| Scorable planets (positive radius and flux) | 5,688 | `data/processed/planets.csv` (`scorable`) |
| Assessed planets | 5,688 | `data/processed/planets_assessed.csv` |
| Assessed TESS candidates | 4,992 | `data/processed/toi_assessed.csv` |
| Planets selected for an offline radius estimate | 1,517 (1,168 with a numeric estimate) | `data/processed/radius_estimates.csv` |
| PHL catalogue rows matched to a planet here by name | 5,566 of 5,599 | `scripts/validate_phl.py` |
| Matched planets that are scorable and used for validation | 5,080 | `data/processed/phl_comparison.csv` |

### Habitable-zone classes (5,688 assessed planets)

| Class | Planets |
| --- | ---: |
| `too_hot` | 4,982 |
| `too_cold` | 363 |
| `conservative` | 178 |
| `optimistic_only` | 91 |
| `undetermined` | 74 |

`conservative` is inside both the conservative and optimistic limits,
`optimistic_only` is inside the optimistic limit only, `undetermined` means the
stellar temperature was missing or outside the fitted range — no extrapolation.

### Evidence tiers (5,688 assessed planets)

| Tier | Planets | Rule (`exophca/science/assessment.py`) |
| --- | ---: | --- |
| A | 18 | inside the conservative habitable zone and rocky-likely |
| B | 52 | inside the conservative zone with an uncertain size regime, or inside only the optimistic zone with radius ≤ 2.5 R⊕ |
| C | 199 | inside a habitable zone but not rocky-likely |
| D | 5,345 | outside both zone limits (too hot or too cold) |
| U | 74 | undetermined (zone undetermined or radius missing) |

Evidence quality splits 3,341 high / 1,175 medium / 1,172 low in the same file.
The tier rules are stated in `exophca/science/assessment.py` and on the Methods
page.

### Agreement with the published PHL ESI

`data/reference/hwc.csv` holds 5,599 PHL rows. Of those, 5,566 matched a
planet here by name (measured by `scripts/validate_phl.py`; the match step
itself is not stored in a committed CSV), and 5,080 of the matched planets are
scorable here, which is what `data/processed/phl_comparison.csv` contains.
From that file:

- Spearman rank correlation with the published PHL ESI: **0.7287**
  (Pearson 0.8307).
- On the 68-planet potentially-habitable subset: Spearman **0.9393**.
- Radius agreement: median relative difference 0.0000, 95th percentile 0.0176.
- Flux agreement: median relative difference 0.0541, 95th percentile 0.6577 over
  the 5,049 planets with a positive finite archive flux.

The flux tail is real: insolation is derived differently in the two catalogues
for some planets, and the comparison says so rather than hiding it.

### Radius model

`models/radius_model.joblib` predicts `log10(pl_rade)` from mass, orbital period
and stellar parameters, for planets whose radius the archive had to model. It is
run offline by `scripts/estimate_radii.py`; the app reads the resulting CSV.
Of the 1,517 planets selected, 1,168 have a numeric estimate and 349 are written
with NaN and an explicit reason (a missing feature column), so the omission is
visible instead of the table silently shrinking. Full numbers and the decision
record are in `models/radius_model_meta.json`.

- Training selection: 1,658 rows (`row_filters`), split by host star into
  1,069 train / 274 calibration / 315 test rows; 258 held-out test hosts.
- Chosen model: random forest (`max_features=1.0`, `min_samples_leaf=3`), after
  a seeded search over the three learners in `cv_learners` (kNN, random forest,
  LightGBM) and comparison against the four baselines in `cv_baselines`
  (median, power law, piecewise power law, ridge). The rule and every score are
  in `models/radius_model_meta.json` (`decision`).
- **Clean — cross-validated only.** 5-fold `GroupKFold` by host: **MAE
  0.1131 dex, R² 0.8154** (best baseline, piecewise power law: MAE 0.1332,
  R² 0.7621). These folds saw only train and calibration rows; nothing below
  influenced them.
- **Selection-influenced.** Held-out test split: **MAE 0.1177 dex, RMSE
  0.1833, R² 0.7931, median absolute percentage error 15.1%** (best baseline:
  MAE 0.1278, R² 0.7603). This split was consulted **twice** — once to choose
  the interval method, once to confirm that the random forest is the
  recommended learner — so it is not an untouched holdout and these are not
  clean generalisation estimates.
- **Selection-influenced.** Prediction intervals at a nominal 90%: split
  conformal, half-width **0.2940 dex** — the method itself was selected on
  these same test rows. Test coverage **0.9111** overall, but only **0.8732**
  in the `mass < 10 M⊕` stratum.

**Interval-method history, stated because the meta file records both steps.**
Conformalised quantile regression (CQR) was recommended on cross-validation —
narrower intervals and a smaller worst-stratum shortfall. On the held-out test
split its worst stratum (`mass < 10 M⊕`) covered only **0.7746** against a
nominal 0.90, while split conformal covered 0.8732, so the final recommendation
in `models/radius_model_meta.json` is **split conformal**, not CQR. Coverage was
allowed to outrank width, and no model was refitted to make that check — but a
**selection** was made on the same rows, which is why the test-split figures
above are labelled selection-influenced.

**Which numbers are clean.** Only the 5-fold `GroupKFold` figures are clean:
MAE 0.1131, R² 0.8154 and the baseline comparison were computed on train and
calibration rows and had no say in any choice. MAE 0.1177, R² 0.7931, coverage
0.9111 and the 0.2940 dex half-width all come from rows that were also used to
choose the interval method and to confirm the model recommendation. This
release has **no untouched holdout**; adding one is the remaining fix.

### The base score

The base score is an **ESI-style radius-flux similarity**, not the published
Earth Similarity Index:

```text
score = sqrt(
    (1 - abs((R - 1) / (R + 1)))^0.57
    * (1 - abs((S - 1) / (S + 1)))^0.70
)
```

with `R` in Earth radii, `S` in Earth-relative flux, and the result clipped to
[0, 1] (`exophca/science/esi.py:calculate_esi`).

It is **not** the published ESI. Schulze-Makuch et al. (2011) use four
parameters, with weights radius 0.57, bulk density 1.07, escape velocity 0.70 and
surface temperature 5.58. EXOPHCA keeps two of those terms: radius with its
published 0.57 weight, and flux with 0.70 — the published escape-velocity weight
**borrowed** for flux, because the paper defines no flux weight at all.

**There are no bonuses.** The score is bounded in [0, 1] and nothing is added to
it. "Host is K-class" and "radius in 1.0–1.5 R⊕" are computed as informational
flags only (`exophca/science/esi.py:classify_flags`) and shown as Yes/No without
touching the number; `app/views/calculator.py` prints "The informational flags
above do not alter or add to the similarity score."

**What the score adds over its own two columns — measured, not asserted.**
Ranked against a plain unweighted proximity to Earth in the same two
properties, `|log10 R| + |log10 S|`, with no exponents and no borrowed weight:

| Quantity | Value |
| --- | ---: |
| Planets with a usable radius and flux | 5,641 |
| Spearman ρ between the two orderings | 0.992 |
| Planets whose tier band changes between the two orderings | 26 (0.46%) |
| Top-band (A) planets shared by both orderings | 17 of 18 |

The bands are the committed tier sizes (A 18, B 52, C 199, D the rest) applied
by rank in each ordering.

**Read that as a null result.** A rank correlation of 0.992 against a
reordering that uses the same two inputs and no published weights means the
score is very close to a monotone transform of radius and flux: it carries
almost no ordering information of its own. Treat it as a **descriptive column**,
not a ranking instrument, and do not rank planets "by habitability
compatibility" with it. The 0.7287 agreement with the published PHL ESI is
partly agreement about one shared input, for the same reason.

## Limitations

- **Similarity is not habitability, and none of this is evidence of life.**
  Radius and insolation are two numbers out of many that would matter.
- **Scores are not probabilities.** The point scores are similarity measures to
  Earth in a two-property space — closer to 1 means closer to Earth — not
  distances. Only the labelled probability columns are probabilities,
  and each carries its `uncertainty_status`.
- **The archive is biased.** Detection methods favour large planets on short
  orbits around bright, nearby stars; the training selection for the radius model
  is almost entirely transits (about 1,608 Transit vs 29 Radial Velocity, per
  `models/radius_model_meta.json`). The population in the files is not the
  population of planets.
- **Modelled radii assume `sin(i) = 1`.** Planets with a minimum mass have a
  radius estimated under that assumption; the true radius may be larger. Those
  rows are labelled and the Detail page says so.
- **The radius model is trained on a narrow sample.** It learns
  `log10(pl_rade)` only from planets whose radius is measured and whose mass is
  measured and positive: `models/radius_model_meta.json` → `row_filters` keeps
  4,742 measured-radius rows, 4,359 with a radius error, 1,695 with a measured
  mass, and 1,658 with all five features. Those rows are almost all transits
  (about 1,608 Transit vs 29 Radial Velocity, same file's `caveats`), so the fit
  describes transiting planets with measured masses, not the population it is
  later applied to.
- **Most planets it is applied to have only a minimum mass.** In
  `data/processed/radius_estimates.csv`, `mass_is_minimum` is True for 870 of the
  1,168 rows with a numeric estimate (1,517 rows in total), so `sin(i) = 1` is
  assumed and the true radius may be larger than the estimate.
- **Almost half the estimates are outside the training range.** `ood_flag` is
  True for 565 of the 1,168 numeric estimates (48%), meaning at least one feature
  lies outside the 1st–99th percentile range of the training features (the
  `caveats` entry in the meta file). Those rows are extrapolations.
- **The interval cannot separate rocky from larger planets.** The nominal 90%
  interval is a fixed multiplicative factor, a split-conformal half-width of
  0.2940 dex (a factor of 10^0.294 ≈ 1.97 up and down) applied to every
  prediction. For 267 of the 1,168 numeric estimates the interval
  (`interval_straddles_1p6`) crosses the 1.6 R⊕ rocky threshold, so for those
  planets the estimate cannot tell a rocky planet from a larger one.
- **The gain over a simple power law is small.** On the held-out test split the
  random forest reaches MAE 0.1177 dex against 0.1278 for the best baseline, a
  piecewise power law (`test_metrics` in the meta file) — a 0.0101 dex
  difference, roughly one thirtieth of the interval half-width.
- **Habitable-zone limits are one-dimensional.** They are flux boundaries for
  Earth-like atmospheres around stars in the fitted temperature range
  (Kopparapu et al.), with no clouds, tides, rotation, magnetic field or stellar
  activity. Stellar temperatures outside the range are reported as
  `undetermined` rather than extrapolated.
- **Boundary cases are flagged, not resolved.** Planets close to a class or size
  boundary are marked `boundary_sensitive`; the flag is visible rather than
  silently rounded.
- **The data is a snapshot.** Exports from July 2026; re-downloading changes the
  numbers (see above).
- **The PHL comparison is a validation exercise.** 5,566 of the 5,599 PHL rows
  matched a planet here by name and 5,080 of those are scorable, so the
  correlation covers a matched subset rather than all 6,319 planets; the 33
  unmatched PHL rows and any mismatch in derived flux limit how strongly it can
  be read.
- **TESS candidates are unconfirmed.** `toi_assessed.csv` rows are Objects of
  Interest, not confirmed planets, and the file says so on every row.

## Repository map

```text
app/
  EXOPHCA.py                 Streamlit entry point, navigation and version guard
  views/                     One module per page (overview, explorer, detail, compare,
                             diagrams, calculator, tess_candidates, methods, architecture)
  assets/architecture.html   Self-contained architecture diagram embedded by the Architecture page
  data_access.py             CSV loading and column checks for every page
  query.py, presets.py, plot_data.py, labelling.py, batch.py,
  calculator_widgets.py, reference_earth.py, methods_content.py, ui.py
exophca/
  science/esi.py             The ESI-style radius-flux score and the informational flags
  science/habitable_zone.py  Kopparapu limits and HZ classification
  science/uncertainty.py     Monte Carlo propagation and probability columns
  science/assessment.py       Tier/evidence rules and the single DISCLAIMER constant
  data/validate.py           Column, range and duplicate checks
  data/clean.py              Raw export -> planets.csv / toi.csv
  data/provenance.py         sha256 and retrieval-date provenance JSON
  ml/radius_model.py         Training, cross-validation, calibration and intervals
  ml/predict.py              Artifact loading, feature contract and prediction
scripts/
  fetch_data.py              Download the two raw archive tables (TAP)
  build_dataset.py           Raw exports -> processed CSVs + provenance
  generate_oracle.py         Kopparapu reference oracle from data/reference/hzcalc.py
  hz_report.py               Habitable-zone report and oracle check
  assess.py                  Uncertainty propagation and tier assessment
  train_radius_model.py      Trains and evaluates models/radius_model.joblib
  estimate_radii.py          Offline radius estimates for modelled-radius planets
  validate_phl.py            PHL comparison and validation report
  make_golden_baseline.py    Golden ESI baseline (legacy input, see above)
  run_app.ps1                Launcher that uses the project virtualenv
data/
  PSCD.csv, TOI_2026.csv     Raw archive exports (not tracked; fetch_data.py)
  processed/                 Every CSV and provenance file the app reads
  reference/                 PHL catalogue, hzcalc oracle, solar system, golden baseline
                             and the pinned source fixture
models/
  radius_model.joblib        Trained artifact (read offline, never by the app)
  radius_model_meta.json     Metrics, splits, decisions and interval history
tests/                       303 unittest tests plus tests/data/batch_example.csv
docs/                        Not published: the report, presentation and the working
                             records are kept outside this repository
notebooks/archive/           The original exploratory notebook (kept for provenance)
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

From the repository root, with the virtualenv active or its interpreter used
directly. The suite is 303 tests and covers the science modules, the data
pipeline, the model artifact, every Streamlit page (via `AppTest`), and cold
starts in fresh processes.

One command runs the whole check — `pip check`, `compileall` and the suite — and
exits non-zero if any step fails:

```bash
python scripts/check.py
```

Two tests need the untracked raw exports and skip with an explanatory message
when those files are absent, and `test_golden_baseline_regression` also skips
(with a `different archive:` message) when `data/PSCD.csv` is not the July 2026
export whose sha256 is pinned in `data/reference/golden_source.json`, because its
golden file is pinned to those rows:

```
raw archive not present: run scripts/fetch_data.py (or download the pscomppars
table from the NASA Exoplanet Archive TAP service) to create data/PSCD.csv
```

On a CI runner — a fresh clone, so no project virtualenv and no working records —
two further tests skip: `test_the_project_virtualenv_interpreter_exists`, because
the clone has no project virtualenv, and `test_phase5_validation_report_exists`,
because the phase-5 working records are kept outside the repository. Both still
run and assert on a developer machine where those files are present.

```bash
# a subset, for a quicker loop
python -m unittest tests.test_scoring tests.test_habitable_zone tests.test_architecture -v
```

## Architecture page

The app's **Architecture** page (`app/views/architecture.py`) embeds
`app/assets/architecture.html`: a self-contained diagram of how the raw exports,
the `exophca` modules, the scripts and the app views connect, with the path of
each arrow verifiable in the repository. Open it in the running app under
Reference → Architecture.

## Screenshots — to be captured

The seven captures below are planned, not taken: the app has not been launched
with a browser to record them, and no placeholder image stands in for one. Each
line names the page and the state the capture should show.

[SCREENSHOT: Overview — headline counts, dataset provenance and the disclaimer]

[SCREENSHOT: Planet Explorer — the filterable planet table with uncertainty labels]

[SCREENSHOT: Planet Detail — one planet's score, criteria, interval and caveats]

[SCREENSHOT: Compare — two planets side by side against Earth]

[SCREENSHOT: Diagrams — mass–radius and habitable-zone figures]

[SCREENSHOT: Calculator — formula calculator with solar-system and archive presets]

[SCREENSHOT: Architecture — the embedded architecture diagram]

## Project materials

The written report and the presentation are produced separately, outside this
repository, and will be added once the project is finished. They are not part
of the published tree, so a clone contains only the code, the data and the
documentation that describes them accurately.

The raw working records for every phase — the validation report, the model
report and the audit trail — are likewise kept outside the repository. They are
process history, not documentation of the shipped system.

## License

MIT. See `LICENSE`. Third-party data and papers keep their own terms; see Data
sources and credits.
