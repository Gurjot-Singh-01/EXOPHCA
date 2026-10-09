# EXOPHCA

**Exoplanet Habitability Compatibility Analysis** is an educational project for
exploring how known exoplanets and TESS Objects of Interest compare with Earth.
It combines a formula-based Earth Similarity Index (ESI) score with a
Random Forest regression estimate for candidates with limited measurements.

> **Important:** Neither score is a probability that a planet is habitable or
> hosts life. The model is an exploratory estimate trained on confirmed-planet
> data and is affected by detection and selection biases.

## What it does

- **Measured properties:** calculates a formula-based score from planet radius
  and stellar insolation, displaying the unadjusted ESI separately from the
  project's two custom adjustments.
- **Limited candidate properties:** uses a `RandomForestRegressor` to estimate
  the formula-derived score it was trained on from orbital period, stellar
  effective temperature, and distance.
- **Exploration:** compares user inputs with the bundled exoplanet and TESS
  candidate datasets and includes an explanatory glossary.

The two tabs use different methods and their scores are **not directly
comparable**. The tab-one range checks are educational filters, not a physical
climate model or evidence of life.

## Scoring and model-input checks

The base Earth Similarity Index (ESI) uses planet radius `R` in Earth radii and
stellar insolation `S` in Earth-relative flux:

```text
ESI = sqrt(
    (1 - abs((R - 1) / (R + 1)))^0.57
    * (1 - abs((S - 1) / (S + 1)))^0.7
)
```

For a positive, finite radius and a finite, nonnegative insolation, base ESI is
in `[0, 1]`, with Earth scoring 1.
EXOPHCA shows this base value separately from its custom-adjusted score. The
custom adjustment is `+0.05` for a K-class host and `+0.05` when
`1 < R <= 1.5`; both may apply. The adjusted score is the unscaled sum of base
ESI and those adjustments, in `[0, 1.1)` for valid inputs. It is not capped
or rescaled. The existing size/insolation screen is presented as an
illustrative project filter, not a physical habitability test.

The model predicts the project's formula-derived adjusted score as a training
target. It does **not** estimate the probability of extraterrestrial life or
habitability, and it has not been calibrated as a probability. The saved model
expects exactly these raw numeric columns, in this order, with no scaling or
other preprocessing:

| Feature | Unit | Observed range in the 5,613-row training selection |
| --- | --- | ---: |
| `pl_orbper` | days | 0.1768913–402,000,000 |
| `st_teff` | K | 2,375–40,000 |
| `sy_dist` | pc | 1.30119–3,460.51 |
| `pl_rade` (ESI input) | Earth radii | 0.3098–25 |
| `pl_insol` (ESI input) | Earth flux | 0–636,352.1258 |

These are observed minima and maxima, not scientifically chosen plausibility
limits. The unusually large maximum orbital period is retained as observed in
the training selection; values outside any listed range trigger an
extrapolation warning rather than being silently clipped or rejected. Model
inputs must be present, finite, and positive. Radius must be positive;
insolation may be zero but not negative. Both ESI inputs are also checked
against the observed training selection ranges.

Candidate scores used for comparisons and rankings are recomputed at app
startup using the same loaded model artifact and validated input contract as
the user's prediction. Precomputed scores in the candidate CSV are not used
for the live comparison. The app reports how many bundled candidates fall
outside one or more observed training ranges; those comparison predictions
are extrapolations.

## Model evaluation

The three-feature regressor used by the app was reported at approximately
**R² = 0.811** and **RMSE = 0.0821** on its held-out evaluation split. The
prepared dataset contains 5,613 rows. These metrics measure agreement with the
project's formula-derived training target; they do not establish predictive
accuracy for life or habitability. The project also documents a richer
feature-set result separately; that model is not the one used for the TESS
candidate predictions in this app.

## Run locally

Use Python 3.12 (the clean environment and tests were verified with Python
3.12.6). The app model and preprocessed CSV datasets are included in the
repository; no training run or NASA API credentials are needed to launch the
app.

```bash
git clone https://github.com/Gurjot-Singh-01/EXOPHCA.git
cd EXOPHCA
```

Create and activate a virtual environment using the Python 3.12 launcher:

**Windows PowerShell**

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

With the environment active, install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Start the app from the repository root:

```bash
python -m streamlit run app/EXOPHCA.py
```

Streamlit prints the local URL to open in a browser.

## Deploy

The app is suitable for Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud and select the repository,
   branch, and `app/EXOPHCA.py` as the main file.
3. Keep the repository's `.python-version` and `requirements.txt` so the
   deployment uses the tested runtime and dependencies.

The app loads the compatibility model and its CSV inputs from repository paths
relative to the app file. Keep `models/rf_compat_model.pkl`,
`data/planets_for_app.csv`, and `data/toi_for_app.csv` available in the
deployment.

## Tests

Run the Streamlit smoke tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The notebook in [`notebooks/Project_v1.ipynb`](notebooks/Project_v1.ipynb)
contains the exploratory/training workflow. Jupyter is included in
`requirements.txt`; start it with `python -m jupyter lab` and provide the
original NASA archive input `PSCD.csv` in the notebook's working directory.
That large raw input is not bundled. The notebook's data-fetch/reproduction
workflow still needs cleanup. The app does not depend on running the notebook.

## Data and project materials

- The project uses public data from the
  [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- The [training report](docs/EXOPHCA_Report.docx) and
  [presentation](docs/EXOPHCA_Presentation.pptx) describe the academic work.

## Project layout

```text
app/EXOPHCA.py                 Streamlit application
app/scoring.py                 ESI formula and validated model inference
data/planets_for_app.csv       Prepared known-planet comparison data
data/toi_for_app.csv           Prepared TESS candidate comparison data
models/rf_compat_model.pkl     Regressor used by the app
notebooks/Project_v1.ipynb     Exploratory analysis and model training
tests/test_app.py              Application smoke tests
tests/test_scoring.py          Scoring and inference validation tests
```

## Acknowledgements

Created by [Gurjot Singh](https://github.com/Gurjot-Singh-01) as an educational
and research project. See the linked report and presentation for project
context and methodology.
