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
  and stellar insolation, adds the project's documented superhabitability
  adjustments, and reports its simplified rocky-planet and insolation-range
  checks.
- **Limited candidate properties:** estimates a compatibility score from
  orbital period, stellar effective temperature, and distance using a
  `RandomForestRegressor`.
- **Exploration:** compares user inputs with the bundled exoplanet and TESS
  candidate datasets and includes an explanatory glossary.

The two tabs use different methods and their scores are **not directly
comparable**. The tab-one range checks are educational filters, not a physical
climate model or evidence of life.

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
data/planets_for_app.csv       Prepared known-planet comparison data
data/toi_for_app.csv           Prepared TESS candidate comparison data
models/rf_compat_model.pkl     Regressor used by the app
notebooks/Project_v1.ipynb     Exploratory analysis and model training
tests/test_app.py              Application smoke tests
```

## Acknowledgements

Created by [Gurjot Singh](https://github.com/Gurjot-Singh-01) as an educational
and research project. See the linked report and presentation for project
context and methodology.
