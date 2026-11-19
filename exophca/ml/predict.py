"""Radius predictions from the saved artifact, for the data pipeline only.

This module is the **only** place that reads ``models/radius_model.joblib``.
``scripts/estimate_radii.py`` uses it to write
``data/processed/radius_estimates.csv``; the Streamlit app reads that CSV and
never imports this module, so the app never loads a pickled model.

The artifact is a dict of a point model plus the interval that was calibrated
for it. Two interval methods exist and the artifact names the one to use:
``split_conformal`` (a symmetric half-width in dex) and ``cqr`` (two quantile
models plus a conformal quantile in dex). Predictions are made in log10 radius
and converted to Earth radii at the end, so the interval is the interval in
``log10 R`` and its width is reported in dex.

Rows that cannot be predicted - a missing or non-positive feature - get NaN
estimates and a reason, never a guess.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# The committed artifact is a pickle, and unpickling executes code from the file,
# so a replaced artifact is a supply-chain surface. Its digest is recorded here,
# next to the loader that checks it, and imported by tests/test_ml_predict.py so
# there is one source of truth rather than two copies that can drift apart.
#
# The guard protects this repository's own artifact. A fork that copies the file
# gets the same check, but the check does not make a copied artifact trustworthy:
# retrain instead.
EXPECTED_ARTIFACT_SHA256 = (
    "6b3d62e901a0598b03bb637b86b53ce003d7637857b975a0ade9e767849cc2de"
)

ROCKY_THRESHOLD_REARTH = 1.6

# The artifact stores the five model features; each is built from one archive
# column. The two log features are logged raw values, the rest are passed as is.
FEATURE_SOURCES = {
    "log10_pl_bmasse": "pl_bmasse",
    "log10_pl_orbper": "pl_orbper",
    "st_teff": "st_teff",
    "st_mass": "st_mass",
    "st_rad": "st_rad",
}

# A minimum mass (as opposed to a measured one) means the radius was estimated
# assuming sin(i) = 1; the true mass, and so the true radius, may be larger.
MINIMUM_MASS_LABELS = ("Msini", "Msin(i)/sin(i)")

# Libraries whose version can change a prediction or an interval.
CHECKED_LIBRARIES = ("scikit-learn", "lightgbm", "numpy")

OUTPUT_COLUMNS = [
    "pl_name",
    "radius_estimate",
    "radius_lower",
    "radius_upper",
    "mass_is_minimum",
    "ood_flag",
    "interval_straddles_1p6",
    "reason",
]


def _installed_version(distribution: str) -> str:
    """Installed version of a distribution, or ``not installed``."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def version_warnings(stored: dict | None) -> list[str]:
    """Compare the artifact's recorded library versions with the installed ones.

    A difference is reported, not raised: the caller decides whether to use the
    artifact. scikit-learn and lightgbm are the libraries that can change a
    prediction; numpy is included because it backs both.
    """
    stored = stored or {}
    warnings: list[str] = []
    for distribution in CHECKED_LIBRARIES:
        expected = stored.get(distribution)
        actual = _installed_version(distribution)
        if expected is None:
            warnings.append(
                f"the artifact records no {distribution} version; installed {actual}"
            )
            continue
        if expected != actual:
            warnings.append(
                f"{distribution} {actual} differs from the {expected} the artifact "
                "was trained with; predictions and intervals may differ from the "
                "reported metrics"
            )
    return warnings


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, read in chunks so a large artifact is not held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radius_artifact(
    path: str | Path,
    expected_sha256: str | None = EXPECTED_ARTIFACT_SHA256,
) -> tuple[dict | None, list[str]]:
    """Load the artifact and return ``(artifact, warnings)``.

    Never raises for a version mismatch, and never raises when a library the
    artifact needs is missing: it returns ``(None, warnings)`` and the caller
    decides what to do. The file itself is expected to exist - the pipeline that
    calls this is the one that trained it.

    Before anything is unpickled, the file's sha256 is compared with
    ``expected_sha256`` (the committed artifact's digest by default). A mismatch
    is refused with a warning rather than loaded, because loading a pickle runs
    its code. Pass ``expected_sha256=None`` only to read a file whose provenance
    you already trust, such as a temporary artifact written by a test.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return None, [f"no radius artifact at {artifact_path}"]

    if expected_sha256 is not None:
        actual = file_sha256(artifact_path)
        if actual != expected_sha256:
            return None, [
                f"refusing to load {artifact_path}: its sha256 is {actual}, but "
                f"{expected_sha256} was expected. The artifact is a pickle, so "
                "loading a replaced file would execute its code; restore the "
                "committed artifact or retrain the model instead of loading it."
            ]

    try:
        artifact = joblib.load(artifact_path)
    except (ImportError, ModuleNotFoundError) as error:
        return None, [
            f"the artifact at {artifact_path} could not be loaded because a library "
            f"it needs is missing ({error}); install the versions recorded in "
            "models/radius_model_meta.json"
        ]

    return artifact, version_warnings(artifact.get("library_versions"))


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[list[str]]]:
    """Return the model matrix and, per row, the list of unusable feature names."""
    matrix = np.full((len(df), len(features)), np.nan, dtype=float)
    problems: list[list[str]] = [[] for _ in range(len(df))]

    for index, name in enumerate(features):
        source = FEATURE_SOURCES.get(name)
        if source is None:
            raise ValueError(
                f"the artifact asks for feature {name!r}, which this module cannot "
                f"build; known features are {sorted(FEATURE_SOURCES)}"
            )
        if source not in df.columns:
            raise ValueError(f"the input table has no column {source!r}")

        raw = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(raw)
        non_positive = ~missing & (raw <= 0)
        for row in np.flatnonzero(missing):
            problems[row].append(f"{source} is missing")
        for row in np.flatnonzero(non_positive):
            problems[row].append(f"{source} is not positive")

        usable = ~(missing | non_positive)
        values = raw.copy()
        if name.startswith("log10_"):
            values = np.where(usable, np.log10(np.where(usable, raw, 1.0)), np.nan)
        matrix[:, index] = np.where(usable, values, np.nan)

    return matrix, problems


def _bounds_mask(matrix: np.ndarray, features: list[str], bounds: dict) -> np.ndarray:
    """True where any feature lies outside its stored training bounds."""
    flags = np.zeros(matrix.shape[0], dtype=bool)
    for index, name in enumerate(features):
        limits = bounds.get(name)
        if not limits:
            continue
        low, high = float(limits[0]), float(limits[1])
        column = matrix[:, index]
        flags |= (column < low) | (column > high)
    return flags


def predict_radius(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Predict radius and its 90% interval for every row of ``df``.

    Returns one row per input, in input order, with the columns of
    ``OUTPUT_COLUMNS``. Estimates are in Earth radii. A row with a missing or
    non-positive feature gets NaN estimates, ``False`` for both flags and the
    reason in ``reason``; a predicted row has an empty ``reason``.
    """
    if artifact is None:
        raise ValueError("no radius artifact was loaded; nothing can be predicted")

    features = list(artifact.get("features") or [])
    if not features:
        raise ValueError("the artifact records no feature list")

    point_model = artifact.get("point_model")
    if point_model is None:
        raise ValueError("the artifact records no point model")

    method = str(artifact.get("recommended_interval_method") or "split_conformal").strip()
    if method not in ("split_conformal", "cqr"):
        raise ValueError(f"unknown interval method {method!r} in the artifact")

    matrix, problems = _feature_matrix(df, features)
    names = df["pl_name"].astype(str).to_numpy() if "pl_name" in df.columns else np.array(
        [str(index) for index in range(len(df))]
    )
    mass_labels = (
        df["mass_provenance"].astype(str).to_numpy()
        if "mass_provenance" in df.columns
        else np.array([""] * len(df))
    )
    mass_is_minimum = np.isin(mass_labels, MINIMUM_MASS_LABELS)

    complete = np.array([not row for row in problems], dtype=bool)
    fit_rows = np.flatnonzero(complete)

    log_point = np.full(len(df), np.nan, dtype=float)
    log_lower = np.full(len(df), np.nan, dtype=float)
    log_upper = np.full(len(df), np.nan, dtype=float)

    if len(fit_rows):
        X = matrix[fit_rows]
        point = np.asarray(point_model.predict(X), dtype=float)
        if method == "cqr":
            quantile = float(artifact["cqr_quantile_dex"])
            lower = np.asarray(artifact["cqr_lower_model"].predict(X), dtype=float) - quantile
            upper = np.asarray(artifact["cqr_upper_model"].predict(X), dtype=float) + quantile
        else:
            half_width = float(artifact["split_conformal_half_width_dex"])
            lower = point - half_width
            upper = point + half_width
        log_point[fit_rows] = point
        log_lower[fit_rows] = lower
        log_upper[fit_rows] = upper

    ood = _bounds_mask(matrix, features, artifact.get("feature_bounds") or {})
    ood &= complete

    estimate = np.power(10.0, log_point)
    lower_earth = np.power(10.0, log_lower)
    upper_earth = np.power(10.0, log_upper)
    straddles = complete & (lower_earth < ROCKY_THRESHOLD_REARTH) & (
        ROCKY_THRESHOLD_REARTH < upper_earth
    )

    reasons = np.array(
        [
            "" if not row else "no estimate: " + "; ".join(row)
            for row in problems
        ],
        dtype=object,
    )

    return pd.DataFrame(
        {
            "pl_name": names,
            "radius_estimate": estimate,
            "radius_lower": lower_earth,
            "radius_upper": upper_earth,
            "mass_is_minimum": mass_is_minimum.astype(bool),
            "ood_flag": ood.astype(bool),
            "interval_straddles_1p6": straddles.astype(bool),
            "reason": reasons,
        },
        columns=OUTPUT_COLUMNS,
    )
