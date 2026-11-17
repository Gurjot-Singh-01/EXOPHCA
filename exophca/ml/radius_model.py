"""Radius estimation from mass, orbit and host-star properties.

This module is deliberately independent of the habitability score: it estimates a
planet's radius from properties that do not depend on the measured radius, so it
can be applied to planets whose archive radius is flagged as "modelled".

The evaluation pipeline has two stages, both in ``scripts/train_radius_model.py``:
stage 1 filters rows, makes the host-grouped splits and reports the baselines;
stage 2 adds the learners, the decision rule, both interval methods and the
single test-set evaluation.

DOMAIN CAVEATS
--------------
- The training selection is almost entirely transit planets (about 1608 Transit
  vs 29 Radial Velocity rows). The per-split RV counts are printed by the
  training script, never assumed.
- The planets this model is intended for (modelled radius, mostly Radial
  Velocity discoveries) usually have Msini, a MINIMUM mass. The model is trained
  on true masses, so any prediction for an Msini planet assumes sin(i) = 1 and
  the true radius may differ. Callers must carry the ``mass_is_minimum`` flag
  through to the output.
- Out-of-distribution flag: True when any feature lies outside the 1st to 99th
  percentile range of the training features. See ``feature_bounds``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import date
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # LightGBM is optional: the pipeline reports and skips it when absent.
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    LGBMRegressor = None  # type: ignore[assignment]
    LIGHTGBM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration. Every scientific or evaluation choice lives here.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_HOST_FRACTION = 0.20
CALIBRATION_HOST_FRACTION = 0.20
CV_FOLDS = 5
INTERVAL_COVERAGE = 0.90
# A stratum may sit at most this far below nominal coverage before it counts as
# under-covered (about two binomial standard errors at the stratum sizes here).
INTERVAL_TOLERANCE = 0.02

TARGET = "pl_rade"
TARGET_LOG = "log10_pl_rade"
GROUP_COLUMN = "hostname"
MASS_COLUMN = "pl_bmasse"

RAW_FEATURES = ["pl_bmasse", "pl_orbper", "st_teff", "st_mass", "st_rad"]
FEATURES = ["log10_pl_bmasse", "log10_pl_orbper", "st_teff", "st_mass", "st_rad"]

# Columns that must never enter the feature matrix: they either are the target,
# are derived from it, or would leak the archive radius/insolation.
FORBIDDEN_FEATURES = (
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_radelim",
    "pl_dens",
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_eqt",
    "pl_orbsmax",
    "sy_dist",
    "disc_year",
    "discoverymethod",
    "base_score",
    "final_score",
    "radius_provenance",
    "mass_provenance",
    "hz_class",
    "tier",
    "evidence_quality",
    "uncertainty_status",
    "scorable",
    "p_conservative_hz",
    "p_any_hz",
    "p_radius_le_1p6",
    "p_rocky_and_conservative",
)

# Forbidden names that must be rejected by prefix as well (any score/probability).
FORBIDDEN_PREFIXES = ("p_", "score", "prob")

# Mass strata used for the honest sub-report. 10 and 100 Earth masses belong to
# the middle stratum; only values strictly above 100 are in the last one.
MASS_LOW_BOUND = 10.0
MASS_HIGH_BOUND = 100.0
MASS_STRATA = ("mass < 10", "10 <= mass <= 100", "mass > 100")

# Piecewise power law: the breakpoints are FIXED IN ADVANCE, not tuned.
PIECEWISE_BREAKPOINTS = (10.0, 100.0)
PIECEWISE_SEGMENTS = ("mass < 10", "10 <= mass < 100", "mass >= 100")

# Learners compared with the baselines. Hyper-parameters are searched with a
# seeded random search scored by the same host-grouped CV used everywhere else.
LEARNER_NAMES = ("knn", "random_forest", "lightgbm")
LEARNER_SPACES: dict[str, dict[str, list]] = {
    "knn": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "random_forest": {
        "max_features": [1.0, 0.6, 0.4],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_depth": [None, 8, 12],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
}
LEARNER_TRIALS = {"knn": 24, "random_forest": 32, "lightgbm": 32}
RANDOM_FOREST_TREES = 400

# Fixed configuration for the conditional quantile models used by CQR.
CQR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
}

CAVEATS = (
    "The training selection is almost entirely transit planets (about 1608 "
    "Transit vs 29 Radial Velocity); per-split RV counts are reported.",
    "The model is trained on true masses. Planets with a modelled radius are "
    "mostly RV discoveries with Msini (a minimum mass); predictions for them "
    "assume sin(i) = 1 and the true radius may differ.",
    "Out-of-distribution flag: True when any feature lies outside the 1st-99th "
    "percentile range of the training features.",
    "This model estimates radius only. It says nothing about habitability.",
)


def package_versions() -> dict[str, str]:
    """Return the versions of the runtime and libraries that affect results."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "lightgbm"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


# ---------------------------------------------------------------------------
# Data loading and filters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterStep:
    label: str
    rows: int


def load_planets(path: str | Path) -> pd.DataFrame:
    """Load the processed planet table (read only)."""
    return pd.read_csv(path)


def filter_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FilterStep]]:
    """Apply the training-row filters and return the rows plus the count trail.

    Filters, in order: measured radius provenance; a present non-zero radius
    error; measured (not minimum) mass; all five raw features present and > 0.
    """
    steps = [FilterStep("raw rows", len(df))]

    mask = df["radius_provenance"] == "measured"
    steps.append(FilterStep('radius_provenance == "measured"', int(mask.sum())))

    mask = mask & df["pl_radeerr1"].notna() & (df["pl_radeerr1"] != 0)
    steps.append(FilterStep("pl_radeerr1 present and != 0", int(mask.sum())))

    mask = mask & (df["mass_provenance"] == "Mass")
    steps.append(FilterStep('mass_provenance == "Mass"', int(mask.sum())))

    for column in RAW_FEATURES:
        mask = mask & df[column].notna() & (df[column] > 0)
    steps.append(FilterStep("all five features present and > 0", int(mask.sum())))

    selected = df.loc[mask].copy()
    return selected, steps


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Return the model matrix in the fixed FEATURES order."""
    log_mass = np.log10(df["pl_bmasse"].to_numpy(dtype=float))
    log_period = np.log10(df["pl_orbper"].to_numpy(dtype=float))
    matrix = np.column_stack(
        [
            log_mass,
            log_period,
            df["st_teff"].to_numpy(dtype=float),
            df["st_mass"].to_numpy(dtype=float),
            df["st_rad"].to_numpy(dtype=float),
        ]
    )
    return matrix


def build_target(df: pd.DataFrame) -> np.ndarray:
    return np.log10(df[TARGET].to_numpy(dtype=float))


def assert_no_forbidden_features(features: Sequence[str]) -> None:
    """Raise if the feature list contains a forbidden or leaky column."""
    for name in features:
        if name in FORBIDDEN_FEATURES:
            raise ValueError(f"Forbidden feature in use: {name}")
        if name.startswith(FORBIDDEN_PREFIXES):
            raise ValueError(f"Forbidden feature (score/probability): {name}")


# Fail at import time if the declared feature list ever becomes leaky.
assert_no_forbidden_features(FEATURES)


# ---------------------------------------------------------------------------
# Host-grouped splits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSummary:
    train_hosts: tuple[str, ...]
    calibration_hosts: tuple[str, ...]
    test_hosts: tuple[str, ...]
    test_host_hash: str

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_hosts),
            "calibration": len(self.calibration_hosts),
            "test": len(self.test_hosts),
        }


def host_list_hash(hosts: Iterable[str]) -> str:
    """Stable SHA-256 of a sorted host list."""
    canonical = "\n".join(sorted(str(host) for host in hosts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_by_host(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    test_fraction: float = TEST_HOST_FRACTION,
    calibration_fraction: float = CALIBRATION_HOST_FRACTION,
) -> tuple[pd.DataFrame, SplitSummary]:
    """Assign train/calibration/test by host star with a fixed seed.

    Hosts are sorted, permuted with `seed`, then `round(test_fraction * n)` hosts
    form the test set; `round(calibration_fraction * remaining)` of the remainder
    form the calibration set; the rest is training data.
    """
    hosts = np.array(sorted(df[GROUP_COLUMN].unique()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(hosts))

    n_test = int(round(test_fraction * len(hosts)))
    test_hosts = hosts[order[:n_test]]
    remaining = hosts[order[n_test:]]
    n_calibration = int(round(calibration_fraction * len(remaining)))
    calibration_hosts = remaining[:n_calibration]
    train_hosts = remaining[n_calibration:]

    test_set = set(test_hosts.tolist())
    calibration_set = set(calibration_hosts.tolist())
    train_set = set(train_hosts.tolist())
    if test_set & calibration_set or test_set & train_set or calibration_set & train_set:
        raise AssertionError("Host splits overlap; this would leak between splits.")

    split = np.where(
        df[GROUP_COLUMN].isin(test_set),
        "test",
        np.where(df[GROUP_COLUMN].isin(calibration_set), "calibration", "train"),
    )
    out = df.copy()
    out["split"] = split
    if not set(out["split"].unique()) <= {"train", "calibration", "test"}:
        raise AssertionError("Unexpected split label.")

    summary = SplitSummary(
        train_hosts=tuple(sorted(train_set)),
        calibration_hosts=tuple(sorted(calibration_set)),
        test_hosts=tuple(sorted(test_set)),
        test_host_hash=host_list_hash(test_set),
    )
    return out, summary


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, hosts and RV-detected rows per split."""
    rows = []
    for name in ("train", "calibration", "test"):
        part = df[df["split"] == name]
        rows.append(
            {
                "split": name,
                "rows": int(len(part)),
                "hosts": int(part[GROUP_COLUMN].nunique()),
                "rv_rows": int((part["discoverymethod"] == "Radial Velocity").sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics and mass strata
# ---------------------------------------------------------------------------
def mass_stratum(mass_earth: float) -> str | None:
    """Return the stratum label for a mass in Earth masses."""
    if not np.isfinite(mass_earth):
        return None
    if mass_earth < MASS_LOW_BOUND:
        return MASS_STRATA[0]
    if mass_earth <= MASS_HIGH_BOUND:
        return MASS_STRATA[1]
    return MASS_STRATA[2]


def stratum_labels(masses: np.ndarray) -> np.ndarray:
    labels = np.empty(len(masses), dtype=object)
    for i, value in enumerate(masses):
        labels[i] = mass_stratum(float(value)) or "undetermined"
    return labels


def regression_metrics(y_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R2 on log10 radius plus median absolute percentage error."""
    y_log = np.asarray(y_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    residual = y_pred_log - y_log
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    true_radius = 10.0**y_log
    pred_radius = 10.0**y_pred_log
    medape = float(np.median(np.abs(pred_radius - true_radius) / true_radius) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MedAPE": medape}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the median training log10 radius for every row."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MedianBaseline":
        self.constant_ = float(np.median(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.constant_)


class PowerLawBaseline(BaseEstimator, RegressorMixin):
    """log10 R = a + b * log10 M, ordinary least squares on mass only."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PowerLawBaseline":
        log_mass = np.asarray(X, dtype=float)[:, 0]
        self.slope_, self.intercept_ = np.polyfit(log_mass, y, 1)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_mass = np.asarray(X, dtype=float)[:, 0]
        return self.intercept_ + self.slope_ * log_mass


def make_ridge_baseline() -> Pipeline:
    """Ridge regression on all five features, standardised."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def piecewise_segment(mass_earth: float) -> str | None:
    """Return the fixed piecewise segment a mass (in Earth masses) belongs to."""
    if not np.isfinite(mass_earth):
        return None
    if mass_earth < PIECEWISE_BREAKPOINTS[0]:
        return PIECEWISE_SEGMENTS[0]
    if mass_earth < PIECEWISE_BREAKPOINTS[1]:
        return PIECEWISE_SEGMENTS[1]
    return PIECEWISE_SEGMENTS[2]


class PiecewisePowerLawBaseline(BaseEstimator, RegressorMixin):
    """log10 R = a_k + b_k * log10 M, one line per FIXED mass segment.

    Breakpoints are 10 and 100 Earth masses, chosen in advance and never tuned.
    Each line is fitted only on the rows of its own segment, so a fold with no
    usable rows in a segment falls back to that segment's constant median (and
    records the fallback) instead of extrapolating a line fitted elsewhere.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PiecewisePowerLawBaseline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        log_mass = X[:, 0]
        masses = 10.0**log_mass
        self.segments_: dict[str, tuple[float, float]] = {}
        self.fallbacks_: dict[str, int] = {}
        self.segment_rows_: dict[str, int] = {}
        global_median = float(np.median(y))
        for label in PIECEWISE_SEGMENTS:
            mask = np.array([piecewise_segment(float(m)) == label for m in masses])
            self.segment_rows_[label] = int(mask.sum())
            usable = mask.sum() >= 2 and np.unique(log_mass[mask]).size >= 2
            if usable:
                slope, intercept = np.polyfit(log_mass[mask], y[mask], 1)
                self.segments_[label] = (float(intercept), float(slope))
            else:
                constant = (
                    float(np.median(y[mask])) if mask.sum() else global_median
                )
                self.segments_[label] = (constant, 0.0)
                self.fallbacks_[label] = int(mask.sum())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_mass = X[:, 0]
        out = np.empty(X.shape[0], dtype=float)
        for i, value in enumerate(log_mass):
            label = piecewise_segment(float(10.0**value))
            intercept, slope = self.segments_[label]
            out[i] = intercept + slope * value
        return out


# ---------------------------------------------------------------------------
# Learners and their seeded random search
# ---------------------------------------------------------------------------
def make_learner(name: str, params: dict):
    """Build a learner from a sampled parameter dictionary."""
    if name == "knn":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("knn", KNeighborsRegressor(**params)),
            ]
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=RANDOM_FOREST_TREES,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
        )
    if name == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not installed in this interpreter.")
        return LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            **params,
        )
    raise ValueError(f"Unknown learner: {name}")


def sample_parameter_sets(
    space: dict[str, list], n_trials: int, rng: np.random.Generator
) -> list[dict]:
    """Sample parameter dictionaries without replacement when the grid allows it."""
    names = list(space)
    total = 1
    for values in space.values():
        total *= len(values)
    indices = rng.permutation(total)[: min(n_trials, total)]
    combos = []
    for index in indices:
        remainder = int(index)
        combo = {}
        for name in reversed(names):
            size = len(space[name])
            combo[name] = space[name][remainder % size]
            remainder //= size
        combos.append({name: combo[name] for name in names})
    return combos


def _mean_absolute_error(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))))


def random_search_learner(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS,
    trials: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Seeded random search scored by host-grouped CV MAE.

    Returns the ranked candidate table and the winning parameter dictionary.
    Ties break on the fold SD and then on the parameter values, so the winner is
    deterministic.
    """
    n_trials = LEARNER_TRIALS[name] if trials is None else trials
    candidates = sample_parameter_sets(LEARNER_SPACES[name], n_trials, np.random.default_rng(seed))
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    rows = []
    for trial, params in enumerate(candidates):
        fold_maes = []
        for train_idx, valid_idx in folds:
            estimator = make_learner(name, params)
            estimator.fit(X[train_idx], y[train_idx])
            fold_maes.append(_mean_absolute_error(y[valid_idx], estimator.predict(X[valid_idx])))
        fold_maes = np.asarray(fold_maes, dtype=float)
        rows.append(
            {
                "trial": trial,
                "params": json.dumps(params, sort_keys=True),
                "mae_mean": float(fold_maes.mean()),
                "mae_sd": float(fold_maes.std(ddof=1)),
                "mae_se": float(fold_maes.std(ddof=1) / np.sqrt(n_splits)),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["mae_mean", "mae_sd", "params"], ignore_index=True
    )
    best_params = {name_: value for name_, value in json.loads(table.iloc[0]["params"]).items()}
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table, best_params


def out_of_fold_predictions(
    factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = CV_FOLDS
) -> tuple[np.ndarray, list[tuple[np.ndarray, object]]]:
    """Out-of-fold predictions plus the fitted fold models for permutation tests."""
    pred = np.full(len(y), np.nan)
    fitted = []
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        estimator = factory()
        estimator.fit(X[train_idx], y[train_idx])
        pred[valid_idx] = estimator.predict(X[valid_idx])
        fitted.append((valid_idx, estimator))
    if np.isnan(pred).any():
        raise AssertionError("Out-of-fold predictions leave rows uncovered.")
    return pred, fitted


def stratum_metrics(
    y: np.ndarray, pred: np.ndarray, masses: np.ndarray
) -> pd.DataFrame:
    """MAE/RMSE/R2/MedAPE per mass stratum, plus an 'all rows' line first."""
    labels = stratum_labels(np.asarray(masses, dtype=float))
    rows = []
    for label in ("all rows", *MASS_STRATA):
        mask = np.ones(len(y), dtype=bool) if label == "all rows" else labels == label
        if not mask.any():
            continue
        metrics = regression_metrics(y[mask], pred[mask])
        rows.append(
            {
                "stratum": label,
                "n": int(mask.sum()),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MedAPE": metrics["MedAPE"],
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_oof(
    fitted: list[tuple[np.ndarray, object]],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """MAE increase when one feature is permuted inside each validation fold.

    Uses the already-fitted fold models from ``out_of_fold_predictions`` and only
    rows that were held out for those models, so the whole procedure stays on the
    training split.
    """
    rng = np.random.default_rng(seed)
    # Fold order is not row order, so align the truth to the concatenation.
    order = np.concatenate([idx for idx, _ in fitted])
    y_ordered = np.asarray(y, dtype=float)[order]
    baseline = _mean_absolute_error(
        y_ordered,
        np.concatenate([model.predict(X[idx]) for idx, model in fitted]),
    )

    rows = []
    for column, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            preds = []
            for idx, model in fitted:
                shuffled = X.copy()
                shuffled[idx, column] = X[rng.permutation(idx), column]
                preds.append(model.predict(shuffled[idx]))
            deltas.append(_mean_absolute_error(y_ordered, np.concatenate(preds)) - baseline)
        deltas = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": name,
                "mae_increase_mean": float(deltas.mean()),
                "mae_increase_sd": (
                    float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                ),
                "n_repeats": n_repeats,
            }
        )
    table = pd.DataFrame(rows).sort_values("mae_increase_mean", ascending=False, ignore_index=True)
    table.attrs["baseline_mae"] = baseline
    return table


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------
def grouped_cv_scores(
    factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold CV and return per-fold and mean/SD summaries."""
    folds = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
        estimator = factory()
        estimator.fit(X[train_idx], y[train_idx])
        predictions = estimator.predict(X[valid_idx])
        metrics = regression_metrics(y[valid_idx], predictions)
        metrics["fold"] = fold
        metrics["n_valid"] = int(len(valid_idx))
        folds.append(metrics)

    fold_df = pd.DataFrame(folds)
    summary = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "R2", "MedAPE"],
            "mean": [fold_df[m].mean() for m in ("MAE", "RMSE", "R2", "MedAPE")],
            "sd": [fold_df[m].std(ddof=1) for m in ("MAE", "RMSE", "R2", "MedAPE")],
            "se": [
                fold_df[m].std(ddof=1) / np.sqrt(n_splits)
                for m in ("MAE", "RMSE", "R2", "MedAPE")
            ],
        }
    )
    return fold_df, summary


# ---------------------------------------------------------------------------
# Prediction intervals: split conformal and conformalised quantile regression
# ---------------------------------------------------------------------------
def conformal_order_statistic(
    scores: np.ndarray, coverage: float = INTERVAL_COVERAGE
) -> tuple[float, float, int]:
    """Split-conformal quantile: the ceil((n+1) * coverage)-th smallest score.

    Returns (quantile, achieved nominal level k / (n + 1), k).
    """
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    if n == 0:
        raise ValueError("Cannot calibrate an interval on zero residuals.")
    k = int(np.ceil((n + 1) * coverage))
    k = min(max(k, 1), n)
    return float(ordered[k - 1]), float(k / (n + 1)), k


def split_conformal_interval(
    pred: np.ndarray, quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval in log10 radius around the point prediction."""
    pred = np.asarray(pred, dtype=float)
    return pred - quantile, pred + quantile


def make_quantile_model(alpha: float, params: dict | None = None):
    """LightGBM quantile regressor used by CQR."""
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("CQR needs LightGBM, which is not installed here.")
    config = dict(CQR_PARAMS if params is None else params)
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=RANDOM_SEED,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        **config,
    )


def fit_cqr(X: np.ndarray, y: np.ndarray, coverage: float = INTERVAL_COVERAGE):
    """Fit the lower/upper conditional quantile models for CQR."""
    alpha = 1.0 - coverage
    lower_model = make_quantile_model(alpha / 2.0)
    upper_model = make_quantile_model(1.0 - alpha / 2.0)
    lower_model.fit(X, y)
    upper_model.fit(X, y)
    return lower_model, upper_model


def cqr_conformity_scores(lower_model, upper_model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CQR scores: how far outside [q_lo, q_hi] each calibration point falls."""
    return np.maximum(lower_model.predict(X) - y, y - upper_model.predict(X))


def cqr_interval(lower_model, upper_model, X: np.ndarray, quantile: float):
    """CQR interval widened by the calibrated conformity quantile."""
    return (
        lower_model.predict(X) - quantile,
        upper_model.predict(X) + quantile,
    )


def interval_table(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    masses: np.ndarray,
) -> pd.DataFrame:
    """Coverage and interval width overall and per mass stratum."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (lower <= y) & (y <= upper)
    width_dex = upper - lower
    width_radius = 10.0**upper - 10.0**lower
    labels = stratum_labels(np.asarray(masses, dtype=float))

    rows = []
    for label in ("all rows", *MASS_STRATA):
        mask = np.ones(len(y), dtype=bool) if label == "all rows" else labels == label
        if not mask.any():
            continue
        rows.append(
            {
                "stratum": label,
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_width_dex": float(width_dex[mask].mean()),
                "mean_width_rearth": float(width_radius[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def interval_cv_comparison(
    X: np.ndarray,
    y: np.ndarray,
    masses: np.ndarray,
    groups: np.ndarray,
    point_factory,
    coverage: float = INTERVAL_COVERAGE,
    n_splits: int = CV_FOLDS,
    calibration_fraction: float = CALIBRATION_HOST_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Compare both interval methods on the training split only.

    Each outer fold's training hosts are split again into fit/calibration hosts,
    so the outer validation rows are never used for fitting or calibrating. The
    test split is not touched, which is what lets the interval method be chosen
    before the single test evaluation.
    """
    collected = {
        method: {"lower": [], "upper": [], "y": [], "mass": []}
        for method in ("split_conformal", "cqr")
    }

    for fold, (train_idx, valid_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, groups)
    ):
        train_groups = np.array(sorted(set(groups[train_idx])))
        rng = np.random.default_rng(seed + fold)
        order = rng.permutation(len(train_groups))
        n_calibration = int(round(calibration_fraction * len(train_groups)))
        calibration_groups = set(train_groups[order[:n_calibration]].tolist())

        fit_mask = np.array([g not in calibration_groups for g in groups[train_idx]])
        cal_mask = ~fit_mask
        fit_idx = train_idx[fit_mask]
        cal_idx = train_idx[cal_mask]
        if len(cal_idx) < 10:
            raise AssertionError(f"Calibration fold {fold} is too small to calibrate.")

        point_model = point_factory()
        point_model.fit(X[fit_idx], y[fit_idx])

        # Split conformal: absolute residuals on the calibration hosts.
        cal_residual = np.abs(point_model.predict(X[cal_idx]) - y[cal_idx])
        q, _, _ = conformal_order_statistic(cal_residual, coverage)
        lower, upper = split_conformal_interval(point_model.predict(X[valid_idx]), q)
        collected["split_conformal"]["lower"].append(lower)
        collected["split_conformal"]["upper"].append(upper)

        # CQR: conditional quantiles plus the conformity score quantile.
        lower_model, upper_model = fit_cqr(X[fit_idx], y[fit_idx], coverage)
        scores = cqr_conformity_scores(lower_model, upper_model, X[cal_idx], y[cal_idx])
        q_cqr, _, _ = conformal_order_statistic(scores, coverage)
        lower_cqr, upper_cqr = cqr_interval(lower_model, upper_model, X[valid_idx], q_cqr)
        collected["cqr"]["lower"].append(lower_cqr)
        collected["cqr"]["upper"].append(upper_cqr)

        for method in collected:
            collected[method]["y"].append(y[valid_idx])
            collected[method]["mass"].append(np.asarray(masses, dtype=float)[valid_idx])

    out = {}
    for method, parts in collected.items():
        out[method] = interval_table(
            np.concatenate(parts["y"]),
            np.concatenate(parts["lower"]),
            np.concatenate(parts["upper"]),
            np.concatenate(parts["mass"]),
        )
    return out


def interval_check(
    table: pd.DataFrame,
    coverage: float = INTERVAL_COVERAGE,
    tolerance: float = INTERVAL_TOLERANCE,
) -> dict:
    """Check overall coverage and the worst stratum against the nominal level."""
    strata = table[table["stratum"] != "all rows"]
    worst = strata.loc[strata["coverage"].idxmin()]
    overall = float(table.iloc[0]["coverage"])
    shortfall = max(0.0, coverage - float(worst["coverage"]))
    return {
        "overall": overall,
        "worst_stratum": str(worst["stratum"]),
        "worst_coverage": float(worst["coverage"]),
        "shortfall": shortfall,
        "passes": bool(overall >= coverage and shortfall <= tolerance),
    }


def recommend_interval_method(
    tables: dict[str, pd.DataFrame],
    coverage: float = INTERVAL_COVERAGE,
    tolerance: float = INTERVAL_TOLERANCE,
) -> tuple[str, str]:
    """Apply the pre-declared interval-selection rule and explain the outcome.

    Rule, fixed before looking at the numbers:
      1. a method qualifies only if its overall coverage reaches the nominal level
         AND no mass stratum falls more than `tolerance` below it;
      2. of the qualifiers, take the narrower mean interval width;
      3. if none qualifies, take the smaller worst-stratum shortfall;
      4. widths within 0.5 percent are a tie, and split conformal wins ties because
         it makes no model for the interval width.
    """
    checks = {method: interval_check(table, coverage, tolerance) for method, table in tables.items()}
    width = {method: float(table.iloc[0]["mean_width_dex"]) for method, table in tables.items()}
    qualifiers = [method for method in tables if checks[method]["passes"]]

    if qualifiers:
        ordered = sorted(qualifiers, key=lambda m: (width[m], m != "split_conformal"))
        winner = ordered[0]
        if len(ordered) > 1:
            runner_up = ordered[1]
            ratio = width[runner_up] / width[winner] if width[winner] > 0 else float("inf")
            if ratio <= 1.005 and winner != "split_conformal":
                winner = "split_conformal"
                reason = (
                    "both methods meet the coverage requirement and their widths are within "
                    f"0.5 percent ({width['split_conformal']:.3f} vs {width['cqr']:.3f} dex), "
                    "so split conformal wins the tie: it needs no conditional quantile model."
                )
            else:
                reason = (
                    f"both methods meet the coverage requirement; {winner} has the narrower "
                    f"mean width ({width[winner]:.3f} vs {width[runner_up]:.3f} dex), with "
                    f"worst-stratum coverage {checks[winner]['worst_coverage']:.3f} in "
                    f"'{checks[winner]['worst_stratum']}' versus "
                    f"{checks[runner_up]['worst_coverage']:.3f} for {runner_up}."
                )
        else:
            reason = (
                f"{winner} is the only method meeting the coverage requirement "
                f"(overall {checks[winner]['overall']:.3f}, worst stratum "
                f"{checks[winner]['worst_coverage']:.3f}); the other method under-covers out "
                "of sample."
            )
    else:
        ordered = sorted(
            tables,
            key=lambda m: (checks[m]["shortfall"], -checks[m]["overall"], m != "split_conformal"),
        )
        winner = ordered[0]
        loser = ordered[1]
        reason = (
            "neither method is uniformly calibrated: both meet overall coverage but each "
            "under-covers in at least one mass stratum beyond the tolerance. "
            f"{winner} has the smaller worst-stratum shortfall "
            f"({checks[winner]['shortfall']:.3f} in '{checks[winner]['worst_stratum']}' vs "
            f"{checks[loser]['shortfall']:.3f} in '{checks[loser]['worst_stratum']}'), and "
            f"the narrower mean width ({width[winner]:.3f} vs {width[loser]:.3f} dex)."
        )
    return winner, reason


def feature_bounds(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """1st-99th percentile bounds of every feature, from the training split only."""
    X = build_features(train_df)
    return {
        name: (float(np.percentile(X[:, i], 1)), float(np.percentile(X[:, i], 99)))
        for i, name in enumerate(FEATURES)
    }


def out_of_distribution(features: np.ndarray, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """True where any feature lies outside its training 1st-99th percentile range."""
    flags = np.zeros(features.shape[0], dtype=bool)
    for i, name in enumerate(FEATURES):
        low, high = bounds[name]
        column = features[:, i]
        flags |= (column < low) | (column > high)
    return flags


def read_meta(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_evaluation_allowed(meta: dict | None, force: bool = False) -> tuple[bool, str]:
    """Decide whether the test split may be evaluated again."""
    if meta is None or not meta.get("test_evaluated", False):
        return True, ""
    if force:
        return True, "WARNING: --force passed; the test split is being re-evaluated."
    return (
        False,
        "Refusing to evaluate on the test split again: it was already evaluated "
        f"on {meta.get('test_evaluated_on', 'an earlier run')} "
        f"(host hash {meta.get('test_host_hash', 'unknown')}). Pass --force to override.",
    )


def meta_stub(seed: int = RANDOM_SEED) -> dict:
    """Fields common to every stage of the training report."""
    versions = package_versions()
    return {
        "generated_on": date.today().isoformat(),
        "python": sys.version.split()[0],
        "library_versions": versions,
        "lightgbm_used": versions.get("lightgbm", "not installed") != "not installed",
        "random_seed": seed,
        "features": list(FEATURES),
        "raw_features": list(RAW_FEATURES),
        "target": f"log10({TARGET})",
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "test_host_fraction": TEST_HOST_FRACTION,
        "calibration_host_fraction": CALIBRATION_HOST_FRACTION,
        "cv_folds": CV_FOLDS,
        "target_interval_coverage": INTERVAL_COVERAGE,
        "caveats": list(CAVEATS),
    }
