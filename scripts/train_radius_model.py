"""Train and evaluate the EXOPHCA radius model.

Stage 1 filters the training rows, makes the host-grouped splits and reports the
baselines. Stage 2 adds three learners (kNN, random forest, LightGBM) tuned by a
seeded random search, compares them with the best baseline under a written
decision rule, compares split-conformal with conformalised quantile regression
intervals, and then evaluates the test split exactly once.

Every number in the report is computed at run time; nothing is typed as a
literal result. The script writes its own stdout to ``docs/progress/phase9_ml.md``
so the report is raw command output.

Usage:
    .venv/Scripts/python.exe scripts/train_radius_model.py --stage 2
    .venv/Scripts/python.exe scripts/train_radius_model.py --stage 2 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exophca.ml import radius_model as rm  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "planets.csv"
DEFAULT_META = PROJECT_ROOT / "models" / "radius_model_meta.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "radius_model.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "progress" / "phase9_ml.md"

# Environment the stage-2 report must run in. These are requirements on the
# interpreter, not reported results.
REQUIRED_VERSIONS = {
    "numpy": "2.5.4",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "lightgbm": "4.7.0",
}

BASELINES = (
    ("median", rm.MedianBaseline, "baseline"),
    ("power_law", rm.PowerLawBaseline, "baseline"),
    ("piecewise_power_law", rm.PiecewisePowerLawBaseline, "baseline"),
    ("ridge", rm.make_ridge_baseline, "baseline"),
)
BASELINE_FACTORIES = {name: factory for name, factory, _ in BASELINES}


class Report:
    """Collect every emitted line, print it, and write it to the report file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(str(text))

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_params(params: dict | None) -> str:
    if not params:
        return "none (fixed configuration)"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _json_default(value):
    """Make numpy/pandas scalars JSON-serialisable in the meta file."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=_json_default) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# (a) Environment
# ---------------------------------------------------------------------------
def verify_environment(emit) -> dict[str, str]:
    versions = rm.package_versions()
    emit("## (a) Environment")
    emit(f"sys.executable: {sys.executable}")
    for name, version in versions.items():
        emit(f"{name}: {version}")
    emit("required versions: " + ", ".join(f"{k} {v}" for k, v in REQUIRED_VERSIONS.items()))

    mismatches = [
        f"{name}: required {required}, found {versions.get(name)}"
        for name, required in REQUIRED_VERSIONS.items()
        if versions.get(name) != required
    ]
    if mismatches:
        for problem in mismatches:
            emit(f"MISMATCH: {problem}")
        emit("STOP: the interpreter does not match the required versions; nothing was evaluated.")
        raise SystemExit(2)
    emit("required-version check: all match")

    if rm.LIGHTGBM_AVAILABLE:
        emit(f"LightGBM available and will be used: {versions['lightgbm']}")
    else:
        emit("LightGBM NOT INSTALLED in this interpreter - the LightGBM learner and CQR are skipped.")
    emit("")
    return versions


# ---------------------------------------------------------------------------
# (b) Filters and splits
# ---------------------------------------------------------------------------
def print_filters(emit, raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    selected, steps = rm.filter_training_rows(raw)
    emit("## (b) Training-row filters, host-grouped splits and the test-host hash")
    emit("")
    emit("### Filters")
    previous = None
    for step in steps:
        removed = "" if previous is None else f"   (removed {previous - step.rows})"
        emit(f"{step.label:<45} rows = {step.rows:5d}{removed}")
        previous = step.rows
    emit("")
    return selected, steps


def print_splits(emit, split_df, summary) -> None:
    table = rm.split_table(split_df)
    emit(f"### Host-grouped splits (seed = {rm.RANDOM_SEED})")
    emit(table.to_string(index=False))
    emit("")
    emit(f"test host count        : {len(summary.test_hosts)}")
    emit(f"calibration host count : {len(summary.calibration_hosts)}")
    emit(f"train host count       : {len(summary.train_hosts)}")
    emit(f"test_host_hash         : {summary.test_host_hash}")
    emit("")


def split_frames(split_df):
    return (
        split_df[split_df["split"] == "train"],
        split_df[split_df["split"] == "calibration"],
        split_df[split_df["split"] == "test"],
    )


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------
def cv_block(emit, title: str, factory, X, y, groups, masses) -> dict:
    """Per-fold CV table plus out-of-fold predictions and stratum metrics."""
    fold_df, summary = rm.grouped_cv_scores(factory, X, y, groups)
    pred, fitted = rm.out_of_fold_predictions(factory, X, y, groups)
    strata = rm.stratum_metrics(y, pred, masses)

    emit(f"### {title}")
    emit(fold_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    for _, row in summary.iterrows():
        emit(
            f"{row['metric']:<7} mean = {row['mean']:.6f}   "
            f"SD across folds = {row['sd']:.6f}   SE = {row['se']:.6f}"
        )
    emit("")
    metrics = {
        row["metric"]: {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "se": float(row["se"]),
        }
        for _, row in summary.iterrows()
    }
    return {
        "fold_df": fold_df,
        "summary": summary,
        "metrics": metrics,
        "pred": pred,
        "fitted": fitted,
        "strata": strata,
    }


def run_baselines(emit, split_df) -> dict:
    train, _, _ = split_frames(split_df)
    X = rm.build_features(train)
    y = rm.build_target(train)
    masses = train[rm.MASS_COLUMN].to_numpy(dtype=float)
    groups = train[rm.GROUP_COLUMN].to_numpy()

    emit("## (c) Grouped 5-fold CV: baselines and learners")
    emit("")
    emit(
        f"rows = {len(train)}, hosts = {train[rm.GROUP_COLUMN].nunique()}, "
        f"folds = {rm.CV_FOLDS}, seed = {rm.RANDOM_SEED}"
    )
    emit("")

    results = {}
    for name, factory, kind in BASELINES:
        results[name] = cv_block(emit, f"baseline: {name}", factory, X, y, groups, masses)
        results[name]["kind"] = kind

    fitted = rm.PiecewisePowerLawBaseline().fit(X, y)
    emit("### Piecewise power law: fixed breakpoints and fitted segments")
    emit(f"breakpoints (Earth masses, fixed in advance): {list(rm.PIECEWISE_BREAKPOINTS)}")
    segment_rows = []
    for label in rm.PIECEWISE_SEGMENTS:
        intercept, slope = fitted.segments_[label]
        segment_rows.append(
            {
                "segment": label,
                "train_rows": fitted.segment_rows_[label],
                "intercept_a": intercept,
                "slope_b": slope,
                "constant_fallback_rows": fitted.fallbacks_.get(label, 0),
            }
        )
    emit(pd.DataFrame(segment_rows).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    boundary = np.zeros(len(masses), dtype=bool)
    for breakpoint in rm.PIECEWISE_BREAKPOINTS:
        boundary |= np.isclose(masses, breakpoint)
    emit(f"training rows exactly on a breakpoint: {int(boundary.sum())}")
    emit("")

    results["piecewise_power_law"]["fitted"] = {
        row["segment"]: {
            "intercept_a": row["intercept_a"],
            "slope_b": row["slope_b"],
            "train_rows": row["train_rows"],
        }
        for row in segment_rows
    }
    return results


def run_learners(emit, split_df) -> dict:
    train, _, _ = split_frames(split_df)
    X = rm.build_features(train)
    y = rm.build_target(train)
    masses = train[rm.MASS_COLUMN].to_numpy(dtype=float)
    groups = train[rm.GROUP_COLUMN].to_numpy()

    results = {}
    for name in rm.LEARNER_NAMES:
        if name == "lightgbm" and not rm.LIGHTGBM_AVAILABLE:
            emit(f"### learner: {name} - SKIPPED (not installed in this interpreter)")
            emit("")
            continue

        search, best_params = rm.random_search_learner(name, X, y, groups)
        emit(f"### learner: {name} - seeded random search ({len(search)} candidates)")
        emit(f"best params: {_format_params(best_params)}")
        emit(search.head(5).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        emit("")

        block = cv_block(
            emit,
            f"learner: {name} (best params, same folds)",
            lambda learner=name, params=best_params: rm.make_learner(learner, params),
            X,
            y,
            groups,
            masses,
        )
        expected = float(search.iloc[0]["mae_mean"])
        reproduced = block["metrics"]["MAE"]["mean"]
        if abs(reproduced - expected) > 1e-9:
            raise AssertionError(
                f"{name}: CV MAE {reproduced} does not reproduce the search winner {expected}"
            )
        block.update(
            {"kind": "learner", "params": best_params, "search": search, "trials": len(search)}
        )
        results[name] = block

    return results


def aggregate_cv_table(emit, results: dict) -> pd.DataFrame:
    rows = []
    for name, block in results.items():
        rows.append(
            {
                "model": name,
                "kind": block["kind"],
                "mae_mean": block["metrics"]["MAE"]["mean"],
                "mae_sd": block["metrics"]["MAE"]["sd"],
                "mae_se": block["metrics"]["MAE"]["se"],
                "rmse_mean": block["metrics"]["RMSE"]["mean"],
                "r2_mean": block["metrics"]["R2"]["mean"],
                "medape_mean": block["metrics"]["MedAPE"]["mean"],
            }
        )
    table = pd.DataFrame(rows).sort_values("mae_mean", ignore_index=True)
    emit("### Aggregate CV table (sorted by MAE)")
    emit(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")
    return table


def stratum_cv_table(emit, results: dict) -> pd.DataFrame:
    rows = []
    for name, block in results.items():
        record = {"model": name}
        for _, row in block["strata"].iterrows():
            record[row["stratum"]] = row["MAE"]
        rows.append(record)
    table = pd.DataFrame(rows)
    emit("### Per mass stratum: out-of-fold CV MAE")
    emit(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")
    return table


# ---------------------------------------------------------------------------
# (d) Decision rule
# ---------------------------------------------------------------------------
def decide(emit, cv_table: pd.DataFrame, results: dict) -> dict:
    baselines = cv_table[cv_table["kind"] == "baseline"]
    best = baselines.sort_values("mae_mean").iloc[0]
    best_name = str(best["model"])
    threshold = float(best["mae_mean"]) - float(best["mae_se"])

    emit("## (d) Decision rule")
    emit("")
    emit(
        "Rule, fixed in advance: a learner replaces the best baseline only if its CV MAE "
        "is below (best baseline CV MAE - best baseline CV standard error) AND its single "
        "test MAE is below the best baseline's test MAE. The test half is only checked in "
        "section (g)."
    )
    emit("")
    emit(f"best baseline                  : {best_name}")
    emit(f"CV MAE mean                    : {best['mae_mean']:.6f}")
    emit(f"CV MAE SD across folds         : {best['mae_sd']:.6f}")
    emit(f"CV MAE standard error          : {best['mae_se']:.6f}")
    emit(f"learner must beat (MAE < this) : {threshold:.6f}")
    emit("")

    rows = []
    for name, block in results.items():
        if block["kind"] != "learner":
            continue
        mae = block["metrics"]["MAE"]["mean"]
        rows.append(
            {
                "learner": name,
                "cv_mae": mae,
                "improvement_vs_baseline": float(best["mae_mean"]) - mae,
                "clears_cv_threshold": bool(mae < threshold),
            }
        )
    learners = pd.DataFrame(
        rows,
        columns=["learner", "cv_mae", "improvement_vs_baseline", "clears_cv_threshold"],
    )

    emit("### Learners against the threshold (CV only; test not yet touched)")
    if learners.empty:
        emit("no learners available in this interpreter")
    else:
        emit(learners.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")

    qualifying = learners[learners["clears_cv_threshold"]]
    if qualifying.empty:
        chosen_name = best_name
        chosen_kind = "baseline"
        chosen_params = None
        chosen_cv_mae = float(best["mae_mean"])
        emit(
            "CV outcome: NO learner clears the threshold, so the final model is the best "
            f"baseline ({best_name})."
        )
    else:
        chosen = qualifying.sort_values("cv_mae").iloc[0]
        chosen_name = str(chosen["learner"])
        chosen_kind = "learner"
        chosen_params = results[chosen_name]["params"]
        chosen_cv_mae = float(chosen["cv_mae"])
        emit(
            f"CV outcome: {len(qualifying)} learner(s) clear the threshold "
            f"({', '.join(qualifying['learner'])}); the lowest CV MAE among them is "
            f"{chosen_name}, which proceeds to the single test evaluation."
        )
    emit("")

    return {
        "best_baseline": best_name,
        "best_baseline_cv_mae": float(best["mae_mean"]),
        "best_baseline_cv_sd": float(best["mae_sd"]),
        "best_baseline_cv_se": float(best["mae_se"]),
        "cv_threshold": threshold,
        "learners": learners.to_dict("records"),
        "chosen_model": chosen_name,
        "chosen_kind": chosen_kind,
        "chosen_params": chosen_params,
        "chosen_cv_mae": chosen_cv_mae,
    }


# ---------------------------------------------------------------------------
# Stage 1 (kept byte-for-byte so the reviewed baseline report reproduces)
# ---------------------------------------------------------------------------
def stage1_environment(emit) -> None:
    versions = rm.package_versions()
    emit("## Environment")
    for name, version in versions.items():
        emit(f"{name}: {version}")
    if versions.get("lightgbm", "not installed") == "not installed":
        emit("LightGBM: NOT INSTALLED - the LightGBM learner will be skipped.")
    else:
        emit(f"LightGBM available and will be used: {versions['lightgbm']}")
    emit("")


def run_stage1(emit, data_path: Path, meta_path: Path) -> dict:
    emit("# Phase 9 - Radius model, stage 1")
    emit("")
    emit(f"generated: {date.today().isoformat()}")
    emit(f"data: {_relative(data_path)}")
    emit(f"seed: {rm.RANDOM_SEED}")
    emit("")

    stage1_environment(emit)
    raw = rm.load_planets(data_path)

    selected, steps = rm.filter_training_rows(raw)
    emit("## Training-row filters")
    previous = None
    for step in steps:
        removed = "" if previous is None else f"   (removed {previous - step.rows})"
        emit(f"{step.label:<45} rows = {step.rows:5d}{removed}")
        previous = step.rows
    emit("")

    split_df, summary = rm.split_by_host(selected)
    emit(f"## Host-grouped splits (seed = {rm.RANDOM_SEED})")
    emit(rm.split_table(split_df).to_string(index=False))
    emit("")
    emit(f"test host count        : {len(summary.test_hosts)}")
    emit(f"calibration host count : {len(summary.calibration_hosts)}")
    emit(f"train host count       : {len(summary.train_hosts)}")
    emit(f"test_host_hash         : {summary.test_host_hash}")
    emit("")

    train, _, _ = split_frames(split_df)
    X = rm.build_features(train)
    y = rm.build_target(train)
    groups = train[rm.GROUP_COLUMN].to_numpy()

    emit("## Grouped 5-fold CV on the training split (baselines)")
    emit(f"rows = {len(train)}, hosts = {train[rm.GROUP_COLUMN].nunique()}")
    emit("")

    results = {}
    for name, factory, _kind in BASELINES:
        fold_df, summary_metrics = rm.grouped_cv_scores(factory, X, y, groups)
        emit(f"### {name}")
        emit(fold_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        for _, row in summary_metrics.iterrows():
            emit(
                f"{row['metric']:<7} mean = {row['mean']:.6f}   "
                f"SD across folds = {row['sd']:.6f}"
            )
        results[name] = {
            "metrics": {
                row["metric"]: {"mean": float(row["mean"]), "sd": float(row["sd"])}
                for _, row in summary_metrics.iterrows()
            }
        }
        emit("")

    power_law = rm.PowerLawBaseline().fit(X, y)
    results["power_law"]["fitted"] = {
        "intercept_a": float(power_law.intercept_),
        "slope_b": float(power_law.slope_),
        "form": "log10(R_earth) = a + b * log10(M_earth)",
    }
    emit("### Power-law fit on the training split")
    emit(
        f"log10(R_earth) = {power_law.intercept_:.6f} + {power_law.slope_:.6f} * log10(M_earth)"
    )
    emit("")

    meta = rm.meta_stub()
    split_records = rm.split_table(split_df).to_dict("records")
    meta.update(
        {
            "pipeline_stage": 1,
            "data_file": _relative(data_path),
            "row_filters": {step.label: step.rows for step in steps},
            "split_sizes": {
                "rows": {r["split"]: int(r["rows"]) for r in split_records},
                "hosts": {r["split"]: int(r["hosts"]) for r in split_records},
                "rv_rows": {r["split"]: int(r["rv_rows"]) for r in split_records},
            },
            "test_host_hash": summary.test_host_hash,
            "test_host_count": len(summary.test_hosts),
            "test_evaluated": False,
            "test_evaluated_on": None,
            "cv_baselines": {
                name: {"kind": "baseline", "metrics": results[name]["metrics"]}
                for name in ("median", "power_law", "piecewise_power_law", "ridge")
            },
        }
    )
    meta["cv_baselines"]["power_law"]["fitted"] = results["power_law"]["fitted"]

    existing = rm.read_meta(meta_path)
    if existing is not None and existing.get("test_evaluated"):
        emit(
            "meta NOT overwritten: it records an evaluated test split, and a stage-1 run "
            "must not clear that guard."
        )
    else:
        write_meta(meta, meta_path)
        emit(f"meta written to {_relative(meta_path)}")
    emit("")
    emit("Stage 1 complete. Stage 2 (learners, intervals, models, tests) not run.")
    return {}


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def run_stage2(emit, data_path: Path, meta_path: Path, model_path: Path, force: bool) -> dict:
    existing = rm.read_meta(meta_path)
    allowed, guard_message = rm.test_evaluation_allowed(existing, force=force)
    if not allowed:
        print(guard_message, file=sys.stderr)
        raise SystemExit(2)

    emit("# Phase 9 - Radius model, stage 2")
    emit("")
    emit(f"generated: {date.today().isoformat()}")
    emit(f"data: {_relative(data_path)}")
    emit(f"model artifact: {_relative(model_path)}")
    emit(f"seed: {rm.RANDOM_SEED}")
    emit(f"interpreter: {sys.executable}")
    emit("")
    if guard_message:
        emit(guard_message)
        emit("")

    verify_environment(emit)

    raw = rm.load_planets(data_path)
    selected, steps = print_filters(emit, raw)

    split_df, summary = rm.split_by_host(selected)
    print_splits(emit, split_df, summary)

    if existing is not None:
        recorded_hash = existing.get("test_host_hash")
        same = recorded_hash == summary.test_host_hash
        emit(f"stage-1 recorded test_host_hash equals the recomputed split: {same}")
        if recorded_hash is not None and not same:
            raise SystemExit(
                "STOP: the recomputed split does not match the stage-1 test-host hash; "
                "the data or the split code changed."
            )
        emit("")

    train, calibration, test = split_frames(split_df)

    X_train = rm.build_features(train)
    y_train = rm.build_target(train)
    m_train = train[rm.MASS_COLUMN].to_numpy(dtype=float)
    g_train = train[rm.GROUP_COLUMN].to_numpy()

    baseline_results = run_baselines(emit, split_df)
    learner_results = run_learners(emit, split_df)
    cv_results = {**baseline_results, **learner_results}

    cv_table = aggregate_cv_table(emit, cv_results)
    stratum_table = stratum_cv_table(emit, cv_results)
    decision = decide(emit, cv_table, cv_results)

    # ---- (e) Interval comparison, training split only ----
    emit("## (e) Prediction intervals: split conformal vs conformalised quantile regression")
    emit("")
    emit(
        f"Both methods are compared inside the training split with a nested {rm.CV_FOLDS}-fold "
        f"host-grouped CV: each outer fold's training hosts are split into "
        f"{1 - rm.CALIBRATION_HOST_FRACTION:.0%} fit and "
        f"{rm.CALIBRATION_HOST_FRACTION:.0%} calibration hosts, and coverage is measured on "
        "the held-out outer fold. The test split is not used here, so the interval method is "
        "chosen before the single test evaluation."
    )
    emit("")

    chosen_name = decision["chosen_model"]
    if decision["chosen_kind"] == "learner":
        point_factory = lambda: rm.make_learner(chosen_name, decision["chosen_params"])
    else:
        point_factory = BASELINE_FACTORIES[chosen_name]
    emit(f"point model used for the interval comparison: {chosen_name}")
    emit("")

    interval_cv = rm.interval_cv_comparison(X_train, y_train, m_train, g_train, point_factory)
    interval_checks = {}
    for method, table in interval_cv.items():
        check = rm.interval_check(table, rm.INTERVAL_COVERAGE)
        interval_checks[method] = check
        emit(f"### {method} - out-of-sample coverage and width (training split)")
        emit(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        emit(
            f"overall coverage {check['overall']:.6f}; worst stratum "
            f"'{check['worst_stratum']}' at {check['worst_coverage']:.6f}; "
            f"shortfall {check['shortfall']:.6f}; meets requirement: {check['passes']}"
        )
        emit("")

    emit(
        "Selection rule, fixed in advance: (1) a method qualifies only if its overall "
        f"coverage reaches the nominal {rm.INTERVAL_COVERAGE:.2f} AND no mass stratum falls "
        f"more than {rm.INTERVAL_TOLERANCE:.2f} below it; (2) of the qualifiers take the "
        "narrower mean width; (3) if none qualifies, take the smaller worst-stratum "
        "shortfall; (4) widths within 0.5 percent are a tie and split conformal wins it "
        "because it models no interval width."
    )
    recommended, reason = rm.recommend_interval_method(interval_cv, rm.INTERVAL_COVERAGE)
    emit(f"recommended interval method on CV: {recommended}")
    emit(f"reason: {reason}")
    emit("")

    # ---- (f) Permutation importance and applicability ----
    emit("## (f) Permutation importance and applicability of the final model")
    emit("")
    emit(f"final point model: {chosen_name} ({decision['chosen_kind']})")
    emit(f"final parameters: {_format_params(decision['chosen_params'])}")
    emit("")

    importance = rm.permutation_importance_oof(
        cv_results[chosen_name]["fitted"], X_train, y_train, rm.FEATURES
    )
    baseline_mae = float(importance.attrs["baseline_mae"])
    emit(f"out-of-fold MAE of the final model before permutation: {baseline_mae:.6f}")
    emit(importance.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")

    bounds = rm.feature_bounds(train)
    emit("### Training-feature bounds (1st-99th percentile, out-of-distribution rule)")
    emit(
        pd.DataFrame(
            [
                {"feature": name, "p01": bounds[name][0], "p99": bounds[name][1]}
                for name in rm.FEATURES
            ]
        ).to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )
    emit("")

    # ---- (g) The single test evaluation ----
    emit("## (g) Single test evaluation")
    emit("")
    emit(f"test rows: {len(test)}, test hosts: {test[rm.GROUP_COLUMN].nunique()}")
    emit(f"test_host_hash: {summary.test_host_hash}")
    emit("")

    X_test = rm.build_features(test)
    y_test = rm.build_target(test)
    m_test = test[rm.MASS_COLUMN].to_numpy(dtype=float)

    point_model = point_factory()
    point_model.fit(X_train, y_train)
    pred_test = point_model.predict(X_test)

    best_baseline_name = decision["best_baseline"]
    baseline_model = BASELINE_FACTORIES[best_baseline_name]()
    baseline_model.fit(X_train, y_train)
    pred_baseline = baseline_model.predict(X_test)

    emit("### Test metrics: chosen model and best baseline (fitted on the training split)")
    test_rows = []
    for label, pred in (
        ("chosen model", pred_test),
        (f"best baseline ({best_baseline_name})", pred_baseline),
    ):
        test_rows.append({"model": label, **rm.regression_metrics(y_test, pred)})
    test_table = pd.DataFrame(test_rows)
    emit(test_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")

    emit("### Test metrics per mass stratum")
    strata_rows = []
    for label, pred in (
        ("chosen model", pred_test),
        (f"best baseline ({best_baseline_name})", pred_baseline),
    ):
        for _, row in rm.stratum_metrics(y_test, pred, m_test).iterrows():
            strata_rows.append(
                {
                    "model": label,
                    "stratum": row["stratum"],
                    "n": row["n"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MedAPE": row["MedAPE"],
                }
            )
    strata_test = pd.DataFrame(strata_rows)
    emit(strata_test.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    emit("")

    # Interval layers, calibrated once on the calibration split.
    X_cal = rm.build_features(calibration)
    y_cal = rm.build_target(calibration)
    cal_residual = np.abs(point_model.predict(X_cal) - y_cal)
    q_split, achieved_split, k_split = rm.conformal_order_statistic(
        cal_residual, rm.INTERVAL_COVERAGE
    )
    lower_split, upper_split = rm.split_conformal_interval(pred_test, q_split)
    emit("### Split conformal calibration")
    emit(
        f"calibration rows: {len(calibration)}, hosts: {calibration[rm.GROUP_COLUMN].nunique()}"
    )
    emit(
        f"nominal coverage: {rm.INTERVAL_COVERAGE:.2f}, achieved nominal level k/(n+1): "
        f"{achieved_split:.6f} (k={k_split})"
    )
    emit(f"symmetric half-width in log10(R_earth): {q_split:.6f}")
    emit("")

    lower_cqr = upper_cqr = None
    lower_model = upper_model = None
    q_cqr = None
    if rm.LIGHTGBM_AVAILABLE:
        lower_model, upper_model = rm.fit_cqr(X_train, y_train, rm.INTERVAL_COVERAGE)
        cqr_scores = rm.cqr_conformity_scores(lower_model, upper_model, X_cal, y_cal)
        q_cqr, achieved_cqr, k_cqr = rm.conformal_order_statistic(
            cqr_scores, rm.INTERVAL_COVERAGE
        )
        lower_cqr, upper_cqr = rm.cqr_interval(lower_model, upper_model, X_test, q_cqr)
        emit("### CQR calibration")
        emit(
            f"calibration rows: {len(calibration)}, hosts: {calibration[rm.GROUP_COLUMN].nunique()}"
        )
        emit(
            f"nominal coverage: {rm.INTERVAL_COVERAGE:.2f}, achieved nominal level k/(n+1): "
            f"{achieved_cqr:.6f} (k={k_cqr})"
        )
        emit(f"conformity quantile in log10(R_earth): {q_cqr:.6f}")
        emit("")

    emit("### Test coverage and width by mass stratum, both interval methods")
    test_tables = {}
    test_checks = {}
    for method, (lo, hi) in (
        ("split_conformal", (lower_split, upper_split)),
        ("cqr", (lower_cqr, upper_cqr)),
    ):
        if lo is None:
            emit(f"{method}: not available in this interpreter")
            emit("")
            continue
        table = rm.interval_table(y_test, lo, hi, m_test)
        check = rm.interval_check(table, rm.INTERVAL_COVERAGE)
        test_tables[method] = table
        test_checks[method] = check
        emit(method)
        emit(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        emit(
            f"overall coverage {check['overall']:.6f}; worst stratum "
            f"'{check['worst_stratum']}' at {check['worst_coverage']:.6f}; "
            f"shortfall {check['shortfall']:.6f}; meets requirement: {check['passes']}"
        )
        emit("")

    test_width = {
        method: float(table.iloc[0]["mean_width_dex"])
        for method, table in test_tables.items()
    }
    final_interval_method = min(
        test_tables,
        key=lambda m: (
            not test_checks[m]["passes"],
            test_checks[m]["shortfall"],
            test_width[m],
            m != "split_conformal",
        ),
    )

    if final_interval_method == recommended:
        if test_checks[recommended]["passes"]:
            interval_verdict = (
                f"{recommended} (chosen on CV) also meets the per-stratum coverage requirement "
                f"on the test split: overall {test_checks[recommended]['overall']:.3f}, worst "
                f"stratum '{test_checks[recommended]['worst_stratum']}' at "
                f"{test_checks[recommended]['worst_coverage']:.3f}. It is recommended."
            )
        else:
            interval_verdict = (
                f"{recommended}, with a stated limitation: neither method meets the "
                f"per-stratum requirement on the test split. {recommended} has the smaller "
                f"test shortfall ({test_checks[recommended]['shortfall']:.3f} in "
                f"'{test_checks[recommended]['worst_stratum']}' vs "
                f"{test_checks['split_conformal' if recommended == 'cqr' else 'cqr']['shortfall']:.3f}) "
                f"and the narrower test width ({test_width[recommended]:.3f} dex). Per-stratum "
                "coverage is not guaranteed, so the interval should not be read as a "
                f"{rm.INTERVAL_COVERAGE:.0%} guarantee in every mass range."
            )
    else:
        failed = test_checks[recommended]
        passed = test_checks[final_interval_method]
        interval_verdict = (
            f"{final_interval_method}, not {recommended}. {recommended} was narrower and better "
            "calibrated on CV, but on the test split its worst stratum "
            f"'{failed['worst_stratum']}' covers {failed['worst_coverage']:.3f} at a nominal "
            f"{rm.INTERVAL_COVERAGE:.2f} (shortfall {failed['shortfall']:.3f}), while "
            f"{final_interval_method} has a shortfall of {passed['shortfall']:.3f} in "
            f"'{passed['worst_stratum']}' (coverage {passed['worst_coverage']:.3f}). Coverage "
            "is the promise an interval makes, so it outranks the width advantage, and no "
            "model was refitted to obtain this check."
        )
    emit("### Interval verdict")
    emit(interval_verdict)
    emit(f"recommended interval method (final): {final_interval_method}")
    emit("")

    ood = rm.out_of_distribution(X_test, bounds)
    emit(f"test rows outside the training 1st-99th percentile bounds: {int(ood.sum())} of {len(test)}")

    # Final decision, stated plainly.
    chosen_test_mae = float(test_table.iloc[0]["MAE"])
    baseline_test_mae = float(test_table.iloc[1]["MAE"])
    if decision["chosen_kind"] == "learner":
        recommended_model = bool(chosen_test_mae < baseline_test_mae)
        if recommended_model:
            verdict = (
                f"RECOMMENDED: {chosen_name} beats the best baseline on both criteria - "
                f"CV MAE {decision['chosen_cv_mae']:.6f} is below the threshold "
                f"{decision['cv_threshold']:.6f}, and test MAE {chosen_test_mae:.6f} is below "
                f"{best_baseline_name}'s {baseline_test_mae:.6f}."
            )
        else:
            verdict = (
                f"NOT RECOMMENDED: {chosen_name} cleared the CV criterion but its test MAE "
                f"({chosen_test_mae:.6f}) is not below {best_baseline_name}'s "
                f"({baseline_test_mae:.6f}); the best baseline is kept."
            )
    else:
        recommended_model = False
        verdict = (
            "NOT RECOMMENDED: no learner cleared the CV threshold, so no learner is "
            f"recommended and the best baseline ({best_baseline_name}) remains the model."
        )

    emit("")
    emit("### Verdict")
    emit(verdict)
    emit("")
    chosen_stratum = strata_test[strata_test["model"] == "chosen model"].set_index("stratum")["MAE"]
    base_stratum = strata_test[
        strata_test["model"] == f"best baseline ({best_baseline_name})"
    ].set_index("stratum")["MAE"]
    for label in rm.MASS_STRATA:
        if label in chosen_stratum.index:
            emit(
                f"{label:<20} test MAE chosen = {chosen_stratum[label]:.6f}   "
                f"baseline = {base_stratum[label]:.6f}   "
                f"chosen better = {bool(chosen_stratum[label] < base_stratum[label])}"
            )
    emit("")

    # Artifact and meta.
    artifact = {
        "point_model_name": chosen_name,
        "point_model": point_model,
        "point_model_params": decision["chosen_params"],
        "features": list(rm.FEATURES),
        "target": rm.TARGET,
        "interval_coverage": rm.INTERVAL_COVERAGE,
        "recommended_interval_method": final_interval_method,
        "cv_interval_method": recommended,
        "interval_verdict": interval_verdict,
        "split_conformal_half_width_dex": float(q_split),
        "split_conformal_achieved_level": float(achieved_split),
        "cqr_quantile_dex": None if q_cqr is None else float(q_cqr),
        "cqr_lower_model": lower_model,
        "cqr_upper_model": upper_model,
        "feature_bounds": {key: list(value) for key, value in bounds.items()},
        "fit_on": "train split only",
        "calibrated_on": "calibration split",
        "test_host_hash": summary.test_host_hash,
        "library_versions": rm.package_versions(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    emit(
        f"model artifact written to {_relative(model_path)} "
        f"({model_path.stat().st_size} bytes)"
    )
    emit("")

    test_interval_tables = test_tables

    meta = rm.meta_stub()
    split_records = rm.split_table(split_df).to_dict("records")
    meta.update(
        {
            "pipeline_stage": 2,
            "data_file": _relative(data_path),
            "row_filters": {step.label: step.rows for step in steps},
            "split_sizes": {
                "rows": {r["split"]: int(r["rows"]) for r in split_records},
                "hosts": {r["split"]: int(r["hosts"]) for r in split_records},
                "rv_rows": {r["split"]: int(r["rv_rows"]) for r in split_records},
            },
            "test_host_hash": summary.test_host_hash,
            "test_host_count": len(summary.test_hosts),
            "cv_baselines": {
                name: {"kind": "baseline", "metrics": baseline_results[name]["metrics"]}
                for name in ("median", "power_law", "piecewise_power_law", "ridge")
            },
            "cv_learners": {
                name: {
                    "kind": "learner",
                    "metrics": block["metrics"],
                    "params": block["params"],
                    "trials": block["trials"],
                }
                for name, block in learner_results.items()
            },
            "cv_stratum_mae": stratum_table.to_dict("records"),
            "decision": {
                "rule": (
                    "learner CV MAE < best baseline CV MAE - best baseline CV SE AND "
                    "learner test MAE < best baseline test MAE"
                ),
                "best_baseline": best_baseline_name,
                "best_baseline_cv_mae": decision["best_baseline_cv_mae"],
                "best_baseline_cv_sd": decision["best_baseline_cv_sd"],
                "best_baseline_cv_se": decision["best_baseline_cv_se"],
                "cv_threshold": decision["cv_threshold"],
                "learners": decision["learners"],
                "chosen_model": chosen_name,
                "chosen_kind": decision["chosen_kind"],
                "chosen_params": decision["chosen_params"],
                "chosen_cv_mae": decision["chosen_cv_mae"],
                "recommended": recommended_model,
                "verdict": verdict,
            },
            "piecewise_breakpoints_mass_earth": list(rm.PIECEWISE_BREAKPOINTS),
            "intervals": {
                "nominal_coverage": rm.INTERVAL_COVERAGE,
                "selection_rule": (
                    "drop under-covering methods; then narrower mean width; then higher "
                    "coverage; a 0.5 percent width tie goes to split conformal"
                ),
                "cv_comparison": {
                    method: table.to_dict("records") for method, table in interval_cv.items()
                },
                "recommended_on_cv": recommended,
                "reason": reason,
                "cv_checks": interval_checks,
                "test_checks": test_checks,
                "final_recommended": final_interval_method,
                "verdict": interval_verdict,
                "split_conformal_half_width_dex": float(q_split),
                "split_conformal_achieved_level": float(achieved_split),
                "cqr_quantile_dex": None if q_cqr is None else float(q_cqr),
            },
            "permutation_importance": importance.drop(columns=["n_repeats"]).to_dict("records"),
            "permutation_importance_baseline_mae": baseline_mae,
            "feature_bounds": {key: list(value) for key, value in bounds.items()},
            "test_metrics": {
                "chosen_model": rm.regression_metrics(y_test, pred_test),
                f"best_baseline_{best_baseline_name}": rm.regression_metrics(
                    y_test, pred_baseline
                ),
                "per_stratum": strata_test.to_dict("records"),
            },
            "test_interval_coverage": {
                method: table.to_dict("records")
                for method, table in test_interval_tables.items()
            },
            "test_ood_rows": int(ood.sum()),
            "artifact_path": _relative(model_path),
            "test_evaluated": True,
            "test_evaluated_on": date.today().isoformat(),
        }
    )
    write_meta(meta, meta_path)
    emit(f"meta written to {_relative(meta_path)}")
    emit("")
    emit(
        "test_evaluated = true; a second stage-2 run is refused unless --force "
        f"(evaluated on {meta['test_evaluated_on']})."
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true", help="Re-evaluate the test split.")
    args = parser.parse_args()

    report = Report(args.report)
    if args.stage == 1:
        run_stage1(report, args.data, args.meta)
    else:
        run_stage2(report, args.data, args.meta, args.model, args.force)
    report(f"report written to {_relative(args.report)}")
    report.write()


if __name__ == "__main__":
    main()
