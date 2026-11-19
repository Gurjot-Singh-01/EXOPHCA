"""Shadow holdout: one unbiased evaluation of the radius recipe, released artifact untouched.

Why this exists
---------------
The published test-split figures (MAE 0.1177, R2 0.7931, coverage 0.9111) come
from rows that were consulted twice: once to choose the interval method, once to
confirm the learner. They are not a clean generalisation estimate, and
re-running ``scripts/train_radius_model.py`` cannot make them one - the split is
seeded and deterministic, so a re-run re-draws the same test hosts and repeats
the same selection.

This script evaluates the *recipe* on the one host set in the repository that no
model was ever fitted on: the calibration hosts. The released artifact records
that the point model was fitted on the train split and the conformal quantile on
the calibration split, and nothing was ever fitted on the calibration hosts'
rows. Their per-host results have never been published.

Design, in order
----------------
1. Re-derive the released training rows, reproduce the released host split, and
   assert the recomputed ``test_host_hash`` equals the one in
   ``models/radius_model_meta.json``. If it does not, stop.
2. Pool = the released train rows. Split the pool again by host, with a fixed
   seed and no test fraction, into pool-fit and pool-calibration hosts.
3. Re-run the release's own selection procedure inside the pool, by importing
   the trainer module and calling its functions - never a copy of the rule:
   ``run_baselines``, ``run_learners``, ``aggregate_cv_table``,
   ``stratum_cv_table``, ``decide``, then ``interval_cv_comparison`` and
   ``recommend_interval_method``.
4. Fit the chosen point model on the pool-fit rows and calibrate the conformal
   quantile on the pool-calibration rows, exactly as the trainer does.
5. Evaluate once on the holdout: the released calibration hosts.

What this is not
----------------
The calibration hosts produced the released model's conformal half-width
(0.29404 dex), so they are not untouched *by the released artifact*. This
measures the recipe on hosts no model was fitted on. It is not an external
distribution-shift test, and it does not retroactively make the published test
numbers unbiased.

Never written by this script: ``models/radius_model.joblib``,
``models/radius_model_meta.json``, anything under ``data/processed/`` or
``data/reference/``, ``exophca/**``, and every README figure other than the new
holdout subsection. Two new files are produced:
``models/radius_model_holdout.json`` and ``docs/progress/phase9_holdout.md``.

Usage::

    python scripts/holdout_evaluation.py            # refuses a second time
    python scripts/holdout_evaluation.py --force    # re-run, justified in the output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from exophca.ml import radius_model as rm  # noqa: E402
import train_radius_model as trainer  # noqa: E402  (imported, never modified)

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "planets.csv"
DEFAULT_META = PROJECT_ROOT / "models" / "radius_model_meta.json"
DEFAULT_OUT = PROJECT_ROOT / "models" / "radius_model_holdout.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "progress" / "phase9_holdout.md"

# Fixed before the run, recorded in the output. Choosing it after seeing a
# metric would defeat the point of the exercise.
SHADOW_SEED = 20261119
SHADOW_CALIBRATION_FRACTION = 0.20

# The files that implement the recipe. Their hash is recorded so a recipe
# changed after the number was seen is visible in the output.
RECIPE_FILES = (
    "exophca/ml/radius_model.py",
    "scripts/train_radius_model.py",
    "scripts/holdout_evaluation.py",
)


class Recorder:
    """Collects the trainer's printed output instead of sending it to a console."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        self.lines.append(str(message))

    def text(self) -> str:
        return "\n".join(self.lines)


def recipe_hash() -> str:
    """sha256 over the recipe's source files, in a fixed order."""
    digest = hashlib.sha256()
    for name in RECIPE_FILES:
        path = PROJECT_ROOT / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _r(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate the holdout when a result already exists.",
    )
    args = parser.parse_args()

    # ---- The lock, using the release's own rule -----------------------------
    existing = None
    if args.out.is_file():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        # Present the held-out file in the shape the release's own guard reads,
        # so the lock is the same rule rather than a second implementation.
        existing = {
            "test_evaluated": bool(previous.get("holdout_evaluated")),
            "test_evaluated_on": previous.get("holdout_evaluated_on"),
            "test_host_hash": previous.get("holdout", {}).get("host_hash"),
        }
    allowed, guard_message = rm.test_evaluation_allowed(existing, force=args.force)
    if not allowed:
        print(guard_message, file=sys.stderr)
        raise SystemExit(2)
    if guard_message:
        print(guard_message, file=sys.stderr)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))

    # ---- 1. Reproduce the released rows and split ---------------------------
    raw = rm.load_planets(args.data)
    selected, _steps = rm.filter_training_rows(raw)
    released_split, released_summary = rm.split_by_host(selected)

    recorded_hash = meta.get("test_host_hash")
    if released_summary.test_host_hash != recorded_hash:
        raise SystemExit(
            "STOP: the recomputed test-host hash does not match "
            f"models/radius_model_meta.json ({released_summary.test_host_hash} vs "
            f"{recorded_hash}). The data or the split code changed; the holdout "
            "would not describe the released model."
        )

    recorded_rows = (meta.get("split_sizes") or {}).get("rows") or {}
    recomputed_rows = {
        "train": int((released_split["split"] == "train").sum()),
        "calibration": int((released_split["split"] == "calibration").sum()),
        "test": int((released_split["split"] == "test").sum()),
    }
    if recomputed_rows != {key: int(value) for key, value in recorded_rows.items()}:
        raise SystemExit(
            "STOP: the recomputed split sizes do not match the released report: "
            f"{recomputed_rows} vs {recorded_rows}"
        )

    # ---- 2. The pool and its fit/calibration split --------------------------
    pool = released_split[released_split["split"] == "train"].copy()
    pool_split, pool_summary = rm.split_by_host(
        pool,
        seed=SHADOW_SEED,
        test_fraction=0.0,
        calibration_fraction=SHADOW_CALIBRATION_FRACTION,
    )
    pool_fit, pool_calibration, pool_empty = trainer.split_frames(pool_split)
    if len(pool_empty):
        raise SystemExit("STOP: the pool split produced test rows; it must not.")

    # ---- 3. Re-run the release's selection procedure inside the pool --------
    recorder = Recorder()
    emit = recorder

    emit("## Shadow holdout - recipe re-run inside the train pool")
    emit("")
    emit(f"released test_host_hash reproduced: {released_summary.test_host_hash}")
    emit(f"pool rows: {len(pool)}, hosts: {pool[rm.GROUP_COLUMN].nunique()}")
    emit(
        f"pool fit rows: {len(pool_fit)} ({pool_fit[rm.GROUP_COLUMN].nunique()} hosts), "
        f"pool calibration rows: {len(pool_calibration)} "
        f"({pool_calibration[rm.GROUP_COLUMN].nunique()} hosts)"
    )
    emit(f"shadow seed: {SHADOW_SEED}")
    emit("")

    baseline_results = trainer.run_baselines(emit, pool_split)
    learner_results = trainer.run_learners(emit, pool_split)
    cv_results = {**baseline_results, **learner_results}
    cv_table = trainer.aggregate_cv_table(emit, cv_results)
    trainer.stratum_cv_table(emit, cv_results)
    decision = trainer.decide(emit, cv_table, cv_results)

    chosen_name = decision["chosen_model"]
    if decision["chosen_kind"] == "learner":
        point_factory = lambda: rm.make_learner(chosen_name, decision["chosen_params"])
    else:
        point_factory = trainer.BASELINE_FACTORIES[chosen_name]

    X_fit = rm.build_features(pool_fit)
    y_fit = rm.build_target(pool_fit)
    m_fit = pool_fit[rm.MASS_COLUMN].to_numpy(dtype=float)
    g_fit = pool_fit[rm.GROUP_COLUMN].to_numpy()

    interval_cv = rm.interval_cv_comparison(X_fit, y_fit, m_fit, g_fit, point_factory)
    interval_recommended, interval_reason = rm.recommend_interval_method(
        interval_cv, rm.INTERVAL_COVERAGE
    )
    emit("")
    emit(f"interval method recommended inside the pool: {interval_recommended}")
    emit(f"reason: {interval_reason}")
    emit("")

    # ---- 4. Fit and calibrate on the pool -----------------------------------
    point_model = point_factory()
    point_model.fit(X_fit, y_fit)

    X_cal = rm.build_features(pool_calibration)
    y_cal = rm.build_target(pool_calibration)
    cal_residual = np.abs(point_model.predict(X_cal) - y_cal)
    q_split, achieved_split, k_split = rm.conformal_order_statistic(
        cal_residual, rm.INTERVAL_COVERAGE
    )

    cqr_lower_model = cqr_upper_model = None
    q_cqr = None
    if rm.LIGHTGBM_AVAILABLE:
        cqr_lower_model, cqr_upper_model = rm.fit_cqr(X_fit, y_fit, rm.INTERVAL_COVERAGE)
        cqr_scores = rm.cqr_conformity_scores(
            cqr_lower_model, cqr_upper_model, X_cal, y_cal
        )
        q_cqr, _, _ = rm.conformal_order_statistic(cqr_scores, rm.INTERVAL_COVERAGE)

    # ---- 5. The single holdout evaluation -----------------------------------
    holdout = released_split[released_split["split"] == "calibration"].copy()
    X_hold = rm.build_features(holdout)
    y_hold = rm.build_target(holdout)
    m_hold = holdout[rm.MASS_COLUMN].to_numpy(dtype=float)

    pred_hold = point_model.predict(X_hold)
    point_metrics = rm.regression_metrics(y_hold, pred_hold)
    per_stratum = rm.stratum_metrics(y_hold, pred_hold, m_hold)

    bounds = rm.feature_bounds(pool_fit)
    ood = rm.out_of_distribution(X_hold, bounds)

    intervals = {}
    lower_split, upper_split = rm.split_conformal_interval(pred_hold, q_split)
    intervals["split_conformal"] = rm.interval_table(y_hold, lower_split, upper_split, m_hold)
    if cqr_lower_model is not None:
        lower_cqr, upper_cqr = rm.cqr_interval(
            cqr_lower_model, cqr_upper_model, X_hold, q_cqr
        )
        intervals["cqr"] = rm.interval_table(y_hold, lower_cqr, upper_cqr, m_hold)

    interval_checks = {
        method: rm.interval_check(table, rm.INTERVAL_COVERAGE)
        for method, table in intervals.items()
    }

    baseline_name = decision["best_baseline"]
    baseline_model = trainer.BASELINE_FACTORIES[baseline_name]()
    baseline_model.fit(X_fit, y_fit)
    baseline_metrics = rm.regression_metrics(y_hold, baseline_model.predict(X_hold))

    emit("## Holdout metrics (single evaluation)")
    emit(f"rows: {len(holdout)}, hosts: {holdout[rm.GROUP_COLUMN].nunique()}")
    emit(f"point model: {chosen_name} ({decision['chosen_kind']})")
    emit(
        f"MAE {point_metrics['MAE']:.6f}  RMSE {point_metrics['RMSE']:.6f}  "
        f"R2 {point_metrics['R2']:.6f}  MedAPE {point_metrics['MedAPE']:.4f}"
    )
    emit(
        f"best baseline ({baseline_name}) MAE {baseline_metrics['MAE']:.6f}, "
        f"R2 {baseline_metrics['R2']:.6f}"
    )
    emit(f"out-of-distribution rows: {int(ood.sum())} of {len(holdout)}")
    for method, table in intervals.items():
        check = interval_checks[method]
        emit(
            f"{method}: coverage {check['overall']:.6f}, worst stratum "
            f"'{check['worst_stratum']}' {check['worst_coverage']:.6f}, mean width "
            f"{float(table.iloc[0]['mean_width_dex']):.6f} dex"
        )
    emit("")

    # ---- The written record ------------------------------------------------
    released_metrics = (meta.get("test_metrics") or {}).get("chosen_model") or {}
    released_intervals = meta.get("test_interval_coverage") or {}

    result = {
        "holdout_evaluated": True,
        "holdout_evaluated_on": date.today().isoformat(),
        "script": "scripts/holdout_evaluation.py",
        "recipe_hash": recipe_hash(),
        "recipe_files": list(RECIPE_FILES),
        "shadow_seed": SHADOW_SEED,
        "shadow_calibration_fraction": SHADOW_CALIBRATION_FRACTION,
        "python": sys.version.split()[0],
        "library_versions": rm.package_versions(),
        "released_test_host_hash": released_summary.test_host_hash,
        "reproduced_test_host_hash": released_summary.test_host_hash,
        "pool": {
            "description": "the released train split",
            "rows": int(len(pool)),
            "hosts": int(pool[rm.GROUP_COLUMN].nunique()),
            "fit_rows": int(len(pool_fit)),
            "fit_hosts": int(pool_fit[rm.GROUP_COLUMN].nunique()),
            "calibration_rows": int(len(pool_calibration)),
            "calibration_hosts": int(pool_calibration[rm.GROUP_COLUMN].nunique()),
            "pool_host_hash": pool_summary.test_host_hash,
        },
        "holdout": {
            "description": "the released calibration split - the only hosts no model was fitted on",
            "rows": int(len(holdout)),
            "hosts": int(holdout[rm.GROUP_COLUMN].nunique()),
            "host_hash": rm.host_list_hash(set(holdout[rm.GROUP_COLUMN])),
            "out_of_distribution_rows": int(ood.sum()),
        },
        "recipe_decision": decision,
        "interval_recommended_on_cv_in_pool": interval_recommended,
        "interval_reason_in_pool": interval_reason,
        "interval_cv_in_pool": {
            method: table.to_dict("records") for method, table in interval_cv.items()
        },
        "calibration": {
            "nominal_coverage": rm.INTERVAL_COVERAGE,
            "split_conformal_half_width_dex": float(q_split),
            "split_conformal_achieved_level": float(achieved_split),
            "split_conformal_order_statistic_k": int(k_split),
            "cqr_quantile_dex": None if q_cqr is None else float(q_cqr),
        },
        "holdout_point_metrics": point_metrics,
        "holdout_baseline_metrics": baseline_metrics,
        "holdout_baseline_name": baseline_name,
        "holdout_per_stratum": per_stratum.to_dict("records"),
        "holdout_interval_tables": {
            method: table.to_dict("records") for method, table in intervals.items()
        },
        "holdout_interval_checks": interval_checks,
        "released_test_metrics_for_comparison": {
            "chosen_model": released_metrics,
            "split_conformal_coverage": released_intervals.get("split_conformal"),
            "out_of_distribution_rows": meta.get("test_ood_rows"),
        },
        "caveat": (
            "The calibration hosts produced the released model's conformal half-width "
            "(0.29404 dex), so they are not untouched by the released artifact. This "
            "measures the recipe on hosts no model was fitted on. It is not an external "
            "distribution-shift test, and it does not make the published test numbers "
            "unbiased."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # A human-readable copy, under the gitignored working-record tree.
    report_lines = [
        "# Phase 9 - shadow holdout",
        "",
        f"Generated {result['holdout_evaluated_on']} by `scripts/holdout_evaluation.py`.",
        f"Recipe hash: `{result['recipe_hash']}`.",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Holdout rows / hosts | {len(holdout)} / {holdout[rm.GROUP_COLUMN].nunique()} |",
        f"| MAE (dex) | {point_metrics['MAE']:.4f} |",
        f"| RMSE (dex) | {point_metrics['RMSE']:.4f} |",
        f"| R2 | {point_metrics['R2']:.4f} |",
        f"| MedAPE (%) | {point_metrics['MedAPE']:.1f} |",
        f"| Best baseline MAE (dex) | {baseline_metrics['MAE']:.4f} |",
        f"| Out-of-distribution rows | {int(ood.sum())} |",
        "",
        "## The trainer's own output, captured",
        "",
        "```text",
        recorder.text(),
        "```",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines), encoding="utf-8")

    print(recorder.text())
    print(f"holdout json written to {args.out}")
    print(f"working record written to {args.report}")
    for method, check in interval_checks.items():
        print(
            f"HOLDOUT {method}: MAE {point_metrics['MAE']:.4f} R2 "
            f"{point_metrics['R2']:.4f} coverage {check['overall']:.4f} "
            f"worst {check['worst_coverage']:.4f} ({check['worst_stratum']})"
        )


if __name__ == "__main__":
    main()
