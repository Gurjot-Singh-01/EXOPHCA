"""Write ``data/processed/radius_estimates.csv`` from the saved radius model.

The app never loads ``models/radius_model.joblib``: this script runs the model
once, offline, and writes a small table that the Planet Detail page reads. The
selection is every planet whose archive radius is ``modelled`` or
``derived_unclear`` - the rows the archive had to estimate rather than measure.

A selected planet that lacks a usable feature is still written, with NaN
estimates and its reason, so the omission is visible in the artifact instead of
silently shrinking the table. The counts printed below separate the two groups.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from exophca.ml.predict import load_radius_artifact, predict_radius  # noqa: E402

PLANETS_PATH = ROOT / "data" / "processed" / "planets.csv"
ARTIFACT_PATH = ROOT / "models" / "radius_model.joblib"
OUTPUT_PATH = ROOT / "data" / "processed" / "radius_estimates.csv"

SELECTED_PROVENANCE = ("modelled", "derived_unclear")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    planets = pd.read_csv(PLANETS_PATH)
    selected = planets[planets["radius_provenance"].isin(SELECTED_PROVENANCE)].copy()
    selected = selected.sort_values("pl_name", kind="stable").reset_index(drop=True)

    artifact, warnings = load_radius_artifact(ARTIFACT_PATH)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if artifact is None:
        print("NOT DONE: the radius artifact could not be loaded")
        return 1

    estimates = predict_radius(selected, artifact)
    estimates = estimates.round(
        {"radius_estimate": 6, "radius_lower": 6, "radius_upper": 6}
    )
    estimates.to_csv(OUTPUT_PATH, index=False)

    predicted = estimates["radius_estimate"].notna()
    skipped = estimates[~predicted]

    print(f"selection: radius_provenance in {SELECTED_PROVENANCE}")
    print(f"selected: {len(selected)}")
    for provenance, count in selected["radius_provenance"].value_counts().items():
        print(f"  {provenance}: {count}")
    print(f"written: {OUTPUT_PATH.relative_to(ROOT).as_posix()}")
    print(f"predicted: {int(predicted.sum())}")
    print(f"skipped: {len(skipped)}")
    for reason, count in skipped["reason"].value_counts().items():
        print(f"  {reason}: {count}")
    print(f"mass_is_minimum: {int(estimates.loc[predicted, 'mass_is_minimum'].sum())}")
    print(f"ood_flag: {int(estimates.loc[predicted, 'ood_flag'].sum())}")
    print(
        "interval_straddles_1p6: "
        f"{int(estimates.loc[predicted, 'interval_straddles_1p6'].sum())}"
    )
    print(f"sha256: {sha256(OUTPUT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
