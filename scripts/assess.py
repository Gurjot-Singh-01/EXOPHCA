"""Script to run uncertainty propagation and rule-based assessment on planets and TOIs."""

import time
from pathlib import Path
import pandas as pd
import numpy as np

from exophca.science.habitable_zone import classify_hz
from exophca.science.uncertainty import propagate_uncertainties
from exophca.science.assessment import assess_planets, DISCLAIMER


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    processed_dir = data_dir / 'processed'

    planets_csv = processed_dir / 'planets.csv'
    toi_csv = processed_dir / 'toi.csv'

    print("=== Loading processed datasets ===")
    df_planets = pd.read_csv(planets_csv)
    df_toi = pd.read_csv(toi_csv)

    # Filter to scorable rows
    df_planets_scorable = df_planets[df_planets['scorable'] == True].copy()
    df_toi_scorable = df_toi[df_toi['scorable'] == True].copy()

    # Ensure hz_class is computed for both if not present
    if 'hz_class' not in df_planets_scorable.columns:
        df_planets_scorable['hz_class'] = [
            classify_hz(row['pl_insol'], row['st_teff'])
            for _, row in df_planets_scorable.iterrows()
        ]
    if 'hz_class' not in df_toi_scorable.columns:
        df_toi_scorable['hz_class'] = [
            classify_hz(row['pl_insol'], row['st_teff'])
            for _, row in df_toi_scorable.iterrows()
        ]

    print(f"\n=== Running Monte Carlo uncertainty propagation on {len(df_planets_scorable)} planets ===")
    t0 = time.time()
    df_planets_mc = propagate_uncertainties(df_planets_scorable, n_draws=5000, seed=42, is_toi=False)
    t1 = time.time()
    planets_runtime = t1 - t0

    print(f"\n=== Running Monte Carlo uncertainty propagation on {len(df_toi_scorable)} TOIs ===")
    t2 = time.time()
    df_toi_mc = propagate_uncertainties(df_toi_scorable, n_draws=5000, seed=42, is_toi=True)
    toi_runtime = time.time() - t2

    print(f"Total MC runtime: planets={planets_runtime:.2f}s, TOIs={toi_runtime:.2f}s")

    print("\n=== Running rule-based assessment ===")
    df_planets_assessed = assess_planets(df_planets_mc, is_toi=False)
    df_toi_assessed = assess_planets(df_toi_mc, is_toi=True)

    # Select exact requested columns in order
    planets_cols = [
        'pl_name', 'status', 'st_teff', 'star_class', 'pl_rade', 'pl_insol',
        'radius_provenance', 'uncertainty_status', 'base_score', 'base_score_median',
        'base_score_p16', 'base_score_p84', 'hz_class', 'p_conservative_hz',
        'p_any_hz', 'p_radius_le_1p6', 'p_rocky_and_conservative', 'size_regime',
        'tier', 'evidence_quality', 'boundary_sensitive', 'criteria'
    ]

    toi_cols = [
        'toi', 'status', 'st_teff', 'pl_rade', 'pl_insol',
        'uncertainty_status', 'base_score', 'base_score_median',
        'base_score_p16', 'base_score_p84', 'hz_class', 'p_conservative_hz',
        'p_any_hz', 'p_radius_le_1p6', 'p_rocky_and_conservative', 'size_regime',
        'tier', 'evidence_quality', 'boundary_sensitive', 'criteria'
    ]

    # Ensure all columns exist
    for c in planets_cols:
        if c not in df_planets_assessed.columns:
            df_planets_assessed[c] = np.nan
    for c in toi_cols:
        if c not in df_toi_assessed.columns:
            df_toi_assessed[c] = np.nan

    planets_out_path = processed_dir / 'planets_assessed.csv'
    toi_out_path = processed_dir / 'toi_assessed.csv'

    df_planets_assessed[planets_cols].to_csv(planets_out_path, index=False)
    df_toi_assessed[toi_cols].to_csv(toi_out_path, index=False)
    print(f"Saved assessed outputs to {planets_out_path} and {toi_out_path}")

    # Now generate the exact raw output for docs/progress/phase4_assessment.md
    target_planets = [
        "Teegarden's Star b",
        "TOI-700 d",
        "TRAPPIST-1 e",
        "Kepler-452 b",
        "Proxima Centauri b",  # note check exact spelling in dataset
        "LP 890-9 c"
    ]

    # Let's inspect how Proxima Centauri b is named in df_planets_assessed
    matching_names = []
    for tp in target_planets:
        matches = df_planets_assessed[df_planets_assessed['pl_name'].str.contains(tp.split()[0], case=False, na=False)]
        if len(matches) > 0:
            matching_names.append(matches.iloc[0]['pl_name'])
        else:
            matching_names.append(tp)

    print("\n--- Generating phase4_assessment.md content ---")
    report_lines = []
    report_lines.append(DISCLAIMER)
    report_lines.append("")

    report_lines.append("### (a) Tier and Evidence Quality Value Counts")
    report_lines.append("\nplanets_assessed.csv - Tier value_counts:")
    report_lines.append(df_planets_assessed['tier'].value_counts().to_string())
    report_lines.append("\nplanets_assessed.csv - Evidence Quality value_counts:")
    report_lines.append(df_planets_assessed['evidence_quality'].value_counts().to_string())

    report_lines.append("\ntoi_assessed.csv - Tier value_counts:")
    report_lines.append(df_toi_assessed['tier'].value_counts().to_string())
    report_lines.append("\ntoi_assessed.csv - Evidence Quality value_counts:")
    report_lines.append(df_toi_assessed['evidence_quality'].value_counts().to_string())

    report_lines.append("\n### (b) Crosstab of Tier by Evidence Quality for planets_assessed.csv")
    ct_planets = pd.crosstab(df_planets_assessed['tier'], df_planets_assessed['evidence_quality'], margins=True)
    report_lines.append(ct_planets.to_string())

    report_lines.append("\n### (c) Uncertainty Status Value Counts")
    report_lines.append("\nplanets_assessed.csv - uncertainty_status value_counts:")
    report_lines.append(df_planets_assessed['uncertainty_status'].value_counts().to_string())
    report_lines.append("\ntoi_assessed.csv - uncertainty_status value_counts:")
    report_lines.append(df_toi_assessed['uncertainty_status'].value_counts().to_string())

    report_lines.append("\n### (d) Assessment Columns for Key Planets")
    key_subset = df_planets_assessed[df_planets_assessed['pl_name'].isin([
        "Teegarden's Star b", "TOI-700 d", "TRAPPIST-1 e", "Kepler-452 b", "Proxima Centauri b", "LP 890-9 c"
    ])]
    if len(key_subset) == 0:
        # Try fuzzy or partial matches
        key_subset = df_planets_assessed[df_planets_assessed['pl_name'].str.contains("Teegarden|TOI-700|TRAPPIST-1 e|Kepler-452|Proxima|LP 890-9 c", regex=True, na=False)]

    report_lines.append(key_subset[planets_cols].to_string(index=False))

    report_lines.append(f"\n### (e) Monte Carlo Runtime")
    report_lines.append(f"Planets MC runtime (5688 planets, 5000 draws): {planets_runtime:.2f} seconds.")
    report_lines.append(f"TOIs MC runtime (4992 TOIs, 5000 draws): {toi_runtime:.2f} seconds.")

    report_lines.append(f"\n### (f) Number of Planets with boundary_sensitive True")
    b_sens_planets = int(df_planets_assessed['boundary_sensitive'].sum())
    b_sens_tois = int(df_toi_assessed['boundary_sensitive'].sum())
    report_lines.append(f"planets_assessed boundary_sensitive=True count: {b_sens_planets}")
    report_lines.append(f"toi_assessed boundary_sensitive=True count: {b_sens_tois}")

    doc_path = project_root / 'docs' / 'progress' / 'phase4_assessment.md'
    doc_path.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"Saved assessment report to {doc_path}")


if __name__ == '__main__':
    main()
