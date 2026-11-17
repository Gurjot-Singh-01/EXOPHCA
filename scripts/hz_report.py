"""Script for Phase 3 Habitable Zone report and verification tasks."""

from pathlib import Path
import numpy as np
import pandas as pd

from exophca.science.habitable_zone import hz_limits, classify_hz


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    processed_dir = data_dir / 'processed'
    reference_dir = data_dir / 'reference'
    docs_progress_dir = project_root / 'docs' / 'progress'

    planets_csv = processed_dir / 'planets.csv'
    df_planets = pd.read_csv(planets_csv)

    # Task 1: Read planets.csv, keep rows where scorable is True, classify with classify_hz
    scorable_mask = df_planets['scorable'] == True
    df_scorable = df_planets[scorable_mask].copy()

    hz_classes = []
    for _, row in df_scorable.iterrows():
        hz_classes.append(classify_hz(row['pl_insol'], row['st_teff']))
    df_scorable['hz_class'] = hz_classes

    hz_output_columns = [
        'pl_name', 'st_teff', 'pl_insol', 'pl_rade',
        'radius_provenance', 'base_score', 'star_class', 'hz_class'
    ]
    df_planets_hz = df_scorable[hz_output_columns]
    planets_hz_path = processed_dir / 'planets_hz.csv'
    df_planets_hz.to_csv(planets_hz_path, index=False)
    print(f"Task 1: Saved {planets_hz_path} with {len(df_planets_hz)} rows.")

    # Task 2: Oracle comparison (a)
    oracle_csv = reference_dir / 'hz_fluxes_oracle.csv'
    df_oracle = pd.read_csv(oracle_csv)

    grid_teffs = [2600, 3600, 4600, 5800, 6600, 7200]
    limits_map = {
        'recent_venus': 'optimistic_inner',
        'runaway_greenhouse': 'conservative_inner',
        'maximum_greenhouse': 'conservative_outer',
        'early_mars': 'optimistic_outer'
    }

    print("\n=== Task 2: Oracle Comparison Table ===")
    print(f"{'Teff':<6} | {'Limit Name':<22} | {'Module Value':<14} | {'Oracle Value':<14} | {'Abs Diff':<14}")
    print("-" * 78)

    max_abs_diff = 0.0
    all_oracle_teffs = df_oracle['teff'].values
    all_oracle_limits = {
        'recent_venus': df_oracle['recent_venus'].values,
        'runaway_greenhouse': df_oracle['runaway_greenhouse'].values,
        'maximum_greenhouse': df_oracle['maximum_greenhouse'].values,
        'early_mars': df_oracle['early_mars'].values,
    }

    # Compute max abs diff across all 26 grid rows
    for i, t in enumerate(all_oracle_teffs):
        mod_limits = hz_limits(t)
        for col, mod_key in limits_map.items():
            mod_val = mod_limits[mod_key]
            oracle_val = all_oracle_limits[col][i]
            diff = abs(mod_val - oracle_val)
            if diff > max_abs_diff:
                max_abs_diff = diff

    # Print requested subset table
    for t in grid_teffs:
        mod_limits = hz_limits(float(t))
        oracle_row = df_oracle[df_oracle['teff'] == t].iloc[0]
        for col, mod_key in limits_map.items():
            mod_val = mod_limits[mod_key]
            oracle_val = oracle_row[col]
            diff = abs(mod_val - oracle_val)
            print(f"{t:<6} | {col:<22} | {mod_val:<14.6f} | {oracle_val:<14.6f} | {diff:<14.2e}")

    print(f"\nMaximum absolute difference across all 26 grid temperatures and 4 limits: {max_abs_diff:.2e}")

    # Task 3: (b) hz_class value_counts, crosstab, missing/out-of-range stats
    print("\n=== Task 3: Scorable Planets Habitable Zone Summary ===")
    print("hz_class value_counts (scorable planets):")
    print(df_scorable['hz_class'].value_counts().to_string())

    print("\nCrosstab of star_class by hz_class:")
    ct = pd.crosstab(df_scorable['star_class'], df_scorable['hz_class'], margins=True)
    print(ct.to_string())

    # Check st_teff missing, < 2600, > 7200 on scorable planets
    missing_teff = df_scorable['st_teff'].isna().sum()
    below_2600 = df_scorable['st_teff'] < 2600.0
    above_7200 = df_scorable['st_teff'] > 7200.0
    count_below_2600 = below_2600.sum()
    count_above_7200 = above_7200.sum()

    print(f"\nScorable planets with st_teff missing: {missing_teff}")
    print(f"Scorable planets with st_teff below 2600 K: {count_below_2600}")
    print(f"Scorable planets with st_teff above 7200 K: {count_above_7200}")

    if count_below_2600 > 0:
        print("Planet names with st_teff below 2600 K:")
        for name in df_scorable[below_2600]['pl_name'].tolist():
            print(f" - {name}")
    else:
        print("Planet names with st_teff below 2600 K: None")

    # Task 4: (c) Old gate planets (scorable AND 0.35 <= pl_insol <= 1.1 AND pl_rade < 1.75)
    print("\n=== Task 4: Old Gate Planets (22 Planets) ===")
    old_gate_mask = (
        df_scorable['scorable']
        & (df_scorable['pl_insol'] >= 0.35)
        & (df_scorable['pl_insol'] <= 1.1)
        & (df_scorable['pl_rade'] < 1.75)
    )
    df_old_gate = df_scorable[old_gate_mask].sort_values(by='base_score', ascending=False).copy()
    print(f"Count of planets satisfying old gate: {len(df_old_gate)}")

    old_gate_display_cols = [
        'pl_name', 'st_teff', 'pl_insol', 'pl_rade',
        'radius_provenance', 'base_score', 'hz_class'
    ]
    print(df_old_gate[old_gate_display_cols].to_string(index=False))

    print("\nValue counts of hz_class for these old gate planets:")
    print(df_old_gate['hz_class'].value_counts().to_string())


if __name__ == '__main__':
    main()
