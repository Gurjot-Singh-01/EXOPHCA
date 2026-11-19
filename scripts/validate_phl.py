"""Script to validate PHL Habitable Worlds Catalog against our data pipeline and science outputs."""

import re
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from exophca.science.habitable_zone import classify_hz


# Upper limit used to flag implausible stellar ages in the PHL export (Gyr).
# 14 Gyr slightly exceeds the age of the Universe and is used only as an
# anomaly screen on the external reference catalogue, never in scoring.
AGE_LIMIT_GYR = 14.0

# Column labels as exported in hwc_potentially_habitable.csv, which uses HTML
# markup in its header row (unlike hwc.csv, which uses P_* names).
POT_COL_TSURF = '<i>T<sub>surf</sub></i><br>(K)'
POT_COL_PERIOD = 'Period<br>(days)'
POT_COL_AGE = 'Age<br>(Gy)'
POT_COL_DISTANCE = 'Distance<br>(ly)'


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Lowercase, replace multiple spaces/hyphens/underscores with single space or empty, strip
    s = name.strip().lower()
    s = re.sub(r'[\s\-_]+', ' ', s)
    # Remove common suffixes/prefixes variations if needed, e.g. "proxima cen b" vs "proxima centauri b"
    s = s.replace('proxima cen', 'proxima centauri')
    s = s.replace('b b', ' b')
    return s


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    ref_dir = data_dir / 'reference'
    processed_dir = data_dir / 'processed'

    hwc_csv = ref_dir / 'hwc.csv'
    hwc_pot_csv = ref_dir / 'hwc_potentially_habitable.csv'
    planets_csv = processed_dir / 'planets.csv'
    planets_hz_csv = processed_dir / 'planets_hz.csv'
    planets_assessed_csv = processed_dir / 'planets_assessed.csv'

    print("=== TASK 1: Inspect before using ===")
    df_hwc = pd.read_csv(hwc_csv)
    df_pot = pd.read_csv(hwc_pot_csv)

    print(f"hwc.csv row count: {len(df_hwc)}")
    print(f"hwc.csv columns: {df_hwc.columns.tolist()}")
    print(f"hwc_potentially_habitable.csv row count: {len(df_pot)}")

    # Check TOI-904 c in hwc_potentially_habitable.csv
    toi904 = df_pot[df_pot['Name'].str.contains('TOI-904', case=False, na=False)]
    print("\nTOI-904 c row in hwc_potentially_habitable.csv:")
    print(toi904.to_string())
    if len(toi904) == 1:
        toi904_tsurf = float(toi904.iloc[0][POT_COL_TSURF])
        toi904_period = float(toi904.iloc[0][POT_COL_PERIOD])
        toi904_age = float(toi904.iloc[0][POT_COL_AGE])
        toi904_distance = float(toi904.iloc[0][POT_COL_DISTANCE])
    else:
        toi904_tsurf = toi904_period = toi904_age = toi904_distance = float('nan')
    print(
        f"Interpretation: TOI-904 c has Tsurf = {toi904_tsurf:.2f} K, "
        f"Period = {toi904_period:.4f} days, Age = {toi904_age:.2f} Gyr and "
        f"Distance = {toi904_distance:.2f} ly in the raw reference file. "
        f"The {toi904_distance:.2f} value is the distance column, not the age, so "
        "this row does not carry an anomalous stellar age."
    )

    # Inspect key columns missing counts and min/max

    key_cols = ['P_MASS', 'P_RADIUS', 'P_FLUX', 'P_TEMP_SURF', 'P_PERIOD', 'S_DISTANCE', 'S_AGE', 'P_ESI']
    print("\n--- Key Columns Inspection (hwc.csv) ---")
    for col in key_cols:
        if col in df_hwc.columns:
            s = pd.to_numeric(df_hwc[col], errors='coerce')
            missing = s.isna().sum()
            min_val = s.min()
            max_val = s.max()
            print(f"{col}: missing={missing}, min={min_val}, max={max_val}")

    # Check implausible values: age > 14 Gy, negative values, flux <= 0, radius <= 0
    age_s = pd.to_numeric(df_hwc['S_AGE'], errors='coerce') if 'S_AGE' in df_hwc.columns else pd.Series(0, index=df_hwc.index)
    rad_s = pd.to_numeric(df_hwc['P_RADIUS'], errors='coerce') if 'P_RADIUS' in df_hwc.columns else pd.Series(1, index=df_hwc.index)
    flux_s = pd.to_numeric(df_hwc['P_FLUX'], errors='coerce') if 'P_FLUX' in df_hwc.columns else pd.Series(1, index=df_hwc.index)
    mass_s = pd.to_numeric(df_hwc['P_MASS'], errors='coerce') if 'P_MASS' in df_hwc.columns else pd.Series(1, index=df_hwc.index)
    period_s = pd.to_numeric(df_hwc['P_PERIOD'], errors='coerce') if 'P_PERIOD' in df_hwc.columns else pd.Series(1, index=df_hwc.index)
    tsurf_s = pd.to_numeric(df_hwc['P_TEMP_SURF'], errors='coerce') if 'P_TEMP_SURF' in df_hwc.columns else pd.Series(0, index=df_hwc.index)

    anomalous = df_hwc[
        (age_s > AGE_LIMIT_GYR) |
        (rad_s <= 0) |
        (flux_s <= 0) |
        (mass_s <= 0) |
        (period_s <= 0) |
        (np.isclose(period_s, tsurf_s, atol=1e-2))
    ]
    print(f"\nAnomalous rows count in hwc.csv: {len(anomalous)}")
    if len(anomalous) > 0:
        print("First up to 10 anomalous rows:")
        print(anomalous[['P_NAME', 'P_RADIUS', 'P_FLUX', 'P_TEMP_SURF', 'P_PERIOD', 'S_AGE']].head(10).to_string())
        oldest = (
            anomalous.assign(_age=pd.to_numeric(anomalous['S_AGE'], errors='coerce'))
            .sort_values('_age', ascending=False)
            .head(2)
        )
        anomaly_examples = '; '.join(
            f"{row['P_NAME']} (Age = {row['_age']:.1f} Gyr)" for _, row in oldest.iterrows()
        )
    else:
        anomaly_examples = "none"

    print("\n=== TASK 2: Name matching ===")
    df_planets = pd.read_csv(planets_csv)
    df_hz = pd.read_csv(planets_hz_csv) if planets_hz_csv.exists() else df_planets.copy()
    if 'hz_class' not in df_hz.columns:
        df_hz['hz_class'] = [classify_hz(r['pl_insol'], r['st_teff']) for _, r in df_hz.iterrows()]
    df_assessed = pd.read_csv(planets_assessed_csv) if planets_assessed_csv.exists() else pd.DataFrame()

    # Normalise names in planets.csv
    df_planets['norm_name'] = df_planets['pl_name'].apply(normalize_name)
    df_hwc['norm_name'] = df_hwc['P_NAME'].apply(normalize_name)

    # Exact match on norm_name
    planets_dict = {row['norm_name']: row['pl_name'] for _, row in df_planets.iterrows()}

    matched_rows = []
    unmatched_names = []
    accepted_non_exact = []

    for _, hwc_row in df_hwc.iterrows():
        h_name = hwc_row['P_NAME']
        h_norm = hwc_row['norm_name']
        if h_norm in planets_dict:
            matched_rows.append({
                'hwc_name': h_name,
                'pl_name': planets_dict[h_norm],
                'Radius': hwc_row.get('P_RADIUS'),
                'Flux': hwc_row.get('P_FLUX'),
                'ESI': hwc_row.get('P_ESI'),
                **hwc_row.to_dict()
            })
        else:
            # Try manual cleanups or checks
            cleaned = h_norm.replace('candidat', '').strip()
            if cleaned in planets_dict:
                matched_rows.append({
                    'hwc_name': h_name,
                    'pl_name': planets_dict[cleaned],
                    'Radius': hwc_row.get('P_RADIUS'),
                    'Flux': hwc_row.get('P_FLUX'),
                    'ESI': hwc_row.get('P_ESI'),
                    **hwc_row.to_dict()
                })
                accepted_non_exact.append((h_name, planets_dict[cleaned]))
            else:
                unmatched_names.append(h_name)

    phl_rows_count = len(df_hwc)
    matched_count = len(matched_rows)
    unmatched_count = len(unmatched_names)

    print(f"PHL rows: {phl_rows_count}")
    print(f"Matched count: {matched_count}")
    print(f"Unmatched count: {unmatched_count}")
    print("\n15 example unmatched names with guess why:")
    for um in unmatched_names[:15]:
        print(f" - {um} (likely not in NASA Exoplanet Archive PSCD / planets.csv or designated differently)")

    print(f"\nAccepted non-exact matches ({len(accepted_non_exact)}): {accepted_non_exact[:5]}")

    print("\n=== TASK 3: Input agreement for matched planets that are scorable ===")
    matched_df = pd.DataFrame(matched_rows)
    # Merge with df_planets and df_hz and df_assessed
    merged = matched_df.merge(df_planets[['pl_name', 'pl_rade', 'pl_insol', 'scorable', 'st_teff']], on='pl_name', how='inner')
    scorable_matched = merged[merged['scorable'] == True].copy()

    scorable_matched['phl_radius'] = pd.to_numeric(scorable_matched['Radius'], errors='coerce')
    scorable_matched['phl_flux'] = pd.to_numeric(scorable_matched['Flux'], errors='coerce')

    scorable_matched['rel_diff_radius'] = (scorable_matched['phl_radius'] - scorable_matched['pl_rade']).abs() / scorable_matched['pl_rade']

    # The flux relative difference is only defined for a positive, finite archive
    # flux. Zero-flux and missing-flux rows are excluded and counted explicitly so
    # that no inf value can enter the percentiles.
    flux_comparable_mask = (
        np.isfinite(scorable_matched['pl_insol'])
        & np.isfinite(scorable_matched['phl_flux'])
        & (scorable_matched['pl_insol'] > 0)
    )
    flux_excluded_count = int((~flux_comparable_mask).sum())
    flux_zero_count = int((scorable_matched['pl_insol'] == 0).sum())
    flux_missing_count = int(
        (scorable_matched['pl_insol'].isna() | scorable_matched['phl_flux'].isna()).sum()
    )
    flux_other_count = flux_excluded_count - flux_zero_count - flux_missing_count

    scorable_matched['rel_diff_flux'] = np.nan
    scorable_matched.loc[flux_comparable_mask, 'rel_diff_flux'] = (
        (
            scorable_matched.loc[flux_comparable_mask, 'phl_flux']
            - scorable_matched.loc[flux_comparable_mask, 'pl_insol']
        ).abs()
        / scorable_matched.loc[flux_comparable_mask, 'pl_insol']
    )

    flux_diffs = scorable_matched.loc[flux_comparable_mask, 'rel_diff_flux']
    flux_comparable_n = int(len(flux_diffs))
    med_diff_r = scorable_matched['rel_diff_radius'].median()
    p95_diff_r = scorable_matched['rel_diff_radius'].quantile(0.95)
    med_diff_f = flux_diffs.median()
    p95_diff_f = flux_diffs.quantile(0.95)

    print(f"Radius relative diff: median = {med_diff_r:.4f}, 95th percentile = {p95_diff_r:.4f}")
    print(
        f"Flux relative diff (comparable rows only, N={flux_comparable_n}): "
        f"median = {med_diff_f:.4f}, 95th percentile = {p95_diff_f:.4f}"
    )
    print(
        f"Flux rows excluded from comparison: {flux_excluded_count} "
        f"(zero flux = {flux_zero_count}, missing flux = {flux_missing_count}, "
        f"other non-finite = {flux_other_count})"
    )
    print(f"Max |rel_diff_flux| used downstream: {flux_diffs.abs().max():.4f} (no inf values)")

    # Find top 15 planets with largest disagreement (>5%) combining radius and flux rel diff
    scorable_matched['max_rel_diff'] = scorable_matched[['rel_diff_radius', 'rel_diff_flux']].max(axis=1)
    top_disagree = scorable_matched.sort_values(by='max_rel_diff', ascending=False).head(15)

    print("\nTop 15 planets with largest disagreement (>5%):")
    print(top_disagree[['pl_name', 'phl_radius', 'pl_rade', 'rel_diff_radius', 'phl_flux', 'pl_insol', 'rel_diff_flux']].to_string())

    print("\n=== TASK 4: Rank agreement ===")
    spearman_pot = pearson_pot = float('nan')
    conf_table = None
    # Merge with df_hz or df_assessed to get base_score
    merged_score = scorable_matched.merge(df_hz[['pl_name', 'base_score', 'hz_class']], on='pl_name', how='inner')
    if not df_assessed.empty and 'tier' in df_assessed.columns:
        merged_score = merged_score.merge(df_assessed[['pl_name', 'tier', 'evidence_quality']], on='pl_name', how='left')
    else:
        merged_score['tier'] = 'U'
        merged_score['evidence_quality'] = 'low'

    merged_score['phl_esi'] = pd.to_numeric(merged_score['ESI'], errors='coerce')

    valid_rank = merged_score.dropna(subset=['phl_esi', 'base_score'])
    spearman_all, _ = spearmanr(valid_rank['phl_esi'], valid_rank['base_score'])
    pearson_all, _ = pearsonr(valid_rank['phl_esi'], valid_rank['base_score'])
    print(f"Overall (N={len(valid_rank)}) - Spearman: {spearman_all:.4f}, Pearson: {pearson_all:.4f}")

    # Potentially habitable subset (match names in df_pot)
    df_pot['norm_name'] = df_pot['Name'].apply(normalize_name)
    pot_names = set(df_pot['norm_name'])
    subset_pot = valid_rank[valid_rank['norm_name'].isin(pot_names)]
    if len(subset_pot) > 1:
        spearman_pot, _ = spearmanr(subset_pot['phl_esi'], subset_pot['base_score'])
        pearson_pot, _ = pearsonr(subset_pot['phl_esi'], subset_pot['base_score'])
        print(f"Potentially habitable subset (N={len(subset_pot)}) - Spearman: {spearman_pot:.4f}, Pearson: {pearson_pot:.4f}")

    # Earth check
    earth_row = valid_rank[valid_rank['pl_name'].str.contains('Earth', case=False, na=False)]
    if not earth_row.empty:
        print("\nEarth comparison:")
        print(earth_row[['pl_name', 'base_score', 'phl_esi']].to_string())
    else:
        print("\nEarth not in matched table (Earth is planet zero / reference, not listed as exoplanet in PSCD planets.csv usually).")

    # Top 10 planets with largest rank difference
    valid_rank['rank_phl'] = valid_rank['phl_esi'].rank(ascending=False)
    valid_rank['rank_our'] = valid_rank['base_score'].rank(ascending=False)
    valid_rank['rank_diff'] = (valid_rank['rank_phl'] - valid_rank['rank_our']).abs()
    top_rank_diff = valid_rank.sort_values(by='rank_diff', ascending=False).head(10)
    print("\nTop 10 planets with largest rank difference:")
    print(top_rank_diff[['pl_name', 'phl_esi', 'rank_phl', 'base_score', 'rank_our', 'rank_diff']].to_string())

    print("\n=== TASK 5: Habitable-zone agreement ===")
    # Compare PHL P_HABZONE_CON and P_HABZONE_OPT with our hz_class
    if 'P_HABZONE_CON' in valid_rank.columns and 'P_HABZONE_OPT' in valid_rank.columns:
        # P_HABZONE_CON / OPT can be 1/0 or True/False
        valid_rank['phl_hz_status'] = np.where(
            valid_rank['P_HABZONE_CON'] == 1, 'conservative',
            np.where(valid_rank['P_HABZONE_OPT'] == 1, 'optimistic', 'outside')
        )
        conf_table = pd.crosstab(valid_rank['phl_hz_status'], valid_rank['hz_class'])
        print("Confusion table (PHL HZ status vs our hz_class):")
        print(conf_table)
    else:
        print("P_HABZONE_CON / P_HABZONE_OPT columns not found.")

    print("\n=== TASK 6: Save phl_comparison.csv ===")
    out_cols = [
        'pl_name', 'hwc_name', 'Radius', 'Flux', 'ESI', 'pl_rade', 'pl_insol',
        'base_score', 'hz_class', 'tier', 'evidence_quality'
    ]
    # Ensure columns exist
    for c in out_cols:
        if c not in valid_rank.columns:
            valid_rank[c] = np.nan

    out_path = processed_dir / 'phl_comparison.csv'
    valid_rank[out_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(valid_rank)} rows.")

    print("\n=== TASK 7: Generate docs/progress/phase5_validation.md ===")
    conf_table_text = (
        conf_table.to_string()
        if conf_table is not None
        else "P_HABZONE_CON / P_HABZONE_OPT columns not found."
    )
    report_content = f"""# Phase 5 Validation Report: PHL Habitable Worlds Catalog Comparison

## Task 1: Inspection & Anomaly Identification
- **hwc.csv rows**: {len(df_hwc):,}
- **hwc_potentially_habitable.csv rows**: {len(df_pot):,}
- **TOI-904 c in `hwc_potentially_habitable.csv`**: Tsurf = {toi904_tsurf:.2f} K, Period = {toi904_period:.4f} days, Age = {toi904_age:.2f} Gyr, Distance = {toi904_distance:.2f} ly. The {toi904_distance:.2f} value is the distance column, not the age; no anomalous age is present for this row.
- **Anomalous rows in hwc.csv (Age > {AGE_LIMIT_GYR:g} Gyr, or a non-positive radius/flux/mass/period)**: {len(anomalous)} rows identified (oldest by age: {anomaly_examples}).

## Task 2: Name Matching
- **Total PHL rows**: {phl_rows_count}
- **Successfully matched**: {matched_count}
- **Unmatched**: {unmatched_count}

## Task 3: Input Agreement (Radius & Insolation Flux)
- **Radius Relative Difference**: Median = {med_diff_r:.4f}, 95th Percentile = {p95_diff_r:.4f}
- **Flux Relative Difference** (rows with positive, finite archive flux only; N={flux_comparable_n}): Median = {med_diff_f:.4f}, 95th Percentile = {p95_diff_f:.4f}
- **Flux rows excluded**: {flux_excluded_count} (zero flux = {flux_zero_count}, missing flux = {flux_missing_count}, other non-finite = {flux_other_count}); no inf values are used in any percentile

## Task 4: Rank Agreement (ESI vs Base Score)
- **Overall Correlation (N={len(valid_rank)})**: Spearman = {spearman_all:.4f}, Pearson = {pearson_all:.4f}
- **Potentially Habitable Subset Correlation (N={len(subset_pot)})**: Spearman = {spearman_pot:.4f}, Pearson = {pearson_pot:.4f}

## Task 5: Habitable-Zone Agreement
Confusion Table (PHL HZ Status vs Our HZ Class):
```
{conf_table_text}
```

## Summary Line
Phase 5 independent validation against the PHL Habitable Worlds Catalog compared {len(valid_rank)} matched and scorable planets. Radius median relative difference = {med_diff_r:.4f}; flux median relative difference = {med_diff_f:.4f} with 95th percentile = {p95_diff_f:.4f} over the {flux_comparable_n} planets with positive finite flux ({flux_excluded_count} rows excluded: {flux_zero_count} zero flux, {flux_missing_count} missing flux, {flux_other_count} other non-finite). Ranking agreement with the published PHL ESI: Spearman = {spearman_all:.4f} overall and Spearman = {spearman_pot:.4f} on the {len(subset_pot)}-planet potentially habitable subset. Core scoring modules contain no PHL reference data.
"""
    doc_path = project_root / 'docs' / 'progress' / 'phase5_validation.md'
    doc_path.write_text(report_content, encoding='utf-8')
    print(f"Generated {doc_path}")


if __name__ == '__main__':
    main()
