"""EXOPHCA data cleaning and dataset building module."""

import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from exophca.science.esi import calculate_esi, classify_flags, STAR_CLASSES
from exophca.data.validate import validate_dataframe


PLANET_REQUIRED_COLUMNS = [
    'pl_name', 'hostname', 'discoverymethod', 'disc_year',
    'pl_rade', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax', 'pl_bmasse',
    'st_teff', 'st_rad', 'st_mass', 'sy_dist'
]

TOI_REQUIRED_COLUMNS = [
    'toi', 'tid', 'tfopwg_disp', 'pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt',
    'st_teff', 'st_rad', 'st_dist'
]


def classify_star(teff: float) -> str:
    """Classify star based on effective temperature (K) using exact historical bin edges."""
    if pd.isna(teff):
        return 'G (Sun-like)'  # default fallback if missing
    if teff >= 7500:
        return 'A (hot)'
    elif teff >= 6000:
        return 'F'
    elif teff >= 5200:
        return 'G (Sun-like)'
    elif teff >= 3700:
        return 'K'
    else:
        return 'M (red dwarf)'


def clean_html_tags(text: Any) -> str:
    """Strip HTML tags from reference links."""
    if pd.isna(text):
        return ""
    clean = re.sub(r'<.*?>', '', str(text))
    return clean.strip()


def clean_planets_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean PSCD.csv dataset keeping all rows, computing scorable flags, provenance, and base scores."""
    df = df_raw.copy()

    # Handle optional columns gracefully if absent
    optional_cols = [
        'pl_radeerr1', 'pl_radeerr2', 'pl_radelim',
        'pl_insolerr1', 'pl_insolerr2', 'pl_insollim',
        'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassprov',
        'pl_dens', 'st_lum', 'st_spectype', 'pl_controv_flag', 'cb_flag',
        'pl_rade_reflink'
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Clean radius_reference
    df['radius_reference'] = df['pl_rade_reflink'].apply(clean_html_tags)

    # Scorable flag: radius and flux both present and non-null
    has_radius = df['pl_rade'].notna() & (df['pl_rade'] > 0)
    has_flux = df['pl_insol'].notna() & (df['pl_insol'] >= 0)
    df['scorable'] = has_radius & has_flux

    # Base score using exophca.science.esi.calculate_esi
    base_scores = []
    for _, row in df.iterrows():
        if row['scorable']:
            try:
                score = calculate_esi(row['pl_rade'], row['pl_insol'])
                base_scores.append(score)
            except Exception:
                base_scores.append(np.nan)
        else:
            base_scores.append(np.nan)
    df['base_score'] = base_scores

    # Radius provenance classification
    # Rules: "missing" if pl_rade is missing; "modelled" if radius_reference is "Calculated Value" AND no radius uncertainty AND discoverymethod is Radial Velocity;
    # "derived_unclear" for other "Calculated Value" cases; "measured" otherwise.
    calc_val_mask = df['radius_reference'].str.contains('Calculated Value', na=False)
    is_rv = df['discoverymethod'].eq('Radial Velocity')
    no_rad_err = df['pl_radeerr1'].isna()

    modelled_mask = has_radius & calc_val_mask & no_rad_err & is_rv
    derived_unclear_mask = has_radius & calc_val_mask & ~modelled_mask
    measured_mask = has_radius & ~calc_val_mask

    radius_prov = np.select(
        [~has_radius, modelled_mask, derived_unclear_mask],
        ['missing', 'modelled', 'derived_unclear'],
        default='measured'
    )
    df['radius_provenance'] = radius_prov

    # Mass provenance
    if 'pl_bmassprov' in df.columns:
        df['mass_provenance'] = df['pl_bmassprov'].fillna('Unknown')
    else:
        df['mass_provenance'] = 'Unknown'

    # Approximate flux recovery (Stefan-Boltzmann) - stored SEPARATELY, NOT used for scoring
    # eqt / 255 ^ 4
    has_eqt = df['pl_eqt'].notna() & (df['pl_eqt'] > 0)
    df['insol_recovered_approx'] = np.where(has_eqt, (df['pl_eqt'] / 255.0) ** 4, np.nan)
    df['insol_recovered_flag'] = df['pl_insol'].isna() & has_eqt

    # Kepler's third law semi-major-axis recovery stored separately
    # a^3 = P^2 * M_star (P in years, a in AU, M in solar masses)
    has_orbper = df['pl_orbper'].notna() & (df['pl_orbper'] > 0)
    has_st_mass = df['st_mass'].notna() & (df['st_mass'] > 0)
    period_years = df['pl_orbper'] / 365.25
    calc_orbsmax = np.where(has_orbper & has_st_mass, (period_years**2 * df['st_mass']) ** (1/3), np.nan)
    df['orbsmax_recovered'] = np.where(df['pl_orbsmax'].isna(), calc_orbsmax, np.nan)
    df['orbsmax_recovered_flag'] = df['pl_orbsmax'].isna() & has_orbper & has_st_mass

    # Star class
    df['star_class'] = df['st_teff'].apply(classify_star)

    # Select final requested columns in order
    output_columns = [
        'pl_name', 'hostname', 'discoverymethod', 'disc_year',
        'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 'pl_radelim',
        'pl_insol', 'pl_insolerr1', 'pl_insolerr2', 'pl_insollim',
        'pl_eqt', 'pl_orbper', 'pl_orbsmax', 'pl_bmasse',
        'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassprov',
        'pl_dens', 'st_teff', 'st_rad', 'st_mass', 'st_lum', 'st_spectype',
        'sy_dist', 'pl_controv_flag', 'cb_flag', 'radius_reference',
        'scorable', 'base_score', 'radius_provenance', 'mass_provenance',
        'insol_recovered_approx', 'insol_recovered_flag',
        'orbsmax_recovered', 'orbsmax_recovered_flag', 'star_class'
    ]

    # Ensure all columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[output_columns]


def clean_toi_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean TOI dataset filtering to PC/APC, computing scorable and base scores."""
    df = df_raw.copy()

    # Rename st_dist to sy_dist if present
    if 'st_dist' in df.columns and 'sy_dist' not in df.columns:
        df['sy_dist'] = df['st_dist']

    # Handle error columns gracefully
    error_cols = ['pl_radeerr1', 'pl_radeerr2', 'pl_insolerr1', 'pl_insolerr2']
    for col in error_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Filter to tfopwg_disp in {PC, APC}
    if 'tfopwg_disp' in df.columns:
        df = df[df['tfopwg_disp'].isin(['PC', 'APC'])].copy()

    # Scorable flag
    has_radius = df['pl_rade'].notna() & (df['pl_rade'] > 0)
    has_flux = df['pl_insol'].notna() & (df['pl_insol'] >= 0)
    df['scorable'] = has_radius & has_flux

    # Base score
    base_scores = []
    for _, row in df.iterrows():
        if row['scorable']:
            try:
                score = calculate_esi(row['pl_rade'], row['pl_insol'])
                base_scores.append(score)
            except Exception:
                base_scores.append(np.nan)
        else:
            base_scores.append(np.nan)
    df['base_score'] = base_scores

    toi_output_columns = [
        'toi', 'tid', 'tfopwg_disp', 'pl_orbper', 'pl_rade',
        'pl_radeerr1', 'pl_radeerr2', 'pl_insol', 'pl_insolerr1', 'pl_insolerr2',
        'pl_eqt', 'st_teff', 'st_rad', 'sy_dist', 'scorable', 'base_score'
    ]

    for col in toi_output_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[toi_output_columns]
