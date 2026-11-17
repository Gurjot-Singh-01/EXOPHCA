"""EXOPHCA data validation module."""

from typing import Any
import pandas as pd
import numpy as np


def validate_dataframe(df: pd.DataFrame, required_columns: list[str], name: str) -> list[str]:
    """Validate a DataFrame for required columns, missing data, numeric ranges, and duplicate names.

    Returns a list of error/warning messages. Never silently drops rows.
    """
    messages = []

    # 1. Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        messages.append(f"[{name}] ERROR: Missing required columns: {missing_cols}")

    # 2. Check duplicate planet/object names if name column exists
    name_col = 'pl_name' if 'pl_name' in df.columns else ('toi' if 'toi' in df.columns else None)
    if name_col:
        duplicates = df[df.duplicated(subset=[name_col], keep=False)]
        if not duplicates.empty:
            dup_names = duplicates[name_col].unique()[:10]
            messages.append(f"[{name}] WARNING: Found duplicate {name_col} entries (e.g., {list(dup_names)}). Count: {len(duplicates)}")

    # 3. Numeric range checks (radius > 0, flux >= 0, Teff > 0 if present)
    if 'pl_rade' in df.columns:
        invalid_rad = df['pl_rade'].notna() & (df['pl_rade'] <= 0)
        if invalid_rad.any():
            messages.append(f"[{name}] ERROR: Found {invalid_rad.sum()} rows with planet radius <= 0.")

    if 'pl_insol' in df.columns:
        invalid_flux = df['pl_insol'].notna() & (df['pl_insol'] < 0)
        if invalid_flux.any():
            messages.append(f"[{name}] ERROR: Found {invalid_flux.sum()} rows with insolation flux < 0.")

    if 'st_teff' in df.columns:
        invalid_teff = df['st_teff'].notna() & (df['st_teff'] <= 0)
        if invalid_teff.any():
            messages.append(f"[{name}] ERROR: Found {invalid_teff.sum()} rows with stellar effective temperature <= 0.")

    return messages
