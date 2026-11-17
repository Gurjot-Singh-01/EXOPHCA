"""Kopparapu et al. habitable zone flux limits module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

# Coefficients from Kopparapu et al. (2013, ApJ 765, 131; 2014, ApJL 787, L29)
# Order: [recent_venus, runaway_greenhouse, maximum_greenhouse, early_mars, runaway_5me, runaway_0p1me]
HZ_CONSTANTS = {
    "citation": "Kopparapu et al. (2013, ApJ 765, 131; 2014, ApJL 787, L29)",
    "s_sun": np.array([1.776, 1.107, 0.356, 0.320, 1.188, 0.99]),
    "a": np.array([2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]),
    "b": np.array([2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]),
    "c": np.array([-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]),
    "d": np.array([-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]),
}


def hz_limits(teff: Union[float, np.ndarray]) -> Dict[str, Any]:
    """Calculate habitable zone flux limits for a given stellar effective temperature (K).

    Returns conservative_inner (runaway greenhouse), conservative_outer (maximum greenhouse),
    optimistic_inner (recent Venus), optimistic_outer (early Mars), and in_valid_range.
    """
    is_scalar = not isinstance(teff, (np.ndarray, pd.Series, list))
    if is_scalar:
        teff_arr = np.array([float(teff) if teff is not None else np.nan])
    else:
        teff_arr = np.asarray(teff, dtype=float)

    valid_mask = np.isfinite(teff_arr) & (teff_arr >= 2600.0) & (teff_arr <= 7200.0)

    # Safe T calculation avoiding RuntimeWarning for invalid/NaN values
    safe_teff = np.where(valid_mask, teff_arr, 5780.0)
    T = safe_teff - 5780.0
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    s_sun = HZ_CONSTANTS["s_sun"]
    a = HZ_CONSTANTS["a"]
    b = HZ_CONSTANTS["b"]
    c = HZ_CONSTANTS["c"]
    d = HZ_CONSTANTS["d"]

    seff = (
        s_sun[None, :]
        + a[None, :] * T[:, None]
        + b[None, :] * T2[:, None]
        + c[None, :] * T3[:, None]
        + d[None, :] * T4[:, None]
    )

    recent_venus = seff[:, 0]
    runaway_gh = seff[:, 1]
    max_gh = seff[:, 2]
    early_mars = seff[:, 3]

    nan_arr = np.full_like(teff_arr, np.nan, dtype=float)
    recent_venus = np.where(valid_mask, recent_venus, nan_arr)
    runaway_gh = np.where(valid_mask, runaway_gh, nan_arr)
    max_gh = np.where(valid_mask, max_gh, nan_arr)
    early_mars = np.where(valid_mask, early_mars, nan_arr)

    if is_scalar:
        return {
            "conservative_inner": float(runaway_gh[0]),
            "conservative_outer": float(max_gh[0]),
            "optimistic_inner": float(recent_venus[0]),
            "optimistic_outer": float(early_mars[0]),
            "in_valid_range": bool(valid_mask[0]),
        }
    else:
        return {
            "conservative_inner": runaway_gh,
            "conservative_outer": max_gh,
            "optimistic_inner": recent_venus,
            "optimistic_outer": early_mars,
            "in_valid_range": valid_mask,
        }


def classify_hz(flux: Union[float, np.ndarray], teff: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """Classify planetary position relative to habitable zone boundaries.

    Classes:
    - "undetermined": missing flux or Teff, or Teff out of range (<2600 or >7200)
    - "too_hot": flux above optimistic inner (recent Venus limit)
    - "too_cold": flux below optimistic outer (early Mars limit)
    - "conservative": conservative_outer <= flux <= conservative_inner
    - "optimistic_only": inside optimistic range (early_mars <= flux <= recent_venus) but not conservative
    - "outside_hz": outside optimistic habitable zone

    Boundary convention: Inclusive (<= and >=).
    """
    limits = hz_limits(teff)
    is_scalar = not isinstance(flux, (np.ndarray, pd.Series, list)) and not isinstance(teff, (np.ndarray, pd.Series, list))

    if is_scalar:
        if flux is None or teff is None or not np.isfinite(float(flux)) or not np.isfinite(float(teff)) or not limits["in_valid_range"]:
            return "undetermined"
        f = float(flux)
        ci = limits["conservative_inner"]
        co = limits["conservative_outer"]
        oi = limits["optimistic_inner"]
        oo = limits["optimistic_outer"]

        if not np.isfinite(ci) or not np.isfinite(co) or not np.isfinite(oi) or not np.isfinite(oo):
            return "undetermined"

        if f > oi:
            return "too_hot"
        elif f < oo:
            return "too_cold"
        elif co <= f <= ci:
            return "conservative"
        elif oo <= f <= oi:
            return "optimistic_only"
        else:
            return "outside_hz"
    else:
        flux_arr = np.asarray(flux, dtype=float)
        teff_arr = np.asarray(teff, dtype=float)
        valid = np.isfinite(flux_arr) & np.isfinite(teff_arr) & limits["in_valid_range"]

        ci = limits["conservative_inner"]
        co = limits["conservative_outer"]
        oi = limits["optimistic_inner"]
        oo = limits["optimistic_outer"]

        results = np.full(flux_arr.shape, "undetermined", dtype=object)

        too_hot_mask = valid & (flux_arr > oi)
        too_cold_mask = valid & (flux_arr < oo)
        conservative_mask = valid & ~too_hot_mask & ~too_cold_mask & (flux_arr >= co) & (flux_arr <= ci)
        opt_only_mask = valid & ~too_hot_mask & ~too_cold_mask & ~conservative_mask & (flux_arr >= oo) & (flux_arr <= oi)

        results[too_hot_mask] = "too_hot"
        results[too_cold_mask] = "too_cold"
        results[conservative_mask] = "conservative"
        results[opt_only_mask] = "optimistic_only"
        results[~valid] = "undetermined"

        return results
