"""Monte Carlo uncertainty propagation and probabilistic metrics module for EXOPHCA."""

import time
import numpy as np
import pandas as pd

from exophca.science.habitable_zone import classify_hz


def sample_twopiece_normal(
    nominal: np.ndarray,
    err1: np.ndarray,
    err2: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    clip_min: float = 0.0,
) -> np.ndarray:
    """Sample from a two-piece normal distribution using vectorized generator draws.

    Asymmetric errors:
    - err1: positive error (upper bound deviation, >= 0)
    - err2: negative error (lower bound deviation, stored as negative number or positive magnitude)

    With probability weight = err1 / (err1 + abs(err2)), draw nominal + err1 * |z|,
    else draw nominal - abs(err2) * |z|, where z is standard normal N(0, 1).
    Truncated/clipped at clip_min.

    Returns:
        Array of shape (n_draws, n_rows).
    """
    n_rows = len(nominal)
    # z draws: (n_draws, n_rows)
    z = rng.standard_normal((n_draws, n_rows))
    abs_z = np.abs(z)

    e1 = np.abs(err1)
    e2 = np.abs(err2)
    sum_e = e1 + e2
    # Handle cases where both errors are zero or NaN
    zero_mask = (sum_e == 0) | np.isnan(sum_e)

    p_upper = np.where(zero_mask, 0.5, e1 / np.maximum(sum_e, 1e-12))

    # Uniform random draws for deciding upper vs lower branch: (n_draws, n_rows)
    u = rng.random((n_draws, n_rows))
    upper_branch = u < p_upper[None, :]

    # Sample values: shape (n_draws, n_rows)
    samples = np.where(
        upper_branch,
        nominal[None, :] + e1[None, :] * abs_z,
        nominal[None, :] - e2[None, :] * abs_z,
    )

    if zero_mask.any():
        samples[:, zero_mask] = nominal[None, zero_mask]

    if clip_min is not None:
        samples = np.maximum(samples, clip_min)

    return samples


def propagate_uncertainties(
    df: pd.DataFrame,
    n_draws: int = 5000,
    seed: int = 42,
    is_toi: bool = False,
) -> pd.DataFrame:
    """Run Monte Carlo uncertainty propagation on scorable planets or TOIs.

    Semantics:
    - base_score: deterministic score from the archive values.
    - base_score_median, base_score_p16, base_score_p84: percentiles (50th, 16th, 84th) of the Monte Carlo score distribution computed from full draws. For symmetric errors, the median is close to the deterministic score.
    - Treat a reported error of exactly 0.0 or NaN as missing (placeholder).

    Limitations and Choices:
    - Teff uncertainty is not available in PSCD or TOI datasets. HZ flux limits are
      computed at the reported Teff and treated as fixed.
    - Missing or zero errors: held fixed. Uncertainty status is categorized explicitly.
    - Two-piece normal distribution used for asymmetric radius/flux errors.
    - Truncated at physical bounds: radius > 0 (clip_min=1e-6), flux >= 0 (clip_min=0.0).
    - Runtime is measured and logged.
    """
    start_time = time.time()
    n_rows = len(df)

    # Extract nominal values
    r_nom = df['pl_rade'].to_numpy(dtype=float)
    f_nom = df['pl_insol'].to_numpy(dtype=float)
    teff_nom = df['st_teff'].to_numpy(dtype=float)

    # Extract errors (handle missing columns gracefully)
    r_err1 = df['pl_radeerr1'].to_numpy(dtype=float) if 'pl_radeerr1' in df.columns else np.full(n_rows, np.nan)
    r_err2 = df['pl_radeerr2'].to_numpy(dtype=float) if 'pl_radeerr2' in df.columns else np.full(n_rows, np.nan)
    f_err1 = df['pl_insolerr1'].to_numpy(dtype=float) if 'pl_insolerr1' in df.columns else np.full(n_rows, np.nan)
    f_err2 = df['pl_insolerr2'].to_numpy(dtype=float) if 'pl_insolerr2' in df.columns else np.full(n_rows, np.nan)

    # Treat reported error of exactly 0.0 as missing (placeholder)
    r_err1 = np.where(r_err1 == 0.0, np.nan, r_err1)
    r_err2 = np.where(r_err2 == 0.0, np.nan, r_err2)
    f_err1 = np.where(f_err1 == 0.0, np.nan, f_err1)
    f_err2 = np.where(f_err2 == 0.0, np.nan, f_err2)

    # Limit flags
    r_lim = df['pl_radelim'].to_numpy(dtype=float) if 'pl_radelim' in df.columns else np.zeros(n_rows)
    f_lim = df['pl_insollim'].to_numpy(dtype=float) if 'pl_insollim' in df.columns else np.zeros(n_rows)

    # Determine uncertainty status per planet
    has_r_err = np.isfinite(r_err1) & np.isfinite(r_err2) & ((r_err1 != 0) | (r_err2 != 0))
    has_f_err = np.isfinite(f_err1) & np.isfinite(f_err2) & ((f_err1 != 0) | (f_err2 != 0))
    has_r_lim = np.isfinite(r_lim) & (r_lim != 0)
    has_f_lim = np.isfinite(f_lim) & (f_lim != 0)

    status = np.empty(n_rows, dtype=object)
    for i in range(n_rows):
        r_l = has_r_lim[i]
        f_l = has_f_lim[i]
        r_e = has_r_err[i]
        f_e = has_f_err[i]

        if r_l or f_l:
            status[i] = "limit_value"
        elif r_e and f_e:
            status[i] = "full"
        elif r_e and not f_e:
            status[i] = "radius_only"
        elif not r_e and f_e:
            status[i] = "flux_only"
        else:
            status[i] = "none"

    # Pre-allocate output arrays
    base_score_med = np.empty(n_rows, dtype=float)
    base_score_p16 = np.empty(n_rows, dtype=float)
    base_score_p84 = np.empty(n_rows, dtype=float)
    p_conservative_hz = np.empty(n_rows, dtype=float)
    p_any_hz = np.empty(n_rows, dtype=float)
    p_radius_le_1p6 = np.empty(n_rows, dtype=float)
    p_rocky_and_conservative = np.empty(n_rows, dtype=float)

    # Monte Carlo draws per planet using per-planet generator
    for i in range(n_rows):
        rng = np.random.default_rng([seed, int(i)])

        # Radius samples
        if has_r_lim[i] or not has_r_err[i]:
            r_samples = np.full(n_draws, r_nom[i])
        else:
            r_samples = sample_twopiece_normal(
                np.array([r_nom[i]]),
                np.array([r_err1[i]]),
                np.array([r_err2[i]]),
                n_draws,
                rng,
                clip_min=1e-6,
            )[:, 0]

        # Flux samples
        if has_f_lim[i] or not has_f_err[i]:
            f_samples = np.full(n_draws, f_nom[i])
        else:
            f_samples = sample_twopiece_normal(
                np.array([f_nom[i]]),
                np.array([f_err1[i]]),
                np.array([f_err2[i]]),
                n_draws,
                rng,
                clip_min=0.0,
            )[:, 0]

        # Vectorized calculations across draws for planet i
        r_comp = (1.0 - np.abs((r_samples - 1.0) / (r_samples + 1.0))) ** 0.57
        f_comp = (1.0 - np.abs((f_samples - 1.0) / (f_samples + 1.0))) ** 0.7
        scores = np.sqrt(r_comp * f_comp)
        scores = np.clip(scores, 0.0, 1.0)

        base_score_med[i] = np.median(scores)
        base_score_p16[i] = np.percentile(scores, 16)
        base_score_p84[i] = np.percentile(scores, 84)

        # Habitable zone classification across draws
        teff_val = teff_nom[i]
        hz_classes = classify_hz(f_samples, teff_val)
        is_cons = (hz_classes == "conservative")
        is_any = (hz_classes == "conservative") | (hz_classes == "optimistic_only")

        p_conservative_hz[i] = np.mean(is_cons)
        p_any_hz[i] = np.mean(is_any)

        is_le_1p6 = (r_samples <= 1.6)
        p_radius_le_1p6[i] = np.mean(is_le_1p6)
        p_rocky_and_conservative[i] = np.mean(is_le_1p6 & is_cons)

    elapsed = time.time() - start_time
    print(f"Monte Carlo propagation completed for {n_rows:,} rows in {elapsed:.2f} seconds ({n_draws} draws).")

    out_df = df.copy()
    out_df['uncertainty_status'] = status
    out_df['base_score_median'] = base_score_med
    out_df['base_score_p16'] = base_score_p16
    out_df['base_score_p84'] = base_score_p84
    out_df['p_conservative_hz'] = p_conservative_hz
    out_df['p_any_hz'] = p_any_hz
    out_df['p_radius_le_1p6'] = p_radius_le_1p6
    out_df['p_rocky_and_conservative'] = p_rocky_and_conservative

    return out_df
