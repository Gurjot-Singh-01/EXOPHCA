"""Rule-based assessment module for exoplanet habitability compatibility (no single score)."""

import pandas as pd
import numpy as np

DISCLAIMER = "Similarity and habitable-zone position are not evidence of habitability or life."

# Size regime thresholds following published conservative and optimistic samples:
# - rocky_likely: R <= 1.6
# - uncertain: 1.6 < R <= 2.5
# - not_rocky_likely: R > 2.5


def determine_size_regime(r: float) -> str:
    if pd.isna(r) or r <= 0:
        return "undetermined"
    elif r <= 1.6:
        return "rocky_likely"
    elif r <= 2.5:
        return "uncertain"
    else:
        return "not_rocky_likely"


def determine_tier(hz_class: str, size_regime: str) -> str:
    """Determine habitability compatibility tier according to exact rules:
    A: conservative HZ and rocky_likely
    B: (conservative HZ and uncertain) OR (optimistic_only HZ and radius <= 2.5)
    C: any HZ (conservative or optimistic_only) and not_rocky_likely
    D: outside HZ (too_hot or too_cold)
    U: undetermined (hz_class undetermined or radius missing)
    """
    if hz_class in ("undetermined",) or size_regime == "undetermined":
        return "U"

    if hz_class in ("too_hot", "too_cold", "outside_hz"):
        return "D"

    # HZ is conservative or optimistic_only
    if hz_class == "conservative":
        if size_regime == "rocky_likely":
            return "A"
        elif size_regime == "uncertain":
            return "B"
        elif size_regime == "not_rocky_likely":
            return "C"
    elif hz_class == "optimistic_only":
        if size_regime in ("rocky_likely", "uncertain"):
            return "B"  # radius <= 2.5 covers rocky_likely and uncertain (1.6 to 2.5)
        elif size_regime == "not_rocky_likely":
            return "C"

    return "U"


def determine_evidence_quality(
    radius_provenance: str,
    has_radius_error: bool,
    has_flux_error: bool,
    is_toi: bool = False,
) -> str:
    """Determine evidence quality according to exact rules:
    - high: radius_provenance == "measured" AND a nonzero radius error is present AND a nonzero flux error is present
    - medium: radius_provenance == "measured" AND (radius error missing OR flux error missing)
    - low: radius_provenance is "modelled" or "derived_unclear", OR a "measured" radius with NO radius error and NO flux error.
    - none: radius missing
    A planet with a modelled radius can NEVER be "high" or "medium".
    TOI planets: radius_provenance is treated as "measured", max evidence is "medium".
    """
    if radius_provenance == "missing":
        return "none"

    if radius_provenance in ("modelled", "derived_unclear"):
        return "low"

    # Measured radius
    if is_toi:
        if has_radius_error and has_flux_error:
            return "high"
        elif has_radius_error or has_flux_error:
            return "medium"
        else:
            return "low"
    else:
        if radius_provenance == "measured":
            if has_radius_error and has_flux_error:
                return "high"
            elif has_radius_error or has_flux_error:
                return "medium"
            else:
                return "low"

    return "low"


def assess_planets(df: pd.DataFrame, is_toi: bool = False) -> pd.DataFrame:
    """Assess planets or TOIs and attach assessment columns."""
    out = df.copy()

    statuses = []
    size_regimes = []
    tiers = []
    evidence_qualities = []
    boundary_sensitives = []
    criteria_list = []

    for _, row in out.iterrows():
        status = "unconfirmed candidate" if is_toi else "confirmed"
        statuses.append(status)

        r = row.get('pl_rade', np.nan)
        hz_cls = row.get('hz_class', 'undetermined')
        prov = row.get('radius_provenance', 'measured' if is_toi else 'missing')

        # Check radius error presence
        r_err1 = row.get('pl_radeerr1', np.nan)
        r_err2 = row.get('pl_radeerr2', np.nan)
        has_r_err = pd.notna(r_err1) and pd.notna(r_err2) and (r_err1 != 0 or r_err2 != 0)

        # Check flux error presence
        f_err1 = row.get('pl_insolerr1', np.nan)
        f_err2 = row.get('pl_insolerr2', np.nan)
        has_f_err = pd.notna(f_err1) and pd.notna(f_err2) and (f_err1 != 0 or f_err2 != 0)

        size_regime = determine_size_regime(r)
        size_regimes.append(size_regime)

        tier = determine_tier(hz_cls, size_regime)
        tiers.append(tier)

        ev = determine_evidence_quality(prov, has_r_err, has_f_err, is_toi=is_toi)
        evidence_qualities.append(ev)

        # Boundary sensitive rule:
        # boundary_sensitive = True when p_conservative_hz (or p_any_hz for optimistic_only planets) is below 0.7 for a planet whose point classification is in that zone.
        p_cons = row.get('p_conservative_hz', 1.0)
        p_any = row.get('p_any_hz', 1.0)

        bound_sens = False
        if hz_cls == "conservative" and p_cons < 0.7:
            bound_sens = True
        elif hz_cls == "optimistic_only" and p_any < 0.7:
            bound_sens = True
        boundary_sensitives.append(bound_sens)

        # Criteria text
        hz_desc = f"HZ: {hz_cls}"
        size_desc = f"size: {size_regime} ({prov})"
        ev_desc = f"evidence: {ev} ({'radius modelled' if prov in ('modelled', 'derived_unclear') else 'radius measured'})"
        crit = f"{hz_desc}; {size_desc}; {ev_desc}"
        criteria_list.append(crit)

    out['status'] = statuses
    out['size_regime'] = size_regimes
    out['tier'] = tiers
    out['evidence_quality'] = evidence_qualities
    out['boundary_sensitive'] = boundary_sensitives
    out['criteria'] = criteria_list

    return out
