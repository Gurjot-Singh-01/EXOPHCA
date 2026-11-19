"""The 'Methods and limitations' text, shared by the Detail expander and the page.

Both places must say the same thing, so the wording lives here once. Nothing
here computes a value: the formula, the weights, the temperature range and the
evidence rules describe the behaviour of ``exophca.science`` and the
uncertainty pipeline that produced ``data/processed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import data_access  # noqa: E402
from exophca.science.assessment import DISCLAIMER  # noqa: E402
from exophca.science.habitable_zone import HZ_CONSTANTS  # noqa: E402

LIMITS = [
    (
        "Similarity is not habitability",
        "The score compares a radius and an insolation flux with Earth's. It uses no "
        "atmosphere, albedo, surface pressure, rotation, magnetic field or "
        "temperature measurement, so a high score cannot be read as a chance of "
        "liquid water, of life, or of anything being habitable.",
    ),
    (
        "Some radii are modelled, not measured",
        "Where an archive radius is absent the radius is estimated from a "
        "mass-radius relation. Those rows are labelled 'radius modelled, not "
        "measured' wherever the radius appears, score lower on evidence quality, and "
        "carry the uncertainty of the relation, which is not propagated here.",
    ),
    (
        "TESS candidates have no flux uncertainty",
        "The unconfirmed TESS candidates carry a radius error but no insolation-flux "
        "error, so their flux is held fixed and their uncertainty_status is "
        "flux_only. Every probability that depends on the flux is therefore "
        "conditional on a fixed input, and a candidate is not a confirmed planet.",
    ),
    (
        "The sample is not a census",
        "Transit and radial-velocity surveys find large, close-in planets far more "
        "easily than small, distant ones, and bright stars more easily than faint "
        "ones. The assessed set is what has been measured well enough to score, not "
        "a representative sample of planets.",
    ),
    (
        "The score also ignores the star",
        "Stellar type enters the habitable-zone class through the effective "
        "temperature, but not the score. A planet around a flare star and an "
        "identical one around a quiet G star receive the same score.",
    ),
]

# The null result about the base score, stated once and reused. Every surface
# that shows the score shows this sentence beside it, so the score cannot read as
# a ranking instrument on one page and as a descriptive column on another.
SCORE_NULL_RESULT = (
    "Ranked against a plain unweighted proximity to Earth in the same two "
    "properties (`|log10 R| + |log10 S|`, no exponents and no borrowed weight), "
    "the ordering agrees almost perfectly - Spearman rho = 0.992 over the 5,641 "
    "planets with a usable radius and flux - so the score is very close to a "
    "monotone transform of radius and flux and carries almost no ordering "
    "information of its own. That is a null result: read the score as a "
    "descriptive column, not as a habitability ranking."
)


def _f(value, digits: int = 3) -> str:
    """Format a number from the training report; a missing one stays visible."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


def _test_mae(test_metrics: dict, name: str) -> object:
    """The test MAE recorded under ``name``, or under a ``best_baseline_`` key."""
    block = test_metrics.get(name)
    if isinstance(block, dict) and "MAE" in block:
        return block["MAE"]
    for key, value in test_metrics.items():
        if key.startswith("best_baseline_") and isinstance(value, dict):
            return value.get("MAE")
    return None


def render_radius_model() -> None:
    """The radius model, with every number read from its training report."""
    meta = data_access.load_radius_model_meta()
    if meta is None:
        st.info(
            "The radius model's training report is missing at "
            "`models/radius_model_meta.json`, so nothing is claimed about the "
            "model here."
        )
        return

    decision = meta.get("decision") or {}
    intervals = meta.get("intervals") or {}
    test_metrics = meta.get("test_metrics") or {}
    chosen = decision.get("chosen_model", "the chosen model")
    baseline = decision.get("best_baseline", "the best baseline")
    method = intervals.get("final_recommended", "unknown")
    cv_method = intervals.get("recommended_on_cv", "unknown")

    st.markdown("#### Radius model (mass-radius estimate)")
    st.markdown(
        "The archive's modelled radii come from a published mass-radius "
        "relation. As a cross-check this project trained and calibrated its own "
        "model, which estimates log10 radius from log10 mass, log10 orbital "
        "period, effective temperature, stellar mass and stellar radius. It is a "
        "radius estimator only: it says nothing about habitability, and its "
        "output never changes a score, tier, class or probability."
    )

    row_filters = list((meta.get("row_filters") or {}).items())
    split_sizes = meta.get("split_sizes") or {}
    rows = (split_sizes.get("rows") or {})
    if row_filters:
        label, count = row_filters[-1]
        st.markdown(
            f"**Training data.** {count:,} planets pass the training filter "
            f"(`{label}`): a measured radius, a measured mass, and all five "
            "features present and positive. Hosts are split so that no host "
            f"appears in two splits - {rows.get('train', 0):,} rows for "
            f"training, {rows.get('calibration', 0):,} for interval "
            f"calibration and {rows.get('test', 0):,} for the test, "
            "held out by host."
        )

    st.markdown(
        f"**Point model.** `{chosen}` was chosen on cross-validation with a MAE "
        f"of {_f(decision.get('chosen_cv_mae'))} dex, against "
        f"{_f(decision.get('best_baseline_cv_mae'))} dex for the best baseline, "
        f"`{baseline}`. On the held-out test hosts the MAE is "
        f"{_f(_test_mae(test_metrics, 'chosen_model'))} dex for `{chosen}` and "
        f"{_f(_test_mae(test_metrics, baseline))} dex for `{baseline}`."
    )

    half_width = intervals.get("split_conformal_half_width_dex")
    nominal = intervals.get("nominal_coverage")
    st.markdown(
        f"**Interval.** The reported interval is `{method}`, at a nominal "
        f"{_f(nominal, 2)} coverage and a half-width of {_f(half_width)} dex "
        "around the point estimate; it is a fixed multiplicative factor in "
        "radius, not a per-planet error bar."
    )

    coverage = (meta.get("test_interval_coverage") or {}).get(method) or []
    if coverage:
        st.markdown(f"Test coverage by mass stratum, `{method}`:")
        lines = [
            "| mass stratum | planets | coverage | mean width (dex) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for entry in coverage:
            lines.append(
                f"| {entry.get('stratum', 'unknown')} | {entry.get('n', 0):,} | "
                f"{_f(entry.get('coverage'))} | {_f(entry.get('mean_width_dex'))} |"
            )
        st.markdown("\n".join(lines))

    checks = (intervals.get("test_checks") or {}).get(cv_method)
    if checks:
        st.markdown(
            "**Why the interval method changed.** The pre-declared rule chose "
            f"`{cv_method}` on cross-validation, where it was the narrower and "
            "better-calibrated of the two methods. After the test split showed "
            f"`{cv_method}` covering {_f(checks.get('worst_coverage'))} of the "
            f"planets in `{checks.get('worst_stratum', 'the worst stratum')}` at "
            f"a nominal {_f(nominal, 2)}, the recommendation was changed to "
            f"`{method}`. That decision was taken with the test split already "
            "seen, so the test coverage of split conformal is **not** fully "
            "independent evidence for it."
        )
    else:
        st.markdown(
            "**Why the interval method changed.** The recommendation in the "
            "report is "
            f"`{method}`, chosen after the earlier cross-validation choice of "
            f"`{cv_method}` under-covered on the test split. Because that change "
            "followed the test evaluation, the test coverage of "
            f"`{method}` is **not** fully independent evidence for it."
        )

    st.markdown(
        "**Minimum masses.** The model was trained on true masses. Most planets "
        "with a modelled radius have only a minimum mass (Msini), so a "
        "prediction for them assumes sin(i) = 1 and the true radius may be "
        "larger. The Detail page marks those estimates."
    )


def render(*, include_radius_model: bool = False) -> None:
    """Write the methods and limitations sections into the current container."""

    st.markdown("#### The score: ESI-style radius-flux similarity")
    st.latex(
        r"\mathrm{score} = \sqrt{\left(1 - \left|\frac{R-1}{R+1}\right|\right)^{0.57}"
        r" \cdot \left(1 - \left|\frac{S-1}{S+1}\right|\right)^{0.70}}"
    )
    st.markdown(
        "`score = sqrt((1 - |(R - 1) / (R + 1)|)^0.57 × "
        "(1 - |(S - 1) / (S + 1)|)^0.7)`, where `R` is the planet radius in Earth "
        "radii and `S` the insolation flux in Earth fluxes. The result lies in "
        "[0, 1] and Earth scores 1.0."
    )
    st.markdown(
        "**Weights.** 0.57 is the published radius weight of the Earth Similarity "
        "Index (Schulze-Makuch et al. 2011). The second weight, 0.70, is that "
        "paper's *escape-velocity* weight, borrowed here for insolation flux; it is "
        "**not** a published flux weight. The published index also uses bulk density "
        "and surface temperature, which are not available for most of this sample, "
        "so this is an ESI-*style* two-parameter similarity and is labelled as such "
        "everywhere in the app."
    )

    st.markdown(
        "**What the score adds over its own two columns.** Measured, not asserted. "
        + SCORE_NULL_RESULT
        + " Banding by rank with the committed tier sizes, 26 of those planets "
        "(0.46%) land in a different band and 17 of the 18 top-band planets are "
        "the same ones. The 0.7287 agreement with the published PHL ESI is partly "
        "agreement about one shared input, for the same reason."
    )

    st.markdown("#### The habitable zone")
    st.markdown(
        f"Flux limits come from {HZ_CONSTANTS['citation']}, evaluated at the star's "
        "reported effective temperature. The conservative band runs from the runaway "
        "greenhouse limit (inner) to the maximum greenhouse limit (outer); the "
        "optimistic band widens it to the recent-Venus limit (inner) and the "
        "early-Mars limit (outer). Boundaries are inclusive."
    )
    st.markdown(
        "The fits are valid for effective temperatures of 2600-7200 K. Outside that "
        "range the limits are **not** extrapolated: the class becomes *undetermined* "
        "and no habitable-zone claim is made. The temperature uncertainty is not "
        "available in either archive table, so the limits are computed at the "
        "reported temperature and held fixed."
    )

    st.markdown("#### The uncertainty")
    st.markdown(
        "Where the archive publishes asymmetric errors, radius and flux are sampled "
        "from a two-piece normal distribution (5,000 draws per planet, seed 42, "
        "truncated at radius > 0 and flux \u2265 0). Reported errors of exactly zero "
        "and missing errors are treated as missing and the value is held fixed. The "
        "spread of the resulting score distribution gives the 16th-84th percentile "
        "range; the same draws give four probabilities: in the conservative zone, in "
        "either zone, at radius \u2264 1.6 R\u2295, and rocky-sized inside the "
        "conservative zone."
    )
    st.markdown(
        "Each row records which inputs were actually varied in `uncertainty_status`: "
        "`full` (radius and flux), `radius_only`, `flux_only`, `none`, or "
        "`limit_value` (an archive limit value was held fixed). The app prints that "
        "status with every probability, and any probability that depends on an input "
        "that was not varied is labelled *conditional on a fixed input (uncertainty "
        "unknown)* instead of being shown as a plain number."
    )

    st.markdown("#### Evidence rules")
    st.markdown(
        "- **high**: measured radius, a published radius error and a published flux "
        "error.\n"
        "- **medium**: measured radius with only one of the two errors.\n"
        "- **low**: modelled or unclear radius provenance, or a measured radius "
        "with neither error.\n"
        "- **none**: no radius.\n\n"
        "A modelled radius can never be high or medium. TESS candidates have no flux "
        "error, so none of them reaches high."
    )

    st.markdown("#### Compatibility tier")
    st.markdown(
        "- **A**: conservative zone and radius \u2264 1.6 R\u2295 (likely rocky).\n"
        "- **B**: conservative zone with an intermediate size (1.6-2.5 R\u2295), or "
        "the optimistic zone only with radius \u2264 2.5 R\u2295.\n"
        "- **C**: inside a zone but larger than 2.5 R\u2295 (unlikely to be rocky).\n"
        "- **D**: outside both zones (too hot or too cold).\n"
        "- **U**: undetermined - no usable radius, or a stellar temperature outside "
        "the fitted range.\n\n"
        "A planet is also flagged **boundary sensitive** when its point class is in "
        "a zone but fewer than 70% of the Monte Carlo draws land in that zone."
    )

    if include_radius_model:
        render_radius_model()

    st.markdown("#### Main limitations")
    for title, explanation in LIMITS:
        st.markdown(f"**{title}.** {explanation}")

    st.markdown("#### References")
    st.markdown(
        "- Schulze-Makuch, D. et al. (2011), *A Two-Tiered Approach to Assessing the "
        "Habitability of Exoplanets*, Astrobiology 11, 1041.\n"
        "- Kopparapu, R. K. et al. (2013), *Habitable Zones Around Main-sequence "
        "Stars: New Estimates*, ApJ 765, 131.\n"
        "- Kopparapu, R. K. et al. (2014), *Habitable Zones Around Main-sequence "
        "Stars: Dependence on Planetary Mass*, ApJL 787, L29."
    )

    st.markdown(f"**Disclaimer.** {DISCLAIMER}")
