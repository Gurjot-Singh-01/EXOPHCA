"""Planet Detail: one assessed planet, its numbers and every caveat that belongs.\n\nEvery probability is printed through ``labelling.format_probability``, so no\nnumber appears without its ``uncertainty_status`` and a probability built on an\nunsampled input is always labelled conditional. A modelled radius is named as\nsuch, and where the offline model produced an estimate this page shows it from\ndata/processed/radius_estimates.csv; the model artifact models/radius_model.joblib is
never loaded by the app.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import data_access  # noqa: E402
import labelling  # noqa: E402
import methods_content  # noqa: E402
import query  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Planet Detail",
    "One assessed planet: its score with the 16th-84th percentile range, its "
    "habitable-zone class, the evidence behind it, and the caveats that travel "
    "with each number.",
    eyebrow="Explore",
)

ui.disclaimer()


def long_date(iso: str | None) -> str:
    if not iso:
        return "unknown"
    try:
        return date.fromisoformat(iso).strftime("%d %B %Y")
    except ValueError:
        return iso


frame = data_access.load_or_stop(data_access.load_planet_explorer, "Planet explorer table")
provenance = data_access.load_or_stop(data_access.load_provenance, "Provenance records")
estimates = data_access.load_or_stop(
    data_access.load_radius_estimates, "ML radius estimates"
)
retrieval_date = provenance["pscd"]["retrieval_date"]

names = sorted(frame["pl_name"].dropna().unique())
if not names:
    st.error("The assessed planet table is empty.")
    st.stop()

DEFAULT_PLANET = "Kepler-452 b"

search_col, select_col = st.columns([1, 2])
search = search_col.text_input(
    "Find a planet",
    value="",
    placeholder="e.g. Kepler-452, TRAPPIST-1, Proxima",
    help=(
        "Case-insensitive substring match, read literally, on the planet name or "
        "its host star; it narrows the list below."
    ),
)
# The search reads the same columns as the Explorer's: the planet name and its
# host star, as a case-insensitive literal substring with surrounding spaces
# ignored (no regular expressions), so "Barnard's star" finds Barnard b.
matches = sorted(
    query.apply_filters(frame, search=search)["pl_name"].dropna().unique()
)
if not matches:
    st.warning(
        "**No planet matches.** No assessed planet name or host star contains "
        f"\u201c{search.strip()}\u201d. Check the spelling, or clear the box to see "
        f"all {len(names):,} assessed planets."
    )
    st.stop()

# The index must be a position in the *filtered* list. Kepler-452 b sits at
# position 3,778 of the 5,688 names, so reusing that position after a search
# narrowed the list to one option raised StreamlitValueOutOfRangeError.
default_index = matches.index(DEFAULT_PLANET) if DEFAULT_PLANET in matches else 0
chosen = select_col.selectbox(
    "Planet (all {count:,} scorable planets)".format(count=len(names)),
    matches,
    index=default_index,
)

row = frame[frame["pl_name"] == chosen].iloc[0]
estimate_rows = estimates[estimates["pl_name"] == chosen]
estimate = estimate_rows.iloc[0] if len(estimate_rows) else None
status = labelling.normalise_status(row["uncertainty_status"])
hz_class = row["hz_class"]
tier = row["tier"]
evidence = row["evidence_quality"]

def text_or_dash(value, fallback: str = "unknown") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return fallback
    return str(value)


def render_ml_estimate(estimate, row) -> None:
    """The indicative radius estimate, or the reason there is none.

    A planet with a measured radius was not selected by the offline model, so it
    has no row and nothing is drawn. A modelled planet without a usable estimate
    says so and gives the reason rather than quietly showing the placeholder.
    """
    if estimate is None or pd.isna(estimate["radius_estimate"]):
        if row["radius_provenance"] in ("modelled", "derived_unclear"):
            reason = ""
            if estimate is not None:
                reason = text_or_dash(estimate["reason"], "").strip()
            suffix = f" {reason.capitalize()}." if reason else ""
            st.info(
                "**No ML radius estimate for this planet.**"
                f"{suffix} The archive value above is unchanged and still carries "
                "the uncertainty of the mass-radius relation."
            )
        return

    estimate_radius = float(estimate["radius_estimate"])
    lower = float(estimate["radius_lower"])
    upper = float(estimate["radius_upper"])
    archive = row["pl_rade"]
    archive_text = (
        f"{archive:.2f} R\u2295"
        if pd.notna(archive)
        else "not recorded"
    )

    st.markdown("#### ML radius estimate (indicative)")
    st.markdown(
        f"**{estimate_radius:.2f} R\u2295**, with a 90% interval of "
        f"**{lower:.2f}\u2013{upper:.2f} R\u2295**. The archive value is "
        f"{archive_text} (radius provenance: {row['radius_provenance']})."
    )
    if bool(estimate["interval_straddles_1p6"]):
        st.markdown(
            "The interval spans the 1.6 R\u2295 rocky threshold, so the model "
            "cannot tell a rocky planet from a larger one."
        )
    if bool(estimate["mass_is_minimum"]):
        st.warning(
            "**Minimum-mass input.** Computed from a minimum mass, assumes "
            "sin(i) = 1; the true radius may differ."
        )
    if bool(estimate["ood_flag"]):
        st.warning(
            "**Out of distribution.** At least one input lies outside the "
            "1st-99th percentile range the model saw in training, so this is an "
            "extrapolation."
        )
    st.caption(
        "This estimate does not change the score, tier, class or probabilities. "
        "The model estimates radius only and says nothing about habitability."
    )


discovered = (
    f"discovered {int(row['disc_year'])}"
    if pd.notna(row["disc_year"])
    else "discovery year unknown"
)
st.subheader(chosen)
st.caption(
    f"{row['status']} · {discovered} by "
    f"{text_or_dash(row['discoverymethod'], 'unknown method')} · host "
    f"{text_or_dash(row['hostname'])}"
)

hz_label = ui.HZ_LABELS.get(hz_class, hz_class)

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Score",
    f"{row['base_score']:.3f}",
    help="The ESI-style similarity computed from the archive radius and flux.",
)
# The range is written without spaces around the dash: at four columns the
# metric font sets "0.855 - 0.920" a few pixels wider than the column, which
# clipped the last digit at desktop width.
metric_columns[1].metric(
    "16th-84th percentile",
    f"{row['base_score_p16']:.3f}\u2013{row['base_score_p84']:.3f}",
    help="Monte Carlo range of the score, from the published radius and flux errors.",
)
metric_columns[2].metric("Compatibility tier", tier)
metric_columns[3].metric(
    "Flux",
    f"{row['pl_insol']:.3f} S⊕",
    help="Insolation flux relative to Earth.",
)

# The habitable-zone class is a long label, so it is shown as a chip rather
# than as a metric value, which Streamlit truncates in a narrow column.
st.markdown(
    f'<div class="ex-eyebrow">Habitable-zone class</div>',
    unsafe_allow_html=True,
)
ui.badge(hz_label, color=ui.HZ_COLORS.get(hz_class, ui.NEUTRAL))
st.write("")

badge_columns = st.columns(4)
with badge_columns[0]:
    st.caption("Evidence quality")
    ui.badge(evidence, color=ui.TIER_COLORS.get(tier, ui.NEUTRAL))
with badge_columns[1]:
    st.caption("Uncertainty status")
    ui.badge(status, color=ui.ACCENT)
with badge_columns[2]:
    st.caption("Radius provenance")
    ui.badge(row["radius_provenance"], color=ui.ACCENT_SOFT)
with badge_columns[3]:
    st.caption("Boundary sensitive")
    ui.badge("yes" if row["boundary_sensitive"] else "no", color=ui.NEUTRAL)

st.write("")

left, right = st.columns(2)

with left:
    st.markdown("#### Score and its range")
    st.markdown(
        f"The point score **{row['base_score']:.3f}** is the deterministic formula on "
        f"the archive values; the 16th-84th percentile range is "
        f"**{row['base_score_p16']:.3f} – {row['base_score_p84']:.3f}**. The range is "
        "not a confidence interval for habitability: it only propagates the "
        "published radius and flux errors, and only for the inputs the "
        "uncertainty_status below says were sampled."
    )
    st.caption(
        labelling.status_text(status)
        + " · the flux limits use the reported stellar temperature, whose "
        "uncertainty is not propagated."
    )

    st.markdown("#### Criteria recorded by the assessment")
    st.text(row["criteria"])

with right:
    st.markdown("#### Why this class")
    st.markdown(
        f"At an insolation of **{row['pl_insol']:.3f} S⊕** and a stellar effective "
        f"temperature of **{row['st_teff']:.0f} K** ({row['star_class']}), the "
        f"habitable-zone class is **{hz_label}**. "
        "Conservative means between the runaway-greenhouse and maximum-greenhouse "
        "limits; optimistic-only widens that band to the recent-Venus and early-Mars "
        "limits. This is a statement about received flux, not about water or life."
    )

    st.markdown("#### Radius and its provenance")
    st.markdown(labelling.radius_text(row["pl_rade"], row["radius_provenance"]))
    st.markdown(labelling.radius_reference(row["radius_provenance"]))
    render_ml_estimate(estimate, row)

st.write("")
st.markdown("#### Probabilistic criteria")
st.caption(
    "Each probability is printed with its uncertainty_status. Where an input it "
    "depends on was held fixed at its archive value, the number is labelled "
    "“conditional on a fixed input (uncertainty unknown)” and is not an "
    "unqualified probability."
)
for name, label in labelling.PROBABILITY_LABELS.items():
    st.markdown(
        f"- **{label}:** {labelling.format_probability(name, row[name], status)}"
    )

if row["boundary_sensitive"]:
    probability = "p_conservative_hz" if hz_class == "conservative" else "p_any_hz"
    st.warning(
        "**Boundary sensitive.** The point classification is "
        f"{ui.HZ_LABELS.get(hz_class, hz_class)}, but "
        f"{labelling.format_probability(probability, row[probability], status)} — "
        "below the 0.7 threshold, so the class is an edge call rather than a firm "
        "one."
    )

with st.expander("Methods and limitations"):
    methods_content.render()

st.write("")
st.caption(
    f"Data: {provenance['pscd']['label']}, retrieved {long_date(retrieval_date)} "
    f"(timestamp `{provenance['pscd']['retrieval_timestamp']}`, sha256 "
    f"`{provenance['pscd']['sha256_prefix']}…`)."
)
st.caption(
    "References: Schulze-Makuch et al. (2011), Astrobiology 11, 1041, for the "
    "ESI-style form and the 0.57 radius weight · Kopparapu et al. (2013), ApJ 765, "
    "131, and Kopparapu et al. (2014), ApJL 787, L29, for the habitable-zone flux "
    "limits (valid for 2600-7200 K)."
)
