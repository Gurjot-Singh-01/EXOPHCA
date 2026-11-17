"""Overview: what the data are, when they were retrieved, and what the scores mean."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st  # noqa: E402

import data_access  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Overview",
    "Physical similarity to Earth and a position inside the habitable zone, "
    "computed from archive radius and insolation flux. Neither is a probability "
    "that a planet is habitable.",
    eyebrow="EXOPHCA",
)

ui.disclaimer()

provenance = data_access.load_or_stop(data_access.load_provenance, "Provenance records")
assessed = data_access.load_or_stop(data_access.load_planets_assessed, "Planet assessments")
toi_assessed = data_access.load_or_stop(data_access.load_toi_assessed, "TESS candidate assessments")


def _long_date(iso: str | None) -> str:
    if not iso:
        return "unknown"
    try:
        return date.fromisoformat(iso).strftime("%d %B %Y")
    except ValueError:
        return iso


st.subheader("Datasets")
pscd = provenance["pscd"]
toi = provenance["toi"]

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown(f"**{pscd['label']}**")
        st.caption(f"source file: `{pscd['original_file']}` · retrieved {_long_date(pscd['retrieval_date'])}")
        ui.metric_row(
            [
                ("Archive rows", f"{pscd['row_count']:,}"),
                ("Assessed planets", f"{len(assessed):,}"),
            ]
        )
        st.caption(
            f"{pscd['column_count']} archive columns · sha256 `{pscd['sha256_prefix']}…` · "
            f"retrieval timestamp `{pscd['retrieval_timestamp']}`"
        )

with right:
    with st.container(border=True):
        st.markdown(f"**{toi['label']}**")
        st.caption(f"source file: `{toi['original_file']}` · retrieved {_long_date(toi['retrieval_date'])}")
        ui.metric_row(
            [
                ("Archive rows", f"{toi['row_count']:,}"),
                ("Assessed candidates", f"{len(toi_assessed):,}"),
            ]
        )
        st.caption(
            f"{toi['column_count']} archive columns · sha256 `{toi['sha256_prefix']}…` · "
            f"retrieval timestamp `{toi['retrieval_timestamp']}`"
        )

st.caption(
    "Assessed rows are the planets with a positive radius and insolation flux and a "
    "stellar effective temperature inside the range where the habitable-zone limits are "
    "valid. The raw archive tables also contain rows with missing or unusable values."
)
st.write("")

st.subheader("How the assessed planets divide up")
hz_counts = assessed["hz_class"].value_counts()
tier_counts = assessed["tier"].value_counts()

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("**Habitable-zone class**")
    st.plotly_chart(
        ui.bar_chart(
            [ui.HZ_LABELS.get(name, name) for name in hz_counts.index],
            hz_counts.to_numpy(),
            colors=[ui.HZ_COLORS.get(name, ui.NEUTRAL) for name in hz_counts.index],
        )
    )
with chart_right:
    st.markdown("**Compatibility tier**")
    tier_descriptions = {
        "A": "conservative zone, likely rocky",
        "B": "zone edge or intermediate size",
        "C": "in a zone, too large to be rocky",
        "D": "outside both zones",
        "U": "undetermined",
    }
    st.plotly_chart(
        ui.bar_chart(
            [f"{tier} — {tier_descriptions.get(tier, '')}" for tier in tier_counts.index],
            tier_counts.to_numpy(),
            colors=[ui.TIER_COLORS.get(tier, ui.NEUTRAL) for tier in tier_counts.index],
        )
    )

st.write("")

st.subheader("What the score means — and what it does not")
meaning_left, meaning_right = st.columns(2)
with meaning_left:
    st.markdown(
        """
        **The score.** A number in [0, 1] built from two measured quantities:
        planet radius and the starlight the planet receives, both compared with
        Earth. Earth scores 1.0. A score of 0.9 means the planet's size and
        sunlight are close to Earth's; it says nothing about its surface,
        atmosphere or temperature. The formula is the published Earth Similarity
        Index shape with the radius weight 0.57 and the escape-velocity weight
        0.70 borrowed for insolation flux — that second weight is not a
        published flux weight.

        **The habitable-zone class.** Where the planet's insolation flux falls
        relative to the Kopparapu et al. (2013, 2014) limits for its star's
        temperature. *Conservative* means inside the band between the runaway
        greenhouse and maximum greenhouse limits; *optimistic only* widens that
        band to the recent-Venus and early-Mars limits. This is a statement
        about received flux, assuming an Earth-like atmosphere — not about
        water, air or life.
        """
    )
with meaning_right:
    st.markdown(
        """
        **What it does not mean.** The score is not a probability of life, not a
        probability of habitability, and not a detection. It does not use
        atmospheric measurements, and most planets in this dataset have none.
        The tier and the evidence quality describe how well the inputs are
        measured, not how promising the planet is.

        **Where the numbers are uncertain.** Where the archive reports errors,
        each score is accompanied by a Monte Carlo 16th–84th percentile range
        and four probabilities. Those probabilities are only as good as the
        inputs that were varied: the app always shows the accompanying
        `uncertainty_status`, and where an input was held fixed at its archive
        value the probability is labelled *conditional on a fixed input*.

        **Radii.** Some planets have no measured radius; theirs is estimated
        from a mass–radius relation. Those rows are labelled *radius modelled,
        not measured* wherever a radius appears.
        """
    )

GLOSSARY = {
    "Insolation flux": "How much energy a planet receives from its star, relative to "
    "Earth (Earth = 1.0). A planet with 2.0 receives twice Earth's sunlight; 0.5 "
    "receives half.",
    "Earth radii (R⊕)": "A planet's size compared with Earth. 1.0 is Earth-sized; "
    "11.2 is Jupiter-sized.",
    "Habitable zone": "The band of insolation where a planet with an Earth-like "
    "atmosphere could plausibly hold liquid water on its surface.",
    "ESI-style radius-flux similarity": "The score shown throughout the app: radius "
    "weight 0.57 (published) and insolation weight 0.70 (the published "
    "escape-velocity weight borrowed for flux), in [0, 1] with Earth = 1.0.",
    "Stellar effective temperature (st_teff)": "The surface temperature of the host "
    "star in kelvin; the Sun is about 5,778 K. It sets the habitable-zone flux "
    "limits and, outside 2600-7200 K, the limits are not evaluated.",
    "Star class": "A rough temperature class: A and F stars are hotter than the Sun, "
    "G stars are Sun-like, K and M stars are cooler and longer-lived.",
    "Tier": "A-D and U. A is conservative habitable zone and likely rocky; B is a "
    "zone edge or an intermediate size; C is in a zone but too large to be rocky; "
    "D is outside both zones; U is undetermined.",
    "Evidence quality": "high / medium / low, from whether the radius and flux are "
    "measured and whether their errors are published. It says nothing about "
    "habitability.",
    "Uncertainty status": "Which inputs the Monte Carlo sampled: full (both), "
    "radius_only, flux_only, none, or limit_value. Probabilities built on an "
    "unsampled input are labelled conditional.",
    "Radius provenance": "measured, modelled (estimated from a mass-radius "
    "relation) or derived_unclear.",
    "Boundary sensitive": "True when the planet sits in a zone but the Monte Carlo "
    "probability of being in that zone is below 0.7.",
}

with st.expander("Glossary of terms"):
    for term, explanation in GLOSSARY.items():
        st.markdown(f"**{term}** — {explanation}")

ui.disclaimer()
st.caption(
    "Source: NASA Exoplanet Archive (PSCompPars and TESS Objects of Interest), "
    "retrieved as recorded above."
)
