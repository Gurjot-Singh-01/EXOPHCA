"""Diagrams: the habitable-zone plane and the mass-radius plane, every planet drawn."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st  # noqa: E402

import data_access  # noqa: E402
import plot_data  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Diagrams",
    "Where the assessed planets sit against the habitable-zone limits and how "
    "their radii compare with their masses. Every point is drawn: nothing is "
    "sampled or thinned.",
    eyebrow="Research",
)

ui.disclaimer()

frame = data_access.load_or_stop(data_access.load_planet_explorer, "Planet explorer table")

# ------------------------------------------------------------ habitable zone
st.subheader("Habitable zone: insolation flux against host temperature")
bands = plot_data.hz_band_frame()
hz_points, hz_counts = plot_data.hz_points(frame)

st.plotly_chart(plot_data.hz_band_figure(hz_points, bands))
st.caption(
    f"{hz_counts['drawn']:,} points drawn of {hz_counts['available']:,} assessed "
    f"planets; {hz_counts['left_out']:,} are left out. Why: "
    f"{hz_counts['teff_missing']:,} have no host effective temperature, "
    f"{hz_counts['teff_out_of_range']:,} have a host temperature outside the fitted "
    f"2600-7200 K range, and {hz_counts['flux_invalid']:,} have a missing or "
    f"non-positive flux. {hz_counts['multiple']:,} planets fall into more than one of "
    "those groups, so the three counts add up to more than the "
    f"{hz_counts['left_out']:,} planets actually left out. The Kopparapu limits are "
    "not extrapolated, so those planets cannot be placed against the bands and are "
    "left out rather than plotted at a guess."
)
st.caption(
    "Bands are computed from `hz_limits` over 2600-7200 K: the conservative band "
    "runs from the runaway-greenhouse limit (inner) to the maximum-greenhouse limit "
    "(outer), and the optimistic band widens it to the recent-Venus and early-Mars "
    "limits. Colour is the habitable-zone class, as in the Explorer: "
    + "; ".join(
        f"{ui.HZ_LABELS[name]} = {ui.HZ_COLOR_NAMES.get(name, name)}"
        for name in ui.HZ_LABELS
    )
    + ". "
    + ui.PROVENANCE_LEGEND
)

# ------------------------------------------------------------ mass-radius
st.subheader("Mass against radius")
mr_points, mr_counts = plot_data.mass_radius_points(frame)

st.plotly_chart(plot_data.mass_radius_figure(mr_points))
st.caption(
    f"{mr_counts['drawn']:,} points drawn of {mr_counts['available']:,} assessed "
    f"planets, both axes logarithmic. {mr_counts['missing']:,} have no mass or no "
    f"radius and {mr_counts['non_positive']:,} have a non-positive value, which a "
    "log axis cannot show. Colour is the radius provenance, and the hover text "
    "gives the mass provenance."
)
st.caption(
    "Read the flat band of modelled giants with care: where a radius was not "
    "measured it was estimated from the planet's mass through a mass-radius "
    "relation, so it lies on that relation by construction. The band is the shape "
    "of the relation, not a measurement, and its scatter is not propagated into the "
    "score here."
)
