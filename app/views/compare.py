"""Compare: two to four planets side by side, with Earth as the fixed reference."""

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
import labelling  # noqa: E402
import plot_data  # noqa: E402
import reference_earth  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Compare",
    "Up to three assessed planets against Earth, on the same numbers: the score "
    "with its 16th-84th percentile range, size and provenance, habitable-zone "
    "class, tier and evidence.",
    eyebrow="Research",
)

ui.disclaimer()

frame = data_access.load_or_stop(data_access.load_planet_explorer, "Planet explorer table")
provenance = data_access.load_or_stop(data_access.load_provenance, "Provenance records")
retrieval_date = provenance["pscd"]["retrieval_date"]

names = sorted(frame["pl_name"].dropna().unique())
DEFAULT_SELECTION = [name for name in ("Kepler-452 b", "Proxima Cen b") if name in names]
if not DEFAULT_SELECTION:
    DEFAULT_SELECTION = names[:2]

EARTH = reference_earth.earth_reference()
EARTH_LABEL = EARTH["pl_name"]

chosen = st.multiselect(
    "Planets to compare (up to 3; Earth is always shown)",
    names,
    default=DEFAULT_SELECTION,
    max_selections=3,
    help="Choose two or three planets. Earth is added automatically as the reference row.",
)

st.caption(
    "Earth is the reference row by definition: the formula gives it a score of "
    "1.000, and its radius and flux are 1.000 R⊕ and 1.000 S⊕. Its "
    f"habitable-zone class and tier are not typed in either: `classify_hz` at "
    f"{EARTH['pl_insol']:.1f} S⊕ and {EARTH['st_teff']:.0f} K returns "
    f"{EARTH['hz_class']!r} and the tier rules on that class and the "
    f"{EARTH['size_regime']!r} size regime return {EARTH['tier']!r}. It has no "
    "Monte Carlo range."
)

# ---------------------------------------------------------------- rows
EARTH_PROVENANCE_TEXT = (
    "Reference row: Earth defines the scale, so its radius and flux are 1.000 by "
    "construction rather than read from the archive."
)

table = plot_data.compare_rows(frame, chosen, earth=EARTH)
rows = table.to_dict("records")

# ---------------------------------------------------------------- warning
modelled = [
    row["Planet"]
    for row in rows[1:]
    if frame.loc[frame["pl_name"] == row["Planet"], "radius_provenance"].iloc[0]
    in ("modelled", "derived_unclear")
]
if modelled:
    if len(modelled) == 1:
        names_text = modelled[0]
    elif len(modelled) == 2:
        names_text = f"{modelled[0]} and {modelled[1]}"
    else:
        names_text = ", ".join(modelled[:-1]) + f" and {modelled[-1]}"

    plural = len(modelled) > 1
    verb = "have" if plural else "has"
    pronoun = "they are" if plural else "it is"
    subject = "each planet" if plural else "the planet"
    possessive = "Their" if plural else "Its"
    st.warning(
        f"**Modelled radius.** {names_text} {verb} no measured radius: {pronoun} "
        f"estimated from {subject}'s mass ({labelling.MODELLED_RADIUS_TEXT}). "
        f"{possessive} size, score and size-dependent probabilities therefore inherit "
        "the scatter of a mass-radius relation on top of the measurement error."
    )

# ---------------------------------------------------------------- chart
st.plotly_chart(plot_data.score_range_figure(table))
st.caption(
    "The bar spans the 16th-84th percentile of the Monte Carlo score distribution "
    "and the tick marks the point score. The dashed line is the Earth reference at "
    "1.000 and the diamond is Earth's own row. The range propagates the published "
    "radius and flux errors only; it is not an interval for habitability."
)

# ---------------------------------------------------------------- table
columns = [
    "Planet",
    "Score",
    "Radius",
    "Radius provenance",
    "HZ class",
    "Tier",
    "Evidence",
    "uncertainty_status",
] + list(labelling.PROBABILITY_DEPENDENCIES)
shown = table[columns].copy()
shown["Score"] = shown["Score"].map(lambda value: f"{value:.3f}")

st.dataframe(
    shown,
    hide_index=True,
    column_config={
        "Score": st.column_config.TextColumn("Score", width="small"),
        "Radius": st.column_config.TextColumn("Radius", width="medium"),
        "Radius provenance": st.column_config.TextColumn(
            "Radius provenance",
            width="large",
            help=(
                "A tag naming where the radius comes from. The full sentence for "
                "each selected planet is listed under the table, where it has the "
                "width to be read without truncation."
            ),
        ),
        "HZ class": st.column_config.TextColumn("Habitable-zone class", width="medium"),
        "Tier": st.column_config.TextColumn("Tier", width="small"),
        "Evidence": st.column_config.TextColumn("Evidence", width="small"),
        "uncertainty_status": st.column_config.TextColumn(
            "uncertainty_status", width="small"
        ),
        **{
            name: st.column_config.TextColumn(label, width="large")
            for name, label in labelling.PROBABILITY_LABELS.items()
        },
    },
)

st.caption(
    "Every probability is printed with its uncertainty_status; where the input it "
    "depends on was held fixed, the cell says "
    f"\u201c{labelling.CONDITIONAL_TEXT}\u201d instead of showing a bare number. "
    f"Data retrieved {retrieval_date}."
)

# The provenance sentences do not fit a table cell, so they are listed in full
# where the text has the whole width to itself.
st.markdown("##### Radius provenance, in full")
provenance_lines = []
for row in rows:
    sentence = (
        EARTH_PROVENANCE_TEXT
        if row["Planet"] == EARTH_LABEL
        else labelling.radius_reference(
            frame.loc[frame["pl_name"] == row["Planet"], "radius_provenance"].iloc[0]
        )
    )
    provenance_lines.append(f"- **{row['Planet']}** ({row['Radius']}) \u2014 {sentence}")
st.markdown("\n".join(provenance_lines))
