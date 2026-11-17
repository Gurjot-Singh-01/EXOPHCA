"""Calculator: the exact ESI-style score and habitable-zone class for one planet.

The single-planet behaviour is unchanged from the original app: the same inputs,
the same validation, the same Kopparapu limits, the same messages. Presets are
computed from the reference tables instead of being typed in - the Solar System
from ``data/reference/solar_system.csv``, the archive from
``data/processed/planets_assessed.csv`` - the Size Comparison is a true-to-scale
drawing, and a batch section scores a CSV held only in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

import batch  # noqa: E402
import calculator_widgets as widgets  # noqa: E402
import data_access  # noqa: E402
import presets  # noqa: E402
import ui  # noqa: E402
from exophca.science.assessment import determine_size_regime  # noqa: E402
from exophca.science.esi import (  # noqa: E402
    calculate_esi,
    classify_flags,
    validate_nonnegative_finite,
    validate_positive_finite,
)
from exophca.science.habitable_zone import classify_hz, hz_limits  # noqa: E402

ui.page_header(
    "Calculator",
    "Enter precise radius and insolation flux to get the exact score and the "
    "habitable-zone class for a star of known temperature. Presets are read from "
    "the processed planet table; a whole CSV can be scored in one go.",
    eyebrow="Tools",
)

planets_df = data_access.load_or_stop(data_access.load_planets_assessed, "Planet assessments")

# ---------- Presets, computed from the reference tables ----------
GROUPS = presets.preset_groups(planets_df)
NO_PRESET = "-- Select a preset --"
STAR_CLASSES = ["G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)"]
JUPITER = GROUPS[presets.GROUP_SOLAR].get("Jupiter")


st.markdown("### Enter known planet properties")
st.caption("Use this when you have precise radius and insolation flux measurements.")

inputs, results = st.columns([1, 1], gap="large")

with inputs:
    group = st.radio(
        "Preset group",
        list(GROUPS),
        horizontal=True,
        key="calc_preset_group",
        help="The Solar System group is computed from the NASA reference table; "
        "the archive group is every scored planet in the processed data.",
    )
    choices = list(GROUPS[group])
    preset_choice = st.selectbox(
        "Or load a known planet:",
        [NO_PRESET] + choices,
        key=f"calc_preset_{group}",
        help=f"{len(choices):,} planets in this group. Type a name to search the list.",
    )
    preset = GROUPS[group].get(preset_choice)

    if group == presets.GROUP_SOLAR:
        jupiter_text = (
            f"Jupiter is {JUPITER['pl_rade']:.2f} R\u2295" if JUPITER else "Jupiter"
        )
        st.caption(
            "Radii are mean (volumetric) radii, not equatorial radii, so "
            f"{jupiter_text} rather than the familiar equatorial 11.2. Radius is "
            "the mean radius divided by Earth's mean radius and flux is (Earth's "
            "semimajor axis \u00f7 the planet's) squared, both computed from "
            f"`{presets.SOLAR_SYSTEM_LABEL}`; Teff is the Sun's and the host "
            "class is G for every Solar System preset."
        )
    else:
        st.caption(
            f"All {len(choices):,} scored planets are available - type a name to "
            "search. Radius, flux, Teff and star class are read from "
            f"`{presets.ARCHIVE_LABEL}`, including the stellar temperature, so no "
            "value is typed in."
        )

    pl_rade = st.number_input(
        "Planet Radius (Earth radii)",
        value=preset["pl_rade"] if preset else 1.0,
        step=0.01,
        key=f"calc_radius_{group}_{preset_choice}",
    )
    pl_insol = st.number_input(
        "Insolation Flux (Earth flux = 1.0)",
        min_value=0.0,
        value=preset["pl_insol"] if preset else 1.0,
        step=0.01,
        format="%.4f",
        key=f"calc_flux_{group}_{preset_choice}",
    )
    preset_star_class = preset["star_class"] if preset else None
    star_class = st.selectbox(
        "Host Star Class",
        STAR_CLASSES,
        index=(STAR_CLASSES.index(preset_star_class) if preset_star_class in STAR_CLASSES else 0),
        key=f"calc_star_{group}_{preset_choice}",
    )
    st_teff = st.number_input(
        "Stellar Effective Temperature (K, optional)",
        min_value=0.0,
        value=(preset["st_teff"] if preset and preset["st_teff"] else None),
        step=50.0,
        format="%.1f",
        help="Used only for the habitable-zone classification. Leave empty to skip it.",
        key=f"calc_teff_{group}_{preset_choice}",
    )
    if preset:
        st.caption(f"Preset values read from {preset['source']}.")

    exact_clicked = st.button("Calculate Compatibility Scores", key="exact")

# A preset is already scored on selection; the button scores whatever is in the
# boxes after an edit.
score_requested = exact_clicked or preset is not None

exact_input_error = None
validated_radius = None
validated_insolation = None
esi = None
flags = None
try:
    validated_radius = validate_positive_finite("Planet radius", pl_rade)
    validated_insolation = validate_nonnegative_finite("Insolation flux", pl_insol)
    esi = calculate_esi(validated_radius, validated_insolation)
    flags = classify_flags(validated_radius, star_class)
except ValueError as error:
    exact_input_error = str(error)

with results:
    if not score_requested:
        st.info(
            "Results appear here when you choose a preset or press **Calculate "
            "Compatibility Scores**. Nothing is assumed about radius or flux: both "
            "must be positive and finite (flux may be zero)."
        )
    elif exact_input_error:
        st.error(exact_input_error)
    else:
        st.metric("ESI-style radius-flux similarity", f"{esi:.3f}")
        st.text(f"Host is K-class: {'Yes' if flags['host_is_k_class'] else 'No'}")
        st.text(f"Radius in 1.0–1.5 R⊕: {'Yes' if flags['radius_in_1_to_1p5'] else 'No'}")
        st.caption(
            "ESI-style radius-flux similarity = sqrt((1 - |(R - 1)/(R + 1)|)^0.57 × "
            "(1 - |(S - 1)/(S + 1)|)^0.7), where R is radius in Earth radii "
            "and S is insolation in Earth flux. For R > 0 and S >= 0, similarity "
            "range: [0, 1]. "
            "The informational flags above do not alter or add to the similarity score."
        )

        st.text(f"Size regime: {determine_size_regime(validated_radius)}")

        st.markdown("#### Habitable-zone position")
        if st_teff is None:
            st.info(
                "Habitable zone undetermined — no stellar effective temperature was "
                "provided, so the Kopparapu et al. flux limits cannot be evaluated."
            )
        else:
            teff_value = float(st_teff)
            limits = hz_limits(teff_value)
            if not limits["in_valid_range"]:
                st.info(
                    "Habitable zone undetermined — the Kopparapu et al. (2013, 2014) "
                    "limits are fitted over 2600-7200 K and "
                    f"{teff_value:.0f} K is outside that range. The limits are "
                    "not extrapolated."
                )
            else:
                hz_class = classify_hz(validated_insolation, teff_value)
                hz_labels = {
                    "conservative": "conservative habitable zone",
                    "optimistic_only": "optimistic habitable zone only",
                    "too_hot": "too hot (inside the recent-Venus limit)",
                    "too_cold": "too cold (outside the early-Mars limit)",
                    "undetermined": "undetermined",
                }
                # The class is a sentence, not a number: as an st.metric value it
                # is set in the metric font and clipped with an ellipsis in a
                # narrow column, so it is shown as wrapping text instead.
                st.markdown(
                    '<div class="ex-eyebrow">Habitable-zone class</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{hz_labels.get(hz_class, hz_class)}**")
                st.caption(
                    f"At {teff_value:.0f} K the conservative zone is "
                    f"{limits['conservative_outer']:.3f}-{limits['conservative_inner']:.3f} S⊕ "
                    f"and the optimistic zone is "
                    f"{limits['optimistic_outer']:.3f}-{limits['optimistic_inner']:.3f} S⊕. "
                    "Habitable-zone position is not evidence of habitability or life."
                )

if score_requested and exact_input_error is None:
    st.divider()
    visual_left, visual_right = st.columns(2)
    with visual_left:
        st.markdown("#### Size Comparison")
        comparison = widgets.size_comparison(pl_rade, pl_insol)
        st.html(comparison["html"])
        st.caption(
            "Drawn to scale: both circles use the same linear scale, so the "
            f"planet's diameter is {comparison['ratio']:.3f}× Earth's - exactly its "
            "radius in Earth radii. "
            + (
                comparison["note"] + " Both circles were reduced by the same factor, "
                "so the ratio is unchanged."
                if comparison["note"]
                else ""
            )
        )
    with visual_right:
        st.markdown("#### System Preview")
        st.caption(
            "Illustration, not to scale: the star's size and the star-planet "
            "separation are decoration. The planet's circle grows linearly with its "
            "radius (capped to the panel); its colour is the insolation flux "
            "(blue = cold, red = hot)."
        )
        st.html(widgets.system_illustration(pl_rade, pl_insol, star_class)["html"])

    st.markdown("#### Where This Planet Lands")
    unknown = planets_df.dropna(subset=["pl_insol", "pl_rade"])
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.scatter(
        unknown["pl_insol"],
        unknown["pl_rade"],
        color="#9AA0A6",
        alpha=0.35,
        s=12,
        label="Assessed planets",
    )
    axis.scatter(
        [pl_insol], [pl_rade], color="#2F6B7A", s=150, edgecolor="black", zorder=5, label="Your planet"
    )
    axis.axvline(1.0, color="#C9A227", linestyle="--", alpha=0.6)
    axis.axhline(1.0, color="#4C7A4C", linestyle="--", alpha=0.6)
    axis.set_xscale("log")
    axis.set_xlabel("Insolation Flux (log scale, Earth = 1.0)")
    axis.set_ylabel("Planet Radius (Earth radii)")
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)

# ---------------------------------------------------------------- batch
st.divider()
st.markdown("### Batch: score a CSV of planets")
st.caption(
    "Columns: **pl_rade** and **pl_insol** are required; **st_teff** and **name** "
    "are optional. At most 5,000 rows. Rows are rejected - with the line number "
    "and the reason - when a value is missing, not a number, a radius is not "
    "positive or a flux is negative. A missing or out-of-range stellar "
    "temperature is not an error: the habitable-zone class becomes "
    "`undetermined`, because the Kopparapu limits are not extrapolated."
)
st.caption(
    "The uploaded file is processed in memory and is **not stored anywhere**: no "
    "copy is written to disk, to a temporary folder or to the session, and only "
    "the scored rows you download leave the page."
)

uploaded = st.file_uploader("Planets CSV", type=["csv"], accept_multiple_files=False)

if uploaded is None:
    st.info("Choose a CSV to score. Nothing is uploaded or saved until you do.")
else:
    result = batch.score_batch(uploaded.getvalue())

    summary = (
        f"**{result.accepted:,} of {result.total_rows:,} rows scored**"
        if result.total_rows
        else "**No rows were read**"
    )
    st.markdown(summary + (f"; {result.rejected:,} rejected." if result.rejected else "."))

    if result.errors:
        shown = result.reported_errors
        st.error(
            f"{len(result.errors):,} row error(s). "
            f"The first {len(shown)} are listed below."
        )
        st.code("\n".join(str(error) for error in shown), language="text")

    if result.accepted:
        scored = batch.export_frame(result)
        st.dataframe(
            scored,
            hide_index=True,
            height=320,
            column_config={
                "pl_rade": st.column_config.NumberColumn("Radius (R⊕)", format="%.3f"),
                "pl_insol": st.column_config.NumberColumn("Flux (S⊕)", format="%.4f"),
                "st_teff": st.column_config.NumberColumn("Teff (K)", format="%.0f"),
                "base_score": st.column_config.NumberColumn("Score", format="%.3f"),
                "disclaimer": st.column_config.TextColumn("Disclaimer", width="large"),
            },
        )
        st.caption(
            "Rows whose Teff is missing or outside 2600-7200 K are scored but their "
            "habitable-zone class is `undetermined`."
        )
        st.download_button(
            "Download scored rows (CSV)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="exophca_batch_scored.csv",
            mime="text/csv",
            key="batch_export",
        )

ui.disclaimer()
