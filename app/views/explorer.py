"""Planet Explorer: filter the assessed planets and export the filtered table.

Every control in the filter panel lives inside one form, so the panel has a
single, visible commit point: **Apply filters** (pressing Enter in any box also
submits the form). A bare ``st.text_input`` commits only on Enter or blur, which
is why a typed search could sit in the box while the table still showed every
planet. The line above the table always reports how many planets the *applied*
filters matched.
"""

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
import query  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Planet Explorer",
    "Every assessed planet with its score, habitable-zone class, evidence tier and "
    "provenance. Sort by any column heading; filter below; export the filtered rows "
    "as CSV.",
    eyebrow="Explore",
)

ui.disclaimer()

frame = data_access.load_or_stop(data_access.load_planet_explorer, "Planet explorer table")
provenance = data_access.load_or_stop(data_access.load_provenance, "Provenance records")
retrieval_date = provenance["pscd"]["retrieval_date"]

EVIDENCE_ORDER = ["high", "medium", "low"]
TIER_ORDER = ["A", "B", "C", "D", "U"]
HZ_ORDER = ["conservative", "optimistic_only", "too_hot", "too_cold", "undetermined"]
STATUS_ORDER = ["full", "radius_only", "flux_only", "none", "limit_value"]

# The keys of every widget in the filter form, and the value "Reset filters"
# puts back into each one. The defaults are assigned rather than dropped so the
# browser also empties the boxes it is holding: a widget whose key is simply
# deleted can keep showing the text the user typed.
FILTER_DEFAULTS = {
    "exo_search": "",
    "exo_methods": [],
    "exo_star_classes": [],
    "exo_provenance": [],
    "exo_evidence": [],
    "exo_tiers": [],
    "exo_hz": [],
    "exo_status": [],
    "exo_score": (0.0, 1.0),
    "exo_sort": "base_score",
    "exo_order": "Descending",
    "exo_probabilities": False,
}

FILTER_HINT = (
    "Change any control, then press **Apply filters** (or press Enter in a box): "
    "typing alone does not change the table, and nothing is filtered until the "
    "form is applied. **Reset filters** clears every control. The line above the "
    "table always reports how many planets the applied filters matched."
)


def _ordered(values, order):
    """Present values in the canonical order, then anything unexpected alphabetically."""
    values = list(values)
    present = [value for value in order if value in values]
    extra = sorted(value for value in values if value not in order)
    return present + extra


# A reset has to be honoured before the widgets are created: the session-state
# value of a widget cannot be changed once that widget exists in the same run.
if st.session_state.pop("_explorer_reset_requested", False):
    for key, default in FILTER_DEFAULTS.items():
        st.session_state[key] = default

with st.expander("Filters", expanded=True):
    st.caption(FILTER_HINT)

    with st.form("explorer_filters"):
        search = st.text_input(
            "Planet name contains",
            value="",
            key="exo_search",
            placeholder="e.g. TRAPPIST, Kepler-452, Proxima, Barnard's star",
            help=(
                "Case-insensitive substring match, read literally (no regular "
                "expressions) with surrounding spaces ignored. It matches the "
                "planet name or its host star, so a host name finds all of that "
                "star's planets."
            ),
        )

        first, second, third = st.columns(3)
        methods = first.multiselect(
            "Discovery method",
            _ordered(frame["discoverymethod"].dropna().unique(), []),
            key="exo_methods",
        )
        star_classes = second.multiselect(
            "Star class",
            _ordered(
                frame["star_class"].dropna().unique(),
                ["G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)"],
            ),
            key="exo_star_classes",
        )
        provenance_values = third.multiselect(
            "Radius provenance",
            _ordered(
                frame["radius_provenance"].dropna().unique(),
                ["measured", "modelled", "derived_unclear"],
            ),
            key="exo_provenance",
        )

        fourth, fifth, sixth = st.columns(3)
        evidence = fourth.multiselect(
            "Evidence quality",
            _ordered(frame["evidence_quality"].dropna().unique(), EVIDENCE_ORDER),
            key="exo_evidence",
        )
        tiers = fifth.multiselect(
            "Tier",
            _ordered(frame["tier"].dropna().unique(), TIER_ORDER),
            key="exo_tiers",
        )
        hz_classes = sixth.multiselect(
            "Habitable-zone class",
            _ordered(frame["hz_class"].dropna().unique(), HZ_ORDER),
            key="exo_hz",
        )

        seventh, eighth, ninth = st.columns(3)
        statuses = seventh.multiselect(
            "Uncertainty status",
            _ordered(frame["uncertainty_status"].dropna().unique(), STATUS_ORDER),
            key="exo_status",
            help=(
                "full: radius and flux both sampled · radius_only: flux held fixed · "
                "flux_only: radius held fixed · none: both held fixed · "
                "limit_value: an archive limit value was held fixed."
            ),
        )
        score_range = eighth.slider(
            "Score range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key="exo_score",
            help="Inclusive bounds on the point score.",
        )
        sort_by = ninth.selectbox(
            "Sort by",
            query.SORTABLE_COLUMNS,
            index=query.SORTABLE_COLUMNS.index("base_score"),
            key="exo_sort",
        )

        order_column, probability_column, spacer_column = st.columns([1, 1, 1])
        descending = order_column.radio(
            "Order",
            ["Descending", "Ascending"],
            horizontal=True,
            index=0,
            key="exo_order",
        ) == "Descending"
        show_probabilities = probability_column.toggle(
            "Show probabilistic criteria",
            value=False,
            key="exo_probabilities",
            help=(
                "Adds the four Monte Carlo probabilities as text. Each one is printed "
                "with its uncertainty_status, and labelled conditional when an input it "
                "depends on was held fixed."
            ),
        )

        apply_column, reset_column = st.columns([1, 1])
        apply_column.form_submit_button("Apply filters", type="primary")
        reset_clicked = reset_column.form_submit_button("Reset filters")
        spacer_column.caption("Nothing changes until you apply.")

    if reset_clicked:
        st.session_state["_explorer_reset_requested"] = True
        st.rerun()

filtered = query.apply_filters(
    frame,
    search=search,
    discovery_methods=methods,
    star_classes=star_classes,
    radius_provenance=provenance_values,
    evidence_quality=evidence,
    tiers=tiers,
    hz_classes=hz_classes,
    uncertainty_status=statuses,
    score_range=score_range,
)

shown = query.sort_frame(filtered, sort_by, descending=descending)
table = query.with_radius_note(shown)[query.DISPLAY_COLUMNS].copy()

if show_probabilities:
    labelled = labelling.labelled_probability_columns(shown)
    for name in labelling.PROBABILITY_DEPENDENCIES:
        table[name] = labelled[name].to_numpy()

st.markdown(
    f"**{len(shown):,} of {len(frame):,} planets** match the current filters."
)
if (search or methods or star_classes or provenance_values or evidence or tiers
        or hz_classes or statuses or score_range != (0.0, 1.0)):
    st.caption(
        "Filters are applied. Press **Reset filters** in the panel above to see "
        "every assessed planet again."
    )

if shown.empty:
    if search.strip():
        st.warning(
            f"**No planet matches.** No planet name or host star contains "
            f"“{search.strip()}”. Check the spelling, widen the other filters, or "
            f"press **Reset filters** to see all {len(frame):,} assessed planets."
        )
    else:
        st.warning(
            "**No planet matches the current filters.** Widen them, or press "
            f"**Reset filters** in the panel above, to see all {len(frame):,} "
            "assessed planets."
        )

column_config = {
    "pl_name": st.column_config.TextColumn("Planet", width="medium"),
    "pl_rade": st.column_config.NumberColumn("Radius (R⊕)", format="%.3f"),
    "radius_note": st.column_config.TextColumn("Radius provenance", width="medium"),
    "pl_insol": st.column_config.NumberColumn("Insolation (S⊕)", format="%.4f"),
    "base_score": st.column_config.NumberColumn("Score", format="%.3f"),
    "base_score_p16": st.column_config.NumberColumn("Score p16", format="%.3f"),
    "base_score_p84": st.column_config.NumberColumn("Score p84", format="%.3f"),
    "boundary_sensitive": st.column_config.CheckboxColumn("Boundary sensitive"),
    "disc_year": st.column_config.NumberColumn("Discovered", format="%d"),
}
for name, label in labelling.PROBABILITY_LABELS.items():
    column_config[name] = st.column_config.TextColumn(label, width="large")

st.dataframe(
    table,
    hide_index=True,
    height=520,
    column_config={
        name: config for name, config in column_config.items() if name in table.columns
    },
)

with st.expander("Radius–insolation plane for the filtered planets"):
    chartable, dropped = query.plot_points(shown)
    if chartable.empty:
        st.caption(
            "No filtered planet has both a positive radius and flux to plot."
        )
    else:
        st.plotly_chart(ui.hz_scatter(chartable))
        st.caption(
            f"{len(chartable):,} points drawn — every one of the {len(shown):,} "
            "filtered planets with a positive radius and flux. No sampling cap is "
            f"applied and nothing is thinned. {dropped:,} filtered planets have a "
            "missing or non-positive radius or flux and cannot be placed on a log "
            "axis; they remain in the table and the export."
        )
        st.caption(
            "Colour is the habitable-zone class: "
            + "; ".join(
                f"{ui.HZ_LABELS[name]} = {ui.HZ_COLOR_NAMES.get(name, name)} "
                f"({ui.HZ_COLORS[name]})"
                for name in _ordered(chartable["hz_class"].unique(), HZ_ORDER)
            )
            + ". "
            + ui.PROVENANCE_LEGEND
        )

st.divider()

ui.disclaimer()
export_frame = query.build_export_frame(shown, retrieval_date)
st.caption(
    f"The CSV holds the {len(export_frame):,} filtered rows with provenance and "
    "uncertainty columns (radius provenance, evidence quality, score percentiles, "
    "boundary flag, uncertainty status, a per-row probability caveat), the project "
    f"disclaimer and the dataset retrieval date ({retrieval_date}) in every row, "
    "and the retrieval date in the file name."
)
st.download_button(
    "Download filtered table (CSV)",
    data=export_frame.to_csv(index=False).encode("utf-8"),
    file_name=query.export_filename(retrieval_date),
    mime="text/csv",
    key="explorer_export",
)
