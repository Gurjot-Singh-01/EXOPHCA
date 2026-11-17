"""TESS Candidates (unconfirmed): filter the TOI assessments and export them.

Candidates are scored with the same rules as the confirmed planets, but no TESS
object has a published flux uncertainty, so every flux-dependent probability is
conditional on a fixed input. The labelling rule prints that per value and the
uncertainty_status column records it per row.
"""

from __future__ import annotations

import math
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
    "TESS Candidates (Unconfirmed)",
    "Objects of interest from the TESS mission, assessed with the same rules as "
    "the confirmed planets but not yet validated. Filter them below; export the "
    "filtered rows with their provenance.",
    eyebrow="Tools",
)

ui.disclaimer()

frame = data_access.load_or_stop(data_access.load_toi_assessed, "TESS candidate assessments")
provenance = data_access.load_or_stop(data_access.load_provenance, "Provenance records")
retrieval_date = provenance["toi"]["retrieval_date"]

st.info(
    "⚠️ Unconfirmed candidates. Every row carries the status "
    "`unconfirmed candidate`. **No TESS candidate has a published flux "
    "uncertainty**, so the flux-dependent probabilities are conditional on a "
    "fixed input; the uncertainty_status column and the probability text record "
    "this per row."
)

scorable = frame[frame["pl_rade"].notna() & (frame["pl_rade"] > 0)]
if len(scorable) < len(frame):
    st.caption(
        f"{len(frame) - len(scorable):,} of {len(frame):,} candidates have no usable "
        "radius and are not shown."
    )

EVIDENCE_ORDER = ["high", "medium", "low"]
TIER_ORDER = ["A", "B", "C", "D", "U"]
HZ_ORDER = ["conservative", "optimistic_only", "too_hot", "too_cold", "undetermined"]


def _ordered(values, order):
    present = [value for value in order if value in values]
    return present + sorted(value for value in values if value not in order)


with st.expander("Filters", expanded=True):
    first, second, third = st.columns(3)
    tiers = first.multiselect(
        "Tier", _ordered(set(scorable["tier"].dropna()), TIER_ORDER), key="toi_tiers"
    )
    hz_classes = second.multiselect(
        "Habitable-zone class",
        _ordered(set(scorable["hz_class"].dropna()), HZ_ORDER),
        key="toi_hz",
    )
    evidence = third.multiselect(
        "Evidence quality",
        _ordered(set(scorable["evidence_quality"].dropna()), EVIDENCE_ORDER),
        key="toi_evidence",
    )

    radius_low = math.floor(float(scorable["pl_rade"].min()) * 100) / 100
    radius_high = math.ceil(float(scorable["pl_rade"].max()) * 100) / 100
    teff_values = scorable["st_teff"].dropna()
    teff_low = float(math.floor(teff_values.min() / 100.0) * 100)
    teff_high = float(math.ceil(teff_values.max() / 100.0) * 100)

    fourth, fifth, sixth = st.columns([1, 1, 1])
    radius_range = fourth.slider(
        "Radius range (R⊕)",
        min_value=float(radius_low),
        max_value=float(radius_high),
        value=(float(radius_low), float(radius_high)),
        step=0.01,
        key="toi_radius",
    )
    teff_range = fifth.slider(
        "Host Teff range (K)",
        min_value=teff_low,
        max_value=teff_high,
        value=(teff_low, teff_high),
        step=50.0,
        key="toi_teff",
    )
    boundary_choice = sixth.selectbox(
        "Boundary sensitive",
        ["Any", "Flagged only", "Not flagged"],
        key="toi_boundary",
        help=(
            "Flagged when the point classification sits inside a habitable-zone "
            "band but fewer than 70% of the Monte Carlo draws agree."
        ),
    )

boundary_flag = {"Any": None, "Flagged only": True, "Not flagged": False}[boundary_choice]

filtered = query.filter_toi(
    scorable,
    tiers=tiers,
    hz_classes=hz_classes,
    evidence_quality=evidence,
    radius_range=None
    if radius_range == (float(radius_low), float(radius_high))
    else radius_range,
    teff_range=None if teff_range == (teff_low, teff_high) else teff_range,
    boundary_sensitive=boundary_flag,
)

missing_teff = int(scorable["st_teff"].isna().sum())
st.markdown(
    f"**{len(filtered):,} of {len(scorable):,} candidates** match the current filters."
)
st.caption(
    "The temperature filter applies once it is narrowed; at its widest setting the "
    f"{missing_teff:,} candidates without a published effective temperature stay in "
    "the table with an `undetermined` habitable-zone class."
)

# ---------------------------------------------------------------- table
shown = filtered.sort_values("base_score", ascending=False).copy()
table = shown[
    [
        "toi",
        "status",
        "st_teff",
        "pl_rade",
        "pl_insol",
        "base_score",
        "hz_class",
        "tier",
        "evidence_quality",
        "uncertainty_status",
    ]
].copy()
labelled = labelling.labelled_probability_columns(shown)
for name in labelling.PROBABILITY_DEPENDENCIES:
    table[name] = labelled[name].to_numpy()

column_config = {
    "toi": st.column_config.TextColumn("TOI", width="small"),
    "status": st.column_config.TextColumn("Status", width="medium"),
    "st_teff": st.column_config.NumberColumn("Teff (K)", format="%.0f"),
    "pl_rade": st.column_config.NumberColumn("Radius (R⊕)", format="%.3f"),
    "pl_insol": st.column_config.NumberColumn("Insolation (S⊕)", format="%.4f"),
    "base_score": st.column_config.NumberColumn("Score", format="%.3f"),
}
for name, label in labelling.PROBABILITY_LABELS.items():
    column_config[name] = st.column_config.TextColumn(label, width="large")

st.dataframe(table, hide_index=True, height=520, column_config=column_config)

st.caption(
    "Every probability is printed with its uncertainty_status. "
    f"Where an input was held fixed the cell says “{labelling.CONDITIONAL_TEXT}” "
    "instead of showing a bare number. For TESS candidates the flux is never "
    "sampled: no flux uncertainty is published, so every flux-dependent "
    "probability is conditional."
)

st.divider()
ui.disclaimer()

export = query.build_toi_export_frame(filtered, retrieval_date)
st.caption(
    f"The CSV holds the {len(export):,} filtered rows with the unconfirmed-candidate "
    "status, the radius provenance and uncertainty columns, a per-row probability "
    f"caveat, the project disclaimer and the retrieval date ({retrieval_date}) in "
    "every row, and that date in the file name."
)
st.download_button(
    "Download filtered candidates (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=query.export_filename(retrieval_date, prefix="exophca_tess_candidates"),
    mime="text/csv",
    key="toi_export",
)
