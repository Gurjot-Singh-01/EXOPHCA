"""Data preparation and Plotly figures for the Compare and Diagrams pages.

Pure Python (no Streamlit), so the band edges, the counts and the figures can be
tested directly. Every figure draws **all** the rows it is given: there is no
sampling anywhere in this module.

The habitable-zone bands come from ``exophca.science.habitable_zone.hz_limits``
evaluated over its fitted 2600-7200 K range. Nothing is extrapolated: rows whose
stellar temperature falls outside that range, or whose flux is missing or not
positive, are counted and left out rather than drawn against the bands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ui  # noqa: E402
from exophca.science.habitable_zone import hz_limits  # noqa: E402
from reference_earth import EARTH_LABEL as EARTH_REFERENCE_LABEL  # noqa: E402

# The fitted validity range of the Kopparapu limits (exophca/science/habitable_zone).
HZ_TEFF_MIN = 2600.0
HZ_TEFF_MAX = 7200.0
BAND_SAMPLES = 481

# Band fills: low alpha so the points stay readable on white and on near-black.
BAND_FILLS = {
    "conservative": "rgba(47,107,122,0.20)",
    "optimistic": "rgba(192,138,46,0.15)",
}
BAND_EDGES = {
    "conservative": ui.ACCENT,
    "optimistic": "#C08A2E",
}

# In the mass-radius diagram colour *is* the radius provenance.
PROVENANCE_COLORS = {
    "measured": ui.ACCENT,
    "modelled": "#C08A2E",
    "derived_unclear": "#8E8E93",
}
PROVENANCE_SYMBOLS = {
    "measured": "circle",
    "modelled": "circle-open",
    "derived_unclear": "diamond-open",
}
PROVENANCE_ORDER = ("measured", "modelled", "derived_unclear")
PROVENANCE_LABELS = {
    "measured": "Radius measured",
    "modelled": "Radius modelled",
    "derived_unclear": "Radius of unclear derivation",
}


def hz_band_frame(teff=None, n: int = BAND_SAMPLES) -> pd.DataFrame:
    """The four habitable-zone flux limits over the fitted temperature range.

    With no ``teff`` the frame is a regular grid of ``n`` temperatures between
    2600 K and 7200 K; passing temperatures evaluates exactly those.
    """
    if teff is None:
        temperatures = np.linspace(HZ_TEFF_MIN, HZ_TEFF_MAX, n)
    else:
        temperatures = np.asarray(teff, dtype=float)
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("Habitable-zone bands need finite temperatures.")
    if np.any(temperatures < HZ_TEFF_MIN) or np.any(temperatures > HZ_TEFF_MAX):
        raise ValueError(
            f"Habitable-zone bands are only defined for {HZ_TEFF_MIN:.0f}-"
            f"{HZ_TEFF_MAX:.0f} K; the limits are not extrapolated."
        )

    limits = hz_limits(temperatures)
    if not np.all(np.asarray(limits["in_valid_range"])):
        raise ValueError("Habitable-zone bands requested outside the fitted range.")

    return pd.DataFrame(
        {
            "st_teff": temperatures,
            "optimistic_inner": np.asarray(limits["optimistic_inner"], dtype=float),
            "conservative_inner": np.asarray(limits["conservative_inner"], dtype=float),
            "conservative_outer": np.asarray(limits["conservative_outer"], dtype=float),
            "optimistic_outer": np.asarray(limits["optimistic_outer"], dtype=float),
        }
    )


def hz_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows that can be drawn against the bands: finite flux in the fitted range."""
    teff = frame["st_teff"]
    flux = frame["pl_insol"]
    return (
        teff.notna()
        & flux.notna()
        & (flux > 0)
        & (teff >= HZ_TEFF_MIN)
        & (teff <= HZ_TEFF_MAX)
    )


def hz_plot_counts(frame: pd.DataFrame) -> dict[str, int]:
    """How many rows are drawn and why the rest are not, by category.

    The three reasons can overlap (a row may have no temperature *and* a
    non-positive flux), so ``teff_missing + teff_out_of_range + flux_invalid``
    can exceed ``left_out``; ``multiple`` counts the rows in more than one group
    and ``left_out`` is the number of distinct rows that are not drawn.
    """
    teff = frame["st_teff"]
    flux = frame["pl_insol"]
    teff_missing = teff.isna()
    teff_out_of_range = teff.notna() & ((teff < HZ_TEFF_MIN) | (teff > HZ_TEFF_MAX))
    flux_invalid = flux.isna() | (flux <= 0)

    groups = (
        teff_missing.astype(int) + teff_out_of_range.astype(int) + flux_invalid.astype(int)
    )
    left_out = groups > 0
    return {
        "available": int(len(frame)),
        "drawn": int((~left_out).sum()),
        "teff_missing": int(teff_missing.sum()),
        "teff_out_of_range": int(teff_out_of_range.sum()),
        "flux_invalid": int(flux_invalid.sum()),
        "multiple": int((groups > 1).sum()),
        "left_out": int(left_out.sum()),
    }


def hz_points(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """The plottable rows and the counts for the caption."""
    counts = hz_plot_counts(frame)
    return frame[hz_mask(frame)], counts


def mass_radius_counts(frame: pd.DataFrame) -> dict[str, int]:
    """How many rows have both a positive mass and a positive radius."""
    mass = frame["pl_bmasse"]
    radius = frame["pl_rade"]
    missing = mass.isna() | radius.isna()
    non_positive = (~missing) & ((mass <= 0) | (radius <= 0))
    drawn = ~missing & ~non_positive
    return {
        "available": int(len(frame)),
        "drawn": int(drawn.sum()),
        "missing": int(missing.sum()),
        "non_positive": int(non_positive.sum()),
    }


def mass_radius_points(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows with a positive mass and radius, plus the counts for the caption."""
    counts = mass_radius_counts(frame)
    mass = frame["pl_bmasse"]
    radius = frame["pl_rade"]
    usable = (
        mass.notna() & radius.notna() & (mass > 0) & (radius > 0)
    )
    return frame[usable], counts


def _hz_point_traces(figure: go.Figure, points: pd.DataFrame) -> None:
    """One Scattergl series per (habitable-zone class, radius provenance)."""
    order = [name for name in ui.HZ_LABELS if name in set(points["hz_class"])]
    order += sorted(set(points["hz_class"]) - set(order))

    for hz_class in order:
        label = ui.HZ_LABELS.get(hz_class, hz_class)
        color = ui.HZ_COLORS.get(hz_class, ui.NEUTRAL)
        first = True
        for provenance, (symbol, width) in ui.PROVENANCE_MARKERS.items():
            subset = points[
                (points["hz_class"] == hz_class)
                & (points["radius_provenance"] == provenance)
            ]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scattergl(
                    x=subset["st_teff"],
                    y=subset["pl_insol"],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=first,
                    marker=dict(
                        size=7,
                        symbol=symbol,
                        color=color,
                        opacity=0.85,
                        line=dict(width=width, color=ui.MARKER_OUTLINE),
                    ),
                    customdata=subset[
                        ["pl_name", "base_score", "radius_provenance", "uncertainty_status"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>Teff %{x:.0f} K · flux %{y:.3g} S⊕"
                        "<br>score %{customdata[1]:.3f} · radius %{customdata[2]}"
                        " · uncertainty_status %{customdata[3]}<extra></extra>"
                    ),
                )
            )
            first = False


def hz_band_figure(
    points: pd.DataFrame, bands: pd.DataFrame | None = None, *, height: int = 580
) -> go.Figure:
    """Insolation flux against host temperature, with both habitable-zone bands."""
    bands = hz_band_frame() if bands is None else bands
    figure = go.Figure()

    # Optimistic band: early Mars (outer) to recent Venus (inner).
    figure.add_trace(
        go.Scatter(
            x=bands["st_teff"],
            y=bands["optimistic_outer"],
            mode="lines",
            name="Early Mars limit (optimistic outer)",
            line=dict(color=BAND_EDGES["optimistic"], width=1.4, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bands["st_teff"],
            y=bands["optimistic_inner"],
            mode="lines",
            name="Recent Venus limit (optimistic inner)",
            line=dict(color=BAND_EDGES["optimistic"], width=1.4, dash="dot"),
            fill="tonexty",
            fillcolor=BAND_FILLS["optimistic"],
        )
    )
    # Conservative band: maximum greenhouse (outer) to runaway greenhouse (inner).
    figure.add_trace(
        go.Scatter(
            x=bands["st_teff"],
            y=bands["conservative_outer"],
            mode="lines",
            name="Maximum greenhouse limit (conservative outer)",
            line=dict(color=BAND_EDGES["conservative"], width=1.8),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bands["st_teff"],
            y=bands["conservative_inner"],
            mode="lines",
            name="Runaway greenhouse limit (conservative inner)",
            line=dict(color=BAND_EDGES["conservative"], width=1.8),
            fill="tonexty",
            fillcolor=BAND_FILLS["conservative"],
        )
    )

    _hz_point_traces(figure, points)

    y_ticks, y_labels = ui.power_of_ten_ticks(
        float(points["pl_insol"].min()), float(points["pl_insol"].max())
    )
    figure.update_xaxes(
        title="Host effective temperature (K)",
        range=[HZ_TEFF_MIN - 100, HZ_TEFF_MAX + 100],
    )
    figure.update_yaxes(
        type="log",
        title="Insolation flux (S⊕, log scale)",
        tickvals=y_ticks,
        ticktext=y_labels,
    )
    figure.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        hovermode="closest",
    )
    return figure


def mass_radius_figure(points: pd.DataFrame, *, height: int = 560) -> go.Figure:
    """Radius against mass on log axes, coloured by radius provenance."""
    figure = go.Figure()

    present = [name for name in PROVENANCE_ORDER if name in set(points["radius_provenance"])]
    present += sorted(set(points["radius_provenance"]) - set(present))

    for provenance in present:
        subset = points[points["radius_provenance"] == provenance]
        if subset.empty:
            continue
        color = PROVENANCE_COLORS.get(provenance, ui.NEUTRAL)
        symbol = PROVENANCE_SYMBOLS.get(provenance, "x")
        width = 1.3 if symbol != "circle" else 0.6
        figure.add_trace(
            go.Scattergl(
                x=subset["pl_bmasse"],
                y=subset["pl_rade"],
                mode="markers",
                name=PROVENANCE_LABELS.get(provenance, provenance),
                marker=dict(
                    size=7,
                    symbol=symbol,
                    color=color,
                    opacity=0.85,
                    line=dict(width=width, color=color if symbol != "circle" else ui.MARKER_OUTLINE),
                ),
                customdata=subset[["pl_name", "mass_provenance", "hostname"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>mass %{x:.3g} M⊕ · radius %{y:.2f} R⊕"
                    "<br>radius provenance: " + provenance
                    + " · mass provenance: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    x_ticks, x_labels = ui.power_of_ten_ticks(
        float(points["pl_bmasse"].min()), float(points["pl_bmasse"].max())
    )
    y_ticks, y_labels = ui.power_of_ten_ticks(
        float(points["pl_rade"].min()), float(points["pl_rade"].max())
    )
    figure.update_xaxes(
        type="log",
        title="Planet mass (M⊕, log scale)",
        tickvals=x_ticks,
        ticktext=x_labels,
    )
    figure.update_yaxes(
        type="log",
        title="Planet radius (R⊕, log scale)",
        tickvals=y_ticks,
        ticktext=y_labels,
    )
    figure.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        hovermode="closest",
    )
    return figure


def compare_rows(frame: pd.DataFrame, chosen, *, earth: dict | None = None) -> pd.DataFrame:
    """The Compare table: Earth's computed reference row, then each planet.

    Earth's score, class and tier come from :mod:`reference_earth`, i.e. from the
    same science functions an archive planet goes through; nothing about Earth is
    typed in here.
    """
    import labelling  # local: keeps the module import list short
    import reference_earth

    earth = reference_earth.earth_reference() if earth is None else earth
    earth_label = earth["pl_name"]
    earth_probability_text = "reference row - no Monte Carlo (uncertainty_status: none)"

    rows = [
        {
            "label": earth_label,
            "Planet": earth_label,
            "Score": float(earth["base_score"]),
            "p16": float(earth["base_score"]),
            "p84": float(earth["base_score"]),
            "Radius": labelling.radius_text(earth["pl_rade"], "measured"),
            "Radius provenance": "Reference (by definition)",
            "HZ class": ui.HZ_LABELS[earth["hz_class"]],
            "Tier": earth["tier"],
            "Evidence": "reference",
            "uncertainty_status": earth["uncertainty_status"],
            **{
                name: earth_probability_text
                for name in labelling.PROBABILITY_DEPENDENCIES
            },
        }
    ]

    for planet in chosen:
        row = frame[frame["pl_name"] == planet].iloc[0]
        status = labelling.normalise_status(row["uncertainty_status"])
        entry = {
            "label": planet,
            "Planet": planet,
            "Score": float(row["base_score"]),
            "p16": float(row["base_score_p16"]),
            "p84": float(row["base_score_p84"]),
            "Radius": labelling.radius_text(row["pl_rade"], row["radius_provenance"]),
            "Radius provenance": labelling.radius_short(row["radius_provenance"]),
            "HZ class": ui.HZ_LABELS.get(row["hz_class"], row["hz_class"]),
            "Tier": row["tier"],
            "Evidence": row["evidence_quality"],
            "uncertainty_status": status,
        }
        for name in labelling.PROBABILITY_DEPENDENCIES:
            entry[name] = labelling.format_probability(name, row[name], status)
        rows.append(entry)

    return pd.DataFrame(rows)


def score_range_figure(
    rows: pd.DataFrame,
    *,
    height: int | None = None,
    earth_label: str | None = EARTH_REFERENCE_LABEL,
) -> go.Figure:
    """Score with its 16th-84th percentile range as horizontal bars.

    ``rows`` needs ``label``, ``Score``, ``p16`` and ``p84``. A bar spans the
    percentile range and a vertical tick marks the point score. A dashed
    vertical line at 1.000 names the reference, and the Earth row carries a
    diamond so the reference is visible even though its own range is zero-width
    (the tick of a zero-width bar would otherwise be the only mark).
    """
    labels = list(rows["label"])
    widths = (rows["p84"] - rows["p16"]).clip(lower=0.0)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=widths,
            y=labels,
            base=rows["p16"],
            orientation="h",
            name="16th-84th percentile",
            marker_color=ui.ACCENT,
            opacity=0.45,
            hovertemplate="%{y}<br>%{base:.3f} to %{x:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["Score"],
            y=labels,
            mode="markers",
            name="Point score",
            marker=dict(symbol="line-ns", size=18, color="#6E6E73", line=dict(width=1)),
            hovertemplate="%{y}<br>score %{x:.3f}<extra></extra>",
        )
    )
    if earth_label is not None and earth_label in labels:
        earth_score = float(rows.loc[rows["label"] == earth_label, "Score"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=[earth_score],
                y=[earth_label],
                mode="markers",
                name=f"Earth reference ({earth_score:.3f})",
                marker=dict(
                    symbol="diamond",
                    size=13,
                    color=ui.ACCENT,
                    line=dict(width=1.6, color=ui.MARKER_OUTLINE),
                ),
                hovertemplate=f"{earth_label}<br>score {earth_score:.3f}<extra></extra>",
            )
        )

    figure.add_shape(
        type="line",
        x0=1.0,
        x1=1.0,
        y0=0.0,
        y1=1.0,
        xref="x",
        yref="paper",
        line=dict(color=ui.MARKER_OUTLINE, width=1.2, dash="dash"),
    )
    figure.add_annotation(
        x=1.0,
        y=1.0,
        xref="x",
        yref="paper",
        text="Earth = 1.000",
        showarrow=False,
        xanchor="right",
        xshift=-6,
        yanchor="bottom",
        font=dict(size=11, color=ui.MARKER_OUTLINE),
    )

    # A little room past 1 so neither the reference line nor the diamond is
    # clipped by the axis edge.
    figure.update_xaxes(title="ESI-style score", range=[0.0, 1.06])
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(
        height=height if height is not None else 60 + 46 * max(len(labels), 1),
        margin=dict(l=4, r=4, t=26, b=4),
        bargap=0.45,
        showlegend=False,
    )
    return figure
