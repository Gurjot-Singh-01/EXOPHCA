"""Shared presentation helpers: accent colour, theme-safe CSS, page chrome.

Plotly figures deliberately carry no background, font or template of their own
and are rendered with Streamlit's own Plotly theme, so axes and labels follow
the light or dark appearance instead of fighting it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from exophca.science.assessment import DISCLAIMER  # noqa: E402

# The one accent used for interactive and emphasis elements (matches
# .streamlit/config.toml). Mid-tone so it reads on both light and dark.
ACCENT = "#2F6B7A"
ACCENT_SOFT = "#7FA7B1"
NEUTRAL = "#9AA0A6"

# One hue per habitable-zone class, chosen to stay apart from each other on
# both a white and a near-black background (mid lightness, distinct hue, and a
# grey outline on every marker so a colour never has to carry the shape alone).
HZ_COLORS = {
    "conservative": ACCENT,
    "optimistic_only": "#C08A2E",
    "too_hot": "#B5453C",
    "too_cold": "#4A6FA5",
    "undetermined": "#8E8E93",
}

# Radius provenance is encoded as marker fill, not colour, so colour keeps one
# meaning: ``symbol`` per provenance, ``width`` of the outline.
HZ_COLOR_NAMES = {
    "conservative": "teal",
    "optimistic_only": "amber",
    "too_hot": "brick red",
    "too_cold": "blue",
    "undetermined": "grey",
}

PROVENANCE_MARKERS = {
    "measured": ("circle", 0.6),
    "modelled": ("circle-open", 1.3),
    "derived_unclear": ("diamond-open", 1.3),
}
UNKNOWN_PROVENANCE_MARKER = ("x", 1.3)
PROVENANCE_LEGEND = (
    "Radius provenance is the marker fill: a filled circle is a measured "
    "radius, an open circle is a modelled radius, an open diamond is a radius "
    "of unclear derivation and a cross has an unknown provenance."
)

# Neutral outline so every marker keeps its edge against either background.
MARKER_OUTLINE = "#6E6E73"

TIER_COLORS = {
    "A": ACCENT,
    "B": ACCENT_SOFT,
    "C": "#7A8CA3",
    "D": NEUTRAL,
    "U": "#C4705A",
}

HZ_LABELS = {
    "conservative": "Conservative habitable zone",
    "optimistic_only": "Optimistic habitable zone only",
    "too_hot": "Too hot",
    "too_cold": "Too cold",
    "undetermined": "Undetermined",
}


def inject_css() -> None:
    """Typography and spacing only - no colours, so light and dark both work."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.75rem; padding-bottom: 3.5rem; max-width: 1250px;}
        h1, h2, h3 {letter-spacing: -0.012em;}
        h1 {font-weight: 650;}
        [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        [data-testid="stDataFrame"] {font-variant-numeric: tabular-nums;}
        .ex-eyebrow {text-transform: uppercase; letter-spacing: 0.14em;
                     font-size: 0.72rem; opacity: 0.65; margin-bottom: 0.2rem;}
        .ex-lede {max-width: 62rem; opacity: 0.82;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, *, color: str = ACCENT) -> None:
    """An inline chip: theme text colour, neutral border, colored dot only.

    No colour is used for the text itself, so the chip stays readable in both
    light and dark themes.
    """
    st.markdown(
        '<span style="display:inline-block; padding:0.16rem 0.62rem; '
        "margin:0 0.4rem 0.4rem 0; border:1px solid rgba(127,127,127,0.45); "
        'border-radius:999px; font-size:0.82rem; white-space:nowrap;">'
        f'<span style="color:{color};">\u25cf</span> {text}</span>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.markdown(f'<div class="ex-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="ex-lede">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def disclaimer() -> None:
    """The project's single disclaimer constant, shown verbatim."""
    st.info(DISCLAIMER)


def bar_chart(
    labels,
    values,
    *,
    colors=None,
    x_title: str = "Planets",
    height: int = 260,
    horizontal: bool = True,
) -> go.Figure:
    """A restrained single-series bar chart that follows the app theme."""
    labels = list(labels)
    values = [int(value) for value in values]
    palette = list(colors) if colors is not None else [ACCENT] * len(labels)

    if horizontal:
        figure = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=palette,
                text=values,
                textposition="outside",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            )
        )
        figure.update_layout(xaxis_title=x_title, yaxis_title=None)
        figure.update_yaxes(autorange="reversed")
    else:
        figure = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=palette,
                text=values,
                textposition="outside",
                hovertemplate="%{x}: %{y:,}<extra></extra>",
            )
        )
        figure.update_layout(xaxis_title=None, yaxis_title=x_title)

    figure.update_layout(
        height=height,
        margin=dict(l=4, r=20, t=8, b=4),
        showlegend=False,
        bargap=0.38,
    )
    return figure


def power_of_ten_ticks(low: float, high: float, *, max_ticks: int = 9):
    """Plain powers of ten for a log axis: ``10^-4, 10^-2, 1, 10^2 ...``.

    Returns ``(tickvals, ticktext)``. Plotly's default SI prefixes (``100\u00b5``,
    ``10k``, ``1M``) hide the magnitude of a log axis, so every log axis in the
    app uses these explicit powers instead.
    """
    lo = float(min(low, high))
    hi = float(max(low, high))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi <= 0:
        return [], []

    e_min = math.floor(math.log10(lo))
    e_max = math.ceil(math.log10(hi))
    if e_max <= e_min:
        return [], []

    step = 1
    while (e_max - e_min) // step + 1 > max_ticks:
        step += 1
    exponents = [e for e in range(e_min, e_max + 1) if e % step == 0]
    if not exponents:
        return [], []

    tickvals = [10.0**exponent for exponent in exponents]
    ticktext = ["1" if exponent == 0 else f"10^{exponent}" for exponent in exponents]
    return tickvals, ticktext


def hz_scatter(points, *, height: int = 520) -> go.Figure:
    """Radius against insolation, coloured by habitable-zone class.

    One ``Scattergl`` series per (habitable-zone class, radius provenance) so a
    modelled radius is an open marker rather than a differently coloured one.
    No sampling and no cap: every row passed in is drawn.
    """
    figure = go.Figure()
    if points is None or len(points) == 0:
        return figure

    order = [name for name in HZ_LABELS if name in set(points["hz_class"])]
    order += sorted(set(points["hz_class"]) - set(order))

    for hz_class in order:
        label = HZ_LABELS.get(hz_class, hz_class)
        color = HZ_COLORS.get(hz_class, NEUTRAL)
        first = True
        for provenance, (symbol, width) in PROVENANCE_MARKERS.items():
            subset = points[
                (points["hz_class"] == hz_class)
                & (points["radius_provenance"] == provenance)
            ]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scattergl(
                    x=subset["pl_insol"],
                    y=subset["pl_rade"],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=first,
                    marker=dict(
                        size=7,
                        symbol=symbol,
                        color=color,
                        opacity=0.85,
                        line=dict(width=width, color=MARKER_OUTLINE),
                    ),
                    customdata=subset[
                        ["pl_name", "base_score", "radius_provenance", "uncertainty_status"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>flux %{x:.3g} S\u2295 \u00b7 radius "
                        "%{y:.2f} R\u2295<br>score %{customdata[1]:.3f} \u00b7 radius "
                        "%{customdata[2]} \u00b7 uncertainty_status %{customdata[3]}"
                        "<extra></extra>"
                    ),
                )
            )
            first = False

        unknown = points[
            (points["hz_class"] == hz_class)
            & (~points["radius_provenance"].isin(list(PROVENANCE_MARKERS)))
        ]
        if not unknown.empty:
            symbol, width = UNKNOWN_PROVENANCE_MARKER
            figure.add_trace(
                go.Scattergl(
                    x=unknown["pl_insol"],
                    y=unknown["pl_rade"],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=first,
                    marker=dict(
                        size=7,
                        symbol=symbol,
                        color=color,
                        opacity=0.85,
                        line=dict(width=width, color=MARKER_OUTLINE),
                    ),
                    customdata=unknown[
                        ["pl_name", "base_score", "radius_provenance", "uncertainty_status"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>flux %{x:.3g} S\u2295 \u00b7 radius "
                        "%{y:.2f} R\u2295<br>score %{customdata[1]:.3f} \u00b7 radius "
                        "%{customdata[2]} \u00b7 uncertainty_status %{customdata[3]}"
                        "<extra></extra>"
                    ),
                )
            )

    x_ticks, x_labels = power_of_ten_ticks(
        points["pl_insol"].min(), points["pl_insol"].max()
    )
    y_ticks, y_labels = power_of_ten_ticks(
        points["pl_rade"].min(), points["pl_rade"].max()
    )
    figure.update_xaxes(
        type="log",
        title="Insolation flux (S\u2295, log scale)",
        tickvals=x_ticks,
        ticktext=x_labels,
    )
    figure.update_yaxes(
        type="log",
        title="Radius (R\u2295, log scale)",
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


def metric_row(items) -> None:
    """A row of metrics from (label, value, help) triples."""
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        column.metric(label, value, help=help_text)
