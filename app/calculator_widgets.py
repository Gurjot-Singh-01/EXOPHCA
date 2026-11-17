"""Drawing helpers for the Calculator's two panels (pure: no Streamlit).

The Size Comparison panel is drawn **to scale**: a circle's diameter is linear
in the planet's radius, so the planet's drawn diameter divided by Earth's drawn
diameter is exactly R / 1 R-Earth. When the planet would not fit the panel both
circles are shrunk by the same factor (the ratio is unchanged) and the shrink is
stated as "Earth shown at X% of the planet's scale to fit".

The System Preview is an illustration: the star's size and the star-planet
separation are decoration, so the panel is labelled "Illustration, not to
scale". Its planet circle still grows linearly with radius, capped to the panel.
"""

from __future__ import annotations

import numpy as np

# Earth's drawn diameter at full scale, and the widest circle the panel accepts.
EARTH_DIAMETER_PX = 90.0
MAX_PANEL_DIAMETER_PX = 200.0
STAR_DIAMETER_PX = 130.0
SYSTEM_MAX_DIAMETER_PX = 110.0

EARTH_COLOR = "#4a90d9"

STAR_COLORS = {
    "A (hot)": "#9bb0ff",
    "F": "#f8f7ff",
    "G (Sun-like)": "#fff2a1",
    "K": "#ffd2a1",
    "M (red dwarf)": "#ff8c69",
}


def get_planet_color(insol) -> str:
    """Cold-blue to hot-red, log-scaled and centred on Earth's flux."""
    log_insol = np.log10(max(float(insol), 1e-4))
    t = float(np.clip((log_insol + 3) / 6, 0, 1))
    if t < 0.5:
        r = int(80 + (255 - 80) * (t / 0.5))
        g = int(120 + (230 - 120) * (t / 0.5))
        b = 255
    else:
        r = 255
        g = int(230 - (230 - 60) * ((t - 0.5) / 0.5))
        b = int(255 - 255 * ((t - 0.5) / 0.5))
    return f"rgb({r},{g},{b})"


def get_star_color(star_class) -> str:
    return STAR_COLORS.get(star_class, "#fff2a1")


def planet_diameter_px(radius, *, max_px: float = MAX_PANEL_DIAMETER_PX) -> tuple[float, float]:
    """Drawn diameter for a planet, linear in radius, plus the shrink factor.

    Returns ``(diameter_px, factor)``; ``factor`` is 1.0 unless the circle had to
    be shrunk to fit ``max_px``.
    """
    value = float(radius)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("A drawn planet needs a positive, finite radius.")
    full = EARTH_DIAMETER_PX * value
    if full > max_px:
        factor = max_px / full
        return max_px, factor
    return full, 1.0


def size_comparison(pl_rade, pl_insol) -> dict:
    """Earth beside the planet, both circles at one linear scale."""
    radius = float(pl_rade)
    planet_px, factor = planet_diameter_px(radius, max_px=MAX_PANEL_DIAMETER_PX)
    earth_px = EARTH_DIAMETER_PX * factor
    shrunk = factor < 1.0
    note = (
        f"Earth shown at {100.0 * earth_px / planet_px:.1f}% of the planet's scale "
        "to fit."
        if shrunk
        else ""
    )

    html = f"""
    <style>
        .size-comparison {{ margin:0; display:flex; justify-content:center; align-items:flex-end;
                            gap: 50px; min-height:240px; background:transparent; }}
        .size-comparison .earth {{
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {EARTH_COLOR};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .size-comparison .planet {{
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {get_planet_color(pl_insol)};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .size-comparison .label {{ font-size: 12px; text-align:center; opacity: 0.7; margin-top: 8px; }}
        .size-comparison .col {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; }}
    </style>
    <div class="size-comparison">
        <div class="col">
            <div class="earth" style="width: {earth_px:.3f}px; height: {earth_px:.3f}px;"></div>
            <div class="label">Earth (1.000 R&oplus;)</div>
        </div>
        <div class="col">
            <div class="planet" style="width: {planet_px:.3f}px; height: {planet_px:.3f}px;"></div>
            <div class="label">This planet ({radius:.2f} R&oplus;)</div>
        </div>
    </div>
    """
    return {
        "html": html,
        "earth_px": earth_px,
        "planet_px": planet_px,
        "factor": factor,
        "ratio": planet_px / earth_px,
        "note": note,
    }


def system_illustration(pl_rade, pl_insol, star_class) -> dict:
    """Host star and planet side by side - an illustration, not to scale."""
    radius = float(pl_rade)
    planet_px, factor = planet_diameter_px(radius, max_px=SYSTEM_MAX_DIAMETER_PX)
    star_color = get_star_color(star_class)
    planet_color = get_planet_color(pl_insol)

    html = f"""
    <style>
        .system-scene {{ margin:0; display:flex; justify-content:center; align-items:center;
                         gap: 60px; min-height:220px; background:transparent; }}
        .system-scene .system-object {{ text-align:center; }}
        .system-scene .star {{
            width: {STAR_DIAMETER_PX}px; height: {STAR_DIAMETER_PX}px; border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.8), {star_color} 60%);
            box-shadow: 0 0 40px 10px {star_color}66;
        }}
        .system-scene .planet {{
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {planet_color};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .system-scene .label {{ font-size: 12px; text-align:center; opacity: 0.7; margin-top: 8px; }}
    </style>
    <div class="system-scene">
        <div class="system-object">
            <div class="star"></div>
            <div class="label">Host Star ({star_class})</div>
        </div>
        <div class="system-object">
            <div class="planet" style="width: {planet_px:.3f}px; height: {planet_px:.3f}px;"></div>
            <div class="label">Planet ({radius:.2f} R&oplus;)</div>
        </div>
    </div>
    """
    return {"html": html, "planet_px": planet_px, "factor": factor, "star_px": STAR_DIAMETER_PX}
