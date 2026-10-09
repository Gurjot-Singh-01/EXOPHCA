import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scoring import (
    FEATURE_UNITS,
    MODEL_FEATURES,
    TRAINING_RANGES,
    calculate_custom_adjustment,
    calculate_esi,
    predict_compatibility,
    training_range_warnings,
    validate_nonnegative_finite,
    validate_positive_finite,
)

# Resolve paths relative to this file's location
SCRIPT_DIR = Path(__file__).resolve().parent.parent  # Go up one level from app/ to repo root
MODEL_DIR = SCRIPT_DIR / "models"
DATA_DIR = SCRIPT_DIR / "data"

st.set_page_config(page_title="EXOPHCA", page_icon="🪐", layout="centered")


def resolve_project_file(filename, search_dirs=None):
    """Find a project file in the standard locations, including the repo root."""
    candidates = []
    default_dirs = [MODEL_DIR, DATA_DIR, SCRIPT_DIR]
    if search_dirs:
        default_dirs = list(search_dirs) + default_dirs

    for directory in default_dirs:
        candidate = directory / filename
        candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched: {', '.join(str(p) for p in candidates)}"
    )


def get_planet_color(insol):
    # Map insolation to a cold->hot color gradient (log scale, centered near Earth=1.0)
    log_insol = np.log10(max(insol, 0.0001))
    # Clamp to a reasonable visual range: -3 (very cold) to 3 (very hot)
    t = np.clip((log_insol + 3) / 6, 0, 1)
    # Blue (cold) -> white/yellow (Earth-ish) -> red (hot)
    if t < 0.5:
        r = int(80 + (255 - 80) * (t / 0.5))
        g = int(120 + (230 - 120) * (t / 0.5))
        b = 255
    else:
        r = 255
        g = int(230 - (230 - 60) * ((t - 0.5) / 0.5))
        b = int(255 - 255 * ((t - 0.5) / 0.5))
    return f"rgb({r},{g},{b})"

def get_star_color(star_class):
    star_colors = {
        "A (hot)": "#9bb0ff",       # blue-white
        "F": "#f8f7ff",             # white
        "G (Sun-like)": "#fff2a1",  # yellow-white
        "K": "#ffd2a1",             # orange
        "M (red dwarf)": "#ff8c69", # red-orange
    }
    return star_colors.get(star_class, "#fff2a1")

def generate_system_html(pl_rade, pl_insol, star_class):
    star_color = get_star_color(star_class)
    planet_color = get_planet_color(pl_insol)

    star_px = 130
    planet_px = int(np.clip(30 + pl_rade * 12, 30, 100))

    html = f"""
    <style>
        .system-scene {{ margin:0; display:flex; justify-content:center; align-items:center;
                         gap: 60px; min-height:220px; background:transparent; }}
        .system-object {{ text-align:center; }}
        .star {{
            width: {star_px}px; height: {star_px}px; border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.8), {star_color} 60%);
            box-shadow: 0 0 40px 10px {star_color}66;
        }}
        .planet {{
            width: {planet_px}px; height: {planet_px}px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {planet_color};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .label {{ font-size: 12px; text-align:center; color: #ccc; margin-top: 8px; }}
    </style>
    <div class="system-scene">
        <div class="system-object">
            <div class="star"></div>
            <div class="label">Host Star ({star_class})</div>
        </div>
        <div class="system-object">
            <div class="planet"></div>
            <div class="label">Planet ({pl_rade:.2f} R⊕)</div>
        </div>
    </div>
    """
    return html

def generate_size_comparison_html(pl_rade, pl_insol):
    planet_color = get_planet_color(pl_insol)
    earth_color = "#4a90d9"  # Earth's own insolation = 1.0

    earth_px = 90
    # Scale relative to Earth, but cap so huge planets don't blow past the container
    scale_factor = np.clip(pl_rade, 0.1, 6.0)  # cap the effective ratio at 6x Earth's size
    planet_px = int(earth_px * (scale_factor ** 0.5))  # sqrt scaling keeps huge planets visually reasonable
    planet_px = min(planet_px, 200)  # hard cap so it never overflows the box

    html = f"""
    <style>
        .size-comparison {{ margin:0; display:flex; justify-content:center; align-items:flex-end;
                            gap: 50px; min-height:240px; background:transparent; }}
        .earth {{
            width: {earth_px}px; height: {earth_px}px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {earth_color};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .planet {{
            width: {planet_px}px; height: {planet_px}px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), transparent 40%), {planet_color};
            box-shadow: inset -10px -10px 25px rgba(0,0,0,0.5);
        }}
        .label {{ font-size: 12px; text-align:center; color: #ccc; margin-top: 8px; }}
        .col {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; }}
    </style>
    <div class="size-comparison">
        <div class="col">
            <div class="earth"></div>
            <div class="label">Earth (1.0 R⊕)</div>
        </div>
        <div class="col">
            <div class="planet"></div>
            <div class="label">This planet ({pl_rade:.2f} R⊕)</div>
        </div>
    </div>
    """
    return html


@st.cache_resource
def load_compatibility_model():
    return joblib.load(resolve_project_file("rf_compat_model.pkl", [MODEL_DIR]))


@st.cache_data
def load_csv(filename):
    return pd.read_csv(resolve_project_file(filename, [DATA_DIR]))


rf_compat = load_compatibility_model()
planets_df = load_csv("planets_for_app.csv")
toi_df = load_csv("toi_for_app.csv")
toi_df["predicted_score"] = predict_compatibility(
    rf_compat,
    toi_df.loc[:, list(MODEL_FEATURES)],
)
outside_training_range = pd.Series(False, index=toi_df.index)
for feature in MODEL_FEATURES:
    lower, upper = TRAINING_RANGES[feature]
    outside_training_range |= (
        (toi_df[feature] < lower) | (toi_df[feature] > upper)
    )
candidate_extrapolation_count = int(outside_training_range.sum())

st.title("🪐 EXOPHCA")
st.subheader("Exoplanet Habitability Compatibility Analysis")
st.write(
    "Explore two experimental measures of Earth similarity. These scores are not "
    "probabilities of life or confirmation that a planet is habitable."
)

# ---------- Preset planet data ----------
PRESETS = {
    "-- Select a preset --": None,
    "Mercury": {"pl_rade": 0.383, "pl_insol": 6.68, "star_class": "G (Sun-like)"},
    "Venus": {"pl_rade": 0.949, "pl_insol": 1.91, "star_class": "G (Sun-like)"},
    "Earth": {"pl_rade": 1.0, "pl_insol": 1.0, "star_class": "G (Sun-like)"},
    "Mars": {"pl_rade": 0.532, "pl_insol": 0.431, "star_class": "G (Sun-like)"},
    "Jupiter": {"pl_rade": 11.21, "pl_insol": 0.0369, "star_class": "G (Sun-like)"},
    "Saturn": {"pl_rade": 9.45, "pl_insol": 0.0109, "star_class": "G (Sun-like)"},
    "Uranus": {"pl_rade": 4.01, "pl_insol": 0.00272, "star_class": "G (Sun-like)"},
    "Neptune": {"pl_rade": 3.88, "pl_insol": 0.00111, "star_class": "G (Sun-like)"},
    "Pluto": {"pl_rade": 0.187, "pl_insol": 0.00064, "star_class": "G (Sun-like)"},
    "Kepler-442 b": {"pl_rade": 1.34, "pl_insol": 0.70, "star_class": "K"},
    "Kepler-438 b": {"pl_rade": 1.12, "pl_insol": 1.40, "star_class": "K"},
    "TRAPPIST-1 e": {"pl_rade": 0.92, "pl_insol": 0.66, "star_class": "M (red dwarf)"},
    "Proxima Centauri b": {"pl_rade": 1.07, "pl_insol": 0.65, "star_class": "M (red dwarf)"},
}

TOI_PRESETS = {
    "-- Select a preset --": None,
    (
        "TOI-7347.01 (formula-ranked lead; unconfirmed TESS candidate that "
        "fails the strict habitability gate)"
    ): {"pl_orbper": 60.834488, "st_teff": 3736.0, "sy_dist": 227.337},
    "TOI-6714.01 (closest system)": {"pl_orbper": 4.767442, "st_teff": 2824.0, "sy_dist": 26.741},
    "TOI-2094.01": {"pl_orbper": 18.793175, "st_teff": 3457.0, "sy_dist": 50.0248},
    "TOI-450.01": {"pl_orbper": 10.714866, "st_teff": 3054.0, "sy_dist": 53.5063},
    "Earth (hypothetical, nearby placeholder)": {"pl_orbper": 365.25, "st_teff": 5778.0, "sy_dist": 10.0},
}

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Full Data (Exact Score)", "🔭 Indirect Prediction (Candidates)", "ℹ️ About", "📖 Glossary"])
# ---------- TAB 1: Exact ESI calculation ----------
with tab1:
    st.markdown("### Enter known planet properties")
    st.caption("Use this when you have precise radius and insolation flux measurements.")

    preset_choice = st.selectbox("Or load a known planet:", list(PRESETS.keys()))
    preset = PRESETS[preset_choice]

    pl_rade = st.number_input("Planet Radius (Earth radii)",
                                value=preset["pl_rade"] if preset else 1.0, step=0.01)
    pl_insol = st.number_input("Insolation Flux (Earth flux = 1.0)", min_value=0.0,
                                 value=preset["pl_insol"] if preset else 1.0, step=0.01, format="%.4f")
    star_class = st.selectbox("Host Star Class", ["G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)"],
                                index=["G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)"].index(preset["star_class"]) if preset else 0)

    exact_clicked = st.button("Calculate Compatibility Scores", key="exact")
    exact_input_error = None
    try:
        validated_radius = validate_positive_finite("Planet radius", pl_rade)
        validated_insolation = validate_nonnegative_finite("Insolation flux", pl_insol)
        esi = calculate_esi(validated_radius, validated_insolation)
        bonus = calculate_custom_adjustment(validated_radius, star_class)
    except ValueError as error:
        exact_input_error = str(error)

    if exact_clicked and exact_input_error:
        st.error(exact_input_error)
    if exact_clicked and exact_input_error is None:
        final_score = esi + bonus
        for warning in training_range_warnings(
            {"pl_rade": validated_radius, "pl_insol": validated_insolation}
        ):
            st.warning(warning)
        in_hz = 0.35 <= pl_insol <= 1.1
        is_rocky = pl_rade < 1.75

        if final_score > 1:
            score_delta = f"+{final_score - 1:.3f} vs Earth"
        elif final_score < 1:
            score_delta = f"{final_score - 1:.3f} vs Earth"
        else:
            score_delta = None

        st.metric(
            "Base ESI (radius + insolation)",
            f"{esi:.3f}",
        )
        st.metric(
            "Custom research-inspired adjustment",
            f"+{bonus:.3f}",
        )
        st.metric(
            "Custom-adjusted score (base ESI + project bonus)",
            f"{final_score:.3f}",
            delta=score_delta,
        )
        st.caption(
            "Base ESI = sqrt((1 - |(R - 1)/(R + 1)|)^0.57 × "
            "(1 - |(S - 1)/(S + 1)|)^0.7), where R is radius in Earth radii "
            "and S is insolation in Earth flux. For R > 0 and S >= 0, base ESI "
            "range: [0, 1]. "
            "The unscaled custom-adjusted score adds 0.05 for a K-class host "
            "and 0.05 when 1 < R ≤ 1.5; for valid inputs its range is "
            "[0, 1.1)."
        )

        if in_hz and is_rocky:
            st.success("✅ Passes the project's illustrative size/insolation screen")
        else:
            reasons = []
            if not in_hz:
                reasons.append("insolation flux outside habitable zone range (0.35–1.1)")
            if not is_rocky:
                reasons.append("radius suggests a gas giant, not rocky")
            st.warning(f"⚠️ Does not pass the project's illustrative screen — {', '.join(reasons)}.")

        if bonus > 0:
            st.info(
                f"Custom research-inspired adjustment: +{bonus:.2f} "
                f"({star_class} host / favorable size). This is separate from base ESI."
            )

        # --- Size comparison visual ---
        st.markdown("#### Size Comparison")
        size_html = generate_size_comparison_html(pl_rade, pl_insol)
        st.html(size_html)

        st.markdown("#### System Preview")
        st.caption("Star color reflects representative spectral-class temperature; planet color reflects insolation flux (blue = cold, red = hot). Sizes are roughly proportional.")
        system_html = generate_system_html(pl_rade, pl_insol, star_class)
        st.html(system_html)
        # --- Scatter plot vs known planets ---
        st.markdown("#### Where This Planet Lands")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(planets_df['pl_insol'], planets_df['pl_rade'], color='lightgrey', alpha=0.4, label='Known planets')
        ax2.scatter([pl_insol], [pl_rade], color='red', s=150, edgecolor='black', zorder=5, label='Your planet')
        ax2.axvline(1.0, color='gold', linestyle='--', alpha=0.6)
        ax2.axhline(1.0, color='green', linestyle='--', alpha=0.6)
        ax2.set_xscale('log')
        ax2.set_xlabel('Insolation Flux (log scale, Earth = 1.0)')
        ax2.set_ylabel('Planet Radius (Earth radii)')
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

# ---------- TAB 2: Indirect ML prediction ----------
with tab2:
    st.markdown("### Enter limited/indirect planet properties")
    st.caption("Use this for unconfirmed candidates where only orbital period, stellar temperature, and distance are known.")
    st.caption(
        "Model contract: raw numeric inputs in the order pl_orbper (days), "
        "st_teff (K), sy_dist (pc); no scaling or other preprocessing is applied."
    )
    range_text = "; ".join(
        f"{name} [{TRAINING_RANGES[name][0]:g}, {TRAINING_RANGES[name][1]:g}] "
        f"{FEATURE_UNITS[name]}"
        for name in MODEL_FEATURES
    )
    st.caption(
        "Observed training-data ranges (outside-range values are extrapolations, "
        f"not hard limits): {range_text}."
    )

    toi_preset_choice = st.selectbox("Or load a real TESS candidate:", list(TOI_PRESETS.keys()))
    toi_preset = TOI_PRESETS[toi_preset_choice]

    pl_orbper = st.number_input("Orbital Period (days)",
                                  value=toi_preset["pl_orbper"] if toi_preset else 365.25, step=0.1)
    st_teff = st.number_input("Stellar Effective Temperature (K)",
                                value=toi_preset["st_teff"] if toi_preset else 5778.0, step=10.0)
    sy_dist = st.number_input("Distance from Earth (parsecs)",
                                value=toi_preset["sy_dist"] if toi_preset else 100.0, step=1.0)
    prediction_clicked = st.button("Predict Compatibility Score", key="indirect")
    prediction_input_error = None
    try:
        feature_values = {
            "pl_orbper": validate_positive_finite("Orbital period", pl_orbper),
            "st_teff": validate_positive_finite("Stellar effective temperature", st_teff),
            "sy_dist": validate_positive_finite("Distance from Earth", sy_dist),
        }
    except ValueError as error:
        prediction_input_error = str(error)

    if prediction_clicked and prediction_input_error:
        st.error(prediction_input_error)
    if prediction_clicked and prediction_input_error is None:
        for warning in training_range_warnings(feature_values):
            st.warning(warning)
        predicted_score = predict_compatibility(rf_compat, feature_values)[0]

        st.metric(
            "Model estimate of its training-target score",
            f"{predicted_score:.3f}",
            delta="Estimate, not exact",
        )
        candidate_scores = toi_df["predicted_score"]
        candidates_scoring_at_least_as_high = int(
            (candidate_scores >= predicted_score).sum()
        )
        st.caption(
            "This is a model estimate, not a probability of life or habitability. "
            "It estimates the formula-derived score used as its training target and "
            "has not been calibrated as a probability. Candidate comparison scores "
            "are recomputed from the same loaded model artifact and raw-input contract. "
            f"{candidate_extrapolation_count:,} of {len(toi_df):,} bundled candidate "
            "inputs fall outside at least one observed training range, so those "
            "comparison predictions are extrapolations. "
            "Tab 1 and Tab 2 use different methods and their scores are not directly "
            f"comparable. {candidates_scoring_at_least_as_high:,} of "
            f"{len(toi_df):,} bundled TESS candidates score this highly or higher."
        )
        # --- Scatter plot vs real TOI candidates ---
        st.markdown("#### Where This Candidate Lands")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(toi_df['pl_orbper'], toi_df['st_teff'], c=toi_df['predicted_score'], 
                     cmap='RdYlBu_r', alpha=0.6, s=40, label='Real TOI candidates')
        ax3.scatter([pl_orbper], [st_teff], color='black', s=200, marker='*', 
                     edgecolor='white', linewidth=1.5, zorder=5, label='Your input')
        ax3.set_xscale('log')
        ax3.set_xlabel('Orbital Period (days, log scale)')
        ax3.set_ylabel('Stellar Effective Temperature (K)')
        ax3.legend()
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Model estimate of training-target score')
        st.pyplot(fig3)
        plt.close(fig3)

# ---------- TAB 3: About ----------
with tab3:
    st.markdown("""
    ### About EXOPHCA

    This tool compares exoplanets with Earth using two complementary approaches, built on
    NASA Exoplanet Archive data (PSCompPars + TESS Objects of Interest).

    **1. Base ESI and custom-adjusted score**
    The base ESI uses radius R (Earth radii) and insolation S (Earth-relative flux):
    sqrt((1 - |(R - 1)/(R + 1)|)^0.57 × (1 - |(S - 1)/(S + 1)|)^0.7).
    For positive finite inputs its range is (0, 1], with Earth = 1.0. Separately,
    the project adds 0.05 for a K-class host and 0.05 when 1 < R <= 1.5.
    The unscaled adjusted score is in (0, 1.1); it is not capped or rescaled.
    Neither score is a measure of the probability that life exists.

    **2. Indirect Prediction (Machine Learning)**
    For candidate comparisons, this model estimates the formula-derived score used as its
    training target, using orbital period (days), stellar temperature (K), and distance
    (pc), in that feature order. It does not predict the probability of extraterrestrial
    life or habitability and has not been calibrated as a probability. Its live candidate
    comparisons are recomputed using the same loaded artifact and raw-input contract.

    **A note on methodology — avoiding data leakage:**
    Early versions of this model used radius and insolation as *both* the training features
    *and* the basis of the target score, producing a nearly perfect (and meaningless) R² of
    0.99. This model excludes those target-defining variables. Its reported held-out metrics
    measure agreement with the formula-derived target and do not establish generalization
    to new populations or life-detection ability.

    **Earth example:** with a 365.25-day period, 5,778 K stellar temperature, and a
    10 pc placeholder distance, the saved model predicts 0.551. Earth's base ESI from
    radius and insolation is 1.0. These are outputs from different methods and inputs;
    the difference is not evidence about measured habitability.
    """)


with tab4:
    st.markdown("### Glossary of Terms")
    st.caption("Plain-language explanations of the terms used throughout this tool.")

    terms = {
        "Insolation Flux": "How much energy/sunlight a planet receives from its star, "
            "measured relative to Earth (Earth = 1.0). A planet with 2.0 receives twice "
            "Earth's sunlight; 0.5 receives half.",
        "Earth Radii (R⊕)": "A planet's size compared to Earth. 1.0 means the same size as "
            "Earth; 11.2 means the same size as Jupiter.",
        "Habitable Zone": "The range of distance (or insolation) from a star where a planet "
            "could plausibly have liquid water on its surface — not too hot, not too cold.",
        "Earth Similarity Index (ESI)": "The unadjusted radius-and-insolation formula score, "
            "in [0, 1] for valid finite inputs, with Earth = 1.0. EXOPHCA separately shows "
            "a custom-adjusted score that can exceed 1 because it adds project-defined bonuses.",
        "Superhabitability": "The idea that some planets could be even more suitable for life "
            "than Earth — for example, orbiting a longer-lived star, or being slightly larger "
            "with a longer-lasting atmosphere.",
        "Stellar Effective Temperature (st_teff)": "The surface temperature of a star, measured in "
            "Kelvin (K). Our Sun is about 5,778 K. Hotter stars appear blue-white; cooler stars "
            "appear orange-red. This temperature determines a star's spectral class.",
        "Star Class (spectral type)": "A category based on a star's temperature: hotter stars "
            "(A, F) burn brighter and faster; cooler stars (K, M) burn dimmer but live much longer. "
            "Our Sun is class G.",
        "K-dwarf / M-dwarf": "Cooler, smaller stars than our Sun. M-dwarfs ('red dwarfs') are the "
            "most common stars in the galaxy; K-dwarfs are considered especially promising for "
            "long-term habitability due to their stability and longevity.",
        "Orbital Period": "How many days a planet takes to complete one full orbit around its star.",
        "Rocky vs Gas Giant": "Rocky planets have a solid surface (like Earth); gas giants "
            "(like Jupiter) are mostly gas with no solid surface — generally not considered "
            "candidates for life as we know it.",
        "Indirect Prediction": "An estimate made using only partial, early-observation data "
            "(orbital period, star temperature, distance) — used for planets not yet fully "
            "measured, like newly discovered candidates.",
        "Data Leakage": "A modeling mistake where a model is accidentally given information "
            "that lets it 'cheat' — for example, if the answer is secretly hidden inside the "
            "input data. This project specifically avoids it in the indirect model.",
        "R² (R-squared)": "A score from 0 to 1 showing how well a model's predictions match "
            "reality. Closer to 1.0 means better predictive accuracy; closer to 0 means "
            "the model isn't explaining much.",
        "Parsec": "A unit of astronomical distance, about 3.26 light-years.",
    }

    for term, explanation in terms.items():
        with st.expander(term):
            st.write(explanation)
