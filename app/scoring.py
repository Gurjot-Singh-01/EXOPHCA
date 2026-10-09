"""Scoring and inference validation for the EXOPHCA Streamlit app."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_FEATURES = ("pl_orbper", "st_teff", "sy_dist")
FEATURE_UNITS = {
    "pl_orbper": "days",
    "st_teff": "K",
    "sy_dist": "pc",
    "pl_rade": "Earth radii",
    "pl_insol": "Earth flux",
}
TRAINING_RANGES = {
    "pl_orbper": (0.1768913, 402_000_000.0),
    "st_teff": (2375.0, 40_000.0),
    "sy_dist": (1.30119, 3460.51),
    "pl_rade": (0.3098, 25.0),
    "pl_insol": (0.0, 636_352.1258),
}
SCORE_UPPER_BOUND = 1.1
STAR_CLASSES = ("G (Sun-like)", "K", "M (red dwarf)", "F", "A (hot)")


def validate_positive_finite(name: str, value: Any) -> float:
    """Return a positive finite number or raise a user-facing validation error."""
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a numeric value.") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return number


def validate_nonnegative_finite(name: str, value: Any) -> float:
    """Return a nonnegative finite number or raise a user-facing validation error."""
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a numeric value.") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if number < 0:
        raise ValueError(f"{name} must be zero or greater.")
    return number


def calculate_esi(radius_earth: Any, insolation_earth: Any) -> float:
    """Calculate the uncapped ESI from radius and insolation relative to Earth."""
    radius = validate_positive_finite("Planet radius", radius_earth)
    insolation = validate_nonnegative_finite("Insolation flux", insolation_earth)

    radius_component = (
        1 - abs((radius - 1.0) / (radius + 1.0))
    ) ** 0.57
    insolation_component = (
        1 - abs((insolation - 1.0) / (insolation + 1.0))
    ) ** 0.7
    return float((radius_component * insolation_component) ** 0.5)


def calculate_custom_adjustment(radius_earth: Any, star_class: Any) -> float:
    """Return the two documented project bonuses, independently of base ESI."""
    radius = validate_positive_finite("Planet radius", radius_earth)
    if star_class is None or star_class not in STAR_CLASSES:
        raise ValueError(f"Host star class must be one of: {', '.join(STAR_CLASSES)}.")

    adjustment = 0.05 if star_class == "K" else 0.0
    if 1.0 < radius <= 1.5:
        adjustment += 0.05
    return adjustment


def training_range_warnings(values: Mapping[str, Any]) -> list[str]:
    """Describe values outside observed training-data minima/maxima."""
    warnings = []
    for name, value in values.items():
        if name not in TRAINING_RANGES:
            raise ValueError(f"No training range is recorded for {name}.")
        validator = (
            validate_nonnegative_finite
            if name == "pl_insol"
            else validate_positive_finite
        )
        number = validator(name, value)
        lower, upper = TRAINING_RANGES[name]
        if number < lower or number > upper:
            unit = FEATURE_UNITS.get(name, "Earth-relative units")
            warnings.append(
                f"{name} = {number:g} {unit} is outside the observed training "
                f"range [{lower:g}, {upper:g}] {unit}; treat the result as extrapolation."
            )
    return warnings


def _validate_model_contract(model: Any) -> None:
    if not isinstance(model, RandomForestRegressor):
        raise ValueError(
            "The loaded artifact must be the unwrapped RandomForestRegressor; "
            "no preprocessing pipeline is expected."
        )
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or tuple(feature_names) != MODEL_FEATURES:
        raise ValueError(
            "Model feature names/order do not match "
            f"{', '.join(MODEL_FEATURES)}."
        )
    if getattr(model, "n_features_in_", None) != len(MODEL_FEATURES):
        raise ValueError(f"The model must accept exactly {len(MODEL_FEATURES)} features.")


def _validated_feature_frame(features: Any) -> pd.DataFrame:
    if isinstance(features, Mapping):
        if set(features) != set(MODEL_FEATURES):
            raise ValueError(
                "Prediction input must contain exactly these features: "
                f"{', '.join(MODEL_FEATURES)}."
            )
        values = {
            name: validate_positive_finite(name, features[name])
            for name in MODEL_FEATURES
        }
        return pd.DataFrame([values], columns=MODEL_FEATURES)

    if not isinstance(features, pd.DataFrame):
        raise ValueError("Prediction inputs must be a feature mapping or pandas DataFrame.")
    if tuple(features.columns) != MODEL_FEATURES:
        raise ValueError(
            "Prediction columns and order must be exactly: "
            f"{', '.join(MODEL_FEATURES)}."
        )
    if features.empty:
        raise ValueError("At least one prediction row is required.")

    validated_rows = [
        {
            name: validate_positive_finite(name, row[name])
            for name in MODEL_FEATURES
        }
        for _, row in features.iterrows()
    ]
    return pd.DataFrame(validated_rows, columns=MODEL_FEATURES)


def predict_compatibility(model: Any, features: Any) -> np.ndarray:
    """Validate the model/input contract and return finite training-target estimates."""
    _validate_model_contract(model)
    frame = _validated_feature_frame(features)
    predictions = np.asarray(model.predict(frame), dtype=float)
    if predictions.shape != (len(frame),):
        raise ValueError("The model must return exactly one score per input row.")
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned a non-finite compatibility score.")
    if ((predictions < 0.0) | (predictions >= SCORE_UPPER_BOUND)).any():
        raise ValueError(
            f"The model returned a score outside the permitted training-target range "
            f"[0, {SCORE_UPPER_BOUND})."
        )
    return predictions
