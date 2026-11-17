"""EXOPHCA science module: ESI-style radius-flux similarity calculations and validation."""

from math import isfinite
from typing import Any

import numpy as np


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
    """Calculate the ESI-style radius-flux similarity score (capped at [0, 1]).

    Note on methodology:
    The published Earth Similarity Index (Schulze-Makuch et al. 2011) uses four
    parameters with weights: radius 0.57, bulk density 1.07, escape velocity 0.70,
    and surface temperature 5.58.
    This project uses radius with weight 0.57 and insolation flux with weight 0.70,
    where 0.70 is the published escape-velocity weight borrowed for flux and
    is NOT a published flux weight.
    ESI-style similarity strictly measures physical similarity to Earth from radius and flux alone,
    naturally bounded in [0, 1] with no bonuses included.
    """
    radius = validate_positive_finite("Planet radius", radius_earth)
    insolation = validate_nonnegative_finite("Insolation flux", insolation_earth)

    radius_component = (
        1 - abs((radius - 1.0) / (radius + 1.0))
    ) ** 0.57
    insolation_component = (
        1 - abs((insolation - 1.0) / (insolation + 1.0))
    ) ** 0.7
    score = float((radius_component * insolation_component) ** 0.5)
    return float(np.clip(score, 0.0, 1.0))


def classify_flags(radius_earth: Any, star_class: Any) -> dict[str, bool]:
    """Return labelled boolean project flags independently of base ESI."""
    radius = validate_positive_finite("Planet radius", radius_earth)
    if star_class is None or star_class not in STAR_CLASSES:
        raise ValueError(f"Host star class must be one of: {', '.join(STAR_CLASSES)}.")

    return {
        "host_is_k_class": (star_class == "K"),
        "radius_in_1_to_1p5": (1.0 < radius <= 1.5),
    }
