"""Display rules for probabilistic numbers and radius provenance.

The app may show four Monte Carlo probabilities and, separately, a radius.
Both need qualification before they are shown:

* every probability carries its ``uncertainty_status``;
* when the variable a probability depends on was *not* sampled (held fixed at
  its archive value), the number is labelled ``conditional on a fixed input
  (uncertainty unknown)`` and never shown bare;
* a modelled radius is labelled ``radius modelled, not measured``.

The functions here are pure (stdlib + no Streamlit) so the rule can be tested
directly, and the same functions feed the pages, the Explorer table and the CSV
export.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

CONDITIONAL_TEXT = "conditional on a fixed input (uncertainty unknown)"
MODELLED_RADIUS_TEXT = "radius modelled, not measured"
DERIVED_RADIUS_TEXT = "radius derived, provenance unclear"
MEASURED_RADIUS_TEXT = "radius measured"
MISSING_RADIUS_TEXT = "radius missing"

# Which physical input each Monte Carlo probability is computed from. This is
# read off exophca/science/uncertainty.py: the score percentiles use both
# inputs, the habitable-zone probabilities use flux alone (at fixed Teff), and
# the size probabilities use radius alone.
PROBABILITY_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "p_conservative_hz": ("flux",),
    "p_any_hz": ("flux",),
    "p_radius_le_1p6": ("radius",),
    "p_rocky_and_conservative": ("radius", "flux"),
}

PROBABILITY_LABELS: dict[str, str] = {
    "p_conservative_hz": "P(conservative habitable zone)",
    "p_any_hz": "P(any habitable zone)",
    "p_radius_le_1p6": "P(radius ≤ 1.6 R⊕)",
    "p_rocky_and_conservative": "P(rocky size and conservative zone)",
}

# uncertainty_status values produced by exophca/science/uncertainty.py:
#   full        - radius and flux errors both present, both sampled
#   radius_only - only the radius was sampled; flux held fixed
#   flux_only   - only the flux was sampled; radius held fixed
#   none        - no usable errors; both held fixed
#   limit_value - radius and/or flux is an archive limit value; held fixed
SAMPLED_INPUTS: dict[str, frozenset[str]] = {
    "full": frozenset({"radius", "flux"}),
    "radius_only": frozenset({"radius"}),
    "flux_only": frozenset({"flux"}),
    "none": frozenset(),
    "limit_value": frozenset(),
}

STATUS_DESCRIPTIONS: dict[str, str] = {
    "full": "radius and flux both sampled",
    "radius_only": "radius sampled, flux held fixed",
    "flux_only": "flux sampled, radius held fixed",
    "none": "radius and flux both held fixed",
    "limit_value": "an archive limit value was held fixed",
}


def normalise_status(status: Any) -> str:
    """Return a known uncertainty_status, defaulting to the strictest one.

    An unrecognised or missing status is treated as "nothing was sampled" so
    that an unknown status can never make a probability look unconditional.
    """
    if isinstance(status, str) and status in SAMPLED_INPUTS:
        return status
    return "none"


def status_text(status: Any) -> str:
    """Readable ``uncertainty_status: ...`` fragment shown beside a probability."""
    key = normalise_status(status)
    return f"uncertainty_status: {key} ({STATUS_DESCRIPTIONS[key]})"


def fixed_inputs(probability: str, status: Any) -> tuple[str, ...]:
    """Inputs that this probability depends on but that were held fixed."""
    dependencies = PROBABILITY_DEPENDENCIES[probability]
    sampled = SAMPLED_INPUTS[normalise_status(status)]
    return tuple(name for name in dependencies if name not in sampled)


def depends_on_uncertain_input(probability: str, status: Any) -> bool:
    """True when the probability is conditional on an unsampled input."""
    return bool(fixed_inputs(probability, status))


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.3f}"


def format_probability(probability: str, value: Any, status: Any) -> str:
    """Format one probability together with its uncertainty status.

    With ``uncertainty_status == "full"`` the number is shown with the status
    it was computed under. Otherwise the conditional phrase is always present.
    """
    if probability not in PROBABILITY_DEPENDENCIES:
        raise KeyError(f"Unknown probability: {probability!r}")

    number = _number(value)
    status_key = normalise_status(status)
    if depends_on_uncertain_input(probability, status_key):
        fixed = " and ".join(fixed_inputs(probability, status_key))
        return (
            f"{number} — {CONDITIONAL_TEXT} ({fixed} held fixed; "
            f"uncertainty_status: {status_key})"
        )
    return f"{number} — uncertainty_status: {status_key} ({STATUS_DESCRIPTIONS[status_key]})"


def radius_text(radius: Any, provenance: Any) -> str:
    """Format a radius with its provenance, always naming a modelled radius."""
    if radius is None:
        return MISSING_RADIUS_TEXT
    try:
        value = float(radius)
    except (TypeError, ValueError):
        return MISSING_RADIUS_TEXT
    if not math.isfinite(value) or value <= 0:
        return MISSING_RADIUS_TEXT

    base = f"{value:.3f} R⊕"
    if provenance == "modelled":
        return f"{base} — {MODELLED_RADIUS_TEXT}"
    if provenance == "derived_unclear":
        return f"{base} — {DERIVED_RADIUS_TEXT}"
    if provenance == "measured":
        return f"{base} — {MEASURED_RADIUS_TEXT}"
    return f"{base} — radius provenance unknown"


RADIUS_REFERENCE = {
    "measured": (
        "Measured radius: a published radius (for a transit, from the transit "
        "depth) with a published uncertainty."
    ),
    "modelled": (
        "Modelled radius: no published radius exists, so it was estimated from the "
        "planet's mass through a mass-radius relation. It carries that relation's "
        "scatter in addition to the measurement error."
    ),
    "derived_unclear": (
        "Radius of unclear derivation: the archive radius comes from a derivation "
        "whose method is not recorded in the processed columns."
    ),
}


def radius_reference(provenance: Any) -> str:
    """A sentence explaining where this radius came from."""
    return RADIUS_REFERENCE.get(
        provenance, "Radius provenance is not recorded for this row."
    )


# Table cells need a tag short enough to read without truncation; the full
# sentence is listed underneath the table.
RADIUS_SHORT = {
    "measured": "Measured radius",
    "modelled": "Modelled radius, not measured",
    "derived_unclear": "Derived radius, provenance unclear",
}


def radius_short(provenance: Any) -> str:
    """A short provenance tag for a table cell."""
    return RADIUS_SHORT.get(provenance, "Radius provenance unknown")


def conditional_probabilities(status: Any) -> list[str]:
    """Probabilities that must be labelled conditional for this status."""
    return [
        name
        for name in PROBABILITY_DEPENDENCIES
        if depends_on_uncertain_input(name, status)
    ]


def probability_caveat(row: Mapping[str, Any]) -> str:
    """One-sentence caveat covering every probability in a row (used in CSVs)."""
    status = row.get("uncertainty_status")
    conditional = conditional_probabilities(status)
    if not conditional:
        return (
            "every probability is computed with both radius and flux sampled "
            f"({status_text(status)})"
        )
    fixed: set[str] = set()
    for name in conditional:
        fixed.update(fixed_inputs(name, status))
    return (
        f"{CONDITIONAL_TEXT}: {', '.join(conditional)} "
        f"({' and '.join(sorted(fixed))} held fixed; uncertainty_status: "
        f"{normalise_status(status)})"
    )


def labelled_probability_columns(frame, status_column: str = "uncertainty_status"):
    """Return a copy of ``frame`` with each probability as a labelled string."""
    out = frame.copy()
    for name in PROBABILITY_DEPENDENCIES:
        if name not in out.columns:
            continue
        out[name] = [
            format_probability(name, value, status)
            for value, status in zip(out[name], out[status_column])
        ]
    return out


def iter_probability_names() -> Iterable[str]:
    return iter(PROBABILITY_DEPENDENCIES)
