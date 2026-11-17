"""Earth as the reference row, computed rather than typed in.

Earth is not a row in the archive extract (the NASA table holds observed
exoplanets only), so its reference values are defined here and everything
derived from them is computed by the same science modules the archive planets
go through:

* the score comes from :func:`exophca.science.esi.calculate_esi`,
* the habitable-zone class from :func:`exophca.science.habitable_zone.classify_hz`
  at 1.0 S-Earth and 5772 K,
* the size regime and tier from :func:`exophca.science.assessment.determine_size_regime`
  and :func:`exophca.science.assessment.determine_tier`.

Nothing about Earth's class or tier is written down here.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from exophca.science.assessment import determine_size_regime, determine_tier  # noqa: E402
from exophca.science.esi import calculate_esi  # noqa: E402
from exophca.science.habitable_zone import classify_hz  # noqa: E402

# The reference definition: Earth's radius and flux are the units of the
# formula, and 5772 K is the adopted solar effective temperature.
EARTH_RADIUS = 1.0
EARTH_FLUX = 1.0
EARTH_TEFF = 5772.0
# Only used to preselect the Calculator's star-class box; it is a property of
# the 5772 K definition, not an archive value.
EARTH_STAR_CLASS = "G (Sun-like)"
EARTH_LABEL = "Earth (reference)"


def earth_reference() -> dict:
    """Earth's reference row, with class, regime and tier computed."""
    hz_class = classify_hz(EARTH_FLUX, EARTH_TEFF)
    size_regime = determine_size_regime(EARTH_RADIUS)
    return {
        "pl_name": EARTH_LABEL,
        "pl_rade": EARTH_RADIUS,
        "pl_insol": EARTH_FLUX,
        "st_teff": EARTH_TEFF,
        "base_score": calculate_esi(EARTH_RADIUS, EARTH_FLUX),
        "hz_class": hz_class,
        "size_regime": size_regime,
        "tier": determine_tier(hz_class, size_regime),
        "uncertainty_status": "none",
        "star_class": EARTH_STAR_CLASS,
    }
