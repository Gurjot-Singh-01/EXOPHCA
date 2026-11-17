"""Calculator presets, computed from the reference tables - never typed in.

Two groups are offered:

* **Solar System** - the eight planets, computed from
  ``data/reference/solar_system.csv`` (NASA NSSDCA volumetric mean radii and
  semimajor axes). Radius is ``mean_radius_km / Earth's mean radius`` and
  insolation is ``(Earth's semimajor axis / a) ** 2``, both taken from the file's
  own Earth row, so no value is hard-coded here. Every Solar System preset uses
  the Sun's effective temperature and a G host class.
* **Archive planets** - every scorable planet in
  ``data/processed/planets_assessed.csv`` with that row's radius, flux, stellar
  temperature and star class. The four quick picks sit at the top of the group.

Because the Solar System radii are *volumetric mean* radii, Jupiter is about
10.97 R-Earth rather than its equatorial 11.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SOLAR_SYSTEM_PATH = PROJECT_ROOT / "data" / "reference" / "solar_system.csv"
SOLAR_SYSTEM_LABEL = "data/reference/solar_system.csv"
ARCHIVE_LABEL = "data/processed/planets_assessed.csv"

# The Sun, shared by all eight Solar System presets.
SOLAR_TEFF_K = 5772.0
SOLAR_STAR_CLASS = "G (Sun-like)"

# The original quick picks. Earth is one of them, but it is not an archive row:
# it is computed in the Solar System group rather than repeated in both.
QUICK_PICKS = ("Earth", "Proxima Cen b", "TRAPPIST-1 e", "Kepler-452 b", "TOI-700 d")
ARCHIVE_QUICK_PICKS = ("Proxima Cen b", "TRAPPIST-1 e", "Kepler-452 b", "TOI-700 d")

GROUP_SOLAR = "Solar System"
GROUP_ARCHIVE = "Archive planets"

SOLAR_SOURCE = (
    f"{SOLAR_SYSTEM_LABEL} - NASA NSSDCA fact-sheet values; radius = mean radius "
    "/ Earth's mean radius, flux = (Earth's semimajor axis / semimajor axis) squared"
)


def _number(value):
    """A finite float, or None for a missing value."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if pd.notna(number) else None


def solar_system_presets(path: Path | str = SOLAR_SYSTEM_PATH) -> dict[str, dict]:
    """``{name: preset}`` for the eight planets, computed from the reference CSV.

    Earth's own row supplies both denominators, so Earth comes out at exactly
    1.000 R-Earth and 1.000 S-Earth by construction.
    """
    table = pd.read_csv(path)
    missing = {"name", "mean_radius_km", "semimajor_axis_km"} - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing the column(s): {', '.join(sorted(missing))}"
        )
    earth = table.loc[table["name"] == "Earth"]
    if earth.empty:
        raise ValueError(f"{path} has no Earth row to scale the other planets against")

    earth_radius_km = float(earth.iloc[0]["mean_radius_km"])
    earth_axis_km = float(earth.iloc[0]["semimajor_axis_km"])
    if earth_radius_km <= 0 or earth_axis_km <= 0:
        raise ValueError(f"{path}: Earth's radius and semimajor axis must be positive")

    presets: dict[str, dict] = {}
    for row in table.itertuples(index=False):
        radius_km = _number(row.mean_radius_km)
        axis_km = _number(row.semimajor_axis_km)
        if radius_km is None or axis_km is None or radius_km <= 0 or axis_km <= 0:
            continue
        presets[str(row.name)] = {
            "pl_rade": radius_km / earth_radius_km,
            "pl_insol": (earth_axis_km / axis_km) ** 2,
            "st_teff": SOLAR_TEFF_K,
            "star_class": SOLAR_STAR_CLASS,
            "mean_radius_km": radius_km,
            "semimajor_axis_km": axis_km,
            "source": SOLAR_SOURCE,
            "group": GROUP_SOLAR,
        }
    return presets


def archive_presets(
    assessed: pd.DataFrame, quick_picks=ARCHIVE_QUICK_PICKS
) -> dict[str, dict]:
    """``{name: preset}`` for every scorable archive planet, quick picks first.

    Values come from the row itself, including ``st_teff`` and ``star_class``.
    A quick pick that is absent from the table is simply not offered.
    """
    records = assessed.drop_duplicates("pl_name").set_index("pl_name")
    present = set(records.index)
    leading = [name for name in quick_picks if name in present]
    ordered = leading + sorted(present - set(leading), key=str.casefold)

    presets: dict[str, dict] = {}
    for name in ordered:
        row = records.loc[name]
        presets[name] = {
            "pl_rade": _number(row["pl_rade"]),
            "pl_insol": _number(row["pl_insol"]),
            "st_teff": _number(row["st_teff"]),
            "star_class": row["star_class"],
            "source": ARCHIVE_LABEL,
            "group": GROUP_ARCHIVE,
        }
    return presets


def preset_groups(
    assessed: pd.DataFrame, *, solar_path: Path | str = SOLAR_SYSTEM_PATH
) -> dict[str, dict[str, dict]]:
    """The two menu groups, in display order."""
    return {
        GROUP_SOLAR: solar_system_presets(solar_path),
        GROUP_ARCHIVE: archive_presets(assessed),
    }


def all_presets(
    assessed: pd.DataFrame, *, solar_path: Path | str = SOLAR_SYSTEM_PATH
) -> dict[str, dict]:
    """Both groups in one mapping, for scoring every preset in one pass."""
    merged: dict[str, dict] = {}
    for group in preset_groups(assessed, solar_path=solar_path).values():
        merged.update(group)
    return merged
