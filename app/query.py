"""Pure dataframe helpers for the Explorer page: join, filter, sort, export.

Kept free of Streamlit so the filtering and export behaviour can be tested
directly and reused by any page.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import labelling  # noqa: E402
from exophca.science.assessment import DISCLAIMER  # noqa: E402

ASSESSED_COLUMNS = [
    "pl_name",
    "status",
    "st_teff",
    "star_class",
    "pl_rade",
    "pl_insol",
    "radius_provenance",
    "uncertainty_status",
    "base_score",
    "base_score_p16",
    "base_score_p84",
    "hz_class",
    "tier",
    "evidence_quality",
    "boundary_sensitive",
    "criteria",
] + list(labelling.PROBABILITY_DEPENDENCIES)

# Columns taken from data/processed/planets.csv (the archive table). Only what
# the Explorer needs; the assessed file already carries star_class and st_teff.
ARCHIVE_COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_bmasse",
    "mass_provenance",
    "pl_orbper",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_insolerr1",
    "pl_insolerr2",
]

# What the table shows. Probabilities are deliberately absent here: they are
# only displayed through the labelled toggle, never as bare numbers. The radius
# is followed immediately by its provenance sentence so a modelled radius is
# never shown as a bare number.
DISPLAY_COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "star_class",
    "pl_rade",
    "radius_note",
    "pl_insol",
    "uncertainty_status",
    "base_score",
    "base_score_p16",
    "base_score_p84",
    "hz_class",
    "tier",
    "evidence_quality",
    "boundary_sensitive",
]

SORTABLE_COLUMNS = [
    "pl_name",
    "discoverymethod",
    "disc_year",
    "star_class",
    "pl_rade",
    "pl_insol",
    "base_score",
    "base_score_p16",
    "base_score_p84",
    "hz_class",
    "tier",
    "evidence_quality",
    "uncertainty_status",
]

# The name search reads these columns. The host name is included because it is
# not always a prefix of the planet name ("Barnard's star" vs "Barnard b").
SEARCH_COLUMNS = ["pl_name", "hostname"]

# Required in any export so provenance and uncertainty travel with the data.
REQUIRED_EXPORT_COLUMNS = [
    "radius_provenance",
    "radius_note",
    "uncertainty_status",
    "evidence_quality",
    "base_score_p16",
    "base_score_p84",
    "boundary_sensitive",
    "criteria",
    "probability_caveat",
    "data_retrieval_date",
]


def join_planets(assessed: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Join the assessed table to the archive table on ``pl_name`` (one-to-one)."""
    missing_assessed = [c for c in ASSESSED_COLUMNS if c not in assessed.columns]
    if missing_assessed:
        raise ValueError(
            "planets_assessed.csv is missing required columns: "
            + ", ".join(missing_assessed)
        )
    missing_archive = [c for c in ARCHIVE_COLUMNS if c not in archive.columns]
    if missing_archive:
        raise ValueError(
            "planets.csv is missing required columns: " + ", ".join(missing_archive)
        )

    merged = assessed[ASSESSED_COLUMNS].merge(
        archive[ARCHIVE_COLUMNS],
        on="pl_name",
        how="left",
        validate="one_to_one",
    )
    merged["boundary_sensitive"] = merged["boundary_sensitive"].fillna(False).astype(bool)
    return merged


def _matches(series: pd.Series, values: Sequence[Any]) -> bool:
    return not values or series.isin(list(values))


def apply_filters(
    frame: pd.DataFrame,
    *,
    search: str = "",
    discovery_methods: Iterable[str] = (),
    star_classes: Iterable[str] = (),
    radius_provenance: Iterable[str] = (),
    evidence_quality: Iterable[str] = (),
    tiers: Iterable[str] = (),
    hz_classes: Iterable[str] = (),
    uncertainty_status: Iterable[str] = (),
    score_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Filter the explorer table. An empty selection means "no filter".

    The name search is a case-insensitive, literal substring match with
    surrounding whitespace ignored, over both the planet name and its host star
    (``SEARCH_COLUMNS``). ``regex=False`` is what makes a name such as
    ``WD 1856+534 b`` match itself instead of being read as a pattern.
    """
    out = frame
    text = (search or "").strip()
    if text:
        matches = out["pl_name"].str.contains(text, case=False, regex=False, na=False)
        for column in SEARCH_COLUMNS[1:]:
            if column not in out.columns:
                continue
            values = out[column].astype("string").fillna("")
            matches = matches | values.str.contains(
                text, case=False, regex=False, na=False
            )
        out = out[matches]

    selections = {
        "discoverymethod": list(discovery_methods),
        "star_class": list(star_classes),
        "radius_provenance": list(radius_provenance),
        "evidence_quality": list(evidence_quality),
        "tier": list(tiers),
        "hz_class": list(hz_classes),
        "uncertainty_status": list(uncertainty_status),
    }
    for column, values in selections.items():
        if values:
            out = out[_matches(out[column], values)]

    if score_range is not None:
        low, high = score_range
        scores = out["base_score"]
        out = out[scores.notna() & (scores >= low) & (scores <= high)]

    return out


def filter_toi(
    frame: pd.DataFrame,
    *,
    tiers: Iterable[str] = (),
    hz_classes: Iterable[str] = (),
    evidence_quality: Iterable[str] = (),
    radius_range: tuple[float, float] | None = None,
    teff_range: tuple[float, float] | None = None,
    boundary_sensitive: bool | None = None,
) -> pd.DataFrame:
    """Filter the unconfirmed TESS candidates. An empty selection means "no filter"."""
    out = frame
    selections = {
        "tier": list(tiers),
        "hz_class": list(hz_classes),
        "evidence_quality": list(evidence_quality),
    }
    for column, values in selections.items():
        if values:
            out = out[_matches(out[column], values)]

    if radius_range is not None:
        low, high = radius_range
        radii = out["pl_rade"]
        out = out[radii.notna() & (radii >= low) & (radii <= high)]

    if teff_range is not None:
        low, high = teff_range
        teff = out["st_teff"]
        out = out[teff.notna() & (teff >= low) & (teff <= high)]

    if boundary_sensitive is not None:
        flags = out["boundary_sensitive"].fillna(False).astype(bool)
        out = out[flags] if boundary_sensitive else out[~flags]

    return out


def build_toi_export_frame(frame: pd.DataFrame, retrieval_date: str | None) -> pd.DataFrame:
    """The TESS candidates as a CSV: status, provenance, caveat and disclaimer."""
    out = frame.copy()
    out["radius_note"] = [
        labelling.radius_text(radius, "measured") for radius in out["pl_rade"]
    ]
    out["radius_provenance"] = "measured"
    out["probability_caveat"] = [
        labelling.probability_caveat(row) for _, row in out.iterrows()
    ]
    out["data_retrieval_date"] = retrieval_date or "unknown"
    out["disclaimer"] = DISCLAIMER

    columns = [
        "toi",
        "status",
        "st_teff",
        "pl_rade",
        "radius_note",
        "radius_provenance",
        "pl_insol",
        "base_score",
        "base_score_p16",
        "base_score_p84",
        "hz_class",
        "tier",
        "evidence_quality",
        "boundary_sensitive",
        "uncertainty_status",
        "p_conservative_hz",
        "p_any_hz",
        "p_radius_le_1p6",
        "p_rocky_and_conservative",
        "probability_caveat",
        "criteria",
        "data_retrieval_date",
        "disclaimer",
    ]
    missing = [name for name in columns if name not in out.columns]
    if missing:
        raise ValueError("TESS export is missing required columns: " + ", ".join(missing))
    return out[columns]


def sort_frame(
    frame: pd.DataFrame, column: str = "base_score", descending: bool = True
) -> pd.DataFrame:
    """Sort by one column, keeping missing values last."""
    if column not in frame.columns:
        return frame
    return frame.sort_values(column, ascending=not descending, na_position="last")


def with_radius_note(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the provenance sentence shown beside every radius."""
    out = frame.copy()
    out["radius_note"] = [
        labelling.radius_text(radius, provenance)
        for radius, provenance in zip(out["pl_rade"], out["radius_provenance"])
    ]
    return out


def plot_points(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows that can be drawn on the radius-insolation plane, plus the drops.

    ``charts`` holds every row with a positive radius and flux - no sampling and
    no cap - and ``dropped`` counts the rows with a missing or non-positive
    radius or flux that cannot be placed on a log axis.
    """
    usable = frame.dropna(subset=["pl_rade", "pl_insol"])
    usable = usable[(usable["pl_rade"] > 0) & (usable["pl_insol"] > 0)]
    return usable, int(len(frame) - len(usable))


def export_filename(retrieval_date: str | None, prefix: str = "exophca_planets") -> str:
    """`exophca_planets_filtered_2026-07-11.csv` (date from the provenance JSON)."""
    suffix = retrieval_date or "unknown-retrieval-date"
    return f"{prefix}_filtered_{suffix}.csv"


def build_export_frame(frame: pd.DataFrame, retrieval_date: str | None) -> pd.DataFrame:
    """Columns for the CSV download, including provenance and uncertainty.

    Numeric probabilities stay numeric so the file is analysable, and each row
    carries ``probability_caveat`` plus ``uncertainty_status`` so no probability
    leaves the app without its qualification.
    """
    out = with_radius_note(frame)
    out["probability_caveat"] = [labelling.probability_caveat(row) for _, row in out.iterrows()]
    out["data_retrieval_date"] = retrieval_date or "unknown"
    out["disclaimer"] = DISCLAIMER

    columns = [
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "star_class",
        "pl_rade",
        "radius_note",
        "radius_provenance",
        "pl_insol",
        "base_score",
        "base_score_p16",
        "base_score_p84",
        "hz_class",
        "tier",
        "evidence_quality",
        "boundary_sensitive",
        "uncertainty_status",
        "p_conservative_hz",
        "p_any_hz",
        "p_radius_le_1p6",
        "p_rocky_and_conservative",
        "probability_caveat",
        "criteria",
        "data_retrieval_date",
        "disclaimer",
    ]
    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise ValueError("Export is missing required columns: " + ", ".join(missing))
    return out[columns]
