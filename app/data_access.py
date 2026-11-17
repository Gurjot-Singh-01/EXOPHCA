"""Cached data access for the EXOPHCA Streamlit app.

All file paths are resolved from this module's location, so the app works
regardless of the directory `streamlit run` was started from. Every loader is
cached with ``st.cache_data``; a missing or malformed file raises
``DataFileMissing`` with a message that names the path, which the pages turn
into a readable error instead of a traceback.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import query  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PLANETS_ASSESSED_PATH = PROCESSED_DIR / "planets_assessed.csv"
PLANETS_PATH = PROCESSED_DIR / "planets.csv"
TOI_ASSESSED_PATH = PROCESSED_DIR / "toi_assessed.csv"
PSCD_PROVENANCE_PATH = PROCESSED_DIR / "PSCD.provenance.json"
TOI_PROVENANCE_PATH = PROCESSED_DIR / "TOI_2026.provenance.json"

# The radius model is run offline by scripts/estimate_radii.py; the app reads
# these two files and never the pickled model itself.
RADIUS_ESTIMATES_PATH = PROCESSED_DIR / "radius_estimates.csv"
RADIUS_MODEL_META_PATH = MODELS_DIR / "radius_model_meta.json"

RADIUS_ESTIMATE_COLUMNS = [
    "pl_name",
    "radius_estimate",
    "radius_lower",
    "radius_upper",
    "mass_is_minimum",
    "ood_flag",
    "interval_straddles_1p6",
    "reason",
]

DATASET_LABELS = {
    "pscd": "NASA Exoplanet Archive - Planetary Systems Composite Parameters",
    "toi": "NASA Exoplanet Archive - TESS Objects of Interest",
}

_REQUIRED_ASSESSED = query.ASSESSED_COLUMNS
_REQUIRED_ARCHIVE = query.ARCHIVE_COLUMNS
_REQUIRED_TOI = ["toi", "pl_rade", "pl_insol", "base_score", "hz_class", "tier", "uncertainty_status"]


class DataFileMissing(RuntimeError):
    """A processed data file (or a column the app relies on) is missing."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv_checked(path: Path, required_columns, label: str) -> pd.DataFrame:
    """Read a processed CSV and verify the columns the app depends on."""
    if not path.exists():
        raise DataFileMissing(
            f"{label} not found at {_relative(path)}. Rebuild it with the data "
            "pipeline before starting the app."
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataFileMissing(f"{label} at {_relative(path)} could not be read: {error}") from error

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataFileMissing(
            f"{label} at {_relative(path)} is missing column(s): {', '.join(missing)}."
        )
    return frame


@st.cache_data(show_spinner=False)
def load_planets_assessed() -> pd.DataFrame:
    """The 5,688 assessed confirmed planets (one row per planet)."""
    return read_csv_checked(
        PLANETS_ASSESSED_PATH, _REQUIRED_ASSESSED, "planets_assessed.csv"
    )


@st.cache_data(show_spinner=False)
def load_planets() -> pd.DataFrame:
    """The archive table (6,319 rows) used for host, method and mass columns."""
    return read_csv_checked(PLANETS_PATH, _REQUIRED_ARCHIVE, "planets.csv")


@st.cache_data(show_spinner=False)
def load_toi_assessed() -> pd.DataFrame:
    """The 4,992 assessed, unconfirmed TESS candidates."""
    return read_csv_checked(TOI_ASSESSED_PATH, _REQUIRED_TOI, "toi_assessed.csv")


@st.cache_data(show_spinner=False)
def load_planet_explorer() -> pd.DataFrame:
    """Assessed planets joined to their host and discovery columns."""
    return query.join_planets(load_planets_assessed(), load_planets())


def _flag(value) -> bool:
    """A boolean column read back from CSV, which stores it as text."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


@st.cache_data(show_spinner=False)
def load_radius_estimates() -> pd.DataFrame:
    """The precomputed ML radius estimates, one row per selected planet.

    Empty when the file has not been built yet, so the Detail page can fall back
    to saying no estimate is available rather than failing.
    """
    if not RADIUS_ESTIMATES_PATH.exists():
        return pd.DataFrame(columns=RADIUS_ESTIMATE_COLUMNS)

    frame = read_csv_checked(
        RADIUS_ESTIMATES_PATH, RADIUS_ESTIMATE_COLUMNS, "radius_estimates.csv"
    )
    for column in ("mass_is_minimum", "ood_flag", "interval_straddles_1p6"):
        frame[column] = frame[column].map(_flag).astype(bool)
    frame["reason"] = frame["reason"].fillna("")
    return frame


@st.cache_data(show_spinner=False)
def load_radius_model_meta() -> dict | None:
    """The radius model's training report, or None when it was not built."""
    if not RADIUS_MODEL_META_PATH.exists():
        return None
    try:
        return json.loads(RADIUS_MODEL_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_retrieval_date(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    for pattern in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestamp, pattern).date().isoformat()
        except ValueError:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_provenance() -> dict[str, dict]:
    """Provenance blocks for both source files, with a parsed retrieval date."""
    blocks: dict[str, dict] = {}
    for key, path in (("pscd", PSCD_PROVENANCE_PATH), ("toi", TOI_PROVENANCE_PATH)):
        if not path.exists():
            raise DataFileMissing(
                f"provenance file {path.name} not found at {_relative(path)}."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataFileMissing(
                f"provenance file {path.name} at {_relative(path)} could not be read: {error}"
            ) from error

        timestamp = raw.get("retrieval_timestamp")
        blocks[key] = {
            "label": DATASET_LABELS[key],
            "path": _relative(path),
            "original_file": raw.get("original_file"),
            "source_filename": raw.get("source_filename"),
            "retrieval_timestamp": timestamp,
            "retrieval_date": _parse_retrieval_date(timestamp),
            "row_count": raw.get("row_count"),
            "column_count": raw.get("column_count"),
            "sha256_prefix": (raw.get("sha256") or "")[:12],
        }
    return blocks


def load_or_stop(loader, label: str):
    """Run a cached loader, turning a missing file into a readable message."""
    try:
        return loader()
    except DataFileMissing as error:
        st.error(f"**{label} could not be loaded.** {error}")
        st.stop()
    except ValueError as error:  # join / schema mismatch
        st.error(f"**{label} could not be prepared.** {error}")
        st.stop()
