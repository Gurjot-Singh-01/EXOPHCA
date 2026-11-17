"""EXOPHCA entry point: page configuration, theme and navigation.

Run it with the project interpreter, which has the versions declared in
``pyproject.toml``:

    .\\.venv\\Scripts\\python.exe -m streamlit run app/EXOPHCA.py

Starting the app with another interpreter is checked below. A machine-wide
Python can carry an older Streamlit that lacks an API a page uses, and the
failure then arrives as a ``TypeError`` inside one page rather than as a
sentence about the version. ``pyproject.toml`` is the only place the minimum
version is written; this file reads it instead of keeping a second copy.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def declared_streamlit_floor() -> str | None:
    """The minimum Streamlit version from ``pyproject.toml``, or ``None``.

    Returns ``None`` when the file cannot be read or declares no Streamlit
    requirement, so an unreadable project file can never block the app.
    """
    try:
        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None

    for requirement in project.get("dependencies", []):
        match = re.match(r"\s*streamlit\s*>=\s*([0-9][0-9.]*)", requirement)
        if match:
            return match.group(1)
    return None


def release_tuple(version: str) -> tuple[int, ...]:
    """``"1.2.3"`` -> ``(1, 2, 3)``; a pre-release suffix is ignored."""
    match = re.match(r"\s*(\d+(?:\.\d+)*)", version)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def at_least(installed: tuple[int, ...], floor: tuple[int, ...]) -> bool:
    """Compare releases after padding the shorter one with zeros.

    Padding matters: ``(1, 66)`` means the same release as ``(1, 66, 0)``, and a
    plain tuple comparison would call the shorter one older.
    """
    width = max(len(installed), len(floor))
    padded_installed = installed + (0,) * (width - len(installed))
    padded_floor = floor + (0,) * (width - len(floor))
    return padded_installed >= padded_floor


def check_streamlit_version() -> None:
    """Stop with a plain message when this interpreter's Streamlit is too old."""
    floor = declared_streamlit_floor()
    installed_text = st.__version__
    if floor is None:
        return
    if at_least(release_tuple(installed_text), release_tuple(floor)):
        return

    st.error(
        f"This app needs Streamlit {floor} or newer; you are running "
        f"{installed_text} from {sys.executable}. Start it with: "
        r".\.venv\Scripts\python.exe -m streamlit run app/EXOPHCA.py"
    )
    st.stop()


import ui  # noqa: E402

st.set_page_config(
    page_title="EXOPHCA",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Before any page is built: an older Streamlit would fail inside a page with a
# TypeError that says nothing about the version.
check_streamlit_version()

ui.inject_css()

st.sidebar.markdown("#### 🪐 EXOPHCA")
st.sidebar.caption("Exoplanet Habitability Compatibility Analysis")

navigation = st.navigation(
    {
        "Explore": [
            st.Page(
                "views/overview.py",
                title="Overview",
                icon=":material/space_dashboard:",
                default=True,
            ),
            st.Page(
                "views/explorer.py",
                title="Planet Explorer",
                icon=":material/table_view:",
            ),
            st.Page(
                "views/detail.py",
                title="Planet Detail",
                icon=":material/travel_explore:",
            ),
        ],
        "Research": [
            st.Page(
                "views/compare.py",
                title="Compare",
                icon=":material/compare_arrows:",
            ),
            st.Page(
                "views/diagrams.py",
                title="Diagrams",
                icon=":material/scatter_plot:",
            ),
        ],
        "Tools": [
            st.Page(
                "views/calculator.py",
                title="Calculator",
                icon=":material/calculate:",
            ),
            st.Page(
                "views/tess_candidates.py",
                title="TESS Candidates",
                icon=":material/satellite_alt:",
            ),
        ],
        "Reference": [
            st.Page(
                "views/methods.py",
                title="Methods and limitations",
                icon=":material/menu_book:",
            ),
            st.Page(
                "views/architecture.py",
                title="Architecture",
                icon=":material/account_tree:",
            ),
        ],
    },
    position="sidebar",
)

navigation.run()
