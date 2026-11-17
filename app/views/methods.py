"""Methods and limitations: how every number in the app is produced, and its limits."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st  # noqa: E402

import methods_content  # noqa: E402
import ui  # noqa: E402

ui.page_header(
    "Methods and limitations",
    "The score formula, the habitable-zone limits, how the uncertainty was "
    "propagated, the evidence rules behind each tier, and what none of it can tell "
    "you.",
    eyebrow="Reference",
)

ui.disclaimer()
st.write("")

methods_content.render(include_radius_model=True)

st.divider()
st.caption(
    "The same text is available as an expander at the bottom of the Planet Detail "
    "page, and the Calculator page evaluates the same formula and the same limits "
    "for values you enter yourself."
)
