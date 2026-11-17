"""Architecture: one interactive diagram of how the project works.

The diagram is a single self-contained HTML file, ``app/assets/architecture.html``:
inline CSS, inline SVG and inline JavaScript, no CDN, no web fonts and no
external images, so it renders with no network access. It is embedded in an
iframe, which keeps its hover cards and its own horizontal scrolling away from
the page layout at phone width.

The file is authored as an Archify architecture specification and rendered by
the Archify CLI (v3.2.0, Apache-2.0, https://github.com/tt-a1i/archify). The
specification is not shipped: the rendered HTML carries the whole diagram.

Embedding is version-tolerant. ``st.iframe`` gained its ``alt`` argument in
Streamlit 1.66, so on an older Streamlit that argument is simply not passed;
if ``st.iframe`` is absent or rejects the call, the file's contents go through
``st.components.v1.html`` instead. The page renders the diagram either way.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st  # noqa: E402
from streamlit.components.v1 import html as components_html  # noqa: E402

import ui  # noqa: E402

DIAGRAM_PATH = APP_DIR / "assets" / "architecture.html"
# The diagram's own document measured 931 px tall at 1280 px in the browser
# (inner scrollHeight, 792 px wide); 20 px of margin keeps the longest hover
# card clear of the frame, so the iframe never grows a second scrollbar at
# desktop width. At phone width the diagram still scrolls inside the frame.
DIAGRAM_HEIGHT = 981

ALT_TEXT = (
    "EXOPHCA architecture: archives, offline scripts, science modules, radius "
    "model and the app pages that read the results."
)


def _accepts_alt(iframe) -> bool:
    """Whether this Streamlit's ``st.iframe`` has an ``alt`` parameter.

    ``alt`` arrived in Streamlit 1.66; an older release raises ``TypeError`` for
    it. The signature is inspected once here rather than by trial and error.
    """
    try:
        return "alt" in inspect.signature(iframe).parameters
    except (TypeError, ValueError):
        return False


def embed_diagram() -> str:
    """Embed the diagram; return the API that was used, for the test suite."""
    iframe = getattr(st, "iframe", None)
    if iframe is not None:
        try:
            if _accepts_alt(iframe):
                st.iframe(DIAGRAM_PATH, height=DIAGRAM_HEIGHT, alt=ALT_TEXT)
            else:
                st.iframe(DIAGRAM_PATH, height=DIAGRAM_HEIGHT)
            return "st.iframe"
        except TypeError:
            # This Streamlit's st.iframe does not take the call at all.
            pass

    # Read only on this path: the normal one lets Streamlit serve the file.
    components_html(
        DIAGRAM_PATH.read_text(encoding="utf-8"),
        height=DIAGRAM_HEIGHT,
        scrolling=True,
    )
    return "st.components.v1.html"

ui.page_header(
    "Architecture",
    "One diagram of the whole project: the raw archives, the offline steps that "
    "turn them into CSV files, the science modules, the radius model branch and "
    "the pages that read the results.",
    eyebrow="Reference",
)

ui.disclaimer()

st.caption(
    "Every box is a real file in this repository; hover or click one to see its "
    "path and what it does. The diagram is one self-contained HTML file with "
    "inline CSS, SVG and JavaScript, so it needs no network access to display."
)

if not DIAGRAM_PATH.exists():
    st.error(
        f"The diagram file is missing: {DIAGRAM_PATH.relative_to(PROJECT_ROOT)}. "
        "It is committed with the project; restore it from version control."
    )
    st.stop()

# ``st.iframe`` reads the local HTML file and embeds it with JavaScript enabled;
# it replaces ``st.components.v1.html``, which this Streamlit version deprecates.
embed_diagram()

st.divider()

st.markdown("#### How it was made")
st.markdown(
    "The diagram was authored as an Archify **architecture** specification and "
    "rendered by the Archify command-line tool, which also ran its schema, "
    "delivery and real-browser checks. Archify is free and open source under the "
    "**Apache-2.0** licence, with a separate MIT licence file shipped alongside "
    "it in its package: "
    "https://github.com/tt-a1i/archify. Its update check was disabled for this "
    "render (`ARCHIFY_UPDATE_CHECK_DISABLED=1`). The rendered HTML is a single "
    "file; the viewer runtime, its styles and its icons are inlined, and the "
    "only links that leave the page are the source links to the exact commit of "
    "this repository."
)
