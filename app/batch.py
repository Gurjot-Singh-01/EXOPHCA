"""Batch scoring for the Calculator: validate an uploaded CSV in memory.

Nothing in this module touches the filesystem. The caller hands over the bytes
of the uploaded file (or its text) and gets a result back, so an upload is never
written to disk.

A payload larger than ``MAX_UPLOAD_BYTES`` is refused before it is decoded or
parsed, and the row cap is enforced by the reader rather than after it, so an
upload can never allocate more than the cap allows.

A row is rejected when the radius or the flux is missing or not a number, when
the radius is not positive, when the flux is negative, or when a supplied
stellar temperature is not a number. A missing or out-of-range temperature is
not an error: the row is scored and its habitable-zone class is
``undetermined``, because the Kopparapu limits are not extrapolated.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _path in (str(PROJECT_ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from exophca.science.assessment import DISCLAIMER, determine_size_regime  # noqa: E402
from exophca.science.esi import calculate_esi  # noqa: E402
from exophca.science.habitable_zone import classify_hz  # noqa: E402

MAX_ROWS = 5000
MAX_UPLOAD_BYTES = 2_000_000
MAX_REPORTED_ERRORS = 20
REQUIRED_COLUMNS = ("pl_rade", "pl_insol")
OPTIONAL_COLUMNS = ("st_teff", "name")

RESULT_COLUMNS = [
    "name",
    "pl_rade",
    "pl_insol",
    "st_teff",
    "base_score",
    "hz_class",
    "size_regime",
    "disclaimer",
]


@dataclass(frozen=True)
class RowError:
    """One rejected row. ``line`` is the 1-based line in the file (header = 1)."""

    line: int
    column: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}, {self.column}: {self.message}"


@dataclass
class BatchResult:
    """Accepted rows, scored; plus the row-level errors and the row counts."""

    frame: pd.DataFrame
    errors: list[RowError] = field(default_factory=list)
    total_rows: int = 0

    @property
    def accepted(self) -> int:
        return len(self.frame)

    @property
    def rejected(self) -> int:
        return self.total_rows - self.accepted

    @property
    def ok(self) -> bool:
        return bool(self.accepted) and not self.errors

    @property
    def reported_errors(self) -> list[RowError]:
        return self.errors[:MAX_REPORTED_ERRORS]

    def error_lines(self, limit: int = MAX_REPORTED_ERRORS) -> list[str]:
        return [str(error) for error in self.errors[:limit]]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _payload_size(source) -> int:
    """The payload's size in bytes, measured without decoding or parsing it.

    A ``str`` payload is measured in characters, which is a lower bound on its
    UTF-8 byte length: one character never encodes to less than one byte.
    """
    if source is None:
        return 0
    if isinstance(source, (bytes, bytearray, memoryview)):
        return len(source)
    return len(str(source))


def _as_text(source) -> str:
    if source is None:
        return ""
    if isinstance(source, (bytes, bytearray)):
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return bytes(source).decode(encoding)
            except UnicodeDecodeError:
                continue
        return ""
    return str(source)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value):
    """A finite float, or None when the value is missing or not a number."""
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fail(message: str, column: str = "file", total_rows: int = 0) -> BatchResult:
    return BatchResult(
        frame=_empty_frame(),
        errors=[RowError(0, column, message)],
        total_rows=total_rows,
    )


def score_batch(source, *, max_rows: int = MAX_ROWS) -> BatchResult:
    """Validate and score an uploaded CSV held in memory."""
    size = _payload_size(source)
    if size > MAX_UPLOAD_BYTES:
        return _fail(
            f"the uploaded file is {size:,} bytes; at most "
            f"{MAX_UPLOAD_BYTES:,} bytes are accepted"
        )

    text = _as_text(source).strip()
    if not text:
        return _fail("the uploaded file is empty")

    try:
        # One row past the cap is enough to know the file is too long, so the
        # reader never materialises more than the cap allows.
        raw = pd.read_csv(io.StringIO(text), nrows=max_rows + 1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        return _fail(f"the file could not be read as CSV ({error})")

    raw.columns = [str(column).strip() for column in raw.columns]
    missing = [name for name in REQUIRED_COLUMNS if name not in raw.columns]
    if missing:
        return _fail(
            "missing required column(s): "
            f"{', '.join(missing)}. Required: pl_rade and pl_insol; "
            "optional: st_teff and name.",
            column="header",
            total_rows=len(raw),
        )

    if len(raw) > max_rows:
        # The frame stops at the cap, so count the rows in the text instead. A
        # quoted newline can make this an overcount; the file is refused either
        # way, and the arithmetic is only used for the message.
        total_rows = max(len(raw), text.count("\n"))
        return _fail(
            f"{total_rows:,} data rows; at most {max_rows:,} rows are accepted",
            total_rows=total_rows,
        )
    if len(raw) == 0:
        return _fail("the file has a header but no data rows", column="header")

    has_teff = "st_teff" in raw.columns
    has_name = "name" in raw.columns

    accepted: list[dict] = []
    errors: list[RowError] = []

    for position, record in raw.iterrows():
        line = int(position) + 2  # the header is line 1
        row_errors: list[RowError] = []

        radius = _number(record["pl_rade"])
        if radius is None:
            row_errors.append(
                RowError(
                    line,
                    "pl_rade",
                    "radius is missing"
                    if _is_missing(record["pl_rade"])
                    else f"radius {record['pl_rade']!r} is not a number",
                )
            )
        elif radius <= 0:
            row_errors.append(
                RowError(line, "pl_rade", f"radius must be positive, got {radius:g}")
            )

        flux = _number(record["pl_insol"])
        if flux is None:
            row_errors.append(
                RowError(
                    line,
                    "pl_insol",
                    "flux is missing"
                    if _is_missing(record["pl_insol"])
                    else f"flux {record['pl_insol']!r} is not a number",
                )
            )
        elif flux < 0:
            row_errors.append(
                RowError(line, "pl_insol", f"flux must not be negative, got {flux:g}")
            )

        teff = None
        if has_teff:
            teff = _number(record["st_teff"])
            if teff is None and not _is_missing(record["st_teff"]):
                row_errors.append(
                    RowError(
                        line,
                        "st_teff",
                        f"stellar temperature {record['st_teff']!r} is not a number",
                    )
                )

        if row_errors:
            errors.extend(row_errors)
            continue

        accepted.append(
            {
                "name": str(record["name"]).strip() if has_name else f"row {line}",
                "pl_rade": radius,
                "pl_insol": flux,
                "st_teff": teff if teff is not None else np.nan,
                "base_score": calculate_esi(radius, flux),
                "hz_class": classify_hz(flux, teff if teff is not None else np.nan),
                "size_regime": determine_size_regime(radius),
                "disclaimer": DISCLAIMER,
            }
        )

    frame = pd.DataFrame(accepted, columns=RESULT_COLUMNS)
    return BatchResult(frame=frame, errors=errors, total_rows=len(raw))


def export_frame(result: BatchResult) -> pd.DataFrame:
    """The scored rows as offered for download, disclaimer included."""
    frame = result.frame.copy()
    if "disclaimer" not in frame.columns:
        frame["disclaimer"] = DISCLAIMER
    return frame[RESULT_COLUMNS]
