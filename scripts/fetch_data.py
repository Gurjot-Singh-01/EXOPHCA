"""Download the NASA Exoplanet Archive tables the data pipeline reads.

``scripts/build_dataset.py`` expects two raw exports under ``data/``:

    data/PSCD.csv      the ``pscomppars`` table (confirmed planets)
    data/TOI_2026.csv  the ``toi`` table (TESS Objects of Interest)

Neither is tracked by git. Both come from the archive's TAP service, so this
script asks the archive for the full table and writes the result under exactly
those names.

A file that already exists is left untouched unless ``--force`` is passed: the
raw exports behind the reported results must not be replaced by accident.

A fresh download is the archive as it stands today. The numbers in
``docs/`` and README were produced from exports taken in July 2026, so row
counts, hashes and every derived figure may differ after a re-download.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --table toi
    python scripts/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# name -> (destination file, TAP query). The queries ask for whole tables, the
# same way the archive's own "Download Table" button does.
TABLES = {
    "pscomppars": {
        "destination": "PSCD.csv",
        "query": "select * from pscomppars",
        "label": "NASA Exoplanet Archive pscomppars (confirmed planets)",
    },
    "toi": {
        "destination": "TOI_2026.csv",
        "query": "select * from toi",
        "label": "NASA Exoplanet Archive toi (TESS Objects of Interest)",
    },
}

# A download is only accepted when the columns the pipeline validates are
# present; otherwise the archive returned an error page or a partial table.
REQUIRED_COLUMNS = {
    "pscomppars": [
        "pl_name", "hostname", "discoverymethod", "disc_year",
        "pl_rade", "pl_insol", "pl_eqt", "pl_orbper", "pl_orbsmax", "pl_bmasse",
        "st_teff", "st_rad", "st_mass", "sy_dist",
    ],
    "toi": [
        "toi", "tid", "tfopwg_disp", "pl_orbper", "pl_rade", "pl_insol", "pl_eqt",
        "st_teff", "st_rad", "st_dist",
    ],
}


def request_url(query: str) -> str:
    """The TAP synchronous query URL that returns the table as CSV."""
    return f"{TAP_SYNC}?query={urllib.parse.quote(query)}&format=csv"


def banner_lines(downloaded_at: datetime) -> list[str]:
    """The two ``#`` lines written above the header row.

    The archive's web exports carry the same two lines, and
    ``exophca.data.provenance.parse_header_timestamp`` reads the retrieval
    timestamp from the second one. TAP ``format=csv`` omits them, so without
    this the provenance JSON would record the first data row as the date.
    """
    return [
        "# Downloaded from the NASA Exoplanet Archive TAP service by scripts/fetch_data.py",
        f"# {downloaded_at.strftime('%a %b %d %H:%M:%S %Y')}",
    ]


def download(url: str, destination: Path, timeout: float, header: list[str]) -> str:
    """Stream ``url`` to ``destination`` and return the sha256 of what was written."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "EXOPHCA fetch_data"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with open(destination, "wb") as handle:
            for line in header:
                encoded = (line + "\n").encode("utf-8")
                digest.update(encoded)
                handle.write(encoded)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def archive_comment_lines(path: Path) -> list[str]:
    """The leading ``#`` lines the archive puts above the CSV header."""
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    return lines


def describe(path: Path, required: list[str]) -> dict:
    """Row/column counts and the missing required columns, if any."""
    import pandas as pd

    frame = pd.read_csv(path, comment="#", low_memory=False)
    missing = [column for column in required if column not in frame.columns]
    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "missing_required_columns": missing,
    }


def fetch_one(table: str, data_dir: Path, force: bool, timeout: float) -> dict | None:
    """Download one table. Returns its provenance record, or None when skipped."""
    spec = TABLES[table]
    destination = data_dir / spec["destination"]
    url = request_url(spec["query"])

    if destination.exists() and not force:
        print(f"{destination.name}: already present, left untouched (pass --force to replace)")
        return None

    partial = destination.with_name(destination.name + ".part")
    print(f"{destination.name}: querying {spec['query']!r}")
    retrieved_at = datetime.now()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        sha256 = download(url, partial, timeout, banner_lines(retrieved_at))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{table}: download failed: {error}") from error

    try:
        facts = describe(partial, REQUIRED_COLUMNS[table])
    except Exception as error:  # noqa: BLE001 - any parse failure means a bad file
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{table}: downloaded data could not be read as CSV: {error}") from error

    if facts["missing_required_columns"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{table}: downloaded table is missing required columns "
            f"{facts['missing_required_columns']}; nothing was written"
        )

    partial.replace(destination)
    print(
        f"{destination.name}: {facts['row_count']} rows, "
        f"{facts['column_count']} columns, sha256 {sha256[:16]}…"
    )

    return {
        "table": table,
        "source": spec["label"],
        "query": spec["query"],
        "request_url": url,
        "file": str(destination.relative_to(data_dir.parent)).replace("\\", "/"),
        "downloaded_at_utc": started,
        "sha256": sha256,
        "byte_length": destination.stat().st_size,
        "row_count": facts["row_count"],
        "column_count": facts["column_count"],
        "archive_header": archive_comment_lines(destination)[:2],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--table",
        choices=sorted(TABLES),
        action="append",
        help="download only this table (repeatable); default is both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing data file instead of leaving it untouched",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="network timeout in seconds per request (default: 180)",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = args.table or sorted(TABLES)
    records: list[dict] = []
    failures: list[str] = []

    print(f"Data directory: {data_dir}")
    for table in selected:
        try:
            record = fetch_one(table, data_dir, args.force, args.timeout)
        except RuntimeError as error:
            print(f"FAILED {error}", file=sys.stderr)
            failures.append(str(error))
            continue
        if record is not None:
            records.append(record)

    if records:
        provenance_path = data_dir / "fetch_provenance.json"
        document = {
            "service": "NASA Exoplanet Archive TAP (sync)",
            "endpoint": TAP_SYNC,
            "note": (
                "The reported results were produced from exports taken in July 2026. "
                "A fresh download is the archive as it stands on the date recorded "
                "here, so row counts, hashes and every derived figure may differ."
            ),
            "downloads": records,
        }
        existing: list[dict] = []
        if provenance_path.is_file():
            try:
                existing = json.loads(provenance_path.read_text(encoding="utf-8")).get(
                    "downloads", []
                )
            except (OSError, ValueError):
                existing = []
        replaced = {record["table"] for record in records}
        kept = [record for record in existing if record.get("table") not in replaced]
        document["downloads"] = kept + records
        provenance_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"Provenance written to {provenance_path}")

    if failures:
        print(f"{len(failures)} table(s) failed; nothing was written for them", file=sys.stderr)
        return 1

    print("Done. Next: python scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
