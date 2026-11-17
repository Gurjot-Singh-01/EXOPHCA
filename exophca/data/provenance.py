"""EXOPHCA provenance generation module."""

import hashlib
import json
from pathlib import Path
import pandas as pd


def parse_header_timestamp(file_path: Path) -> str:
    """Extract the second header line timestamp from NASA Exoplanet Archive CSV files."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(5)]
        if len(lines) >= 2:
            second_line = lines[1].strip()
            # Usually starts with '# ' followed by timestamp string like 'Sat Jul 11 08:35:38 2026'
            if second_line.startswith('#'):
                return second_line.lstrip('#').strip()
            return second_line
    except Exception:
        pass
    return "Unknown"


def generate_provenance_json(file_path: Path, output_json_path: Path, source_name_desc: str) -> dict:
    """Generate provenance JSON metadata for a raw data file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    byte_length = path.stat().st_size

    # Compute sha256
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # Read dataframe to get row and column count
    df = pd.read_csv(path, comment='#', low_memory=False)
    row_count, col_count = df.shape

    timestamp = parse_header_timestamp(path)

    provenance_data = {
        "source_filename": source_name_desc,
        "original_file": path.name,
        "sha256": file_hash,
        "byte_length": byte_length,
        "row_count": row_count,
        "column_count": col_count,
        "retrieval_timestamp": timestamp
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as jf:
        json.dump(provenance_data, jf, indent=4)

    return provenance_data
