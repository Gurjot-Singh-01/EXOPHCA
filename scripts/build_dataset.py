"""Script to build processed datasets and provenance metadata offline."""

from pathlib import Path
import pandas as pd

from exophca.data.validate import validate_dataframe
from exophca.data.clean import (
    clean_planets_dataset, clean_toi_dataset,
    PLANET_REQUIRED_COLUMNS, TOI_REQUIRED_COLUMNS
)
from exophca.data.provenance import generate_provenance_json


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    pscd_path = data_dir / 'PSCD.csv'
    toi_path = data_dir / 'TOI_2026.csv'

    print("=== Step 0: Loading raw datasets ===")
    df_pscd_raw = pd.read_csv(pscd_path, comment='#', low_memory=False)
    print(f"Loaded PSCD.csv: shape {df_pscd_raw.shape}")

    df_toi_raw = pd.read_csv(toi_path, comment='#', low_memory=False)
    print(f"Loaded TOI_2026.csv: shape {df_toi_raw.shape}")

    print("\n=== Step 1: Validating raw datasets ===")
    pscd_errors = validate_dataframe(df_pscd_raw, PLANET_REQUIRED_COLUMNS, "PSCD.csv")
    for err in pscd_errors:
        print(err)

    toi_errors = validate_dataframe(df_toi_raw, TOI_REQUIRED_COLUMNS, "TOI_2026.csv")
    for err in toi_errors:
        print(err)

    print("\n=== Step 2: Cleaning PSCD dataset -> planets.csv ===")
    df_planets = clean_planets_dataset(df_pscd_raw)
    planets_out = processed_dir / 'planets.csv'
    df_planets.to_csv(planets_out, index=False)
    print(f"Saved {planets_out}: {len(df_planets)} rows, {len(df_planets.columns)} columns.")
    print(f"Scorable count: {df_planets['scorable'].sum()}")
    print("Radius provenance value counts:")
    print(df_planets['radius_provenance'].value_counts())
    print("Mass provenance value counts:")
    print(df_planets['mass_provenance'].value_counts(dropna=False))

    print("\n=== Step 3: Cleaning TOI dataset -> toi.csv ===")
    df_toi = clean_toi_dataset(df_toi_raw)
    toi_out = processed_dir / 'toi.csv'
    df_toi.to_csv(toi_out, index=False)
    print(f"Saved {toi_out}: {len(df_toi)} rows, {len(df_toi.columns)} columns.")
    print(f"Scorable count: {df_toi['scorable'].sum()}")

    print("\n=== Step 4: Generating Provenance JSONs ===")
    pscd_prov = generate_provenance_json(
        pscd_path,
        processed_dir / 'PSCD.provenance.json',
        "pscomppars (file renamed PSCD.csv)"
    )
    print("PSCD provenance generated:", pscd_prov)

    toi_prov = generate_provenance_json(
        toi_path,
        processed_dir / 'TOI_2026.provenance.json',
        "TOI_2026.csv"
    )
    print("TOI provenance generated:", toi_prov)

    print("\nDataset build complete successfully.")


if __name__ == '__main__':
    main()
