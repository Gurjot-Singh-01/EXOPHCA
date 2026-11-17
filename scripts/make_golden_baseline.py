"""Generate golden baseline CSV for planet base ESI scores (radius-flux formula only, no custom adjustments)."""

from pathlib import Path
import pandas as pd
from exophca.science.esi import calculate_esi

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    ref_dir = data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    planets_path = data_dir / "planets_for_app.csv"
    print(f"Reading planets from {planets_path}")
    df = pd.read_csv(planets_path)

    results = []
    for row in df.itertuples(index=False):
        # calculate_esi computes the base ESI from radius and insolation
        base = calculate_esi(row.pl_rade, row.pl_insol)
        results.append({
            "pl_name": row.pl_name,
            "pl_rade": row.pl_rade,
            "pl_insol": row.pl_insol,
            "base_score": base
        })

    golden_df = pd.DataFrame(results)
    out_path = ref_dir / "golden_base_score.csv"
    golden_df.to_csv(out_path, index=False)
    print(f"Wrote {len(golden_df)} rows to {out_path}")

    # Verify Earth (radius 1.0, insolation 1.0)
    earth_rows = golden_df[golden_df["pl_name"].str.lower() == "earth"]
    if not earth_rows.empty:
        earth_score = earth_rows.iloc[0]["base_score"]
        print(f"Earth base_score verified: {earth_score}")
        assert earth_score == 1.0, f"Earth base score expected 1.0, got {earth_score}"
    else:
        # Check calculation explicitly for (1.0, 1.0)
        s = calculate_esi(1.0, 1.0)
        print(f"Explicit (1.0, 1.0) base_score verified: {s}")
        assert s == 1.0

if __name__ == "__main__":
    main()
