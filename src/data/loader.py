"""
loader.py — Phase 1, Prompt 1.1
Loads and merges D9 CIBIL + Internal Bank on PROSPECTID.
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RAW_CIBIL_FILE, RAW_INTERNAL_FILE, RAW_UNSEEN_FILE,
    CIBIL_SHEET, INTERNAL_SHEET, UNSEEN_SHEET,
)

def load_raw() -> pd.DataFrame:
    """
    Load D9 CIBIL + Internal Bank, verify join key, merge inner join.
    Returns merged DataFrame. Raises on any join anomaly.
    """
    print("── Loading D9 CIBIL ────────────────────────────────────────")
    cibil = pd.read_excel(RAW_CIBIL_FILE, sheet_name=CIBIL_SHEET)
    print(f"  Shape:   {cibil.shape[0]:,} rows × {cibil.shape[1]} cols")
    print(f"  Nulls:   {cibil.isnull().sum().sum()} total")
    print(f"  PROSPECTID: min={cibil['PROSPECTID'].min()}  max={cibil['PROSPECTID'].max()}")

    print("\n── Loading D9 Internal Bank ────────────────────────────────")
    internal = pd.read_excel(RAW_INTERNAL_FILE, sheet_name=INTERNAL_SHEET)
    print(f"  Shape:   {internal.shape[0]:,} rows × {internal.shape[1]} cols")
    print(f"  Nulls:   {internal.isnull().sum().sum()} total")
    print(f"  PROSPECTID: min={internal['PROSPECTID'].min()}  max={internal['PROSPECTID'].max()}")

    print("\n── Join key validation ─────────────────────────────────────")
    cibil_ids    = set(cibil['PROSPECTID'])
    internal_ids = set(internal['PROSPECTID'])
    only_cibil   = cibil_ids - internal_ids
    only_internal= internal_ids - cibil_ids
    print(f"  IDs only in CIBIL:    {len(only_cibil)}")
    print(f"  IDs only in Internal: {len(only_internal)}")
    print(f"  CIBIL duplicates:     {cibil['PROSPECTID'].duplicated().sum()}")
    print(f"  Internal duplicates:  {internal['PROSPECTID'].duplicated().sum()}")
    print(f"  CIBIL nulls in ID:    {cibil['PROSPECTID'].isnull().sum()}")
    print(f"  Internal nulls in ID: {internal['PROSPECTID'].isnull().sum()}")

    if only_cibil or only_internal:
        raise ValueError(
            f"Join key mismatch: {len(only_cibil)} IDs only in CIBIL, "
            f"{len(only_internal)} only in Internal"
        )

    print("\n── Merging (inner join on PROSPECTID) ──────────────────────")
    merged = pd.merge(cibil, internal, on='PROSPECTID', how='inner')
    rows_lost = len(cibil) - len(merged)
    print(f"  CIBIL rows:    {len(cibil):,}")
    print(f"  Merged rows:   {len(merged):,}")
    print(f"  Rows lost:     {rows_lost}  ← must be 0")
    print(f"  Merged cols:   {merged.shape[1]}")

    if rows_lost != 0:
        raise ValueError(f"Inner join lost {rows_lost} rows — investigate PROSPECTID mismatch")

    print(f"\n  ✅ Perfect join — zero rows lost")
    return merged


def load_unseen() -> pd.DataFrame:
    """Load the 100-row holdout scoring set."""
    df = pd.read_excel(RAW_UNSEEN_FILE, sheet_name=UNSEEN_SHEET)
    print(f"Unseen dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(f"\nFinal merged shape: {df.shape}")
    print(f"Target distribution:\n{df['Approved_Flag'].value_counts()}")
