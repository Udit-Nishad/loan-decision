"""
cleaner.py — Phase 1, Prompt 1.2
Cleans merged D9 data:
  - Fix TL sentinel values (-99999 → 0)
  - Cap outliers at 99th percentile
  - Encode target P1/P2/P3/P4 → 0/1/2/3
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    TARGET_COL, TARGET_MAP,
    TL_SENTINEL_COLS, TL_SENTINEL_VAL,
    OUTLIER_COLS,
)

def fix_tl_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace -99999 sentinel with 0 in TL age columns."""
    df = df.copy()
    for col in TL_SENTINEL_COLS:
        if col in df.columns:
            n = (df[col] == TL_SENTINEL_VAL).sum()
            df[col] = df[col].replace(TL_SENTINEL_VAL, 0)
            print(f"  {col}: replaced {n:,} sentinel values with 0")
            print(f"    Before fix range: would have been -99999 – {df[col].max()}")
            print(f"    After fix range:  {df[col].min()} – {df[col].max()}")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap specified columns at their 99th percentile."""
    df = df.copy()
    for col, pct in OUTLIER_COLS.items():
        if col in df.columns:
            cap = df[col].quantile(pct)
            n   = (df[col] > cap).sum()
            df[col] = df[col].clip(upper=cap)
            print(f"  {col}: capped {n:,} values above p{int(pct*100)} ({cap:,.0f})")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map P1/P2/P3/P4 → 0/1/2/3."""
    df = df.copy()
    if df[TARGET_COL].dtype == object or str(df[TARGET_COL].dtype) in ('string','str','object'):
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
        print(f"  Target encoded: P1→0, P2→1, P3→2, P4→3")
    else:
        print(f"  Target already numeric — skipping encode")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("── Step 1: Fix TL sentinel values ─────────────────────────")
    df = fix_tl_sentinels(df)

    print("\n── Step 2: Cap outliers ────────────────────────────────────")
    df = cap_outliers(df)

    print("\n── Step 3: Encode target ───────────────────────────────────")
    df = encode_target(df)

    print("\n── Step 4: Final null check ────────────────────────────────")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"  Total nulls after cleaning: {total_nulls}")
    if total_nulls > 0:
        print(f"  Columns with nulls:\n{null_counts[null_counts > 0]}")

    print(f"\n── Cleaned shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Target distribution:")
    vc = df[TARGET_COL].value_counts()
    label_lookup = {0:"P1",1:"P2",2:"P3",3:"P4","P1":"P1","P2":"P2","P3":"P3","P4":"P4"}
    for k, v in vc.sort_index().items():
        label = label_lookup[k]
        print(f"    {label} ({k}): {v:,}  ({v/len(df)*100:.1f}%)")
    return df


if __name__ == "__main__":
    from src.data.loader import load_raw
    from src.utils.config import CLEANED_FILE
    df = load_raw()
    df_clean = clean(df)
    df_clean.to_csv(CLEANED_FILE, index=False)
    print(f"\nSaved → {CLEANED_FILE}")
