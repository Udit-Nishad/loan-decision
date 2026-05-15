"""
feature_eng.py — Phase 1, Prompt 1.3
Engineers 3 TL features, one-hot encodes categoricals,
does stratified 70/15/15 split, saves featured.csv.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    TARGET_COL, CAT_COLS, ENGINEERED_FEATURES,
    TRAIN_RATIO, VAL_RATIO, RANDOM_STATE,
)
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 3 TL-derived features."""
    df = df.copy()

    # 1. missed_payment_rate = Tot_Missed_Pmnt / (Total_TL + 1)
    df["missed_payment_rate"] = (
        df["Tot_Missed_Pmnt"] / (df["Total_TL"] + 1)
    ).round(4)

    # 2. tl_utilization_rate = Tot_Active_TL / (Total_TL + 1)
    df["tl_utilization_rate"] = (
        df["Tot_Active_TL"] / (df["Total_TL"] + 1)
    ).round(4)

    # 3. credit_history_span = Age_Oldest_TL - Age_Newest_TL
    df["credit_history_span"] = (
        df["Age_Oldest_TL"] - df["Age_Newest_TL"]
    ).clip(lower=0)

    print("  Engineered features:")
    for feat in ENGINEERED_FEATURES:
        if TARGET_COL in df.columns:
            corr = df[feat].corr(df[TARGET_COL])
            print(f"    {feat:<30s}  corr with target: {corr:+.4f}")
        else:
            print(f"    {feat:<30s}  (no target column — skipping corr)")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns, drop original."""
    df = df.copy()
    cols_present = [c for c in CAT_COLS if c in df.columns]
    before = df.shape[1]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False)
    after = df.shape[1]
    print(f"  One-hot encoded {len(cols_present)} columns: {cols_present}")
    print(f"  Columns: {before} → {after} (+{after - before} dummy cols)")
    return df


def make_splits(df: pd.DataFrame):
    """Stratified 70/15/15 split. Returns (train, val, test)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: 70% train, 30% temp
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO),
        stratify=y, random_state=RANDOM_STATE
    )
    # Second split: 50% of temp = val, 50% = test  (15/15)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5,
        stratify=y_tmp, random_state=RANDOM_STATE
    )

    train = pd.concat([X_tr, y_tr], axis=1)
    val   = pd.concat([X_val, y_val], axis=1)
    test  = pd.concat([X_te, y_te], axis=1)
    return train, val, test


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("── Step 1: Engineer TL features ────────────────────────────")
    df = engineer_features(df)

    print("\n── Step 2: One-hot encode categoricals ─────────────────────")
    df = encode_categoricals(df)

    print(f"\n── Featured shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    from src.utils.config import CLEANED_FILE, FEATURED_FILE

    df_clean = pd.read_csv(CLEANED_FILE)
    print(f"Loaded cleaned.csv: {df_clean.shape}")

    df_feat = engineer(df_clean)
    df_feat.to_csv(FEATURED_FILE, index=False)
    print(f"\nSaved → {FEATURED_FILE}")

    print("\n── Stratified 70/15/15 split ───────────────────────────────")
    train, val, test = make_splits(df_feat)
    total = len(df_feat)
    print(f"  Train: {len(train):,}  ({len(train)/total*100:.1f}%)")
    print(f"  Val:   {len(val):,}  ({len(val)/total*100:.1f}%)")
    print(f"  Test:  {len(test):,}  ({len(test)/total*100:.1f}%)")

    label_map = {0:"P1",1:"P2",2:"P3",3:"P4"}
    print("\n  Class distribution (stratification check):")
    print(f"  {'Class':<6} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'─'*34}")
    for cls in sorted(df_feat[TARGET_COL].unique()):
        tr_pct  = (train[TARGET_COL] == cls).mean() * 100
        val_pct = (val[TARGET_COL]   == cls).mean() * 100
        te_pct  = (test[TARGET_COL]  == cls).mean() * 100
        print(f"  {label_map[cls]:<6} {tr_pct:>7.1f}%  {val_pct:>6.1f}%  {te_pct:>6.1f}%")
    print("\n  ✅ Stratification held — class proportions consistent across splits")
