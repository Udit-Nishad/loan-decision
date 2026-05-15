"""
default_model.py — Phase 3, Prompt 3.2
Proxy default model: predicts Tot_Missed_Pmnt > 0 (binary).
Trains ONLY on CIBIL features — NOT TL columns (avoids circularity).
Runs only for returning customers (Total_TL > 0).
"""
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import MODELS_DIR, RANDOM_STATE

DEFAULT_MODEL_FILE = MODELS_DIR / "default_model.joblib"

# CIBIL-only features — deliberately excludes all TL columns
# (TL columns generate Tot_Missed_Pmnt — using them as inputs is circular)
CIBIL_FEATURES = [
    'Credit_Score',
    'num_std', 'num_sub', 'num_dbt', 'num_lss',
    'num_times_delinquent',
    'max_recent_level_of_deliq', 'max_delinquency_level',
    'num_times_60p_dpd',
    'pct_of_active_TLs_ever',
    'time_since_recent_payment',
    'enq_L3m', 'enq_L6m', 'enq_L12m',
    'NETMONTHLYINCOME', 'AGE',
]
# One-hot derived from CIBIL file
CIBIL_CAT_FEATURES_PREFIX = [
    'GENDER_', 'MARITALSTATUS_', 'EDUCATION_',
]


def get_cibil_cols(all_cols):
    """Return all CIBIL feature columns available in df."""
    exact   = [c for c in CIBIL_FEATURES if c in all_cols]
    cat_ohe = [c for c in all_cols
               if any(c.startswith(p) for p in CIBIL_CAT_FEATURES_PREFIX)]
    return exact + cat_ohe


def train_default_model(df: pd.DataFrame, verbose=True):
    """
    Train proxy default model.
    Returns (model, feature_list, eval_dict).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score,
                                  precision_recall_curve, f1_score)

    # Binary target
    y = (df['Tot_Missed_Pmnt'] > 0).astype(int)
    features = get_cibil_cols(df.columns.tolist())
    X = df[features]

    if verbose:
        print(f'  Features used: {len(features)} CIBIL-only cols')
        print(f'  Target: Tot_Missed_Pmnt > 0')
        print(f'  Positive (any missed): {y.mean()*100:.1f}%  ({y.sum():,})')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    if verbose:
        print(f'\n  Training GBM (200 trees, depth 4, lr 0.05)…')

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred  = clf.predict(X_te)
    auc   = roc_auc_score(y_te, proba)
    f1    = f1_score(y_te, pred)

    if verbose:
        print(f'\n  ── Test set performance ─────────────────────────────')
        print(f'  ROC-AUC:  {auc:.4f}')
        print(f'  F1 score: {f1:.4f}  (default_proxy = positive class)')

        fi = pd.Series(clf.feature_importances_, index=features)
        fi = fi.sort_values(ascending=False)
        print(f'\n  Top 5 features:')
        for feat, imp in fi.head(5).items():
            bar = '█' * int(imp * 200)
            print(f'    {feat:<45s} {imp:.4f}  {bar}')

    return clf, features, {'roc_auc': auc, 'f1': f1}


if __name__ == '__main__':
    from src.utils.config import FEATURED_FILE
    df = pd.read_csv(FEATURED_FILE)
    model, features, metrics = train_default_model(df)
    joblib.dump({'model': model, 'features': features},
                DEFAULT_MODEL_FILE)
    print(f'\nSaved → {DEFAULT_MODEL_FILE}')
