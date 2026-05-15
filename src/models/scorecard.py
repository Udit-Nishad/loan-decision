"""
scorecard.py — Weight of Evidence (WoE) Scorecard Model
Regulatory baseline model. RBI IRB-compliant.
Binary target: P1 (Reject) vs rest.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import roc_curve
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (
    FEATURED_FILE, MODELS_DIR, RANDOM_STATE as RANDOM_SEED,
    TARGET_MAP as TARGET_ENCODING, TARGET_IMAP as TARGET_DECODING,
    FEATURE_PLAIN_ENGLISH,
)
from src.utils.logger import get_logger

log = get_logger("ScorecardModel")

SCORECARD_FEATURES = [
    "Credit_Score", "num_times_delinquent", "NETMONTHLYINCOME",
    "enq_L6m", "Time_With_Curr_Empr", "num_deliq_12mts",
    "max_delinquency_level", "pct_of_active_TLs_ever", "num_std",
    "recent_stress", "emp_stability", "income_deliq_ratio",
    "credit_maturity", "enq_velocity", "num_lss", "num_dbt",
]


class WoEBinner:
    """Custom Weight of Evidence binner — no external library needed."""

    def __init__(self, n_bins=10):
        self.n_bins   = n_bins
        self.bins_    = None
        self.woe_map_ = {}
        self.iv_      = 0.0
        self.woe_table_ = None

    def fit(self, x, y_binary):
        quantiles  = np.linspace(0, 1, self.n_bins + 1)
        bin_edges  = x.quantile(quantiles).unique()
        bin_edges[0]  = -np.inf
        bin_edges[-1] =  np.inf
        self.bins_ = sorted(set(bin_edges))

        labels = pd.cut(x, bins=self.bins_, labels=False, include_lowest=True)
        df_tmp = pd.DataFrame({"bin": labels, "y": y_binary})

        total_ev  = max(y_binary.sum(), 1)
        total_nev = max((1 - y_binary).sum(), 1)
        records   = []
        iv_total  = 0.0

        for b in sorted(df_tmp["bin"].dropna().unique()):
            mask   = df_tmp["bin"] == b
            n      = mask.sum()
            ev     = df_tmp.loc[mask, "y"].sum()
            nev    = n - ev
            pev    = max(ev,  0.5) / total_ev
            pnev   = max(nev, 0.5) / total_nev
            woe    = np.log(pev / pnev)
            iv     = (pev - pnev) * woe
            self.woe_map_[b] = woe
            iv_total += iv
            lo = self.bins_[int(b)]
            hi = self.bins_[int(b) + 1]
            label = f"({lo:.1f},{hi:.1f}]" if lo != -np.inf else f"≤{hi:.1f}"
            records.append({"bin": int(b), "range": label, "n_total": int(n),
                            "n_events": int(ev), "n_non_events": int(nev),
                            "woe": round(woe, 4), "iv": round(iv, 4)})

        self.iv_        = round(iv_total, 4)
        self.woe_table_ = pd.DataFrame(records)
        return self

    def transform(self, x):
        idx = pd.cut(x, bins=self.bins_, labels=False, include_lowest=True)
        return idx.map(self.woe_map_).fillna(0.0)

    def fit_transform(self, x, y_binary):
        return self.fit(x, y_binary).transform(x)


class ScorecardModel:
    """WoE Scorecard + Logistic Regression. Outputs 300–900 CIBIL-style score."""

    def __init__(self, n_bins=10):
        self.n_bins        = n_bins
        self.binners_      = {}
        self.lr_model_     = None
        self.feature_cols_ = None
        self.iv_report_    = {}
        self.score_offset_ = 600
        self.score_factor_ = 50
        self._is_fitted    = False

    def _binary(self, y):
        return (y == TARGET_ENCODING["P1"]).astype(int)

    def fit(self, X_train, y_train):
        log.info("=" * 55)
        log.info("TRAINING SCORECARD MODEL")
        log.info("=" * 55)
        y_bin = self._binary(y_train)
        log.info(f"Event rate (P1): {y_bin.mean()*100:.1f}%  ({y_bin.sum():,} events)")

        self.feature_cols_ = [f for f in SCORECARD_FEATURES if f in X_train.columns]
        X_woe = pd.DataFrame(index=X_train.index)
        for feat in self.feature_cols_:
            b = WoEBinner(self.n_bins)
            X_woe[feat] = b.fit_transform(X_train[feat], y_bin)
            self.binners_[feat]   = b
            self.iv_report_[feat] = b.iv_

        iv_sorted = sorted(self.iv_report_.items(), key=lambda x: -x[1])
        log.info("Information Value (IV):")
        for feat, iv in iv_sorted:
            s = ("STRONG" if iv > 0.3 else "MEDIUM" if iv > 0.1
                 else "WEAK" if iv > 0.02 else "USELESS")
            log.info(f"  {FEATURE_PLAIN_ENGLISH.get(feat,feat):<45} IV={iv:.4f} [{s}]")

        self.lr_model_ = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=1.0, random_state=RANDOM_SEED, solver="lbfgs")
        self.lr_model_.fit(X_woe, y_bin)
        self._is_fitted = True

        auc_tr = roc_auc_score(y_bin, self.lr_model_.predict_proba(X_woe)[:, 1])
        log.info(f"Train Gini: {2*auc_tr-1:.4f}")
        return self

    def _woe_transform(self, X):
        X_woe = pd.DataFrame(index=X.index)
        for feat in self.feature_cols_:
            X_woe[feat] = self.binners_[feat].transform(X[feat])
        return X_woe

    def predict_proba(self, X):
        return self.lr_model_.predict_proba(self._woe_transform(X))

    def predict_score(self, X):
        pd_p = self.predict_proba(X)[:, 1]
        s = self.score_offset_ + self.score_factor_ * np.log(
            (1 - pd_p + 1e-9) / (pd_p + 1e-9))
        return np.clip(s, 300, 900).round(0).astype(int)

    def evaluate(self, X, y, name="Val"):
        y_bin = self._binary(y)
        prob  = self.predict_proba(X)[:, 1]
        auc_v = roc_auc_score(y_bin, prob)
        gini  = 2 * auc_v - 1

        df_ks = pd.DataFrame({"y": y_bin, "p": prob}).sort_values("p", ascending=False)
        df_ks["ce"] = df_ks["y"].cumsum()     / max(y_bin.sum(), 1)
        df_ks["cn"] = (1-df_ks["y"]).cumsum() / max((1-y_bin).sum(), 1)
        ks = (df_ks["ce"] - df_ks["cn"]).abs().max()

        prec, rec, _ = precision_recall_curve(y_bin, prob)
        auprc = auc(rec, prec)

        metrics = {"gini": round(gini,4), "ks": round(ks,4),
                   "auprc": round(auprc,4), "auc": round(auc_v,4)}
        log.info(f"[{name}] Gini={gini:.4f} | KS={ks:.4f} | AUPRC={auprc:.4f}")
        return metrics

    def get_scorecard_table(self):
        coefs = dict(zip(self.feature_cols_, self.lr_model_.coef_[0]))
        rows  = []
        for feat in self.feature_cols_:
            for _, r in self.binners_[feat].woe_table_.iterrows():
                rows.append({
                    "Feature":     FEATURE_PLAIN_ENGLISH.get(feat, feat),
                    "Bin Range":   r["range"],
                    "N (Train)":   r["n_total"],
                    "Event Rate%": round(r["n_events"]/max(r["n_total"],1)*100,1),
                    "WoE":         r["woe"],
                    "LR Coeff":    round(coefs.get(feat,0), 4),
                    "Points":      round(r["woe"] * coefs.get(feat,0) * self.score_factor_, 1),
                    "IV":          round(self.binners_[feat].iv_, 4),
                })
        return pd.DataFrame(rows)

    def save(self, path=None):
        p = Path(path) if path else MODELS_DIR / "scorecard.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        log.info(f"Scorecard saved → {p}")
        return str(p)

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else MODELS_DIR / "scorecard.joblib"
        return joblib.load(p)
