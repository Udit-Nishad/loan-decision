"""
ensemble.py — Phase 2, Prompt 2.2
Three-layer GBM ensemble:
  Layer 1: RF(200) + ExtraTrees(200)  — base learners
  Layer 2: GradientBoosting(200)      — meta-learner on val predictions
  Layer 3: WoE Scorecard              — 10% weight fallback
Final blend: GBM 70% + ExtraTrees 20% + Scorecard 10%
"""
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    TARGET_COL, TARGET_IMAP, MODELS_DIR,
    ENSEMBLE_MODEL_FILE, SCORECARD_FILE, TRANSFORMER_FILE,
)


class LoanEnsemble:
    """
    Three-layer ensemble for loan approval decisions.
    Weights: GBM 0.70, ExtraTrees 0.20, Scorecard 0.10
    """
    WEIGHTS = {"gbm": 0.70, "et": 0.20, "scorecard": 0.10}
    CLASSES  = [0, 1, 2, 3]   # P1, P2, P3, P4

    def __init__(self):
        self.rf          = None
        self.et          = None
        self.meta        = None   # GradientBoosting meta-learner
        self.scorecard   = None
        self.feature_cols = None
        self.sc_features  = None
        self.is_fitted    = False

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val, y_val,
            X_sc=None, y_sc=None):
        """
        Two-stage training:
          Stage 1 — base learners on (X_train, y_train)
          Stage 2 — meta-learner on val predictions
        """
        from sklearn.ensemble import (
            RandomForestClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
        )

        self.feature_cols = list(X_train.columns)

        print("  Training RF base learner (200 trees)…")
        self.rf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42)
        self.rf.fit(X_train, y_train)

        print("  Training ExtraTrees base learner (200 trees)…")
        self.et = ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=42)
        self.et.fit(X_train, y_train)

        print("  Building meta-features from val set…")
        rf_val = self.rf.predict_proba(X_val)   # (n, 4)
        et_val = self.et.predict_proba(X_val)   # (n, 4)
        meta_X = np.hstack([rf_val, et_val])    # (n, 8)

        print("  Training GBM meta-learner (200 estimators)…")
        self.meta = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42)
        self.meta.fit(meta_X, y_val)

        self.is_fitted = True
        print("  ✅ Ensemble fitted")

    def _meta_features(self, X):
        rf_p = self.rf.predict_proba(X)
        et_p = self.et.predict_proba(X)
        return np.hstack([rf_p, et_p])

    def predict_proba(self, X):
        """Returns (n, 4) probability array. Blends all layers."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        X_arr = X[self.feature_cols] if hasattr(X, 'columns') else X

        # GBM meta probability
        meta_X   = self._meta_features(X_arr)
        gbm_prob = self.meta.predict_proba(meta_X)   # (n, 4)

        # ExtraTrees direct probability
        et_prob  = self.et.predict_proba(X_arr)      # (n, 4)

        # Scorecard probability (if fitted)
        if self.scorecard is not None and self.sc_features is not None:
            try:
                X_sc = X_arr[self.sc_features] if hasattr(X_arr, 'columns') else X_arr
                sc_prob = self._scorecard_proba(X_sc)
                blended = (self.WEIGHTS["gbm"]       * gbm_prob
                         + self.WEIGHTS["et"]        * et_prob
                         + self.WEIGHTS["scorecard"] * sc_prob)
            except Exception:
                blended = (0.70 * gbm_prob + 0.30 * et_prob)
        else:
            blended = (0.70 * gbm_prob + 0.30 * et_prob)

        # Normalise rows to sum to 1
        blended = blended / blended.sum(axis=1, keepdims=True)
        return blended

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _scorecard_proba(self, X_sc):
        """Wrapper around loaded scorecard logistic regression."""
        if hasattr(self.scorecard, 'predict_proba'):
            p = self.scorecard.predict_proba(X_sc)
            if p.shape[1] == 4:
                return p
        # Fallback — uniform
        return np.ones((len(X_sc), 4)) / 4

    def attach_scorecard(self, scorecard_model, sc_features):
        self.scorecard  = scorecard_model
        self.sc_features = sc_features

    def save(self, path=None):
        path = path or ENSEMBLE_MODEL_FILE
        joblib.dump(self, path)
        print(f"  Saved ensemble → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or ENSEMBLE_MODEL_FILE
        return joblib.load(path)
