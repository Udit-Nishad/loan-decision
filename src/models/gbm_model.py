"""
gbm_model.py — RandomForest + ExtraTrees Stacked Ensemble
Model A — Primary 4-class classifier (60% weight in final ensemble).
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, confusion_matrix,
                              roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import (
    FEATURED_FILE, MODELS_DIR, RANDOM_STATE as RANDOM_SEED,
    TARGET_MAP as TARGET_ENCODING, TARGET_IMAP as TARGET_DECODING,
    FEATURE_PLAIN_ENGLISH,
)
DECISION_MAP = TARGET_ENCODING
from src.utils.logger import get_logger

log = get_logger("GBMEnsemble")
CLASSES     = [0, 1, 2, 3]
CLASS_NAMES = ["P1", "P2", "P3", "P4"]
CLASS_COLORS= {"P1":"#B71C1C","P2":"#2E7D32","P3":"#F57F17","P4":"#E65100"}


class GBMEnsemble:
    def __init__(self):
        self.rf_model_ = None
        self.et_model_ = None
        self.meta_model_ = None
        self.feature_cols_ = None
        self.feature_importances_ = None
        self._is_fitted = False

    def _get_feature_cols(self, X):
        exclude = {"PROSPECTID","Approved_Flag","target_numeric"}
        return [c for c in X.columns if c not in exclude
                and str(X[c].dtype).startswith(("float","int"))]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        log.info("="*55)
        log.info("TRAINING GBM ENSEMBLE (RF + ExtraTrees stacked)")
        log.info("="*55)
        self.feature_cols_ = self._get_feature_cols(X_train)
        X_tr = X_train[self.feature_cols_].values
        y_tr = y_train.values
        n = len(X_tr)
        log.info(f"Samples: {n:,} | Features: {len(self.feature_cols_)}")

        # 5-fold OOF
        log.info("5-fold OOF cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        oof_rf = np.zeros((n, 4))
        oof_et = np.zeros((n, 4))
        fold_f1s = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr), 1):
            rf_f = RandomForestClassifier(n_estimators=100, max_depth=12,
                min_samples_leaf=15, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1)
            et_f = ExtraTreesClassifier(n_estimators=100, max_depth=12,
                min_samples_leaf=15, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1)
            rf_f.fit(X_tr[tr_idx], y_tr[tr_idx])
            et_f.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_rf[val_idx] = rf_f.predict_proba(X_tr[val_idx])
            oof_et[val_idx] = et_f.predict_proba(X_tr[val_idx])
            fp = np.argmax(0.6*oof_rf[val_idx]+0.4*oof_et[val_idx], axis=1)
            ff = f1_score(y_tr[val_idx], fp, average="macro")
            fold_f1s.append(ff)
            log.info(f"  Fold {fold}/5  macro F1: {ff:.4f}")

        log.info(f"CV macro F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")

        # Meta-learner
        log.info("Training meta-learner...")
        meta_X = np.hstack([oof_rf, oof_et])
        self.meta_model_ = LogisticRegression(class_weight="balanced",
            max_iter=500, C=0.5, random_state=RANDOM_SEED,
            solver="lbfgs")
        self.meta_model_.fit(meta_X, y_tr)

        # Retrain on full data
        log.info("Retraining base models on full data...")
        self.rf_model_ = RandomForestClassifier(n_estimators=200, max_depth=14,
            min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1)
        self.et_model_ = ExtraTreesClassifier(n_estimators=200, max_depth=14,
            min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1)
        self.rf_model_.fit(X_tr, y_tr)
        log.info("  RF done")
        self.et_model_.fit(X_tr, y_tr)
        log.info("  ET done")

        self.feature_importances_ = pd.Series(
            self.rf_model_.feature_importances_, index=self.feature_cols_
        ).sort_values(ascending=False)
        self._is_fitted = True
        log.info("GBM Ensemble training complete")

        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val, "Validation")
        return self

    def predict_proba(self, X):
        Xv = X[self.feature_cols_].values
        meta_in = np.hstack([self.rf_model_.predict_proba(Xv),
                             self.et_model_.predict_proba(Xv)])
        return self.meta_model_.predict_proba(meta_in)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y, split_name="Validation"):
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        y_arr = y.values
        macro   = f1_score(y_arr, preds, average="macro")
        per_cls = f1_score(y_arr, preds, average=None, labels=CLASSES)
        y_bin   = label_binarize(y_arr, classes=CLASSES)
        aucs = {CLASS_NAMES[i]: round(roc_auc_score(y_bin[:,i], proba[:,i]),4)
                for i in range(4) if y_bin[:,i].sum()>0}
        metrics = {"macro_f1": round(macro,4),
                   "per_class_f1": {CLASS_NAMES[i]: round(per_cls[i],4) for i in range(4)},
                   "auc_ovr": aucs}
        log.info(f"[{split_name}] Macro F1: {macro:.4f}")
        for cls in CLASS_NAMES:
            log.info(f"  {cls} F1={metrics['per_class_f1'][cls]:.4f}  AUC={aucs.get(cls,'N/A')}")
        return metrics

    def get_feature_importance_table(self, top_n=20):
        top = self.feature_importances_.head(top_n)
        return pd.DataFrame({"Feature": top.index,
            "Plain Name": [FEATURE_PLAIN_ENGLISH.get(f,f) for f in top.index],
            "Importance": top.values.round(4),
            "Importance %": (top.values/top.values.sum()*100).round(1)})

    def plot_results(self, X_test, y_test, save_path=None):
        proba = self.predict_proba(X_test)
        preds = np.argmax(proba, axis=1)
        y_arr = y_test.values
        fig, axes = plt.subplots(1, 3, figsize=(20,6), facecolor="#F8F9FA")
        fig.suptitle("GBM Ensemble — Performance Dashboard",
                     fontsize=16, fontweight="bold", color="#1B3A6B", y=1.02)
        NAVY = "#1B3A6B"

        # Confusion matrix
        ax = axes[0]
        cm = confusion_matrix(y_arr, preds, labels=CLASSES, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    linewidths=0.5, annot_kws={"size":12,"weight":"bold"})
        ax.set_title("Confusion Matrix (Normalised)", fontweight="bold", color=NAVY)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

        # Feature importance
        ax = axes[1]
        top15 = self.feature_importances_.head(15)
        names = [FEATURE_PLAIN_ENGLISH.get(f,f) for f in top15.index]
        eng = {"credit_maturity","income_deliq_ratio","deliq_acceleration",
               "recent_stress","enq_velocity","emp_stability"}
        cols = ["#0D7377" if f in eng else "#1B3A6B" for f in top15.index]
        ax.barh(range(len(top15)), top15.values, color=cols,
                edgecolor="white", linewidth=1, height=0.7)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Top 15 Feature Importances\n(Teal=Engineered)",
                     fontweight="bold", color=NAVY)
        ax.set_facecolor("#FAFAFA")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # PR curves
        ax = axes[2]
        y_bin = label_binarize(y_arr, classes=CLASSES)
        for i, (cls, color) in enumerate(CLASS_COLORS.items()):
            if y_bin[:,i].sum() > 0:
                prec, rec, _ = precision_recall_curve(y_bin[:,i], proba[:,i])
                auprc = auc(rec, prec)
                ax.plot(rec, prec, color=color, lw=2, label=f"{cls} (AUPRC={auprc:.3f})")
        ax.set_title("Per-Class Precision-Recall", fontweight="bold", color=NAVY)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_facecolor("#FAFAFA")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        plt.tight_layout()
        out = save_path or str(MODELS_DIR / "gbm_performance.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"GBM chart saved -> {out}")
        return out

    def save(self, path=None):
        p = Path(path) if path else MODELS_DIR / "gbm_ensemble.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        log.info(f"GBM saved -> {p}")
        return str(p)

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else MODELS_DIR / "gbm_ensemble.joblib"
        return joblib.load(p)
