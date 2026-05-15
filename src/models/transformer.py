"""
transformer.py — FT-Transformer (PyTorch) with ExtraTrees fallback.
Model B in the ensemble — captures non-linear feature interactions.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import FEATURED_FILE, MODELS_DIR, RANDOM_STATE as RANDOM_SEED
from src.utils.logger import get_logger

log = get_logger("Transformer")

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDL, TensorDataset
    TORCH_AVAILABLE = True
    log.info("PyTorch available — FT-Transformer will be trained")

    class FTTransformerNet(nn.Module):
        def __init__(self, n_features, d_model=64, n_heads=4,
                     n_layers=3, dropout=0.1, n_classes=4):
            super().__init__()
            self.embed = nn.Linear(1, d_model)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4, dropout=dropout,
                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model//2),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model//2, n_classes))

        def forward(self, x):
            tokens = self.embed(x.unsqueeze(-1))
            out    = self.transformer(tokens)
            return self.head(out.mean(dim=1))

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        def forward(self, logits, targets):
            ce  = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            pt  = torch.exp(-ce)
            return (((1-pt)**self.gamma)*ce).mean()

except ImportError:
    log.info("PyTorch not available — ExtraTrees fallback will be used")


CLASS_NAMES = ["P1","P2","P3","P4"]

TRANSFORMER_FEATURES = [
    "Credit_Score","NETMONTHLYINCOME","Time_With_Curr_Empr","AGE",
    "num_times_delinquent","num_deliq_6mts","num_deliq_12mts",
    "enq_L6m","enq_L3m","enq_L12m",
    "deliq_acceleration","enq_velocity","income_deliq_ratio",
    "credit_maturity","recent_stress","emp_stability",
    "num_std","num_lss","num_dbt","num_sub",
    "pct_of_active_TLs_ever","max_unsec_exposure_inPct",
    "time_since_first_deliquency","time_since_recent_payment",
    "max_delinquency_level",
]


class TabTransformerModel:
    def __init__(self, n_epochs=30, batch_size=512, lr=1e-3, dropout=0.1):
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.dropout    = dropout
        self.feature_cols_ = None
        self._torch_model  = None
        self._fallback_et  = None
        self._feat_mean    = None
        self._feat_std     = None
        self._is_torch     = False

    def _get_feature_cols(self, X):
        avail = [f for f in TRANSFORMER_FEATURES if f in X.columns]
        if len(avail) < 5:
            exc = {"PROSPECTID","Approved_Flag","target_numeric"}
            avail = [c for c in X.columns if c not in exc
                     and str(X[c].dtype).startswith(("float","int"))]
        return avail

    def _fit_fallback(self, X_train, y_train, X_val, y_val):
        log.info("Training ExtraTrees fallback for Transformer slot...")
        self._fallback_et = ExtraTreesClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        self._fallback_et.fit(X_train[self.feature_cols_].values, y_train.values)
        self._is_torch = False
        if X_val is not None:
            preds = self._fallback_et.predict(X_val[self.feature_cols_].values)
            f1 = f1_score(y_val.values, preds, average="macro")
            log.info(f"[ExtraTrees Fallback] Val Macro F1: {f1:.4f}")

    def _fit_torch(self, X_train, y_train, X_val, y_val):
        log.info("Training FT-Transformer (PyTorch)...")
        X_np = X_train[self.feature_cols_].values.astype(np.float32)
        self._feat_mean = X_np.mean(0)
        self._feat_std  = X_np.std(0) + 1e-8
        X_np = (X_np - self._feat_mean) / self._feat_std

        X_t   = torch.tensor(X_np, dtype=torch.float32)
        y_t   = torch.tensor(y_train.values, dtype=torch.long)
        ds    = TensorDataset(X_t, y_t)
        dl    = TorchDL(ds, batch_size=self.batch_size, shuffle=True)

        n_features = len(self.feature_cols_)
        self._torch_model = FTTransformerNet(
            n_features=n_features, d_model=64, n_heads=4,
            n_layers=3, dropout=self.dropout)
        opt  = torch.optim.Adam(self._torch_model.parameters(), lr=self.lr)
        loss_fn = FocalLoss(gamma=2.0)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs)

        best_f1, wait, patience = 0, 0, 5
        for epoch in range(1, self.n_epochs+1):
            self._torch_model.train()
            tot = 0
            for Xb, yb in dl:
                opt.zero_grad()
                l = loss_fn(self._torch_model(Xb), yb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(self._torch_model.parameters(), 1.0)
                opt.step()
                tot += l.item()
            sch.step()
            if X_val is not None and epoch % 5 == 0:
                vf1 = self._eval_torch(X_val, y_val)
                log.info(f"  Epoch {epoch}/{self.n_epochs}  loss={tot/len(dl):.4f}  val_F1={vf1:.4f}")
                if vf1 > best_f1:
                    best_f1 = vf1; wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        log.info(f"  Early stop at epoch {epoch}")
                        break
        self._is_torch = True
        log.info(f"FT-Transformer done. Best val F1: {best_f1:.4f}")

    def _eval_torch(self, X_val, y_val):
        X_np = X_val[self.feature_cols_].values.astype(np.float32)
        X_np = (X_np - self._feat_mean) / self._feat_std
        self._torch_model.eval()
        with torch.no_grad():
            preds = self._torch_model(torch.tensor(X_np)).argmax(1).numpy()
        return f1_score(y_val.values, preds, average="macro")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        log.info("="*55)
        log.info("TRAINING TRANSFORMER MODEL")
        log.info("="*55)
        self.feature_cols_ = self._get_feature_cols(X_train)
        log.info(f"Features: {len(self.feature_cols_)} | PyTorch: {TORCH_AVAILABLE}")
        if TORCH_AVAILABLE:
            self._fit_torch(X_train, y_train, X_val, y_val)
        else:
            self._fit_fallback(X_train, y_train, X_val, y_val)
        if X_val is not None:
            self.evaluate(X_val, y_val, "Validation")
        return self

    def predict_proba(self, X):
        if self._is_torch and self._torch_model is not None:
            X_np = X[self.feature_cols_].values.astype(np.float32)
            X_np = (X_np - self._feat_mean) / self._feat_std
            self._torch_model.eval()
            with torch.no_grad():
                logits = self._torch_model(torch.tensor(X_np))
                return torch.softmax(logits, dim=1).numpy()
        return self._fallback_et.predict_proba(X[self.feature_cols_].values)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y, split_name="Validation"):
        proba   = self.predict_proba(X)
        preds   = np.argmax(proba, axis=1)
        macro   = f1_score(y.values, preds, average="macro")
        per_cls = f1_score(y.values, preds, average=None)
        metrics = {"macro_f1": round(macro,4),
                   "per_class_f1":{CLASS_NAMES[i]:round(per_cls[i],4) for i in range(4)}}
        log.info(f"[{split_name}] Transformer Macro F1: {macro:.4f}")
        for cls, f1v in metrics["per_class_f1"].items():
            log.info(f"  {cls} F1={f1v:.4f}")
        return metrics

    def save(self, path=None):
        p = Path(path) if path else MODELS_DIR / "transformer.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        log.info(f"Transformer saved -> {p}")
        return str(p)

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else MODELS_DIR / "transformer.joblib"
        return joblib.load(p)
