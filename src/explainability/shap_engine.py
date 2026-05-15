"""
shap_engine.py — Phase 9 (SHAP upgrade)

PRIMARY path  (when shap is installed):
    Uses shap.TreeExplainer for exact, per-applicant Shapley values.
    - GBM / RF / ExtraTrees: TreeExplainer (fast, exact)
    - LoanEnsemble: runs TreeExplainer on the GBM meta-learner
    - Values are per-class; we extract the predicted class slice.

FALLBACK path (when shap is not installed):
    Uses model.feature_importances_ with benchmark-based direction inference.
    Produces qualitatively similar output, less precise magnitudes.

Install SHAP locally:
    pip install shap
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import FEATURE_PLAIN_ENGLISH

# ── Try to import SHAP ────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Direction helpers ─────────────────────────────────────────────────────────
APPROVE_POSITIVE = {
    'Credit_Score', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr',
    'num_std', 'pct_of_active_TLs_ever', 'time_since_recent_payment',
    'time_since_recent_deliquency', 'time_since_recent_enq',
    'credit_history_span', 'AGE',
    'income_verified', 'salary_credit_months', 'avg_monthly_balance',
    'utility_payment_score', 'upi_monthly_txn_count', 'has_aadhaar_linked',
    'monthly_income',
}

APPROVE_NEGATIVE = {
    'num_lss', 'num_dbt', 'num_sub', 'num_times_delinquent',
    'num_times_60p_dpd', 'max_delinquency_level', 'max_recent_level_of_deliq',
    'Tot_Missed_Pmnt', 'missed_payment_rate',
    'enq_L3m', 'enq_L6m', 'enq_L12m', 'foir',
}

BENCHMARKS = {
    'Credit_Score': 700,        'NETMONTHLYINCOME': 30_000,
    'num_std': 3,               'num_lss': 0,
    'num_dbt': 0,               'num_times_delinquent': 0,
    'num_times_60p_dpd': 0,     'Tot_Missed_Pmnt': 0,
    'missed_payment_rate': 0.0, 'enq_L3m': 0,
    'enq_L6m': 1,               'enq_L12m': 2,
    'foir': 0.40,               'pct_of_active_TLs_ever': 0.80,
    'credit_history_span': 60,  'income_verified': 1,
    'salary_credit_months': 6,  'utility_payment_score': 0.75,
    'avg_monthly_balance': 5_000,
}

# Map from LoanEnsemble class → 4 output classes
TARGET_IMAP = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}


class SHAPEngine:
    """
    Per-applicant feature attribution engine.

    When shap is available:
        TreeExplainer gives exact Shapley values — positive = pushes toward
        the predicted class, negative = pushes away from it.

    When shap is not available:
        Falls back to normalised feature_importances_ with direction
        inferred from benchmark comparisons.
    """

    def __init__(self, model, feature_cols: list, model_type: str = 'main'):
        self.model        = model
        self.feature_cols = feature_cols
        self.model_type   = model_type
        self._explainer   = None
        self._method      = None

        # ── Resolve model for SHAP / fallback ────────────────────────────────
        # MUST use ET/RF (106 features). GBM meta was trained on 8 meta-features.
        self._full_model = self._resolve_full_feature_model(model)

        # ── Pre-build SHAP explainer ───────────────────────────────────────────
        if SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(self._full_model)
                self._method    = 'shap_tree'
            except Exception as e:
                warnings.warn(f'TreeExplainer failed ({e}), falling back')
                self._method = 'feature_importances'
        else:
            self._method = 'feature_importances'

        # ── Fallback importances ──────────────────────────────────────────────
        fim = self._full_model
        if hasattr(fim, 'feature_importances_') and                 len(fim.feature_importances_) == len(feature_cols):
            self._base_importance = np.array(fim.feature_importances_)
        else:
            self._base_importance = np.ones(len(feature_cols)) / len(feature_cols)

    def _resolve_full_feature_model(self, model):
        """Return an estimator trained on the full feature set (not meta-features)."""
        # dict from retrain.py
        if isinstance(model, dict):
            gbm_e = model.get('gbm_ensemble', model)
            if isinstance(gbm_e, dict):
                return gbm_e.get('et') or gbm_e.get('rf') or model
            return gbm_e
        # LoanEnsemble object
        if hasattr(model, 'et') and model.et is not None:
            return model.et
        if hasattr(model, 'rf') and model.rf is not None:
            return model.rf
        return model

    # ── SHAP attribution ──────────────────────────────────────────────────────

    def _shap_attribute(self, X_df: pd.DataFrame,
                         pred_class: int, top_n: int) -> list:
        """
        Compute SHAP values for one applicant row.
        Returns list of factor dicts sorted by |shap_value| descending.
        """
        sv = self._explainer.shap_values(X_df)

        # sv shape depends on model type:
        #   GBM multiclass  → list of (1, n_features) arrays, one per class
        #   GBM binary      → (1, n_features) array
        #   RF multiclass   → list of (1, n_features) arrays
        if isinstance(sv, list):
            # Multiclass — take the predicted class slice
            class_idx = min(pred_class, len(sv) - 1)
            sv_row    = sv[class_idx][0]
        else:
            sv_row = sv[0]

        results = []
        for i, feat in enumerate(self.feature_cols):
            shap_val = float(sv_row[i])
            raw_val  = float(X_df.iloc[0][feat])

            if abs(shap_val) < 1e-6:
                continue

            label     = FEATURE_PLAIN_ENGLISH.get(feat,
                            feat.replace('_', ' ').title())
            bench     = BENCHMARKS.get(feat)
            # For SHAP: positive value = pushes toward predicted class
            # We remap to approval direction:
            #   predicted class = P2 → positive shap = good
            #   predicted class = P1 → positive shap = bad (pushes toward reject)
            if pred_class == 1:   # P2 Approve
                direction = 'positive' if shap_val > 0 else 'negative'
            elif pred_class == 0:  # P1 Reject — invert
                direction = 'negative' if shap_val > 0 else 'positive'
            else:                  # P3/P4 — use value-vs-benchmark
                direction = self._bench_direction(feat, raw_val, bench)

            results.append({
                'feature':           feat,
                'label':             label,
                'value':             raw_val,
                'benchmark':         bench,
                'direction':         direction,
                'magnitude':         round(abs(shap_val), 5),
                'shap_value':        round(shap_val, 5),
                'supports_decision': True,
                'description':       self._describe(feat, raw_val, bench, direction),
                'method':            'shap',
            })

        results.sort(key=lambda x: x['magnitude'], reverse=True)
        return results[:top_n]

    # ── Fallback attribution ──────────────────────────────────────────────────

    def _fallback_attribute(self, feature_dict: dict, top_n: int) -> list:
        """Uses model.feature_importances_ — global, not per-applicant."""
        total_imp = self._base_importance.sum()
        if total_imp == 0:
            return []
        norm_imp = self._base_importance / total_imp

        results = []
        for i, feat in enumerate(self.feature_cols):
            val = feature_dict.get(feat)
            if val is None:
                continue
            imp = float(norm_imp[i])
            if imp < 0.001:
                continue
            bench     = BENCHMARKS.get(feat)
            direction = self._bench_direction(feat, val, bench)
            label     = FEATURE_PLAIN_ENGLISH.get(feat,
                            feat.replace('_', ' ').title())
            results.append({
                'feature':           feat,
                'label':             label,
                'value':             val,
                'benchmark':         bench,
                'direction':         direction,
                'magnitude':         round(imp, 5),
                'shap_value':        None,
                'supports_decision': True,
                'description':       self._describe(feat, val, bench, direction),
                'method':            'feature_importances',
            })

        results.sort(key=lambda x: x['magnitude'], reverse=True)
        return results[:top_n]

    # ── Public API ────────────────────────────────────────────────────────────

    def attribute(self, feature_dict: dict, top_n: int = 8,
                  pred_class: int = 1) -> list:
        """
        Main attribution method.

        Args:
            feature_dict: flat dict of feature values for one applicant
            top_n:        number of top factors to return
            pred_class:   predicted class index (0=P1,1=P2,2=P3,3=P4)
                          used to orient SHAP values correctly
        Returns:
            list of factor dicts sorted by magnitude descending
        """
        if self._method == 'shap_tree':
            try:
                # Build aligned DataFrame
                row = {f: feature_dict.get(f, 0.0) for f in self.feature_cols}
                X   = pd.DataFrame([row])[self.feature_cols]
                return self._shap_attribute(X, pred_class, top_n)
            except Exception as e:
                warnings.warn(f'SHAP attribution failed ({e}), using fallback')

        return self._fallback_attribute(feature_dict, top_n)

    @property
    def method(self) -> str:
        return self._method

    # ── Direction & description helpers ──────────────────────────────────────

    def _bench_direction(self, feat, val, bench) -> str:
        if feat in APPROVE_POSITIVE:
            if bench is not None:
                return 'positive' if float(val) >= float(bench) else 'negative'
            return 'positive' if float(val) > 0 else 'neutral'
        if feat in APPROVE_NEGATIVE:
            if bench is not None:
                return 'positive' if float(val) <= float(bench) else 'negative'
            return 'positive' if float(val) == 0 else 'negative'
        return 'neutral'

    def _describe(self, feat, val, bench, direction) -> str:
        label = FEATURE_PLAIN_ENGLISH.get(feat, feat)
        try:
            v = float(val)
        except (TypeError, ValueError):
            return f'{label}: {val}'

        if feat == 'Credit_Score':
            tier = ('Excellent' if v >= 750 else 'Good' if v >= 700
                    else 'Fair' if v >= 650 else 'Poor' if v >= 550 else 'Very Poor')
            return f'CIBIL score {int(v)} — {tier}'
        if feat == 'NETMONTHLYINCOME':
            return f'Monthly income ₹{v:,.0f}'
        if feat == 'Tot_Missed_Pmnt':
            return (f'{int(v)} missed payment(s)' if v > 0
                    else 'No missed payments on internal accounts')
        if feat == 'foir':
            return f'FOIR {v:.0%} — {"elevated" if v > 0.5 else "acceptable"}'
        if feat == 'enq_L3m':
            return (f'{int(v)} enquiry(ies) in last 3 months' if v > 0
                    else 'No enquiries in last 3 months')
        if feat == 'credit_history_span':
            return f'Credit history span {int(v)} months'
        if feat == 'income_verified':
            return 'Income verified' if v == 1 else 'Income NOT verified'
        if feat == 'utility_payment_score':
            return f'Utility payment score {v:.0%}'
        if feat == 'num_lss':
            return (f'{int(v)} loss-classified account(s)' if v > 0
                    else 'No loss-classified accounts')
        if feat == 'num_times_delinquent':
            return (f'{int(v)} delinquency instance(s) on record' if v > 0
                    else 'No delinquency on record')
        if feat == 'pct_of_active_TLs_ever':
            return f'{v:.0%} of trade lines ever active'
        if feat == 'time_since_recent_enq':
            return (f'{int(v)} months since last bureau enquiry'
                    if v < 500 else 'No recent bureau enquiry')
        if bench is not None:
            arrow = '↑' if direction == 'positive' else '↓'
            return f'{label}: {v:.3g} {arrow} (benchmark: {bench})'
        return f'{label}: {v:.3g}'

    # ── Improvement roadmap ───────────────────────────────────────────────────

    def improvement_roadmap(self, feature_dict: dict, decision: str) -> list:
        if decision not in ('P1', 'P3'):
            return []
        steps = []
        fd    = feature_dict

        def add(p, action, impact, timeline):
            steps.append({'priority': p, 'action': action,
                          'impact': impact, 'timeline': timeline})

        cs   = float(fd.get('Credit_Score',          0) or 0)
        lss  = float(fd.get('num_lss',               0) or 0)
        dbt  = float(fd.get('num_dbt',               0) or 0)
        miss = float(fd.get('Tot_Missed_Pmnt',       0) or 0)
        enq3 = float(fd.get('enq_L3m',              0) or 0)
        foir = float(fd.get('foir',                  0) or 0)
        iv   = int(fd.get('income_verified',         1) or 1)
        ups  = float(fd.get('utility_payment_score', 1) or 1)
        sc   = int(fd.get('salary_credit_months',    0) or 0)

        if lss > 0:
            add(1,
                f'Resolve {int(lss)} loss-classified account(s) — contact '
                'the recovery team to settle outstanding dues immediately.',
                'CRITICAL — blocks all approvals until resolved',
                '3–12 months depending on settlement')
        if dbt > 0:
            add(1 if lss == 0 else 2,
                f'Regularise {int(dbt)} doubtful account(s) by clearing '
                'overdue amounts and upgrading account classification.',
                'HIGH — removes mandatory manual review (P4) flag',
                '2–6 months')
        if 0 < cs < 700:
            gap = int(700 - cs)
            add(2,
                f'Raise CIBIL score from {int(cs)} to at least 700 (+{gap} pts): '
                'pay all EMIs on time, keep credit card utilisation below 30%, '
                'avoid new loan applications for 6 months.',
                f'HIGH — {gap} point gap to standard approval threshold',
                '6–18 months of consistent repayment')
        if miss > 0:
            add(2,
                f'Clear {int(miss)} missed payment(s) immediately and set up '
                'auto-pay on all accounts to prevent future misses.',
                'HIGH — directly reduces missed_payment_rate feature',
                '1–3 months')
        if foir > 0.50:
            add(3,
                f'Reduce FOIR from {foir:.0%} to below 50%: (a) request a smaller '
                'loan amount, (b) extend tenure to lower the EMI, or '
                '(c) foreclose an existing loan to free up income.',
                'MEDIUM — required to upgrade from P3 to P2',
                'Immediate if loan parameters adjusted')
        if enq3 >= 3:
            add(3,
                f'Pause all credit applications for 6 months. {int(enq3)} bureau '
                'enquiries in 3 months signals credit-hungry behaviour to lenders.',
                'MEDIUM — reduces enquiry risk score',
                '3–6 month cooling-off period')
        if iv == 0:
            add(2,
                'Verify income via Account Aggregator consent or submit 3 months '
                'of salary credit bank statements to the branch.',
                'HIGH — unverified income automatically prevents P2 approval',
                'Within days')
        if ups < 0.5:
            add(4,
                f'Improve utility payment score from {ups:.0%} to above 75%: '
                'set up auto-pay for electricity, water and mobile bills.',
                'MEDIUM — builds alternative credit history',
                '3–6 months of consistent payments')
        if sc < 3:
            add(4,
                f'Maintain consistent salary credits in the same bank account '
                f'for at least 3 months (currently {sc} months observed).',
                'MEDIUM — enables income pattern verification',
                '3 months')

        steps.sort(key=lambda x: x['priority'])
        return steps


# ── Convenience wrapper ───────────────────────────────────────────────────────

def explain_decision(output, feature_dict: dict,
                     main_model=None, thinfile_model=None,
                     feature_cols: list = None) -> dict:
    """
    Top-level explain function. Auto-selects model and engine.

    Args:
        output:          PredictionOutput from router
        feature_dict:    flat feature dict for this applicant
        main_model:      optional pre-loaded LoanEnsemble
        thinfile_model:  optional pre-loaded thin-file GBM
        feature_cols:    optional feature list (auto-loaded if None)

    Returns:
        { factors, roadmap, method }
    """
    import joblib
    from src.utils.config import MODELS_DIR
    from src.utils.config import TARGET_IMAP

    # ── Rule engine — still generate roadmap + attempt factor attribution ────
    if output.model_used == 'rule_engine':
        # Even rule-engine decisions benefit from improvement roadmap
        # and best-effort factor attribution.
        try:
            # Try to build factors from main model
            if main_model is None:
                try:
                    ensemble     = joblib.load(MODELS_DIR / 'final_ensemble.joblib')
                    main_model   = ensemble
                    if isinstance(ensemble, dict):
                        feature_cols = ensemble.get('feature_names', [])
                    else:
                        feature_cols = ensemble.feature_cols
                except Exception:
                    pass

            if main_model is not None and feature_cols:
                engine = SHAPEngine(main_model, feature_cols, 'main')
                inv_map    = {v: k for k, v in TARGET_IMAP.items()}
                pred_class = inv_map.get(output.decision, 0)
                factors = engine.attribute(feature_dict, top_n=8, pred_class=pred_class)
                roadmap = engine.improvement_roadmap(feature_dict, output.decision)
                return {'factors': factors, 'roadmap': roadmap, 'method': 'rule_engine+fallback'}
        except Exception:
            pass

        # Minimal fallback — at least generate roadmap from feature data
        _engine = SHAPEngine.__new__(SHAPEngine)
        _engine.feature_cols = feature_cols or []
        _engine._base_importance = np.ones(max(len(feature_cols or []), 1))
        roadmap = _engine.improvement_roadmap(feature_dict, output.decision)
        return {'factors': [], 'roadmap': roadmap, 'method': 'rule_engine'}

    # ── Select model ──────────────────────────────────────────────────────────
    if output.model_used == 'thin_file':
        # Always load thin-file bundle to get the correct 22 feature columns.
        # The caller may pass main-ensemble feature_cols (106 features) which
        # would cause a shape mismatch — so we always use the bundle's cols.
        try:
            bundle         = joblib.load(MODELS_DIR / 'thin_file_model.joblib')
            thinfile_model = bundle['model']
            feature_cols   = bundle['features']
        except Exception:
            if thinfile_model is None:
                raise
            # Fallback: use what was passed in (may still error later)
        engine = SHAPEngine(thinfile_model, feature_cols, 'thin_file')
        # Thin-file is binary — class 1 = creditworthy
        pred_class = 1 if output.decision in ('P2', 'P3') else 0

    else:
        if main_model is None:
            try:
                ensemble     = joblib.load(MODELS_DIR / 'final_ensemble.joblib')
                main_model   = ensemble
                if isinstance(ensemble, dict):
                    feature_cols = ensemble.get('feature_names', [])
                else:
                    feature_cols = ensemble.feature_cols
            except Exception:
                bundle       = joblib.load(MODELS_DIR / 'default_model.joblib')
                main_model   = bundle['model']
                feature_cols = bundle['features']
        engine = SHAPEngine(main_model, feature_cols, 'main')
        # Map decision label back to class index
        inv_map    = {v: k for k, v in TARGET_IMAP.items()}
        pred_class = inv_map.get(output.decision, 1)

    factors = engine.attribute(feature_dict, top_n=8, pred_class=pred_class)
    roadmap = engine.improvement_roadmap(feature_dict, output.decision)

    return {
        'factors': factors,
        'roadmap': roadmap,
        'method':  engine.method,
    }
