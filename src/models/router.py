"""
router.py — Phase 8, Prompt 8.2
Routes each application to the correct model and blends all outputs
into a unified PredictionOutput.

Flow:
  1. Rule engine  →  if fires, return immediately
  2. Credit_Score == 0  →  thin-file model
     Credit_Score >= 300 →  main ensemble
     Credit_Score 1-299  →  main ensemble + elevated risk flag
  3. Returning customers (Total_TL > 0)  →  also run proxy default model
  4. FOIR rule (already in rule engine, duplicated here for clarity)
  5. Thin-file policy limits
  6. Return unified PredictionOutput
"""
import sys, time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    TARGET_IMAP, ENSEMBLE_MODEL_FILE, THINFILE_MODEL_FILE, MODELS_DIR,
    RULE_MIN_CREDIT_SCORE, RULE_MAX_FOIR_REJECT, RULE_MAX_FOIR_COND,
    RULE_THINFILE_MAX_LOAN, RULE_THINFILE_MAX_FOIR,
)
from src.models.rule_engine   import RuleEngine
from src.api.feature_builder  import build_main_features, build_thin_file_features
from src.api.schemas          import PredictionOutput, make_confidence

# ── Lazy model loading ────────────────────────────────────────────────────────
_main_model      = None
_thinfile_bundle = None
_default_bundle  = None
_rule_engine     = RuleEngine()

def _load_main():
    global _main_model
    if _main_model is None:
        raw = joblib.load(ENSEMBLE_MODEL_FILE)
        # retrain.py saves a plain dict; router expects a LoanEnsemble object
        if isinstance(raw, dict) and 'gbm_ensemble' in raw:
            from src.models.ensemble import LoanEnsemble
            ens = LoanEnsemble()
            gbm_e = raw['gbm_ensemble']
            ens.rf   = gbm_e['rf']
            ens.et   = gbm_e['et']
            ens.meta = gbm_e['gbm_meta']
            ens.feature_cols = raw.get('feature_names', [])
            ens.is_fitted = True
            # Attach scorecard if present
            sc = raw.get('scorecard')
            if sc is not None:
                ens.scorecard    = sc.get('lr')
                ens.sc_features  = sc.get('score_cols')
            _main_model = ens
        else:
            _main_model = raw
    return _main_model

def _load_thinfile():
    global _thinfile_bundle
    if _thinfile_bundle is None:
        _thinfile_bundle = joblib.load(THINFILE_MODEL_FILE)
    return _thinfile_bundle

def _load_default():
    global _default_bundle
    if _default_bundle is None:
        path = MODELS_DIR / "default_model.joblib"
        _default_bundle = joblib.load(path)
    return _default_bundle


# ── Improvement actions lookup ────────────────────────────────────────────────
def _improvement_actions(decision: str, features: dict, model_used: str) -> list:
    actions = []
    if decision not in ('P1', 'P3'):
        return actions

    cs   = float(features.get('Credit_Score', 0) or 0)
    foir = float(features.get('foir', 0) or 0)
    lss  = float(features.get('num_lss', 0) or 0)
    dbt  = float(features.get('num_dbt', 0) or 0)
    miss = float(features.get('Tot_Missed_Pmnt', 0) or 0)
    iv   = int(features.get('income_verified', 1) or 1)
    ups  = float(features.get('utility_payment_score', 1) or 1)

    if lss > 0:
        actions.append('Resolve loss-classified accounts with the bank before reapplying')
    if cs > 0 and cs < RULE_MIN_CREDIT_SCORE:
        actions.append(f'Improve CIBIL score from {int(cs)} to at least {RULE_MIN_CREDIT_SCORE} '
                       f'by maintaining zero missed payments for 6+ months')
    if foir > RULE_MAX_FOIR_COND:
        actions.append(f'Reduce FOIR from {foir:.0%} to below 50% by closing existing loans '
                       f'or requesting a smaller loan amount / longer tenure')
    if miss > 0:
        actions.append(f'Clear {int(miss)} outstanding missed payment(s) and maintain clean '
                       f'repayment for 3+ months')
    if dbt > 0:
        actions.append('Resolve doubtful accounts flagged in your credit history')

    # Thin-file specific
    if model_used == 'thin_file':
        if iv == 0:
            actions.append('Link your bank account via Account Aggregator or provide '
                           '3 months of salary credit statements to verify income')
        if ups < 0.5:
            actions.append('Set up auto-pay for electricity, water and mobile bills to '
                           'build utility payment history')
        sc = features.get('salary_credit_months', 0) or 0
        if sc < 3:
            actions.append('Maintain consistent salary credits for at least 3 months '
                           'before reapplying')

    return actions if actions else ['Maintain clean repayment history and reapply in 6 months']


# ─────────────────────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────────────────────

def predict(request) -> PredictionOutput:
    """
    Full prediction pipeline for one application.
    Returns PredictionOutput with decision, probability, explanation.
    """
    t0          = time.time()
    feat_dict   = request.to_feature_dict()
    rules_fired = []
    model_used  = 'main_ensemble'

    # ── Step 0: Compute FOIR upfront so rule engine can use it ───────────────
    income   = float(request.NETMONTHLYINCOME or 0)
    loan_amt = float(request.loan_amount or 0)
    tenure   = float(request.loan_tenure_months or 12)
    existing = float(getattr(request, 'existing_emi', 0) or 0)
    new_emi  = round(loan_amt / tenure, 2) if tenure > 0 else 0
    feat_dict['foir'] = min(round((existing + new_emi) / income, 3), 1.0) if income > 0 else 1.0

    # ── Step 1: Rule engine ───────────────────────────────────────────────────
    rule_result = _rule_engine.evaluate(feat_dict)
    if rule_result is not None:
        forced_decision, rule_name, reason = rule_result
        rules_fired.append(rule_name)
        return PredictionOutput(
            decision=forced_decision,
            probability=0.95,
            confidence='HIGH',
            model_used='rule_engine',
            rules_fired=rules_fired,
            explanation=reason,
            improvement_actions=_improvement_actions(
                forced_decision, feat_dict, 'rule_engine'),
            default_probability=None,
            prospectid=request.PROSPECTID,
            processing_ms=round((time.time()-t0)*1000, 1),
        )

    cs         = float(request.Credit_Score or 0)
    total_tl   = float(request.Total_TL or 0)
    is_thin    = (cs == 0)

    # ── Step 2: Route to correct model ───────────────────────────────────────
    if is_thin:
        # ── Thin-file path ────────────────────────────────────────────────────
        model_used = 'thin_file'
        bundle     = _load_thinfile()
        clf        = bundle['model']
        encoders   = bundle['encoders']
        feat_cols  = bundle['features']

        tf_feats = build_thin_file_features(request)

        # Encode categoricals
        from src.utils.config import THINFILE_CAT_COLS
        row = dict(tf_feats)
        for c in THINFILE_CAT_COLS:
            if c in encoders and c in row:
                try:
                    row[c] = encoders[c].transform([str(row[c])])[0]
                except ValueError:
                    row[c] = 0

        X      = pd.DataFrame([row])[feat_cols]
        proba  = clf.predict_proba(X)[0]
        cw_p   = proba[1]   # creditworthy probability

        # Map to P1/P2/P3 (thin-file never gets P4)
        if cw_p >= 0.75:
            decision, prob = 'P2', cw_p
        elif cw_p >= 0.55:
            decision, prob = 'P3', cw_p
        else:
            decision, prob = 'P1', 1 - cw_p

        # Thin-file post-model policy rules
        loan_amt = float(request.loan_amount or 0)
        foir_val = tf_feats.get('foir', 0)
        iv       = int(request.income_verified or 0)

        if loan_amt > RULE_THINFILE_MAX_LOAN:
            decision = 'P1'
            rules_fired.append('RULE_08_THINFILE_LOAN_LIMIT')
        elif foir_val > RULE_THINFILE_MAX_FOIR and decision == 'P2':
            decision = 'P3'
            rules_fired.append('RULE_09_THINFILE_FOIR')
        elif iv == 0 and decision == 'P2':
            decision = 'P3'
            rules_fired.append('RULE_10_THINFILE_UNVERIFIED_INCOME')

        feat_dict.update(tf_feats)

    else:
        # ── Main ensemble path ────────────────────────────────────────────────
        if cs < 300:
            rules_fired.append('FLAG_LOW_BUREAU_SCORE')

        main_feats = build_main_features(request)
        model      = _load_main()
        fc         = model.feature_cols
        X          = pd.DataFrame([main_feats])[fc]
        proba_4    = model.predict_proba(X)[0]    # [P1, P2, P3, P4]
        pred_cls   = int(np.argmax(proba_4))
        decision   = TARGET_IMAP[pred_cls]
        prob       = float(proba_4[pred_cls])

        # Post-model rule 5: recent_stress downgrade
        rule5 = _rule_engine.evaluate(feat_dict, model_prediction=decision)
        if rule5 is not None:
            decision, rule5_name, _ = rule5
            rules_fired.append(rule5_name)
            prob = min(prob, 0.75)

        # Proxy default model for returning customers
        default_prob = None
        if total_tl > 0:
            try:
                dbundle  = _load_default()
                dmodel   = dbundle['model']
                dfeats   = dbundle['features']
                X_d      = pd.DataFrame([main_feats])[[f for f in dfeats
                                                        if f in main_feats]]
                # Align columns
                for f in dfeats:
                    if f not in X_d.columns:
                        X_d[f] = 0.0
                X_d       = X_d[dfeats]
                default_prob = float(dmodel.predict_proba(X_d)[0][1])
            except Exception:
                default_prob = None

        feat_dict.update(main_feats)

        return PredictionOutput(
            decision=decision,
            probability=round(prob, 4),
            confidence=make_confidence(prob),
            model_used=model_used,
            rules_fired=rules_fired,
            explanation=_build_explanation(decision, feat_dict, model_used, rules_fired),
            improvement_actions=_improvement_actions(decision, feat_dict, model_used),
            default_probability=round(default_prob, 4) if default_prob is not None else None,
            prospectid=request.PROSPECTID,
            processing_ms=round((time.time()-t0)*1000, 1),
        )

    # Thin-file return path
    return PredictionOutput(
        decision=decision,
        probability=round(prob, 4),
        confidence=make_confidence(prob),
        model_used=model_used,
        rules_fired=rules_fired,
        explanation=_build_explanation(decision, feat_dict, model_used, rules_fired),
        improvement_actions=_improvement_actions(decision, feat_dict, model_used),
        default_probability=None,
        prospectid=request.PROSPECTID,
        processing_ms=round((time.time()-t0)*1000, 1),
    )


def _build_explanation(decision: str, feats: dict,
                        model_used: str, rules_fired: list) -> str:
    cs   = float(feats.get('Credit_Score', 0) or 0)
    inc  = float(feats.get('NETMONTHLYINCOME', 0) or feats.get('monthly_income', 0) or 0)
    foir = float(feats.get('foir', 0) or 0)
    miss = float(feats.get('Tot_Missed_Pmnt', 0) or 0)

    if rules_fired and 'rule_engine' in model_used:
        return feats.get('_rule_reason', 'Application decided by bank policy rule.')

    label = {'P1':'rejected','P2':'approved','P3':'conditionally approved',
             'P4':'referred to Credit Committee'}.get(decision, decision)

    if model_used == 'thin_file':
        iv  = int(feats.get('income_verified', 0) or 0)
        src = feats.get('income_source','')
        return (f'Thin-file application {label}. '
                f'Income source: {src}. '
                f'Income {"verified" if iv else "unverified"}. '
                f'FOIR: {foir:.0%}.')
    else:
        return (f'Application {label} by credit scoring model. '
                f'CIBIL score: {int(cs)}. '
                f'Net monthly income: ₹{inc:,.0f}. '
                f'Missed payments on file: {int(miss)}.')
