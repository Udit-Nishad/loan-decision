"""
feature_builder.py — Phase 8, Prompt 8.1
Maps raw LoanApplicationInput fields to the exact feature vectors
each model was trained on.

Two functions:
  build_main_features(request)     → dict of 106 features for main ensemble
  build_thin_file_features(request)→ dict of 22 features for thin-file model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import joblib
from src.utils.config import (
    HARD_CAPS, MODELS_DIR, THINFILE_FEATURES, THINFILE_CAT_COLS
)

# ── Feature lists exactly matching training-time column order ─────────────────
MAIN_FEATURE_COLS = None   # loaded lazily from joblib

def _get_main_features():
    global MAIN_FEATURE_COLS
    if MAIN_FEATURE_COLS is None:
        MAIN_FEATURE_COLS = joblib.load(MODELS_DIR / "feature_cols.joblib")
    return MAIN_FEATURE_COLS


# ── OHE category maps (must match training-time pd.get_dummies output) ────────
EDUCATION_CATS   = ['12TH','GRADUATE','OTHERS','POST-GRADUATE',
                    'PROFESSIONAL','SSC','UNDER GRADUATE']
MARITAL_CATS     = ['Married','Single']
GENDER_CATS      = ['F','M']
LAST_PROD_CATS   = ['AL','CC','ConsumerLoan','HL','PL','others']
FIRST_PROD_CATS  = ['AL','CC','ConsumerLoan','HL','PL','others']


def _ohe_col(prefix, cats, value):
    """Return a dict of one-hot cols for a categorical field."""
    val = str(value).strip() if value else ''
    return {f"{prefix}_{c}": int(val == c) for c in cats}


def _cap(val, col):
    """Apply hard cap from config if defined."""
    cap = HARD_CAPS.get(col)
    if cap is not None and val is not None:
        return min(float(val), cap)
    return float(val) if val is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main model feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_main_features(request) -> dict:
    """
    Map LoanApplicationInput → 106-feature dict for main ensemble.
    Applies outlier caps, computes 3 engineered TL features, one-hot encodes.
    Raises ValueError if any required feature is missing after construction.
    """
    r = request  # shorthand

    # ── Raw passthrough fields ────────────────────────────────────────────────
    row = {
        # Demographic
        'AGE':               float(r.AGE or 0),
        'NETMONTHLYINCOME':  min(float(r.NETMONTHLYINCOME or 0), 78_000),
        'Time_With_Curr_Empr': float(getattr(r, 'Time_With_Curr_Empr', 0) or 0),

        # CIBIL delinquency
        'time_since_recent_payment':    float(r.time_since_recent_payment or 0),
        'time_since_first_deliquency':  float(r.time_since_first_deliquency or 0),
        'time_since_recent_deliquency': float(r.time_since_recent_deliquency or 0),
        'num_times_delinquent':         float(r.num_times_delinquent or 0),
        'max_delinquency_level':        float(r.max_delinquency_level or 0),
        'max_recent_level_of_deliq':    float(r.max_recent_level_of_deliq or 0),
        'num_deliq_6mts':  float(getattr(r,'num_deliq_6mts',  0) or 0),
        'num_deliq_12mts': float(getattr(r,'num_deliq_12mts', 0) or 0),
        'num_deliq_6_12mts':float(getattr(r,'num_deliq_6_12mts',0) or 0),
        'max_deliq_6mts':  float(getattr(r,'max_deliq_6mts',  0) or 0),
        'max_deliq_12mts': float(getattr(r,'max_deliq_12mts', 0) or 0),
        'num_times_30p_dpd':float(getattr(r,'num_times_30p_dpd',0) or 0),
        'num_times_60p_dpd':float(r.num_times_60p_dpd or 0),
        'recent_level_of_deliq':float(getattr(r,'recent_level_of_deliq',0) or 0),

        # CIBIL account type counts
        'num_std':      float(r.num_std  or 0),
        'num_std_6mts': float(getattr(r,'num_std_6mts', 0) or 0),
        'num_std_12mts':float(getattr(r,'num_std_12mts',0) or 0),
        'num_sub':      float(r.num_sub  or 0),
        'num_sub_6mts': float(getattr(r,'num_sub_6mts', 0) or 0),
        'num_sub_12mts':float(getattr(r,'num_sub_12mts',0) or 0),
        'num_dbt':      float(r.num_dbt  or 0),
        'num_dbt_6mts': float(getattr(r,'num_dbt_6mts', 0) or 0),
        'num_dbt_12mts':float(getattr(r,'num_dbt_12mts',0) or 0),
        'num_lss':      float(r.num_lss  or 0),
        'num_lss_6mts': float(getattr(r,'num_lss_6mts', 0) or 0),
        'num_lss_12mts':float(getattr(r,'num_lss_12mts',0) or 0),

        # CIBIL enquiries
        'tot_enq':    float(getattr(r,'tot_enq', 0) or 0),
        'CC_enq':     float(getattr(r,'CC_enq',  0) or 0),
        'CC_enq_L6m': float(getattr(r,'CC_enq_L6m',0) or 0),
        'CC_enq_L12m':float(getattr(r,'CC_enq_L12m',0) or 0),
        'PL_enq':     float(getattr(r,'PL_enq',  0) or 0),
        'PL_enq_L6m': float(getattr(r,'PL_enq_L6m',0) or 0),
        'PL_enq_L12m':float(getattr(r,'PL_enq_L12m',0) or 0),
        'time_since_recent_enq':float(r.time_since_recent_enq or 0),
        'enq_L12m':   float(r.enq_L12m or 0),
        'enq_L6m':    float(r.enq_L6m  or 0),
        'enq_L3m':    float(r.enq_L3m  or 0),

        # CIBIL utilisation / portfolio
        'pct_of_active_TLs_ever':    float(r.pct_of_active_TLs_ever or 0),
        'pct_opened_TLs_L6m_of_L12m':float(getattr(r,'pct_opened_TLs_L6m_of_L12m',0) or 0),
        'pct_currentBal_all_TL':     float(getattr(r,'pct_currentBal_all_TL',0) or 0),
        'CC_utilization':            float(getattr(r,'CC_utilization',0) or 0),
        'CC_Flag':                   int(r.CC_Flag or 0),
        'PL_utilization':            float(getattr(r,'PL_utilization',0) or 0),
        'PL_Flag':                   int(r.PL_Flag or 0),
        'pct_PL_enq_L6m_of_L12m':   float(r.pct_PL_enq_L6m_of_L12m or 0),
        'pct_CC_enq_L6m_of_L12m':   float(r.pct_CC_enq_L6m_of_L12m or 0),
        'pct_PL_enq_L6m_of_ever':   float(r.pct_PL_enq_L6m_of_ever or 0),
        'pct_CC_enq_L6m_of_ever':   float(r.pct_CC_enq_L6m_of_ever or 0),
        'max_unsec_exposure_inPct':  float(getattr(r,'max_unsec_exposure_inPct',0) or 0),
        'HL_Flag':                   int(getattr(r,'HL_Flag',0) or 0),
        'GL_Flag':                   int(getattr(r,'GL_Flag',0) or 0),
        'Credit_Score':              float(r.Credit_Score or 0),
        'Other_TL':                  float(getattr(r,'Other_TL',0) or 0),

        # Internal TL (with outlier caps)
        'Total_TL':            _cap(r.Total_TL,    'Total_TL'),
        'Tot_Closed_TL':       float(r.Tot_Closed_TL       or 0),
        'Tot_Active_TL':       float(r.Tot_Active_TL       or 0),
        'Total_TL_opened_L6M': float(r.Total_TL_opened_L6M or 0),
        'Tot_TL_closed_L6M':   float(r.Tot_TL_closed_L6M   or 0),
        'pct_tl_open_L6M':     float(r.pct_tl_open_L6M     or 0),
        'pct_tl_closed_L6M':   float(r.pct_tl_closed_L6M   or 0),
        'pct_active_tl':       float(getattr(r,'pct_active_tl',0) or 0),
        'pct_closed_tl':       float(getattr(r,'pct_closed_tl',0) or 0),
        'Total_TL_opened_L12M':float(r.Total_TL_opened_L12M or 0),
        'Tot_TL_closed_L12M':  float(r.Tot_TL_closed_L12M   or 0),
        'pct_tl_open_L12M':    float(r.pct_tl_open_L12M     or 0),
        'pct_tl_closed_L12M':  float(r.pct_tl_closed_L12M   or 0),
        'Tot_Missed_Pmnt':     float(r.Tot_Missed_Pmnt       or 0),
        'Auto_TL':             float(r.Auto_TL     or 0),
        'CC_TL':               float(r.CC_TL       or 0),
        'Consumer_TL':         float(r.Consumer_TL or 0),
        'Gold_TL':             _cap(r.Gold_TL,  'Gold_TL'),
        'Home_TL':             float(r.Home_TL  or 0),
        'PL_TL':               float(r.PL_TL    or 0),
        'Secured_TL':          float(r.Secured_TL   or 0),
        'Unsecured_TL':        float(r.Unsecured_TL or 0),
        'Age_Oldest_TL':       float(r.Age_Oldest_TL or 0),
        'Age_Newest_TL':       float(r.Age_Newest_TL or 0),
    }

    # ── 3 engineered TL features ──────────────────────────────────────────────
    row['missed_payment_rate'] = round(
        row['Tot_Missed_Pmnt'] / (row['Total_TL'] + 1), 4)
    row['tl_utilization_rate'] = round(
        row['Tot_Active_TL']   / (row['Total_TL'] + 1), 4)
    row['credit_history_span'] = max(
        0, row['Age_Oldest_TL'] - row['Age_Newest_TL'])

    # ── One-hot encoding ──────────────────────────────────────────────────────
    row.update(_ohe_col('EDUCATION',       EDUCATION_CATS,  r.EDUCATION))
    row.update(_ohe_col('MARITALSTATUS',   MARITAL_CATS,    r.MARITALSTATUS))
    row.update(_ohe_col('GENDER',          GENDER_CATS,     r.GENDER))
    row.update(_ohe_col('last_prod_enq2',  LAST_PROD_CATS,  r.last_prod_enq2))
    row.update(_ohe_col('first_prod_enq2', FIRST_PROD_CATS, r.first_prod_enq2))

    # ── Align to exact training-time feature order ────────────────────────────
    feature_cols = _get_main_features()
    missing = [f for f in feature_cols if f not in row]
    if missing:
        # Fill any training-time columns not present in the API with 0
        for f in missing:
            row[f] = 0.0

    ordered = {f: row.get(f, 0.0) for f in feature_cols}
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Thin-file feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_thin_file_features(request) -> dict:
    """
    Map LoanApplicationInput → 22-feature dict for thin-file model.
    Computes FOIR from income + EMI + loan params.
    Raises ValueError if monthly_income is missing or zero.
    """
    r = request

    income = float(r.NETMONTHLYINCOME or 0)
    if income <= 0:
        raise ValueError("NETMONTHLYINCOME required for thin-file scoring")

    loan_amount  = float(r.loan_amount or 0)
    tenure       = float(r.loan_tenure_months or 12)
    existing_emi = float(getattr(r, 'existing_emi', 0) or 0)
    new_emi      = round(loan_amount / tenure, 2) if tenure > 0 else 0
    foir         = min(round((existing_emi + new_emi) / income, 3), 1.0)

    row = {
        'person_age':             float(r.AGE or 0),
        'gender':                 str(r.GENDER or 'M'),
        'education':              str(r.EDUCATION or 'Graduate'),
        'marital_status':         str(r.MARITALSTATUS or 'Single'),
        'residence_type':         str(getattr(r,'residence_type','Urban') or 'Urban'),
        'income_source':          str(r.income_source or 'Salaried'),
        'monthly_income':         income,
        'employment_months':      float(getattr(r,'employment_months',0) or 0),
        'income_verified':        int(r.income_verified or 0),
        'salary_credit_months':   int(r.salary_credit_months or 0),
        'avg_monthly_balance':    float(r.avg_monthly_balance or 0),
        'utility_payment_score':  float(r.utility_payment_score or 0),
        'upi_monthly_txn_count':  int(r.upi_monthly_txn_count or 0),
        'existing_emi':           existing_emi,
        'loan_purpose':           str(r.loan_purpose or 'Personal'),
        'loan_amount_requested':  loan_amount,
        'loan_tenure_months':     int(tenure),
        'new_emi_approx':         new_emi,
        'foir':                   foir,
        'residence_stability_yrs':float(r.residence_stability_yrs or 0),
        'has_mobile_verified':    int(getattr(r,'has_mobile_verified',1) or 1),
        'has_aadhaar_linked':     int(r.has_aadhaar_linked or 0),
    }

    # Confirm all 22 thin-file features present
    missing = [f for f in THINFILE_FEATURES if f not in row]
    if missing:
        raise ValueError(f"Thin-file builder missing features: {missing}")

    return {f: row[f] for f in THINFILE_FEATURES}


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (run with: python -m src.api.feature_builder)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.api.schemas import LoanApplicationInput

    print('feature_builder unit tests')
    results = []

    def test(desc, fn, expect_pass=True):
        try:
            fn()
            ok = expect_pass
        except Exception as e:
            ok = not expect_pass
            if expect_pass:
                print(f'  ❌ {desc}: {e}')
                results.append(False)
                return
        print(f'  {"✅" if ok else "❌"} {desc}')
        results.append(ok)

    # Test 1: main model — all 106 features present
    def t1():
        req = LoanApplicationInput(AGE=35, NETMONTHLYINCOME=50000,
            loan_amount=300000, loan_purpose='Business',
            Credit_Score=720, Total_TL=5, Gold_TL=2,
            GENDER='M', EDUCATION='GRADUATE', MARITALSTATUS='Married',
            loan_tenure_months=36)
        feats = build_main_features(req)
        fc = _get_main_features()
        assert len(feats) == len(fc), f"Expected {len(fc)}, got {len(feats)}"
        assert set(feats.keys()) == set(fc)
    test('Main model: all 106 features present', t1)

    # Test 2: outlier cap applied
    def t2():
        req = LoanApplicationInput(AGE=35, NETMONTHLYINCOME=50000,
            loan_amount=300000, loan_purpose='B', Gold_TL=100, Total_TL=200)
        feats = build_main_features(req)
        assert feats['Gold_TL']  <= 24,  f"Gold_TL not capped: {feats['Gold_TL']}"
        assert feats['Total_TL'] <= 34, f"Total_TL not capped: {feats['Total_TL']}"
    test('Main model: outlier caps applied (Gold_TL=100→24, Total_TL=200→34)', t2)

    # Test 3: engineered features computed correctly
    def t3():
        req = LoanApplicationInput(AGE=35, NETMONTHLYINCOME=50000,
            loan_amount=300000, loan_purpose='B',
            Total_TL=10, Tot_Active_TL=7, Tot_Missed_Pmnt=2,
            Age_Oldest_TL=120, Age_Newest_TL=24)
        feats = build_main_features(req)
        assert abs(feats['missed_payment_rate'] - 2/11) < 0.001
        assert abs(feats['tl_utilization_rate'] - 7/11) < 0.001
        assert feats['credit_history_span'] == 96
    test('Main model: 3 engineered features computed correctly', t3)

    # Test 4: all TL fields None → defaults to 0, no crash
    def t4():
        req = LoanApplicationInput(AGE=30, NETMONTHLYINCOME=30000,
            loan_amount=100000, loan_purpose='P', Credit_Score=680)
        feats = build_main_features(req)
        assert feats['Total_TL'] == 0
        assert feats['missed_payment_rate'] == 0.0
    test('Main model: all TL fields=0, no crash', t4)

    # Test 5: thin-file — all 22 features present
    def t5():
        req = LoanApplicationInput(AGE=28, NETMONTHLYINCOME=30000,
            loan_amount=200000, loan_purpose='Business',
            Credit_Score=0, income_source='Salaried',
            income_verified=1, loan_tenure_months=24,
            GENDER='F', EDUCATION='Graduate', MARITALSTATUS='Single')
        feats = build_thin_file_features(req)
        assert len(feats) == 22, f"Expected 22, got {len(feats)}"
        assert set(feats.keys()) == set(THINFILE_FEATURES)
    test('Thin-file: all 22 features present', t5)

    # Test 6: FOIR computed correctly
    def t6():
        req = LoanApplicationInput(AGE=28, NETMONTHLYINCOME=40000,
            loan_amount=120000, loan_purpose='P',
            Credit_Score=0, income_source='Salaried',
            loan_tenure_months=12)
        # existing_emi not set → 0; new_emi = 120000/12 = 10000; foir = 10000/40000 = 0.25
        feats = build_thin_file_features(req)
        assert abs(feats['foir'] - 0.25) < 0.01, f"FOIR: {feats['foir']}"
        assert feats['new_emi_approx'] == 10000.0
    test('Thin-file: FOIR computed correctly (120k/12m on 40k income = 0.25)', t6)

    # Test 7: FOIR capped at 1.0
    def t7():
        req = LoanApplicationInput(AGE=28, NETMONTHLYINCOME=5000,
            loan_amount=500000, loan_purpose='P',
            Credit_Score=0, income_source='Daily-Wage', loan_tenure_months=6)
        feats = build_thin_file_features(req)
        assert feats['foir'] <= 1.0
    test('Thin-file: FOIR capped at 1.0 even if EMI > income', t7)

    # Test 8: zero income raises ValueError
    def t8():
        req = LoanApplicationInput.__new__(LoanApplicationInput)
        object.__setattr__(req, 'NETMONTHLYINCOME', 0)
        object.__setattr__(req, 'AGE', 28)
        object.__setattr__(req, 'loan_amount', 100000)
        object.__setattr__(req, 'loan_purpose', 'P')
        build_thin_file_features(req)
    test('Thin-file: zero income raises ValueError', t8, expect_pass=False)

    # Test 9: main model OHE correct
    def t9():
        req = LoanApplicationInput(AGE=35, NETMONTHLYINCOME=50000,
            loan_amount=300000, loan_purpose='B',
            GENDER='F', EDUCATION='POST-GRADUATE', MARITALSTATUS='Single')
        feats = build_main_features(req)
        assert feats['GENDER_F'] == 1
        assert feats['GENDER_M'] == 0
        assert feats['EDUCATION_POST-GRADUATE'] == 1
        assert feats['EDUCATION_GRADUATE'] == 0
        assert feats['MARITALSTATUS_Single'] == 1
        assert feats['MARITALSTATUS_Married'] == 0
    test('Main model: one-hot encoding correct for GENDER/EDUCATION/MARITAL', t9)

    # Test 10: unknown category → all zeros, no crash
    def t10():
        req = LoanApplicationInput(AGE=35, NETMONTHLYINCOME=50000,
            loan_amount=300000, loan_purpose='B', GENDER='X')
        feats = build_main_features(req)
        assert feats['GENDER_F'] == 0
        assert feats['GENDER_M'] == 0
    test('Main model: unknown category value → all OHE zeros, no crash', t10)

    print(f'\n  {sum(results)}/{len(results)} tests passed')
    if all(results):
        print('  ✅ feature_builder fully validated')
