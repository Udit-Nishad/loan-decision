from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = DATA_DIR / "models"

RAW_CIBIL_FILE    = RAW_DIR / "External_Cibil_Dataset.xlsx"
RAW_INTERNAL_FILE = RAW_DIR / "Internal_Bank_Dataset.xlsx"
RAW_UNSEEN_FILE   = RAW_DIR / "Unseen_Dataset.xlsx"
RAW_THINFILE_FILE = RAW_DIR / "synthetic_thin_file.csv"

CIBIL_SHEET    = "case_study2"
INTERNAL_SHEET = "case_study1"
UNSEEN_SHEET   = "Sheet1"

CLEANED_FILE  = PROCESSED_DIR / "cleaned.csv"
FEATURED_FILE = PROCESSED_DIR / "featured.csv"

ENSEMBLE_MODEL_FILE = MODELS_DIR / "final_ensemble.joblib"
THINFILE_MODEL_FILE = MODELS_DIR / "thin_file_model.joblib"
SCORECARD_FILE      = MODELS_DIR / "scorecard.joblib"
TRANSFORMER_FILE    = MODELS_DIR / "transformer.joblib"

TARGET_COL  = "Approved_Flag"
TARGET_MAP  = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
TARGET_IMAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}

TL_SENTINEL_COLS = ["Age_Oldest_TL", "Age_Newest_TL"]
TL_SENTINEL_VAL  = -99999

OUTLIER_COLS = {
    "NETMONTHLYINCOME": 0.99,
    "Gold_TL":          0.99,
    "Total_TL":         0.99,
    "Unsecured_TL":     0.99,
}
HARD_CAPS = {"Gold_TL": 24, "Total_TL": 34}

TL_COLS = [
    "ACCOUNT_TYPE","Total_TL","Tot_Closed_TL","Tot_Active_TL",
    "Total_TL_opened_L6M","Tot_TL_closed_L6M",
    "pct_tl_open_L6M","pct_tl_closed_L6M",
    "pct_active_tl_open_L6M","pct_closed_tl_open_L6M",
    "Total_TL_opened_L12M","Tot_TL_closed_L12M",
    "pct_tl_open_L12M","pct_tl_closed_L12M",
    "pct_active_tl_open_L12M","pct_closed_tl_open_L12M",
    "Tot_Missed_Pmnt",
    "Auto_TL","CC_TL","Consumer_TL","Gold_TL",
    "Home_TL","PL_TL","Secured_TL","Unsecured_TL",
]

ENGINEERED_FEATURES = ["missed_payment_rate","tl_utilization_rate","credit_history_span"]
CAT_COLS = ["EDUCATION","MARITALSTATUS","GENDER","last_prod_enq2","first_prod_enq2"]

THINFILE_FEATURES = [
    "person_age","gender","education","marital_status","residence_type",
    "income_source","monthly_income","employment_months","income_verified",
    "salary_credit_months","avg_monthly_balance","utility_payment_score",
    "upi_monthly_txn_count","existing_emi","loan_purpose",
    "loan_amount_requested","loan_tenure_months","new_emi_approx","foir",
    "residence_stability_yrs","has_mobile_verified","has_aadhaar_linked",
]
THINFILE_TARGET   = "creditworthy"
THINFILE_CAT_COLS = ["gender","education","marital_status",
                     "residence_type","income_source","loan_purpose"]

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_STATE = 42

RULE_MIN_CREDIT_SCORE  = 550
RULE_MAX_FOIR_REJECT   = 0.60
RULE_MAX_FOIR_COND     = 0.50
RULE_THINFILE_MAX_LOAN = 1_000_000
RULE_THINFILE_MAX_FOIR = 0.30

# ── Plain-English feature labels (for SHAP / decision reports) ────────────────
FEATURE_PLAIN_ENGLISH = {
    'Credit_Score':                  'CIBIL Credit Score',
    'NETMONTHLYINCOME':              'Net Monthly Income (₹)',
    'AGE':                           'Applicant Age',
    'Time_With_Curr_Empr':           'Tenure with Current Employer (months)',
    'num_std':                       'Standard (Performing) Accounts',
    'num_sub':                       'Sub-Standard Accounts',
    'num_dbt':                       'Doubtful Accounts',
    'num_lss':                       'Loss-Classified Accounts',
    'num_times_delinquent':          'Total Delinquency Count',
    'num_times_60p_dpd':             'Times 60+ Days Past Due',
    'max_delinquency_level':         'Highest Delinquency Level Ever',
    'max_recent_level_of_deliq':     'Most Recent Delinquency Level',
    'Tot_Missed_Pmnt':               'Total Missed Payments (Internal)',
    'missed_payment_rate':           'Missed Payment Rate',
    'credit_history_span':           'Credit History Span (months)',
    'tl_utilization_rate':           'Trade Line Utilisation Rate',
    'pct_of_active_TLs_ever':        '% Active Trade Lines (Ever)',
    'enq_L3m':                       'Enquiries in Last 3 Months',
    'enq_L6m':                       'Enquiries in Last 6 Months',
    'enq_L12m':                      'Enquiries in Last 12 Months',
    'time_since_recent_enq':         'Months Since Last Enquiry',
    'time_since_recent_payment':     'Months Since Last Payment',
    'time_since_recent_deliquency':  'Months Since Last Delinquency',
    'Total_TL':                      'Total Trade Lines',
    'Tot_Active_TL':                 'Active Trade Lines',
    'Tot_Closed_TL':                 'Closed Trade Lines',
    'Gold_TL':                       'Gold Loan Trade Lines',
    'Secured_TL':                    'Secured Trade Lines',
    'Unsecured_TL':                  'Unsecured Trade Lines',
    'pct_PL_enq_L6m_of_L12m':        '% Personal Loan Enquiries (6m/12m)',
    'pct_CC_enq_L6m_of_L12m':        '% Credit Card Enquiries (6m/12m)',
    # Thin-file
    'income_source':                 'Income Source',
    'income_verified':               'Income Verified (0/1)',
    'salary_credit_months':          'Months of Salary Credits Observed',
    'avg_monthly_balance':           'Average Monthly Balance (₹)',
    'utility_payment_score':         'Utility Payment Score (0–1)',
    'upi_monthly_txn_count':         'UPI Transactions per Month',
    'foir':                          'Fixed Obligation to Income Ratio (FOIR)',
    'has_aadhaar_linked':            'Aadhaar Linked (0/1)',
    'monthly_income':                'Monthly Income (₹)',
}

# ── Origination logging ───────────────────────────────────────────────────────
import os
DATA_DIR   = Path(__file__).resolve().parents[2] / 'data'
LOG_DB     = DATA_DIR / 'origination_log.db'
REJECT_LOG = DATA_DIR / 'rejection_log.db'
