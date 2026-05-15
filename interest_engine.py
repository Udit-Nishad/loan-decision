"""
interest_engine.py — Loan Interest Rate Calculator

Rate = Repo Rate + Operational Margin + Risk Premium

  Repo Rate:          5.50%  (RBI policy rate — update via REPO_RATE constant)
  Operational Margin: 3.00%  (fixed bank cost — update via OP_MARGIN constant)
  Risk Premium:       varies by decision class × loan purpose × applicant profile

Risk Premium bands (basis points above base):
  P2 (Approve):        75–300 bps depending on loan type + credit score
  P3 (Conditional):   250–450 bps — elevated due to conditional approval
  P4 (Refer):         350–500 bps — credit committee level risk
  P1 (Reject):        N/A — no rate offered

Loan purpose multipliers applied on top of the class-level premium:
  Home Loan / Home Renovation  — lowest risk  (secured)   → −50 bps
  Gold Loan                    — very low risk (secured)   → −75 bps
  Vehicle                      — moderate     (semi-secured) → +0 bps
  Education                    — moderate     (productive use) → +25 bps
  Business                     — higher       (income volatility) → +75 bps
  Personal / Medical / Other   — highest      (unsecured)  → +125 bps
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_RATE  = 5.50   # % — update when RBI changes policy rate
OP_MARGIN  = 3.00   # % — bank operational spread (fixed)

# Base risk premium per decision class (%)
_BASE_RISK_PREMIUM = {
    'P2': 1.50,   # 150 bps — standard approval
    'P3': 3.25,   # 325 bps — conditional approval, elevated risk
    'P4': 4.25,   # 425 bps — refer, high risk
    'P1': None,   # no rate — rejected
}

# Adjustment per loan purpose (percentage points, + = higher rate)
_PURPOSE_ADJ = {
    'gold loan':        -0.75,
    'gold':             -0.75,
    'home loan':        -0.50,
    'home renovation':  -0.50,
    'home':             -0.50,
    'vehicle':           0.00,
    'car':               0.00,
    'two wheeler':       0.00,
    'education':         0.25,
    'medical':           0.75,
    'business':          0.75,
    'personal':          1.25,
    'other':             1.25,
}

# Credit score bonus: good score lowers risk premium
# Applied only for P2; P3/P4 get smaller relief
def _score_adj(credit_score: int, decision: str) -> float:
    """Return negative adjustment (rate relief) for strong credit score."""
    if credit_score == 0:   # thin-file — no CIBIL
        return 0.25         # slight premium for unknown credit
    if credit_score >= 750:
        return -0.50 if decision == 'P2' else -0.25
    if credit_score >= 700:
        return -0.25 if decision == 'P2' else -0.10
    if credit_score >= 650:
        return 0.00
    if credit_score >= 600:
        return 0.25
    return 0.50             # sub-600 score


# Thin-file income source adjustment
_INCOME_ADJ = {
    'salaried':      0.00,
    'self-employed': 0.50,
    'gig':           0.75,
    'daily-wage':    1.00,
}

# Tenure adjustment: longer tenure = higher risk
def _tenure_adj(months: int) -> float:
    if months <= 12:  return -0.25
    if months <= 24:  return  0.00
    if months <= 36:  return  0.10
    if months <= 48:  return  0.20
    return 0.30                      # 60 months


@dataclass
class InterestRateBreakdown:
    decision:           str
    loan_purpose:       str
    loan_amount:        float
    loan_tenure_months: int
    credit_score:       int

    repo_rate:          float        # always 5.50%
    operational_margin: float        # always 3.00%
    base_risk_premium:  float        # class-level base
    purpose_adj:        float        # loan purpose adjustment
    score_adj:          float        # credit score relief/penalty
    tenure_adj:         float        # tenure adjustment
    income_adj:         float        # thin-file income source adj
    total_risk_premium: float        # sum of all risk components

    annual_rate:        float        # final rate = repo + op + risk
    monthly_rate:       float        # annual / 12
    emi:                float        # standard EMI
    total_payable:      float        # EMI × tenure
    total_interest:     float        # total_payable − loan_amount
    effective_cost:     float        # total_interest / loan_amount × 100

    rate_band:          str          # BEST / STANDARD / ELEVATED / HIGH
    rate_rationale:     str          # one-line explanation


def calculate(
    decision:           str,
    loan_purpose:       str,
    loan_amount:        float,
    loan_tenure_months: int,
    credit_score:       int  = 0,
    income_source:      str  = 'salaried',
    default_probability: Optional[float] = None,
) -> Optional[InterestRateBreakdown]:
    """
    Calculate the all-in interest rate for an approved/conditional application.
    Returns None for P1 (rejected) decisions.

    Args:
        decision:            'P1'|'P2'|'P3'|'P4'
        loan_purpose:        free-text purpose from application
        loan_amount:         in INR
        loan_tenure_months:  6|12|24|36|48|60
        credit_score:        CIBIL score (0 = no bureau record)
        income_source:       for thin-file path
        default_probability: proxy PD from default model (optional)

    Returns:
        InterestRateBreakdown or None
    """
    if decision == 'P1':
        return None

    base_rp = _BASE_RISK_PREMIUM[decision]

    # Purpose adjustment
    purpose_key = (loan_purpose or 'personal').lower().strip()
    p_adj = _PURPOSE_ADJ.get(purpose_key, 1.25)  # default: personal rate
    # Try partial match
    if purpose_key not in _PURPOSE_ADJ:
        for k, v in _PURPOSE_ADJ.items():
            if k in purpose_key:
                p_adj = v
                break

    # Score adjustment
    s_adj = _score_adj(credit_score, decision)

    # Tenure adjustment
    t_adj = _tenure_adj(loan_tenure_months)

    # Income source adjustment (thin-file)
    inc_key = (income_source or 'salaried').lower().strip()
    i_adj   = _INCOME_ADJ.get(inc_key, 0.50)

    # If we have a default probability, use it to fine-tune risk premium
    # PD > 40% → +50 bps; PD 20–40% → +25 bps; PD < 20% → 0
    dp_adj = 0.0
    if default_probability is not None:
        if default_probability > 0.40:
            dp_adj = 0.50
        elif default_probability > 0.20:
            dp_adj = 0.25

    total_risk = round(base_rp + p_adj + s_adj + t_adj + i_adj + dp_adj, 2)
    # Floor: never below 0.50% risk premium
    total_risk = max(total_risk, 0.50)

    annual_rate   = round(REPO_RATE + OP_MARGIN + total_risk, 2)
    monthly_rate  = annual_rate / 100 / 12
    n             = loan_tenure_months

    # Standard reducing balance EMI formula
    if monthly_rate > 0:
        emi = loan_amount * monthly_rate * (1 + monthly_rate)**n / ((1 + monthly_rate)**n - 1)
    else:
        emi = loan_amount / n

    emi           = round(emi, 2)
    total_payable = round(emi * n, 2)
    total_interest= round(total_payable - loan_amount, 2)
    effective_cost= round(total_interest / loan_amount * 100, 2)

    # Rate band
    if annual_rate < 10.5:
        band = 'BEST'
    elif annual_rate < 13.0:
        band = 'STANDARD'
    elif annual_rate < 16.0:
        band = 'ELEVATED'
    else:
        band = 'HIGH'

    # Rationale
    parts = [f'Base: {REPO_RATE}% repo + {OP_MARGIN}% margin + {base_rp}% class risk']
    if p_adj != 0:
        sign = '+' if p_adj > 0 else ''
        parts.append(f'{sign}{p_adj}% for {loan_purpose}')
    if s_adj != 0:
        sign = '+' if s_adj > 0 else ''
        parts.append(f'{sign}{s_adj}% credit score adjustment')
    if dp_adj > 0:
        parts.append(f'+{dp_adj}% default probability premium')
    if t_adj != 0:
        parts.append(f'+{t_adj}% tenure adjustment')

    return InterestRateBreakdown(
        decision=decision,
        loan_purpose=loan_purpose,
        loan_amount=loan_amount,
        loan_tenure_months=loan_tenure_months,
        credit_score=credit_score,
        repo_rate=REPO_RATE,
        operational_margin=OP_MARGIN,
        base_risk_premium=base_rp,
        purpose_adj=p_adj,
        score_adj=s_adj,
        tenure_adj=t_adj,
        income_adj=i_adj,
        total_risk_premium=total_risk,
        annual_rate=annual_rate,
        monthly_rate=round(monthly_rate * 100, 4),
        emi=emi,
        total_payable=total_payable,
        total_interest=total_interest,
        effective_cost=effective_cost,
        rate_band=band,
        rate_rationale='  ·  '.join(parts),
    )
