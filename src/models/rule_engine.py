"""
rule_engine.py — Phase 4, Prompt 4.1
Hard override rules that run BEFORE any model prediction.
Returns (forced_decision, rule_fired, reason_text) or None.

Rules (in priority order):
  1. num_lss > 0              → P1 (loss-classified accounts)
  2. Credit_Score < 550       → P1 (below minimum threshold)
  3. NETMONTHLYINCOME <= 0    → P1 (no income)
  4. num_dbt > 0              → P4 (doubtful accounts — human review)
  5. recent_stress=1 + P2     → P3 (downgrade clean approval)
  6. FOIR > 0.60              → P1 (unaffordable)
  7. FOIR > 0.50              → P3 (conditional — verify affordability)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RULE_MIN_CREDIT_SCORE,
    RULE_MAX_FOIR_REJECT,
    RULE_MAX_FOIR_COND,
    RULE_THINFILE_MAX_LOAN,
    RULE_THINFILE_MAX_FOIR,
)


class RuleEngine:
    """
    Non-negotiable hard rules that override model output completely.
    Instantiate once, call .evaluate(features, model_prediction) per application.
    """

    def evaluate(self, features: dict, model_prediction: str = None):
        """
        Evaluate all rules against the application feature dict.

        Args:
            features: dict of raw feature values for one application
            model_prediction: optional — 'P1'/'P2'/'P3'/'P4' from model
                              Required for Rule 5 (downgrade logic)

        Returns:
            (decision, rule_name, reason_text)  if a rule fires
            None                                if no rule fires
        """

        # ── Rule 1: Loss-classified accounts ─────────────────────────────────
        num_lss = float(features.get('num_lss', 0) or 0)
        if num_lss > 0:
            return (
                'P1',
                'RULE_01_LOSS_ACCOUNT',
                f'Application automatically rejected: applicant has '
                f'{int(num_lss)} loss-classified account(s) in their '
                f'credit history. Loss accounts are an immediate '
                f'disqualifier under bank credit policy.'
            )

        # ── Rule 2: Credit score below minimum ───────────────────────────────
        credit_score = float(features.get('Credit_Score', 0) or 0)
        if 0 < credit_score < RULE_MIN_CREDIT_SCORE:
            return (
                'P1',
                'RULE_02_LOW_CREDIT_SCORE',
                f'Application automatically rejected: CIBIL score of '
                f'{int(credit_score)} is below the minimum threshold of '
                f'{RULE_MIN_CREDIT_SCORE}. The applicant should work on '
                f'improving their credit score before reapplying.'
            )

        # ── Rule 3: No income ─────────────────────────────────────────────────
        income = float(features.get('NETMONTHLYINCOME', 0) or 0)
        if income <= 0:
            return (
                'P1',
                'RULE_03_NO_INCOME',
                f'Application automatically rejected: net monthly income '
                f'is zero or negative (₹{income:,.0f}). A verified income '
                f'source is required to assess repayment capacity.'
            )

        # ── Rule 4: Doubtful accounts ─────────────────────────────────────────
        num_dbt = float(features.get('num_dbt', 0) or 0)
        if num_dbt > 0:
            return (
                'P4',
                'RULE_04_DOUBTFUL_ACCOUNT',
                f'Application referred to Credit Committee: applicant has '
                f'{int(num_dbt)} doubtful account(s) requiring manual review. '
                f'Doubtful accounts indicate potential recovery risk and '
                f'cannot be auto-decisioned.'
            )

        # ── Rule 5: Recent stress downgrade ───────────────────────────────────
        recent_stress = int(features.get('recent_stress', 0) or 0)
        if recent_stress == 1 and model_prediction == 'P2':
            return (
                'P3',
                'RULE_05_RECENT_STRESS_DOWNGRADE',
                f'Approval downgraded to Conditional: applicant shows '
                f'recent delinquency signals despite a passing credit score. '
                f'Loan officer should verify current repayment status and '
                f'apply income/collateral conditions before disbursement.'
            )

        # ── Rule 6: FOIR too high — reject ────────────────────────────────────
        foir = float(features.get('foir', 0) or 0)
        if foir > RULE_MAX_FOIR_REJECT:
            return (
                'P1',
                'RULE_06_FOIR_REJECT',
                f'Application automatically rejected: Fixed Obligation to '
                f'Income Ratio (FOIR) of {foir:.1%} exceeds the maximum '
                f'permissible limit of {RULE_MAX_FOIR_REJECT:.0%}. '
                f'Applicant cannot afford additional EMI obligations.'
            )

        # ── Rule 7: FOIR elevated — conditional ───────────────────────────────
        if foir > RULE_MAX_FOIR_COND:
            return (
                'P3',
                'RULE_07_FOIR_CONDITIONAL',
                f'Approval conditionally approved: FOIR of {foir:.1%} is '
                f'elevated (threshold: {RULE_MAX_FOIR_COND:.0%}). Loan '
                f'officer must verify income documents and may need to '
                f'reduce loan amount or tenure to bring FOIR below 50%.'
            )

        # ── Thin-file policy limits ────────────────────────────────────────────
        # Only applies when Credit_Score == 0 (thin-file applicants)
        if credit_score == 0:
            loan_amount = float(features.get('loan_amount', 0) or
                                features.get('loan_amount_requested', 0) or 0)
            if loan_amount > RULE_THINFILE_MAX_LOAN:
                return (
                    'P1',
                    'RULE_08_THINFILE_LOAN_LIMIT',
                    f'Application rejected: loan amount of ₹{loan_amount:,.0f} '
                    f'exceeds the maximum limit of ₹{RULE_THINFILE_MAX_LOAN:,} '
                    f'for first-time borrowers with no credit history. '
                    f'Please apply for a smaller amount.'
                )

            thinfile_foir = float(features.get('foir', 0) or 0)
            if thinfile_foir > RULE_THINFILE_MAX_FOIR:
                return (
                    'P3',
                    'RULE_09_THINFILE_FOIR',
                    f'Thin-file application conditionally approved: FOIR of '
                    f'{thinfile_foir:.1%} exceeds the {RULE_THINFILE_MAX_FOIR:.0%} '
                    f'limit for first-time borrowers. Income must be '
                    f'independently verified before disbursement.'
                )

        return None  # No rule fired — let the model decide

    def explain(self, features: dict, model_prediction: str = None) -> str:
        """
        Return plain-English explanation for a loan officer.
        If no rule fires, returns empty string.
        """
        result = self.evaluate(features, model_prediction)
        if result is None:
            return ''
        _, rule_name, reason = result
        return reason

    def get_all_fired(self, features: dict, model_prediction: str = None) -> list:
        """
        Return list of all rule names that would fire.
        (Useful for audit logs — records all violations, not just first.)
        """
        fired = []
        checks = [
            ('num_lss', lambda f: float(f.get('num_lss', 0) or 0) > 0,
             'RULE_01_LOSS_ACCOUNT'),
            ('Credit_Score', lambda f: 0 < float(f.get('Credit_Score', 0) or 0) < RULE_MIN_CREDIT_SCORE,
             'RULE_02_LOW_CREDIT_SCORE'),
            ('NETMONTHLYINCOME', lambda f: float(f.get('NETMONTHLYINCOME', 0) or 0) <= 0,
             'RULE_03_NO_INCOME'),
            ('num_dbt', lambda f: float(f.get('num_dbt', 0) or 0) > 0,
             'RULE_04_DOUBTFUL_ACCOUNT'),
            ('foir_reject', lambda f: float(f.get('foir', 0) or 0) > RULE_MAX_FOIR_REJECT,
             'RULE_06_FOIR_REJECT'),
            ('foir_cond', lambda f: RULE_MAX_FOIR_COND < float(f.get('foir', 0) or 0) <= RULE_MAX_FOIR_REJECT,
             'RULE_07_FOIR_CONDITIONAL'),
        ]
        for _, condition, rule_name in checks:
            try:
                if condition(features):
                    fired.append(rule_name)
            except Exception:
                pass
        return fired
