"""
schemas.py — Phase 7, Prompt 7.1
Data contracts for the loan decisioning API.

Written with dataclasses + manual validators so it runs without pydantic
in this environment. When deployed locally with pydantic installed,
swap the base class to BaseModel — field names and validators are identical.

Covers:
  - LoanApplicationInput  (main + thin-file fields, cross-field validators)
  - PredictionOutput
  - BatchPredictRequest / BatchPredictResponse / BatchSummary
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    def __init__(self, errors: List[Dict]):
        self.errors = errors
        msg = "; ".join(f"{e['field']}: {e['msg']}" for e in errors)
        super().__init__(msg)


def _check(condition: bool, field: str, msg: str, errors: list):
    if not condition:
        errors.append({"field": field, "msg": msg})


# ─────────────────────────────────────────────────────────────────────────────
# LoanApplicationInput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoanApplicationInput:
    # ── Required core fields ──────────────────────────────────────────────────
    AGE:              float
    NETMONTHLYINCOME: float
    loan_amount:      float
    loan_purpose:     str

    # ── Optional demographics ─────────────────────────────────────────────────
    GENDER:           Optional[str]   = None
    MARITALSTATUS:    Optional[str]   = None
    EDUCATION:        Optional[str]   = None
    CITY:             Optional[str]   = None

    # ── CIBIL bureau fields (all optional — absent = new/thin-file) ───────────
    Credit_Score:                    Optional[float] = None
    num_std:                         Optional[float] = None
    num_sub:                         Optional[float] = None
    num_dbt:                         Optional[float] = None
    num_lss:                         Optional[float] = None
    num_times_delinquent:            Optional[float] = None
    max_recent_level_of_deliq:       Optional[float] = None
    max_delinquency_level:           Optional[float] = None
    num_times_60p_dpd:               Optional[float] = None
    pct_of_active_TLs_ever:          Optional[float] = None
    time_since_recent_payment:       Optional[float] = None
    time_since_recent_deliquency:    Optional[float] = None
    time_since_first_deliquency:     Optional[float] = None
    time_since_recent_enq:           Optional[float] = None
    enq_L3m:                         Optional[float] = None
    enq_L6m:                         Optional[float] = None
    enq_L12m:                        Optional[float] = None
    tot_enq:                         Optional[float] = None
    pct_PL_enq_L6m_of_L12m:         Optional[float] = None
    pct_PL_enq_L6m_of_ever:         Optional[float] = None
    pct_CC_enq_L6m_of_L12m:         Optional[float] = None
    pct_CC_enq_L6m_of_ever:         Optional[float] = None
    last_prod_enq2:                  Optional[str]   = None
    first_prod_enq2:                 Optional[str]   = None
    CC_Flag:                         Optional[int]   = None
    PL_Flag:                         Optional[int]   = None
    AGE_cibil:                       Optional[float] = None
    GENDER_cibil:                    Optional[str]   = None
    Time_With_Curr_Empr:             Optional[float] = None
    recent_stress:                   Optional[int]   = None

    # ── Internal TL fields (default=0 → "no trade line") ─────────────────────
    ACCOUNT_TYPE:            float = 0
    Total_TL:                float = 0
    Tot_Closed_TL:           float = 0
    Tot_Active_TL:           float = 0
    Total_TL_opened_L6M:     float = 0
    Tot_TL_closed_L6M:       float = 0
    pct_tl_open_L6M:         float = 0
    pct_tl_closed_L6M:       float = 0
    pct_active_tl_open_L6M:  float = 0
    pct_closed_tl_open_L6M:  float = 0
    Total_TL_opened_L12M:    float = 0
    Tot_TL_closed_L12M:      float = 0
    pct_tl_open_L12M:        float = 0
    pct_tl_closed_L12M:      float = 0
    pct_active_tl_open_L12M: float = 0
    pct_closed_tl_open_L12M: float = 0
    Tot_Missed_Pmnt:         float = 0
    Auto_TL:                 float = 0
    CC_TL:                   float = 0
    Consumer_TL:             float = 0
    Gold_TL:                 float = 0
    Home_TL:                 float = 0
    PL_TL:                   float = 0
    Secured_TL:              float = 0
    Unsecured_TL:            float = 0
    Age_Oldest_TL:           float = 0
    Age_Newest_TL:           float = 0

    # ── Thin-file specific fields (default=None → not applicable for CIBIL) ───
    income_source:          Optional[str]   = None
    income_verified:        Optional[int]   = None
    salary_credit_months:   Optional[int]   = None
    avg_monthly_balance:    Optional[float] = None
    utility_payment_score:  Optional[float] = None
    upi_monthly_txn_count:  Optional[int]   = None
    residence_stability_yrs:Optional[float] = None
    has_mobile_verified:    Optional[int]   = None
    has_aadhaar_linked:     Optional[int]   = None
    loan_tenure_months:     Optional[int]   = None
    residence_type:         Optional[str]   = None
    employment_months:      Optional[float] = None
    existing_emi:           Optional[float] = None

    # ── Computed / passed through ─────────────────────────────────────────────
    PROSPECTID: Optional[int] = None

    def __post_init__(self):
        """Cross-field validation — runs automatically on construction."""
        errors = []

        # Rule 1: income required
        _check(self.NETMONTHLYINCOME > 0,
               "NETMONTHLYINCOME", "Must be greater than 0", errors)

        # Rule 2: age sanity
        _check(18 <= self.AGE <= 75,
               "AGE", "Must be between 18 and 75", errors)

        # Rule 3: loan amount required
        _check(self.loan_amount > 0,
               "loan_amount", "Must be greater than 0", errors)

        # Rule 4: impossible combination — score=0 but TL > 0
        cs = self.Credit_Score or 0
        if cs == 0 and self.Total_TL > 0:
            errors.append({
                "field": "Credit_Score/Total_TL",
                "msg": "Credit_Score=0 with Total_TL>0 is impossible: "
                       "a bureau score of 0 means no credit history, "
                       "but Total_TL>0 means trade lines exist."
            })

        # Rule 5: thin-file loan cap = ₹10,00,000
        if cs == 0 and self.loan_amount > 1_000_000:
            errors.append({
                "field": "loan_amount",
                "msg": f"Thin-file applicants (Credit_Score=0) are limited "
                       f"to ₹10,00,000. Requested: ₹{self.loan_amount:,.0f}"
            })

        # Rule 6: tenure validation for thin-file
        if self.loan_tenure_months is not None:
            _check(self.loan_tenure_months in [6, 12, 24, 36, 48, 60],
                   "loan_tenure_months",
                   "Must be one of: 6, 12, 24, 36, 48, 60 months", errors)

        if errors:
            raise ValidationError(errors)

    @classmethod
    def from_dict(cls, d: dict) -> "LoanApplicationInput":
        """Construct from a raw dict (e.g. JSON body), ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def is_thin_file(self) -> bool:
        """True if this applicant has no CIBIL history."""
        return (self.Credit_Score is None or self.Credit_Score == 0)

    def to_feature_dict(self) -> dict:
        """Flat dict of all fields for downstream feature builders."""
        import dataclasses
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionOutput:
    decision:            str           # P1 / P2 / P3 / P4
    probability:         float         # confidence in the decision class
    confidence:          str           # HIGH / MEDIUM / LOW
    model_used:          str           # main_ensemble / thin_file / rule_engine
    rules_fired:         List[str]     # list of rule IDs that fired
    explanation:         str           # plain-English reason for the decision
    improvement_actions: List[str]     # actionable steps for P1/P3
    default_probability: Optional[float] = None   # from proxy default model
    prospectid:          Optional[int]   = None
    processing_ms:       Optional[float] = None

    DECISION_LABELS = {
        "P1": "Reject",
        "P2": "Approve",
        "P3": "Conditional Approve",
        "P4": "Refer to Credit Committee",
    }

    def label(self) -> str:
        return self.DECISION_LABELS.get(self.decision, self.decision)

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["decision_label"] = self.label()
        return d


def make_confidence(probability: float) -> str:
    if probability >= 0.85:
        return "HIGH"
    elif probability >= 0.65:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Batch schemas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchSummary:
    total:          int
    approved:       int   # P2
    conditional:    int   # P3
    referred:       int   # P4
    rejected:       int   # P1
    thin_file_count:int
    main_model_count:int
    rule_engine_count:int
    processing_ms:  float
    approval_rate:  float  # (P2 + P3) / total

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class BatchPredictRequest:
    applications: List[LoanApplicationInput]

    def __post_init__(self):
        if len(self.applications) == 0:
            raise ValidationError([{"field": "applications",
                                    "msg": "Batch must contain at least 1 application"}])
        if len(self.applications) > 1000:
            raise ValidationError([{"field": "applications",
                                    "msg": "Batch limit is 1000 applications per request"}])

    @classmethod
    def from_list(cls, records: list) -> "BatchPredictRequest":
        apps = [LoanApplicationInput.from_dict(r) for r in records]
        return cls(applications=apps)


@dataclass
class BatchPredictResponse:
    results:  List[PredictionOutput]
    summary:  BatchSummary

    @classmethod
    def build(cls, outputs: List[PredictionOutput],
              start_time: float) -> "BatchPredictResponse":
        elapsed = (time.time() - start_time) * 1000
        total   = len(outputs)
        summary = BatchSummary(
            total=total,
            approved=sum(1 for o in outputs if o.decision == "P2"),
            conditional=sum(1 for o in outputs if o.decision == "P3"),
            referred=sum(1 for o in outputs if o.decision == "P4"),
            rejected=sum(1 for o in outputs if o.decision == "P1"),
            thin_file_count=sum(1 for o in outputs if "thin_file" in o.model_used),
            main_model_count=sum(1 for o in outputs if o.model_used == "main_ensemble"),
            rule_engine_count=sum(1 for o in outputs if o.model_used == "rule_engine"),
            processing_ms=round(elapsed, 1),
            approval_rate=round(
                sum(1 for o in outputs if o.decision in ["P2","P3"]) / max(total, 1), 4),
        )
        for o in outputs:
            o.processing_ms = round(elapsed / total, 2)
        return cls(results=outputs, summary=summary)

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary.to_dict(),
        }
