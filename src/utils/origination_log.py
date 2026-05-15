"""
origination_log.py — Phase 11

SQLite-backed origination log for every loan application decision.

TWO DATABASES:
  origination_log.db  — full decision record for every application
  rejection_log.db    — extended record for rejections (P1) and conditionals (P3/P4)
                        used to instrument reject inference for future model retraining

WHY THIS MATTERS:
  The current default model uses Tot_Missed_Pmnt as a proxy target because we have
  no observed 90+ DPD outcomes for rejected applicants. Once origination logging
  runs for 12–18 months, we will have:
    - Re-application records (rejected applicants who applied again elsewhere)
    - Bureau pull refresh showing subsequent delinquency
    - Outcome linkage via PROSPECTID across time
  This is the primary upgrade path from the proxy model to a true forward-looking
  default model.

SCHEMA:
  applications table (origination_log.db):
    - One row per decision
    - All input features (JSON), decision, probabilities, rules fired
    - Timestamps, model version, processing time

  rejections table (rejection_log.db):
    - One row per P1/P3/P4 decision
    - All rejection reasons (rules + top risk factors)
    - Re-application tracking fields
    - Bureau refresh scheduled date
    - Future outcome fields (populated by the retraining pipeline)
"""
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import LOG_DB, REJECT_LOG
from src.utils.logger import get_logger

log = get_logger('OriginationLog')

MODEL_VERSION = '1.0.0-phase11'

# ── Schema definitions ────────────────────────────────────────────────────────

_APPLICATIONS_DDL = """
CREATE TABLE IF NOT EXISTS applications (
    id                   TEXT PRIMARY KEY,
    prospectid           INTEGER,
    applicant_name       TEXT,
    application_ts       TEXT NOT NULL,
    loan_amount          REAL NOT NULL,
    loan_purpose         TEXT,
    loan_tenure_months   INTEGER,

    -- Decision
    decision             TEXT NOT NULL CHECK(decision IN ('P1','P2','P3','P4')),
    decision_label       TEXT NOT NULL,
    probability          REAL NOT NULL,
    confidence           TEXT NOT NULL,
    model_used           TEXT NOT NULL,
    rules_fired          TEXT,           -- JSON list
    default_probability  REAL,
    explanation          TEXT,

    -- Key features (denormalised for quick querying)
    credit_score         INTEGER,
    net_monthly_income   REAL,
    age                  INTEGER,
    total_tl             INTEGER,
    tot_missed_pmnt      INTEGER,
    num_lss              INTEGER,
    num_dbt              INTEGER,
    foir                 REAL,

    -- Full input (JSON)
    input_features       TEXT,           -- JSON dict

    -- Metadata
    model_version        TEXT NOT NULL,
    processing_ms        REAL,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_app_decision   ON applications(decision);
CREATE INDEX IF NOT EXISTS idx_app_ts         ON applications(application_ts);
CREATE INDEX IF NOT EXISTS idx_app_prospectid ON applications(prospectid);
CREATE INDEX IF NOT EXISTS idx_app_model      ON applications(model_used);
"""

_REJECTIONS_DDL = """
CREATE TABLE IF NOT EXISTS rejections (
    id                      TEXT PRIMARY KEY,
    application_id          TEXT NOT NULL,    -- FK to applications.id
    prospectid              INTEGER,
    application_ts          TEXT NOT NULL,
    decision                TEXT NOT NULL,    -- P1, P3, or P4

    loan_amount             REAL,
    loan_purpose            TEXT,

    -- Why rejected / conditional
    primary_rule            TEXT,             -- first rule that fired
    all_rules_fired         TEXT,             -- JSON list
    top_risk_factors        TEXT,             -- JSON list of {feature, direction, magnitude}
    rejection_reason_text   TEXT,             -- human-readable

    -- Key risk snapshot at time of application
    credit_score            INTEGER,
    net_monthly_income      REAL,
    tot_missed_pmnt         INTEGER,
    num_lss                 INTEGER,
    num_dbt                 INTEGER,
    foir                    REAL,
    default_probability     REAL,

    -- Re-application tracking
    reapplication_eligible_date  TEXT,        -- 6 months after rejection for P1
    bureau_refresh_due_date      TEXT,        -- scheduled next bureau pull

    -- Future outcome fields (populated by retraining pipeline 12–18 months later)
    outcome_observed        INTEGER DEFAULT 0,  -- 0=pending, 1=observed
    outcome_defaulted       INTEGER,            -- NULL=unknown, 0=no, 1=yes (90+DPD)
    outcome_observed_at     TEXT,
    outcome_source          TEXT,               -- 'bureau_refresh' | 'reapplication' | 'internal'

    -- Metadata
    model_version           TEXT,
    created_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rej_application_id ON rejections(application_id);
CREATE INDEX IF NOT EXISTS idx_rej_decision        ON rejections(decision);
CREATE INDEX IF NOT EXISTS idx_rej_eligible_date   ON rejections(reapplication_eligible_date);
CREATE INDEX IF NOT EXISTS idx_rej_outcome         ON rejections(outcome_observed);
CREATE INDEX IF NOT EXISTS idx_rej_prospectid      ON rejections(prospectid);
"""

# ── DB helpers ────────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')   # concurrent reads during writes
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_databases():
    """Create both databases and their schemas. Idempotent."""
    with _connect(LOG_DB) as conn:
        conn.executescript(_APPLICATIONS_DDL)
    with _connect(REJECT_LOG) as conn:
        conn.executescript(_REJECTIONS_DDL)
    log.info(f'Databases initialised: {LOG_DB.name}, {REJECT_LOG.name}')


# ── Write ─────────────────────────────────────────────────────────────────────

def log_application(result: dict, input_dict: dict) -> str:
    """
    Write one application decision to origination_log.db.
    Also writes to rejection_log.db for P1/P3/P4.

    Args:
        result:     dict returned by _run_single()
        input_dict: raw input dict (before schema parsing)

    Returns:
        application_id (str)
    """
    init_databases()

    app_id = result.get('application_id', str(uuid.uuid4())[:12].upper())
    now    = datetime.utcnow().isoformat()
    ts     = result.get('timestamp', now)

    # ── Denormalised key features ─────────────────────────────────────────────
    def _f(key, default=None):
        return input_dict.get(key, default)

    foir_val = None
    if _f('NETMONTHLYINCOME') and _f('loan_amount') and _f('loan_tenure_months'):
        try:
            emi = float(_f('loan_amount')) / float(_f('loan_tenure_months'))
            foir_val = round(emi / float(_f('NETMONTHLYINCOME')), 4)
        except (TypeError, ZeroDivisionError):
            pass

    # ── Write to applications ─────────────────────────────────────────────────
    with _connect(LOG_DB) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO applications
            (id, prospectid, applicant_name, application_ts,
             loan_amount, loan_purpose, loan_tenure_months,
             decision, decision_label, probability, confidence,
             model_used, rules_fired, default_probability, explanation,
             credit_score, net_monthly_income, age, total_tl,
             tot_missed_pmnt, num_lss, num_dbt, foir,
             input_features, model_version, processing_ms, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            app_id,
            _f('PROSPECTID'),
            input_dict.get('applicant_name'),
            ts,
            result.get('loan_amount'),
            result.get('loan_purpose'),
            _f('loan_tenure_months'),
            result['decision'],
            result['decision_label'],
            result['probability'],
            result['confidence'],
            result['model_used'],
            json.dumps(result.get('rules_fired', [])),
            result.get('default_probability'),
            result.get('explanation'),
            _f('Credit_Score'),
            _f('NETMONTHLYINCOME'),
            _f('AGE'),
            _f('Total_TL'),
            _f('Tot_Missed_Pmnt'),
            _f('num_lss'),
            _f('num_dbt'),
            foir_val,
            json.dumps({k: v for k, v in input_dict.items()
                        if k not in ('applicant_name',) and v is not None}),
            MODEL_VERSION,
            result.get('processing_ms'),
            now,
        ))

    # ── Write rejections for P1/P3/P4 ────────────────────────────────────────
    if result['decision'] in ('P1', 'P3', 'P4'):
        _log_rejection(app_id, result, input_dict, foir_val, now, ts)

    log.debug(f'Logged [{app_id}] decision={result["decision"]} '
              f'model={result["model_used"]}')
    return app_id


def _log_rejection(app_id: str, result: dict, input_dict: dict,
                   foir_val, now: str, ts: str):
    """Write extended rejection record."""

    rules    = result.get('rules_fired', [])
    factors  = result.get('top_factors', [])
    decision = result['decision']

    primary_rule = rules[0] if rules else None

    # Re-application eligibility: 6 months for P1, 3 months for P3/P4
    months_wait = 6 if decision == 'P1' else 3
    eligible_dt = (datetime.utcnow() + timedelta(days=30 * months_wait)).date().isoformat()
    bureau_dt   = (datetime.utcnow() + timedelta(days=90)).date().isoformat()

    # Top risk factors — only negative ones
    risk_factors = [
        {'feature': f['feature'], 'label': f['label'],
         'magnitude': f['magnitude'], 'value': f.get('value')}
        for f in factors if f.get('direction') == 'negative'
    ][:5]

    def _f(key, default=None):
        return input_dict.get(key, default)

    with _connect(REJECT_LOG) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO rejections
            (id, application_id, prospectid, application_ts, decision,
             loan_amount, loan_purpose,
             primary_rule, all_rules_fired, top_risk_factors,
             rejection_reason_text,
             credit_score, net_monthly_income, tot_missed_pmnt,
             num_lss, num_dbt, foir, default_probability,
             reapplication_eligible_date, bureau_refresh_due_date,
             model_version, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()),
            app_id,
            _f('PROSPECTID'),
            ts,
            decision,
            result.get('loan_amount'),
            result.get('loan_purpose'),
            primary_rule,
            json.dumps(rules),
            json.dumps(risk_factors),
            result.get('explanation'),
            _f('Credit_Score'),
            _f('NETMONTHLYINCOME'),
            _f('Tot_Missed_Pmnt'),
            _f('num_lss'),
            _f('num_dbt'),
            foir_val,
            result.get('default_probability'),
            eligible_dt,
            bureau_dt,
            MODEL_VERSION,
            now,
        ))


# ── Read / analytics ──────────────────────────────────────────────────────────

def get_decision_summary(days: int = 30) -> dict:
    """
    Return a summary of recent decisions for monitoring dashboards.

    Returns:
        {
            total, by_decision, by_model, approval_rate,
            conditional_rate, avg_processing_ms, period_days
        }
    """
    init_databases()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with _connect(LOG_DB) as conn:
        rows = conn.execute("""
            SELECT decision, model_used, COUNT(*) as n,
                   AVG(processing_ms) as avg_ms
            FROM   applications
            WHERE  application_ts >= ?
            GROUP  BY decision, model_used
        """, (since,)).fetchall()

    total         = sum(r['n'] for r in rows)
    by_decision   = {}
    by_model      = {}
    avg_ms_total  = 0
    ms_count      = 0

    for r in rows:
        by_decision[r['decision']] = by_decision.get(r['decision'], 0) + r['n']
        by_model[r['model_used']]  = by_model.get(r['model_used'],  0) + r['n']
        if r['avg_ms']:
            avg_ms_total += r['avg_ms'] * r['n']
            ms_count     += r['n']

    p2 = by_decision.get('P2', 0)
    p3 = by_decision.get('P3', 0)

    return {
        'total':            total,
        'by_decision':      by_decision,
        'by_model':         by_model,
        'approval_rate':    round(p2 / total, 4) if total else 0.0,
        'conditional_rate': round(p3 / total, 4) if total else 0.0,
        'avg_processing_ms':round(avg_ms_total / ms_count, 1) if ms_count else 0.0,
        'period_days':      days,
    }


def get_rejection_summary(days: int = 30) -> dict:
    """
    Rejection analysis — top rejection rules and risk factors.
    Used by the model monitoring dashboard.
    """
    init_databases()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with _connect(REJECT_LOG) as conn:
        rows = conn.execute("""
            SELECT primary_rule, decision, COUNT(*) as n,
                   AVG(credit_score) as avg_cs,
                   AVG(foir) as avg_foir
            FROM   rejections
            WHERE  application_ts >= ?
            GROUP  BY primary_rule, decision
            ORDER  BY n DESC
        """, (since,)).fetchall()

        pending = conn.execute("""
            SELECT COUNT(*) as n FROM rejections
            WHERE outcome_observed = 0
            AND   application_ts   >= ?
        """, (since,)).fetchone()['n']

    return {
        'total_rejections': sum(r['n'] for r in rows),
        'pending_outcomes': pending,
        'by_rule': [
            {'rule':     r['primary_rule'] or 'model',
             'decision': r['decision'],
             'count':    r['n'],
             'avg_credit_score': round(r['avg_cs'], 1) if r['avg_cs'] else None,
             'avg_foir':         round(r['avg_foir'], 3) if r['avg_foir'] else None}
            for r in rows
        ],
        'period_days': days,
    }


def get_pending_outcomes(limit: int = 500) -> list:
    """
    Return rejection records where outcome has not yet been observed.
    Used by the retraining pipeline to schedule bureau refreshes.
    """
    init_databases()
    cutoff = (datetime.utcnow() - timedelta(days=365)).isoformat()

    with _connect(REJECT_LOG) as conn:
        rows = conn.execute("""
            SELECT id, application_id, prospectid, application_ts,
                   decision, credit_score, default_probability,
                   bureau_refresh_due_date, reapplication_eligible_date
            FROM   rejections
            WHERE  outcome_observed  = 0
            AND    application_ts   <= ?
            ORDER  BY application_ts ASC
            LIMIT  ?
        """, (cutoff, limit)).fetchall()

    return [dict(r) for r in rows]


def record_outcome(rejection_id: str, defaulted: bool,
                   source: str = 'bureau_refresh'):
    """
    Record a real-world outcome for a rejected applicant.
    Called by the retraining pipeline when bureau refresh data arrives.

    Args:
        rejection_id: rejections.id
        defaulted:    True if applicant went 90+ DPD within 12 months
        source:       how we observed the outcome
    """
    init_databases()
    now = datetime.utcnow().isoformat()
    with _connect(REJECT_LOG) as conn:
        conn.execute("""
            UPDATE rejections
            SET    outcome_observed  = 1,
                   outcome_defaulted = ?,
                   outcome_observed_at = ?,
                   outcome_source    = ?
            WHERE  id = ?
        """, (1 if defaulted else 0, now, source, rejection_id))
    log.info(f'Outcome recorded: {rejection_id} defaulted={defaulted} source={source}')


def get_application(app_id: str) -> Optional[dict]:
    """Retrieve a single application record by ID."""
    init_databases()
    with _connect(LOG_DB) as conn:
        row = conn.execute(
            'SELECT * FROM applications WHERE id = ?', (app_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d['rules_fired']    = json.loads(d['rules_fired'] or '[]')
    d['input_features'] = json.loads(d['input_features'] or '{}')
    return d
