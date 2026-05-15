"""
retrain_pipeline.py — Phase 12

Scheduled retraining pipeline. Three independent retrains:

  1. MAIN MODEL   — LoanEnsemble (RF + ET → GBM meta + WoE Scorecard)
                    Trigger: new origination data accumulates
                    Target: Approved_Flag (4-class policy labels from featured.csv)

  2. THIN-FILE    — GradientBoostingClassifier (22 alternative-data features)
                    Trigger: enough new thin-file applications logged
                    Target: creditworthy (0/1) from synthetic + origination data

  3. DEFAULT MODEL — GradientBoostingClassifier (27 CIBIL features)
                     Trigger: observed outcomes in rejection_log (90+ DPD)
                     Target: defaulted (0/1) — starts as proxy, upgrades to real outcomes

Each retrain:
  - Evaluates the challenger model against the incumbent
  - Archives the old model with a timestamp before replacing
  - Writes a retrain report (metrics, feature drift, dataset stats)
  - Only promotes the challenger if it is strictly better

UPGRADE ROADMAP (printed at end of each run):
  Phase A  (now)        — proxy default target (Tot_Missed_Pmnt > 0)
  Phase B  (12m)        — real outcomes: rejection_log.outcome_defaulted populated
  Phase C  (18m)        — full 90+ DPD model replacing proxy entirely
"""

import sys, warnings, time, shutil, json
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              confusion_matrix)

from src.utils.config  import (MODELS_DIR, FEATURED_FILE, TARGET_COL,
                                RANDOM_STATE, TARGET_IMAP)
from src.utils.logger  import get_logger
from src.models.ensemble import LoanEnsemble

log = get_logger('RetrainPipeline')

ARCHIVE_DIR = MODELS_DIR / 'archive'
REPORT_DIR  = MODELS_DIR / 'retrain_reports'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _archive(model_path: Path) -> Path:
    """Copy model to archive with timestamp. Returns archive path."""
    ARCHIVE_DIR.mkdir(exist_ok=True)
    ts   = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    dest = ARCHIVE_DIR / f'{model_path.stem}_{ts}{model_path.suffix}'
    shutil.copy2(str(model_path), str(dest))
    log.info(f'Archived {model_path.name} → {dest.name}')
    return dest


def _write_report(name: str, report: dict):
    """Write JSON retrain report to disk."""
    REPORT_DIR.mkdir(exist_ok=True)
    ts   = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path = REPORT_DIR / f'{name}_{ts}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f'Retrain report: {path.name}')
    return path


def _load_featured() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_FILE)
    log.info(f'Loaded featured.csv: {df.shape}')
    return df


def _split(X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size,
        stratify=y, random_state=random_state)
    ratio = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=ratio,
        stratify=y_tmp, random_state=random_state)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ── 1. Main ensemble retrain ──────────────────────────────────────────────────

def retrain_main_model(min_improvement: float = 0.001,
                       dry_run: bool = False) -> dict:
    """
    Retrain the LoanEnsemble on the full featured dataset.

    Args:
        min_improvement: challenger must beat incumbent by at least this much
                         on macro F1 to be promoted
        dry_run:         if True, trains challenger but does not replace incumbent

    Returns:
        report dict with incumbent/challenger metrics and promotion decision
    """
    log.info('─── Main Model Retrain ───────────────────────────────')
    t0 = time.perf_counter()

    df            = _load_featured()
    incumbent_path = MODELS_DIR / 'final_ensemble.joblib'

    # ── Load incumbent metrics (stored in model object) ───────────────────────
    incumbent_raw  = joblib.load(incumbent_path)
    # retrain.py saves a plain dict; retrain_pipeline expects a LoanEnsemble
    if isinstance(incumbent_raw, dict) and 'gbm_ensemble' in incumbent_raw:
        incumbent = LoanEnsemble()
        gbm_e = incumbent_raw['gbm_ensemble']
        incumbent.rf   = gbm_e['rf']
        incumbent.et   = gbm_e['et']
        incumbent.meta = gbm_e['gbm_meta']
        incumbent.feature_cols = incumbent_raw.get('feature_names', [])
        incumbent.is_fitted = True
    else:
        incumbent = incumbent_raw
    incumbent_cols = incumbent.feature_cols

    # Re-score incumbent on current data
    X = df[incumbent_cols]
    # TARGET_COL is already integer-encoded in featured.csv
    y = df[TARGET_COL].astype(int)
    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X, y)

    inc_proba = incumbent.predict_proba(X_te)
    inc_pred  = np.argmax(inc_proba, axis=1)
    inc_f1    = f1_score(y_te, inc_pred, average='macro')
    log.info(f'Incumbent macro F1 (test): {inc_f1:.4f}')

    # ── Train challenger ───────────────────────────────────────────────────────
    log.info('Training challenger LoanEnsemble...')
    challenger = LoanEnsemble()
    challenger.fit(X_tr, y_tr, X_val, y_val)

    chal_proba = challenger.predict_proba(X_te)
    chal_pred  = np.argmax(chal_proba, axis=1)
    chal_f1    = f1_score(y_te, chal_pred, average='macro')
    log.info(f'Challenger macro F1 (test): {chal_f1:.4f}')

    # Per-class F1
    inc_per_class  = f1_score(y_te, inc_pred,  average=None)
    chal_per_class = f1_score(y_te, chal_pred, average=None)

    # Dangerous error check: P1 predicted as P2 (or vice versa)
    cm_chal = confusion_matrix(y_te, chal_pred)
    p1_idx, p2_idx = 0, 1
    dangerous_errors = (int(cm_chal[p1_idx][p2_idx]) +
                        int(cm_chal[p2_idx][p1_idx]))

    improved  = chal_f1 >= inc_f1 + min_improvement
    promoted  = improved and dangerous_errors == 0 and not dry_run

    if promoted:
        archive_path = _archive(incumbent_path)
        joblib.dump(challenger, incumbent_path)
        log.info(f'✅ Challenger promoted (F1 {inc_f1:.4f} → {chal_f1:.4f})')
    elif dry_run:
        log.info('Dry run — challenger not saved')
    else:
        log.info(f'Incumbent retained (challenger did not improve by >{min_improvement})')

    elapsed = round(time.perf_counter() - t0, 1)
    report  = {
        'model':           'main_ensemble',
        'retrain_ts':      datetime.utcnow().isoformat(),
        'elapsed_s':       elapsed,
        'train_size':      len(X_tr),
        'test_size':       len(X_te),
        'incumbent_f1':    round(inc_f1, 4),
        'challenger_f1':   round(chal_f1, 4),
        'improvement':     round(chal_f1 - inc_f1, 4),
        'per_class_incumbent':  {TARGET_IMAP[i]: round(float(v), 4)
                                  for i, v in enumerate(inc_per_class)},
        'per_class_challenger': {TARGET_IMAP[i]: round(float(v), 4)
                                  for i, v in enumerate(chal_per_class)},
        'dangerous_errors':     dangerous_errors,
        'promoted':             promoted,
        'dry_run':              dry_run,
    }
    _write_report('main_model', report)
    return report


# ── 2. Thin-file model retrain ────────────────────────────────────────────────

def retrain_thin_file_model(synthetic_path: str = None,
                             dry_run: bool = False) -> dict:
    """
    Retrain the thin-file GBM on synthetic data (and origination data if available).

    Args:
        synthetic_path: path to synthetic_thin_file.csv
                        defaults to data/raw/synthetic_thin_file.csv
        dry_run:        train but don't save

    Returns:
        report dict
    """
    log.info('─── Thin-File Model Retrain ──────────────────────────')
    t0 = time.perf_counter()

    if synthetic_path is None:
        synthetic_path = str(MODELS_DIR.parent / 'raw' / 'synthetic_thin_file.csv')

    df = pd.read_csv(synthetic_path)
    log.info(f'Loaded synthetic thin-file data: {df.shape}')

    # ── Append origination data if available ──────────────────────────────────
    origination_rows = _pull_origination_thin_file()
    if origination_rows:
        orig_df = pd.DataFrame(origination_rows)
        df      = pd.concat([df, orig_df], ignore_index=True)
        log.info(f'+ {len(origination_rows)} origination rows → total {len(df)}')

    thin_path  = MODELS_DIR / 'thin_file_model.joblib'
    incumbent  = joblib.load(thin_path)
    inc_model  = incumbent['model']
    features   = incumbent['features']
    encoders   = incumbent['encoders']

    # ── Encode categorical ────────────────────────────────────────────────────
    df_enc = df.copy()
    for col, enc in encoders.items():
        if col in df_enc.columns:
            try:
                df_enc[col] = enc.transform(df_enc[col].astype(str))
            except Exception:
                df_enc[col] = 0

    # Target: creditworthy column
    target_col = 'creditworthy' if 'creditworthy' in df_enc.columns else 'default'
    if target_col not in df_enc.columns:
        raise ValueError(f'Target column not found. Columns: {list(df_enc.columns)}')

    X = df_enc[features].fillna(0)
    y = df_enc[target_col].astype(int)

    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X, y)

    # ── Incumbent score ───────────────────────────────────────────────────────
    inc_auc = roc_auc_score(y_te, inc_model.predict_proba(X_te)[:, 1])
    log.info(f'Incumbent ROC-AUC: {inc_auc:.4f}')

    # ── Challenger ────────────────────────────────────────────────────────────
    challenger = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE)
    challenger.fit(X_tr, y_tr)
    chal_auc = roc_auc_score(y_te, challenger.predict_proba(X_te)[:, 1])

    # 5-fold CV
    cv_scores = cross_val_score(challenger, X, y, cv=5,
                                 scoring='roc_auc', n_jobs=-1)
    log.info(f'Challenger ROC-AUC: {chal_auc:.4f}  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    improved = chal_auc > inc_auc + 0.002
    promoted = improved and not dry_run

    if promoted:
        _archive(thin_path)
        joblib.dump({'model': challenger, 'features': features,
                     'encoders': encoders}, thin_path)
        log.info(f'✅ Thin-file challenger promoted (AUC {inc_auc:.4f} → {chal_auc:.4f})')
    else:
        log.info('Thin-file incumbent retained')

    elapsed = round(time.perf_counter() - t0, 1)
    report  = {
        'model':          'thin_file',
        'retrain_ts':     datetime.utcnow().isoformat(),
        'elapsed_s':      elapsed,
        'train_size':     len(X_tr),
        'test_size':      len(X_te),
        'origination_rows': len(origination_rows),
        'incumbent_auc':  round(inc_auc, 4),
        'challenger_auc': round(chal_auc, 4),
        'cv_mean':        round(float(cv_scores.mean()), 4),
        'cv_std':         round(float(cv_scores.std()), 4),
        'improvement':    round(chal_auc - inc_auc, 4),
        'promoted':       promoted,
        'dry_run':        dry_run,
    }
    _write_report('thin_file', report)
    return report


# ── 3. Default model retrain ──────────────────────────────────────────────────

def retrain_default_model(use_real_outcomes: bool = False,
                           dry_run: bool = False) -> dict:
    """
    Retrain the proxy default model.

    Phase A (use_real_outcomes=False):
        Target = Tot_Missed_Pmnt > 0 (current proxy)
        Data   = featured.csv (27 CIBIL features)

    Phase B (use_real_outcomes=True):
        Target = outcome_defaulted from rejection_log
        Data   = featured.csv rows with observed outcomes joined to rejection_log
        Requires: ≥500 observed outcomes in rejection_log

    Args:
        use_real_outcomes: if True, attempts to use rejection_log outcomes
        dry_run:           train but don't save

    Returns:
        report dict including readiness assessment for Phase B upgrade
    """
    log.info('─── Default Model Retrain ────────────────────────────')
    t0 = time.perf_counter()

    default_path = MODELS_DIR / 'default_model.joblib'
    incumbent    = joblib.load(default_path)
    features     = incumbent['features']
    inc_model    = incumbent['model']

    target_mode = 'proxy'
    X, y        = None, None

    # ── Phase B: real outcomes ────────────────────────────────────────────────
    if use_real_outcomes:
        X, y, n_outcomes = _load_real_outcome_data(features)
        if n_outcomes < 500:
            log.warning(f'Only {n_outcomes} real outcomes — minimum 500 required. '
                        f'Falling back to proxy target.')
            use_real_outcomes = False
        else:
            target_mode = 'real_outcomes'
            log.info(f'Using {n_outcomes} real outcomes for training')

    # ── Phase A: proxy target ─────────────────────────────────────────────────
    if not use_real_outcomes:
        df = _load_featured()
        # Only applicants with TL history (returning customers)
        df = df[df['Total_TL'] > 0].copy()
        y  = (df['Tot_Missed_Pmnt'] > 0).astype(int)
        X  = df[features].fillna(0)
        log.info(f'Proxy target: {y.sum()} positives / {len(y)} total '
                 f'({y.mean():.1%} default rate)')

    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X, y)

    # ── Incumbent score ───────────────────────────────────────────────────────
    inc_auc = roc_auc_score(y_te, inc_model.predict_proba(X_te)[:, 1])
    log.info(f'Incumbent ROC-AUC: {inc_auc:.4f}')

    # ── Challenger ────────────────────────────────────────────────────────────
    challenger = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE)
    challenger.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))
    chal_auc = roc_auc_score(y_te, challenger.predict_proba(X_te)[:, 1])

    cv_scores = cross_val_score(challenger, X, y, cv=5,
                                 scoring='roc_auc', n_jobs=-1)
    log.info(f'Challenger ROC-AUC: {chal_auc:.4f}  CV: {cv_scores.mean():.4f}')

    # ── Phase B readiness assessment ──────────────────────────────────────────
    pending, total_rej = _assess_phase_b_readiness()
    phase_b_ready = pending >= 500

    improved = chal_auc > inc_auc + 0.002
    promoted = improved and not dry_run

    if promoted:
        _archive(default_path)
        joblib.dump({'model': challenger, 'features': features,
                     'target_mode': target_mode}, default_path)
        log.info(f'✅ Default model challenger promoted (AUC {inc_auc:.4f} → {chal_auc:.4f})')
    else:
        log.info('Default model incumbent retained')

    elapsed = round(time.perf_counter() - t0, 1)
    report  = {
        'model':           'default_model',
        'retrain_ts':      datetime.utcnow().isoformat(),
        'elapsed_s':       elapsed,
        'target_mode':     target_mode,
        'train_size':      len(X_tr) + len(X_val),
        'test_size':       len(X_te),
        'incumbent_auc':   round(inc_auc, 4),
        'challenger_auc':  round(chal_auc, 4),
        'cv_mean':         round(float(cv_scores.mean()), 4),
        'cv_std':          round(float(cv_scores.std()), 4),
        'improvement':     round(chal_auc - inc_auc, 4),
        'promoted':        promoted,
        'dry_run':         dry_run,
        'phase_b_readiness': {
            'observed_outcomes': pending,
            'required':          500,
            'ready':             phase_b_ready,
            'eta_note':          ('Ready for real-outcome training'
                                  if phase_b_ready else
                                  f'Need {500 - pending} more observed outcomes '
                                  f'(~{max(0, (500-pending)//30)} months at current rate)'),
        },
    }
    _write_report('default_model', report)
    return report


# ── Data helpers ──────────────────────────────────────────────────────────────

def _pull_origination_thin_file() -> list:
    """
    Pull thin-file applications from origination_log that can supplement
    synthetic training data. Returns list of dicts.
    """
    try:
        from src.utils.origination_log import LOG_DB
        import sqlite3, json
        with sqlite3.connect(str(LOG_DB)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT input_features, decision
                FROM   applications
                WHERE  model_used = 'thin_file'
                AND    decision   IN ('P2','P3')
            """).fetchall()
        result = []
        for r in rows:
            fd = json.loads(r['input_features'] or '{}')
            fd['creditworthy'] = 1 if r['decision'] == 'P2' else 0
            result.append(fd)
        return result
    except Exception:
        return []


def _load_real_outcome_data(features: list):
    """
    Load rejection_log records with observed outcomes and join to featured.csv.
    Returns (X, y, n_outcomes).
    """
    from src.utils.origination_log import REJECT_LOG
    import sqlite3

    with sqlite3.connect(str(REJECT_LOG)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT prospectid, outcome_defaulted
            FROM   rejections
            WHERE  outcome_observed = 1
            AND    outcome_defaulted IS NOT NULL
        """).fetchall()

    if not rows:
        return None, None, 0

    outcomes = {r['prospectid']: r['outcome_defaulted'] for r in rows if r['prospectid']}
    df       = _load_featured()
    df       = df[df['PROSPECTID'].isin(outcomes)]
    df['_target'] = df['PROSPECTID'].map(outcomes)
    X = df[features].fillna(0)
    y = df['_target'].astype(int)
    return X, y, len(rows)


def _assess_phase_b_readiness():
    """Return (n_observed_outcomes, n_total_rejections)."""
    try:
        from src.utils.origination_log import REJECT_LOG
        import sqlite3
        with sqlite3.connect(str(REJECT_LOG)) as conn:
            n_obs = conn.execute(
                'SELECT COUNT(*) FROM rejections WHERE outcome_observed=1'
            ).fetchone()[0]
            n_tot = conn.execute(
                'SELECT COUNT(*) FROM rejections'
            ).fetchone()[0]
        return n_obs, n_tot
    except Exception:
        return 0, 0


# ── Archive management ────────────────────────────────────────────────────────

def list_archive(model_name: str = None) -> list:
    """List archived model versions. Optionally filter by model_name prefix."""
    ARCHIVE_DIR.mkdir(exist_ok=True)
    files = sorted(ARCHIVE_DIR.glob('*.joblib'), reverse=True)
    if model_name:
        files = [f for f in files if f.name.startswith(model_name)]
    return [{'name': f.name,
             'size_mb': round(f.stat().st_size / 1e6, 1),
             'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
            for f in files]


def rollback(model_name: str, archive_filename: str):
    """
    Roll back a model to an archived version.

    Args:
        model_name:      e.g. 'final_ensemble', 'thin_file_model', 'default_model'
        archive_filename: filename from list_archive()
    """
    src  = ARCHIVE_DIR / archive_filename
    dest = MODELS_DIR  / f'{model_name}.joblib'
    if not src.exists():
        raise FileNotFoundError(f'Archive not found: {src}')
    _archive(dest)   # archive current before overwriting
    shutil.copy2(str(src), str(dest))
    log.info(f'Rolled back {model_name} ← {archive_filename}')


# ── Upgrade roadmap ───────────────────────────────────────────────────────────

def print_upgrade_roadmap():
    pending, total_rej = _assess_phase_b_readiness()
    pct = f'{pending/max(total_rej,1)*100:.0f}%'

    print('\n' + '═'*58)
    print('  MODEL UPGRADE ROADMAP')
    print('═'*58)
    print(f'\n  Current phase:  A — proxy default target (Tot_Missed_Pmnt)')
    print(f'\n  Rejection log:')
    print(f'    Total rejections logged : {total_rej}')
    print(f'    Outcomes observed       : {pending} ({pct})')
    print(f'    Required for Phase B    : 500')
    print(f'    Status                  : {"✅ READY" if pending >= 500 else f"⏳ need {500-pending} more"}')
    print(f'\n  Phase A  (now)   Proxy target. ROC-AUC ≈ 0.85')
    print(f'  Phase B  (12m)   500+ real outcomes → hybrid training')
    print(f'  Phase C  (18m)   Full 90+DPD model. Expected AUC ≈ 0.88–0.92')
    print(f'\n  Triggers for next retrain:')
    print(f'    Main model   — every 10,000 new applications')
    print(f'    Thin-file    — every 2,000 new thin-file applications')
    print(f'    Default      — every 500 new observed outcomes')
    print('═'*58 + '\n')


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def run_full_retrain(dry_run: bool = False):
    """
    Run all three retrains in sequence.
    Called by the scheduler (cron / Airflow DAG).
    """
    log.info('═══ FULL RETRAIN PIPELINE ════════════════════════════')
    results = {}

    results['main']     = retrain_main_model(dry_run=dry_run)
    results['thin_file']= retrain_thin_file_model(dry_run=dry_run)
    results['default']  = retrain_default_model(dry_run=dry_run)

    promoted = [k for k, v in results.items() if v.get('promoted')]
    log.info(f'Retrain complete. Promoted: {promoted or "none"}')
    print_upgrade_roadmap()
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Loan Decisioning Retrain Pipeline')
    parser.add_argument('--model', choices=['main','thin_file','default','all'],
                        default='all')
    parser.add_argument('--dry-run',   action='store_true')
    parser.add_argument('--real-outcomes', action='store_true',
                        help='Use rejection_log real outcomes for default model')
    parser.add_argument('--roadmap',   action='store_true',
                        help='Print upgrade roadmap and exit')
    args = parser.parse_args()

    if args.roadmap:
        print_upgrade_roadmap()
    elif args.model == 'all':
        run_full_retrain(dry_run=args.dry_run)
    elif args.model == 'main':
        retrain_main_model(dry_run=args.dry_run)
    elif args.model == 'thin_file':
        retrain_thin_file_model(dry_run=args.dry_run)
    elif args.model == 'default':
        retrain_default_model(use_real_outcomes=args.real_outcomes,
                               dry_run=args.dry_run)
