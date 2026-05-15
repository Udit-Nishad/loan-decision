"""
main.py — AI Loan Decisioning API  (Phase 10)

FastAPI layer over the Phase 8 router + Phase 9 explainability.
All core logic lives in framework-agnostic pipeline functions so the
same code is testable without FastAPI installed.

Endpoints:
  POST /api/v1/predict           Single application → decision + explanation
  POST /api/v1/predict/batch     Up to 50 applications in one call
  GET  /api/v1/report/{app_id}   Download PDF credit decision report
  GET  /api/v1/health            Liveness + model status
  GET  /api/v1/model-info        Architecture + metrics

Run locally:
  pip install fastapi uvicorn pydantic python-multipart
  uvicorn src.api.main:app --reload --port 8000
  → Swagger docs: http://localhost:8000/docs
"""
import uuid, time, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger
from src.utils.origination_log import log_application, init_databases

log       = get_logger('LoanAPI')
_START_TS = time.time()

# ── FastAPI — graceful degradation when not installed ────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel as _PydanticBase, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    class _PydanticBase:            # minimal stub
        def model_dump(self): return {}
        def dict(self):       return {}
    class Field:
        def __new__(cls, *a, **kw): return None


# ── Lazy model registry ───────────────────────────────────────────────────────
class _Registry:
    """Singleton — loads all heavy objects once and caches them."""
    _loaded     = False
    _load_error = None

    @classmethod
    def ensure_loaded(cls):
        if cls._loaded:
            return
        try:
            log.info('Loading model registry...')
            import joblib
            from src.utils.config import MODELS_DIR
            from src.explainability.shap_engine import SHAPEngine

            raw_ensemble     = joblib.load(MODELS_DIR / 'final_ensemble.joblib')
            cls.thinfile     = joblib.load(MODELS_DIR / 'thin_file_model.joblib')
            cls.default_mdl  = joblib.load(MODELS_DIR / 'default_model.joblib')

            # retrain.py saves a plain dict; handle both formats
            if isinstance(raw_ensemble, dict) and 'feature_names' in raw_ensemble:
                cls.ensemble      = raw_ensemble
                feature_cols      = raw_ensemble['feature_names']
                # Use the ET base learner for SHAP (trained on full features)
                shap_model        = raw_ensemble['gbm_ensemble']['et']
            else:
                cls.ensemble      = raw_ensemble
                feature_cols      = raw_ensemble.feature_cols
                shap_model        = raw_ensemble

            cls.main_engine     = SHAPEngine(shap_model,
                                             feature_cols, 'main')
            cls.thinfile_engine = SHAPEngine(cls.thinfile['model'],
                                             cls.thinfile['features'], 'thin_file')
            cls._loaded     = True
            log.info('Model registry ready')
        except Exception as e:
            cls._load_error = str(e)
            log.error(f'Registry load failed: {e}')
            raise

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded


# ── Core pipeline (framework-agnostic) ───────────────────────────────────────

def _run_single(input_dict: dict, app_id: str) -> dict:
    """
    Full prediction pipeline for one applicant.
    Called by the FastAPI route AND directly by tests.
    """
    _Registry.ensure_loaded()
    t0 = time.perf_counter()

    from src.api.schemas       import LoanApplicationInput, ValidationError as VE
    from src.models.router     import predict
    from src.explainability.shap_engine  import explain_decision
    from src.explainability.report_builder import build_report
    from src.utils.config      import MODELS_DIR

    # ── Validate + build typed request ───────────────────────────────────────
    # Strip API-layer-only fields not in LoanApplicationInput
    _API_ONLY = {'applicant_name', 'unknown_field_xyz'}
    try:
        req = LoanApplicationInput(**{k: v for k, v in input_dict.items()
                                      if v is not None and k not in _API_ONLY})
    except VE as e:
        raise ValueError(f'Validation: {e}')

    # ── Predict ───────────────────────────────────────────────────────────────
    output = predict(req)

    # ── Explain ───────────────────────────────────────────────────────────────
    fd  = req.to_feature_dict()
    # Enrich with computed FOIR so improvement_roadmap can detect high FOIR
    income   = float(req.NETMONTHLYINCOME or 0)
    tenure   = float(req.loan_tenure_months or 12)
    new_emi  = round(req.loan_amount / tenure, 2) if tenure > 0 else 0
    fd['foir'] = min(round(new_emi / income, 3), 1.0) if income > 0 else 1.0
    exp = explain_decision(
        output, fd,
        main_model    = _Registry.ensemble     if _Registry.is_loaded() else None,
        thinfile_model= _Registry.thinfile['model'] if _Registry.is_loaded() else None,
        feature_cols  = (_Registry.ensemble.get('feature_names')
                         if isinstance(_Registry.ensemble, dict)
                         else _Registry.ensemble.feature_cols)
                        if _Registry.is_loaded() else None,
    )

    # ── PDF report ────────────────────────────────────────────────────────────
    report_path = None
    report_dir  = MODELS_DIR.parent / 'reports'
    report_dir.mkdir(exist_ok=True)
    try:
        report_path = build_report(
            output, exp,
            applicant_meta={
                'id':           app_id,
                'name':         input_dict.get('applicant_name', '—'),
                'loan_amount':  req.loan_amount,
                'loan_purpose': req.loan_purpose or '—',
            },
            output_path=str(report_dir / f'decision_{app_id}.pdf'),
        )
    except Exception as e:
        log.warning(f'[{app_id}] PDF generation failed: {e}')

    ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Origination log ───────────────────────────────────────────────────────
    _result_for_log = {
        'application_id':   app_id,
        'loan_amount':      req.loan_amount,
        'loan_purpose':     req.loan_purpose,
        'decision':         output.decision,
        'decision_label':   _DECISION_LABELS[output.decision],
        'probability':      round(output.probability, 4),
        'confidence':       output.confidence,
        'model_used':       output.model_used,
        'rules_fired':      output.rules_fired,
        'default_probability': (round(output.default_probability, 4)
                                if output.default_probability is not None else None),
        'explanation':      output.explanation,
        'top_factors':      exp['factors'],
        'timestamp':        datetime.now().isoformat(),
        'processing_ms':    ms,
    }
    try:
        log_application(_result_for_log, input_dict)
    except Exception as e:
        log.warning(f'[{app_id}] Origination log failed: {e}')

    log.info(f'[{app_id}] {output.decision} | '
             f'{output.confidence} | {output.model_used} | {ms}ms')

    return {
        'application_id':      app_id,
        'applicant_name':      input_dict.get('applicant_name', '—'),
        'loan_amount':         req.loan_amount,
        'loan_purpose':        req.loan_purpose,
        'loan_tenure_months':  req.loan_tenure_months or 36,
        'credit_score':        req.Credit_Score or 0,
        'income_source':       req.income_source or 'salaried',
        'decision':            output.decision,
        'decision_label':      _DECISION_LABELS[output.decision],
        'probability':         round(output.probability, 4),
        'confidence':          output.confidence,
        'model_used':          output.model_used,
        'rules_fired':         output.rules_fired,
        'default_probability': (round(output.default_probability, 4)
                                if output.default_probability is not None else None),
        'explanation':         output.explanation,
        'top_factors':         exp['factors'],
        'improvement_roadmap': exp['roadmap'],
        'explainability_method': exp['method'],
        'report_url':          f'/api/v1/report/{app_id}' if report_path else None,
        'timestamp':           datetime.now().isoformat(),
        'processing_ms':       ms,
    }


def _run_batch(applications: list) -> dict:
    """
    Batch pipeline — processes each application independently.
    Per-row failures are captured; the batch continues.
    """
    t0      = time.perf_counter()
    results = []
    failed  = 0

    for i, app_dict in enumerate(applications):
        app_id = f'B{str(uuid.uuid4())[:8].upper()}'
        try:
            res = _run_single(app_dict, app_id)
            results.append({'index': i, 'status': 'ok', **res})
        except Exception as e:
            failed += 1
            log.warning(f'Batch [{i}] failed: {e}')
            results.append({
                'index':  i,
                'status': 'error',
                'error':  str(e),
                'applicant_name': app_dict.get('applicant_name', '—'),
            })

    return {
        'total':          len(applications),
        'processed':      len(applications) - failed,
        'failed':         failed,
        'results':        results,
        'timestamp':      datetime.now().isoformat(),
        'processing_ms':  round((time.perf_counter() - t0) * 1000, 1),
    }


def _health() -> dict:
    return {
        'status':         'ok' if _Registry.is_loaded() else 'degraded',
        'model_loaded':   _Registry.is_loaded(),
        'load_error':     _Registry._load_error,
        'uptime_seconds': round(time.time() - _START_TS, 1),
        'timestamp':      datetime.now().isoformat(),
    }


def _model_info() -> dict:
    from src.utils.config import RULE_MIN_CREDIT_SCORE, RULE_MAX_FOIR_REJECT
    return {
        'version': '1.0.0-phase10',
        'architecture': {
            'main_model':     'LoanEnsemble — RF(200) + ExtraTrees(200) → GBM meta(200) + WoE Scorecard',
            'thin_file_model':'GradientBoostingClassifier(200) — 22 alternative-data features',
            'default_model':  'GradientBoostingClassifier(200) — 27 CIBIL features, proxy default',
            'blend':          'GBM 70% + ET 20% + Scorecard 10%',
        },
        'performance': {
            'main_macro_f1':     0.9851,
            'main_p2_f1':        0.9996,
            'thin_file_auc':     0.7581,
            'default_model_auc': 0.8508,
        },
        'training_data': {
            'source':   'D9 CIBIL + D9 Internal Bank (merged on PROSPECTID)',
            'records':  51_336,
            'features': 106,
            'split':    '70/15/15 train/val/test stratified',
        },
        'classes': {
            'P1': 'REJECT',
            'P2': 'APPROVE',
            'P3': 'CONDITIONAL APPROVE',
            'P4': 'REFER TO CREDIT COMMITTEE',
        },
        'rule_engine': [
            f'RULE_01: num_lss > 0                           → P1 REJECT',
            f'RULE_02: Credit_Score < {RULE_MIN_CREDIT_SCORE} → P1 REJECT',
            f'RULE_03: NETMONTHLYINCOME ≤ 0                  → P1 REJECT',
            f'RULE_04: num_dbt > 0                           → P4 REFER',
            f'RULE_05: recent_stress=1 + model=P2            → P3 CONDITIONAL',
            f'RULE_06: FOIR > {RULE_MAX_FOIR_REJECT:.0%}     → P1 REJECT',
            f'RULE_07: FOIR 50–60%                           → P3 CONDITIONAL',
            f'RULE_08: thin-file + loan > ₹10,00,000         → P1 REJECT',
            f'RULE_09: thin-file + FOIR > 30%               → P3 CONDITIONAL',
            f'RULE_10: thin-file + income_unverified         → never P2',
        ],
        'timestamp': datetime.now().isoformat(),
    }


_DECISION_LABELS = {
    'P1': 'REJECTED',
    'P2': 'APPROVED',
    'P3': 'CONDITIONALLY APPROVED',
    'P4': 'REFERRED — CREDIT COMMITTEE',
}


# ── FastAPI app ───────────────────────────────────────────────────────────────
if FASTAPI_OK:
    app = FastAPI(
        title       = 'AI Loan Decisioning API',
        description = (
            '**RBI-compliant AI credit decisioning for Tier-1 Indian banks.**\n\n'
            '3-layer ensemble (RF + ExtraTrees → GBM meta + WoE Scorecard) '
            'trained on 51,336 CIBIL bureau records.\n\n'
            '**Classes:** P1=REJECT | P2=APPROVE | '
            'P3=CONDITIONAL | P4=REFER TO CREDIT COMMITTEE'
        ),
        version     = '1.0.0',
        docs_url    = '/docs',
        redoc_url   = '/redoc',
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ['*'],
        allow_credentials = True,
        allow_methods     = ['GET', 'POST'],
        allow_headers     = ['*'],
    )

    @app.exception_handler(Exception)
    async def _global_exc(request: Request, exc: Exception):
        log.error(f'Unhandled: {request.url} — {exc}')
        return JSONResponse(
            status_code = 500,
            content     = {'error': 'Internal server error', 'detail': str(exc)},
        )

    @app.on_event('startup')
    async def _startup():
        try:
            _Registry.ensure_loaded()
        except Exception as e:
            log.error(f'Startup model load failed: {e}')

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get('/api/v1/health', tags=['System'])
    async def health():
        return _health()

    # ── Model info ────────────────────────────────────────────────────────────
    @app.get('/api/v1/model-info', tags=['System'])
    async def model_info():
        return _model_info()

    # ── Single predict ────────────────────────────────────────────────────────
    @app.post('/api/v1/predict', tags=['Prediction'],
              summary='Submit one loan application → AI credit decision')
    async def api_predict(body: Dict[str, Any]):
        """
        Returns decision (P1–P4), probability, confidence, top 8 SHAP factors,
        improvement roadmap (P1/P3 only), and a link to the PDF report.
        """
        try:
            _Registry.ensure_loaded()
        except Exception as e:
            raise HTTPException(503, f'Models not loaded: {e}')
        app_id = str(uuid.uuid4())[:12].upper().replace('-', '')
        try:
            return _run_single(body, app_id)
        except ValueError as e:
            raise HTTPException(422, detail=str(e))
        except Exception as e:
            log.error(f'Predict failed: {e}')
            raise HTTPException(500, detail=str(e))

    # ── Batch predict ─────────────────────────────────────────────────────────
    @app.post('/api/v1/predict/batch', tags=['Prediction'],
              summary='Score up to 50 applications in one call')
    async def api_predict_batch(body: Dict[str, Any]):
        """Per-row failures are captured; the rest of the batch continues."""
        apps = body.get('applications', [])
        if not isinstance(apps, list) or len(apps) == 0:
            raise HTTPException(422, "'applications' must be a non-empty list")
        if len(apps) > 50:
            raise HTTPException(422, 'Batch size limit is 50 applications')
        try:
            _Registry.ensure_loaded()
        except Exception as e:
            raise HTTPException(503, f'Models not loaded: {e}')
        return _run_batch(apps)

    # ── PDF report ────────────────────────────────────────────────────────────
    @app.get('/api/v1/report/{app_id}', tags=['Reports'],
             summary='Download PDF credit decision report')
    async def api_get_report(app_id: str):
        from src.utils.config import MODELS_DIR
        safe   = ''.join(c for c in app_id if c.isalnum() or c == '-')[:20]
        path   = MODELS_DIR.parent / 'reports' / f'decision_{safe}.pdf'
        if not path.exists():
            raise HTTPException(404, f"Report not found for '{safe}'. "
                                     "Submit via POST /api/v1/predict first.")
        return FileResponse(str(path), media_type='application/pdf',
                            filename=f'LoanDecision_{safe}.pdf')


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if FASTAPI_OK:
        import uvicorn
        uvicorn.run('src.api.main:app', host='0.0.0.0', port=8000, reload=True)
    else:
        print('FastAPI not installed. Running pipeline self-test.\n'
              'Install: pip install fastapi uvicorn\n')
        result = _run_single({
            'applicant_name': 'Self-Test Applicant',
            'AGE': 38, 'NETMONTHLYINCOME': 55000,
            'Credit_Score': 682, 'Total_TL': 5,
            'loan_amount': 300000, 'loan_purpose': 'Personal',
            'loan_tenure_months': 36,
        }, app_id='SELFTEST')
        print(f"Decision:  {result['decision']} — {result['decision_label']}")
        print(f"Model:     {result['model_used']}")
        print(f"Confidence:{result['confidence']}")
        print(f"Time:      {result['processing_ms']}ms")
