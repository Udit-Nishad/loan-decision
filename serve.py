"""
serve.py — Flask server for the AI Loan Decisioning System.

Usage (from the loan-decisioning/ directory):
    python serve.py

Then open:
    http://localhost:8000

PyCharm: Right-click serve.py → Run 'serve'
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
for pkg, imp in [('flask','flask'), ('pandas','pandas'), ('numpy','numpy'),
                 ('sklearn','sklearn'), ('joblib','joblib'), ('reportlab','reportlab')]:
    try:
        __import__(imp)
    except ImportError:
        _missing.append(pkg)

if _missing:
    print(f'\n❌  Missing packages. Run:\n\n    pip install {" ".join(_missing)}\n')
    sys.exit(1)

# ── Model check ───────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / 'data' / 'models'
if not (MODELS_DIR / 'final_ensemble.joblib').exists():
    print('\n❌  final_ensemble.joblib not found. Run:\n\n    python retrain.py\n')
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings, uuid, json
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_file, send_from_directory, Response

from src.api.main import (
    _run_single, _run_batch, _health, _model_info, _Registry,
)

# Map frontend field names → LoanApplicationInput field names
# The frontend was built before Phase 7 finalised the schema
_FIELD_MAP = {
    'num_deliq_6mts':        'num_times_delinquent',
    'num_deliq_12mts':       'num_times_delinquent',
    'num_deliq_6_12mts':     None,   # drop — not in schema
    'max_deliq_6mts':        'max_recent_level_of_deliq',
    'max_deliq_12mts':       'max_recent_level_of_deliq',
    'recent_level_of_deliq': 'max_recent_level_of_deliq',
    'num_times_30p_dpd':     None,
    'num_times_60p_dpd':     'num_times_60p_dpd',
    'HL_Flag':               None,   # not in schema — drop
    'GL_Flag':               None,
    'Time_With_Curr_Empr':   'Time_With_Curr_Empr',
    'loan_tenure_months':    'loan_tenure_months',
}

def _normalise(body: dict) -> dict:
    """Translate frontend field names to schema field names."""
    out = {}
    seen_delinquent = None
    for k, v in body.items():
        mapped = _FIELD_MAP.get(k, k)   # default: keep as-is
        if mapped is None:
            continue   # drop unknown / legacy field
        # If multiple frontend fields map to same schema field, take max
        if mapped in ('num_times_delinquent', 'max_recent_level_of_deliq'):
            prev = out.get(mapped, 0)
            out[mapped] = max(prev, int(v or 0))
        else:
            out[mapped] = v
    return out
from src.utils.config import MODELS_DIR as _MDIR
from interest_engine import calculate as _calc_rate, REPO_RATE, OP_MARGIN

def _shape_response(result: dict) -> dict:
    """
    Translate the Phase 10 pipeline response to the shape the
    frontend index.html expects (built against the Phase 2 API).
    """
    decision = result.get('decision', 'P1')
    prob      = result.get('probability', 0.0)
    factors   = result.get('top_factors', [])
    rules     = result.get('rules_fired', [])

    # Decision label the frontend's decisionClass() understands
    decision_text = {
        'P1': 'REJECT',
        'P2': 'APPROVE',
        'P3': 'CONDITIONAL APPROVE',
        'P4': 'REFER TO CREDIT COMMITTEE',
    }.get(decision, decision)

    # Per-class probabilities — frontend iterates d.ensemble_proba
    ensemble_proba = {'P1': 0.0, 'P2': 0.0, 'P3': 0.0, 'P4': 0.0}
    ensemble_proba[decision] = prob
    remaining = (1.0 - prob) / 3
    for k in ensemble_proba:
        if k != decision:
            ensemble_proba[k] = round(remaining, 4)
    ensemble_proba[decision] = round(prob, 4)

    # Top factors → frontend shape
    def _icon(direction):
        return '✅' if direction == 'positive' else ('⚠️' if direction == 'negative' else '—')

    shaped_factors = [
        {
            'feature':     f.get('feature', ''),
            'plain_name':  f.get('label', f.get('feature', '')),
            'banker_text': f.get('description', ''),
            'icon':        _icon(f.get('direction', 'neutral')),
            'importance':  f.get('magnitude', 0.0),
            'direction':   f.get('direction', 'neutral'),
            'value':       f.get('value', 0),
        }
        for f in factors
    ]

    # Improvement steps as plain text
    roadmap = result.get('improvement_roadmap', [])
    steps = [f"[{s['priority']}] {s['action']}  ->  {s['timeline']}" for s in roadmap]
    improvement_text = chr(10).join(steps) if steps else ''

    # ── Interest rate ─────────────────────────────────────────────────────────
    rate_info = None
    if decision in ('P2', 'P3', 'P4'):
        try:
            _r = _calc_rate(
                decision            = decision,
                loan_purpose        = result.get('loan_purpose', 'Personal'),
                loan_amount         = result.get('loan_amount', 0),
                loan_tenure_months  = result.get('loan_tenure_months', 36),
                credit_score        = result.get('credit_score', 0),
                income_source       = result.get('income_source', 'salaried'),
                default_probability = result.get('default_probability'),
            )
            if _r:
                rate_info = {
                    'annual_rate':        _r.annual_rate,
                    'repo_rate':          _r.repo_rate,
                    'operational_margin': _r.operational_margin,
                    'risk_premium':       _r.total_risk_premium,
                    'rate_band':          _r.rate_band,
                    'emi':                _r.emi,
                    'total_payable':      _r.total_payable,
                    'total_interest':     _r.total_interest,
                    'effective_cost':     _r.effective_cost,
                    'rate_rationale':     _r.rate_rationale,
                    'breakdown':          f'{_r.repo_rate}% (Repo) + {_r.operational_margin}% (OpEx) + {_r.total_risk_premium}% (Risk) = {_r.annual_rate}%',
                }
        except Exception:
            pass

    return {
        # Fields the frontend reads directly
        'application_id':   result.get('application_id', ''),
        'applicant_name':   result.get('applicant_name', ''),
        'loan_amount':      result.get('loan_amount', 0),
        'loan_purpose':     result.get('loan_purpose', ''),
        'predicted_class':  decision,
        'decision':         decision_text,
        'confidence_score': round(prob * 100, 1),
        'scorecard_points': 650 + int((prob - 0.5) * 200),   # synthetic scorecard score
        'scorecard_pd':     round(1.0 - prob, 4),
        'ensemble_proba':   ensemble_proba,
        'gbm_proba':        ensemble_proba,
        'rule_triggered':   len(rules) > 0,
        'rule_name':        rules[0] if rules else None,
        'rule_reason':      result.get('explanation', ''),
        'top_factors':      shaped_factors,
        'improvement_steps': improvement_text,
        'report_url':       result.get('report_url', ''),
        'timestamp':        result.get('timestamp', ''),
        'processing_ms':    result.get('processing_ms', 0),
        # Also include the full Phase 10 fields for completeness
        'model_used':       result.get('model_used', ''),
        'confidence':       result.get('confidence', ''),
        'probability':      prob,
        'explanation':      result.get('explanation', ''),
        'default_probability': result.get('default_probability'),
        'improvement_roadmap': roadmap,
        'rate_info':           rate_info,
    }


# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(ROOT / 'frontend'))

FRONTEND = ROOT / 'frontend'


# ── Warm up models on start ───────────────────────────────────────────────────
def _warmup():
    print('\n  Loading models...', flush=True)
    try:
        _Registry.ensure_loaded()
        print('  [OK] Models loaded\n', flush=True)
    except Exception as e:
        print(f'  [FAIL] Model load failed: {e}\n', flush=True)


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(FRONTEND), 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(str(FRONTEND), filename)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify(_health())


# ── Model info ────────────────────────────────────────────────────────────────
@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    return jsonify(_model_info())


# ── Single predict ────────────────────────────────────────────────────────────
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True, silent=True) or {}
    try:
        _Registry.ensure_loaded()
    except Exception as e:
        return jsonify({'error': f'Models not loaded: {e}'}), 503

    app_id = str(uuid.uuid4())[:12].upper().replace('-', '')
    body   = _normalise(body)
    try:
        result = _run_single(body, app_id)
        return jsonify(_shape_response(result))
    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Batch predict ─────────────────────────────────────────────────────────────
@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch():
    body = request.get_json(force=True, silent=True) or {}
    apps = body if isinstance(body, list) else body.get('applications', [])
    if not apps:
        return jsonify({'error': "'applications' must be a non-empty list"}), 422
    if len(apps) > 50:
        return jsonify({'error': 'Batch limit is 50'}), 422
    try:
        _Registry.ensure_loaded()
    except Exception as e:
        return jsonify({'error': f'Models not loaded: {e}'}), 503
    batch = _run_batch(apps)
    batch['results'] = [
        {**r, **_shape_response(r)} if r.get('status') == 'ok' else r
        for r in batch.get('results', [])
    ]
    return jsonify(batch)


# ── PDF report ────────────────────────────────────────────────────────────────
@app.route('/api/v1/report/<app_id>', methods=['GET'])
def get_report(app_id):
    safe = ''.join(c for c in app_id if c.isalnum() or c == '-')[:20]
    path = _MDIR.parent / 'reports' / f'decision_{safe}.pdf'
    if not path.exists():
        return jsonify({'error': f"Report not found for '{safe}'"}), 404
    return send_file(str(path), mimetype='application/pdf',
                     download_name=f'LoanDecision_{safe}.pdf')


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))

    print()
    print('  +------------------------------------------+')
    print('  |   AI Loan Decisioning System  v1.0       |')
    print('  +------------------------------------------+')
    print(f'  |   Open -> http://localhost:{port}            |')
    print('  +------------------------------------------+')

    _warmup()

    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
