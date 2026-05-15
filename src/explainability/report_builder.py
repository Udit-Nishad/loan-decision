"""
report_builder.py — Phase 9 (upgraded)

5-section PDF credit decision report:
  1. Executive Decision Summary
  2. Top Driving Factors (SHAP-weighted table)
  3. Default Risk Assessment (proxy PD + gauge)
  4. Improvement Roadmap — What To Do To Improve Your Chances
  5. Regulatory Audit Trail
"""
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from interest_engine import calculate as _calc_rate

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak, KeepTogether
)

# ── Colour palette ────────────────────────────────────────────────────────────
BANK_BLUE    = colors.HexColor('#003366')
APPROVE_GRN  = colors.HexColor('#1A7F4B')
REJECT_RED   = colors.HexColor('#C0392B')
COND_ORG     = colors.HexColor('#D68910')
REFER_PURPLE = colors.HexColor('#6C3483')
LIGHT_GREY   = colors.HexColor('#F2F3F4')
MID_GREY     = colors.HexColor('#AAB7B8')
DARK_GREY    = colors.HexColor('#555555')

DECISION_COLOUR = {
    'P1': REJECT_RED, 'P2': APPROVE_GRN,
    'P3': COND_ORG,   'P4': REFER_PURPLE,
}
DECISION_LABEL = {
    'P1': 'REJECTED',
    'P2': 'APPROVED',
    'P3': 'CONDITIONALLY APPROVED',
    'P4': 'REFERRED — CREDIT COMMITTEE',
}
RISK_LABEL = {
    'P1': 'HIGH RISK',
    'P2': 'LOW RISK',
    'P3': 'MODERATE RISK',
    'P4': 'ELEVATED RISK',
}


def _styles():
    base = getSampleStyleSheet()
    return {
        'title':   ParagraphStyle('title',  parent=base['Title'],
                                  fontSize=16, textColor=BANK_BLUE, spaceAfter=4),
        'bank':    ParagraphStyle('bank',   parent=base['Normal'],
                                  fontSize=10, textColor=BANK_BLUE,
                                  fontName='Helvetica-Bold', spaceAfter=2),
        'h2':      ParagraphStyle('h2',     parent=base['Heading2'],
                                  fontSize=11, textColor=BANK_BLUE,
                                  spaceBefore=14, spaceAfter=4),
        'h3':      ParagraphStyle('h3',     parent=base['Heading3'],
                                  fontSize=10, textColor=BANK_BLUE,
                                  spaceBefore=8, spaceAfter=3),
        'body':    ParagraphStyle('body',   parent=base['Normal'],
                                  fontSize=9,  leading=14),
        'small':   ParagraphStyle('small',  parent=base['Normal'],
                                  fontSize=8,  leading=12, textColor=DARK_GREY),
        'bold':    ParagraphStyle('bold',   parent=base['Normal'],
                                  fontSize=9,  leading=14,
                                  fontName='Helvetica-Bold'),
        'centre':  ParagraphStyle('centre', parent=base['Normal'],
                                  fontSize=9,  alignment=TA_CENTER),
        'improve': ParagraphStyle('improve',parent=base['Normal'],
                                  fontSize=9,  leading=15,
                                  leftIndent=10),
    }


def _tbl_style(header_colour=BANK_BLUE, header_text=colors.white):
    return TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), header_colour),
        ('TEXTCOLOR',    (0, 0), (-1, 0), header_text),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0), 9),
        ('FONTSIZE',     (0, 1), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, LIGHT_GREY]),
        ('GRID',         (0, 0), (-1,-1), 0.4, MID_GREY),
        ('VALIGN',       (0, 0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1,-1), 6),
        ('RIGHTPADDING', (0, 0), (-1,-1), 6),
        ('TOPPADDING',   (0, 0), (-1,-1), 4),
        ('BOTTOMPADDING',(0, 0), (-1,-1), 4),
    ])


def _bar(magnitude: float, width: int = 24) -> str:
    filled = max(1, int(magnitude * width * 12))
    return '█' * min(filled, width)


def _pd_gauge(pd_prob: float) -> str:
    """Text-based PD gauge: 0% ░░░░░░░░░░ 100%"""
    filled = int(pd_prob * 20)
    bar = '█' * filled + '░' * (20 - filled)
    return f'0%  {bar}  100%   ({pd_prob*100:.1f}%)'


def build_report(output, explain_result: dict,
                 applicant_meta: dict = None,
                 output_path: str = None) -> str:
    """
    Build a 5-section PDF decision report.

    Args:
        output:         PredictionOutput from router
        explain_result: dict from explain_decision()
        applicant_meta: optional dict with name, id, loan_amount, loan_purpose
        output_path:    where to save the PDF (default: auto-named in /tmp)

    Returns:
        str — path to the generated PDF
    """
    if output_path is None:
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = applicant_meta.get('id', 'UNKNOWN') if applicant_meta else 'UNKNOWN'
        output_path = f'/tmp/decision_{pid}_{ts}.pdf'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc   = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm,  bottomMargin=2*cm)
    S     = _styles()
    story = []
    now   = datetime.now().strftime('%d %b %Y  %H:%M')
    meta  = applicant_meta or {}

    dec_col   = DECISION_COLOUR.get(output.decision, BANK_BLUE)
    dec_label = DECISION_LABEL.get(output.decision, output.decision)
    risk_lbl  = RISK_LABEL.get(output.decision, '—')

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — Executive Decision Summary
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph('TIER-1 BANK — AI CREDIT DECISION REPORT', S['bank']))
    story.append(Paragraph('Credit Decisioning System  ·  Confidential', S['small']))
    story.append(Spacer(1, 0.2*cm))
    story.append(HRFlowable(width='100%', thickness=2, color=BANK_BLUE, spaceAfter=8))

    summary_data = [
        ['Field', 'Value'],
        ['Application ID',          str(meta.get('id', output.prospectid or '—'))],
        ['Applicant Name',           meta.get('name', '—')],
        ['Loan Amount Requested',    f"₹{meta.get('loan_amount', 0):,.0f}"],
        ['Loan Purpose',             meta.get('loan_purpose', '—')],
        ['Report Generated',         now],
    ]
    t = Table(summary_data, colWidths=[6*cm, 11*cm])
    t.setStyle(_tbl_style())
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Decision banner
    banner_data = [
        ['CREDIT DECISION', 'RISK RATING', 'MODEL CONFIDENCE'],
        [dec_label,          risk_lbl,       f'{output.probability*100:.1f}%  ({output.confidence})'],
    ]
    bt = Table(banner_data, colWidths=[7*cm, 5*cm, 5*cm])
    bs = TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), BANK_BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0), 8),
        ('BACKGROUND',  (0,1), (-1,1), dec_col),
        ('TEXTCOLOR',   (0,1), (-1,1), colors.white),
        ('FONTNAME',    (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,1), (-1,1), 11),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.white),
        ('ROUNDEDCORNERS', [4]),
    ])
    bt.setStyle(bs)
    story.append(bt)
    story.append(Spacer(1, 0.3*cm))

    # Model info + rules
    model_data = [
        ['Model Used',    'Rules Fired',                          'Processing Time'],
        [output.model_used.replace('_',' ').title(),
         ', '.join(output.rules_fired) if output.rules_fired else 'None',
         f'{output.processing_ms} ms'],
    ]
    mt = Table(model_data, colWidths=[5*cm, 8*cm, 4*cm])
    mt.setStyle(_tbl_style(DARK_GREY))
    story.append(mt)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(f'<b>Decision Basis:</b> {output.explanation}', S['body']))

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — Top Driving Factors
    # ══════════════════════════════════════════════════════════════
    factors = explain_result.get('factors', [])
    if factors:
        story.append(Paragraph('TOP DRIVING FACTORS', S['h2']))
        story.append(Paragraph(
            'Features with the highest influence on this decision, ranked by magnitude. '
            '<b>✅</b> = supports approval &nbsp;&nbsp; <b>⚠</b> = risk signal.',
            S['small']))
        story.append(Spacer(1, 0.2*cm))

        factor_data = [['#', 'Factor', 'Applicant Value', 'Signal', 'Weight', 'Contribution']]
        for i, f in enumerate(factors, 1):
            icon = '✅' if f['direction'] == 'positive' else ('⚠' if f['direction'] == 'negative' else '—')
            bench_txt = f"  (benchmark: {f['benchmark']})" if f.get('benchmark') is not None else ''
            factor_data.append([
                str(i),
                f['label'],
                f['description'] + bench_txt,
                icon,
                f'{f["magnitude"]*100:.1f}%',
                _bar(f['magnitude']),
            ])

        ft = Table(factor_data, colWidths=[0.6*cm, 4*cm, 6*cm, 0.8*cm, 1.2*cm, 4.4*cm])
        fs = _tbl_style()
        for i, f in enumerate(factors, 1):
            c_idx = 3
            if f['direction'] == 'negative':
                fs.add('TEXTCOLOR', (c_idx, i), (c_idx, i), REJECT_RED)
                fs.add('FONTNAME',  (c_idx, i), (c_idx, i), 'Helvetica-Bold')
            elif f['direction'] == 'positive':
                fs.add('TEXTCOLOR', (c_idx, i), (c_idx, i), APPROVE_GRN)
        ft.setStyle(fs)
        story.append(ft)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3 — Default Risk Assessment
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph('DEFAULT RISK ASSESSMENT', S['h2']))

    # Approval probability bar
    ap = output.probability
    ap_data = [
        ['Approval Probability', f'{ap*100:.1f}%', _pd_gauge(ap)],
    ]

    # Default probability (proxy or real)
    dp = output.default_probability
    if dp is not None:
        dp_risk = 'LOW' if dp < 0.2 else ('MODERATE' if dp < 0.4 else ('HIGH' if dp < 0.6 else 'VERY HIGH'))
        dp_col  = APPROVE_GRN if dp < 0.2 else (COND_ORG if dp < 0.4 else REJECT_RED)
        risk_data = [
            ['Metric', 'Value', 'Risk Gauge', 'Assessment'],
            ['Approval Probability',
             f'{ap*100:.1f}%',
             _pd_gauge(ap),
             'HIGH' if ap > 0.85 else ('MEDIUM' if ap > 0.65 else 'LOW')],
            ['Proxy Default Probability (12m)',
             f'{dp*100:.1f}%',
             _pd_gauge(dp),
             dp_risk],
        ]
        rt = Table(risk_data, colWidths=[5*cm, 2*cm, 7.5*cm, 2.5*cm])
        rs = _tbl_style()
        # Colour the default probability row
        rs.add('TEXTCOLOR', (3, 2), (3, 2), dp_col)
        rs.add('FONTNAME',  (3, 2), (3, 2), 'Helvetica-Bold')
        rt.setStyle(rs)
        story.append(rt)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            f'<b>Note on Default Probability:</b> This is a proxy estimate based on '
            f'Tot_Missed_Pmnt patterns on internal accounts. It reflects the applicant\'s '
            f'historical missed payment tendency, not a forward-looking 90+ DPD prediction. '
            f'A real outcome model will be available once 500+ observed outcomes are logged '
            f'(Phase B upgrade, approx. 12–18 months).',
            S['small']))
    else:
        # No TL history — thin-file applicant
        rt = Table(
            [['Metric', 'Value', 'Risk Gauge'],
             ['Approval Probability', f'{ap*100:.1f}%', _pd_gauge(ap)]],
            colWidths=[5*cm, 2*cm, 10*cm]
        )
        rt.setStyle(_tbl_style())
        story.append(rt)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            '<b>Note:</b> No trade line history found for this applicant. '
            'Default probability model requires at least one active trade line. '
            'This applicant was evaluated on alternative data signals (income verification, '
            'utility payments, salary credits).',
            S['small']))

    # ══════════════════════════════════════════════════════════════
    # SECTION 3.5 — Interest Rate Breakdown
    # ══════════════════════════════════════════════════════════════
    if output.decision in ('P2', 'P3', 'P4'):
        rate = _calc_rate(
            decision            = output.decision,
            loan_purpose        = meta.get('loan_purpose', 'Personal'),
            loan_amount         = meta.get('loan_amount', 0),
            loan_tenure_months  = getattr(output, 'loan_tenure_months', 36) or 36,
            credit_score        = getattr(output, 'credit_score', 0) or 0,
            income_source       = getattr(output, 'income_source', 'salaried') or 'salaried',
            default_probability = output.default_probability,
        )
        if rate:
            story.append(Paragraph('INDICATIVE INTEREST RATE', S['h2']))
            story.append(Paragraph(
                'Rate = RBI Repo Rate + Operational Margin + Risk Premium. '
                'Risk premium is determined by approval class, loan purpose, '
                'credit score, tenure, and proxy default probability.',
                S['small']))
            story.append(Spacer(1, 0.2*cm))

            # Rate breakdown table
            band_col = {'''+'BEST'+''': APPROVE_GRN, '''+'STANDARD'+''': BANK_BLUE,
                        '''+'ELEVATED'+''': COND_ORG, '''+'HIGH'+''': REJECT_RED}.get(rate.rate_band, BANK_BLUE)
            rate_data = [
                ['Component', 'Rate', 'Notes'],
                ['RBI Repo Rate',        f'{rate.repo_rate:.2f}%',          'RBI policy rate — fixed'],
                ['Operational Margin',   f'{rate.operational_margin:.2f}%', 'Bank cost spread — fixed'],
                ['Risk Premium',         f'{rate.total_risk_premium:.2f}%', rate.rate_rationale[:60]],
                ['TOTAL ANNUAL RATE',    f'{rate.annual_rate:.2f}%',        f'Band: {rate.rate_band}'],
            ]
            rt2 = Table(rate_data, colWidths=[5*cm, 3*cm, 9*cm])
            rs2 = _tbl_style()
            rs2.add('BACKGROUND',  (0, 4), (-1, 4), band_col)
            rs2.add('TEXTCOLOR',   (0, 4), (-1, 4), colors.white)
            rs2.add('FONTNAME',    (0, 4), (-1, 4), 'Helvetica-Bold')
            rs2.add('FONTSIZE',    (0, 4), (-1, 4), 10)
            rt2.setStyle(rs2)
            story.append(rt2)
            story.append(Spacer(1, 0.3*cm))

            # EMI breakdown
            emi_data = [
                ['Loan Amount', 'Tenure', 'Monthly EMI', 'Total Payable', 'Total Interest', 'Effective Cost'],
                [f'₹{rate.loan_amount:,.0f}',
                 f'{rate.loan_tenure_months} months',
                 f'₹{rate.emi:,.0f}',
                 f'₹{rate.total_payable:,.0f}',
                 f'₹{rate.total_interest:,.0f}',
                 f'{rate.effective_cost:.1f}%'],
            ]
            et2 = Table(emi_data, colWidths=[3*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 2.5*cm])
            et2.setStyle(_tbl_style(DARK_GREY))
            story.append(et2)
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                '<b>Note:</b> This is an indicative rate only. Final rate is subject to '
                'credit committee approval, documentation verification, and prevailing '
                'RBI MCLR guidelines at the time of disbursement.',
                S['small']))

    # ══════════════════════════════════════════════════════════════
    # SECTION 4 — Improvement Roadmap
    # ══════════════════════════════════════════════════════════════
    roadmap = explain_result.get('roadmap', [])
    if roadmap:
        story.append(PageBreak())
        story.append(Paragraph('HOW TO IMPROVE YOUR APPROVAL CHANCES', S['h2']))
        story.append(Paragraph(
            'The following steps are tailored to this applicant\'s credit profile. '
            'Completing higher-priority steps first will have the greatest impact. '
            'Re-apply after the recommended waiting period.',
            S['body']))
        story.append(Spacer(1, 0.3*cm))

        for step in roadmap:
            priority_col = {1: REJECT_RED, 2: COND_ORG, 3: BANK_BLUE, 4: DARK_GREY}.get(step['priority'], DARK_GREY)
            priority_lbl = {1: 'CRITICAL', 2: 'HIGH', 3: 'MEDIUM', 4: 'LOW'}.get(step['priority'], f"P{step['priority']}")

            step_block = [
                [Paragraph(f'<b>Priority {step["priority"]} — {priority_lbl}</b>', S['bold']),
                 Paragraph(f'<b>Expected Impact:</b> {step["impact"]}', S['small']),
                 Paragraph(f'<b>Timeline:</b> {step["timeline"]}', S['small'])],
                [Paragraph(step['action'], S['improve']), '', ''],
            ]
            st = Table(step_block, colWidths=[4*cm, 7*cm, 6*cm])
            ss = TableStyle([
                ('BACKGROUND',   (0,0), (-1,0), LIGHT_GREY),
                ('FONTNAME',     (0,0), (0, 0), 'Helvetica-Bold'),
                ('TEXTCOLOR',    (0,0), (0, 0), priority_col),
                ('SPAN',         (0,1), (-1,1)),
                ('BACKGROUND',   (0,1), (-1,1), colors.white),
                ('GRID',         (0,0), (-1,-1), 0.3, MID_GREY),
                ('LEFTPADDING',  (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING',   (0,0), (-1,-1), 5),
                ('BOTTOMPADDING',(0,0), (-1,-1), 5),
                ('VALIGN',       (0,0), (-1,-1), 'TOP'),
            ])
            st.setStyle(ss)
            story.append(KeepTogether(st))
            story.append(Spacer(1, 0.2*cm))

        # Re-application eligibility
        months = 6 if output.decision == 'P1' else 3
        from datetime import timedelta
        eligible = (datetime.now() + timedelta(days=30*months)).strftime('%d %b %Y')
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            f'<b>Re-Application Eligibility:</b> {eligible}  '
            f'({months} months from today)',
            S['bold']))
        story.append(Paragraph(
            'Applying before this date without addressing the above points is unlikely '
            'to change the outcome and may create additional bureau enquiries.',
            S['small']))

    elif output.decision == 'P2':
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            '✅  <b>No improvement steps required.</b>  '
            'This application meets all standard approval criteria.',
            S['body']))

    # ══════════════════════════════════════════════════════════════
    # SECTION 5 — Regulatory Audit Trail
    # ══════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('REGULATORY AUDIT TRAIL', S['h2']))

    audit_data = [
        ['Item', 'Detail'],
        ['Report Generated',     now],
        ['Final Decision',       dec_label],
        ['Model Version',        'v1.0 — Phase 12 Build'],
        ['Model Used',           output.model_used.replace('_',' ').title()],
        ['Explainability Method',explain_result.get('method', '—')],
        ['Rules Evaluated',      ', '.join(output.rules_fired) if output.rules_fired else 'None fired'],
        ['Default Prob Source',  'proxy (Tot_Missed_Pmnt)' if output.default_probability else 'N/A — no TL history'],
        ['Fair Lending Note',
         'This model does not use gender, religion, caste, national origin, or '
         'marital status as credit decision features. These fields are captured '
         'only for regulatory reporting and CIBIL bureau matching per RBI guidelines.'],
        ['Reject Inference',
         'Approval model trained on approved + rejected applicants. Rejected '
         'applicants labelled via Tot_Missed_Pmnt proxy on linked internal accounts.'],
        ['Data Sources',
         'D9 CIBIL Bureau + D9 Internal Bank Trade Lines + '
         'Alternative data (thin-file path only)'],
        ['Model Approval Status','Pending — awaiting Model Risk Management sign-off'],
        ['Loan Officer Review',  '________________________'],
        ['Date of Review',       '________________________'],
        ['Signature',            '________________________'],
    ]
    at = Table(audit_data, colWidths=[5*cm, 12*cm])
    at.setStyle(_tbl_style())
    story.append(at)

    doc.build(story)
    return output_path
