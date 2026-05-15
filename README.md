---
title: Loan Decision
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

# AI Loan Decisioning System

An AI-powered loan decisioning system using a 3-layer ensemble model (Random Forest + ExtraTrees → GBM meta + WoE Scorecard) trained on 51,336 CIBIL bureau records.

## Features
- Single and batch loan application scoring
- SHAP-based explainability
- PDF decision reports
- 4-class output: P1 (Reject), P2 (Approve), P3 (Conditional), P4 (Refer)
