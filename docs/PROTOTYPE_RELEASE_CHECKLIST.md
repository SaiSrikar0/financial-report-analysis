# FinCast Prototype Release Checklist

Last updated: 2026-04-09

## 1. Release Readiness Checklist

### Documentation

- [x] README includes setup, run modes, architecture summary, and prototype limitations
- [x] Core docs exist in docs/ for architecture, progress, and data integration
- [x] Prototype smoke test command is documented

### Environment and Dependencies

- [x] Virtual environment selected and active
- [x] requirements.txt updated with compatible dependency ranges
- [x] `pip check` passes with no broken requirements
- [x] Required environment keys are present in `.env`

### Application Runtime

- [x] Streamlit entrypoint is `app.py`
- [x] Dashboard can load with authentication gate in place
- [x] Core analysis artifacts available under `analysis/reports/`

### Pipeline and Model Artifacts

- [x] SVR outputs present (`svr_evaluation_metrics.csv`, `svr_future_predictions.csv`)
- [x] SHAP outputs present (`phase_5_shap_global_importance.csv`, `phase_5_shap_local_explanations.csv`)
- [x] Recommendation JSON artifacts generated for sample tickers

### Quality Gate

- [x] Smoke test added: `python scripts/smoke_test.py`
- [x] Smoke test currently passes in workspace environment
- [ ] Optional: add automated CI run for smoke test

## 2. Demo Runbook (10-12 Minutes)

### Demo Goal

Show end-to-end value: ingest/prepare data, forecast growth, explain model behavior, and produce recommendations in one flow.

### Suggested Sequence

1. Open app:
   - `streamlit run app.py`
2. Authenticate with test user.
3. Show historical analysis panel and key KPIs.
4. Move to SVR section:
   - show evaluation metrics
   - show future prediction output and target-gap view
5. Move to SHAP section:
   - show global importance
   - show one local explanation for a ticker
6. Move to recommendation output:
   - show generated recommendation JSON and rationale consistency with SHAP drivers
7. Close with limitations and next milestone plan.

### Backup Commands (if UI issue)

- Full CLI pipeline: `python scripts/run.py all`
- Smoke test: `python scripts/smoke_test.py`

## 3. Known Prototype Constraints

- Model performance is baseline and needs further feature and hyperparameter work.
- Observability and testing are lightweight (smoke-test level, not full CI test matrix).
- Recommendation quality depends on upstream artifact quality and available external API credentials.
- Security hardening and production deployment controls are not finalized.

## 4. Recommended Next Milestones

1. Add a CI workflow to run smoke test on each pull request.
2. Add focused unit tests for ETL transformation and model input validation.
3. Add model performance tracking and drift checks across runs.
4. Add deployment profile and environment templates for staging/production.
