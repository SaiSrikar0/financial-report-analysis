# FinCast

FinCast is an app-first financial analytics and forecasting prototype built with Streamlit, Supabase, SVR, and SHAP.

## Prototype Scope

FinCast currently delivers an end-to-end prototype that can:

- ingest financial data from API and uploads (CSV, Excel, JSON, PDF)
- validate and transform data into ML-ready features
- train and evaluate an SVR model for growth-rate forecasting
- explain predictions with SHAP (global + local)
- generate grounded AI recommendations from model outputs
- present analysis, prediction, and explainability in one dashboard

## Architecture Summary

```text
Data Retrieval (API / uploads)
  -> ETL (extract, transform, load)
  -> Supabase tables (standard_table, category_table)
  -> Analysis + Feature Engineering
  -> SVR Forecasting (Phase 4)
  -> SHAP Explainability (Phase 5)
  -> LLM Recommendations (Phase 6)
  -> Streamlit App (app.py)
```

Main modules:

- `app.py`: Streamlit dashboard entrypoint (recommended runtime)
- `etl/`: extraction, transformation, loading, validation
- `analysis/`: historical analysis, feature analytics, recommendation engine
- `models/`: SVR training pipeline + SHAP explainability
- `auth/`: Supabase authentication helpers
- `scripts/run.py`: phase orchestrator for CLI runs

## Quick Start

### 1) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create `.env` in the project root:

```env
ALPHAVANTAGE_API_KEY=your_key
ALPHAVANTAGE_BASE_URL=https://www.alphavantage.co/query

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

GROQ_API_KEY=your_groq_key
```

### 4) Run the app (primary path)

```bash
streamlit run app.py
```

## Run Modes

### App mode (recommended)

- launch Streamlit with `app.py`
- authenticate
- upload/use data
- review analysis, predictions, SHAP, and recommendations

### CLI mode (maintenance and phased runs)

Run the full pipeline:

```bash
python scripts/run.py all
```

Run a single phase:

```bash
python scripts/run.py 4 --target-growth 10
python scripts/run.py 5 --shap-nsamples 200
python scripts/run.py 6
```

Additional maintenance utilities:

- `scripts/maintenance/debug_database.py`
- `scripts/maintenance/fix_rls_policies.py`

## Validation and Smoke Test

Use the lightweight smoke test to verify basic wiring after setup:

```bash
python scripts/smoke_test.py
```

This verifies imports, required directories, expected report artifacts, and environment variable presence.

## Documentation

See detailed project docs in `docs/`:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROGRESS.md`
- `docs/UPLOADED_DATA_ANALYSIS_GUIDE.md`
- `docs/SVR_DATA_RETRIEVAL_ARCHITECTURE.md`
- `docs/USER_DATA_INTEGRATION.md`
- `docs/DASHBOARD_FIX_INSTRUCTIONS.md`
- `docs/PROTOTYPE_RELEASE_CHECKLIST.md`

## Known Prototype Limitations

- model performance is baseline and still under optimization
- production hardening (full test coverage, CI, deployment guardrails) is pending
- recommendation quality depends on available upstream artifacts and keys

## License

Internal/project use unless a separate license file is added.
