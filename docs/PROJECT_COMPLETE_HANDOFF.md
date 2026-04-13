# FinCast Complete Project Summary and Handoff Guide

**Last Updated:** 2026-04-09

This document is the single handoff reference for new team members. It explains what the project does, how it works, how the implementation is organized, and how the prototype was brought from the initial assessment state to the finished product.

---

## 1. Executive Summary

FinCast is an app-first financial analytics and forecasting prototype. It ingests financial data, standardizes it into analysis-ready tables, runs historical and feature analysis, trains an SVR model to forecast next-period growth, explains the model with SHAP, and generates recommendation narratives grounded in model outputs.

The project is designed so that the main experience lives in the Streamlit app. CLI scripts exist for maintenance, phased execution, and validation, but the product is primarily meant to be used through the dashboard.

### What the prototype currently delivers

- Data ingestion from uploaded files and API-backed workflows
- ETL normalization into ML-ready and LLM-ready tables
- Historical analysis, trend analysis, ratio analysis, peer comparison, and insight extraction
- SVR-based growth-rate forecasting with target-gap analysis
- SHAP-based global and local explainability
- Recommendation generation from structured outputs
- An authenticated Streamlit dashboard that ties the workflow together

### Core business idea

The key idea is not just to predict future performance, but to explain why a prediction is made and convert that explanation into actionable guidance.

---

## 2. Problem Statement

Traditional financial reporting tools usually provide one of two things:

1. historical reporting with limited forecasting, or
2. forecasting models with limited explanation.

FinCast combines both. It is meant to answer:

- What has happened historically?
- What patterns exist in the financials?
- What is likely to happen next?
- Why is that prediction being made?
- What should a user do about it?

This makes the prototype useful for analysts, product reviewers, or future engineers who want a single pipeline from raw data to narrative recommendation.

---

## 3. System Overview

### High-level flow

```text
Data Source / Uploads
        ->
ETL (extract, transform, validate, load)
        ->
Supabase tables
        ->
Analysis modules
        ->
SVR model training and prediction
        ->
SHAP explainability
        ->
Recommendation generation
        ->
Streamlit dashboard
```

### Main layers

| Layer | Role |
|---|---|
| Data retrieval | Pull or accept source financial data |
| ETL | Clean, normalize, and load data |
| Analysis | Historical, trend, peer, and feature analysis |
| ML | Train SVR and produce growth forecasts |
| Explainability | Use SHAP to expose model drivers |
| Recommendations | Convert structured outputs into narrative advice |
| Dashboard | Present the complete workflow to the user |

---

## 4. Architecture and Implementation

### 4.1 Data ingestion

The project supports financial data coming from API-backed retrieval and user uploads. The repository already contains raw and staged data paths, and the implementation supports a structured JSON-based raw store as well as CSV-based staged tables.

Relevant locations:

- [data/raw/financial_data_raw.json](../data/raw/financial_data_raw.json)
- [data/staged/standard_table.csv](../data/staged/standard_table.csv)
- [data/staged/category_table.csv](../data/staged/category_table.csv)

### 4.2 ETL pipeline

The ETL layer is responsible for converting heterogeneous financial input into stable downstream tables.

The core ETL responsibilities are:

- extract data from supported sources
- validate the shape and required fields
- normalize fields and dates
- engineer ratios and growth features
- load the results into Supabase

Important implementation files:

- [etl/extract.py](../etl/extract.py)
- [etl/transform.py](../etl/transform.py)
- [etl/load.py](../etl/load.py)
- [etl/validator.py](../etl/validator.py)

### 4.3 Analysis layer

The analysis layer produces the non-ML analytical foundation for the app and the recommendation engine.

This includes:

- historical performance analysis
- trend analysis
- peer comparison
- insight extraction
- feature analysis
- outlier treatment
- preprocessing support
- uploaded-data auto-analysis for newly added company data

Important implementation files:

- [analysis/data_connection.py](../analysis/data_connection.py)
- [analysis/historical_performance.py](../analysis/historical_performance.py)
- [analysis/trend_analysis.py](../analysis/trend_analysis.py)
- [analysis/peer_comparison.py](../analysis/peer_comparison.py)
- [analysis/insights.py](../analysis/insights.py)
- [analysis/feature_analysis.py](../analysis/feature_analysis.py)
- [analysis/timeseries_analysis.py](../analysis/timeseries_analysis.py)
- [analysis/outlier_treatment.py](../analysis/outlier_treatment.py)
- [analysis/feature_preprocessing.py](../analysis/feature_preprocessing.py)
- [analysis/auto_analysis.py](../analysis/auto_analysis.py)
- [analysis/recommendation_engine.py](../analysis/recommendation_engine.py)

### 4.4 Machine learning layer

The ML layer uses Support Vector Regression to forecast next-period growth rate. The important design choice is that the model predicts growth rate rather than absolute revenue, because growth rate is easier to reason about in business terms and works better for gap analysis against a target.

Core responsibilities:

- prepare ML-ready features
- split data in a time-aware way
- train and tune SVR
- evaluate holdout performance
- create future predictions
- compute target gap and confidence intervals

Important implementation file:

- [models/svr_pipeline.py](../models/svr_pipeline.py)

### 4.5 Explainability layer

The explainability layer uses SHAP to answer why the model produced its outputs.

Outputs include:

- global feature importance
- local explanations for individual companies
- future prediction explainability artifacts

Important implementation file:

- [models/explainability.py](../models/explainability.py)

### 4.6 Recommendation layer

The recommendation engine converts structured model outputs and analysis bundles into natural language output.

It consumes:

- SVR predicted growth and confidence intervals
- target gap and status
- SHAP feature contributions
- key financial ratios and peer context

The current implementation is grounded in the generated artifacts and is designed to remain useful even when the LLM provider is unavailable or rate-limited.

Important implementation file:

- [analysis/recommendation_engine.py](../analysis/recommendation_engine.py)

### 4.7 Dashboard layer

The dashboard is the main end-user entrypoint. It handles authentication, uploaded data review, historical analysis, SVR outputs, SHAP views, and recommendations.

Main entrypoint:

- [app.py](../app.py)

### 4.8 CLI orchestration

The CLI orchestrator exists for running phases individually or as a full pipeline.

Main entrypoint:

- [scripts/run.py](../scripts/run.py)

---

## 5. Data Model and Key Outputs

### 5.1 Core tables

| Table | Purpose |
|---|---|
| `standard_table` | ML-ready feature table |
| `category_table` | Recommendation-oriented context table |
| `uploaded_files` | Tracks user uploaded raw data |
| recommendation outputs | Stores generated recommendation results |

### 5.2 Engineered features

The feature engineering layer derives financial ratios and growth indicators used by both analysis and ML.

Common features include:

- profit margin
- operating margin
- debt-to-asset ratio
- asset efficiency
- revenue growth
- net income growth

### 5.3 Generated artifacts

The pipeline writes reports under [analysis/reports](../analysis/reports).

Examples include:

- SVR metrics and predictions
- SHAP global and local explanation files
- peer rankings and historical analysis summaries
- preprocessing and feature-analysis outputs

These artifacts are the bridge between the backend pipeline and the dashboard.

---

## 6. User Flow in the App

The normal user experience is:

1. Open the Streamlit app.
2. Authenticate.
3. Upload or select data.
4. Review analysis and derived features.
5. Train or inspect the SVR outputs.
6. Review SHAP explanations.
7. Generate or inspect recommendations.

The app is intentionally structured to keep the workflow visible and interactive rather than hiding everything behind a single script.

---

## 7. Development History: Task 0 to Finished Product

This section is the practical story of how the prototype was brought to its current finished state.

### Task 0: Establish the final prototype state

The starting point was the realization that the project was already effectively a usable prototype. From that point, the goal changed from building core features to making the system understandable, runnable, and handoff-ready.

Key intent at this stage:

- confirm the prototype had enough substance to hand off
- identify gaps in documentation and runtime reliability
- decide what still needed packaging for a new contributor

### Task 1: Rewrite the README for onboarding

The initial README was converted into a clearer project introduction with:

- a concise prototype scope
- architecture summary
- app-first run guidance
- CLI run examples
- smoke test instructions
- known prototype limitations

Outcome: a new team member can now understand the project at a glance and know where to start.

### Task 2: Validate dependencies and environment setup

The environment was checked to ensure the project could actually run in its current venv.

What was done:

- configured the workspace Python environment
- checked installed packages
- ran `pip check`
- identified compatibility issues with `httpx` and `websockets`
- pinned compatible versions in [requirements.txt](../requirements.txt)
- reinstalled the affected packages
- reran `pip check` until the environment was clean

Outcome: dependency management is now predictable and reproducible.

### Task 3: Add a smoke test

A new smoke test script was created to give future contributors a quick validation command.

The smoke test checks:

- critical imports
- folder structure
- required `.env` keys
- presence of expected report artifacts

Outcome: the project now has a low-cost validation step for onboarding, debugging, and release checks.

### Task 4: Create a release checklist and demo notes

To support handoff, a release checklist and demo runbook were added.

This document helps answer:

- what is complete
- what remains optional
- how to demo the system
- what the known limitations are
- what the next milestone should be

Outcome: the project now has a repeatable release and demo reference.

### Cleanup step: remove cache noise

Running the smoke test surfaced tracked `__pycache__` files. Those were removed from version control so future diffs stay clean.

Outcome: the repository stopped tracking generated bytecode artifacts.

### Packaging step: split commits by concern

The final work was committed in three separate groups:

1. docs, dependency cleanup, smoke test, and release checklist
2. application and ETL behavior changes
3. report snapshot artifacts

Outcome: the history is easier to review, and changes are isolated by purpose.

### Publishing step: push to GitHub

The final commit set was pushed to the remote repository.

Outcome: the prototype is now visible and shareable outside the local workspace.

---

## 8. What Each Final Commit Covers

This is useful if a future maintainer wants to understand where the current state came from.

| Commit | Purpose |
|---|---|
| `fcd1360` | Prototype docs, smoke test, dependency fixes, and cache cleanup |
| `ba17fcd` | Upload-to-training flow improvements and Supabase write robustness |
| `4f91e04` | Snapshot update for generated analysis reports |

---

## 9. How to Run the Project

### Recommended app path

```bash
streamlit run app.py
```

### CLI pipeline path

Run all phases:

```bash
python scripts/run.py all
```

Run a specific phase:

```bash
python scripts/run.py 4 --target-growth 10
python scripts/run.py 5 --shap-nsamples 200
python scripts/run.py 6
```

### Smoke test

```bash
python scripts/smoke_test.py
```

---

## 10. Environment and Configuration

The project expects a `.env` file in the repository root with at least:

```env
ALPHAVANTAGE_API_KEY=...
ALPHAVANTAGE_BASE_URL=...
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
GROQ_API_KEY=...
```

The prototype also relies on the local Python environment and the pinned package set in [requirements.txt](../requirements.txt).

---

## 11. Current Status

The project is currently in a strong prototype state.

### What is complete

- data flow and ETL
- analysis pipeline
- SVR model and SHAP explainability
- recommendation generation
- dashboard integration
- documentation and release support
- smoke test validation
- GitHub push and commit history organization

### What is still prototype-level

- model performance is baseline and can be improved
- CI is not yet fully automated
- more unit and integration tests are needed
- deployment hardening is still pending

---

## 12. How a New Team Member Should Approach the Codebase

If you are new to the project, read and do these in order:

1. Read this document first.
2. Read [README.md](../README.md) for quick setup.
3. Read [docs/PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the fuller architecture story.
4. Read [docs/PROGRESS.md](PROGRESS.md) for milestone history.
5. Run the smoke test.
6. Launch the app.
7. Inspect [analysis/reports](../analysis/reports) to understand the generated artifacts.
8. Trace the flow through [app.py](../app.py), [analysis/recommendation_engine.py](../analysis/recommendation_engine.py), and [models/svr_pipeline.py](../models/svr_pipeline.py).

This path gives the fastest understanding of both the user experience and the backend implementation.

---

## 13. Recommended Next Improvements

If the next contributor wants to improve the prototype, the best next targets are:

- add CI to run the smoke test automatically
- add focused unit tests for ETL and model preparation
- make the recommendation pipeline more resilient to API failures
- improve model performance and feature selection
- add clearer deployment/staging instructions
- standardize report regeneration so snapshots are reproducible

---

## 14. Final Takeaway

FinCast is a prototype that already connects the full path from raw financial data to explanation and recommendation. The work completed in this workspace did not just add features; it also made the project understandable, testable, and handoff-ready.

If you are extending the project, your job is not to rebuild the core pipeline. Your job is to improve reliability, model quality, test coverage, deployment readiness, and the quality of the outputs.