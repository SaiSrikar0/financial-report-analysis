# FinCast — Financial Forecasting and Recommendation System

**Last Updated:** March 12, 2026

---

## 1. Project Overview

FinCast is an end‑to‑end financial analytics platform that automates financial data collection, performs in‑depth analysis, forecasts future performance, and generates explainable, actionable recommendations.

| Layer | Responsibility |
|---|---|
| Data Retrieval | Alpha Vantage API ingestion |
| ETL Pipeline | Extract → Transform → Load into Supabase |
| Analysis Engine | Feature engineering, trend/peer/outlier analysis |
| ML Forecasting | SVR growth‑rate prediction with confidence intervals |
| Explainability | SHAP global + local feature contribution analysis |
| Recommendation Engine | LLM‑based narrative generation (Phase 6) |
| Dashboard | Interactive Streamlit visualization |

**Core contribution:** Target‑aware financial forecasting with explainable, data‑driven recommendations — going beyond traditional historical reporting tools.

---

## 2. Objectives

- Automate financial data collection from reliable financial APIs
- Build a flexible ETL pipeline for heterogeneous financial data
- Engineer financial features and analyze historical statements
- Predict future growth rates using SVR with gap analysis vs targets
- Explain predictions using SHAP feature contributions
- Generate actionable LLM‑based natural language recommendations
- Visualize all layers in an interactive multi‑tab dashboard


---

## 3. System Architecture

```
Financial Data Sources (Alpha Vantage API / Uploaded Data)
                │
                ▼
        Data Retrieval Layer
        data_retrieval/retrieve_api.py
                │
                ▼
            ETL Pipeline
      Extract → Transform → Load
      etl/extract.py → etl/transform.py → etl/load.py
                │
                ▼
     Supabase PostgreSQL Database
     standard_table (ML) + category_table (LLM)
                │
                ▼
    Financial Analysis Engine (analysis/)
    Trend · Peer Comparison · Feature Engineering
                │
                ▼
    SVR Growth-Rate Prediction (models/svr_pipeline.py)
    GridSearchCV · Confidence Intervals · Gap Analysis
                │
                ▼
    SHAP Explainability Layer (models/explainability.py)
    Global Importance · Local Contributions per Company
                │
                ▼
    LLM Recommendation Engine (Phase 6 — pending)
                │
                ▼
    Interactive Streamlit Dashboard (app.py)
```

---

## 4. Data Sources

Financial data is retrieved from the **Alpha Vantage API** for AAPL, MSFT, and GOOGL (2006–2025).

**Collected statements:**
- Income Statement: Revenue, Operating Income, Net Income
- Balance Sheet: Total Assets, Total Liabilities
- Cash Flow Statement: Operating Cash Flow

**Example record:**
```json
{
  "date": "2024-09-30",
  "ticker": "AAPL",
  "revenue": 391035000000,
  "operating_income": 123216000000,
  "net_income": 93736000000,
  "total_assets": 364980000000,
  "total_liabilities": 308030000000,
  "operating_cashflow": 118254000000
}
```

**Dataset scale:** 60 records × 3 companies × 20 years

---

## 5. ETL Pipeline

### 5.1 Extract
Adapter‑based extraction supports: Financial APIs · CSV · Excel · User uploads  
Raw output: `data/raw/financial_data_raw.json`

### 5.2 Transform
**Data cleaning:** missing value handling, type normalization, schema standardization

**Engineered features:**

| Feature | Formula |
|---|---|
| `profit_margin` | `net_income / revenue` |
| `operating_margin` | `operating_income / revenue` |
| `debt_to_asset` | `total_liabilities / total_assets` |
| `asset_efficiency` | `revenue / total_assets` |
| `revenue_growth` | YoY % change in revenue |
| `net_income_growth` | YoY % change in net income |

**Output tables:**
- `data/staged/standard_table.csv` — 60 rows × 14 features for ML training
- `data/staged/category_table.csv` — 60 rows with business‑context categories for LLM

### 5.3 Load
Batch‑insert both tables into Supabase (PostgreSQL).  
Tables: `standard_table` (ML), `category_table` (recommendations)

---

## 6. Machine Learning Model (Phase 4)

**Model:** Support Vector Regression (SVR) via Scikit‑learn

**Why SVR:** Handles structured numerical data, captures nonlinear relationships, works well on moderate‑sized datasets.

**Prediction task:** Forecast **next‑period revenue growth rate (% YoY)** — not absolute values, enabling interpretable gap analysis vs targets.

**Input features (14):**
`revenue`, `operating_income`, `net_income`, `total_assets`, `total_liabilities`, `operating_cashflow`, `profit_margin`, `operating_margin`, `debt_to_asset`, `asset_efficiency`, `revenue_growth`, `net_income_growth`, `ticker_AAPL`, `ticker_GOOGL`

**Training strategy:**
- Time‑aware chronological split (no data leakage)
- GridSearchCV hyperparameter optimization
- Best params: kernel=`linear`, C=`1`, epsilon=`0.01`, gamma=`scale`

**Evaluation results:**
| Metric | Value |
|---|---|
| MAE | 106.40% |
| RMSE | 165.18% |
| R² | −0.0957 |

**Gap analysis:** Predicted growth vs target (10%) → shortfall/surplus with 95% confidence intervals

---

## 7. SHAP Explainability (Phase 5)

SHAP (SHapley Additive exPlanations) is applied to the trained SVR model to explain prediction drivers.

**Global importance:** Mean absolute SHAP values ranked across all features  
**Top features:** `asset_efficiency` (8.39), `total_assets` (5.14), `profit_margin` (5.06)

**Local explanations:** Per‑company feature contributions showing which factors increase or decrease the predicted growth rate

**Outputs:**
- `analysis/reports/phase_5_shap_global_importance.csv`
- `analysis/reports/phase_5_shap_local_explanations.csv`
- `analysis/reports/phase_5_shap_future_predictions.csv`

---

## 8. Recommendation Engine (Phase 6 — Pending)

An LLM converts structured analysis outputs into natural language recommendations.

**LLM inputs:**
- SVR predicted growth rate and confidence interval
- Gap vs target (shortfall/surplus)
- Top positive and negative SHAP feature contributions
- Key financial ratios

**LLM output example:**
> The company shows strong revenue growth and healthy profitability. However, the high debt ratio indicates reliance on external financing. Reducing liabilities or improving asset utilization could improve financial stability.

**Constraints:** LLM generates narrative only — no numeric prediction. No RAG. Grounded strictly in model outputs.

---

## 9. Dashboard (Phase 7 — Pending)

Interactive Streamlit dashboard with three tabs:

| Tab | Content |
|---|---|
| 📊 Historical Analysis | KPI cards · Revenue trend · Bar comparisons · 3D scatter · Data table |
| 🎯 SVR Predictions | Model metrics · Hyperparameters · CI visualization · Predicted vs actual |
| 🔍 SHAP Explainability | Global feature importance · Local contributions per company |

**Sidebar controls:** Company filter · Year range · Data source toggle · Metric selector

---

## 10. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas · NumPy |
| Data Retrieval | Alpha Vantage API |
| Database | Supabase (PostgreSQL) |
| Machine Learning | Scikit‑learn (SVR + GridSearchCV) |
| Explainability | SHAP (KernelExplainer) |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Environment | python‑dotenv |

---

## 11. Project Folder Structure

```
financial-report-analysis/
├── app.py                          ← Streamlit dashboard (Phases 1–5 integrated)
├── run.py                          ← Unified CLI orchestrator
├── data/
│   ├── raw/financial_data_raw.json ← Source data (60 records)
│   └── staged/
│       ├── standard_table.csv      ← ML-ready (60 rows × 14 features)
│       └── category_table.csv      ← LLM-ready categories
├── data_retrieval/retrieve_api.py  ← Alpha Vantage API adapter
├── etl/
│   ├── extract.py                  ← Multi-source extraction adapters
│   ├── transform.py                ← Feature engineering pipeline
│   └── load.py                     ← Supabase batch loader
├── analysis/
│   ├── data_connection.py
│   ├── historical_performance.py
│   ├── trend_analysis.py
│   ├── peer_comparison.py
│   ├── insights.py
│   ├── feature_analysis.py
│   ├── timeseries_analysis.py
│   ├── outlier_treatment.py
│   ├── feature_preprocessing.py
│   └── reports/                    ← All generated CSVs + PNGs
├── models/
│   ├── svr_pipeline.py             ← Phase 4: SVR training + gap analysis
│   └── explainability.py           ← Phase 5: SHAP global + local explanations
└── scripts/
    ├── run_feature_analysis.py
    └── clear_and_reload.py
```

---

## 12. Phase Summary

| Phase | Description | Status |
|---|---|---|
| 1 | Project Planning & Design | ✅ Complete |
| 2 | ETL Pipeline | ✅ Complete |
| 3.1 | Financial Analysis | ✅ Complete |
| 3.2 | Feature Analysis | ✅ Complete |
| 4 | SVR Forecasting | ✅ Complete |
| 5 | SHAP Explainability | ✅ Complete |
| 6 | LLM Recommendations | ⏳ Pending |
| 7 | Streamlit Dashboard | 🔄 In Progress |
| 8 | Testing & Validation | ⏳ Pending |
| 9 | Documentation & Deployment | ⏳ Pending |
