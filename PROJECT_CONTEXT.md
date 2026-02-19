# FinCast: Financial Report Analysis and Recommendation System

## Project Context & Documentation

Abstract – Financial Report Analysis (FinCast)
1. Problem / Need
Financial reports are complex, heterogeneous, and time‑consuming to analyze manually.

Existing systems focus mainly on historical analysis and lack future prediction and explainable insights.

2. Objective
To develop an automated system that analyzes financial reports, predicts future performance, and provides explainable recommendations for decision‑making.

3. Methodology
Financial data is collected from multiple real‑world sources and processed using an ETL pipeline.

Key financial features are analyzed and used to train a Support Vector Regression (SVR) model for forecasting.

Explainable AI techniques and an LLM are used to generate interpretable, actionable recommendations.

4. Results / Benefits
Accurate financial forecasts with clear trend visualization.

Improved transparency through explainable insights.

Reduced manual effort and better decision support.

5. Conclusion / Impact
The project transforms traditional financial analysis into an intelligent, explainable, and future‑oriented decision‑support system.


## Project Overview

### Problem Statement

Financial reports such as income statements, balance sheets, and cash flow statements form the backbone of corporate decision‑making. However, these reports are typically large, complex, and heterogeneous, making manual analysis difficult, time‑consuming, and error‑prone.

**Key Challenges:**
- Traditional financial analysis approaches primarily focus on historical performance
- Lack of robust mechanisms for forecasting future outcomes
- Existing tools fail to provide explainable insights
- Difficulty translating numerical results into actionable business decisions

**Solution Need:** An automated, intelligent system capable of analyzing financial reports, predicting future performance, and supporting informed financial decision‑making through explainable insights.

---

## Proposed Solution

An end‑to‑end Financial Report Analysis system that:

1. **Automates** financial data collection, cleaning, and integration using a structured ETL pipeline
2. **Extracts and analyzes** key financial indicators to evaluate historical performance
3. **Applies** machine learning techniques to forecast future profit or growth
4. **Uses** explainable AI to generate interpretable, business‑oriented recommendations

---

## System Workflow

```
Financial Data Collection
    ↓
ETL Pipeline (Extract, Transform, Normalize)
    ↓
Centralized Database (PostgreSQL via Supabase)
    ↓
Financial Analysis & Feature Engineering
    ↓
SVR Model Training
    ↓
Prediction
    ↓
LLM‑Based Recommendation Generation
    ↓
User‑Readable Output & Visualizations
```

---

## Data Collection

### Sources
- Publicly available and reliable financial data sources
- Financial APIs
- Structured CSV / Excel datasets

### Dataset Strategy
- Multiple companies supported (multi‑company datasets)
- Historical financial statements (income, balance sheet, cash flow)
- Time‑ordered data (no random shuffling)

### Assumption
"Real‑time data" refers to real‑world, periodically updated financial data, not live streaming feeds.

---

## ETL Pipeline Design

### Extraction
- Data collected from APIs and structured files
- Adapter‑based extraction
- All inputs converted into a unified internal raw format (JSON)

### Transformation
- Data cleaning and normalization
- Mapping‑based standardization
- Feature engineering (revenue trends, profit margins, growth rates, expense patterns)

### Loading
- Integrated and cleaned data stored in a centralized PostgreSQL database via Supabase

---

## Database Design

### Design Principle
Hybrid schema to handle heterogeneous company financial structures without schema explosion.

### Tables

#### 1. Standardized Financial Table (for ML)
**Fields:**
- total_revenue
- operating_expenses
- net_profit
- growth_rate
- profit_margin

**Used for:**
- Model training
- Forecasting
- Comparative analysis

#### 2. Flexible Breakdown Table (for Analysis & Recommendations)
**Fields:**
- company_id
- year
- income/expense type
- category
- amount

**Used for:**
- Detailed financial analysis
- Recommendation generation

---

## Financial Analysis

**Key Indicators Computed:**
- Revenue trends
- Profit margins
- Growth rates
- Expense patterns

**Purpose:**
- Feature engineering to support predictive modeling

---

## Machine Learning

### Model Used
**Support Vector Regression (SVR)**

### Purpose
Predict future financial performance (profit or growth)

### Training Strategy
- Time‑aware train–test split
- Historical data used for training
- Evaluation using regression metrics (MAE, RMSE, R²)

### Prediction & Gap Identification
- Predicted values are compared against predefined financial targets
- System identifies:
  - Prediction shortfall or surplus
  - Magnitude of deviation
- Enables decision‑oriented analysis rather than raw forecasting

---

## Explainable AI Integration

### Meaning
Explainable AI is implemented through feature contribution analysis, identifying how individual financial features influence the model's prediction.

### Role
- Improves transparency and interpretability
- Provides analytical grounding for recommendations
- Prevents black‑box decision‑making

---

## Recommendation Generation (LLM‑Based)

### LLM Role
Converts analytical outputs into natural‑language recommendations

**Generates business‑oriented insights based on:**
- Predictions
- Target gaps
- Feature contributions

### Explicit Constraints
- LLM does not perform numeric prediction
- No Retrieval‑Augmented Generation (RAG)
- Recommendations are grounded strictly in model outputs and analysis

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| **Programming Language** | Python |
| **Database** | PostgreSQL (via Supabase) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit‑learn (SVR) |
| **Frontend / Interface** | Streamlit |
| **Explainable AI Layer** | LLM API |

---

## Key Features (Prototype Scope)

1. Automated financial data integration through an ETL pipeline
2. Multi‑company dataset support
3. Standardized and flexible financial data storage
4. In‑depth financial performance analysis and feature engineering
5. Predictive financial forecasting using Support Vector Regression
6. Visual representation of financial data
7. Explainable AI‑based recommendation generation
8. Prediction gap identification
9. User‑readable analytical output

---

## Unique Feature (Core Contribution)

**Target‑aware financial forecasting with explainable, data‑driven recommendations.**

This differentiates the system from traditional tools that stop at reporting or prediction.

---

## Constraints

- Academic prototype scope
- No live streaming data
- No dynamic database schema creation
- No GPU dependency
- No deep learning requirement
- No RAG usage

---

## Current Project Status

✅ Problem statement and solution finalized  
✅ System workflow defined and visualized  
✅ Features and technologies finalized  
✅ PPT prepared and aligned with implementation  
🚀 Ready for implementation, demonstration, and evaluation

---

## Project Structure

```
etl/
├── data/
│   ├── raw/          # Raw financial data files
│   └── staged/       # Processed/staged data
└── scripts/
    ├── extract.py    # Data extraction logic
    ├── transform.py  # Data transformation logic
    └── load.py       # Data loading logic
```

---

## Next Steps

1. Implement ETL pipeline components
2. Set up Supabase database connection
3. Develop financial analysis module
4. Train SVR model with historical data
5. Integrate explainable AI layer
6. Develop LLM-based recommendation system
7. Create Streamlit interface
8. Testing and validation
9. Documentation and presentation preparation

---

**Document Created:** January 22, 2026  
**Last Updated:** January 22, 2026
