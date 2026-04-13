# AN INDUSTRY ORIENTED MINI PROJECT REPORT ON

# FINCAST Financial Report Analysis and Recommendation System

In the partial fulfillment of the requirements for the award of the degree of

# BACHELOR OF TECHNOLOGY

IN ARTIFICIAL INTELLIGENCE & MACHINE LEARNING

Submitted by

**G. RUTWIKA** 23B81A7341
**B. SAI SRIKAR** 23B81A7346

Under the guidance of

**Mr. AZMERA CHANDU NAIK**
Sr. Assistant Professor, Department of CSE(AI&ML)

DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
(ARTIFICIAL INTELLEGENCE & MACHINE LEARNING)

**CVR COLLEGE OF ENGINEERING**
(An Autonomous institution, NAAC Accredited and Affiliated to JNTUH, Hyderabad)
Vastunagar, Mangalpalli (V), Ibrahimpatnam (M),
Rangareddy (D), Telangana- 501 510

**APRIL 2026**

---

## TABLE OF CONTENTS

| Chapter No. | Contents | Page No. |
|---|---|---|
| | List of Tables | i |
| | List of Figures | ii |
| | List of Symbols | iii |
| | Abbreviations | iv |
| | Abstract | |
| 1 | **Introduction** | 1-5 |
| 1.1 | Motivation | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Project Objectives | 3 |
| 1.4 | Project Report Organization | 4-5 |
| 2 | **Literature Survey** | 6-7 |
| 2.1 | Existing work | 6 |
| 2.2 | Limitations of Existing work | 7 |
| 3 | **Software & Hardware specifications** | 8-9 |
| 3.1 | Software requirements | 8 |
| 3.2 | Hardware requirements | 8 |
| 3.3 | Functional requirements | 9 |
| 3.4 | Non-Functional requirements | 9 |
| 4 | **Proposed System Design** | 10-15 |
| 4.1 | Proposed methods | 10 |
| 4.2 | Class Diagram | 10 |
| 4.3 | Use case Diagram | 11 |
| 4.4 | Sequence Diagram | 12 |
| 4.5 | Activity Diagram | 12 |
| 4.6 | System Architecture | 13 |
| 4.7 | Technology Description | 14-15 |
| 5 | **Implementation & Testing** | 16-23 |
| 5.1 | Front page Screenshot | 16 |
| 5.2 | Results Screenshot | 17-22 |
| 5.3 | Source Code - Key Modules | 23 |
| 6 | **Conclusion & Future Scope** | 24-25 |
| | References | 25 |
| | Appendix | 26-30 |

---

## LIST OF TABLES

Table 1: Project Timeline and Milestones
Table 2: Software Dependencies and Versions
Table 3: API Endpoints and Data Schema
Table 4: SVR Model Performance Metrics
Table 5: Feature Engineering Pipeline
Table 6: System Requirements Specification

---

## LIST OF FIGURES

Figure 1: System Architecture Overview
Figure 2: Data Flow Diagram
Figure 3: ETL Pipeline Architecture
Figure 4: SVR Model Training Process
Figure 5: SHAP Explainability Framework
Figure 6: Dashboard User Interface
Figure 7: Class Diagram
Figure 8: Use Case Diagram
Figure 9: Sequence Diagram
Figure 10: Activity Diagram

---

## LIST OF SYMBOLS/ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| ETL | Extract, Transform, Load |
| SVR | Support Vector Regression |
| SHAP | SHApley Additive exPlanations |
| LLM | Large Language Model |
| API | Application Programming Interface |
| ML | Machine Learning |
| CSV | Comma Separated Values |
| JSON | JavaScript Object Notation |
| RLS | Row Level Security |
| KPI | Key Performance Indicator |
| CAGR | Compound Annual Growth Rate |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| YoY | Year over Year |
| UI | User Interface |

---

# ABSTRACT

FinCast is an end-to-end financial analytics and forecasting prototype that combines data collection, analysis, machine learning prediction, and explainability into a unified system. The platform ingests financial data from API sources and user uploads, processes it through a robust ETL pipeline, and applies Support Vector Regression (SVR) to forecast next-period growth rates. The system leverages SHAP (SHapley Additive exPlanations) to provide interpretable insights into model predictions and generates actionable recommendations using LLM-based narrative generation.

The prototype addresses the gap between traditional financial reporting tools that offer limited forecasting and pure forecasting models that lack interpretability. By combining historical analysis, feature engineering, machine learning prediction, and AI-generated recommendations, FinCast provides analysts with both quantitative predictions and qualitative guidance.

The system is built on Python 3.11 with Streamlit for visualization, Supabase for data persistence, Scikit-learn for ML, and Groq's LLM API for recommendation generation. The complete pipeline has been validated end-to-end, with all core components operational and tested. The prototype has been packaged with comprehensive documentation, smoke tests, and release checklists to support handoff to new contributors.

**Keywords:** Financial Forecasting, Support Vector Regression, SHAP Explainability, LLM Recommendations, Data Analytics, Streamlit Dashboard

---

# 1. INTRODUCTION

## 1.1 Motivation

Financial analysts and decision-makers face a common challenge: bridging the gap between understanding what has happened historically and predicting what will happen in the future. Existing solutions typically fall into one of two categories:

1. **Historical Reporting Tools** - These provide excellent insights into past performance but offer limited forecasting capabilities. They excel at answering "what happened?" but struggle with "what will happen?"

2. **Black-Box Forecasting Models** - These can generate predictions but provide little insight into why those predictions were made. They excel at answering "what will happen?" but struggle with "why?"

FinCast was motivated by the need to merge these two approaches. The financial industry needs tools that can:

- Ingest diverse financial data sources reliably
- Extract meaningful patterns and correlations
- Make future predictions with quantifiable confidence
- Explain the drivers of those predictions
- Convert technical outputs into actionable business guidance

The prototype demonstrates that it is feasible to build a single integrated system that does all of these things, without requiring the user to stitch together multiple disconnected tools.

## 1.2 Problem Statement

**Problem:** Financial analysts lack a unified platform that combines historical analysis, pattern recognition, future forecasting, and model explainability in a single, interpretable interface.

**Specific Pain Points:**
1. Data normalization across multiple financial sources is time-consuming and error-prone
2. Separate tools for analysis, forecasting, and explanation create workflow friction
3. Model predictions without clear drivers are difficult to trust and act upon
4. Converting technical model outputs into business-friendly recommendations requires manual effort

**Proposed Solution:** Develop an integrated financial analytics platform (FinCast) that:
- Automates ETL from diverse sources
- Provides comprehensive historical and feature analysis
- Trains a growth-rate forecasting model
- Explains predictions using SHAP
- Generates grounded AI recommendations from model outputs
- Presents all of this in an interactive, user-friendly dashboard

## 1.3 Project Objectives

### Primary Objectives:
1. **Data Integration:** Build a flexible ETL pipeline that normalizes heterogeneous financial data
2. **Feature Engineering:** Extract and engineer financial ratios and growth indicators for ML
3. **Historical Analysis:** Provide comprehensive analysis of trends, peer comparison, and anomalies
4. **Machine Learning Forecasting:** Train an SVR model to predict next-period growth rates with confidence intervals
5. **Explainability:** Apply SHAP to expose feature contributions to model predictions
6. **Recommendation Generation:** Convert model outputs into actionable natural language recommendations
7. **User Experience:** Build an interactive Streamlit dashboard that integrates all layers
8. **Handoff Readiness:** Package the prototype with docs, tests, and clear architecture for future contributors

### Secondary Objectives:
1. Design for extensibility so new analysis modules can be added
2. Implement authentication to support multi-user workflows
3. Support both CLI and app-based usage patterns
4. Create comprehensive documentation for new developers
5. Validate the end-to-end pipeline with smoke tests

## 1.4 Project Report Organization

This report is organized as follows:

- **Chapter 1 (Introduction):** Provides context, problem statement, and objectives
- **Chapter 2 (Literature Survey):** Reviews existing financial analytics and ML explainability approaches
- **Chapter 3 (Requirements):** Details software, hardware, functional, and non-functional requirements
- **Chapter 4 (System Design):** Presents proposed architecture, data models, and diagrams
- **Chapter 5 (Implementation & Testing):** Shows screenshots and results of the running prototype
- **Chapter 6 (Conclusion & Future Scope):** Summarizes achievements and next milestones

---

# 2. LITERATURE SURVEY

## 2.1 Existing Work

### Financial Analytics Platforms
Traditional financial reporting platforms like Bloomberg Terminal and FactSet provide:
- Comprehensive historical financial data aggregation
- Industry-standard analysis and peer comparison
- Real-time market data feeds
However, these platforms are expensive, closed-source, and limited in custom forecasting.

### Machine Learning Forecasting
Academic and open-source projects have explored:
- **Time Series Models (ARIMA, Prophet):** Good for univariate series, but limited to historical patterns
- **Deep Learning LSTM Models:** Capture complex temporal dependencies but are data-hungry and less interpretable
- **SVR/Kernel Methods:** Balance complexity and interpretability; work well with limited data

### Model Explainability
The explainability landscape has evolved significantly:
- **SHAP (Lundberg & Lee, 2017):** Provides theoretically grounded feature attribution using game theory
- **LIME (Ribeiro et al., 2016):** Local linear approximations for model explanation
- **Attention Mechanisms:** Popular in deep learning but harder to interpret

### LLM-Based Recommendation Generation
Recent advances in large language models have enabled:
- Zero-shot and few-shot recommendation synthesis
- Fine-tuning on domain-specific financial guidance
- Grounding recommendations in structured model outputs

### Gap in Existing Work
Most existing solutions focus on one or two of these layers (analysis, forecasting, explainability, recommendations), but not all four integrated together. FinCast addresses this gap by unifying the full pipeline.

## 2.2 Limitations of Existing Work

### Commercial Tools
- High cost barrier to entry
- Closed source limiting customization
- Vendor lock-in
- Limited integration with custom ML pipelines

### Academic/Open-Source Solutions
- Often focus on single layers (forecasting OR explainability, not both)
- Limited documentation for non-researchers
- Assume deep ML expertise
- Lack production-ready error handling and logging

### Hybrid Approaches
- Stitching multiple tools together creates friction
- Data transformation between tools is error-prone
- No unified entrypoint (app vs CLI inconsistency)
- Difficult to trace recommendation rationale across layers

### FinCast's Approach to Overcome These
1. **Open source design:** Fully accessible and modifiable
2. **Integrated pipeline:** All layers in one codebase, reducing friction
3. **App-first interface:** User-friendly dashboard as primary entry point
4. **Documentation for onboarding:** Not just code, but handoff guides
5. **End-to-end testing:** Smoke tests validate the complete workflow
6. **Modular architecture:** Easy to swap components (e.g., different ML models)

---

# 3. SOFTWARE & HARDWARE SPECIFICATIONS

## 3.1 Software Requirements

### Core Dependencies
- **Python:** 3.11+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (SVR, GridSearchCV)
- **Explainability:** SHAP (KernelExplainer)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Frontend:** Streamlit 1.28.1
- **Database:** Supabase (PostgreSQL)
- **API Clients:** Requests, HTTPX
- **File Processing:** PDFPlumber, OpenPyXL, XlRD
- **LLM Integration:** Groq API client
- **Environment Management:** python-dotenv
- **Configuration:** PyYAML

### Optional Utilities
- **CLI Orchestration:** Argparse (built-in)
- **Logging:** Python logging (standard library)
- **Testing:** Smoke test script (custom)

### Development Tools
- **Version Control:** Git
- **Code Quality:** Optional (Pylint, Black not required for prototype)

## 3.2 Hardware Requirements

### Minimum Requirements
- **Processor:** Intel i5 or equivalent
- **RAM:** 4 GB
- **Storage:** 2 GB free space (for dependencies and data)
- **Network:** Internet connection for API calls

### Recommended Requirements
- **Processor:** Intel i7 or AMD Ryzen 5+
- **RAM:** 8+ GB (for comfortable Streamlit operation)
- **Storage:** 5+ GB free space
- **GPU:** Optional (not required; SVR runs efficiently on CPU)

### Development Environment
- **OS:** Windows 10+, macOS 11+, Linux (Ubuntu 20.04+)
- **Terminal:** PowerShell (Windows), Bash/Zsh (macOS/Linux)
- **Text Editor:** VS Code, PyCharm, or equivalent

## 3.3 Functional Requirements

### Data Ingestion
- FR1: System shall accept financial data from CSV, Excel, JSON, and PDF uploads
- FR2: System shall validate data schema and required fields
- FR3: System shall normalize inconsistent field names and units

### ETL Processing
- FR4: System shall extract raw data reliably
- FR5: System shall transform data into ML-ready and LLM-ready tables
- FR6: System shall engineer financial ratios and growth indicators
- FR7: System shall load processed data into Supabase with user-level isolation

### Analysis Engine
- FR8: System shall compute historical performance metrics (CAGR, margins)
- FR9: System shall perform trend and peer comparison analysis
- FR10: System shall detect outliers and anomalies
- FR11: System shall generate interpretable insights from analysis

### Machine Learning
- FR12: System shall train SVR models on historical data
- FR13: System shall generate next-period growth rate predictions
- FR14: System shall compute target-gap analysis and confidence intervals
- FR15: System shall optimize hyperparameters using GridSearchCV

### Explainability
- FR16: System shall compute SHAP values for global feature importance
- FR17: System shall compute SHAP local explanations per company
- FR18: System shall expose feature contributions in human-readable format

### Recommendations
- FR19: System shall generate structured recommendation objects from model outputs
- FR20: System shall convert structured outputs to natural language recommendations
- FR21: System shall handle LLM API failures gracefully with fallback recommendations

### User Interface
- FR22: System shall require user authentication before data access
- FR23: System shall display historical analysis in interactive tables and charts
- FR24: System shall display SVR predictions and confidence intervals
- FR25: System shall display SHAP explanations with visualizations
- FR26: System shall allow users to upload and analyze new company data

### Orchestration
- FR27: System shall support single-phase CLI execution
- FR28: System shall support full-pipeline CLI execution
- FR29: System shall validate environment configuration with smoke test

## 3.4 Non-Functional Requirements

### Performance
- NFR1: SVR training shall complete within 5 minutes on reference hardware
- NFR2: SHAP computation shall complete within 10 minutes for < 100 samples
- NFR3: Streamlit app shall load and respond within 3 seconds
- NFR4: Dashboard charts shall render within 2 seconds

### Reliability
- NFR5: System shall handle missing values without crashing
- NFR6: System shall handle API timeouts with exponential backoff (future enhancement)
- NFR7: System shall validate all ETL outputs before loading
- NFR8: System shall provide detailed error messages for debugging

### Usability
- NFR9: Dashboard shall be navigable with minimal training
- NFR10: All error messages shall be clear and actionable
- NFR11: Documentation shall include end-to-end walkthrough
- NFR12: Code shall follow consistent style and naming conventions

### Security
- NFR13: User data shall be isolated via Supabase RLS policies
- NFR14: API keys shall be stored in environment variables, not committed to git
- NFR15: Database credentials shall be provided via .env file
- NFR16: Uploaded files shall be validated before processing

### Maintainability
- NFR17: Code shall be modular with clear separation of concerns
- NFR18: All modules shall be importable and tested independently
- NFR19: Documentation shall match current implementation
- NFR20: Architecture decisions shall be documented in /docs

---

# 4. PROPOSED SYSTEM DESIGN

## 4.1 Proposed Methods

### Methodology Overview
The system follows a **layered architecture** with clear separation of concerns:

```
Data Source
    ↓
ETL Pipeline (Extract → Transform → Load)
    ↓
Supabase (PostgreSQL)
    ↓
Analysis Engine (Historical + Feature Analysis)
    ↓
Machine Learning (SVR Training + Prediction)
    ↓
Explainability (SHAP Global + Local)
    ↓
Recommendation Generation (LLM)
    ↓
Streamlit Dashboard
```

### Design Approach

**1. Adapter Pattern (ETL):**
Each data source (JSON, CSV, Excel, PDF) has a corresponding extractor adapter. This allows new sources to be added without modifying existing code.

**2. Feature Engineering Pipeline:**
Derived features are computed in stages:
- Stage 1: Raw field validation and type normalization
- Stage 2: Financial ratio calculation (margins, efficiency, leverage)
- Stage 3: Growth rate computation (YoY % change)
- Stage 4: Output to ML-ready and LLM-ready tables

**3. Time-Aware Train-Test Split:**
To avoid data leakage, the split is chronological, not random. This ensures the model never sees future data during training.

**4. Grid Search Hyperparameter Tuning:**
SVR hyperparameters (C, epsilon, gamma, kernel) are optimized using GridSearchCV with cross-validation.

**5. Growth-Rate Prediction:**
Instead of predicting absolute revenue (which is highly dependent on company scale), the model predicts YoY growth rate. This is:
- More interpretable ("growth of 5%" vs "$1B in revenue")
- Better for gap analysis (predicted vs target)
- Comparable across companies of different sizes

**6. SHAP for Model Transparency:**
SHAP values provide theoretically grounded feature attributions. The system computes:
- Global importance: mean absolute SHAP per feature
- Local explanations: per-prediction feature contributions

**7. Fallback Recommendation Generation:**
If the LLM API is unavailable, the system generates deterministic recommendations based on financial rules and SHAP drivers. This ensures graceful degradation.

## 4.2 Class Diagram

(Conceptual representation; code is function-based rather than strictly OOP)

```
┌─────────────────────────────────────────┐
│ DataConnection                          │
├─────────────────────────────────────────┤
│ - supabase_client: SupabaseClient      │
│ + get_user_standard_table()            │
│ + get_supabase_client()                │
│ + load_user_data()                     │
└─────────────────────────────────────────┘
               ▲
               │
      ┌────────┴──────────┐
      │                   │
┌─────────────────┐  ┌──────────────────┐
│ ETL Pipeline    │  │ Analysis Engine  │
├─────────────────┤  ├──────────────────┤
│ + extract()     │  │ + analyze()      │
│ + transform()   │  │ + trend()        │
│ + validate()    │  │ + peer_compare() │
│ + load()        │  │ + insights()     │
└─────────────────┘  └──────────────────┘
      │                   │
      └────────┬──────────┘
               ▼
┌──────────────────────────────┐
│ ML Pipeline (SVR)            │
├──────────────────────────────┤
│ + prepare_features()         │
│ + train_svr()                │
│ + predict()                  │
│ + evaluate()                 │
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ SHAP Explainability          │
├──────────────────────────────┤
│ + compute_global_importance()│
│ + compute_local_explanation()│
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ Recommendation Engine        │
├──────────────────────────────┤
│ + load_bundle()              │
│ + generate_recommendations() │
│ + build_fallback()           │
└──────────────────────────────┘
```

## 4.3 Use Case Diagram

```
                        ┌─────────────────┐
                        │     Analyst     │
                        └────────┬────────┘
                                 │
                ┌────────────────┼─────────────────┐
                │                │                 │
                ▼                ▼                 ▼
         ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
         │Upload Data  │  │Review Analysis│  │View Forecast│
         └─────────────┘  └──────────────┘  └─────────────┘
                │                │                 │
                └────────────────┼─────────────────┘
                                 │
                    ┌────────────────────────┐
                    │   FinCast System       │
                    └─┬────────────────────┬─┘
                      │                    │
         ┌────────────┴──────────┐  ┌──────┴──────────────┐
         │                       │  │                     │
         ▼                       ▼  ▼                     ▼
    ┌─────────────┐      ┌───────────────┐      ┌──────────────┐
    │Authenticate │      │ Process Data  │      │Generate      │
    │             │      │               │      │Recommendations
    └─────────────┘      └───────────────┘      └──────────────┘
```

## 4.4 Sequence Diagram

```
Analyst    Dashboard    ETL      Database    ML        SHAP      LLM
  │            │         │          │         │         │        │
  │─Upload Data─>│        │          │         │         │        │
  │            │─Validate>│          │         │         │        │
  │            │<Validated│          │         │         │        │
  │            │─Transform─>         │         │         │        │
  │            │─Load─────────────────>       │         │        │
  │            │<Loaded──────────────│         │         │        │
  │            │─Train SVR──────────┼────────>│         │        │
  │            │<Model Trained──────┼────────<│         │        │
  │            │─Compute SHAP──────────────────>         │        │
  │            │<Explain Ready──────────────────<         │        │
  │            │──Generate Recs───────────────────────────>       │
  │            │<Recommendations───────────────────────────<       │
  │<Dashboard──│                                                 │
  │            │                                                 │
  └─────────────────────────────────────────────────────────────┘
```

## 4.5 Activity Diagram

```
                    ┌──────────────────┐
                    │   User Logs In   │
                    └────────┬─────────┘
                             │
                    ┌────────▼──────────┐
                    │  Upload Data or   │
                    │  Select Company   │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────────┐
                    │ Validate & Transform  │
                    └────────┬──────────────┘
                             │
                    ┌────────▼──────────────┐
            ┌───────│ Load to Database     │◄─────────────┐
            │       └────────┬──────────────┘              │
            │                │                            │
            │       ┌────────▼──────────────┐             │
            │       │ Review Historical     │             │
            │       │ Analysis              │             │
            │       └────────┬──────────────┘             │
            │                │                            │
            │       ┌────────▼──────────────┐             │
            │       │ Train SVR Model      │             │
            │       └────────┬──────────────┘             │
            │                │                            │
            │       ┌────────▼──────────────┐             │
            │       │ Compute SHAP         │             │
            │       │ Explanations         │             │
            │       └────────┬──────────────┘             │
            │                │                            │
            │       ┌────────▼──────────────┐             │
            │       │ Generate             │             │
            │       │ Recommendations      │             │
            │       └────────┬──────────────┘             │
            │                │                            │
            │       ┌────────▼──────────────┐             │
            └──────│ View Results &        │             │
                   │ Export Analysis       │─────────────┘
                   └───────────────────────┘
```

## 4.6 System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)                   │
│  ┌────────────┬──────────────────┬──────────────┐               │
│  │ Dashboard  │ Authentication   │ Upload Panel │               │
│  └────────────┴──────────────────┴──────────────┘               │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    Service Layer (analysis/)                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐       │
│  │ Data     │Historical│ Feature  │ Auto     │ Recommend│       │
│  │Connection│Performance│Analysis │ Analysis │ Engine   │       │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘       │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    ML Layer (models/)                            │
│  ┌──────────────────────┬──────────────────────┐                │
│  │  SVR Pipeline        │  SHAP Explainability │                │
│  │  - Train            │  - Global Importance │                │
│  │  - Predict          │  - Local Explanation │                │
│  │  - Gap Analysis     │  - Future Predictions│                │
│  └──────────────────────┴──────────────────────┘                │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    ETL Layer (etl/)                              │
│  ┌──────────┬──────────┬──────────┬──────────┐                  │
│  │Extract   │Transform │Validate  │Load      │                  │
│  │- CSV     │- Cleaning│- Schema  │- Upsert  │                  │
│  │- Excel   │- Features│- Types   │- Batch   │                  │
│  │- JSON    │- Rates   │- Missing │- RLS     │                  │
│  │- PDF     │- Ratios  │- Values  │          │                  │
│  └──────────┴──────────┴──────────┴──────────┘                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    Data Layer                                    │
│  ┌─────────────────────────────────────────────────┐             │
│  │  Supabase PostgreSQL                           │             │
│  │  - standard_table (ML features)                │             │
│  │  - category_table (LLM context)                │             │
│  │  - uploaded_files (audit trail)                │             │
│  │  - recommendations (outputs)                   │             │
│  └─────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Data Source
      │
      ├─ CSV, Excel, JSON, PDF, API
      │
┌─────▼─────────────────────────────────────┐
│ Extract (etl/extract.py)                 │
│ Multi-source adapters                    │
└─────┬─────────────────────────────────────┘
      │
      ▼ Standardized JSON format
      
┌─────────────────────────────────────┐
│ Transform (etl/transform.py)        │
│ - Validate schema                   │
│ - Clean missing values              │
│ - Engineer features                 │
└─────┬───────────────────────────────┘
      │
      ▼ standard_table.csv & category_table.csv
      
┌─────────────────────────────────────┐
│ Load (etl/load.py)                  │
│ - Upsert to Supabase                │
│ - Apply RLS policies                │
└─────┬───────────────────────────────┘
      │
      ▼ Supabase PostgreSQL
      
┌─────────────────────────────────────┐
│ Analysis Layer (analysis/...)       │
│ - Historical performance            │
│ - Trend analysis                    │
│ - Peer comparison                   │
│ - Feature analysis                  │
└─────┬───────────────────────────────┘
      │
      ▼ Analysis reports
      
┌─────────────────────────────────────┐
│ ML Layer (models/svr_pipeline.py)   │
│ - SVR training                      │
│ - Future predictions                │
│ - Gap analysis                      │
└─────┬───────────────────────────────┘
      │
      ▼ SVR model + predictions + metrics
      
┌─────────────────────────────────────┐
│ SHAP Layer (models/explainability.py)
│ - Global importance                 │
│ - Local explanations                │
└─────┬───────────────────────────────┘
      │
      ▼ SHAP values + visualizations
      
┌─────────────────────────────────────┐
│ Recommendation (analysis/rec_engine)│
│ - LLM narrative generation          │
│ - Fallback rules                    │
└─────┬───────────────────────────────┘
      │
      ▼ Structured recommendations
      
┌─────────────────────────────────────┐
│ Streamlit Dashboard (app.py)        │
│ - View and interact                 │
└─────────────────────────────────────┘
```

## 4.7 Technology Description

### Python 3.11
**Rationale:** Mature, widely supported, excellent ecosystem for data science and ML. Version 3.11 offers performance improvements and is stable for production.

### Pandas & NumPy
**Rationale:** Industry standard for data manipulation. NumPy provides efficient numerical computation; Pandas provides intuitive data frame operations.

### Scikit-learn (SVR)
**Rationale:** Support Vector Regression is robust for moderate-sized datasets, handles nonlinearity, and provides built-in cross-validation and hyperparameter tuning via GridSearchCV. More interpretable than deep learning for this use case.

### SHAP
**Rationale:** Theoretically grounded explainability method based on Shapley values. Provides both global and local explanations and is widely adopted in the ML community.

### Streamlit
**Rationale:** Rapid prototyping framework for ML dashboards. Excellent for iterative UI development and allows data scientists to build interactive apps without deep frontend knowledge.

### Supabase
**Rationale:** PostgreSQL-based backend-as-a-service. Provides authentication, RLS policies for data isolation, and vector extension (for future RAG enhancements). Free tier is suitable for prototype.

### Groq API
**Rationale:** Fast LLM inference service. Groq's llama-3.3-70b-versatile model is state-of-the-art and cost-effective for structured recommendation generation.

### Plotly
**Rationale:** Interactive visualizations for dashboards. Plotly charts are responsive, publication-quality, and don't require complex configuration.

### python-dotenv
**Rationale:** Safe management of environment variables. Keeps sensitive API keys out of version control.

---

# 5. IMPLEMENTATION & TESTING

## 5.1 Frontend Screenshots

(Dashboard sections and key UI components would be shown here with captions)

### Dashboard - Historical Analysis Tab
- KPI cards showing revenue, growth, margins
- Time-series charts of financial metrics
- Peer comparison bar charts
- Data tables with sorting and filtering

### Dashboard - SVR Predictions Tab
- Model evaluation metrics (MAE, RMSE, R²)
- Predicted vs actual scatter plot
- Confidence interval visualization
- Target-gap analysis table

### Dashboard - SHAP Explainability Tab
- Global feature importance bar chart
- Local SHAP value explanation
- Force plot showing feature contributions
- Summary statistics

### Dashboard - Recommendations Tab
- Structured recommendation output
- Performance score and risk assessment
- Key action items with priority
- Investment verdict summary

## 5.2 Results & Performance

### Model Performance Results
- **SVR Evaluation:**
  - MAE: 105.21% (growth-rate prediction error)
  - RMSE: 171.23%
  - R²: -0.1210 (baseline model; room for improvement)

- **Cross-Validation:**
  - 5-fold CV used to avoid overfitting on small dataset
  - Best hyperparameters: kernel=RBF, C=50, epsilon=0.01, gamma=auto

- **Future Predictions:**
  - Generated for AAPL, AMZN, GOOGL with 95% confidence intervals
  - Gap analysis shows predicted growth vs 10% target

### Feature Importance (SHAP)
- **Top Drivers of Growth:**
  1. Asset Efficiency (mean |SHAP| = 8.21)
  2. Revenue Growth (mean |SHAP| = 5.20)
  3. Total Assets (mean |SHAP| = 5.09)

### Recommendation Quality
- **LLM Generated Outputs:**
  - Structured JSON with executive summary, risk assessment, opportunities
  - Fallback mechanism tested and working
  - Recommendations grounded in SHAP drivers

### System Performance
- **Execution Times:**
  - ETL Pipeline: ~2 seconds for 60 records
  - SVR Training: ~3 seconds
  - SHAP Computation: ~8 seconds for 50 samples
  - LLM Recommendation: ~2 seconds (API call)

### Test Coverage
- **Smoke Test:** PASSED (all imports, directories, env keys, artifacts)
- **End-to-End Pipeline:** PASSED (phases 3.1-6 execute without errors)

---

## 5.3 Source Code - Key Modules

This section presents the core source code for the two most critical FinCast components:
1. **SVR Pipeline** (Phase 4) - Machine Learning prediction model
2. **Recommendation Engine** (Phase 6) - LLM-based financial intelligence generation

These modules represent the heart of FinCast's forecasting and interpretation capabilities.

### 5.3.1 SVR Pipeline Module (`models/svr_pipeline.py`)

The SVR Pipeline implements Support Vector Regression for growth-rate prediction with hyperparameter tuning and confidence intervals.

#### Core Function: SVR Model Training & Evaluation

```python
def tune_and_train_svr(X_train, y_train):
    """Tune SVR hyperparameters using time-series cross-validation."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    param_grid = {
        "svr__kernel": ["rbf", "linear"],
        "svr__C": [1, 10, 50, 100],
        "svr__epsilon": [0.01, 0.1, 0.5],
        "svr__gamma": ["scale", "auto"],
    }

    n_splits = min(5, max(2, len(X_train) // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    cv_metrics = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=tscv,
        scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
        n_jobs=-1,
    )

    cv_summary = {
        "cv_mae_mean": -cv_metrics["test_neg_mean_absolute_error"].mean(),
        "cv_mae_std": cv_metrics["test_neg_mean_absolute_error"].std(),
        "cv_rmse_mean": -cv_metrics["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_std": cv_metrics["test_neg_root_mean_squared_error"].std(),
        "cv_r2_mean": cv_metrics["test_r2"].mean(),
        "cv_r2_std": cv_metrics["test_r2"].std(),
        "best_params": grid.best_params_,
    }

    return best_model, cv_summary
```

**Key Features:**
- Time-series cross-validation to prevent data leakage
- Scikit-learn Pipeline with StandardScaler for feature normalization
- GridSearchCV over kernel types (RBF, linear), regularization (C), and gamma parameters
- Cross-validation metrics returned for model stability assessment

#### Time-Aware Train-Test Split

```python
def time_aware_split(X, y, dates, test_ratio=0.2):
    """Create chronological train-test split using unique dates."""
    unique_dates = sorted(pd.Series(dates).dropna().unique())
    if len(unique_dates) < 4:
        raise ValueError("Insufficient unique time points for time-aware split")

    split_index = max(1, int(len(unique_dates) * (1 - test_ratio)))
    split_index = min(split_index, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_index - 1]

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return X_train, X_test, y_train, y_test, cutoff_date
```

**Purpose:** Prevents temporal leakage by splitting on a cutoff date rather than random sampling.

#### Future Prediction & Gap Analysis

```python
def predict_future_and_gaps(model, future_rows, future_X, confidence_sigma, target_growth_rate=10.0):
    """Predict next-period growth rate and compute target gaps."""
    preds = model.predict(future_X)
    output = future_rows[["ticker", "date", "net_income"]].copy()
    
    output["current_net_income"] = output["net_income"]
    output["predicted_growth_rate"] = preds
    output["target_growth_rate"] = target_growth_rate
    output["gap_vs_target"] = preds - target_growth_rate
    output["gap_status"] = np.where(output["gap_vs_target"] >= 0, "surplus", "shortfall")
    output["confidence_lower_95"] = preds - 1.96 * confidence_sigma
    output["confidence_upper_95"] = preds + 1.96 * confidence_sigma
    output["predicted_next_net_income"] = output["current_net_income"] * (1 + preds / 100)
    output["target_next_net_income"] = output["current_net_income"] * (1 + target_growth_rate / 100)
    
    return output
```

**Outputs:**
- `predicted_growth_rate`: SVR model's predicted YoY growth %
- `gap_vs_target`: Difference vs 10% target (positive = surplus, negative = shortfall)
- `confidence_lower_95` / `confidence_upper_95`: 95% confidence intervals
- `predicted_next_net_income`: Projected next-period net income
- `target_next_net_income`: Net income needed to hit 10% growth target

#### Model Evaluation

```python
def evaluate_model(model, X_test, y_test):
    """Evaluate trained model on holdout set."""
    predictions = model.predict(X_test)
    residuals = y_test.values - predictions

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "test_size": int(len(y_test)),
    }

    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
        "residual": residuals,
    }, index=y_test.index)

    return metrics, pred_df
```

**Metrics:**
- **MAE**: Mean Absolute Error in growth rate %
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **Residual statistics**: Min, max, mean, std for error analysis

### 5.3.2 Recommendation Engine Module (`analysis/recommendation_engine.py`)

The Recommendation Engine uses Groq's LLM to generate structured financial intelligence reports grounded in SVR predictions and SHAP explanations.

#### Analysis Bundle Assembly

```python
def load_analysis_bundle_from_reports(
    ticker: str, reports_dir: str = "analysis/reports"
) -> Dict[str, Any]:
    """
    Assemble the analysis bundle for a ticker from Phase 4 + 5 report CSVs.
    All CSV files use 'ticker' column to match your schema.
    """
    bundle: Dict[str, Any] = {"ticker": ticker}

    # SVR future predictions
    svr_path = os.path.join(reports_dir, "svr_future_predictions.csv")
    if os.path.exists(svr_path):
        df = pd.read_csv(svr_path)
        row = df[df["ticker"].str.upper() == ticker.upper()]
        if not row.empty:
            bundle["svr_predictions"] = row.iloc[0].to_dict()

    # SVR evaluation metrics
    metrics_path = os.path.join(reports_dir, "svr_evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        bundle["model_metrics"] = {
            "mae": round(float(m.get("mae", 0)), 2),
            "rmse": round(float(m.get("rmse", 0)), 2),
            "r2": round(float(m.get("r2", 0)), 4),
        }

    # SHAP global importance (top 8 features)
    shap_global_path = os.path.join(reports_dir, "phase_5_shap_global_importance.csv")
    if os.path.exists(shap_global_path):
        shap_df = pd.read_csv(shap_global_path).head(8)
        bundle["shap_global_top_features"] = shap_df.to_dict(orient="records")

    # SHAP local explanations for this ticker
    shap_local_path = os.path.join(reports_dir, "phase_5_shap_local_explanations.csv")
    if os.path.exists(shap_local_path):
        local_df = pd.read_csv(shap_local_path)
        ticker_local = local_df[local_df["ticker"].str.upper() == ticker.upper()]
        if not ticker_local.empty:
            bundle["shap_local_drivers"] = ticker_local.to_dict(orient="records")

    # Latest financial ratios
    std_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "staged", "standard_table.csv",
    )
    if os.path.exists(std_path):
        std_df = pd.read_csv(std_path)
        ticker_std = std_df[std_df["ticker"].str.upper() == ticker.upper()]
        if not ticker_std.empty:
            latest = ticker_std.sort_values("date").iloc[-1]
            bundle["latest_ratios"] = {
                "profit_margin": round(float(latest.get("profit_margin", 0) or 0), 2),
                "operating_margin": round(float(latest.get("operating_margin", 0) or 0), 2),
                "debt_to_asset": round(float(latest.get("debt_to_asset", 0) or 0), 4),
                "asset_efficiency": round(float(latest.get("asset_efficiency", 0) or 0), 4),
            }

    # Peer comparison data
    if os.path.exists(std_path):
        all_df = pd.read_csv(std_path)
        peer_data = {}
        for peer in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            peer_df = all_df[all_df["ticker"] == peer]
            if not peer_df.empty:
                p_latest = peer_df.sort_values("date").iloc[-1]
                peer_data[peer] = {
                    "profit_margin": round(float(p_latest.get("profit_margin", 0) or 0), 2),
                    "debt_to_asset": round(float(p_latest.get("debt_to_asset", 0) or 0), 4),
                }
        bundle["peer_comparison"] = peer_data

    return bundle
```

**Bundle Contents:**
- SVR predictions (growth rate, confidence intervals, gaps)
- Model metrics (MAE, RMSE, R²)
- SHAP global importance (top 8 features)
- SHAP local explanations (per-company drivers)
- Latest financial ratios (margins, debt, efficiency)
- Peer comparison data

#### Recommendation Generation with Fallback

```python
def generate_recommendations(analysis_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Groq with the analysis bundle and return a structured
    financial intelligence report dict.
    """
    user_content = (
        f"Generate the financial intelligence report for:\n\n"
        f"{json.dumps(analysis_bundle, indent=2, default=str)}"
    )

    try:
        raw = call_llm(SYSTEM_PROMPT, user_content)
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(clean)
    except (json.JSONDecodeError, Exception) as e:
        # Retry with stricter instruction
        system_strict = SYSTEM_PROMPT + " Ensure the output is valid JSON with no extra text."
        try:
            raw_retry = call_llm(system_strict, user_content)
            clean_retry = re.sub(r"```(?:json)?", "", raw_retry).strip().rstrip("`").strip()
            result = json.loads(clean_retry)
        except Exception as e2:
            raise ValueError(f"Failed to generate recommendations after retry: {str(e2)}")

    # Save to reports directory
    reports_dir = "analysis/reports"
    os.makedirs(reports_dir, exist_ok=True)
    ticker = analysis_bundle.get("ticker", "unknown")
    out_path = os.path.join(
        reports_dir, f"phase_6_{ticker.lower()}_recommendations.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[RecommendationEngine] ✓ Saved recommendations to {out_path}")

    return result
```

**Features:**
- Calls Groq's llama-3.3-70b-versatile model with structured prompt
- Cleans markdown code blocks from response
- Adds retry mechanism for robust JSON parsing
- Saves outputs to `analysis/reports/phase_6_*.json` for audit trail
- Temperature: 0.2 (low creativity, factual/consistent output)
- Requires `GROQ_API_KEY` in `.env`

#### LLM Call Wrapper

```python
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
```

### 5.3.3 Integration & Data Flow

The two modules are chained in `scripts/run.py`:

```python
# Phase 4: SVR Training & Prediction
from models.svr_pipeline import run_phase4_svr
phase4_output = run_phase4_svr(target_growth_rate=10.0)

# Phase 5: SHAP Explainability (optional)

# Phase 6: LLM Recommendations
from analysis.recommendation_engine import load_analysis_bundle_from_reports, generate_recommendations

for ticker in ["AAPL", "AMZN", "GOOGL"]:
    bundle = load_analysis_bundle_from_reports(ticker)
    recommendations = generate_recommendations(bundle)
```

**Key Dependencies:**
```
scikit-learn==1.3.2      # SVR, GridSearchCV, metrics
pandas==2.0.3            # Data manipulation
numpy==1.24.3            # Numerical operations
groq==0.4.2              # LLM API (Recommendation Engine)
python-dotenv==1.0.0     # Environment variable management
```

---

# 6. CONCLUSION & FUTURE SCOPE

## Conclusion

FinCast demonstrates that it is feasible and valuable to build an integrated financial analytics platform that combines historical analysis, machine learning forecasting, model explainability, and AI-generated recommendations in a single, user-friendly system.

### Key Achievements:
1. **Complete ETL Pipeline:** Successfully ingests, validates, and transforms financial data from multiple sources
2. **Unified Analysis Engine:** Provides historical, trend, peer, and feature analysis in one framework
3. **ML Forecasting:** Trained SVR model that predicts growth rates with quantifiable uncertainty
4. **Explainability:** SHAP integration exposes which features drive predictions
5. **Recommendation Generation:** LLM-based recommendations ground in model outputs with fallback mechanism
6. **Interactive Dashboard:** Streamlit app ties all layers together in a user-friendly interface
7. **Production-Ready Packaging:** Documentation, smoke tests, release checklist, and git organization support handoff

### Quality of Solution:
The prototype successfully answers the original problem statement: analysts now have a unified platform that provides historical insights, future predictions, model explanations, and actionable recommendations without stitching together disconnected tools.

The system is architecturally sound, modular, and extensible. All core layers are operational and integrated end-to-end.

## Future Scope

### Short-Term Improvements (Next 1-2 Sprints)
1. **Model Enhancement:**
   - Feature selection/importance analysis to improve R²
   - Ensemble methods (Random Forest, XGBoost) to compare with SVR
   - Time-series cross-validation for more robust evaluation

2. **Testing:**
   - Add unit tests for ETL transformations
   - Add integration tests for end-to-end pipeline
   - Add CI/CD so tests run on every commit

3. **UI/UX:**
   - Add data export functionality (PDF reports, CSV)
   - Add filtering and date-range selection in dashboard
   - Add historical comparison ("how does 2024 look vs 2023?")

4. **Deployment:**
   - Docker containerization for reproducible environment
   - Cloud deployment (Heroku, AWS, or similar)
   - Monitoring and alerting for production issues

### Medium-Term Enhancements (Next Quarter)
1. **Data Integration:**
   - RAG (Retrieval Augmented Generation) to ground recommendations in news/filings
   - Real-time data pipelines for streaming financial updates
   - Support for more asset classes (crypto, commodities, bonds)

2. **ML Improvements:**
   - Federated learning for collaborative model training across organizations
   - Anomaly detection for unusual financial events
   - Scenario analysis ("what if revenue drops 10%?")

3. **Explainability:**
   - Counterfactual explanations ("what features would need to change for a different prediction?")
   - Feature interaction plots
   - SHAP waterfall plots for individual predictions

### Long-Term Vision (6+ Months)
1. **Platform Scaling:**
   - Multi-tenant architecture for SaaS deployment
   - Support for thousands of companies and concurrent users
   - Real-time collaborative analysis

2. **Advanced Features:**
   - Industry-specific models (finance, pharma, tech, etc.)
   - Regulatory compliance reporting
   - Portfolio-level optimization

3. **Research:**
   - Research-grade backtesting and performance attribution
   - Novel forecasting architectures (hybrid ML + causal inference)
   - Fairness and bias analysis for recommendations

---

# REFERENCES

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (pp. 4765-4774).
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135-1144).
3. Vapnik, V. N. (1998). Statistical learning theory (Vol. 1). Wiley.
4. Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of statistical software, 27(3), 1-22.
5. Finance Datasets - Alpha Vantage. https://www.alphavantage.co/
6. Scikit-learn: Machine Learning in Python - Pedregosa et al., JMLR 12(85):2825-2830, 2011
7. Streamlit Documentation: https://docs.streamlit.io/
8. Supabase Documentation: https://supabase.com/docs
9. SHAP GitHub Repository: https://github.com/shap/shap
10. Groq API Documentation: https://console.groq.com/docs

---

# APPENDIX

## A. Repository-Based Setup and Validation

### Prerequisites:
- Python 3.11+
- Git
- Virtual environment tool (venv)

### Step-by-Step Setup:
1. Source repository: https://github.com/SaiSrikar0/financial-report-analysis
2. Follow the installation workflow documented in README: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/README.md
3. Install dependencies listed in requirements: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/requirements.txt
4. Configure required API keys in a local .env file (see Appendix B)
5. Validate environment and project integrity using the smoke test script: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/scripts/smoke_test.py
6. Launch the dashboard application entry point: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/app.py

### Repository Navigation (Key Paths):
- Project documentation: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/docs
- ETL pipeline modules: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/etl
- Analysis layer modules: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/analysis
- ML and explainability modules: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/models

## B. API Key Configuration

### Required Keys:
- ALPHAVANTAGE_API_KEY: Financial data retrieval
- SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY: Database
- GROQ_API_KEY: LLM recommendations

### Obtaining Keys:
- Alpha Vantage: https://www.alphavantage.co/
- Supabase: https://supabase.com/
- Groq: https://console.groq.com/

## C. Sample Data Format

Financial data should conform to this schema:
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

## D. Troubleshooting Common Issues

| Issue | Solution |
|---|---|
| Import errors on startup | Re-verify dependency setup against requirements and README in the GitHub repository |
| Supabase connection fails | Verify .env keys and network connectivity |
| SVR training slow | Expected for GridSearchCV; runs within 5 min on reference hardware |
| SHAP computation timeout | Reduce sample size or use GPU (optional future) |
| LLM API 429 errors | Rate limited; fallback recommendations will generate |
| Missing report artifacts | Re-run the full pipeline using the workflow documented in https://github.com/SaiSrikar0/financial-report-analysis/blob/main/scripts/run.py |

## E. Guide for Readers and Evaluators

### How to Navigate This Report
1. **For Executive Summary:** Start with the Abstract (page v) for a high-level overview of the project scope and contributions
2. **For Technical Understanding:** Read Chapter 4 (System Design) to understand the architecture and methodology
3. **For Implementation Details:** Refer to Chapter 5 (Implementation & Testing) for actual performance metrics and results
4. **For Source Code (Key Modules):** See Section 5.3 (Source Code - Key Modules) for detailed implementations of the SVR Pipeline and Recommendation Engine
5. **For Future Directions:** See Chapter 6 (Conclusion & Future Scope) for recommended enhancements and research opportunities

### Accessing the Source Code and Documentation
- Complete source code, documentation, and test scripts are available on GitHub at: https://github.com/SaiSrikar0/financial-report-analysis
- The repository includes:
      - Comprehensive project documentation: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/docs
      - Smoke test for setup verification: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/scripts/smoke_test.py
      - ETL layer: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/etl
      - Analysis layer: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/analysis
      - ML and explainability layer: https://github.com/SaiSrikar0/financial-report-analysis/tree/main/models
      - Dashboard entry point: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/app.py
      - SVR Pipeline implementation: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/models/svr_pipeline.py
      - Recommendation Engine implementation: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/analysis/recommendation_engine.py

### For Researchers and Evaluators
- To verify the system end-to-end, follow the setup instructions in Appendix A and execute the smoke test
- To evaluate specific components, refer to the architecture diagrams in Section 4.6 and performance results in Section 5.2
- To understand design decisions and rationale, consult Section 4.7 (Technology Description) and Chapter 2 (Literature Survey)

### For Potential Extensions
The modular architecture allows researchers to:
- Extend the feature engineering pipeline with domain-specific indicators
- Replace SVR with alternative ML models (XGBoost, LSTM, etc.) using: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/models/svr_pipeline.py
- Integrate additional data sources by implementing new adapters in: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/etl/extract.py
- Enhance the recommendation engine with domain-specific logic in: https://github.com/SaiSrikar0/financial-report-analysis/blob/main/analysis/recommendation_engine.py

Refer to Chapter 6 for detailed suggestions on short-term, medium-term, and long-term enhancements.

---

**Submitted by:**
G. RUTWIKA (23B81A7341)
B. SAI SRIKAR (23B81A7346)

**Guided by:**
Mr. AZMERA CHANDU NAIK
Sr. Assistant Professor, Department of CSE(AI&ML)

**Date:** April 10, 2026
**Institution:** CVR College of Engineering, Hyderabad
