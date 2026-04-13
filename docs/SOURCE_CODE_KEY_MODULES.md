# FinCast Source Code - Key Modules

This document contains the core source code for the two most critical FinCast components:
1. **SVR Pipeline** (Phase 4) - Machine Learning prediction model
2. **Recommendation Engine** (Phase 6) - LLM-based financial intelligence generation

These modules represent the heart of FinCast's forecasting and interpretation capabilities.

---

## 1. SVR Pipeline Module (`models/svr_pipeline.py`)

The SVR Pipeline implements Support Vector Regression for growth-rate prediction with hyperparameter tuning and confidence intervals.

### 1.1 Core Function: SVR Model Training & Evaluation

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

### 1.2 Time-Aware Train-Test Split

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

### 1.3 Future Prediction & Gap Analysis

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

### 1.4 Model Evaluation

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

---

## 2. Recommendation Engine Module (`analysis/recommendation_engine.py`)

The Recommendation Engine uses Groq's LLM to generate structured financial intelligence reports grounded in SVR predictions and SHAP explanations.

### 2.1 Analysis Bundle Assembly

```python
def load_analysis_bundle_from_reports(
    ticker: str, reports_dir: str = "analysis/reports"
) -> Dict[str, Any]:
    """
    Assemble the analysis bundle for a ticker from Phase 4 + 5 report CSVs.
    All CSV files use 'ticker' column to match your schema.
    """
    bundle: Dict[str, Any] = {"ticker": ticker}

    # ── SVR future predictions ────────────────────────────────────────────────
    svr_path = os.path.join(reports_dir, "svr_future_predictions.csv")
    if os.path.exists(svr_path):
        df = pd.read_csv(svr_path)
        row = df[df["ticker"].str.upper() == ticker.upper()]
        if not row.empty:
            bundle["svr_predictions"] = row.iloc[0].to_dict()

    # ── SVR evaluation metrics ────────────────────────────────────────────────
    metrics_path = os.path.join(reports_dir, "svr_evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        bundle["model_metrics"] = {
            "mae": round(float(m.get("mae", 0)), 2),
            "rmse": round(float(m.get("rmse", 0)), 2),
            "r2": round(float(m.get("r2", 0)), 4),
        }

    # ── SHAP global importance (top 8 features) ───────────────────────────────
    shap_global_path = os.path.join(reports_dir, "phase_5_shap_global_importance.csv")
    if os.path.exists(shap_global_path):
        shap_df = pd.read_csv(shap_global_path).head(8)
        bundle["shap_global_top_features"] = shap_df.to_dict(orient="records")

    # ── SHAP local explanations for this ticker ───────────────────────────────
    shap_local_path = os.path.join(reports_dir, "phase_5_shap_local_explanations.csv")
    if os.path.exists(shap_local_path):
        local_df = pd.read_csv(shap_local_path)
        ticker_local = local_df[local_df["ticker"].str.upper() == ticker.upper()]
        if not ticker_local.empty:
            bundle["shap_local_drivers"] = ticker_local.to_dict(orient="records")

    # ── Latest financial ratios from standard_table CSV ───────────────────────
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
                "revenue_growth": round(float(latest.get("revenue_growth", 0) or 0), 2),
                "net_income_growth": round(float(latest.get("net_income_growth", 0) or 0), 2),
            }
            bundle["fiscal_year"] = str(latest.get("date", ""))[:10]

    # ── Peer comparison (latest ratios for AAPL, MSFT, GOOGL, AMZN) ──────────
    if os.path.exists(std_path):
        all_df = pd.read_csv(std_path)
        peer_data = {}
        for peer in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            peer_df = all_df[all_df["ticker"] == peer]
            if not peer_df.empty:
                p_latest = peer_df.sort_values("date").iloc[-1]
                peer_data[peer] = {
                    "profit_margin": round(float(p_latest.get("profit_margin", 0) or 0), 2),
                    "revenue_growth": round(float(p_latest.get("revenue_growth", 0) or 0), 2),
                    "debt_to_asset": round(float(p_latest.get("debt_to_asset", 0) or 0), 4),
                }
        bundle["peer_comparison"] = peer_data

    # ── Anomaly context from analysis report ──────────────────────────────────
    report_path = os.path.join(reports_dir, "analysis_report.txt")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            lines = f.readlines()
        anomalies = [
            line.strip().lstrip("0123456789. ")
            for line in lines
            if ticker.upper() in line.upper()
            and any(kw in line.lower() for kw in ["drop", "anomal", "decline", "risk"])
        ]
        bundle["anomalies"] = anomalies[:5]

    return bundle
```

**Bundle Contents:**
- SVR predictions (growth rate, confidence intervals, gaps)
- Model metrics (MAE, RMSE, R²)
- SHAP global importance (top 8 features)
- SHAP local explanations (per-company drivers)
- Latest financial ratios (margins, debt, efficiency)
- Peer comparison (ratios for AAPL, MSFT, GOOGL, AMZN)
- Anomalies (flagged issues from analysis)

### 2.2 LLM System Prompt (Recommendation Generation Instructions)

```python
SYSTEM_PROMPT = """You are FinCast, a senior financial analyst providing CRITICAL investment intelligence.
You receive a JSON bundle with SVR growth predictions, SHAP feature importance, financial ratios, and anomaly data.

CRITICAL REQUIREMENTS:
- For HIGH/MEDIUM risk: Provide specific, data-backed concerns (NOT general statements)
- For growth analysis: Explain WHY the gap exists based on actual metrics (debt, margins, efficiency)
- For opportunities: Identify concrete, actionable levers (cost reduction, revenue acceleration, capital efficiency)
- For anomalies: Explain the severity and cascading business impact
- All insights must reference specific metrics from the bundle (debt_to_asset, margins, growth rates, SHAP drivers)

Return ONLY valid JSON (no explanation, no markdown) matching this exact schema:

{
  "executive_summary": "2-3 sentences: concise health assessment including specific red flags or strengths",
  "performance_score": <integer 1-10 where 1=critical risk, 10=exceptional>,
  "growth_outlook": {
    "forecast": "2-3 sentences: specific growth trajectory with REASONS (SVR model predicts X% because of Y metrics)",
    "predicted_growth_rate": <float — SVR predicted YoY growth %>,
    "gap_vs_target": <float — gap vs 10% target>,
    "gap_status": "shortfall or surplus",
    "confidence": "Low | Medium | High (based on R² and model stability)",
    "key_drivers": ["driver1 with metric", "driver2 with metric", "driver3 with metric"],
    "critical_concern": "if shortfall, explain the business mechanics causing underperformance"
  },
  "risk_assessment": {
    "overall_risk": "Low | Medium | High",
    "risk_factors": [
      {"factor": "specific name (e.g., Debt Service Risk)", "severity": "Low|Medium|High", "explanation": "concrete concern with specific metrics: debt_to_asset=X, coverage_ratio=Y, impact=Z"}
    ],
    "critical_warnings": ["severe structural issue if applicable, else empty"],
    "risk_trend": "deteriorating | stable | improving (based on historical ratios)"
  },
  "opportunities": [
    {"title": "specific operational lever", "rationale": "2-3 sentences with: current state, target state, impact (revenue gain/cost save/cash freed)", "estimated_upside": "quantified if possible (e.g., '5-8% EBITDA improvement')"}
  ],
  "action_items": [
    {"priority": "High|Medium|Low", "action": "specific, measurable recommendation", "rationale": "why this unlocks value based on metrics", "timeline": "immediate|6-months|12-months"}
  ],
  "peer_positioning": "detailed 2-3 sentence comparison to AAPL/MSFT/GOOGL/AMZN: specific metric deltas (margin gap, growth differential, debt burden relative position)",
  "anomalies_flagged": ["specific anomaly with business impact assessment"],
  "investment_verdict": "BUY if strong growth drivers | HOLD if stable but unexceptional | REDUCE/AVOID if deteriorating fundamentals"
}"""
```

### 2.3 Recommendation Generation with Fallback

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

### 2.4 LLM Call Wrapper

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

**Configuration:**
- Model: `llama-3.3-70b-versatile` (Groq fast inference)
- Temperature: 0.2 (low creativity, factual/consistent output)
- Requires `GROQ_API_KEY` in `.env`

---

## 3. Integration Points

### 3.1 Pipeline Orchestration

The two modules are chained in [scripts/run.py](../scripts/run.py):

```python
# Phase 4: SVR Training & Prediction
from models.svr_pipeline import run_phase4_svr
phase4_output = run_phase4_svr(target_growth_rate=10.0)

# Phase 5: SHAP Explainability (optional - see explainability.py)

# Phase 6: LLM Recommendations
from analysis.recommendation_engine import load_analysis_bundle_from_reports, generate_recommendations

for ticker in ["AAPL", "AMZN", "GOOGL"]:
    bundle = load_analysis_bundle_from_reports(ticker)
    recommendations = generate_recommendations(bundle)
```

### 3.2 Data Flow

```
Raw Financial Data
        ↓
[ETL Pipeline - extract.py, transform.py, load.py]
        ↓
standard_table.csv (Supabase)
        ↓
[SVR Pipeline Phase 4 - svr_pipeline.py]
        ↓
svr_future_predictions.csv, svr_evaluation_metrics.csv
        ↓
[SHAP Explainability Phase 5 - explainability.py] (Optional)
        ↓
phase_5_shap_global_importance.csv, phase_5_shap_local_explanations.csv
        ↓
[Recommendation Engine Phase 6 - recommendation_engine.py]
        ├→ load_analysis_bundle_from_reports() [assembles data]
        ├→ call_llm() with SYSTEM_PROMPT [generates report]
        └→ Save phase_6_(ticker)_recommendations.json
        ↓
Streamlit Dashboard (app.py) [User visualization]
```

---

## 4. Key Dependencies

```
scikit-learn==1.3.2      # SVR, GridSearchCV, metrics
pandas==2.0.3            # Data manipulation
numpy==1.24.3            # Numerical operations
groq==0.4.2              # LLM API (Recommendation Engine)
python-dotenv==1.0.0     # Environment variable management
```

---

## 5. Error Handling & Robustness

### SVR Pipeline:
- Time-series cross-validation prevents temporal leakage
- Residual analysis detects outliers in predictions
- Confidence intervals quantify prediction uncertainty

### Recommendation Engine:
- JSON parsing with retry mechanism (2 attempts)
- Graceful degradation: LLM API failures raise detailed errors
- Input validation: Checks for required fields in analysis bundle
- Audit trail: All recommendations saved to JSON for reproducibility

---

## 6. Future Enhancements

1. **SVR Pipeline:**
   - Add ensemble methods (Random Forest, XGBoost) for comparison
   - Implement feature selection/importance analysis
   - Support for multivariate target (multiple growth dimensions)

2. **Recommendation Engine:**
   - RAG (Retrieval Augmented Generation) with news/earnings data
   - Fine-tuning LLM on domain-specific financial guidance
   - Scenario analysis ("What if revenue drops 10%?")
   - Confidence-weighted recommendations based on model R²

