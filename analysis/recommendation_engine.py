"""
Phase 6: LLM-based recommendation engine.
Reads Phase 4 + Phase 5 report CSVs and calls Claude to generate
structured financial intelligence reports.
Schema is aligned with your actual report column names (ticker, not company).
"""

import json
import os
import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"  # change here only if needed


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_periods_per_year_from_target(period_target_pct: float, annual_target_pct: float = 10.0) -> float:
    """Infer periods/year from per-period target using compound growth math."""
    p = _safe_float(period_target_pct, annual_target_pct) / 100.0
    a = _safe_float(annual_target_pct, 10.0) / 100.0
    if p <= -0.9999 or a <= -0.9999:
        return 1.0
    if abs(p) < 1e-9:
        return 1.0
    if p >= a:
        return 1.0
    try:
        periods = np.log1p(a) / np.log1p(p)
        if not np.isfinite(periods) or periods <= 0:
            return 1.0
        return float(max(1.0, min(365.0, periods)))
    except Exception:
        return 1.0


def _annualize_period_growth(period_growth_pct: float, periods_per_year: float) -> float:
    period_decimal = _safe_float(period_growth_pct, 0.0) / 100.0
    n = max(1.0, _safe_float(periods_per_year, 1.0))
    if period_decimal <= -0.9999:
        return -100.0
    annual_decimal = (1.0 + period_decimal) ** n - 1.0
    return annual_decimal * 100.0


def _score_from_signals(pred_growth_annual: float, gap_annual: float, r2: float, overall_risk: str) -> int:
    """Deterministic 1-10 score based on annualized growth, model fit, and risk."""
    score = 5

    if pred_growth_annual >= 12:
        score += 3
    elif pred_growth_annual >= 8:
        score += 2
    elif pred_growth_annual >= 4:
        score += 1
    elif pred_growth_annual < 0:
        score -= 2

    if gap_annual >= 0:
        score += 1
    elif gap_annual < -5:
        score -= 1

    if r2 >= 0.7:
        score += 1
    elif r2 < 0.3:
        score -= 1

    risk = str(overall_risk or "Medium").strip().lower()
    if risk == "low":
        score += 1
    elif risk == "high":
        score -= 1

    return int(max(1, min(10, score)))


def _normalize_recommendation(result: Dict[str, Any], analysis_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Align LLM output to actual model signals and add deterministic score."""
    svr = analysis_bundle.get("svr_predictions", {}) or {}
    metrics = analysis_bundle.get("model_metrics", {}) or {}

    growth_outlook = result.get("growth_outlook") or {}
    risk_assessment = result.get("risk_assessment") or {}

    predicted_period = _safe_float(
        svr.get("predicted_growth_rate", growth_outlook.get("predicted_growth_rate", 0.0)),
        0.0,
    )
    target_period = _safe_float(svr.get("target_growth_rate", 10.0), 10.0)
    gap_period = _safe_float(
        svr.get("gap_vs_target", predicted_period - target_period),
        predicted_period - target_period,
    )
    gap_status = "surplus" if gap_period >= 0 else "shortfall"

    periods_per_year = _infer_periods_per_year_from_target(target_period, annual_target_pct=10.0)
    predicted_annual = _annualize_period_growth(predicted_period, periods_per_year)
    target_annual = _annualize_period_growth(target_period, periods_per_year)
    gap_annual = predicted_annual - target_annual

    r2 = _safe_float(metrics.get("r2", 0.0), 0.0)
    if r2 >= 0.7:
        confidence = "High"
    elif r2 >= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    overall_risk = risk_assessment.get("overall_risk", "Medium")
    performance_score = _score_from_signals(predicted_annual, gap_annual, r2, overall_risk)

    growth_outlook["predicted_growth_rate"] = float(predicted_period)
    growth_outlook["target_growth_rate"] = float(target_period)
    growth_outlook["gap_vs_target"] = float(gap_period)
    growth_outlook["gap_status"] = gap_status
    growth_outlook["confidence"] = confidence
    growth_outlook["periods_per_year"] = round(periods_per_year, 2)
    growth_outlook["predicted_growth_rate_annualized"] = float(predicted_annual)
    growth_outlook["target_growth_rate_annualized"] = float(target_annual)
    growth_outlook["gap_vs_target_annualized"] = float(gap_annual)

    result["growth_outlook"] = growth_outlook
    result["performance_score"] = performance_score
    return result


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are FinCast, a senior financial analyst providing CRITICAL investment intelligence.
You receive a JSON bundle with SVR growth predictions, SHAP feature importance, financial ratios, and anomaly data.

CRITICAL REQUIREMENTS:
- For HIGH/MEDIUM risk: Provide specific, data-backed concerns (NOT general statements)
- For growth analysis: Explain WHY the gap exists based on actual metrics (debt, margins, efficiency)
- For opportunities: Identify concrete, actionable levers (cost reduction, revenue acceleration, capital efficiency)
- For anomalies: Explain the severity and cascading business impact
- All insights must reference specific metrics from the bundle (debt_to_asset, margins, growth rates, SHAP drivers)
- Use the provided `target_growth_rate` and `gap_vs_target` from the bundle; do not assume a fixed 10% target.

Return ONLY valid JSON (no explanation, no markdown) matching this exact schema:

{
  "executive_summary": "2-3 sentences: concise health assessment including specific red flags or strengths",
  "performance_score": <integer 1-10 where 1=critical risk, 10=exceptional>,
  "growth_outlook": {
    "forecast": "2-3 sentences: specific growth trajectory with REASONS (SVR model predicts X% because of Y metrics)",
    "predicted_growth_rate": <float - SVR predicted YoY growth %>,
    "gap_vs_target": <float - gap vs configured target from bundle>,
    "gap_status": "shortfall or surplus",
    "confidence": "Low | Medium | High (based on R2 and model stability)",
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


def load_analysis_bundle_from_reports(
    ticker: str, reports_dir: str = "analysis/reports"
) -> Dict[str, Any]:
    """
    Assemble the analysis bundle for a ticker from Phase 4 + 5 report CSVs.
    All CSV files use 'ticker' column to match your schema.
    """
    bundle: Dict[str, Any] = {"ticker": ticker}

    svr_path = os.path.join(reports_dir, "svr_future_predictions.csv")
    if os.path.exists(svr_path):
        df = pd.read_csv(svr_path)
        row = df[df["ticker"].str.upper() == ticker.upper()]
        if not row.empty:
            bundle["svr_predictions"] = row.iloc[0].to_dict()

    metrics_path = os.path.join(reports_dir, "svr_evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        bundle["model_metrics"] = {
            "mae": round(float(m.get("mae", 0)), 2),
            "rmse": round(float(m.get("rmse", 0)), 2),
            "r2": round(float(m.get("r2", 0)), 4),
        }

    shap_global_path = os.path.join(reports_dir, "phase_5_shap_global_importance.csv")
    if os.path.exists(shap_global_path):
        shap_df = pd.read_csv(shap_global_path).head(8)
        bundle["shap_global_top_features"] = shap_df.to_dict(orient="records")

    shap_local_path = os.path.join(reports_dir, "phase_5_shap_local_explanations.csv")
    if os.path.exists(shap_local_path):
        local_df = pd.read_csv(shap_local_path)
        ticker_local = local_df[local_df["ticker"].str.upper() == ticker.upper()]
        if not ticker_local.empty:
            bundle["shap_local_drivers"] = ticker_local.to_dict(orient="records")

    std_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "staged",
        "standard_table.csv",
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


def generate_recommendations(analysis_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Groq with the analysis bundle and return a structured
    financial intelligence report dict.
    """
    user_content = (
        "Generate the financial intelligence report for:\n\n"
        f"{json.dumps(analysis_bundle, indent=2, default=str)}"
    )

    try:
        raw = call_llm(SYSTEM_PROMPT, user_content)
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(clean)
    except (json.JSONDecodeError, Exception):
        system_strict = SYSTEM_PROMPT + " Ensure the output is valid JSON with no extra text."
        try:
            raw_retry = call_llm(system_strict, user_content)
            clean_retry = re.sub(r"```(?:json)?", "", raw_retry).strip().rstrip("`").strip()
            result = json.loads(clean_retry)
        except Exception as e2:
            raise ValueError(f"Failed to generate recommendations after retry: {str(e2)}")

    result = _normalize_recommendation(result, analysis_bundle)

    reports_dir = "analysis/reports"
    os.makedirs(reports_dir, exist_ok=True)
    ticker = analysis_bundle.get("ticker", "unknown")
    out_path = os.path.join(reports_dir, f"phase_6_{ticker.lower()}_recommendations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[RecommendationEngine] Saved recommendations to {out_path}")

    return result
