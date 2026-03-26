"""
Two-prompt LLM extraction layer.
Prompt 1: Map arbitrary raw financial records → FinCast standard schema.
Prompt 2: Derive category/classification fields.
Schema matches your actual standard_table and category_table columns.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"  # change here only if needed


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


# Matches your actual standard_table schema
STANDARD_SCHEMA = {
    "date": "string — fiscal period end date as YYYY-MM-DD (use fiscal year end or quarter end)",
    "ticker": "string — company ticker symbol or short name supplied by user",
    "revenue": "float — total revenue in USD (convert millions/billions as needed)",
    "operating_income": "float — operating income/profit in USD",
    "net_income": "float — net income after tax in USD",
    "operating_cashflow": "float — operating cash flow in USD (null if unavailable)",
    "total_assets": "float — total assets in USD",
    "total_liabilities": "float — total liabilities in USD",
}

# Matches your actual category_table schema
CATEGORY_SCHEMA = {
    "ticker": "string — same ticker as standard schema",
    "date": "string — same date as standard schema",
    "sector": "string — e.g. Technology, Healthcare, Finance, Retail, Energy",
    "category": "string — one of: High Growth, Moderate Growth, Stable, Unknown",
    "risk_level": "string — one of: High Risk, Medium Risk, Low Risk, Unknown",
    "revenue": "float — same revenue as standard schema",
    "operating_income": "float — same operating_income as standard schema",
    "net_income": "float — same net_income as standard schema",
}


def _parse_json(text: str) -> Any:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON: {clean}")


def extract_standard_schema(
    raw_records: List[Dict], ticker: str = "UNKNOWN"
) -> List[Dict]:
    """
    Prompt 1: Normalize arbitrary raw financial records to FinCast standard schema.
    Handles variable column names, unit differences, date formats.
    """
    system = (
        "You are a financial data normalization expert. "
        "Convert raw financial records into a strictly defined schema. "
        "Rules: "
        "- All monetary values must be in full USD (if values appear to be in millions, multiply by 1,000,000; billions by 1,000,000,000). "
        "- date must be YYYY-MM-DD format (fiscal year/quarter end). "
        "- ticker is provided by the user — use it exactly as given. "
        "- If a field cannot be determined, use null. "
        "- Return ONLY valid JSON. No explanation. No markdown."
    )

    user = (
        f"Ticker: {ticker}\n\n"
        f"Target schema (field: description):\n{json.dumps(STANDARD_SCHEMA, indent=2)}\n\n"
        f"Raw records (first 30 shown):\n{json.dumps(raw_records[:30], indent=2)}\n\n"
        "Convert ALL records to the target schema. Return a JSON array only."
    )

    response = call_llm(system, user)
    try:
        return _parse_json(response)
    except ValueError:
        # Retry with stricter instruction
        system_strict = system + " Ensure the output is valid JSON with no extra text."
        response_retry = call_llm(system_strict, user)
        return _parse_json(response_retry)


def extract_category_schema(
    standard_records: List[Dict], ticker: str = "UNKNOWN"
) -> List[Dict]:
    """
    Prompt 2: Derive category_table fields from standard records.
    Matches your category_table schema exactly.
    """
    system = (
        "You are a financial analyst computing classifications from financial records. "
        "Rules: "
        "- sector: infer from company name/ticker (Technology, Healthcare, Finance, Retail, Energy, etc.) "
        "- category: High Growth (revenue_growth>10%), Moderate Growth (0-10%), Stable (<0% or no prior year), Unknown "
        "- risk_level: High Risk (debt_to_asset>0.7), Medium Risk (0.4-0.7), Low Risk (<0.4), Unknown "
        "- Compute revenue_growth as YoY % change between consecutive years for same ticker "
        "- Compute debt_to_asset = total_liabilities / total_assets "
        "- revenue, operating_income, net_income: copy from standard records unchanged "
        "- Return ONLY valid JSON. No explanation. No markdown."
    )

    user = (
        f"Ticker context: {ticker}\n\n"
        f"Target schema (field: description):\n{json.dumps(CATEGORY_SCHEMA, indent=2)}\n\n"
        f"Standard records:\n{json.dumps(standard_records, indent=2)}\n\n"
        "Compute and return the category schema as a JSON array only."
    )

    response = call_llm(system, user)
    try:
        return _parse_json(response)
    except ValueError:
        # Retry with stricter instruction
        system_strict = system + " Ensure the output is valid JSON with no extra text."
        response_retry = call_llm(system_strict, user)
        return _parse_json(response_retry)


def run_extraction_pipeline(
    raw_records: List[Dict], ticker: str = "UNKNOWN"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Full two-prompt pipeline.
    Returns (standard_records, category_records).
    """
    print(f"[LLM Extractor] Prompt 1: normalising schema for {ticker}...")
    standard = extract_standard_schema(raw_records, ticker)
    # Ensure ticker field is set correctly on all records
    for rec in standard:
        if not rec.get("ticker"):
            rec["ticker"] = ticker
    print(f"[LLM Extractor] ✓ {len(standard)} standard records extracted.")

    print("[LLM Extractor] Prompt 2: deriving category/classification fields...")
    category = extract_category_schema(standard, ticker)
    for rec in category:
        if not rec.get("ticker"):
            rec["ticker"] = ticker
    print(f"[LLM Extractor] ✓ {len(category)} category records derived.")

    return standard, category