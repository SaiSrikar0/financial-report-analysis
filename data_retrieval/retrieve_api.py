"""
retrieval_api.py

Fetch financial statements from Alpha Vantage API and store raw JSON
into data/raw without modifying the ETL pipeline.
"""

import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = os.getenv("ALPHAVANTAGE_BASE_URL")

if not API_KEY:
    raise ValueError("Missing ALPHAVANTAGE_API_KEY in .env")

print(API_KEY)
print(BASE_URL)


# -----------------------------
# API fetch helper
# -----------------------------
def fetch_statement(symbol: str, function_name: str):
    url = f"{BASE_URL}?function={function_name}&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    annual = data.get("annualReports", [])
    quarterly = data.get("quarterlyReports", [])

    combined = annual + quarterly

    seen = set()
    unique = []
    for r in combined:
        d = r.get("fiscalDateEnding")
        if d and d not in seen:
            seen.add(d)
            unique.append(r)

    return unique


# -----------------------------
# Merge financial statements
# -----------------------------
def merge_financials(symbol: str):
    income = fetch_statement(symbol, "INCOME_STATEMENT")
    balance = fetch_statement(symbol, "BALANCE_SHEET")
    cashflow = fetch_statement(symbol, "CASH_FLOW")

    def index_by_year(records):
        return {r["fiscalDateEnding"]: r for r in records}

    inc_map = index_by_year(income)
    bal_map = index_by_year(balance)
    cf_map = index_by_year(cashflow)

    years = set(inc_map) & set(bal_map) & set(cf_map)

    merged = []

    for year in sorted(years):
        inc = inc_map[year]
        bal = bal_map[year]
        cf = cf_map[year]

        record = {
            "date": year,
            "ticker": symbol,

            # Income
            "revenue": float(inc.get("totalRevenue", 0) or 0),
            "operating_income": float(inc.get("operatingIncome", 0) or 0),
            "net_income": float(inc.get("netIncome", 0) or 0),

            # Balance
            "total_assets": float(bal.get("totalAssets", 0) or 0),
            "total_liabilities": float(bal.get("totalLiabilities", 0) or 0),

            # Cashflow
            "operating_cashflow": float(cf.get("operatingCashflow", 0) or 0)
        }

        merged.append(record)

    return merged


# -----------------------------
# Save to raw folder
# -----------------------------
def save_raw_data(records, filename="financial_data_raw.json"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    path = os.path.join(raw_dir, filename)

    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {path}")
    return path


# -----------------------------
# Main execution
# -----------------------------
def fetch_and_store(symbols=None):
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    all_data = []

    for sym in symbols:
        print(f"Fetching financials for {sym}...")
        try:
            merged = merge_financials(sym)
            all_data.extend(merged)
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

    return save_raw_data(all_data)


if __name__ == "__main__":
    fetch_and_store()