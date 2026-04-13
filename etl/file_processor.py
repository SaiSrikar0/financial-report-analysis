"""
File processor for user-uploaded financial reports.
Supports CSV, Excel, PDF, and JSON.
"""

import json
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any

SUPPORTED_TYPES = ["csv", "xlsx", "xls", "pdf", "json"]

DIRECT_STANDARD_COLUMNS = [
    "date",
    "ticker",
    "revenue",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cashflow",
]


def process_upload(uploaded_file) -> List[Dict[str, Any]]:
    """
    Accept a Streamlit UploadedFile object.
    Returns a list of raw dicts representing rows/pages.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return _from_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return _from_excel(uploaded_file)
    elif name.endswith(".pdf"):
        return _from_pdf(uploaded_file)
    elif name.endswith(".json"):
        return _from_json(uploaded_file)
    else:
        raise ValueError(
            f"Unsupported file type: {name}. Supported: {SUPPORTED_TYPES}"
        )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def has_direct_standard_schema(raw_records: List[Dict[str, Any]]) -> bool:
    """Return True when records already match FinCast's tabular schema."""
    if not raw_records:
        return False

    df = pd.DataFrame(raw_records)
    if df.empty:
        return False

    df = _normalise_columns(df)
    return all(col in df.columns for col in DIRECT_STANDARD_COLUMNS)


def build_direct_standard_and_category_records(
    raw_records: List[Dict[str, Any]],
    ticker_fallback: str = "UNKNOWN",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically convert structured tabular uploads into
    standard/category records without LLM extraction.
    """
    df = pd.DataFrame(raw_records)
    df = _normalise_columns(df)

    missing = [c for c in DIRECT_STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build direct records. Missing required columns: {missing}"
        )

    std = df[DIRECT_STANDARD_COLUMNS].copy()
    std["date"] = pd.to_datetime(std["date"], errors="coerce")

    numeric_cols = [
        "revenue",
        "operating_income",
        "net_income",
        "total_assets",
        "total_liabilities",
        "operating_cashflow",
    ]
    for col in numeric_cols:
        std[col] = pd.to_numeric(std[col], errors="coerce")

    std["ticker"] = (
        std["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": None, "NAN": None, "NONE": None})
    )
    if ticker_fallback:
        std["ticker"] = std["ticker"].fillna(ticker_fallback)

    std = std.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    # Convert dates back to string for JSON-style records consumed downstream.
    std["date"] = std["date"].dt.strftime("%Y-%m-%d")

    standard_records = std.where(pd.notnull(std), None).to_dict(orient="records")

    # Build category rows with deterministic defaults/classification.
    calc = std.copy()
    calc["date_dt"] = pd.to_datetime(calc["date"], errors="coerce")
    calc = calc.sort_values(["ticker", "date_dt"], kind="stable")
    calc["revenue_growth"] = calc.groupby("ticker")["revenue"].pct_change() * 100
    calc["debt_to_asset"] = calc["total_liabilities"] / (calc["total_assets"] + 1e-9)

    category = calc[["ticker", "date", "revenue", "operating_income", "net_income"]].copy()
    category["sector"] = "Unknown"
    category["category"] = np.select(
        [
            calc["revenue_growth"] > 10,
            (calc["revenue_growth"] > 0) & (calc["revenue_growth"] <= 10),
            calc["revenue_growth"].notna(),
        ],
        ["High Growth", "Moderate Growth", "Stable"],
        default="Unknown",
    )
    category["risk_level"] = np.select(
        [
            calc["debt_to_asset"] > 0.7,
            (calc["debt_to_asset"] > 0.4) & (calc["debt_to_asset"] <= 0.7),
            calc["debt_to_asset"].notna(),
        ],
        ["High Risk", "Medium Risk", "Low Risk"],
        default="Unknown",
    )

    category = category[
        [
            "ticker",
            "date",
            "sector",
            "category",
            "risk_level",
            "revenue",
            "operating_income",
            "net_income",
        ]
    ]
    category_records = category.where(pd.notnull(category), None).to_dict(orient="records")

    return standard_records, category_records


def _from_csv(f) -> List[Dict]:
    df = pd.read_csv(f)
    df = _normalise_columns(df)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _from_excel(f) -> List[Dict]:
    xl = pd.ExcelFile(f)
    best_df, best_score = None, -1
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        score = df.select_dtypes(include="number").shape[1]
        if score > best_score:
            best_score, best_df = score, df
    best_df = _normalise_columns(best_df)
    return best_df.where(pd.notnull(best_df), None).to_dict(orient="records")


def _from_pdf(f) -> List[Dict]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required: pip install pdfplumber")

    results = []
    with pdfplumber.open(f) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if not table or not table[0]:
                        continue
                    headers = [
                        str(h).strip().lower().replace(" ", "_") if h else f"col_{j}"
                        for j, h in enumerate(table[0])
                    ]
                    for row in table[1:]:
                        record = {
                            headers[j]: (row[j] if j < len(row) else None)
                            for j in range(len(headers))
                        }
                        record["_source_page"] = i + 1
                        results.append(record)
            else:
                text = page.extract_text() or ""
                if text.strip():
                    results.append(
                        {"_source_page": i + 1, "_raw_text": text.strip()}
                    )
    return results


def _from_json(f) -> List[Dict]:
    data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return []