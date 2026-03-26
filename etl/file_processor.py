"""
File processor for user-uploaded financial reports.
Supports CSV, Excel, PDF, and JSON.
"""

import json
import io
import pandas as pd
from typing import List, Dict, Any

SUPPORTED_TYPES = ["csv", "xlsx", "xls", "pdf", "json"]


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