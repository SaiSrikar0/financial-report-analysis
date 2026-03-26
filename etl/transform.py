"""ETL transform layer: clean, engineer features, and stage ML/LLM tables."""

import json
import os

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "revenue",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cashflow",
]


def _load_raw_df(raw_path: str) -> pd.DataFrame:
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "revenue", "operating_income", "net_income",
        "total_assets", "total_liabilities", "operating_cashflow",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "ticker", "revenue", "net_income"]).copy()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} incomplete rows")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    out = df.copy()
    out["profit_margin"] = (out["net_income"] / (out["revenue"] + eps)) * 100
    out["operating_margin"] = (out["operating_income"] / (out["revenue"] + eps)) * 100
    out["revenue_growth"] = out.groupby("ticker")["revenue"].pct_change() * 100
    out["net_income_growth"] = out.groupby("ticker")["net_income"].pct_change() * 100
    out["asset_efficiency"] = out["revenue"] / (out["total_assets"] + eps)
    out["debt_to_asset"] = out["total_liabilities"] / (out["total_assets"] + eps)
    return out


def _build_standard_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date", "ticker", "revenue", "operating_income", "net_income",
        "operating_cashflow", "total_assets", "total_liabilities",
        "profit_margin", "operating_margin", "revenue_growth",
        "net_income_growth", "asset_efficiency", "debt_to_asset",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def _build_category_table(df: pd.DataFrame) -> pd.DataFrame:
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology",
        "GOOGL": "Technology", "AMZN": "Technology",
    }

    cat = df[
        ["ticker", "date", "revenue", "operating_income", "net_income",
         "revenue_growth", "debt_to_asset"]
    ].copy()
    cat["sector"] = cat["ticker"].map(sector_map).fillna("Unknown")

    cat["category"] = np.select(
        [
            cat["revenue_growth"] > 10,
            (cat["revenue_growth"] > 0) & (cat["revenue_growth"] <= 10),
            cat["revenue_growth"].notna(),
        ],
        ["High Growth", "Moderate Growth", "Stable"],
        default="Unknown",
    )

    cat["risk_level"] = np.select(
        [
            cat["debt_to_asset"] > 0.7,
            (cat["debt_to_asset"] > 0.4) & (cat["debt_to_asset"] <= 0.7),
            cat["debt_to_asset"].notna(),
        ],
        ["High Risk", "Medium Risk", "Low Risk"],
        default="Unknown",
    )

    return cat[
        ["ticker", "date", "sector", "category", "risk_level",
         "revenue", "operating_income", "net_income"]
    ]


def _sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def transform_data(raw_path=None):
    etl_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(etl_dir)

    if raw_path is None:
        raw_path = os.path.join(base_dir, "data", "raw", "financial_data_raw.json")

    staged_dir = os.path.join(base_dir, "data", "staged")
    os.makedirs(staged_dir, exist_ok=True)

    print(f"Loading raw data from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = _load_raw_df(raw_path)
    print(f"Loaded {len(df)} records from raw data")

    df = _clean_data(df)
    df = _engineer_features(df)

    standard_df = _sanitize_for_csv(_build_standard_table(df))
    category_df = _sanitize_for_csv(_build_category_table(df))

    standard_path = os.path.join(staged_dir, "standard_table.csv")
    category_path = os.path.join(staged_dir, "category_table.csv")

    standard_df.to_csv(standard_path, index=False)
    category_df.to_csv(category_path, index=False)

    print(f"Standard table saved to {standard_path}")
    print(f"Category table saved to {category_path}")

    return standard_path, category_path


# ── New: dynamic transform for user-uploaded LLM-extracted data ───────────────

def transform_dynamic(
    standard_records: list,
    category_records: list,
    user_id: str = "anonymous",
    ticker: str = None,
) -> dict:
    """
    Transform LLM-extracted records into DataFrames matching your exact
    standard_table and category_table schemas. Tags every row with user_id.

    Args:
        standard_records: output from llm_extractor.extract_standard_schema()
        category_records: output from llm_extractor.extract_category_schema()
        user_id: Supabase auth user UUID (or 'predefined' for built-in data)
        ticker: Company ticker (optional, used to fill any NULL ticker values as backup)

    Returns:
        {"standard_table": pd.DataFrame, "category_table": pd.DataFrame}
    """
    eps = 1e-9

    # ── Standard table ────────────────────────────────────────────────────────
    std_df = pd.DataFrame(standard_records)

    # Normalise types
    std_df["date"] = pd.to_datetime(std_df.get("date"), errors="coerce")
    numeric_cols = [
        "revenue", "operating_income", "net_income",
        "operating_cashflow", "total_assets", "total_liabilities",
    ]
    for col in numeric_cols:
        if col in std_df.columns:
            std_df[col] = pd.to_numeric(std_df[col], errors="coerce")
        else:
            std_df[col] = np.nan

    # ── Ensure ticker is not NULL (belt-and-suspenders) ──────────────────────
    if "ticker" in std_df.columns:
        # First try to fill missing tickers from records that have them
        if std_df["ticker"].isna().any():
            valid_tickers = std_df[std_df["ticker"].notna()]["ticker"].unique()
            if len(valid_tickers) > 0:
                # Use the first valid ticker value from the records
                std_df["ticker"].fillna(valid_tickers[0], inplace=True)
            elif ticker:
                # Fallback to explicit ticker parameter if all records had NULL
                std_df["ticker"].fillna(ticker, inplace=True)
    
    # Fallback: if ticker column still doesn't exist or is all NULL, use parameter
    if ticker and ("ticker" not in std_df.columns or std_df["ticker"].isna().all()):
        std_df["ticker"] = ticker

    std_df = std_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Engineer features (mirrors _engineer_features)
    std_df["profit_margin"] = (
        std_df["net_income"] / (std_df["revenue"] + eps)
    ) * 100
    std_df["operating_margin"] = (
        std_df["operating_income"] / (std_df["revenue"] + eps)
    ) * 100
    std_df["revenue_growth"] = (
        std_df.groupby("ticker")["revenue"].pct_change() * 100
    )
    std_df["net_income_growth"] = (
        std_df.groupby("ticker")["net_income"].pct_change() * 100
    )
    std_df["asset_efficiency"] = std_df["revenue"] / (std_df["total_assets"] + eps)
    std_df["debt_to_asset"] = (
        std_df["total_liabilities"] / (std_df["total_assets"] + eps)
    )

    std_df["user_id"] = user_id
    std_df = std_df.replace([np.inf, -np.inf], np.nan)

    # Select and order columns to match standard_table schema
    std_cols = [
        "date", "ticker", "revenue", "operating_income", "net_income",
        "operating_cashflow", "total_assets", "total_liabilities",
        "profit_margin", "operating_margin", "revenue_growth",
        "net_income_growth", "asset_efficiency", "debt_to_asset",
        "user_id",
    ]
    std_df = std_df[[c for c in std_cols if c in std_df.columns]]

    # ── Category table ────────────────────────────────────────────────────────
    cat_df = pd.DataFrame(category_records)

    # Ensure all required category_table columns exist
    if "date" not in cat_df.columns and "date" in std_df.columns:
        cat_df["date"] = std_df["date"].values

    cat_required = ["sector", "category", "risk_level", "revenue",
                    "operating_income", "net_income"]
    for col in cat_required:
        if col not in cat_df.columns:
            cat_df[col] = "Unknown" if col in ("sector", "category", "risk_level") else np.nan

    cat_df["date"] = pd.to_datetime(cat_df.get("date"), errors="coerce")
    cat_df["user_id"] = user_id

    cat_cols = [
        "ticker", "date", "sector", "category", "risk_level",
        "revenue", "operating_income", "net_income", "user_id",
    ]
    cat_df = cat_df[[c for c in cat_cols if c in cat_df.columns]]

    return {"standard_table": std_df, "category_table": cat_df}


if __name__ == "__main__":
    transform_data()