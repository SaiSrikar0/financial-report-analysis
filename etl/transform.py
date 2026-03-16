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
        "revenue",
        "operating_income",
        "net_income",
        "total_assets",
        "total_liabilities",
        "operating_cashflow",
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

    # Keep percentage-based margins and growth to preserve downstream compatibility.
    out["profit_margin"] = (out["net_income"] / (out["revenue"] + eps)) * 100
    out["operating_margin"] = (out["operating_income"] / (out["revenue"] + eps)) * 100
    out["revenue_growth"] = out.groupby("ticker")["revenue"].pct_change() * 100
    out["net_income_growth"] = out.groupby("ticker")["net_income"].pct_change() * 100
    out["asset_efficiency"] = out["revenue"] / (out["total_assets"] + eps)
    out["debt_to_asset"] = out["total_liabilities"] / (out["total_assets"] + eps)

    return out


def _build_standard_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "ticker",
        "revenue",
        "operating_income",
        "net_income",
        "operating_cashflow",
        "total_assets",
        "total_liabilities",
        "profit_margin",
        "operating_margin",
        "revenue_growth",
        "net_income_growth",
        "asset_efficiency",
        "debt_to_asset",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def _build_category_table(df: pd.DataFrame) -> pd.DataFrame:
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
    }

    cat = df[["ticker", "date", "revenue", "operating_income", "net_income", "revenue_growth", "debt_to_asset"]].copy()
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

    cat = cat[["ticker", "date", "sector", "category", "risk_level", "revenue", "operating_income", "net_income"]]
    return cat


def _sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.replace([np.inf, -np.inf], np.nan)
    return clean_df


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

    print(f"Standard table (for ML/SVR) saved to {standard_path}")
    print(f"Category table (for LLM) saved to {category_path}")

    return standard_path, category_path


if __name__ == "__main__":
    transform_data()