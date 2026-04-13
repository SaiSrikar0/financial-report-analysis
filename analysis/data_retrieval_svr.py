"""
Data retrieval and validation for uploaded data analysis.
Handles JSON stored data and ensures all required SVR parameters are present.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Tuple, Optional
import streamlit as st


# Required financial fields for SVR model training
REQUIRED_SVR_FIELDS = [
    "date",
    "ticker",
    "revenue",
    "net_income",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "operating_cashflow",
]

OPTIONAL_SVR_FIELDS = [
    "profit_margin",
    "operating_margin",
    "revenue_growth",
    "net_income_growth",
    "asset_efficiency",
    "debt_to_asset",
]


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """
    Robust numeric coercion for messy financial uploads.
    Handles: $1,234, (123), 12%, 1.2B, 450M, 23K, textual separators.
    """
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan}, regex=False)

    # (1234) -> -1234
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    # Remove currency symbols, commas and whitespace inside numbers
    s = s.str.replace(r"[,$€£₹\s]", "", regex=True)

    # Preserve percentage as numeric value (e.g., 12% -> 12)
    s = s.str.replace("%", "", regex=False)

    multipliers = pd.Series(1.0, index=s.index)
    suffix_map = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    suffix = s.str.extract(r"([KMBTkmbt])$", expand=False).str.upper()
    for key, mult in suffix_map.items():
        multipliers = multipliers.where(suffix != key, mult)

    s = s.str.replace(r"[KMBTkmbt]$", "", regex=True)
    num = pd.to_numeric(s, errors="coerce")
    return num * multipliers


def _coerce_date_series(series: pd.Series) -> pd.Series:
    """Parse common date variants, including year-only and quarter formats."""
    s = series.astype(str).str.strip()

    # Quarter formats: 2024Q1, Q1-2024, 2024-Q1
    q1 = s.str.extract(r"^(\d{4})[-\s]?Q([1-4])$", expand=True)
    q2 = s.str.extract(r"^Q([1-4])[-\s]?(\d{4})$", expand=True)

    quarter_map = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}
    out = pd.to_datetime(s, errors="coerce")

    q1_mask = q1[0].notna() & q1[1].notna() & out.isna()
    if q1_mask.any():
        q_dates = q1.loc[q1_mask].apply(lambda r: f"{r[0]}-{quarter_map.get(r[1], '12-31')}", axis=1)
        out.loc[q1_mask] = pd.to_datetime(q_dates, errors="coerce")

    q2_mask = q2[0].notna() & q2[1].notna() & out.isna()
    if q2_mask.any():
        q_dates = q2.loc[q2_mask].apply(lambda r: f"{r[1]}-{quarter_map.get(r[0], '12-31')}", axis=1)
        out.loc[q2_mask] = pd.to_datetime(q_dates, errors="coerce")

    # Year-only: 2024 -> 2024-12-31
    year_only = s.str.extract(r"^(\d{4})$", expand=False)
    y_mask = year_only.notna() & out.isna()
    if y_mask.any():
        out.loc[y_mask] = pd.to_datetime(year_only.loc[y_mask] + "-12-31", errors="coerce")

    return out


def _normalize_raw_data_fields(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize field names in raw uploaded data to match standard schema.
    Handles common variations: Revenue, REVENUE, revenue_total, revenue_sales, etc.
    Also ensures ticker and date fields exist and have valid values.
    
    Args:
        df: Raw DataFrame from uploaded file
        ticker: Company ticker to ensure in output
        
    Returns:
        DataFrame with normalized field names and valid ticker values
    """
    normalized = df.copy()
    
    print(f"[_normalize_raw_data_fields] ✓ Input shape: {normalized.shape}")
    print(f"[_normalize_raw_data_fields] ✓ Input columns: {normalized.columns.tolist()}")
    
    # Lowercase all columns for case-insensitive matching
    normalized.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", col.lower().strip())).strip("_")
        for col in normalized.columns
    ]
    
    # Map common field name variations to standard names
    field_mappings = {
        # Date field
        'date': ['date', 'period', 'fiscal_date', 'report_date', 'year', 'quarter', 'fiscal_year'],
        'ticker': ['ticker', 'symbol', 'company', 'company_name', 'stock', 'stock_symbol'],
        'revenue': ['revenue', 'total_revenue', 'sales', 'net_sales', 'operating_revenue', 'turnover'],
        'net_income': ['net_income', 'income', 'earnings', 'net_profit', 'ni', 'profit_after_tax', 'pat'],
        'operating_income': ['operating_income', 'operating_profit', 'ebit', 'operating_earnings', 'op_income'],
        'total_assets': ['total_assets', 'assets', 'total_asset', 'asset_total'],
        'total_liabilities': ['total_liabilities', 'liabilities', 'total_liability', 'liability_total'],
        'operating_cashflow': ['operating_cashflow', 'operating_cash_flow', 'cash_from_operations', 'ocf', 'cashflow_from_operations'],
    }
    
    for target_field, source_variants in field_mappings.items():
        if target_field not in normalized.columns:
            for variant in source_variants:
                if variant in normalized.columns:
                    normalized = normalized.rename(columns={variant: target_field})
                    print(f"[_normalize_raw_data_fields] → Mapped '{variant}' → '{target_field}'")
                    break
    
    # CRITICAL: Ensure every single row has a valid ticker
    # First check if ticker column exists
    if 'ticker' not in normalized.columns:
        print(f"[_normalize_raw_data_fields] ⚠️ Ticker column not found - creating from parameter: {ticker}")
        normalized.insert(0, 'ticker', ticker)
    else:
        # Column exists - fill ANY null values
        null_count = normalized['ticker'].isna().sum()
        if null_count > 0:
            print(f"[_normalize_raw_data_fields] ⚠️ Found {null_count} NULL tickers - filling from parameter: {ticker}")
            normalized['ticker'] = normalized['ticker'].fillna(ticker)
        
        # Also check for empty strings
        if (normalized['ticker'] == '').any():
            print(f"[_normalize_raw_data_fields] ⚠️ Found empty string tickers - replacing with: {ticker}")
            normalized.loc[normalized['ticker'] == '', 'ticker'] = ticker
    
    # Final verification
    null_after = normalized['ticker'].isna().sum()
    if null_after > 0:
        print(f"[_normalize_raw_data_fields] ✗ CRITICAL: Still have {null_after} NULL tickers after filling!")
        print(f"[_normalize_raw_data_fields] ✗ Setting all tickers to: {ticker}")
        normalized['ticker'] = ticker
    else:
        print(f"[_normalize_raw_data_fields] ✓ All {len(normalized)} rows have valid ticker: {ticker}")
    
    # Ensure date is datetime
    if 'date' in normalized.columns:
        normalized['date'] = _coerce_date_series(normalized['date'])
    
    # Convert numeric fields
    numeric_fields = ['revenue', 'net_income', 'operating_income', 'total_assets', 
                     'total_liabilities', 'operating_cashflow']
    for field in numeric_fields:
        if field in normalized.columns:
            normalized[field] = _coerce_numeric_series(normalized[field])
    
    print(f"[_normalize_raw_data_fields] ✓ Output shape: {normalized.shape}")
    print(f"[_normalize_raw_data_fields] ✓ Output columns: {normalized.columns.tolist()}")
    
    return normalized


def _engineer_svr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer calculated features needed for SVR training.
    Mirrors the feature engineering from transform_dynamic().
    
    These features are often more predictive than raw values for ML models.
    
    Args:
        df: DataFrame with required raw fields (must have date, ticker, revenue, etc.)
        
    Returns:
        DataFrame with engineered features added
    """
    
    work_df = df.copy()
    eps = 1e-9  # Small constant to avoid division by zero
    
    # Financial ratios
    work_df["profit_margin"] = (
        work_df["net_income"] / (work_df["revenue"] + eps)
    ) * 100
    
    work_df["operating_margin"] = (
        work_df["operating_income"] / (work_df["revenue"] + eps)
    ) * 100
    
    # Growth rates (groupby ticker to handle multiple periods)
    work_df["revenue_growth"] = (
        work_df.groupby("ticker")["revenue"].pct_change() * 100
    )
    
    work_df["net_income_growth"] = (
        work_df.groupby("ticker")["net_income"].pct_change() * 100
    )
    
    # Efficiency and leverage ratios
    work_df["asset_efficiency"] = (
        work_df["revenue"] / (work_df["total_assets"] + eps)
    )
    
    work_df["debt_to_asset"] = (
        work_df["total_liabilities"] / (work_df["total_assets"] + eps)
    )
    
    # Replace infinite values with NaN
    work_df = work_df.replace([np.inf, -np.inf], np.nan)
    
    return work_df


def retrieve_uploaded_data_by_ticker(
    ticker: str, 
    user_id: str,
    supabase_client
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Retrieve uploaded data for a specific ticker from database.
    PRIORITIZES raw uploaded_files data over standard_table to avoid ticker NULL issues.
    Normalizes field names to match standard schema.
    ENGINEERS calculated features (growth rates, margins, ratios).
    
    Args:
        ticker: Company ticker
        user_id: Authenticated user ID
        supabase_client: Supabase client with user session set
        
    Returns:
        Tuple of (DataFrame, source) or (None, error_message)
    """
    
    # Strategy 1: Try uploaded_files table FIRST (raw data, most reliable)
    try:
        response = supabase_client.table("uploaded_files").select(
            "file_content, ticker"
        ).eq("user_id", user_id).limit(100).execute()
        
        matched_rows = []
        if response.data and len(response.data) > 0:
            matched_rows = [
                r for r in response.data
                if str(r.get("ticker", "")).upper() == ticker.upper()
            ]

        if matched_rows:
            file_content = matched_rows[0].get("file_content", [])
            if isinstance(file_content, list) and len(file_content) > 0:
                df = pd.DataFrame(file_content)
                # Step 1: Normalize field names in raw data
                df = _normalize_raw_data_fields(df, ticker.upper())
                # Step 2: Engineer calculated features (growth, margins, ratios)
                df = _engineer_svr_features(df)
                # Step 3: Sort by date for growth calculations to be correct
                if "date" in df.columns:
                    df = df.sort_values("date").reset_index(drop=True)
                print(f"[retrieve_uploaded_data] ✓ Loaded {len(df)} records from uploaded_files (raw)")
                print(f"[retrieve_uploaded_data] ✓ Engineered features: profit_margin, operating_margin, revenue_growth, net_income_growth, asset_efficiency, debt_to_asset")
                return df, "uploaded_files (raw + engineered)"
    except Exception as e:
        print(f"[retrieve_uploaded_data] Note: uploaded_files query failed: {e}")
    
    # Strategy 2: Fallback to standard_table (already normalized + engineered)
    try:
        response = supabase_client.table("standard_table").select(
            "*"
        ).eq("ticker", ticker.upper()).execute()
        
        if response.data and len(response.data) > 0:
            df = pd.DataFrame(response.data)
            print(f"[retrieve_uploaded_data] ✓ Loaded {len(df)} records from standard_table (pre-engineered)")
            return df, "standard_table"
    except Exception as e:
        print(f"[retrieve_uploaded_data] Note: standard_table query failed: {e}")
    
    return None, f"No data found for ticker {ticker}"


def validate_and_prepare_svr_data(
    df: pd.DataFrame,
    ticker: str,
    min_records: int = 2
) -> Tuple[pd.DataFrame, list]:
    """
    Validate that data has all required fields for SVR training.
    Returns cleaned DataFrame and list of validation messages/warnings.
    
    Args:
        df: Input DataFrame from database/upload
        ticker: Expected ticker value
        min_records: Minimum records required for training
        
    Returns:
        Tuple of (validated_df, validation_messages)
    """
    
    messages = []
    
    # Check record count
    if len(df) < min_records:
        messages.append(f"⚠️ Only {len(df)} records (need ≥{min_records})")
    
    # Check required fields
    missing_fields = [f for f in REQUIRED_SVR_FIELDS if f not in df.columns]
    if missing_fields:
        messages.append(f"⚠️ Missing fields: {', '.join(missing_fields)}")
    
    # Check ticker consistency
    if "ticker" in df.columns:
        tickers = df["ticker"].unique()
        if len(tickers) > 1:
            messages.append(f"⚠️ Multiple tickers found: {', '.join(tickers)}")
    
    # Normalize data types
    work_df = df.copy()
    
    # Convert date to datetime
    if "date" in work_df.columns:
        work_df["date"] = _coerce_date_series(work_df["date"])
        work_df = work_df.dropna(subset=["date"])
    
    # Convert numeric financial fields only (never coerce ticker/date to numeric)
    numeric_fields = [
        "revenue",
        "net_income",
        "operating_income",
        "total_assets",
        "total_liabilities",
        "operating_cashflow",
    ] + OPTIONAL_SVR_FIELDS

    for field in numeric_fields:
        if field in work_df.columns:
            work_df[field] = _coerce_numeric_series(work_df[field])

    # Ensure ticker strings are preserved/normalized
    if "ticker" in work_df.columns:
        work_df["ticker"] = work_df["ticker"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    
    # Remove rows with NaN in required fields
    for field in REQUIRED_SVR_FIELDS:
        if field in work_df.columns:
            initial_count = len(work_df)
            work_df = work_df.dropna(subset=[field])
            removed = initial_count - len(work_df)
            if removed > 0:
                messages.append(f"ℹ️ Removed {removed} rows with missing '{field}'")
    
    if len(work_df) == 0:
        messages.append(
            "❌ CRITICAL: DataFrame is empty after removing rows with missing required fields. "
            "Check your CSV has these columns: date, ticker, revenue, net_income, "
            "operating_income, total_assets, total_liabilities, operating_cashflow"
        )
        from etl.validator import print_engineered_features_report
        print_engineered_features_report(False, messages)
        return None, messages
    
    # Use validator.py to check engineered features are present and valid
    from etl.validator import validate_engineered_features, print_engineered_features_report
    
    is_valid, validation_issues = validate_engineered_features(work_df)
    messages.extend(validation_issues)
    
    if not is_valid:
        print_engineered_features_report(is_valid, validation_issues)
        return None, messages
    
    # Check if we still have enough data
    if len(work_df) < min_records:
        messages.append(f"❌ CRITICAL: Only {len(work_df)} valid records (need ≥{min_records})")
        return None, messages
    
    # Sort by date
    if "date" in work_df.columns:
        work_df = work_df.sort_values("date").reset_index(drop=True)
    
    messages.append(f"✅ Prepared {len(work_df)} clean records for SVR training")
    print_engineered_features_report(True, messages)
    
    return work_df, messages


def load_and_validate_training_data(
    ticker: str,
    user_id: str,
    supabase_client,
    standard_records: Optional[list] = None,
    category_records: Optional[list] = None,
) -> Tuple[Optional[pd.DataFrame], list]:
    """
    Complete data loading and validation pipeline for SVR training.
    Intelligently handles both uploaded and normalized data.
    
    Args:
        ticker: Company ticker
        user_id: User ID
        supabase_client: Supabase client
        standard_records: Optional pre-extracted standard records (from upload flow)
        category_records: Optional pre-extracted category records
        
    Returns:
        Tuple of (prepared_df, validation_messages)
    """
    
    all_messages = []
    
    # If records provided (from upload flow), use them directly
    if standard_records and len(standard_records) > 0:
        all_messages.append("ℹ️ Using provided uploaded records")
        df = pd.DataFrame(standard_records)
        if "ticker" not in df.columns:
            df["ticker"] = ticker
        else:
            df["ticker"] = df["ticker"].fillna(ticker)

        # Coerce core types for upload-flow records (these may not include
        # engineered fields yet because transform_dynamic is a separate step).
        if "date" in df.columns:
            df["date"] = _coerce_date_series(df["date"])

        numeric_fields = [
            "revenue",
            "net_income",
            "operating_income",
            "total_assets",
            "total_liabilities",
            "operating_cashflow",
        ]
        for field in numeric_fields:
            if field in df.columns:
                df[field] = _coerce_numeric_series(df[field])

        missing_engineered = [f for f in OPTIONAL_SVR_FIELDS if f not in df.columns]
        if missing_engineered:
            all_messages.append("ℹ️ Engineering SVR features from uploaded records")
            df = _engineer_svr_features(df)

        df = df.replace([np.inf, -np.inf], np.nan)
    else:
        # Otherwise retrieve from database
        all_messages.append("→ Retrieving from database...")
        df, source_or_error = retrieve_uploaded_data_by_ticker(
            ticker, user_id, supabase_client
        )
        
        if df is None:
            all_messages.append(f"❌ {source_or_error}")
            return None, all_messages
        
        all_messages.append(f"✓ Source: {source_or_error}")
    
    # Validate and prepare
    all_messages.append("→ Validating data completeness...")
    prepared_df, validation_msgs = validate_and_prepare_svr_data(df, ticker)
    all_messages.extend(validation_msgs)
    
    if prepared_df is None:
        return None, all_messages
    
    # Final checks
    if len(prepared_df) < 2:
        all_messages.append("❌ CRITICAL: Insufficient data for model training (need at least 2 periods)")
        return None, all_messages
    
    all_messages.append(f"✅ Data Ready: {len(prepared_df)} records with all required fields")
    
    return prepared_df, all_messages


def display_validation_report(messages: list, container=None):
    """Display validation messages in Streamlit."""
    if container is None:
        container = st
    
    for msg in messages:
        if "✓" in msg or "✅" in msg:
            container.success(msg)
        elif "❌" in msg or "CRITICAL" in msg:
            container.error(msg)
        elif "⚠️" in msg:
            container.warning(msg)
        elif "ℹ️" in msg or "→" in msg:
            container.info(msg)
        else:
            container.write(msg)
