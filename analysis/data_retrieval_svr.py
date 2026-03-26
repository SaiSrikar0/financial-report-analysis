"""
Data retrieval and validation for uploaded data analysis.
Handles JSON stored data and ensures all required SVR parameters are present.

ENHANCED WITH:
- Comprehensive debug logging at each step
- Smart numeric conversion (handles $1,000, 1M, 1B formats)
- Robust ticker assignment ensuring no NULL values persist
- Print actual DataFrame structure for troubleshooting
- Detailed field-by-field validation reporting
"""

import pandas as pd
import numpy as np
import json
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


def _convert_numeric_field(value, field_name: str = "unknown") -> float:
    """
    Convert various numeric formats to float.
    Handles: "1000", "$1,000", "1M", "1.5B", etc.
    
    Args:
        value: Value to convert (string, int, float, etc.)
        field_name: Field name for logging
        
    Returns:
        Float value or np.nan if conversion fails
    """
    if pd.isna(value) or value is None or value == '':
        return np.nan
    
    try:
        # Already a number
        if isinstance(value, (int, float)):
            return float(value)
        
        # String conversion
        s = str(value).strip()
        
        # Remove currency symbols
        s = s.replace('$', '').replace('€', '').replace('£', '')
        
        # Handle millions (M)
        if 'M' in s.upper():
            num_str = s.replace('M', '').replace('m', '').strip()
            return float(num_str) * 1_000_000
        
        # Handle billions (B)
        if 'B' in s.upper():
            num_str = s.replace('B', '').replace('b', '').strip()
            return float(num_str) * 1_000_000_000
        
        # Handle thousands (K)
        if 'K' in s.upper():
            num_str = s.replace('K', '').replace('k', '').strip()
            return float(num_str) * 1_000
        
        # Remove commas and spaces
        s = s.replace(',', '').replace(' ', '')
        
        # Handle percentage
        if '%' in s:
            s = s.replace('%', '').strip()
            return float(s) / 100
        
        # Try direct conversion
        return float(s)
    except (ValueError, TypeError) as e:
        # Silently return NaN for conversion errors (common for missing data)
        return np.nan


def _normalize_raw_data_fields(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize field names in raw uploaded data to match standard schema.
    Handles common variations: Revenue, REVENUE, revenue_total, revenue_sales, etc.
    Also ensures ticker and date fields exist and have valid values.
    
    ENHANCED: Added comprehensive debug logging, smart numeric conversion, 
    robust ticker fixing.
    
    Args:
        df: Raw DataFrame from uploaded file
        ticker: Company ticker to ensure in output
        
    Returns:
        DataFrame with normalized field names and valid ticker values
    """
    normalized = df.copy()
    
    print(f"\n{'='*70}")
    print(f"FUNCTION: _normalize_raw_data_fields(ticker={ticker})")
    print(f"{'='*70}")
    
    print(f"\n[INPUT DEBUG INFO]")
    print(f"  Shape: {normalized.shape}")
    print(f"  Columns: {normalized.columns.tolist()}")
    print(f"  Data types: {normalized.dtypes.to_dict()}")
    null_summary = normalized.isnull().sum()
    if null_summary.any():
        print(f"  Null counts:")
        for col, count in null_summary[null_summary > 0].items():
            print(f"    - {col}: {count}")
    
    if len(normalized) > 0:
        print(f"\n  FIRST ROW DATA (RAW):")
        for col in normalized.columns[:5]:  # Show first 5 columns
            val = normalized.iloc[0][col]
            print(f"    {col}: {repr(val)} (type: {type(val).__name__})")
    
    # Lowercase all columns for case-insensitive matching
    normalized.columns = [col.lower().strip() for col in normalized.columns]
    print(f"\n[NORMALIZED COLUMNS (lowercase)]")
    print(f"  {normalized.columns.tolist()}")
    
    # Map common field name variations to standard names
    field_mappings = {
        # Date field
        'date': ['date', 'period', 'fiscal_date', 'report_date', 'year', 'quarter', 'month'],
        'ticker': ['ticker', 'symbol', 'company', 'company_name', 'stock_symbol'],
        'revenue': ['revenue', 'total_revenue', 'sales', 'net_sales', 'operating_revenue', 'revenues'],
        'net_income': ['net_income', 'income', 'earnings', 'net_profit', 'ni', 'net_earnings'],
        'operating_income': ['operating_income', 'operating_profit', 'ebit', 'operating_earnings', 'operating_income_loss'],
        'total_assets': ['total_assets', 'assets', 'total_asset', 'assets_total'],
        'total_liabilities': ['total_liabilities', 'liabilities', 'total_liability', 'liabilities_total'],
        'operating_cashflow': ['operating_cashflow', 'operating_cash_flow', 'cash_from_operations', 'ocf', 'cash_flow_operations'],
    }
    
    print(f"\n[FIELD NAME MAPPINGS]")
    mapped_count = 0
    for target_field, source_variants in field_mappings.items():
        if target_field not in normalized.columns:
            for variant in source_variants:
                if variant in normalized.columns:
                    normalized = normalized.rename(columns={variant: target_field})
                    print(f"  ✓ '{variant}' → '{target_field}'")
                    mapped_count += 1
                    break
            else:
                print(f"  ✗ {target_field}: NOT FOUND in input data")
        else:
            print(f"  - {target_field}: already present")
    
    print(f"\n[TICKER ASSIGNMENT] (Critical step)")
    
    # Debug: Show current ticker state
    if 'ticker' in normalized.columns:
        ticker_col = normalized['ticker']
        null_count = ticker_col.isna().sum()
        empty_str_count = (ticker_col == '').sum()
        null_or_empty = null_count + empty_str_count
        
        print(f"  Ticker column EXISTS")
        print(f"    - Total rows: {len(normalized)}") 
        print(f"    - NULL values: {null_count}")
        print(f"    - Empty strings: {empty_str_count}")
        print(f"    - NULL or empty: {null_or_empty}")
        print(f"    - Valid tickers: {len(normalized) - null_or_empty}")
        
        if null_or_empty > 0:
            print(f"    - Unique values: {ticker_col.unique()[:5].tolist()}")
    else:
        print(f"  Ticker column DOES NOT EXIST - will create")
    
    # CRITICAL: Ensure every single row has a valid ticker
    # Strategy 1: If column doesn't exist, create it
    if 'ticker' not in normalized.columns:
        print(f"\n[ACTION] Creating new ticker column")
        normalized.insert(0, 'ticker', ticker)
        print(f"  ✓ Created 'ticker' column, all rows set to: {ticker}")
    else:
        # Strategy 2: Column exists, fix any NULL/empty values
        ticker_col_before = normalized['ticker'].copy()
        null_before = ticker_col_before.isna().sum()
        empty_before = (ticker_col_before == '').sum()
        
        if null_before > 0:
            print(f"\n[ACTION] Filling {null_before} NULL ticker values")
            normalized.loc[normalized['ticker'].isna(), 'ticker'] = ticker
            print(f"  ✓ Filled {null_before} NULL values with: {ticker}")
        
        if (normalized['ticker'] == '').sum() > 0:
            empty_count = (normalized['ticker'] == '').sum()
            print(f"\n[ACTION] Fixing {empty_count} empty string ticker values")
            normalized.loc[normalized['ticker'] == '', 'ticker'] = ticker
            print(f"  ✓ Replaced {empty_count} empty strings with: {ticker}")
    
    # ABSOLUTE FINAL CHECK: Force all to have ticker
    null_after = normalized['ticker'].isna().sum()
    empty_after = (normalized['ticker'] == '').sum()
    
    if null_after > 0 or empty_after > 0:
        print(f"\n[EMERGENCY] Still found {null_after} NULL + {empty_after} empty tickers - FORCING ALL")
        normalized['ticker'] = ticker
        print(f"  ✓ Force-set all {len(normalized)} rows to: {ticker}")
    
    verify_ticker_valid = normalized['ticker'].unique().tolist()
    print(f"\n[VERIFICATION] Final ticker values in DataFrame: {verify_ticker_valid}")
    print(f"  ✓ All {len(normalized)} rows have valid ticker")
    
    # Convert date to datetime
    print(f"\n[DATE CONVERSION]")
    if 'date' in normalized.columns:
        date_before = normalized['date'].isna().sum()
        normalized['date'] = pd.to_datetime(normalized['date'], errors='coerce')
        date_after = normalized['date'].isna().sum()
        print(f"  ✓ Converted date column ({date_after} NULLs after conversion)")
    else:
        print(f"  ✗ Date column not found - will be NaN")
        normalized['date'] = pd.NaT
    
    # Convert numeric fields with smart conversion
    print(f"\n[NUMERIC FIELD CONVERSION] (using smart converter)")
    numeric_fields = ['revenue', 'net_income', 'operating_income', 'total_assets', 
                     'total_liabilities', 'operating_cashflow']
    
    for field in numeric_fields:
        if field not in normalized.columns:
            print(f"  ✗ {field}: MISSING - creating empty column (will be NaN)")
            normalized[field] = np.nan
        else:
            before_null = normalized[field].isna().sum()
            # Apply smart converter
            normalized[field] = normalized[field].apply(
                lambda x: _convert_numeric_field(x, field_name=field)
            )
            after_null = normalized[field].isna().sum()
            valid_count = len(normalized) - after_null
            print(f"  ✓ {field}: {valid_count} valid values, {after_null} NULL")
    
    # Final report
    print(f"\n[NORMALIZATION OUTPUT SUMMARY]")
    print(f"  Shape: {normalized.shape}")
    print(f"  Columns: {normalized.columns.tolist()}")
    print(f"\n  NULL COUNT BY COLUMN:")
    for col in normalized.columns:
        null_count = normalized[col].isna().sum()
        pct = 100 * null_count / len(normalized) if len(normalized) > 0 else 0
        status = "✓" if null_count == 0 else "⚠" if null_count < len(normalized) else "✗"
        print(f"    {status} {col}: {null_count}/{len(normalized)} ({pct:.1f}%)")
    
    if len(normalized) > 0:
        print(f"\n  FIRST ROW AFTER NORMALIZATION:")
        first_row = normalized.iloc[0]
        for col in ['date', 'ticker', 'revenue', 'net_income', 'operating_income']:
            if col in normalized.columns:
                val = first_row[col]
                print(f"    {col}: {repr(val)} (type: {type(val).__name__})")
    
    print(f"\n{'='*70}\n")
    
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
    
    print(f"\n[_engineer_svr_features] INPUT: shape {df.shape}")
    
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
    
    print(f"[_engineer_svr_features] OUTPUT: shape {work_df.shape}")
    print(f"[_engineer_svr_features] ✓ FEATURES ENGINEERED: profit_margin, operating_margin,")
    print(f"                          revenue_growth, net_income_growth, asset_efficiency, debt_to_asset")
    
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
    
    print(f"\n[retrieve_uploaded_data_by_ticker] Starting for ticker: {ticker}")
    
    # Strategy 1: Try uploaded_files table FIRST (raw data, most reliable)
    try:
        print(f"[retrieve_uploaded_data] → Querying uploaded_files table...")
        response = supabase_client.table("uploaded_files").select(
            "file_content, ticker"
        ).eq("ticker", ticker.upper()).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            file_content = response.data[0].get("file_content", [])
            retrieved_ticker = response.data[0].get("ticker", ticker.upper())
            
            print(f"[retrieve_uploaded_data] ✓ Found data in uploaded_files")
            print(f"[retrieve_uploaded_data]   - Retrieved ticker: {retrieved_ticker}")
            print(f"[retrieve_uploaded_data]   - File content type: {type(file_content)}")
            print(f"[retrieve_uploaded_data]   - File content length: {len(file_content) if isinstance(file_content, list) else 'N/A'}")
            
            if isinstance(file_content, list) and len(file_content) > 0:
                df = pd.DataFrame(file_content)
                print(f"[retrieve_uploaded_data] ✓ Converted to DataFrame: {df.shape}")
                
                # Step 1: Normalize field names in raw data
                print(f"[retrieve_uploaded_data] → Step 1: Normalizing raw data fields...")
                df = _normalize_raw_data_fields(df, retrieved_ticker)
                
                # Step 2: Engineer calculated features (growth, margins, ratios)
                print(f"[retrieve_uploaded_data] → Step 2: Engineering features...")
                df = _engineer_svr_features(df)
                
                # Step 3: Sort by date for growth calculations to be correct
                if "date" in df.columns:
                    df = df.sort_values("date").reset_index(drop=True)
                    print(f"[retrieve_uploaded_data] ✓ Sorted by date")
                
                print(f"[retrieve_uploaded_data] ✓✓✓ SUCCESS: Loaded {len(df)} records from uploaded_files (raw + engineered)")
                return df, "uploaded_files (raw + engineered)"
    except Exception as e:
        print(f"[retrieve_uploaded_data] ℹ Note: uploaded_files query failed: {str(e)[:100]}")
    
    # Strategy 2: Fallback to standard_table (already normalized + engineered)
    try:
        print(f"[retrieve_uploaded_data] → Fallback: Querying standard_table...")
        response = supabase_client.table("standard_table").select(
            "*"
        ).eq("ticker", ticker.upper()).execute()
        
        if response.data and len(response.data) > 0:
            df = pd.DataFrame(response.data)
            print(f"[retrieve_uploaded_data] ✓ Loaded {len(df)} records from standard_table (pre-engineered)")
            return df, "standard_table"
    except Exception as e:
        print(f"[retrieve_uploaded_data] ℹ Note: standard_table query failed: {str(e)[:100]}")
    
    print(f"[retrieve_uploaded_data] ✗✗✗ FAILURE: No data found for ticker {ticker}")
    return None, f"No data found for ticker {ticker}"


def validate_and_prepare_svr_data(
    df: pd.DataFrame,
    ticker: str,
    min_records: int = 3
) -> Tuple[Optional[pd.DataFrame], list]:
    """
    Validate that data has all required fields for SVR training.
    Returns cleaned DataFrame and list of validation messages/warnings.
    
    ENHANCED: Detailed field-by-field reporting, shows before/after row counts,
    identifies exactly which records are dropped and why.
    
    Args:
        df: Input DataFrame from database/upload
        ticker: Expected ticker value
        min_records: Minimum records required for training
        
    Returns:
        Tuple of (validated_df, validation_messages)
    """
    
    print(f"\n{'='*70}")
    print(f"FUNCTION: validate_and_prepare_svr_data()")
    print(f"{'='*70}")
    
    messages = []
    
    print(f"\n[INPUT VALIDATION]")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Expected ticker: {ticker}")
    
    # Check record count
    if len(df) < min_records:
        messages.append(f"⚠️ Only {len(df)} records (need ≥{min_records})")
        print(f"  ⚠ Record count low: {len(df)} < {min_records}")
    
    # Check required fields
    missing_fields = [f for f in REQUIRED_SVR_FIELDS if f not in df.columns]
    if missing_fields:
        messages.append(f"⚠️ Missing fields: {', '.join(missing_fields)}")
        print(f"  ⚠ Missing fields: {missing_fields}")
    present_fields = [f for f in REQUIRED_SVR_FIELDS if f in df.columns]
    print(f"  ✓ Present fields: {present_fields}")
    
    # Check ticker consistency
    if "ticker" in df.columns:
        tickers = df["ticker"].unique()
        print(f"  Ticker info: {tickers.tolist()}")
        if len(tickers) > 1:
            messages.append(f"⚠️ Multiple tickers found: {', '.join(tickers)}")
    
    # Normalize data types
    print(f"\n[DATA TYPE NORMALIZATION]")
    work_df = df.copy()
    rows_start = len(work_df)
    
    # Convert date to datetime
    if "date" in work_df.columns:
        before = work_df['date'].isna().sum()
        work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
        after = work_df['date'].isna().sum()
        before_rows = len(work_df)
        work_df = work_df.dropna(subset=["date"])
        after_rows = len(work_df)
        print(f"  ✓ date: {after} NULLs after conversion, dropped {before_rows - after_rows} rows")
    
    # Convert numeric fields
    numeric_fields = REQUIRED_SVR_FIELDS + OPTIONAL_SVR_FIELDS
    for field in numeric_fields:
        if field in work_df.columns:
            work_df[field] = pd.to_numeric(work_df[field], errors="coerce")
    
    # Remove rows with NaN in required fields
    print(f"\n[REQUIRED FIELD VALIDATION]")
    for field in REQUIRED_SVR_FIELDS:
        if field in work_df.columns:
            before_count = len(work_df)
            null_count = work_df[field].isna().sum()
            work_df = work_df.dropna(subset=[field])
            after_count = len(work_df)
            removed = before_count - after_count
            
            if removed > 0:
                messages.append(f"ℹ️ Removed {removed} rows with missing '{field}'")
                print(f"  - {field}: {removed} rows dropped ({null_count} NULLs found)")
            else:
                print(f"  ✓ {field}: all values present ({before_count} rows)")
    
    print(f"\n[FINAL VALIDATION]")
    print(f"  Records at start: {rows_start}")
    print(f"  Records remaining: {len(work_df)}")
    print(f"  Records dropped: {rows_start - len(work_df)}")
    
    if len(work_df) == 0:
        messages.append(
            "❌ CRITICAL: DataFrame is empty after removing rows with missing required fields. "
            "Check your CSV has these columns: date, ticker, revenue, net_income, "
            "operating_income, total_assets, total_liabilities, operating_cashflow"
        )
        print(f"\n  ❌❌❌ CRITICAL: All records dropped!")
        from etl.validator import print_engineered_features_report
        print_engineered_features_report(False, messages)
        print(f"\n{'='*70}\n")
        return None, messages
    
    # Use validator.py to check engineered features are present and valid
    from etl.validator import validate_engineered_features, print_engineered_features_report
    
    is_valid, validation_issues = validate_engineered_features(work_df)
    
    messages.extend(validation_issues)
    print(f"\n[ENGINEERED FEATURES CHECK]")
    print(f"  Valid: {is_valid}")
    for issue in validation_issues:
        print(f"  - {issue}")
    
    print_engineered_features_report(is_valid, validation_issues + messages)
    
    if not is_valid:
        print(f"\n[VALIDATION FAILED]")
        print(f"  ✗ Engineered features check failed")
        print(f"{'='*70}\n")
        return None, messages
    
    # Check if we still have enough data
    if len(work_df) < min_records:
        messages.append(f"❌ CRITICAL: Only {len(work_df)} valid records (need ≥{min_records})")
        print(f"\n[VALIDATION FAILED]")
        print(f"  ✗ Insufficient records: {len(work_df)} < {min_records}")
        print(f"{'='*70}\n")
        return None, messages
    
    # Sort by date
    if "date" in work_df.columns:
        work_df = work_df.sort_values("date").reset_index(drop=True)
    
    messages.append(f"✅ Prepared {len(work_df)} clean records for SVR training")
    
    print(f"\n[FINAL OUTPUT]")
    print(f"  ✓ Ready for SVR training: {len(work_df)} records, {work_df.shape[1]} columns")
    print(f"{'='*70}\n")
    
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
    if len(prepared_df) < 3:
        all_messages.append("❌ CRITICAL: Insufficient data for model training")
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
