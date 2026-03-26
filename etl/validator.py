"""
Validation gate — runs before the ML pipeline to catch sparse or broken data.
Prevents silent SVR failures on user-uploaded datasets.
"""

from typing import List, Dict, Tuple
import pandas as pd

# Fields that absolutely must be present for the SVR pipeline to run
REQUIRED_FOR_ML = ["revenue", "net_income", "total_assets", "date", "ticker"]

# Fields used in analysis modules; missing ones trigger warnings, not hard failures
REQUIRED_FOR_ANALYSIS = ["operating_income", "total_liabilities", "operating_cashflow"]

# Engineered features that SVR model expects (calculated during feature engineering)
ENGINEERED_FEATURES = [
    "profit_margin",
    "operating_margin", 
    "revenue_growth",
    "net_income_growth",
    "asset_efficiency",
    "debt_to_asset"
]

MIN_RECORDS = 3  # Minimum years/periods needed for trend analysis
MAX_NULL_PCT_FOR_ENGINEERED = 0.5  # Allow up to 50% NULL for growth features (first period has no growth)


def validate(standard_records: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Returns (is_valid, list_of_issues).
    Issues prefixed with 'CRITICAL' cause is_valid=False.
    """
    issues = []

    if len(standard_records) < MIN_RECORDS:
        issues.append(
            f"CRITICAL: Only {len(standard_records)} records found. "
            f"Need at least {MIN_RECORDS} periods for trend analysis."
        )

    if not standard_records:
        return False, issues

    # Check null rates for required fields
    total = len(standard_records)
    for field in REQUIRED_FOR_ML:
        null_count = sum(1 for r in standard_records if r.get(field) is None)
        null_pct = null_count / total
        if null_pct > 0.3:
            issues.append(
                f"CRITICAL: '{field}' is null in {null_pct:.0%} of records — "
                f"SVR pipeline will fail."
            )

    for field in REQUIRED_FOR_ANALYSIS:
        null_count = sum(1 for r in standard_records if r.get(field) is None)
        null_pct = null_count / total
        if null_pct > 0.5:
            issues.append(
                f"WARNING: '{field}' is null in {null_pct:.0%} of records — "
                f"some analysis modules will be skipped."
            )

    # Check for duplicate dates per ticker
    from collections import Counter
    date_keys = [
        (r.get("ticker"), r.get("date"))
        for r in standard_records
        if r.get("date")
    ]
    duplicates = [k for k, v in Counter(date_keys).items() if v > 1]
    if duplicates:
        issues.append(
            f"WARNING: {len(duplicates)} duplicate (ticker, date) pairs detected. "
            f"Records may have been doubled during LLM extraction."
        )

    # Check revenue values are plausible (not all zero)
    revenues = [r.get("revenue") for r in standard_records if r.get("revenue")]
    if revenues and max(revenues) == 0:
        issues.append(
            "CRITICAL: All revenue values are zero — data extraction likely failed."
        )

    is_valid = not any(i.startswith("CRITICAL") for i in issues)
    return is_valid, issues


def print_validation_report(is_valid: bool, issues: List[str]):
    if is_valid and not issues:
        print("[Validator] ✅ Data passed validation. Proceeding to analysis pipeline.")
        return
    if is_valid:
        print("[Validator] ✅ Passed with warnings:")
    else:
        print("[Validator] ❌ Validation failed:")
    for issue in issues:
        prefix = "  ❌" if issue.startswith("CRITICAL") else "  ⚠️ "
        print(f"{prefix} {issue}")


def validate_engineered_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that engineered/calculated features are present and not excessively NULL.
    Called AFTER feature engineering during SVR data preparation.
    
    Args:
        df: DataFrame with raw fields + engineered features
        
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    if df is None:
        issues.append("CRITICAL: DataFrame is None — data retrieval failed completely.")
        return False, issues
    
    total = len(df)
    
    if total == 0:
        issues.append(
            "CRITICAL: DataFrame is empty after data retrieval and cleaning. "
            "This usually means all rows were removed due to missing required fields. "
            "Check that your uploaded data has: date, ticker, revenue, net_income, "
            "operating_income, total_assets, total_liabilities, operating_cashflow"
        )
        return False, issues
    
    # Check required raw fields are present
    missing_raw = [f for f in REQUIRED_FOR_ML if f not in df.columns]
    if missing_raw:
        issues.append(
            f"CRITICAL: Missing required fields for feature engineering: {', '.join(missing_raw)}. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
    
    # Check engineered features are present
    missing_engineered = [f for f in ENGINEERED_FEATURES if f not in df.columns]
    if missing_engineered:
        issues.append(
            f"CRITICAL: Missing engineered features: {', '.join(missing_engineered)}. "
            f"Feature engineering pipeline may have failed."
        )
    
    # Check null rates for required raw fields first (before engineered)
    for field in REQUIRED_FOR_ML:
        if field in df.columns:
            null_count = df[field].isna().sum()
            null_pct = null_count / total
            if null_count > 0:
                issues.append(
                    f"ℹ️ Field '{field}': {null_count}/{total} rows have NULL ({null_pct:.0%})"
                )
    
    # Check null rates for engineered features
    for field in ENGINEERED_FEATURES:
        if field in df.columns:
            null_count = df[field].isna().sum()
            null_pct = null_count / total
            
            # Growth features can have ~50% NULL (first period has no prior value to compare)
            # Other features should be mostly populated
            if field in ["revenue_growth", "net_income_growth"]:
                max_allowed = MAX_NULL_PCT_FOR_ENGINEERED
                if null_pct > max_allowed:
                    issues.append(
                        f"CRITICAL: '{field}' is null in {null_pct:.0%} of records "
                        f"(max allowed: {max_allowed:.0%} for growth features). SVR cannot train."
                    )
                elif null_pct > 0:
                    issues.append(
                        f"ℹ️ Growth feature '{field}': {null_count}/{total} rows NULL ({null_pct:.0%}) - normal for first period"
                    )
            else:
                max_allowed = 0.3
                if null_pct > max_allowed:
                    issues.append(
                        f"CRITICAL: '{field}' is null in {null_pct:.0%} of records "
                        f"(max allowed: {max_allowed:.0%}). SVR cannot train."
                    )
                elif null_pct > 0:
                    issues.append(
                        f"ℹ️ Feature '{field}': {null_count}/{total} rows NULL ({null_pct:.0%})"
                    )
    
    # Check that after removing NULLs from required fields, we still have enough records
    required_fields = REQUIRED_FOR_ML + [f for f in ENGINEERED_FEATURES if f in df.columns]
    clean_df = df.dropna(subset=required_fields)
    clean_count = len(clean_df)
    
    if clean_count < MIN_RECORDS:
        issues.append(
            f"CRITICAL: After removing rows with any NULL values, "
            f"only {clean_count} records remain (need ≥{MIN_RECORDS}). "
            f"Cannot train SVR model. Your data may not have enough periods."
        )
    elif clean_count < total:
        # Just a warning if we lose some records but still have enough
        dropped = total - clean_count
        issues.append(
            f"✓ Data ready: Using {clean_count} clean records for SVR training "
            f"({dropped} rows with NULL values will be skipped)"
        )
    else:
        issues.append(
            f"✓ All {clean_count} records are complete with no NULL values"
        )
    
    is_valid = not any(i.startswith("CRITICAL") for i in issues)
    return is_valid, issues


def print_engineered_features_report(is_valid: bool, issues: List[str]):
    """Pretty-print validation results for engineered features."""
    if is_valid and not issues:
        print("[Validator] ✅ Engineered features validated. Ready for SVR training.")
        return
    if is_valid:
        print("[Validator] ✅ Features validated with notes:")
    else:
        print("[Validator] ❌ Feature validation failed:")
    for issue in issues:
        if issue.startswith("CRITICAL"):
            prefix = "  ❌"
        elif issue.startswith("ℹ️"):
            prefix = "  ℹ️"
        else:
            prefix = "  ⚠️"
        print(f"{prefix} {issue}")