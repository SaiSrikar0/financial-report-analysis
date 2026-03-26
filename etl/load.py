"""ETL load layer for Supabase/PostgreSQL."""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

BATCH_SIZE = 100


def get_supabase_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL/SUPABASE_KEY (or supabase_url/supabase_key) in .env"
        )
    return create_client(url, key)


def get_supabase_admin_client():
    """
    Returns Supabase client with service role key for backend operations.
    Bypasses RLS, allowing inserts with specific user_id values.
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env. "
            "Get service role key from Supabase → Settings → API → Service role"
        )
    return create_client(url, service_key)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    data = df.copy()
    for col in data.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ):
        data[col] = data[col].astype(str)
    data = data.where(pd.notnull(data), other=None)
    return data.to_dict(orient="records")


def _batch_upsert(client, table_name: str, records: list[dict]):
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        client.table(table_name).upsert(batch).execute()
        end = min(i + BATCH_SIZE, total)
        print(f"Upserted rows {i + 1}-{end} of {total} into {table_name}")


def load_to_supabase(staged_path: str, table_name: str):
    if not os.path.isabs(staged_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        staged_path = os.path.join(project_root, staged_path)

    if not os.path.exists(staged_path):
        raise FileNotFoundError(f"File not found: {staged_path}")

    print(f"Loading {staged_path} -> {table_name}")
    df = pd.read_csv(staged_path)
    records = _df_to_records(df)

    client = get_supabase_client()
    _batch_upsert(client, table_name, records)
    print(f"Finished loading {len(records)} rows into '{table_name}'")


# ── New: user-scoped loader for uploaded data ─────────────────────────────────

def store_uploaded_file(
    filename: str,
    raw_records: list,
    ticker: str,
    supabase_client,
    user_id: str,
) -> bool:
    """
    Store raw uploaded file as JSON in uploaded_files table.
    This preserves the original data before any LLM processing.
    
    Uses admin client (service role) to bypass RLS for backend operations.
    
    Args:
        filename: Original filename
        raw_records: List of raw extracted records
        ticker: Company ticker
        supabase_client: Supabase client (unused, uses admin client instead)
        user_id: Authenticated user's ID
    
    Returns: True if stored successfully
    """
    if not user_id or user_id == "predefined":
        print(f"[store_uploaded_file] ✗ CRITICAL: Invalid user_id: '{user_id}'. User must be authenticated.")
        return False
    
    if not raw_records:
        print(f"[store_uploaded_file] ✗ CRITICAL: No records to store.")
        return False
    
    try:
        # Use admin client (service role) to bypass RLS
        admin_client = get_supabase_admin_client()
        record = {
            "user_id": user_id,
            "filename": filename,
            "file_content": raw_records,  # Stored as JSONB
            "ticker": ticker,
        }
        print(f"[store_uploaded_file] DEBUG: Attempting to insert for user_id={user_id[:8]}..., ticker={ticker}, records={len(raw_records)}")
        response = admin_client.table("uploaded_files").insert(record).execute()
        print(f"[store_uploaded_file] ✓ Stored {len(raw_records)} records for {ticker} (user={user_id[:8]}...)")
        return True
    except Exception as e:
        print(f"[store_uploaded_file] ✗ CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def store_recommendation_results(
    ticker: str,
    recommendation_json: dict,
    supabase_client,
    user_id: str,
) -> bool:
    """
    Store LLM-generated recommendations as JSON in recommendation_results table.
    
    Uses admin client (service role) to bypass RLS for backend operations.
    
    Args:
        ticker: Company ticker
        recommendation_json: Full recommendation dict from Phase 6
        supabase_client: Supabase client (unused, uses admin client instead)
        user_id: Authenticated user's ID
    
    Returns: True if stored successfully
    """
    if not user_id or user_id == "predefined":
        print(f"[store_recommendation_results] ✗ Invalid user_id: '{user_id}'")
        return False
    
    try:
        # Use admin client (service role) to bypass RLS
        admin_client = get_supabase_admin_client()
        record = {
            "user_id": user_id,
            "ticker": ticker,
            "recommendation_json": recommendation_json,
            "performance_score": recommendation_json.get("performance_score"),
            "overall_risk": recommendation_json.get("risk_assessment", {}).get("overall_risk"),
            "predicted_growth_rate": recommendation_json.get("growth_outlook", {}).get("predicted_growth_rate"),
        }
        admin_client.table("recommendation_results").insert(record).execute()
        print(f"[store_recommendation_results] ✓ Stored recommendations for {ticker}")
        return True
    except Exception as e:
        print(f"[store_recommendation_results] ✗ Error: {type(e).__name__}: {e}")
        return False


def load_user_data(
    standard_df: pd.DataFrame,
    category_df: pd.DataFrame,
    supabase_client,
    user_id: str,
) -> bool:
    """
    Load user-uploaded (LLM-transformed) DataFrames into Supabase.
    Tags every record with user_id for row-level isolation.
    Uses upsert so re-uploads don't create duplicates.

    Returns True if both tables loaded successfully.
    """

    def _prepare(df: pd.DataFrame) -> list[dict]:
        d = df.copy()
        
        # Ensure user_id column is set and is not None/empty
        if not user_id or user_id == "predefined":
            raise ValueError(f"Invalid user_id: {user_id}. User must be authenticated.")
        d["user_id"] = user_id
        
        # CRITICAL: Ensure ticker is not NULL (required for SVR training)
        if "ticker" in d.columns:
            if d["ticker"].isna().any():
                print(f"[load_user_data] ⚠️ WARNING: Found {d['ticker'].isna().sum()} rows with NULL ticker!")
            if d["ticker"].isna().all():
                raise ValueError(
                    "Cannot load data: ALL records have NULL ticker. "
                    "Ticker must be set before upserting."
                )
        elif d.shape[0] > 0:
            raise ValueError("Cannot load data: 'ticker' column not found in DataFrame!")
        
        # Serialise datetimes
        for col in d.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns:
            d[col] = d[col].dt.strftime("%Y-%m-%d")
        
        # Replace inf with nan first
        d = d.replace([np.inf, -np.inf], np.nan)
        
        # Convert all NaN/None to None (Python None, not numpy.nan)
        d = d.where(pd.notnull(d), other=None)
        
        # Extra safety: explicitly handle any remaining NaN in dict conversion
        records = d.to_dict(orient="records")
        for record in records:
            for key, val in record.items():
                if pd.isna(val):
                    record[key] = None
        
        return records

    errors = []

    # standard_table
    try:
        std_records = _prepare(standard_df)
        _batch_upsert(supabase_client, "standard_table", std_records)
        print(
            f"[load_user_data] ✓ Loaded {len(std_records)} rows "
            f"into standard_table for user {user_id[:8]}…"
        )
    except Exception as e:
        errors.append(f"standard_table: {e}")
        print(f"[load_user_data] ✗ standard_table error: {e}")

    # category_table
    try:
        cat_records = _prepare(category_df)
        _batch_upsert(supabase_client, "category_table", cat_records)
        print(
            f"[load_user_data] ✓ Loaded {len(cat_records)} rows "
            f"into category_table for user {user_id[:8]}…"
        )
    except Exception as e:
        errors.append(f"category_table: {e}")
        print(f"[load_user_data] ✗ category_table error: {e}")

    return len(errors) == 0


if __name__ == "__main__":
    from transform import transform_data

    standard_path, category_path = transform_data()

    print("\n--- Loading Standard Table (for ML/SVR) ---")
    load_to_supabase(standard_path, "standard_table")

    print("\n--- Loading Category Table (for LLM Recommendations) ---")
    load_to_supabase(category_path, "category_table")