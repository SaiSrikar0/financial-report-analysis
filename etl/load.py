"""ETL load layer for Supabase/PostgreSQL."""

import os
import hashlib
import json

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


def _batch_upsert(client, table_name: str, records: list[dict], on_conflict: str | None = None):
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        try:
            if on_conflict:
                client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
            else:
                client.table(table_name).upsert(batch).execute()
        except Exception as e:
            # Fallback for environments where unique constraints were not migrated yet.
            msg = str(e).lower()
            if "no unique" in msg or "there is no unique or exclusion constraint" in msg:
                print(
                    f"[_batch_upsert] ⚠️ Upsert not available for {table_name} without unique constraint. "
                    "Falling back to insert."
                )
                client.table(table_name).insert(batch).execute()
            else:
                raise
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
        print("[store_uploaded_file] ✗ CRITICAL: No records to store.")
        return False

    # Prefer service-role client for backend inserts; fall back to provided client.
    client = None
    try:
        client = get_supabase_admin_client()
    except Exception as e:
        print(f"[store_uploaded_file] ⚠️ Admin client unavailable, using session client: {e}")
        client = supabase_client

    if client is None:
        print("[store_uploaded_file] ✗ CRITICAL: No Supabase client available for insert.")
        return False

    try:
        record = {
            "user_id": user_id,
            "filename": filename,
            "file_content": raw_records,  # Stored as JSONB
            "ticker": str(ticker).upper(),
        }
        print(
            f"[store_uploaded_file] DEBUG: inserting user_id={user_id[:8]}..., "
            f"ticker={record['ticker']}, records={len(raw_records)}"
        )
        client.table("uploaded_files").insert(record).execute()
        print(f"[store_uploaded_file] ✓ Stored {len(raw_records)} records for {record['ticker']} (user={user_id[:8]}...)")
        return True
    except Exception as e:
        print(f"[store_uploaded_file] ✗ CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _stable_records_hash(raw_records: list) -> str:
    """Compute deterministic hash for JSON-like record list."""
    payload = json.dumps(raw_records or [], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_duplicate_uploaded_file(
    user_id: str,
    ticker: str,
    raw_records: list,
) -> bool:
    """
    Check if same user already uploaded the same ticker with identical content.
    """
    if not user_id or not ticker or raw_records is None:
        return False

    target_hash = _stable_records_hash(raw_records)

    try:
        admin_client = get_supabase_admin_client()
        response = (
            admin_client.table("uploaded_files")
            .select("id,file_content")
            .eq("user_id", user_id)
            .eq("ticker", str(ticker).upper())
            .limit(200)
            .execute()
        )

        rows = response.data if response and response.data else []
        for row in rows:
            existing_hash = _stable_records_hash(row.get("file_content") or [])
            if existing_hash == target_hash:
                return True
        return False
    except Exception as e:
        print(f"[is_duplicate_uploaded_file] Warning: {type(e).__name__}: {e}")
        # Fail-open to avoid blocking uploads on transient DB errors.
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
    return_details: bool = False,
) -> bool | dict:
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
            # Normalize ticker values so conflict-key dedupe is consistent.
            d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
            d["ticker"] = d["ticker"].replace({"": None, "NAN": None, "NONE": None})
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

        # De-duplicate by the same conflict key used in Supabase upsert.
        # Without this, PostgreSQL raises:
        # "ON CONFLICT DO UPDATE command cannot affect row a second time"
        # when a single insert payload contains repeated (ticker, date, user_id).
        conflict_cols = ["ticker", "date", "user_id"]
        if all(col in d.columns for col in conflict_cols):
            before_dedupe = len(d)
            d = d.drop_duplicates(subset=conflict_cols, keep="last").copy()
            dropped = before_dedupe - len(d)
            if dropped > 0:
                print(
                    f"[load_user_data] ⚠️ Deduplicated {dropped} rows "
                    f"on conflict key ({', '.join(conflict_cols)})."
                )
        
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

    write_client = None
    try:
        write_client = get_supabase_admin_client()
        print("[load_user_data] Using admin client for writes.")
    except Exception as e:
        print(f"[load_user_data] Admin client unavailable, using provided client: {e}")
        write_client = supabase_client

    if write_client is None:
        error = "No Supabase write client available"
        if return_details:
            return {"success": False, "errors": [error]}
        return False

    # standard_table
    try:
        std_records = _prepare(standard_df)
        _batch_upsert(write_client, "standard_table", std_records, on_conflict="ticker,date,user_id")
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
        _batch_upsert(write_client, "category_table", cat_records, on_conflict="ticker,date,user_id")
        print(
            f"[load_user_data] ✓ Loaded {len(cat_records)} rows "
            f"into category_table for user {user_id[:8]}…"
        )
    except Exception as e:
        errors.append(f"category_table: {e}")
        print(f"[load_user_data] ✗ category_table error: {e}")

    success = len(errors) == 0
    if return_details:
        return {"success": success, "errors": errors}
    return success


def delete_user_uploaded_data(
    user_id: str,
    ticker: str,
    uploaded_file_id: str | int | None = None,
) -> dict:
    """
    Delete uploaded data for a user/ticker.

    Behavior:
    - If uploaded_file_id is provided, delete that specific row from uploaded_files.
    - Always delete user-scoped derived rows for the ticker from standard_table,
      category_table, and recommendation_results.

    Returns a summary dict with success flag and per-table counts (best effort).
    """
    if not user_id or user_id == "predefined":
        return {"success": False, "error": f"Invalid user_id: {user_id}"}
    if not ticker:
        return {"success": False, "error": "Missing ticker"}

    admin_client = get_supabase_admin_client()
    ticker_upper = str(ticker).upper()

    result = {
        "success": True,
        "deleted": {
            "uploaded_files": 0,
            "standard_table": 0,
            "category_table": 0,
            "recommendation_results": 0,
        },
        "errors": [],
    }

    try:
        q = admin_client.table("uploaded_files").delete().eq("user_id", user_id)
        if uploaded_file_id is not None:
            q = q.eq("id", uploaded_file_id)
        else:
            q = q.eq("ticker", ticker_upper)
        resp = q.execute()
        result["deleted"]["uploaded_files"] = len(resp.data) if resp and resp.data else 0
    except Exception as e:
        result["success"] = False
        result["errors"].append(f"uploaded_files: {e}")

    for table_name in ["standard_table", "category_table", "recommendation_results"]:
        try:
            resp = (
                admin_client.table(table_name)
                .delete()
                .eq("user_id", user_id)
                .eq("ticker", ticker_upper)
                .execute()
            )
            result["deleted"][table_name] = len(resp.data) if resp and resp.data else 0
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"{table_name}: {e}")

    return result


if __name__ == "__main__":
    from transform import transform_data

    standard_path, category_path = transform_data()

    print("\n--- Loading Standard Table (for ML/SVR) ---")
    load_to_supabase(standard_path, "standard_table")

    print("\n--- Loading Category Table (for LLM Recommendations) ---")
    load_to_supabase(category_path, "category_table")