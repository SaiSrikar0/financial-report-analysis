# Data Connection Module
# Handles all Supabase queries for analysis

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_supabase_client():
    """Initialize and return Supabase client"""
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    anon_key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    use_service_role = _is_truthy(os.getenv("SUPABASE_USE_SERVICE_ROLE", "0"))

    key = anon_key
    if use_service_role and service_role_key:
        key = service_role_key

    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL/SUPABASE_KEY (or supabase_url/supabase_key) in .env"
        )
    
    client = create_client(url, key)
    
    # Import here to avoid circular imports
    try:
        import streamlit as st
        # If user is authenticated (logged in), set their session on the client
        # This ensures RLS policies use auth.uid() correctly
        if "session" in st.session_state and st.session_state["session"]:
            session = st.session_state["session"]
            client.auth.set_session(session.access_token, session.refresh_token)
    except Exception:
        pass
    
    return client


def get_table_data(table_name="standard_table"):
    """Fetch data from specified Supabase table."""
    try:
        supabase = get_supabase_client()
        response = supabase.table(table_name).select("*").execute()
        if not response.data:
            raise ValueError(f"No rows returned from Supabase table: {table_name}")

        df = pd.DataFrame(response.data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "ticker" in df.columns and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"✗ Error loading {table_name} from Supabase: {e}")
        print(f"→ Falling back to local staged CSV for {table_name}")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(base_dir, "data", "staged", f"{table_name}.csv")
        if not os.path.exists(local_path):
            raise

        df = pd.read_csv(local_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "ticker" in df.columns and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df


def get_standard_table_data():
    """Fetch standard_table (ML features)."""
    return get_table_data("standard_table")


def get_category_table_data():
    """Fetch category_table (business context)."""
    return get_table_data("category_table")


def get_analysis_data():
    """
    Fetch both tables and merge for comprehensive analysis.
    Returns:
        Tuple of (standard_df, category_df, merged_df)
    """
    print("\n--- Loading Data from Supabase ---")
    standard_df = get_standard_table_data()
    category_df = get_category_table_data()

    merged_df = standard_df.merge(
        category_df[["ticker", "date", "sector", "category", "risk_level"]],
        on=["ticker", "date"],
        how="left",
    )
    print(f"✓ Merged datasets: {len(merged_df)} records\n")
    return standard_df, category_df, merged_df


def get_company_data(ticker):
    """Fetch data for a specific company."""
    try:
        standard_df, category_df, _ = get_analysis_data()
        company_data = standard_df[standard_df["ticker"] == ticker].copy()
        if len(company_data) == 0:
            raise ValueError(f"No data found for ticker: {ticker}")
        print(f"✓ Loaded {len(company_data)} records for {ticker}\n")
        return company_data
    except Exception as e:
        print(f"✗ Error loading data for {ticker}: {str(e)}")
        raise


def get_companies_list():
    """Get list of unique companies in dataset."""
    try:
        standard_df, _, _ = get_analysis_data()
        companies = sorted(standard_df["ticker"].unique().tolist())
        print(f"✓ Found {len(companies)} companies: {', '.join(companies)}\n")
        return companies
    except Exception as e:
        print(f"✗ Error fetching companies list: {str(e)}")
        raise


# ── New: user-scoped query functions ──────────────────────────────────────────

# Predefined tickers that are always available for peer comparison
PREDEFINED_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN"}


def load_user_standard_table(
    user_id: str, supabase_client=None
) -> pd.DataFrame:
    """
    Load standard_table rows visible to a given user.
    Includes:
      - Rows tagged with this user_id (their uploads)
      - Rows tagged 'predefined' (built-in AAPL/MSFT/GOOGL/AMZN data)
    Falls back to local CSV + full table if user_id column doesn't exist yet.
    """
    client = supabase_client or get_supabase_client()
    try:
        user_rows = (
            client.table("standard_table")
            .select("*")
            .eq("user_id", user_id)
            .execute()
            .data
        )
        pred_rows = (
            client.table("standard_table")
            .select("*")
            .eq("user_id", "predefined")
            .execute()
            .data
        )
        all_rows = (user_rows or []) + (pred_rows or [])
        if not all_rows:
            # No user_id column or empty table — fall back to full table
            return get_standard_table_data()
        df = pd.DataFrame(all_rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "ticker" in df.columns and "date" in df.columns:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[DataConnection] user_id query failed ({e}), loading full table.")
        return get_standard_table_data()


def load_user_category_table(
    user_id: str, supabase_client=None
) -> pd.DataFrame:
    """
    Load category_table rows visible to a given user.
    Same scoping logic as load_user_standard_table.
    """
    client = supabase_client or get_supabase_client()
    try:
        user_rows = (
            client.table("category_table")
            .select("*")
            .eq("user_id", user_id)
            .execute()
            .data
        )
        pred_rows = (
            client.table("category_table")
            .select("*")
            .eq("user_id", "predefined")
            .execute()
            .data
        )
        all_rows = (user_rows or []) + (pred_rows or [])
        if not all_rows:
            return get_category_table_data()
        df = pd.DataFrame(all_rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        print(f"[DataConnection] category user_id query failed ({e}), loading full table.")
        return get_category_table_data()


def get_user_tickers(user_id: str, supabase_client=None) -> list:
    """
    Return sorted list of tickers accessible to a user
    (their uploads from uploaded_files + predefined AAPL/MSFT/GOOGL/AMZN).
    """
    if not supabase_client:
        supabase_client = get_supabase_client()
    
    tickers = set(PREDEFINED_TICKERS)
    
    # Fetch tickers from user's uploaded files
    try:
        response = supabase_client.table("uploaded_files").select("ticker").eq("user_id", user_id).execute()
        if response.data:
            user_tickers = [row["ticker"] for row in response.data if row.get("ticker")]
            tickers.update(user_tickers)
    except Exception as e:
        print(f"[get_user_tickers] Warning: Could not fetch from uploaded_files: {e}")
    
    return sorted(list(tickers))