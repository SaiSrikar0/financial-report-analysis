# Supabase User-Data Mapping Implementation Guide

## Your Setup ✓
- Tables: `standard_table`, `category_table`
- Both have `user_id` column + RLS policies
- Auth module: Already working (signup, login, session management)
- Auth method: Email + password via Supabase Auth

---

## WORKING SNIPPETS

### 1. Insert Data with User ID (Automatic Mapping)

```python
# File: etl/load.py or custom insert function
from auth.supabase_auth import get_user_id
from analysis.data_connection import get_supabase_client
import streamlit as st

def insert_standard_table_with_user_id(records: list):
    """
    Insert records into standard_table with automatic user_id attachment.
    
    records: List of dicts with financial data
    """
    user_id = get_user_id()  # Get logged-in user's ID
    
    # Attach user_id to each record
    for rec in records:
        rec["user_id"] = user_id
    
    try:
        supabase = get_supabase_client()
        response = supabase.table("standard_table").insert(records).execute()
        print(f"✓ Inserted {len(records)} records for user {user_id}")
        st.success(f"✓ {len(records)} records saved!")
        return True
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        st.error(f"Insert failed: {e}")
        return False


def insert_category_table_with_user_id(records: list):
    """
    Insert records into category_table with automatic user_id attachment.
    """
    user_id = get_user_id()
    
    for rec in records:
        rec["user_id"] = user_id
    
    try:
        supabase = get_supabase_client()
        response = supabase.table("category_table").insert(records).execute()
        print(f"✓ Inserted {len(records)} category records for user {user_id}")
        return True
    except Exception as e:
        print(f"✗ Category insert failed: {e}")
        return False
```

---

### 2. Fetch Data with RLS (User-Filtered)

```python
# File: analysis/data_connection.py (enhanced)
import pandas as pd
from supabase import create_client
from auth.supabase_auth import get_user_id
import os
from dotenv import load_dotenv

def get_user_standard_table():
    """
    Fetch ONLY logged-in user's data from standard_table.
    RLS policy automatically filters to:
    - user_id = auth.uid()  OR  user_id = 'predefined'
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")
    supabase = create_client(url, key)
    
    try:
        # RLS will automatically filter!
        response = supabase.table("standard_table").select("*").execute()
        df = pd.DataFrame(response.data)
        print(f"✓ Fetched {len(df)} records for user (+ predefined)")
        return df
    except Exception as e:
        print(f"✗ Fetch failed: {e}")
        return pd.DataFrame()


def get_user_category_table():
    """
    Fetch ONLY logged-in user's data from category_table.
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")
    supabase = create_client(url, key)
    
    try:
        response = supabase.table("category_table").select("*").execute()
        df = pd.DataFrame(response.data)
        print(f"✓ Fetched {len(df)} category records for user")
        return df
    except Exception as e:
        print(f"✗ Category fetch failed: {e}")
        return pd.DataFrame()


def get_user_tickers():
    """
    Get list of tickers the user has uploaded/created.
    """
    df = get_user_standard_table()
    if df.empty:
        return []
    # Exclude 'predefined' data if desired
    user_tickers = df[df["user_id"] != "predefined"]["ticker"].unique().tolist()
    return sorted(user_tickers)
```

---

### 3. Dashboard Integration

```python
# File: app.py (integration point)
import streamlit as st
from auth.supabase_auth import is_authenticated, get_user_id, get_user_email
from analysis.data_connection import get_user_standard_table, get_user_tickers

if not is_authenticated():
    st.stop()

user_id = get_user_id()
user_email = get_user_email()

# ── Debug: Show user context ──────────────────────────────
with st.sidebar:
    st.markdown(f"**User:** {user_email}")
    st.markdown(f"**ID:** `{user_id}`")

# ── Fetch user's data ──────────────────────────────────────
st.header("Your Financial Data")

try:
    df = get_user_standard_table()
    tickers = get_user_tickers()
    
    if df.empty:
        st.info("No data yet. Upload financial records to get started.")
    else:
        st.metric("Total Records", len(df))
        st.metric("Tickers", len(tickers))
        st.write(f"Your tickers: {', '.join(tickers)}")
        
        # Display data
        st.dataframe(df, use_container_width=True)
        
except Exception as e:
    st.error(f"Failed to load data: {e}")
```

---

### 4. ETL Pipeline with User Mapping

```python
# File: etl/extract.py or main ETL orchestrator
from auth.supabase_auth import get_user_id
from etl.llm_extractor import run_extraction_pipeline
from etl.load import insert_standard_table_with_user_id, insert_category_table_with_user_id

def upload_and_process_file(uploaded_file, ticker: str):
    """
    User uploads file → Extract/Transform → Load with user_id
    """
    user_id = get_user_id()
    print(f"\n[ETL] Processing for user {user_id}")
    
    # 1. Extract raw data
    raw_records = extract_from_file(uploaded_file)  # Your extraction logic
    
    # 2. LLM Schema normalization (uses Groq now)
    standard_records, category_records = run_extraction_pipeline(raw_records, ticker)
    
    # 3. Load with automatic user_id attachment
    success = insert_standard_table_with_user_id(standard_records)
    if success:
        insert_category_table_with_user_id(category_records)
        print(f"✓ ETL complete. Data saved for {ticker}")
    
    return success
```

---

## DEBUGGING CHECKLIST

### ✓ Verify User ID After Login
```python
# Add to app.py to debug
if st.button("Debug: Show Session"):
    st.json({
        "user_id": st.session_state.get("user_id"),
        "user_email": st.session_state.get("user_email"),
        "user_object": str(st.session_state.get("user")),
    })
```

### ✓ Check Session Persistence
```python
# In Streamlit sidebar
import streamlit as st
st.session_state  # Shows all session vars
```

### ✓ Verify RLS in Supabase Console
1. Go to Supabase → Authentication → Users
2. Copy a user's ID (UUID format)
3. Go to SQL Editor → Run:
```sql
SELECT * FROM standard_table WHERE user_id = 'YOUR_USER_ID_HERE';
```

### ✓ Test Insert → Immediate Fetch
```python
# Streamlit test page
if st.button("Test Insert"):
    test_record = [{"date": "2024-01-01", "ticker": "TEST", "revenue": 100.0}]
    insert_standard_table_with_user_id(test_record)
    
    df = get_user_standard_table()
    st.write("Fetched back:", df)
```

---

## KEY POINTS

1. **Auth Session is Persistent:** Streamlit session_state holds user_id automatically after login
2. **RLS Handles Filtering:** When you query, Supabase RLS automatically filters by auth.uid()
3. **Insert Pattern:** Always attach `user_id = get_user_id()` before insert
4. **Fetch Pattern:** Simple `.select("*")` — RLS does the filtering server-side
5. **Predefined Data:** Remains visible to all users (via `user_id = 'predefined'` policy)

---

## NEXT STEPS

1. **Update `etl/load.py`** to use `insert_*_with_user_id()` functions
2. **Update `analysis/data_connection.py`** to replace fetch functions with user-filtered versions
3. **Test login → upload → fetch** flow end-to-end
4. **Monitor Supabase logs** for RLS denials (if any permissions issues)

Your existing auth module + RLS tables already support this. Just ensure data operations include user_id attachment.
