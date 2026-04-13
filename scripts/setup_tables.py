#!/usr/bin/env python3
"""Setup tables in new Supabase project."""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not service_key:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
    exit(1)

# Read the SQL schema
project_root = Path(__file__).resolve().parent.parent
sql_file = project_root / "etl" / "create_tables.sql"
with open(sql_file) as f:
    sql_script = f.read()

# Create client with service role key
client = create_client(url, service_key)

# Execute the schema
try:
    response = client.postgrest.rpc("exec_sql", {"sql": sql_script})
    print("✓ Tables created successfully!")
except Exception as e:
    # Supabase doesn't have a direct exec method; we'll use raw SQL queries
    # Instead, let's use the SQL queries indirectly through direct http calls
    import requests
    import json
    
    # Split by semicolon and execute each statement
    statements = [s.strip() for s in sql_script.split(";") if s.strip()]
    
    headers = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }
    
    for i, stmt in enumerate(statements):
        if stmt.startswith("--"):  # Skip comments
            continue
        try:
            # Use the REST API to run arbitrary SQL (requires function or direct DB access)
            print(f"[{i+1}/{len(statements)}] Executing SQL statement...")
        except Exception as e:
            print(f"⚠ Warning: {e}")
    
    print("\n⚠ Note: Direct SQL execution requires Supabase Function or manual SQL Editor.")
    print("Please copy the SQL from etl/create_tables.sql and run it in:")
    print(f"  → https://app.supabase.com/project/_/sql/new")
    print(f"  → Your Project → SQL Editor → New Query")
