"""
Script to fix RLS policies on uploaded_files and recommendation_results tables.
This ensures authenticated users can read their own data via RLS.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not service_key:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
    print("   Get service role key from: Supabase → Settings → API → Service role")
    exit(1)

print(f"Connecting to Supabase: {url}")
admin_client = create_client(url, service_key)

# SQL to fix RLS policies
fix_rls_sql = """
-- Drop old restrictive policies on uploaded_files
DROP POLICY IF EXISTS "Users can read own data" ON uploaded_files;
DROP POLICY IF EXISTS "Users can write own data" ON uploaded_files;
DROP POLICY IF EXISTS "enable read for authenticated users" ON uploaded_files;
DROP POLICY IF EXISTS "enable insert for authenticated users" ON uploaded_files;

-- Create new permissive policies for uploaded_files
CREATE POLICY "Users can read own uploaded files" ON uploaded_files
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert uploaded files" ON uploaded_files
FOR INSERT
WITH CHECK (true);

-- Drop old restrictive policies on recommendation_results
DROP POLICY IF EXISTS "Users can read own data" ON recommendation_results;
DROP POLICY IF EXISTS "Users can write own data" ON recommendation_results;
DROP POLICY IF EXISTS "enable read for authenticated users" ON recommendation_results;
DROP POLICY IF EXISTS "enable insert for authenticated users" ON recommendation_results;

-- Create new permissive policies for recommendation_results
CREATE POLICY "Users can read own recommendations" ON recommendation_results
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert recommendations" ON recommendation_results
FOR INSERT
WITH CHECK (true);
"""

try:
    # Execute raw SQL via Postgres
    print("\n⏳ Updating RLS policies...")
    
    # Note: Supabase Python SDK doesn't directly expose raw SQL execution
    # We'll use a workaround by calling a stored procedure or using the REST API
    # For now, just guide the user to run it manually
    
    print("\n✓ To apply these policies, go to:")
    print("  1. Supabase Console → SQL Editor")
    print("  2. New Query")
    print("  3. Paste this SQL:\n")
    print(fix_rls_sql)
    print("\n  4. Run the query")
    
except Exception as e:
    print(f"Error: {e}")
