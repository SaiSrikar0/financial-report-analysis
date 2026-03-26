# 🔧 CRITICAL FIX: Dashboard Not Showing Uploaded Files

## Problem
✗ Data uploads successfully BUT dashboard shows "No uploaded files yet"  
✗ Tickets don't appear in AI Recommendations

## Root Cause
The RLS (Row-Level Security) policies block authenticated users from reading their own data on the frontend.

## Why This Happened
- Your data IS stored in Supabase (verified with admin key)
- But the RLS policy doesn't allow the frontend to READ it
- Frontend uses unauthenticated key → RLS blocks access
- Admin key can READ (service role bypasses RLS)

## Solution: Update RLS Policies

### Step 1: Open Supabase Console
Go to: https://supabase.com → Your Project → SQL Editor

### Step 2: Create New Query
Click "New Query" and paste this SQL (it safely removes old policies first):

```sql
-- Safely drop all old policies on uploaded_files
DROP POLICY IF EXISTS "Users can read own data" ON uploaded_files;
DROP POLICY IF EXISTS "Users can write own data" ON uploaded_files;
DROP POLICY IF EXISTS "enable read for authenticated users" ON uploaded_files;
DROP POLICY IF EXISTS "enable insert for authenticated users" ON uploaded_files;
DROP POLICY IF EXISTS "Users can read own uploaded files" ON uploaded_files;
DROP POLICY IF EXISTS "Service role can insert uploaded files" ON uploaded_files;

-- Create fresh policies for uploaded_files
CREATE POLICY "Users can read own uploaded files" ON uploaded_files
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert uploaded files" ON uploaded_files
FOR INSERT
WITH CHECK (true);

-- Safely drop all old policies on recommendation_results
DROP POLICY IF EXISTS "Users can read own data" ON recommendation_results;
DROP POLICY IF EXISTS "Users can write own data" ON recommendation_results;
DROP POLICY IF EXISTS "enable read for authenticated users" ON recommendation_results;
DROP POLICY IF EXISTS "enable insert for authenticated users" ON recommendation_results;
DROP POLICY IF EXISTS "Users can read own recommendations" ON recommendation_results;
DROP POLICY IF EXISTS "Service role can insert recommendations" ON recommendation_results;

-- Create fresh policies for recommendation_results
CREATE POLICY "Users can read own recommendations" ON recommendation_results
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert recommendations" ON recommendation_results
FOR INSERT
WITH CHECK (true);
```

### Step 3: Run Query
Click "Run" or press Ctrl+Enter

### Step 4: Verify
You should see: `Query executed successfully with no results`

### Step 5: Restart App
In your terminal, stop the app (Ctrl+C) and restart:
```bash
streamlit run app.py
```

### Step 6: Test
1. Login
2. Upload a file
3. Go to Dashboard → "📁 Uploaded Data" tab
4. You should now see your uploaded file listed

## Technical Details
The code changes I made:
- ✓ Updated `auth/supabase_auth.py` to set auth session on client
- ✓ Updated `analysis/data_connection.py` to use authenticated session
- ✓ Updated `analysis/uploaded_data_analytics.py` to trust RLS filtering

These changes ensure the frontend client:
1. Knows who the authenticated user is
2. Passes that info to RLS policies
3. RLS allows reads for `auth.uid()::text = user_id`
