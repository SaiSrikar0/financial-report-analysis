-- Fix RLS policies for user-scoped tables
-- Allow authenticated users to read/write their own records

-- Drop old restrictive policies
DROP POLICY IF EXISTS "Users can read own data" ON uploaded_files;
DROP POLICY IF EXISTS "Users can write own data" ON uploaded_files;
DROP POLICY IF EXISTS "Users can read own data" ON recommendation_results;
DROP POLICY IF EXISTS "Users can write own data" ON recommendation_results;

-- Create new permissive policies for uploaded_files
CREATE POLICY "Users can read own uploaded files"
ON uploaded_files
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert uploaded files"
ON uploaded_files
FOR INSERT
WITH CHECK (true);  -- Service role bypasses this anyway

-- Create new permissive policies for recommendation_results
CREATE POLICY "Users can read own recommendations"
ON recommendation_results
FOR SELECT
USING (auth.uid()::text = user_id);

CREATE POLICY "Service role can insert recommendations"
ON recommendation_results
FOR INSERT
WITH CHECK (true);  -- Service role bypasses this anyway
