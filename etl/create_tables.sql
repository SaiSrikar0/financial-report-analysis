-- FinCast: Simple Table Creation Script
-- Run this in Supabase SQL Editor

-- Standard Table for ML/SVR Model Training
CREATE TABLE IF NOT EXISTS standard_table (
    id BIGSERIAL PRIMARY KEY,
    date TEXT,
    ticker TEXT,
    revenue FLOAT,
    operating_income FLOAT,
    net_income FLOAT,
    operating_cashflow FLOAT,
    total_assets FLOAT,
    total_liabilities FLOAT,
    profit_margin FLOAT,
    operating_margin FLOAT,
    revenue_growth FLOAT,
    net_income_growth FLOAT,
    asset_efficiency FLOAT,
    debt_to_asset FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Category Table for LLM Recommendations
CREATE TABLE IF NOT EXISTS category_table (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT,
    date TEXT,
    sector TEXT,
    category TEXT,
    risk_level TEXT,
    revenue FLOAT,
    operating_income FLOAT,
    net_income FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- MIGRATION: Add user_id column for multi-user data isolation
-- Run this AFTER tables already exist
-- ============================================================

ALTER TABLE standard_table ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'predefined';
ALTER TABLE category_table  ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'predefined';

-- Tag existing rows as predefined data
UPDATE standard_table SET user_id = 'predefined' WHERE user_id IS NULL;
UPDATE category_table  SET user_id = 'predefined' WHERE user_id IS NULL;

-- Composite unique constraints (needed for upsert)
ALTER TABLE standard_table DROP CONSTRAINT IF EXISTS std_ticker_date_user_uq;
ALTER TABLE standard_table ADD CONSTRAINT std_ticker_date_user_uq
    UNIQUE (ticker, date, user_id);

ALTER TABLE category_table DROP CONSTRAINT IF EXISTS cat_ticker_date_user_uq;
ALTER TABLE category_table ADD CONSTRAINT cat_ticker_date_user_uq
    UNIQUE (ticker, date, user_id);

-- Enable Row Level Security
ALTER TABLE standard_table ENABLE ROW LEVEL SECURITY;
ALTER TABLE category_table  ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if re-running
DROP POLICY IF EXISTS "user_isolation_standard" ON standard_table;
DROP POLICY IF EXISTS "user_isolation_category" ON category_table;

-- Users see their own data + predefined reference data
CREATE POLICY "user_isolation_standard" ON standard_table
    FOR ALL USING (
        user_id = auth.uid()::text
        OR user_id = 'predefined'
    );

CREATE POLICY "user_isolation_category" ON category_table
    FOR ALL USING (
        user_id = auth.uid()::text
        OR user_id = 'predefined'
    );

-- ============================================================
-- ADDITIONAL TABLES: User uploads and LLM recommendations
-- ============================================================

-- Raw uploaded files (stores original uploaded JSON/CSV)
CREATE TABLE IF NOT EXISTS uploaded_files (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT,
    file_content JSONB,
    ticker TEXT,
    upload_date TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- LLM Recommendation results (stores Phase 6 output)
CREATE TABLE IF NOT EXISTS recommendation_results (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    ticker TEXT,
    recommendation_json JSONB,
    performance_score INT,
    overall_risk TEXT,
    predicted_growth_rate FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable Row Level Security on new tables
ALTER TABLE uploaded_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_results ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if re-running
DROP POLICY IF EXISTS "user_isolation_uploaded" ON uploaded_files;
DROP POLICY IF EXISTS "user_isolation_recommendations" ON recommendation_results;

-- RLS Policies: Users see only their own data
CREATE POLICY "user_isolation_uploaded" ON uploaded_files
    FOR ALL USING (user_id = auth.uid()::text);

CREATE POLICY "user_isolation_recommendations" ON recommendation_results
    FOR ALL USING (user_id = auth.uid()::text);