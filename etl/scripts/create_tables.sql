-- FinCast: Simple Table Creation Script
-- Run this in Supabase SQL Editor

-- Standard Table for ML/SVR Model Training
CREATE TABLE standard_table (
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
CREATE TABLE category_table (
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
