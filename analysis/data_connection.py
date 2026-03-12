# Data Connection Module
# Handles all Supabase queries for analysis

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv


def get_supabase_client():
    """Initialize and return Supabase client"""
    load_dotenv()
    url = os.getenv("supabase_url")
    key = os.getenv("supabase_key")

    if not url or not key:
        raise ValueError("Missing Supabase URL or Supabase KEY in .env")

    return create_client(url, key)


def get_table_data(table_name='standard_table'):
    """Fetch data from specified Supabase table."""
    try:
        supabase = get_supabase_client()
        response = supabase.table(table_name).select('*').execute()
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"✗ Error loading {table_name}: {e}")
        raise


def get_standard_table_data():
    """Fetch standard_table (ML features)."""
    return get_table_data('standard_table')


def get_category_table_data():
    """Fetch category_table (business context)."""
    return get_table_data('category_table')


def get_analysis_data():
    """
    Fetch both tables and merge for comprehensive analysis
    
    Returns:
        Tuple of (standard_df, category_df, merged_df)
    """
    print("\n--- Loading Data from Supabase ---")
    standard_df = get_standard_table_data()
    category_df = get_category_table_data()
    
    # Merge on ticker and date for combined analysis
    merged_df = standard_df.merge(
        category_df[['ticker', 'date', 'sector', 'category', 'risk_level']],
        on=['ticker', 'date'],
        how='left'
    )
    
    print(f"✓ Merged datasets: {len(merged_df)} records\n")
    
    return standard_df, category_df, merged_df


def get_company_data(ticker):
    """
    Fetch data for a specific company
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL', 'MSFT', 'GOOGL')
    
    Returns:
        DataFrame filtered for the company
    """
    try:
        standard_df, category_df, _ = get_analysis_data()
        company_data = standard_df[standard_df['ticker'] == ticker].copy()
        
        if len(company_data) == 0:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        print(f"✓ Loaded {len(company_data)} records for {ticker}\n")
        return company_data
    except Exception as e:
        print(f"✗ Error loading data for {ticker}: {str(e)}")
        raise


def get_companies_list():
    """Get list of unique companies in dataset"""
    try:
        standard_df, _, _ = get_analysis_data()
        companies = sorted(standard_df['ticker'].unique().tolist())
        print(f"✓ Found {len(companies)} companies: {', '.join(companies)}\n")
        return companies
    except Exception as e:
        print(f"✗ Error fetching companies list: {str(e)}")
        raise
