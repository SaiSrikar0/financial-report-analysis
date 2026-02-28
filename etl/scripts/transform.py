#import necessary libraries
import os
import json
import pandas as pd
from extract import extract_data

# Field Mapping Configuration
FIELD_MAPPINGS = {
    'standard_fields': {
        'date': 'transaction_date',
        'ticker': 'symbol',
        'revenue': 'total_revenue',
        'operating_income': 'operating_income',
        'net_income': 'profit',
        'operating_cashflow': 'cashflow',
        'total_assets': 'assets',
        'total_liabilities': 'liabilities',
        'profit_margin': 'profit_margin',
        'operating_margin': 'operating_margin',
        'revenue_growth': 'revenue_growth',
        'net_income_growth': 'income_growth',
        'asset_efficiency': 'asset_efficiency',
        'debt_to_asset': 'debt_ratio'
    },
    'category_fields': {
        'ticker': 'symbol',
        'date': 'transaction_date',
        'sector': 'industry_sector',
        'category': 'financial_category',
        'risk_level': 'risk_rating',
        'revenue': 'total_revenue',
        'operating_income': 'operating_income',
        'net_income': 'profit'
    }
}

def apply_field_mapping(data, mapping):
    """Apply field mapping transformation"""
    mapped_data = []
    for record in data:
        mapped_record = {}
        for target_field, source_field in mapping.items():
            # Use original field name if exists, otherwise use mapped name
            if source_field in record:
                mapped_record[target_field] = record[source_field]
            elif target_field in record:
                mapped_record[target_field] = record[target_field]
        if mapped_record:  # Only add if mapping found fields
            mapped_data.append(mapped_record)
    return mapped_data

def transform_data(raw_path=None):
    # Get the etl directory (parent of scripts)
    etl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(etl_dir)  # Project root
    
    # If raw_path not provided, look in the data/raw directory (project root)
    if raw_path is None:
        raw_path = os.path.join(base_dir, 'data', 'raw', 'financial_data_raw.json')
    
    # Output to etl/data/staged
    staged_dir = os.path.join(etl_dir, 'data', 'staged')
    os.makedirs(staged_dir, exist_ok = True)
    
    # Load raw JSON data
    print(f"Loading raw data from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    print(f"Loaded {len(df)} records from raw data")
    
    #handling missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        if col in df.columns and not df[col].empty:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    #feature engineering for financial data
    # Calculate profit margin if we have revenue and net_income
    if 'revenue' in df.columns and 'net_income' in df.columns:
        df['profit_margin'] = (df['net_income'] / df['revenue']) * 100
    
    # Calculate operating margin
    if 'revenue' in df.columns and 'operating_income' in df.columns:
        df['operating_margin'] = (df['operating_income'] / df['revenue']) * 100
    
    # Calculate growth metrics (group by ticker to get year-over-year changes)
    if 'ticker' in df.columns and 'date' in df.columns and 'revenue' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change() * 100
        df['net_income_growth'] = df.groupby('ticker')['net_income'].pct_change() * 100
    
    # Calculate asset efficiency ratio
    if 'total_assets' in df.columns and 'revenue' in df.columns:
        df['asset_efficiency'] = df['revenue'] / df['total_assets']
    
    # Calculate debt-to-asset ratio
    if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
        df['debt_to_asset'] = df['total_liabilities'] / df['total_assets']
    
    # Mapping-based Transform: Create Standard Table (for ML/SVR)
    standard_data = apply_field_mapping(df.to_dict('records'), FIELD_MAPPINGS['standard_fields'])
    standard_df = pd.DataFrame(standard_data) if standard_data else df.copy()
    
    # Mapping-based Transform: Create Category Table (for LLM Recommendations)
    # Extract categorical/classification features
    category_df = df.copy()
    
    # Add sector mapping based on ticker
    sector_map = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology'
    }
    
    if 'ticker' in category_df.columns:
        category_df['sector'] = category_df['ticker'].map(sector_map)
        # Determine growth category based on revenue growth
        category_df['category'] = category_df.apply(
            lambda row: 'High Growth' if pd.notna(row.get('revenue_growth')) and row.get('revenue_growth', 0) > 10 
            else 'Moderate Growth' if pd.notna(row.get('revenue_growth')) and row.get('revenue_growth', 0) > 0
            else 'Stable' if pd.notna(row.get('revenue_growth'))
            else 'Unknown', axis=1
        )
        # Determine risk level based on debt ratio
        category_df['risk_level'] = category_df.apply(
            lambda row: 'High Risk' if pd.notna(row.get('debt_to_asset')) and row.get('debt_to_asset', 0) > 0.7
            else 'Medium Risk' if pd.notna(row.get('debt_to_asset')) and row.get('debt_to_asset', 0) > 0.4
            else 'Low Risk' if pd.notna(row.get('debt_to_asset'))
            else 'Unknown', axis=1
        )
    
    category_data = apply_field_mapping(category_df.to_dict('records'), FIELD_MAPPINGS['category_fields'])
    category_df = pd.DataFrame(category_data) if category_data else category_df
    
    #save transformed data - Standard Table
    # Replace NaN and Inf values with None for JSON compatibility
    standard_df = standard_df.replace([float('inf'), float('-inf')], None)
    standard_df = standard_df.where(pd.notnull(standard_df), None)
    
    standard_path = os.path.join(staged_dir, 'standard_table.csv')
    standard_df.to_csv(standard_path, index = False)
    print(f"Standard table (for ML/SVR) saved to {standard_path}")
    
    #save transformed data - Category Table
    # Replace NaN and Inf values with None for JSON compatibility
    category_df = category_df.replace([float('inf'), float('-inf')], None)
    category_df = category_df.where(pd.notnull(category_df), None)
    
    category_path = os.path.join(staged_dir, 'category_table.csv')
    category_df.to_csv(category_path, index = False)
    print(f"Category table (for LLM) saved to {category_path}")
    
    return standard_path, category_path

if __name__ == "__main__":
    # Transform existing raw data from data/raw directory
    standard_path, category_path = transform_data()