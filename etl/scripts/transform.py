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
        'open_price': 'open',
        'close_price': 'close',
        'volume': 'trading_volume',
        'revenue': 'total_revenue',
        'net_income': 'profit'
    },
    'category_fields': {
        'ticker': 'symbol',
        'sector': 'industry_sector',
        'category': 'financial_category',
        'risk_level': 'risk_rating'
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

def transform_data(raw_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_dir = os.path.join(base_dir, 'data', 'staged')
    os.makedirs(staged_dir, exist_ok = True)
    
    # Load raw JSON data
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    
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
    if 'open_price' in df.columns and 'close_price' in df.columns:
        df['price_change'] = df['close_price'] - df['open_price']
        df['price_change_pct'] = (df['price_change'] / df['open_price']) * 100
    
    if 'revenue' in df.columns and 'net_income' in df.columns:
        df['profit_margin'] = (df['net_income'] / df['revenue']) * 100
    
    # Mapping-based Transform: Create Standard Table (for ML/SVR)
    standard_data = apply_field_mapping(df.to_dict('records'), FIELD_MAPPINGS['standard_fields'])
    standard_df = pd.DataFrame(standard_data) if standard_data else df.copy()
    
    # Mapping-based Transform: Create Category Table (for LLM Recommendations)
    # Extract categorical/classification features
    category_df = df.copy()
    if 'ticker' in category_df.columns:
        # Add sample categorization (customize based on your data)
        category_df['sector'] = 'Technology'  # TODO: Add actual sector mapping
        category_df['category'] = 'Growth Stock'  # TODO: Add actual categorization
        category_df['risk_level'] = 'Medium'  # TODO: Add risk assessment logic
    
    category_data = apply_field_mapping(category_df.to_dict('records'), FIELD_MAPPINGS['category_fields'])
    category_df = pd.DataFrame(category_data) if category_data else category_df
    
    #save transformed data - Standard Table
    standard_path = os.path.join(staged_dir, 'standard_table.csv')
    standard_df.to_csv(standard_path, index = False)
    print(f"Standard table (for ML/SVR) saved to {standard_path}")
    
    #save transformed data - Category Table
    category_path = os.path.join(staged_dir, 'category_table.csv')
    category_df.to_csv(category_path, index = False)
    print(f"Category table (for LLM) saved to {category_path}")
    
    return standard_path, category_path

if __name__ == "__main__":
    raw_path = extract_data()
    transform_data(raw_path)