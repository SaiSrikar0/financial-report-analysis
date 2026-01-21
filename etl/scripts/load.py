#import necessary libraries
import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from transform import transform_data
from extract import extract_data

# Initialize Supabase Client
def get_supabase_client():
    load_dotenv()
    url = os.getenv("supabase_url")
    key = os.getenv("supabase_key")

    if not url or not key:
        raise ValueError("Missing Supabase URL or Supabase KEY in .env")

    return create_client(url, key)

# Create financial tables (if not exists)
def create_tables_if_not_exists():
    try:
        supabase = get_supabase_client()

        # Standard Table for ML/SVR
        create_standard_table_sql = """
        CREATE TABLE IF NOT EXISTS standard_table (
            id BIGSERIAL PRIMARY KEY,
            transaction_date DATE,
            symbol TEXT,
            open FLOAT,
            close FLOAT,
            trading_volume BIGINT,
            total_revenue FLOAT,
            profit FLOAT,
            price_change FLOAT,
            price_change_pct FLOAT,
            profit_margin FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Category Table for LLM Recommendations
        create_category_table_sql = """
        CREATE TABLE IF NOT EXISTS category_table (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            industry_sector TEXT,
            financial_category TEXT,
            risk_rating TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            supabase.rpc("execute_sql", {"query": create_standard_table_sql}).execute()
            print("Table 'standard_table' created or already exists.")
            supabase.rpc("execute_sql", {"query": create_category_table_sql}).execute()
            print("Table 'category_table' created or already exists.")
        except Exception as e:
            print(f"RPC failed: {e}")
            print("Assuming tables already exist or will auto-create.")

    except Exception as e:
        print(f"Error creating tables: {e}")
        print("Continuing with data insertion...")

# Load CSV into Supabase
def load_to_supabase(staged_path: str, table_name: str):

    # Convert to absolute path if needed
    if not os.path.isabs(staged_path):
        staged_path = os.path.abspath(os.path.join(os.path.dirname(__file__), staged_path))

    print(f"Looking for the data file at: {staged_path}")

    if not os.path.exists(staged_path):
        print(f"Error: File not found at {staged_path}")
        print("Run transform.py first.")
        return

    try:
        supabase = get_supabase_client()

        df = pd.read_csv(staged_path)
        total_rows = len(df)
        batch_size = 50

        print(f"Loading {total_rows} rows into table '{table_name}'...")

        # Insert in batches
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i: i + batch_size].copy()

            # Replace NaN with None
            batch = batch.where(pd.notnull(batch), None)
            records = batch.to_dict("records")
            try:
                supabase.table(table_name).insert(records).execute()

                end = min(i + batch_size, total_rows)
                print(f"Inserted rows {i + 1} – {end} of {total_rows}")

            except Exception as e:
                print(f"Error in batch {i // batch_size + 1}: {str(e)}")
                continue

        print(f"Finished loading data into '{table_name}'.")

    except Exception as e:
        print(f"Error loading data: {e}")

# Main Execution
if __name__ == "__main__":

    # Extract and transform data
    raw_path = extract_data()
    standard_path, category_path = transform_data(raw_path)

    # Create tables
    create_tables_if_not_exists()
    
    # Load Standard Table (for ML/SVR)
    print("\n--- Loading Standard Table (for ML/SVR) ---")
    load_to_supabase(standard_path, "standard_table")
    
    # Load Category Table (for LLM Recommendations)
    print("\n--- Loading Category Table (for LLM Recommendations) ---")
    load_to_supabase(category_path, "category_table")