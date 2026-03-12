"""Clear Supabase tables and reload with new data."""
import pandas as pd
import numpy as np
from analysis.data_connection import get_supabase_client

def clear_table(client, table_name):
    """Delete all rows from a table."""
    try:
        # Get all IDs
        response = client.table(table_name).select('id').execute()
        if response.data:
            ids = [row['id'] for row in response.data]
            print(f"Deleting {len(ids)} rows from {table_name}...")
            for row_id in ids:
                client.table(table_name).delete().eq('id', row_id).execute()
            print(f"✓ Cleared {table_name}")
        else:
            print(f"✓ {table_name} already empty")
    except Exception as e:
        print(f"✗ Error clearing {table_name}: {e}")

def sanitize_row(row):
    """Replace inf/nan values with None for JSON compliance."""
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                cleaned[key] = None
            else:
                cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned

def load_table(client, csv_path, table_name, batch_size=50):
    """Load CSV data into Supabase table."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\nLoading {len(df)} rows into {table_name}...")
        
        records = df.to_dict('records')
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cleaned_batch = [sanitize_row(row) for row in batch]
            
            try:
                client.table(table_name).insert(cleaned_batch).execute()
                total_inserted += len(batch)
                print(f"  ✓ Inserted rows {i+1}–{min(i+batch_size, len(records))}")
            except Exception as e:
                print(f"  ✗ Batch {i//batch_size + 1} failed: {e}")
        
        print(f"✓ Successfully loaded {total_inserted}/{len(records)} rows into {table_name}")
        return total_inserted
    except Exception as e:
        print(f"✗ Error loading {table_name}: {e}")
        return 0

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CLEARING AND RELOADING SUPABASE TABLES")
    print("="*70)
    
    client = get_supabase_client()
    
    # Clear tables
    print("\n[1/4] Clearing standard_table...")
    clear_table(client, 'standard_table')
    
    print("\n[2/4] Clearing category_table...")
    clear_table(client, 'category_table')
    
    # Reload tables
    print("\n[3/4] Reloading standard_table...")
    load_table(client, 'etl/data/staged/standard_table.csv', 'standard_table')
    
    print("\n[4/4] Reloading category_table...")
    load_table(client, 'etl/data/staged/category_table.csv', 'category_table')
    
    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    response = client.table('standard_table').select('ticker').execute()
    df = pd.DataFrame(response.data)
    print(f"\nstandard_table: {len(df)} rows")
    print(f"Companies: {sorted(df['ticker'].unique())}")
    print(f"Distribution:\n{df['ticker'].value_counts().sort_index()}")
    
    print("\n✓ Database reload complete!")
