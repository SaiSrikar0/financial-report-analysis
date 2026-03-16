"""ETL load layer for Supabase/PostgreSQL."""

import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

BATCH_SIZE = 100


def get_supabase_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL/SUPABASE_KEY (or supabase_url/supabase_key) in .env")

    return create_client(url, key)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    data = df.copy()
    for col in data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
        data[col] = data[col].astype(str)
    data = data.where(pd.notnull(data), other=None)
    return data.to_dict(orient="records")


def _batch_upsert(client, table_name: str, records: list[dict]):
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        client.table(table_name).upsert(batch).execute()
        end = min(i + BATCH_SIZE, total)
        print(f"Upserted rows {i + 1}-{end} of {total} into {table_name}")


def load_to_supabase(staged_path: str, table_name: str):
    if not os.path.isabs(staged_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        staged_path = os.path.join(project_root, staged_path)

    if not os.path.exists(staged_path):
        raise FileNotFoundError(f"File not found: {staged_path}")

    print(f"Loading {staged_path} -> {table_name}")
    df = pd.read_csv(staged_path)
    records = _df_to_records(df)

    client = get_supabase_client()
    _batch_upsert(client, table_name, records)
    print(f"Finished loading {len(records)} rows into '{table_name}'")


if __name__ == "__main__":
    from transform import transform_data

    standard_path, category_path = transform_data()

    print("\n--- Loading Standard Table (for ML/SVR) ---")
    load_to_supabase(standard_path, "standard_table")

    print("\n--- Loading Category Table (for LLM Recommendations) ---")
    load_to_supabase(category_path, "category_table")