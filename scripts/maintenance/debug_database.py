"""Debug script to check database contents."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY")
    exit(1)

print(f"Connecting to: {url}")

# Try with anon key first
print("\n--- Using ANON KEY ---")
client = create_client(url, key)

try:
    response = client.table("uploaded_files").select("*").execute()
    print(f"✓ Uploaded files (anon key): {len(response.data)} records")
    for row in response.data[:3]:
        print(f"  - user_id: {row.get('user_id', 'N/A')[:20]}...")
        print(f"    ticker: {row.get('ticker', 'N/A')}")
        print(f"    filename: {row.get('filename', 'N/A')}")
        print(f"    file_content type: {type(row.get('file_content', 'N/A'))}")
        if isinstance(row.get('file_content'), list):
            print(f"    file_content records: {len(row['file_content'])}")
except Exception as e:
    print(f"✗ Error with anon key: {e}")

# Try with service role key
if service_key:
    print("\n--- Using SERVICE ROLE KEY ---")
    admin_client = create_client(url, service_key)
    try:
        response = admin_client.table("uploaded_files").select("*").execute()
        print(f"✓ Uploaded files (service key): {len(response.data)} records")
        for row in response.data[:3]:
            print(f"  - user_id: {row.get('user_id', 'N/A')[:20]}...")
            print(f"    ticker: {row.get('ticker', 'N/A')}")
            print(f"    filename: {row.get('filename', 'N/A')}")
            print(f"    file_content type: {type(row.get('file_content', 'N/A'))}")
            if isinstance(row.get('file_content'), list):
                print(f"    file_content records: {len(row['file_content'])}")
    except Exception as e:
        print(f"✗ Error with service key: {e}")
else:
    print("⚠️  Service role key not in .env")
