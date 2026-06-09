from config import get_secret

print("SUPABASE_URL =", get_secret("SUPABASE_URL"))
print("SUPABASE_KEY =", bool(get_secret("SUPABASE_KEY")))
print("SERVICE_ROLE =", bool(get_secret("SUPABASE_SERVICE_ROLE_KEY")))