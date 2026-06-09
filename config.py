import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import streamlit as st
except ImportError:
    st = None


def get_secret(key, default=None):
    if st:
        try:
            return st.secrets[key]
        except Exception:
            pass

    return os.getenv(key, default)