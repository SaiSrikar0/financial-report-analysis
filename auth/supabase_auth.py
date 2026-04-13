"""Supabase Auth module for FinCast portal."""

import os
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


def _get_client():
    """Create a fresh Supabase client, optionally with authenticated session."""
    url = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
    key = os.getenv("SUPABASE_KEY") or os.getenv("supabase_key")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL/SUPABASE_KEY in .env")
    
    client = create_client(url, key)
    
    # If user is authenticated, set their session on the client
    # This ensures RLS policies use auth.uid() correctly for subsequent requests
    try:
        if "session" in st.session_state and st.session_state["session"]:
            session = st.session_state["session"]
            client.auth.set_session(session.access_token, session.refresh_token)
    except Exception:
        pass
    
    return client


def login(email: str, password: str) -> bool:
    try:
        res = _get_client().auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        st.session_state["user"] = res.user
        st.session_state["session"] = res.session
        st.session_state["user_id"] = res.user.id
        st.session_state["user_email"] = res.user.email
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False


def signup(email: str, password: str) -> bool:
    try:
        _get_client().auth.sign_up({"email": email, "password": password})
        st.success("Account created! Check your email to verify before logging in.")
        return True
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return False


def logout():
    try:
        _get_client().auth.sign_out()
    except Exception:
        pass
    for key in ["user", "session", "user_id", "user_email", "recommendations"]:
        st.session_state.pop(key, None)


def restore_session() -> bool:
    """
    Restore session from Supabase on page reload.
    Allows users to stay logged in after page refresh.
    """
    if "user" in st.session_state:
        return True  # Already authenticated in current session
    
    try:
        user = _get_client().auth.get_user()
        if user and user.user:
            st.session_state["user"] = user.user
            st.session_state["user_id"] = user.user.id
            st.session_state["user_email"] = user.user.email
            return True
    except Exception as e:
        print(f"[AUTH] restore_session failed: {type(e).__name__}: {e}")
    return False


def is_authenticated() -> bool:
    if "user" not in st.session_state:
        # Try to restore from Supabase
        restored = restore_session()
        if not restored:
            print(f"[AUTH] is_authenticated: no cached user, restore_session returned False")
        return restored
    try:
        session = st.session_state.get("session")
        if session and hasattr(session, "refresh_token"):
            try:
                refreshed = _get_client().auth.refresh_session(session.refresh_token)
                st.session_state["session"] = refreshed.session
            except Exception as e:
                print(f"[AUTH] Session refresh failed: {type(e).__name__}: {e}")
                logout()
                return False
        return True
    except Exception as e:
        print(f"[AUTH] is_authenticated exception: {type(e).__name__}: {e}")
        logout()
        return False


def get_user_id() -> str:
    return st.session_state.get("user_id", "predefined")


def get_user_email() -> str:
    return st.session_state.get("user_email", "")


def render_login_page():
    """Render the login/signup page with FinCast branding."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        :root {
            --bg-0: #080f2c; --bg-1: #0d1844;
            --panel: rgba(20,33,78,0.82); --panel-border: rgba(154,196,255,0.22);
            --text-strong: #eaf1ff; --text-soft: #9fb3db;
            --accent-a: #44d1ff; --accent-b: #2ee6a8; --accent-c: #3c82ff;
        }
        .stApp {
            background: radial-gradient(1150px 460px at 18% -8%, rgba(89,127,255,0.35), transparent 65%),
                        radial-gradient(900px 500px at 98% 8%, rgba(46,230,168,0.18), transparent 62%),
                        linear-gradient(145deg, var(--bg-0), var(--bg-1));
            color: var(--text-strong);
            font-family: 'Manrope', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align:center; padding: 40px 0 20px;'>
                <div style='display:inline-flex; align-items:center; gap:12px; margin-bottom:12px;'>
                    <div style='width:48px; height:48px; border-radius:14px;
                                background: linear-gradient(155deg, #44d1ff, #2ee6a8);
                                display:grid; place-items:center;
                                font-weight:900; font-size:1.2rem; color:#052135;
                                box-shadow:0 0 20px rgba(68,209,255,0.4);'>FC</div>
                    <span style='font-size:2rem; font-weight:800; color:#eaf1ff;
                                 font-family:Space Grotesk,sans-serif;'>FinCast</span>
                </div>
                <p style='color:#9fb3db; font-size:0.95rem; margin:0;'>
                    Financial Intelligence Platform
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input(
                "Password", type="password", key="login_pw", placeholder="••••••••"
            )
            if st.button("Login", use_container_width=True, type="primary"):
                if email and password:
                    if login(email, password):
                        st.rerun()
                else:
                    st.warning("Please enter email and password.")

        with tab_signup:
            s_email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            s_password = st.text_input(
                "Password (min 6 chars)", type="password", key="signup_pw"
            )
            if st.button("Create Account", use_container_width=True):
                if s_email and s_password:
                    signup(s_email, s_password)
                else:
                    st.warning("Please fill in all fields.")