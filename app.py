"""
FinCast Dashboard — app.py
Integrates Phases 1-6: Auth, ETL Upload, Historical Analysis,
SVR Predictions, SHAP Explainability, LLM Recommendations.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FinCast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth ───────────────────────────────────────────────────────────────────────
from auth.supabase_auth import (
    render_login_page,
    is_authenticated,
    logout,
    get_user_id,
    get_user_email,
)

if not is_authenticated():
    render_login_page()
    st.stop()

user_id = get_user_id()
user_email = get_user_email()

# ── Lazy imports (only after auth passes) ─────────────────────────────────────
from analysis.data_connection import (
    get_supabase_client,
    load_user_standard_table,
    get_user_tickers,
)
from analysis.recommendation_engine import (
    load_analysis_bundle_from_reports,
    generate_recommendations,
)

# ── Theme ──────────────────────────────────────────────────────────────────────
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-0: #080f2c;
            --bg-1: #0d1844;
            --panel: rgba(20, 33, 78, 0.82);
            --panel-border: rgba(154, 196, 255, 0.22);
            --text-strong: #eaf1ff;
            --text-soft: #9fb3db;
            --accent-a: #44d1ff;
            --accent-b: #2ee6a8;
            --accent-c: #3c82ff;
            --danger: #ff7b8f;
            --shadow: 0 14px 32px rgba(0, 0, 0, 0.35);
        }

        .stApp {
            background:
                radial-gradient(1150px 460px at 18% -8%, rgba(89, 127, 255, 0.35), transparent 65%),
                radial-gradient(900px 500px at 98% 8%, rgba(46, 230, 168, 0.18), transparent 62%),
                linear-gradient(145deg, var(--bg-0), var(--bg-1));
            color: var(--text-strong);
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 16, 43, 0.98), rgba(12, 20, 55, 0.94));
            border-right: 1px solid rgba(161, 196, 255, 0.18);
            box-shadow: inset -1px 0 0 rgba(78, 110, 198, 0.24);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-strong) !important;
            font-family: 'Space Grotesk', sans-serif;
        }

        .brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            background: linear-gradient(90deg, rgba(14, 26, 72, 0.9), rgba(11, 21, 61, 0.9));
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
            box-shadow: var(--shadow);
            animation: drop-in 0.5s ease;
        }

        .brand-left {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            letter-spacing: 0.2px;
            font-size: 1.26rem;
            font-family: 'Space Grotesk', sans-serif;
        }

        .brand-logo {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(155deg, var(--accent-a), var(--accent-b));
            box-shadow: 0 0 16px rgba(68, 209, 255, 0.36);
            display: grid;
            place-items: center;
            color: #052135;
            font-size: 1rem;
            font-weight: 900;
        }

        .brand-icons {
            display: flex;
            gap: 8px;
            color: var(--text-soft);
            font-size: 0.92rem;
        }

        .metric-card {
            background: linear-gradient(155deg, rgba(20, 33, 78, 0.88), rgba(27, 44, 100, 0.7));
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 14px 16px;
            min-height: 112px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            animation: rise 0.65s ease;
        }

        .metric-card::after {
            content: "";
            position: absolute;
            inset: auto -15% -55% -15%;
            height: 85px;
            background: radial-gradient(circle at center, rgba(68, 209, 255, 0.24), transparent 65%);
        }

        .metric-label {
            color: var(--text-soft);
            font-size: 0.85rem;
            letter-spacing: 0.35px;
            text-transform: uppercase;
            font-family: 'Space Grotesk', sans-serif;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            color: var(--text-strong);
        }

        .panel {
            background: linear-gradient(170deg, rgba(18, 30, 73, 0.9), rgba(14, 25, 63, 0.88));
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 10px 12px 8px;
            box-shadow: var(--shadow);
            animation: rise 0.65s ease;
        }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.02rem;
            color: var(--text-strong);
            margin: 2px 0 8px;
            letter-spacing: 0.25px;
        }

        .subtle {
            color: var(--text-soft);
            font-size: 0.84rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow);
            background: rgba(9, 18, 52, 0.72);
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes drop-in {
            from { opacity: 0; transform: translateY(-8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .metric-value { font-size: 1.45rem; }
            .brand-row { padding: 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


# ── Helper functions (all unchanged from original) ─────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_svr_predictions() -> dict:
    reports_path = Path("analysis/reports")
    data = {}
    metrics_file = reports_path / "svr_evaluation_metrics.csv"
    if metrics_file.exists():
        data["metrics"] = pd.read_csv(metrics_file).iloc[0].to_dict()
    pred_file = reports_path / "svr_future_predictions.csv"
    if pred_file.exists():
        data["predictions"] = pd.read_csv(pred_file)
    test_file = reports_path / "svr_test_predictions.csv"
    if test_file.exists():
        data["test_predictions"] = pd.read_csv(test_file)
    params_file = reports_path / "svr_best_params.csv"
    if params_file.exists():
        data["params"] = pd.read_csv(params_file).iloc[0].to_dict()
    return data


@st.cache_data
def load_shap_data() -> dict:
    reports_path = Path("analysis/reports")
    data = {}
    global_file = reports_path / "phase_5_shap_global_importance.csv"
    if global_file.exists():
        data["global_importance"] = pd.read_csv(global_file)
    local_file = reports_path / "phase_5_shap_local_explanations.csv"
    if local_file.exists():
        data["local_explanations"] = pd.read_csv(local_file)
    future_file = reports_path / "phase_5_shap_future_predictions.csv"
    if future_file.exists():
        data["future_predictions"] = pd.read_csv(future_file)
    return data


def metric_tile(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def infer_ticker_from_upload(raw_records: list, filename: str) -> tuple[str, str]:
    """Infer ticker from raw records first, then fall back to filename stem."""
    candidate_keys = [
        "ticker",
        "symbol",
        "stock_symbol",
        "company_ticker",
        "security_symbol",
    ]

    counts = {}
    for rec in raw_records or []:
        if not isinstance(rec, dict):
            continue
        for key in candidate_keys:
            val = rec.get(key)
            if val is None:
                continue
            txt = str(val).strip().upper()
            if not txt:
                continue
            if len(txt) <= 12 and " " not in txt:
                counts[txt] = counts.get(txt, 0) + 1

    if counts:
        ticker = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return ticker, "file data"

    stem = Path(filename).stem.upper()
    parts = [p for p in "".join(ch if ch.isalnum() else " " for ch in stem).split() if p]
    if parts:
        return parts[0][:12], "filename"

    return "UNKNOWN", "fallback"


def has_svr_predictions(ticker: str) -> bool:
    """Check if SVR predictions exist for a ticker."""
    svr_path = "analysis/reports/svr_future_predictions.csv"
    if not os.path.exists(svr_path):
        return False
    try:
        df = pd.read_csv(svr_path)
        return len(df[df["ticker"].str.upper() == ticker.upper()]) > 0
    except:
        return False


def run_svr_training_button(
    ticker: str,
    standard_records: list = None,
    category_records: list = None,
    user_id: str = None,
    navigate_to_recommendations: bool = False,
) -> bool:
    """
    Render a training button and run SVR+SHAP pipeline when clicked.
    Shows detailed progress and clear error messages.
    Returns True if training was run successfully.
    """
    if st.button(f"🚀 Train SVR for {ticker}", type="primary", use_container_width=True, key=f"train_svr_{ticker}"):
        with st.status("🔄 Training SVR Model...", expanded=True) as status_analysis:
            try:
                from analysis.auto_analysis import run_uploaded_analysis_pipeline, display_analysis_progress
                
                st.write("📊 Initializing training pipeline...")
                supabase = get_supabase_client()
                final_user_id = user_id or get_user_id()
                
                st.write(f"🎯 Training for: **{ticker}**")
                st.write("📈 Loading and engineering data...")
                
                # Run the unified training pipeline
                result = run_uploaded_analysis_pipeline(
                    ticker,
                    final_user_id,
                    supabase,
                    standard_records=standard_records,
                    category_records=category_records,
                )
                
                # Display progress
                display_analysis_progress(result, st)
                
                if result["success"]:
                    status_analysis.update(label="✅ Training Complete!", state="complete")
                    st.balloons()
                    st.success(f"✅ **{ticker}** trained successfully!")
                    st.info(f"📊 Results saved to `analysis/reports/`")

                    if navigate_to_recommendations:
                        st.session_state["nav_page"] = "🤖 AI Recommendations"
                        st.session_state["preferred_ticker"] = ticker
                        st.rerun()

                    return True
                else:
                    # Detailed error display
                    status_analysis.update(label="❌ Training Failed", state="error")
                    
                    st.error("❌ **Training could not complete**")
                    
                    # Show messages without nested expander
                    if result.get("messages"):
                        st.write("**Details:**")
                        for msg in result["messages"]:
                            if "CRITICAL" in msg or "❌" in msg:
                                st.error(f"• {msg}")
                            elif "⚠️" in msg:
                                st.warning(f"• {msg}")
                            else:
                                st.info(f"• {msg}")
                    
                    st.markdown("---")
                    st.markdown("**🔧 Troubleshooting:**")
                    st.markdown("""
- Ensure your CSV has all required columns: `date`, `ticker`, `revenue`, `net_income`, `operating_income`, `total_assets`, `total_liabilities`, `operating_cashflow`
- Check that data has at least 3 periods (years/quarters)
- Make sure numeric values are valid numbers (not text like "$1,000")
- Try uploading the CSV file again and ensure the ticker matches the company name
                    """)
                    return False
            except Exception as e:
                status_analysis.update(label="✗ Training Error", state="error")
                st.error(f"**Pipeline Error:** {e}")
                
                st.code(str(e), language="python")
                
                st.warning("Please check your data and try again. Contact support if issue persists.")
                return False
    return False


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=16, b=14),
        font=dict(family="Manrope, sans-serif", color="#dce7ff", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#a9bfe7"),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(112,145,212,0.22)",
            zeroline=False, linecolor="rgba(112,145,212,0.32)",
            tickfont=dict(color="#a9bfe7"),
            title=dict(font=dict(color="#a9bfe7")),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(112,145,212,0.22)",
            zeroline=False, linecolor="rgba(112,145,212,0.32)",
            tickfont=dict(color="#a9bfe7"),
            title=dict(font=dict(color="#a9bfe7")),
        ),
    )
    return fig


def build_light_recommendation(filtered: pd.DataFrame, ticker: str, year: int) -> dict:
    company_df = filtered[filtered["ticker"] == ticker].sort_values("date")
    row = company_df[company_df["date"].dt.year == year]
    if row.empty:
        return {"status": "Unavailable", "strengths": [], "risks": [], "actions": []}

    row = row.iloc[-1]
    prev_row = company_df[company_df["date"].dt.year < year].tail(1)

    revenue = float(row.get("revenue", 0))
    profit_margin = float(row.get("profit_margin", 0))
    debt_ratio = float(row.get("debt_ratio", 0))
    operating_cashflow = float(row.get("operating_cashflow", 0))
    net_income = float(row.get("net_income", 0))
    cash_ratio = operating_cashflow / abs(net_income) if net_income else 0.0

    revenue_growth = None
    if not prev_row.empty:
        prev_revenue = float(prev_row.iloc[0].get("revenue", 0))
        if prev_revenue:
            revenue_growth = ((revenue - prev_revenue) / abs(prev_revenue)) * 100

    strengths, risks, actions = [], [], []

    if profit_margin >= 0.15:
        strengths.append(f"Healthy profitability with ~{profit_margin*100:.1f}% margin.")
    elif profit_margin >= 0.05:
        strengths.append(f"Positive profitability ({profit_margin*100:.1f}%), room to expand.")
        actions.append("Improve gross margin mix and operating efficiency.")
    else:
        risks.append(f"Weak profitability ({profit_margin*100:.1f}%).")
        actions.append("Prioritize cost control and low-margin segment review.")

    if debt_ratio <= 0.4:
        strengths.append(f"Manageable leverage (debt ratio {debt_ratio*100:.1f}%).")
    elif debt_ratio <= 0.6:
        risks.append(f"Moderate leverage pressure (debt ratio {debt_ratio*100:.1f}%).")
        actions.append("Limit new debt and improve debt servicing from operating cashflow.")
    else:
        risks.append(f"High leverage risk (debt ratio {debt_ratio*100:.1f}%).")
        actions.append("Create a debt reduction plan and consider refinancing.")

    if cash_ratio >= 1.0:
        strengths.append("Operating cashflow sufficiently supports earnings.")
    else:
        risks.append("Cash conversion is weak relative to reported net income.")
        actions.append("Tighten receivables and working-capital cycles.")

    if revenue_growth is not None:
        if revenue_growth > 8:
            strengths.append(f"Strong YoY revenue momentum (+{revenue_growth:.1f}%).")
        elif revenue_growth < 0:
            risks.append(f"Revenue contraction detected ({revenue_growth:.1f}% YoY).")
            actions.append("Investigate demand, pricing, and product-mix drivers.")

    if net_income > 0 and profit_margin >= 0.05:
        status = "Profitable"
    elif profit_margin >= 0:
        status = "Break-Even"
    else:
        status = "Loss"

    return {"status": status, "strengths": strengths, "risks": risks, "actions": actions}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">FC</div>
                <div>FinCast</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Logged in as `{user_email or user_id[:12]}`")
    st.divider()

    nav_options = [
        "📊 Dashboard",
        "📤 Upload Data",
        "🤖 AI Recommendations",
    ]
    current_page = st.session_state.get("nav_page", "📊 Dashboard")
    if current_page not in nav_options:
        current_page = "📊 Dashboard"

    page = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(current_page),
        key="nav_page",
        label_visibility="collapsed",
    )
    st.divider()

    # Existing sidebar controls (only shown on Dashboard page)
    if page == "📊 Dashboard":
        data_path = "data/raw/financial_data_raw.json"
        _df_raw = load_data(data_path)

        if not _df_raw.empty:
            st.markdown("### Filters")
            tickers_available = sorted(_df_raw["ticker"].dropna().unique().tolist())
            selected_tickers = st.multiselect(
                "Company", tickers_available, default=tickers_available
            )
            min_year = int(_df_raw["date"].dt.year.min())
            max_year = int(_df_raw["date"].dt.year.max())
            year_range = st.slider(
                "Year Range", min_year, max_year,
                (max(min_year, 2015), max_year),
            )
            data_source = st.radio(
                "Data Source", ["Internal", "External"], horizontal=True
            )
            
            metric_choice = st.selectbox(
                "Select Metric",
                ["Revenue", "Net Income", "Operating Income",
                 "Cashflow", "Profit Margin", "Debt Ratio"],
                index=0,
            )
            
            # Train SVR section
            st.divider()
            st.markdown("### 🚀 Train SVR")
            
            # Get list of uploaded files for this user
            try:
                supabase = get_supabase_client()
                uploaded_response = supabase.table("uploaded_files").select(
                    "ticker, filename"
                ).eq("user_id", user_id).limit(50).execute()
                uploaded_files = uploaded_response.data if uploaded_response and uploaded_response.data else []
                
                if uploaded_files:
                    # Create options for dropdown
                    options = [f"{f['ticker']} - {f['filename'][:30]}" for f in uploaded_files]
                    selected_option = st.selectbox(
                        "Select file to train",
                        options,
                        key="train_svr_dropdown"
                    )
                    
                    # Extract ticker from selected option
                    if selected_option:
                        selected_ticker = selected_option.split(" - ")[0]
                        
                        # Show training button using unified function
                        run_svr_training_button(
                            ticker=selected_ticker,
                            user_id=user_id
                        )
                else:
                    st.info("📤 Upload data first to train SVR")
            except Exception as e:
                st.warning(f"Could not load uploaded files: {str(e)[:50]}")

    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD (original 3 tabs, fully preserved)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    # Brand header
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">FC</div>
                <div>FinCast</div>
            </div>
            <div class="brand-icons">Analytics Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── NEW: Display User-Uploaded Data Analytics ──────────────────────────────
    from analysis.uploaded_data_analytics import (
        display_uploaded_files_section,
        display_svr_analysis_for_ticker,
        display_shap_analysis_for_ticker,
    )
    
    st.subheader("Your Data Analytics")
    
    uploaded_tab, predefined_tab = st.tabs(["📁 Uploaded Data", "📊 Predefined Reference"])
    
    with uploaded_tab:
        display_uploaded_files_section(user_id)
        st.divider()
        
        # Show SVR + SHAP for user's uploaded tickers
        user_tickers = get_user_tickers(user_id)
        if user_tickers:
            selected_user_ticker = st.selectbox(
                "Analyze uploaded ticker",
                user_tickers,
                key="user_ticker_select"
            )
            
            # Check if this ticker has SVR predictions
            ticker_has_predictions = has_svr_predictions(selected_user_ticker)
            
            if not ticker_has_predictions:
                st.warning(f"⚠️ **SVR Analysis Not Available** for {selected_user_ticker}")
                st.write(f"Click the button below to train the analysis pipeline for this ticker.")
                run_svr_training_button(selected_user_ticker, user_id=user_id)
            else:
                col_svr, col_shap = st.columns(2)
                with col_svr:
                    display_svr_analysis_for_ticker(selected_user_ticker)
                with col_shap:
                    display_shap_analysis_for_ticker(selected_user_ticker)
        else:
            st.info("No uploaded files yet. Go to 📤 Upload Data to get started.")
    
    with predefined_tab:
        st.markdown("**Predefined Reference Data Dashboard**")
        st.info("Analysis of reference tickers (AAPL, MSFT, GOOGL, AMZN)")
        st.divider()

    data_path = "data/raw/financial_data_raw.json"
    df = load_data(data_path)
    if df.empty:
        st.stop()

    # Apply sidebar filters (defined above in sidebar block)
    if "selected_tickers" not in dir():
        selected_tickers = sorted(df["ticker"].dropna().unique().tolist())
        year_range = (int(df["date"].dt.year.min()), int(df["date"].dt.year.max()))
        data_source = "Internal"
        metric_choice = "Revenue"

    if not selected_tickers:
        st.warning("Select at least one company to render the dashboard.")
        st.stop()

    filtered = df[df["ticker"].isin(selected_tickers)].copy()
    filtered = filtered[
        (filtered["date"].dt.year >= year_range[0])
        & (filtered["date"].dt.year <= year_range[1])
    ]

    if "operating_expenses" not in filtered.columns:
        filtered["operating_expenses"] = (
            filtered["revenue"] - filtered["operating_income"]
        ).clip(lower=0)

    if filtered.empty:
        st.warning("No records found for selected filters.")
        st.stop()

    filtered["profit_margin"] = filtered["net_income"] / filtered["revenue"].replace(0, pd.NA)
    filtered["debt_ratio"] = (
        filtered["total_liabilities"] / filtered["total_assets"].replace(0, pd.NA)
    )

    tab_historical, tab_predictions, tab_explainability = st.tabs(
        ["📊 Historical Analysis", "🎯 SVR Predictions", "🔍 SHAP Explainability"]
    )

    # ── TAB 1: HISTORICAL ANALYSIS ─────────────────────────────────────────────
    with tab_historical:
        total_revenue = filtered["revenue"].sum()
        total_net_income = filtered["net_income"].sum()
        avg_profit_margin = filtered["profit_margin"].mean(skipna=True)
        avg_debt_ratio = filtered["debt_ratio"].mean(skipna=True)

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            metric_tile("Revenue", f"${total_revenue/1_000_000:.1f}M")
        with c2:
            metric_tile("Net Income", f"${total_net_income/1_000_000:.1f}M")
        with c3:
            metric_tile("Profit Margin", f"{(avg_profit_margin or 0)*100:.1f}%")
        with c4:
            metric_tile("Debt Ratio", f"{(avg_debt_ratio or 0)*100:.1f}%")

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Revenue Trends by Company</div>",
            unsafe_allow_html=True,
        )
        line_fig = px.line(
            filtered.sort_values("date"), x="date", y="revenue",
            color="ticker", markers=True,
            color_discrete_sequence=["#3c82ff", "#44d1ff", "#2ee6a8", "#9d8dff", "#ff9c66"],
        )
        line_fig.update_traces(line=dict(width=3), marker=dict(size=7))
        style_figure(line_fig)
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3, gap="medium")

        with b1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Revenue vs Expenses</div>",
                unsafe_allow_html=True,
            )
            rev_exp = (
                filtered.groupby("ticker", as_index=False)[
                    ["revenue", "operating_expenses"]
                ].sum()
                .melt(
                    id_vars="ticker",
                    value_vars=["revenue", "operating_expenses"],
                    var_name="metric",
                    value_name="value",
                )
            )
            fig_rev_exp = px.bar(
                rev_exp, x="ticker", y="value", color="metric",
                barmode="group",
                color_discrete_map={
                    "revenue": "#3c82ff", "operating_expenses": "#2ee6a8"
                },
            )
            style_figure(fig_rev_exp)
            st.plotly_chart(fig_rev_exp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Assets vs Liabilities</div>",
                unsafe_allow_html=True,
            )
            ass_liab = (
                filtered.groupby("ticker", as_index=False)[
                    ["total_assets", "total_liabilities"]
                ].sum()
                .melt(
                    id_vars="ticker",
                    value_vars=["total_assets", "total_liabilities"],
                    var_name="metric",
                    value_name="value",
                )
            )
            fig_ass_liab = px.bar(
                ass_liab, x="ticker", y="value", color="metric",
                barmode="group",
                color_discrete_map={
                    "total_assets": "#44d1ff", "total_liabilities": "#2ee6a8"
                },
            )
            style_figure(fig_ass_liab)
            st.plotly_chart(fig_ass_liab, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b3:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Cashflow vs Income</div>",
                unsafe_allow_html=True,
            )
            cf_income = (
                filtered.groupby("ticker", as_index=False)[
                    ["operating_cashflow", "net_income"]
                ].sum()
                .melt(
                    id_vars="ticker",
                    value_vars=["operating_cashflow", "net_income"],
                    var_name="metric",
                    value_name="value",
                )
            )
            fig_cf = px.bar(
                cf_income, x="ticker", y="value", color="metric",
                barmode="group",
                color_discrete_map={
                    "operating_cashflow": "#3c82ff", "net_income": "#44d1ff"
                },
            )
            style_figure(fig_cf)
            st.plotly_chart(fig_cf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        low, high = st.columns([1.1, 2.2], gap="medium")

        with low:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Financial Landscape</div>",
                unsafe_allow_html=True,
            )
            scatter3d = px.scatter_3d(
                filtered, x="revenue", y="debt_ratio", z="net_income",
                color="ticker", size="total_assets", hover_name="ticker",
                opacity=0.92,
                color_discrete_sequence=["#3c82ff", "#44d1ff", "#2ee6a8", "#9d8dff", "#ff9c66"],
            )
            scatter3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                    bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(112,145,212,0.2)", title="Revenue", color="#a9bfe7"),
                    yaxis=dict(gridcolor="rgba(112,145,212,0.2)", title="Debt Ratio", color="#a9bfe7"),
                    zaxis=dict(gridcolor="rgba(112,145,212,0.2)", title="Net Income", color="#a9bfe7"),
                ),
                legend=dict(font=dict(color="#a9bfe7")),
                font=dict(family="Manrope, sans-serif", color="#dce7ff"),
            )
            st.plotly_chart(scatter3d, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with high:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Financial Overview</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='subtle'>Source: {data_source}  |  "
                f"Focus Metric: {metric_choice}  |  Rows: {len(filtered)}</div>",
                unsafe_allow_html=True,
            )
            table = filtered.sort_values(
                ["date", "ticker"], ascending=[False, True]
            ).copy()
            table["year"] = table["date"].dt.year
            table = table[
                ["year", "ticker", "revenue", "net_income",
                 "operating_expenses", "total_assets",
                 "total_liabilities", "operating_cashflow"]
            ].rename(
                columns={
                    "ticker": "company", "operating_expenses": "expenses",
                    "total_assets": "assets", "total_liabilities": "liabilities",
                    "operating_cashflow": "cashflow",
                }
            )
            styled = table.style.format(
                {
                    "revenue": "${:,.0f}", "net_income": "${:,.0f}",
                    "expenses": "${:,.0f}", "assets": "${:,.0f}",
                    "liabilities": "${:,.0f}", "cashflow": "${:,.0f}",
                }
            )
            st.dataframe(styled, use_container_width=True, height=350)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: SVR PREDICTIONS (unchanged) ─────────────────────────────────────
    with tab_predictions:
        svr_data = load_svr_predictions()

        if not svr_data:
            st.warning("Phase 4 SVR data not available. Run `python run.py 4`.")
        else:
            if "metrics" in svr_data:
                metrics = svr_data["metrics"]
                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_tile("MAE", f"{metrics.get('mae', 0):.2f}%")
                with m2:
                    metric_tile("RMSE", f"{metrics.get('rmse', 0):.2f}%")
                with m3:
                    metric_tile("R² Score", f"{metrics.get('r2', 0):.4f}")

            if "params" in svr_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Best Model Hyperparameters</div>",
                    unsafe_allow_html=True,
                )
                params = svr_data["params"]
                param_df = pd.DataFrame(
                    [
                        {"Parameter": "Kernel",             "Value": params.get("svr__kernel", "rbf")},
                        {"Parameter": "C (regularization)", "Value": f"{params.get('svr__C', 1):.2f}"},
                        {"Parameter": "Epsilon",            "Value": f"{params.get('svr__epsilon', 0.01):.4f}"},
                        {"Parameter": "Gamma",              "Value": str(params.get("svr__gamma", "scale"))},
                    ]
                )
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if "predictions" in svr_data:
                preds = svr_data["predictions"]
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Future Growth Rate Predictions (Next Fiscal Year)</div>",
                    unsafe_allow_html=True,
                )
                pred_cols = [
                    "ticker", "predicted_growth_rate", "target_growth_rate",
                    "gap_vs_target", "gap_status",
                    "confidence_lower_95", "confidence_upper_95",
                ]
                pred_table = preds[[c for c in pred_cols if c in preds.columns]].copy()
                pred_table.columns = [
                    c.replace("_", " ").title() for c in pred_table.columns
                ]
                float_cols = [
                    c for c in pred_table.columns
                    if c not in ("Ticker", "Gap Status")
                ]
                fmt = {c: "{:.2f}%" for c in float_cols}
                st.dataframe(
                    pred_table.style.format(fmt, na_rep="N/A"),
                    use_container_width=True,
                    hide_index=True,
                )

                fig_ci = go.Figure()
                for _, row in preds.iterrows():
                    ticker = row["ticker"]
                    predicted = row["predicted_growth_rate"]
                    ci_lower = row.get("confidence_lower_95", predicted - 10)
                    ci_upper = row.get("confidence_upper_95", predicted + 10)
                    fig_ci.add_trace(
                        go.Scatter(
                            x=[ci_lower, ci_upper], y=[ticker, ticker],
                            mode="lines",
                            line=dict(width=10, color="#44d1ff"),
                            name=f"{ticker} (95% CI)",
                        )
                    )
                    fig_ci.add_trace(
                        go.Scatter(
                            x=[predicted], y=[ticker], mode="markers",
                            marker=dict(size=12, color="#ff9c66"),
                            name=f"{ticker} (Predicted)",
                            showlegend=False,
                        )
                    )
                fig_ci.add_vline(
                    x=10, line_dash="dash", line_color="#2ee6a8",
                    annotation_text="Target (10%)",
                )
                style_figure(fig_ci)
                fig_ci.update_xaxes(title_text="Growth Rate (%)")
                fig_ci.update_yaxes(title_text="Ticker")
                st.plotly_chart(fig_ci, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if "test_predictions" in svr_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Test Set: Predicted vs Actual Growth Rates</div>",
                    unsafe_allow_html=True,
                )
                test_preds = svr_data["test_predictions"]
                actual_col = "actual" if "actual" in test_preds.columns else None
                predicted_col = "predicted" if "predicted" in test_preds.columns else None

                if actual_col and predicted_col:
                    color_col = "ticker" if "ticker" in test_preds.columns else None
                    fig_pva = px.scatter(
                        test_preds, x=actual_col, y=predicted_col,
                        color=color_col,
                        hover_name=color_col,
                        color_discrete_sequence=[
                            "#3c82ff", "#44d1ff", "#2ee6a8", "#9d8dff", "#ff9c66"
                        ],
                    )
                    min_val = min(
                        test_preds[actual_col].min(),
                        test_preds[predicted_col].min(),
                    )
                    max_val = max(
                        test_preds[actual_col].max(),
                        test_preds[predicted_col].max(),
                    )
                    fig_pva.add_trace(
                        go.Scatter(
                            x=[min_val, max_val], y=[min_val, max_val],
                            mode="lines",
                            line=dict(dash="dash", color="#a9bfe7", width=2),
                            name="Perfect Prediction",
                        )
                    )
                    style_figure(fig_pva)
                    fig_pva.update_xaxes(title_text="Actual Growth Rate (%)")
                    fig_pva.update_yaxes(title_text="Predicted Growth Rate (%)")
                    st.plotly_chart(fig_pva, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: SHAP EXPLAINABILITY (unchanged) ─────────────────────────────────
    with tab_explainability:
        shap_data = load_shap_data()

        if not shap_data:
            st.warning("Phase 5 SHAP data not available. Run `python run.py 5`.")
        else:
            if "global_importance" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Global Feature Importance (Mean Absolute SHAP)</div>",
                    unsafe_allow_html=True,
                )
                global_imp = shap_data["global_importance"].sort_values(
                    "mean_abs_shap", ascending=True
                )
                fig_global = px.bar(
                    global_imp, x="mean_abs_shap", y="feature",
                    color="mean_abs_shap", orientation="h",
                    color_continuous_scale=["#3c82ff", "#44d1ff", "#2ee6a8"],
                )
                style_figure(fig_global)
                fig_global.update_xaxes(title_text="Mean Absolute SHAP Value")
                fig_global.update_yaxes(title_text="Feature")
                st.plotly_chart(fig_global, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if "local_explanations" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Local Feature Contributions by Ticker</div>",
                    unsafe_allow_html=True,
                )
                local_exp = shap_data["local_explanations"]
                tickers_in_shap = sorted(local_exp["ticker"].unique())
                selected_ticker = st.selectbox("Select Ticker", tickers_in_shap)

                ticker_exp = local_exp[
                    local_exp["ticker"] == selected_ticker
                ].sort_values("shap_value")

                fig_local = px.bar(
                    ticker_exp, x="shap_value", y="feature",
                    color="direction",
                    color_discrete_map={
                        "increases_prediction": "#2ee6a8",
                        "decreases_prediction": "#ff7b8f",
                    },
                    title=f"SHAP Feature Contributions — {selected_ticker}",
                )
                style_figure(fig_local)
                fig_local.update_xaxes(title_text="SHAP Value (Impact on Prediction)")
                fig_local.update_yaxes(title_text="Feature")
                st.plotly_chart(fig_local, use_container_width=True)
                st.markdown(
                    f"<div class='subtle'>{len(ticker_exp)} features contributing to "
                    f"{selected_ticker}'s prediction</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            if "future_predictions" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>SHAP Summary for Future Predictions</div>",
                    unsafe_allow_html=True,
                )
                future_exp = shap_data["future_predictions"].drop_duplicates(
                    subset=["ticker"]
                )
                summary_data = []
                for _, row in future_exp.iterrows():
                    summary_data.append(
                        {
                            "Ticker": row["ticker"],
                            "Prediction": (
                                f"{row.get('predicted_growth_rate', 0):.2f}%"
                                if "predicted_growth_rate" in row
                                else "N/A"
                            ),
                            "Base Value": f"{row.get('shap_expected_value', 0):.4f}",
                        }
                    )
                if summary_data:
                    st.dataframe(
                        pd.DataFrame(summary_data),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Data":
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">FC</div>
                <div>Upload Financial Report</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='subtle'>Supported formats: CSV, Excel (.xlsx/.xls), PDF, JSON. "
        "Any column format accepted — Claude normalises the schema automatically.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader(
        "Financial Report File",
        type=["csv", "xlsx", "xls", "pdf", "json"],
        label_visibility="visible",
    )

    if uploaded_file:
        if st.button("Process & Load Data", type="primary", use_container_width=True):
            from etl.file_processor import process_upload
            from etl.llm_extractor import run_extraction_pipeline
            from etl.transform import transform_dynamic
            from etl.load import store_uploaded_file, is_duplicate_uploaded_file
            from etl.validator import validate, print_validation_report

            with st.status("Processing your file...", expanded=True) as status:

                # Step 1: Extract raw records
                st.write("📂 Extracting data from file...")
                try:
                    raw_records = process_upload(uploaded_file)
                    st.write(f"✅ Extracted {len(raw_records)} raw records.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    status.update(label="Extraction failed", state="error")
                    st.stop()

                ticker_clean, source = infer_ticker_from_upload(raw_records, uploaded_file.name)
                st.write(f"🏷️ Detected ticker: **{ticker_clean}** (source: {source})")
                if ticker_clean == "UNKNOWN":
                    st.error(
                        "Could not infer ticker from file content or filename. "
                        "Please include a ticker/symbol column or use a clearer filename."
                    )
                    status.update(label="Ticker detection failed", state="error")
                    st.stop()

                # Duplicate check: same user + ticker + identical file content
                st.write("🧠 Checking for duplicate upload...")
                try:
                    duplicate_exists = is_duplicate_uploaded_file(
                        user_id=user_id,
                        ticker=ticker_clean,
                        raw_records=raw_records,
                    )
                except Exception:
                    duplicate_exists = False

                if duplicate_exists:
                    st.warning(
                        "Already present. Please check the recommendations. "
                        "No duplicate upload was added."
                    )
                    status.update(label="Already present", state="complete")
                    st.session_state["nav_page"] = "🤖 AI Recommendations"
                    st.session_state["preferred_ticker"] = ticker_clean
                    st.rerun()

                # Step 2: Store raw file in uploaded_files table
                st.write("☁️  Storing raw upload...")
                st.caption(f"DEBUG: user_id = `{user_id}` | email = `{user_email}`")
                try:
                    supabase = get_supabase_client()
                    success = store_uploaded_file(
                        filename=uploaded_file.name,
                        raw_records=raw_records,
                        ticker=ticker_clean,
                        supabase_client=supabase,
                        user_id=user_id,
                    )
                    if success:
                        st.write("✅ Raw file stored securely.")
                    else:
                        st.warning("⚠️  Could not store raw file. Continuing... (Check terminal logs for details)")
                except Exception as e:
                    st.error(f"Failed to store file: {e}")
                    status.update(label="Storage failed", state="error")
                    st.stop()

                # Step 3: LLM schema normalisation
                st.write("🤖 Running LLM schema normalisation (2-prompt pipeline)...")
                try:
                    standard_records, category_records = run_extraction_pipeline(
                        raw_records, ticker=ticker_clean
                    )
                    st.write(
                        f"✅ Normalised {len(standard_records)} records to standard schema."
                    )
                except Exception as e:
                    st.error(f"LLM extraction failed: {e}")
                    status.update(label="LLM extraction failed", state="error")
                    st.stop()

                # Step 4: Validate
                st.write("🔍 Validating data completeness...")
                is_valid, issues = validate(standard_records)
                for issue in issues:
                    if issue.startswith("CRITICAL"):
                        st.error(f"❌ {issue}")
                    else:
                        st.warning(f"⚠️  {issue}")
                if not is_valid:
                    st.error("Data failed critical validation. Cannot proceed to pipeline.")
                    status.update(label="Validation failed", state="error")
                    st.stop()
                st.write("✅ Validation passed.")

                # Step 5: Transform + feature engineering
                st.write("⚙️  Engineering features...")
                try:
                    tables = transform_dynamic(
                        standard_records, category_records, user_id=user_id, ticker=ticker_clean
                    )
                    st.write(
                        f"✅ Standard table: {len(tables['standard_table'])} rows | "
                        f"Category table: {len(tables['category_table'])} rows"
                    )
                except Exception as e:
                    st.error(f"Transform failed: {e}")
                    status.update(label="Transform failed", state="error")
                    st.stop()

                # Step 6: Store transformed tables to Supabase
                st.write("💾 Storing to database...")
                try:
                    from etl.load import load_user_data
                    import pandas as pd
                    
                    standard_df = pd.DataFrame(tables["standard_table"])
                    category_df = pd.DataFrame(tables["category_table"])
                    supabase = get_supabase_client()
                    
                    success = load_user_data(
                        standard_df, 
                        category_df, 
                        supabase, 
                        user_id
                    )
                    
                    if success:
                        st.write("✅ Data stored to database.")
                    else:
                        st.warning("⚠️  Could not store all data to database (see logs).")
                except Exception as e:
                    st.warning(f"⚠️  Storage warning: {e}")

                status.update(label="✅ Data ready for analysis!", state="complete")

            st.markdown("\n---\n")
            
            st.success(
                f"✅ **{ticker_clean}** data successfully stored! "
                f"**{len(tables['standard_table'])} rows** are now in your secure database."
            )
            
            # Show training section
            st.markdown("### Next: Train Analysis Pipeline")
            st.write(
                f"Before generating AI recommendations, train the analysis pipeline for **{ticker_clean}**. "
                f"This generates growth predictions and feature importance analytics."
            )
            
            # Use the unified training button
            run_svr_training_button(
                ticker_clean,
                standard_records=standard_records,
                category_records=category_records,
                user_id=user_id,
                navigate_to_recommendations=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Recommendations":
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">FC</div>
                <div>AI Financial Recommendations</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Available tickers — from SVR predictions + user uploads
    svr_path = "analysis/reports/svr_future_predictions.csv"
    available_tickers = []
    if os.path.exists(svr_path):
        available_tickers = pd.read_csv(svr_path)["ticker"].unique().tolist()

    # Also include user-uploaded tickers
    user_tickers = get_user_tickers(user_id)
    all_tickers = sorted(set(available_tickers + user_tickers))

    if not all_tickers:
        st.info(
            "No data available for recommendations. "
            "Run `python run.py 4` to generate SVR predictions, or upload a report first."
        )
        st.stop()

    preferred_ticker = st.session_state.get("preferred_ticker")
    default_index = 0
    if preferred_ticker and preferred_ticker in all_tickers:
        default_index = all_tickers.index(preferred_ticker)

    selected_ticker = st.selectbox(
        "Select Ticker for Analysis", all_tickers, index=default_index
    )

    # ── Check if SVR predictions exist ─────────────────────────────────────────
    has_predictions = has_svr_predictions(selected_ticker)
    
    if not has_predictions:
        st.warning(f"⚠️ **SVR Analysis Not Available**")
        st.write(f"The ticker **{selected_ticker}** hasn't been trained yet. Click below to train it first.")
        run_svr_training_button(selected_ticker, user_id=user_id)
        st.stop()
    
    col_gen, col_info = st.columns([2, 1])
    with col_gen:
        gen_btn = st.button(
            "Generate AI Recommendations",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            "Powered by Claude · Reads Phase 4+5 outputs · "
            "Generates risk, growth & action narratives"
        )

    if gen_btn:
        with st.spinner(f"Analyzing {selected_ticker}..."):
            try:
                from etl.load import store_recommendation_results
                
                bundle = load_analysis_bundle_from_reports(selected_ticker)
                if not bundle.get("svr_predictions"):
                    st.warning(
                        f"No SVR predictions found for {selected_ticker}. "
                        "Run `python run.py 4` first."
                    )
                    st.stop()
                recs = generate_recommendations(bundle)
                
                # Store recommendations in Supabase
                supabase = get_supabase_client()
                success = store_recommendation_results(
                    ticker=selected_ticker,
                    recommendation_json=recs,
                    supabase_client=supabase,
                    user_id=user_id,
                )
                if success:
                    st.toast("✓ Recommendations saved to database")
                else:
                    st.warning("⚠️  Could not save to database, but showing results")
                
                st.session_state["recommendations"] = recs
                st.session_state["rec_ticker"] = selected_ticker
            except Exception as e:
                st.error(f"Recommendation generation failed: {e}")
                st.stop()

    # ── Display recommendations ────────────────────────────────────────────────
    recs = st.session_state.get("recommendations")
    rec_ticker = st.session_state.get("rec_ticker", selected_ticker)

    if recs:
        # ── Executive summary + score ──────────────────────────────────────────
        score = recs.get("performance_score", 0)
        score_color = (
            "#2ee6a8" if score >= 7
            else "#ffb347" if score >= 5
            else "#ff7b8f"
        )
        col_s, col_sc = st.columns([3, 1])
        with col_s:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Executive Summary</div>",
                unsafe_allow_html=True,
            )
            st.info(recs.get("executive_summary", ""))
            st.markdown("</div>", unsafe_allow_html=True)
        with col_sc:
            metric_tile(
                f"{rec_ticker} Health Score",
                f"{score}/10",
            )

        st.divider()

        # ── Growth outlook + risk ──────────────────────────────────────────────
        col_g, col_r = st.columns(2)
        with col_g:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Growth Outlook</div>",
                unsafe_allow_html=True,
            )
            growth = recs.get("growth_outlook", {})
            st.write(growth.get("forecast", ""))
            conf = growth.get("confidence", "")
            conf_icon = (
                "🟢" if conf == "High" else "🟡" if conf == "Medium" else "🔴"
            )
            pred_rate = growth.get("predicted_growth_rate", "N/A")
            gap = growth.get("gap_vs_target", "N/A")
            gap_status = growth.get("gap_status", "")
            st.caption(
                f"{conf_icon} Confidence: **{conf}** | "
                f"Predicted: **{pred_rate}%** | "
                f"Gap vs target: **{gap}%** ({gap_status})"
            )
            if growth.get("key_drivers"):
                st.write("**Key Drivers:**")
                for d in growth["key_drivers"]:
                    st.write(f"  · {d}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Risk Assessment</div>",
                unsafe_allow_html=True,
            )
            risk = recs.get("risk_assessment", {})
            risk_level = risk.get("overall_risk", "Medium")
            risk_icon = (
                "🟢" if risk_level == "Low"
                else "🟡" if risk_level == "Medium"
                else "🔴"
            )
            st.write(f"{risk_icon} Overall Risk: **{risk_level}**")
            for factor in risk.get("risk_factors", []):
                sev = factor.get("severity", "")
                sev_icon = (
                    "🔴" if sev == "High"
                    else "🟡" if sev == "Medium"
                    else "🟢"
                )
                with st.expander(f"{sev_icon} {factor.get('factor', '')}"):
                    st.write(factor.get("explanation", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ── Action items ───────────────────────────────────────────────────────
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Recommended Actions</div>",
            unsafe_allow_html=True,
        )
        for item in recs.get("action_items", []):
            priority = item.get("priority", "Medium")
            p_icon = (
                "🔴" if priority == "High"
                else "🟡" if priority == "Medium"
                else "🟢"
            )
            with st.expander(f"{p_icon} [{priority}] {item.get('action', '')}"):
                st.write(item.get("rationale", ""))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Opportunities ──────────────────────────────────────────────────────
        opps = recs.get("opportunities", [])
        if opps:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Growth Opportunities</div>",
                unsafe_allow_html=True,
            )
            for opp in opps:
                st.success(
                    f"**{opp.get('title', '')}** — {opp.get('rationale', '')}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Peer positioning ───────────────────────────────────────────────────
        peer_pos = recs.get("peer_positioning", "")
        if peer_pos:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Peer Positioning vs AAPL/MSFT/GOOGL/AMZN</div>",
                unsafe_allow_html=True,
            )
            st.write(peer_pos)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Anomalies ──────────────────────────────────────────────────────────
        anomalies = recs.get("anomalies_flagged", [])
        if anomalies:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>Flagged Anomalies</div>",
                unsafe_allow_html=True,
            )
            for a in anomalies:
                st.warning(f"⚠️  {a}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Export ─────────────────────────────────────────────────────────────
        st.divider()
        st.download_button(
            label="⬇️  Download JSON Report",
            data=json.dumps(recs, indent=2, default=str),
            file_name=f"{rec_ticker}_fincast_report.json",
            mime="application/json",
            use_container_width=True,
        )