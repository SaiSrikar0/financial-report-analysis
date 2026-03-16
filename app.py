import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np


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
    """Load Phase 4 SVR model outputs."""
    reports_path = Path("analysis/reports")
    data = {}
    
    # Load evaluation metrics
    metrics_file = reports_path / "svr_evaluation_metrics.csv"
    if metrics_file.exists():
        data["metrics"] = pd.read_csv(metrics_file).iloc[0].to_dict()
    
    # Load future predictions
    pred_file = reports_path / "svr_future_predictions.csv"
    if pred_file.exists():
        data["predictions"] = pd.read_csv(pred_file)
    
    # Load test predictions
    test_file = reports_path / "svr_test_predictions.csv"
    if test_file.exists():
        data["test_predictions"] = pd.read_csv(test_file)
    
    # Load best params
    params_file = reports_path / "svr_best_params.csv"
    if params_file.exists():
        data["params"] = pd.read_csv(params_file).iloc[0].to_dict()
    
    return data


@st.cache_data
def load_shap_data() -> dict:
    """Load Phase 5 SHAP explainability outputs."""
    reports_path = Path("analysis/reports")
    data = {}
    
    # Load global importance
    global_file = reports_path / "phase_5_shap_global_importance.csv"
    if global_file.exists():
        data["global_importance"] = pd.read_csv(global_file)
    
    # Load local explanations
    local_file = reports_path / "phase_5_shap_local_explanations.csv"
    if local_file.exists():
        data["local_explanations"] = pd.read_csv(local_file)
    
    # Load future predictions with SHAP
    future_file = reports_path / "phase_5_shap_future_predictions.csv"
    if future_file.exists():
        data["future_predictions"] = pd.read_csv(future_file)
    
    return data


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
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes drop-in {
            from {
                opacity: 0;
                transform: translateY(-8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .metric-value {
                font-size: 1.45rem;
            }

            .brand-row {
                padding: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=16, b=14),
        font=dict(family="Manrope, sans-serif", color="#dce7ff", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#a9bfe7"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(112,145,212,0.22)",
            zeroline=False,
            linecolor="rgba(112,145,212,0.32)",
            tickfont=dict(color="#a9bfe7"),
            title=dict(font=dict(color="#a9bfe7")),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(112,145,212,0.22)",
            zeroline=False,
            linecolor="rgba(112,145,212,0.32)",
            tickfont=dict(color="#a9bfe7"),
            title=dict(font=dict(color="#a9bfe7")),
        ),
    )
    return fig


def build_light_recommendation(
    filtered: pd.DataFrame,
    ticker: str,
    year: int,
) -> dict:
    """Generate lightweight rule-based guidance from currently filtered data."""
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

    strengths = []
    risks = []
    actions = []

    if profit_margin >= 0.15:
        strengths.append(f"Healthy profitability with ~{profit_margin*100:.1f}% margin.")
    elif profit_margin >= 0.05:
        strengths.append(f"Positive profitability ({profit_margin*100:.1f}%), but with room to expand margins.")
        actions.append("Improve gross margin mix and operating efficiency to push profit margin above 15%.")
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
        actions.append("Create a debt reduction plan and consider refinancing expensive liabilities.")

    if cash_ratio >= 1.0:
        strengths.append("Operating cashflow sufficiently supports earnings.")
    else:
        risks.append("Cash conversion is weak relative to reported net income.")
        actions.append("Tighten receivables and working-capital cycles to improve cash conversion.")

    if revenue_growth is not None:
        if revenue_growth > 8:
            strengths.append(f"Strong YoY revenue momentum (+{revenue_growth:.1f}%).")
        elif revenue_growth < 0:
            risks.append(f"Revenue contraction detected ({revenue_growth:.1f}% YoY).")
            actions.append("Investigate demand, pricing, and product-mix drivers behind revenue decline.")

    if net_income > 0 and profit_margin >= 0.05:
        status = "Profitable"
    elif profit_margin >= 0:
        status = "Break-Even"
    else:
        status = "Loss"

    return {
        "status": status,
        "strengths": strengths,
        "risks": risks,
        "actions": actions,
    }


def main() -> None:
    st.set_page_config(page_title="FinCast Dashboard", layout="wide")
    inject_theme()

    data_path = "data/raw/financial_data_raw.json"
    df = load_data(data_path)
    if df.empty:
        return

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">FC</div>
                <div>FinCast</div>
            </div>
            <div class="brand-icons">Bell  |  Panels  |  Settings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Filters")
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    selected_tickers = st.sidebar.multiselect("Company", tickers, default=tickers)

    min_year = int(df["date"].dt.year.min())
    max_year = int(df["date"].dt.year.max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (max(min_year, 2015), max_year))

    data_source = st.sidebar.radio("Data Source", ["Internal", "External"], horizontal=True)
    st.sidebar.button("Upload Dataset", use_container_width=True)
    enable_light_reco = st.sidebar.checkbox(
        "Enable Quick Recommendations",
        value=False,
        help="Optional lightweight advisory panel (rule-based, no LLM / Phase 6 dependency).",
    )
    metric_choice = st.sidebar.selectbox(
        "Select Metric",
        options=["Revenue", "Net Income", "Operating Income", "Cashflow", "Profit Margin", "Debt Ratio"],
        index=0,
    )

    if not selected_tickers:
        st.warning("Select at least one company to render the dashboard.")
        return

    filtered = df[df["ticker"].isin(selected_tickers)].copy()
    filtered = filtered[
        (filtered["date"].dt.year >= year_range[0]) & (filtered["date"].dt.year <= year_range[1])
    ]

    # Raw financial data contains operating income; derive expenses for UI parity.
    if "operating_expenses" not in filtered.columns:
        filtered["operating_expenses"] = (filtered["revenue"] - filtered["operating_income"]).clip(lower=0)

    if filtered.empty:
        st.warning("No records found for selected filters.")
        return

    filtered["profit_margin"] = filtered["net_income"] / filtered["revenue"].replace(0, pd.NA)
    filtered["debt_ratio"] = filtered["total_liabilities"] / filtered["total_assets"].replace(0, pd.NA)

    if enable_light_reco:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Recommendation")
        reco_company = st.sidebar.selectbox("Company for advisory", selected_tickers, key="quick_reco_company")
        reco_years = sorted(
            filtered.loc[filtered["ticker"] == reco_company, "date"].dt.year.unique().tolist(),
            reverse=True,
        )

        if reco_years:
            reco_year = st.sidebar.selectbox("Year", reco_years, key="quick_reco_year")
            recommendation = build_light_recommendation(filtered, reco_company, int(reco_year))

            st.sidebar.markdown(f"Status: **{recommendation['status']}**")
            with st.sidebar.expander("Strengths", expanded=False):
                if recommendation["strengths"]:
                    for item in recommendation["strengths"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- No major strengths flagged for the selected slice.")

            with st.sidebar.expander("Risks", expanded=False):
                if recommendation["risks"]:
                    for item in recommendation["risks"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- No major risks flagged for the selected slice.")

            with st.sidebar.expander("Actions", expanded=True):
                if recommendation["actions"]:
                    for item in recommendation["actions"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- Maintain current strategy and monitor trend changes.")

    # Create tabs
    tab_historical, tab_predictions, tab_explainability = st.tabs(
        ["📊 Historical Analysis", "🎯 SVR Predictions", "🔍 SHAP Explainability"]
    )

    # ========== TAB 1: HISTORICAL ANALYSIS ==========
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
        st.markdown("<div class='section-title'>Revenue Trends by Company</div>", unsafe_allow_html=True)
        line_fig = px.line(
            filtered.sort_values("date"),
            x="date",
            y="revenue",
            color="ticker",
            markers=True,
            color_discrete_sequence=["#3c82ff", "#44d1ff", "#2ee6a8", "#9d8dff", "#ff9c66"],
        )
        line_fig.update_traces(line=dict(width=3), marker=dict(size=7))
        style_figure(line_fig)
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3, gap="medium")

        with b1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Revenue vs Expenses</div>", unsafe_allow_html=True)
            rev_exp = (
                filtered.groupby("ticker", as_index=False)[["revenue", "operating_expenses"]].sum().melt(
                    id_vars="ticker", value_vars=["revenue", "operating_expenses"], var_name="metric", value_name="value"
                )
            )
            fig_rev_exp = px.bar(
                rev_exp,
                x="ticker",
                y="value",
                color="metric",
                barmode="group",
                color_discrete_map={"revenue": "#3c82ff", "operating_expenses": "#2ee6a8"},
            )
            style_figure(fig_rev_exp)
            st.plotly_chart(fig_rev_exp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Assets vs Liabilities</div>", unsafe_allow_html=True)
            ass_liab = (
                filtered.groupby("ticker", as_index=False)[["total_assets", "total_liabilities"]].sum().melt(
                    id_vars="ticker",
                    value_vars=["total_assets", "total_liabilities"],
                    var_name="metric",
                    value_name="value",
                )
            )
            fig_ass_liab = px.bar(
                ass_liab,
                x="ticker",
                y="value",
                color="metric",
                barmode="group",
                color_discrete_map={"total_assets": "#44d1ff", "total_liabilities": "#2ee6a8"},
            )
            style_figure(fig_ass_liab)
            st.plotly_chart(fig_ass_liab, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b3:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Cashflow vs Income</div>", unsafe_allow_html=True)
            cf_income = (
                filtered.groupby("ticker", as_index=False)[["operating_cashflow", "net_income"]].sum().melt(
                    id_vars="ticker",
                    value_vars=["operating_cashflow", "net_income"],
                    var_name="metric",
                    value_name="value",
                )
            )
            fig_cf_income = px.bar(
                cf_income,
                x="ticker",
                y="value",
                color="metric",
                barmode="group",
                color_discrete_map={"operating_cashflow": "#3c82ff", "net_income": "#44d1ff"},
            )
            style_figure(fig_cf_income)
            st.plotly_chart(fig_cf_income, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        low, high = st.columns([1.1, 2.2], gap="medium")

        with low:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Financial Landscape</div>", unsafe_allow_html=True)
            scatter3d = px.scatter_3d(
                filtered,
                x="revenue",
                y="debt_ratio",
                z="net_income",
                color="ticker",
                size="total_assets",
                hover_name="ticker",
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
            st.markdown("<div class='section-title'>Financial Overview</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='subtle'>Source: {data_source}  |  Focus Metric: {metric_choice}  |  Rows: {len(filtered)}</div>",
                unsafe_allow_html=True,
            )
            table = filtered.sort_values(["date", "ticker"], ascending=[False, True]).copy()
            table["year"] = table["date"].dt.year
            table = table[
                [
                    "year",
                    "ticker",
                    "revenue",
                    "net_income",
                    "operating_expenses",
                    "total_assets",
                    "total_liabilities",
                    "operating_cashflow",
                ]
            ].rename(
                columns={
                    "ticker": "company",
                    "operating_expenses": "expenses",
                    "total_assets": "assets",
                    "total_liabilities": "liabilities",
                    "operating_cashflow": "cashflow",
                }
            )

            styled = table.style.format(
                {
                    "revenue": "${:,.0f}",
                    "net_income": "${:,.0f}",
                    "expenses": "${:,.0f}",
                    "assets": "${:,.0f}",
                    "liabilities": "${:,.0f}",
                    "cashflow": "${:,.0f}",
                }
            )
            st.dataframe(styled, use_container_width=True, height=350)
            st.markdown("</div>", unsafe_allow_html=True)

    # ========== TAB 2: SVR PREDICTIONS (PHASE 4) ==========
    with tab_predictions:
        svr_data = load_svr_predictions()
        
        if not svr_data:
            st.warning("Phase 4 SVR data not available. Run Phase 4 to generate predictions.")
        else:
            # Model Performance Metrics
            if "metrics" in svr_data:
                metrics = svr_data["metrics"]
                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_tile("MAE", f"{metrics.get('mae', 0):.2f}%")
                with m2:
                    metric_tile("RMSE", f"{metrics.get('rmse', 0):.2f}%")
                with m3:
                    metric_tile("R² Score", f"{metrics.get('r2', 0):.4f}")
            
            # Best Hyperparameters
            if "params" in svr_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Best Model Hyperparameters</div>", unsafe_allow_html=True)
                params = svr_data["params"]
                param_df = pd.DataFrame(
                    [
                        {"Parameter": "Kernel", "Value": params.get("svr__kernel", "linear")},
                        {"Parameter": "C (regularization)", "Value": f"{params.get('svr__C', 1):.2f}"},
                        {"Parameter": "Epsilon", "Value": f"{params.get('svr__epsilon', 0.01):.4f}"},
                        {"Parameter": "Gamma", "Value": str(params.get("svr__gamma", "scale"))},
                    ]
                )
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Future Predictions with Confidence Intervals
            if "predictions" in svr_data:
                preds = svr_data["predictions"]
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Future Growth Rate Predictions (Next Fiscal Year)</div>", unsafe_allow_html=True)
                
                pred_table = preds[[
                    "ticker",
                    "predicted_growth_rate",
                    "target_growth_rate",
                    "gap_vs_target",
                    "gap_status",
                    "confidence_lower_95",
                    "confidence_upper_95"
                ]].copy()
                
                pred_table.columns = [
                    "Company",
                    "Predicted Growth %",
                    "Target Growth %",
                    "Gap %",
                    "Status",
                    "CI Lower",
                    "CI Upper"
                ]
                
                styled_preds = pred_table.style.format({
                    "Predicted Growth %": "{:.2f}%",
                    "Target Growth %": "{:.2f}%",
                    "Gap %": "{:.2f}%",
                    "CI Lower": "{:.2f}%",
                    "CI Upper": "{:.2f}%",
                })
                st.dataframe(styled_preds, use_container_width=True, hide_index=True)
                
                # Confidence Interval Visualization
                fig_ci = go.Figure()
                for _, row in preds.iterrows():
                    ticker = row["ticker"]
                    predicted = row["predicted_growth_rate"]
                    ci_lower = row["confidence_lower_95"]
                    ci_upper = row["confidence_upper_95"]
                    
                    fig_ci.add_trace(go.Scatter(
                        x=[ci_lower, ci_upper],
                        y=[ticker, ticker],
                        mode="lines",
                        line=dict(width=10, color="#44d1ff"),
                        name=f"{ticker} (95% CI)"
                    ))
                    
                    fig_ci.add_trace(go.Scatter(
                        x=[predicted],
                        y=[ticker],
                        mode="markers",
                        marker=dict(size=12, color="#ff9c66"),
                        name=f"{ticker} (Predicted)",
                        showlegend=False
                    ))
                
                fig_ci.add_vline(x=10, line_dash="dash", line_color="#2ee6a8", annotation_text="Target (10%)")
                style_figure(fig_ci)
                fig_ci.update_xaxes(title_text="Growth Rate (%)")
                fig_ci.update_yaxes(title_text="Company")
                st.plotly_chart(fig_ci, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Test Predictions (Predicted vs Actual)
            if "test_predictions" in svr_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Test Set: Predicted vs Actual Growth Rates</div>", unsafe_allow_html=True)
                
                test_preds = svr_data["test_predictions"]
                actual_col = "actual_growth_rate" if "actual_growth_rate" in test_preds.columns else "actual"
                predicted_col = "predicted_growth_rate" if "predicted_growth_rate" in test_preds.columns else "predicted"
                residual_col = "residual" if "residual" in test_preds.columns else None

                if actual_col in test_preds.columns and predicted_col in test_preds.columns:
                    hover_cfg = {residual_col: ":.2f"} if residual_col else None
                    fig_pva = px.scatter(
                        test_preds,
                        x=actual_col,
                        y=predicted_col,
                        color="ticker",
                        hover_name="ticker",
                        hover_data=hover_cfg,
                        color_discrete_sequence=["#3c82ff", "#44d1ff", "#2ee6a8", "#9d8dff", "#ff9c66"],
                    )

                    # Add perfect prediction line
                    min_val = min(test_preds[actual_col].min(), test_preds[predicted_col].min())
                    max_val = max(test_preds[actual_col].max(), test_preds[predicted_col].max())
                    fig_pva.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(dash="dash", color="#a9bfe7", width=2),
                        name="Perfect Prediction"
                    ))

                    style_figure(fig_pva)
                    fig_pva.update_xaxes(title_text="Actual Growth Rate (%)")
                    fig_pva.update_yaxes(title_text="Predicted Growth Rate (%)")
                    st.plotly_chart(fig_pva, use_container_width=True)
                else:
                    st.warning(
                        "Could not render test prediction scatter: expected actual/predicted columns were not found."
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    # ========== TAB 3: SHAP EXPLAINABILITY (PHASE 5) ==========
    with tab_explainability:
        shap_data = load_shap_data()
        
        if not shap_data:
            st.warning("Phase 5 SHAP data not available. Run Phase 5 to generate explainability insights.")
        else:
            # Global Feature Importance
            if "global_importance" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Global Feature Importance (Mean Absolute SHAP)</div>", unsafe_allow_html=True)
                
                global_imp = shap_data["global_importance"].sort_values("mean_abs_shap", ascending=True)
                
                fig_global = px.bar(
                    global_imp,
                    x="mean_abs_shap",
                    y="feature",
                    color="mean_abs_shap",
                    orientation="h",
                    color_continuous_scale=["#3c82ff", "#44d1ff", "#2ee6a8"],
                )
                
                style_figure(fig_global)
                fig_global.update_xaxes(title_text="Mean Absolute SHAP Value")
                fig_global.update_yaxes(title_text="Feature")
                st.plotly_chart(fig_global, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Local Explanations by Company
            if "local_explanations" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Local Feature Contributions by Company</div>", unsafe_allow_html=True)
                
                local_exp = shap_data["local_explanations"]
                companies = sorted(local_exp["ticker"].unique())
                selected_company = st.selectbox("Select Company", companies)
                
                company_exp = local_exp[local_exp["ticker"] == selected_company].sort_values("shap_value")
                
                fig_local = px.bar(
                    company_exp,
                    x="shap_value",
                    y="feature",
                    color="direction",
                    color_discrete_map={
                        "increases_prediction": "#2ee6a8",
                        "decreases_prediction": "#ff7b8f"
                    },
                    title=f"SHAP Feature Contributions for {selected_company}"
                )
                
                style_figure(fig_local)
                fig_local.update_xaxes(title_text="SHAP Value (Impact on Prediction)")
                fig_local.update_yaxes(title_text="Feature")
                fig_local.update_layout(showlegend=True, legend=dict(title="Direction"))
                st.plotly_chart(fig_local, use_container_width=True)
                
                # Display feature details
                st.markdown(f"<div class='subtle'>{len(company_exp)} features contributing to {selected_company}'s prediction</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # SHAP Summary for Future Predictions
            if "future_predictions" in shap_data:
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>SHAP Summary for Future Predictions</div>", unsafe_allow_html=True)
                
                future_exp = shap_data["future_predictions"].drop_duplicates(subset=["ticker"])
                summary_data = []
                
                for _, row in future_exp.iterrows():
                    summary_data.append({
                        "Company": row["ticker"],
                        "Prediction": f"{row.get('predicted_growth_rate', 'N/A'):.2f}%" if "predicted_growth_rate" in row else "N/A",
                        "Top Positive Factor": row.get("top_positive_feature", "N/A"),
                        "Top Negative Factor": row.get("top_negative_feature", "N/A")
                    })
                
                summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
                if not summary_df.empty:
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Future prediction SHAP summary not fully loaded.")
                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
