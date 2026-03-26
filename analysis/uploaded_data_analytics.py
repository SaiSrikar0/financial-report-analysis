"""Analytics for user-uploaded data files."""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from analysis.data_connection import get_supabase_client
from etl.load import delete_user_uploaded_data


def get_user_uploaded_files(user_id: str) -> list:
    """Fetch list of uploaded files for user with robust error handling."""
    try:
        # Use the Supabase client - auth context is set during login via st.session_state
        client = get_supabase_client()
        
        # Try to fetch with explicit user_id filter (double-safe with RLS)
        # RLS policies enforce user_id filtering on the server side
        response = client.table("uploaded_files").select(
            "id,filename,ticker,created_at,file_content"
        ).eq("user_id", user_id).limit(100).execute()
        
        files = response.data if response and response.data else []
        print(f"[get_user_uploaded_files] ✓ Loaded {len(files)} files for user {user_id[:8]}...")
        return files
    except Exception as e:
        print(f"[get_user_uploaded_files] Error: {type(e).__name__}: {e}")
        # Don't show warning - silently return empty list for better UX
        return []


def display_uploaded_files_section(user_id: str):
    """Display uploaded files with data visualizations and analytics."""
    uploaded_files = get_user_uploaded_files(user_id)
    
    if not uploaded_files:
        st.info("No uploaded files yet. Go to 📤 Upload Data to get started.")
        return
    
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-left">
                <div class="brand-logo">📁</div>
                <div>Your Uploaded Data</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # List uploaded files
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Files Uploaded</div>",
        unsafe_allow_html=True,
    )
    
    files_df = pd.DataFrame([
        {
            "Ticker": f.get("ticker", "UNKNOWN"),
            "Filename": f.get("filename", "Unknown"),
            "Records": len(f.get("file_content", [])) if isinstance(f.get("file_content"), list) else 0,
            "Uploaded": (f.get("created_at") or f.get("upload_date") or "N/A")[:10]
        }
        for f in uploaded_files
    ])
    
    st.dataframe(files_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Select file to analyze
    selected_idx = st.selectbox(
        "Select a file to analyze",
        range(len(uploaded_files)),
        format_func=lambda i: f"{uploaded_files[i]['ticker']} - {uploaded_files[i]['filename']}"
    )
    
    selected_file = uploaded_files[selected_idx]
    ticker = selected_file["ticker"]
    file_id = selected_file.get("id")
    raw_content = selected_file.get("file_content", [])

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Manage Uploaded Data</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Delete the selected upload and its related ticker data "
        "(standard/category/recommendation records for your user)."
    )

    confirm_delete = st.checkbox(
        f"Confirm delete for {ticker} ({selected_file.get('filename', 'file')})",
        key=f"confirm_delete_{file_id}_{ticker}",
    )
    delete_btn = st.button(
        "🗑️ Delete Selected Upload",
        type="secondary",
        use_container_width=False,
        key=f"delete_upload_{file_id}_{ticker}",
    )

    if delete_btn:
        if not confirm_delete:
            st.warning("Tick the confirmation box before deleting data.")
        else:
            with st.spinner("Deleting uploaded data..."):
                delete_result = delete_user_uploaded_data(
                    user_id=user_id,
                    ticker=ticker,
                    uploaded_file_id=file_id,
                )

            if delete_result.get("success"):
                counts = delete_result.get("deleted", {})
                st.success(
                    "Deleted successfully: "
                    f"uploaded_files={counts.get('uploaded_files', 0)}, "
                    f"standard_table={counts.get('standard_table', 0)}, "
                    f"category_table={counts.get('category_table', 0)}, "
                    f"recommendation_results={counts.get('recommendation_results', 0)}"
                )

                if (
                    st.session_state.get("rec_ticker", "").upper() == str(ticker).upper()
                ):
                    st.session_state.pop("recommendations", None)
                    st.session_state.pop("rec_ticker", None)

                st.rerun()
            else:
                st.error("Delete failed. See details below.")
                for err in delete_result.get("errors", []):
                    st.error(f"- {err}")

    st.markdown("</div>", unsafe_allow_html=True)
    
    if not raw_content:
        st.warning("No data in this file.")
        return
    
    # Convert to DataFrame
    if isinstance(raw_content, list):
        data_df = pd.DataFrame(raw_content)
    else:
        st.warning("Invalid data format.")
        return
    
    # Display data statistics
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-title'>Data Summary: {ticker}</div>",
        unsafe_allow_html=True,
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(data_df))
    with col2:
        st.metric("Columns", len(data_df.columns))
    with col3:
        numeric_cols = data_df.select_dtypes(include=['number']).columns
        st.metric("Numeric Fields", len(numeric_cols))
    with col4:
        if 'revenue' in data_df.columns:
            try:
                total_rev = data_df['revenue'].sum()
                st.metric("Total Revenue", f"${total_rev/1e6:.1f}M" if total_rev > 0 else "N/A")
            except:
                st.metric("Total Revenue", "N/A")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations for key metrics
    numeric_cols = data_df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Key Metrics Distribution</div>",
            unsafe_allow_html=True,
        )
        
        # Select metrics to visualize
        selected_metrics = st.multiselect(
            "Select metrics to visualize",
            numeric_cols,
            default=[col for col in ['revenue', 'net_income', 'operating_income'] if col in numeric_cols][:2]
        )
        
        if selected_metrics:
            # Create line chart
            if 'date' in data_df.columns:
                plot_df = data_df[['date'] + selected_metrics].copy()
                plot_df['date'] = pd.to_datetime(plot_df['date'], errors='coerce')
                plot_df = plot_df.dropna(subset=['date']).sort_values('date')
                
                fig = px.line(
                    plot_df,
                    x='date',
                    y=selected_metrics,
                    markers=True,
                    title=f"{ticker} - Metric Trends"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,33,78,0.3)",
                    font=dict(family="Manrope", color="#dce7ff"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # If no date column, show bar chart
                plot_df = data_df[selected_metrics].head(10)
                fig = px.bar(
                    plot_df,
                    y=selected_metrics,
                    title=f"{ticker} - Metric Comparison (First 10 Records)"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,33,78,0.3)",
                    font=dict(family="Manrope", color="#dce7ff"),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show raw data preview
    with st.expander("📋 View Raw Data"):
        st.dataframe(data_df, use_container_width=True)


def display_svr_analysis_for_ticker(ticker: str):
    """Display SVR predictions for a specific ticker if available."""
    reports_dir = "analysis/reports"
    svr_path = os.path.join(reports_dir, "svr_future_predictions.csv")
    
    if not os.path.exists(svr_path):
        st.info("SVR predictions not available. Run `python run.py 4` first.")
        return
    
    try:
        df = pd.read_csv(svr_path)
        row = df[df["ticker"].str.upper() == ticker.upper()]
        
        if row.empty:
            st.warning(f"No SVR predictions found for {ticker}.")
            return
        
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-title'>SVR Growth Predictions: {ticker}</div>",
            unsafe_allow_html=True,
        )
        
        prediction = row.iloc[0].to_dict()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Growth Rate", f"{prediction.get('predicted_growth_rate', 'N/A'):.2f}%" if isinstance(prediction.get('predicted_growth_rate'), (int, float)) else "N/A")
        with col2:
            st.metric("Target Growth", "10%")
        with col3:
            gap = prediction.get('predicted_growth_rate', 0) - 10
            st.metric("Gap vs Target", f"{gap:.2f}%", delta=gap)
        
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load SVR data: {e}")


def display_shap_analysis_for_ticker(ticker: str):
    """Display SHAP feature importance if available."""
    reports_dir = "analysis/reports"
    shap_path = os.path.join(reports_dir, "phase_5_shap_global_importance.csv")
    
    if not os.path.exists(shap_path):
        st.info("SHAP analysis not available. Run `python run.py 5` first.")
        return
    
    try:
        df = pd.read_csv(shap_path)
        
        if df.empty:
            st.warning("No SHAP data found.")
            return
        
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>Feature Importance (SHAP Analysis)</div>",
            unsafe_allow_html=True,
        )
        
        # Display top features
        top_features = df.head(8)
        
        fig = px.bar(
            top_features,
            x=top_features.columns[1] if len(top_features.columns) > 1 else top_features.columns[0],
            y=top_features.columns[0],
            orientation='h',
            title="Top 8 Most Important Features (by SHAP value)"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,33,78,0.3)",
            font=dict(family="Manrope", color="#dce7ff"),
            xaxis_title="SHAP Impact",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load SHAP data: {e}")
