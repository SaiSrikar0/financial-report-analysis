import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def load_data(path: str) -> pd.DataFrame:
	p = Path(path)
	if not p.exists():
		st.error(f"Data file not found: {path}")
		return pd.DataFrame()
	with p.open("r", encoding="utf-8") as f:
		data = json.load(f)
	df = pd.DataFrame(data)
	# normalize types
	df["date"] = pd.to_datetime(df["date"])
	return df


def set_light_theme():
	# simple light background and gentle card styling
	st.markdown(
		"""
		<style>
		.stApp {
			linear-gradient : {
			deg: 135;}
			background-color: #ECC4F5;
		}
		.metric {
			padding: 8px;
		}
		.card {
			background: blue;
			border-radius: 8px;
			padding: 8px;
			box-shadow: 0 1px 3px rgba(0,0,0,0.08);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def main():
	st.set_page_config(page_title="Financial Report Analysis", layout="wide")
	set_light_theme()

	st.title("FinCast — Financial Report Analysis")

	data_path = "data/raw/financial_data_raw.json"
	df = load_data(data_path)
	if df.empty:
		return

	# Sidebar filters
	st.sidebar.header("Filters")
	tickers = sorted(df["ticker"].unique())
	selected = st.sidebar.multiselect("Company", tickers, default=tickers[:2])

	min_year = int(df["date"].dt.year.min())
	max_year = int(df["date"].dt.year.max())
	year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

	filtered = df[df["ticker"].isin(selected)]
	filtered = filtered[(filtered["date"].dt.year >= year_range[0]) & (filtered["date"].dt.year <= year_range[1])]

	# Top metrics for selected set
	total_revenue = filtered["revenue"].sum()
	total_net = filtered["net_income"].sum()
	profit_margin = (filtered["net_income"].sum() / filtered["revenue"].sum()) if filtered["revenue"].sum() else 0
	debt_ratio = (filtered["total_liabilities"].sum() / filtered["total_assets"].sum()) if filtered["total_assets"].sum() else 0

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Revenue", f"${total_revenue:,.0f}")
	col2.metric("Net Income", f"${total_net:,.0f}")
	col3.metric("Profit Margin", f"{profit_margin*100:.1f}%")
	col4.metric("Debt Ratio", f"{debt_ratio*100:.1f}%")

	# Revenue trends line chart
	st.markdown("### Revenue Trends by Company")
	fig_line = px.line(filtered, x="date", y="revenue", color="ticker", markers=True)
	fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20))
	st.plotly_chart(fig_line, use_container_width=True)

	# Three small charts in a row
	c1, c2, c3 = st.columns(3)

	with c1:
		st.markdown("#### Revenue vs Operating Income")
		bar = filtered.groupby("ticker")[["revenue", "operating_income"]].sum().reset_index()
		fig_bar = px.bar(bar, x="ticker", y=["revenue", "operating_income"], barmode="group")
		st.plotly_chart(fig_bar, use_container_width=True)

	with c2:
		st.markdown("#### Assets vs Liabilities")
		al = filtered.groupby("ticker")[["total_assets", "total_liabilities"]].sum().reset_index()
		fig_al = px.bar(al, x="ticker", y=["total_assets", "total_liabilities"], barmode="group")
		st.plotly_chart(fig_al, use_container_width=True)

	with c3:
		st.markdown("#### Cashflow vs Net Income")
		cn = filtered.groupby("ticker")[["operating_cashflow", "net_income"]].sum().reset_index()
		fig_cn = px.bar(cn, x="ticker", y=["operating_cashflow", "net_income"], barmode="group")
		st.plotly_chart(fig_cn, use_container_width=True)

	# 3D scatter (revenue, debt ratio, net income)
	st.markdown("### 3D View: Revenue / Debt Ratio / Net Income")
	scatter_df = filtered.copy()
	scatter_df["debt_ratio"] = scatter_df["total_liabilities"] / scatter_df["total_assets"]
	fig_scatter = px.scatter_3d(scatter_df, x="revenue", y="debt_ratio", z="net_income", color="ticker", size="revenue", hover_name="ticker")
	fig_scatter.update_layout(margin=dict(l=0, r=0, t=20, b=0))
	st.plotly_chart(fig_scatter, use_container_width=True)

	# Financial table
	st.markdown("### Financial Overview")
	table = filtered.sort_values(["ticker", "date"]) \
		[["date", "ticker", "revenue", "net_income", "operating_income", "total_assets", "total_liabilities", "operating_cashflow"]]
	st.dataframe(table)


if __name__ == "__main__":
	main()
