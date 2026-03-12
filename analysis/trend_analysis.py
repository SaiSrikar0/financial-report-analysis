# Trend Analysis
# Phase 3.1: Components 2 & 3

import pandas as pd
import numpy as np
from .data_connection import get_standard_table_data, get_company_data


def analyze_trends(ticker=None):
    """Analyze revenue, profit, and growth trends."""
    print("\n" + "=" * 60)
    print("PHASE 3.1.2: TREND ANALYSIS")
    print("=" * 60)
    
    df = get_company_data(ticker) if ticker else get_standard_table_data()
    companies = [ticker] if ticker else sorted(df['ticker'].unique())
    
    trends_report = {}
    for company in companies:
        data = df[df['ticker'] == company].sort_values('date')
        if len(data) < 2:
            continue
        
        trends_report[company] = {
            'revenue_trend': classify_trend(data['revenue'].values),
            'profit_trend': classify_trend(data['net_income'].values),
            'growth_trend': classify_trend(data['revenue_growth'].dropna().values),
            'latest_year': data['date'].iloc[-1].year,
            'revenue_latest': data['revenue'].iloc[-1],
            'profit_latest': data['net_income'].iloc[-1],
        }
        
        t = trends_report[company]
        print(f"\n{company}:")
        print(f"  Revenue: {t['revenue_trend']} | Profit: {t['profit_trend']} | Growth: {t['growth_trend']}")
        print(f"  Latest (FY {t['latest_year']}): Rev=${t['revenue_latest']/1e9:.1f}B, Profit=${t['profit_latest']/1e9:.1f}B")
    
    return trends_report


def classify_trend(values):
    """
    Classify trend as Increasing, Decreasing, or Stable
    
    Args:
        values: Array of values over time
    
    Returns:
        Trend classification
    """
    if len(values) < 2:
        return "Insufficient Data"
    
    # Calculate simple linear regression
    x = np.arange(len(values))
    y = np.array(values)
    
    # Remove NaN values
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return "Insufficient Data"
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope = np.polyfit(x_clean, y_clean, 1)[0]
    
    # Determine trend based on slope and volatility
    mean_val = np.mean(y_clean)
    if mean_val == 0:
        return "Neutral"
    
    slope_pct = (slope / mean_val) * 100  # As percentage of mean
    
    if slope_pct > 5:
        return "Strong Increase"
    elif slope_pct > 1:
        return "Moderate Increase"
    elif slope_pct > -1:
        return "Stable"
    elif slope_pct > -5:
        return "Moderate Decrease"
    else:
        return "Strong Decrease"


def calculate_ratios(ticker=None):
    """Calculate comprehensive financial ratios."""
    print("\n" + "=" * 60)
    print("PHASE 3.1.3: RATIO ANALYSIS")
    print("=" * 60)
    
    df = get_company_data(ticker) if ticker else get_standard_table_data()
    companies = [ticker] if ticker else sorted(df['ticker'].unique())
    
    ratios_report = {}
    for company in companies:
        company_data = df[df['ticker'] == company].sort_values('date')
        ratios = {'ticker': company, **calculate_ratio_metrics(company_data)}
        ratios_report[company] = ratios
        print_ratio_summary(company, ratios)
    
    return ratios_report


def calculate_ratio_metrics(company_data):
    """Calculate all ratio types in one pass."""
    # Profitability
    profitability = {
        'profit_margin_avg': company_data['profit_margin'].mean(),
        'profit_margin_latest': company_data['profit_margin'].iloc[-1],
        'operating_margin_avg': company_data['operating_margin'].mean(),
        'operating_margin_latest': company_data['operating_margin'].iloc[-1],
    }
    
    # Efficiency
    efficiency = {
        'asset_efficiency_avg': company_data['asset_efficiency'].mean(),
        'asset_efficiency_latest': company_data['asset_efficiency'].iloc[-1],
        'asset_turnover_trend': company_data['asset_efficiency'].iloc[-1] - company_data['asset_efficiency'].iloc[0],
    }
    
    # Leverage
    leverage = {
        'debt_to_asset_avg': company_data['debt_to_asset'].mean(),
        'debt_to_asset_latest': company_data['debt_to_asset'].iloc[-1],
        'debt_to_asset_improvement': company_data['debt_to_asset'].iloc[0] - company_data['debt_to_asset'].iloc[-1],
        'equity_to_asset_latest': 1 - company_data['debt_to_asset'].iloc[-1],
    }
    
    # Cash flow
    total_revenue = company_data['revenue'].sum()
    total_cashflow = company_data['operating_cashflow'].sum()
    cash_flow = {
        'total_cashflow': total_cashflow,
        'avg_annual_cashflow': company_data['operating_cashflow'].mean(),
        'cashflow_to_revenue': (total_cashflow / total_revenue) * 100 if total_revenue > 0 else 0,
        'latest_year_cashflow': company_data['operating_cashflow'].iloc[-1],
    }
    
    return {
        'profitability_ratios': profitability,
        'efficiency_ratios': efficiency,
        'leverage_ratios': leverage,
        'cash_flow_ratios': cash_flow,
    }


def print_ratio_summary(ticker, ratios):
    """Print formatted ratio summary"""
    prof = ratios['profitability_ratios']
    eff = ratios['efficiency_ratios']
    lev = ratios['leverage_ratios']
    cf = ratios['cash_flow_ratios']
    
    print(f"\n{ticker} - Financial Ratios:")
    print("-" * 60)
    print(f"  Profitability:")
    print(f"    Profit Margin (Avg): {prof['profit_margin_avg']:.2f}% (Latest: {prof['profit_margin_latest']:.2f}%)")
    print(f"    Operating Margin (Avg): {prof['operating_margin_avg']:.2f}% (Latest: {prof['operating_margin_latest']:.2f}%)")
    
    print(f"  Efficiency:")
    print(f"    Asset Turnover (Avg): {eff['asset_efficiency_avg']:.2f}x (Latest: {eff['asset_efficiency_latest']:.2f}x)")
    print(f"    Trend: {eff['asset_turnover_trend']:+.2f}x")
    
    print(f"  Leverage:")
    print(f"    Debt-to-Assets (Latest): {lev['debt_to_asset_latest']:.2f} (Avg: {lev['debt_to_asset_avg']:.2f})")
    print(f"    Improvement: {lev['debt_to_asset_improvement']:+.2f}")
    
    print(f"  Cash Flow:")
    print(f"    Total Operating Cashflow: ${cf['total_cashflow']/1e9:.1f}B")
    print(f"    Avg Annual: ${cf['avg_annual_cashflow']/1e9:.1f}B")
    print(f"    Cashflow to Revenue: {cf['cashflow_to_revenue']:.1f}%")


if __name__ == "__main__":
    # Test modules
    trends = analyze_trends()
    ratios = calculate_ratios()
