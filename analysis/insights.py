# Key Insights & Anomaly Detection
# Phase 3.1: Components 5 & 6

import pandas as pd
import numpy as np
from .data_connection import get_standard_table_data


def extract_key_insights():
    """
    Extract actionable insights from financial data
    
    Returns:
        List of insight strings
    """
    print("\n" + "=" * 60)
    print("PHASE 3.1.5 & 6: KEY INSIGHTS & ANOMALY DETECTION")
    print("=" * 60)
    
    df = get_standard_table_data()
    companies = sorted(df['ticker'].unique())
    
    insights = []
    
    # 1. Overall Market Insights
    print("\n--- Market-Wide Insights ---")
    market_insights = analyze_market_trends(df, companies)
    insights.extend(market_insights)
    for insight in market_insights:
        print(f"  • {insight}")
    
    # 2. Company-Specific Insights
    print("\n--- Company-Specific Insights ---")
    for company in companies:
        company_data = df[df['ticker'] == company].sort_values('date')
        company_insights = analyze_company_trends(company_data, company)
        insights.extend(company_insights)
        for insight in company_insights:
            print(f"  • {insight}")
    
    # 3. Anomalies
    print("\n--- Detected Anomalies ---")
    anomalies = detect_anomalies(df, companies)
    insights.extend(anomalies)
    if anomalies:
        for anomaly in anomalies:
            print(f"  ⚠ {anomaly}")
    else:
        print("  ✓ No significant anomalies detected")
    
    return insights


def analyze_market_trends(df, companies):
    """Analyze overall market trends"""
    insights = []
    
    # Market size growth
    latest_total_revenue = df.sort_values('date').groupby('ticker')['revenue'].tail(1).sum()
    earliest_total_revenue = df.sort_values('date').groupby('ticker')['revenue'].head(1).sum()
    market_growth = ((latest_total_revenue - earliest_total_revenue) / earliest_total_revenue) * 100
    
    insights.append(f"Total market (AAPL+MSFT+GOOGL) revenue grew {market_growth:.1f}% from 2006 to 2025")
    
    # Profitability trends
    avg_profit_margin = df['profit_margin'].mean()
    insights.append(f"Average profit margin across all companies: {avg_profit_margin:.2f}%")
    
    # Leverage analysis
    avg_debt_ratio = df['debt_to_asset'].mean()
    latest_avg_debt = df.sort_values('date').groupby('ticker')['debt_to_asset'].tail(1).mean()
    insights.append(f"Average debt ratio decreased from {df.sort_values('date').groupby('ticker')['debt_to_asset'].head(1).mean():.2f} to {latest_avg_debt:.2f}")
    
    return insights


def analyze_company_trends(company_data, ticker):
    """Analyze company-specific trends"""
    insights = []
    
    # Revenue trajectory
    first_revenue = company_data['revenue'].iloc[0]
    last_revenue = company_data['revenue'].iloc[-1]
    revenue_multiplier = last_revenue / first_revenue
    insights.append(f"{ticker}: Revenue grew {revenue_multiplier:.1f}x (${first_revenue/1e9:.1f}B → ${last_revenue/1e9:.1f}B)")
    
    # Profitability trajectory
    first_margin = company_data['profit_margin'].iloc[0]
    last_margin = company_data['profit_margin'].iloc[-1]
    insights.append(f"{ticker}: Profit margin at {last_margin:.2f}% (from {first_margin:.2f}%)")
    
    # Debt reduction success
    first_debt = company_data['debt_to_asset'].iloc[0]
    last_debt = company_data['debt_to_asset'].iloc[-1]
    if first_debt > last_debt:
        debt_reduction = ((first_debt - last_debt) / first_debt) * 100
        insights.append(f"{ticker}: Reduced debt-to-assets by {debt_reduction:.1f}% (stronger balance sheet)")
    
    # Growth stability
    growth_rates = company_data['revenue_growth'].dropna().values
    if len(growth_rates) > 0:
        avg_growth = np.mean(growth_rates)
        growth_volatility = np.std(growth_rates)
        insights.append(f"{ticker}: Average revenue growth {avg_growth:.1f}% (volatility: {growth_volatility:.1f}%)")
    
    return insights


def detect_anomalies(df, companies):
    """Detect unusual patterns or anomalies"""
    anomalies = []
    
    for company in companies:
        company_data = df[df['ticker'] == company].sort_values('date')
        
        # 1. Sudden profit margin changes
        profit_margins = company_data['profit_margin'].values
        margin_changes = np.diff(profit_margins)
        large_margin_drops = np.where(margin_changes < -5)[0]
        
        if len(large_margin_drops) > 0:
            for idx in large_margin_drops:
                year = company_data.iloc[idx]['date'].year
                change = margin_changes[idx]
                anomalies.append(
                    f"{company} (FY {year}): Profit margin dropped {change:.2f}% - investigate profitability decline"
                )
        
        # 2. Negative growth or profitability
        negative_income = company_data[company_data['net_income'] < 0]
        if len(negative_income) > 0:
            anomalies.append(f"{company} showed negative net income in {len(negative_income)} years")
        
        # 3. Asset efficiency degradation
        efficiency = company_data['asset_efficiency'].values
        recent_efficiency = efficiency[-1]
        historical_avg = np.mean(efficiency[:-1])
        
        if recent_efficiency < (historical_avg * 0.8):  # 20% below average
            anomalies.append(
                f"{company}: Asset efficiency degraded to {recent_efficiency:.2f}x (below historical avg {historical_avg:.2f}x)"
            )
        
        # 4. High debt ratio
        latest_debt = company_data['debt_to_asset'].iloc[-1]
        if latest_debt > 0.7:
            anomalies.append(f"{company}: High debt-to-assets ratio ({latest_debt:.2f}) - potential financial risk")
    
    return anomalies


def generate_insights_report(output_path='analysis_report.txt'):
    """Generate a comprehensive insights report"""
    insights = extract_key_insights()
    
    report = "FINCAST - FINANCIAL ANALYSIS REPORT\n"
    report += "=" * 60 + "\n"
    report += f"Generated: 2026-02-26\n"
    report += f"Total Insights Generated: {len(insights)}\n"
    report += "=" * 60 + "\n\n"
    
    report += "KEY INSIGHTS AND FINDINGS:\n"
    report += "-" * 60 + "\n"
    for i, insight in enumerate(insights, 1):
        report += f"{i}. {insight}\n"
    
    report += "\n" + "=" * 60 + "\n"
    report += "END OF REPORT\n"
    
    return report


if __name__ == "__main__":
    # Test module
    insights = extract_key_insights()
    report = generate_insights_report()
    print("\n" + report)
