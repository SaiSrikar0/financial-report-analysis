# Historical Performance Analysis
# Phase 3.1: Component 1

import pandas as pd
import numpy as np
from .data_connection import get_standard_table_data, get_company_data


def analyze_historical_performance(ticker=None):
    """
    Analyze historical performance for company/companies
    
    Args:
        ticker: Specific ticker or None for all companies
    
    Returns:
        Dictionary with performance metrics
    """
    print("=" * 60)
    print("PHASE 3.1.1: HISTORICAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if ticker:
        df = get_company_data(ticker)
        companies = [ticker]
    else:
        df = get_standard_table_data()
        companies = sorted(df['ticker'].unique())
    
    performance_report = {}
    
    for company in companies:
        company_data = df[df['ticker'] == company].sort_values('date')
        
        if len(company_data) == 0:
            continue
        
        performance = calculate_performance_metrics(company_data, company)
        performance_report[company] = performance
        print_performance_summary(company, performance)
    
    return performance_report


def calculate_performance_metrics(company_data, ticker):
    """
    Calculate comprehensive performance metrics for a company
    
    Args:
        company_data: DataFrame with company records
        ticker: Company ticker
    
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        'ticker': ticker,
        'period_start': company_data['date'].min(),
        'period_end': company_data['date'].max(),
        'num_years': len(company_data),
    }
    
    # Revenue Analysis
    first_revenue = company_data['revenue'].iloc[0]
    last_revenue = company_data['revenue'].iloc[-1]
    metrics['revenue_cagr'] = calculate_cagr(first_revenue, last_revenue, len(company_data))
    metrics['revenue_start'] = first_revenue
    metrics['revenue_end'] = last_revenue
    metrics['revenue_total_growth'] = ((last_revenue - first_revenue) / first_revenue) * 100
    
    # Profitability Analysis
    first_net_income = company_data['net_income'].iloc[0]
    last_net_income = company_data['net_income'].iloc[-1]
    metrics['net_income_cagr'] = calculate_cagr(first_net_income, last_net_income, len(company_data))
    metrics['net_income_start'] = first_net_income
    metrics['net_income_end'] = last_net_income
    metrics['profit_margin_avg'] = company_data['profit_margin'].mean()
    metrics['profit_margin_last'] = company_data['profit_margin'].iloc[-1]
    
    # Operating Performance
    metrics['operating_margin_avg'] = company_data['operating_margin'].mean()
    metrics['operating_margin_trend'] = (
        company_data['operating_margin'].iloc[-1] - company_data['operating_margin'].iloc[0]
    )
    
    # Cash Flow Analysis
    metrics['total_cashflow'] = company_data['operating_cashflow'].sum()
    metrics['avg_annual_cashflow'] = company_data['operating_cashflow'].mean()
    
    # Balance Sheet Health
    first_debt_ratio = company_data['debt_to_asset'].iloc[0]
    last_debt_ratio = company_data['debt_to_asset'].iloc[-1]
    metrics['debt_ratio_start'] = first_debt_ratio
    metrics['debt_ratio_end'] = last_debt_ratio
    metrics['debt_ratio_improvement'] = first_debt_ratio - last_debt_ratio
    
    # Asset Efficiency
    metrics['asset_efficiency_avg'] = company_data['asset_efficiency'].mean()
    metrics['asset_efficiency_trend'] = (
        company_data['asset_efficiency'].iloc[-1] - company_data['asset_efficiency'].iloc[0]
    )
    
    # Performance Classification
    metrics['performance_class'] = classify_performance(metrics)
    
    return metrics


def calculate_cagr(start_value, end_value, num_periods):
    """
    Calculate Compound Annual Growth Rate
    
    Args:
        start_value: Initial value
        end_value: Final value
        num_periods: Number of periods
    
    Returns:
        CAGR percentage
    """
    if start_value <= 0 or end_value <= 0:
        return None
    
    cagr = (pow(end_value / start_value, 1 / num_periods) - 1) * 100
    return cagr


def classify_performance(metrics):
    """
    Classify overall performance based on multiple factors
    
    Returns:
        Performance classification
    """
    # Score based on revenue growth
    revenue_growth = metrics['revenue_total_growth']
    
    # Score based on profitability
    profit_margin = metrics['profit_margin_last']
    
    # Score based on debt improvement
    debt_improvement = metrics['debt_ratio_improvement']
    
    # Calculate composite score
    score = 0
    if revenue_growth > 50:
        score += 3  # Excellent growth
    elif revenue_growth > 20:
        score += 2  # Good growth
    elif revenue_growth > 0:
        score += 1  # Moderate growth
    else:
        score -= 1  # Declining
    
    if profit_margin > 20:
        score += 2
    elif profit_margin > 10:
        score += 1
    
    if debt_improvement > 0:
        score += 1
    
    if score >= 5:
        return "Excellent"
    elif score >= 2:
        return "Good"
    elif score >= 0:
        return "Moderate"
    else:
        return "Concerning"


def print_performance_summary(ticker, metrics):
    """Print formatted performance summary"""
    print(f"\n{ticker} - {metrics['period_start'].strftime('%Y-%m-%d')} to {metrics['period_end'].strftime('%Y-%m-%d')}")
    print("-" * 60)
    print(f"  Performance Class: {metrics['performance_class']}")
    print(f"  Revenue CAGR: {metrics['revenue_cagr']:.2f}%")
    print(f"  Revenue Growth: ${metrics['revenue_start']/1e9:.1f}B → ${metrics['revenue_end']/1e9:.1f}B ({metrics['revenue_total_growth']:.1f}%)")
    print(f"  Net Income CAGR: {metrics['net_income_cagr']:.2f}%")
    print(f"  Avg Profit Margin: {metrics['profit_margin_avg']:.2f}%")
    print(f"  Current Profit Margin: {metrics['profit_margin_last']:.2f}%")
    print(f"  Avg Operating Margin: {metrics['operating_margin_avg']:.2f}%")
    print(f"  Total Operating Cashflow: ${metrics['total_cashflow']/1e9:.1f}B")
    print(f"  Debt-to-Assets: {metrics['debt_ratio_start']:.2f} → {metrics['debt_ratio_end']:.2f} (Δ {metrics['debt_ratio_improvement']:.2f})")
    print(f"  Asset Efficiency (Avg): {metrics['asset_efficiency_avg']:.2f}x")


if __name__ == "__main__":
    # Test the module
    report = analyze_historical_performance()
