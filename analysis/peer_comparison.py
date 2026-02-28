# Peer Comparison Analysis
# Phase 3.1: Component 4

import pandas as pd
import numpy as np
from .data_connection import get_standard_table_data


def compare_peers():
    """
    Compare all companies across key metrics
    
    Returns:
        Dictionary with comparison matrices
    """
    print("\n" + "=" * 60)
    print("PHASE 3.1.4: PEER COMPARISON ANALYSIS")
    print("=" * 60)
    
    df = get_standard_table_data()
    companies = sorted(df['ticker'].unique())
    
    if len(companies) < 2:
        print("✗ Need at least 2 companies for peer comparison")
        return
    
    # Get latest year data for each company
    latest_data = df.sort_values('date').groupby('ticker').tail(1)
    
    comparison = {}
    
    # Latest Financial Metrics Comparison
    print("\n--- Latest Financial Metrics (Most Recent Year) ---")
    metrics_df = latest_data[['ticker', 'date', 'revenue', 'operating_income', 'net_income', 'operating_cashflow']].copy()
    metrics_df['revenue_billions'] = metrics_df['revenue'] / 1e9
    metrics_df['net_income_billions'] = metrics_df['net_income'] / 1e9
    
    print(metrics_df[['ticker', 'date', 'revenue_billions', 'net_income_billions']].to_string(index=False))
    
    # Profitability Comparison
    print("\n--- Profitability Metrics (Latest Year) ---")
    profit_df = latest_data[['ticker', 'profit_margin', 'operating_margin']].copy()
    profit_df = profit_df.sort_values('profit_margin', ascending=False)
    print(profit_df.to_string(index=False))
    
    comparison['profitability'] = {
        'leader': profit_df.iloc[0]['ticker'],
        'profit_margin_range': (profit_df['profit_margin'].min(), profit_df['profit_margin'].max()),
        'operating_margin_range': (profit_df['operating_margin'].min(), profit_df['operating_margin'].max()),
    }
    
    # Growth Comparison (Average historical growth rates)
    print("\n--- Growth Rates (Average Historical) ---")
    growth_df = pd.DataFrame()
    for company in companies:
        company_data = df[df['ticker'] == company]
        avg_revenue_growth = company_data['revenue_growth'].mean()
        avg_income_growth = company_data['net_income_growth'].mean()
        
        growth_df = pd.concat([growth_df, pd.DataFrame({
            'ticker': [company],
            'avg_revenue_growth': [avg_revenue_growth],
            'avg_income_growth': [avg_income_growth],
        })], ignore_index=True)
    
    growth_df = growth_df.sort_values('avg_revenue_growth', ascending=False)
    print(growth_df[['ticker', 'avg_revenue_growth', 'avg_income_growth']].to_string(index=False))
    
    comparison['growth'] = {
        'leader': growth_df.iloc[0]['ticker'],
        'revenue_growth_leader_rate': growth_df.iloc[0]['avg_revenue_growth'],
    }
    
    # Efficiency Comparison
    print("\n--- Efficiency Ratios (Latest Year) ---")
    efficiency_df = latest_data[['ticker', 'asset_efficiency', 'debt_to_asset']].copy()
    efficiency_df = efficiency_df.sort_values('asset_efficiency', ascending=False)
    print(efficiency_df.to_string(index=False))
    
    comparison['efficiency'] = {
        'leader': efficiency_df.iloc[0]['ticker'],
        'asset_efficiency_range': (efficiency_df['asset_efficiency'].min(), efficiency_df['asset_efficiency'].max()),
    }
    
    # Financial Health Comparison
    print("\n--- Financial Health (Latest Year) ---")
    health_df = latest_data[['ticker', 'debt_to_asset']].copy()
    health_df['equity_ratio'] = 1 - health_df['debt_to_asset']
    health_df = health_df.sort_values('debt_to_asset')  # Lower debt is better
    
    print(health_df[['ticker', 'debt_to_asset', 'equity_ratio']].to_string(index=False))
    
    comparison['financial_health'] = {
        'lowest_debt': health_df.iloc[0]['ticker'],
        'debt_ratio_range': (health_df['debt_to_asset'].min(), health_df['debt_to_asset'].max()),
    }
    
    # Overall Competitive Position
    print("\n--- Competitive Positioning ---")
    print_competitive_summary(comparison, companies)
    
    return comparison


def print_competitive_summary(comparison, companies):
    """Print competitive positioning summary"""
    print(f"Profitability Leader: {comparison['profitability']['leader']}")
    print(f"Growth Leader: {comparison['growth']['leader']}")
    print(f"Efficiency Leader: {comparison['efficiency']['leader']}")
    print(f"Financial Health Leader (Lowest Debt): {comparison['financial_health']['lowest_debt']}")
    
    print("\nTimeline: 2006-2025 (20 years)")
    print(f"Companies Analyzed: {', '.join(companies)}")


def get_peer_rankings():
    """
    Get ranking of peers across different dimensions
    
    Returns:
        DataFrame with rankings
    """
    df = get_standard_table_data()
    latest_data = df.sort_values('date').groupby('ticker').tail(1)
    
    rankings = pd.DataFrame({
        'Company': latest_data['ticker'].values,
        'Profitability (Profit Margin %)': latest_data['profit_margin'].values,
        'Asset Efficiency': latest_data['asset_efficiency'].values,
        'Financial Health (1 - Debt Ratio)': (1 - latest_data['debt_to_asset']).values,
    })
    
    # Rank each column (1 = best)
    for col in rankings.columns:
        if col != 'Company':
            rankings[f'{col} Rank'] = rankings[col].rank(ascending=False).astype(int)
    
    return rankings


if __name__ == "__main__":
    # Test module
    comparison = compare_peers()
    rankings = get_peer_rankings()
    print("\n" + rankings.to_string(index=False))
