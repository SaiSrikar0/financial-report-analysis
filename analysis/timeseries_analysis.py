"""
Time Series Analysis Module - Phase 3.2
Decomposes time-series patterns, detects seasonality, and identifies trends
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from .data_connection import get_standard_table_data


def decompose_timeseries(df, ticker=None):
    """
    Decompose time-series into trend, seasonal, and residual components.
    
    Args:
        df: DataFrame with date and financial metrics
        ticker: Company ticker (optional, for filtering)
        
    Returns:
        dict with decomposition components for each metric
    """
    if ticker:
        df = df[df['ticker'] == ticker].copy()
    
    df = df.sort_values('fiscal_year')
    
    decomposition_results = {}
    metrics = ['total_revenue', 'net_profit', 'operating_expenses']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        values = df[metric].fillna(0).values
        n = len(values)
        
        if n < 3:  # Need minimum points for decomposition
            continue
        
        # Trend: fit linear regression
        X = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        trend = model.predict(X)
        
        # Detrended values
        detrended = values - trend
        
        # Seasonal: extract repeating pattern (simplified)
        # For financial data, we look for patterns every 4 quarters or 1 year
        period = min(4, n // 2)
        seasonal = np.zeros(n)
        if period > 1:
            for i in range(n):
                seasonal[i] = np.mean([detrended[j] for j in range(i % period, n, period)])
        
        # Residual
        residual = values - trend - seasonal
        
        decomposition_results[metric] = {
            'original': values,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'trend_strength': 1 - (np.var(residual) / np.var(trend + residual)) if np.var(trend + residual) > 0 else 0,
            'seasonal_strength': 1 - (np.var(residual) / np.var(seasonal + residual)) if np.var(seasonal + residual) > 0 else 0
        }
    
    return decomposition_results


def detect_seasonality(df, ticker=None, period=4):
    """
    Detect seasonal patterns in financial data.
    
    Args:
        df: DataFrame with date and metrics
        ticker: Company ticker (optional)
        period: Period for seasonality check (default 4 quarters)
        
    Returns:
        dict with seasonality detection results
    """
    if ticker:
        df = df[df['ticker'] == ticker].copy()
    
    df = df.sort_values('fiscal_year')
    
    seasonality_results = {}
    metrics = ['total_revenue', 'net_profit', 'profit_margin']
    
    for metric in metrics:
        if metric not in df.columns or len(df) < period:
            continue
        
        values = df[metric].fillna(0).values
        
        # Check for regular patterns
        seasonal_indices = []
        for i in range(0, len(values) - period):
            segment = values[i:i+period]
            seasonal_indices.append(np.std(segment))
        
        avg_volatility = np.mean(seasonal_indices) if seasonal_indices else 0
        
        # Detect if pattern repeats
        is_seasonal = avg_volatility > np.std(values) * 0.5 if np.std(values) > 0 else False
        
        seasonality_results[metric] = {
            'is_seasonal': is_seasonal,
            'volatility': avg_volatility,
            'overall_std': np.std(values),
            'pattern_strength': avg_volatility / (np.std(values) + 1e-10)
        }
    
    return seasonality_results


def identify_growth_periods(df, ticker=None):
    """
    Identify periods of growth, decline, and stability.
    
    Args:
        df: DataFrame with date and metrics
        ticker: Company ticker (optional)
        
    Returns:
        dict with period classifications
    """
    if ticker:
        df = df[df['ticker'] == ticker].copy()
    
    df = df.sort_values('fiscal_year')
    
    periods = []
    years = df['fiscal_year'].values
    revenues = df['total_revenue'].fillna(0).values
    
    for i in range(1, len(revenues)):
        yoy_growth = ((revenues[i] - revenues[i-1]) / (revenues[i-1] + 1e-10)) * 100
        
        if abs(revenues[i]) < 1e-10:
            period_type = "No data"
        elif yoy_growth > 10:
            period_type = "Strong growth"
        elif yoy_growth > 0:
            period_type = "Moderate growth"
        elif yoy_growth > -5:
            period_type = "Stable/slight decline"
        else:
            period_type = "Significant decline"
        
        periods.append({
            'year': years[i],
            'prev_year': years[i-1],
            'yoy_growth_pct': yoy_growth,
            'period_type': period_type
        })
    
    return periods


def calculate_trend_slope(df, ticker=None):
    """
    Calculate slope (rate of change) for key metrics.
    
    Args:
        df: DataFrame with financial data
        ticker: Company ticker (optional)
        
    Returns:
        dict with trend slopes for each metric
    """
    if ticker:
        df = df[df['ticker'] == ticker].copy()
    
    df = df.sort_values('fiscal_year')
    
    trend_slopes = {}
    metrics = ['total_revenue', 'net_profit', 'operating_expenses', 'profit_margin']
    
    for metric in metrics:
        if metric not in df.columns or len(df) < 2:
            continue
        
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[metric].fillna(0).values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slopes[metric] = {
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'slope_interpretation': "increasing" if model.coef_[0] > 0 else "decreasing",
            'r_squared': model.score(X, y)
        }
    
    return trend_slopes


def run_timeseries_analysis():
    """
    Execute complete time-series analysis.
    
    Returns:
        dict with all analysis results
    """
    print("\n" + "="*60)
    print("PHASE 3.2: TIME-SERIES ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading time-series data...")
    try:
        df = get_standard_table_data()
        print(f"✓ Data loaded: {len(df)} records")
        companies = df['ticker'].unique()
        print(f"✓ Companies: {', '.join(companies)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Decompose time-series
    print("\n[2/4] Decomposing time-series by company...")
    decomposition_by_company = {}
    try:
        for company in companies:
            decomp = decompose_timeseries(df, ticker=company)
            decomposition_by_company[company] = decomp
            print(f"✓ {company}: Decomposed {len(decomp)} metrics")
            for metric, result in decomp.items():
                print(f"  - {metric}: trend_strength={result['trend_strength']:.3f}, seasonal_strength={result['seasonal_strength']:.3f}")
    except Exception as e:
        print(f"✗ Error in decomposition: {e}")
    
    # Detect seasonality
    print("\n[3/4] Detecting seasonality patterns...")
    seasonality_by_company = {}
    try:
        for company in companies:
            seasonality = detect_seasonality(df, ticker=company)
            seasonality_by_company[company] = seasonality
            seasonal_count = sum(1 for v in seasonality.values() if v['is_seasonal'])
            print(f"✓ {company}: Seasonal patterns detected in {seasonal_count}/{len(seasonality)} metrics")
    except Exception as e:
        print(f"✗ Error in seasonality detection: {e}")
    
    # Identify growth periods and trend slopes
    print("\n[4/4] Identifying growth periods and trend slopes...")
    growth_periods_by_company = {}
    trend_slopes_by_company = {}
    try:
        for company in companies:
            growth = identify_growth_periods(df, ticker=company)
            trends = calculate_trend_slope(df, ticker=company)
            
            growth_periods_by_company[company] = growth
            trend_slopes_by_company[company] = trends
            
            print(f"\n✓ {company} Growth Analysis:")
            strong_growth = sum(1 for p in growth if "growth" in p['period_type'].lower())
            print(f"  - Growth periods: {strong_growth}/{len(growth)}")
            print(f"  - Trend slopes:")
            for metric, trend in trends.items():
                print(f"    * {metric}: {trend['slope_interpretation']} (slope={trend['slope']:.2f})")
    except Exception as e:
        print(f"✗ Error in growth analysis: {e}")
    
    return {
        'raw_data': df,
        'decomposition_results': decomposition_by_company,
        'seasonality_results': seasonality_by_company,
        'growth_periods': growth_periods_by_company,
        'trend_slopes': trend_slopes_by_company
    }


if __name__ == "__main__":
    results = run_timeseries_analysis()
    if results:
        print("\n" + "="*60)
        print("Time-Series Analysis Complete!")
        print("="*60)
