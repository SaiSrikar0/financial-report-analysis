"""
Outlier Treatment Module - Phase 3.2
Detects statistical outliers and provides treatment recommendations
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .data_connection import get_analysis_data


def detect_statistical_outliers(df, method='iqr', threshold=1.5):
    """
    Detect outliers using statistical methods.
    
    Args:
        df: DataFrame with numerical features
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: For IQR: multiplier (1.5 standard), for z-score: threshold (typically 3)
        
    Returns:
        dict with outlier detection results
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
        
        outlier_indices = df[outlier_mask].index.tolist() if len(df[outlier_mask]) > 0 else []
        
        outliers[col] = {
            'method': method,
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(data) * 100) if len(data) > 0 else 0,
            'bounds': {
                'lower': lower_bound if method == 'iqr' else None,
                'upper': upper_bound if method == 'iqr' else None
            },
            'outlier_values': data[outlier_mask].tolist(),
            'outlier_indices': outlier_indices
        }
    
    return outliers


def detect_extreme_values(df, percentile=95):
    """
    Detect extreme values using percentile-based approach.
    
    Args:
        df: DataFrame with numerical features
        percentile: Percentile threshold (e.g., 95 = top 5%)
        
    Returns:
        dict with extreme value detection
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    extreme_values = {}
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        lower_pct = 100 - percentile
        upper_pct = percentile
        
        lower_bound = data.quantile(lower_pct / 100)
        upper_bound = data.quantile(upper_pct / 100)
        
        extreme_low = data[data < lower_bound]
        extreme_high = data[data > upper_bound]
        
        extreme_values[col] = {
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'extreme_low_count': len(extreme_low),
            'extreme_high_count': len(extreme_high),
            'extreme_values_total': len(extreme_low) + len(extreme_high)
        }
    
    return extreme_values


def flag_anomalies(df):
    """
    Flag data anomalies like NaN, Inf, or zero values.
    
    Args:
        df: DataFrame to check
        
    Returns:
        dict with anomaly flags
    """
    anomalies = {}
    
    for col in df.columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum() if df[col].dtype in ['float64', 'float32'] else 0
        zero_count = (df[col] == 0).sum() if df[col].dtype in ['int64', 'int32', 'float64', 'float32'] else 0
        
        anomalies[col] = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_count': zero_count,
            'total_anomalies': nan_count + inf_count + zero_count,
            'anomaly_percentage': ((nan_count + inf_count + zero_count) / len(df) * 100) if len(df) > 0 else 0
        }
    
    return anomalies


def generate_treatment_recommendations(outliers, extreme_values, anomalies):
    """
    Generate treatment recommendations for detected outliers and anomalies.
    
    Args:
        outliers: Outlier detection results
        extreme_values: Extreme value detection results
        anomalies: Anomaly flags
        
    Returns:
        List of treatment recommendations
    """
    recommendations = []
    
    # Outlier recommendations
    for col, result in outliers.items():
        if result['outlier_count'] > 0:
            pct = result['outlier_percentage']
            if pct > 10:
                recommendations.append({
                    'column': col,
                    'issue': f"High outlier percentage ({pct:.1f}%)",
                    'treatment': "Investigate data quality; consider separate analysis for outlier subset",
                    'severity': 'high'
                })
            elif pct > 5:
                recommendations.append({
                    'column': col,
                    'issue': f"Moderate outlier percentage ({pct:.1f}%)",
                    'treatment': "Apply robust scaling or IOR transformation",
                    'severity': 'medium'
                })
            else:
                recommendations.append({
                    'column': col,
                    'issue': f"Low outlier count ({result['outlier_count']})",
                    'treatment': "Safe to use; consider Winsorization if needed",
                    'severity': 'low'
                })
    
    # Anomaly recommendations
    for col, result in anomalies.items():
        if result['total_anomalies'] > 0:
            if result['nan_count'] > 0:
                recommendations.append({
                    'column': col,
                    'issue': f"Missing values ({result['nan_count']})",
                    'treatment': "Impute using median/mean or forward-fill for time-series",
                    'severity': 'medium'
                })
            if result['inf_count'] > 0:
                recommendations.append({
                    'column': col,
                    'issue': f"Infinite values ({result['inf_count']})",
                    'treatment': "Replace with max/min finite values",
                    'severity': 'high'
                })
    
    return recommendations


def run_outlier_treatment():
    """
    Execute complete outlier detection and treatment analysis.
    
    Returns:
        dict with all outlier analysis results
    """
    print("\n" + "="*60)
    print("PHASE 3.2: OUTLIER TREATMENT")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    try:
        _, _, df = get_analysis_data()  # Unpack tuple (standard_df, category_df, merged_df)
        print(f"✓ Data loaded: {len(df)} records, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Detect statistical outliers (IQR method)
    print("\n[2/4] Detecting outliers using IQR method...")
    try:
        outliers_iqr = detect_statistical_outliers(df, method='iqr', threshold=1.5)
        outlier_columns = [col for col, result in outliers_iqr.items() if result['outlier_count'] > 0]
        print(f"✓ IQR method: {len(outlier_columns)} columns with outliers")
        for col in outlier_columns[:5]:  # Show first 5
            count = outliers_iqr[col]['outlier_count']
            pct = outliers_iqr[col]['outlier_percentage']
            print(f"  - {col}: {count} outliers ({pct:.1f}%)")
    except Exception as e:
        print(f"✗ Error: {e}")
        outliers_iqr = {}
    
    # Detect extreme values
    print("\n[3/4] Detecting extreme values (95th percentile)...")
    try:
        extreme_values = detect_extreme_values(df, percentile=95)
        extreme_cols = [col for col, result in extreme_values.items() 
                       if result['extreme_values_total'] > 0]
        print(f"✓ Extreme values detected in {len(extreme_cols)} columns")
        for col in extreme_cols[:5]:
            total = extreme_values[col]['extreme_values_total']
            print(f"  - {col}: {total} extreme values")
    except Exception as e:
        print(f"✗ Error: {e}")
        extreme_values = {}
    
    # Flag anomalies and generate recommendations
    print("\n[4/4] Flagging anomalies and generating recommendations...")
    try:
        anomalies = flag_anomalies(df)
        anomaly_cols = [col for col, result in anomalies.items() 
                       if result['total_anomalies'] > 0]
        print(f"✓ Anomalies detected in {len(anomaly_cols)} columns")
        for col in anomaly_cols[:5]:
            total = anomalies[col]['total_anomalies']
            pct = anomalies[col]['anomaly_percentage']
            print(f"  - {col}: {total} anomalies ({pct:.1f}%)")
        
        # Generate recommendations
        recommendations = generate_treatment_recommendations(outliers_iqr, extreme_values, anomalies)
        print(f"\n✓ Treatment recommendations generated: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. [{rec['severity'].upper()}] {rec['column']}: {rec['treatment']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        recommendations = []
    
    return {
        'raw_data': df,
        'outliers_iqr': outliers_iqr,
        'extreme_values': extreme_values,
        'anomalies': anomalies,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    results = run_outlier_treatment()
    if results:
        print("\n" + "="*60)
        print("Outlier Treatment Analysis Complete!")
        print("="*60)
