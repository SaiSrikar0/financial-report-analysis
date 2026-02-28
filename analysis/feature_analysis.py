"""
Feature Analysis Module - Phase 3.2
Calculates feature correlations, importance, and redundancy analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from .data_connection import get_analysis_data


def calculate_correlations(df):
    """
    Calculate correlation matrix between all numerical features.
    
    Args:
        df: DataFrame with numerical features
        
    Returns:
        dict with correlation matrix and high-correlation pairs
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold: 0.7
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs
    }


def calculate_feature_importance(df, target_col='net_profit'):
    """
    Calculate feature importance using Random Forest regression.
    
    Args:
        df: DataFrame with features
        target_col: Target variable for importance calculation
        
    Returns:
        dict with feature importance rankings and scores
    """
    if target_col not in df.columns:
        # Try alternative target columns
        possible_targets = ['net_profit', 'total_revenue', 'profit_margin']
        target_col = next((col for col in possible_targets if col in df.columns), None)
        if target_col is None:
            raise ValueError("No suitable target column found for feature importance")
    
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col != target_col]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    
    # Get importance scores
    importance_scores = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_scores['cumulative_importance'] = importance_scores['importance'].cumsum()
    importance_scores['cumulative_importance_pct'] = (
        importance_scores['cumulative_importance'] / importance_scores['importance'].sum() * 100
    )
    
    return {
        'feature_importance': importance_scores,
        'model': rf_model,
        'target_column': target_col
    }


def identify_redundant_features(corr_matrix, threshold=0.8):
    """
    Identify redundant features based on high correlation.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold for redundancy
        
    Returns:
        List of redundant feature pairs
    """
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                redundant_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j],
                    'recommendation': f"Consider removing {corr_matrix.columns[j]} (highly correlated with {corr_matrix.columns[i]})"
                })
    return redundant_pairs


def analyze_feature_variance(df):
    """
    Analyze variance and standard deviation of features.
    
    Args:
        df: DataFrame with numerical features
        
    Returns:
        DataFrame with variance statistics
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    variance_stats = pd.DataFrame({
        'feature': numerical_cols,
        'mean': df[numerical_cols].mean(),
        'std_dev': df[numerical_cols].std(),
        'min': df[numerical_cols].min(),
        'max': df[numerical_cols].max(),
        'coefficient_of_variation': (df[numerical_cols].std() / df[numerical_cols].mean().abs()).replace([np.inf, -np.inf], 0)
    }).sort_values('std_dev', ascending=False)
    
    return variance_stats


def run_feature_analysis():
    """
    Execute complete feature analysis.
    
    Returns:
        dict with all analysis results
    """
    print("\n" + "="*60)
    print("PHASE 3.2: FEATURE ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data from Supabase...")
    try:
        _, _, df = get_analysis_data()  # Unpack tuple (standard_df, category_df, merged_df)
        print(f"✓ Data loaded: {len(df)} records")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Calculate correlations
    print("\n[2/5] Calculating feature correlations...")
    try:
        corr_results = calculate_correlations(df)
        print(f"✓ Correlation matrix calculated: {corr_results['correlation_matrix'].shape}")
        if corr_results['high_correlation_pairs']:
            print(f"✓ High correlation pairs found: {len(corr_results['high_correlation_pairs'])}")
        else:
            print("✓ No high correlations (>0.7) detected")
    except Exception as e:
        print(f"✗ Error: {e}")
        corr_results = None
    
    # Calculate feature importance
    print("\n[3/5] Calculating feature importance...")
    try:
        importance_results = calculate_feature_importance(df)
        print(f"✓ Feature importance calculated")
        print(f"✓ Top 5 important features:")
        for idx, row in importance_results['feature_importance'].head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f} ({row['cumulative_importance_pct']:.1f}% cumulative)")
    except Exception as e:
        print(f"✗ Error: {e}")
        importance_results = None
    
    # Identify redundant features
    print("\n[4/5] Identifying redundant features...")
    try:
        if corr_results:
            redundant = identify_redundant_features(corr_results['correlation_matrix'], threshold=0.8)
            print(f"✓ Redundant features identified: {len(redundant)}")
            for pair in redundant:
                print(f"  - {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
        else:
            redundant = []
    except Exception as e:
        print(f"✗ Error: {e}")
        redundant = []
    
    # Analyze variance
    print("\n[5/5] Analyzing feature variance...")
    try:
        variance_stats = analyze_feature_variance(df)
        print(f"✓ Variance analysis complete")
        print(f"✓ Top 5 highest variance features:")
        for idx, row in variance_stats.head(5).iterrows():
            print(f"  - {row['feature']}: σ={row['std_dev']:.4f}, CV={row['coefficient_of_variation']:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        variance_stats = None
    
    return {
        'raw_data': df,
        'correlation_results': corr_results,
        'importance_results': importance_results,
        'redundant_features': redundant,
        'variance_stats': variance_stats
    }


if __name__ == "__main__":
    results = run_feature_analysis()
    if results:
        print("\n" + "="*60)
        print("Feature Analysis Complete!")
        print("="*60)
