"""
Feature Preprocessing Module - Phase 3.2
Prepares features for ML model training (scaling, normalization, encoding)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from .data_connection import get_analysis_data


def scale_features(df, method='standard', exclude_cols=None):
    """
    Scale numerical features using specified method.
    
    Args:
        df: DataFrame with features to scale
        method: 'standard' (z-score), 'minmax', or 'robust'
        exclude_cols: List of column names to exclude from scaling
        
    Returns:
        tuple: (scaled_df, scaler_object)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    df_copy = df.copy()
    
    # Select numerical columns to scale
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scale_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Choose scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    df_copy[scale_cols] = scaler.fit_transform(df_copy[scale_cols])
    
    return df_copy, scaler


def handle_missing_values(df, method='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with potential missing values
        method: 'mean', 'median', 'forward_fill', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if method == 'mean':
        df_copy = df_copy.fillna(df_copy.mean())
    elif method == 'median':
        df_copy = df_copy.fillna(df_copy.median())
    elif method == 'forward_fill':
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        df_copy = df_copy.dropna()
    
    return df_copy


def remove_constant_features(df, threshold=0.0):
    """
    Remove features with zero or near-zero variance.
    
    Args:
        df: DataFrame
        threshold: Variance threshold (0.0 = exact zero variance)
        
    Returns:
        tuple: (df with constant features removed, list of removed columns)
    """
    df_copy = df.copy()
    removed_cols = []
    
    for col in df.select_dtypes(include=[np.number]).columns:
        variance = df_copy[col].var()
        if variance <= threshold:
            removed_cols.append(col)
            df_copy = df_copy.drop(columns=[col])
    
    return df_copy, removed_cols


def fix_infinite_values(df):
    """
    Replace infinite values with max/min finite values.
    
    Args:
        df: DataFrame potentially containing infinite values
        
    Returns:
        DataFrame with infinite values fixed
    """
    df_copy = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        # Find infinite values
        inf_mask = np.isinf(df_copy[col])
        
        if inf_mask.any():
            # Get finite values
            finite_values = df_copy[col][~inf_mask]
            
            if len(finite_values) > 0:
                max_val = finite_values.max()
                min_val = finite_values.min()
                
                # Replace positive inf with max, negative inf with min
                df_copy[col] = df_copy[col].replace(np.inf, max_val)
                df_copy[col] = df_copy[col].replace(-np.inf, min_val)
    
    return df_copy


def encode_categorical(df, categorical_cols=None):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names (defaults to object dtype columns)
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_copy = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply one-hot encoding
    for col in categorical_cols:
        if col in df_copy.columns:
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy = df_copy.drop(columns=[col])
    
    return df_copy


def prepare_ml_dataset(df, scale_method='standard', handle_missing='mean'):
    """
    Prepare complete ML-ready dataset through full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        scale_method: Scaling method for features
        handle_missing: Method to handle missing values
        
    Returns:
        dict with processed data and preprocessing steps
    """
    df_processed = df.copy()
    preprocessing_log = []
    
    # Step 1: Fix infinite values
    initial_inf = np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
    df_processed = fix_infinite_values(df_processed)
    preprocessing_log.append({
        'step': 'Fix infinite values',
        'infinite_values_fixed': initial_inf
    })
    
    # Step 2: Handle missing values
    initial_na = df_processed.isna().sum().sum()
    df_processed = handle_missing_values(df_processed, method=handle_missing)
    preprocessing_log.append({
        'step': 'Handle missing values',
        'missing_values_handled': initial_na,
        'method': handle_missing
    })
    
    # Step 3: Encode categorical features
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_processed = encode_categorical(df_processed, categorical_cols)
        preprocessing_log.append({
            'step': 'Encode categorical features',
            'categorical_features': categorical_cols
        })
    
    # Step 4: Remove constant features
    initial_cols = len(df_processed.columns)
    df_processed, removed_cols = remove_constant_features(df_processed, threshold=0.0)
    preprocessing_log.append({
        'step': 'Remove constant features',
        'constant_features_removed': len(removed_cols),
        'removed_columns': removed_cols
    })
    
    # Step 5: Scale features
    exclude_from_scaling = ['ticker', 'company', 'fiscal_year', 'category', 'growth_category']
    df_processed, scaler = scale_features(df_processed, method=scale_method, exclude_cols=exclude_from_scaling)
    preprocessing_log.append({
        'step': 'Scale features',
        'method': scale_method,
        'features_scaled': len(df_processed.select_dtypes(include=[np.number]).columns)
    })
    
    return {
        'processed_data': df_processed,
        'preprocessing_steps': preprocessing_log,
        'scaler': scaler,
        'initial_shape': df.shape,
        'final_shape': df_processed.shape,
        'features_list': df_processed.columns.tolist()
    }


def run_feature_preprocessing():
    """
    Execute complete feature preprocessing pipeline.
    
    Returns:
        dict with preprocessing results
    """
    print("\n" + "="*60)
    print("PHASE 3.2: FEATURE PREPROCESSING")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading raw data...")
    try:
        _, _, df = get_analysis_data()  # Unpack tuple (standard_df, category_df, merged_df)
        print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Apply preprocessing pipeline
    print("\n[2/5] Fixing infinite values...")
    try:
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        df = fix_infinite_values(df)
        print(f"✓ Fixed {inf_count} infinite values")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[3/5] Handling missing values...")
    try:
        na_count = df.isna().sum().sum()
        df = handle_missing_values(df, method='median')
        print(f"✓ Handled {na_count} missing values (using median)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[4/5] Encoding categorical features...")
    try:
        categorical = df.select_dtypes(include=['object']).columns.tolist()
        df = encode_categorical(df, categorical_cols=categorical)
        print(f"✓ Encoded {len(categorical)} categorical features: {', '.join(categorical)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[5/5] Scaling and final preparation...")
    try:
        results = prepare_ml_dataset(df, scale_method='standard', handle_missing='median')
        print(f"✓ Preprocessing complete!")
        print(f"  - Shape: {results['initial_shape']} → {results['final_shape']}")
        print(f"  - Features prepared: {len(results['features_list'])}")
        
        print(f"\n  Preprocessing pipeline steps:")
        for i, step in enumerate(results['preprocessing_steps'], 1):
            print(f"  {i}. {step['step']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    return results


if __name__ == "__main__":
    results = run_feature_preprocessing()
    if results:
        print("\n" + "="*60)
        print("Feature Preprocessing Complete!")
        print("="*60)
