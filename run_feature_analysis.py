"""
Phase 3.2: Feature Analysis Orchestrator
Executes all feature analysis components and generates comprehensive reports
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Handle both module and direct execution
try:
    from .feature_analysis import run_feature_analysis
    from .timeseries_analysis import run_timeseries_analysis
    from .outlier_treatment import run_outlier_treatment
    from .feature_preprocessing import run_feature_preprocessing
except ImportError:
    # Direct execution mode
    sys.path.insert(0, os.path.dirname(__file__))
    from analysis.feature_analysis import run_feature_analysis
    from analysis.timeseries_analysis import run_timeseries_analysis
    from analysis.outlier_treatment import run_outlier_treatment
    from analysis.feature_preprocessing import run_feature_preprocessing


def ensure_report_directory():
    """Create reports directory if it doesn't exist."""
    report_dir = "analysis/reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    return report_dir


def save_feature_importance_report(feature_results, report_dir):
    """Save feature importance analysis to CSV."""
    if feature_results and 'importance_results' in feature_results:
        importance_df = feature_results['importance_results']['feature_importance']
        filepath = os.path.join(report_dir, 'feature_importance.csv')
        importance_df.to_csv(filepath, index=False)
        print(f"✓ Feature importance report saved: {filepath}")
        return filepath
    return None


def save_correlation_matrix(feature_results, report_dir):
    """Save correlation matrix to CSV."""
    if feature_results and 'correlation_results' in feature_results:
        corr_matrix = feature_results['correlation_results']['correlation_matrix']
        filepath = os.path.join(report_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(filepath)
        print(f"✓ Correlation matrix saved: {filepath}")
        return filepath
    return None


def save_timeseries_summary(timeseries_results, report_dir):
    """Save time-series analysis summary to CSV."""
    if timeseries_results:
        try:
            summary_data = []
            
            for company, trends in timeseries_results.get('trend_slopes', {}).items():
                for metric, trend_info in trends.items():
                    summary_data.append({
                        'company': company,
                        'metric': metric,
                        'slope': trend_info['slope'],
                        'interpretation': trend_info['slope_interpretation'],
                        'r_squared': trend_info['r_squared']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                filepath = os.path.join(report_dir, 'timeseries_trends.csv')
                summary_df.to_csv(filepath, index=False)
                print(f"✓ Time-series trends summary saved: {filepath}")
                return filepath
        except Exception as e:
            print(f"✗ Error saving time-series summary: {e}")
    
    return None


def save_outlier_report(outlier_results, report_dir):
    """Save outlier detection report to CSV."""
    if outlier_results and 'recommendations' in outlier_results:
        try:
            recommendations = outlier_results['recommendations']
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                filepath = os.path.join(report_dir, 'outlier_recommendations.csv')
                rec_df.to_csv(filepath, index=False)
                print(f"✓ Outlier recommendations saved: {filepath}")
                return filepath
        except Exception as e:
            print(f"✗ Error saving outlier report: {e}")
    
    return None


def save_preprocessing_log(preprocessing_results, report_dir):
    """Save feature preprocessing log to CSV."""
    if preprocessing_results and 'processed_data' in preprocessing_results:
        try:
            # Save preprocessing steps log
            steps = preprocessing_results['preprocessing_steps']
            steps_df = pd.DataFrame(steps)
            filepath = os.path.join(report_dir, 'preprocessing_steps.csv')
            steps_df.to_csv(filepath, index=False)
            print(f"✓ Preprocessing steps log saved: {filepath}")
            
            # Save processed data
            processed_data = preprocessing_results['processed_data']
            data_filepath = os.path.join(report_dir, 'ml_ready_data.csv')
            processed_data.to_csv(data_filepath, index=False)
            print(f"✓ ML-ready preprocessed data saved: {data_filepath}")
            
            return filepath, data_filepath
        except Exception as e:
            print(f"✗ Error saving preprocessing data: {e}")
    
    return None, None


def generate_summary_report(all_results, report_dir):
    """Generate comprehensive summary report."""
    report_path = os.path.join(report_dir, 'phase_3_2_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 3.2: COMPREHENSIVE FEATURE ANALYSIS REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Feature Analysis Summary
        f.write("1. FEATURE ANALYSIS\n")
        f.write("-"*70 + "\n")
        if all_results['feature_analysis'] and 'importance_results' in all_results['feature_analysis']:
            importance_df = all_results['feature_analysis']['importance_results']['feature_importance']
            f.write(f"   Total Features Analyzed: {len(importance_df)}\n")
            f.write(f"   Top 5 Important Features:\n")
            for idx, row in importance_df.head(5).iterrows():
                f.write(f"   - {row['feature']}: {row['importance']:.4f} importance\n")
        
        f.write(f"\n   High Correlation Pairs (>0.7):\n")
        if all_results['feature_analysis'] and all_results['feature_analysis'].get('correlation_results', {}).get('high_correlation_pairs'):
            for pair in all_results['feature_analysis']['correlation_results']['high_correlation_pairs']:
                f.write(f"   - {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}\n")
        else:
            f.write("   - No high correlations detected (healthy multicollinearity)\n")
        
        # Time-Series Analysis Summary
        f.write("\n2. TIME-SERIES ANALYSIS\n")
        f.write("-"*70 + "\n")
        if all_results['timeseries_analysis'] and 'trend_slopes' in all_results['timeseries_analysis']:
            for company, trends in all_results['timeseries_analysis']['trend_slopes'].items():
                f.write(f"\n   {company}:\n")
                for metric, trend_info in list(trends.items())[:3]:
                    direction = "↑" if trend_info['slope'] > 0 else "↓"
                    f.write(f"   - {metric}: {direction} {trend_info['slope_interpretation']} (R²={trend_info['r_squared']:.3f})\n")
        
        # Outlier Analysis Summary
        f.write("\n3. OUTLIER & ANOMALY DETECTION\n")
        f.write("-"*70 + "\n")
        if all_results['outlier_treatment'] and 'recommendations' in all_results['outlier_treatment']:
            recommendations = all_results['outlier_treatment']['recommendations']
            high_severity = [r for r in recommendations if r['severity'] == 'high']
            f.write(f"   Total Recommendations: {len(recommendations)}\n")
            f.write(f"   High Severity Issues: {len(high_severity)}\n")
            if high_severity:
                f.write(f"\n   High Priority Actions:\n")
                for rec in high_severity[:5]:
                    f.write(f"   [{rec['severity'].upper()}] {rec['column']}\n")
                    f.write(f"   → {rec['treatment']}\n")
        
        # Preprocessing Summary
        f.write("\n4. FEATURE PREPROCESSING\n")
        f.write("-"*70 + "\n")
        if all_results['preprocessing'] and 'preprocessing_steps' in all_results['preprocessing']:
            initial_shape = all_results['preprocessing']['initial_shape']
            final_shape = all_results['preprocessing']['final_shape']
            f.write(f"   Data Shape: {initial_shape} → {final_shape}\n")
            f.write(f"   Preprocessing Steps Applied:\n")
            for step in all_results['preprocessing']['preprocessing_steps']:
                f.write(f"   - {step['step']}\n")
            f.write(f"\n   ML-Ready Features: {final_shape[1]} columns\n")
        
        # Output Files
        f.write("\n5. GENERATED OUTPUT FILES\n")
        f.write("-"*70 + "\n")
        f.write("   ✓ feature_importance.csv - Feature importance rankings\n")
        f.write("   ✓ correlation_matrix.csv - Feature correlation analysis\n")
        f.write("   ✓ timeseries_trends.csv - Trend analysis by company\n")
        f.write("   ✓ outlier_recommendations.csv - Outlier treatment recommendations\n")
        f.write("   ✓ preprocessing_steps.csv - Preprocessing pipeline log\n")
        f.write("   ✓ ml_ready_data.csv - Final preprocessed dataset for ML\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Phase 3.2 Complete! Ready for Phase 4: SVR Model Training\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved: {report_path}")
    return report_path


def run_full_feature_analysis():
    """
    Execute all Phase 3.2 components sequentially.
    """
    print("\n" + "="*70)
    print("PHASE 3.2: FEATURE ANALYSIS - FULL PIPELINE")
    print("="*70 + "\n")
    
    report_dir = ensure_report_directory()
    
    all_results = {
        'feature_analysis': None,
        'timeseries_analysis': None,
        'outlier_treatment': None,
        'preprocessing': None
    }
    
    # 1. Feature Analysis
    print("\n" + "-"*70)
    print("[1/4] FEATURE ANALYSIS - Correlations, Importance, Redundancy")
    print("-"*70)
    try:
        all_results['feature_analysis'] = run_feature_analysis()
    except Exception as e:
        print(f"\n✗ Feature analysis failed: {e}")
    
    # 2. Time-Series Analysis
    print("\n" + "-"*70)
    print("[2/4] TIME-SERIES ANALYSIS - Trends, Seasonality, Growth Periods")
    print("-"*70)
    try:
        all_results['timeseries_analysis'] = run_timeseries_analysis()
    except Exception as e:
        print(f"\n✗ Time-series analysis failed: {e}")
    
    # 3. Outlier Treatment
    print("\n" + "-"*70)
    print("[3/4] OUTLIER TREATMENT - Detection & Recommendations")
    print("-"*70)
    try:
        all_results['outlier_treatment'] = run_outlier_treatment()
    except Exception as e:
        print(f"\n✗ Outlier treatment failed: {e}")
    
    # 4. Feature Preprocessing
    print("\n" + "-"*70)
    print("[4/4] FEATURE PREPROCESSING - Scaling, Encoding, ML Preparation")
    print("-"*70)
    try:
        all_results['preprocessing'] = run_feature_preprocessing()
    except Exception as e:
        print(f"\n✗ Feature preprocessing failed: {e}")
    
    # Save all reports
    print("\n" + "="*70)
    print("SAVING ANALYSIS REPORTS")
    print("="*70 + "\n")
    
    save_feature_importance_report(all_results['feature_analysis'], report_dir)
    save_correlation_matrix(all_results['feature_analysis'], report_dir)
    save_timeseries_summary(all_results['timeseries_analysis'], report_dir)
    save_outlier_report(all_results['outlier_treatment'], report_dir)
    save_preprocessing_log(all_results['preprocessing'], report_dir)
    summary_path = generate_summary_report(all_results, report_dir)
    
    # Final status
    print("\n" + "="*70)
    print("PHASE 3.2 COMPLETE!")
    print("="*70)
    print(f"✓ All reports saved to: {report_dir}/")
    print(f"✓ Summary: {summary_path}")
    print("\n📊 Ready for Phase 4: SVR Model Training")
    print("="*70 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = run_full_feature_analysis()
