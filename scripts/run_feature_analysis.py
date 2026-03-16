"""Phase 3.2: Feature Analysis Orchestrator"""

import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.feature_analysis import run_feature_analysis
from analysis.timeseries_analysis import run_timeseries_analysis
from analysis.outlier_treatment import run_outlier_treatment
from analysis.feature_preprocessing import run_feature_preprocessing


def save_results(results, report_dir="analysis/reports"):
    """Save all feature analysis results to CSV files."""
    os.makedirs(report_dir, exist_ok=True)
    
    # Feature importance & correlation
    if results['feature_analysis']:
        if 'importance_results' in results['feature_analysis']:
            results['feature_analysis']['importance_results']['feature_importance'].to_csv(
                f"{report_dir}/feature_importance.csv", index=False
            )
        if 'correlation_results' in results['feature_analysis']:
            results['feature_analysis']['correlation_results']['correlation_matrix'].to_csv(
                f"{report_dir}/correlation_matrix.csv"
            )
    
    # Time-series trends
    if results['timeseries_analysis'] and 'trend_slopes' in results['timeseries_analysis']:
        trends_data = []
        for company, trends in results['timeseries_analysis']['trend_slopes'].items():
            for metric, info in trends.items():
                trends_data.append({'company': company, 'metric': metric, **info})
        if trends_data:
            pd.DataFrame(trends_data).to_csv(f"{report_dir}/timeseries_trends.csv", index=False)
    
    # Outlier recommendations
    if results['outlier_treatment'] and 'recommendations' in results['outlier_treatment']:
        if results['outlier_treatment']['recommendations']:
            pd.DataFrame(results['outlier_treatment']['recommendations']).to_csv(
                f"{report_dir}/outlier_recommendations.csv", index=False
            )
    
    # Preprocessing steps & ML-ready data
    if results['preprocessing'] and 'processed_data' in results['preprocessing']:
        pd.DataFrame(results['preprocessing']['preprocessing_steps']).to_csv(
            f"{report_dir}/preprocessing_steps.csv", index=False
        )
        results['preprocessing']['processed_data'].to_csv(
            f"{report_dir}/ml_ready_data.csv", index=False
        )
    
    # Summary report
    with open(f"{report_dir}/phase_3_2_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\nPHASE 3.2: FEATURE ANALYSIS SUMMARY\n{'='*70}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if results['feature_analysis']:
            f.write("✓ Feature importance rankings saved\n")
            f.write("✓ Correlation matrix saved\n")
        if results['timeseries_analysis']:
            f.write("✓ Time-series trend analysis saved\n")
        if results['outlier_treatment']:
            f.write("✓ Outlier recommendations saved\n")
        if results['preprocessing']:
            f.write("✓ ML-ready dataset saved\n")
        
        f.write(f"\n{'='*70}\nPhase 3.2 Complete! Ready for Phase 4\n{'='*70}\n")


def run_full_feature_analysis():
    """Execute all Phase 3.2 components."""
    print(f"\n{'='*70}\nPHASE 3.2: FEATURE ANALYSIS\n{'='*70}\n")
    
    results = {}
    
    steps = [
        ("feature_analysis", "Feature Analysis", run_feature_analysis),
        ("timeseries_analysis", "Time-Series Analysis", run_timeseries_analysis),
        ("outlier_treatment", "Outlier Treatment", run_outlier_treatment),
        ("preprocessing", "Feature Preprocessing", run_feature_preprocessing),
    ]
    
    for i, (key, name, func) in enumerate(steps, 1):
        print(f"\n[{i}/4] {name}...")
        try:
            results[key] = func()
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[key] = None
    
    print(f"\n{'='*70}\nSaving reports...\n{'='*70}\n")
    save_results(results)
    print(f"✓ All reports saved to: analysis/reports/\n")
    
    return results


if __name__ == "__main__":
    run_full_feature_analysis()
