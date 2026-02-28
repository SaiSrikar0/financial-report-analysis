#!/usr/bin/env python3
"""
FINCAST - Phase 3.1: Financial Analysis Module
Orchestrates all analysis components and generates comprehensive report
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data_connection import get_analysis_data, get_companies_list
from analysis.historical_performance import analyze_historical_performance
from analysis.trend_analysis import analyze_trends, calculate_ratios
from analysis.peer_comparison import compare_peers, get_peer_rankings
from analysis.insights import extract_key_insights, generate_insights_report


def run_full_analysis():
    """Run complete Phase 3.1 analysis"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "FINCAST - PHASE 3.1: FINANCIAL ANALYSIS")
    print("=" * 70)
    print(f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 1. Load data
        print("\n[1/6] Loading data from Supabase...")
        standard_df, category_df, merged_df = get_analysis_data()
        companies = get_companies_list()
        
        # 2. Historical Performance Analysis
        print("\n[2/6] Running historical performance analysis...")
        performance_report = analyze_historical_performance()
        
        # 3. Trend Analysis
        print("\n[3/6] Running trend analysis...")
        trends = analyze_trends()
        
        # 4. Ratio Analysis
        print("\n[4/6] Calculating financial ratios...")
        ratios = calculate_ratios()
        
        # 5. Peer Comparison
        print("\n[5/6] Comparing peer companies...")
        peer_comparison = compare_peers()
        peer_rankings = get_peer_rankings()
        
        # 6. Key Insights & Anomalies
        print("\n[6/6] Extracting insights and detecting anomalies...")
        insights = extract_key_insights()
        
        # 7. Generate Report
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 70)
        
        print(f"\n✓ Data loaded: {len(standard_df)} records from {len(companies)} companies")
        print(f"✓ Performance analysis: {len(performance_report)} companies analyzed")
        print(f"✓ Trends identified: {len(trends)} trend classifications")
        print(f"✓ Ratios calculated: {len(ratios)} companies")
        print(f"✓ Peer comparison: {len(peer_rankings)} companies ranked")
        print(f"✓ Insights extracted: {len(insights)} actionable insights")
        
        return {
            'standard_df': standard_df,
            'category_df': category_df,
            'merged_df': merged_df,
            'performance_report': performance_report,
            'trends': trends,
            'ratios': ratios,
            'peer_comparison': peer_comparison,
            'peer_rankings': peer_rankings,
            'insights': insights,
        }
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_analysis_results(results, output_dir='analysis/reports'):
    """Save analysis results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save insights report
    report = generate_insights_report()
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save peer rankings
    rankings_path = os.path.join(output_dir, 'peer_rankings.csv')
    results['peer_rankings'].to_csv(rankings_path, index=False)
    print(f"✓ Peer rankings saved to: {rankings_path}")
    
    # Save merged dataframe
    merged_path = os.path.join(output_dir, 'analysis_data.csv')
    results['merged_df'].to_csv(merged_path, index=False)
    print(f"✓ Analysis data saved to: {merged_path}")


if __name__ == "__main__":
    # Run analysis
    results = run_full_analysis()
    
    if results:
        # Save results
        save_analysis_results(results)
        
        print("\n" + "=" * 70)
        print("Phase 3.1 Complete! Ready for Phase 4: SVR Model Training")
        print("=" * 70)
    else:
        print("\n✗ Analysis failed. Check error messages above.")
        sys.exit(1)
