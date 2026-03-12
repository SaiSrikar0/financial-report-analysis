#!/usr/bin/env python3
"""
FinCast - Main Orchestrator
Run complete analysis pipeline or specific phases
"""

import argparse
import sys


def run_phase3_1():
    """Execute Phase 3.1: Financial Analysis"""
    from analysis.data_connection import get_analysis_data
    from analysis.historical_performance import analyze_historical_performance
    from analysis.trend_analysis import analyze_trends, calculate_ratios
    from analysis.peer_comparison import compare_peers, get_peer_rankings
    from analysis.insights import generate_insights_report
    
    print("\n" + "="*70)
    print("PHASE 3.1: FINANCIAL ANALYSIS")
    print("="*70 + "\n")
    
    steps = [
        ("Loading data", get_analysis_data),
        ("Historical performance", analyze_historical_performance),
        ("Trend analysis", analyze_trends),
        ("Ratio analysis", calculate_ratios),
        ("Peer comparison", compare_peers),
        ("Insights extraction", generate_insights_report),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}...")
        try:
            func()
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "="*70)
    print("PHASE 3.1 COMPLETE")
    print("="*70)


def run_phase3_2():
    """Execute Phase 3.2: Feature Analysis"""
    from scripts.run_feature_analysis import run_full_feature_analysis
    run_full_feature_analysis()


def run_phase4(target_growth_rate=10.0):
    """Execute Phase 4: SVR Model Training"""
    from models.svr_pipeline import run_phase4_svr
    run_phase4_svr(target_growth_rate=target_growth_rate)


def run_phase5(shap_nsamples=200):
    """Execute Phase 5: SHAP Explainability"""
    from models.explainability import run_phase5_explainability
    run_phase5_explainability(shap_nsamples=shap_nsamples)


def run_all_phases(target_growth_rate=10.0, shap_nsamples=200):
    """Execute complete pipeline: Phases 3.1, 3.2, 4, and 5"""
    print("\n" + "="*70)
    print("FINCAST - COMPLETE PIPELINE EXECUTION")
    print("="*70)
    
    run_phase3_1()
    run_phase3_2()
    run_phase4(target_growth_rate)
    run_phase5(shap_nsamples)
    
    print("\n" + "="*70)
    print("ALL PHASES COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="FinCast Analysis Pipeline")
    parser.add_argument(
        "phase",
        choices=["3.1", "3.2", "4", "5", "all"],
        nargs="?",
        default="all",
        help="Phase to run: 3.1 (analysis), 3.2 (features), 4 (ML), 5 (XAI), or all"
    )
    parser.add_argument(
        "--target-growth",
        type=float,
        default=10.0,
        help="Target growth rate for Phase 4 (default: 10.0%%)"
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=200,
        help="Number of SHAP sampling evaluations for Phase 5 (default: 200)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.phase == "3.1":
            run_phase3_1()
        elif args.phase == "3.2":
            run_phase3_2()
        elif args.phase == "4":
            run_phase4(args.target_growth)
        elif args.phase == "5":
            run_phase5(args.shap_nsamples)
        else:
            run_all_phases(args.target_growth, args.shap_nsamples)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
