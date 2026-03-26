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

    print("\n" + "=" * 70)
    print("PHASE 3.1: FINANCIAL ANALYSIS")
    print("=" * 70 + "\n")

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

    print("\n" + "=" * 70)
    print("PHASE 3.1 COMPLETE")
    print("=" * 70)


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


def run_phase6():
    """Execute Phase 6: LLM Recommendation Engine"""
    from analysis.recommendation_engine import (
        load_analysis_bundle_from_reports,
        generate_recommendations,
    )
    import os, json

    print("\n" + "=" * 70)
    print("PHASE 6: LLM RECOMMENDATION ENGINE")
    print("=" * 70)

    # Determine tickers from SVR predictions
    svr_path = "analysis/reports/svr_future_predictions.csv"
    if os.path.exists(svr_path):
        import pandas as pd
        tickers = pd.read_csv(svr_path)["ticker"].unique().tolist()
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        print(f"SVR predictions not found — using default tickers: {tickers}")

    print(f"Generating recommendations for: {tickers}\n")

    for ticker in tickers:
        print(f"[Phase 6] Processing {ticker}...")
        try:
            bundle = load_analysis_bundle_from_reports(ticker)
            if not bundle.get("svr_predictions"):
                print(
                    f"  ⚠️  No SVR predictions for {ticker}. "
                    "Run Phase 4+5 first."
                )
                continue
            recs = generate_recommendations(bundle)
            score = recs.get("performance_score", "N/A")
            risk = recs.get("risk_assessment", {}).get("overall_risk", "N/A")
            growth = recs.get("growth_outlook", {}).get("predicted_growth_rate", "N/A")
            print(f"  ✓ Score={score}/10 | Risk={risk} | Growth={growth}%")
        except Exception as e:
            print(f"  ✗ {ticker} failed: {e}")

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)


def run_all_phases(target_growth_rate=10.0, shap_nsamples=200):
    """Execute complete pipeline: Phases 3.1, 3.2, 4, 5, 6"""
    print("\n" + "=" * 70)
    print("FINCAST - COMPLETE PIPELINE EXECUTION")
    print("=" * 70)

    run_phase3_1()
    run_phase3_2()
    run_phase4(target_growth_rate)
    run_phase5(shap_nsamples)
    run_phase6()

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="FinCast Analysis Pipeline")
    parser.add_argument(
        "phase",
        choices=["3.1", "3.2", "4", "5", "6", "all"],
        nargs="?",
        default="all",
        help="Phase to run: 3.1, 3.2, 4, 5, 6, or all",
    )
    parser.add_argument(
        "--target-growth",
        type=float,
        default=10.0,
        help="Target growth rate for Phase 4 (default: 10.0%%)",
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=200,
        help="Number of SHAP sampling evaluations for Phase 5 (default: 200)",
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
        elif args.phase == "6":
            run_phase6()
        else:
            run_all_phases(args.target_growth, args.shap_nsamples)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()