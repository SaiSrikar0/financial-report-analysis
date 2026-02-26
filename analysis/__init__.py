# FinCast Analysis Module
# Phase 3: Financial Analysis & Insights

from .data_connection import get_analysis_data
from .historical_performance import analyze_historical_performance
from .trend_analysis import analyze_trends, calculate_ratios
from .peer_comparison import compare_peers
from .insights import extract_key_insights

__all__ = [
    'get_analysis_data',
    'analyze_historical_performance',
    'analyze_trends',
    'calculate_ratios',
    'compare_peers',
    'extract_key_insights',
]
