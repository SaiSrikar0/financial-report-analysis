from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from analysis.recommendation_engine import (  # noqa: E402
    _normalize_recommendation,
    load_analysis_bundle_from_reports,
)
from models.svr_pipeline import predict_future_and_gaps  # noqa: E402


class DummyModel:
    def __init__(self, predictions: list[float]):
        self._predictions = np.array(predictions, dtype=float)

    def predict(self, future_x):
        return self._predictions[: len(future_x)]


class RecommendationEngineTests(unittest.TestCase):
    def test_low_confidence_mode_caps_score_and_rewrites_output(self):
        analysis_bundle = {
            "svr_predictions": {
                "predicted_growth_rate_period_pct": 3.0,
                "target_growth_rate_period_pct": 10.0,
                "gap_vs_target_period_pct": -7.0,
                "predicted_growth_rate_annualized_pct": 38.0,
                "target_growth_rate_annualized_pct": 10.0,
                "gap_vs_target_annualized_pct": 28.0,
                "period_type": "monthly",
                "periods_per_year": 12,
                "model_reliability": "Low",
                "model_reliability_reason": "r2_below_threshold",
                "beats_naive": False,
            },
            "model_metrics": {
                "r2": 0.11,
                "mae": 8.5,
                "rmse": 9.4,
                "residual_std": 4.2,
                "test_size": 6,
                "model_reliability": "Low",
                "model_reliability_reason": "r2_below_threshold",
                "beats_naive": False,
                "naive_mae": 7.8,
            },
        }

        result = {
            "executive_summary": "Base summary",
            "performance_score": 9,
            "investment_verdict": "BUY",
            "growth_outlook": {
                "forecast": "Initial forecast",
                "predicted_growth_rate": 3.0,
                "gap_vs_target": -7.0,
                "gap_status": "shortfall",
                "confidence": "High",
                "key_drivers": ["momentum"],
                "critical_concern": "weak fit",
            },
            "risk_assessment": {
                "overall_risk": "Low",
                "critical_warnings": [],
            },
            "opportunities": [{"title": "Original opportunity"}],
            "action_items": [{"priority": "Low", "action": "Original action"}],
        }

        normalized = _normalize_recommendation(result, analysis_bundle)

        self.assertTrue(normalized["growth_outlook"]["low_confidence_mode"])
        self.assertTrue(normalized["reliability_gate"]["triggered"])
        self.assertEqual(normalized["investment_verdict"], "HOLD")
        self.assertLessEqual(normalized["performance_score"], 4)
        self.assertEqual(normalized["growth_outlook"]["confidence"], "Low")
        self.assertIn("Forecast reliability is low", normalized["risk_assessment"]["critical_warnings"][0])
        self.assertEqual(normalized["opportunities"][0]["title"], "Model validation before directional action")
        self.assertEqual(normalized["action_items"][-1]["priority"], "High")

    def test_high_quality_mode_preserves_directional_output(self):
        analysis_bundle = {
            "svr_predictions": {
                "predicted_growth_rate_period_pct": 11.0,
                "target_growth_rate_period_pct": 10.0,
                "gap_vs_target_period_pct": 1.0,
                "predicted_growth_rate_annualized_pct": 16.0,
                "target_growth_rate_annualized_pct": 10.0,
                "gap_vs_target_annualized_pct": 6.0,
                "period_type": "monthly",
                "periods_per_year": 12,
                "model_reliability": "High",
                "model_reliability_reason": "strong_fit_and_beats_naive",
                "beats_naive": True,
            },
            "model_metrics": {
                "r2": 0.67,
                "mae": 2.1,
                "rmse": 2.9,
                "residual_std": 1.0,
                "test_size": 24,
                "model_reliability": "High",
                "model_reliability_reason": "strong_fit_and_beats_naive",
                "beats_naive": True,
                "naive_mae": 5.2,
            },
        }

        result = {
            "executive_summary": "Strong summary",
            "performance_score": 6,
            "investment_verdict": "BUY",
            "growth_outlook": {
                "forecast": "Initial forecast",
                "predicted_growth_rate": 11.0,
                "gap_vs_target": 1.0,
                "gap_status": "surplus",
                "confidence": "Medium",
                "key_drivers": ["growth"],
                "critical_concern": "none",
            },
            "risk_assessment": {
                "overall_risk": "Low",
                "critical_warnings": [],
            },
            "opportunities": [{"title": "Original opportunity"}],
            "action_items": [{"priority": "Low", "action": "Original action"}],
        }

        normalized = _normalize_recommendation(result, analysis_bundle)

        self.assertFalse(normalized["growth_outlook"]["low_confidence_mode"])
        self.assertFalse(normalized["reliability_gate"]["triggered"])
        self.assertEqual(normalized["investment_verdict"], "BUY")
        self.assertGreaterEqual(normalized["performance_score"], 4)
        self.assertEqual(normalized["growth_outlook"]["confidence"], "Medium")

    def test_load_analysis_bundle_reads_new_metrics_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir)
            pd.DataFrame(
                [
                    {
                        "ticker": "QNRX",
                        "predicted_growth_rate": 2.5,
                        "target_growth_rate": 0.8,
                        "gap_vs_target": 1.7,
                        "gap_status": "surplus",
                        "period_type": "monthly",
                        "periods_per_year": 12,
                        "predicted_growth_rate_period_pct": 2.5,
                        "target_growth_rate_period_pct": 0.8,
                        "gap_vs_target_period_pct": 1.7,
                        "predicted_growth_rate_annualized_pct": 36.1,
                        "target_growth_rate_annualized_pct": 10.0,
                        "gap_vs_target_annualized_pct": 26.1,
                        "model_reliability": "Medium",
                        "model_reliability_reason": "adequate_fit_and_beats_naive",
                        "beats_naive": True,
                        "benchmark_naive_mae": 7.44,
                    }
                ]
            ).to_csv(reports_dir / "svr_future_predictions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "mae": 3.54,
                        "rmse": 4.25,
                        "r2": 0.4963,
                        "residual_std": 4.15,
                        "test_size": 34,
                        "model_reliability": "Medium",
                        "model_reliability_reason": "adequate_fit_and_beats_naive",
                        "beats_naive": True,
                        "naive_mae": 7.44,
                    }
                ]
            ).to_csv(reports_dir / "svr_evaluation_metrics.csv", index=False)

            bundle = load_analysis_bundle_from_reports("QNRX", reports_dir=str(reports_dir))

        self.assertEqual(bundle["ticker"], "QNRX")
        self.assertEqual(bundle["svr_predictions"]["period_type"], "monthly")
        self.assertTrue(bundle["model_metrics"]["beats_naive"])
        self.assertEqual(bundle["model_metrics"]["test_size"], 34)


class SvrPipelineTests(unittest.TestCase):
    def test_predict_future_and_gaps_adds_canonical_fields(self):
        future_rows = pd.DataFrame(
            {
                "ticker": ["QNRX", "LYRX"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "net_income": [100.0, 200.0],
            }
        )
        future_x = pd.DataFrame({"feature_1": [1.0, 2.0]})

        output = predict_future_and_gaps(
            model=DummyModel([5.0, 12.0]),
            future_rows=future_rows,
            future_X=future_x,
            confidence_sigma=1.5,
            target_growth_rate=2.0,
            periods_per_year=12.0,
            trend_priors={"QNRX": 8.0},
            blend_weight=0.5,
        )

        self.assertEqual(output.loc[0, "period_type"], "monthly")
        self.assertAlmostEqual(output.loc[0, "predicted_growth_rate_period_pct"], 6.5)
        self.assertAlmostEqual(output.loc[0, "gap_vs_target_period_pct"], 4.5)
        self.assertIn("predicted_growth_rate_annualized_pct", output.columns)
        self.assertIn("gap_vs_target_annualized_pct", output.columns)
        self.assertGreater(output.loc[0, "predicted_growth_rate_annualized_pct"], output.loc[0, "predicted_growth_rate_period_pct"])

    def test_assess_model_reliability_flags_underperforming_models(self):
        from models.svr_pipeline import assess_model_reliability

        test_metrics = {
            "mae": 9.0,
            "r2": 0.32,
            "test_size": 10,
        }
        benchmark_rows = [
            {"model": "NaiveLast", "mae": 8.0, "rmse": 9.0, "r2": -1.0},
            {"model": "LinearRegression", "mae": 6.0, "rmse": 7.0, "r2": 0.3},
        ]

        reliability = assess_model_reliability(test_metrics, benchmark_rows)

        self.assertEqual(reliability["model_reliability"], "Low")
        self.assertEqual(reliability["model_reliability_reason"], "does_not_beat_naive")
        self.assertFalse(reliability["beats_naive"])


if __name__ == "__main__":
    unittest.main()