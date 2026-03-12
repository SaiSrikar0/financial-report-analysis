"""
Phase 5: Explainable AI layer using SHAP for SVR growth-rate predictions.
Generates global and local feature contribution outputs.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is required for Phase 5 explainability. Install with: pip install shap"
    ) from exc

from models.svr_pipeline import (
    build_supervised_dataset,
    load_modeling_data,
    time_aware_split,
    tune_and_train_svr,
)


def _ensure_report_dir(report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)


def _safe_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def _build_explainer_and_values(
    model,
    X_train_scaled: pd.DataFrame,
    X_eval_scaled: pd.DataFrame,
    nsamples: int,
):
    # KernelExplainer works with non-tree models (SVR) and accepts a callable predict fn.
    background = _safe_sample(X_train_scaled, n=min(25, len(X_train_scaled)))
    explainer = shap.KernelExplainer(model.named_steps["svr"].predict, background)
    shap_values = explainer.shap_values(X_eval_scaled, nsamples=nsamples)
    expected_value = float(np.ravel(explainer.expected_value)[0])
    return explainer, np.asarray(shap_values), expected_value


def _save_global_outputs(
    shap_values_eval: np.ndarray,
    X_eval_scaled: pd.DataFrame,
    report_dir: str,
):
    mean_abs = np.abs(shap_values_eval).mean(axis=0)
    global_df = pd.DataFrame(
        {
            "feature": X_eval_scaled.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    global_csv_path = os.path.join(report_dir, "phase_5_shap_global_importance.csv")
    global_df.to_csv(global_csv_path, index=False)

    chart_path = os.path.join(report_dir, "phase_5_shap_global_importance.png")
    plt.figure(figsize=(10, 6))
    top_df = global_df.head(12).iloc[::-1]
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.title("Phase 5 SHAP Global Importance (Top 12 Features)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=140)
    plt.close()

    return global_df, global_csv_path, chart_path


def _save_local_outputs(
    model,
    explainer,
    expected_value: float,
    future_rows: pd.DataFrame,
    future_X_scaled: pd.DataFrame,
    report_dir: str,
    nsamples: int,
):
    future_preds = model.predict(future_X_scaled.values)
    shap_values_future = np.asarray(explainer.shap_values(future_X_scaled.values, nsamples=nsamples))

    local_rows = []
    for idx, (_, future_row) in enumerate(future_rows.reset_index(drop=True).iterrows()):
        ticker = future_row["ticker"]
        row_contrib = pd.Series(shap_values_future[idx], index=future_X_scaled.columns)
        top_contrib = row_contrib.reindex(row_contrib.abs().sort_values(ascending=False).index).head(8)

        for feature, contribution in top_contrib.items():
            local_rows.append(
                {
                    "ticker": ticker,
                    "predicted_growth_rate": float(future_preds[idx]),
                    "feature": feature,
                    "feature_value_scaled": float(future_X_scaled.iloc[idx][feature]),
                    "shap_value": float(contribution),
                    "direction": "increases_prediction" if contribution >= 0 else "decreases_prediction",
                }
            )

    local_df = pd.DataFrame(local_rows)
    local_csv_path = os.path.join(report_dir, "phase_5_shap_local_explanations.csv")
    local_df.to_csv(local_csv_path, index=False)

    prediction_df = pd.DataFrame(
        {
            "ticker": future_rows["ticker"].values,
            "current_net_income": future_rows["net_income"].values,
            "predicted_growth_rate": future_preds,
            "shap_expected_value": expected_value,
        }
    )
    pred_csv_path = os.path.join(report_dir, "phase_5_shap_future_predictions.csv")
    prediction_df.to_csv(pred_csv_path, index=False)

    return local_csv_path, pred_csv_path, prediction_df


def _save_summary(
    report_dir: str,
    data_source: str,
    cutoff_date: str,
    global_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_paths: dict,
):
    summary_path = os.path.join(report_dir, "phase_5_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 5: EXPLAINABLE AI (SHAP) SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {data_source}\n")
        f.write(f"Cutoff Date (time-aware split): {cutoff_date}\n\n")

        f.write("1. GLOBAL FEATURE IMPACT (Top 10 by mean |SHAP|)\n")
        f.write("-" * 70 + "\n")
        for _, row in global_df.head(10).iterrows():
            f.write(f"- {row['feature']}: {row['mean_abs_shap']:.6f}\n")

        f.write("\n2. LOCAL EXPLANATIONS (Future predictions)\n")
        f.write("-" * 70 + "\n")
        for _, row in prediction_df.iterrows():
            f.write(
                f"- {row['ticker']}: predicted growth {row['predicted_growth_rate']:.2f}%"
                f" (base value {row['shap_expected_value']:.4f})\n"
            )

        f.write("\nGenerated Files:\n")
        for name, path in output_paths.items():
            f.write(f"- {name}: {path}\n")

    return summary_path


def run_phase5_explainability(report_dir="analysis/reports", shap_nsamples=200):
    """Execute Phase 5 SHAP explainability workflow on the SVR model."""
    print("\n" + "=" * 70)
    print("PHASE 5: EXPLAINABLE AI LAYER (SHAP)")
    print("=" * 70)

    _ensure_report_dir(report_dir)

    print("\n[1/5] Loading modeling data...")
    df, data_source = load_modeling_data()
    data = build_supervised_dataset(df, target_col="net_income", horizon=1)
    X, y = data["X"], data["y"]
    dates = data["model_df"]["date"]
    print(f"✓ Dataset ready: {X.shape[0]} rows × {X.shape[1]} features")

    print("\n[2/5] Re-training best SVR with time-aware split...")
    X_train, _, y_train, _, cutoff = time_aware_split(X, y, dates, test_ratio=0.2)
    model, cv_summary = tune_and_train_svr(X_train, y_train)
    print(f"✓ Best params: {cv_summary['best_params']}")

    print("\n[3/5] Preparing SHAP inputs...")
    scaler = model.named_steps["scaler"]
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_eval_scaled = _safe_sample(X_train_scaled, n=min(30, len(X_train_scaled)))

    future_rows = data["future_rows"].reset_index(drop=True)
    future_X = data["future_X"]
    future_X_scaled = pd.DataFrame(
        scaler.transform(future_X),
        columns=future_X.columns,
        index=future_X.index,
    )
    print(f"✓ SHAP eval rows: {len(X_eval_scaled)} | future rows: {len(future_X_scaled)}")

    print("\n[4/5] Computing SHAP values (global + local)...")
    explainer, shap_values_eval, expected_value = _build_explainer_and_values(
        model=model,
        X_train_scaled=X_train_scaled,
        X_eval_scaled=X_eval_scaled,
        nsamples=shap_nsamples,
    )

    global_df, global_csv_path, global_chart_path = _save_global_outputs(
        shap_values_eval=shap_values_eval,
        X_eval_scaled=X_eval_scaled,
        report_dir=report_dir,
    )

    local_csv_path, pred_csv_path, prediction_df = _save_local_outputs(
        model=model.named_steps["svr"],
        explainer=explainer,
        expected_value=expected_value,
        future_rows=future_rows,
        future_X_scaled=future_X_scaled,
        report_dir=report_dir,
        nsamples=shap_nsamples,
    )

    output_paths = {
        "global_importance_csv": global_csv_path,
        "global_importance_chart": global_chart_path,
        "local_explanations_csv": local_csv_path,
        "future_predictions_csv": pred_csv_path,
    }

    print("\n[5/5] Writing summary report...")
    summary_path = _save_summary(
        report_dir=report_dir,
        data_source=data_source,
        cutoff_date=str(pd.to_datetime(cutoff).date()),
        global_df=global_df,
        prediction_df=prediction_df,
        output_paths=output_paths,
    )

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print(f"✓ Reports saved in: {report_dir}")
    print(f"✓ Summary report: {summary_path}")

    return {
        "summary_path": summary_path,
        "output_paths": output_paths,
        "global_importance": global_df,
        "future_predictions": prediction_df,
    }
