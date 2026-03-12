"""
Phase 4: SVR modeling pipeline.
Implements training, evaluation, and prediction gap analysis.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from analysis.data_connection import get_standard_table_data
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.data_connection import get_standard_table_data


def load_modeling_data():
    """Load data for Phase 4 modeling from Supabase or local staged fallback."""
    try:
        df = get_standard_table_data()
        source = "supabase"
    except Exception:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(base_dir, "etl", "data", "staged", "standard_table.csv")
        if not os.path.exists(fallback_path):
            raise
        df = pd.read_csv(fallback_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        source = "local_csv"

    required_cols = ["ticker", "date", "net_income"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for modeling: {col}")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df, source


def build_supervised_dataset(df, target_col="net_income", horizon=1):
    """Build supervised dataset for growth-rate prediction (Strategy 5)."""
    work_df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Growth-rate target: % YoY change
    work_df[f"{target_col}_next"] = work_df.groupby("ticker")[target_col].shift(-horizon)
    work_df["target"] = ((work_df[f"{target_col}_next"] - work_df[target_col]) / work_df[target_col]) * 100
    
    target_name = "target"
    feature_exclude = {"date", "ticker", target_name, f"{target_col}_next"}
    numerical_features = [
        col for col in work_df.select_dtypes(include=[np.number]).columns
        if col not in feature_exclude
    ]

    model_df_raw = work_df.dropna(subset=[target_name]).copy()
    metadata = model_df_raw[["date", "ticker"]].copy()
    model_df = pd.get_dummies(model_df_raw, columns=["ticker"], drop_first=True)

    ticker_dummy_cols = [col for col in model_df.columns if col.startswith("ticker_")]
    feature_cols = numerical_features + ticker_dummy_cols

    model_df = model_df.dropna(subset=feature_cols)

    X = model_df[feature_cols].copy()
    y = model_df[target_name].copy()

    last_rows = work_df.groupby("ticker", as_index=False).tail(1).copy()
    future_df = pd.get_dummies(last_rows, columns=["ticker"], drop_first=True)
    for col in ticker_dummy_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    future_X = future_df[feature_cols].copy()

    metadata = metadata.loc[model_df.index].copy()

    return {
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "model_df": model_df,
        "metadata": metadata,
        "future_rows": last_rows,
        "future_X": future_X,
        "target_name": target_name,
    }


def time_aware_split(X, y, dates, test_ratio=0.2):
    """Create chronological train-test split using unique dates."""
    unique_dates = sorted(pd.Series(dates).dropna().unique())
    if len(unique_dates) < 4:
        raise ValueError("Insufficient unique time points for time-aware split")

    split_index = max(1, int(len(unique_dates) * (1 - test_ratio)))
    split_index = min(split_index, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_index - 1]

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return X_train, X_test, y_train, y_test, cutoff_date


def tune_and_train_svr(X_train, y_train):
    """Tune SVR hyperparameters using time-series cross-validation."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    param_grid = {
        "svr__kernel": ["rbf", "linear"],
        "svr__C": [1, 10, 50, 100],
        "svr__epsilon": [0.01, 0.1, 0.5],
        "svr__gamma": ["scale", "auto"],
    }

    n_splits = min(5, max(2, len(X_train) // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    cv_metrics = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=tscv,
        scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
        n_jobs=-1,
    )

    cv_summary = {
        "cv_mae_mean": -cv_metrics["test_neg_mean_absolute_error"].mean(),
        "cv_mae_std": cv_metrics["test_neg_mean_absolute_error"].std(),
        "cv_rmse_mean": -cv_metrics["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_std": cv_metrics["test_neg_root_mean_squared_error"].std(),
        "cv_r2_mean": cv_metrics["test_r2"].mean(),
        "cv_r2_std": cv_metrics["test_r2"].std(),
        "best_params": grid.best_params_,
    }

    return best_model, cv_summary


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model on holdout set."""
    predictions = model.predict(X_test)
    residuals = y_test.values - predictions

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "test_size": int(len(y_test)),
    }

    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
        "residual": residuals,
    }, index=y_test.index)

    return metrics, pred_df


def predict_future_and_gaps(model, future_rows, future_X, confidence_sigma, target_growth_rate=10.0):
    """Predict next-period growth rate and compute target gaps."""
    preds = model.predict(future_X)
    output = future_rows[["ticker", "date", "net_income"]].copy()
    
    output["current_net_income"] = output["net_income"]
    output["predicted_growth_rate"] = preds
    output["target_growth_rate"] = target_growth_rate
    output["gap_vs_target"] = preds - target_growth_rate
    output["gap_status"] = np.where(output["gap_vs_target"] >= 0, "surplus", "shortfall")
    output["confidence_lower_95"] = preds - 1.96 * confidence_sigma
    output["confidence_upper_95"] = preds + 1.96 * confidence_sigma
    output["predicted_next_net_income"] = output["current_net_income"] * (1 + preds / 100)
    output["target_next_net_income"] = output["current_net_income"] * (1 + target_growth_rate / 100)
    
    return output


def save_phase4_outputs(report_dir, summary, pred_df, future_df):
    """Persist Phase 4 outputs to report files."""
    os.makedirs(report_dir, exist_ok=True)

    metrics_df = pd.DataFrame([summary["test_metrics"]])
    cv_df = pd.DataFrame([{
        "cv_mae_mean": summary["cv_summary"]["cv_mae_mean"],
        "cv_mae_std": summary["cv_summary"]["cv_mae_std"],
        "cv_rmse_mean": summary["cv_summary"]["cv_rmse_mean"],
        "cv_rmse_std": summary["cv_summary"]["cv_rmse_std"],
        "cv_r2_mean": summary["cv_summary"]["cv_r2_mean"],
        "cv_r2_std": summary["cv_summary"]["cv_r2_std"],
    }])

    best_params_df = pd.DataFrame([summary["cv_summary"]["best_params"]])

    metrics_path = os.path.join(report_dir, "svr_evaluation_metrics.csv")
    cv_path = os.path.join(report_dir, "svr_cross_validation.csv")
    params_path = os.path.join(report_dir, "svr_best_params.csv")
    pred_path = os.path.join(report_dir, "svr_test_predictions.csv")
    future_path = os.path.join(report_dir, "svr_future_predictions.csv")
    summary_path = os.path.join(report_dir, "phase_4_summary.txt")

    metrics_df.to_csv(metrics_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    best_params_df.to_csv(params_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    future_df.to_csv(future_path, index=False)

    chart1_path = os.path.join(report_dir, "svr_actual_vs_predicted.png")
    chart2_path = os.path.join(report_dir, "svr_residuals.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.8)
    min_val = min(pred_df["actual"].min(), pred_df["predicted"].min())
    max_val = max(pred_df["actual"].max(), pred_df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title("SVR Test Set: Actual vs Predicted")
    plt.xlabel("Actual Net Income")
    plt.ylabel("Predicted Net Income")
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["predicted"], pred_df["residual"], alpha=0.8)
    plt.axhline(y=0, linestyle="--")
    plt.title("SVR Residual Analysis")
    plt.xlabel("Predicted Net Income")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=140)
    plt.close()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 4: SVR GROWTH-RATE PREDICTION (STRATEGY 5)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {summary['data_source']}\n")
        f.write(f"Train Size: {summary['train_size']} | Test Size: {summary['test_size']}\n")
        f.write(f"Split Cutoff Date: {summary['cutoff_date']}\n\n")

        f.write("1. MODEL EVALUATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"MAE:  {summary['test_metrics']['mae']:.4f}\n")
        f.write(f"RMSE: {summary['test_metrics']['rmse']:.4f}\n")
        f.write(f"R²:   {summary['test_metrics']['r2']:.4f}\n")
        f.write(f"Residual Mean: {summary['test_metrics']['residual_mean']:.4f}\n")
        f.write(f"Residual Std:  {summary['test_metrics']['residual_std']:.4f}\n\n")

        f.write("2. CROSS-VALIDATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"CV MAE (mean):  {summary['cv_summary']['cv_mae_mean']:.4f}\n")
        f.write(f"CV RMSE (mean): {summary['cv_summary']['cv_rmse_mean']:.4f}\n")
        f.write(f"CV R² (mean):   {summary['cv_summary']['cv_r2_mean']:.4f}\n")
        f.write("Best Params:\n")
        for key, value in summary["cv_summary"]["best_params"].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("3. FUTURE PREDICTION & GAP ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for _, row in future_df.iterrows():
            f.write(
                f"{row['ticker']}:\n"
                f"  Current Net Income: ${row['current_net_income']/1e9:.2f}B\n"
                f"  Predicted Growth Rate: {row['predicted_growth_rate']:.2f}%\n"
                f"  Target Growth Rate: {row['target_growth_rate']:.2f}%\n"
                f"  Gap: {row['gap_vs_target']:+.2f}% ({row['gap_status']})\n"
                f"  95% CI: [{row['confidence_lower_95']:.2f}%, {row['confidence_upper_95']:.2f}%]\n\n"
            )

        f.write("\nGenerated Files:\n")
        f.write("- svr_evaluation_metrics.csv\n")
        f.write("- svr_cross_validation.csv\n")
        f.write("- svr_best_params.csv\n")
        f.write("- svr_test_predictions.csv\n")
        f.write("- svr_future_predictions.csv\n")
        f.write("- svr_actual_vs_predicted.png\n")
        f.write("- svr_residuals.png\n")
        f.write("- phase_4_summary.txt\n")

    return {
        "metrics_path": metrics_path,
        "cv_path": cv_path,
        "params_path": params_path,
        "pred_path": pred_path,
        "future_path": future_path,
        "summary_path": summary_path,
        "chart1_path": chart1_path,
        "chart2_path": chart2_path,
    }


def run_phase4_svr(target_growth_rate=10.0, report_dir="analysis/reports"):
    """Execute full Phase 4 workflow (Strategy 5: Growth-Rate Prediction)."""
    print("\n" + "=" * 70)
    print("PHASE 4: SVR MODEL - GROWTH-RATE PREDICTION")
    print("=" * 70)

    print("\n[1/6] Loading modeling data...")
    df, data_source = load_modeling_data()
    print(f"✓ Loaded {len(df)} records from {data_source}")

    print("\n[2/6] Building supervised dataset (% growth prediction)...")
    data = build_supervised_dataset(df, target_col="net_income", horizon=1)
    X, y = data["X"], data["y"]
    dates = data["model_df"]["date"]
    print(f"✓ Supervised dataset: {X.shape[0]} rows × {X.shape[1]} features")

    print("\n[3/6] Time-aware train-test split...")
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, dates, test_ratio=0.2)
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)} | Cutoff: {pd.to_datetime(cutoff).date()}")

    print("\n[4/6] Hyperparameter tuning + cross-validation...")
    model, cv_summary = tune_and_train_svr(X_train, y_train)
    print(f"✓ Best params: {cv_summary['best_params']}")

    print("\n[5/6] Holdout evaluation and residual analysis...")
    test_metrics, pred_df = evaluate_model(model, X_test, y_test)
    print(f"✓ MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")

    meta_test = data["metadata"].loc[X_test.index]
    pred_df["date"] = pd.to_datetime(meta_test["date"]).dt.strftime("%Y-%m-%d").values
    pred_df["ticker"] = meta_test["ticker"].values
    pred_df = pred_df.reset_index(drop=True)

    print("\n[6/6] Future prediction and target gap analysis...")
    future_df = predict_future_and_gaps(
        model=model,
        future_rows=data["future_rows"],
        future_X=data["future_X"],
        confidence_sigma=test_metrics["residual_std"],
        target_growth_rate=target_growth_rate,
    )
    print(f"✓ Future predictions generated for {len(future_df)} companies")

    summary = {
        "data_source": data_source,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cutoff_date": str(pd.to_datetime(cutoff).date()),
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }

    output_paths = save_phase4_outputs(report_dir, summary, pred_df, future_df)
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"✓ Reports saved in: {report_dir}")
    print(f"✓ Summary report: {output_paths['summary_path']}")
    print("\nReady for Phase 5: Explainable AI Layer")

    return {
        "summary": summary,
        "predictions": pred_df,
        "future_predictions": future_df,
        "outputs": output_paths,
    }
