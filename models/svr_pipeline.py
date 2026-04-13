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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
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
        fallback_path = os.path.join(base_dir, "data", "staged", "standard_table.csv")
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

    # Replace infinities before filtering; growth calculations can produce inf
    # when previous period values are zero.
    work_df = work_df.replace([np.inf, -np.inf], np.nan)

    model_df_raw = work_df.dropna(subset=[target_name]).copy()
    metadata = model_df_raw[["date", "ticker"]].copy()
    model_df = pd.get_dummies(model_df_raw, columns=["ticker"], drop_first=True)

    ticker_dummy_cols = [col for col in model_df.columns if col.startswith("ticker_")]
    feature_cols = numerical_features + ticker_dummy_cols

    model_df = model_df.dropna(subset=feature_cols)

    # Keep only finite training rows so downstream scikit-learn calls do not fail
    # on "Input contains NaN, infinity or a value too large".
    finite_mask = np.isfinite(model_df[feature_cols]).all(axis=1) & np.isfinite(model_df[target_name])
    model_df = model_df.loc[finite_mask].copy()

    X = model_df[feature_cols].copy()
    y = model_df[target_name].copy()

    if len(X) == 0:
        raise ValueError(
            "No finite supervised rows available after cleaning. "
            "Check for zero-denominator growth values or non-numeric financial fields."
        )

    last_rows = work_df.groupby("ticker", as_index=False).tail(1).copy()
    future_df = pd.get_dummies(last_rows, columns=["ticker"], drop_first=True)
    for col in ticker_dummy_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    future_X = future_df[feature_cols].copy()
    future_X = future_X.replace([np.inf, -np.inf], np.nan).fillna(0)

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


def _build_metrics(y_true, predictions):
    """Build consistent regression metrics across SVR and benchmark models."""
    residuals = y_true.values - predictions
    r2_value = r2_score(y_true, predictions) if len(y_true) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_value) if pd.notna(r2_value) else np.nan,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "test_size": int(len(y_true)),
    }


def evaluate_benchmark_models(X_train, y_train, X_test, y_test):
    """Evaluate simple baselines to validate SVR adds value over naive models."""
    rows = []

    # Time-series naive baseline: predict last observed growth each step.
    history = list(y_train.values)
    naive_last_preds = []
    for actual in y_test.values:
        pred = history[-1] if history else float(np.mean(y_train.values))
        naive_last_preds.append(pred)
        history.append(actual)
    naive_last_preds = np.array(naive_last_preds, dtype=float)
    naive_last_metrics = _build_metrics(y_test, naive_last_preds)
    rows.append({"model": "NaiveLast", **naive_last_metrics})

    benchmark_models = {
        "NaiveMean": DummyRegressor(strategy="mean"),
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in benchmark_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _build_metrics(y_test, preds)
        rows.append({"model": name, **metrics})

    return rows


def train_candidate_models(X_train, y_train):
    """Train candidate regressors used for champion selection."""
    candidates = {}

    svr_model, cv_summary = _train_svr_adaptive(X_train, y_train)
    candidates["SVR"] = {
        "model": svr_model,
        "cv_summary": cv_summary,
    }

    lin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    lin_model.fit(X_train, y_train)
    candidates["LinearRegression"] = {
        "model": lin_model,
        "cv_summary": None,
    }

    rf_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    candidates["RandomForest"] = {
        "model": rf_model,
        "cv_summary": None,
    }

    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(X_train, y_train)
    candidates["NaiveMean"] = {
        "model": mean_model,
        "cv_summary": None,
    }

    return candidates


def select_champion_model(candidates, X_test, y_test):
    """Pick the best holdout model, prioritizing higher R2 then lower MAE."""
    rows = []
    for name, payload in candidates.items():
        model = payload["model"]
        metrics, _ = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, "metrics": metrics, "payload": payload})

    def _rank_key(row):
        r2_value = row["metrics"].get("r2", np.nan)
        r2_rank = float(r2_value) if np.isfinite(r2_value) else -np.inf
        mae_rank = -float(row["metrics"].get("mae", np.inf))
        return (r2_rank, mae_rank)

    ranked = sorted(rows, key=_rank_key, reverse=True)
    winner = ranked[0]
    return {
        "name": winner["model"],
        "model": winner["payload"]["model"],
        "metrics": winner["metrics"],
        "cv_summary": winner["payload"]["cv_summary"],
        "ranking": [
            {
                "model": row["model"],
                "r2": float(row["metrics"].get("r2", np.nan)),
                "mae": float(row["metrics"].get("mae", np.nan)),
                "rmse": float(row["metrics"].get("rmse", np.nan)),
            }
            for row in ranked
        ],
    }


def assess_model_reliability(test_metrics, benchmark_rows):
    """Assess whether model quality is good enough for high-confidence narratives."""
    r2_value = float(test_metrics.get("r2", np.nan))
    test_size = int(test_metrics.get("test_size", 0) or 0)
    svr_mae = float(test_metrics.get("mae", np.inf))

    benchmark_df = pd.DataFrame(benchmark_rows)
    naive_row = benchmark_df[benchmark_df["model"] == "NaiveLast"]
    naive_mae = float(naive_row["mae"].iloc[0]) if not naive_row.empty else np.inf
    beats_naive = bool(np.isfinite(naive_mae) and svr_mae <= naive_mae)

    best_benchmark = (
        benchmark_df.sort_values("mae", ascending=True).iloc[0]["model"]
        if not benchmark_df.empty else "unknown"
    )

    if test_size < 8:
        status = "Low"
        reason = "insufficient_holdout_samples"
    elif pd.isna(r2_value):
        status = "Low"
        reason = "invalid_r2"
    elif r2_value < 0.2:
        status = "Low"
        reason = "r2_below_threshold"
    elif not beats_naive:
        status = "Low"
        reason = "does_not_beat_naive"
    elif r2_value >= 0.5:
        status = "High"
        reason = "strong_fit_and_beats_naive"
    else:
        status = "Medium"
        reason = "adequate_fit_and_beats_naive"

    return {
        "model_reliability": status,
        "model_reliability_reason": reason,
        "beats_naive": beats_naive,
        "naive_mae": float(naive_mae) if np.isfinite(naive_mae) else np.nan,
        "best_benchmark_model": str(best_benchmark),
    }


def predict_future_and_gaps(
    model,
    future_rows,
    future_X,
    confidence_sigma,
    target_growth_rate=10.0,
    periods_per_year=1.0,
    trend_priors=None,
    blend_weight=1.0,
):
    """Predict next-period growth rate and compute target gaps."""
    preds = np.asarray(model.predict(future_X), dtype=float)
    output = future_rows[["ticker", "date", "net_income"]].copy()

    if trend_priors:
        priors = (
            output["ticker"]
            .astype(str)
            .str.upper()
            .map(trend_priors)
            .astype(float)
            .values
        )
        valid_prior = np.isfinite(priors)
        w = float(max(0.0, min(1.0, blend_weight)))
        preds = np.where(valid_prior, (w * preds) + ((1.0 - w) * priors), preds)
    
    output["current_net_income"] = output["net_income"]
    output["predicted_growth_rate"] = preds
    output["target_growth_rate"] = target_growth_rate
    output["gap_vs_target"] = preds - target_growth_rate
    output["gap_status"] = np.where(output["gap_vs_target"] >= 0, "surplus", "shortfall")
    output["confidence_lower_95"] = preds - 1.96 * confidence_sigma
    output["confidence_upper_95"] = preds + 1.96 * confidence_sigma
    output["predicted_next_net_income"] = output["current_net_income"] * (1 + preds / 100)
    output["target_next_net_income"] = output["current_net_income"] * (1 + target_growth_rate / 100)

    period_label = _period_type_from_frequency(periods_per_year)
    output["period_type"] = period_label
    output["periods_per_year"] = float(periods_per_year)

    # Explicit semantics fields (new canonical names).
    output["predicted_growth_rate_period_pct"] = output["predicted_growth_rate"]
    output["target_growth_rate_period_pct"] = output["target_growth_rate"]
    output["gap_vs_target_period_pct"] = output["gap_vs_target"]

    output["predicted_growth_rate_annualized_pct"] = output["predicted_growth_rate"].apply(
        lambda x: _period_to_annual_growth(float(x), float(periods_per_year))
    )
    output["target_growth_rate_annualized_pct"] = output["target_growth_rate"].apply(
        lambda x: _period_to_annual_growth(float(x), float(periods_per_year))
    )
    output["gap_vs_target_annualized_pct"] = (
        output["predicted_growth_rate_annualized_pct"]
        - output["target_growth_rate_annualized_pct"]
    )
    
    return output


def _infer_periods_per_year(dates: pd.Series) -> float:
    """Infer data frequency as periods/year from median date spacing."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().sort_values()
    if len(d) < 2:
        return 1.0

    deltas = d.diff().dropna().dt.days
    if deltas.empty:
        return 1.0

    median_days = float(deltas.median())
    if median_days <= 2:
        return 365.0
    if median_days <= 8:
        return 52.0
    if median_days <= 16:
        return 24.0
    if median_days <= 45:
        return 12.0
    if median_days <= 120:
        return 4.0
    if median_days <= 240:
        return 2.0
    return 1.0


def _annual_to_period_target(annual_target_pct: float, periods_per_year: float) -> float:
    """Convert annual growth target percent to per-period percent."""
    if periods_per_year <= 0:
        return annual_target_pct
    annual_decimal = annual_target_pct / 100.0
    per_period_decimal = (1.0 + annual_decimal) ** (1.0 / periods_per_year) - 1.0
    return per_period_decimal * 100.0


def _period_to_annual_growth(period_growth_pct: float, periods_per_year: float) -> float:
    """Convert period growth percent to annualized growth percent via compounding."""
    if periods_per_year <= 0:
        return period_growth_pct
    period_decimal = period_growth_pct / 100.0
    if period_decimal <= -0.9999:
        return -100.0
    annual_decimal = (1.0 + period_decimal) ** periods_per_year - 1.0
    return annual_decimal * 100.0


def _period_type_from_frequency(periods_per_year: float) -> str:
    """Human-friendly period label inferred from periods/year."""
    rounded = int(round(periods_per_year))
    if rounded >= 300:
        return "daily"
    if rounded >= 48:
        return "weekly"
    if rounded >= 10:
        return "monthly"
    if rounded >= 3:
        return "quarterly"
    return "yearly"


def _select_prediction_horizon(periods_per_year: float) -> int:
    """Choose a forecast horizon that matches the source frequency."""
    if periods_per_year >= 300:
        return 30  # daily data -> approximately monthly horizon
    if periods_per_year >= 48:
        return 4   # weekly data -> approximately monthly horizon
    if periods_per_year >= 10:
        return 3   # monthly data -> approximately quarterly horizon
    return 1


def _effective_periods_per_year(base_periods_per_year: float, horizon: int) -> float:
    """Convert raw frequency into forecast-period frequency."""
    h = max(1, int(horizon))
    return float(max(1.0, base_periods_per_year / h))


def _build_recent_growth_priors(metadata: pd.DataFrame, y: pd.Series, lookback: int = 6):
    """Estimate per-ticker prior growth from recent realized target growth."""
    if metadata is None or len(metadata) == 0 or y is None or len(y) == 0:
        return {}

    prior_df = metadata.copy()
    prior_df = prior_df.assign(target_growth=y.values)
    prior_df["ticker"] = prior_df["ticker"].astype(str).str.upper()
    prior_df["date"] = pd.to_datetime(prior_df["date"], errors="coerce")
    prior_df = prior_df.dropna(subset=["date", "target_growth"])
    if prior_df.empty:
        return {}

    priors = {}
    for ticker, grp in prior_df.sort_values("date").groupby("ticker"):
        tail = grp["target_growth"].tail(max(2, int(lookback)))
        if len(tail) > 0:
            priors[ticker] = float(np.median(tail.values))
    return priors


def _build_level_trend_priors(df: pd.DataFrame, horizon: int):
    """Estimate per-ticker horizon growth from long-run net income trend."""
    if df is None or len(df) == 0:
        return {}

    work = df.copy()
    if "ticker" not in work.columns or "date" not in work.columns or "net_income" not in work.columns:
        return {}

    work["ticker"] = work["ticker"].astype(str).str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["net_income"] = pd.to_numeric(work["net_income"], errors="coerce")
    work = work.dropna(subset=["ticker", "date", "net_income"]).sort_values(["ticker", "date"])

    priors = {}
    h = max(1, int(horizon))

    for ticker, grp in work.groupby("ticker"):
        series = grp["net_income"].astype(float).values
        if len(series) < 3:
            continue
        first = float(series[0])
        last = float(series[-1])
        steps = len(series) - 1
        if first <= 0 or last <= 0 or steps <= 0:
            continue
        per_step = (last / first) ** (1.0 / steps) - 1.0
        horizon_growth = ((1.0 + per_step) ** h - 1.0) * 100.0
        if np.isfinite(horizon_growth):
            priors[ticker] = float(horizon_growth)

    return priors


def _merge_growth_priors(recent_priors, level_priors, recent_weight: float = 0.4):
    """Combine short-run and long-run growth priors into one per ticker prior."""
    merged = {}
    w_recent = float(max(0.0, min(1.0, recent_weight)))

    all_tickers = set(recent_priors.keys()) | set(level_priors.keys())
    for ticker in all_tickers:
        recent = recent_priors.get(ticker, np.nan)
        level = level_priors.get(ticker, np.nan)
        has_recent = np.isfinite(recent)
        has_level = np.isfinite(level)

        if has_recent and has_level:
            merged[ticker] = float((w_recent * recent) + ((1.0 - w_recent) * level))
        elif has_recent:
            merged[ticker] = float(recent)
        elif has_level:
            merged[ticker] = float(level)

    return merged


def save_phase4_outputs(report_dir, summary, pred_df, future_df, benchmark_df=None):
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
    benchmark_path = os.path.join(report_dir, "svr_benchmark_comparison.csv")
    summary_path = os.path.join(report_dir, "phase_4_summary.txt")

    metrics_df.to_csv(metrics_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    best_params_df.to_csv(params_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    future_df.to_csv(future_path, index=False)
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df.to_csv(benchmark_path, index=False)

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

        if summary.get("benchmark_rows"):
            f.write("3. BENCHMARK COMPARISON\n")
            f.write("-" * 70 + "\n")
            f.write("Model\t\tMAE\tRMSE\tR²\n")
            for row in summary["benchmark_rows"]:
                f.write(
                    f"{row['model']:<16}{row['mae']:.4f}\t{row['rmse']:.4f}\t{row['r2']:.4f}\n"
                )
            rel = summary.get("reliability", {})
            f.write("\n")
            f.write(f"Reliability: {rel.get('model_reliability', 'Low')}\n")
            f.write(f"Reason: {rel.get('model_reliability_reason', 'unknown')}\n")
            f.write(f"Beats Naive: {rel.get('beats_naive', False)}\n")
            f.write(f"SVR MAE vs Naive MAE: {summary['test_metrics']['mae']:.4f} vs {rel.get('naive_mae', np.nan):.4f}\n\n")

        f.write("4. FUTURE PREDICTION & GAP ANALYSIS\n")
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
        f.write("- svr_benchmark_comparison.csv\n")
        f.write("- svr_actual_vs_predicted.png\n")
        f.write("- svr_residuals.png\n")
        f.write("- phase_4_summary.txt\n")

    return {
        "metrics_path": metrics_path,
        "cv_path": cv_path,
        "params_path": params_path,
        "pred_path": pred_path,
        "future_path": future_path,
        "benchmark_path": benchmark_path,
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
    base_periods = _infer_periods_per_year(df["date"])
    horizon = _select_prediction_horizon(base_periods)
    effective_periods = _effective_periods_per_year(base_periods, horizon)

    data = build_supervised_dataset(df, target_col="net_income", horizon=horizon)
    X, y = data["X"], data["y"]
    dates = data["model_df"]["date"]
    print(
        f"✓ Supervised dataset: {X.shape[0]} rows × {X.shape[1]} features "
        f"(horizon={horizon} period(s))"
    )

    print("\n[3/6] Time-aware train-test split...")
    X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, dates, test_ratio=0.2)
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)} | Cutoff: {pd.to_datetime(cutoff).date()}")

    print("\n[4/6] Training candidate models (SVR + benchmarks)...")
    candidates = train_candidate_models(X_train, y_train)
    svr_cv_summary = candidates["SVR"]["cv_summary"]
    print(f"✓ SVR params: {svr_cv_summary['best_params']}")

    print("\n[5/6] Holdout evaluation and residual analysis...")
    champion = select_champion_model(candidates, X_test, y_test)
    model = champion["model"]
    cv_summary = champion["cv_summary"] if champion["cv_summary"] is not None else svr_cv_summary
    test_metrics, pred_df = evaluate_model(model, X_test, y_test)
    test_metrics["selected_model"] = champion["name"]

    benchmark_rows = evaluate_benchmark_models(X_train, y_train, X_test, y_test)
    reliability = assess_model_reliability(test_metrics, benchmark_rows)
    test_metrics.update(reliability)
    print(
        f"✓ Champion={champion['name']} | MAE={test_metrics['mae']:.4f}, "
        f"RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}"
    )

    meta_test = data["metadata"].loc[X_test.index]
    pred_df["date"] = pd.to_datetime(meta_test["date"]).dt.strftime("%Y-%m-%d").values
    pred_df["ticker"] = meta_test["ticker"].values
    pred_df = pred_df.reset_index(drop=True)

    print("\n[6/6] Future prediction and target gap analysis...")
    period_target_growth = _annual_to_period_target(target_growth_rate, effective_periods)
    recent_priors = _build_recent_growth_priors(data["metadata"].loc[X_train.index], y_train)
    level_priors = _build_level_trend_priors(df, horizon)
    trend_priors = _merge_growth_priors(recent_priors, level_priors, recent_weight=0.4)
    blend_weight = float(max(0.35, min(0.85, 0.35 + 0.45 * float(test_metrics["r2"]))))

    future_df = predict_future_and_gaps(
        model=model,
        future_rows=data["future_rows"],
        future_X=data["future_X"],
        confidence_sigma=test_metrics["residual_std"],
        target_growth_rate=period_target_growth,
        periods_per_year=effective_periods,
        trend_priors=trend_priors,
        blend_weight=blend_weight,
    )
    future_df["forecast_horizon_periods"] = int(horizon)
    future_df["model_reliability"] = test_metrics["model_reliability"]
    future_df["model_reliability_reason"] = test_metrics["model_reliability_reason"]
    future_df["selected_model"] = test_metrics["selected_model"]
    future_df["model_r2"] = test_metrics["r2"]
    future_df["benchmark_naive_mae"] = test_metrics["naive_mae"]
    future_df["beats_naive"] = test_metrics["beats_naive"]
    print(f"✓ Future predictions generated for {len(future_df)} companies")

    summary = {
        "data_source": data_source,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cutoff_date": str(pd.to_datetime(cutoff).date()),
        "forecast_horizon_periods": int(horizon),
        "effective_periods_per_year": float(effective_periods),
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "benchmark_rows": benchmark_rows,
        "model_ranking": champion["ranking"],
        "trend_blend_weight": blend_weight,
        "reliability": reliability,
    }

    benchmark_df = pd.DataFrame(benchmark_rows)
    output_paths = save_phase4_outputs(report_dir, summary, pred_df, future_df, benchmark_df=benchmark_df)
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


def _chronological_holdout_split(X, y, dates):
    """Fallback split for very small datasets: train on all but last row."""
    order = pd.Series(dates).sort_values().index
    X_sorted = X.loc[order]
    y_sorted = y.loc[order]
    d_sorted = pd.to_datetime(pd.Series(dates).loc[order])

    if len(X_sorted) < 2:
        raise ValueError("Need at least 2 supervised rows for holdout split")

    split_idx = len(X_sorted) - 1
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    cutoff_date = d_sorted.iloc[split_idx - 1]

    return X_train, X_test, y_train, y_test, cutoff_date


def _train_svr_adaptive(X_train, y_train):
    """Use grid search when possible, otherwise fit a compact baseline SVR."""
    if len(X_train) >= 8 and y_train.nunique() > 1:
        return tune_and_train_svr(X_train, y_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
    ])
    pipeline.fit(X_train, y_train)

    cv_summary = {
        "cv_mae_mean": np.nan,
        "cv_mae_std": np.nan,
        "cv_rmse_mean": np.nan,
        "cv_rmse_std": np.nan,
        "cv_r2_mean": np.nan,
        "cv_r2_std": np.nan,
        "best_params": {
            "svr__kernel": "rbf",
            "svr__C": 10,
            "svr__epsilon": 0.1,
            "svr__gamma": "scale",
            "mode": "baseline_small_dataset",
        },
    }
    return pipeline, cv_summary


def run_phase4_svr_for_ticker(
    ticker,
    standard_df,
    target_growth_rate=10.0,
    report_dir="analysis/reports",
):
    """Train Phase 4 SVR for one uploaded ticker and update shared report files."""
    if standard_df is None or len(standard_df) == 0:
        raise ValueError(f"No training data provided for {ticker}")

    df = standard_df.copy()
    if "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for ticker {ticker}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    base_periods = _infer_periods_per_year(df["date"])
    horizon = _select_prediction_horizon(base_periods)
    effective_periods = _effective_periods_per_year(base_periods, horizon)

    data = build_supervised_dataset(df, target_col="net_income", horizon=horizon)
    X, y = data["X"], data["y"]
    dates = pd.to_datetime(data["model_df"]["date"], errors="coerce")

    period_target_growth = _annual_to_period_target(target_growth_rate, effective_periods)

    if len(X) < 2:
        # Not enough rows to fit SVR; produce a deterministic fallback from
        # observed growth so recommendations remain ticker-specific.
        observed_growth = float(y.iloc[0]) if len(y) == 1 else 0.0
        confidence_sigma = max(1.0, abs(observed_growth) * 0.2)

        future_df = data["future_rows"][["ticker", "date", "net_income"]].copy()
        future_df = future_df[
            future_df["ticker"].astype(str).str.upper() == str(ticker).upper()
        ].copy()
        if future_df.empty:
            raise ValueError(f"Could not generate future prediction row for {ticker}")

        future_df["current_net_income"] = future_df["net_income"]
        future_df["predicted_growth_rate"] = observed_growth
        future_df["target_growth_rate"] = period_target_growth
        future_df["gap_vs_target"] = observed_growth - period_target_growth
        future_df["gap_status"] = np.where(
            future_df["gap_vs_target"] >= 0, "surplus", "shortfall"
        )
        future_df["confidence_lower_95"] = observed_growth - 1.96 * confidence_sigma
        future_df["confidence_upper_95"] = observed_growth + 1.96 * confidence_sigma
        future_df["predicted_next_net_income"] = future_df["current_net_income"] * (1 + observed_growth / 100)
        future_df["target_next_net_income"] = future_df["current_net_income"] * (1 + period_target_growth / 100)
        future_df["period_type"] = _period_type_from_frequency(effective_periods)
        future_df["periods_per_year"] = float(effective_periods)
        future_df["forecast_horizon_periods"] = int(horizon)
        future_df["predicted_growth_rate_period_pct"] = future_df["predicted_growth_rate"]
        future_df["target_growth_rate_period_pct"] = future_df["target_growth_rate"]
        future_df["gap_vs_target_period_pct"] = future_df["gap_vs_target"]
        future_df["predicted_growth_rate_annualized_pct"] = _period_to_annual_growth(observed_growth, effective_periods)
        future_df["target_growth_rate_annualized_pct"] = _period_to_annual_growth(period_target_growth, effective_periods)
        future_df["gap_vs_target_annualized_pct"] = (
            future_df["predicted_growth_rate_annualized_pct"]
            - future_df["target_growth_rate_annualized_pct"]
        )
        future_df["model_reliability"] = "Low"
        future_df["model_reliability_reason"] = "insufficient_training_rows"
        future_df["model_r2"] = np.nan
        future_df["benchmark_naive_mae"] = np.nan
        future_df["beats_naive"] = False

        os.makedirs(report_dir, exist_ok=True)
        future_path = os.path.join(report_dir, "svr_future_predictions.csv")
        if os.path.exists(future_path):
            existing = pd.read_csv(future_path)
            if "ticker" in existing.columns:
                existing = existing[
                    existing["ticker"].astype(str).str.upper() != str(ticker).upper()
                ]
            merged = pd.concat([existing, future_df], ignore_index=True)
        else:
            merged = future_df
        merged.to_csv(future_path, index=False)

        metrics_path = os.path.join(report_dir, "svr_evaluation_metrics.csv")
        pd.DataFrame([
            {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "residual_mean": np.nan,
                "residual_std": np.nan,
                "residual_min": np.nan,
                "residual_max": np.nan,
                "test_size": 0,
            }
        ]).to_csv(metrics_path, index=False)

        return {
            "ticker": ticker,
            "future_path": future_path,
            "metrics_path": metrics_path,
            "test_metrics": {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "test_size": 0,
                "model_reliability": "Low",
                "model_reliability_reason": "insufficient_training_rows",
                "naive_mae": np.nan,
                "beats_naive": False,
            },
            "cv_summary": {"best_params": {"mode": "fallback_single_sample"}},
            "cutoff_date": str(pd.to_datetime(data["future_rows"]["date"]).max().date()),
        }

    unique_dates = sorted(pd.Series(dates).dropna().unique())
    if len(unique_dates) >= 4:
        X_train, X_test, y_train, y_test, cutoff = time_aware_split(X, y, dates, test_ratio=0.2)
    else:
        X_train, X_test, y_train, y_test, cutoff = _chronological_holdout_split(X, y, dates)

    candidates = train_candidate_models(X_train, y_train)
    champion = select_champion_model(candidates, X_test, y_test)
    model = champion["model"]
    svr_cv_summary = candidates["SVR"]["cv_summary"]
    cv_summary = champion["cv_summary"] if champion["cv_summary"] is not None else svr_cv_summary

    test_metrics, _ = evaluate_model(model, X_test, y_test)
    test_metrics["selected_model"] = champion["name"]

    benchmark_rows = evaluate_benchmark_models(X_train, y_train, X_test, y_test)
    reliability = assess_model_reliability(test_metrics, benchmark_rows)
    test_metrics.update(reliability)

    residual_std = float(test_metrics.get("residual_std", 0.0) or 0.0)
    confidence_sigma = residual_std if residual_std > 0 else 1.0
    recent_priors = _build_recent_growth_priors(data["metadata"].loc[X_train.index], y_train)
    level_priors = _build_level_trend_priors(df, horizon)
    trend_priors = _merge_growth_priors(recent_priors, level_priors, recent_weight=0.4)
    blend_weight = float(max(0.35, min(0.85, 0.35 + 0.45 * float(test_metrics["r2"]))))

    future_df = predict_future_and_gaps(
        model=model,
        future_rows=data["future_rows"],
        future_X=data["future_X"],
        confidence_sigma=confidence_sigma,
        target_growth_rate=period_target_growth,
        periods_per_year=effective_periods,
        trend_priors=trend_priors,
        blend_weight=blend_weight,
    )
    future_df["forecast_horizon_periods"] = int(horizon)
    future_df["model_reliability"] = test_metrics["model_reliability"]
    future_df["model_reliability_reason"] = test_metrics["model_reliability_reason"]
    future_df["selected_model"] = test_metrics["selected_model"]
    future_df["model_r2"] = test_metrics["r2"]
    future_df["benchmark_naive_mae"] = test_metrics["naive_mae"]
    future_df["beats_naive"] = test_metrics["beats_naive"]

    future_df = future_df[
        future_df["ticker"].astype(str).str.upper() == str(ticker).upper()
    ].copy()

    if future_df.empty:
        raise ValueError(f"Could not generate future prediction row for {ticker}")

    os.makedirs(report_dir, exist_ok=True)
    future_path = os.path.join(report_dir, "svr_future_predictions.csv")

    if os.path.exists(future_path):
        existing = pd.read_csv(future_path)
        if "ticker" in existing.columns:
            existing = existing[
                existing["ticker"].astype(str).str.upper() != str(ticker).upper()
            ]
        merged = pd.concat([existing, future_df], ignore_index=True)
    else:
        merged = future_df

    merged.to_csv(future_path, index=False)

    benchmark_path = os.path.join(report_dir, "svr_benchmark_comparison.csv")
    benchmark_df = pd.DataFrame(benchmark_rows)
    if not benchmark_df.empty:
        benchmark_df["ticker"] = str(ticker).upper()
        benchmark_df["model_reliability"] = test_metrics["model_reliability"]
        benchmark_df["model_reliability_reason"] = test_metrics["model_reliability_reason"]

        if os.path.exists(benchmark_path):
            existing_bench = pd.read_csv(benchmark_path)
            if "ticker" in existing_bench.columns:
                existing_bench = existing_bench[
                    existing_bench["ticker"].astype(str).str.upper() != str(ticker).upper()
                ]
            benchmark_merged = pd.concat([existing_bench, benchmark_df], ignore_index=True)
        else:
            benchmark_merged = benchmark_df
        benchmark_merged.to_csv(benchmark_path, index=False)

    metrics_path = os.path.join(report_dir, "svr_evaluation_metrics.csv")
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)

    return {
        "ticker": ticker,
        "future_path": future_path,
        "metrics_path": metrics_path,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "cutoff_date": str(pd.to_datetime(cutoff).date()),
        "benchmark_path": benchmark_path,
    }
