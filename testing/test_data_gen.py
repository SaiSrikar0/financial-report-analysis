import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent

tickers = ["META", "NFLX", "NVDA", "ADBE", "INTC", "AMD", "CRM", "ORCL"]
dates = pd.date_range(start="2000-01-01", periods=200, freq="Q")

# ---------- PERFECT DATASET ----------
perfect_data = []

for _ in range(10000):
    ticker = np.random.choice(tickers)
    date = np.random.choice(dates)

    revenue = np.random.uniform(100, 800) * 1e9
    operating_income = revenue * np.random.uniform(0.25, 0.5)
    net_income = operating_income * np.random.uniform(0.7, 0.9)
    total_assets = revenue * np.random.uniform(2.0, 4.0)
    total_liabilities = total_assets * np.random.uniform(0.2, 0.5)
    cashflow = net_income * np.random.uniform(1.0, 1.3)

    perfect_data.append([date, ticker, revenue, operating_income,
                         net_income, total_assets, total_liabilities, cashflow])

perfect_df = pd.DataFrame(perfect_data, columns=[
    "date", "ticker", "revenue", "operating_income", "net_income",
    "total_assets", "total_liabilities", "operating_cashflow"
])

perfect_df.to_csv(OUTPUT_DIR / "perfect_dataset_10000.csv", index=False)


# ---------- BAD DATASET ----------
bad_data = []

for _ in range(10000):
    ticker = np.random.choice(tickers)
    date = np.random.choice(dates)

    revenue = np.random.uniform(50, 300) * 1e9
    operating_income = revenue * np.random.uniform(-0.1, 0.1)
    net_income = operating_income * np.random.uniform(-1.5, 0.5)
    total_assets = revenue * np.random.uniform(1.0, 2.0)
    total_liabilities = total_assets * np.random.uniform(0.7, 1.2)
    cashflow = net_income * np.random.uniform(-1.0, 0.5)

    bad_data.append([date, ticker, revenue, operating_income,
                     net_income, total_assets, total_liabilities, cashflow])

bad_df = pd.DataFrame(bad_data, columns=perfect_df.columns)
bad_df.to_csv(OUTPUT_DIR / "bad_dataset_10000.csv", index=False)


# ---------- MESSY DATASET ----------
messy_data = []

for _ in range(10000):
    ticker = np.random.choice(tickers)
    date = np.random.choice(dates)

    revenue = np.random.uniform(50, 500) * 1e9

    # Introduce missing values randomly
    if np.random.rand() < 0.1:
        revenue = None

    operating_income = revenue * np.random.uniform(0.1, 0.4) if revenue else None
    net_income = operating_income * np.random.uniform(0.5, 0.9) if operating_income else None
    total_assets = (revenue or np.random.uniform(50, 300) * 1e9) * np.random.uniform(1.5, 3.0)
    total_liabilities = total_assets * np.random.uniform(0.4, 0.9)
    cashflow = net_income * np.random.uniform(0.7, 1.2) if net_income else None

    # Add noise / anomalies
    if np.random.rand() < 0.05:
        total_assets *= 10  # extreme outlier

    messy_data.append([date, ticker, revenue, operating_income,
                       net_income, total_assets, total_liabilities, cashflow])

messy_df = pd.DataFrame(messy_data, columns=perfect_df.columns)
messy_df.to_csv(OUTPUT_DIR / "messy_dataset_10000.csv", index=False)


# ---------- THREE NEW DATASETS (NEW COMPANY/TICKER NAMES) ----------
# Note: "perfect" is intentionally engineered for very strong profitability,
# stable leverage and smooth growth, which is the best chance to reach top score.

new_columns = [
    "date", "ticker", "revenue", "operating_income", "net_income",
    "total_assets", "total_liabilities", "operating_cashflow",
]


def generate_perfect_aurx(n_rows: int = 10000) -> pd.DataFrame:
    ticker = "AURX"
    dates = pd.date_range(start="1999-01-01", periods=n_rows, freq="D")

    rows = []
    base_revenue = 150e9
    for i, dt in enumerate(dates):
        trend = (1.00035 ** i)
        revenue = base_revenue * trend * np.random.uniform(0.997, 1.003)
        operating_income = revenue * np.random.uniform(0.36, 0.44)
        net_income = operating_income * np.random.uniform(0.78, 0.88)
        total_assets = revenue * np.random.uniform(2.2, 2.9)
        total_liabilities = total_assets * np.random.uniform(0.18, 0.28)
        cashflow = net_income * np.random.uniform(1.08, 1.25)
        rows.append([
            dt, ticker, revenue, operating_income, net_income,
            total_assets, total_liabilities, cashflow,
        ])

    return pd.DataFrame(rows, columns=new_columns)


def generate_bad_crsx(n_rows: int = 10000) -> pd.DataFrame:
    ticker = "CRSX"
    dates = pd.date_range(start="1999-01-01", periods=n_rows, freq="D")

    rows = []
    base_revenue = 240e9
    for i, dt in enumerate(dates):
        trend = (0.9997 ** i)
        revenue = base_revenue * trend * np.random.uniform(0.96, 1.03)
        operating_income = revenue * np.random.uniform(-0.18, 0.03)
        net_income = operating_income * np.random.uniform(0.5, 1.8)
        total_assets = revenue * np.random.uniform(1.1, 1.8)
        total_liabilities = total_assets * np.random.uniform(0.9, 1.45)
        cashflow = net_income * np.random.uniform(-1.3, 0.4)

        # Add frequent stress spikes to force risk-heavy recommendations.
        if np.random.rand() < 0.12:
            operating_income *= np.random.uniform(1.5, 2.5)
            net_income *= np.random.uniform(1.8, 3.0)
            cashflow *= np.random.uniform(1.5, 2.8)
            total_liabilities *= np.random.uniform(1.2, 1.8)

        rows.append([
            dt, ticker, revenue, operating_income, net_income,
            total_assets, total_liabilities, cashflow,
        ])

    return pd.DataFrame(rows, columns=new_columns)


def generate_messy_znqt(n_rows: int = 10000) -> pd.DataFrame:
    ticker = "ZNQT"
    dates = pd.date_range(start="1999-01-01", periods=n_rows, freq="D")

    rows = []
    for dt in dates:
        revenue = np.random.uniform(60, 420) * 1e9

        if np.random.rand() < 0.12:
            revenue = None

        if revenue is None:
            operating_income = None
            net_income = None
            cashflow = None
            total_assets = np.random.uniform(70, 350) * 1e9
        else:
            operating_income = revenue * np.random.uniform(0.06, 0.34)
            net_income = operating_income * np.random.uniform(0.45, 0.9)
            cashflow = net_income * np.random.uniform(0.6, 1.3)
            total_assets = revenue * np.random.uniform(1.4, 3.4)

        total_liabilities = total_assets * np.random.uniform(0.35, 0.95)

        # Outliers and formatting noise to test robust parsing.
        if np.random.rand() < 0.05 and total_assets is not None:
            total_assets *= np.random.uniform(5, 15)

        if np.random.rand() < 0.05 and net_income is not None:
            net_income = -abs(net_income)

        # Occasionally write string-formatted numerics as uploaded real-world data does.
        if np.random.rand() < 0.06 and revenue is not None:
            revenue = f"${revenue:,.0f}"
        if np.random.rand() < 0.04 and total_liabilities is not None:
            total_liabilities = f"({total_liabilities:,.0f})"

        rows.append([
            dt, ticker, revenue, operating_income, net_income,
            total_assets, total_liabilities, cashflow,
        ])

    return pd.DataFrame(rows, columns=new_columns)


def generate_high_health_lyrx_100(n_rows: int = 100) -> pd.DataFrame:
    """
    Generate a compact, high-quality dataset designed to produce strong
    health-scoring signals (smooth growth, strong margins, low leverage).
    """
    ticker = "LYRX"
    dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")

    rows = []
    base_revenue = 220e9
    for i, dt in enumerate(dates):
        # Steady daily growth with light noise to avoid a perfectly flat synthetic pattern.
        trend = 1.0012 ** i
        revenue = base_revenue * trend * np.random.uniform(0.999, 1.001)

        # Healthy operating + net margins.
        operating_income = revenue * np.random.uniform(0.40, 0.47)
        net_income = operating_income * np.random.uniform(0.84, 0.90)

        # Efficient balance sheet and low debt burden.
        total_assets = revenue * np.random.uniform(1.7, 2.1)
        total_liabilities = total_assets * np.random.uniform(0.10, 0.18)

        # Strong operating cash conversion.
        operating_cashflow = net_income * np.random.uniform(1.12, 1.30)

        rows.append([
            dt,
            ticker,
            revenue,
            operating_income,
            net_income,
            total_assets,
            total_liabilities,
            operating_cashflow,
        ])

    return pd.DataFrame(rows, columns=new_columns)


def generate_high_health_qnrx_200(n_rows: int = 200) -> pd.DataFrame:
    """
    Generate a 200-row high-health dataset with strong and stable trend,
    designed to produce a high recommendation score in current pipeline.
    """
    ticker = "QNRX"
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="D")
    rng = np.random.default_rng(2026)

    rows = []
    base_revenue = 180e9
    for i, dt in enumerate(dates):
        # Smooth compounding with very low noise improves learnability and holdout fit.
        trend = 1.0016 ** i
        revenue = base_revenue * trend * rng.uniform(0.9994, 1.0006)

        # Strong but realistic profitability profile.
        operating_income = revenue * rng.uniform(0.41, 0.47)
        net_income = operating_income * rng.uniform(0.86, 0.91)

        # Conservative leverage and healthy cash generation.
        total_assets = revenue * rng.uniform(1.8, 2.2)
        total_liabilities = total_assets * rng.uniform(0.08, 0.15)
        operating_cashflow = net_income * rng.uniform(1.15, 1.32)

        rows.append([
            dt,
            ticker,
            revenue,
            operating_income,
            net_income,
            total_assets,
            total_liabilities,
            operating_cashflow,
        ])

    return pd.DataFrame(rows, columns=new_columns)


aurx_perfect_df = generate_perfect_aurx(10000)
crsx_bad_df = generate_bad_crsx(10000)
znqt_messy_df = generate_messy_znqt(10000)
lyrx_high_health_100_df = generate_high_health_lyrx_100(100)
qnrx_high_health_200_df = generate_high_health_qnrx_200(200)

aurx_perfect_df.to_csv(OUTPUT_DIR / "perfect_dataset_aurx_10000.csv", index=False)
crsx_bad_df.to_csv(OUTPUT_DIR / "bad_dataset_crsx_10000.csv", index=False)
znqt_messy_df.to_csv(OUTPUT_DIR / "messy_dataset_znqt_10000.csv", index=False)
lyrx_high_health_100_df.to_csv(OUTPUT_DIR / "high_health_dataset_lyrx_100.csv", index=False)
qnrx_high_health_200_df.to_csv(OUTPUT_DIR / "high_health_dataset_qnrx_200.csv", index=False)


print("✅ All datasets generated successfully!")