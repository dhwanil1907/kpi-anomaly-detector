from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def load_local_store_sales(data_dir: str | Path, grain: str = "global") -> pd.DataFrame:
    """
    Load the Kaggle Store Sales dataset from a local directory.
    Minimal columns used: date, store_nbr, sales, onpromotion, transactions, oil, holidays.
    Args:
        data_dir: folder with CSVs (train.csv, transactions.csv, oil.csv, holidays_events.csv, stores.csv)
        grain: "global" (aggregate by date) or "store" (keep store_nbr)
    """
    data_dir = Path(data_dir)
    train_fp = data_dir / "train.csv"
    trans_fp = data_dir / "transactions.csv"
    oil_fp = data_dir / "oil.csv"
    hol_fp = data_dir / "holidays_events.csv"
    stores_fp = data_dir / "stores.csv"

    if not train_fp.exists():
        raise FileNotFoundError(f"Missing file: {train_fp}")

    use_cols = ["date", "store_nbr", "sales", "onpromotion"]
    df = pd.read_csv(train_fp, parse_dates=["date"])[use_cols]

    if grain == "global":
        df = df.groupby("date", as_index=False).agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )

    if trans_fp.exists():
        trans = pd.read_csv(trans_fp, parse_dates=["date"])
        if grain == "global":
            trans = trans.groupby("date", as_index=False)["transactions"].sum()
            df = df.merge(trans, on="date", how="left")
        else:
            df = df.merge(trans, on=["date", "store_nbr"], how="left")

    if oil_fp.exists():
        oil = pd.read_csv(oil_fp, parse_dates=["date"]).rename(columns={"dcoilwtico": "oil_price"})
        oil = oil.sort_values("date").set_index("date").ffill().reset_index()
        df = df.merge(oil, on="date", how="left")

    if hol_fp.exists():
        hol = pd.read_csv(hol_fp, parse_dates=["date"])
        hol = hol[hol["transferred"] == False]
        hol["is_holiday"] = 1
        hol = hol.groupby("date", as_index=False)["is_holiday"].max()
        df = df.merge(hol, on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    if stores_fp.exists() and grain != "global":
        stores = pd.read_csv(stores_fp)
        df = df.merge(stores, on="store_nbr", how="left")

    for c in ["transactions", "oil_price"]:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    sort_cols = ["date"] + (["store_nbr"] if grain != "global" else [])
    return df.sort_values(sort_cols).reset_index(drop=True)


def make_synthetic(days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic multi-KPI dataset with injected anomalies."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=days)
    sales = 200 + np.sin(np.linspace(0, 15, days)) * 20 + rng.normal(0, 5, days)
    revenue = sales * (10 + rng.normal(0, 0.5, days))
    cost = revenue * rng.uniform(0.6, 0.8, days)
    engagement = rng.normal(100, 10, days)
    oil_price = 60 + np.sin(np.linspace(0, 30, days)) * 5 + rng.normal(0, 1, days)

    # anomalies
    idx = rng.choice(len(dates), 12, replace=False)
    sales[idx] *= rng.uniform(0.5, 1.6, len(idx))

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "revenue": revenue,
            "cost": cost,
            "engagement": engagement,
            "transactions": (sales * rng.uniform(0.8, 1.2)).astype(int),
            "oil_price": oil_price,
            "is_holiday": (pd.Series(dates).dt.dayofweek >= 5).astype(int),
        }
    )
    return df


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), parse_dates=["date"])