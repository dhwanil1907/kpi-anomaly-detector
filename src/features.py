from __future__ import annotations
import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col])
        df["dow"] = d.dt.dayofweek
        df["month"] = d.dt.month
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_lag_roll(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    df = df.copy()
    if target not in df.columns:
        return df
    df[f"{target}_lag_7"] = df[target].shift(7)
    df[f"{target}_lag_28"] = df[target].shift(28)
    df[f"{target}_rollmean_7"] = df[target].rolling(7).mean()
    df[f"{target}_rollstd_7"] = df[target].rolling(7).std()
    df[f"{target}_pct_change_1"] = df[target].pct_change(1)
    df[f"{target}_pct_change_7"] = df[target].pct_change(7)
    return df


def build_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Add common time features + lags/rolling, drop NA rows created by lags."""
    out = add_time_features(df)
    out = add_lag_roll(out, target)
    return out.dropna().reset_index(drop=True)