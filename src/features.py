# features.py — Feature engineering utilities

import pandas as pd

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features (day of week, month, weekend flag)."""
    out = df.copy()
    d = pd.to_datetime(out[date_col], errors="coerce")
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def add_lag_roll(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add lag, rolling mean/std, and percent change features for a target column."""
    out = df.copy()
    if target not in out.columns:
        return out
    out[f"{target}_lag_7"] = out[target].shift(7)
    out[f"{target}_lag_28"] = out[target].shift(28)
    out[f"{target}_rollmean_7"] = out[target].rolling(7).mean()
    out[f"{target}_rollstd_7"] = out[target].rolling(7).std()
    out[f"{target}_pct_change_1"] = out[target].pct_change(1)
    out[f"{target}_pct_change_7"] = out[target].pct_change(7)
    return out

def build_features(df: pd.DataFrame, date_col: str, target: str) -> pd.DataFrame:
    """Full feature-building pipeline."""
    tmp = add_time_features(df, date_col)
    tmp = add_lag_roll(tmp, target)
    return tmp.dropna().reset_index(drop=True)