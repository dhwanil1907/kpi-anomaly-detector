from __future__ import annotations
from typing import Tuple, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


# --------- Statistical detectors ---------
def zscore_flags(series: pd.Series, thresh: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Return z-scores and boolean mask where |z| >= thresh.
    """
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        z = pd.Series(np.zeros(len(series)), index=series.index)
    else:
        z = (series - mu) / sigma
    mask = z.abs() >= thresh
    return z, mask


def iqr_flags(series: pd.Series, k: float = 1.5) -> Tuple[pd.Series, Tuple[float, float]]:
    """
    Return mask of outliers using IQR rule and the (lo, hi) bounds.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (series < lo) | (series > hi)
    return mask, (lo, hi)


# --------- Isolation Forest (ML) ---------
def iforest_train(
    X: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 300,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples="auto",
        n_jobs=-1,
    )
    model.fit(X)
    return model


def iforest_infer(model: IsolationForest, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (decision_function, flags) where flags is 1 for anomaly.
    """
    decision = model.decision_function(X)  # higher is more normal
    preds = model.predict(X)               # -1 anomaly, 1 normal
    flags = (preds == -1).astype(int)
    return decision, flags


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(Path(path))


# --------- Prophet residuals (optional) ---------
def prophet_residual_flags(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    sigma_k: float = 3.0,
) -> Optional[pd.DataFrame]:
    """
    Fit Prophet (if available) on (date_col, target_col), compute residuals,
    and return a DataFrame with [date, yhat, resid, anom_prophet].
    Returns None if Prophet is not installed or fails.
    """
    try:
        from prophet import Prophet
    except Exception:
        return None

    try:
        dfp = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).copy()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
        m.fit(dfp)
        fcst = m.predict(dfp)[["ds", "yhat"]]
        merged = dfp.merge(fcst, on="ds", how="left")
        merged["resid"] = merged["y"] - merged["yhat"]
        std = merged["resid"].std(ddof=0)
        merged["anom_prophet"] = (merged["resid"].abs() >= sigma_k * std).astype(int)
        return merged.rename(columns={"ds": date_col})
    except Exception:
        return None