# detect.py — Statistical & ML-based anomaly detectors

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

RANDOM_SEED = 42

def zscore_flags(series: pd.Series, threshold: float = 3.0):
    """Return z-scores and binary anomaly mask."""
    z = (series - series.mean()) / series.std(ddof=0)
    return z, (z.abs() >= threshold).astype(int)

def iqr_flags(series: pd.Series, k: float = 1.5):
    """Flag outliers using IQR method."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = ((series < lo) | (series > hi)).astype(int)
    return mask, (lo, hi)

def _clean_X(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Replace inf/NaN and ensure numeric array."""
    if isinstance(X, pd.DataFrame):
        X = (
            X.replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0)
            .to_numpy()
        )
    X = np.asarray(X)
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Feature matrix contains NaN/Inf after cleaning.")
    return X

def run_iforest(X, contamination: float = 0.02, n_estimators: int = 300, random_state: int = RANDOM_SEED):
    """Isolation Forest detector."""
    Xc = _clean_X(X)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    model.fit(Xc)
    score = model.decision_function(Xc)
    flags = (model.predict(Xc) == -1).astype(int)
    return score, flags

def run_ocsvm(X, nu_val: float = 0.02):
    """One-Class SVM detector."""
    Xc = _clean_X(X)
    pipe = Pipeline([("scaler", StandardScaler()), ("ocsvm", OneClassSVM(kernel="rbf", gamma="scale", nu=nu_val))])
    pipe.fit(Xc)
    Xs = pipe.named_steps["scaler"].transform(Xc)
    model = pipe.named_steps["ocsvm"]
    score = model.decision_function(Xs)
    flags = (model.predict(Xs) == -1).astype(int)
    return score, flags

def run_lof(X, contamination: float = 0.02, n_neighbors: int = 20):
    """Local Outlier Factor detector."""
    Xc = _clean_X(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    flags = (lof.fit_predict(Xc) == -1).astype(int)
    score = -lof.negative_outlier_factor_
    return score, flags

def vote_consensus(df_flags: pd.DataFrame, k: int = 2) -> pd.Series:
    """Consensus voting: flag rows where at least k detectors agree."""
    vote = df_flags.sum(axis=1)
    return (vote >= k).astype(int)