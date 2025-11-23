# app.py
# =========================================================
# Streamlit KPI Anomaly Dashboard
# - Use processed files OR upload any CSV
# - Auto feature-engineering
# - Multiple detectors: IsolationForest, One-Class SVM (optional), LOF
# - Interactive plots + downloadable anomalies
# =========================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ------------- Utils ---------------------------------------------------------

RANDOM_SEED = 42

def find_default_processed():
    """
    Try common locations for processed files created by the notebook.
    """
    candidates = []
    cwd = Path.cwd()
    for base in [cwd, cwd.parent, cwd.parent.parent]:
        candidates.append(base / "data" / "processed" / "clean_data.csv")
        candidates.append(base / "data" / "processed" / "anomalies.csv")
    # Return dir if clean_data exists
    for p in candidates:
        if p.name == "clean_data.csv" and p.exists():
            return p.parent
    return None


def coerce_datetime(df: pd.DataFrame):
    """
    Make a best-effort to coerce a date-like column to datetime.
    """
    for cand in ["date", "ds", "timestamp", "time", "Date", "DATE"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            if df[cand].notna().any():
                return cand
    # if not found, try the first datetime-convertible column
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.7:
                df[col] = converted
                return col
        except Exception:
            pass
    return None


def add_time_features(df: pd.DataFrame, date_col: str):
    out = df.copy()
    d = pd.to_datetime(out[date_col], errors="coerce")
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_lag_roll(df: pd.DataFrame, target: str):
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


def build_features(df: pd.DataFrame, date_col: str, target: str):
    tmp = add_time_features(df, date_col)
    tmp = add_lag_roll(tmp, target)
    tmp = tmp.dropna().reset_index(drop=True)
    return tmp


def zscore_flags(series: pd.Series, threshold: float = 3.0):
    z = (series - series.mean()) / series.std(ddof=0)
    return z, (z.abs() >= threshold).astype(int)


def iqr_flags(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = ((series < lo) | (series > hi)).astype(int)
    return mask, (lo, hi)


def run_iforest(X: np.ndarray, contamination: float, n_estimators: int):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_SEED,
    )
    model.fit(X)
    score = model.decision_function(X)          # higher = more normal
    flags = (model.predict(X) == -1).astype(int)  # 1 = anomaly
    return score, flags


def run_ocsvm(X: np.ndarray, nu_val: float):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", gamma="scale", nu=nu_val)),
    ])
    pipe.fit(X)
    Xs = pipe.named_steps["scaler"].transform(X)
    model = pipe.named_steps["ocsvm"]
    score = model.decision_function(Xs)
    flags = (model.predict(Xs) == -1).astype(int)
    return score, flags


def run_lof(X: np.ndarray, contamination: float, n_neighbors: int = 20):
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    flags = (lof.fit_predict(X) == -1).astype(int)
    score = -lof.negative_outlier_factor_
    return score, flags


def plot_series_with_anoms(df: pd.DataFrame, date_col: str, value_col: str, flag_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode="lines", name=value_col))
    if flag_col in df.columns:
        mask = df[flag_col] == 1
        fig.add_trace(go.Scatter(
            x=df.loc[mask, date_col],
            y=df.loc[mask, value_col],
            mode="markers",
            name="Anomaly",
            marker=dict(size=8, symbol="circle-open")
        ))
    fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col)
    return fig


# ------------- Streamlit UI --------------------------------------------------

st.set_page_config(page_title="KPI Anomaly Detector", layout="wide")
st.title("📈 KPI Anomaly Detector")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose data source", ["Use processed files", "Upload CSV"], index=0)

    contamination = st.slider("Contamination (expected anomaly %)", 0.005, 0.10, 0.02, 0.005)
    n_estimators = st.slider("IsolationForest trees", 100, 700, 300, 50)
    use_ocsvm = st.checkbox("Enable One-Class SVM (slower)", value=False)
    use_lof = st.checkbox("Enable Local Outlier Factor", value=True)
    consensus_k = st.slider("Consensus: min # detectors to flag", 1, 3, 2)

    st.caption("Tip: start with IF only. Then toggle OCSVM/LOF to compare. Adjust contamination to control strictness.")


# ------------- Load Data -----------------------------------------------------

df_input = None
source_label = ""

if mode == "Use processed files":
    processed_dir = find_default_processed()
    if processed_dir is None:
        st.warning("Couldn't find `data/processed/clean_data.csv`. Upload a CSV instead.")
    else:
        fp = processed_dir / "clean_data.csv"
        df_input = pd.read_csv(fp)
        source_label = f"Processed file: {fp}"
else:
    up = st.file_uploader("Upload a CSV (must include a date column and at least 1 numeric KPI)", type=["csv"])
    if up is not None:
        try:
            df_input = pd.read_csv(up)
            source_label = f"Uploaded file: {up.name}"
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if df_input is None:
    st.stop()

st.success(f"✅ Loaded data — {source_label}")
st.write("Shape:", df_input.shape)
st.dataframe(df_input.head(10), use_container_width=True)


# ------------- Date + KPI selection -----------------------------------------

date_col = coerce_datetime(df_input)
if date_col is None:
    st.error("Could not auto-detect a date/time column. Please rename your time column to 'date'.")
    st.stop()

# reorder by date
df_input = df_input.sort_values(date_col).reset_index(drop=True)

# KPI column selection
numeric_cols = [c for c in df_input.columns if c != date_col and pd.api.types.is_numeric_dtype(df_input[c])]
if not numeric_cols:
    st.error("No numeric KPI columns found in the data.")
    st.stop()

kpi = st.selectbox("Select KPI to analyze", numeric_cols, index=0)

st.markdown(f"**Date column:** `{date_col}` • **KPI:** `{kpi}`")
st.write(f"Date range: {df_input[date_col].min().date()} → {df_input[date_col].max().date()}")


# ------------- Feature Engineering ------------------------------------------

df_feat = build_features(df_input[[date_col] + numeric_cols], date_col=date_col, target=kpi)

# features used for ML
feature_cols = [
    kpi, f"{kpi}_lag_7", f"{kpi}_lag_28", f"{kpi}_rollmean_7", f"{kpi}_rollstd_7",
    f"{kpi}_pct_change_1", f"{kpi}_pct_change_7", "dow", "month", "is_weekend"
]
# add other numeric context columns if present
for c in ["onpromotion", "transactions", "oil_price"]:
    if c in df_feat.columns:
        feature_cols.append(c)

feature_cols = [c for c in feature_cols if c in df_feat.columns]
if not feature_cols:
    st.error("No feature columns available for modeling. Check your input data.")
    st.stop()

# Clean numeric features before model training
X = df_feat[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(method="ffill").fillna(method="bfill").fillna(0).to_numpy()

# Defensive check again
if np.isnan(X).any() or np.isinf(X).any():
    st.error("❌ Still found NaN/Inf in feature matrix. Check your input CSV for missing values or gaps.")
    st.stop()

# ------------- Baselines (Z-score / IQR) ------------------------------------

z, zmask = zscore_flags(df_feat[kpi], 3.0)
iqr_mask, _ = iqr_flags(df_feat[kpi], 1.5)

df_det = df_feat[[date_col, kpi]].copy()
df_det["zscore"] = z
df_det["anom_zscore"] = zmask
df_det["anom_iqr"] = iqr_mask


# ------------- ML Models -----------------------------------------------------

# Isolation Forest (always on)
if_score, if_flags = run_iforest(X, contamination=contamination, n_estimators=n_estimators)
df_det["iforest_score"] = if_score
df_det["anom_iforest"] = if_flags

# One-Class SVM (optional)
if use_ocsvm:
    try:
        ocsvm_score, ocsvm_flags = run_ocsvm(X, nu_val=float(contamination))
        df_det["ocsvm_score"] = ocsvm_score
        df_det["anom_ocsvm"] = ocsvm_flags
    except Exception as e:
        st.warning(f"OCSVM failed: {e}")

# LOF (optional)
if use_lof:
    try:
        lof_score, lof_flags = run_lof(X, contamination=contamination, n_neighbors=20)
        df_det["lof_score"] = lof_score
        df_det["anom_lof"] = lof_flags
    except Exception as e:
        st.warning(f"LOF failed: {e}")

# consensus
flag_cols = [c for c in ["anom_iforest", "anom_ocsvm", "anom_lof", "anom_zscore", "anom_iqr"] if c in df_det.columns]
df_det["vote_sum"] = df_det[flag_cols].sum(axis=1)
df_det["anom_consensus"] = (df_det["vote_sum"] >= consensus_k).astype(int)

st.subheader("Model Summary")
counts = df_det[flag_cols + ["anom_consensus"]].sum().sort_values(ascending=False)
st.write("Anomaly counts by detector:")

fig_counts = px.bar(
    x=counts.index,
    y=counts.values,
    title="Anomaly counts by detector",
    labels={"x": "Detector", "y": "Anomalies"},
)
st.plotly_chart(fig_counts, width="stretch")


# ------------- Plots ---------------------------------------------------------

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Time Series with Anomalies")
    fig = plot_series_with_anoms(
        df_det, date_col=date_col, value_col=kpi,
        flag_col="anom_consensus", title=f"{kpi} — Consensus anomalies (k ≥ {consensus_k})"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Top Anomalies")
    top = (df_det[df_det["anom_consensus"] == 1]
           .sort_values([date_col], ascending=True)
           .tail(25)
           .loc[:, [date_col, kpi, "vote_sum"] + flag_cols])
    st.dataframe(top, use_container_width=True, height=520)


# ------------- Download ------------------------------------------------------

st.subheader("Download Results")
anom_only = df_det[df_det["anom_consensus"] == 1].copy()
csv_bytes = anom_only.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download anomalies (CSV)",
    data=csv_bytes,
    file_name=f"anomalies_{kpi}.csv",
    mime="text/csv"
)

st.caption("Tip: try different KPIs, contamination levels, and consensus thresholds to explore behavior.")