# =========================================================
# Streamlit KPI Anomaly Dashboard
# - Use processed files OR upload any CSV
# - Auto feature-engineering via src/
# - Multiple detectors: IsolationForest, One-Class SVM, LOF
# - Interactive plots + downloadable anomalies
# =========================================================

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
from src import (
    build_features, run_iforest, run_ocsvm, run_lof,
    zscore_flags, iqr_flags,
    plot_series_with_anoms, counts_bar
)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------
# Streamlit UI setup
# ---------------------------------------------------------
st.set_page_config(page_title="KPI Anomaly Detector", layout="wide")
st.title("📈 KPI Anomaly Detector")

# Sidebar controls
with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose data source", ["Use processed files", "Upload CSV"], index=0)
    contamination = st.slider("Contamination (expected anomaly %)", 0.005, 0.10, 0.02, 0.005)
    n_estimators = st.slider("IsolationForest trees", 100, 700, 300, 50)
    use_ocsvm = st.checkbox("Enable One-Class SVM (slower)", value=False)
    use_lof = st.checkbox("Enable Local Outlier Factor", value=True)
    consensus_k = st.slider("Consensus: min # detectors to flag", 1, 3, 2)
    st.caption("Tip: Start with IsolationForest only, then enable OCSVM/LOF for comparison.")

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
def find_default_processed():
    """Try to locate default processed data directory."""
    cwd = Path.cwd()
    for base in [cwd, cwd.parent, cwd.parent.parent]:
        candidate = base / "data" / "processed" / "clean_data.csv"
        if candidate.exists():
            return candidate.parent
    return None

def coerce_datetime(df: pd.DataFrame):
    """Find and convert a date-like column."""
    for cand in ["date", "ds", "timestamp", "time", "Date", "DATE"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            if df[cand].notna().any():
                return cand
    return None

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
st.dataframe(df_input.head(10), width="stretch")

# ---------------------------------------------------------
# Date + KPI selection
# ---------------------------------------------------------
date_col = coerce_datetime(df_input)
if date_col is None:
    st.error("Could not auto-detect a date/time column. Please rename it to 'date'.")
    st.stop()

df_input = df_input.sort_values(date_col).reset_index(drop=True)
numeric_cols = [c for c in df_input.columns if c != date_col and pd.api.types.is_numeric_dtype(df_input[c])]
if not numeric_cols:
    st.error("No numeric KPI columns found in the data.")
    st.stop()

kpi = st.selectbox("Select KPI to analyze", numeric_cols, index=0)
st.markdown(f"**Date column:** `{date_col}` • **KPI:** `{kpi}`")
st.write(f"Date range: {df_input[date_col].min().date()} → {df_input[date_col].max().date()}")

# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------
df_feat = build_features(df_input[[date_col] + numeric_cols], date_col=date_col, target=kpi)

feature_cols = [
    kpi, f"{kpi}_lag_7", f"{kpi}_lag_28", f"{kpi}_rollmean_7", f"{kpi}_rollstd_7",
    f"{kpi}_pct_change_1", f"{kpi}_pct_change_7", "dow", "month", "is_weekend"
]
for c in ["onpromotion", "transactions", "oil_price"]:
    if c in df_feat.columns:
        feature_cols.append(c)

feature_cols = [c for c in feature_cols if c in df_feat.columns]
if not feature_cols:
    st.error("No feature columns available for modeling. Check your input data.")
    st.stop()

# ---------------------------------------------------------
# Baselines (Z-score / IQR)
# ---------------------------------------------------------
z, zmask = zscore_flags(df_feat[kpi], 3.0)
iqr_mask, _ = iqr_flags(df_feat[kpi], 1.5)

df_det = df_feat[[date_col, kpi]].copy()
df_det["zscore"] = z
df_det["anom_zscore"] = zmask
df_det["anom_iqr"] = iqr_mask

# ---------------------------------------------------------
# ML Models
# ---------------------------------------------------------
if_score, if_flags = run_iforest(df_feat[feature_cols], contamination=contamination, n_estimators=n_estimators)
df_det["iforest_score"] = if_score
df_det["anom_iforest"] = if_flags

if use_ocsvm:
    try:
        ocsvm_score, ocsvm_flags = run_ocsvm(df_feat[feature_cols], nu_val=float(contamination))
        df_det["ocsvm_score"] = ocsvm_score
        df_det["anom_ocsvm"] = ocsvm_flags
    except Exception as e:
        st.warning(f"OCSVM failed: {e}")

if use_lof:
    try:
        lof_score, lof_flags = run_lof(df_feat[feature_cols], contamination=contamination, n_neighbors=20)
        df_det["lof_score"] = lof_score
        df_det["anom_lof"] = lof_flags
    except Exception as e:
        st.warning(f"LOF failed: {e}")

flag_cols = [c for c in ["anom_iforest", "anom_ocsvm", "anom_lof", "anom_zscore", "anom_iqr"] if c in df_det.columns]
df_det["vote_sum"] = df_det[flag_cols].sum(axis=1)
df_det["anom_consensus"] = (df_det["vote_sum"] >= consensus_k).astype(int)

# ---------------------------------------------------------
# Summary visualization
# ---------------------------------------------------------
st.subheader("Model Summary")
counts = df_det[flag_cols + ["anom_consensus"]].sum().sort_values(ascending=False)
st.write("Anomaly counts by detector:")
st.plotly_chart(counts_bar(counts), width="stretch")

# ---------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Time Series with Anomalies")
    fig = plot_series_with_anoms(
        df=df_det, date_col=date_col, value_col=kpi,
        flag_col="anom_consensus", title=f"{kpi} — Consensus anomalies (k ≥ {consensus_k})"
    )
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Top Anomalies")
    top = (df_det[df_det["anom_consensus"] == 1]
           .sort_values([date_col], ascending=True)
           .tail(25)
           .loc[:, [date_col, kpi, "vote_sum"] + flag_cols])
    st.dataframe(top, width="stretch", height=520)

# ---------------------------------------------------------
# Download results
# ---------------------------------------------------------
st.subheader("Download Results")
anom_only = df_det[df_det["anom_consensus"] == 1].copy()
csv_bytes = anom_only.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download anomalies (CSV)",
    data=csv_bytes,
    file_name=f"anomalies_{kpi}.csv",
    mime="text/csv"
)

st.caption("Tip: Try different KPIs, contamination levels, and consensus thresholds to explore behavior.")