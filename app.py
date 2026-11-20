# =========================================================
# Streamlit KPI Anomaly Dashboard
# =========================================================

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
from src import (
    build_features, feature_columns, run_iforest, run_ocsvm, run_lof,
    zscore_flags, iqr_flags,
    plot_series_with_anoms, counts_bar
)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="KPI Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1f2937;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.9rem !important; font-weight: 700 !important; color: #f9fafb !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; color: #6b7280 !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1f2937; border-radius: 10px; overflow: hidden;
}

[data-testid="stDownloadButton"] button {
    background: #2563eb !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.5rem 1.25rem !important;
}
[data-testid="stDownloadButton"] button:hover { background: #1d4ed8 !important; }

.section-label {
    font-size: 0.68rem; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;
}
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-red   { background:#450a0a; color:#fca5a5; border:1px solid #991b1b; }
.badge-green { background:#052e16; color:#86efac; border:1px solid #166534; }
.badge-blue  { background:#0c1a3a; color:#93c5fd; border:1px solid #1e40af; }
.badge-yellow{ background:#422006; color:#fcd34d; border:1px solid #92400e; }

.info-card {
    background: #161b27; border: 1px solid #1f2937; border-radius: 12px;
    padding: 1.1rem 1.25rem; margin-bottom: 1rem;
}
.info-card h4 { color: #f9fafb; font-size: 0.9rem; font-weight: 600; margin: 0 0 0.4rem 0; }
.info-card p  { color: #9ca3af; font-size: 0.82rem; margin: 0; line-height: 1.6; }

.finding-card {
    background: linear-gradient(135deg, #0c1a3a 0%, #161b27 100%);
    border: 1px solid #1e40af; border-radius: 12px; padding: 1.2rem 1.4rem;
}
.finding-card h3 { color: #93c5fd; font-size: 1rem; font-weight: 600; margin: 0 0 0.5rem 0; }
.finding-card p  { color: #bfdbfe; font-size: 0.85rem; margin: 0; line-height: 1.7; }

.severity-high   { color: #ef4444; font-weight: 700; }
.severity-medium { color: #f59e0b; font-weight: 600; }
.severity-low    { color: #3b82f6; font-weight: 500; }

hr { border-color: #1f2937; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div style="padding: 1.25rem 0 0.25rem 0;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
        <span style="font-size:1.75rem;">📈</span>
        <span style="font-size:1.55rem; font-weight:700; color:#f9fafb; letter-spacing:-0.02em;">
            KPI Anomaly Detector
        </span>
    </div>
    <p style="color:#6b7280; font-size:0.85rem; margin:0;">
        Automatically finds unusual patterns in stock data — helpful for spotting crashes, rallies, or data errors.
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown('<p class="section-label">Data Source</p>', unsafe_allow_html=True)
    mode = st.radio("", ["Use processed files", "Upload CSV"], index=0, label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Sensitivity</p>', unsafe_allow_html=True)
    contamination = st.slider(
        "How many anomalies to expect",
        0.005, 0.10, 0.02, 0.005,
        help="2% means ~1 in 50 days is flagged. Higher = more flags, lower = stricter.",
        format="%.3f"
    )
    n_estimators = st.slider(
        "Model precision (trees)",
        100, 700, 300, 50,
        help="More trees = more accurate but slower. 300 is a good balance."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Detection Methods</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="margin-bottom:0.5rem;">
        <h4>🌲 Isolation Forest <span class="badge badge-green">Always On</span></h4>
        <p>Randomly isolates data points. Anomalies are easier to isolate and get flagged.</p>
    </div>
    """, unsafe_allow_html=True)

    use_lof = st.checkbox("📍 Local Outlier Factor", value=True,
                          help="Flags points that are far from their neighbours in behaviour space.")
    st.markdown('<p style="color:#6b7280;font-size:0.75rem;margin:-0.5rem 0 0.75rem 1.5rem;">Compares each day to nearby days — great at catching local spikes.</p>', unsafe_allow_html=True)

    use_ocsvm = st.checkbox("🔷 One-Class SVM (slower)", value=False,
                            help="Learns a boundary around normal behaviour. Anything outside = anomaly.")
    st.markdown('<p style="color:#6b7280;font-size:0.75rem;margin:-0.5rem 0 0.75rem 1.5rem;">Most accurate but takes longer to run.</p>', unsafe_allow_html=True)

    st.markdown('<p style="color:#6b7280;font-size:0.75rem;margin:0.25rem 0 0.5rem 0;">📊 Statistical baselines (Z-Score & IQR) always run — no configuration needed.</p>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">How Many Methods Must Agree?</p>', unsafe_allow_html=True)
    consensus_k = st.slider("Agreement threshold", 1, 3, 2,
                             help="1 = flag if ANY method catches it (sensitive). 3 = all must agree (strict).")
    agree_label = {1: "Sensitive — catches more, may include false alarms",
                   2: "Balanced — good default",
                   3: "Strict — only very clear anomalies"}
    st.markdown(f'<p style="color:#9ca3af;font-size:0.75rem;">{agree_label[consensus_k]}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
def find_default_processed():
    cwd = Path.cwd()
    for base in [cwd, cwd.parent, cwd.parent.parent]:
        candidate = base / "data" / "processed" / "clean_data.csv"
        if candidate.exists():
            return candidate.parent
    return None

def coerce_datetime(df: pd.DataFrame):
    for cand in ["date", "ds", "timestamp", "time", "Date", "DATE"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            if df[cand].notna().any():
                return cand
    return None

df_input     = None
source_label = ""

if mode == "Use processed files":
    processed_dir = find_default_processed()
    if processed_dir is None:
        st.warning("⚠️ No processed data found. Run the notebook first, or switch to Upload CSV.")
        st.stop()
    fp       = processed_dir / "clean_data.csv"
    df_input = pd.read_csv(fp)
    # Keep only the raw columns; features are rebuilt below, and the engineered
    # columns saved by the notebook would otherwise show up as selectable KPIs.
    raw_cols = [c for c in ["date", "Open", "High", "Low", "Close", "Volume"] if c in df_input.columns]
    df_input = df_input[raw_cols]
    source_label = fp.name
else:
    up = st.file_uploader(
        "Upload a CSV — needs a **date** column and at least one **number** column (e.g. price, sales, traffic)",
        type=["csv"]
    )
    if up is not None:
        try:
            df_input     = pd.read_csv(up)
            source_label = up.name
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if df_input is None:
    st.markdown("""
    <div style="text-align:center; padding:5rem 0; color:#374151;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">📂</div>
        <p style="font-size:1.05rem; font-weight:600; color:#6b7280;">No data loaded yet</p>
        <p style="font-size:0.85rem; color:#4b5563;">
            Run the notebook first to generate processed files,<br>or switch to <strong>Upload CSV</strong> in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# Date + KPI selection
# ---------------------------------------------------------
date_col = coerce_datetime(df_input)
if date_col is None:
    st.error("Could not detect a date column. Please rename it to `date`.")
    st.stop()

df_input     = df_input.sort_values(date_col).reset_index(drop=True)
numeric_cols = [c for c in df_input.columns
                if c != date_col and pd.api.types.is_numeric_dtype(df_input[c])]
if not numeric_cols:
    st.error("No numeric KPI columns found.")
    st.stop()

kpi_col, info_col = st.columns([1, 3])
with kpi_col:
    kpi = st.selectbox("📊 Choose a metric to analyse", numeric_cols, index=0)
with info_col:
    date_min = df_input[date_col].min()
    date_max = df_input[date_col].max()
    st.markdown(f"""
    <div style="display:flex; gap:10px; align-items:center; margin-top:1.75rem; flex-wrap:wrap;">
        <span class="badge badge-green">✅ {source_label}</span>
        <span class="badge badge-blue">📅 {date_min.strftime('%b %d, %Y')} → {date_max.strftime('%b %d, %Y')}</span>
        <span class="badge badge-blue">🗂 {df_input.shape[0]:,} rows</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------
df_feat = build_features(df_input[[date_col] + numeric_cols], date_col=date_col, target=kpi)

feature_cols = feature_columns(df_feat, kpi)

if not feature_cols:
    st.error("No feature columns available. Check your input data.")
    st.stop()

# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------
# Statistical baselines run on 1-day % change: raw levels trend, so they would never flag anything
stat_series = df_feat[f"{kpi}_pct_change_1"]
z, zmask    = zscore_flags(stat_series, 3.0)
iqr_mask, _ = iqr_flags(stat_series, 1.5)

df_det = df_feat[[date_col, kpi]].copy()
df_det["zscore"]      = z
df_det["anom_zscore"] = zmask
df_det["anom_iqr"]    = iqr_mask

with st.spinner("🌲 Running Isolation Forest..."):
    if_score, if_flags      = run_iforest(df_feat[feature_cols], contamination=contamination, n_estimators=n_estimators)
    df_det["iforest_score"] = if_score
    df_det["anom_iforest"]  = if_flags

if use_ocsvm:
    with st.spinner("🔷 Running One-Class SVM..."):
        try:
            ocsvm_score, ocsvm_flags = run_ocsvm(df_feat[feature_cols], nu_val=float(contamination))
            df_det["ocsvm_score"]    = ocsvm_score
            df_det["anom_ocsvm"]     = ocsvm_flags
        except Exception as e:
            st.warning(f"OCSVM failed: {e}")

if use_lof:
    with st.spinner("📍 Running Local Outlier Factor..."):
        try:
            lof_score, lof_flags = run_lof(df_feat[feature_cols], contamination=contamination, n_neighbors=20)
            df_det["lof_score"]  = lof_score
            df_det["anom_lof"]   = lof_flags
        except Exception as e:
            st.warning(f"LOF failed: {e}")

flag_cols            = [c for c in ["anom_iforest", "anom_ocsvm", "anom_lof", "anom_zscore", "anom_iqr"]
                        if c in df_det.columns]
df_det["vote_sum"]       = df_det[flag_cols].sum(axis=1)
df_det["anom_consensus"] = (df_det["vote_sum"] >= consensus_k).astype(int)

# ---------------------------------------------------------
# Key numbers
# ---------------------------------------------------------
n_total   = len(df_det)
n_anomaly = int(df_det["anom_consensus"].sum())
pct       = n_anomaly / max(1, n_total) * 100
days_span = (date_max - date_min).days

# Ties on vote_sum are broken by Isolation Forest score (lower = more anomalous)
worst_idx  = df_det.sort_values(["vote_sum", "iforest_score"], ascending=[False, True]).index[0]
worst_day  = df_det.loc[worst_idx, date_col] if n_anomaly > 0 else None
worst_val  = df_det.loc[worst_idx, kpi]      if n_anomaly > 0 else None

# ---------------------------------------------------------
# Plain-English Summary
# ---------------------------------------------------------
if n_anomaly > 0 and worst_day is not None:
    worst_str = pd.to_datetime(worst_day).strftime("%B %d, %Y")
    summary_text = (
        f"Out of <strong>{n_total:,} trading days</strong> analysed, "
        f"<strong>{n_anomaly} days ({pct:.1f}%)</strong> showed unusual behaviour. "
        f"The most anomalous day was <strong>{worst_str}</strong>, "
        f"when <strong>{kpi}</strong> was <strong>{worst_val:,.2f}</strong> — "
        f"flagged by {int(df_det['vote_sum'].max())} out of {len(flag_cols)} detectors. "
        f"These are days where the price or volume moved in a way that doesn't match the surrounding pattern."
    )
else:
    summary_text = f"No anomalies found with the current settings. Try lowering the agreement threshold or increasing the contamination rate."

st.markdown(f"""
<div class="finding-card">
    <h3>🔍 What We Found</h3>
    <p>{summary_text}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Metric cards
# ---------------------------------------------------------
st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("📅 Days Analysed",    f"{n_total:,}",
          help="Total trading days in the selected date range")
m2.metric("🚨 Anomalies Found",  f"{n_anomaly}",
          delta=f"{pct:.1f}% of all days", delta_color="inverse",
          help="Days flagged as unusual by the consensus rule")
m3.metric("📆 Date Range",       f"{days_span:,} days",
          help="Total calendar days covered")
m4.metric("🤖 Active Detectors", f"{len(flag_cols)}",
          help="Number of detection methods currently running")

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main chart with annotations
# ---------------------------------------------------------
st.markdown('<p class="section-label">📉 Time Series — Red dots are anomalous days</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#6b7280;font-size:0.8rem;margin:-0.4rem 0 0.75rem 0;">Each red dot is a day where the model detected an unusual move. Hover over any point for details.</p>', unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_det[date_col], y=df_det[kpi],
    mode="lines", name=kpi,
    line=dict(color="#3b82f6", width=1.5),
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>" + kpi + ": %{y:.2f}<extra></extra>",
))

mask = df_det["anom_consensus"] == 1
if mask.any():
    hover_votes = df_det.loc[mask, "vote_sum"].astype(int)
    severity    = hover_votes.map({1: "Low", 2: "Medium", 3: "High", 4: "High", 5: "High"}).fillna("High")

    fig.add_trace(go.Scatter(
        x=df_det.loc[mask, date_col],
        y=df_det.loc[mask, kpi],
        mode="markers",
        name="⚠️ Anomaly",
        marker=dict(
            color=hover_votes.map({1: "#f59e0b", 2: "#ef4444", 3: "#dc2626", 4: "#b91c1c", 5: "#7f1d1d"}).fillna("#7f1d1d"),
            size=9, symbol="circle",
            line=dict(color="#fca5a5", width=1.5),
        ),
        customdata=np.stack([hover_votes, severity], axis=1),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            + kpi + ": <b>%{y:.2f}</b><br>"
            "Detectors flagged: <b>%{customdata[0]}</b><br>"
            "Severity: <b>%{customdata[1]}</b><extra></extra>"
        ),
    ))

# Annotate the single worst anomaly
if n_anomaly > 0 and worst_day is not None:
    fig.add_annotation(
        x=worst_day, y=worst_val,
        text=f"  Most anomalous<br>  {pd.to_datetime(worst_day).strftime('%b %d, %Y')}",
        showarrow=True, arrowhead=2,
        arrowcolor="#ef4444", arrowwidth=1.5,
        font=dict(color="#fca5a5", size=11),
        bgcolor="#450a0a", bordercolor="#991b1b", borderwidth=1,
        ax=60, ay=-50,
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(family="Inter", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False,
               title=dict(text="Date", font=dict(color="#6b7280", size=11))),
    yaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False,
               title=dict(text=kpi, font=dict(color="#6b7280", size=11))),
    legend=dict(bgcolor="#161b27", bordercolor="#1f2937", borderwidth=1, font=dict(size=11)),
    margin=dict(l=10, r=10, t=20, b=10),
    height=430,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# How it works (non-technical explainer, collapsible)
# ---------------------------------------------------------
with st.expander("🧠 How does anomaly detection work? (click to expand)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Step 1 · Build Context</h4>
            <p>For each day, we calculate what "normal" looks like — using the past 7 and 28 days of data,
            rolling averages, and day-of-week patterns. This gives each day a rich fingerprint.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Step 2 · Run 5 Detectors</h4>
            <p>Five different methods each independently decide if a day looks unusual.
            They use different logic — some statistical, some machine learning — so they catch different types of anomalies.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>Step 3 · Vote & Decide</h4>
            <p>A day is only flagged if <strong>at least {k} detectors agree</strong> it's unusual.
            This reduces false alarms — like a panel of experts that must reach consensus.</p>
        </div>
        """.format(k=consensus_k), unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.75rem; padding:0.9rem 1.1rem; background:#0f1117;
                border:1px solid #1f2937; border-radius:10px;">
        <p style="color:#6b7280; font-size:0.8rem; margin:0; line-height:1.8;">
            <strong style="color:#9ca3af;">🌲 Isolation Forest</strong> — Randomly splits the data; anomalies are isolated faster. &nbsp;|&nbsp;
            <strong style="color:#9ca3af;">📍 Local Outlier Factor</strong> — Compares each day to its neighbours. &nbsp;|&nbsp;
            <strong style="color:#9ca3af;">🔷 One-Class SVM</strong> — Learns the boundary of "normal" and flags anything outside. &nbsp;|&nbsp;
            <strong style="color:#9ca3af;">📐 Z-Score</strong> — Flags daily changes more than 3 standard deviations from the mean change. &nbsp;|&nbsp;
            <strong style="color:#9ca3af;">📦 IQR</strong> — Flags daily changes outside the typical range of changes.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Bottom row
# ---------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-label">Which detectors fired?</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.78rem;margin:-0.4rem 0 0.75rem 0;">Red = final consensus count. Higher bars = more days flagged by that method.</p>', unsafe_allow_html=True)

    counts     = df_det[flag_cols + ["anom_consensus"]].sum().sort_values(ascending=True)
    bar_colors = ["#ef4444" if c == "anom_consensus" else "#3b82f6" for c in counts.index]
    labels     = (counts.index
                  .str.replace("anom_", "")
                  .str.replace("_", " ")
                  .str.replace("iforest", "Isolation Forest")
                  .str.replace("ocsvm", "One-Class SVM")
                  .str.replace("lof", "Local Outlier Factor")
                  .str.replace("zscore", "Z-Score")
                  .str.replace("iqr", "IQR (Range)")
                  .str.replace("consensus", "✅ Consensus (Final)")
                  .str.title()
                  .str.replace("✅ Consensus (Final)", "✅ Consensus (Final)"))

    bar_fig = go.Figure(go.Bar(
        x=counts.values, y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#9ca3af", size=11),
    ))
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(family="Inter", color="#9ca3af", size=11),
        xaxis=dict(gridcolor="#1f2937", title=dict(text="Days Flagged", font=dict(size=10, color="#6b7280"))),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=50, t=10, b=10),
        height=270, showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with right:
    st.markdown('<p class="section-label">Top anomalous days</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.78rem;margin:-0.4rem 0 0.75rem 0;">Sorted by how many detectors agreed. More votes = more confident it\'s a real anomaly.</p>', unsafe_allow_html=True)

    top = (df_det[df_det["anom_consensus"] == 1]
           .sort_values("vote_sum", ascending=False)
           .head(20)
           .copy()
           .reset_index(drop=True))

    if not top.empty:
        top["Date"]     = pd.to_datetime(top[date_col]).dt.strftime("%b %d, %Y")
        top[kpi]        = top[kpi].round(2)
        top["Severity"] = top["vote_sum"].map(lambda v: "🔴 High" if v >= 3 else ("🟡 Medium" if v == 2 else "🔵 Low"))
        top["Votes"]    = top["vote_sum"].astype(int)

        display_cols = ["Date", kpi, "Votes", "Severity"]
        st.dataframe(top[display_cols], use_container_width=True, height=270, hide_index=True)
    else:
        st.info("No anomalies detected with current settings. Try lowering the agreement threshold.")

# ---------------------------------------------------------
# Download
# ---------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
anom_only = df_det[df_det["anom_consensus"] == 1].copy()
csv_bytes = anom_only.to_csv(index=False).encode("utf-8")

dl_col, tip_col = st.columns([1, 3])
with dl_col:
    st.download_button(
        label="⬇️  Download Anomalies CSV",
        data=csv_bytes,
        file_name=f"anomalies_{kpi}.csv",
        mime="text/csv",
    )
with tip_col:
    st.markdown(f"""
    <p style="color:#6b7280; font-size:0.8rem; margin-top:0.6rem;">
        The CSV includes the date, {kpi} value, anomaly scores from each detector,
        and the final consensus flag — ready for further analysis or reporting.
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
<p style="color:#374151; font-size:0.7rem; margin-top:1.5rem; padding-bottom:1rem;">
    KPI Anomaly Detector · Isolation Forest · One-Class SVM · LOF · Z-Score · IQR · Built with Streamlit
</p>
""", unsafe_allow_html=True)
