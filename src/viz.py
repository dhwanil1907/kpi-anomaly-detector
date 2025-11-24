# viz.py — Visualization utilities (Plotly)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_series_with_anoms(df: pd.DataFrame, date_col: str, value_col: str, flag_col: str, title: str):
    """Plot time series with anomaly markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode="lines", name=value_col))
    if flag_col in df.columns:
        mask = df[flag_col] == 1
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, date_col],
                y=df.loc[mask, value_col],
                mode="markers",
                name="Anomaly",
                marker=dict(size=8, symbol="circle-open"),
            )
        )
    fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col)
    return fig

def counts_bar(counts: pd.Series):
    """Create a labeled bar chart of anomaly counts per detector."""
    df = counts.reset_index()
    df.columns = ["Detector", "Anomalies"]
    fig = px.bar(df, x="Detector", y="Anomalies", text="Anomalies", title="Anomaly Counts by Detector")
    fig.update_traces(textposition="outside")
    return fig