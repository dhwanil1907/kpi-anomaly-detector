from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def line_with_anomalies(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "sales",
    flag_col: str = "anom_iforest",
    title: str | None = None,
):
    """
    Return a Plotly Figure with a line and anomaly markers (if flag_col exists).
    """
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
    fig.update_layout(
        title=title or f"{value_col} with anomalies ({flag_col})",
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols=None):
    """
    Return a Matplotlib/Seaborn heatmap Axes for numeric correlations.
    """
    if cols is None:
        cols = df.select_dtypes("number").columns
    corr = df[cols].corr()
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    ax.set_title("Correlation Heatmap")
    return ax