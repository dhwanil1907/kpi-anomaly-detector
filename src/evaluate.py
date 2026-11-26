# evaluate.py — Validation against known market events

import pandas as pd
import numpy as np

KNOWN_EVENTS = {
    "2020-02-24": "COVID fear begins — S&P worst day since 2018",
    "2020-02-27": "COVID crash accelerates",
    "2020-03-09": "Oil price war + COVID — circuit breaker triggered",
    "2020-03-12": "COVID pandemic declared — market panic",
    "2020-03-16": "Worst single day of COVID crash",
    "2020-03-18": "COVID volatility spike",
    "2021-01-27": "Reddit short squeeze mania",
    "2022-01-24": "Fed rate hike fears — AAPL flash crash intraday",
    "2022-09-13": "CPI inflation shock — worst day since 2020",
    "2022-09-23": "Fed 75bps hike — market selloff",
    "2024-08-05": "Yen carry trade unwind — global selloff",
    "2024-08-02": "Recession fears + Buffett sells AAPL",
    "2020-03-02": "COVID first big bounce — extreme volatility",
    "2020-03-13": "COVID partial recovery — extreme volatility",
    "2023-03-13": "SVB bank collapse fear",
}

def _top_k(df_det, date_col, score_col, k, tiebreak_col="iforest_score"):
    """
    Top-K rows by score. Vote counts are small integers, so ties are common;
    break them by the detector score (lower = more anomalous) instead of row order,
    which would favour the earliest dates.
    """
    has_tiebreak = tiebreak_col in df_det.columns
    df = df_det[[date_col, score_col] + ([tiebreak_col] if has_tiebreak else [])].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    by = [score_col] + ([tiebreak_col] if has_tiebreak else []) + [date_col]
    ascending = [False] + ([True] if has_tiebreak else []) + [True]
    return df.sort_values(by, ascending=ascending).head(k).reset_index(drop=True)

def precision_at_k(df_det, date_col, score_col="vote_sum", k=10, window_days=1):
    """
    Precision@K: of the top-K most anomalous days,
    what fraction fall within window_days of a known market event?
    Returns (precision_float, detail_dataframe).
    """
    event_dates = pd.to_datetime(list(KNOWN_EVENTS.keys()))
    top_k = _top_k(df_det, date_col, score_col, k)

    def is_near_event(d):
        return any(abs((d - e).days) <= window_days for e in event_dates)

    def get_label(d):
        # Closest event within the window, so a day is not labelled with a neighbouring event
        gap, label = min(
            ((abs((d - pd.Timestamp(k2)).days), v) for k2, v in KNOWN_EVENTS.items()),
            key=lambda pair: pair[0],
        )
        return label if gap <= window_days else "—"

    top_k["is_known_event"] = top_k[date_col].apply(is_near_event)
    top_k["event_label"]    = top_k[date_col].apply(get_label)
    return top_k["is_known_event"].mean(), top_k

def recall_at_k(df_det, date_col, score_col="vote_sum", k=10, window_days=1):
    """
    Recall@K: of all known events in our data range,
    what fraction did we catch in the top-K flags?
    """
    event_dates = pd.to_datetime(list(KNOWN_EVENTS.keys()))
    dates = pd.to_datetime(df_det[date_col])
    data_min, data_max = dates.min(), dates.max()
    events_in_range = [e for e in event_dates if data_min <= e <= data_max]
    if not events_in_range:
        return 0.0
    top_k_dates = _top_k(df_det, date_col, score_col, k)[date_col]
    caught = sum(
        any(abs((t - e).days) <= window_days for t in top_k_dates)
        for e in events_in_range
    )
    return caught / len(events_in_range)

def summarize_validation(df_det, date_col, score_col="vote_sum", ks=(5, 10, 15, 20)):
    """Print full validation table for multiple K values."""
    print("=" * 58)
    print("  VALIDATION — Precision & Recall vs Known Market Events")
    print("=" * 58)
    print(f"  {'K':>4}   {'Precision@K':>12}   {'Recall@K':>10}   {'Caught':>6}")
    print("-" * 58)
    detail = None
    for k in ks:
        p, d = precision_at_k(df_det, date_col, score_col, k=k, window_days=1)
        r    = recall_at_k(df_det, date_col, score_col, k=k, window_days=1)
        n    = d["is_known_event"].sum()
        print(f"  {k:>4}   {p:>11.1%}   {r:>9.1%}   {n:>4}/{k}")
        if k == 10:
            detail = d
    print("=" * 58)
    return detail
