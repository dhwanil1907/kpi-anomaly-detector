from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def summarize_flags(df_flags: pd.DataFrame, flag_cols=None) -> Dict[str, float]:
    """
    Quick stats: count and percent for each anomaly flag column.
    """
    if flag_cols is None:
        flag_cols = [c for c in df_flags.columns if c.startswith("anom_")]
    total = len(df_flags)
    out = {}
    for c in flag_cols:
        n = int(df_flags[c].sum())
        out[f"{c}_count"] = n
        out[f"{c}_pct"] = round(100.0 * n / total, 2) if total else 0.0
    return out


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute precision/recall/F1 for binary labels (1=anomaly).
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}