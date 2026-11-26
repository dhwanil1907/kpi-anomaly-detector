# explain.py — SHAP-based model explainability

import numpy as np
import pandas as pd


def get_shap_values(model, X: np.ndarray, feature_names: list):
    """
    Compute SHAP values for an IsolationForest model.
    Returns (shap_values array, explainer).
    """
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def shap_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Global feature importance: mean absolute SHAP value per feature.
    Returns DataFrame sorted by importance descending.
    """
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_top_anomalies(shap_values, df_det, feature_cols, date_col="date", target="Close", n=5):
    """
    Print a human-readable SHAP explanation for the top-N most anomalous rows.
    """
    # Ties on vote_sum are broken by Isolation Forest score (lower = more anomalous)
    by = ["vote_sum"] + (["iforest_score"] if "iforest_score" in df_det.columns else [])
    top_idx = df_det.sort_values(by, ascending=[False] + [True] * (len(by) - 1)).head(n).index.tolist()
    print("=" * 60)
    print(f"  SHAP Explanation — Top {n} Most Anomalous Days")
    print("=" * 60)
    for rank, idx in enumerate(top_idx, 1):
        date_str = pd.to_datetime(df_det.loc[idx, date_col]).strftime("%Y-%m-%d")
        val      = df_det.loc[idx, target]
        votes    = int(df_det.loc[idx, "vote_sum"])
        row_shap = dict(zip(feature_cols, shap_values[idx]))
        top3     = sorted(row_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"\n  #{rank}  {date_str}  {target}={val:.2f}  votes={votes}")
        for feat, sv in top3:
            # IsolationForest SHAP explains the anomaly score, where lower = more anomalous
            direction = "→ anomaly" if sv < 0 else "→ normal"
            print(f"      {feat:<30s}  SHAP={sv:+.4f}  {direction}")
    print("=" * 60)
