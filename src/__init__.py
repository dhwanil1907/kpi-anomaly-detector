# src package initializer: re-exports the loaders, features, detectors, evaluation and plots

from .config import load_config
from .io import find_data_dir, load_processed, read_csv_auto, load_yfinance
from .features import add_time_features, add_lag_roll, build_features, feature_columns
from .detect import zscore_flags, iqr_flags, run_iforest, run_ocsvm, run_lof, vote_consensus
from .viz import plot_series_with_anoms, counts_bar
from .evaluate import precision_at_k, recall_at_k, summarize_validation, KNOWN_EVENTS
from .explain import get_shap_values, shap_feature_importance, explain_top_anomalies

__all__ = [
    "load_config",
    "find_data_dir", "load_processed", "read_csv_auto", "load_yfinance",
    "add_time_features", "add_lag_roll", "build_features", "feature_columns",
    "zscore_flags", "iqr_flags", "run_iforest", "run_ocsvm", "run_lof", "vote_consensus",
    "plot_series_with_anoms", "counts_bar",
    "precision_at_k", "recall_at_k", "summarize_validation", "KNOWN_EVENTS",
    "get_shap_values", "shap_feature_importance", "explain_top_anomalies",
]