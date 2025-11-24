# kpi_anomaly package initializer

# Exports all key modules for convenient importing

from .config import load_config
from .io import find_data_dir, load_processed, read_csv_auto
from .features import add_time_features, add_lag_roll, build_features
from .detect import zscore_flags, iqr_flags, run_iforest, run_ocsvm, run_lof, vote_consensus
from .viz import plot_series_with_anoms, counts_bar

__all__ = [
    "load_config",
    "find_data_dir", "load_processed", "read_csv_auto",
    "add_time_features", "add_lag_roll", "build_features",
    "zscore_flags", "iqr_flags", "run_iforest", "run_ocsvm", "run_lof", "vote_consensus",
    "plot_series_with_anoms", "counts_bar",
]