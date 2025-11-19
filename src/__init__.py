"""
kpi_anomaly: utilities for loading, featurizing, detecting anomalies,
evaluating, and visualizing KPI time series.
"""

from .config import load_config, ensure_dirs
from .io import (
    load_local_store_sales,
    make_synthetic,
    save_processed,
    load_processed,
)
from .features import add_time_features, add_lag_roll, build_features
from .detect import (
    zscore_flags,
    iqr_flags,
    iforest_train,
    iforest_infer,
    prophet_residual_flags,
    save_model,
    load_model,
)
from .evaluate import summarize_flags, precision_recall_f1
from .viz import line_with_anomalies, correlation_heatmap