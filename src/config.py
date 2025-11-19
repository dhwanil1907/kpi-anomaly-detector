from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

# Optional: YAML support
try:
    import yaml  # for .yaml/.yml configs
except Exception:
    yaml = None


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load a YAML or JSON config file.
    If no path is provided or file missing, return default settings.
    """
    default = {
        "data": {
            "input_dir": "../data/raw",
            "processed": "../data/processed/clean_data.csv",
        },
        "preprocess": {
            "grain": "global",
            "freq": "D",
            "rolling_windows": [7, 28],
        },
        "features": {
            "target": "sales",
            "lags": [7, 28],
            "rolling": [7, 28],
            "pct_change": [1, 7],
        },
        "models": {
            "statistical": {"zscore_threshold": 3.0, "iqr_k": 1.5},
            "ml": {
                "method": "isolation_forest",
                "contamination": 0.02,
                "n_estimators": 300,
                "random_state": 42,
            },
            "forecast": {
                "method": "prophet",
                "sigma_k": 3.0,
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "seasonality_mode": "additive",
            },
        },
        "evaluate": {
            "metrics": ["precision", "recall", "f1"],
            "save_report": True,
            "output_path": "../data/processed/evaluation_report.json",
        },
        "visualize": {
            "default_kpi": "sales",
            "flag_priority": "anom_iforest",
            "color_theme": "plotly",
            "save_plots": "../plots/",
        },
        "runtime": {
            "random_seed": 42,
            "log_level": "INFO",
            "use_synthetic_if_missing": True,
        },
    }

    # --- no config file provided ---
    if not path:
        return default

    path = Path(path)
    if not path.exists():
        print(f"⚠️ Config file not found at {path}, using defaults.")
        return default

    # --- YAML (.yaml/.yml) ---
    if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            cfg_file = yaml.safe_load(f)
    # --- JSON (.json) ---
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg_file = json.load(f)
    else:
        print(f"⚠️ Unsupported config format for {path}. Using defaults.")
        return default

    # --- Merge file config over defaults ---
    def merge_dicts(a: dict, b: dict) -> dict:
        merged = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    return merge_dicts(default, cfg_file or {})


def ensure_dirs(*paths: str | Path) -> None:
    """
    Create directories if they don't exist.
    Useful for saving processed data or model artifacts.
    """
    for p in paths:
        P = Path(p)
        target = P if P.suffix == "" else P.parent
        target.mkdir(parents=True, exist_ok=True)