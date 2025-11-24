
# config.py — YAML/JSON configuration loader

from pathlib import Path
from typing import Any, Dict
import json

try:
    import yaml 
except Exception:
    yaml = None

def _merge(a: dict, b: dict) -> dict:
    """Recursively merge two dictionaries."""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML or JSON config. Return defaults if missing."""
    default = {
        "data": {
            "input_dir": "data/raw",
            "processed": "data/processed/clean_data.csv",
        },
        "models": {
            "ml": {"contamination": 0.02, "n_estimators": 300, "random_state": 42}
        },
    }
    if not path:
        return default

    p = Path(path)
    if not p.exists():
        return default

    if p.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        return _merge(default, yaml.safe_load(p.read_text(encoding="utf-8")))
    if p.suffix.lower() == ".json":
        return _merge(default, json.loads(p.read_text(encoding="utf-8")))
    return default