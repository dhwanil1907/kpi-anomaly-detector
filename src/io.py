# io.py — data input/output utilities

from pathlib import Path
import pandas as pd

def find_data_dir(start: Path | None = None) -> Path | None:
    """Search upward from current directory for a folder named 'data'."""
    start = start or Path.cwd()
    for base in [start, start.parent, start.parent.parent]:
        if (base / "data").exists():
            return base / "data"
    return None

def load_processed(base: Path | None = None):
    """Load processed clean and anomaly datasets."""
    data_dir = base or find_data_dir()
    if data_dir is None:
        raise FileNotFoundError("Could not locate 'data' directory.")
    clean = pd.read_csv(data_dir / "processed" / "clean_data.csv", parse_dates=["date"])
    anom = pd.read_csv(data_dir / "processed" / "anomalies.csv", parse_dates=["date"])
    return clean, anom

def read_csv_auto(file_or_path) -> pd.DataFrame:
    """Read a CSV and automatically convert date/time columns to datetime."""
    df = pd.read_csv(file_or_path)
    for cand in ["date", "ds", "timestamp", "time", "Date", "DATE"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
    return df