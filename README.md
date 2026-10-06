# KPI Anomaly Detector

An end-to-end **unsupervised anomaly detection system** for financial time-series data, built on real AAPL stock data (2019–present). Combines 5 ML/statistical detectors with SHAP explainability, stationarity-aware feature engineering, and validation against 15 documented real-world market events.

Includes an interactive Streamlit dashboard and a reusable `src/` Python package.

---

## Results

| Metric | Value |
|---|---|
| Dataset | AAPL · 1,950 trading days · Jan 2019 – Oct 2026 |
| ADF p-value (raw Close) | 0.9476 → ❌ Non-stationary |
| ADF p-value (log returns) | 0.0000 → ✅ Stationary |
| Consensus anomalies detected | 20 / 1,922 days (1.04%) |
| Precision@5 | 60% (3/5 flagged days matched known events) |
| Precision@10 | 50% (5/10 flagged days matched known events) |
| Recall@10 | 33.3% (5 of 15 known events caught in top 10) |
| SHAP top feature | `Close_pct_change_1` (importance = 0.236) |
| Model stability agreement | 67% between first/second half splits |

Known events caught in top 10: COVID crash (Feb 28, Mar 2, Mar 13 2020), US-China trade war spike (May 13 2019), COVID recovery bounce (Jul 31 2020).

## Highlights

- **Precision@5 of 60%** — 3 of the top 5 most anomalous days matched documented real-world market events, with zero labeled training data
- **SHAP explainability** — every anomaly comes with per-feature attribution: *why* was this day flagged?
- **Statistically validated** — ADF test confirmed raw Close is non-stationary (p=0.9476); log returns are stationary (p=0.0000)
- **5-detector ensemble** with consensus voting — 20 high-confidence anomalies out of 1,922 trading days
- **Live data** — pulls from Yahoo Finance via `yfinance`, always up to date

---

## Detection Models

| Model | Type | Notes |
|---|---|---|
| Isolation Forest | ML | Randomly partitions features; anomalies isolate faster |
| One-Class SVM | ML | Learns boundary of normal; flags anything outside |
| Local Outlier Factor | ML | Compares local density to neighbours |
| Z-Score | Statistical | Flags values > 3σ from mean |
| IQR | Statistical | Flags values outside [Q1−1.5×IQR, Q3+1.5×IQR] |

---

## Features

### Feature Engineering Pipeline
- **Log returns** (`ln(Pₜ/Pₜ₋₁)`) — stationary transformation of raw price
- **Log volume** — reduces skew from extreme trading volume spikes
- **Lag features** — 7-day and 28-day price memory
- **Rolling statistics** — 7-day mean and std (local trend + volatility)
- **Momentum** — 1-day and 7-day percent change
- **Calendar features** — day-of-week, month, weekend flag

### Model Validation
- **ADF (Augmented Dickey-Fuller) test** — confirms raw price is non-stationary; log returns pass stationarity check
- **Precision@K** — of top-K flagged days, what % match a known market event?
- **Recall@K** — of all known events in the data range, what % did the model catch?
- **15 ground-truth events** including COVID crash (Mar 2020), Fed rate hikes (2022), Yen carry trade unwind (Aug 2024)

### SHAP Explainability
- Global feature importance (mean |SHAP| per feature)
- Per-anomaly explanation: which features drove the flag and in which direction
- SHAP scatter plot: feature value vs SHAP contribution across all days

### Streamlit Dashboard
- Live data from Yahoo Finance — no CSV needed
- Select any numeric KPI column to analyse
- Severity labels: 🔴 High / 🟡 Medium / 🔵 Low
- Plain-English summary card (*"31 anomalous days found — most anomalous: March 16, 2020"*)
- Annotated time-series chart with hover tooltips
- Detector agreement bar chart
- Download anomalies as CSV

---

## Project Structure

```
kpi-anomaly-detector/
│
├── app.py                      # Streamlit dashboard
│
├── src/
│   ├── __init__.py             # Public package API
│   ├── io.py                   # yfinance loader + CSV helpers
│   ├── features.py             # Log returns, lags, rolling, calendar features
│   ├── detect.py               # IsolationForest, OCSVM, LOF, Z-Score, IQR
│   ├── evaluate.py             # Precision@K, Recall@K, known market events
│   ├── explain.py              # SHAP feature importance + anomaly explanations
│   ├── viz.py                  # Plotly time-series + bar chart utilities
│   └── config.py               # YAML/JSON config loader with safe defaults
│
├── notebook/
│   └── kpi.ipynb               # Full EDA → features → detection → SHAP → validation
│
├── configs/
│   └── default.yaml            # Ticker, date range, model hyperparameters
│
├── data/
│   └── processed/              # clean_data.csv, anomalies.csv (git-ignored)
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/dhwanil1907/kpi-anomaly-detector
cd kpi-anomaly-detector

python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Running the Notebook

```bash
cd notebook
jupyter notebook kpi.ipynb
```

Run cells top to bottom. The notebook covers:
1. Environment setup + config
2. Data load from Yahoo Finance (AAPL, 2019–today)
3. ADF stationarity test + log return transformation
4. Feature engineering
5. Anomaly detection (5 models)
6. Detector comparison + consensus voting
7. Model validation: Precision@K and Recall@K vs known events
8. SHAP explainability: feature importance + per-day explanations
9. Save processed outputs

---

## How It Works

### 1. Data Ingestion
Pulls OHLCV data directly from Yahoo Finance via `yfinance`. No manual CSV download needed.

### 2. Stationarity Check
ADF test confirms raw `Close` price is non-stationary (p > 0.05). Log returns are computed and confirmed stationary (p < 0.05) before being fed to ML models.

### 3. Feature Engineering
Each day gets a rich feature vector: log return, log volume, lagged prices, rolling mean/std, momentum, and calendar context — giving models the temporal pattern needed to define "normal".

### 4. Detection
Five independent methods each produce a binary anomaly flag. A day is only marked anomalous if **≥ k detectors agree** (configurable: 1 = sensitive, 2 = balanced, 3 = strict).

### 5. Validation
Top-K flagged days are compared against a curated list of 15 known market events. Precision@K and Recall@K quantify performance without requiring labeled training data.

### 6. Explainability
SHAP values attribute each anomaly flag to specific features — e.g., *"the March 2020 flag was driven by a −12% log return (SHAP +0.08) and 3× normal volume (SHAP +0.06)"*.

---

## Configuration

Edit `configs/default.yaml` to change ticker, date range, or model settings:

```yaml
data:
  ticker: "AAPL"
  start: "2019-01-01"

preprocess:
  target: "Close"

models:
  ml:
    contamination: 0.02
    n_estimators: 300
```

---

## Tech Stack

`Python` · `scikit-learn` · `yfinance` · `SHAP` · `statsmodels` · `Streamlit` · `Plotly` · `pandas` · `numpy`

---

## Author

**Dhwanil Ranpura**
Machine Learning · Data Science · Python Development
