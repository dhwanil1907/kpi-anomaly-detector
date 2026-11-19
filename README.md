# KPI Anomaly Detector

Unsupervised anomaly detection on AAPL daily data (2019 to present). Five detectors (Isolation Forest, One-Class SVM, LOF, z-score, IQR) vote on which days look unusual. SHAP explains the Isolation Forest flags, and the top-ranked days are checked against a hand-written list of 15 market events. There is a Streamlit dashboard and a small `src/` package.

---

## Results

| Metric | Value |
|---|---|
| Dataset | AAPL, 1,950 trading days, Jan 2019 to Oct 2026 |
| ADF p-value, raw Close | 0.9476 (non-stationary) |
| ADF p-value, log returns | 0.0000 (stationary) |
| Consensus anomalies | 20 of 1,922 days (1.04%) |
| Precision@5 | 60% (3 of 5) |
| Precision@10 | 50% (5 of 10) |
| Recall@10 | 33.3% (5 of 15 events) |
| Top SHAP feature | `Close_pct_change_1` (mean abs SHAP 0.236) |
| Stability | 67% agreement between first-half and second-half fits |

Matches are counted within +/- 1 day of an event date. See [Limitations](#limitations) before reading much into these.

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
- Severity labels (High / Medium / Low) based on how many detectors agree
- Summary card with the anomaly count and the most anomalous day
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
Pulls OHLCV data from Yahoo Finance via `yfinance`.

### 2. Stationarity Check
ADF test confirms raw `Close` price is non-stationary (p > 0.05). Log returns are computed and confirmed stationary (p < 0.05) before being fed to ML models.

### 3. Feature Engineering
Each day gets a feature vector: log return, log volume, lagged prices, rolling mean/std, momentum, and calendar fields.

### 4. Detection
Five independent methods each produce a binary anomaly flag. A day is only marked anomalous if **≥ k detectors agree** (configurable: 1 = sensitive, 2 = balanced, 3 = strict).

### 5. Validation
Top-K flagged days are compared against the 15 dates in `KNOWN_EVENTS` (`src/evaluate.py`). There are no labels, so this is the only check on whether the flags mean anything.

### 6. Explainability
SHAP values from the Isolation Forest attribute each flag to features. `explain_top_anomalies` prints the top three drivers for the most anomalous days.

---

## Configuration

`configs/default.yaml` holds the ticker, start date and Isolation Forest settings, and `src.config.load_config()` reads it. The Streamlit app does not load it yet, so the dashboard uses its own sidebar controls.

---

## Limitations

- **Hand-picked ground truth.** `KNOWN_EVENTS` has 15 dates, and 6 of them are in March 2020. A model that only catches the COVID crash scores well on precision.
- **Small sample.** Recall@10 of 33% is 5 events out of 15, so one more or fewer hit moves it a lot.
- **Close-to-event matching.** A flag counts if it falls within a day of an event, which makes matches easier than an exact-date test.
- **SHAP covers one model.** The explanations are for the Isolation Forest only, not for the consensus vote.
- **Fixed contamination.** The ML detectors assume about 2% of days are anomalous, so the count of flags is partly set by that choice.

---

## Tech Stack

`Python` · `scikit-learn` · `yfinance` · `SHAP` · `statsmodels` · `Streamlit` · `Plotly` · `pandas` · `numpy`

---

## Author

**Dhwanil Ranpura**
Machine Learning · Data Science · Python Development
