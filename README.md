# KPI Anomaly Detector

A modular Python project for detecting anomalies in business KPIs (sales, energy usage, web traffic, etc.) using unsupervised machine learning and statistical methods.  
Includes a fully interactive Streamlit dashboard and a reusable `src/kpi_anomaly` Python package.

---

## Features

### Anomaly Detection Models
- Isolation Forest  
- One-Class SVM  
- Local Outlier Factor (LOF)  
- Z-Score Outlier Detection  
- IQR Outlier Method  

### Dashboard (Streamlit)
- Upload any CSV  
- Auto-detect date/timestamp column  
- Automatic feature engineering (lags, rolling statistics, percent change, calendar features)  
- Compare multiple detectors  
- Visualize anomalies on a time-series  
- Download anomalies as CSV  

### Modular Python Package (`src/kpi_anomaly`)
- `features.py`: Builds lag/rolling/calendar features  
- `detect.py`: ML detectors + statistical baselines  
- `viz.py`: Plotly visualization utilities  
- `io.py`: Data loading/helpers  
- `config.py`: YAML/JSON configuration loader  

### Notebook for Exploration
- `notebooks/01_eda.ipynb` shows:
  - EDA  
  - Feature engineering  
  - Model testing  
  - Detector comparison  
  - Stability analysis  

---

## Project Structure

kpi-anomaly-detector/
│
├── app.py                     # Streamlit dashboard
│
├── src/
│   └── kpi_anomaly/
│       ├── init.py
│       ├── io.py
│       ├── features.py
│       ├── detect.py
│       ├── viz.py
│       └── config.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── data/
│   ├── raw/                   # local datasets (ignored by Git)
│   └── processed/             # clean_data.csv, anomalies.csv
│
├── configs/
│   └── default.yaml
│
├── requirements.txt
├── .gitattributes             # prevents notebooks from affecting language stats
├── .gitignore
└── README.md

---

## Installation

git clone 
cd kpi-anomaly-detector

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .

---

## Running the Streamlit App

streamlit run app.py

Open the URL displayed in the console, usually:

http://localhost:8501

---

## Uploading a CSV (Important)

Your CSV should contain:
- A **date** or **timestamp** column  
- At least **one numeric column** (KPI)

Compatible datasets include:
- PJM Electricity data (`PJME_hourly.csv`, etc.)  
- Stock data (AAPL/MSFT from Yahoo Finance)  
- Web traffic logs  
- Sales or revenue data  
- Any KPI time series  

The app will:
1. Detect the date column  
2. Automatically engineer features  
3. Run all detectors  
4. Highlight anomalies  
5. Provide CSV export  

---

## How It Works

### 1. Feature Engineering
Generated automatically:
- Lag features  
- Rolling windows  
- Percent change  
- Day-of-week  
- Month  
- Weekend indicator  

### 2. Detection Models
Each model outputs:
- A score  
- A binary anomaly flag  

### 3. Consensus Logic
You choose how many detectors must agree:
- 1 → Sensitive  
- 2 → Balanced  
- 3 → Strict  

---

## Why This Project Matters

- Demonstrates real-world unsupervised machine learning  
- Streamlit UI provides business-friendly insights  
- Reusable Python package for anomaly detection  
- Clean architecture for production-style work  
- Strong portfolio project showcasing:
  - Python package design  
  - Data engineering  
  - ML modeling  
  - Visualization  
  - Software structure  

---

## Author

**Dhwanil**  
Machine Learning · Data Engineering · Python Development  
(Add your GitHub & LinkedIn links here)


⸻

If you want, I can also:
	•	Create a cover image/banner for your GitHub repo
	•	Generate a shorter README
	•	Add badges (Python, Streamlit, license, etc.)

Just tell me!
