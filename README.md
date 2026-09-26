# Adaptive Real-Time Financial Transaction Fraud Detection
**XGBoost · Kafka · Spark Structured Streaming · Explainable AI · Drift Detection · Automated Retraining**

Final-year B.Tech CSE project.

## Current phase
**Phase 1 – Dataset selection & understanding** and **Phase 2 – Complete Exploratory Data Analysis.**
No model, streaming component, API or UI exists yet. Later phases are intentionally not implemented.

## Dataset
PaySim – Synthetic Financial Dataset for Fraud Detection
https://www.kaggle.com/datasets/ealaxi/paysim1
File expected at: `data/raw/PS_20174392719_1491204439457_log.csv`
(Path is configurable: edit `DATA_PATH` in the notebook or set the environment variable `PAYSIM_DATA_PATH`.)

## Folder structure
```
fraud-detection-system/
├── data/
│   ├── raw/          <- put the PaySim CSV here (never modified)
│   └── processed/    <- empty until Phase 3
├── notebooks/
│   └── 02_exploratory_data_analysis.ipynb
├── reports/
│   ├── figures/      <- PNG figures written by the notebook
│   ├── tables/       <- CSV tables written by the notebook
│   └── eda_summary.json
├── models/           <- empty until Phase 4
├── src/              <- empty until Phase 3
├── requirements.txt
└── README.md
```

## Setup
```bash
cd fraud-detection-system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pyarrow                # optional, lowers memory use
```
Recommended: 8 GB free RAM (the CSV has ~6.36 million rows).

## Run the EDA
1. Download the dataset from Kaggle, unzip it, and place the CSV in `data/raw/`.
2. Start Jupyter from the project root: `jupyter notebook`
3. Open `notebooks/02_exploratory_data_analysis.ipynb`.
4. Run **Kernel → Restart & Run All**. (Expect a few minutes on a laptop.)

## Outputs
* `reports/tables/*.csv` – summary, data quality, fraud distribution, type/amount/time/account/balance analyses, correlation, outliers, cardinality, leakage audit, prediction-time availability, feature candidates, temporal split plan, etc.
* `reports/figures/*.png` – 16 figures (`01_…` to `16_…`).
* `reports/eda_summary.json` – machine-readable summary (all values computed from the CSV).

## What Phase 2 accomplishes
Understands the data (quality, imbalance, types, amounts, time, accounts, balances, outliers, correlations), audits **leakage** and **prediction-time availability**, proposes a chronological split, lists candidate features, and sketches the streaming-replay and drift-monitoring plans — without training anything.

## What Phase 3 will do next
Preprocessing and feature engineering using only prediction-time-safe information (past-only behavioural features), a chronological train/validation/test split, and the resolution of the open decisions from this EDA (e.g. `isFlaggedFraud`, balance fields). XGBoost V1 is trained after that.
