# Fraud Detection System

Final-year B.Tech CSE project implementing Phases 1-6 of an adaptive financial fraud-detection pipeline.

## Current implementation

1. **Phase 1:** PaySim dataset selection and validation
2. **Phase 2:** Exploratory Data Analysis
3. **Phase 3:** Past-only feature engineering and leakage audit
4. **Phase 4:** Chronological XGBoost Model V1 training and evaluation
5. **Phase 5:** FastAPI deployment of Model V1
6. **Phase 6:** Live transaction simulation with delayed ground truth

Kafka, Spark Structured Streaming, SHAP, counterfactual explanations, drift monitoring, Model V2, automated retraining, and a React dashboard are not implemented yet.

## Dataset and generated data

The project uses the PaySim synthetic financial dataset:

<https://www.kaggle.com/datasets/ealaxi/paysim1>

Download `PS_20174392719_1491204439457_log.csv` and place it at:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

The raw dataset is excluded from Git. Phase 3 generates `data/processed/featured_transactions.parquet`, which is also excluded from Git and must be regenerated locally when needed.

The dataset contains approximately 6.36 million rows, so a machine with at least 8 GB of free RAM is recommended.

## Setup

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pyarrow
```

The repository pins `pandas==2.2.3` because PySpark 4.2.0 still warns on pandas 3.x. The Spark Kafka connector is configured for the installed Spark 4.2.0 / Scala 2.13 runtime via the `spark.jars.packages` setting in `src/streaming/spark_streaming.py`.

The repository's existing local environment may also be used if it already contains the declared dependencies.

## Notebook execution order

Run the notebooks from the repository root in this order:

1. `02_exploratory_data_analysis.ipynb`
2. `03_feature_engineering.ipynb`
3. `04_xgboost_model_v1.ipynb`
4. `05_model_deployment.ipynb`
5. Start FastAPI.
6. `06_live_transaction_simulation.ipynb`

Phase 4 uses a chronological train/validation/test split and selects the classification threshold on validation data. Do not retrain Model V1 as part of Phase 5 or Phase 6.

## Run FastAPI

From the repository root:

```powershell
python -m uvicorn src.api.app:app --reload
```

The service is available at:

- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/docs>

The Phase 5 API loads Model V1 once and maintains development-only process-local account history.

## Phase 6 historical warm-up

Before the selected live evaluation window, Phase 6 resets the development API state and replays only the chronological prefix needed for accounts present in the live window. Warm-up transactions reconstruct prediction-time account state; they are not counted as live predictions and their labels are never sent to the API.

The live request excludes `isFraud`. Ground truth is stored separately and becomes available using a logical delay timestamp.

## Evaluation interpretation

Phase 4's full chronological test-set metrics are the primary Model V1 performance evidence.

Phase 6 retains a selected 100-transaction fraud-containing demonstration so the delayed-label workflow can be shown with both classes. Its metrics describe that selected live demonstration and should not be interpreted as an unbiased estimate of overall Model V1 generalization.

## Outputs

Reports are written under `reports/`, including feature metadata, Model V1 metrics, test predictions, live prediction logs, delayed ground truth, merged evaluation data, and Phase 6 figures. The raw dataset and processed parquet remain local-only artifacts.
