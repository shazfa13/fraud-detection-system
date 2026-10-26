# Fraud Intelligence Control Center

This optional Streamlit application is a read-only project viewer except for the explicit single-transaction action on **Transaction Analyzer**. It does not start or restart Kafka, Spark, or FastAPI; it does not train, approve, promote, or roll back models.

## Start the UI

From the project root in PowerShell:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-ui.txt
streamlit run dashboard.py
```

The API defaults to `http://127.0.0.1:8000`. Set `FRAUD_API_BASE_URL` in the dashboard terminal if the existing API uses a different address. Sidebar navigation includes Overview, Transaction Explorer, Live Monitoring, Model Performance, System Health, Transaction Analyzer, Delayed Labels, Drift & Alerts, Retraining Lab, System Architecture, and Reports & Documentation.

## Actual data sources

- **Overview counts:** Stage 2 SQLite `predictions` rows, grouped by stored model prediction and with the registry's recorded time range. These are predictions, not verified outcomes. If the DB is missing/unreadable, counts are shown as unavailable.
- **Transaction Explorer:** bounded recent rows read from the configured API log, known versioned API logs, and saved streaming CSV outputs. Source names are retained; potential overlaps are not silently deduplicated.
- **Live Monitoring:** saved Stage 1 heartbeat, prediction/error JSONL tails, Stage 2 registry count, and historical Phase 12 evidence. The page does not test broker connectivity or claim a running heartbeat means new events are flowing. Heartbeat is considered fresh for 30 seconds.
- **Model Performance:** saved `reports/model_v1_vs_v2_comparison.json`, model metadata, thresholds, and feature importance CSV. These are historical synthetic PaySim evaluation results.
- **Drift & Alerts:** the most recent persisted Stage 3 report through the monitor's read-only SQLite accessor. A report older than 24 hours is marked stale. The existing aggregate report does not store per-feature PSI and adjusted p-values, so the page does not fabricate them.
- **Delayed Labels:** Stage 2's read-only SQLite statistics and grouped label/submission provenance. No credential values are displayed; synthetic labels remain separate.
- **Retraining Lab:** Stage 4's existing read-only eligibility collector, the shared active-model resolver, and the latest saved Stage 6 demonstration report. Synthetic demo candidate metrics are labeled ineligible for genuine promotion.
- **System Architecture:** native UI diagram and Phase 14 workflow/report references.
- **Reports & Documentation:** only existing allowlisted files are offered, after path and size checks.

## Transaction Analyzer behavior

The Analyzer collects exactly the seven raw fields accepted by the FastAPI request schema: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, and `oldbalanceDest`. The backend's `/state/preview` computes and returns the 34-feature model vector in metadata order. It never submits `isFraud` or `event_id` as model inputs.

Choose **Preview only** to call `/state/preview`; it does not consume account history. Choose **Submit prediction** to call the existing `/predict` endpoint. That explicit action mutates the API process's account state and appends to its configured prediction log. Use an isolated API log path when doing a manual demo if preserving the existing API log is important. The response contains a model prediction, probability, score, risk level, and model version—not ground truth.

## Safety and limitations

- SQLite viewers use read-only connections and do not initialize schemas.
- The shared active-model resolver provides model version, threshold, and ordered features. The Streamlit app does not load its own estimator.
- Saved status/report data may be old or absent. Missing data is labeled unavailable; historical evidence is not called live service health.
- The current project is an academic demonstration on synthetic PaySim data. Genuine label collection and production-population validation remain separate requirements.
- The UI dependencies stay in `requirements-ui.txt`; backend dependencies are not changed by the dashboard.
