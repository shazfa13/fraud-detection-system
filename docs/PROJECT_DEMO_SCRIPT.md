# Project demonstration script

This walkthrough uses the current role-based Streamlit navigation and saved evidence. Start FastAPI only if you intend to exercise transaction preview/prediction. Kafka/Spark and monitoring are optional, separate services. A prediction, producer run, label submission, or monitor invocation persists state; use preview and saved reports for a non-mutating talk.

## Before the presentation

From the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

The UI requires `requirements-ui.txt`. Configure an admin username/hash as documented in [DASHBOARD_README.md](../DASHBOARD_README.md), then select Admin sign-in. To run FastAPI separately (needed by Transaction Risk Check and Analyzer):

```powershell
$env:FRAUD_ACTIVE_MODEL_VERSION = 'v2'
$env:FRAUD_PREDICTION_LOG_PATH = Join-Path $env:TEMP ('fraud-demo-' + [guid]::NewGuid().ToString('N') + '.jsonl')
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. Expected model selection in this workspace is V2, 34 features, threshold 0.53; an activation manifest or environment override can change the resolved bundle. Uvicorn is not listed in `requirements.txt`; verify it is installed in the chosen environment before launch.

## Suggested admin walkthrough

1. **Admin Overview:** explain that counts are saved classifier decisions, and historical evidence is distinct from the cached FastAPI health check.
2. **Transaction Analyzer:** choose **Preview only** and use the synthetic illustration below. It does not add the current event to history.

   ```text
   Step: 5
   Type: TRANSFER
   Amount: 1500.00
   Sender reference: DEMO-SENDER-1
   Sender balance before: 5000.00
   Receiver reference: DEMO-RECEIVER-1
   Receiver balance before: 0.00
   ```

   The API constructs the 34 model features. Never send `isFraud` to a prediction endpoint.
3. **Transaction Explorer:** show search/filter/export over saved outputs. A class-1 result means predicted class 1, not confirmed fraud.
4. **Counterfactual Explainability:** if the processed Parquet data is available, use an exact event ID to show a fresh resolver-based score, optional SHAP contribution view and approximate bounded hypothetical change. State that neither proves cause.
5. **Pipeline Evidence:** show persisted Stage 1/2/3 and historical integration records. Explain that a heartbeat is not a broker health check and the page does not start services.
6. **Drift & Alerts / Delayed Labels:** explain that drift is feature-distribution evidence, while retraining eligibility requires provenance-verified genuine labels.
7. **Model Performance:** present the saved same-split V1/V2 evaluation only as synthetic PaySim benchmark results. V2's 0.53 threshold test metrics are precision 0.998468, recall 0.999489, F1 0.998978, PR-AUC 0.999490, ROC-AUC 0.999682 on 181,624 rows with 1,956 fraud labels.
8. **Model Orchestrator:** show that current real candidate creation is blocked and no real run is present. The explicit simulation action is separate and writes new synthetic artifacts; use saved reports if you do not want writes.
9. **System Health, System Architecture, Reports & Documentation:** distinguish API/resolver/file status from Kafka/Spark and use saved reference files as needed.

The current admin groups are Dashboard; Transactions; Monitoring & Data Quality; Model Lifecycle; Project Reference. The page previously called Live Monitoring is currently **Pipeline Evidence**. Retraining Lab is not in active navigation.

## Optional streaming demonstration

Only present an active stream if a Kafka broker is already configured, the `fraud-transactions` topic has one partition, Java/Spark and the connector are available, and new persistent writes are intended. Follow [STAGE1_STREAMING.md](STAGE1_STREAMING.md) and [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md). The producer repeats a finite held-out PaySim slice with synthetic IDs. It does not create new real transactions. Never start a second consumer on `checkpoints/stage1_kafka_spark`.

## Demonstrating prediction versus outcome

The prediction API does not establish an outcome. Historical model metrics compare classifications with PaySim's synthetic `isFraud` target. Stage 2 is the later-label audit path, but the inspected setup has no configured trusted source and no eligible genuine labels. Do not call PaySim labels or generated simulation labels genuine investigation results.

## Side effects to avoid during a routine talk

- **Analyzer Submit prediction / user analysis:** writes account history and an API JSONL record; use the temporary log above if intentionally demonstrated.
- **Stage 1 producer/consumer:** publishes messages, appends outputs, updates the registry and advances the checkpoint.
- **Stage 2 label API/submission:** initializes or writes SQLite state.
- **Monitoring CLI:** persists a new run, alerts and requests.
- **Orchestrator simulation:** writes a new candidate and report set under its isolated simulation directories.
- **Promotion/rollback:** changes active manifest state and is not part of this presentation walkthrough.

For a current code/artifact overview, see [the UI guide](UI_USER_GUIDE.md), [project workflow](PROJECT_WORKFLOW.md), [results](PROJECT_RESULTS_AND_LIMITATIONS.md), and [simulation details](DRIFT_SIMULATION.md).
