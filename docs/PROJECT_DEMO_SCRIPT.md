# Project Demonstration Script

This sequence demonstrates the actual Streamlit UI and saved evidence without starting Kafka/Spark or running retraining. The workspace currently has user-created persistent streaming/database outputs; avoid mutating those during a presentation unless new writes are intentional.

## Before the presentation

Use PowerShell, from the project root:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\Activate.ps1
```

The optional UI dependency is declared in `requirements-ui.txt`. It was not installed or changed during this audit. If Streamlit is already available, start the UI in its own terminal:

```powershell
streamlit run dashboard.py
```

The dashboard defaults to `http://127.0.0.1:8000` for FastAPI. Check `FRAUD_API_BASE_URL` if your API uses another port. To run the existing API in a separate terminal and isolate any Analyzer prediction log from the repository, use:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\Activate.ps1
$env:FRAUD_PREDICTION_LOG_PATH = Join-Path $env:TEMP ('fraud-demo-' + [guid]::NewGuid().ToString('N') + '.jsonl')
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In a third terminal, check API health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected fields: `status`, `model_loaded`, `model_version`, `feature_count` (34), and `threshold` (resolver-selected; expected V2/.53 with current configuration). The venv's historical runbook records Uvicorn available, but the dependency file does not declare it; if this fresh command fails, use the existing environment setup documented in `docs/DEMO_RUNBOOK.md`.

### Optional streaming demonstration

Only do this when Kafka is already installed/configured and you intentionally want to append new synthetic events, predictions, Stage 2 rows, and checkpoints. The repository does not include Kafka broker scripts, so use the existing local installation and existing KRaft configuration; do not format storage, delete/recreate topics, or reuse a checkpoint with a second consumer.

1. Start the existing Kafka broker and verify topic `fraud-transactions` on `127.0.0.1:9092`, exactly one partition.
2. Start one Spark consumer in its own terminal:

   ```powershell
   Set-Location D:\fraud-detection-system\fraud-detection-system
   .\venv\Scripts\Activate.ps1
   python -m src.streaming.stage1_streaming
   ```

3. Only if you intend a continuous synthetic replay, start the producer in another terminal:

   ```powershell
   Set-Location D:\fraud-detection-system\fraud-detection-system
   .\venv\Scripts\Activate.ps1
   python -m src.streaming.stage1_producer --delay 2
   ```

The producer loops forever over a finite held-out PaySim slice, assigning fresh synthetic IDs; it does not create new real transactions. Stop producer, Spark, and broker with Ctrl+C in their own terminals. Keep topic and checkpoint intact. Kafka/Spark were not live-tested in this documentation task.

## Recommended sidebar order and speaking notes

1. **Overview.** Point to registry counts, resolver-selected model, recent saved activity, and cached API health. Say: “Counts are saved model decisions, and prediction classes are not confirmed outcomes.” Current saved count at audit: 2,228; all class 0. Historical Phase 12 evidence is not a current broker status.
2. **Transaction Analyzer — preview.** Enter this schema-valid synthetic illustration:

   ```text
   Step: 5
   Transaction type: TRANSFER
   Amount: 1500.0
   Origin account: C1234567890
   Origin balance before: 5000.0
   Destination account: C9876543210
   Destination balance before: 0.0
   Action: Preview only
   ```

   Click **Run selected action**. Show the ordered 34-feature vector. Say: “The request carries seven transaction-time fields; the API derives model features. This is synthetic input for the demo, not a real customer transaction.” Do not include `isFraud`.

3. **Transaction Analyzer — optional prediction.** For a prediction, select **Submit prediction** and click Run selected action. Show class, probability, risk score, level, model. Say: “This is the model's score, not an investigation result.” This action updates process-local account history and writes to the API log; use the temp log above. Probability/class may change with state/model version. No class is guaranteed.

4. **Live Monitoring.** Show Stage 1 heartbeat freshness, Stage 2 totals, Stage 3 persisted report, and Phase 12 evidence. At audit time the heartbeat was stale (more than the 30-second window), although the status file said running; the Stage 3 report was historical Oct 25 evidence with 31/34 features drifted and no eligible labels. Say: “This screen reads files and saved monitor state. It does not poll Kafka or Spark.”

5. **Transaction Explorer.** To show a saved class-1 output, choose Predicted class `1` and Model version `V2`. The audited V2 log has a historical result at `2026-10-19T07:23:39.313398+00:00`, probability `0.570683`, score `57.0683`, MEDIUM. The record has no transaction ID/type/amount; explicitly call it a saved model prediction, not confirmed fraud. To show a saved class-0 example, search `PHASE12-20261019075835` to inspect the historical V2 integration row if present. Download action is optional and only downloads the filtered view. Say: “These are logged predictions; source overlap is retained and no ground truth is inferred.”

6. **Model Performance.** Show V1/V2 historical held-out comparison. Point out V2 at .53 achieved saved precision .998468, recall .999489, F1 .998978 on 181,624 synthetic PaySim test rows; one false negative and three false positives. Say: “These are offline synthetic test metrics, not live stream accuracy.”

7. **Drift & Alerts.** Show persisted drift rate 31/34 (91.18%) against 15%, report timestamp, alert, and insufficient-label result. Say: “This is a saved synthetic-window shift signal; drift is not proof of fraud, and the monitor did not train a model.” Do not run the monitoring CLI during the presentation; it writes SQLite state.

8. **Delayed Labels.** Show 2,227 unresolved predictions, 0 genuine labels, and 1 synthetic test label. Say: “The delayed-label path records provenance; synthetic labels cannot be used for genuine retraining.” Do not start the label API or submit a label; its startup and submissions persist data.

9. **Retraining Lab.** Show the minimum 100 labels/10 fraud gate, current blocked status, and Stage 6 saved demo candidate ineligible for genuine promotion. Say: “The candidate run proves a bounded synthetic training demo only. Genuine retraining is blocked by verified label requirements.” Do not run `--train`, promotion, or rollback commands.

10. **System Architecture.** Trace API / Stage 1 → feature builder → shared model resolver → outputs; then labels → monitoring → guarded retraining handoff. Say: “Drift requests review; it does not automatically retrain or deploy.”

11. **Reports & Documentation / System Health.** Download a saved report if useful. Finish at System Health: show the actual `/health` response if available, the resolver-selected model and artifact presence. Say: “Health reports FastAPI only. Kafka and Spark are not probed here.”

## Demonstrating confirmed versus predicted fraud

The Analyzer cannot establish a confirmed outcome. V1/V2 report metrics compare predictions with known `isFraud` labels on synthetic historical test partitions. Delayed labels are the mechanism for later outcomes, but current Stage 2 has no trusted genuine source configured and its one label is synthetic. Do not relabel the Phase 12 class-0 example as legitimate ground truth; it is only a predicted class. If you want to show the historical record, it contains `step=379`, `type=PAYMENT`, `amount=4349.85`, V2 probability approximately .4293, class 0, risk level MEDIUM. The CSV lacks account identifiers and pre-transaction balances, so it is not a complete Analyzer input.

## What is safe and what writes state

Safe, read-only UI actions: navigate, refresh cached dashboard values, expand saved sections, filter/search/sort Explorer, download report/filtered data, use Analyzer **Preview only**. Prediction action writes the configured API JSONL and changes that API process's in-memory account state. Starting Kafka/Spark producer/consumer writes logs, Stage 2 registration, offsets/checkpoint; continuous producer publishes synthetic messages indefinitely. Stage 2 API initializes/writes SQLite. Monitoring CLI persists monitor runs/alerts. Stage 6 training writes candidate artifacts; Stage 5 promotion/rollback changes model activation state. Keep these out of a routine talk.

## Backup plan

If FastAPI is unavailable, skip live Analyzer response and show the saved Phase 12 row and `/health` schema in the code/runbook; label both as historical/code contract. Use `Preview only` only when API is responding. If Kafka/Spark is unavailable, do not claim a live stream: show saved Stage 1 records, stale heartbeat, and historical Phase 12 PASS evidence with timestamps. Continue through model metrics, drift, delayed labels, and lifecycle pages, which read local reports/databases. If the UI itself is unavailable, use these Markdown guides and saved CSV/JSON evidence via PowerShell `Get-Content` or `Import-Csv`; do not try to repair it by clearing data or resetting Kafka.
