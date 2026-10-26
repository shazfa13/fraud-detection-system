# Local demo runbook

This runbook describes the existing Windows PowerShell setup. Running the live pipeline appends prediction/error outputs, updates Stage 2/3 SQLite stores, and advances the existing Kafka checkpoint/topic. This documentation task did not start these services. Do not run them against this worktree unless those persistent updates are intended.

## Prerequisites

- Windows PowerShell, the repository at `D:\fraud-detection-system\fraud-detection-system`, and its existing `venv`.
- The PaySim CSV at `data/raw/PS_20174392719_1491204439457_log.csv` for the producer.
- A locally installed/configured Kafka broker at `127.0.0.1:9092`, with topic `fraud-transactions` configured for one partition.
- Java/JDK for Spark. The configured Spark Kafka connector may be fetched from Maven Central on first launch. The Windows Spark setup expects `tools/hadoop/bin/winutils.exe`; check the console message if it is missing.
- The current environment was Python 3.12.7 and had Uvicorn 0.54.0. `requirements.txt` declares FastAPI but does not list Uvicorn; verify it is available in a fresh environment before using the Uvicorn command. No dependencies were installed or changed for Phase 14.

From a PowerShell terminal:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\Activate.ps1
```

Install the declared project dependencies only when preparing a separate environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install pyarrow
```

PyArrow is commented as optional in the requirements file, but Parquet workflows such as Stage 6 need a Parquet engine. The current repo's data and environment were already present. `requirements-ui.txt` is optional for the Streamlit dashboard.

## Environment variables

| Variable | Purpose | Current/default behavior |
|---|---|---|
| `FRAUD_ACTIVE_MODEL_VERSION` | Startup selection when there is no `models/active_model_manifest.json` | Defaults to `v2`; set to `v1` to select V1. Restart API and Spark processes after a change. An activation manifest, when present, is authoritative. |
| `FRAUD_PREDICTION_LOG_PATH` | Optional FastAPI `/predict` JSONL destination | Defaults under `reports/`; set a new `%TEMP%` path for a disposable API smoke test. |
| `FRAUD_TRUSTED_LABEL_SOURCES` | Comma-separated Stage 2 genuine source allowlist | Empty by default; do not configure a source without real authorization. |
| `FRAUD_TRUSTED_LABEL_TOKEN` | Shared local credential for trusted submissions | Must be at least 32 characters; keep private. This is not identity management. |

Kafka address/topic, Stage 1 output paths, and checkpoints are constants in `src/streaming/config.py`, not environment overrides in this implementation.

## FastAPI inference (separate terminal)

The API loads model artifacts once at startup. For a temporary prediction log and default V2 selection:

```powershell
$env:FRAUD_ACTIVE_MODEL_VERSION = 'v2'
$env:FRAUD_PREDICTION_LOG_PATH = Join-Path $env:TEMP ('fraud-api-demo-' + [guid]::NewGuid().ToString('N') + '.jsonl')
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Check the health endpoint and interactive schema:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
Start-Process 'http://127.0.0.1:8000/docs'
```

`/health` reports `status`, `model_loaded`, `model_version`, `feature_count`, and `threshold`. The model input schema requires `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, and `oldbalanceDest`; `isFlaggedFraud` is optional. `isFraud` is rejected as an extra field. A `/predict` call updates this process's in-memory account history and appends to the configured log, so use only a disposable demo request. There is no transaction ID field in the FastAPI request schema. Stop the API with Ctrl+C in its terminal.

## Kafka and Stage 1 Spark stream

The repository does not include a configured Kafka installation or Kafka scripts. Use the existing broker installation/configuration for this machine; the Phase 1 guide shows the Windows KRaft start command with a placeholder Kafka path. Do not format the broker storage. First verify that the existing broker and `fraud-transactions` topic are reachable and that the topic has the configured single partition. Do not delete or recreate the topic to troubleshoot.

Start services in separate terminals and in this order:

1. Start the existing Kafka broker using its already configured KRaft properties (see [STAGE1_STREAMING.md](STAGE1_STREAMING.md)).
2. Start exactly one Stage 1 consumer against the existing checkpoint:

   ```powershell
   Set-Location D:\fraud-detection-system\fraud-detection-system
   .\venv\Scripts\Activate.ps1
   $env:FRAUD_ACTIVE_MODEL_VERSION = 'v2'
   python -m src.streaming.stage1_streaming
   ```

   Keep it running. It uses `checkpoints/stage1_kafka_spark`; never run another consumer concurrently against that path. It starts from the configured Kafka topic/checkpoint state.

3. Only when you intend to publish synthetic messages, run the continuous producer in another terminal:

   ```powershell
   Set-Location D:\fraud-detection-system\fraud-detection-system
   .\venv\Scripts\Activate.ps1
   $env:FRAUD_ACTIVE_MODEL_VERSION = 'v2'
   python -m src.streaming.stage1_producer --delay 2
   ```

   This producer loops indefinitely over a finite held-out PaySim slice and assigns new synthetic event IDs. It does **not** create new underlying transactions. Ctrl+C stops it. Every published message can advance the existing topic and produce new log/database/checkpoint state.

An explicit bounded *offline* selection is available and does not publish to Kafka:

```powershell
$sample = Join-Path $env:TEMP ('stage1-validation-' + [guid]::NewGuid().ToString('N') + '.jsonl')
python -m src.streaming.stage1_producer --validation-sample 20 --output $sample
```

To inspect live outputs while intentionally running Stage 1:

```powershell
Get-Content .\data\stage1_streaming\predictions.jsonl -Tail 10
Get-Content .\data\stage1_streaming\errors.jsonl -Tail 10
Get-Content .\data\stage1_streaming\stream_status.json
```

The prediction/error files are append-only. Expected error examples include rejected malformed events and `stage2_store_failure` when the secondary SQLite registry is unavailable; the Stage 1 guide explains replay recovery. Do not use log deletion as a troubleshooting step. Stop the producer first, then stop Spark with Ctrl+C, and stop the broker with Ctrl+C in its own terminal. Leave the checkpoint and topic intact for restart recovery.

## Delayed-label API (optional, persistent)

The Stage 2 API runs on loopback port `8769`:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\Activate.ps1
uvicorn src.labels.api:app --host 127.0.0.1 --port 8769
```

Its startup initializes the default `data/stage2/prediction_labels.sqlite3`, and submissions write labels/audit rows there. Do not launch it just for a read-only presentation. For workflow testing, use the temporary-database tests. If an operator intentionally submits a `synthetic_test` label, it remains synthetic and persists; it never becomes genuine verified ground truth. See [STAGE2_LABELS.md](STAGE2_LABELS.md) for the payload, authorization rules, and queries. Ctrl+C stops the API; it does not undo database writes.

## Monitoring and retraining inspection

Stage 3's `python -m src.monitoring.cli` writes a monitoring run, alerts, and request records to the persistent Stage 3 database. Do not run it as a harmless health check. Review the saved report or the existing dashboard for read-only inspection; execute the monitor only when a new persisted observation is intended.

The Stage 4 eligibility command reads the Stage 2 database read-only but writes a report. Use a unique temporary report path; do not pass `--train-candidate` for an eligibility check:

```powershell
$report = Join-Path $env:TEMP ('stage4-eligibility-' + [guid]::NewGuid().ToString('N') + '.json')
python -m src.retraining.stage4_workflow --report $report
Get-Content $report -Raw
```

Given the current absence of eligible genuine labels, the expected result is `insufficient_eligible_labels`; it should not create a candidate. Stage 5's inventory is read-only:

```powershell
python -m src.retraining.stage5_workflow status --candidate models\candidates
```

Stage 6 `--preflight` is read-only. `--train` writes a new demonstration candidate from synthetic PaySim data and is not a genuine retraining run; promotion is blocked. Only run it when you deliberately want those new candidate artifacts. See the [Stage 4](STAGE4_RETRAINING.md), [Stage 5](STAGE5_MODEL_PROMOTION.md), and [Stage 6](STAGE6_RETRAINING_DEMONSTRATION.md) guides.

## Tests and references

Run tests from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage*.py" -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Phase 13 recorded 57 passing Stage 1–6 tests and 107 passing tests overall. These tests use fixtures for writable state; they do not substitute for live broker/service testing. `pytest` is absent from the current venv and was not installed.

## Common issues

- **API fails to start:** confirm model/metadata/threshold files and `reports/feature_engineering_summary.json` are present. Verify the selected version and inspect the startup error. Current `requirements.txt` does not declare Uvicorn even though the existing venv has it.
- **HTTP 422:** check the request schema and field types; do not send `isFraud` in the prediction request.
- **Spark cannot connect or find the Kafka connector:** confirm the broker, existing topic, Java, and connector dependency availability. The first connector launch may need Maven Central access.
- **Windows Spark reports missing `winutils.exe`:** follow the message and the Spark setup note in [STAGE1_STREAMING.md](STAGE1_STREAMING.md); do not alter a live checkpoint.
- **No genuine labels or Stage 4 says insufficient:** this is the expected current gate. Configure only a genuinely authorized source and token; synthetic labels cannot satisfy it.
- **Spark restarts or output duplicates:** Stage 1 is at-least-once; the append sink can duplicate an event if interruption occurs between append and checkpoint commit. Preserve the checkpoint and use Kafka partition/offset for delivery identity.

## Detailed references

- [System workflow and architecture](PROJECT_WORKFLOW.md)
- [Results and limitations](PROJECT_RESULTS_AND_LIMITATIONS.md)
- [Presentation outline and viva preparation](PROJECT_PRESENTATION.md)
- [Stage 1 streaming](STAGE1_STREAMING.md), [Stage 2 labels](STAGE2_LABELS.md), [Stage 3 monitoring](STAGE3_MONITORING.md)
- [Stage 4 retraining gates](STAGE4_RETRAINING.md), [Stage 5 promotion and rollback](STAGE5_MODEL_PROMOTION.md), [Stage 6 demonstration](STAGE6_RETRAINING_DEMONSTRATION.md), [Stage 7 validation](STAGE7_END_TO_END_VALIDATION.md)
- [V2 training report](../reports/model_v2_training_report.json), [Phase 11 V1/V2 comparison](../reports/model_v1_vs_v2_comparison.json), [Stage 3 drift report](../reports/drift_report.json), [Phase 12 deployment report](../reports/model_v2_deployment_report.json), [Phase 10 dataset report](../reports/v2_retraining_dataset_report.json), [Stage 7 validation JSON](../reports/stage7_end_to_end_validation_20261025T150813Z.json)

## Project status and future work

The current active model is V2 at threshold `0.53` with the exact 34-feature contract. The Phase 13 Stage 7 report records 107 passing unit tests and integrity checks; live Kafka/Spark/API services and the persistent Stage 3 monitor were not rerun in Phase 13. Stage 6 demonstration artifacts remain separate and unpromoted. The project is an academic prototype, not a production-ready banking system.

Before any genuine adaptive retraining claim, the project needs an authorized genuine label source, sufficient correctly matched labels (including fraud examples), a comparable evaluation cohort, calibrated acceptance and drift policies, and an explicitly reviewed candidate. Future engineering work also includes cross-process/distributed account-state management, operational monitoring and capacity testing, and the final academic defense materials.
