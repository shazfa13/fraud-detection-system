# Adaptive Real-Time Financial Transaction Fraud Detection System

An academic transaction-fraud detection prototype built on synthetic PaySim data. It combines a chronological Kafka–Spark prediction pipeline, FastAPI inference, delayed-label tracking, drift monitoring, and guarded offline model lifecycle tools. It is **not a production banking system**: genuine labels are not configured, and retraining is not automatically connected to streaming or monitoring.

## Problem and objectives

Digital transaction streams are imbalanced and time-dependent. A useful prototype must score transactions consistently, prevent target/future-data leakage, preserve a clear audit trail, and avoid retraining or activating a model on unverified outcomes. This project demonstrates those controls with historical and streaming workflows while keeping prediction, monitoring, labels, and candidate management as distinct components.

## Capabilities

- FastAPI validates and scores a transaction using a startup-loaded V1 or V2 XGBoost model.
- Stage 1 publishes PaySim demo events to Kafka; Spark Structured Streaming validates, orders, featurizes, and scores micro-batches.
- The model receives a strict ordered 34-feature vector. `isFraud` is the target; `event_id` and `step` are audit/order fields, not model inputs.
- Prediction/error JSONL and stream health are persisted. Stage 1 also registers predictions in the Stage 2 SQLite index.
- Stage 2 accepts delayed labels with source/authorization checks, immutable first-label behavior, and auditable duplicate/conflict handling.
- Stage 3 separately records feature drift, eligible-label performance, alerts, and retraining requests; an alert does not launch training.
- Stage 4 requires genuine verified-label eligibility. Stage 5 evaluates a separate candidate and requires explicit approval for activation, with guarded rollback.
- Stage 6 demonstrates bounded candidate training on historical synthetic data. Its candidate is marked ineligible for genuine promotion and Stage 5 rejects it.
- Stage 7 records end-to-end regression and integrity validation. See the [Stage 7 report](reports/stage7_end_to_end_validation_20261025T150813Z.md).

## Technology stack

| Technology | Role |
|---|---|
| Python 3.12 | Application, data preparation, tests, and workflow tools |
| Pandas / NumPy / PyArrow | Feature/data preparation and Parquet access; PyArrow is optional in requirements but needed for Parquet workflows |
| XGBoost | Binary classification over tabular transaction features |
| FastAPI / Pydantic / Uvicorn | REST endpoints and request validation; Uvicorn is present in the current venv but is not declared in `requirements.txt` |
| Apache Kafka (`kafka-python`) | Transaction event transport |
| Apache Spark Structured Streaming | Micro-batch ingestion and inference |
| SciPy / scikit-learn | Statistical drift tests and evaluation metrics |
| Streamlit (optional) | Existing local dashboard; installed separately via `requirements-ui.txt` |

See [`requirements.txt`](requirements.txt), [`requirements-ui.txt`](requirements-ui.txt), and the [demo runbook](docs/DEMO_RUNBOOK.md) for environment details.

## Architecture and workflow

```text
PaySim demo producer -> Kafka topic -> Spark micro-batches -> event validation
  -> shared 34-feature builder / process-local AccountState -> active-model resolver
  -> XGBoost scoring -> prediction JSONL + Stage 2 prediction registry

REST client -> FastAPI /predict -> same feature builder and active model -> response + API log

Delayed label submitter -> Stage 2 API / SQLite <- Stage 1 prediction registry
Stage 1 feature logs + Stage 2 eligible labels + V1/V2 reference data
  -> Stage 3 monitor -> persisted alerts and retraining requests
  -> Stage 4 eligibility gates -> separate candidate -> Stage 5 evaluation
  -> explicit human approval -> activation manifest; rollback is a separate guarded action
```

The prediction flow and the delayed-label/retraining flow are separate. Monitoring requests are evidence only. The detailed [architecture diagram and lifecycle](docs/PROJECT_WORKFLOW.md) show the implemented connections and their status. The Spark stream uses one Kafka partition because historical account state is process-local. It uses the existing checkpoint; never start concurrent consumers against that checkpoint.

## Features and model versions

The model contract is defined in metadata order and contains these 34 numeric features:

`hour_of_day`, `amount`, `log_amount`, `is_large_amount`, `amount_to_sender_balance_ratio`, `amount_ge_sender_balance`, `sender_remaining_if_debited`, `amount_to_dest_balance_ratio`, `sender_balance_before`, `sender_balance_zero_before`, `dest_balance_before`, `dest_balance_zero_before`, `sender_prev_txn_count`, `sender_prev_total_amount`, `sender_prev_mean_amount`, `sender_prev_max_amount`, `dest_prev_txn_count`, `dest_prev_total_amount`, `dest_prev_mean_amount`, `dest_prev_max_amount`, `amount_vs_sender_hist_mean`, `amount_vs_dest_hist_mean`, `sender_hours_since_prev_txn`, `dest_hours_since_prev_txn`, `dest_txn_count_last_1h`, `dest_txn_count_last_24h`, `dest_txn_count_last_168h`, `dest_amount_sum_last_24h`, `dest_amount_sum_last_168h`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`.

The feature builder uses pre-transaction balances and past-only sender/destination state. Post-transaction balances are not model inputs. See [`STAGE3_MONITORING.md`](docs/STAGE3_MONITORING.md) and the metadata files for the exact order and provenance.

The active model is **V2**, threshold **0.53**. Without an activation manifest, `FRAUD_ACTIVE_MODEL_VERSION` selects V1 or V2 and defaults to V2. If a valid `models/active_model_manifest.json` exists, it is authoritative. FastAPI and the Spark consumer resolve the model at process startup; restart both to load a changed selection. The current model artifacts are in `models/`; the demonstration candidates under `models/candidates/stage6_demo/` are not active.

## V2 evaluation results

The V2 training report uses `data/processed/v2_retraining_dataset.parquet`: 5,382,204 synthetic PaySim rows (7,649 fraud, 5,374,555 legitimate). Chronological splits were train through step 422 (5,015,193 rows), validation through 565 (185,387 rows), and held-out test through 743 (181,624 rows, 1,956 fraud). Threshold 0.53 was selected on validation data. The values below are the saved **test** results, not live-system performance.

| Model on V2 test split | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.60 | 0.991820 | 0.991820 | 0.991820 | 0.998165 | 0.999927 | 16 / 16 |
| V2 | 0.53 | 0.998468 | 0.999489 | 0.998978 | 0.999490 | 0.999682 | 3 / 1 |

V2's precision, recall, F1, and PR-AUC are higher on this split; V1's ROC-AUC is slightly higher. The dataset is simulator-generated, so these scores do not establish performance on real financial transactions. More context is in [PROJECT_RESULTS_AND_LIMITATIONS.md](docs/PROJECT_RESULTS_AND_LIMITATIONS.md).

## Labels, monitoring, and model lifecycle

The source dataset is the [PaySim synthetic financial dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). Local producer and preprocessing workflows expect `data/raw/PS_20174392719_1491204439457_log.csv`.

Stage 2 stores predictions and delayed labels in `data/stage2/prediction_labels.sqlite3`. Trusted labels require an explicit source allowlist and a private token; synthetic predictions cannot receive trusted-verified labels. The current documented setup has no genuine source configured and no eligible genuine verified cohort. Stage 4's current minimum is 100 verified labels, including 10 fraud labels, so it blocks genuine candidate creation.

Stage 3 reads prediction feature vectors, a model-specific synthetic reference dataset, and eligible labels; it stores monitoring runs, alerts, and requests in `data/stage3/monitoring.sqlite3`. A historical 31/34 feature-drift alert came from a synthetic comparison window and is not production evidence. Stage 3 never trains or promotes a model.

Stage 5 candidate validation and promotion are separate offline operations. Human approval is bound to a particular candidate and report; activation is not automatic, and rollback is explicit. Stage 6's latest candidate was evaluated on the same 15,000-row synthetic sample as V2. Candidate thresholded precision/recall/F1 were below V2's; the candidate remains marked demonstration-only and was not promoted. See [Stage 4](docs/STAGE4_RETRAINING.md), [Stage 5](docs/STAGE5_MODEL_PROMOTION.md), and [Stage 6](docs/STAGE6_RETRAINING_DEMONSTRATION.md).

## Repository map

```text
data/                 Raw/processed PaySim files and local Stage 1–3 state
models/               Frozen V1/V2 artifacts, offline candidates, activation records
notebooks/            Historical analysis and Phases 2–12
src/api/              FastAPI, feature builder, AccountState
src/labels/           Delayed-label API, SQLite store, CLI
src/monitoring/       Stage 3 monitor, drift/reference sampling, CLI
src/retraining/       Active resolver, eligibility, candidate evaluation/activation
src/streaming/        Kafka producers and Spark consumers
reports/              Evaluation, integration, monitoring, and validation records
docs/                 Stage guides, workflow, results, demo runbook, presentation
tests/                unittest coverage for streaming and lifecycle stages
```

The repo's `tools/kafka.tgz` is an archive, not an installed broker. The local Kafka installation and broker configuration are machine-specific. The active service paths and safe startup order are in [`DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md); it also explains which commands write persistent state.

## Setup and run

Use Python 3.12 and PowerShell from the repository root. A fresh environment can be prepared with:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pyarrow
```

The current project environment has Uvicorn, although `requirements.txt` does not declare it. Check `python -m uvicorn --version` before starting the API in a new environment. Optional dashboard dependencies are in `requirements-ui.txt`.

Start FastAPI with V2 (the default) using `python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000`; check `/health` and `/docs`. For the current streaming path, start the configured Kafka broker, then one consumer with `python -m src.streaming.stage1_streaming`, then optionally the continuous synthetic producer with `python -m src.streaming.stage1_producer --delay 2`. The producer loops indefinitely and publishes synthetic PaySim events. Those commands append to existing topic/log/checkpoint/database state; see the [runbook](docs/DEMO_RUNBOOK.md) before starting them. Phase 14 did not run services.

Run the tests from the project venv:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage*.py" -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Phase 13 recorded 57 passing Stage 1–6 tests and 107 passing tests overall. The venv does not include `pytest`; it was not installed. The tests establish the exercised code paths and guard behavior, not production readiness or live broker reliability.

## Reports and detailed guides

- [System workflow and Mermaid architecture](docs/PROJECT_WORKFLOW.md)
- [Evaluation results and limitations](docs/PROJECT_RESULTS_AND_LIMITATIONS.md)
- [Demo runbook](docs/DEMO_RUNBOOK.md)
- [12-slide outline and viva questions](docs/PROJECT_PRESENTATION.md)
- [Stage 1 streaming](docs/STAGE1_STREAMING.md) · [Stage 2 labels](docs/STAGE2_LABELS.md) · [Stage 3 monitoring](docs/STAGE3_MONITORING.md)
- [Stage 4 eligibility](docs/STAGE4_RETRAINING.md) · [Stage 5 promotion/rollback](docs/STAGE5_MODEL_PROMOTION.md) · [Stage 6 demonstration](docs/STAGE6_RETRAINING_DEMONSTRATION.md) · [Stage 7 validation](docs/STAGE7_END_TO_END_VALIDATION.md)
- [V2 training report](reports/model_v2_training_report.json) · [V1/V2 comparison](reports/model_v1_vs_v2_comparison.json) · [V2 deployment report](reports/model_v2_deployment_report.json) · [dataset report](reports/v2_retraining_dataset_report.json) · [drift report](reports/drift_report.json) · [Stage 7 report](reports/stage7_end_to_end_validation_20261025T150813Z.json)

## Limitations and future work

PaySim is synthetic. Genuine investigation/chargeback labels and a verified source are not available in the current setup. Stage 3's statistical limits and Stage 4's eligibility minima need calibration with a real, authorized, sufficiently large cohort. Account history is process-local, and the single-partition design limits scaling. The live Stage 1 producer repeats a finite historical slice, and streaming append/checkpoint behavior is at-least-once. Phase 13 did not repeat live Kafka/Spark or a persistent monitoring run. The lifecycle tools are guarded and offline, not an automated adaptive retraining service. Future work begins with authorized real-label integration, representative evaluation, operational policy review, distributed state design, and controlled live validation.
