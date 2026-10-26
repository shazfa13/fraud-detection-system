# Fraud Intelligence UI User Guide

This guide follows the Streamlit implementation in `dashboard.py`, `ui/components.py`, `pages/`, `services/`, and the FastAPI/Stage 1/Stage 2/Stage 3 code. The sidebar order is Overview, Transaction Analyzer, Live Monitoring, Transaction Explorer, Model Performance, Drift & Alerts, Delayed Labels, Retraining Lab, System Architecture, Reports & Documentation, and System Health.

## What the displayed values mean

The dashboard is a viewer, not a service supervisor. It reads saved JSON/JSONL/CSV files and SQLite databases, reads model metadata through the shared resolver, and makes a short cached `GET /health` request to FastAPI. Only Transaction Analyzer has an explicit prediction action. It never infers ground truth from a prediction. The project uses synthetic PaySim transaction data; “live” can mean a currently running local pipeline processing synthetic replay, not live bank traffic.

The UI contains no fabricated prediction probabilities or performance metrics in the inspected paths. Metric values come from the API response, the SQLite registry, or saved reports. Some text is deliberately static: page headings, explanatory copy, fallback labels such as `—`/`Unknown`, and the architectural diagram. Empty/missing values may display those fallbacks; they are not measured values. The Explorer normalizer uses `—` when an input record has no transaction ID/type, and `Unknown` when model/risk metadata is absent. The UI also uses examples/placeholders in input boxes; they are not stored transactions.

At audit time the local registry had 2,228 predictions (0 predicted fraud, 2,228 predicted legitimate), registered from 2026-10-25 11:39:08 UTC through 2026-10-26 15:59:12 UTC. These are saved model outputs, not verified outcomes. Stage 2 had 2,227 unresolved predictions, zero genuine verified labels, one synthetic test label, one duplicate submission, zero conflicting labels, and no configured trusted source. The latest Stage 3 report was generated 2026-10-25 12:31:18 UTC: 31 of 34 tested features were flagged as drifted (91.18%, against 15%); it had zero eligible verified labels and overall status `insufficient_data`. These are workspace snapshots, not guaranteed values on a later run.

The Explorer combines a separate, capped history from older API logs and historical streaming outputs. At audit time its newest 500 normalized rows included six class-1 outputs (one saved V2 API-log output, five V1 outputs) and 494 class-0 outputs. This is not the same cohort as the current Stage 2 registry. It gives you a saved class-1 prediction to show, but the logs lack transaction IDs and the outputs are not verified fraud outcomes.

The Stage 1 status file recorded V2/topic `fraud-transactions` as running at 2026-10-26 16:00:35 UTC, but the UI treats heartbeats older than 30 seconds as stale. At the audit check (16:06 UTC), that heartbeat was stale. Requests to local FastAPI port 8000 and Streamlit port 8501 timed out, so current HTTP availability could not be confirmed. Kafka/Spark are not probed by the dashboard. Do not describe every displayed value as “live”: distinguish current API health, recent process heartbeat, saved records, historical evaluation, and synthetic source data.

## Shared sidebar behavior

Each page calls `configure_page`, which sets the title/theme and renders all 11 `st.page_link` links. The native Streamlit page navigation is hidden. “Refresh dashboard data” clears Streamlit's cached data and reruns the page; it does not restart services or create predictions. API health uses `st.cache_data(ttl=15)`, so it may be up to 15 seconds old. The bottom sidebar also displays the API-served model or `Unavailable` and the fixed academic/synthetic PaySim note.

## 1. Overview

**Purpose and source.** A read-only summary of Stage 2 registry counts, the resolver-selected model, short-cached API health, recent normalized prediction history, and the newest readable Phase 12 integration evidence. Implemented in `dashboard.py`.

**What to inspect.** The four registry cards are total, predicted class 1, predicted class 0, and predicted-fraud share. The range is the minimum/maximum registry registration timestamp. These counts describe classifier decisions, not confirmed fraud. Model selection shows the shared resolver's version, threshold, feature count, and resolver source. FastAPI reachable is a live-but-cached `/health` response; it does not check Kafka/Spark. Recent activity is up to 8 of the first 100 normalized rows from API and known streaming report files, newest timestamp first. The pipeline evidence pill is a historical Phase 12 report, not runtime health.

**Actions.** Click Refresh dashboard data in the sidebar to clear UI cache and rerun. There are no form fields or write actions on this page.

**How to explain it.** “This is an audit summary: prediction counts and saved evidence are kept separate from the current API health check. A predicted class is not a verified outcome.”

## 2. Transaction Analyzer

**Purpose and dependencies.** Calls the existing FastAPI `/state/preview` or `/predict` endpoint via `services/api_client.py`; model features are created by `src/api/app.py`. The page itself does not load an estimator. FastAPI must be reachable at `FRAUD_API_BASE_URL` or the default `http://127.0.0.1:8000`.

**Exact UI fields.** Step (whole integer >= 1); Transaction type (select box populated from `reports/feature_engineering_summary.json`, currently the frozen categories `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`); Amount (finite number >= 0); Origin account (non-empty string); Origin balance before (finite number >= 0); Destination account (non-empty string); Destination balance before (finite number >= 0). Action is either Preview only or Submit prediction. Click Run selected action to submit the form. The UI sends exactly `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, `oldbalanceDest`. `isFraud`, `event_id`, and post-transaction balances are not sent.

**Copy-ready schema-valid example (synthetic illustration, not a real row):**

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

This matches the FastAPI schema example and is structurally valid. It is not a source-dataset row, and no particular class/probability is promised. Preview calls `/state/preview` and returns the ordered 34-feature vector without changing account history. Select Submit prediction to call preview, check the returned feature order, check `/health`, then call `/predict`; the result shows predicted class, fraud probability, probability-times-100 risk score, risk level, and model version. A prediction mutates the API process-local account state and appends an output-only JSONL record at `FRAUD_PREDICTION_LOG_PATH` or `reports/model_<version>_predictions_log.jsonl`. It does not return confirmed ground truth. Avoid submitting a prediction during a talk unless that write is intended. No transaction ID is accepted by this API form.

**Interpretation.** Threshold is applied to probability to produce class 0/1. Risk bands in current metadata are LOW 0–<30, MEDIUM 30–<70, HIGH 70–100. Risk score is not independently calibrated. The UI shows the threshold from a health check immediately before prediction and warns if the model version changed between checks.

**Expected failures.** Missing/non-numeric input or a type outside the frozen list is rejected in the UI/API. An unreachable service produces an API error. A model/feature-order mismatch blocks submission. The API rejects extra keys (`extra=forbid`), including `isFraud`.

## 3. Live Monitoring

**Purpose and source.** A composite read-only view of Stage 1 heartbeat/predictions/errors, Stage 2 registry/label counts, Stage 3's saved report, and the latest Phase 12 evidence. It does not connect to Kafka, launch Spark, or poll a broker.

**Sections.** Stage 1 process state is computed from `data/stage1_streaming/stream_status.json`; `running` is only shown as fresh when the heartbeat is less than 30 seconds old. Recent JSONL predictions and errors are saved history and can remain visible after a process stops. Stage 2 registration counts and label disposition metrics are read from `data/stage2/prediction_labels.sqlite3`; they are not accuracy or throughput. Stage 3 displays its persisted run status, drift status, eligible verified labels, performance status/metrics if available, alerts, and retraining-request decision. Phase 12 shows persisted Kafka/Spark test checks and output CSV, labeled historical. Broker connectivity at the bottom explicitly says it was not checked.

**Actions/inputs.** No input, filter, or write button. Refresh only clears Streamlit cache. If no Stage 3 report exists, the page suggests `python -m src.monitoring.cli`; that command writes monitoring data and should not be used as a harmless presentation refresh.

**Audit snapshot.** Stage 1 status file is now stale by the page's rule. The Stage 3 run is historical and reports feature drift alongside insufficient verified-label performance data. Phase 12's PASS is historical evidence from Oct 19, not proof a broker is running now.

## 4. Transaction Explorer

**Purpose/source.** Searches up to 500 normalized rows from the configured API log, known V1/V2 logs, Phase 7 and Phase 12 CSVs, plus versioned Phase 12 output CSVs. Source names are preserved; overlapping records are not automatically deduplicated. It does not read raw dataset rows or labels.

**Controls and order.** Choose Recorded source (All sources or a discovered source); enter optional Search text (matches any displayed/normalized field, including ID, type, model, source); select optional Predicted class, Risk level, and Model version multiselect values; choose Sort records by timestamp, transaction ID, probability, or risk score. Selected filters combine. Timestamp sorting defaults newest first; other sort columns are ascending. Exact duplicate values in the filtered frame trigger a warning but remain visible. The table shows records. Download filtered records saves the current filtered table as `transaction_predictions.csv`.

**Meaning/limitations.** Missing transaction IDs display `—`; normalized rows can retain source fields only when present. CSV/log record counts are bounded and not complete audit export. The download contains prediction output and available IDs/timestamps; no verified outcome is inferred.

**Showing a saved class-1 output.** Select **Predicted class = 1** and **Model version = V2**. The audited V2 log includes a historical class-1 result timestamped `2026-10-19T07:23:39.313398+00:00`, probability `0.5706832408905029`, risk score `57.0683`, risk level MEDIUM. Its transaction ID/type/amount are absent. Call it a saved model prediction, not a confirmed fraud case. For a saved class-0 example use the Phase 12 integration row described at the end of this guide.

## 5. Model Performance

**Purpose/source.** Reads V2 metadata and threshold, the shared active-model resolver, `reports/model_v1_vs_v2_comparison.json`, and the saved V1 feature-importance CSV. Metrics are historical held-out synthetic PaySim evaluation, not live monitoring performance.

**What is shown.** Resolver-selected version; feature count from V2 metadata; V2 threshold from the V2 threshold file; optional expander with the ordered 34 features. V1 and V2 rows compare precision, recall, F1, PR-AUC, ROC-AUC, confusion counts, threshold, and row count on the same V2 test split (steps 566–743; 181,624 rows, 1,956 fraud). The historical report records V1 threshold .60 and V2 .53. Below that, the historical V1 feature-importance CSV is charted if readable. Lifecycle readiness labels say automatic retraining is not operational and promotion needs explicit Stage 5 approval.

**Current saved comparison.** V1: precision/recall/F1 0.991820; PR-AUC 0.998165; ROC-AUC 0.999927; TP/TN/FP/FN 1,940/179,652/16/16. V2: precision 0.998468; recall 0.999489; F1 0.998978; PR-AUC 0.999490; ROC-AUC 0.999682; TP/TN/FP/FN 1,955/179,665/3/1. V2's saved Phase 11 decision is `V2_CANDIDATE_FOR_DEPLOYMENT`; this is not an external production approval.

**Actions.** Only expand feature contract; no filters or model modifications. Missing/malformed report produces an info/warning message.

## 6. Drift & Alerts

**Purpose/source.** Displays the latest persisted Stage 3 report read through the read-only monitor report accessor. It does not run detection or write alerts.

**What is shown.** Freshness label is based on the saved `generated_at` timestamp (<24h versus >24h). Cards show saved model/threshold, recent records read, reference sample size, and drifted/tested features. Decision metrics show drift rate, configured threshold, and recorded decision. Selected feature names and any persisted per-feature output are displayed only if actually stored. Current saved aggregate does not include per-feature PSI/p-values, so the page explains their absence. Alerts table lists IDs, type, window, drift-rate evidence, and whether newly recorded. Retraining request status/reason is an auditable request only.

**Current saved result.** 31/34 = 91.18%, above the 15% configured drift-rate gate, but generated Oct 25 12:31 UTC and stale at the audit time. Stage 3's overall status is `insufficient_data`; verified-label performance has zero eligible transactions. This is historical synthetic PaySim window evidence, not a live production drift finding.

**Actions.** No filters or buttons. Do not invoke the monitoring CLI during a demo unless you intend its persistent SQLite writes.

## 7. Delayed Labels

**Purpose/source.** Read-only Stage 2 registry statistics and label/submission provenance from SQLite. The page never edits labels or reveals credentials.

**Metrics.** Predictions registered, unresolved predictions, resolved genuine labels, synthetic test labels, identical duplicate submissions, conflicting submissions, unmatched pending labels. The provenance table groups `source_type`, `label_source`, `status`, and count. The audit table groups accepted/duplicate/rejected/conflict dispositions. Synthetic test labels are explicitly excluded from genuine retraining eligibility.

**Audit snapshot.** 2,228 predictions; 2,227 pending; zero genuine verified; one synthetic label; one duplicate; zero conflicting labels; zero unmatched pending submissions; no genuine source configured. Do not claim the dashboard has a confirmed fraud outcome. There are no form inputs or write actions. SQLite readers are read-only.

## 8. Retraining Lab

**Purpose/source.** Read-only genuine-label eligibility via Stage 4 collector, active model snapshot, saved Stage 6 demonstration report, and lifecycle directory/manifest inventory. The page explicitly offers no train/approve/promote/rollback action.

**Sections.** Active version, threshold, 34-feature count, and resolver source. Eligibility reports labels examined, genuine eligible labels, eligible fraud labels, minimum labels/fraud (100/10), source/state breakdowns, and exclusion reasons. Count gate status is not a substitute for later policy gates. Stage 6 shows saved candidate status, candidate threshold, evaluation rows, promotion eligibility, same-row active/candidate comparison, promotion status, and dataset provenance. Rollback inventory lists the manifest, promotion-history directory, backup, and legacy `models/active` contents. Presence alone does not prove validity.

**Current interpretation.** No trusted source/token is configured; the current one synthetic test label does not qualify. The saved Stage 6 candidate was trained on historical synthetic PaySim samples, status `candidate_trained_not_promoted`, and `eligible_for_genuine_promotion=false`. Stage 4 requires 100 labels including 10 fraud labels; genuine retraining is currently blocked. No UI action changes artifacts.

## 9. System Architecture

**Purpose/source.** Static Streamlit diagram of the implemented request/streaming path and delayed-label/monitoring/lifecycle handoffs, plus the saved workflow guide and Stage 7 report.

**What to explain.** Requests and Stage 1 events pass through validation and a 34-feature builder into the resolver-selected XGBoost model. API predictions return/log output; Stage 1 sends Kafka events through Spark and records JSONL plus the Stage 2 registry. Delayed labels feed monitoring only when trusted and matched. Stage 3 drift/retraining signals do not automatically train or deploy. Stage 4 eligibility and Stage 5 approval/promotion are distinct; Stage 6 is synthetic demonstration only.

**Actions.** Expand the saved Phase 14 workflow guide. Stage 7 status is read from a fixed report. The page does not test or start services.

## 10. Reports & Documentation

**Purpose/source.** Offers an allowlist of known files, checking that each exists, resolves within the project, and is at most 4 MiB. Each card has a Download saved file button; the download is saved bytes, not regenerated content. Missing/oversize/unreadable items are labeled unavailable. Listed files include README, workflow/results/demo/stage guides, Stage 7 report, V1/V2 comparison, and V2 training report.

**Actions.** Click the download button on the desired card. There are no filters or backend writes. Files in this UI allowlist are defined in `pages/10_Reports_Documentation.py`; newly created guides may not appear unless added to that list.

## 11. System Health

**Purpose/source.** Short-cached FastAPI `/health` plus local artifact presence checks and active resolver metadata. It does not probe Kafka/Spark, inspect their processes, or load the estimator from the dashboard.

**What is shown.** FastAPI Online/Offline; if reachable, model-loaded boolean, API model version, feature count, threshold. Then the shared resolver's selected model/version/threshold/source, followed by presence of V1/V2 models, metadata, and V2 threshold. Static explanation clarifies resolver manifest precedence and academic limitation.

**Actions.** No form/button. The sidebar refresh clears cache. A healthy API response does not establish Kafka/Spark health. Conversely, a missing local artifact is a real filesystem state; an unavailable health endpoint means API reachability was not verified.

## Safe demonstration input and label handling

Use the synthetic illustrative payload above with **Preview only** first. Explain that the 34 inputs are derived by the API and that values are not a verified transaction. If showing a prediction, start the API with a disposable `FRAUD_PREDICTION_LOG_PATH` as in `PROJECT_DEMO_SCRIPT.md`, select Submit prediction, and explain that it changes process-local history and writes a prediction log. Do not send `isFraud` to the prediction route. `isFraud` is the training/evaluation target or delayed-label outcome only; a prediction is not truth.

For a reproducible source-data row, inspect the raw PaySim CSV and copy only the seven request fields, excluding `isFraud`, `newbalanceOrig`, and `newbalanceDest`. The raw row is still synthetic PaySim. For saved streaming output, the historical Phase 12 row `PHASE12-20261019075835` was `step=379`, `PAYMENT`, amount `4349.85`, V2 probability `0.4292928`, class 0, MEDIUM; that record lacks account names and pre-transaction balances and cannot be pasted into Analyzer as-is.
