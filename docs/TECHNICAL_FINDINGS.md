# Technical Findings: Fraud Intelligence UI Audit

## Scope and method

Inspected `dashboard.py`, `ui/components.py`, all page modules under `pages/`, `services/api_client.py`, `services/data_service.py`, FastAPI `src/api/app.py`, shared model resolver, Stage 1 streaming producer/consumer/config, Stage 2 label configuration/store/API, Stage 3 monitor/config/drift detector, Stage 4/6 workflows, model metadata/thresholds, local SQLite databases, saved reports, and the existing runbooks. No application code, model, source data, database, checkpoint, Kafka topic, or report was changed. No model was trained, no service was started, and no tests were run for this audit. The Oct 25 Stage 7 report records 107 passing `unittest` tests and 57 focused tests; this was read as saved evidence, not rerun here.

## Framework, navigation, and read/write boundaries

The UI is a multipage Streamlit application. `dashboard.py` is Overview; `pages/1_...py` through `pages/10_...py` implement the other pages. `ui/components.py` hides native sidebar navigation and renders ordered `st.page_link` links. Each page uses `configure_page`. The sidebar API health check is cached for 15 seconds. Streamlit is optional and listed in `requirements-ui.txt`.

Most pages read known data files through path-bounded/size-bounded functions in `services/data_service.py`; SQLite readers use read-only connections for dashboard counts and label tables. Current exception: Stage 4 eligibility called by Retraining Lab reads the Stage 2 database and model/report files; it is an inspection operation. Reports page only exposes its explicit allowlist. The UI does not start Kafka/Spark/FastAPI. Analyzer Preview calls `/state/preview`; Analyzer prediction calls `/state/preview`, `/health`, `/predict`, updates process-local `AccountState`, and appends an output-only JSONL record. Explorer filtering and CSV download are local UI operations.

### Page dependency map

| Sidebar page | Actual data/dependency | Write behavior |
|---|---|---|
| Overview | Stage 2 `predictions` SQLite table; active model resolver; cached FastAPI `/health`; known API logs/streaming CSV; newest Phase 12 JSON | None |
| Transaction Analyzer | `/state/preview`, `/health`, `/predict`; frozen feature summary and resolver | Preview is non-consuming; prediction updates API in-memory account state and writes configured API JSONL |
| Live Monitoring | Stage 1 status/prediction/error files; Stage 2 SQLite; latest Stage 3 monitor DB report; Phase 12 report/CSV | None; the suggested monitoring CLI would persist data |
| Transaction Explorer | known API JSONL and Phase 7/12 CSV output sources | Download only |
| Model Performance | V2 metadata/threshold; saved V1/V2 comparison; V1 importance CSV; label collection report; Phase 12 evidence | None |
| Drift & Alerts | latest Stage 3 persisted report from SQLite | None; monitor CLI writes run/alert/request records |
| Delayed Labels | Stage 2 SQLite label/submission tables and statistics | None; label API submissions are persistent |
| Retraining Lab | resolver; read-only Stage 4 eligibility collector; Stage 6 training report; artifact/manifest inventory | None from UI; separate workflow commands can train/promote/rollback |
| System Architecture | static diagram; saved workflow Markdown and Stage 7 JSON | None |
| Reports & Documentation | fixed allowlist of files under project root, maximum 4 MiB | Downloads only |
| System Health | cached FastAPI `/health`; resolver; artifact existence checks | None |

## Current snapshot and whether UI values are dummy

The UI does not calculate arbitrary/example prediction metrics and then present them as results. Displayed numeric metrics are sourced from the API, local records, or saved reports. Static labels, architecture copy, input placeholders, fallback text (`—`, `Unknown`, `Unavailable`), and historical report content are not live measurements. The Streamlit UI therefore has **some fixed explanatory text and historical/synthetic values, but no evidence of fabricated live prediction numbers** in the inspected implementation.

The values are not all “legit and live” in the ordinary production sense:

- The source transactions are synthetic PaySim. Stage 1 producer tags replayed records `synthetic_paysim_demo`; Stage 6 explicitly records `historical_paysim_synthetic_labels`.
- Current Stage 2 registry snapshot: 2,228 rows, predicted fraud 0, predicted legitimate 2,228. Registration range: Oct 25 11:39:08 UTC–Oct 26 15:59:12 UTC. These are actual saved predictions from this machine, but the count does not prove accuracy, current throughput, or true fraud outcomes.
- Stage 2 label snapshot: 2,227 pending; zero genuine verified; one synthetic test; one duplicate; no conflicting submissions; no unmatched pending; no trusted source configured. Thus there is no confirmed genuine fraud outcome in the current registry.
- Stage 1 status file: says running/V2/topic `fraud-transactions`, PID 18648, timestamp Oct 26 16:00:35 UTC. The UI's 30-second age rule classifies it stale when the page is opened later. Stage 1 prediction JSONL last-write time was 15:59:12 UTC. A file timestamp is evidence of recent writes, not proof a process is healthy now.
- At 16:06 UTC, probes to FastAPI `127.0.0.1:8000` and Streamlit `127.0.0.1:8501` timed out. Kafka is not HTTP-probed. This sandboxed audit could not confirm live HTTP service availability; retry locally in the presentation environment.
- Latest Stage 3 report: Oct 25 12:31:18 UTC, 369 recent records read, 34 tested, 31 drifted, drift rate .9118, configured trigger .15, zero eligible verified labels, overall `insufficient_data`. Historical/synthetic signal; not current real-world drift.
- Phase 12 evidence dated Oct 19 is a saved PASS: one Kafka message sent and one V2 output, step 379 PAYMENT amount 4349.85, probability .4292928, class 0, score 42.9293, MEDIUM. Historical integration evidence, not a current Kafka connection or confirmed legitimate outcome.
- V1/V2 performance report: held-out synthetic PaySim test split, 181,624 rows/1,956 fraud. V2 threshold .53 and V1 .60. V2 saved metrics: precision .998468, recall .999489, F1 .998978, PR-AUC .999490, ROC-AUC .999682, TP/TN/FP/FN 1955/179665/3/1. Historical offline metrics.
- V2 metadata states 34 features, trained Oct 6, and threshold .53 chosen by maximum validation F1. Current shared resolver defaults to V2 because there is no activation manifest and no `FRAUD_ACTIVE_MODEL_VERSION` override in the inspected process environment. V2 is currently the API model contract when API imports in this environment, but a running server was not reachable.

**Answer to the dummy-values question:** there are fixed explanatory/UI strings and the application is fed synthetic PaySim records, but its displayed recorded predictions and metrics come from actual local saved records/reports. They are not all live, and “actual saved output” does not mean “real transaction” or “verified fraud label.” The only live service reading is the cached FastAPI `/health` request; that endpoint could not be reached during this audit. The safest presentation statement is: “These are real outputs and saved measurements from the project’s synthetic PaySim demo/evaluation; historical reports and saved records are labeled as such, and no verified production outcomes are available.”

## Model resolver and prediction path

`src/retraining/active_model.py` first looks for `models/active_model_manifest.json`. If present, it validates version, ordered features, threshold, and hashes and treats the manifest as authoritative. No manifest exists in this workspace. Without one, `FRAUD_ACTIVE_MODEL_VERSION` is honored; default is V2. The resolver validates a shared 34-feature contract against V1 metadata and checks threshold/artifact consistency. It returns paths/metadata but the UI does not load the XGBoost estimator.

V2 uses the same frozen 34 ordered features as V1 (metadata order). The model vector excludes `step` (temporal context), `event_id` (audit identity), and `isFraud` (target/leakage); it also excludes post-transaction balances. Seven UI request fields are converted into temporal, amount, pre-transaction balance, process-local past-only account-history, and transaction-type features. API `/predict` calculates `predict_proba`, assigns class 1 when probability >= active threshold, sets risk score to probability × 100, maps the score to configured LOW/MEDIUM/HIGH bands, logs output-only fields, and returns prediction/probability/score/level/model version. Threshold .53 was chosen for V2 by validation F1; it is a classification cut point, not a claim that probability scores are calibrated.

API request models also allow optional `isFlaggedFraud` (0/1) and post-balance compatibility fields, but the Streamlit client sends only the seven required values. Extra fields are forbidden. `isFraud` must not be supplied to prediction; it is target data for offline evaluation or a later verified-label field.

## Kafka, Spark, monitoring, labels, and retraining

Stage 1 producer reads a finite held-out raw PaySim slice selected after the model's validation boundary, then loops indefinitely when used as a producer. Each replay receives a newly generated synthetic event ID; this does not make an underlying transaction new. Configured Kafka endpoint/topic are `127.0.0.1:9092`/`fraud-transactions`, one partition for chronological process-local account state. Spark Structured Streaming reads the topic, validates events, performs model inference via the same API app feature/model contract, and writes JSONL results plus Stage 2 registry rows. A checkpoint preserves source progress; output append and checkpoint commit are not an exactly-once transaction. The dashboard only reads its heartbeat/output files and explicitly does not check broker reachability.

Stage 2 SQLite stores prediction registry, label provenance, and submission dispositions. Synthetic test outcomes are separate from genuine trusted labels. Trusted source allowlist and shared token are environment-configured; no authorized source/token is currently set. Retraining requires minimum 100 eligible genuine labels including 10 fraud labels, plus later evaluation/policy/approval gates.

Stage 3 compares a sampled training reference with recent Stage 1 feature vectors using PSI plus KS/two-proportion tests with Benjamini-Hochberg correction. Current persisted report notes 31/34 drifted; drift creates alert/retraining request evidence but never runs training. Performance uses only eligible verified labels and is insufficient at zero such labels. Stage 4 candidate eligibility is gated; Stage 5 promotion/rollback is explicit and guarded; Stage 6 is a separate bounded demo candidate flow explicitly marked ineligible for genuine promotion. There is no automatic drift-to-training-to-deployment connection.

## Issues and limitations to explain

1. **Synthetic data:** all PaySim-derived histories/evaluations are not production banking traffic; impressive held-out metrics do not establish real-world generalization.
2. **Prediction counts:** all current Stage 2 model outputs happen to be class 0; this says nothing about true class balance or accuracy without labels.
3. **Live freshness:** dashboard's history pages are disk-backed. Historical Phase 12 evidence and report timestamps must not be described as current services.
4. **Heartbeat:** a status file can remain after process termination; only a <30-second heartbeat is rendered as fresh, and even that does not independently prove Kafka is delivering.
5. **API process state:** account history is a Python in-memory dictionary, lost on restart and not shared across API workers. Do not run multiple independent API instances expecting consistent history.
6. **Streaming replay:** producer is an unbounded replay of a finite held-out slice; one partition constrains scaling; append/checkpoint behavior can duplicate output after interruption.
7. **No genuine labels:** present Stage 2 source is unconfigured; synthetic label remains ineligible. Genuine retraining is blocked.
8. **No automatic retraining/promotion:** a drift alert/request does not train or deploy. Stage 6 is demonstration only and not promotion eligible.
9. **Environment checks:** current local HTTP probes timed out; live FastAPI/UI availability and Kafka/Spark processes remain unverified here.
10. **Analyzer examples:** API schema example is structurally valid but synthetic, and history features depend on current process state; output is not deterministic across warm/cold/restarted state.

## Safe verification performed and follow-up

This audit read saved files and queried SQLite with read-only URI mode for prediction counts. It checked local HTTP endpoints on ports 8000 and 8501 with a short timeout; both timed out. It did not start services, write SQLite, run a prediction, inspect private label tokens, publish Kafka messages, change model files, train a candidate, or execute tests. For a live demo, run the documented `/health` request in the same environment as the UI, inspect the Stage 1 heartbeat timestamp, and only claim a currently fresh stream if recent heartbeat updates continue. Broker reachability requires a broker-aware check in its configured environment; the UI does not supply one.
