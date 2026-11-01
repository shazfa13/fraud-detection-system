# Streamlit UI user guide

This guide describes the current role-aware Streamlit interface in `dashboard.py`, `ui/`, `pages/`, and `services/`. The UI consumes the API and saved project evidence; it does not start Kafka, Spark, or the monitoring CLI.

## Roles and navigation

The public routes are Welcome, Sign in, and Sign Up. A registered User is routed to **Transaction Risk Check**. An Admin gets the grouped sidebar below:

| Group | Active page |
|---|---|
| Dashboard | Admin Overview |
| Transactions | Transaction Analyzer; Transaction Explorer; Counterfactual Explainability |
| Monitoring & Data Quality | Pipeline Evidence; Drift & Alerts; Delayed Labels; System Health |
| Model Lifecycle | Model Performance; Model Orchestrator |
| Project Reference | System Architecture; Reports & Documentation |

The app hides Streamlit's built-in navigation and renders links in `ui/components.py`; protected pages also check the current role. `pages/8_Retraining_Lab.py` remains in the repository but is not registered in current navigation. The old page name **Live Monitoring** is now **Pipeline Evidence**. The admin sidebar refresh clears cached dashboard data and reruns the page; it does not trigger a monitor or inference call.

User registration stores a salted PBKDF2-SHA256 hash and unique case-insensitive username key in `data/user_accounts.sqlite3`. Users cannot select an admin role. Admin credentials come from `FRAUD_ADMIN_USERNAME` and `FRAUD_ADMIN_PASSWORD_HASH` in process environment or Streamlit secrets. See [DASHBOARD_README.md](../DASHBOARD_README.md) for PowerShell setup and secret-safe diagnostics.

## User: Transaction Risk Check

The form collects transaction time step, supported transaction category, amount, sender account reference, sender's pre-transaction balance, receiver account reference, and receiver's pre-transaction balance. It sends the corresponding API fields `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, and `oldbalanceDest` to FastAPI `/analyze`.

The API uses a shared active model bundle and strict 34-feature order. It reads strictly earlier events for those account references from local SQLite, scores before adding the current event, then records it for subsequent history. The portal checks the API model version and threshold before and after analysis and uses the returned feature vector for its local explanation. It shows the probability, thresholded class, risk score/level, model version and an explanatory summary. If optional explanation fails, it can still show the model result.

Use synthetic account references. There is no connection to a financial institution or automatic retrieval of real account history. The prediction is not verified ground truth.

## Admin: page-by-page guide

### Admin Overview

Landing summary of prediction registry totals/classes, resolver model, API reachability, saved Stage 3 state, recent records, alerts and shortcuts. Counts are predictions, not outcomes. The `/health` result is briefly cached. The page does not start or check Kafka/Spark.

### Transaction Analyzer

Inputs are the seven API fields above. The type selector comes from saved feature-engineering configuration. Choose **Preview only** or **Submit prediction**, then **Run selected action**. Preview calls `/state/preview` and returns the ordered vector without inserting the event. Submit calls `/predict` and writes to account history and the configured API prediction JSONL. The result is class, probability, probability-times-100 risk score, risk level, model version and threshold context. No class/probability is guaranteed for example input.

### Transaction Explorer

Reads a bounded set of known API logs and saved streaming/integration reports. Source names are preserved; overlapping rows are not silently deduplicated. Controls include source, text search, predicted class, risk level, model version, sort order, and a filtered CSV download. Missing fields may display placeholders. A class prediction is never presented as confirmed fraud.

### Counterfactual Explainability

Accepts an exact event ID from the processed V2 Parquet data. It computes a fresh score using the shared active resolver, optionally displays SHAP contributions, and searches a bounded set of amount/type changes for a hypothetical decision flip. It does not display a stored historical prediction or change the underlying row. SHAP describes model contributions, not real-world cause; the counterfactual is approximate and may not describe a realistic transaction.

### Pipeline Evidence

This is the current page label for `pages/2_Live_Monitoring.py`. It reads Stage 1 status, predictions/errors, Stage 2 counts, Stage 3 report and historical Phase 12 integration evidence. Fresh heartbeat means the status writer recently updated; it is not a broker reachability or event-arrival check. Browsing is read-only; the CLI suggested by the page persists a new monitor run.

### Drift & Alerts

Reads the latest saved Stage 3 monitoring report, its drift summary/per-feature evidence when available, alerts, and retraining-request status. It does not recompute drift. Some historical reports predate detailed per-feature evidence. Drift is an input-distribution signal and does not imply model-performance decline or automatically start retraining.

### Delayed Labels

Read-only Stage 2 database view of registered predictions, pending labels, genuine/synthetic label counts, provenance and duplicate/conflict/rejection dispositions. It does not submit labels. The local synthetic test labels are not genuine investigative outcomes and do not satisfy retraining gates.

### Model Performance

Shows saved V1/V2 metadata and same-split benchmark reports, thresholds, and feature-importance evidence. These are historical synthetic PaySim metrics, not a fresh evaluation or current stream accuracy. The page does not train or switch models.

### Model Orchestrator

Shows current model/readiness inputs, saved monitoring and label eligibility, existing candidate-run evaluations and any review/promotion evidence. A human decision is separate from promotion; the promotion service rechecks admin identity, review token, artifact/report hashes, current active state and reason before changing the manifest. In the current workspace there are no real orchestrator runs and genuine-label readiness is blocked. The real candidate-start control is disabled in the page. The separate **Run Simulation Demonstration** action executes a synthetic drift/candidate experiment and writes isolated artifacts; it is not genuine retraining and cannot auto-promote.

### System Health

Displays a short-cached FastAPI `/health` result, active resolver bundle, and local artifact presence. It does not probe Kafka/Spark or prove that a streaming process is handling events.

### System Architecture

Shows a static project workflow diagram and saved Stage 7 validation summary. It is explanatory content, not a runtime status check.

### Reports & Documentation

Offers a fixed list of reports and guides if files exist and pass project-root/size checks. Downloads saved files; it does not generate new reports. A newly added Markdown file may not appear until the UI allowlist is updated.

## Sample input

The API includes this schema-shaped example; it is illustrative, not a row from a customer or the dataset:

```json
{
  "step": 5,
  "type": "TRANSFER",
  "amount": 1500.0,
  "nameOrig": "C1234567890",
  "oldbalanceOrg": 5000.0,
  "nameDest": "C9876543210",
  "oldbalanceDest": 0.0,
  "isFlaggedFraud": 0
}
```

Do not add `isFraud`. The target is not available at prediction time and extra request fields are rejected. Post-transaction balances may be accepted by FastAPI for compatibility but are not used to build model features.

## Data and freshness

Pages combine data from distinct sources: SQLite registry rows, model metadata, JSONL/CSV prediction outputs, cached API health, and saved reports. Read each timestamp and source label. Historical Phase 12 checks are not current Kafka/Spark health; prediction counts are not fraud incidence; probabilities are model scores, not confirmed outcomes. Local snapshot numbers in older dated project notes describe only the snapshot on their stated date and should not be repeated as current counts.

## Further reading

- [Dashboard setup, navigation and side effects](../DASHBOARD_README.md)
- [Windows runbook and service prerequisites](DEMO_RUNBOOK.md)
- [System workflow and lifecycle](PROJECT_WORKFLOW.md)
- [Model and simulation results](PROJECT_RESULTS_AND_LIMITATIONS.md)
- [Synthetic drift simulation details](DRIFT_SIMULATION.md)
