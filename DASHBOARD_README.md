# Dashboard guide

The Streamlit app provides public welcome/sign-in/sign-up pages, a user transaction-risk form, and a protected admin workspace. It reads local project evidence and calls FastAPI when a user or administrator requests a transaction analysis. It does not start or supervise Kafka, Spark, or monitoring jobs.

## Start the dashboard

From the repository root in Windows PowerShell:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install pyarrow
.\venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

For an admin login, configure `FRAUD_ADMIN_USERNAME` and `FRAUD_ADMIN_PASSWORD_HASH` in the process environment or Streamlit secrets. Generate a salted PBKDF2-SHA256 hash without placing the password in shell history:

```powershell
$adminHash = & .\venv\Scripts\python.exe -c "from ui.auth import password_hash; import getpass; print(password_hash(getpass.getpass('Admin password: ')))"
$env:FRAUD_ADMIN_USERNAME = 'admin'
$env:FRAUD_ADMIN_PASSWORD_HASH = $adminHash.Trim()
& .\venv\Scripts\python.exe -m ui.auth_diagnostics
```

Then launch Streamlit from the same process environment. The diagnostic reports configuration state without printing the credential/hash. For persistent local setup, place the values in `.streamlit/secrets.toml`; keep that file private. User accounts are created through Sign Up and stored in `data/user_accounts.sqlite3`. Registration cannot create an Admin. The project also accepts `FRAUD_API_BASE_URL` to point the dashboard at a different FastAPI base URL; the default is `http://127.0.0.1:8000`.

The Welcome, Sign in, and Sign Up routes do not need FastAPI. Transaction Risk Check and Transaction Analyzer require the API. Admin evidence pages read local files/databases. The sidebar's API indicator is a short-cached `/health` request, not a Kafka or Spark check.

## Navigation and roles

Built-in Streamlit navigation is hidden. Public routes are available before authentication. After sign-in, a regular user sees **Transaction Risk Check** (the sidebar label is **Analyze a transaction**). An admin sees this grouped navigation:

| Group | Page | Main purpose |
|---|---|---|
| Dashboard | Admin Overview | Summary of prediction registry, model/API status, saved monitor evidence and recent records |
| Transactions | Transaction Analyzer | Preview features or explicitly submit a prediction for new input |
| Transactions | Transaction Explorer | Search/filter saved prediction outputs and download a bounded filtered CSV |
| Transactions | Counterfactual Explainability | Look up a stored event, score it with the current resolver bundle, view optional SHAP and bounded hypothetical changes |
| Monitoring & Data Quality | Pipeline Evidence | Review Stage 1 heartbeat/output, Stage 2 counts, Stage 3 report and historical integration evidence |
| Monitoring & Data Quality | Drift & Alerts | Inspect the saved Stage 3 drift result, feature evidence, alerts and retraining request |
| Monitoring & Data Quality | Delayed Labels | Inspect label-source provenance, pending/resolved counts and submission audit dispositions |
| Monitoring & Data Quality | System Health | Inspect API `/health`, resolver selection and local model artifact presence |
| Model Lifecycle | Model Performance | Read saved V1/V2 benchmark metrics and model metadata |
| Model Lifecycle | Model Orchestrator | Inspect readiness and runs; record review/promote an existing eligible candidate; run the isolated synthetic drift simulation |
| Project Reference | System Architecture | View the workflow diagram and saved validation summary |
| Project Reference | Reports & Documentation | Download allowlisted, existing project reports and guides |

The older source file `pages/8_Retraining_Lab.py` is not registered in the current navigation. Its existence does not make it an active page. The current Model Orchestrator page has no enabled real-candidate training action while eligibility is blocked. Its simulation button runs a separate synthetic demonstration.

Sidebar controls include **Log out** and (for admins) **Refresh dashboard data**. Refresh clears Streamlit's cached data and reruns the page; it does not create monitoring records or predictions. Page-level role checks protect the admin pages as well as role-dependent route registration.

## Transaction Risk Check

The user form collects seven transaction-time values:

- Simulated hour `step` and a supported `type`.
- `amount`.
- Sender reference
ameOrig` and before-transaction balance `oldbalanceOrg`.
- Receiver reference
ameDest` and before-transaction balance `oldbalanceDest`.

The portal sends these fields to FastAPI `/analyze` with an idempotency key. The API validates them, builds the ordered 34 model inputs from request values and strictly prior account history, and returns the prediction plus the exact feature vector/history status used for a local explanation. The current transaction is stored in account history after scoring. The portal checks that the API's model version and threshold remain stable across analysis; SHAP explanation is optional and a prediction remains available if explanation fails.

The result distinguishes fraud probability from the thresholded class. The probability is the classifier output; the binary result is 1 when probability meets the selected model threshold (currently V2, 0.53 under the checked-in configuration). A risk score is probability times 100 and the risk band is a configured display band. These are model estimates, not investigation outcomes. Use synthetic account references; the application cannot fetch real bank history.

## Admin page behavior

- **Admin Overview:** summarizes stored prediction decisions and saved monitor evidence. It does not itself poll the stream or run a prediction.
- **Transaction Analyzer:** accepts seven raw fields. **Preview only** calls `/state/preview` and does not insert history. **Submit prediction** calls the prediction API, mutates account history, and appends to the configured API JSONL log.
- **Transaction Explorer:** combines bounded, known prediction logs/reports while retaining source names. It does not load outcomes or infer truth. Search and filters apply to the displayed records; CSV download exports the filtered view.
- **Counterfactual Explainability:** exact transaction ID lookup in the processed V2 Parquet data, fresh score under the current resolver, optional local SHAP, and an approximate bounded amount/type flip search. It does not change stored rows.
- **Pipeline Evidence:** shows persisted heartbeat/output/error records, Stage 2 registry and Stage 3 report, plus historical Phase 12 results. The heartbeat alone does not prove that Kafka is reachable or events are arriving.
- **Drift & Alerts:** reads saved monitoring evidence; it does not rerun the detector. A drift signal is not proof of model degradation and does not launch training.
- **Delayed Labels:** read-only registry and provenance view. It does not submit or edit labels.
- **Model Performance:** saved historical evaluation, not a fresh evaluation against current traffic.
- **Model Orchestrator:** shows readiness reasons and existing run evidence. Review and promotion are separate actions with identity, decision, artifact/report binding and integrity checks. Current genuine-label requirements block real candidate creation. The simulation button writes separate demonstration artifacts and never promotes them.
- **System Health:** reports a cached API health result and local resolver/artifact state. It does not probe Kafka/Spark.
- **System Architecture:** explanatory diagram and saved report links; it does not inspect running services.
- **Reports & Documentation:** offers a fixed allowlist of files only when present and within the page's size/path restrictions. It downloads saved bytes; it does not regenerate reports.

## Data sources and status meaning

The UI uses the active-model resolver for displayed version/threshold information; it does not load a separate model for regular transaction requests. Prediction counts are class decisions, not confirmed fraud. Stage 2 and Stage 3 cards are persisted SQLite state. Pipeline Evidence and Transaction Explorer read JSON/JSONL/CSV files. Model Performance reads saved reports. Some reports and integration checks are historical. Timestamps and freshness labels should be read before presenting a value as current.

The current implementation uses local SQLite, JSONL and files. The admin allowlist is not a substitute for protecting access to the host, and the API itself has no authentication layer. Bind it to loopback for local demonstrations.

## Persistent effects

| Action | Effect |
|---|---|
| Sign Up | Creates a local user account record |
| Transaction Risk Check / `/analyze` | Stores the analyzed event/history and prediction for idempotent retry; writes prediction log |
| Analyzer Preview | Reads prior history and builds features without inserting the current event |
| Analyzer Submit prediction | Scores, stores the event/history and appends prediction log |
| Refresh dashboard data | Clears UI cache only |
| Orchestrator simulation | Writes a unique synthetic candidate and reports under simulation artifact directories |
| Label API or monitor CLI | Writes their respective Stage 2 or Stage 3 SQLite records; these are separate from ordinary dashboard browsing |

For service startup and data prerequisites see [the demo runbook](docs/DEMO_RUNBOOK.md). For the user-facing screen walkthrough and examples see [the UI user guide](docs/UI_USER_GUIDE.md).
