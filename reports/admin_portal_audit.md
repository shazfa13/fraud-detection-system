# Admin Portal Audit

**Repository:** `D:\fraud-detection-system\fraud-detection-system`  
**Audit date:** 2026-11-01  
**Scope:** Static review of current Streamlit admin navigation, page code, services, backend lifecycle modules, and existing tests. No project code or data was changed for this audit.

## 1. Executive summary

The admin portal has **13 sidebar features**, registered in `ui/components.py:171-185`. Navigation is role-aware: `dashboard.py:13-31` registers the admin route set only for the `admin` role; built-in Streamlit navigation is hidden. Each sidebar page uses `configure_page()`, which requires admin by default (`ui/components.py:150-169`); Model Orchestrator adds an explicit role check.

Most monitoring features are **readers of saved evidence**, not a live control plane. Transaction Analyzer can call FastAPI and write prediction records. Model Orchestrator supports explicit review decisions and promotion when a qualifying run exists. Real candidate training is not currently exposed: “Start candidate retraining” is disabled for every branch (`pages/12_Model_Orchestrator.py:398-406`), and current genuine-label requirements are described as blocked. The simulation button runs a separate synthetic demonstration and is not evidence of production readiness.

Most significant overlaps: Overview/System Health/Orchestrator repeat model and API status; Live Monitoring/Drift & Alerts/Delayed Labels/Orchestrator reuse Stage 1–3 evidence at different detail levels; Retraining Lab/Orchestrator/Model Performance repeat lifecycle status; Overview/Transaction Explorer show summary vs searchable history. Most are defensible, but saved vs live status should be clearer.

## 2. Evidence and status method

- **Verified implementation fact:** established by tracing current source calls.
- **Test evidence:** repository test asserts behavior; dashboard tests often mock services and do not establish live service availability.
- **Static observation:** code path exists, but was not invoked in this audit.
- **Recommendation:** proposal only.

“READ-ONLY” describes page behavior even if its source is unavailable. External API requirements and unmet prerequisites are noted separately.

## 3. Complete admin sidebar inventory

The manual sidebar is rendered in this order by `ui/components.py:171-185`. `dashboard.py:31` hides built-in Streamlit navigation. All 13 entries are admin-only in the app; ordinary Users get a separate one-page portal. Welcome, Sign in, and Sign Up routes are registered but are not admin sidebar links. Overview's five shortcut links are not additional features.

| # | Sidebar label | Page | Audience | Purpose / controls | Backend and data |
|---:|---|---|---|---|---|
| 1 | Overview | `pages/admin_overview.py` | Admin only | Prediction counts/share, model/API/monitor status, latest five records, alerts and shortcuts; read-only. | `services.data_service` registry/model/history; `ui.components.cached_api_health`; `src.orchestrator.drift_adapter`; saved logs and Stage 2 SQLite. |
| 2 | Model Orchestrator | `pages/12_Model_Orchestrator.py` | Admin only | Eligibility, saved drift, runs, candidate metrics, approve/reject, promote, simulation button. | `src.orchestrator` overview/eligibility/approval/promotion, saved run artifacts, V2/label summaries, `src.retraining.drift_simulation_workflow`. |
| 3 | Transaction Analyzer | `pages/5_Transaction_Analyzer.py` | Admin only | Preview model features or explicitly submit a single prediction. | `services.api_client`; FastAPI `/state/preview`, `/predict`; resolver and transaction types. |
| 4 | Counterfactual Explainability | `pages/11_Counterfactual_Explainability.py` | Admin only | Exact event ID lookup, fresh score, optional SHAP, bounded hypothetical decision-flip search. | `services.counterfactual_explainability`; resolver/XGBoost; V2 Parquet; feature engineering report; optional SHAP. |
| 5 | Live Monitoring | `pages/2_Live_Monitoring.py` | Admin only | Saved Stage 1 heartbeat/output, Stage 2 counts, Stage 3 report, errors and Phase 12 evidence; no services started. | `services.data_service`; Stage 2 and Stage 3 SQLite; JSON/JSONL/CSV and Phase 12 reports. |
| 6 | Transaction Explorer | `pages/1_Transaction_Explorer.py` | Admin only | Search, source/class/risk/model filters, sorting, duplicate warning, CSV download (up to 500 rows). | `load_prediction_history`; configured API logs, known reports, streaming outputs, Phase 12 CSVs. |
| 7 | Model Performance | `pages/3_Model_Performance.py` | Admin only | V2 contract/threshold, historical V1/V2 metrics, V1 importance chart and lifecycle notes. | V2 metadata/threshold; saved comparison/importance reports; Phase 12 evidence. |
| 8 | Drift & Alerts | `pages/7_Drift_Alerts.py` | Admin only | Freshness, aggregate/per-feature drift, alerts and recorded request; read-only. | `load_stage3_monitor_report`; Stage 3 SQLite report. |
| 9 | Delayed Labels | `pages/6_Delayed_Labels.py` | Admin only | Genuine/synthetic counts, provenance, submission dispositions; read-only. | `load_stage2_label_stats`, `stage2_label_overview`, `src.labels.store`; Stage 2 SQLite. |
| 10 | Retraining Lab | `pages/8_Retraining_Lab.py` | Admin only | Active resolver, Stage 4 eligibility/exclusions, saved Stage 6 demo metrics, rollback inventory; no actions. | `stage4_workflow.collect_eligible_labels`; Stage 1/2 data; saved Stage 6 report; model/promotion-history files. |
| 11 | System Architecture | `pages/9_System_Architecture.py` | Admin only | Stage diagrams, workflow guide and saved validation summary; read-only. | `docs/PROJECT_WORKFLOW.md`, Stage 7 JSON; `read_json_report`. |
| 12 | Reports & Documentation | `pages/10_Reports_Documentation.py` | Admin only | Fixed documentation/report list; download if contained file exists and is under 4 MiB. | `project_path`; project-relative README/docs/report files. |
| 13 | System Health | `pages/4_System_Health.py` | Admin only | Cached API health, active resolver and local artifact presence; no Kafka/Spark check. | FastAPI `/health`; `cached_api_health`; `load_active_model_snapshot`, `local_artifact_status`. |

Navigation controls are in `ui/components.py:187-201`: Log out and admin-only “Refresh dashboard data” (clears Streamlit cached data and reruns; does not write monitor records). No admin user-management page exists. User self-registration is separate in `pages/0_Signup.py`.

## 4. Page-by-page audit

### 4.1 Overview — READ-ONLY

`pages/admin_overview.py:26-178` shows prediction registry totals/share, resolver model, API reachability, latest saved monitoring state, five recent records, alerts, and shortcuts. It calls `prediction_registry_summary`, `load_active_model_snapshot`, `load_prediction_history`, cached API `/health`, and the saved drift adapter. It never predicts or runs monitoring. Counts/classes are model outputs, not verified outcomes. Useful landing summary; its “Real-time transaction monitoring” hero text overstates the saved-summary behavior and should be corrected.

### 4.2 Model Orchestrator — PARTIALLY FUNCTIONAL; real retraining BLOCKED BY REQUIREMENTS; demo SIMULATION-ONLY

`pages/12_Model_Orchestrator.py:246-500` shows model/drift/labels/eligibility, run count, decision reasons, saved candidate runs and evaluation metrics. Refresh rereads evidence. Existing completed candidate runs can enter a review form (decision + reason + confirmation) and then a separate promotion review (reason + confirmation). The real Start button is disabled for BLOCKED, NOT_REQUIRED and remaining paths (`:398-406`), so candidate creation is not launchable here today.

Read paths use `get_orchestrator_overview`, `evaluate_retraining_eligibility`, approval/promotion status and run artifacts. `approval.record_admin_decision` writes an immutable decision; `promotion.promote_approved_candidate` changes the active manifest only after service-side admin/token/integrity gates and writes audit history. The simulation calls `run_simulation()` only on explicit button click, writing unique isolated synthetic artifacts and stopping at a recommendation. It does not call approval/promotion. This audit did not execute it.

High-value lifecycle console, but real training prerequisites and the absent start path must remain explicit. Existing UI/service tests use fixture data and mocks; they do not prove production readiness.

### 4.3 Transaction Analyzer — PARTIALLY FUNCTIONAL (API dependent)

`pages/5_Transaction_Analyzer.py:11-96` accepts seven transaction-time fields and a form choice between “Preview only” and “Submit prediction.” Preview displays the exact ordered model feature row. Prediction displays class/probability/risk/model and threshold context. Preview uses `/state/preview` without consuming/inserting. Submit calls `/predict`; `src/api/app.py:458-475` processes a non-replayed request and logs the prediction/account history. The page makes this side effect explicit. This audit did not send a live request.

### 4.4 Counterfactual Explainability — PARTIALLY FUNCTIONAL

`pages/11_Counterfactual_Explainability.py:21-164` looks up an exact stored event ID, computes a fresh active-model result, optionally displays SHAP contributions, and performs a bounded hypothetical flip search. `lookup_transaction` requires a unique exact event in V2 Parquet and excludes target labels. No row is changed. Depends on data/model availability and optional SHAP; no execution was attempted here. Distinct from Analyzer: stored row explanation vs new raw transaction input.

### 4.5 Live Monitoring — READ-ONLY

`pages/2_Live_Monitoring.py:15-188` shows a persisted Stage 1 heartbeat (30-second freshness), last-five-minute records, history, Stage 2 counts, saved Stage 3 drift/performance/alerts, errors and historical Phase 12 evidence. Uses JSON/status, Stage 2/3 SQLite and saved CSV/report files. Does not connect to a broker or start Spark/monitor. A fresh heartbeat means the status writer updated, not that transactions are arriving. Consider renaming to “Pipeline Evidence.”

### 4.6 Transaction Explorer — READ-ONLY

`pages/1_Transaction_Explorer.py:10-48` reads up to 500 saved records, filters source/class/risk/model, searches visible fields, sorts, warns on duplicate-valued rows, and downloads CSV. `load_prediction_history` reads known logs/CSVs and may combine overlapping sources; it preserves duplicates and does not infer ground truth. Useful investigation view beyond Overview's latest five.

### 4.7 Model Performance — READ-ONLY

`pages/3_Model_Performance.py:10-58` reads V2 metadata/threshold, historical V1/V2 comparison, V1 feature importance and Phase 12 evidence. The metrics are historical synthetic PaySim evaluation, not current live performance. No model is evaluated or trained on page open. Keep separate from per-run Orchestrator evaluation because cohorts/purpose differ.

### 4.8 Drift & Alerts — READ-ONLY

`pages/7_Drift_Alerts.py:12-73` reads Stage 3 report age, model/threshold, window/reference sizes, drift aggregate/features, insufficiency notes, alerts and retraining-request status. It does not recalculate or write evidence. Some saved report versions omit p-values/per-feature detail; the page reports that limitation. Useful focused drilldown that overlaps Stage 3 material on Live Monitoring.

### 4.9 Delayed Labels — READ-ONLY

`pages/6_Delayed_Labels.py:10-48` displays prediction/pending/genuine/synthetic counts, duplicates/conflicts/unmatched counts, provenance groupings and audit dispositions. Backed by Stage 2 SQLite and `src.labels.store`; no label edits or submissions occur on this page. Genuine/synthetic distinction is important; detailed audit is beyond Live Monitoring's repeated counts.

### 4.10 Retraining Lab — READ-ONLY; Stage 6 is SIMULATION-ONLY

`pages/8_Retraining_Lab.py:10-85` shows active model, Stage 4 eligible-label counts/exclusions, saved Stage 6 candidate metrics, promotion status and rollback inventory. It explicitly offers no training/approval/promotion/rollback/monitor action. Stage 4 calls `validate_feature_contract` and `collect_eligible_labels`; Stage 6 values come from a saved report. Rollback directory presence is not proof of validity. Strong overlap with Orchestrator; merge/simplify while retaining a concise genuine-label gate and Stage 6 reference.

### 4.11 System Architecture — READ-ONLY

`pages/9_System_Architecture.py:9-50` explains pipeline stages, reads workflow documentation and saved Stage 7 status. No operational calls. Useful academic/viva view; distinct from downloading source documents.

### 4.12 Reports & Documentation — READ-ONLY

`pages/10_Reports_Documentation.py:9-51` lists fixed docs and saved reports. `project_path` constrains paths to project root; download offered only for present files under 4 MiB. It does not generate reports. Keep as handover library; group by workflow/date for clarity.

### 4.13 System Health — READ-ONLY

`pages/4_System_Health.py:9-32` displays FastAPI `/health` (cached 15 seconds), resolver model/source/threshold and local artifact presence. It does not load or mutate artifact files and does not probe Kafka/Spark. Useful distinct checks, but overlaps Overview; label API, filesystem/resolver and stream heartbeat separately.

## 5. Redundancy and overlap analysis

| Page A | Page B | Overlap and real difference | Recommendation |
|---|---|---|---|
| Overview | System Health | Both show active model/API. Overview adds counts/alerts/recent rows; Health adds resolver/artifact detail. | KEEP SEPARATE; concise summary + detail link. |
| Overview | Transaction Explorer | Latest five vs searchable/filterable/exportable up-to-500 history. | KEEP SEPARATE; mark summary vs investigation. |
| Overview | Model Orchestrator | Model/API/drift summaries repeat; Orchestrator adds eligibility, run, decisions and promotion. | KEEP SEPARATE; avoid candidate details on Overview. |
| Live Monitoring | Drift & Alerts | Both display saved Stage 3 information; Live also covers Stage 1/2 and Phase 12, Drift is Stage 3 detail. | KEEP SEPARATE or make Drift a drilldown in Pipeline Evidence. |
| Live Monitoring | Delayed Labels | Some Stage 2 counts repeat; Labels adds provenance and submission audit. | KEEP SEPARATE; consider removing repeated metrics from Live. |
| Live Monitoring | System Health | Heartbeat/activity vs API and local resolver status are distinct signals. | KEEP SEPARATE; clarify each signal's source. |
| Orchestrator | Retraining Lab | Active model and eligibility/candidate lifecycle repeat; Lab is read-only Stage 4/6/rollback, Orchestrator owns review/promote. | MERGE/SIMPLIFY Lab into Orchestrator. |
| Orchestrator | Model Performance | Per-run candidate evaluation vs fixed historical V1/V2 test benchmark. | KEEP SEPARATE; label cohort and date. |
| Performance | Retraining Lab | Lifecycle readiness/active model repeats; historical comparison and importance are unique. | SIMPLIFY lifecycle badges in Performance; link to Orchestrator. |
| Architecture | Reports & Documentation | Workflow narrative may repeat; one is visual, other downloadable artifacts. | KEEP SEPARATE; curate duplication. |
| Analyzer | Counterfactual Explainability | Both may show a prediction; Analyzer uses new raw input, XAI uses exact stored event plus SHAP/flip search. | KEEP SEPARATE; label provenance. |
| Counterfactual Explainability | User Portal | Both can explain scores; admin view uses stored event and SHAP, user path explains submitted analysis. | KEEP SEPARATE by audience/data contract. |

No backend service was proven unused. Similar stage modules serve distinct sources/actions; removal based only on naming is not justified. There is no admin label-submission or user-management action duplicated across pages.

## 6. Administrator workflow

1. Welcome → Sign in → choose Admin → authenticate against configured admin username/PBKDF2 hash in environment or Streamlit secrets. Successful auth stores role/username in server session state. Admin cannot self-register.
2. Overview summarizes predictions, API, resolver and saved monitoring evidence; prediction classes are estimates.
3. Inspect Explorer for saved records; use Analyzer to preview a new input or explicitly send a prediction (writes history).
4. Use Counterfactual Explainability for an exact stored event and fresh model interpretation; output is not causal evidence.
5. Use Live Monitoring for saved Stage 1/2/3 and Phase 12 evidence, Drift for Stage 3 detail, and Delayed Labels for label provenance. These views do not run Kafka/Spark/monitoring.
6. Review Model Performance for historical V1/V2 synthetic benchmarks; System Health for API/resolver/artifact availability; Architecture for stage explanation.
7. Retraining Lab and Orchestrator expose eligibility and saved Stage 6 demo. PaySim labels do not satisfy genuine label gates. The real candidate Start button is disabled; operational real retraining cannot proceed from this UI.
8. An admin can explicitly run the isolated simulation in Orchestrator. It ends in recommendation for simulated human review only.
9. For an existing eligible real candidate run, Orchestrator has separate approval/rejection and promotion steps. Approval does not promote; promotion has independent service-side identity, token, integrity, reason, lock and audit checks. Current blocked prerequisites and absent candidate-start action mean this is not a currently end-to-end real retraining journey.
10. No admin user activity/account-management feature exists; the User Portal is a role-specific separate experience.

## 7. Security and permissions

### Positive controls observed

- Role-dependent route registration and page-level `require_role` checks are present (`dashboard.py:13-34`; `ui/components.py:150-169`).
- Admin login uses configured environment/Streamlit secrets, PBKDF2-SHA256 verification and bounded work factors. User signup cannot assign admin role; users use SQLite with case-insensitive username uniqueness. No actual credentials or hashes were inspected.
- Approval service obtains identity from authenticated server-side session, requires non-empty reason and fresh token, rejects duplicate decisions and has create-only writes. Promotion independently requires admin, fresh token, explicit reason and validated artifacts before active manifest change.
- Report download paths are project-contained with a size cap. Simulation is a separate admin-page action and does not call approval/promotion.

### Concerns and boundaries

- No MFA, password reset, admin rotation, role-management UI, or admin account-management page was found. Deployment should protect Streamlit and secrets.
- `run_simulation()` has no independent auth check as an internal Python function; its current only UI entry is admin-gated. Add authorization if later exposed through another boundary.
- FastAPI `/state/reset` and `/state/warmup` route definitions (`src/api/app.py:420-444`) show no authentication dependency. These are not sidebar controls, but should be network-restricted or authenticated in a shared deployment because they mutate state/history.
- Analyzer prediction is stateful/logged, but the UI warns before submit. No other destructive admin-page action was found outside explicit promotion.

## 8. Data and status accuracy

- Registered predictions are model estimates, not ground truth. Stage 2 genuine verified labels and synthetic test labels remain separate.
- V1/V2 benchmark metrics are historical PaySim evaluation; Stage 6 and drift simulation are synthetic. Neither represents current live performance or verified fraud outcomes.
- Stage 1 “recent” uses stored timestamps. Heartbeat freshness confirms writer updates only; it does not prove transactions or broker connectivity.
- Drift & Alerts shows saved report fields and freshness; it does not recalculate. Some report formats lack raw per-feature tests.
- API health is a cached 15-second `/health` check, not Kafka/Spark health. Phase 12 pass is historical saved evidence, not current integration poll.
- Resolver snapshot and API-served model are contextualized separately; Analyzer warns if version changes during call.
- **Copy issue:** Overview's “Real-time transaction monitoring” hero contradicts its saved registry/log/report reads. Recommend “Prediction and pipeline overview.”
- **Current prerequisite issue:** real retraining remains blocked/unlaunchable from the page. Do not describe the synthetic demo as satisfying this gate.

## 9. Tests and validation review

Tests were inspected but **not run**. No workflow was triggered, no service started, and no database/model/report data changed.

| Area | Existing tests | What they establish / gap |
|---|---|---|
| Overview/Orchestrator | `test_orchestrator_admin_dashboard.py`, `test_model_orchestrator_page.py` | AppTest UI assertions; external services/write functions mainly mocked. Approval/promotion UI steps and role navigation covered; no live API or production training proof. |
| Auth/User portal | `test_ui_auth.py`, `test_user_portal_contract.py` | Unit/UI role and auth contracts; not deployment penetration testing. |
| Dashboard readers/API client | `test_dashboard_services.py` | Reader/path/schema behavior and mocked HTTP requests; no running FastAPI validation. |
| Stages 1–3 | `test_stage1_streaming.py`, `test_stage2_labels.py`, `test_stage3_monitoring.py`, `test_stage3_validation.py` | Backend stage tests, not all page render interactions. |
| Lifecycle | `test_stage4_workflow.py`, `test_stage5_workflow.py`, `test_stage6_demo.py`, `test_lifecycle_coordinator.py`, Orchestrator dashboard tests | Workflow helpers and fixtures; synthetic Stage 6; no production-population validation from UI. |
| Counterfactual/XAI | `test_counterfactual_explainability.py`, `test_xai_v2.py` | Helpers/schema/explanation; no proof of page execution with installed SHAP and intended dataset row. |
| Explorer, Analyzer, Health, Drift, Labels, Performance, Architecture, Reports | No dedicated page UI test identified in inspected test inventory. | Related service/backend tests exist; page filters/download/render not directly proven. |

## 10. Redundancy scorecard

| Admin feature | Main responsibility | Functional status | Overlap with | Recommendation |
|---|---|---|---|---|
| Overview | Landing summary | READ-ONLY | Health, Orchestrator, Explorer, Drift | KEEP |
| Model Orchestrator | Eligibility, review, promotion, demo | PARTIAL; real training blocked; demo synthetic | Lab, Performance, Drift | KEEP BUT SIMPLIFY |
| Transaction Analyzer | Single-record API prediction | PARTIAL (API required) | Counterfactual | KEEP |
| Counterfactual Explainability | Stored-event interpretation | PARTIAL (dataset/SHAP dependent) | Analyzer, User Portal | KEEP |
| Live Monitoring | Saved Stage 1–3/Phase 12 | READ-ONLY | Drift, Labels, Health | KEEP BUT SIMPLIFY |
| Transaction Explorer | Filter/export saved predictions | READ-ONLY | Overview | KEEP |
| Model Performance | Historical benchmark | READ-ONLY | Orchestrator, Lab | KEEP |
| Drift & Alerts | Saved Stage 3 detail | READ-ONLY | Live, Orchestrator | KEEP |
| Delayed Labels | Label provenance/audit | READ-ONLY | Live | KEEP |
| Retraining Lab | Stage 4/6/rollback snapshot | READ-ONLY; Stage 6 synthetic | Orchestrator, Performance | MERGE/SIMPLIFY |
| System Architecture | Workflow explanation | READ-ONLY | Reports | KEEP |
| Reports & Documentation | Fixed-list downloads | READ-ONLY | Architecture | KEEP |
| System Health | API/model/artifact state | READ-ONLY | Overview, Live | KEEP BUT SIMPLIFY |

## 11. Prioritized recommendations

### KEEP

- Transaction Analyzer: distinct explicit raw-input/API path; retain write warning.
- Transaction Explorer: useful record filtering and export.
- Model Performance: useful historical benchmark when its cohort/synthetic scope remains explicit.
- Drift & Alerts and Delayed Labels: distinct monitor detail and label provenance.
- Counterfactual Explainability: valuable bounded model interpretation with non-causal caveat.
- Architecture and Reports: visual explanation and downloadable handover serve different tasks.

### KEEP BUT SIMPLIFY

- Overview: correct the “Real-time” hero and link to details instead of repeating them.
- Live Monitoring: rename “Pipeline Evidence” and remove repeated Stage 2 counts if linking to Delayed Labels.
- System Health: label API, resolver/filesystem, heartbeat and broker status as distinct signals.
- Orchestrator: keep explicit lifecycle actions; clearly state real candidate creation is not available while blocked.

### MERGE

- Consolidate Retraining Lab's Stage 4 summary and Stage 6 reference into Orchestrator; keep run decisions/promotion in Orchestrator. Preserve stage explanation in docs. This is the strongest page-level consolidation.
- Optionally place Drift detail beneath a Pipeline Evidence hub; retain separate label provenance because the source and workflow are distinct.

### REMOVE

None recommended. Each page supports a different evidence detail, operator task, or academic explanation; shorter navigation alone is insufficient reason to remove it.

### FIX

- Update Overview hero language and make saved-vs-live distinctions visible in menu labels.
- Add page UI coverage for System Health, Drift, Labels, Performance, Architecture, Reports and Explorer filters/download.
- Protect FastAPI state reset/warmup routes in shared deployments.
- Define genuine trusted label source, verified transaction-to-event mapping, and real candidate creation entry point before claiming operational retraining.

### INVESTIGATE

- Verify deployment network/auth controls for `/state/reset` and `/state/warmup`.
- Verify SHAP availability and exact Parquet lookup in target runtime before promising XAI availability.
- Verify Stage 1 heartbeat production in intended runtime without inferring broker/traffic health from the file alone.

## 12. Proposed admin sidebar (proposal only)

**Current order:** Overview; Model Orchestrator; Transaction Analyzer; Counterfactual Explainability; Live Monitoring; Transaction Explorer; Model Performance; Drift & Alerts; Delayed Labels; Retraining Lab; System Architecture; Reports & Documentation; System Health.

**Proposed (12 entries):**

**Dashboard** — Overview  
**Transactions** — Transaction Analyzer; Transaction Explorer; Counterfactual Explainability  
**Monitoring and data quality** — Pipeline Evidence (renamed Live Monitoring); Drift & Alerts; Delayed Labels; System Health  
**Model lifecycle** — Model Performance; Model Orchestrator (with Retraining Lab's concise eligibility and Stage 6 status consolidated here)  
**Project reference** — System Architecture; Reports & Documentation

All remain admin-only. The count drops by one through consolidation, not removal of Stage 4/6 information. Keep Drift and Delayed Labels as separate drilldowns because their evidence differs.

## 13. Final verdict and limitations

The portal is coherent for an academic/portfolio project and clearly separates predictions, saved monitoring, label provenance, historical evaluation, and human lifecycle controls. Strongest working paths from source are read-only evidence review, explicit single-record API prediction, and guarded decision/promotion services for a candidate that already exists. Real candidate training cannot currently be started from this page, and current label prerequisites are blocked. Synthetic drift and Stage 6 results are demonstration-only.

**Limitations:** static source/test inventory only; tests not run; no FastAPI, Kafka, Spark, monitoring, labels, training, simulation, candidate creation, approval, rejection or promotion invoked. Credentials, tokens and label contents were not inspected. Code paths do not establish external service availability.
## Current navigation status addendum — 2026-11-01

The navigation recommendations in the audit body are a snapshot and have since been partly implemented in `ui/components.py` and `dashboard.py`. The current admin sidebar has 12 pages grouped as follows: Dashboard (Admin Overview); Transactions (Transaction Analyzer, Transaction Explorer, Counterfactual Explainability); Monitoring & Data Quality (Pipeline Evidence, Drift & Alerts, Delayed Labels, System Health); Model Lifecycle (Model Performance, Model Orchestrator); Project Reference (System Architecture, Reports & Documentation). The stream evidence page is now labeled **Pipeline Evidence**. The old `pages/8_Retraining_Lab.py` remains on disk but is not registered in navigation. User routing remains separate with **Transaction Risk Check**.

This supersedes the audit body's old 13-entry flat navigation inventory and proposed rename/consolidation recommendations where those changes have landed. The overview page's current hero title is “Fraud Detection Dashboard” and subtitle “Administrative overview”; the old “Real-time transaction monitoring” copy issue is no longer present. Other findings about read-only evidence pages, role checks, API dependencies, and blocked real training remain subject to the source/test limitations recorded in the audit. The current Model Orchestrator still disables candidate training while readiness is blocked; its simulation button is a separate synthetic path. This addendum does not assert that live services are running.
