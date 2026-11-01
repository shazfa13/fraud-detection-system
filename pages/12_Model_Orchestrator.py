"""Administrator view for saved drift evidence and human-reviewed model lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from services.data_service import PROJECT_ROOT, load_latest_stage6_demo_report
from src.orchestrator import approval, promotion
from src.orchestrator.config import RUNS_ROOT, resolve_run_dir, validate_run_id
from src.orchestrator.eligibility import evaluate_retraining_eligibility
from src.orchestrator.overview import get_orchestrator_overview
from src.orchestrator.promotion import get_promotion_status
from ui import auth
from ui.components import (
    configure_page,
    render_footer,
    render_hero,
    render_section_heading,
    status_pill,
)


configure_page("Model Orchestrator", "◈")
if auth.current_role() != "admin":
    st.error("Administrator access is required for model lifecycle information and controls.")
    st.stop()

render_hero(
    "Model Orchestrator",
    "Drift monitoring, candidate review and explicit promotion.",
    "Review saved evidence and isolated candidates. Page refreshes never start a workflow.",
)

st.markdown("""
<style>
.orch-panel { height:100%; padding:1.05rem 1.15rem; border:1px solid #293445; border-radius:14px; background:#111720; }
.orch-panel h3 { margin:.15rem 0 .55rem; font-size:1.02rem; }
.orch-muted { color:#aeb4c2; font-size:.88rem; line-height:1.5; }
.orch-alert { margin:.35rem 0 .9rem; padding:1rem 1.1rem; border:1px solid #784950; border-left:4px solid #e65b67; border-radius:12px; background:#24171d; }
.orch-alert strong { color:#ff9baa; font-size:1.16rem; letter-spacing:.02em; }
.orch-blocked { border-color:#784950; background:#24171d; }
.orch-good { border-color:#23674d; background:#10251d; }
.orch-warn { border-color:#78602d; background:#292313; }
.orch-count { color:#f5f3ff; font-size:1.65rem; font-weight:700; }
div[data-testid="stProgressBar"] { margin:.35rem 0 .8rem; }
@media(max-width:800px){ .orch-panel { margin-bottom:.45rem; } }
</style>
""", unsafe_allow_html=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _safe_run_dir(run_id: str) -> Path | None:
    try:
        valid_id = validate_run_id(run_id)
        run_dir = resolve_run_dir(valid_id, RUNS_ROOT)
        if (RUNS_ROOT / valid_id).is_symlink() or not run_dir.is_dir():
            return None
        return run_dir
    except (TypeError, ValueError, OSError):
        return None


def _stage_state(run_dir: Path, folder: str, report_name: str, status_key: str) -> str:
    report = _read_json(run_dir / folder / report_name)
    if report is not None:
        return str(report.get(status_key, "AVAILABLE"))
    return "INCOMPLETE" if (run_dir / folder).exists() else "NOT_STARTED"


def _metric_value(metrics: Mapping[str, Any], name: str) -> str:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unavailable"
    if name in {"tp", "fp", "fn", "tn", "rows"}:
        return f"{int(value):,}"
    return f"{float(value):.3f}"


def _render_metrics(title: str, values: Mapping[str, Any]) -> None:
    st.markdown(f"**{title}**")
    keys = ("precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn")
    labels = {"f1": "F1", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "tp": "True positives",
              "fp": "False positives", "fn": "False negatives", "tn": "True negatives"}
    for offset in range(0, len(keys), 5):
        columns = st.columns(min(5, len(keys) - offset))
        for column, key in zip(columns, keys[offset:offset + 5]):
            column.metric(labels.get(key, key.title()), _metric_value(values, key))


def _render_run(run: Mapping[str, Any]) -> None:
    run_id = run.get("run_id")
    if not isinstance(run_id, str):
        return
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        st.error("This run record has an invalid ID or unsafe directory and was not opened.")
        return

    dataset_state = "READY" if all((run_dir / "dataset" / name).is_file() for name in (
        "candidate_dataset.parquet", "dataset_metadata.json", "dataset_validation.json", "split_summary.json")) else _stage_state(run_dir, "dataset", "dataset_validation.json", "status")
    training_report = _read_json(run_dir / "candidate" / "training_report.json")
    candidate_state = str(training_report.get("status", "INCOMPLETE")) if training_report else _stage_state(run_dir, "candidate", "training_report.json", "status")
    evaluation_report = _read_json(run_dir / "evaluation" / "evaluation_report.json")
    evaluation_state = str(evaluation_report.get("evaluation_status", "INCOMPLETE")) if evaluation_report else _stage_state(run_dir, "evaluation", "evaluation_report.json", "evaluation_status")
    try:
        admin_decision = approval.get_admin_decision(run_id, run_root=RUNS_ROOT)
        decision_error = None
    except Exception as exc:
        admin_decision = None
        decision_error = f"{type(exc).__name__}: decision record is unavailable or invalid."
    promotion_state = get_promotion_status(run_id, run_root=RUNS_ROOT, project_root=PROJECT_ROOT)

    with st.expander(f"{run_id} · {run.get('workflow_status', 'UNKNOWN')}", expanded=False):
        st.caption(f"Created: {run.get('created_at_utc', 'unknown')} · Current status: {run.get('workflow_status', 'unknown')}")
        stages = st.columns(5)
        for column, title, state in zip(
            stages,
            ("Data readiness", "Candidate model", "Evaluation", "Review decision", "Model activation"),
            (dataset_state, candidate_state, evaluation_state,
             str(admin_decision.get("decision", "NOT_REVIEWED")) if admin_decision else "NOT_REVIEWED",
             str(promotion_state.get("status", "UNKNOWN"))),
        ):
            column.metric(title, state)

        if decision_error:
            st.error(decision_error)
        if evaluation_report:
            st.markdown("#### Candidate evaluation")
            models = evaluation_report.get("models", {})
            candidate_model = models.get("candidate", {}) if isinstance(models, dict) else {}
            active_model = models.get("active", {}) if isinstance(models, dict) else {}
            st.write(
                f"Status: **{evaluation_report.get('evaluation_status', 'UNKNOWN')}** · "
                f"Recommendation: **{evaluation_report.get('decision_status', 'UNKNOWN')}** · "
                f"Candidate: {candidate_model.get('model_version', 'unknown')} · "
                f"Active model: {active_model.get('model_version', 'unknown')}"
            )
            metrics = evaluation_report.get("metrics", {})
            if isinstance(metrics, dict):
                if isinstance(metrics.get("active"), Mapping):
                    _render_metrics("Active model", metrics["active"])
                if isinstance(metrics.get("candidate"), Mapping):
                    _render_metrics("Candidate model", metrics["candidate"])
            comparison = evaluation_report.get("comparison")
            if isinstance(comparison, Mapping):
                st.markdown("**Candidate minus active comparison**")
                st.json(comparison)
            validation = _read_json(run_dir / "evaluation" / "evaluation_validation.json")
            if validation and validation.get("passed") is not True:
                st.warning(validation.get("failure_reason") or "Evaluation validation reported warnings or failure.")
            for reason in evaluation_report.get("decision_reasons", []) if isinstance(evaluation_report.get("decision_reasons"), list) else []:
                st.caption(f"Evaluation note: {reason}")
        elif evaluation_state not in {"NOT_STARTED", "INCOMPLETE"}:
            st.info(f"Evaluation report status: {evaluation_state}.")

        if admin_decision:
            st.info(f"Administrator decision: {admin_decision.get('decision')} · {admin_decision.get('reason', 'No reason recorded')}. Approval does not activate a model.")
        elif evaluation_report and evaluation_report.get("evaluation_status") == "COMPLETED" and not decision_error:
            approval_review_key = f"model_orchestrator_approval_{run_id}"
            if st.button("Review evaluation for approval or rejection", key=f"model_orchestrator_review_{run_id}"):
                try:
                    review = approval.get_approval_review(run_id, run_root=RUNS_ROOT)
                    st.session_state[approval_review_key] = {"review_token": review["review_token"], "evaluation_status": review["evaluation_status"]}
                except Exception as exc:
                    st.session_state.pop(approval_review_key, None)
                    st.error(f"Candidate review failed closed: {type(exc).__name__}: {exc}")
                st.rerun()
            approval_review = st.session_state.get(approval_review_key)
            if isinstance(approval_review, dict) and approval_review.get("review_token"):
                st.warning("Approval records a human decision only. It does not activate or promote the candidate.")
                with st.form(f"model_orchestrator_decision_form_{run_id}"):
                    decision = st.radio("Administrator decision", ("APPROVED", "REJECTED"), key=f"model_orchestrator_decision_{run_id}")
                    reason = st.text_area("Required decision reason", key=f"model_orchestrator_decision_reason_{run_id}")
                    confirmed = st.checkbox(f"Confirm recording {decision} for this candidate", key=f"model_orchestrator_decision_confirm_{run_id}")
                    submitted = st.form_submit_button("Record approval or rejection")
                if submitted:
                    if not confirmed or not reason.strip():
                        st.error("A non-empty reason and explicit confirmation are required.")
                    else:
                        try:
                            fresh = approval.get_approval_review(run_id, run_root=RUNS_ROOT)
                            saved = approval.record_admin_decision(run_id, decision, reason,
                                                                   review_token=fresh["review_token"], run_root=RUNS_ROOT)
                            st.session_state["model_orchestrator_last_action"] = f"{saved['decision']} decision recorded for {run_id}. Promotion remains separate."
                            st.session_state.pop(approval_review_key, None)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Decision was not recorded: {type(exc).__name__}: {exc}")

        if admin_decision and admin_decision.get("decision") == "APPROVED" and promotion_state.get("status") != "PROMOTED":
            promotion_review_key = f"model_orchestrator_promotion_{run_id}"
            if st.button("Review approved candidate for promotion", key=f"model_orchestrator_promotion_review_{run_id}"):
                try:
                    review = promotion.get_promotion_review(run_id, run_root=RUNS_ROOT, project_root=PROJECT_ROOT)
                    if review.get("promotion_ready") is not True:
                        raise ValueError("; ".join(review.get("validation_failures", [])) or "Promotion readiness was not confirmed.")
                    st.session_state[promotion_review_key] = {
                        "promotion_token": review["promotion_token"],
                        "active_model_version": review["active_model_version"],
                        "candidate_version": review["candidate_version"],
                    }
                except Exception as exc:
                    st.session_state.pop(promotion_review_key, None)
                    st.error(f"Promotion review failed closed: {type(exc).__name__}: {exc}")
                st.rerun()
            promotion_review = st.session_state.get(promotion_review_key)
            if isinstance(promotion_review, dict) and promotion_review.get("promotion_token"):
                st.error(
                    f"Promotion changes the active-model manifest from "
                    f"{promotion_review.get('active_model_version')} to {promotion_review.get('candidate_version')}."
                )
                with st.form(f"model_orchestrator_promotion_form_{run_id}"):
                    promotion_reason = st.text_area("Required promotion reason", key=f"model_orchestrator_promotion_reason_{run_id}")
                    promotion_confirmed = st.checkbox("I explicitly confirm changing the active-model manifest", key=f"model_orchestrator_promotion_confirm_{run_id}")
                    promote = st.form_submit_button("Promote approved candidate")
                if promote:
                    if not promotion_confirmed or not promotion_reason.strip():
                        st.error("Explicit confirmation and a non-empty promotion reason are required.")
                    else:
                        try:
                            fresh = promotion.get_promotion_review(run_id, run_root=RUNS_ROOT, project_root=PROJECT_ROOT)
                            if fresh.get("promotion_ready") is not True:
                                raise ValueError("; ".join(fresh.get("validation_failures", [])) or "Promotion readiness was not confirmed.")
                            result = promotion.promote_approved_candidate(
                                run_id, review_token=fresh["promotion_token"], promotion_reason=promotion_reason,
                                run_root=RUNS_ROOT, project_root=PROJECT_ROOT,
                            )
                            st.session_state["model_orchestrator_last_action"] = f"Promotion service confirmed: {result['status']} for {run_id}."
                            st.session_state.pop(promotion_review_key, None)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Promotion did not complete: {type(exc).__name__}: {exc}")


if st.session_state.get("model_orchestrator_last_action"):
    st.success(st.session_state.pop("model_orchestrator_last_action"))

try:
    overview = get_orchestrator_overview()
    eligibility = evaluate_retraining_eligibility()
except Exception as exc:
    st.error(f"Orchestrator summaries are unavailable: {type(exc).__name__}: {exc}")
    st.stop()

active = overview.get("active_model", {})
drift = overview.get("latest_drift", {})
labels = overview.get("label_readiness", {})
history = overview.get("run_history", {})
runs = history.get("runs", []) if isinstance(history, dict) else []

if st.button("Refresh eligibility and saved evidence", key="model_orchestrator_refresh"):
    try:
        overview = get_orchestrator_overview()
        eligibility = evaluate_retraining_eligibility()
        st.rerun()
    except Exception as exc:
        st.error(f"Refresh failed: {type(exc).__name__}: {exc}")

eligibility_status = eligibility.get("decision_status", "BLOCKED")
drift_value = ("UNAVAILABLE" if drift.get("status") == "NO_SAVED_REPORT" else
               "UNAVAILABLE" if drift.get("status") != "SAVED_REPORT" else
               "DRIFT DETECTED" if drift.get("drift_decision") is True else
               "NO DRIFT" if drift.get("drift_decision") is False else "UNKNOWN")
label_status = "VERIFIED READY" if labels.get("strict_retraining_ready") is True else "NOT READY"
render_section_heading("Model Readiness", "Current active model, saved monitoring evidence, genuine-label availability and eligibility.")
summary = (
    ("Active model", active.get("model_version") or "Unavailable"),
    ("Integrity", active.get("integrity", {}).get("status", "UNKNOWN")),
    ("Drift status", drift_value),
    ("Label readiness", label_status),
    ("Retraining eligibility", eligibility_status),
    ("Orchestrator runs", f"{len(runs):,}"),
)
summary_columns = st.columns(6)
for column, (title, value) in zip(summary_columns, summary):
    column.metric(title, value)

st.caption(f"Active artifact: {(active.get('model_artifact_path') or 'unavailable')} · manifest: {'present' if active.get('manifest_present') else 'absent' if active.get('status') == 'VALID' else 'unavailable'}")

render_section_heading("Monitoring snapshot", "Latest saved monitoring evidence; viewing this page does not run a monitor or change its data.")
if drift.get("status") == "NO_SAVED_REPORT":
    st.markdown(status_pill("Monitoring data unavailable · no saved report", "unavailable"), unsafe_allow_html=True)
    st.info("No drift alert is available because no saved monitoring report was found.")
elif drift.get("status") == "SAVED_REPORT":
    detected = drift.get("drift_decision")
    drift_tone = "blocked" if detected is True else "healthy" if detected is False else "unavailable"
    drift_label = "DRIFT DETECTED" if detected is True else "NO DRIFT DETECTED" if detected is False else "DRIFT STATUS UNKNOWN"
    st.markdown(status_pill(drift_label, drift_tone), unsafe_allow_html=True)
    affected = drift.get("drifted_features") if isinstance(drift.get("drifted_features"), list) else []
    monitoring_window = drift.get("monitoring_window")
    origins = monitoring_window.get("data_origin_counts") if isinstance(monitoring_window, Mapping) else None
    origin_label = drift.get("monitoring_basis") or (", ".join(str(key) for key in origins) if isinstance(origins, Mapping) and origins else "unavailable")
    if detected is True:
        st.markdown(
            f"<div class='orch-alert'><strong>DRIFT DETECTED</strong><div class='orch-muted'>"
            f"{len(affected)} affected feature{'s' if len(affected) != 1 else ''} · saved evidence requires review</div></div>",
            unsafe_allow_html=True,
        )
    fact_cols = st.columns(4)
    fact_cols[0].markdown("**Detected**  \n" + str(drift.get("report_timestamp") or "Unavailable"))
    fact_cols[1].markdown("**Affected features**  \n" + str(len(affected)))
    window_summary = "Unavailable"
    if isinstance(monitoring_window, Mapping):
        start = monitoring_window.get("window_start")
        end = monitoring_window.get("window_end")
        if start or end:
            window_summary = f"{start or 'unknown start'} → {end or 'unknown end'}"
    fact_cols[2].markdown("**Monitoring window**  \n" + window_summary)
    fact_cols[3].markdown("**Data origin**  \n" + str(origin_label))
    st.caption(f"Saved report run: {drift.get('report_run_id') or 'unavailable'}")
    scores = drift.get("feature_scores")
    with st.expander(f"Affected features and scores ({len(affected)})", expanded=False):
        if affected:
            st.markdown("\n".join(f"- `{name}`" for name in affected))
        else:
            st.caption("No affected feature names were included in this saved report.")
        if isinstance(scores, list) and scores:
            st.dataframe(pd.DataFrame(scores), use_container_width=True, hide_index=True)
    with st.expander("Reference and monitoring window details", expanded=False):
        st.markdown("**Reference window**")
        st.json(drift.get("reference_window"))
        st.markdown("**Monitoring window**")
        st.json(drift.get("monitoring_window"))
    warnings = drift.get("warnings", []) if isinstance(drift.get("warnings"), list) else []
    with st.expander(f"Saved report limitations ({len(warnings)})", expanded=False):
        if warnings:
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.caption("No limitations were recorded in the saved summary.")
    if detected is True:
        st.error("Data drift detected. Review the saved evidence and eligibility decision.")
else:
    st.markdown(status_pill("Monitoring data unavailable", "unavailable"), unsafe_allow_html=True)
    st.warning(drift.get("error", "The latest saved monitoring report could not be read."))

render_section_heading("Retraining Eligibility", "This decision uses genuine verified labels and cannot be overridden in the dashboard.")
tone = "healthy" if eligibility_status == "ELIGIBLE" else "historical" if eligibility_status == "NOT_REQUIRED" else "blocked"
st.markdown(status_pill(f"{eligibility_status}", tone), unsafe_allow_html=True)
configuration = eligibility.get("configuration", {})
genuine_count = labels.get("genuinely_verified_labels")
minimum_count = configuration.get("minimum_genuine_labels")
count_known = isinstance(genuine_count, int) and not isinstance(genuine_count, bool)
minimum_known = isinstance(minimum_count, int) and not isinstance(minimum_count, bool) and minimum_count > 0
count_text = f"{genuine_count:,}" if count_known else "Unknown"
minimum_text = f"{minimum_count:,}" if minimum_known else "Unknown"
st.markdown(f"**Genuine labels**  \n`{count_text} / {minimum_text}`")
progress = min(max(genuine_count / minimum_count, 0.0), 1.0) if count_known and minimum_known else 0.0
st.progress(progress, text="Verified label minimum" if count_known and minimum_known else "Verified-label count unavailable")
matching_key = labels.get("matching_key", "UNKNOWN")
mapping_state = "confirmed" if matching_key == "event_id" else "not established"
st.caption(f"Minimum: {minimum_text} · Matching key: {matching_key} · Transaction-to-event ID mapping: {mapping_state}")
st.caption(f"Eligibility rule: {eligibility.get('rule_version', 'unknown')}")
if labels.get("simulated_labels") or labels.get("phase9_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH":
    st.warning("PaySim labels are simulated and do not count as verified ground truth.")
label_reasons = labels.get("readiness_reasons", []) if isinstance(labels.get("readiness_reasons"), list) else []
with st.expander("Label provenance details", expanded=False):
    st.caption(f"Recorded label type: {labels.get('phase9_label_type') or 'unavailable'}")
    source_counts = labels.get("source_type_counts")
    st.markdown("**Label source type counts**")
    if isinstance(source_counts, Mapping):
        st.json(source_counts)
    else:
        st.caption("Source type counts are unavailable.")
    st.markdown("**Label readiness reasons**")
    if label_reasons:
        for reason in label_reasons:
            st.markdown(f"- {reason}")
    else:
        st.caption("No additional label readiness reasons were recorded.")
reasons = eligibility.get("reasons", []) if isinstance(eligibility.get("reasons"), list) else []
if reasons:
    st.markdown("**Why retraining is blocked**")
    for reason in reasons[:2]:
        st.markdown(f"- {reason}")
with st.expander(f"Complete decision reasons ({len(reasons)})", expanded=False):
    for reason in reasons:
        st.markdown(f"- {reason}")
warnings = eligibility.get("warnings", []) if isinstance(eligibility.get("warnings"), list) else []
with st.expander(f"Eligibility warnings ({len(warnings)})", expanded=False):
    if warnings:
        for warning in warnings:
            st.warning(str(warning))
    else:
        st.caption("No additional warnings were recorded.")

render_section_heading("Candidate Model", "Real candidate training remains separate from the synthetic demonstration below.")
if runs:
    st.caption(f"{len(runs):,} saved real candidate run(s) are available for review below.")
else:
    st.info("No real candidate run is available. A candidate cannot be shown until verified data and a supported run are provided.")
if eligibility_status == "BLOCKED":
    st.button("Start candidate retraining", disabled=True, key="model_orchestrator_start_blocked")
    st.caption("Unavailable: current genuine-label eligibility is BLOCKED. Simulated labels do not count toward this requirement.")
elif eligibility_status == "NOT_REQUIRED":
    st.button("Start candidate retraining", disabled=True, key="model_orchestrator_start_not_required")
    st.caption("Unavailable: saved evidence does not currently require retraining.")
else:
    st.button("Start candidate retraining", disabled=True, key="model_orchestrator_start_source_unavailable")
    st.caption("Unavailable from this page: the real workflow requires an existing run plus a trusted candidate dataset and provenance. This UI has no supported real-data/run creation input path; eligibility alone does not create a candidate.")

render_section_heading("Genuine label availability", "Read-only source, mapping and exclusion details used to explain eligibility.")
try:
    from src.labels import config as labels_config
    from src.monitoring import config as monitor_config
    from src.retraining.stage4_workflow import collect_eligible_labels, validate_feature_contract

    _eligible_rows, label_audit = collect_eligible_labels(
        database_path=labels_config.STAGE2_DB_PATH,
        prediction_log_path=monitor_config.STAGE1_JSONL_PATH,
        features=validate_feature_contract(),
        trusted_sources=labels_config.trusted_label_sources(),
        token_configured=bool(labels_config.trusted_label_token()),
    )
    del _eligible_rows
    readiness_cols = st.columns(4)
    readiness_cols[0].metric("Labels examined", label_audit["labels_seen"])
    readiness_cols[1].metric("Eligible genuine labels", label_audit["eligible_label_count"])
    readiness_cols[2].metric("Eligible fraud labels", label_audit["eligible_fraud_count"])
    readiness_cols[3].metric("Minimum labels / fraud", f"{label_audit['minimum_verified_labels']} / {label_audit['minimum_verified_fraud_labels']}")
    with st.expander("Label sources and exclusion reasons"):
        st.json({"label_source_breakdown": label_audit["label_source_breakdown"],
                 "label_state_breakdown": label_audit["label_state_breakdown"],
                 "exclusions": label_audit["exclusions"],
                 "trusted_source_authorization_configured": label_audit["trusted_source_authorization_configured"]})
except Exception as exc:
    st.warning(f"Detailed label readiness is unavailable: {type(exc).__name__}: {exc}")

demo, _demo_error = load_latest_stage6_demo_report(PROJECT_ROOT)
if demo:
    render_section_heading("Saved demonstration candidate", "Synthetic PaySim evidence only; it is not a real candidate or verified performance result.")
    demo_metrics = demo.get("candidate_evaluation_metrics") or {}
    active_metrics = (demo.get("same_rows_v2_comparison") or {}).get("active_metrics") or {}
    demo_cols = st.columns(4)
    demo_cols[0].metric("Demo status", demo.get("status", "Unknown"))
    demo_cols[1].metric("Candidate threshold", demo_metrics.get("threshold", "Unavailable"))
    demo_cols[2].metric("Evaluation rows", demo_metrics.get("rows", "Unavailable"))
    demo_cols[3].metric("Eligible for real activation", "No" if demo.get("eligible_for_genuine_promotion") is False else "Unknown")
    st.caption(f"Candidate: {demo.get('model_version', 'Unknown')} · generated: {demo.get('generated_at_utc', 'Unknown')} · report: {demo.get('_report_path', 'Unavailable')}")
    if demo_metrics and active_metrics:
        st.dataframe(pd.DataFrame([
            {"Model": "Active V2", **active_metrics},
            {"Model": "Synthetic demonstration candidate", **demo_metrics},
        ]), use_container_width=True, hide_index=True)
    demo_promotion = demo.get("promotion") or {}
    st.caption(f"Demonstration activation status: {demo_promotion.get('status', 'not attempted')}")
manifest = PROJECT_ROOT / "models" / "active_model_manifest.json"
history_dir = PROJECT_ROOT / "models" / "promotion_history"
backup_dir = PROJECT_ROOT / "models" / "backup"
active_dir = PROJECT_ROOT / "models" / "active"
with st.expander("Activation and rollback inventory"):
    st.caption(f"Activation manifest: {'present' if manifest.is_file() else 'absent'} · decision history: {'present' if history_dir.is_dir() else 'absent'}")
    try:
        backup_files = sorted(path.name for path in backup_dir.iterdir() if path.is_file()) if backup_dir.is_dir() else []
        active_files = sorted(path.name for path in active_dir.iterdir() if path.is_file()) if active_dir.is_dir() else []
        st.caption("Saved backup files: " + (", ".join(backup_files) if backup_files else "none"))
        st.caption("Legacy active files: " + (", ".join(active_files) if active_files else "none"))
        if history_dir.is_dir():
            st.caption("Activation history entries: " + (", ".join(sorted(item.name for item in history_dir.iterdir())) or "none"))
    except OSError:
        st.caption("Activation inventory could not be read.")
    st.caption("Inventory checks only; a listed file is not a validation that rollback is safe.")

render_section_heading("Simulation Demonstration", "An isolated synthetic workflow for drift detection and candidate review.")
st.info(
    "This demonstration uses synthetic fraud labels and controlled simulated drift to demonstrate candidate model retraining and evaluation. "
    "It does not represent genuine verified fraud data, production banking retraining or deployment readiness. "
    "The candidate is stored separately and is never automatically promoted."
)
simulation_key = "model_orchestrator_simulation_result"
if st.button("Run Simulation Demonstration", key="model_orchestrator_run_simulation"):
    if st.session_state.get("model_orchestrator_simulation_running"):
        st.warning("A simulation run is already in progress.")
    else:
        st.session_state["model_orchestrator_simulation_running"] = True
        try:
            from src.retraining.drift_simulation_workflow import run_simulation
            with st.spinner("Running isolated drift simulation, candidate training, and evaluation…"):
                result = run_simulation()
            st.session_state[simulation_key] = result
            st.success(f"Simulation completed · run {result.get('run_id', 'unknown')}.")
        except Exception as exc:
            st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
        finally:
            st.session_state["model_orchestrator_simulation_running"] = False

simulation_result = st.session_state.get(simulation_key)
if isinstance(simulation_result, Mapping):
    st.markdown("#### Simulation results")
    drift_result = _read_json(Path(simulation_result.get("artifact_paths", {}).get("drift_report", "")))
    evaluation_result = _read_json(Path(simulation_result.get("artifact_paths", {}).get("evaluation_report", "")))
    recommendation_result = _read_json(Path(simulation_result.get("artifact_paths", {}).get("recommendation_report", "")))
    summary_result = _read_json(Path(simulation_result.get("artifact_paths", {}).get("final_summary", "")))
    run_id = simulation_result.get("run_id", "Unavailable")
    if drift_result:
        drifted = int(drift_result.get("features_drifted", 0))
        checked = int(drift_result.get("features_tested", 0))
        drift_cols = st.columns(4)
        drift_cols[0].metric("Drift status", "DRIFT DETECTED" if drift_result.get("drift_detected") else "NO DRIFT")
        drift_cols[1].metric("Affected features", f"{drifted} / {checked}")
        drift_cols[2].metric("Drift percentage", f"{float(drift_result.get('drift_rate', 0)):.1%}")
        drift_cols[3].metric("Run identifier", str(run_id))
    else:
        st.warning("The drift report is unavailable.")
    if evaluation_result:
        metric_map = evaluation_result.get("metrics", {})
        rows = []
        for model, key in (("V2", "V2"), ("Candidate", "candidate")):
            metrics = metric_map.get(key, {})
            rows.append({"Model": model, "Precision": metrics.get("precision"), "Recall": metrics.get("recall"),
                         "F1-score": metrics.get("f1"), "PR-AUC": metrics.get("pr_auc"), "ROC-AUC": metrics.get("roc_auc"),
                         "False positives": metrics.get("fp"), "False negatives": metrics.get("fn")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("The evaluation report is unavailable.")
    if recommendation_result:
        recommendation = recommendation_result.get("recommendation", "UNKNOWN")
        if recommendation == "CANDIDATE_RECOMMENDED_FOR_REVIEW":
            st.success("CANDIDATE_RECOMMENDED_FOR_REVIEW · Recommended for human review within this simulation only.")
        else:
            st.warning(f"{recommendation} · Candidate is not recommended for review.")
        for reason in recommendation_result.get("reasons", []):
            st.caption(str(reason))
    artifact_paths = simulation_result.get("artifact_paths", {})
    st.markdown("**Simulation artifacts**")
    for label, key in (("Candidate model", "candidate_model"), ("Drift report", "drift_report"),
                       ("Evaluation report", "evaluation_report"), ("Comparison CSV", "comparison_csv"),
                       ("Recommendation report", "recommendation_report"), ("Final simulation summary", "final_summary")):
        path = Path(artifact_paths.get(key, "")) if artifact_paths.get(key) else None
        if path is not None and path.is_file():
            st.code(str(path), language=None)
    if summary_result:
        hashes = summary_result.get("v2_hashes_unchanged", {})
        safety_pass = (summary_result.get("active_model") == "V2" and all(hashes.values())
                       and summary_result.get("real_orchestrator_runs_unchanged") is True
                       and summary_result.get("automatic_promotion") == "NO")
        st.success("Safety audit passed: V2 remains active and unchanged; real run history is unchanged; no promotion occurred." if safety_pass
                   else "Safety audit did not confirm every required invariant; inspect the final summary report.")

render_section_heading("Evaluation History and Decisions", "Evaluation, review decisions, and model activation are separate human-controlled actions.")
if not runs:
    st.markdown("<div class='empty-state'><strong>No orchestrator runs</strong><br/>No candidate evaluation, administrator decision, or promotion is available.</div>", unsafe_allow_html=True)
else:
    for run in runs:
        _render_run(run)

render_footer()
