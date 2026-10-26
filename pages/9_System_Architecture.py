"""Native visualization of the implemented system and workflow boundaries."""
from __future__ import annotations

import streamlit as st

from services.data_service import PROJECT_ROOT, read_json_report
from ui.components import configure_page, render_footer, render_hero, render_section_heading

configure_page("System Architecture", "⌘")
render_hero("System Architecture", "How the workflows connect",
            "The prediction path, delayed-label monitoring path, and guarded model lifecycle are separate workflows with explicit hand-offs.")

render_section_heading("Prediction and streaming path", "Implemented path: request/source → validated event → ordered features → shared model bundle → recorded result.")
a, b, c, d, e = st.columns(5)
for col, number, title, detail in zip((a, b, c, d, e), ("01", "02", "03", "04", "05"),
    ("Inputs", "FastAPI", "Feature builder", "XGBoost", "Outputs"),
    ("Transaction API or Stage 1 producer", "/predict and /health", "34 ordered features; process-local history", "Resolver-selected V1 or V2", "Response/log or Kafka → Spark JSONL")):
    with col:
        st.markdown(f"<div class='architecture-node'><small>{number}</small><h4>{title}</h4><p>{detail}</p></div>", unsafe_allow_html=True)
st.markdown("<div class='architecture-strip'><span>Stage 1 producer</span><b>→</b><span>Kafka</span><b>→</b><span>Spark consumer</span><b>→</b><span>Shared features + resolver</span><b>→</b><span>Prediction JSONL + Stage 2 registry</span></div>", unsafe_allow_html=True)

render_section_heading("Delayed labels, monitoring, and lifecycle", "Dashed hand-offs represent later review; drift evidence never triggers training or promotion on its own.")
left, mid, right = st.columns(3)
with left:
    st.markdown("#### Stage 2 · delayed labels")
    st.write("Authorized label submissions are validated and retained in SQLite. Synthetic test labels remain separate from genuine verified labels; identical retries are idempotent and conflicts are audited.")
with mid:
    st.markdown("#### Stage 3 · monitoring")
    st.write("Stage 1 feature vectors are compared with a model-specific training reference. Eligible verified labels are required for performance metrics. Reports, alerts, and retraining requests are persisted for review.")
with right:
    st.markdown("#### Stages 4–6 · model lifecycle")
    st.write("Stage 4 gates on genuine labels. Stage 5 validates and requires human approval to promote or roll back. Stage 6 is a synthetic demonstration and is explicitly ineligible for genuine promotion.")

render_section_heading("Current validation boundaries", "Reused from the Phase 14 workflow guide; distinguish built/tested workflows from genuine live evidence.")
workflow_path = PROJECT_ROOT / "docs" / "PROJECT_WORKFLOW.md"
if workflow_path.is_file():
    try:
        if workflow_path.stat().st_size <= 4 * 1024 * 1024:
            with st.expander("Read the saved Phase 14 workflow guide"):
                st.markdown(workflow_path.read_text(encoding="utf-8")[:12000])
    except OSError:
        st.caption("The Phase 14 workflow guide could not be read.")
workflow, error = read_json_report(PROJECT_ROOT, "reports/stage7_end_to_end_validation_20261025T150813Z.json")
if workflow:
    st.caption(f"Saved Stage 7 validation report exists at reports/stage7_end_to_end_validation_20261025T150813Z.json · status {workflow.get('status', 'not recorded')}.")
else:
    st.caption("The saved Stage 7 report is unavailable; see docs/PROJECT_WORKFLOW.md for documented workflow details.")
st.info("PaySim labels and Stage 6 candidate metrics are synthetic. Genuine label collection and production-population validation remain unresolved requirements. This dashboard does not start Kafka/Spark or run retraining.")
render_footer()
