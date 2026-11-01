"""Human-readable overview of the implemented system boundaries."""
from __future__ import annotations

import streamlit as st

from services.data_service import PROJECT_ROOT, read_json_report
from ui.components import configure_page, render_footer, render_hero, render_section_heading


configure_page("System Architecture", "⌘")
render_hero("System Architecture", "How the system fits together",
            "A high-level view of transaction scoring, outcome verification, monitoring and human-controlled model decisions.")

render_section_heading("Transaction scoring", "A request is validated, converted to the model's ordered inputs, scored and recorded.")
request, api, model, result = st.columns(4)
for column, title, detail in (
    (request, "Transaction", "Single-record input or streaming event"),
    (api, "Prediction service", "Validation, feature construction and account context"),
    (model, "Active model", "Shared resolver selects the current model bundle"),
    (result, "Prediction record", "Score returned and saved with available context"),
):
    with column:
        st.markdown(f"<div class='architecture-node'><h4>{title}</h4><p>{detail}</p></div>", unsafe_allow_html=True)

render_section_heading("Outcomes and monitoring", "Verified outcomes and feature drift are tracked as separate evidence.")
labels, monitor, review = st.columns(3)
with labels:
    st.markdown("#### Delayed outcomes")
    st.write("Authorized label sources are checked and recorded. Synthetic test outcomes remain separate from genuine verified outcomes.")
with monitor:
    st.markdown("#### Monitoring evidence")
    st.write("Saved feature-drift and performance reports support review. Viewing the portal does not run monitoring.")
with review:
    st.markdown("#### Model decisions")
    st.write("Eligibility gates candidate data. Evaluation, human review and activation are separate decisions; the dashboard does not promote automatically.")

render_section_heading("Validation and limitations")
validation, error = read_json_report(PROJECT_ROOT, "reports/stage7_end_to_end_validation_20261025T150813Z.json")
if validation:
    st.caption(f"Saved integration evidence: {validation.get('status', 'Unknown')} · completed {validation.get('completed_at_utc', 'timestamp unavailable')}.")
else:
    st.caption("Saved integration evidence is unavailable.")
st.info("Historical PaySim results and synthetic demonstrations are not genuine real-world fraud evidence. The portal does not start streaming services or run real retraining automatically.")
render_footer()
