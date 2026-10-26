"""Verified project documentation and report downloads."""
from __future__ import annotations

import streamlit as st

from services.data_service import MAX_TEXT_FILE_BYTES, PROJECT_ROOT, project_path
from ui.components import configure_page, render_footer, render_hero, render_section_heading

configure_page("Reports & Documentation", "▤")
render_hero("Reports & Documentation", "Project handover library",
            "Browse existing documentation and saved reports. Each item is checked on disk before a download is offered.")

items = [
    ("Project README", "README.md", "Project scope and setup"),
    ("Workflow and architecture guide", "docs/PROJECT_WORKFLOW.md", "System relationships and safeguards"),
    ("Results and limitations", "docs/PROJECT_RESULTS_AND_LIMITATIONS.md", "Saved evaluation context and remaining limits"),
    ("Demo runbook", "docs/DEMO_RUNBOOK.md", "Local commands and runtime safety"),
    ("Presentation outline", "docs/PROJECT_PRESENTATION.md", "Academic handover and viva notes"),
    ("Stage 1 streaming guide", "docs/STAGE1_STREAMING.md", "Producer, consumer, outputs, and checkpoints"),
    ("Stage 2 delayed-label guide", "docs/STAGE2_LABELS.md", "Label provenance and idempotency"),
    ("Stage 3 monitoring guide", "docs/STAGE3_MONITORING.md", "Drift sampling and monitoring behavior"),
    ("Stage 4 retraining guide", "docs/STAGE4_RETRAINING.md", "Eligibility and candidate training gates"),
    ("Stage 5 promotion guide", "docs/STAGE5_MODEL_PROMOTION.md", "Validation, approval, and rollback"),
    ("Stage 6 demonstration guide", "docs/STAGE6_RETRAINING_DEMONSTRATION.md", "Synthetic candidate limitations"),
    ("Stage 7 validation guide", "docs/STAGE7_END_TO_END_VALIDATION.md", "End-to-end validation evidence"),
    ("Stage 7 saved validation report", "reports/stage7_end_to_end_validation_20261025T150813Z.json", "Recorded report; not regenerated here"),
    ("V1 vs V2 saved comparison", "reports/model_v1_vs_v2_comparison.json", "Historical held-out evaluation metrics"),
    ("V2 saved training report", "reports/model_v2_training_report.json", "Training split and evaluation metadata"),
]

render_section_heading("Available files", "Paths are project-relative and downloads contain the saved file bytes only.")
for title, relative, detail in items:
    path = project_path(PROJECT_ROOT, relative)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"{detail} · `{relative}`")
        if path is None or not path.is_file():
            st.caption("Unavailable: file not found.")
            continue
        try:
            if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                st.caption("Unavailable for dashboard download: file exceeds the 4 MiB read limit.")
                continue
            content = path.read_bytes()
        except OSError:
            st.caption("Unavailable: file could not be read.")
            continue
        st.download_button("Download saved file", content, file_name=path.name,
                           mime="application/json" if path.suffix.lower() == ".json" else "text/plain",
                           key=f"download:{relative}")

render_footer()
