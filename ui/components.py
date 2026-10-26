"""Consistent visual styling and small reusable Streamlit components."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from services.api_client import api_base_url, get_health


CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 85% -5%, rgba(49, 164, 177, .08), transparent 35%), #0a0d14; color: #eef0f7; }
[data-testid="stSidebar"] { background: #0d111a; border-right: 1px solid #222a38; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c8cad8; }
[data-testid="stSidebarNav"] { display: none; }
.block-container { max-width: 1480px; padding-top: 1.8rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.025em; }
[data-testid="stMetric"] { background: linear-gradient(145deg, #151b27, #10151e); border: 1px solid #293445; padding: 1rem 1.1rem; border-radius: 14px; }
[data-testid="stMetricLabel"] { color: #a9acbd; }
.hero { padding: 1.45rem 1.65rem; margin: .2rem 0 1.4rem; border: 1px solid #36334f; border-radius: 18px; background: radial-gradient(circle at 85% 5%, rgba(130,91,255,.16), transparent 36%), linear-gradient(125deg,#171b29,#11151e 65%); }
.hero-kicker { color: #b49eff; text-transform: uppercase; letter-spacing: .14em; font-size: .76rem; font-weight: 700; }
.hero-title { color: #f4f3ff; font-size: 2.1rem; line-height: 1.12; font-weight: 760; margin: .38rem 0 .42rem; }
.hero-copy { color: #b9bbca; font-size: 1rem; margin: 0; max-width: 850px; }
.section-copy { color: #999caf; margin-top: -.55rem; margin-bottom: .8rem; font-size: .92rem; }
.architecture-strip { display: flex; justify-content: space-around; align-items: center; gap: .55rem; flex-wrap: wrap; padding: 1.1rem; background: #111720; border: 1px solid #293445; border-radius: 14px; color: #e8e7f5; }
.architecture-strip b { color: #54c3c2; font-size: 1.2rem; }
.architecture-node { min-height: 150px; height: 100%; padding: .85rem; border: 1px solid #30394a; border-radius: 13px; background: linear-gradient(150deg,#161d29,#11161f); }
.architecture-node small { color: #54c3c2; font-weight: 700; letter-spacing: .08em; }
.architecture-node h4 { margin: .45rem 0; color: #e8e8f2; }
.architecture-node p { color: #aeb4c2; font-size: .86rem; line-height: 1.45; }
.status-row { display: flex; gap: .55rem; flex-wrap: wrap; padding-top: .8rem; }
.pill { display: inline-flex; padding: .32rem .65rem; border: 1px solid #38394b; border-radius: 999px; color: #d8d9e4; background: #171823; font-size: .78rem; }
.pill-healthy { border-color: #23674d; color: #8ae0b4; background: #10251d; }
.pill-offline { border-color: #744040; color: #ffaaaa; background: #281718; }
.pill-warning { border-color: #78602d; color: #f5d28b; background: #292313; }
.pill-historical { border-color: #514183; color: #c4b7ff; background: #1d1930; }
.pill-stale { border-color: #765729; color: #f3ce88; background: #251f15; }
.pill-blocked { border-color: #784950; color: #ffc2c9; background: #28171c; }
.pill-failed { border-color: #8c4141; color: #ffb1b1; background: #2b1719; }
.pill-unavailable { border-color: #48505e; color: #b8c0cc; background: #191f28; }
.empty-state { border: 1px dashed #424355; border-radius: 14px; padding: 1.2rem 1.3rem; color: #b7b9c8; background: #12131b; line-height: 1.65; }
.footer { border-top: 1px solid #292a38; margin-top: 2.4rem; padding-top: .8rem; color: #8f91a2; font-size: .78rem; }
div[data-testid="stDataFrame"] { border: 1px solid #292a3a; border-radius: 12px; }
button[kind="primary"] { background: #7657e8; border-color: #7657e8; }
@media (max-width: 800px) { .hero-title { font-size: 1.7rem; } .block-container { padding-left: 1rem; padding-right: 1rem; } }
</style>
"""


@st.cache_data(ttl=15, show_spinner=False)
def cached_api_health() -> tuple[bool, dict[str, Any] | None, str | None]:
    """Cache a live API health check briefly; this is not Kafka/Spark health."""
    try:
        return True, get_health(base_url=api_base_url()), None
    except Exception as exc:
        return False, None, str(exc)


def configure_page(title: str, icon: str = "◈") -> None:
    """Apply page metadata, theme details, and consistent sidebar identity."""
    st.set_page_config(page_title=f"{title} · Fraud Intelligence", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<div class='hero-kicker'>Fraud Intelligence</div>", unsafe_allow_html=True)
        st.caption("Adaptive Transaction Risk Monitoring")
        st.markdown("<div class='section-copy' style='margin: .9rem 0 .25rem'>CONTROL CENTER</div>", unsafe_allow_html=True)
        navigation = (
            ("dashboard.py", "Overview", ":material/dashboard:"),
            ("pages/5_Transaction_Analyzer.py", "Transaction Analyzer", ":material/search:"),
            ("pages/2_Live_Monitoring.py", "Live Monitoring", ":material/sensors:"),
            ("pages/1_Transaction_Explorer.py", "Transaction Explorer", ":material/manage_search:"),
            ("pages/3_Model_Performance.py", "Model Performance", ":material/monitoring:"),
            ("pages/7_Drift_Alerts.py", "Drift & Alerts", ":material/notifications_active:"),
            ("pages/6_Delayed_Labels.py", "Delayed Labels", ":material/label:"),
            ("pages/8_Retraining_Lab.py", "Retraining Lab", ":material/precision_manufacturing:"),
            ("pages/9_System_Architecture.py", "System Architecture", ":material/account_tree:"),
            ("pages/10_Reports_Documentation.py", "Reports & Documentation", ":material/description:"),
            ("pages/4_System_Health.py", "System Health", ":material/health_and_safety:"),
        )
        for path, label, glyph in navigation:
            st.page_link(path, label=label, icon=glyph, use_container_width=True)
        st.divider()
        ok, health, _ = cached_api_health()
        model = health.get("model_version", "Unknown") if ok and health else "Unavailable"
        st.markdown(status_pill("API online" if ok else "API offline", "healthy" if ok else "offline"), unsafe_allow_html=True)
        st.caption(f"API-served model: {model}")
        st.caption("API reachability is checked separately from stream activity.")
        if st.button("Refresh dashboard data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Academic demonstration · synthetic PaySim data")


def render_refresh_control() -> None:
    """Offer a manual refresh without changing backend or report files."""
    with st.sidebar:
        if st.button("Refresh dashboard data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_hero(title: str, subtitle: str, description: str) -> None:
    """Render a compact high-contrast page header."""
    st.markdown(
        "<div class='hero'>"
        f"<div class='hero-kicker'>{escape(subtitle)}</div>"
        f"<div class='hero-title'>{escape(title)}</div>"
        f"<p class='hero-copy'>{escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, detail: str | None = None) -> None:
    st.markdown(f"### {title}")
    if detail:
        st.markdown(f"<div class='section-copy'>{escape(detail)}</div>", unsafe_allow_html=True)


def render_metric(label: str, value: str, caption: str = "", tone: str = "neutral") -> None:
    """Render a Streamlit metric with a small accessible status cue."""
    st.metric(label, value, help=caption or None)
    if caption:
        st.caption(caption)


def status_pill(label: str, tone: str = "neutral") -> str:
    css = {"healthy": "pill-healthy", "offline": "pill-offline", "warning": "pill-warning",
           "historical": "pill-historical", "stale": "pill-stale", "blocked": "pill-blocked",
           "failed": "pill-failed", "unavailable": "pill-unavailable"}.get(tone, "")
    return f"<span class='pill {css}'>{escape(label)}</span>"


def render_footer() -> None:
    st.markdown(
        "<div class='footer'>Academic and portfolio demonstration using synthetic PaySim data. "
        "Historical evaluation results are not live performance. No automatic retraining or production banking operation is represented.</div>",
        unsafe_allow_html=True,
    )
