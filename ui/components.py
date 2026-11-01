"""Consistent visual styling and small reusable Streamlit components."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from services.api_client import api_base_url, get_health
from ui.auth import current_role, logout, require_role


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
.welcome-hero { position: relative; overflow: hidden; padding: clamp(2.2rem, 7vw, 5.4rem) clamp(1.4rem, 6vw, 5rem); margin: .25rem 0 1.25rem; border: 1px solid #38344f; border-radius: 24px; background: radial-gradient(ellipse at 78% 15%, rgba(111,83,235,.27), transparent 34%), radial-gradient(ellipse at 20% 100%, rgba(49,164,177,.12), transparent 38%), linear-gradient(125deg,#171b29,#0f131d 68%); box-shadow: 0 22px 70px rgba(0,0,0,.25); }
.welcome-hero h1 { color: #f5f3ff; font-size: clamp(2.35rem, 6.3vw, 4.5rem); line-height: 1.02; letter-spacing: -.055em; margin: 1rem 0 .8rem; }
.welcome-hero h1 span { color: #a995ff; }
.welcome-hero > p { color: #d8d5ef; font-size: clamp(1.08rem, 2.2vw, 1.42rem); margin: .35rem 0; }
.welcome-copy { max-width: 690px; color: #b9bbca; line-height: 1.65; margin-top: 1rem; }
.feature-card { height: 100%; min-height: 172px; padding: 1.25rem; border: 1px solid #30394a; border-radius: 17px; background: linear-gradient(150deg,#171e2b,#111720); box-shadow: 0 12px 32px rgba(0,0,0,.12); }
.feature-card h3 { color: #f0edff; margin: .45rem 0; font-size: 1.08rem; }
.feature-card p { color: #aeb4c2; font-size: .91rem; line-height: 1.55; }
.feature-icon { color: #a995ff; font-size: 1.55rem; }
.welcome-foot { color: #9296a6; text-align: center; font-size: .8rem; margin-top: 2rem; }
.stApp:has(.fs-topbar) { background: radial-gradient(ellipse at 0% 0%, rgba(111,40,239,.38), transparent 33%), radial-gradient(ellipse at 100% 0%, rgba(61,49,219,.3), transparent 33%), radial-gradient(ellipse at 100% 100%, rgba(106,35,225,.22), transparent 32%), #070c24; }
.fs-topbar { display: flex; align-items: center; justify-content: space-between; min-height: 76px; margin: -1.4rem -2.5rem 2.4rem; padding: .75rem clamp(1.5rem, 5vw, 4.8rem); border-bottom: 1px solid rgba(116,115,209,.16); background: linear-gradient(90deg,rgba(54,25,129,.32),rgba(8,13,39,.12)); }
.fs-brand { display: flex; align-items: center; gap: .75rem; color: #f7f5ff; }
.fs-brand-mark { display: grid; place-items: center; width: 48px; height: 48px; border: 2px solid #a76eff; border-radius: 16px; color: #c291ff; font-size: 2.1rem; line-height: 1; text-shadow: 0 0 18px rgba(172,111,255,.75); }
.fs-brand strong { display: block; font-size: 1.27rem; letter-spacing: -.025em; }
.fs-brand small { display: block; color: #b2b8e8; font-size: .86rem; }
.fs-tagline { color: #c1c4ed; font-size: .9rem; }
.fs-tagline b { color: #be8aff; padding: 0 .45rem; }
.fs-intro { padding: .15rem 0 0; }
.fs-badge { display: inline-block; padding: .42rem .9rem; color: #c9b4ff; border: 1px solid rgba(144,83,255,.4); border-radius: 999px; background: rgba(69,42,178,.26); font-size: .8rem; letter-spacing: .04em; }
.fs-badge b { color: #9e87ff; padding: 0 .45rem; }
.fs-intro h1 { margin: 1rem 0 .85rem; color: #f7f7ff; font-size: clamp(2.4rem, 4.4vw, 3.55rem); line-height: 1.1; letter-spacing: -.05em; }
.fs-intro h1 span { color: #b080ff; text-shadow: 0 0 24px rgba(137,76,255,.28); }
.fs-lead { max-width: 570px; color: #d0d5f3; font-size: 1.08rem; line-height: 1.65; }
.fs-highlights { display: grid; grid-template-columns: repeat(4,1fr); margin: 1.55rem 0 1.05rem; }
.fs-highlights > div { min-height: 93px; display: flex; flex-direction: column; align-items: center; gap: .45rem; text-align: center; border-right: 1px solid rgba(116,115,209,.2); }
.fs-highlights > div:last-child { border-right: 0; }
.fs-highlight-icon { display: grid; place-items: center; width: 48px; height: 48px; color: #c17aff; border: 1px solid rgba(123,76,255,.5); border-radius: 50%; background: radial-gradient(circle at 35% 20%,rgba(114,75,255,.4),rgba(21,26,87,.45)); font-size: 1.45rem; box-shadow: 0 0 22px rgba(113,64,255,.12); }
.fs-highlights strong { color: #e5e6ff; font-size: .77rem; font-weight: 550; line-height: 1.25; }
.fs-illustration { width: 100%; margin-top: .45rem; }
.fs-illustration svg { display: block; width: 100%; max-height: 320px; overflow: visible; }
.fs-footnote { display: flex; align-items: center; gap: .65rem; margin-top: .1rem; color: #c1c8ee; font-size: .76rem; }
.fs-footnote span { color: #70e4d1; font-size: 1.35rem; }
.fs-footnote i { height: 15px; border-left: 1px solid rgba(172,178,225,.35); }
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fs-panel-icon) { min-height: 650px; display: flex; flex-direction: column; justify-content: center; padding: clamp(1.4rem, 3vw, 2.6rem); border: 1px solid rgba(111,105,228,.55); border-radius: 22px; background: linear-gradient(145deg,rgba(20,27,68,.85),rgba(9,16,46,.88)); box-shadow: 0 18px 70px rgba(0,0,0,.2), inset 0 1px rgba(255,255,255,.025); }
.fs-panel-icon { color: #a978ff; text-align: center; font-size: 3.1rem; line-height: 1.05; text-shadow: 0 0 20px rgba(160,104,255,.65); }
.fs-panel-title { margin: .55rem 0 .3rem; color: #f5f4ff; text-align: center; font-size: clamp(1.45rem, 2.2vw, 1.9rem); letter-spacing: -.035em; }
.fs-panel-title span { color: #a77aff; }
.fs-panel-copy { margin-bottom: 1.35rem; color: #c9ccef; text-align: center; font-size: .94rem; }
div.st-key-welcome_user_login, div.st-key-welcome_signup, div.st-key-welcome_admin_login { margin: .28rem 0; }
div.st-key-welcome_user_login button, div.st-key-welcome_signup button, div.st-key-welcome_admin_login button { position: relative; min-height: 105px; padding: .8rem 1.05rem .8rem 5.3rem; border-radius: 18px; text-align: left; white-space: normal; color: #f7f6ff; background: rgba(18,25,63,.76); border: 1px solid rgba(119,101,227,.58); box-shadow: none; transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease; }
div.st-key-welcome_user_login button { background: linear-gradient(110deg,#6840f4,#5035db); border-color: #7654ff; box-shadow: 0 8px 26px rgba(76,48,224,.22); }
div.st-key-welcome_signup button { border-color: rgba(162,103,255,.7); }
div.st-key-welcome_user_login button:hover, div.st-key-welcome_signup button:hover, div.st-key-welcome_admin_login button:hover { transform: translateY(-2px); border-color: #ba8cff; box-shadow: 0 10px 25px rgba(85,62,205,.2); }
div.st-key-welcome_user_login button::before, div.st-key-welcome_signup button::before, div.st-key-welcome_admin_login button::before { position: absolute; left: 1.05rem; top: 50%; transform: translateY(-50%); display: grid; place-items: center; width: 58px; height: 58px; border-radius: 50%; color: #e3c6ff; background: rgba(193,148,255,.24); font-size: 1.65rem; }
div.st-key-welcome_user_login button::before { content: "♙"; }
div.st-key-welcome_signup button::before { content: "＋"; background: rgba(119,53,210,.3); }
div.st-key-welcome_admin_login button::before { content: "♢"; background: rgba(100,102,204,.3); }
div.st-key-welcome_user_login button p, div.st-key-welcome_signup button p, div.st-key-welcome_admin_login button p { color: #f7f6ff; line-height: 1.55; }
.fs-trust { display: flex; justify-content: center; align-items: center; gap: .55rem; margin-top: 1.25rem; padding-top: 1.1rem; border-top: 1px solid rgba(116,115,209,.24); color: #c4c9eb; font-size: .8rem; }
.fs-trust span { color: #b27bff; font-size: 1.15rem; }
.fs-trust b { color: #8b91c1; font-weight: 400; }
.result-card { padding: 1.45rem 1.65rem; margin: .6rem 0 1rem; border-radius: 18px; background: linear-gradient(140deg,#151b27,#11151e); }
.result-card h2 { margin: .25rem 0; }
.result-card p { color: #b9bbca; margin-bottom: .1rem; }
.result-card strong { font-size: 2.5rem; letter-spacing: -.04em; }
.result-fraud { border: 1px solid #784950; box-shadow: 0 0 35px rgba(230,89,117,.08); }
.result-fraud h2, .result-fraud strong { color: #ff9baa; }
.result-clear { border: 1px solid #23674d; box-shadow: 0 0 35px rgba(79,203,151,.07); }
.result-clear h2, .result-clear strong { color: #8ae0b4; }
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
/* Shared layout refinements for aligned dashboard pages. */
.block-container { max-width: 1360px; padding-top: 1.5rem; }
h1 { margin: 0 0 .85rem; line-height: 1.15; }
h2 { margin: 1.35rem 0 .55rem; line-height: 1.25; }
h3, h4 { line-height: 1.3; }
[data-testid="stMetric"] { min-height: 112px; display: flex; flex-direction: column; justify-content: center; overflow-wrap: anywhere; }
[data-testid="stMetricLabel"] { min-height: 2.4em; white-space: normal; line-height: 1.25; }
[data-testid="stMetricValue"] { font-size: clamp(1.25rem, 2vw, 1.8rem); }
[data-testid="stHorizontalBlock"] { align-items: stretch; gap: 1rem; }
[data-testid="stDataFrame"], [data-testid="stTable"] { width: 100%; max-width: 100%; overflow-x: auto; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { min-height: 2.7rem; margin: .16rem 0; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.45; }
[data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"] { width: 100%; overflow-x: auto; }
@media (max-width: 800px) {
  .block-container { padding-top: 1rem; padding-left: .85rem; padding-right: .85rem; }
  [data-testid="stMetric"] { min-height: 94px; padding: .75rem; }
  [data-testid="stMetricLabel"] { min-height: 0; }
  [data-testid="stHorizontalBlock"] { gap: .55rem; }
}
@media (max-width: 800px) { .hero-title { font-size: 1.7rem; } .block-container { padding-left: 1rem; padding-right: 1rem; } .feature-card { min-height: 0; margin-bottom: .7rem; } .fs-topbar { margin: -1.4rem -1rem 1.4rem; padding: .65rem 1rem; } .fs-tagline { display: none; } .fs-intro h1 { font-size: 2.45rem; } .fs-highlights { grid-template-columns: repeat(2,1fr); row-gap: .9rem; } .fs-highlights > div:nth-child(2) { border-right: 0; } div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fs-panel-icon) { min-height: 0; margin-top: 1rem; } .fs-illustration svg { max-height: 260px; } }
</style>
"""


@st.cache_data(ttl=15, show_spinner=False)
def cached_api_health() -> tuple[bool, dict[str, Any] | None, str | None]:
    """Cache a live API health check briefly; this is not Kafka/Spark health."""
    try:
        return True, get_health(base_url=api_base_url()), None
    except Exception as exc:
        return False, None, str(exc)


def configure_public_page(title: str, icon: str = "◈") -> None:
    """Apply shared styling to public welcome and login pages."""
    st.set_page_config(page_title=f"{title} · Fraud Intelligence", page_icon=icon,
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def configure_page(title: str, icon: str = "◈", required_role: str = "admin") -> None:
    """Apply page metadata, theme details, and consistent sidebar identity."""
    st.set_page_config(page_title=f"{title} · Fraud Intelligence", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    if not require_role(required_role):
        return
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<div class='hero-kicker'>Fraud Intelligence</div>", unsafe_allow_html=True)
        st.caption("Adaptive Transaction Risk Monitoring")
        st.markdown("<div class='section-copy' style='margin: .9rem 0 .25rem'>WORKSPACE</div>", unsafe_allow_html=True)
        if current_role() == "user":
            navigation = (("pages/user_portal.py", "Analyze a transaction", ":material/search:"),)
        else:
            groups = (
                ("Dashboard", (("pages/admin_overview.py", "Admin Overview", ":material/dashboard:"),)),
                ("Transactions", (
                    ("pages/5_Transaction_Analyzer.py", "Transaction Analyzer", ":material/search:"),
                    ("pages/1_Transaction_Explorer.py", "Transaction Explorer", ":material/manage_search:"),
                    ("pages/11_Counterfactual_Explainability.py", "Counterfactual Explainability", ":material/psychology:"),
                )),
                ("Monitoring & Data Quality", (
                    ("pages/2_Live_Monitoring.py", "Pipeline Evidence", ":material/sensors:"),
                    ("pages/7_Drift_Alerts.py", "Drift & Alerts", ":material/notifications_active:"),
                    ("pages/6_Delayed_Labels.py", "Delayed Labels", ":material/label:"),
                    ("pages/4_System_Health.py", "System Health", ":material/health_and_safety:"),
                )),
                ("Model Lifecycle", (
                    ("pages/3_Model_Performance.py", "Model Performance", ":material/monitoring:"),
                    ("pages/12_Model_Orchestrator.py", "Model Orchestrator", ":material/model_training:"),
                )),
                ("Project Reference", (
                    ("pages/9_System_Architecture.py", "System Architecture", ":material/account_tree:"),
                    ("pages/10_Reports_Documentation.py", "Reports & Documentation", ":material/description:"),
                )),
            )
            for group_name, navigation in groups:
                st.markdown(f"<div class='section-copy' style='margin: .85rem 0 .2rem'>{escape(group_name)}</div>", unsafe_allow_html=True)
                for path, label, glyph in navigation:
                    st.page_link(path, label=label, icon=glyph, use_container_width=True)
        if current_role() == "user":
            for path, label, glyph in navigation:
                st.page_link(path, label=label, icon=glyph, use_container_width=True)
        st.divider()
        ok, health, _ = cached_api_health()
        model = health.get("model_version", "Unknown") if ok and health else "Unavailable"
        st.markdown(status_pill("API online" if ok else "API offline", "healthy" if ok else "offline"), unsafe_allow_html=True)
        st.caption(f"API-served model: {model}")
        st.caption("API reachability is checked separately from stream activity.")
        st.caption(f"Signed in as {st.session_state.get('authenticated_username', 'account')} · {current_role().title()}")
        if st.button("Log out", use_container_width=True):
            logout()
            st.switch_page("pages/0_Welcome.py")
        if current_role() == "admin" and st.button("Refresh dashboard data", use_container_width=True):
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
