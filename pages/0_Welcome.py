"""Public FraudShield landing page, styled to match the supplied reference."""
import streamlit as st

from ui.components import configure_public_page


configure_public_page("Welcome", ":material/security:")

st.markdown("""
<style>
.fs-topbar { margin-top: 0; margin-bottom: 1.6rem; padding-top: .8rem; padding-bottom: .8rem; }
.fs-intro h1 { margin-top: .7rem; margin-bottom: .6rem; }
.fs-lead { margin-top: .25rem; margin-bottom: 0; }
.fs-highlights { margin-top: 1.1rem; margin-bottom: .45rem; }
.fs-highlights > div { min-width: 0; padding-inline: .35rem; }
.fs-illustration { margin-top: .2rem; }
.fs-illustration svg { max-height: clamp(235px, 31vh, 285px); }
.fs-footnote { margin-top: 0; }
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fs-panel-icon) {
  box-sizing: border-box;
  width: 100%;
  height: min(690px, calc(100vh - 160px));
  min-height: 0;
  padding: clamp(1.2rem, 2.2vw, 2rem);
}
div.st-key-welcome_user_login button,
div.st-key-welcome_signup button,
div.st-key-welcome_admin_login button {
  min-height: 96px;
  height: auto;
  justify-content: flex-start;
}
div.st-key-welcome_user_login button > div,
div.st-key-welcome_signup button > div,
div.st-key-welcome_admin_login button > div {
  display: block !important;
  flex: 1 1 auto;
  width: 100%;
  text-align: left;
}
div.st-key-welcome_user_login button p,
div.st-key-welcome_signup button p,
div.st-key-welcome_admin_login button p { width: 100%; text-align: left; }
@media (max-width: 800px) {
  .fs-topbar { margin-bottom: 1.2rem; }
  .fs-illustration svg { max-height: 250px; }
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fs-panel-icon) {
    height: auto;
    min-height: 0;
    margin-top: 1rem;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fs-topbar">
  <div class="fs-brand"><span class="fs-brand-mark">♢</span>
    <span><strong>FraudShield</strong><small>Transaction Fraud Detection System</small></span>
  </div>
  <div class="fs-tagline">Safer Transactions <b>✦</b> A Smarter Tomorrow</div>
</div>
""", unsafe_allow_html=True)

intro, access = st.columns([1.05, .95], gap="large")
with intro:
    st.markdown("""
    <section class="fs-intro">
      <div class="fs-badge">AI POWERED <b>•</b> XGBOOST V2</div>
      <h1>Secure transactions.<br><span>Smarter decisions.</span></h1>
      <p class="fs-lead">Analyze transaction patterns with machine learning and understand the estimated fraud risk behind each result.</p>
      <div class="fs-highlights">
        <div><span class="fs-highlight-icon">ϟ</span><strong>Risk<br>Scoring</strong></div>
        <div><span class="fs-highlight-icon">✿</span><strong>ML Powered<br>(XGBoost V2)</strong></div>
        <div><span class="fs-highlight-icon">♢</span><strong>Understandable<br>Explanations</strong></div>
        <div><span class="fs-highlight-icon">♙</span><strong>Role-Based<br>Access</strong></div>
      </div>
      <div class="fs-illustration" aria-label="Illustration of a payment card protected by a shield">
        <svg viewBox="0 0 720 330" role="img" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="card" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#5738ed"/><stop offset="1" stop-color="#171a82"/></linearGradient>
            <linearGradient id="shield" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#dc72ff"/><stop offset=".53" stop-color="#7855ff"/><stop offset="1" stop-color="#38c9ff"/></linearGradient>
            <linearGradient id="wave" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#5129dc" stop-opacity="0"/><stop offset=".5" stop-color="#844dff" stop-opacity=".72"/><stop offset="1" stop-color="#1fc8ef" stop-opacity="0"/></linearGradient>
            <filter id="glow"><feGaussianBlur stdDeviation="12" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>
          <path d="M0 270 C125 190 190 317 327 247 S555 180 720 258" fill="none" stroke="url(#wave)" stroke-width="2"/>
          <path d="M0 288 C135 212 190 330 340 264 S565 198 720 278" fill="none" stroke="url(#wave)" stroke-width="1" opacity=".7"/>
          <path d="M0 306 C128 238 207 342 356 282 S573 220 720 298" fill="none" stroke="url(#wave)" stroke-width="1" opacity=".48"/>
          <ellipse cx="326" cy="264" rx="229" ry="32" fill="#4726d9" opacity=".22" filter="url(#glow)"/>
          <g transform="translate(100 72) rotate(-8 130 92)">
            <rect x="16" y="16" width="250" height="153" rx="20" fill="#10165f" stroke="#573cff" stroke-width="2" opacity=".75"/>
            <rect x="0" y="0" width="250" height="153" rx="20" fill="url(#card)" stroke="#9d70ff" stroke-width="2"/>
            <path d="M0 23 Q0 0 23 0 H227 Q250 0 250 23 V45 H0Z" fill="#171b69" opacity=".65"/>
            <rect x="23" y="58" width="35" height="26" rx="5" fill="#b7b0ff" opacity=".85"/><path d="M39 58v26M23 71h35" stroke="#5545c8" stroke-width="2"/>
            <text x="24" y="117" fill="#d9d4ff" font-size="13" letter-spacing="2">••••  ••••  ••••  1234</text>
            <circle cx="204" cy="126" r="12" fill="#ff6888" opacity=".82"/><circle cx="220" cy="126" r="12" fill="#ffd15f" opacity=".8"/>
          </g>
          <path d="M362 78 L415 99 V147 C415 187 392 211 362 229 C331 211 309 187 309 147 V99Z" fill="#602dff" opacity=".32" filter="url(#glow)"/>
          <path d="M362 69 L409 88 V132 C409 169 389 192 362 208 C335 192 315 169 315 132 V88Z" fill="url(#shield)" stroke="#c9b7ff" stroke-width="3"/>
          <path d="M341 132 L356 147 L385 113" fill="none" stroke="#fff" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
          <g font-family="sans-serif" font-size="15">
            <rect x="470" y="91" width="230" height="48" rx="13" fill="#0d2934" stroke="#42d7b0"/>
            <circle cx="493" cy="115" r="11" fill="#86edc7"/><path d="M488 115l4 4 7-8" fill="none" stroke="#123a38" stroke-width="2.5"/>
            <text x="514" y="120" fill="#a1f2d7">Lower estimated risk</text>
            <rect x="470" y="153" width="230" height="48" rx="13" fill="#29152c" stroke="#ff628a"/>
            <circle cx="493" cy="177" r="11" fill="#ff839b"/><text x="490" y="182" fill="#421a2c" font-weight="bold">!</text>
            <text x="514" y="182" fill="#ff9cac">Higher estimated risk</text>
          </g>
          <circle cx="65" cy="184" r="2.5" fill="#7b5cff"/><circle cx="674" cy="67" r="2" fill="#b96bff"/><circle cx="433" cy="237" r="2" fill="#39e0d1"/><circle cx="288" cy="42" r="2" fill="#754fff"/>
        </svg>
      </div>
      <div class="fs-footnote"><span>∿</span> Powered by Machine Learning <i></i> Built for a Safer Financial Future</div>
    </section>
    """, unsafe_allow_html=True)

with access:
    with st.container():
        st.markdown("""
        <div class="fs-panel-icon">♢</div>
        <h2 class="fs-panel-title">Welcome to <span>FraudShield</span></h2>
        <p class="fs-panel-copy">Choose an option below to get started</p>
        """, unsafe_allow_html=True)
        if st.button("**User Login**  \nAccess your account and analyze transaction risk.  →", key="welcome_user_login", use_container_width=True):
            st.session_state["login_role"] = "user"
            st.switch_page("pages/0_Login.py")
        if st.button("**Sign Up**  \nCreate a new account to get started.  →", key="welcome_signup", use_container_width=True):
            st.switch_page("pages/0_Signup.py")
        if st.button("**Admin Login**  \nAccess the administrative dashboard and monitoring tools.  →", key="welcome_admin_login", use_container_width=True):
            st.session_state["login_role"] = "admin"
            st.switch_page("pages/0_Login.py")
        st.markdown("<div class='fs-trust'><span>♢</span> Secure <b>•</b> Private <b>•</b> Trusted</div>", unsafe_allow_html=True)
