"""Role-aware Streamlit entry point with protected dynamic page registration."""
import streamlit as st

from ui.auth import current_role

welcome = st.Page("pages/0_Welcome.py", title="Welcome", icon=":material/security:", url_path="welcome")
login = st.Page("pages/0_Login.py", title="Sign in", icon=":material/lock:", url_path="login")
signup = st.Page("pages/0_Signup.py", title="Sign Up", icon=":material/person_add:", url_path="signup")
role = current_role()

if role == "admin":
    pages = [
        st.Page("pages/admin_overview.py", title="Admin Overview", icon=":material/dashboard:", url_path="admin"),
        st.Page("pages/12_Model_Orchestrator.py", title="Model Orchestrator", icon=":material/model_training:"),
        welcome,
        login,
        signup,
        st.Page("pages/5_Transaction_Analyzer.py", title="Transaction Analyzer", icon=":material/search:"),
        st.Page("pages/11_Counterfactual_Explainability.py", title="Counterfactual Explainability", icon=":material/psychology:"),
        st.Page("pages/2_Live_Monitoring.py", title="Pipeline Evidence", icon=":material/sensors:"),
        st.Page("pages/1_Transaction_Explorer.py", title="Transaction Explorer", icon=":material/manage_search:"),
        st.Page("pages/3_Model_Performance.py", title="Model Performance", icon=":material/monitoring:"),
        st.Page("pages/7_Drift_Alerts.py", title="Drift & Alerts", icon=":material/notifications_active:"),
        st.Page("pages/6_Delayed_Labels.py", title="Delayed Labels", icon=":material/label:"),
        st.Page("pages/9_System_Architecture.py", title="System Architecture", icon=":material/account_tree:"),
        st.Page("pages/10_Reports_Documentation.py", title="Reports & Documentation", icon=":material/description:"),
        st.Page("pages/4_System_Health.py", title="System Health", icon=":material/health_and_safety:"),
    ]
elif role == "user":
    pages = [st.Page("pages/user_portal.py", title="Transaction Risk Check", icon=":material/verified_user:"), welcome, login, signup]
else:
    pages = [welcome, login, signup]

st.navigation(pages, position="hidden").run()
