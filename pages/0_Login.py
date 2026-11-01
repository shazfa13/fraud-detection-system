"""Role-specific login page for registered users and the single configured Admin."""
import streamlit as st

from ui.auth import authenticate, credential_configuration_error, current_role, login, logout
from ui.components import configure_public_page


configure_public_page("Sign in", ":material/lock:")

if current_role() == "admin":
    st.switch_page("pages/admin_overview.py")
elif current_role() == "user":
    st.switch_page("pages/user_portal.py")

role = st.session_state.get("login_role")
if role not in ("user", "admin"):
    st.markdown("<div class='hero-kicker'>SECURE ACCESS</div><h1>Choose your workspace</h1>", unsafe_allow_html=True)
    user_col, admin_col = st.columns(2, gap="large")
    with user_col:
        st.markdown("<div class='feature-card'><h3>User Login</h3><p>Analyze a transaction and understand its estimated fraud risk.</p></div>", unsafe_allow_html=True)
        if st.button("Continue as User", key="choose_user", type="primary", use_container_width=True):
            st.session_state["login_role"] = "user"
            st.rerun()
    with admin_col:
        st.markdown("<div class='feature-card'><h3>Admin Login</h3><p>Open the technical dashboard, monitoring, evaluation, and administrative tools.</p></div>", unsafe_allow_html=True)
        if st.button("Continue as Admin", key="choose_admin", type="primary", use_container_width=True):
            st.session_state["login_role"] = "admin"
            st.rerun()
else:
    title = "User" if role == "user" else "Admin"
    st.markdown(f"<div class='hero-kicker'>SECURE ACCESS</div><h1>{title} Login</h1>", unsafe_allow_html=True)
    notice = st.session_state.pop("user_registered_notice", None)
    if notice:
        st.success(notice)
    if role == "admin" and credential_configuration_error("admin"):
        st.warning(f"Admin sign-in is unavailable: {credential_configuration_error('admin')}")
        st.markdown("Create `.streamlit/secrets.toml` in the project folder. Generate a password hash in PowerShell, then paste it into the file:")
        st.code(r'''$adminHash = & .\venv\Scripts\python.exe -c "from ui.auth import password_hash; import getpass; print(password_hash(getpass.getpass('Admin password: ')))"
$adminHash.Trim()''', language="powershell")
        st.code('''FRAUD_ADMIN_USERNAME = "admin"
FRAUD_ADMIN_PASSWORD_HASH = "paste_the_generated_pbkdf2_sha256_hash_here"''', language="toml")
        st.caption("Save the file and restart Streamlit. Keep secrets.toml private; it is excluded from Git.")
    else:
        with st.form(f"{role}_login", clear_on_submit=True):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if authenticate(role, username, password):
                login(role, username)
                st.rerun()
            else:
                logout()
                st.error(f"The {title.lower()} username and password did not match. Usernames are case-insensitive for Users and case-sensitive for Admin; passwords are checked exactly as entered.")
    if role == "user" and st.button("Create a User account", key="login_signup"):
        st.switch_page("pages/0_Signup.py")

    back, switch = st.columns(2)
    with back:
        if st.button("Back to welcome"):
            st.session_state.pop("login_role", None)
            st.switch_page("pages/0_Welcome.py")
    with switch:
        if st.button("Switch role"):
            st.session_state.pop("login_role", None)
            st.rerun()
