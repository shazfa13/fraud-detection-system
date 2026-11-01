"""Self-service registration for normal User accounts; Admin is never selectable."""
import streamlit as st

from ui.auth import AuthStorageError, RegistrationError, current_role, register_user
from ui.components import configure_public_page


configure_public_page("Sign Up", ":material/person_add:")

role = current_role()
if role == "admin":
    st.switch_page("pages/admin_overview.py")
elif role == "user":
    st.switch_page("pages/user_portal.py")

st.markdown("<div class='hero-kicker'>GET STARTED</div><h1>Create your User account</h1><p class='section-copy'>Register to analyze transactions and view understandable risk estimates.</p>", unsafe_allow_html=True)
st.caption("Usernames are unique without regard to capitalization. Passwords must contain at least 12 characters. Registration never creates an Admin account.")

with st.form("user_registration", clear_on_submit=True):
    username = st.text_input("Username", autocomplete="username", max_chars=64)
    password = st.text_input("Password", type="password", autocomplete="new-password", max_chars=1024)
    confirmation = st.text_input("Confirm password", type="password", autocomplete="new-password", max_chars=1024)
    submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

if submitted:
    try:
        registered = register_user(username, password, confirmation)
    except RegistrationError as exc:
        st.error(str(exc))
    except AuthStorageError as exc:
        st.error(str(exc))
    else:
        st.session_state["user_registered_notice"] = f"Account created for {registered}. Sign in to continue."
        st.session_state["login_role"] = "user"
        st.switch_page("pages/0_Login.py")

if st.button("Back to welcome"):
    st.switch_page("pages/0_Welcome.py")
