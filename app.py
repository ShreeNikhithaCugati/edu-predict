"""
EduPredict — Student Performance Prediction (ML + DL + XAI)
Main entry point / router.
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

# Page config MUST be the first Streamlit call
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

# On Streamlit Community Cloud there is no .env file, so copy the values from
# the app's "Secrets" panel into environment variables. This must happen
# BEFORE importing database/authentication modules, because database.py reads
# DB_PATH from the environment at import time.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass  # no secrets configured (e.g. local run with .env)

# Now safe to import modules that use Streamlit
from authentication.session import (
    init_session,
    is_logged_in,
    logout_user,
)
from authentication.login import render_login_page
from authentication.signup import render_signup_page
from admin.dashboard import render_admin_dashboard
from teacher.dashboard import render_teacher_dashboard

# Initialize session state
init_session()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎓 EduPredict")
    st.caption("ML + DL + XAI")

    if is_logged_in():
        user = st.session_state.user
        st.success(f"👤 {user['full_name']}")
        st.caption(f"Role: {user['role'].title()}")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        page = st.radio("Menu", ["Login", "Signup"])
        st.session_state["auth_page"] = page

# ---------------- ROUTING ----------------
if not is_logged_in():
    page = st.session_state.get("auth_page", "Login")
    if page == "Login":
        render_login_page()
    else:
        render_signup_page()
    st.stop()

user = st.session_state.user

if user["role"] == "admin":
    render_admin_dashboard()
elif user["role"] == "teacher":
    render_teacher_dashboard()
else:
    st.error("Unknown role. Contact administrator.")
    if st.button("Logout"):
        logout_user()
        st.rerun()