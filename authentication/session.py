"""Session management for Streamlit."""
import os
import time
import streamlit as st

SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))


def init_session():
    """Call at the top of every page."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()
    if "login_time" not in st.session_state:
        st.session_state.login_time = None


def login_user(user: dict):
    """Call after successful password verification."""
    st.session_state.user = user
    st.session_state.last_activity = time.time()
    st.session_state.login_time = time.time()


def logout_user():
    st.session_state.user = None
    st.session_state.last_activity = None
    st.session_state.login_time = None


def is_logged_in() -> bool:
    return st.session_state.get("user") is not None


def check_session_timeout() -> bool:
    """Returns True if session is still valid."""
    if not is_logged_in():
        return False
    elapsed = (time.time() - st.session_state.last_activity) / 60
    if elapsed > SESSION_TIMEOUT_MINUTES:
        logout_user()
        return False
    st.session_state.last_activity = time.time()
    return True


def require_login(role: str | None = None):
    """Use at top of protected pages. Stops page if not allowed."""
    init_session()
    if not is_logged_in():
        st.warning("Please log in first.")
        st.stop()
    if not check_session_timeout():
        st.warning("Session expired. Please log in again.")
        st.stop()
    if role is not None and st.session_state.user["role"] != role:
        st.error("You do not have permission to view this page.")
        st.stop()