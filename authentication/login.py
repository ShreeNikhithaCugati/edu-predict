"""Login page."""
import streamlit as st
from authentication.session import init_session, login_user
from authentication.auth_service import attempt_login


def render_login_page():
    init_session()
    st.markdown("## 🔐 Login")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
            else:
                ok, msg, user = attempt_login(username, password)
                if ok:
                    login_user(user)
                    st.success(f"Welcome, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()
        st.caption("New teacher? Use the Signup page in the sidebar.")