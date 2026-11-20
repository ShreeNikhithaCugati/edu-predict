"""Teacher signup page."""
import streamlit as st
from authentication.session import init_session
from authentication.auth_service import register_teacher


def render_signup_page():
    init_session()
    st.markdown("## 📝 Teacher Registration")
    st.caption("Your account will be reviewed by an administrator before activation.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            teacher_id = st.text_input("Teacher ID")
            email = st.text_input("Email")
            username = st.text_input("Username")
            department = st.text_input("Department")
            subject = st.text_input("Subject Handled")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register", use_container_width=True)

        if submitted:
            ok, msg = register_teacher(
                full_name, teacher_id, email, username,
                department, subject, password, confirm,
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)