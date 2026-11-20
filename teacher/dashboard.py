"""Teacher dashboard — overview + 3 tabs."""
import streamlit as st
from database.database import query
from teacher.student_records import render_student_records
from teacher.prediction import render_prediction_page
from teacher.at_risk import render_at_risk_students
from teacher.reports import render_reports


def render_teacher_dashboard():
    st.title("👨‍🏫 Teacher Dashboard")
    tid = st.session_state.user["id"]

    n_students = query(
        "SELECT COUNT(*) AS n FROM students WHERE assigned_teacher_id=?", (tid,)
    )[0]["n"]
    n_preds = query(
        "SELECT COUNT(*) AS n FROM predictions WHERE teacher_id=?", (tid,)
    )[0]["n"]
    at_risk = query(
        """
        SELECT COUNT(DISTINCT p.student_id) AS n 
        FROM predictions p
        INNER JOIN (
            SELECT student_id, MAX(id) as max_id
            FROM predictions
            WHERE teacher_id = ?
            GROUP BY student_id
        ) latest ON p.id = latest.max_id
        WHERE p.teacher_id = ? AND p.performance_category = 'At Risk'
        """, 
        (tid, tid)
    )[0]["n"]

    c1, c2, c3 = st.columns(3)
    c1.metric("My Students", n_students)
    c2.metric("Predictions Made", n_preds)
    c3.metric("At-Risk Students", at_risk)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🎓 My Students", "🔮 Predict", "📜 Prediction History", "🚨 At-Risk Students"])
    with tab1:
        render_student_records()
    with tab2:
        render_prediction_page()
    with tab3:
        render_reports()
    with tab4:
        render_at_risk_students()