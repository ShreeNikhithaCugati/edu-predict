"""Admin dashboard — overview + tabs for management."""
import streamlit as st
from database.database import query
from admin.teacher_management import render_teacher_management
from admin.student_management import render_student_management
from admin.dataset_management import render_dataset_management
from admin.model_management import render_model_management
from admin.batch_prediction import render_batch_prediction
from admin.teacher_predictions import render_teacher_predictions  # <--- NEW IMPORT
from visualization.charts import (
    performance_distribution_chart,
    department_performance_chart,
    predictions_over_time_chart,
)

def render_admin_dashboard():
    st.title("🛡️ Admin Dashboard")

    # --- Metric Cards ---
    teachers = query("SELECT COUNT(*) AS n FROM users WHERE role='teacher'")[0]["n"]
    pending = query("SELECT COUNT(*) AS n FROM users WHERE role='teacher' AND approval_status='pending'")[0]["n"]
    students = query("SELECT COUNT(*) AS n FROM students")[0]["n"]
    preds = query("SELECT COUNT(*) AS n FROM predictions")[0]["n"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Teachers", teachers)
    c2.metric("Pending Approvals", pending)
    c3.metric("Total Students", students)
    c4.metric("Predictions Made", preds)

    st.divider()

    # --- FILTERS ---
    st.markdown("#### 🔍 Analytics Filters")
    f1, f2 = st.columns(2)
    
    # Fetch unique departments from the students table
    raw_depts = query("SELECT DISTINCT department FROM students WHERE department IS NOT NULL")
    depts = ["All"] + [r["department"] for r in raw_depts]
    
    raw_sems = query("SELECT DISTINCT semester FROM students ORDER BY semester")
    sems = ["All"] + [r["semester"] for r in raw_sems]
    
    with f1:
        selected_dept = st.selectbox("Filter by Department", depts, key="admin_filter_dept")
    with f2:
        selected_sem = st.selectbox("Filter by Semester", sems, key="admin_filter_sem")

    # --- Charts ---
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = performance_distribution_chart(selected_dept, selected_sem)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions found for this filter.")
    with cc2:
        fig = department_performance_chart(selected_dept, selected_sem)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No department data found for this filter.")

    st.divider()

    # --- TABS ---
    # Removed: "Single Predict", "XAI Predict", "Subject-Wise"
    # Added: "Teacher Predictions"
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["👥 Teachers", "🎓 Students", "📊 Dataset", "🤖 Models", "📁 Batch Predict", "🏫 Teacher Predictions"]
    )
    with tab1:
        render_teacher_management()
    with tab2:
        render_student_management()
    with tab3:
        render_dataset_management()
    with tab4:
        render_model_management()
    with tab5:
        render_batch_prediction()
    with tab6:
        render_teacher_predictions()  # <--- NEW TAB