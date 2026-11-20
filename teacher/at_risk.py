"""Teacher: At-Risk Student Intervention Board."""
import streamlit as st
import pandas as pd
from database.database import query
from recommendations.recommendation_engine import generate_recommendations
import json
import os

def render_at_risk_students():
    st.markdown("## 🚨 At-Risk Student Intervention Board")
    st.write("This page displays all students who have been predicted to be 'At Risk' based on their latest academic records.")

    teacher_id = st.session_state.user["id"]

    # FIXED: Use a subquery to fetch only the LATEST At-Risk prediction for each unique student
    query_sql = """
        SELECT p.id, p.student_id, s.student_name, p.predicted_marks, p.prediction_date, p.selected_model
        FROM predictions p
        JOIN students s ON p.student_id = s.student_id
        JOIN (
            SELECT student_id, MAX(id) as max_id
            FROM predictions
            WHERE teacher_id = ? AND performance_category = 'At Risk'
            GROUP BY student_id
        ) latest ON p.id = latest.max_id
        ORDER BY p.predicted_marks ASC
    """
    
    at_risk_data = query(query_sql, (teacher_id,))

    if not at_risk_data:
        st.success("🎉 Excellent! There are currently no students classified as 'At Risk'.")
        return

    st.error(f"⚠️ **Warning:** {len(at_risk_data)} unique student(s) require immediate attention.")

    # Display each at-risk student in an expandable card
    for data in at_risk_data:
        sid = data["student_id"]
        name = data["student_name"]
        marks = data["predicted_marks"]
        date = data["prediction_date"]
        
        with st.expander(f"🚨 {name} ({sid}) — Predicted: {marks:.1f} Marks"):
            c1, c2 = st.columns(2)
            c1.metric("Predicted Final Score", f"{marks:.1f}/100")
            c2.metric("Last Predicted On", date)
            
            st.markdown("### 💡 Actionable Solutions (Based on XAI)")
            
            # Load the student's academic record to generate recommendations
            record = query(
                "SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
                (sid,)
            )
            if record:
                r = record[0]
                # We pass None for SHAP here so the engine relies purely on raw data rules
                recs = generate_recommendations(r, shap_contributions=None)
                for rec in recs:
                    st.markdown(f"- {rec}")
            else:
                st.warning("No detailed academic record found to generate solutions.")
            
            st.divider()
            st.markdown("### 📊 Latest Subject Breakdown")
            json_path = f"data/subject_records/{sid}.json"
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    saved_subject_marks = json.load(f)
                
                # Create a small table of their IA marks
                subj_data = []
                for subj, m in saved_subject_marks.items():
                    avg = round((m[0] + m[1] + m[2]) / 3, 1)
                    subj_data.append({"Subject": subj, "IA1": m[0], "IA2": m[1], "IA3": m[2], "Average": avg})
                st.dataframe(pd.DataFrame(subj_data), use_container_width=True, hide_index=True)
            else:
                st.info("Subject-wise breakdown not available.")