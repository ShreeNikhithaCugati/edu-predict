"""Admin: View predictions by teacher and download reports."""
import streamlit as st
import pandas as pd
from database.database import query
import os
import json
from datetime import datetime

def render_teacher_predictions():
    st.markdown("## 👨‍🏫 Teacher & Student Predictions")
    st.write("View student predictions made by each teacher and download comprehensive reports.")

    # 1. Get all teachers
    teachers = query("SELECT id, full_name, department FROM users WHERE role='teacher' AND approval_status='approved'")
    
    if not teachers:
        st.warning("No approved teachers found.")
        return

    # 2. Select a teacher
    teacher_options = {f"{t['full_name']} ({t['department']})": t['id'] for t in teachers}
    selected_teacher_name = st.selectbox("Select a Teacher", list(teacher_options.keys()))
    selected_teacher_id = teacher_options[selected_teacher_name]

    st.divider()

    # 3. Get students assigned to this teacher
    students = query(
        "SELECT student_id, student_name, department, semester FROM students WHERE assigned_teacher_id = ?",
        (selected_teacher_id,)
    )

    if not students:
        st.info(f"No students assigned to {selected_teacher_name} yet.")
        return

    # 4. Select a student
    student_options = {f"{s['student_name']} ({s['student_id']})": s['student_id'] for s in students}
    selected_student_name = st.selectbox("Select a Student", list(student_options.keys()))
    selected_student_id = student_options[selected_student_name]
    
    # Find the full student record
    student_data = next(s for s in students if s["student_id"] == selected_student_id)

    # 5. Get the latest prediction for this student by this teacher
    latest_pred = query(
        """SELECT * FROM predictions 
           WHERE student_id = ? AND teacher_id = ? 
           ORDER BY prediction_date DESC LIMIT 1""",
        (selected_student_id, selected_teacher_id)
    )

    st.subheader(f"📋 Report for {selected_student_name}")

    if not latest_pred:
        st.warning("No predictions have been made for this student yet.")
        return

    pred = latest_pred[0]
    
    # Display basic info
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Final Marks", f"{pred['predicted_marks']:.1f}")
    c2.metric("Performance Category", pred['performance_category'])
    c3.metric("Prediction Date", pred['prediction_date'])

    # Load subject marks from JSON
    json_path = f"data/subject_records/{selected_student_id}.json"
    saved_subject_marks = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved_subject_marks = json.load(f)

    if saved_subject_marks:
        st.markdown("### 📚 Subject-Wise Marks")
        subj_data = []
        for subj, m in saved_subject_marks.items():
            avg = round((m[0] + m[1] + m[2]) / 3, 2)
            subj_data.append({"Subject": subj, "IA1": m[0], "IA2": m[1], "IA3": m[2], "Average": avg})
        st.dataframe(pd.DataFrame(subj_data), use_container_width=True, hide_index=True)

    # --- GENERATE PDF REPORT ---
    st.divider()
    st.subheader("📄 Download Comprehensive Report")
    
    if st.button("Generate PDF Report", type="primary"):
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "EduPredict - Comprehensive Student Report", ln=True, align="C")
            pdf.ln(5)
            
            # Basic Info
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Student Name: {selected_student_name}", ln=True)
            pdf.cell(0, 8, f"Student ID: {selected_student_id} | Dept: {student_data['department']} | Sem: {student_data['semester']}", ln=True)
            pdf.cell(0, 8, f"Teacher: {selected_teacher_name}", ln=True)
            
            # Prediction
            pdf.set_fill_color(230, 240, 255)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"Prediction Result: {pred['predicted_marks']:.1f} ({pred['performance_category']})", ln=True, fill=True)
            pdf.ln(5)
            
            # Table
            if saved_subject_marks:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Subject-Wise Internal Assessment Marks", ln=True)
                
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_fill_color(240, 240, 240)
                col_widths = [65, 25, 25, 25, 35]
                headers = ["Subject", "IA1", "IA2", "IA3", "Average"]
                for i, h in enumerate(headers):
                    pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
                pdf.ln()
                
                pdf.set_font("Helvetica", "", 10)
                for subj, m in saved_subject_marks.items():
                    avg = round((m[0] + m[1] + m[2]) / 3, 2)
                    row_data = [subj, str(m[0]), str(m[1]), str(m[2]), str(avg)]
                    for i, item in enumerate(row_data):
                        align = "L" if i == 0 else "C"
                        pdf.cell(col_widths[i], 8, item, border=1, align=align)
                    pdf.ln()
                pdf.ln(5)
            
            # Solutions/Recommendations
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Actionable Solutions (Based on XAI)", ln=True)
            pdf.set_font("Helvetica", "", 11)
            
            # Use the recommendation engine (simulated here for PDF context)
            from recommendations.recommendation_engine import generate_recommendations
            # Load the academic record
            rec_record = query("SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1", (selected_student_id,))
            if rec_record:
                recs = generate_recommendations(rec_record[0], shap_contributions=None)
                for rec in recs:
                    # Clean up emojis for PDF
                    clean_rec = rec.encode('ascii', 'ignore').decode('ascii')
                    pdf.multi_cell(0, 8, f"- {clean_rec}")
            
            pdf.ln(10)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 8, f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            
            pdf_bytes = bytes(pdf.output(dest='S'))
            
            st.download_button(
                label="⬇️ Download Report PDF",
                data=pdf_bytes,
                file_name=f"{selected_student_id}_report.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Error generating PDF: {e}")