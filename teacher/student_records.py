"""Teacher: manage academic records for assigned students."""
import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
from database.database import query, execute
from authentication.auth_service import log_activity

SUBJECTS_BY_SEM = {
    1: ["Mathematics-I", "Applied Physics", "Programming in C", "Introduction to Electronics", "Engineering Drawing", "C Lab"],
    2: ["Mathematics-II", "Applied Chemistry", "Python Programming", "Introduction to AI", "Electrical Engineering", "Communication Skills"],
    3: ["Mathematics for Computer Science", "Digital Design and Computer Organization", "Operating Systems", "Data Structures and Applications"],
    4: ["Analysis and Design of Algorithms", "Microcontrollers", "DBMS", "Discrete Mathematics", "Biology for Computer Engineers"],
    5: ["Software Engineering", "Computer Networks", "Theory of Computation", "Web Technology"],
    6: ["Cloud Computing", "Machine Learning", "Professional Electives", "Open Electives"],
    7: ["Internet of Things", "Parallel Computing", "Cryptography and Network Security", "Major Project"],
    8: ["NPTEL Electives", "Open Electives", "Internship"]
}

def render_student_records():
    st.markdown("## 🎓 My Students")

    teacher_id = st.session_state.user["id"]
    
    base_sql = "SELECT * FROM students WHERE assigned_teacher_id = ?"
    params = [teacher_id]
    
    st.write("### 🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)
    
    depts_data = query("SELECT DISTINCT department FROM students WHERE assigned_teacher_id = ?", (teacher_id,))
    dept_options = ["All"] + [d["department"] for d in depts_data if d["department"]]
    
    years_data = query("SELECT DISTINCT academic_year FROM students WHERE assigned_teacher_id = ?", (teacher_id,))
    year_options = ["All"] + [y["academic_year"] for y in years_data if y["academic_year"]]
    
    with f1:
        dept_filter = st.selectbox("Department", dept_options, key="t_dept")
    with f2:
        year_filter = st.selectbox("Batch Year", year_options, key="t_year")
    with f3:
        sem_filter = st.selectbox("Semester", ["All", 1, 2, 3, 4, 5, 6, 7, 8], key="t_sem")
    with f4:
        search_id = st.text_input("Search Student ID", placeholder="e.g. 1CR23...", key="t_search")
        
    if dept_filter != "All":
        base_sql += " AND department = ?"
        params.append(dept_filter)
    if year_filter != "All":
        base_sql += " AND academic_year = ?"
        params.append(year_filter)
    if sem_filter != "All":
        base_sql += " AND semester = ?"
        params.append(sem_filter)
    if search_id:
        base_sql += " AND student_id LIKE ?"
        params.append(f"%{search_id}%")
        
    base_sql += " ORDER BY student_id"
    students = query(base_sql, tuple(params))

    if not students:
        st.warning("No students found matching your filters.")
        return

    st.write(f"**{len(students)} student(s) found**")

    sid = st.selectbox(
        "Select student to view/edit records",
        [s["student_id"] for s in students],
        format_func=lambda x: f"{x} — {next(s['student_name'] for s in students if s['student_id'] == x)}",
        key="t_student_select"
    )
    
    s = next(x for x in students if x["student_id"] == sid)
    semester = s["semester"]
    subjects = SUBJECTS_BY_SEM.get(semester, ["General Subject"])

    st.divider()
    st.subheader(f"📋 {s['student_name']} ({s['student_id']})")
    st.caption(f"Dept: {s['department']} | Semester: {semester} | Year: {s['academic_year']}")

    # Fetch the latest record from DB
    records = query(
        "SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    )
    latest = records[0] if records else None

    # --- PERMANENT STATUS BOX ---
    if latest:
        last_updated_time = latest.get("last_updated") or latest.get("created_at") or "Unknown"
        st.info(f"🕒 **Last Record Update:** {last_updated_time}")
    else:
        st.warning("⚠️ No academic record found for this student yet.")
    # ----------------------------

    json_path = f"data/subject_records/{sid}.json"
    saved_subject_marks = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved_subject_marks = json.load(f)

    with st.form("academic_record_form"):
        st.markdown("**Previous Academic Performance**")
        c1, c2, c3 = st.columns(3)
        with c1:
            prev_marks = st.number_input("Previous Semester Marks (%)", 0.0, 100.0,
                                         value=float(latest["previous_semester_marks"]) if latest and latest["previous_semester_marks"] else 60.0)
        with c2:
            prev_sgpa = st.number_input("Previous SGPA", 0.0, 10.0,
                                        value=float(latest["previous_sgpa"]) if latest and latest["previous_sgpa"] else 6.5)
        with c3:
            backlogs = st.number_input("Previous Backlogs", 0, 20,
                                       value=int(latest["previous_backlogs"]) if latest and latest["previous_backlogs"] else 0)

        st.markdown(f"**Internal Assessment Scores (Semester {semester}) - Max 50 Marks Each**")
        
        subject_ia_marks = {}
        
        for subj in subjects:
            st.markdown(f"#### {subj}")
            sc1, sc2, sc3 = st.columns(3)
            
            default_ia1 = saved_subject_marks.get(subj, [35.0, 38.0, 40.0])[0]
            default_ia2 = saved_subject_marks.get(subj, [35.0, 38.0, 40.0])[1]
            default_ia3 = saved_subject_marks.get(subj, [35.0, 38.0, 40.0])[2]
            
            with sc1:
                ia1 = st.number_input(f"IA1", 0.0, 50.0, round(float(default_ia1), 2), key=f"ia1_{subj}")
            with sc2:
                ia2 = st.number_input(f"IA2", 0.0, 50.0, round(float(default_ia2), 2), key=f"ia2_{subj}")
            with sc3:
                ia3 = st.number_input(f"IA3", 0.0, 50.0, round(float(default_ia3), 2), key=f"ia3_{subj}")
            
            subject_ia_marks[subj] = [ia1, ia2, ia3]
            st.markdown("---")

        st.markdown("### 📊 Calculated Subject Averages")
        
        avg_data = []
        for subj, (ia1, ia2, ia3) in subject_ia_marks.items():
            subj_avg = round((ia1 + ia2 + ia3) / 3, 2)
            avg_data.append({"Subject": subj, "IA1": ia1, "IA2": ia2, "IA3": ia3, "Subject Average": subj_avg})
        
        df_avg = pd.DataFrame(avg_data)
        st.dataframe(df_avg, use_container_width=True, hide_index=True)

        avg_ia1 = round(float(df_avg["IA1"].mean()), 2)
        avg_ia2 = round(float(df_avg["IA2"].mean()), 2)
        avg_ia3 = round(float(df_avg["IA3"].mean()), 2)

        st.markdown("**Other Academic Data**")
        c1, c2, c3 = st.columns(3)
        with c1:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0,
                                         value=float(latest["attendance"]) if latest and latest["attendance"] else 75.0)
            assignment_avg = st.number_input("Assignment Average (%)", 0.0, 100.0,
                                             value=float(latest["assignment_average"]) if latest and latest["assignment_average"] else 70.0)
        with c2:
            assignment_compl = st.number_input("Assignment Completion (%)", 0.0, 100.0,
                                               value=float(latest["assignment_completion"]) if latest and latest["assignment_completion"] else 80.0)
            quiz_avg = st.number_input("Quiz Average (%)", 0.0, 100.0,
                                       value=float(latest["quiz_average"]) if latest and latest["quiz_average"] else 65.0)
        with c3:
            study_hours = st.number_input("Study Hours / Day", 0.0, 24.0,
                                          value=float(latest["study_hours"]) if latest and latest["study_hours"] else 3.0)

        if st.form_submit_button("💾 Save Marks & Subject Records"):
            save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            execute(
                """INSERT INTO academic_records
                   (student_id, previous_semester_marks, previous_sgpa, previous_backlogs,
                    ia1_score, ia1_max_marks, ia2_score, ia2_max_marks,
                    ia3_score, ia3_max_marks, attendance, assignment_average,
                    assignment_completion, quiz_average, study_hours, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, prev_marks, prev_sgpa, backlogs,
                 float(avg_ia1), 50.0, float(avg_ia2), 50.0, float(avg_ia3), 50.0,
                 attendance, assignment_avg, assignment_compl,
                 quiz_avg, study_hours, save_time),
            )
            
            log_activity(teacher_id, "save_academic_record", f"Saved record for {sid} at {save_time}")
            
            os.makedirs("data/subject_records", exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(subject_ia_marks, f, indent=4)
                
            st.rerun()

    # --- NEW: PDF REPORT GENERATION (Dashboard Style) ---
    st.divider()
    st.subheader("📄 Download Student Report")
    
    latest_record = query(
        "SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    )
    
    if not latest_record:
        st.info("No records found. Please save marks first to generate a report.")
    else:
        lr = latest_record[0]
        saved_on_time = lr.get("last_updated") or lr.get("created_at") or "N/A"
        
        # Prepare the data for the table
        table_data = []
        for subj, marks in saved_subject_marks.items():
            ia1_r = round(float(marks[0]), 2)
            ia2_r = round(float(marks[1]), 2)
            ia3_r = round(float(marks[2]), 2)
            avg_r = round((ia1_r + ia2_r + ia3_r) / 3, 2)
            table_data.append([subj, str(ia1_r), str(ia2_r), str(ia3_r), str(avg_r)])
            
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "EduPredict - Student Academic Report", ln=True, align="C")
            pdf.ln(5)
            
            # Student Info
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Student Name: {s['student_name']} ({s['student_id']})", ln=True)
            pdf.cell(0, 8, f"Department: {s['department']} | Semester: {s['semester']} | Year: {s['academic_year']}", ln=True)
            
            # Timestamp Box
            pdf.set_fill_color(230, 240, 255) # Light Blue
            pdf.set_draw_color(200, 220, 255)
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, f" Last Record Update: {saved_on_time}", ln=True, fill=True)
            pdf.ln(5)
            
            # Previous Performance
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Previous Academic Performance", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"- Previous Marks: {lr['previous_semester_marks']}%", ln=True)
            pdf.cell(0, 8, f"- Previous SGPA: {lr['previous_sgpa']}", ln=True)
            pdf.cell(0, 8, f"- Backlogs: {lr['previous_backlogs']}", ln=True)
            pdf.ln(5)
            
            # Subject Averages Table
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Calculated Subject Averages", ln=True)
            
            # Table Header
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(240, 240, 240)
            col_widths = [65, 25, 25, 25, 35]
            headers = ["Subject", "IA1", "IA2", "IA3", "Subject Average"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
            pdf.ln()
            
            # Table Rows
            pdf.set_font("Helvetica", "", 10)
            for row in table_data:
                for i, item in enumerate(row):
                    align = "L" if i == 0 else "C"
                    pdf.cell(col_widths[i], 8, item, border=1, align=align)
                pdf.ln()
            pdf.ln(5)
            
            # Other Data
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Other Academic Data", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"- Attendance: {lr['attendance']}%", ln=True)
            pdf.cell(0, 8, f"- Assignment Average: {lr['assignment_average']}%", ln=True)
            pdf.cell(0, 8, f"- Assignment Completion: {lr['assignment_completion']}%", ln=True)
            pdf.cell(0, 8, f"- Quiz Average: {lr['quiz_average']}%", ln=True)
            pdf.cell(0, 8, f"- Study Hours / Day: {lr['study_hours']}", ln=True)
            
            # Generate PDF bytes
            pdf_bytes = bytes(pdf.output(dest='S'))
            
            st.download_button(
                label="⬇️ Download Detailed PDF Report",
                data=pdf_bytes,
                file_name=f"{s['student_id']}_report.pdf",
                mime="application/pdf",
                type="primary"
            )
        except ImportError:
            st.error("Please install fpdf2 to download PDF: python -m pip install fpdf2")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")