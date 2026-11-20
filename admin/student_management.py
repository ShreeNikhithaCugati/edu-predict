"""Admin: manage student records."""
import streamlit as st
import pandas as pd
from database.database import query, execute
from authentication.auth_service import log_activity


def render_student_management():
    st.markdown("## 🎓 Student Management")

    tab1, tab2, tab3 = st.tabs(["📋 View All", "➕ Add Student", "✏️ Edit / Assign / Delete"])

    # --- Tab 1: View ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            depts_data = query("SELECT DISTINCT department FROM students WHERE department IS NOT NULL")
            dept_options = ["All"] + [d["department"] for d in depts_data]
            dept_filter = st.selectbox("Filter by Department", dept_options, key="view_dept_filter")
        with col2:
            sem_filter = st.selectbox("Semester", ["All", 1, 2, 3, 4, 5, 6, 7, 8], key="view_sem_filter")

        sql = """
            SELECT s.student_id, s.student_name, s.department, 
                   s.semester, s.academic_year, 
                   COALESCE(u.full_name, 'Unassigned') AS assigned_teacher,
                   s.created_at
            FROM students s
            LEFT JOIN users u ON s.assigned_teacher_id = u.id
        """
        params = []
        conds = []
        if dept_filter and dept_filter != "All":
            conds.append("s.department = ?")
            params.append(dept_filter)
        if sem_filter != "All":
            conds.append("s.semester = ?")
            params.append(sem_filter)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY s.student_id"

        students = query(sql, tuple(params))
        
        if students:
            st.metric(label="Total Students Found", value=len(students))
        else:
            st.metric(label="Total Students Found", value="0")

        if students:
            df_students = pd.DataFrame(students)
            df_students.insert(0, "Sl No", range(1, len(df_students) + 1))
            df_students.rename(columns={
                "student_id": "Student ID",
                "student_name": "Student Name",
                "department": "Department",
                "semester": "Semester",
                "academic_year": "Academic Year",
                "assigned_teacher": "Assigned Teacher",
                "created_at": "Registered On"
            }, inplace=True)
            st.dataframe(df_students, use_container_width=True, hide_index=True)
        
    # --- Tab 2: Add ---
    with tab2:
        teachers = query(
            "SELECT id, full_name FROM users WHERE role='teacher' AND approval_status='approved'"
        )
        teacher_opts = {"(none)": None} | {t["full_name"]: t["id"] for t in teachers}

        with st.form("add_student"):
            c1, c2 = st.columns(2)
            with c1:
                student_id = st.text_input("Student ID (e.g. 1CR21CS001)")
                name = st.text_input("Student Name")
                department = st.text_input("Department")
            with c2:
                semester = st.number_input("Semester", 1, 8, 1)
                academic_year = st.text_input("Academic Year (e.g. 2024-25)")
                teacher = st.selectbox("Assign Teacher", list(teacher_opts.keys()))

            if st.form_submit_button("Add Student"):
                if not all([student_id, name, department, academic_year]):
                    st.error("Please fill all required fields.")
                else:
                    try:
                        execute(
                            """INSERT INTO students
                               (student_id, student_name, department, semester,
                                academic_year, assigned_teacher_id)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (student_id, name, department, semester,
                             academic_year, teacher_opts[teacher]),
                        )
                        log_activity(st.session_state.user["id"], "add_student",
                                     f"Added {student_id}")
                        st.success(f"Student {student_id} added.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- Tab 3: Edit / Assign / Delete ---
    with tab3:
        students = query("SELECT * FROM students ORDER BY student_id")
        if not students:
            st.info("No students to edit.")
            return
        
        sid = st.selectbox("Select student",
                           [s["student_id"] for s in students], key="edit_sid_select")
        s = next(x for x in students if x["student_id"] == sid)

        # Get all approved teachers for the dropdown
        all_teachers = query("SELECT id, full_name FROM users WHERE role='teacher' AND approval_status='approved'")
        teacher_opts = {"(Unassigned)": None} | {t["full_name"]: t["id"] for t in all_teachers}
        
        # Find the current teacher's name (if assigned) to set as default
        current_teacher_name = next((name for name, tid in teacher_opts.items() if tid == s["assigned_teacher_id"]), "(Unassigned)")

        with st.form("edit_student"):
            name = st.text_input("Name", s["student_name"])
            dept = st.text_input("Department", s["department"])
            sem = st.number_input("Semester", 1, 8, s["semester"])
            ay = st.text_input("Academic Year", s["academic_year"])
            
            # NEW: Assign Teacher Dropdown
            selected_teacher_name = st.selectbox(
                "Assigned Teacher", 
                list(teacher_opts.keys()), 
                index=list(teacher_opts.keys()).index(current_teacher_name)
            )
            
            if st.form_submit_button("Update Record"):
                execute(
                    """UPDATE students SET student_name=?, department=?,
                       semester=?, academic_year=?, assigned_teacher_id=? WHERE student_id=?""",
                    (name, dept, sem, ay, teacher_opts[selected_teacher_name], sid),
                )
                log_activity(st.session_state.user["id"], "update_student", f"Updated/Assigned {sid}")
                st.success(f"Updated {sid} successfully.")
                st.rerun()

        st.divider()
        st.warning(f"Delete {sid}? This removes all their records and predictions.")
        if st.checkbox("Yes, I understand"):
            if st.button("🗑️ Delete Student"):
                execute("DELETE FROM students WHERE student_id = ?", (sid,))
                log_activity(st.session_state.user["id"], "delete_student",
                             f"Deleted {sid}")
                st.success("Deleted.")
                st.rerun()