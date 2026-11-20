"""Teacher: Prediction History & Reports."""
import streamlit as st
import pandas as pd
from database.database import query, execute
from authentication.auth_service import log_activity

def render_reports():
    st.markdown("## 📜 Prediction History")

    teacher_id = st.session_state.user["id"]
    
    # Fetch all students assigned to this teacher for filtering
    students = query("SELECT student_id, student_name FROM students WHERE assigned_teacher_id = ?", (teacher_id,))
    if not students:
        st.info("No students assigned to you.")
        return

    # --- FILTERS ---
    st.write("### 🔍 Filters")
    f1, f2 = st.columns(2)
    
    with f1:
        sid = st.selectbox(
            "Select student",
            ["All Students"] + [s["student_id"] for s in students],
            format_func=lambda x: f"{x} — {next(s['student_name'] for s in students if s['student_id'] == x)}" if x != "All Students" else "All Students",
            key="history_student"
        )
    with f2:
        sort_order = st.selectbox("Sort by Date", ["Newest First", "Oldest First"])

    # --- QUERY BUILDING ---
    base_sql = "SELECT id, student_id, predicted_marks, performance_category, selected_model, prediction_date FROM predictions WHERE teacher_id = ?"
    params = [teacher_id]
    
    if sid != "All Students":
        base_sql += " AND student_id = ?"
        params.append(sid)
        
    if sort_order == "Newest First":
        base_sql += " ORDER BY prediction_date DESC"
    else:
        base_sql += " ORDER BY prediction_date ASC"
        
    history = query(base_sql, tuple(params))

    if not history:
        st.info("No predictions found for the selected filter.")
        return

    # --- DISPLAY TABLE ---
    df_history = pd.DataFrame(history)
    df_history.columns = ["ID", "Student ID", "Predicted Marks", "Category", "Model Used", "Date & Time"]
    
    st.dataframe(
        df_history[["Student ID", "Predicted Marks", "Category", "Model Used", "Date & Time"]],
        use_container_width=True,
        hide_index=True
    )
    
    # CSV Download
    csv = df_history[["Student ID", "Predicted Marks", "Category", "Model Used", "Date & Time"]].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    # --- DELETE SECTION ---
    st.divider()
    st.markdown("### 🗑️ Delete a Prediction")
    
    pred_ids = df_history["ID"].tolist()
    pred_labels = [f"{row['Date & Time']} | {row['Student ID']} | {row['Predicted Marks']} ({row['Category']})" for _, row in df_history.iterrows()]
    
    selected_pred = st.selectbox(
        "Select a prediction to delete",
        options=pred_ids,
        format_func=lambda x: pred_labels[pred_ids.index(x)],
        key="del_pred_select"
    )
    
    if st.button("❌ Delete Selected Prediction", type="primary"):
        execute("DELETE FROM predictions WHERE id = ?", (selected_pred,))
        log_activity(teacher_id, "delete_prediction", f"Deleted prediction #{selected_pred}")
        st.success("Prediction deleted successfully!")
        st.rerun()