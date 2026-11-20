"""Teacher: prediction page with XAI (Dynamic Subject-Wise)."""
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import plotly.express as px
import json
from datetime import datetime

from database.database import query, execute
from authentication.auth_service import log_activity
from recommendations.recommendation_engine import generate_recommendations
from admin.batch_prediction import SUBJECTS_BY_DEPT_SEM
from teacher.xai_helpers import (
    build_model_input,
    grouped_lime_background,
    grouped_lime_inputs,
    lime_rows_from_explanation,
    make_grouped_predictor,
    shap_frame,
    split_shap,
    subject_shap_rows,
    subject_table,
)

PREDICTION_VERSION = 2


def categorize(marks):
    if marks >= 90: return "Excellent"
    if marks >= 75: return "Good"
    if marks >= 50: return "Average"
    return "At Risk"


def render_prediction_page():
    st.markdown("## 🔮 Performance Prediction")

    teacher_id = st.session_state.user["id"]

    # --- FILTERS ---
    st.write("### 🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)

    depts_data = query("SELECT DISTINCT department FROM students WHERE assigned_teacher_id = ?", (teacher_id,))
    dept_options = ["All"] + [d["department"] for d in depts_data if d["department"]]
    years_data = query("SELECT DISTINCT academic_year FROM students WHERE assigned_teacher_id = ?", (teacher_id,))
    year_options = ["All"] + [y["academic_year"] for y in years_data if y["academic_year"]]

    with f1: dept_filter = st.selectbox("Department", dept_options, key="p_dept")
    with f2: year_filter = st.selectbox("Batch Year", year_options, key="p_year")
    with f3: sem_filter = st.selectbox("Semester", ["All", 1, 2, 3, 4, 5, 6, 7, 8], key="p_sem")
    with f4: search_id = st.text_input("Search Student ID", placeholder="e.g. 1CR23...", key="p_search")

    base_sql = "SELECT student_id, student_name FROM students WHERE assigned_teacher_id = ?"
    params = [teacher_id]
    if dept_filter != "All": base_sql += " AND department = ?"; params.append(dept_filter)
    if year_filter != "All": base_sql += " AND academic_year = ?"; params.append(year_filter)
    if sem_filter != "All": base_sql += " AND semester = ?"; params.append(sem_filter)
    if search_id: base_sql += " AND student_id LIKE ?"; params.append(f"%{search_id}%")
    base_sql += " ORDER BY student_id"

    students = query(base_sql, tuple(params))

    if not students:
        st.warning("No students found matching your filters.")
        return

    sid = st.selectbox(
        "Select student to predict",
        [s["student_id"] for s in students],
        format_func=lambda x: f"{x} — {next(s['student_name'] for s in students if s['student_id'] == x)}",
        key="p_student_select"
    )

    record = query("SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1", (sid,))
    if not record:
        st.warning("No academic record found. Add one under 'My Students' first.")
        return

    r = record[0]

    json_path = f"data/subject_records/{sid}.json"
    saved_subject_marks = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved_subject_marks = json.load(f)

    st.markdown("### 📸 Subject-Wise IA Breakdown")
    if saved_subject_marks:
        chart_data = []
        for subj, marks in saved_subject_marks.items():
            chart_data.append({"Subject": subj, "Test": "IA1", "Score": marks[0]})
            chart_data.append({"Subject": subj, "Test": "IA2", "Score": marks[1]})
            chart_data.append({"Subject": subj, "Test": "IA3", "Score": marks[2]})

        df_chart = pd.DataFrame(chart_data)
        fig = px.bar(df_chart, x="Subject", y="Score", color="Test", barmode="group",
                     title="IA Marks per Subject (Out of 50)", color_discrete_sequence=["#3498db", "#f39c12", "#e74c3c"])
        fig.update_layout(yaxis_range=[0, 55])
        st.plotly_chart(fig, use_container_width=True)

    if "pred_results" not in st.session_state:
        st.session_state["pred_results"] = {}

    if st.button("🚀 Run Prediction & Explain", type="primary"):
        ml_path = "saved_models/dynamic_xgb_model.pkl"
        feat_path = "saved_models/dynamic_features.json"

        if not os.path.exists(ml_path):
            st.error("Dynamic Model not found. Please run retrain_final.py.")
            return

        ml_model = joblib.load(ml_path)
        with open(feat_path, "r") as f:
            dynamic_features = json.load(f)

        if not saved_subject_marks:
            st.warning("No subject-wise marks found. Save marks under 'My Students' first.")
            return

        # --- BUILD THE MODEL ROW THE WAY THE MODEL WAS TRAINED ---
        # Training copied one IA score into every subject column, so all
        # subject columns must carry the student's per-test average. The old
        # code set the other 48 subjects to 0, which the model had never seen
        # and produced a bogus "Applied Physics" explanation.
        base_values = {
            "previous_semester_marks": float(r["previous_semester_marks"]) if r["previous_semester_marks"] else 0.0,
            "previous_sgpa": float(r["previous_sgpa"]) if r["previous_sgpa"] else 0.0,
            "previous_backlogs": int(r["previous_backlogs"]) if r["previous_backlogs"] else 0,
            "attendance": float(r["attendance"]) if r["attendance"] else 0.0,
            "assignment_average": float(r["assignment_average"]) if r["assignment_average"] else 0.0,
            "quiz_average": float(r["quiz_average"]) if r["quiz_average"] else 0.0,
            "study_hours": float(r["study_hours"]) if r["study_hours"] else 0.0,
        }

        X, per_test_avg = build_model_input(base_values, saved_subject_marks, dynamic_features)

        # Prediction
        pred = float(ml_model.predict(X)[0])
        cat = categorize(pred)

        # --- SHAP: behaviour + subject block ---
        explainer = shap.TreeExplainer(ml_model)
        sv = explainer.shap_values(X)[0]
        base_value = float(np.ravel(explainer.expected_value)[0])

        behaviour, subject_total = split_shap(
            dynamic_features, X[0], sv, saved_subject_marks
        )
        subject_rows = subject_shap_rows(saved_subject_marks, subject_total)
        contrib = shap_frame(behaviour, subject_rows)

        # --- SUBJECT TABLE (matches the IA bar graph) ---
        subj_df = subject_table(saved_subject_marks)

        # --- LIME ---
        lime_results = []
        try:
            lime_instance, lime_names, behaviour_features, subject_names = grouped_lime_inputs(
                base_values, saved_subject_marks, dynamic_features
            )
            predict_grouped = make_grouped_predictor(
                ml_model, dynamic_features, len(subject_names)
            )
            training_df = pd.read_csv("data/raw/training_data.csv")
            background = grouped_lime_background(
                training_df, behaviour_features, len(subject_names)
            )

            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                background,
                feature_names=lime_names,
                mode="regression",
                discretize_continuous=True,
                random_state=42,
            )
            exp = explainer_lime.explain_instance(
                lime_instance, predict_grouped, num_features=12
            )
            lime_results = lime_rows_from_explanation(exp, lime_names)
        except Exception as e:
            st.warning(f"LIME unavailable: {e}")

        recs = generate_recommendations(
            r,
            [row for row in behaviour],
            subject_impacts=subj_df.to_dict("records"),
        )

        st.session_state["pred_results"][sid] = {
            "version": PREDICTION_VERSION,
            "pred": pred,
            "cat": cat,
            "contrib": contrib,
            "subjects": subj_df,
            "base_value": base_value,
            "subject_total": subject_total,
            "per_test_avg": per_test_avg,
            "lime": lime_results,
            "recs": recs,
        }

        save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        execute(
            """INSERT INTO predictions (student_id, teacher_id, predicted_marks, performance_category, selected_model, prediction_date)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sid, teacher_id, pred, cat, "Dynamic XGBoost", save_time)
        )
        log_activity(teacher_id, "run_prediction", f"Predicted {pred:.1f} for {sid}")
        st.rerun()

    # --- DISPLAY RESULTS ---
    res = st.session_state["pred_results"].get(sid)
    if res and res.get("version") == PREDICTION_VERSION:
        st.divider()
        st.subheader("📊 Prediction Results")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Final Marks", f"{res['pred']:.1f}")
        c2.metric("Performance Category", res['cat'])

        st.markdown("### 🧠 SHAP Explanation")
        st.caption(
            f"SHAP is measured against the model's average prediction "
            f"({res['base_value']:.1f} marks). The model learned every subject "
            "column as one common IA level, so the subject block is split "
            "across the student's actual subjects."
        )
        if not res["contrib"].empty:
            fig_shap = px.bar(
                res["contrib"], x="SHAP Value", y="Display Feature", orientation="h",
                color="Impact", color_discrete_map={"Positive (Helps)": "#2ecc71", "Negative (Hurts)": "#e74c3c"},
                title="What moved this prediction (marks)", text="SHAP Value"
            )
            fig_shap.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_shap.update_layout(xaxis_title="Impact on Final Marks", yaxis_title="", height=480)
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("No SHAP contributions to display.")

        st.markdown("### 📚 Subject-Wise Impact (from the bar graph)")
        st.caption(
            "Weighted score = 40% IA1 + 20% IA2 + 10% IA3 (out of 50). "
            "The **Focus** column is the test with the most room to improve."
        )
        st.dataframe(
            res["subjects"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Weighted %": st.column_config.ProgressColumn(
                    "Weighted %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.markdown("### 🔍 LIME Explanation (Local Impact)")
        if res['lime']:
            for row in res['lime']:
                weight = row["weight"]
                if weight > 0:
                    st.write(f"- 🟢 **{row['label']}** (Impact: `+{weight:.2f}` marks)")
                else:
                    st.write(f"- 🔴 **{row['label']}** (Impact: `{weight:.2f}` marks)")
        else:
            st.info("LIME results not available.")

        st.markdown("### 💡 Personalized Recommendations")
        for rec in res['recs']:
            st.markdown(f"- {rec}")

        st.caption("⚠️ This is an estimate based on historical patterns, not a guaranteed result.")
    else:
        st.info("👆 Click 'Run Prediction & Explain' to generate results for this student.")