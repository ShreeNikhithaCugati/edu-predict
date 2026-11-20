"""Admin: Batch prediction via CSV upload + Dynamic XAI."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import lime
import lime.lime_tabular
import plotly.express as px
import json

from database.database import query, execute
from recommendations.recommendation_engine import generate_recommendations
from visualization.charts import ia_trend_chart

FEATURE_COLS = [
    "previous_semester_marks", "previous_sgpa", "previous_backlogs",
    "ia1_score", "ia2_score", "ia3_score",
    "attendance", "assignment_average", "assignment_completion",
    "quiz_average", "study_hours",
]

# --- DEPARTMENT-WISE SUBJECT MAPPING ---
SUBJECTS_BY_DEPT_SEM = {
    "CSE": {
        1: ["Mathematics-I", "Applied Physics", "Programming in C", "Introduction to Electronics", "Engineering Drawing", "C Lab"],
        2: ["Mathematics-II", "Applied Chemistry", "Python Programming", "Introduction to AI", "Electrical Engineering", "Communication Skills"],
        3: ["Mathematics for Computer Science", "Digital Design and Computer Organization", "Operating Systems", "Data Structures and Applications"],
        4: ["Analysis and Design of Algorithms", "Microcontrollers", "DBMS", "Discrete Mathematics", "Biology for Computer Engineers"],
        5: ["Software Engineering", "Computer Networks", "Theory of Computation", "Web Technology"],
        6: ["Cloud Computing", "Machine Learning", "Professional Electives", "Open Electives"],
    },
    "ISE": {
        1: ["Mathematics-I", "Applied Physics", "Programming in C", "Introduction to Electronics", "Engineering Drawing", "C Lab"],
        2: ["Mathematics-II", "Applied Chemistry", "Python Programming", "Introduction to AI", "Electrical Engineering", "Communication Skills"],
        3: ["Mathematics for Computer Science", "Digital Design and Computer Organization", "Operating Systems", "Data Structures and Applications"],
        4: ["Analysis and Design of Algorithms", "Advanced Java", "DBMS", "Discrete Mathematics", "Biology for Computer Engineers"],
        5: ["Software Engineering", "Computer Networks", "Theory of Computation", "Data Visualization"],
        6: ["Full Stack Development", "Machine Learning", "Professional Electives", "Open Electives"],
    },
    "ECE": {
        1: ["Mathematics-I", "Applied Physics", "Basic Electrical Engineering", "Electronics Engineering Drawing", "Physics Lab"],
        2: ["Mathematics-II", "Applied Chemistry", "C Programming", "Basic Electronics", "Engineering Mechanics", "Chemistry Lab"],
        3: ["Mathematics", "Electronic Devices", "Digital System Design", "Network Theory", "Probability and Statistics"],
        4: ["Communication Systems", "Analog Circuits", "Signals and Systems", "Verilog HDL", "Computer Architecture"],
        5: ["Digital Signal Processing", "Digital Communication", "Electromagnetic Waves", "Management and Entrepreneurship"],
        6: ["Digital Communication", "Embedded Systems", "Computer Networks", "Professional Electives"],
    }
}

def get_subjects(dept, sem):
    dept = dept.upper().strip() if dept else "CSE"
    if dept in SUBJECTS_BY_DEPT_SEM and sem in SUBJECTS_BY_DEPT_SEM[dept]:
        return SUBJECTS_BY_DEPT_SEM[dept][sem]
    return SUBJECTS_BY_DEPT_SEM["CSE"].get(sem, ["General Subject"])

def categorize(marks):
    if marks >= 85: return "Excellent"
    if marks >= 70: return "Good"
    if marks >= 50: return "Average"
    return "At Risk"

# --- TAB 1: BATCH PREDICTION ---
def render_batch_prediction():
    st.markdown("## 📁 Batch Prediction")
    st.write("Upload a CSV file containing student data to predict final marks for all of them at once.")
    
    model_features = FEATURE_COLS
    template_cols = ["student_id", "student_name"] + model_features
    
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV Template",
        data=csv_template,
        file_name="batch_prediction_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Student Data (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            missing = [c for c in model_features if c not in df.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing}")
                return
            
            if "student_id" not in df.columns:
                st.warning("Warning: 'student_id' column is missing. Using row numbers instead.")
                df["student_id"] = df.index

            st.success(f"Loaded {len(df)} students.")
            st.dataframe(df, use_container_width=True, height=250)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Predicting..."):
                    model_path = "saved_models/xgboost_model.pkl"
                    if not os.path.exists(model_path):
                        st.error("XGBoost model not found. Please train models first.")
                        return
                        
                    model = joblib.load(model_path)
                    
                    X = df[model_features].fillna(df[model_features].median()).to_numpy()
                    preds = model.predict(X)
                    
                    df["Predicted_Final_Marks"] = np.round(preds, 2)
                    df["Performance_Category"] = [categorize(p) for p in preds]
                    
                    st.success("Batch prediction complete!")
                    
                    display_cols = ["student_id"]
                    if "student_name" in df.columns:
                        display_cols.append("student_name")
                    display_cols.extend(["Predicted_Final_Marks", "Performance_Category"])
                    
                    st.dataframe(df[display_cols], use_container_width=True, height=400)
                    
                    result_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=result_csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv",
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")


# --- TAB 2: SINGLE PREDICTION ---
def render_single_prediction():
    st.markdown("## 👤 Single Student Prediction")
    st.write("Enter the academic details to predict the final marks instantly.")
    
    with st.form("admin_single_predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            prev_marks = st.number_input("Previous Semester Marks (%)", 0.0, 100.0, 65.0)
            prev_sgpa = st.number_input("Previous SGPA", 0.0, 10.0, 7.5)
            backlogs = st.number_input("Previous Backlogs", 0, 20, 0)
        with c2:
            ia1 = st.number_input("IA1 Score", 0.0, 50.0, 30.0)
            ia2 = st.number_input("IA2 Score", 0.0, 50.0, 32.0)
            ia3 = st.number_input("IA3 Score", 0.0, 50.0, 35.0)
        with c3:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0)
            assignment_avg = st.number_input("Assignment Avg (%)", 0.0, 100.0, 80.0)
            assignment_compl = st.number_input("Assignment Compl (%)", 0.0, 100.0, 90.0)
            
        c4, c5 = st.columns(2)
        with c4:
            quiz_avg = st.number_input("Quiz Avg (%)", 0.0, 100.0, 75.0)
        with c5:
            study_hours = st.number_input("Study Hours / Day", 0.0, 24.0, 3.5)

        if st.form_submit_button("🚀 Predict Final Marks", type="primary"):
            model_path = "saved_models/xgboost_model.pkl"
            if not os.path.exists(model_path):
                st.error("XGBoost model not found. Please train models first.")
                return
                
            model = joblib.load(model_path)
            
            X = np.array([[
                prev_marks, prev_sgpa, backlogs,
                ia1, ia2, ia3,
                attendance, assignment_avg, assignment_compl,
                quiz_avg, study_hours
            ]])
            
            pred = float(model.predict(X)[0])
            cat = categorize(pred)
            
            st.success("Prediction Successful!")
            st.metric("Predicted Final Marks", f"{pred:.1f}")
            st.info(f"**Performance Category:** {cat}")
            
            if cat == "At Risk":
                st.error("This student is at risk of failing. Immediate intervention recommended.")
            elif cat == "Excellent":
                st.balloons()


# --- TAB 3: DYNAMIC XAI PREDICTION ---
def render_admin_xai_prediction():
    st.markdown("## 🔮 Admin Student Prediction & XAI")
    st.write("Select a student from the database to generate a prediction with full Explainable AI (SHAP & LIME).")
    
    students = query("SELECT student_id, student_name FROM students ORDER BY student_id")
    if not students:
        st.warning("No students found in the database. Add students first.")
        return

    sid = st.selectbox(
        "Select Student",
        [s["student_id"] for s in students],
        format_func=lambda x: f"{x} — {next(s['student_name'] for s in students if s['student_id'] == x)}",
        key="admin_xai_sid"
    )
    
    record = query(
        "SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
        (sid,)
    )
    
    if not record:
        st.warning("No academic record found for this student. A teacher must enter their marks first.")
        return
        
    r = record[0]

    json_path = f"data/subject_records/{sid}.json"
    saved_subject_marks = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved_subject_marks = json.load(f)
    
    st.subheader(f"📋 Current Data for {sid}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IA1 Avg", f"{r['ia1_score']}/{r['ia1_max_marks']}")
    c2.metric("IA2 Avg", f"{r['ia2_score']}/{r['ia2_max_marks']}")
    c3.metric("IA3 Avg", f"{r['ia3_score']}/{r['ia3_max_marks']}")
    c4.metric("Attendance", f"{r['attendance']}%")

    if st.button("🚀 Run Prediction & Explain", type="primary"):
        ml_path = "saved_models/dynamic_xgb_model.pkl"
        feat_path = "saved_models/dynamic_features.json"
        
        if not os.path.exists(ml_path):
            st.error("Dynamic Model not found. Please retrain models in the Models tab.")
            return
            
        ml_model = joblib.load(ml_path)
        with open(feat_path, "r") as f:
            dynamic_features = json.load(f)

        X_dict = {
            "previous_semester_marks": float(r["previous_semester_marks"]) if r["previous_semester_marks"] else 0.0,
            "previous_sgpa": float(r["previous_sgpa"]) if r["previous_sgpa"] else 0.0,
            "previous_backlogs": int(r["previous_backlogs"]) if r["previous_backlogs"] else 0,
            "attendance": float(r["attendance"]) if r["attendance"] else 0.0,
            "assignment_average": float(r["assignment_average"]) if r["assignment_average"] else 0.0,
            "quiz_average": float(r["quiz_average"]) if r["quiz_average"] else 0.0,
            "study_hours": float(r["study_hours"]) if r["study_hours"] else 0.0,
        }

        for subj in ["Data Structures", "Operating Systems", "DBMS", "Computer Networks", "Software Engineering"]:
            marks = saved_subject_marks.get(subj, [0, 0, 0])
            X_dict[f"{subj}_IA1"] = marks[0]
            X_dict[f"{subj}_IA2"] = marks[1]
            X_dict[f"{subj}_IA3"] = marks[2]

        X_df = pd.DataFrame([X_dict])[dynamic_features]
        X = X_df.to_numpy()

        # 1. Prediction
        pred = float(ml_model.predict(X)[0])
        cat = categorize(pred)
        
        st.divider()
        st.subheader("Prediction Results")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Final Marks", f"{pred:.1f}")
        c2.metric("Performance Category", cat)

        # 2. IA Trend Chart
        st.subheader("📊 IA Trajectory")
        fig = ia_trend_chart(sid)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # 3. SHAP
        st.subheader("🧠 SHAP Explanation (Feature Impact)")
        try:
            explainer = shap.TreeExplainer(ml_model)
            sv = explainer.shap_values(X)[0]
            
            contrib = pd.DataFrame({
                "Feature": dynamic_features,
                "Value": X[0],
                "SHAP Value": sv
            }).sort_values("SHAP Value", key=lambda s: s.abs(), ascending=True)
            
            contrib["Impact"] = contrib["SHAP Value"].apply(lambda x: "Positive (Helps)" if x > 0 else "Negative (Hurts)")
            
            fig = px.bar(
                contrib.tail(15), 
                x="SHAP Value", 
                y="Feature",
                orientation="h",
                color="Impact",
                color_discrete_map={"Positive (Helps)": "#2ecc71", "Negative (Hurts)": "#e74c3c"},
                title="How each feature impacted the prediction",
                text="SHAP Value"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Impact on Final Marks", yaxis_title="Feature", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
            contrib = None

        # 4. LIME
        st.subheader("🔍 LIME Explanation (Local Impact)")
        try:
            dummy_bg = np.zeros((50, len(dynamic_features)))
            for i, feat in enumerate(dynamic_features):
                if "IA1" in feat or "IA2" in feat or "IA3" in feat:
                    dummy_bg[:, i] = 30.0
                elif feat == "attendance":
                    dummy_bg[:, i] = 75.0
                elif feat == "previous_semester_marks":
                    dummy_bg[:, i] = 65.0
                elif feat == "previous_sgpa":
                    dummy_bg[:, i] = 6.5
                elif feat == "previous_backlogs":
                    dummy_bg[:, i] = 0
                else:
                    dummy_bg[:, i] = 3.0

            bg_df = pd.DataFrame(dummy_bg, columns=dynamic_features)
            expl = lime.lime_tabular.LimeTabularExplainer(
                bg_df.to_numpy(), feature_names=dynamic_features, mode="regression", discretize_continuous=True
            )
            exp = expl.explain_instance(X[0], ml_model.predict, num_features=5)
            
            for feat, weight in exp.as_list():
                if weight > 0:
                    st.write(f"- 🟢 **{feat}** (Impact: `+{weight:.2f}` marks)")
                else:
                    st.write(f"- 🔴 **{feat}** (Impact: `{weight:.2f}` marks)")
        except Exception as e:
            st.warning(f"LIME unavailable: {e}")

        # 5. Recommendations
        st.subheader("💡 Personalized Recommendations")
        recs = generate_recommendations(r, contrib.to_dict("records") if contrib is not None else None)
        for rec in recs:
            st.markdown(f"- {rec}")


# --- TAB 4: DYNAMIC SUBJECT-WISE PREDICTION ---
def render_subject_wise_prediction():
    st.markdown("## 📚 Subject-Wise & Overall Prediction")
    st.write("Enter IA marks for each subject to predict subject-wise finals and the overall semester performance.")
    
    # We need the student's department and semester
    students = query("SELECT student_id, student_name, department, semester FROM students ORDER BY student_id")
    if not students:
        st.warning("No students found. Add students first.")
        return

    sid = st.selectbox(
        "Select Student for Semester Report",
        [s["student_id"] for s in students],
        format_func=lambda x: f"{x} — {next(s['student_name'] for s in students if s['student_id'] == x)}",
        key="subject_wise_sid"
    )
    
    selected_student = next(s for s in students if s["student_id"] == sid)
    department = selected_student["department"]
    semester = selected_student["semester"]
    
    # Fetch the dynamic subjects based on the student's department AND semester
    subjects = get_subjects(department, semester)
    
    record = query("SELECT * FROM academic_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1", (sid,))
    if not record:
        st.warning("No academic record found. Please enter previous data first.")
        return
        
    r = record[0]
    
    st.info(f"Please enter the 3 IA marks (out of 50) for each of the {len(subjects)} subjects in **{department} Semester {semester}** below.")
    
    with st.form("subject_wise_form"):
        subject_marks = {}
        
        # Create dynamic columns (3 per row) for better layout
        cols = st.columns(3)
        for i, subj in enumerate(subjects):
            with cols[i % 3]:
                st.markdown(f"**{subj}**")
                subject_marks[subj] = (
                    st.number_input(f"IA1 ({subj})", 0.0, 50.0, 35.0, key=f"ia1_{i}"),
                    st.number_input(f"IA2 ({subj})", 0.0, 50.0, 38.0, key=f"ia2_{i}"),
                    st.number_input(f"IA3 ({subj})", 0.0, 50.0, 40.0, key=f"ia3_{i}")
                )
                
        if st.form_submit_button("🚀 Generate Semester Report", type="primary"):
            ml_path = "saved_models/dynamic_xgb_model.pkl"
            feat_path = "saved_models/dynamic_features.json"
            
            if not os.path.exists(ml_path):
                st.error("Dynamic Model not found. Please retrain.")
                return
                
            ml_model = joblib.load(ml_path)
            with open(feat_path, "r") as f:
                dynamic_features = json.load(f)

            st.divider()
            st.subheader(f"📊 Semester {semester} Report for {sid}")
            
            results = []
            total_predicted = 0
            
            for subj, (ia1, ia2, ia3) in subject_marks.items():
                X_dict = {
                    "previous_semester_marks": float(r["previous_semester_marks"]) if r["previous_semester_marks"] else 0.0,
                    "previous_sgpa": float(r["previous_sgpa"]) if r["previous_sgpa"] else 0.0,
                    "previous_backlogs": int(r["previous_backlogs"]) if r["previous_backlogs"] else 0,
                    "attendance": float(r["attendance"]) if r["attendance"] else 0.0,
                    "assignment_average": float(r["assignment_average"]) if r["assignment_average"] else 0.0,
                    "quiz_average": float(r["quiz_average"]) if r["quiz_average"] else 0.0,
                    "study_hours": float(r["study_hours"]) if r["study_hours"] else 0.0,
                }

                # The Dynamic model was trained on 5 specific core subjects. 
                core_subjects = ["Data Structures", "Operating Systems", "DBMS", "Computer Networks", "Software Engineering"]
                avg_all_ia1 = sum(m[0] for m in subject_marks.values()) / len(subjects)
                avg_all_ia2 = sum(m[1] for m in subject_marks.values()) / len(subjects)
                avg_all_ia3 = sum(m[2] for m in subject_marks.values()) / len(subjects)
                
                for s in core_subjects:
                    # Fuzzy match: if the subject name contains the core subject name
                    if s in subj or subj in s:
                        X_dict[f"{s}_IA1"] = ia1
                        X_dict[f"{s}_IA2"] = ia2
                        X_dict[f"{s}_IA3"] = ia3
                    else:
                        X_dict[f"{s}_IA1"] = avg_all_ia1
                        X_dict[f"{s}_IA2"] = avg_all_ia2
                        X_dict[f"{s}_IA3"] = avg_all_ia3

                X_df = pd.DataFrame([X_dict])[dynamic_features]
                X = X_df.to_numpy()
                
                pred = float(ml_model.predict(X)[0])
                total_predicted += pred
                results.append({"Subject": subj, "Predicted Final Marks": round(pred, 2), "Category": categorize(pred)})
                
            overall_avg = total_predicted / len(subjects)
            overall_cat = categorize(overall_avg)
            
            st.markdown("### 🏆 Overall Semester Performance")
            c1, c2 = st.columns(2)
            c1.metric("Predicted Overall Average", f"{overall_avg:.2f}%")
            c2.metric("Overall Category", overall_cat)
            
            st.markdown("### 📚 Subject-Wise Breakdown")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            fig = px.bar(
                df_results, 
                x="Subject", 
                y="Predicted Final Marks",
                color="Category",
                title="Subject-Wise Predicted Performance",
                color_discrete_map={"Excellent": "#2ecc71", "Good": "#3498db", "Average": "#f39c12", "At Risk": "#e74c3c"}
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)