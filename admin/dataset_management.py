"""Admin: upload and validate training dataset."""
import streamlit as st
import pandas as pd
import os
from authentication.auth_service import log_activity

EXPECTED_COLS = [
    "student_id", "previous_semester_marks", "previous_sgpa",
    "previous_backlogs", "ia1_score", "ia2_score", "ia3_score",
    "attendance", "assignment_average", "assignment_completion", # <--- ADD THIS
    "quiz_average", "study_hours", "actual_final_marks",
]

RAW_DIR = "data/raw"


def render_dataset_management():
    st.markdown("## 📊 Dataset Management")

    uploaded = st.file_uploader("Upload training CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            return

        st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Validation")
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLS]
        c1, c2 = st.columns(2)
        with c1:
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success("All expected columns present.")
        with c2:
            if extra:
                st.warning(f"Extra columns (ignored): {extra}")

        st.subheader("Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Missing Values")
        mv = df.isnull().sum()
        st.dataframe(mv[mv > 0].reset_index().rename(
            columns={"index": "column", 0: "missing"}))

        st.subheader("Duplicates")
        st.write(f"Duplicate rows: {df.duplicated().sum()}")

        st.subheader("Target Variable Distribution (Actual Final Marks)")
        import plotly.express as px
        fig = px.histogram(df, x="actual_final_marks", nbins=20, 
                           title="Distribution of Final Marks in Dataset",
                           color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig, use_container_width=True)

        if not missing:
            if st.button("💾 Save as active training dataset"):
                os.makedirs(RAW_DIR, exist_ok=True)
                path = os.path.join(RAW_DIR, "training_data.csv")
                df.to_csv(path, index=False)
                log_activity(st.session_state.user["id"], "upload_dataset",
                             f"Saved {len(df)} rows to {path}")
                st.success(f"Saved to {path}")