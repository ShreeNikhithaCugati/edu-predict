"""Admin: view model registry."""
import streamlit as st
import json
from database.database import query
import subprocess
import sys
import time


def render_model_management():
    st.markdown("## 🤖 Model Registry")
    
    # --- Retrain Button ---
    st.write("Admin Action: Retrain models using the active dataset.")
    if st.button("🔄 Retrain ML & DL Models Now", type="primary"):
        with st.spinner("Training base ML & DL models..."):
            # 1. Train Base Models
            result_ml = subprocess.run([sys.executable, "-m", "machine_learning.train_ml"], capture_output=True, text=True)
            result_dl = subprocess.run([sys.executable, "-m", "deep_learning.train_ann"], capture_output=True, text=True)
            
            if result_ml.returncode != 0:
                st.error(f"Base ML training failed: {result_ml.stderr}")
                return
            
            # 2. Train Dynamic Model
            with st.spinner("Training Dynamic Subject-Wise model..."):
                dynamic_script = """
import pandas as pd, numpy as np, joblib, xgboost as xgb, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('data/raw/training_data.csv')

# Scale IAs to 50-mark system
df['ia1_score'] = df['ia1_score'] / 30.0 * 50.0
df['ia2_score'] = df['ia2_score'] / 30.0 * 50.0
df['ia3_score'] = df['ia3_score'] / 30.0 * 50.0

# Map to subjects (no random noise this time!)
subjects = ['Data Structures', 'Operating Systems', 'DBMS', 'Computer Networks', 'Software Engineering']
for subj in subjects:
    df[f'{subj}_IA1'] = df['ia1_score']
    df[f'{subj}_IA2'] = df['ia2_score']
    df[f'{subj}_IA3'] = df['ia3_score']

# Recompute target to match the 50-mark scale logic
df['actual_final_marks'] = (
    0.20 * df['previous_semester_marks'] +
    0.40 * (df['ia1_score']/50*100) +
    0.20 * (df['ia2_score']/50*100) +
    0.10 * (df['ia3_score']/50*100) +
    0.10 * df['attendance']
).clip(0, 100)

y = df['actual_final_marks']
feature_cols = ['previous_semester_marks', 'previous_sgpa', 'previous_backlogs', 'attendance', 'assignment_average', 'quiz_average', 'study_hours']
for subj in subjects:
    feature_cols.extend([f'{subj}_IA1', f'{subj}_IA2', f'{subj}_IA3'])

X = df[feature_cols].fillna(0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(f'R2: {r2_score(yte, pred):.3f}')

joblib.dump(model, 'saved_models/dynamic_xgb_model.pkl')
with open('saved_models/dynamic_features.json', 'w') as f:
    json.dump(feature_cols, f)
print("Dynamic model updated")
"""
                # Write the script to a temp file and run it
                with open("temp_train_dynamic.py", "w") as f:
                    f.write(dynamic_script)
                
                result_dyn = subprocess.run([sys.executable, "temp_train_dynamic.py"], capture_output=True, text=True)
                
                if result_dyn.returncode != 0:
                    st.error(f"Dynamic Model training failed: {result_dyn.stderr}")
                else:
                    st.success("✅ All models retrained successfully! Teacher dashboard will now use the updated model.")
                    time.sleep(1)
                    st.rerun()

    st.divider()

    # Show the registry
    rows = query("SELECT * FROM model_registry ORDER BY training_date DESC")
    if not rows:
        st.info("No models registered yet. Train models in Phase 5.")
        return

    for r in rows:
        active = "🟢 ACTIVE" if r["active_status"] else "⚪ inactive"
        with st.expander(f"{active} — {r['model_name']} v{r['model_version']} ({r['model_type'].upper()})"):
            st.write(f"**Trained:** {r['training_date']}")
            if r["evaluation_metrics"]:
                try:
                    metrics = json.loads(r["evaluation_metrics"])
                    st.json(metrics)
                except Exception:
                    st.text(r["evaluation_metrics"])