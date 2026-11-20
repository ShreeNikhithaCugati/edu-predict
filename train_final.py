import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from admin.batch_prediction import SUBJECTS_BY_DEPT_SEM

print("Loading data...")
df = pd.read_csv('data/raw/training_data.csv')

# Scale IAs to 50
df['ia1_score'] = df['ia1_score'] / 30.0 * 50.0
df['ia2_score'] = df['ia2_score'] / 30.0 * 50.0
df['ia3_score'] = df['ia3_score'] / 30.0 * 50.0

# Create target
df['actual_final_marks'] = (
    0.20 * df['previous_semester_marks'] +
    0.40 * (df['ia1_score']/50*100) +
    0.20 * (df['ia2_score']/50*100) +
    0.10 * (df['ia3_score']/50*100) +
    0.10 * df['attendance']
).clip(0, 100)

# Collect ALL subjects
all_subjects = set()
for dept, sems in SUBJECTS_BY_DEPT_SEM.items():
    for sem, subjs in sems.items():
        for s in subjs:
            all_subjects.add(s)

feature_cols = ['previous_semester_marks', 'previous_sgpa', 'previous_backlogs', 'attendance', 'assignment_average', 'quiz_average', 'study_hours']

# Create columns for EVERY subject
for subj in all_subjects:
    df[f'{subj}_IA1'] = df['ia1_score']
    df[f'{subj}_IA2'] = df['ia2_score']
    df[f'{subj}_IA3'] = df['ia3_score']
    feature_cols.extend([f'{subj}_IA1', f'{subj}_IA2', f'{subj}_IA3'])

y = df['actual_final_marks']
X = df[feature_cols].fillna(0)

print(f"Training on {len(feature_cols)} features...")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(Xtr, ytr)

print(f"R2 Score: {r2_score(yte, model.predict(Xte)):.3f}")

joblib.dump(model, 'saved_models/dynamic_xgb_model.pkl')
with open('saved_models/dynamic_features.json', 'w') as f:
    json.dump(feature_cols, f)

print("✅ Model retrained with ALL subjects for ALL semesters.")