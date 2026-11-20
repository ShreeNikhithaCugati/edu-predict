import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# The complete dictionary from your app
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

print("Loading data...")
df = pd.read_csv('data/raw/training_data.csv')

# Scale IAs to 50-mark system
df['ia1_score'] = df['ia1_score'] / 30.0 * 50.0
df['ia2_score'] = df['ia2_score'] / 30.0 * 50.0
df['ia3_score'] = df['ia3_score'] / 30.0 * 50.0

# Recompute target
df['actual_final_marks'] = (
    0.20 * df['previous_semester_marks'] +
    0.40 * (df['ia1_score']/50*100) +
    0.20 * (df['ia2_score']/50*100) +
    0.10 * (df['ia3_score']/50*100) +
    0.10 * df['attendance']
).clip(0, 100)

# Create a master list of unique subjects across ALL departments
all_subjects = set()
for dept, sems in SUBJECTS_BY_DEPT_SEM.items():
    for sem, subjs in sems.items():
        for s in subjs:
            all_subjects.add(s)

feature_cols = ['previous_semester_marks', 'previous_sgpa', 'previous_backlogs', 'attendance', 'assignment_average', 'quiz_average', 'study_hours']

# Generate columns for every unique subject
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

pred = model.predict(Xte)
print(f"R2 Score: {r2_score(yte, pred):.3f}")

joblib.dump(model, 'saved_models/dynamic_xgb_model.pkl')
with open('saved_models/dynamic_features.json', 'w') as f:
    json.dump(feature_cols, f)

print("✅ Model retrained with ALL subjects for ALL semesters.")