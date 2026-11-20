
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
