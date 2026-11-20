import pandas as pd
import numpy as np
import json
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the base dataset
df = pd.read_csv("data/raw/training_data.csv")

# Simulate subject marks for the training data 
# (In a real scenario, you would have a CSV with subject columns)
# We will use the averages from the base dataset to create 5 fake subjects
subjects = ["Data Structures", "Operating Systems", "DBMS", "Computer Networks", "Software Engineering"]

for subj in subjects:
    df[f"{subj}_IA1"] = df["ia1_score"] + np.random.normal(0, 2, len(df))
    df[f"{subj}_IA2"] = df["ia2_score"] + np.random.normal(0, 2, len(df))
    df[f"{subj}_IA3"] = df["ia3_score"] + np.random.normal(0, 2, len(df))

# Create a target variable (Final marks)
y = df["actual_final_marks"]

# Feature columns for the new model
feature_cols = ["previous_semester_marks", "previous_sgpa", "previous_backlogs", "attendance", "assignment_average", "quiz_average", "study_hours"]
for subj in subjects:
    feature_cols.extend([f"{subj}_IA1", f"{subj}_IA2", f"{subj}_IA3"])

X = df[feature_cols].fillna(0)

# Train the model
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5)
model.fit(Xtr, ytr)

pred = model.predict(Xte)
print(f"New Dynamic Model R2 Score: {r2_score(yte, pred):.3f}")

# Save the model and feature list
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/dynamic_xgb_model.pkl")
with open("saved_models/dynamic_features.json", "w") as f:
    json.dump(feature_cols, f)

print("Model and features saved successfully.")