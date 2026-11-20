# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""Train ML models, save best, register in model registry."""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from database.database import execute, query

FEATURE_COLS = [
    "previous_semester_marks", "previous_sgpa", "previous_backlogs",
    "ia1_score", "ia2_score", "ia3_score",
    "attendance", "assignment_average", "assignment_completion",
    "quiz_average", "study_hours",
]
TARGET = "actual_final_marks"
DATA_PATH = "data/raw/training_data.csv"
MODELS_DIR = "saved_models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).to_numpy()
    y = df[TARGET].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    Xtr, Xte, ytr, yte = load_data()

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    models = {
        "LinearRegression": (LinearRegression(), True),
        "DecisionTree":     (DecisionTreeRegressor(max_depth=8, random_state=42), False),
        "RandomForest":     (RandomForestRegressor(n_estimators=300, random_state=42), False),
        "XGBoost":          (xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                              max_depth=5, random_state=42), False),
    }

    results = {}
    for name, (m, needs_scale) in models.items():
        Xtr_use = Xtr_s if needs_scale else Xtr
        Xte_use = Xte_s if needs_scale else Xte
        m.fit(Xtr_use, ytr)
        pred = m.predict(Xte_use)
        mt = metrics(yte, pred)
        results[name] = mt
        print(f"{name:18s} MAE={mt['MAE']:.2f}  R2={mt['R2']:.3f}")

        path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pkl")
        joblib.dump(m, path)

        existing = query("SELECT id FROM model_registry WHERE model_name=?", (name,))
        if not existing:
            execute(
                """INSERT INTO model_registry
                   (model_name, model_version, model_type, evaluation_metrics, training_date, active_status)
                   VALUES (?, ?, ?, ?, datetime('now', 'localtime'), 0)""",
                (name, "1.0", "ml", json.dumps(mt)),
            )
        else:
            execute(
                """UPDATE model_registry 
                   SET evaluation_metrics=?, training_date=datetime('now', 'localtime') 
                   WHERE model_name=?""",
                (json.dumps(mt), name),
            )

    if "XGBoost" in results:
        execute("UPDATE model_registry SET active_status=0")
        execute("UPDATE model_registry SET active_status=1 WHERE model_name='XGBoost'")
        print("\nXGBoost chosen as the active production model.")

    return results


if __name__ == "__main__":
    print("Training ML models...\n")
    train_all()
    print("\nDone.")