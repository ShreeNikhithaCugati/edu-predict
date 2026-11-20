# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""Train a plain ANN."""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

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


def build_ann(input_dim):
    m = models.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])
    return m


def train_ann(epochs=150, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)
    Xtr, Xte, ytr, yte = load_data()

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
    else:
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        joblib.dump(scaler, scaler_path)

    model = build_ann(Xtr_s.shape[1])
    es = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    history = model.fit(
        Xtr_s, ytr,
        validation_split=0.15,
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0,
    )

    pred = model.predict(Xte_s, verbose=0).flatten()
    mt = {
        "MAE": float(mean_absolute_error(yte, pred)),
        "MSE": float(mean_squared_error(yte, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
        "R2": float(r2_score(yte, pred)),
    }
    print(f"ANN                MAE={mt['MAE']:.2f}  R2={mt['R2']:.3f}")

    keras_path = os.path.join(MODELS_DIR, "ann_model.keras")
    model.save(keras_path)
    print(f"Saved ANN to {keras_path}")

    existing = query("SELECT id FROM model_registry WHERE model_name='ANN'")
    if not existing:
        execute(
            """INSERT INTO model_registry
               (model_name, model_version, model_type, evaluation_metrics, training_date, active_status)
               VALUES ('ANN', '1.0', 'dl', ?, datetime('now', 'localtime'), 0)""",
            (json.dumps(mt),),
        )
    else:
        execute(
            """UPDATE model_registry 
               SET evaluation_metrics=?, training_date=datetime('now', 'localtime') 
               WHERE model_name='ANN'""",
            (json.dumps(mt),),
        )

    return mt


if __name__ == "__main__":
    print("Training ANN...\n")
    train_ann()
    print("\nDone.")