"""Test that trained models load and predict."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib

MODELS_DIR = "saved_models"
FEATURE_COLS = [
    "previous_semester_marks", "previous_sgpa", "previous_backlogs",
    "ia1_score", "ia2_score", "ia3_score",
    "attendance", "assignment_average", "assignment_completion",
    "quiz_average", "study_hours",
]


def test_models_exist():
    for f in ["xgboost_model.pkl", "randomforest_model.pkl",
              "ann_model.keras", "scaler.pkl"]:
        path = os.path.join(MODELS_DIR, f)
        assert os.path.exists(path), f"Missing model: {f}"


def test_xgboost_predicts():
    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if not os.path.exists(path):
        print("⚠️  XGBoost not found — skipping")
        return
    m = joblib.load(path)
    X = np.array([[70, 7, 0, 20, 22, 18, 85, 75, 80, 70, 3]])
    pred = float(m.predict(X)[0])
    assert 0 <= pred <= 100, f"Prediction out of range: {pred}"


def test_ann_predicts():
    path = os.path.join(MODELS_DIR, "ann_model.keras")
    if not os.path.exists(path):
        print("⚠️  ANN not found — skipping")
        return
    from tensorflow.keras.models import load_model
    m = load_model(path)
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X = scaler.transform([[70, 7, 0, 20, 22, 18, 85, 75, 80, 70, 3]])
    pred = float(m.predict(X, verbose=0)[0][0])
    assert 0 <= pred <= 100, f"ANN prediction out of range: {pred}"


if __name__ == "__main__":
    test_models_exist()
    test_xgboost_predicts()
    test_ann_predicts()
    print("[PASS] Prediction tests passed")