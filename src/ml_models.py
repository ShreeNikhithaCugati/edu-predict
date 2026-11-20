"""Classical ML baselines for trajectory features."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def train_ml_models(X, y, test_size=0.2, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    models = {
        "Ridge": (Ridge(alpha=1.0), Xtr_s, Xte_s),
        "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=random_state), Xtr, Xte),
        "XGBoost": (xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                     random_state=random_state), Xtr, Xte),
    }

    results = {}
    for name, (m, xtr, xte) in models.items():
        m.fit(xtr, ytr)
        pred = m.predict(xte)
        results[name] = {
            "model": m,
            "scaler": scaler if name == "Ridge" else None,
            "MAE": float(mean_absolute_error(yte, pred)),
            "R2": float(r2_score(yte, pred)),
            "y_true": yte,
            "y_pred": pred,
        }
    return results