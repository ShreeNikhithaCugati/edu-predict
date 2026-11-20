import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np

from src.data_prep import load_and_prepare
from src.ml_models import train_ml_models
from src.xai import shap_tree

Xs, Xf, y, ids, names = load_and_prepare("data/tests.csv", "data/labels.csv")
res = train_ml_models(Xf, y)
sv = shap_tree(res["XGBoost"]["model"], Xf, names, plot=False)
print("SHAP shape:", sv.shape)
top = np.argsort(np.abs(sv).mean(0))[::-1][:3]
print("Top features:", [names[i] for i in top])