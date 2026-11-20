import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.data_prep import load_and_prepare
from src.ml_models import train_ml_models
from src.dl_models import build_lstm, build_cnn, build_transformer, train_dl

Xs, Xf, y, ids, names = load_and_prepare("data/tests.csv", "data/labels.csv")
print(f"Data: X_seq={Xs.shape}  X_flat={Xf.shape}  y={y.shape}\n")

print("=== ML baselines ===")
res = train_ml_models(Xf, y)
for name, r in res.items():
    print(f"{name:15s} MAE={r['MAE']:.2f}  R2={r['R2']:.3f}")

print("\n=== DL sequence models ===")
for name, builder in [("LSTM", build_lstm), ("CNN", build_cnn)]:
    m = builder(Xs.shape[1], Xs.shape[2])
    r = train_dl(m, Xs, y)
    print(f"{name:15s} MAE={r['MAE']:.2f}  R2={r['R2']:.3f}")

m, _ = build_transformer(Xs.shape[1], Xs.shape[2])
r = train_dl(m, Xs, y)
print(f"{'Transformer':15s} MAE={r['MAE']:.2f}  R2={r['R2']:.3f}")