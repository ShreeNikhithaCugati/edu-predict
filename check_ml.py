from src.data_prep import load_and_prepare
from src.ml_models import train_ml_models

Xs, Xf, y, ids, names = load_and_prepare("data/tests.csv", "data/labels.csv")
print("X_seq", Xs.shape, "| X_flat", Xf.shape, "| y", y.shape)

res = train_ml_models(Xf, y)
for name, r in res.items():
    print(f"{name:15s} MAE={r['MAE']:.2f}  R2={r['R2']:.3f}")