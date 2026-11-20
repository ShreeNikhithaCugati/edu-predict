import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from src.data_prep import load_and_prepare
from src.dl_models import build_lstm, train_dl

Xs, Xf, y, ids, names = load_and_prepare("data/tests.csv", "data/labels.csv")
m = build_lstm(Xs.shape[1], Xs.shape[2])
r = train_dl(m, Xs, y)
print(f"LSTM            MAE={r['MAE']:.2f}  R2={r['R2']:.3f}")