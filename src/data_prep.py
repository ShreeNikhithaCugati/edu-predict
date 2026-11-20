"""
Builds sequences (n_students, seq_len, n_feats) from long-format data.
Adds trajectory features for ML baselines.
"""
import numpy as np
import pandas as pd

FEATS = ["marks", "attendance", "assignment", "participation"]

def build_sequences(df: pd.DataFrame, seq_len: int = 4):
    df = df.sort_values(["student_id", "test_no"])
    ids, X_seq, X_flat = [], [], []
    for sid, g in df.groupby("student_id"):
        arr = g[FEATS].to_numpy(dtype=float)[:seq_len]
        n_real = len(arr)
        mask = np.ones((n_real, 1))
        if n_real < seq_len:
            pad = np.repeat(arr[-1:], seq_len - n_real, axis=0)
            pad_mask = np.zeros((seq_len - n_real, 1))
            arr = np.vstack([arr, pad])
            mask = np.vstack([mask, pad_mask])
        seq = np.hstack([arr, mask])
        ids.append(sid)
        X_seq.append(seq)
        X_flat.append(arr.flatten())

    X_seq = np.array(X_seq)
    X_flat = np.array(X_flat)
    feat_names_flat = [f"test{t+1}_{f}" for t in range(seq_len) for f in FEATS]
    return X_seq, X_flat, ids, feat_names_flat


def trajectory_features(X_flat: np.ndarray, seq_len: int = 4) -> np.ndarray:
    n = X_flat.shape[0]
    X = X_flat.reshape(n, seq_len, -1)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    delta = X[:, -1, :] - X[:, 0, :]
    t = np.arange(seq_len)
    slopes = np.zeros_like(mean)
    for i in range(n):
        for f in range(X.shape[2]):
            slopes[i, f] = np.polyfit(t, X[i, :, f], 1)[0]
    return np.hstack([X_flat, mean, std, delta, slopes])


def load_and_prepare(tests_csv: str, labels_csv: str, seq_len: int = 4):
    df = pd.read_csv(tests_csv)
    labels = pd.read_csv(labels_csv)
    X_seq, X_flat, ids, flat_names = build_sequences(df, seq_len)
    X_flat_aug = trajectory_features(X_flat, seq_len)

    aug_names = flat_names + [f"mean_{f}" for f in FEATS] \
                          + [f"std_{f}" for f in FEATS] \
                          + [f"delta_{f}" for f in FEATS] \
                          + [f"slope_{f}" for f in FEATS]

    id_to_label = dict(zip(labels["student_id"], labels["final_score"]))
    y = np.array([id_to_label[s] for s in ids])
    return X_seq, X_flat_aug, y, ids, aug_names