"""Deep learning sequence models."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def _compile(m):
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])
    return m


def build_lstm(seq_len, n_feats):
    inp = layers.Input((seq_len, n_feats))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return _compile(models.Model(inp, out))


def build_cnn(seq_len, n_feats):
    inp = layers.Input((seq_len, n_feats))
    x = layers.Conv1D(32, 2, activation="relu", padding="same")(inp)
    x = layers.Conv1D(64, 2, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    return _compile(models.Model(inp, out))


def build_transformer(seq_len, n_feats, d=32, heads=4):
    inp = layers.Input((seq_len, n_feats))
    x = layers.Dense(d)(inp)
    attn_out, attn_scores = layers.MultiHeadAttention(
        num_heads=heads, key_dim=d, name="mha")(x, x, return_attention_scores=True)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    attn_model = models.Model(inp, attn_scores)
    return _compile(m), attn_model


def train_dl(model, X_seq, y, epochs=80, batch_size=32, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X_seq, y, test_size=0.2, random_state=random_state)
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs,
                     batch_size=batch_size, callbacks=[es], verbose=0)
    pred = model.predict(Xte, verbose=0).flatten()
    return {
        "model": model,
        "history": hist.history,
        "MAE": float(mean_absolute_error(yte, pred)),
        "R2": float(r2_score(yte, pred)),
        "y_true": yte,
        "y_pred": pred,
        "X_test": Xte,
    }