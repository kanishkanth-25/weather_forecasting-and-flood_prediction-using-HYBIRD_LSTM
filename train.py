import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json, os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
import joblib

SEQ_LEN = 14


def build_model(seq_len=SEQ_LEN, n_features=2):
    """
    Hybrid CNN–LSTM model.
    - Input: (seq_len, n_features)
    - Branch 1: LSTM
    - Branch 2: 1D CNN
    - Output: next-step prediction for all features (n_features)
    """
    inp = layers.Input(shape=(seq_len, n_features))

    # ----- LSTM branch -----
    x_lstm = layers.LSTM(64, return_sequences=True)(inp)
    x_lstm = layers.Dropout(0.2)(x_lstm)
    x_lstm = layers.LSTM(32)(x_lstm)

    # ----- CNN branch -----
    x_cnn = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x_cnn = layers.MaxPooling1D()(x_cnn)
    x_cnn = layers.Conv1D(32, kernel_size=3, activation="relu")(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D()(x_cnn)

    # ----- Merge branches -----
    x = layers.concatenate([x_lstm, x_cnn])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # IMPORTANT: output dimension = n_features (supports 4 outputs)
    out = layers.Dense(n_features, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def prepare_from_csv(df, seq_len=SEQ_LEN):
    """
    Prepare sequences from CSV for 4 features:
        temp_c, rain_mm, humidity, wind_speed

    X: (samples, seq_len, n_features)
    y: (samples, n_features) -> next-step values for all features
    """
    cols = {c.lower(): c for c in df.columns}

    temp_col = cols.get("temp_c") or cols.get("temperature") or cols.get("temp")
    rain_col = cols.get("rain_mm") or cols.get("rain") or cols.get("precip_mm")
    hum_col = cols.get("humidity") or cols.get("hum")
    wind_col = (
        cols.get("wind_speed")
        or cols.get("windspeed")
        or cols.get("windspeed_kmph")
        or cols.get("wind_kmph")
        or cols.get("wind")
    )

    missing = []
    if temp_col is None:
        missing.append("temperature (temp_c / temperature / temp)")
    if rain_col is None:
        missing.append("rainfall (rain_mm / rain / precip_mm)")
    if hum_col is None:
        missing.append("humidity (humidity / hum)")
    if wind_col is None:
        missing.append("wind speed (wind_speed / windspeed / wind_kmph / wind)")

    if missing:
        raise ValueError("CSV must contain: " + ", ".join(missing))

    # Ensure station column
    if "station" not in df.columns:
        df["station"] = 0

    # Sort by station & date
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    # Next-step targets for ALL 4
    df["next_temp"] = df.groupby("station")[temp_col].shift(-1)
    df["next_rain"] = df.groupby("station")[rain_col].shift(-1)
    df["next_humidity"] = df.groupby("station")[hum_col].shift(-1)
    df["next_wind"] = df.groupby("station")[wind_col].shift(-1)

    df = df.dropna(subset=["next_temp", "next_rain", "next_humidity", "next_wind"])

    X_list, y_list = [], []

    # feature order: [temp, rain, humidity, wind]
    for sid, g in df.groupby("station"):
        arr = g[[temp_col, rain_col, hum_col, wind_col]].values.astype("float32")
        targets = g[
            ["next_temp", "next_rain", "next_humidity", "next_wind"]
        ].values.astype("float32")

        if len(arr) < seq_len + 1:
            continue

        for i in range(len(arr) - seq_len):
            X_list.append(arr[i : i + seq_len])
            y_list.append(targets[i + seq_len - 1])

    if not X_list:
        raise ValueError("Not enough time-series length in CSV for training.")

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y


def main(args):
    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Drop rows with missing core features
    cols = {c.lower(): c for c in df.columns}
    temp_col = cols.get("temp_c") or cols.get("temperature") or cols.get("temp")
    rain_col = cols.get("rain_mm") or cols.get("rain") or cols.get("precip_mm")
    hum_col = cols.get("humidity") or cols.get("hum")
    wind_col = (
        cols.get("wind_speed")
        or cols.get("windspeed")
        or cols.get("windspeed_kmph")
        or cols.get("wind_kmph")
        or cols.get("wind")
    )

    required = [c for c in [temp_col, rain_col, hum_col, wind_col] if c is not None]
    df = df.dropna(subset=required, how="any")

    # ---- Sequence creation ----
    X, y = prepare_from_csv(df, seq_len=SEQ_LEN)
    n_features = X.shape[2]

    # ---- Scaling ----
    X_flat = X.reshape(-1, n_features)
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_flat).reshape(X.shape)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Shuffle
    idx = np.arange(len(X_scaled))
    np.random.shuffle(idx)
    X_scaled = X_scaled[idx]
    y_scaled = y_scaled[idx]

    # Train/val split
    n_train = int(0.85 * len(X_scaled))
    X_train, X_val = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_val = y_scaled[:n_train], y_scaled[n_train:]

    model = build_model(seq_len=SEQ_LEN, n_features=n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        # IMPORTANT: cast Path to str for Windows
        keras.callbacks.ModelCheckpoint(str(outdir / "model.keras"), save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )

    # Save scalers + history
    joblib.dump(x_scaler, outdir / "x_scaler.save")
    joblib.dump(y_scaler, outdir / "y_scaler.save")

    with open(outdir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    # ---- Evaluate on val (unscaled) ----
    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = y_scaler.inverse_transform(y_val)

    # Simple RMSE calculation using numpy (no 'squared' argument)
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # indices: 0=temp, 1=rain, 2=humidity, 3=wind
    rmse_temp = rmse(y_val_true[:, 0], y_val_pred[:, 0])
    rmse_rain = rmse(y_val_true[:, 1], y_val_pred[:, 1])
    rmse_hum  = rmse(y_val_true[:, 2], y_val_pred[:, 2])
    rmse_wind = rmse(y_val_true[:, 3], y_val_pred[:, 3])

    mae_temp = mean_absolute_error(y_val_true[:, 0], y_val_pred[:, 0])
    mae_rain = mean_absolute_error(y_val_true[:, 1], y_val_pred[:, 1])
    mae_hum  = mean_absolute_error(y_val_true[:, 2], y_val_pred[:, 2])
    mae_wind = mean_absolute_error(y_val_true[:, 3], y_val_pred[:, 3])

    metrics = {
        "rmse_temp": float(round(rmse_temp, 4)),
        "rmse_rain": float(round(rmse_rain, 4)),
        "rmse_humidity": float(round(rmse_hum, 4)),
        "rmse_wind": float(round(rmse_wind, 4)),
        "mae_temp": float(round(mae_temp, 4)),
        "mae_rain": float(round(mae_rain, 4)),
        "mae_humidity": float(round(mae_hum, 4)),
        "mae_wind": float(round(mae_wind, 4)),
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Training complete. Model and scalers saved to", outdir)
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    main(args)
