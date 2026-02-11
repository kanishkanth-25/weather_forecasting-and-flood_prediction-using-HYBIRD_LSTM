from pathlib import Path
import numpy as np, pandas as pd, joblib
from tensorflow import keras
import argparse

SEQ_LEN = 14


def load_model_and_scalers(model_dir):
    model_dir = Path(model_dir)
    model = keras.models.load_model(model_dir / "model.keras")
    x_scaler = joblib.load(model_dir / "x_scaler.save")
    y_scaler = joblib.load(model_dir / "y_scaler.save")
    return model, x_scaler, y_scaler


def predict_from_recent(df, model, x_scaler, y_scaler):
    """
    Uses last SEQ_LEN rows of the station time series.

    Expects CSV to contain:
        temp_c, rain_mm, humidity, wind_speed
    Returns: dict with predicted values.
    """
    cols = {c.lower(): c for c in df.columns}

    temp_col = cols.get("temp_c") or cols.get("temperature") or cols.get("temp")
    rain_col = cols.get("rain_mm") or cols.get("rain")
    hum_col = cols.get("humidity") or cols.get("hum")
    wind_col = (
        cols.get("wind_speed")
        or cols.get("windspeed")
        or cols.get("windspeed_kmph")
        or cols.get("wind_kmph")
        or cols.get("wind")
    )

    for name, val in [
        ("temp_c", temp_col),
        ("rain_mm", rain_col),
        ("humidity", hum_col),
        ("wind_speed", wind_col),
    ]:
        if val is None:
            raise ValueError(f"CSV missing {name} column")

    seq = df[[temp_col, rain_col, hum_col, wind_col]].values.astype("float32")[-SEQ_LEN:]
    seq_scaled = x_scaler.transform(seq).reshape(1, SEQ_LEN, -1)
    pred_scaled = model.predict(seq_scaled)
    pred = y_scaler.inverse_transform(pred_scaled)[0]

    return {
        "temp_c": float(pred[0]),
        "rain_mm": float(pred[1]),
        "humidity": float(pred[2]),
        "wind_speed": float(pred[3]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=".")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    model, x_scaler, y_scaler = load_model_and_scalers(args.model_dir)
    pred = predict_from_recent(df, model, x_scaler, y_scaler)
    print("Predicted next day:", pred)
