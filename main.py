# backend/api/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import subprocess, os, json
from typing import Optional
import pandas as pd
import numpy as np
import requests
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import tensorflow as tf

app = FastAPI(title="Tamil Nadu Weather LSTM API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_model"
CSV_PATH = BASE_DIR / "data" / "tamilnadu_weather_2014_2024.csv"
DIST_GEOJSON_PATH = BASE_DIR / "data" / "tn_districts.geojson"

class TrainResponse(BaseModel):
    status: str
    message: str
    path: Optional[str]

# ----------------------------
# HEALTH
# ----------------------------
@app.get("/health")
def health():
    model_exists = (MODEL_DIR / "model.keras").exists()
    return {"ok": True, "model_present": model_exists}

# ----------------------------
# TRAIN MODEL
# ----------------------------
@app.post("/train", response_model=TrainResponse)
def train(epochs: int = 20, batch: int = 32):
    if not CSV_PATH.exists():
        raise HTTPException(status_code=400, detail="CSV not found.")

    cmd = [
        "python",
        str(MODEL_DIR / "train.py"),
        "--csv", str(CSV_PATH),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--outdir", str(MODEL_DIR)
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return TrainResponse(
            status="ok",
            message=proc.stdout[:800],
            path=str(MODEL_DIR / "model.keras"),
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed:\n{e.stdout}\n{e.stderr}")

# ----------------------------
# LIVE WEATHER (OPEN-METEO)
# ----------------------------
WEATHERCODE_MAP = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    80: "Slight rain showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm"
}

def _weathercode_to_text(code):
    return WEATHERCODE_MAP.get(int(code), "Unknown")

@app.get("/live-weather")
def live_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,weathercode",
        "timezone": "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        cw = data.get("current_weather", {})
        hourly = data.get("hourly", {})

        humidity = None
        if "relativehumidity_2m" in hourly and len(hourly["relativehumidity_2m"]) > 0:
            humidity = float(hourly["relativehumidity_2m"][-1])

        return {
            "temp": cw.get("temperature"),
            "wind": cw.get("windspeed"),
            "weathercode": cw.get("weathercode"),
            "description": _weathercode_to_text(cw.get("weathercode", 0)),
            "humidity": humidity,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# METRICS
# ----------------------------
@app.get("/metrics")
def metrics():
    hist_path = MODEL_DIR / "history.json"
    metrics_path = MODEL_DIR / "metrics.json"
    out = {}

    if hist_path.exists():
        try:
            out["history"] = json.loads(hist_path.read_text())
        except:
            out["history"] = None

    if metrics_path.exists():
        try:
            out["metrics"] = json.loads(metrics_path.read_text())
        except:
            out["metrics"] = None

    if not out:
        raise HTTPException(status_code=404, detail="No metrics found")

    return out

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def _nearest_station_to_point(lat, lon, df):
    pts = df[["lat", "lon"]].drop_duplicates().values
    best = None
    best_dist = 1e18
    for lat2, lon2 in pts:
        d = (lat - lat2)**2 + (lon - lon2)**2
        if d < best_dist:
            best_dist = d
            best = (lat2, lon2)
    return best

def classify_weather(temp, rain):
    if rain >= 20: return "Heavy Rain"
    if rain >= 10: return "Moderate Rain"
    if rain >= 2: return "Light Rain"
    if temp >= 36: return "Very Hot"
    if temp >= 32: return "Hot"
    if temp < 22: return "Cool"
    return "Clear / Cloudy"

def flood_alert_level(risk):
    if risk < 0.2: return ("Safe", "#10b981")
    elif risk < 0.4: return ("Caution", "#f59e0b")
    elif risk < 0.7: return ("Warning", "#f97316")
    else: return ("Severe", "#dc2626")

def get_season(month):
    if month in [3,4,5]: return "Summer (Mar–May)"
    if month in [6,7,8,9]: return "Southwest Monsoon (Jun–Sep)"
    if month in [10,11,12]: return "Northeast Monsoon (Oct–Dec)"
    return "Winter (Jan–Feb)"

def generate_explanation(preds):
    temps = [p[0] for p in preds]
    rains = [p[1] for p in preds]
    explanation = []

    if temps[-1] > temps[0]: explanation.append("Temperature rising.")
    elif temps[-1] < temps[0]: explanation.append("Temperature decreasing.")
    else: explanation.append("Temperature stable.")

    if rains[-1] > rains[0]: explanation.append("Rainfall increasing.")
    elif rains[-1] < rains[0]: explanation.append("Rainfall decreasing.")
    else: explanation.append("Rainfall stable.")

    return " ".join(explanation)

# ----------------------------
# PREDICTION API
# ----------------------------
@app.get("/predict-seq")
def predict_seq(
    days: int = Query(7, ge=1, le=14),
    lat: float = Query(None),
    lon: float = Query(None),
):
    """
    Multi-feature hybrid LSTM forecast.

    Uses last 14 days of [temp_c, rain_mm, humidity, wind_speed] from the nearest station
    and autoregressively predicts the next `days` steps for all four variables.
    """
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="lat & lon required")

    if not (MODEL_DIR / "model.keras").exists():
        raise HTTPException(status_code=400, detail="Model not trained")

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    nearest = _nearest_station_to_point(lat, lon, df)
    st_df = df[(df["lat"] == nearest[0]) & (df["lon"] == nearest[1])].sort_values("date")

    SEQ_LEN = 14
    if len(st_df) < SEQ_LEN:
        raise HTTPException(status_code=400, detail="Not enough data for this location")

    # Load scalers/model
    x_scaler = load(MODEL_DIR / "x_scaler.save")
    y_scaler = load(MODEL_DIR / "y_scaler.save")
    model = tf.keras.models.load_model(str(MODEL_DIR / "model.keras"))

    # Initial sequence
    seq_cols = ["temp_c", "rain_mm", "humidity", "wind_speed"]
    for c in seq_cols:
        if c not in st_df.columns:
            raise HTTPException(status_code=400, detail=f"CSV missing '{c}' column")

    seq = st_df[seq_cols].values.astype("float32")[-SEQ_LEN:]
    seq_run = seq.copy()

    preds = []
    out_entries = []

    for i in range(days):
        seq_scaled = x_scaler.transform(seq_run).reshape(1, SEQ_LEN, -1)
        pred_scaled = model.predict(seq_scaled)
        pred = y_scaler.inverse_transform(pred_scaled)[0]

        temp_pred = float(pred[0])
        rain_pred = float(pred[1])
        hum_pred = float(pred[2])
        wind_pred = float(pred[3])
        preds.append([temp_pred, rain_pred, hum_pred, wind_pred])

        flood_risk = round(min(1.0, rain_pred / 10.0), 3)
        alert, color = flood_alert_level(flood_risk)
        weather_type = classify_weather(temp_pred, rain_pred)

        out_entries.append(
            {
                "date": str(
                    (pd.Timestamp.today().normalize() + pd.Timedelta(days=i)).date()
                ),
                "temp_c": round(temp_pred, 2),
                "rain_mm": round(rain_pred, 2),
                "humidity": round(hum_pred, 1),
                "wind_speed": round(wind_pred, 1),
                "flood_risk": flood_risk,
                "flood_alert": alert,
                "flood_color": color,
                "weather_type": weather_type,
            }
        )

        # Slide window
        seq_run = np.vstack(
            [seq_run[1:], [[temp_pred, rain_pred, hum_pred, wind_pred]]]
        )

    explanation = generate_explanation(preds)
    season = get_season(pd.Timestamp.today().month)

    return {
        "station": {"lat": float(nearest[0]), "lon": float(nearest[1])},
        "predictions": out_entries,
        "explanation": explanation,
        "season": season,
    }

# ----------------------------
# DISTRICT FORECAST API
# ----------------------------
# ----------------------------
# DISTRICT FORECAST API
# ----------------------------
@app.get("/district-forecast")
def district_forecast(days: int = 7):
    """
    Returns GeoJSON of TN districts with average 4-parameter forecast
    (temp, rain, humidity, wind_speed) added to `properties.forecast`.
    """
    if not DIST_GEOJSON_PATH.exists():
        raise HTTPException(status_code=400, detail="GeoJSON missing")

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    model = tf.keras.models.load_model(str(MODEL_DIR / "model.keras"))
    x_scaler = load(MODEL_DIR / "x_scaler.save")
    y_scaler = load(MODEL_DIR / "y_scaler.save")
    SEQ_LEN = 14

    geo = json.loads(DIST_GEOJSON_PATH.read_text())

    for f in geo["features"]:
        geom = f["geometry"]
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
        else:
            ring = geom["coordinates"][0][0]

        lats = [c[1] for c in ring]
        lons = [c[0] for c in ring]

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        nearest = _nearest_station_to_point(center_lat, center_lon, df)
        st = df[(df["lat"] == nearest[0]) & (df["lon"] == nearest[1])].sort_values("date")

        if len(st) < SEQ_LEN:
            f["properties"]["forecast"] = None
            continue

        needed_cols = ["temp_c", "rain_mm", "humidity", "wind_speed"]
        for c in needed_cols:
            if c not in st.columns:
                f["properties"]["forecast"] = None
                continue

        seq = st[needed_cols].values.astype("float32")[-SEQ_LEN:]
        seq_run = seq.copy()
        out = []

        for d in range(days):
            seq_scaled = x_scaler.transform(seq_run).reshape(1, SEQ_LEN, -1)
            pred_scaled = model.predict(seq_scaled)
            pred = y_scaler.inverse_transform(pred_scaled)[0]

            temp, rain, hum, wind = map(float, pred)
            out.append([temp, rain, hum, wind])
            seq_run = np.vstack([seq_run[1:], [[temp, rain, hum, wind]]])

        out = np.array(out)
        f["properties"]["forecast"] = {
            "avg_temp": round(float(out[:, 0].mean()), 2),
            "avg_rain": round(float(out[:, 1].mean()), 2),
            "avg_humidity": round(float(out[:, 2].mean()), 1),
            "avg_wind_speed": round(float(out[:, 3].mean()), 1),
        }

    return geo

# ======================================================
# 🚀 ***AI Assistant Backend Route (local)***
# ======================================================

# ======================================================
# 🚀 ***AI Assistant Backend Route (OpenAI GPT-4o)***
# ======================================================


@app.post("/ask-ai")
async def ask_ai(data: dict):
    """
    Local AI assistant placeholder.

    Instead of calling OpenAI/Gemini, this endpoint synthesizes a short
    advisory message using simple rules on the text + (optional) weather info.
    Later, you can replace this logic with a real local LLM call.
    """
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt missing")

    # Simple heuristic-based response
    base_reply = "Here is a quick weather insight for Tamil Nadu: "

    lower = prompt.lower()
    tips = []

    if "rain" in lower or "flood" in lower:
        tips.append("Expect possible heavy showers on high-rainfall days, avoid low-lying areas.")
    if "heat" in lower or "hot" in lower or "temperature" in lower:
        tips.append("Carry water, avoid peak afternoon sun, and plan outdoor work in mornings/evenings.")
    if "wind" in lower or "storm" in lower:
        tips.append("Secure loose items and be cautious with tall structures during strong winds.")
    if "agriculture" in lower or "farm" in lower or "crop" in lower:
        tips.append("Schedule irrigation around low-rain days and watch high-rain days for waterlogging.")
    if not tips:
        tips.append("Use the 7-day forecast panel to time your travel or outdoor work safely.")

    reply = base_reply + " ".join(tips)

    return {"reply": reply}


# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
