# backend/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tamilnadu_weather.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "weather_lstm.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# how many past timesteps you use to predict future
SEQ_LENGTH = 24   # e.g. use past 24 hours

# ---- MULTIVARIATE FEATURES ----
# Change names here if your CSV header is different
FEATURES = [
    "temperature",   # or "temp"
    "humidity",
    "wind_speed",    # or "wind_speed_kmph"
    "rainfall"       # or "rain_fall_mm"
]
N_FEATURES = len(FEATURES)
