import requests
import pandas as pd

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

TN_LOCATIONS = [
    ("Chennai",       13.0827, 80.2707),
    ("Coimbatore",    11.0168, 76.9558),
    ("Madurai",        9.9252, 78.1198),
    ("Tiruchirappalli",10.7905, 78.7047),
    ("Salem",         11.6643, 78.1460),
    ("Tirunelveli",    8.7139, 77.7567),
    ("Erode",         11.3410, 77.7172),
    ("Vellore",       12.9165, 79.1325),
    ("Kanyakumari",    8.0883, 77.5385),
    ("Thanjavur",     10.7867, 79.1378),
]

def fetch_location(name, lat, lon):
    print(f"Downloading for {name} ({lat}, {lon})...")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&daily=temperature_2m_mean,precipitation_sum,"
        "relative_humidity_2m_mean,windspeed_10m_max"
        "&timezone=auto"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temp_c": daily["temperature_2m_mean"],
        "rain_mm": daily["precipitation_sum"],
        "humidity": daily["relative_humidity_2m_mean"],
        "wind_speed": daily["windspeed_10m_max"],
    })

    df["district"] = name
    df["lat"] = lat
    df["lon"] = lon
    return df

def main():
    all_dfs = []
    for name, lat, lon in TN_LOCATIONS:
        try:
            df = fetch_location(name, lat, lon)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed for {name}: {e}")

    if not all_dfs:
        print("No data downloaded!")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["district", "date"])

    out_path = "data/tamilnadu_weather_2014_2024.csv"
    df_all.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    print(df_all.head())

if __name__ == "__main__":
    main()
